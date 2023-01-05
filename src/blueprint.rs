#![allow(clippy::cast_sign_loss)]
#![allow(clippy::inline_always)]

pub mod explore;
pub mod parse;
pub mod trace;

use crate::logic::{GateNetwork, GateType, LogicSim};
use explore::{compile_network, BoardNode};
use parse::VcbPlainBoard;
pub use parse::{VcbInput, VcbParser};
use trace::*;

use anyhow::{anyhow, Context};
use crossterm::style::{Color, Colors, Print, ResetColor, SetColors, Stylize};
use crossterm::{terminal, QueueableCommand};

use std::array::from_fn;
use std::collections::BTreeSet;
use std::io::{stdout, Write};
use std::mem::size_of;
use std::thread::sleep;
use std::time::Duration;

pub struct VcbBoard<T: LogicSim> {
    traces: Vec<Trace>,
    element_ids_internal: Vec<Option<usize>>,
    element_ids_external: Vec<Option<usize>>, // to debug
    elements: Vec<BoardElement>,
    nodes: Vec<BoardNode>,
    pub(crate) logic_sim: T,
    pub(crate) width: usize,
    pub(crate) height: usize,
}
impl<T: LogicSim> VcbBoard<T> {
    fn num_elements(&self) -> usize {
        self.width * self.height
    }
    fn new(plain: VcbPlainBoard, optimize: bool) -> Self {
        fn element_id_to_external_id(
            elements: &[BoardElement],
            nodes: &[BoardNode],
            id: usize,
        ) -> Option<usize> {
            elements[id].id.and_then(|id| nodes[id].network_id)
        }

        let (height, width, nodes, elements, network) = compile_network::<T>(&plain);
        let element_ids_external: Vec<_> = (0..elements.len())
            .map(|id| element_id_to_external_id(&elements, &nodes, id))
            .collect();
        let logic_sim: T = network.compiled(optimize);
        let element_ids: Vec<_> = element_ids_external
            .iter()
            .map(|id| id.map(|id| logic_sim.to_internal_id(id)))
            .collect();

        VcbBoard {
            element_ids_internal: element_ids,
            element_ids_external,
            traces: plain.traces,
            elements,
            nodes,
            logic_sim,
            width,
            height,
        }
    }
}
impl<T: LogicSim> VcbBoard<T> {
    pub fn update_i(&mut self, iterations: usize) {
        self.logic_sim.update_i(iterations);
    }
    #[inline(always)]
    pub fn update(&mut self) {
        self.logic_sim.update();
    }
    /// For regression testing
    /// Get state for every pixel
    #[must_use]
    #[cfg(test)]
    pub fn make_state_vec(&self) -> Vec<bool> {
        (0..self.traces.len())
            .map(|i| self.get_state_element(i))
            .collect()
    }
    #[must_use]
    #[cfg(test)]
    pub(crate) fn make_inner_state_vec(&self) -> Vec<bool> {
        self.logic_sim.get_state_vec()
    }
}

impl<T: LogicSim> VcbBoard<T> {
    /// internal id -> state
    #[must_use]
    fn get_state_internal_id(&self, id: usize) -> bool {
        self.logic_sim.get_state_internal(id)
    }
    /// pixel -> state
    #[must_use]
    fn get_state_element(&self, id: usize) -> bool {
        self.get_internal_id(id)
            .map(|id| self.get_state_internal_id(id))
            .unwrap_or_default()
    }
    /// pixel -> external_id
    #[must_use]
    fn get_external_id(&self, id: usize) -> Option<usize> {
        self.element_ids_external[id]
    }
    /// pixel -> internal_id
    #[must_use]
    fn get_internal_id(&self, id: usize) -> Option<usize> {
        self.element_ids_internal[id]
    }
    /// pixel -> color
    #[must_use]
    fn get_color_element(&self, id: usize) -> [u8; 4] {
        self.traces[id].get_color(self.get_state_element(id))
    }
    /// x,y -> pixel
    fn pixel_id(&self, x: usize, y: usize) -> usize {
        x + y * self.width
    }
}
impl<T: LogicSim> VcbBoard<T> {
    fn print_generic_debug<F: Fn(usize) -> String>(&self, f: F, constrain: bool) {
        let (sx, sy) = terminal::size().unwrap_or((50, 50));
        let (max_print_width, max_print_height) = if constrain {
            (
                self.width.min((sx as usize) / 2),
                self.height.min(sy as usize - 2),
            )
        } else {
            (self.width, self.height)
        };
        println!("\nBoard:");
        for y in 0..max_print_height {
            for x in 0..max_print_width {
                let id = self.pixel_id(x, y);
                let s = f(id);
                let col = self.get_color_element(id);
                let col1: Color = (col[0], col[1], col[2]).into();
                let col2: Color = (255 - col[0], 255 - col[1], 255 - col[2]).into();
                print!("{}", s.on(col1).with(col2));
            }
            println!();
        }
    }
}
impl<T: LogicSim> VcbBoard<T> {
    fn print_inner(&self, debug_inner: Option<bool>, constrain: bool) {
        let format = |id: usize| {
            debug_inner
                .map(|s| {
                    if s {
                        self.get_internal_id(id)
                    } else {
                        self.get_external_id(id)
                    }
                    .map(|s| s % 100)
                })
                .flatten()
                .map(|s| format!("{s:>2}"))
                .unwrap_or("  ".to_string())
        };

        self.print_generic_debug(format, constrain);
    }

    fn print_compact(&self) -> Result<(), std::io::Error> {
        let mut stdout = stdout();
        let (sx, sy) = terminal::size()?;
        let max_print_width = self.width.min(sx as usize);
        let max_print_height = self.height.min(2 * sy as usize - 4);
        stdout.queue(Print(format!(
            "{:?} {:?}\n",
            (sx, sy),
            (self.width, self.height)
        )))?;
        for y in (0..max_print_height).step_by(2) {
            for x in 0..max_print_width {
                let i = x + y * self.width;
                let i2 = x + (y + 1) * self.width;
                //let col = self.elements[i].get_color(self);
                let col = self.get_color_element(i);
                let col2 = (i2 / self.width < self.height)
                    .then(|| self.get_color_element(i2))
                    .unwrap_or(Trace::Empty.to_color_off());
                //let col2 = self
                //    .elements
                //    .get(i2)
                //    .map_or(Trace::Empty.to_color_off(), |s| s.get_color(self));

                stdout.queue(SetColors(Colors::new(
                    (col2[0], col2[1], col2[2]).into(),
                    (col[0], col[1], col[2]).into(),
                )))?;
                stdout.queue(Print("▄"))?;
            }
            stdout.queue(Print("\n"))?;
        }
        stdout.queue(ResetColor)?;
        stdout.flush()?;
        Ok(())
    }
    fn print_using_translation<F: Fn(Trace) -> &'static str>(&self, fun: F, legend: bool) {
        for y in 0..self.height {
            let mut min_print_width = 0;
            for x in 0..self.width {
                let i = x + y * self.width;
                if !matches!(self.elements[i].trace, Trace::Empty) {
                    min_print_width = x;
                }
            }
            min_print_width += 1;
            for x in 0..min_print_width {
                let i = x + y * self.width;
                let trace = self.elements[i].trace;
                let printed_str = fun(trace);
                print!("{printed_str}");
            }
            println!();
        }
        if legend {
            self.print_legend(|x| fun(x).to_string(), |x| format!("{x:?}"));
        }
    }
    pub fn print_regular_emoji(&self, legend: bool) {
        self.print_using_translation(Trace::as_regular_emoji, legend);
    }
    pub fn print_vcb_discord_emoji(&self, legend: bool) {
        self.print_using_translation(Trace::as_discord_emoji, legend);
    }
    pub fn print_debug(&self) {
        self.print_inner(Some(true), false);
        self.print_inner(Some(false), false);
    }
    pub fn print_debug_constrain(&self) {
        self.print_inner(Some(true), true);
        self.print_inner(Some(false), true);
    }
    /// # Errors
    /// cannot print
    pub fn print(&self) -> Result<(), std::io::Error> {
        self.print_compact()
    }
    fn print_legend<F: Fn(Trace) -> String, U: Fn(Trace) -> String>(&self, f1: F, f2: U) {
        fn get_current_traces<T: LogicSim>(b: &VcbBoard<T>) -> Vec<Trace> {
            b.elements
                .iter()
                .map(|e| e.trace)
                .fold(std::collections::HashSet::new(), |mut set, trace| {
                    set.insert(trace);
                    set
                })
                .into_iter()
                .collect()
        }
        for t in get_current_traces(&self) {
            println!("{} = {}", f1(t), f2(t));
        }
    }

    /// # Panics
    /// Cannot set clipboard
    pub fn print_to_clipboard(&self) -> ! {
        use arboard::{Clipboard, ImageData};
        use std::borrow::Cow;

        let color_data: Vec<u8> = self.traces.iter().flat_map(|x| x.to_color_on()).collect();
        let mut clipboard = Clipboard::new().unwrap();
        clipboard
            .set_image(ImageData {
                width: self.width,
                height: self.height,
                bytes: Cow::from(color_data),
            })
            .unwrap();
        println!("Running infinite loop so that clipboard contents are preserved, CTRL-C to force exit...");
        loop {
            sleep(Duration::from_secs(1));
        }
    }
    /// # Panics
    /// very large image
    pub fn print_to_gif(&mut self, limit: usize) {
        #![allow(clippy::cast_precision_loss)]
        #![allow(clippy::cast_possible_truncation)]
        #![allow(clippy::cast_lossless)]
        use image::codecs::gif::GifEncoder;
        use image::{codecs, imageops, Frame, ImageBuffer, RgbaImage};
        use std::collections::HashMap;
        use tempfile::{Builder, NamedTempFile};

        type BoardColorData = Vec<u8>;
        let mut v: Vec<BoardColorData> = Vec::new();
        let mut i = 0;
        let mut map: HashMap<BoardColorData, usize> = HashMap::new();

        println!("Generating colors...");
        let a = loop {
            let color_data: BoardColorData = (0..self.num_elements())
                .flat_map(|i| self.get_color_element(i))
                .collect();

            v.push(color_data.clone()); // optimization is my passion
            match map.insert(color_data, i) {
                None => (),
                Some(k) => {
                    break k;
                },
            };
            if i == limit {
                break 0;
            }
            self.update();
            i += 1;
        };
        let file: NamedTempFile = Builder::new().suffix(".gif").tempfile().unwrap();
        let (file, path) = file.keep().unwrap();

        let image_width = 500_u32;
        let image_height = (image_width as f64 / self.width as f64 * self.height as f64) as u32;

        let frames = v
            .into_iter()
            .enumerate()
            .map(|(iframe, x)| {
                println!("{iframe}/{i}");
                let rgba: RgbaImage = ImageBuffer::from_raw(
                    self.width.try_into().unwrap(),
                    self.height.try_into().unwrap(),
                    x,
                )
                .unwrap();
                let rgba: RgbaImage =
                    imageops::resize(&rgba, image_width, image_height, imageops::Nearest);
                Frame::new(rgba)
            })
            .skip(a);

        println!("Encoding into gif");
        // 1 is slowest and highest quality
        let mut gif_encoder = GifEncoder::new_with_speed(file, 1);
        gif_encoder
            .set_repeat(codecs::gif::Repeat::Infinite)
            .unwrap();
        gif_encoder.encode_frames(frames).unwrap();

        println!("Gif stored at: {path:?}");
    }
}

/// Represents one pixel.
/// It is probably a mistake to
/// make a copy of this type.
struct BoardElement {
    trace: Trace,
    id: Option<usize>,
}
impl BoardElement {
    fn new(trace: Trace) -> Self {
        BoardElement { trace, id: None }
    }
}
