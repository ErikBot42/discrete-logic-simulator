#![allow(clippy::cast_sign_loss)]
#![allow(clippy::inline_always)]

use trace::*;
pub mod parse;
pub mod trace;
use crate::logic::{GateNetwork, GateType, LogicSim};
pub use parse::VcbParser;

use anyhow::{anyhow, Context};
use crossterm::style::{Color, Colors, Print, ResetColor, SetColors, Stylize};
use crossterm::QueueableCommand;

use std::array::from_fn;
use std::collections::BTreeSet;
use std::io::{stdout, Write};
use std::mem::size_of;
use std::thread::sleep;
use std::time::Duration;

#[derive(Clone)]
pub enum VcbInput {
    BlueprintLegacy(String),
    Blueprint(String),
    WorldLegacy(String),
    World(String),
}

/// Represents one gate or trace
struct BoardNode {
    inputs: BTreeSet<usize>,
    outputs: BTreeSet<usize>,
    kind: GateType,
    initial_state: bool,
    network_id: Option<usize>,
}
impl BoardNode {
    #[must_use]
    fn new(trace: Trace) -> Self {
        let (kind, initial_state) = trace.to_gatetype_state();
        BoardNode {
            inputs: BTreeSet::new(),
            outputs: BTreeSet::new(),
            initial_state,
            kind,
            network_id: None,
        }
    }
}


pub struct VcbBoard<T: LogicSim> {
    traces: Vec<Trace>,
    element_ids: Vec<Option<usize>>,
    element_ids_external: Vec<Option<usize>>, // to debug
    elements: Vec<BoardElement>,
    nodes: Vec<BoardNode>,
    pub(crate) logic_sim: T,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<T: LogicSim> VcbBoard<T> {
    fn new(plain: parse::VcbPlainBoard, optimize: bool) -> Self {
        let height = plain.height;
        let width = plain.width;
        let num_elements = width * height;

        let mut nodes = Vec::new();
        let elements = timed!(
            {
                let mut elements: Vec<_> = plain
                    .traces
                    .iter()
                    .cloned()
                    .map(BoardElement::new)
                    .collect();
                for x in 0..num_elements {
                    explore::explore(
                        &mut elements,
                        &mut nodes,
                        width.try_into().unwrap(),
                        x.try_into().unwrap(),
                    );
                }
                elements
            },
            "create elements: {:?}"
        );

        let network = {
            let mut network = GateNetwork::default();

            // add vertexes to network
            for node in &mut nodes {
                node.network_id = Some(network.add_vertex(node.kind, node.initial_state));
            }
            // add edges to network
            for (i, node) in nodes.iter().enumerate() {
                for input in &node.inputs {
                    assert!(nodes[*input].outputs.contains(&i));
                }
                let mut inputs: Vec<usize> = node
                    .inputs
                    .clone()
                    .into_iter()
                    .map(|x| nodes[x].network_id.unwrap())
                    .collect();
                inputs.sort_unstable();
                inputs.dedup();
                network.add_inputs(node.kind, node.network_id.unwrap(), inputs);
            }
            network
        };
        let element_ids_external: Vec<_> = (0..elements.len())
            .map(|id| Self::element_id_to_external_id(&elements, &nodes, id))
            .collect();
        let logic_sim: T = network.compiled(optimize);
        let element_ids: Vec<_> = element_ids_external
            .iter()
            .map(|id| id.map(|id| logic_sim.to_internal_id(id)))
            .collect();

        VcbBoard {
            element_ids,
            element_ids_external,
            traces: plain.traces,
            elements,
            nodes,
            logic_sim,
            width,
            height,
        }
    }
    fn get_state_element(&self, id: usize) -> bool {
        //Self::element_id_to_internal_id(&self.elements, &self.nodes, &self.compiled_network, id)
        self.element_ids[id]
            .map(|id| self.logic_sim.get_state_internal(id))
            .unwrap_or_default()
    }

    // element id to internal compiled network id
    fn element_id_to_internal_id(
        elements: &[BoardElement],
        nodes: &[BoardNode],
        compiled_network: &T,
        id: usize,
    ) -> Option<usize> {
        elements[id]
            .id
            .and_then(|id| nodes[id].network_id)
            .map(|id| compiled_network.to_internal_id(id))
    }

    fn element_id_to_external_id(
        elements: &[BoardElement],
        nodes: &[BoardNode],
        id: usize,
    ) -> Option<usize> {
        elements[id].id.and_then(|id| nodes[id].network_id)
    }

    /// For regression testing
    #[must_use]
    pub fn make_state_vec(&self) -> Vec<bool> {
        let mut a = Vec::new();
        for i in 0..self.elements.len() {
            a.push(self.get_state_element(i));
        }
        a
    }
    #[must_use]
    #[cfg(test)]
    pub(crate) fn make_inner_state_vec(&self) -> Vec<bool> {
        self.logic_sim.get_state_vec()
    }
    pub fn update_i(&mut self, iterations: usize) {
        self.logic_sim.update_i(iterations);
    }
    #[inline(always)]
    pub fn update(&mut self) {
        self.logic_sim.update();
    }

    pub fn print_marked(&self, marked: &[usize]) {
        println!("\nBoard:");
        for y in 0..self.height {
            for x in 0..self.width {
                let i = x + y * self.width;
                let element = &self.elements[i];
                element.print(
                    self,
                    i,
                    element.id.map_or(false, |i| marked.contains(&i)),
                    Some(true),
                );
            }
            println!();
        }
    }
    fn print_inner(&self, debug_inner: Option<bool>) {
        println!("\nBoard:");
        for y in 0..self.height {
            for x in 0..self.width {
                let i = x + y * self.width;
                self.elements[i].print(self, i, false, debug_inner);
            }
            println!();
        }
    }
    fn print_compact(&self) -> Result<(), std::io::Error> {
        let mut stdout = stdout();
        let (sx, sy) = crossterm::terminal::size()?;
        stdout.queue(Print(format!(
            "{:?} {:?}\n",
            (sx, sy),
            (self.width, self.height)
        )))?;
        let max_print_width = self.width.min(sx as usize);
        let max_print_height = self.height.min(2 * sy as usize - 4);
        for y in (0..max_print_height).step_by(2) {
            for x in 0..max_print_width {
                let i = x + y * self.width;
                let i2 = x + (y + 1) * self.width;
                let col = self.elements[i].get_color(self);
                let col2 = self
                    .elements
                    .get(i2)
                    .map_or(Trace::Empty.to_color_off(), |s| s.get_color(self));

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
        self.print_inner(Some(true));
        self.print_inner(Some(false));
    }
    /// # Errors
    /// cannot print
    pub fn print(&self) -> Result<(), std::io::Error> {
        self.print_compact()
    }
    fn print_legend<F: Fn(Trace) -> String, U: Fn(Trace) -> String>(&self, f1: F, f2: U) {
        for t in self.get_current_traces() {
            println!("{} = {}", f1(t), f2(t));
        }
    }
    fn get_current_traces(&self) -> Vec<Trace> {
        self.elements
            .iter()
            .map(|e| e.trace)
            .fold(std::collections::HashSet::new(), |mut set, trace| {
                set.insert(trace);
                set
            })
            .into_iter()
            .collect()
    }

    /// # Panics
    /// Cannot set clipboard
    pub fn print_to_clipboard(&self) -> ! {
        use arboard::{Clipboard, ImageData};
        use std::borrow::Cow;

        let color_data: Vec<u8> = self
            .elements
            .iter()
            .flat_map(|x| x.trace.to_color_on())
            .collect();
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
            let color_data: BoardColorData = self
                .elements
                .iter()
                .flat_map(|x| x.get_color(self))
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
    fn print<T: LogicSim>(
        &self,
        board: &VcbBoard<T>,
        _i: usize,
        marked: bool,
        debug_inner: Option<bool>,
    ) {
        let format = |debug: Option<bool>, id: usize| match debug {
            None => "  ".to_string(),
            Some(true) => {
                format!("{:>2}", board.logic_sim.to_internal_id(id))
            },
            Some(false) => {
                format!("{id:>2}")
            },
        };

        let mut state = false;
        let tmpstr = if let Some(t) = self.id {
            state = board.nodes[t]
                .network_id
                .map(|i| board.logic_sim.get_state(i))
                .unwrap_or_default();
            format(debug_inner, t % 100)
        } else {
            "  ".to_string()
        };
        let col = if state {
            self.trace.to_color_on()
        } else {
            self.trace.to_color_off()
        };
        let col1: Color = (col[0], col[1], col[2]).into();
        let col2: Color = (255 - col[0], 255 - col[1], 255 - col[2]).into();

        let (col1, col2) = if marked { (col2, col1) } else { (col1, col2) };

        print!("{}", tmpstr.on(col1).with(col2));
    }
    fn get_state<T: LogicSim>(&self, board: &VcbBoard<T>) -> bool {
        self.id
            .map(|t| {
                board.nodes[t]
                    .network_id
                    .map(|i| board.logic_sim.get_state(i))
                    .unwrap_or_default()
            })
            .unwrap_or_default()
    }
    fn get_color<T: LogicSim>(&self, board: &VcbBoard<T>) -> [u8; 4] {
        self.trace.get_color::<T>(self.get_state(board))
    }
}


mod explore {
    use super::*;
    fn add_connection(nodes: &mut [BoardNode], connection: (usize, usize), swp_dir: bool) {
        let (start, end) = if swp_dir {
            (connection.1, connection.0)
        } else {
            connection
        };
        assert!(start != end);
        let a = nodes[start].inputs.insert(end);
        let b = nodes[end].outputs.insert(start);
        assert!(a == b);
        assert_ne!(
            nodes[start].kind == GateType::Cluster,
            nodes[end].kind == GateType::Cluster
        );
    }

    // Connect gate with immediate neighbor
    // pre: this is gate, this_x valid, this has id.
    fn connect_id(
        elements: &mut Vec<BoardElement>,
        nodes: &mut Vec<BoardNode>,
        width: i32,
        this_x: i32,
        this_id: usize,
    ) {
        let this = &mut elements[TryInto::<usize>::try_into(this_x).unwrap()];
        let this_kind = this.trace;
        assert!(this_kind.is_gate());
        assert!(this.id.is_some());

        'side: for dx in [1, -1, width, -width] {
            let other_x = this_x + dx;
            if this_x % width != other_x % width && this_x / width != other_x / width {
                continue 'side; // prevent wrapping connections
            }
            let other = unwrap_or_else!(elements.get(other_x as usize), continue 'side);
            let dir = match other.trace {
                Trace::Read => false,
                Trace::Write => true,
                _ => continue,
            };
            let other_id =
                unwrap_or_else!(other.id, { add_new_id(elements, nodes, width, other_x) });
            add_connection(nodes, (this_id, other_id), dir);
        }
    }

    // pre: this trace is logic
    #[must_use]
    fn add_new_id(
        elements: &mut Vec<BoardElement>,
        nodes: &mut Vec<BoardNode>,
        width: i32,
        this_x: i32,
    ) -> usize {
        let this = &elements[TryInto::<usize>::try_into(this_x).unwrap()];
        assert!(this.trace.is_logic());

        // create a new id.
        let this_id = nodes.len();
        nodes.push(BoardNode::new(this.trace));

        // fill with this_id
        fill_id(elements, width, this_x, this_id);
        this_id
    }

    /// floodfill with `id` at `this_x`
    /// pre: logic trace, otherwise id could have been
    /// assigned to nothing, `this_x` valid
    fn fill_id(elements: &mut Vec<BoardElement>, width: i32, this_x: i32, id: usize) {
        let this = &mut elements[this_x as usize];
        let this_kind = this.trace;
        assert!(this_kind.is_logic());
        match this.id {
            None => this.id = Some(id),
            Some(this_id) => {
                assert_eq!(this_id, id);
                return;
            },
        }
        'side: for dx in [1, -1, width, -width] {
            'forward: for ddx in [1, 2] {
                let other_x = this_x + dx * ddx;
                if this_x % width != other_x % width && this_x / width != other_x / width {
                    continue 'side; // prevent wrapping connections
                }
                let other_kind =
                    unwrap_or_else!(elements.get(other_x as usize), continue 'side).trace;
                if other_kind == Trace::Cross {
                    continue 'forward;
                }
                if other_kind.is_same_as(this_kind) {
                    fill_id(elements, width, other_x, id);
                }
                continue 'side;
            }
        }
    }

    // this HAS to be recursive since gates have arbitrary shapes.
    #[inline]
    pub(super) fn explore(
        elements: &mut Vec<BoardElement>,
        nodes: &mut Vec<BoardNode>,
        width: i32,
        this_x: i32,
    ) {
        // if prev_id is Some, merge should be done.
        // don't merge if id already exists, since that wouldn't make sense

        let this = &elements[TryInto::<usize>::try_into(this_x).unwrap()];
        let this_kind = this.trace;

        if !this_kind.is_logic() {
            return;
        }

        let this_id = unwrap_or_else!(this.id, { add_new_id(elements, nodes, width, this_x) });

        assert!(this_kind.is_logic());
        if !this_kind.is_gate() {
            return;
        }

        connect_id(elements, nodes, width, this_x, this_id);
    }
}
