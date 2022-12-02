#![allow(clippy::cast_sign_loss)]

use crate::logic::{CompiledNetwork, GateNetwork, GateType};
use crossterm::style::{Color, Colors, Print, ResetColor, SetColors, Stylize};
use crossterm::QueueableCommand;
use std::collections::BTreeSet;
use std::io::{stdout, Write};
use std::thread::sleep;
use std::time::Duration;

pub enum VcbParseInput {
    VcbBlueprint(String),
    VcbWorld(String),
}
#[derive(Default)]
pub struct VcbParser<const STRATEGY: u8> {}
impl<const STRATEGY: u8> VcbParser<STRATEGY> {
    #[must_use]
    fn make_plain_board_from_blueprint(data: &str) -> anyhow::Result<VcbPlainBoard> {
        let bytes = base64::decode_config(data.trim(), base64::STANDARD)?;
        let data_bytes = &bytes[..bytes.len() - BlueprintFooter::SIZE];
        let footer_bytes: [u8; BlueprintFooter::SIZE] =
            bytes[bytes.len() - BlueprintFooter::SIZE..bytes.len()].try_into()?;
        let footer = BlueprintFooter::from_bytes(footer_bytes);
        assert!(footer.layer == Layer::Logic);
        let data = zstd::bulk::decompress(data_bytes, 1 << 27)?;
        assert!(!data.is_empty());
        assert!(data.len() == footer.count * 4);
        Ok(VcbPlainBoard::from_color_data(
            &data,
            footer.width,
            footer.height,
        ))
    }
    #[must_use]
    fn make_plain_board_from_world(s: &str) -> anyhow::Result<VcbPlainBoard> {
        // Godot uses a custom format, tscn, which cannot be parsed with a json formatter
        let maybe_json = s.split("data = ").skip(1).next().unwrap();
        let s = maybe_json.split("\"layers\": [").skip(1).next().unwrap();
        let s = s.split(']').next().unwrap();
        let mut s = s
            .split("PoolByteArray( ")
            .skip(1)
            .map(|x| x.split(')').next().unwrap())
            .map(|x| {
                x.split(',')
                    .map(|x| str::parse::<u8>(x.trim()))
                    .map(|x| x.unwrap())
                    .collect::<Vec<_>>()
            });
        let bytes = s.next().unwrap();
        let data_bytes = &bytes[..bytes.len() - BoardFooter::SIZE];
        let footer_bytes: [u8; BoardFooter::SIZE] =
            bytes[bytes.len() - BoardFooter::SIZE..bytes.len()].try_into()?;
        let footer = dbg!(BoardFooter::from_bytes(footer_bytes));
        let data = zstd::bulk::decompress(&data_bytes, 1 << 27)?;

        assert_eq!(footer.width, 2048);
        assert_eq!(footer.height, 2048);

        Ok(VcbPlainBoard::from_color_data(
            &data,
            footer.width,
            footer.height,
        ))
    }
    #[must_use]
    fn make_board_from_blueprint(data: &str, optimize: bool) -> anyhow::Result<VcbBoard<STRATEGY>> {
        Ok(VcbBoard::new(
            Self::make_plain_board_from_blueprint(data)?,
            optimize,
        ))
    }
    #[must_use]
    pub fn parse(input: VcbParseInput, optimize: bool) -> anyhow::Result<VcbBoard<STRATEGY>> {
        let plain_board = match input {
            VcbParseInput::VcbBlueprint(b) => Self::make_plain_board_from_blueprint(&b)?,
            VcbParseInput::VcbWorld(w) => Self::make_plain_board_from_world(&w)?,
        };
        {};
        Ok(VcbBoard::new(plain_board, optimize))
    }
    /// # Panics
    /// invalid base64 string, invalid zstd, invalid colors
    #[must_use]
    pub fn parse_to_board(data: &str, optimize: bool) -> VcbBoard<STRATEGY> {
        Self::make_board_from_blueprint(data, optimize).unwrap()
    }
    #[must_use]
    pub fn try_parse_to_board(data: &str, optimize: bool) -> anyhow::Result<VcbBoard<STRATEGY>> {
        Self::make_board_from_blueprint(data, optimize)
    }
}

/// contains the raw footer data for worlds
#[derive(Debug, Default)]
#[repr(C)]
struct BoardFooter {
    height_type: i32,
    height: i32,
    width_type: i32,
    width: i32,
    bytes_type: i32,
    bytes: i32,
}
impl BoardFooter {
    const SIZE: usize = std::mem::size_of::<Self>();
    #[must_use]
    fn from_bytes(bytes: [u8; Self::SIZE]) -> BoardFooterInfo {
        let read_int = |i: usize| i32::from_le_bytes([0, 1, 2, 3].map(|k| bytes[k + (i * 4)]));

        #[rustfmt::skip]
        let footer = BoardFooterInfo::new(&(Self {
            height_type: read_int(0),
            height:      read_int(1),
            width_type:  read_int(2),
            width:       read_int(3),
            bytes_type:  read_int(4),
            bytes:       read_int(5),
        }));
        footer
    }
}

/// Useable cleaned footer data.
#[derive(Debug)]
struct BoardFooterInfo {
    width: usize,
    height: usize,
    count: usize,
}

impl BoardFooterInfo {
    #[must_use]
    fn new(footer: &BoardFooter) -> Self {
        assert_eq!(footer.height_type, 2);
        assert_eq!(footer.width_type, 2);
        assert_eq!(footer.bytes_type, 2);
        assert_eq!(footer.bytes, footer.height * footer.width * 4);
        Self {
            width: footer.width.try_into().unwrap(),
            height: footer.height.try_into().unwrap(),
            count: (footer.width * footer.height).try_into().unwrap(),
        }
    }
}
/// contains the raw footer data for blueprints
#[derive(Debug, Default)]
#[repr(C)]
struct BlueprintFooter {
    height_type: i32,
    height: i32,
    width_type: i32,
    width: i32,
    bytes_type: i32,
    bytes: i32,
    layer_type: i32,
    layer: i32,
}
impl BlueprintFooter {
    const SIZE: usize = std::mem::size_of::<Self>();
    #[must_use]
    fn from_bytes(bytes: [u8; Self::SIZE]) -> BlueprintFooterInfo {
        let read_int = |i: usize| i32::from_le_bytes([0, 1, 2, 3].map(|k| bytes[k + (i * 4)]));

        #[rustfmt::skip]
        let footer = BlueprintFooterInfo::new(&(Self {
            height_type: read_int(0),
            height:      read_int(1),
            width_type:  read_int(2),
            width:       read_int(3),
            bytes_type:  read_int(4),
            bytes:       read_int(5),
            layer_type:  read_int(6),
            layer:       read_int(7),
        }));
        footer
    }
}

/// Useable cleaned footer data.
#[derive(Debug)]
struct BlueprintFooterInfo {
    width: usize,
    height: usize,
    count: usize,
    layer: Layer,
}
impl BlueprintFooterInfo {
    #[must_use]
    fn new(footer: &BlueprintFooter) -> Self {
        assert_eq!(footer.height_type, 2);
        assert_eq!(footer.width_type, 2);
        assert_eq!(footer.bytes_type, 2);
        assert_eq!(footer.bytes, footer.height * footer.width * 4);
        BlueprintFooterInfo {
            width: footer.width.try_into().unwrap(),
            height: footer.height.try_into().unwrap(),
            count: (footer.width * footer.height).try_into().unwrap(),
            layer: match footer.layer {
                65_536 => Layer::Logic,
                131_072 => Layer::On,
                262_144 => Layer::Off,
                _ => panic!(),
            },
        }
    }
}
#[derive(Debug, PartialEq, Eq)]
enum Layer {
    Logic,
    On,
    Off,
}

/// Decoded blueprint or board
struct VcbPlainBoard {
    traces: Vec<Trace>,
    width: usize,
    height: usize,
}
impl VcbPlainBoard {
    #[must_use]
    fn from_color_data(data: &[u8], width: usize, height: usize) -> Self {
        let traces: Vec<_> = data
            .chunks_exact(4)
            .map(|x| Trace::from_raw_color(x.try_into().unwrap()))
            .collect();
        assert_eq!(traces.len(), width * height);
        VcbPlainBoard {
            traces,
            width,
            height,
        }
    }
    #[must_use]
    fn as_color_data(&self) -> Vec<u8> {
        self.traces
            .iter()
            .map(|x| x.to_color_raw())
            .flatten()
            .collect()
    }
}

/// Represents one gate or trace
#[derive(Debug)]
struct BoardNode {
    inputs: BTreeSet<usize>,
    outputs: BTreeSet<usize>,
    kind: GateType,
    network_id: Option<usize>,
}
impl BoardNode {
    #[must_use]
    fn new(trace: Trace) -> Self {
        BoardNode {
            inputs: BTreeSet::new(),
            outputs: BTreeSet::new(),
            kind: trace.to_gate(),
            network_id: None,
        }
    }
}

#[derive(Debug)]
pub struct VcbBoard<const STRATEGY: u8> {
    elements: Vec<BoardElement>,
    nodes: Vec<BoardNode>,
    pub(crate) network: GateNetwork<STRATEGY>,
    pub(crate) compiled_network: CompiledNetwork<STRATEGY>,
    pub width: usize,
    pub height: usize,
}

impl<const STRATEGY: u8> VcbBoard<STRATEGY> {
    /// For regression testing
    #[must_use]
    pub fn make_state_vec(&self) -> Vec<bool> {
        let mut a = Vec::new();
        for i in 0..self.elements.len() {
            a.push(match self.elements[i].id {
                None => false,
                Some(node_id) => match self.nodes[node_id].network_id {
                    None => false,
                    Some(id) => self.compiled_network.get_state(id),
                },
            });
        }
        a
    }
    #[must_use]
    pub(crate) fn make_inner_state_vec(&self) -> Vec<bool> {
        self.compiled_network.get_state_vec()
    }
    #[inline(always)]
    pub fn update(&mut self) {
        self.compiled_network.update();
    }
    fn new(plain_board: VcbPlainBoard, optimize: bool) -> Self {
        let height = plain_board.height;
        let width = plain_board.width;

        let num_elements = width * height;

        let elements: Vec<_> = plain_board
            .traces
            .into_iter()
            .map(BoardElement::from_trace)
            .collect();

        let mut board = VcbBoard {
            elements,
            nodes: Vec::new(),
            width,
            height,
            network: GateNetwork::default(),
            compiled_network: CompiledNetwork::default(),
        };
        let mut counter = 0;
        for x in 0..num_elements {
            board.explore(x.try_into().unwrap(), &mut counter);
        }
        // add vertexes to network
        for node in &mut board.nodes {
            node.network_id = Some(board.network.add_vertex(node.kind));
        }
        // add edges to network
        for i in 0..board.nodes.len() {
            let node = &board.nodes[i];
            for input in &node.inputs {
                assert!(board.nodes[*input].outputs.contains(&i));
            }
            let mut inputs: Vec<usize> = node
                .inputs
                .clone()
                .into_iter()
                .map(|x| board.nodes[x].network_id.unwrap())
                .collect();
            inputs.sort_unstable();
            inputs.dedup();
            board
                .network
                .add_inputs(node.kind, node.network_id.unwrap(), inputs);
        }
        board.compiled_network = board.network.compiled(optimize);

        board
    }
    fn add_connection(nodes: &mut Vec<BoardNode>, connection: (usize, usize), swp_dir: bool) {
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

    /// floodfill with id at given index
    /// pre: logic trace, otherwise id could have been
    /// assigned to nothing, this_x valid
    fn fill_id(
        nodes: &mut Vec<BoardNode>,
        elements: &mut Vec<BoardElement>,
        width: i32,
        this_x: i32,
        id: usize,
    ) {
        let this = &mut elements[this_x as usize];
        let this_kind = this.kind;
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
                // TODO: handle wrapping
                let other_x = this_x + dx * ddx;
                let other_kind =
                    unwrap_or_else!(elements.get(other_x as usize), continue 'side).kind;
                if other_kind == Trace::Cross {
                    continue 'forward;
                }
                if other_kind.is_same_as(this_kind) {
                    Self::fill_id(nodes, elements, width, other_x, id);
                }
                continue 'side;
            }
        }
    }
    // Connect gate with immediate neighbor
    // pre: this is gate, this_x valid, this has id.
    fn connect_id(
        elements: &mut Vec<BoardElement>,
        nodes: &mut Vec<BoardNode>,
        width: i32,
        this_x: i32,
        this_id: usize,
        id_counter: &mut usize,
    ) {
        let this = &mut elements[TryInto::<usize>::try_into(this_x).unwrap()];
        let this_kind = this.kind;
        assert!(this_kind.is_gate());
        assert!(this.id.is_some());

        //let width: i32 = self.width.try_into().unwrap();
        'side: for dx in [1, -1, width, -width] {
            //TODO: handle wrapping
            let other_x = this_x + dx;
            let other = unwrap_or_else!(elements.get(other_x as usize), continue 'side);

            let dir = match other.kind {
                Trace::Read => false,
                Trace::Write => true,
                _ => continue,
            };
            let other_id = unwrap_or_else!(other.id, {
                Self::add_new_id(elements, nodes, width, other_x, id_counter)
            });
            Self::add_connection(nodes, (this_id, other_id), dir);
        }
    }

    // pre: this trace is logic
    #[must_use]
    fn add_new_id(
        elements: &mut Vec<BoardElement>,
        nodes: &mut Vec<BoardNode>,
        width: i32,
        this_x: i32,
        id_counter: &mut usize,
    ) -> usize {
        let this = &elements[TryInto::<usize>::try_into(this_x).unwrap()];
        assert!(this.kind.is_logic());

        // TODO: omit the id_counter

        // create a new id.
        nodes.push(BoardNode::new(this.kind));
        let this_id = *id_counter;
        *id_counter += 1;

        // fill with this_id
        Self::fill_id(nodes, elements, width, this_x, this_id);
        this_id
    }

    // this HAS to be recursive since gates have arbitrary shapes.
    fn explore(&mut self, this_x: i32, id_counter: &mut usize) {
        // if prev_id is Some, merge should be done.
        // don't merge if id already exists, since that wouldn't make sense

        // all here is correct
        let this = &self.elements[TryInto::<usize>::try_into(this_x).unwrap()];
        let this_kind = this.kind;

        if !this_kind.is_logic() {
            return;
        }

        let this_id = unwrap_or_else!(this.id, {
            Self::add_new_id(
                &mut self.elements,
                &mut self.nodes,
                self.width as i32,
                this_x,
                id_counter,
            )
        });

        assert!(this_kind.is_logic());
        if !this_kind.is_gate() {
            return;
        }

        Self::connect_id(
            &mut self.elements,
            &mut self.nodes,
            self.width as i32,
            this_x,
            this_id,
            id_counter,
        );
    }

    pub fn print_marked(&self, marked: &[usize]) {
        println!("\nBoard:");
        for y in 0..self.height {
            for x in 0..self.width {
                let i = x + y * self.width;
                self.elements[i].print(self, i, marked.contains(&i), true);
            }
            println!();
        }
    }
    fn print_inner(&self, debug: bool) {
        println!("\nBoard:");
        for y in 0..self.height {
            for x in 0..self.width {
                let i = x + y * self.width;
                self.elements[i].print(self, i, false, debug);
            }
            println!();
        }
    }
    fn print_compact(&self) -> Result<(), std::io::Error> {
        let mut stdout = stdout();
        let (sx, sy) = crossterm::terminal::size().unwrap();
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
                    .map(|s| s.get_color(self))
                    .unwrap_or(Trace::Empty.to_color_off());

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
                if !matches!(self.elements[i].kind, Trace::Empty) {
                    min_print_width = x
                }
            }
            min_print_width += 1;
            for x in 0..min_print_width {
                let i = x + y * self.width;
                let trace = self.elements[i].kind;
                let printed_str = fun(trace);
                print!("{}", printed_str);
            }
            println!();
        }
        if legend {
            self.print_legend(|x| fun(x).to_string(), |x| format!("{x:?}"));
        }
    }
    pub fn print_regular_emoji(&self, legend: bool) {
        self.print_using_translation(|t| t.as_regular_emoji(), legend);
    }
    pub fn print_vcb_discord_emoji(&self, legend: bool) {
        self.print_using_translation(|t| t.as_discord_emoji(), legend);
    }
    pub fn print_debug(&self) {
        self.print_inner(true);
    }
    pub fn print(&self) {
        self.print_compact().unwrap();
    }
    fn print_legend<F: Fn(Trace) -> String, U: Fn(Trace) -> String>(&self, f1: F, f2: U) {
        for t in self.get_current_traces() {
            println!("{} = {}", f1(t), f2(t))
        }
    }
    fn get_current_traces(&self) -> Vec<Trace> {
        self.elements
            .iter()
            .map(|e| e.kind)
            .fold(std::collections::HashSet::new(), |mut set, trace| {
                set.insert(trace);
                set
            })
            .into_iter()
            .collect()
    }
    pub fn print_to_clipboard(&self) -> ! {
        use arboard::*;
        use std::borrow::Cow;

        let color_data: Vec<u8> = self
            .elements
            .iter()
            .map(|x| x.kind.to_color_on())
            .flatten()
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
            sleep(Duration::from_secs(1))
        }
    }
    pub fn print_to_gif(&mut self) {
        use std::collections::HashMap;
        type BoardColorData = Vec<u8>;
        let mut v: Vec<BoardColorData> = Vec::new();
        let mut i = 0;
        let mut map: HashMap<BoardColorData, usize> = HashMap::new();
        let limit = 500;

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
        use image::codecs::gif::*;
        use image::*;
        use tempfile::*;
        let file: NamedTempFile = Builder::new().suffix(".gif").tempfile().unwrap();
        let (file, path) = file.keep().unwrap();

        let image_width = 500_u32;
        let image_height = (image_width as f64 / self.width as f64 * self.height as f64) as u32;

        let frames = v
            .into_iter()
            .enumerate()
            .map(|(iframe, x)| {
                println!("{iframe}/{i}");
                let rgba: RgbaImage =
                    ImageBuffer::from_raw(self.width as u32, self.height as u32, x).unwrap();
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

/// All color constants used by vcb
#[rustfmt::skip]
mod vcb_colors {
    //                                               r,   g,   b,   w
    pub(crate) const COLOR_GRAY:       [u8; 4] = [  42,  53,  65, 255 ];
    pub(crate) const COLOR_WHITE:      [u8; 4] = [ 159, 168, 174, 255 ];
    pub(crate) const COLOR_RED:        [u8; 4] = [ 161,  85,  94, 255 ];
    pub(crate) const COLOR_ORANGE1:    [u8; 4] = [ 161, 108,  86, 255 ];
    pub(crate) const COLOR_ORANGE2:    [u8; 4] = [ 161, 133,  86, 255 ];
    pub(crate) const COLOR_ORANGE3:    [u8; 4] = [ 161, 152,  86, 255 ];
    pub(crate) const COLOR_YELLOW:     [u8; 4] = [ 153, 161,  86, 255 ];
    pub(crate) const COLOR_GREEN1:     [u8; 4] = [ 136, 161,  86, 255 ];
    pub(crate) const COLOR_GREEN2:     [u8; 4] = [ 108, 161,  86, 255 ];
    pub(crate) const COLOR_CYAN1:      [u8; 4] = [  86, 161, 141, 255 ];
    pub(crate) const COLOR_CYAN2:      [u8; 4] = [  86, 147, 161, 255 ];
    pub(crate) const COLOR_BLUE1:      [u8; 4] = [  86, 123, 161, 255 ];
    pub(crate) const COLOR_BLUE2:      [u8; 4] = [  86,  98, 161, 255 ];
    pub(crate) const COLOR_PURPLE:     [u8; 4] = [ 102,  86, 161, 255 ];
    pub(crate) const COLOR_MAGENTA:    [u8; 4] = [ 135,  86, 161, 255 ];
    pub(crate) const COLOR_PINK:       [u8; 4] = [ 161,  85, 151, 255 ];
    pub(crate) const COLOR_WRITE:      [u8; 4] = [  77,  56,  62, 255 ];
    pub(crate) const COLOR_EMPTY:      [u8; 4] = [   0,   0,   0,   0 ];
    pub(crate) const COLOR_CROSS:      [u8; 4] = [ 102, 120, 142, 255 ];
    pub(crate) const COLOR_READ:       [u8; 4] = [  46,  71,  93, 255 ];
    pub(crate) const COLOR_BUFFER:     [u8; 4] = [ 146, 255,  99, 255 ];
    pub(crate) const COLOR_AND:        [u8; 4] = [ 255, 198,  99, 255 ];
    pub(crate) const COLOR_OR:         [u8; 4] = [  99, 242, 255, 255 ];
    pub(crate) const COLOR_XOR:        [u8; 4] = [ 174, 116, 255, 255 ];
    pub(crate) const COLOR_NOT:        [u8; 4] = [ 255,  98, 138, 255 ];
    pub(crate) const COLOR_NAND:       [u8; 4] = [ 255, 162,   0, 255 ];
    pub(crate) const COLOR_NOR:        [u8; 4] = [  48, 217, 255, 255 ];
    pub(crate) const COLOR_XNOR:       [u8; 4] = [ 166,   0, 255, 255 ];
    pub(crate) const COLOR_LATCHON:    [u8; 4] = [  99, 255, 159, 255 ];
    pub(crate) const COLOR_LATCHOFF:   [u8; 4] = [  56,  77,  71, 255 ];
    pub(crate) const COLOR_CLOCK:      [u8; 4] = [ 255,   0,  65, 255 ];
    pub(crate) const COLOR_LED:        [u8; 4] = [ 255, 255, 255, 255 ];
    pub(crate) const COLOR_ANNOTATION: [u8; 4] = [  58,  69,  81, 255 ];
    pub(crate) const COLOR_FILLER:     [u8; 4] = [ 140, 171, 161, 255 ];
}

#[non_exhaustive]
#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash)]
enum Trace {
    Gray,
    White,
    Red,
    Orange1,
    Orange2,
    Orange3,
    Yellow,
    Green1,
    Green2,
    Cyan1,
    Cyan2,
    Blue1,
    Blue2,
    Purple,
    Magenta,
    Pink,
    Write,
    Empty,
    Cross,
    Read,
    Buffer,
    And,
    Or,
    Xor,
    Not,
    Nand,
    Nor,
    Xnor,
    LatchOn,
    LatchOff,
    Clock,
    Led,
    Annotation,
    Filler,
}
impl Trace {
    #[rustfmt::skip]
    fn to_color_raw(self) -> [u8; 4] {
        match self {
            Trace::Gray       => vcb_colors::COLOR_GRAY,
            Trace::White      => vcb_colors::COLOR_WHITE,
            Trace::Red        => vcb_colors::COLOR_RED,
            Trace::Orange1    => vcb_colors::COLOR_ORANGE1,
            Trace::Orange2    => vcb_colors::COLOR_ORANGE2,
            Trace::Orange3    => vcb_colors::COLOR_ORANGE3,
            Trace::Yellow     => vcb_colors::COLOR_YELLOW,
            Trace::Green1     => vcb_colors::COLOR_GREEN1,
            Trace::Green2     => vcb_colors::COLOR_GREEN2,
            Trace::Cyan1      => vcb_colors::COLOR_CYAN1,
            Trace::Cyan2      => vcb_colors::COLOR_CYAN2,
            Trace::Blue1      => vcb_colors::COLOR_BLUE1,
            Trace::Blue2      => vcb_colors::COLOR_BLUE2,
            Trace::Purple     => vcb_colors::COLOR_PURPLE,
            Trace::Magenta    => vcb_colors::COLOR_MAGENTA,
            Trace::Pink       => vcb_colors::COLOR_PINK,
            Trace::Write      => vcb_colors::COLOR_WRITE,
            Trace::Empty      => vcb_colors::COLOR_EMPTY,
            Trace::Cross      => vcb_colors::COLOR_CROSS,
            Trace::Read       => vcb_colors::COLOR_READ,
            Trace::Buffer     => vcb_colors::COLOR_BUFFER,
            Trace::And        => vcb_colors::COLOR_AND,
            Trace::Or         => vcb_colors::COLOR_OR,
            Trace::Xor        => vcb_colors::COLOR_XOR,
            Trace::Not        => vcb_colors::COLOR_NOT,
            Trace::Nand       => vcb_colors::COLOR_NAND,
            Trace::Nor        => vcb_colors::COLOR_NOR,
            Trace::Xnor       => vcb_colors::COLOR_XNOR,
            Trace::LatchOn    => vcb_colors::COLOR_LATCHON,
            Trace::LatchOff   => vcb_colors::COLOR_LATCHOFF,
            Trace::Clock      => vcb_colors::COLOR_CLOCK,
            Trace::Led        => vcb_colors::COLOR_LED,
            Trace::Annotation => vcb_colors::COLOR_ANNOTATION,
            Trace::Filler     => vcb_colors::COLOR_FILLER,
        }
    }
    fn to_color_on(self) -> [u8; 4] {
        match self {
            Trace::LatchOff => Trace::LatchOn,
            _ => self,
        }
        .to_color_raw()
    }
    fn to_color_off(self) -> [u8; 4] {
        match self {
            Trace::LatchOn => Trace::LatchOff.to_color_raw(),
            _ => {
                if self.is_passive() {
                    self.to_color_on()
                } else {
                    let rgb = self.to_color_raw();
                    let brfac = 60;
                    [
                        ((u32::from(rgb[0]) * brfac) / 255).try_into().unwrap(),
                        ((u32::from(rgb[1]) * brfac) / 255).try_into().unwrap(),
                        ((u32::from(rgb[2]) * brfac) / 255).try_into().unwrap(),
                        rgb[3],
                    ]
                }
            },
        }
    }
    // colors from file format
    #[rustfmt::skip]
    fn from_raw_color(color: [u8; 4]) -> Self {
        let color: [u8; 4] = color.try_into().unwrap();
        match color {
            vcb_colors::COLOR_GRAY       => Trace::Gray,
            vcb_colors::COLOR_WHITE      => Trace::White,
            vcb_colors::COLOR_RED        => Trace::Red,
            vcb_colors::COLOR_ORANGE1    => Trace::Orange1,
            vcb_colors::COLOR_ORANGE2    => Trace::Orange2,
            vcb_colors::COLOR_ORANGE3    => Trace::Orange3,
            vcb_colors::COLOR_YELLOW     => Trace::Yellow,
            vcb_colors::COLOR_GREEN1     => Trace::Green1,
            vcb_colors::COLOR_GREEN2     => Trace::Green2,
            vcb_colors::COLOR_CYAN1      => Trace::Cyan1,
            vcb_colors::COLOR_CYAN2      => Trace::Cyan2,
            vcb_colors::COLOR_BLUE1      => Trace::Blue1,
            vcb_colors::COLOR_BLUE2      => Trace::Blue2,
            vcb_colors::COLOR_PURPLE     => Trace::Purple,
            vcb_colors::COLOR_MAGENTA    => Trace::Magenta,
            vcb_colors::COLOR_PINK       => Trace::Pink,
            vcb_colors::COLOR_WRITE      => Trace::Write,
            vcb_colors::COLOR_EMPTY      => Trace::Empty,
            vcb_colors::COLOR_CROSS      => Trace::Cross,
            vcb_colors::COLOR_READ       => Trace::Read,
            vcb_colors::COLOR_BUFFER     => Trace::Buffer,
            vcb_colors::COLOR_AND        => Trace::And,
            vcb_colors::COLOR_OR         => Trace::Or,
            vcb_colors::COLOR_XOR        => Trace::Xor,
            vcb_colors::COLOR_NOT        => Trace::Not,
            vcb_colors::COLOR_NAND       => Trace::Nand,
            vcb_colors::COLOR_NOR        => Trace::Nor,
            vcb_colors::COLOR_XNOR       => Trace::Xnor,
            vcb_colors::COLOR_LATCHON    => Trace::LatchOn,
            vcb_colors::COLOR_LATCHOFF   => Trace::LatchOff,
            vcb_colors::COLOR_CLOCK      => Trace::Clock,
            vcb_colors::COLOR_LED        => Trace::Led,
            vcb_colors::COLOR_ANNOTATION => Trace::Annotation,
            vcb_colors::COLOR_FILLER     => Trace::Filler,
            _ => panic!("Invalid trace color"),
        }
    }
    fn is_wire(self) -> bool {
        matches!(
            self,
            Trace::Gray
                | Trace::White
                | Trace::Red
                | Trace::Orange1
                | Trace::Orange2
                | Trace::Orange3
                | Trace::Yellow
                | Trace::Green1
                | Trace::Green2
                | Trace::Cyan1
                | Trace::Cyan2
                | Trace::Blue1
                | Trace::Blue2
                | Trace::Purple
                | Trace::Magenta
                | Trace::Pink
                | Trace::Read
                | Trace::Write
        )
    }
    fn is_gate(self) -> bool {
        matches!(
            self,
            Trace::Buffer
                | Trace::And
                | Trace::Or
                | Trace::Xor
                | Trace::Not
                | Trace::Nand
                | Trace::Nor
                | Trace::Xnor
                | Trace::LatchOn
                | Trace::LatchOff
                | Trace::Clock
                | Trace::Led
        )
    }
    fn is_logic(self) -> bool {
        self.is_wire() || self.is_gate()
    }
    fn is_passive(self) -> bool {
        !self.is_logic()
    }

    // is logically same as other, will connect
    fn is_same_as(self, other: Self) -> bool {
        (self == other) || (self.is_wire() && other.is_wire())
    }

    fn to_gate(self) -> GateType {
        if self.is_wire() {
            GateType::Cluster
        } else {
            match self {
                Trace::Buffer | Trace::Or | Trace::Led => GateType::Or,
                Trace::Not | Trace::Nor => GateType::Nor,
                Trace::And => GateType::And,
                Trace::Nand => GateType::Nand,
                Trace::Xor | Trace::LatchOff => GateType::Xor,
                Trace::Xnor | Trace::LatchOn => GateType::Xnor,
                _ => panic!("unsupported logic trace: {self:?}"),
            }
        }
    }
    fn as_regular_emoji(self) -> &'static str {
        match self {
            Trace::Gray => "🔘",
            Trace::White => "🔘",
            Trace::Red => "🔘",
            Trace::Orange1 => "🔘",
            Trace::Orange2 => "🔘",
            Trace::Orange3 => "🔘",
            Trace::Yellow => "🔘",
            Trace::Green1 => "🔘",
            Trace::Green2 => "🔘",
            Trace::Cyan1 => "🔘",
            Trace::Cyan2 => "🔘",
            Trace::Blue1 => "🔘",
            Trace::Blue2 => "🔘",
            Trace::Purple => "🔘",
            Trace::Magenta => "🔘",
            Trace::Pink => "🔘",
            Trace::Write => "✏",
            Trace::Empty => "⬛",
            Trace::Cross => "➕",
            Trace::Read => "👓",
            Trace::Buffer => "🟣",
            Trace::And => "🅰",
            Trace::Or => "🅾",
            Trace::Xor => "✖",
            Trace::Not => "❕",
            Trace::Nand => "🈲",
            Trace::Nor => "🈳",
            Trace::Xnor => "🔶",
            Trace::LatchOn => "🔺",
            Trace::LatchOff => "🔻",
            Trace::Clock => "🥞",
            Trace::Led => "🏁",
            Trace::Annotation => "🥚",
            Trace::Filler => "🌯",
        }
    }

    fn as_discord_emoji(self) -> &'static str {
        match self {
            Trace::Gray => ":t00:",
            Trace::White => ":t01:",
            Trace::Red => ":t02:",
            Trace::Orange1 => ":t03:",
            Trace::Orange2 => ":t04:",
            Trace::Orange3 => ":t05:",
            Trace::Yellow => ":t06:",
            Trace::Green1 => ":t07:",
            Trace::Green2 => ":t08:",
            Trace::Cyan1 => ":t09:",
            Trace::Cyan2 => ":t10:",
            Trace::Blue1 => ":t11:",
            Trace::Blue2 => ":t12:",
            Trace::Purple => ":t13:",
            Trace::Magenta => ":t14:",
            Trace::Pink => ":t15:",
            Trace::Write => ":wr:",
            Trace::Empty => ":pd:",
            Trace::Cross => ":crs:",
            Trace::Read => ":rd:",
            Trace::Buffer => ":bfr:",
            Trace::And => ":and:",
            Trace::Or => ":or:",
            Trace::Xor => ":xor:",
            Trace::Not => ":not:",
            Trace::Nand => ":ina:",
            Trace::Nor => ":nor:",
            Trace::Xnor => ":xnr:",
            Trace::LatchOn => ":lt1:",
            Trace::LatchOff => ":lt0:",
            Trace::Clock => "CLOCK",
            Trace::Led => ":led:",
            Trace::Annotation => ":non:",
            Trace::Filler => ":fil:",
        }
    }
}

/// Represents one pixel.
/// It is probably a mistake to
/// make a copy of this type.
#[derive(Debug)]
struct BoardElement {
    /// Raw color from input file.
    color_on: [u8; 4],
    color_off: [u8; 4],
    kind: Trace,
    id: Option<usize>,
}
impl BoardElement {
    fn from_trace(trace: Trace) -> Self {
        BoardElement {
            color_on: trace.to_color_on(),
            color_off: trace.to_color_off(),
            kind: trace,
            id: None,
        }
    }
    fn new(color: [u8; 4]) -> Self {
        let trace = Trace::from_raw_color(color);
        BoardElement {
            color_on: trace.to_color_on(),
            color_off: trace.to_color_off(),
            kind: trace,
            id: None,
        }
    }
    fn print<const STRATEGY: u8>(
        &self,
        board: &VcbBoard<STRATEGY>,
        _i: usize,
        _marked: bool,
        debug: bool,
    ) {
        let format = |debug, id: usize| {
            if debug {
                format!("{:>2}", id)
            } else {
                format!("  ")
            }
        };

        let mut state = false;
        let tmpstr = if let Some(t) = self.id {
            state = board.nodes[t]
                .network_id
                .map(|i| board.compiled_network.get_state(i))
                .unwrap_or_default();
            format(debug, board.compiled_network.get_inner_id(t) % 100)
        } else {
            "  ".to_string()
        };
        let col = if state { self.color_on } else { self.color_off };
        let col1: Color = (col[0], col[1], col[2]).into();
        let col2: Color = (255 - col[0], 255 - col[1], 255 - col[2]).into();

        print!("{}", tmpstr.on(col1).with(col2));
    }
    fn get_state<const STRATEGY: u8>(&self, board: &VcbBoard<STRATEGY>) -> bool {
        self.id
            .map(|t| {
                board.nodes[t]
                    .network_id
                    .map(|i| board.compiled_network.get_state(i))
                    .unwrap_or_default()
            })
            .unwrap_or_default()
    }
    fn get_color<const STRATEGY: u8>(&self, board: &VcbBoard<STRATEGY>) -> [u8; 4] {
        if self.get_state(board) {
            self.color_on
        } else {
            self.color_off
        }
    }
}
