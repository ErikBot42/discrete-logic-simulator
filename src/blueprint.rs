#![allow(clippy::cast_sign_loss)]

use crate::logic::{CompiledNetwork, GateNetwork, GateType};
use anyhow::{anyhow, Context};
use crossterm::style::{Color, Colors, Print, ResetColor, SetColors, Stylize};
use crossterm::QueueableCommand;
use std::collections::BTreeSet;
use std::io::{stdout, Write};
use std::mem::size_of;
use std::thread::sleep;
use std::time::Duration;

const RGBA_SIZE: usize = 4;

fn zstd_decompress(data: &[u8], num_traces: usize) -> std::io::Result<Vec<u8>> {
    timed!(
        { zstd::bulk::decompress(data, num_traces * RGBA_SIZE) },
        "zstd decompress in: {:?}"
    )
}
fn base64_decode(data: &str) -> Result<Vec<u8>, base64::DecodeError> {
    timed!(
        base64::decode_config(data.trim(), base64::STANDARD),
        "base64 decode in: {:?}"
    )
}

#[derive(Clone)]
pub enum VcbInput {
    BlueprintLegacy(String),
    Blueprint(String),
    WorldLegacy(String),
    World(String),
}
#[derive(Default)]
pub struct VcbParser<const STRATEGY: u8> {}
impl<const STRATEGY: u8> VcbParser<STRATEGY> {
    fn parse_legacy_blueprint(data: &str) -> anyhow::Result<VcbPlainBoard> {
        let bytes = base64_decode(data)?;
        let data_bytes = &bytes
            .get(..bytes.len() - BlueprintFooter::SIZE)
            .context("")?;
        let footer_bytes: [u8; BlueprintFooter::SIZE] = bytes
            .get(bytes.len() - BlueprintFooter::SIZE..bytes.len())
            .context("")?
            .try_into()?;
        let footer = BlueprintFooter::from_bytes(footer_bytes);
        if footer.layer != Layer::Logic {
            return Err(anyhow!("Wrong blueprint layer"));
        };
        let data = zstd_decompress(data_bytes, footer.count)?;
        if data.len() != footer.count * 4 {
            return Err(anyhow!("Mismatch between footer count and data length"));
        }
        VcbPlainBoard::from_color_data(&data, footer.width, footer.height)
    }
    fn parse_blueprint(data: &str) -> anyhow::Result<VcbPlainBoard> {
        if data.get(0..4).context("data too short")? != "VCB+" {
            return Err(anyhow!("Wrong prefix"));
        }
        let bytes = base64_decode(data)?;
        let mut bytes_iter = bytes.iter();
        let header = BlueprintHeader::try_from_bytes(&mut bytes_iter)
            .context("Not enough bytes for header")?;
        loop {
            if bytes.len() == 0 {
                break Err(anyhow!("out of bytes in blueprint string"));
            }
            let block_header = BlueprintBlockHeader::try_from_bytes(&mut bytes_iter)
                .context("Not enough bytes for block header")?;
            let (color_bytes, bytes) = bytes_iter
                .as_slice()
                .split_at(block_header.block_size - BlueprintBlockHeader::SIZE);
            bytes_iter = bytes.iter();
            if block_header.layer == Layer::Logic {
                let color_data =
                    zstd_decompress(color_bytes, block_header.buffer_size_uncompressed)?;
                break VcbPlainBoard::from_color_data(&color_data, header.width, header.height);
            }
        }
    }
    fn parse_legacy_world(s: &str) -> anyhow::Result<VcbPlainBoard> {
        // Godot uses a custom format, tscn, which cannot be parsed with a json formatter
        let maybe_json = s.split("data = ").nth(1).context("")?;
        let s = maybe_json.split("\"layers\": [").nth(1).context("")?;
        let s = s.split(']').next().context("")?;
        let mut s = s
            .split("PoolByteArray( ")
            .skip(1)
            .map(|x| x.split(')').next())
            .map(|x| {
                x.map(|x| {
                    x.split(',')
                        .map(|x| str::parse::<u8>(x.trim()))
                        .collect::<Result<Vec<_>, _>>()
                })
            });
        let bytes = s.next().flatten().context("")??;
        let data_bytes = &bytes.get(..bytes.len() - BoardFooter::SIZE).context("")?;
        let footer_bytes: [u8; BoardFooter::SIZE] = bytes
            .get(bytes.len() - BoardFooter::SIZE..bytes.len())
            .context("")?
            .try_into()?;
        let footer = BoardFooter::from_bytes(footer_bytes);
        let data = zstd_decompress(data_bytes, footer.count)?;
        VcbPlainBoard::from_color_data(&data, footer.width, footer.height)
    }
    fn parse_world(s: &str) -> anyhow::Result<VcbPlainBoard> {
        let parsed = json::parse(s)?;
        let world_str: &json::JsonValue = &parsed["layers"][0];
        if let json::JsonValue::String(data) = world_str {
            let bytes = base64_decode(data)?;
            let (color_data, footer_bytes) = bytes.split_at(bytes.len() - BoardFooter::SIZE);
            let footer = BoardFooter::from_bytes(footer_bytes.try_into().context("")?);
            let data = zstd_decompress(color_data, footer.count)?;
            VcbPlainBoard::from_color_data(&data, footer.width, footer.height)
        } else {
            Err(anyhow!("json parsing went wrong"))
        }
    }

    fn make_board_from_legacy_blueprint(
        data: &str,
        optimize: bool,
    ) -> anyhow::Result<VcbBoard<STRATEGY>> {
        Ok(VcbBoard::new(
            Self::parse_legacy_blueprint(data)?,
            optimize,
        ))
    }
    #[must_use]
    /// # Result
    /// Returns Err if input is invalid or could not parse.
    pub fn parse(input: VcbInput, optimize: bool) -> anyhow::Result<VcbBoard<STRATEGY>> {
        let plain_board = match input {
            VcbInput::BlueprintLegacy(b) => Self::parse_legacy_blueprint(&b)?,
            VcbInput::Blueprint(b) => Self::parse_blueprint(&b)?,
            VcbInput::WorldLegacy(w) => Self::parse_legacy_world(&w)?,
            VcbInput::World(w) => Self::parse_world(&w)?,
        };
        {};
        Ok(VcbBoard::new(plain_board, optimize))
    }
    /// # Panics
    /// invalid base64 string, invalid zstd, invalid colors
    #[must_use]
    #[deprecated(note = "use parse instead, it is more general")]
    pub fn parse_to_board(data: &str, optimize: bool) -> VcbBoard<STRATEGY> {
        Self::make_board_from_legacy_blueprint(data, optimize).unwrap()
    }
    //#[deprecated(note = "use parse instead, it is more general")]
    //pub fn try_parse_to_board(data: &str, optimize: bool) -> anyhow::Result<VcbBoard<STRATEGY>> {
    //    Self::make_board_from_legacy_blueprint(data, optimize)
    //}
}

// New blueprints start with "VCB+".
// Bytes are in big-endian order.
//
// # Header
// 3-byte blueprint identifier (VCB+)
// 3-byte blueprint version
// 6-byte checksum (truncated SHA-1) of the remaining characters of the blueprint string
// 4-byte width
// 4-byte height
//
// 20 bytes???
#[derive(Debug, Default)]
#[repr(C)]
struct BlueprintHeader {
    identifier: [u8; 3],
    version: [u8; 3],
    checksum: [u8; 6],
    width: usize,
    height: usize,
}
impl BlueprintHeader {
    fn try_from_bytes<'a>(data: &mut impl Iterator<Item = &'a u8>) -> Option<Self> {
        let mut n = || data.next().map(|x| *x);
        let identifier = [n()?, n()?, n()?];
        let version = [n()?, n()?, n()?];
        let checksum = [n()?, n()?, n()?, n()?, n()?, n()?];
        let width = u32::from_be_bytes([n()?, n()?, n()?, n()?]) as usize;
        let height = u32::from_be_bytes([n()?, n()?, n()?, n()?]) as usize;
        Some(Self {
            identifier,
            version,
            checksum,
            width,
            height,
        })
    }
}

// # Layer Block(s)
// 4-byte block size
// 4-byte layer id (Logic = 0, Deco On = 1, Deco Off = 2)
// 4-byte uncompressed buffer size
// N-byte zstd-compressed RGBA8 buffer
//
#[derive(Debug)]
struct BlueprintBlockHeader {
    block_size: usize,
    layer: Layer,
    buffer_size_uncompressed: usize,
}
impl BlueprintBlockHeader {
    const SIZE: usize = size_of::<u32>() + size_of::<u32>() + size_of::<u32>();
    fn try_from_bytes<'a>(data: &mut impl Iterator<Item = &'a u8>) -> Option<Self> {
        let mut n = || data.next().map(|x| *x);
        let block_size: usize = u32::from_be_bytes([n()?, n()?, n()?, n()?])
            .try_into()
            .ok()?;
        let layer = match u32::from_be_bytes([n()?, n()?, n()?, n()?]) {
            0 => Some(Layer::Logic),
            1 => Some(Layer::On),
            2 => Some(Layer::Off),
            _ => None,
        }?;
        let buffer_size_uncompressed: usize = u32::from_be_bytes([n()?, n()?, n()?, n()?])
            .try_into()
            .ok()?;
        Some(Self {
            block_size,
            layer,
            buffer_size_uncompressed,
        })
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
        assert_eq!(
            footer.bytes,
            footer.height * footer.width * size_of::<u32>() as i32
        );
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
    /// Infalliable because of fixed size input
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
    fn from_color_data(data: &[u8], width: usize, height: usize) -> anyhow::Result<Self> {
        timed!(
            {
                let traces = data
                    .array_chunks::<4>()
                    .copied()
                    .map(Trace::from_raw_color)
                    .collect::<Option<Vec<_>>>()
                    .context("invalid color found")?;
                if traces.len() == width * height {
                    Ok(VcbPlainBoard {
                        traces,
                        width,
                        height,
                    })
                } else {
                    Err(anyhow!(
                        "Wrong trace len: len: {}, width: {width}, height: {height}",
                        traces.len()
                    ))
                }
            },
            "Create plain board in {:?}"
        )
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
    #[cfg(test)]
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

        let mut nodes = Vec::new();
        let elements = timed!(
            {
                let mut elements: Vec<_> = plain_board
                    .traces
                    .into_iter()
                    .map(BoardElement::new)
                    .collect();
                for x in 0..num_elements {
                    Self::explore(
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
                node.network_id = Some(network.add_vertex(node.kind));
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

        let compiled_network = timed! { network.compiled(optimize), "compiled network in: {:?}"};
        VcbBoard {
            elements,
            nodes,
            compiled_network,
            width,
            height,
        }
    }
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

    /// floodfill with `id` at `this_x`
    /// pre: logic trace, otherwise id could have been
    /// assigned to nothing, `this_x` valid
    fn fill_id(elements: &mut Vec<BoardElement>, width: i32, this_x: i32, id: usize) {
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
                    Self::fill_id(elements, width, other_x, id);
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
                Self::add_new_id(elements, nodes, width, other_x)
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
    ) -> usize {
        let this = &elements[TryInto::<usize>::try_into(this_x).unwrap()];
        assert!(this.kind.is_logic());

        // create a new id.
        let this_id = nodes.len();
        nodes.push(BoardNode::new(this.kind));

        // fill with this_id
        Self::fill_id(elements, width, this_x, this_id);
        this_id
    }

    // this HAS to be recursive since gates have arbitrary shapes.
    #[inline]
    fn explore(
        elements: &mut Vec<BoardElement>,
        nodes: &mut Vec<BoardNode>,
        width: i32,
        this_x: i32,
    ) {
        // if prev_id is Some, merge should be done.
        // don't merge if id already exists, since that wouldn't make sense

        // all here is correct
        let this = &elements[TryInto::<usize>::try_into(this_x).unwrap()];
        let this_kind = this.kind;

        if !this_kind.is_logic() {
            return;
        }

        let this_id = unwrap_or_else!(this.id, {
            Self::add_new_id(elements, nodes, width, this_x)
        });

        assert!(this_kind.is_logic());
        if !this_kind.is_gate() {
            return;
        }

        Self::connect_id(elements, nodes, width, this_x, this_id);
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
                if !matches!(self.elements[i].kind, Trace::Empty) {
                    min_print_width = x;
                }
            }
            min_print_width += 1;
            for x in 0..min_print_width {
                let i = x + y * self.width;
                let trace = self.elements[i].kind;
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
        self.print_inner(true);
    }
    pub fn print(&self) {
        self.print_compact().unwrap();
    }
    fn print_legend<F: Fn(Trace) -> String, U: Fn(Trace) -> String>(&self, f1: F, f2: U) {
        for t in self.get_current_traces() {
            println!("{} = {}", f1(t), f2(t));
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
        use arboard::{Clipboard, ImageData};
        use std::borrow::Cow;

        let color_data: Vec<u8> = self
            .elements
            .iter()
            .flat_map(|x| x.kind.to_color_on())
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
    pub fn print_to_gif(&mut self, limit: usize) {
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
    fn from_raw_color(color: [u8; 4]) -> Option<Self> {
        match color {
            vcb_colors::COLOR_GRAY       => Some(Trace::Gray),
            vcb_colors::COLOR_WHITE      => Some(Trace::White),
            vcb_colors::COLOR_RED        => Some(Trace::Red),
            vcb_colors::COLOR_ORANGE1    => Some(Trace::Orange1),
            vcb_colors::COLOR_ORANGE2    => Some(Trace::Orange2),
            vcb_colors::COLOR_ORANGE3    => Some(Trace::Orange3),
            vcb_colors::COLOR_YELLOW     => Some(Trace::Yellow),
            vcb_colors::COLOR_GREEN1     => Some(Trace::Green1),
            vcb_colors::COLOR_GREEN2     => Some(Trace::Green2),
            vcb_colors::COLOR_CYAN1      => Some(Trace::Cyan1),
            vcb_colors::COLOR_CYAN2      => Some(Trace::Cyan2),
            vcb_colors::COLOR_BLUE1      => Some(Trace::Blue1),
            vcb_colors::COLOR_BLUE2      => Some(Trace::Blue2),
            vcb_colors::COLOR_PURPLE     => Some(Trace::Purple),
            vcb_colors::COLOR_MAGENTA    => Some(Trace::Magenta),
            vcb_colors::COLOR_PINK       => Some(Trace::Pink),
            vcb_colors::COLOR_WRITE      => Some(Trace::Write),
            vcb_colors::COLOR_EMPTY      => Some(Trace::Empty),
            vcb_colors::COLOR_CROSS      => Some(Trace::Cross),
            vcb_colors::COLOR_READ       => Some(Trace::Read),
            vcb_colors::COLOR_BUFFER     => Some(Trace::Buffer),
            vcb_colors::COLOR_AND        => Some(Trace::And),
            vcb_colors::COLOR_OR         => Some(Trace::Or),
            vcb_colors::COLOR_XOR        => Some(Trace::Xor),
            vcb_colors::COLOR_NOT        => Some(Trace::Not),
            vcb_colors::COLOR_NAND       => Some(Trace::Nand),
            vcb_colors::COLOR_NOR        => Some(Trace::Nor),
            vcb_colors::COLOR_XNOR       => Some(Trace::Xnor),
            vcb_colors::COLOR_LATCHON    => Some(Trace::LatchOn),
            vcb_colors::COLOR_LATCHOFF   => Some(Trace::LatchOff),
            vcb_colors::COLOR_CLOCK      => Some(Trace::Clock),
            vcb_colors::COLOR_LED        => Some(Trace::Led),
            vcb_colors::COLOR_ANNOTATION => Some(Trace::Annotation),
            vcb_colors::COLOR_FILLER     => Some(Trace::Filler),
            _ => None,
        }
    }
    #[inline]
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
    #[inline]
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
    #[inline]
    fn is_logic(self) -> bool {
        self.is_wire() || self.is_gate()
    }
    #[inline]
    fn is_passive(self) -> bool {
        !self.is_logic()
    }

    // is logically same as other, will connect
    #[inline]
    fn is_same_as(self, other: Self) -> bool {
        (self == other) || (self.is_wire() && other.is_wire())
    }

    #[inline]
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
    fn new(trace: Trace) -> Self {
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
                format!("{id:>2}")
            } else {
                "  ".to_string()
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
