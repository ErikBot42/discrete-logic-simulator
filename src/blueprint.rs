// blueprint.rs: parsing VCB blueprints
#![allow(clippy::inline_always)]
//#![allow(clippy::upper_case_acronyms)]
#![allow(dead_code)]
#![allow(clippy::cast_sign_loss)]
use crate::logic::{CompiledNetwork, GateNetwork, GateType};
use colored::Colorize;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq, Eq)]
enum Layer {
    Logic,
    On,
    Off,
}
// useable cleaned footer data.
#[derive(Debug)]
struct FooterInfo {
    width: usize,
    height: usize,
    count: usize,
    layer: Layer,
}
impl FooterInfo {
    fn new(footer: &Footer) -> Self {
        FooterInfo {
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

#[non_exhaustive]
#[derive(Debug, PartialEq, Clone, Copy)]
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

struct ColorConstants {}
#[rustfmt::skip]
impl ColorConstants {
    //                                    r,   g,   b,   w
    const COLOR_GRAY:       [u8; 4] = [  42,  53,  65, 255 ];
    const COLOR_WHITE:      [u8; 4] = [ 159, 168, 174, 255 ];
    const COLOR_RED:        [u8; 4] = [ 161,  85,  94, 255 ];
    const COLOR_ORANGE1:    [u8; 4] = [ 161, 108,  86, 255 ];
    const COLOR_ORANGE2:    [u8; 4] = [ 161, 133,  86, 255 ];
    const COLOR_ORANGE3:    [u8; 4] = [ 161, 152,  86, 255 ];
    const COLOR_YELLOW:     [u8; 4] = [ 153, 161,  86, 255 ];
    const COLOR_GREEN1:     [u8; 4] = [ 136, 161,  86, 255 ];
    const COLOR_GREEN2:     [u8; 4] = [ 108, 161,  86, 255 ];
    const COLOR_CYAN1:      [u8; 4] = [  86, 161, 141, 255 ];
    const COLOR_CYAN2:      [u8; 4] = [  86, 147, 161, 255 ];
    const COLOR_BLUE1:      [u8; 4] = [  86, 123, 161, 255 ];
    const COLOR_BLUE2:      [u8; 4] = [  86,  98, 161, 255 ];
    const COLOR_PURPLE:     [u8; 4] = [ 102,  86, 161, 255 ];
    const COLOR_MAGENTA:    [u8; 4] = [ 135,  86, 161, 255 ];
    const COLOR_PINK:       [u8; 4] = [ 161,  85, 151, 255 ];
    const COLOR_WRITE:      [u8; 4] = [  77,  56,  62, 255 ];
    const COLOR_EMPTY:      [u8; 4] = [   0,   0,   0,   0 ];
    const COLOR_CROSS:      [u8; 4] = [ 102, 120, 142, 255 ];
    const COLOR_READ:       [u8; 4] = [  46,  71,  93, 255 ];
    const COLOR_BUFFER:     [u8; 4] = [ 146, 255,  99, 255 ];
    const COLOR_AND:        [u8; 4] = [ 255, 198,  99, 255 ];
    const COLOR_OR:         [u8; 4] = [  99, 242, 255, 255 ];
    const COLOR_XOR:        [u8; 4] = [ 174, 116, 255, 255 ];
    const COLOR_NOT:        [u8; 4] = [ 255,  98, 138, 255 ];
    const COLOR_NAND:       [u8; 4] = [ 255, 162,   0, 255 ];
    const COLOR_NOR:        [u8; 4] = [  48, 217, 255, 255 ];
    const COLOR_XNOR:       [u8; 4] = [ 166,   0, 255, 255 ];
    const COLOR_LATCHON:    [u8; 4] = [  99, 255, 159, 255 ];
    const COLOR_LATCHOFF:   [u8; 4] = [  56,  77,  71, 255 ];
    const COLOR_CLOCK:      [u8; 4] = [ 255,   0,  65, 255 ];
    const COLOR_LED:        [u8; 4] = [ 255, 255, 255, 255 ];
    const COLOR_ANNOTATION: [u8; 4] = [  58,  69,  81, 255 ];
    const COLOR_FILLER:     [u8; 4] = [ 140, 171, 161, 255 ];
}

impl Trace {
    fn to_color_raw(self) -> [u8; 4] {
        match self {
            Trace::Gray => ColorConstants::COLOR_GRAY,
            Trace::White => ColorConstants::COLOR_WHITE,
            Trace::Red => ColorConstants::COLOR_RED,
            Trace::Orange1 => ColorConstants::COLOR_ORANGE1,
            Trace::Orange2 => ColorConstants::COLOR_ORANGE2,
            Trace::Orange3 => ColorConstants::COLOR_ORANGE3,
            Trace::Yellow => ColorConstants::COLOR_YELLOW,
            Trace::Green1 => ColorConstants::COLOR_GREEN1,
            Trace::Green2 => ColorConstants::COLOR_GREEN2,
            Trace::Cyan1 => ColorConstants::COLOR_CYAN1,
            Trace::Cyan2 => ColorConstants::COLOR_CYAN2,
            Trace::Blue1 => ColorConstants::COLOR_BLUE1,
            Trace::Blue2 => ColorConstants::COLOR_BLUE2,
            Trace::Purple => ColorConstants::COLOR_PURPLE,
            Trace::Magenta => ColorConstants::COLOR_MAGENTA,
            Trace::Pink => ColorConstants::COLOR_PINK,
            Trace::Write => ColorConstants::COLOR_WRITE,
            Trace::Empty => ColorConstants::COLOR_EMPTY,
            Trace::Cross => ColorConstants::COLOR_CROSS,
            Trace::Read => ColorConstants::COLOR_READ,
            Trace::Buffer => ColorConstants::COLOR_BUFFER,
            Trace::And => ColorConstants::COLOR_AND,
            Trace::Or => ColorConstants::COLOR_OR,
            Trace::Xor => ColorConstants::COLOR_XOR,
            Trace::Not => ColorConstants::COLOR_NOT,
            Trace::Nand => ColorConstants::COLOR_NAND,
            Trace::Nor => ColorConstants::COLOR_NOR,
            Trace::Xnor => ColorConstants::COLOR_XNOR,
            Trace::LatchOn => ColorConstants::COLOR_LATCHON,
            Trace::LatchOff => ColorConstants::COLOR_LATCHOFF,
            Trace::Clock => ColorConstants::COLOR_CLOCK,
            Trace::Led => ColorConstants::COLOR_LED,
            Trace::Annotation => ColorConstants::COLOR_ANNOTATION,
            Trace::Filler => ColorConstants::COLOR_FILLER,
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
    fn from_raw_color(color: &[u8]) -> Self {
        let color: [u8; 4] = color.try_into().unwrap();
        match color {
            ColorConstants::COLOR_GRAY       => Trace::Gray,
            ColorConstants::COLOR_WHITE      => Trace::White,
            ColorConstants::COLOR_RED        => Trace::Red,
            ColorConstants::COLOR_ORANGE1    => Trace::Orange1,
            ColorConstants::COLOR_ORANGE2    => Trace::Orange2,
            ColorConstants::COLOR_ORANGE3    => Trace::Orange3,
            ColorConstants::COLOR_YELLOW     => Trace::Yellow,
            ColorConstants::COLOR_GREEN1     => Trace::Green1,
            ColorConstants::COLOR_GREEN2     => Trace::Green2,
            ColorConstants::COLOR_CYAN1      => Trace::Cyan1,
            ColorConstants::COLOR_CYAN2      => Trace::Cyan2,
            ColorConstants::COLOR_BLUE1      => Trace::Blue1,
            ColorConstants::COLOR_BLUE2      => Trace::Blue2,
            ColorConstants::COLOR_PURPLE     => Trace::Purple,
            ColorConstants::COLOR_MAGENTA    => Trace::Magenta,
            ColorConstants::COLOR_PINK       => Trace::Pink,
            ColorConstants::COLOR_WRITE      => Trace::Write,
            ColorConstants::COLOR_EMPTY      => Trace::Empty,
            ColorConstants::COLOR_CROSS      => Trace::Cross,
            ColorConstants::COLOR_READ       => Trace::Read,
            ColorConstants::COLOR_BUFFER     => Trace::Buffer,
            ColorConstants::COLOR_AND        => Trace::And,
            ColorConstants::COLOR_OR         => Trace::Or,
            ColorConstants::COLOR_XOR        => Trace::Xor,
            ColorConstants::COLOR_NOT        => Trace::Not,
            ColorConstants::COLOR_NAND       => Trace::Nand,
            ColorConstants::COLOR_NOR        => Trace::Nor,
            ColorConstants::COLOR_XNOR       => Trace::Xnor,
            ColorConstants::COLOR_LATCHON    => Trace::LatchOn,
            ColorConstants::COLOR_LATCHOFF   => Trace::LatchOff,
            ColorConstants::COLOR_CLOCK      => Trace::Clock,
            ColorConstants::COLOR_LED        => Trace::Led,
            ColorConstants::COLOR_ANNOTATION => Trace::Annotation,
            ColorConstants::COLOR_FILLER     => Trace::Filler,
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
}

/// Represents one pixel.
/// It is probably a mistake to
/// make a copy of this type.
#[derive(Debug)]
struct BoardElement<const STRATEGY: u8> {
    /// Raw color from input file.
    color_on: [u8; 4],
    color_off: [u8; 4],
    kind: Trace,
    id: Option<usize>,
}
impl<const STRATEGY: u8> BoardElement<STRATEGY> {
    fn new(color: &[u8]) -> Self {
        let trace = Trace::from_raw_color(color);
        BoardElement {
            color_on: trace.to_color_on(),
            color_off: trace.to_color_off(),
            kind: trace,
            id: None,
        }
    }
    fn print(&self, board: &VcbBoard<STRATEGY>, _i: usize, marked: bool, debug: bool) {
        let mut brfac: u32 = 50;
        let mut state = false;
        let tmpstr = if let Some(t) = self.id {
            let id_to_print = board.compiled_network.get_inner_id(t) % 100;
            if let Some(id) = board.nodes[t].network_id {
                if board.compiled_network.get_state(id) {
                    state = true;
                    brfac = 255;
                };
                if debug {
                    format!("{:>2}", id_to_print)
                } else {
                    format!("  ")
                }
            } else {
                if debug {
                    format!("{:>2}", id_to_print)
                } else {
                    format!("  ")
                }
            }
        } else {
            "  ".to_string()
        };
        let col = if marked {
            (255, 0, 0)
        } else {
            (
                ((u32::from(self.color_on[0]) * brfac) / 255)
                    .try_into()
                    .unwrap(),
                ((u32::from(self.color_on[1]) * brfac) / 255)
                    .try_into()
                    .unwrap(),
                ((u32::from(self.color_on[2]) * brfac) / 255)
                    .try_into()
                    .unwrap(),
            )
        };
        let col = if state { self.color_on } else { self.color_off };

        let tmp = tmpstr.on_truecolor(col[0], col[1], col[2]);
        //.truecolor(
        //u8::MAX - col.0,
        //u8::MAX - col.1,
        //u8::MAX - col.2,);
        print!("{}", tmp);
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
    elements: Vec<BoardElement<STRATEGY>>,
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
    //#[inline(always)]
    //pub fn update_simd(&mut self) {
    //    self.compiled_network.update_simd();
    //}
    #[inline(always)]
    pub fn update(&mut self) {
        self.compiled_network.update();
    }
    fn new(data: &[u8], width: usize, height: usize, optimize: bool) -> Self {
        let num_elements = width * height;
        let mut elements = Vec::with_capacity(num_elements);

        for i in 0..width * height {
            elements.push(BoardElement::new(&data[i * 4..i * 4 + 4]));
        }
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

        // TODO: terminal buffer
        board
    }
    fn add_connection(&mut self, connection: (usize, usize), swp_dir: bool) {
        let (start, end) = if swp_dir {
            (connection.1, connection.0)
        } else {
            connection
        };
        assert!(start != end);
        let a = self.nodes[start].inputs.insert(end);
        let b = self.nodes[end].outputs.insert(start);
        assert!(a == b);
        assert_ne!(
            (self.nodes[start].kind == GateType::Cluster),
            (self.nodes[end].kind == GateType::Cluster)
        );
        //if a {println!("connect: {start}, {end}");}
    }

    // floodfill with id at given index
    // pre: logic trace, otherwise id could have been
    // assigned to nothing, this_x valid
    fn fill_id(&mut self, this_x: i32, id: usize) {
        let this = &mut self.elements[TryInto::<usize>::try_into(this_x).unwrap()];
        let this_kind = this.kind;
        assert!(this_kind.is_logic());
        match this.id {
            None => this.id = Some(id),
            Some(this_id) => {
                assert_eq!(this_id, id);
                return;
            },
        }
        let width: i32 = self.width.try_into().unwrap();
        'side: for dx in [1, -1, width, -width] {
            'forward: for ddx in [1, 2] {
                //TODO: handle wrapping
                let other_x = this_x + dx * ddx;
                let other_kind =
                    unwrap_or_else!(self.elements.get(other_x as usize), continue 'side).kind;
                if other_kind == Trace::Cross {
                    continue 'forward;
                }
                if !other_kind.is_same_as(this_kind) {
                    continue 'side;
                }
                self.fill_id(other_x, id);
                continue 'side;
            }
        }
    }
    // Connect gate with immediate neighbor
    // pre: this is gate, this_x valid, this has id.
    fn connect_id(&mut self, this_x: i32, this_id: usize, id_counter: &mut usize) {
        let this = &mut self.elements[TryInto::<usize>::try_into(this_x).unwrap()];
        let this_kind = this.kind;
        assert!(this_kind.is_gate());
        assert!(this.id.is_some());

        let width: i32 = self.width.try_into().unwrap();
        'side: for dx in [1, -1, width, -width] {
            //TODO: handle wrapping
            let other_x = this_x + dx;
            let other = unwrap_or_else!(self.elements.get(other_x as usize), continue 'side);

            let dir = match other.kind {
                Trace::Read => false,
                Trace::Write => true,
                _ => continue,
            };
            let other_id = unwrap_or_else!(other.id, { self.add_new_id(other_x, id_counter) });
            self.add_connection((this_id, other_id), dir);
        }
    }

    // pre: this trace is logic
    fn add_new_id(&mut self, this_x: i32, id_counter: &mut usize) -> usize {
        let this = &self.elements[TryInto::<usize>::try_into(this_x).unwrap()];
        assert!(this.kind.is_logic());
        // create a new id.
        self.nodes.push(BoardNode::new(this.kind));
        let this_id = *id_counter;
        *id_counter += 1;

        // fill with this_id
        self.fill_id(this_x, this_id);
        this_id
    }

    // this HAS to be recursive since gates have arbitrary shapes.
    fn explore(&mut self, this_x: i32, id_counter: &mut usize) {
        // if prev_id is Some, merge should be done.
        // don't merge if id already exists, since that wouldn't make sense

        // all here is correct
        let this = &self.elements[TryInto::<usize>::try_into(this_x).unwrap()];
        let this_kind = this.kind;

        if !this.kind.is_logic() {
            return;
        }

        let this_id = unwrap_or_else!(this.id, { self.add_new_id(this_x, id_counter) });

        assert!(this_kind.is_logic());
        if !this_kind.is_gate() {
            return;
        }

        self.connect_id(this_x, this_id, id_counter);
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
    pub fn print_debug(&self) {
        self.print_inner(true);
    }
    pub fn print(&self) {
        self.print_inner(false);
    }
}

/// contains the raw footer data.
#[derive(Debug, Default)]
#[repr(C)]
struct Footer {
    height_type: i32,
    height: i32,
    width_type: i32,
    width: i32,
    bytes_type: i32,
    bytes: i32,
    layer_type: i32,
    layer: i32,
}
impl Footer {
    const SIZE: usize = 32; // 8*4 bytes
}
#[derive(Default)]
pub struct VcbParser<const STRATEGY: u8> {}
impl<const STRATEGY: u8> VcbParser<STRATEGY> {
    /// # Panics
    /// invalid base64 string, invalid zstd, invalid colors
    #[must_use]
    pub fn parse_to_board(data: &str, optimize: bool) -> VcbBoard<STRATEGY> {
        let bytes = base64::decode_config(data.trim(), base64::STANDARD).unwrap();
        let data_bytes = &bytes[..bytes.len() - Footer::SIZE];
        let footer_bytes: [u8; Footer::SIZE] = bytes[bytes.len() - Footer::SIZE..bytes.len()]
            .try_into()
            .unwrap();
        let footer = FooterInfo::new(&unsafe {
            std::mem::transmute::<[u8; Footer::SIZE], Footer>(footer_bytes)
        });
        assert!(footer.layer == Layer::Logic);
        let data = zstd::bulk::decompress(data_bytes, 1 << 27).unwrap();
        assert!(!data.is_empty());
        assert!(data.len() == footer.count * 4);
        VcbBoard::new(&data, footer.width, footer.height, optimize)
    }
}
