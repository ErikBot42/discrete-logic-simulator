// blueprint.rs: parsing VCB blueprints
#![allow(clippy::upper_case_acronyms)]
use colored::Colorize;
use std::collections::BTreeSet;
use crate::logic::{GateNetwork, GateType};
//use std::time::Instant;

//use std::hash::{Hash, Hasher};
//use std::collections::HashSet;
//use std::collections::HashMap;

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
            count: (footer.width*footer.height).try_into().unwrap(),
            layer: match footer.layer {
                 65_536 => Layer::Logic,
                131_072 => Layer::On,
                262_144 => Layer::Off,
                _       => panic!(),
            }
        }
    }
}
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
impl Trace {
    fn from_color(color: &[u8]) -> Self {
        let color: [u8; 4] = color.try_into().unwrap();
        match color {
            [42,  53,  65,  255] => Trace::Gray,
            [159, 168, 174, 255] => Trace::White,
            [161, 85,  94,  255] => Trace::Red,
            [161, 108, 86,  255] => Trace::Orange1,
            [161, 133, 86,  255] => Trace::Orange2,
            [161, 152, 86,  255] => Trace::Orange3,
            [153, 161, 86,  255] => Trace::Yellow,
            [136, 161, 86,  255] => Trace::Green1,
            [108, 161, 86,  255] => Trace::Green2,
            [86,  161, 141, 255] => Trace::Cyan1,
            [86,  147, 161, 255] => Trace::Cyan2,
            [86,  123, 161, 255] => Trace::Blue1,
            [86,  98,  161, 255] => Trace::Blue2,
            [102, 86,  161, 255] => Trace::Purple,
            [135, 86,  161, 255] => Trace::Magenta,
            [161, 85,  151, 255] => Trace::Pink,
            [77,  56,  62,  255] => Trace::Write,
            [0,   0,   0,   0]   => Trace::Empty,
            [102, 120, 142, 255] => Trace::Cross,
            [46,  71,  93,  255] => Trace::Read,
            [146, 255, 99,  255] => Trace::Buffer,
            [255, 198, 99,  255] => Trace::And,
            [99,  242, 255, 255] => Trace::Or,
            [174, 116, 255, 255] => Trace::Xor,
            [255, 98,  138, 255] => Trace::Not,
            [255, 162, 0,   255] => Trace::Nand,
            [48,  217, 255, 255] => Trace::Nor,
            [166, 0,   255, 255] => Trace::Xnor,
            [99,  255, 159, 255] => Trace::LatchOn,
            [56,  77,  71,  255] => Trace::LatchOff,
            [255, 0,   65,  255] => Trace::Clock,
            [255, 255, 255, 255] => Trace::Led,
            [58,  69,  81,  255] => Trace::Annotation,
            [140, 171, 161, 255] => Trace::Filler,
            _                    => panic!("Invalid trace color"),
        }
    }
    fn is_wire(self) -> bool {
        matches!(self, Trace::Gray | Trace::White | Trace::Red | Trace::Orange1 | Trace::Orange2 | Trace::Orange3 | Trace::Yellow | Trace::Green1 | Trace::Green2 | Trace::Cyan1 | Trace::Cyan2 | Trace::Blue1 | Trace::Blue2 | Trace::Purple | Trace::Magenta | Trace::Pink | Trace::Read | Trace::Write)
    }
    fn is_gate(self) -> bool {
        matches!(self, Trace::Buffer | Trace::And | Trace::Or | Trace::Xor | Trace::Not | Trace::Nand | Trace::Nor | Trace::Xnor | Trace::LatchOn | Trace::LatchOff | Trace::Clock | Trace::Led)
    }
    fn is_logic(self) -> bool {
        self.is_wire() || self.is_gate()
    }
    // is logically same as other, will connect
    fn is_same_as(self, other: Self) -> bool {
        (self == other) || (self.is_wire() && other.is_wire())
    }

    fn to_gate(self) -> GateType {
        // TODO: single input gate simplification.
        if self.is_wire() {
            GateType::CLUSTER
        }
        else {
            match self {
                Trace::Buffer | Trace::Or | Trace::Led => GateType::OR,
                Trace::Not | Trace::Nor                => GateType::NOR,
                Trace::And                             => GateType::AND,
                Trace::Nand                            => GateType::NAND,
                Trace::Xor | Trace::LatchOff           => GateType::XOR,
                Trace::Xnor | Trace::LatchOn           => GateType::XNOR,
                _                                      => panic!("unsupported logic trace: {self:?}"),
            } 
        }
    }
}

/// Represents one pixel.
/// It is probably a mistake to
/// make a copy of this type.
#[derive(Debug)]
struct BoardElement {
    color: [u8; 4],
    kind: Trace,
    id: Option<usize>, 
}
impl BoardElement {
    fn new(color: &[u8]) -> Self {
        BoardElement {color: color.try_into().unwrap(), kind: Trace::from_color(color), id: None}
    }
    fn print(&self, board: &VcbBoard, _i: usize) {
        let mut brfac: u32 = 100;
        let tmpstr = if let Some(t) = self.id {
            if let Some(id) = board.nodes[t].network_id {
                if board.network.get_state(id) {
                    brfac = 255;
                }; 
                //match board.nodes[t].kind {
                //    GateType::AND => format!(" A"),
                //    GateType::NAND => format!("NA"),
                //    GateType::OR => format!(" O"),
                //    GateType::NOR => format!("NO"),
                //    GateType::XOR => format!(" X"),
                //    GateType::XNOR => format!("NX"),
                //    GateType::CLUSTER => format!(" C"),
                //}
                //format!("{:>2}",t%100)
                format!("{:>2}",t%100)
            } else {format!("{:>2}",t%100)}
        } else {"  ".to_string()};
        print!("{}", tmpstr.on_truecolor(
            ((u32::from(self.color[0]) * brfac) / 255).try_into().unwrap(),
            ((u32::from(self.color[1]) * brfac) / 255).try_into().unwrap(),
            ((u32::from(self.color[2]) * brfac) / 255).try_into().unwrap()));
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
        BoardNode {inputs: BTreeSet::new(), outputs: BTreeSet::new(), kind: trace.to_gate(), network_id: None}
    }
}

#[derive(Debug)]
pub struct VcbBoard {
    elements: Vec<BoardElement>,
    nodes: Vec<BoardNode>,
    network: GateNetwork, //TODO
    width: usize,
    height: usize,
}
impl VcbBoard {
    /// For regression testing
    #[must_use]
    pub fn make_state_vec(&self) -> Vec<bool> {
        let mut a = Vec::new();
        for i in 0..self.elements.len() {
            a.push(match self.elements[i].id {
                None => false,
                Some(node_id) => match self.nodes[node_id].network_id {
                    None => false,
                    Some(id) => self.network.get_state(id),
                }
            });
        };
        a
    }
    pub fn update(&mut self) {
        self.network.update();
    }
    fn new(data: &[u8], width: usize, height: usize) -> Self {
        let num_elements = width*height;
        let mut elements = Vec::with_capacity(num_elements);

        // TODO: iterate through chunks to trick compiler into
        // unrolling the loop
        for i in 0..width*height {
            elements.push(BoardElement::new(&data[i*4..i*4+4])); 
        }
        let mut board = VcbBoard{
            elements,
            nodes: Vec::new(),
            width,
            height,
            network: GateNetwork::default()
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
            for input in &node.inputs{
                assert!(board.nodes[*input].outputs.contains(&i));
            }
            let mut inputs: Vec<usize> = node.inputs.clone().into_iter().map(|x| board.nodes[x].network_id.unwrap()).collect();
            inputs.sort_unstable();
            inputs.dedup();
            board.network.add_inputs(
                node.kind,
                node.network_id.unwrap(),
                inputs,
                );
        }
        board.network.init_network();
        
        // TODO: terminal buffer
        board
    }
    fn add_connection(&mut self, connection: (usize,usize), swp_dir: bool) {
        let (start, end) = if swp_dir {(connection.1,connection.0)} else {connection};
        assert!(start != end);
        let a = self.nodes[start].inputs.insert(end);
        let b = self.nodes[end].outputs.insert(start);
        assert!(a == b);
        assert_ne!((self.nodes[start].kind == GateType::CLUSTER),
                   (self.nodes[end].  kind == GateType::CLUSTER));
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
            Some(this_id) => {assert_eq!(this_id, id); return},
        }
        let width: i32 = self.width.try_into().unwrap();
        'side: for dx in [1,-1,width, -width] {
            'forward: for ddx in [1,2] {
                //TODO: handle wrapping
                let other_x = this_x + dx*ddx;
                let other_kind = unwrap_or_else!(self.elements.get(TryInto::<usize>::try_into(other_x).unwrap()), continue 'side).kind;
                if other_kind == Trace::Cross {continue 'forward}
                if !other_kind.is_same_as(this_kind) {continue 'side}
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
        'side: for dx in [1,-1,width, -width] {
            //TODO: handle wrapping
            let other_x = this_x + dx;
            let other = unwrap_or_else!(self.elements.get(TryInto::<usize>::try_into(other_x).unwrap()), continue 'side);

            let dir = match other.kind {
                Trace::Read => false,
                Trace::Write => true,
                _ => continue,
            };
            let other_id = unwrap_or_else!(other.id, {
                self.add_new_id(other_x, id_counter)
            });
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

        if !this.kind.is_logic() {return}

        let this_id = unwrap_or_else!(this.id, {
            self.add_new_id(this_x, id_counter)
        });

        assert!(this_kind.is_logic());
        if !this_kind.is_gate() {return}

        self.connect_id(this_x, this_id, id_counter);
    }

    pub fn print(&self) {
        println!("\nBoard:");
        for y in 0..self.height{
            for x in 0..self.width {
                let i = x+y*self.width;
                self.elements[i].print(self, i);
            }
            println!();
        }
    }
}
// contains the raw data.
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
pub struct Parser {}
impl Parser {
    /// # Panics
    /// invalid base64 string, invalid zstd, invalid colors
    #[must_use]
    pub fn parse(data: &str) -> VcbBoard {
        let bytes = base64::decode_config(data.trim(), base64::STANDARD).unwrap();
        let data_bytes = &bytes[..bytes.len()-Footer::SIZE];
        let footer_bytes: [u8; Footer::SIZE] = bytes[bytes.len()-Footer::SIZE..bytes.len()].try_into().unwrap();
        let footer = FooterInfo::new(&unsafe { std::mem::transmute::<[u8; Footer::SIZE], Footer>(footer_bytes) });
        assert!(footer.layer == Layer::Logic);
        let data = zstd::bulk::decompress(data_bytes, 1 << 32).unwrap();
        assert!(!data.is_empty());
        assert!(data.len() == footer.count*4);
        VcbBoard::new(&data, footer.width, footer.height)
    }
}

