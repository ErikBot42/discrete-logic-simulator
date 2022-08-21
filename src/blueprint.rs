// blueprint.rs: parsing VCB blueprints

#![allow(clippy::upper_case_acronyms)]
use colored::Colorize;
use std::collections::BTreeSet;
use crate::logic::*;
//use std::time::Instant;

//use std::hash::{Hash, Hasher};
//use std::collections::HashSet;
//use std::collections::HashMap;

#[derive(Debug)]
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
    fn new(footer: Footer) -> Self {
        FooterInfo {
            width: footer.width as usize,
            height: footer.height as usize,
            count: (footer.width*footer.height) as usize,
            layer: match footer.layer {
                65536 => Layer::Logic,
                131072 => Layer::On,
                262144 => Layer::Off,
                _ => panic!(),
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
        match self {
            Trace::Gray | Trace::White | Trace::Red | Trace::Orange1 | Trace::Orange2 | Trace::Orange3 | Trace::Yellow | Trace::Green1 | Trace::Green2 | Trace::Cyan1 | Trace::Cyan2 | Trace::Blue1 | Trace::Blue2 | Trace::Purple | Trace::Magenta | Trace::Pink | Trace::Read | Trace::Write => true,
            _ => false,
        }
    }
    fn is_gate(self) -> bool {
        match self {
            Trace::Buffer | Trace::And | Trace::Or | Trace::Xor | Trace::Not | Trace::Nand | Trace::Nor | Trace::Xnor | Trace::LatchOn | Trace::LatchOff | Trace::Clock | Trace::Led => true,
            _ => false,
        }
    }
    fn should_merge(self, other: Trace) -> bool {
        other == Trace::Cross || self == other || self.is_wire() && other.is_wire()
    }

    /// returns bool if connection should be made, 
    /// containing whether to swap order of the connection
    fn should_invert_connection(self, other: Trace) -> Option<bool> {
        let swp; 
        if self.is_gate() {
            swp = true;
        } else if other.is_gate() {
            swp = false;
        } else {
            return None
        };
        assert_ne!((self.to_gate() == GateType::CLUSTER),
                  (other.to_gate() == GateType::CLUSTER));
        match if swp {other} else {self} {
            Trace::Read => Some(swp),
            Trace::Write => Some(!swp),
            _ => None,
        }
    }
    fn to_gate(self) -> GateType {
        // TODO: single input gate simplification.
        if self.is_wire() {
            GateType::CLUSTER
        }
        else {
            match self {
                Trace::Buffer   => GateType::OR,
                Trace::And      => GateType::AND,
                Trace::Or       => GateType::OR,
                Trace::Xor      => GateType::XOR,
                Trace::Not      => GateType::NOR,
                Trace::Nand     => GateType::NAND,
                Trace::Nor      => GateType::NOR,
                Trace::Xnor     => GateType::XNOR,
                Trace::LatchOn  => GateType::XNOR,
                Trace::LatchOff => GateType::XOR,
                Trace::Clock    => panic!(),
                Trace::Led      => (GateType::OR),
                _ => panic!(),

            } 
        }
    }
}
//fn print_blueprint_data(data: &Vec<u8>, footer: &FooterInfo) 
//{
//    for y in 0..footer.height {
//        for x in 0..footer.width {
//            let i = (x + y*footer.width)*4;
//            let p = "  ".on_truecolor(
//                data[i],
//                data[i+1],
//                data[i+2],
//                );
//            print!("{}",p);
//        }
//        println!();
//    }
//}

/// Represents one pixel.
#[derive(Debug, Copy, Clone)]
struct BoardElement {
    color: [u8; 4],
    kind: Trace,
    //inputs: BTreeSet<usize>,
    //outputs: BTreeSet<usize>,
    // id for the associated node
    id: Option<usize>, 
}
impl BoardElement {
    fn new(color: &[u8]) -> Self {
        BoardElement {color: color.try_into().unwrap(), kind: Trace::from_color(color), id: None}
    }
    fn print(&self, board: &VcbBoard, i: usize) {
        let mut brfac: u32 = 70;
        print!("{}", match self.id {
            Some(t) => match board.nodes[t].network_id {
                Some(id) => {
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
                    format!("{:>2}",i%100)
                }
                None => format!("{:>2}",i%100),    
            }
            None => format!("  ")
        }.on_truecolor(
            (self.color[0] as u32 * brfac / 255) as u8,
            (self.color[1] as u32 * brfac / 255) as u8,
            (self.color[2] as u32 * brfac / 255) as u8));
    }
}
/// Represents one gate or trace
#[derive(Debug)]
struct BoardNode {
    inputs: BTreeSet<usize>,
    outputs: BTreeSet<usize>,
    trace: Trace,
    kind: GateType,
    network_id: Option<usize>,
    // TODO: type
}
impl BoardNode {
    fn new(trace: Trace) -> Self {
        BoardNode {inputs: BTreeSet::new(), outputs: BTreeSet::new(), kind: trace.to_gate(), trace, network_id: None}
    }
}
//connections: HashSet<(usize,usize)>,

#[derive(Debug)]
pub struct VcbBoard {
    elements: Vec<BoardElement>,
    nodes: Vec<BoardNode>,
    pub network: GateNetwork, //TODO
    width: usize,
    height: usize,
}
impl VcbBoard {
    /// For regression testing
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
    fn new(data: Vec<u8>, width: usize, height: usize) -> Self {
        let num_elements = width*height;
        let mut elements = Vec::with_capacity(num_elements);
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
        for x in 0..num_elements {
            board.explore_rec(x as i32, 0, board.nodes.len(), None);
        }
        // add vertexes to network
        for node in &mut board.nodes {
            node.network_id = Some(board.network.add_vertex(node.kind));
        }
        // add edges to network
        for i in 0..board.nodes.len() {
        //for node in &board.nodes {
            let node = &board.nodes[i];
            for input in &node.inputs{
                assert!(board.nodes[*input].outputs.contains(&i));
            }
            board.network.add_inputs(
                node.kind,
                node.network_id.unwrap(),
                node.inputs.clone().into_iter().map(|x| board.nodes[x].network_id.unwrap()).collect()
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
        match self.nodes[start].kind {
            _ => (),
        };
        match self.nodes[end].kind {
            _ => (),
        };
        //if a {println!("connect: {start}, {end}");}
    }
    fn explore_it(&mut self, x: i32, id_counter: &mut usize) {

        // this trace, must be a valid index
        let this = self.elements[x as usize];
        
        let this_is_gate = this.kind.is_gate();
        let this_is_wire = this.kind.is_wire();
        
        // a trace must be either wire or gate to get an id.
        if !this_is_wire && !this_is_gate {return}
        
        // TODO: optimize direction based on iteration order.
        'side: for dx in [1,-1,self.width as i32,-(self.width as i32)] {
            'forward: for ddx in [1,2] { // disallow double crosses.
                let other_x = x + dx*ddx;
                // edge hit case
                if (other_x%self.width as i32 - x%self.width as i32).abs()>ddx {continue 'side}
                let other = match self.elements.get(other_x as usize) {
                    None => continue 'side, 
                    Some(other) => other
                };

                // Can't provide id if neither gate has id. This handles the 
                // case of the other gate being a non active logic gate.
                if other.id == None && this.id == None {continue 'side};
                let other_is_gate = other.kind.is_gate();
                let other_is_wire = this.kind.is_wire();
                if other.kind == Trace::Cross {continue 'forward};
                if !other_is_gate && !other_is_wire {continue 'side}
                assert!(other_is_gate != other_is_wire);
                assert!(this_is_gate != this_is_wire);

                // merge traces if applicable
                if (this.kind == other.kind) || (this_is_wire && other_is_wire) {
                    self.elements[x as usize].id = this.id.or(other.id);
                    self.elements[other_x as usize].id = this.id;
                } else { // cannot connect AND merge
                    if ddx > 1 {continue 'side}; // disallow gate connection through cross.
                    let inv_connection_dir = match if this_is_wire {this.kind} else {other.kind} {
                        Trace::Read => this_is_wire,
                        Trace::Write => !this_is_wire,
                        _ => continue 'side, 
                    };
                    self.add_connection((this.id.unwrap(), other.id.unwrap()),inv_connection_dir); 
                }
            }
        }
        if this.id == None { // new gate
            self.nodes.push(BoardNode::new(this.kind));
            self.elements[x as usize].id = Some(*id_counter);
            *id_counter += 1;
        }
    }

    fn explore_rec(&mut self, mut x: i32, dx: i32, id: usize, prev: Option<(Trace, usize)>) {
        x+=dx;

        let el = match self.elements.get_mut(x as usize) {Some(el) => el, None => {return}};

        if let Some((prev_trace,prev_id)) = prev {
            if !prev_trace.should_merge(el.kind) {
                if let Some(id) = el.id {
                    // Create node connection
                    assert!(id != prev_id);
                    assert!(prev != None);
                    if let Some(v) = prev_trace.should_invert_connection(el.kind) {
                        self.add_connection((id, prev_id),v); 
                        return;
                    }
                };
                return
            }
        };
        // merge with prev 
        match el.kind {
            Trace::Empty => return,
            Trace::Cross => {
                if dx != 0 {
                    self.explore_rec(x,dx,id,prev);
                } 
                return 
            }
            _ => (),
        }
        // assign new id to el
        if el.id == None {
            el.id = Some(id);
            let kind = el.kind;
            // origin of search
            // add id before using it
            if prev == None {
                self.nodes.push(BoardNode::new(kind));
                assert!(id == self.nodes.len()-1);
            }
            if x % self.width as i32 != 0 {
                self.explore_rec(x, -1, id, Some((kind, id))); 
            }
            if x as i32 % self.width as i32 != self.width as i32 -1 {
                self.explore_rec(x, 1, id, Some((kind, id))); 
            }
            self.explore_rec(x,   self.width as i32,  id, Some((kind, id))); 
            self.explore_rec(x, -(self.width as i32), id, Some((kind, id))); 
        };
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
pub struct BlueprintParser {}
impl BlueprintParser {
    pub fn parse(&mut self, data: &str) -> VcbBoard {
        let bytes = base64::decode_config(data.trim(), base64::STANDARD).unwrap();
        let data_bytes = &bytes[..bytes.len()-Footer::SIZE];
        let footer_bytes: [u8; Footer::SIZE] = bytes[bytes.len()-Footer::SIZE..bytes.len()].try_into().unwrap();
        let footer = FooterInfo::new(unsafe { std::mem::transmute::<[u8; Footer::SIZE], Footer>(footer_bytes) });
        let data = zstd::bulk::decompress(data_bytes, 9999999).unwrap();
        assert!(data.len() != 0);
        assert!(data.len() == footer.count*4);
        VcbBoard::new(data, footer.width, footer.height)
    }
}

