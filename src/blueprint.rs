// blueprint.rs: parsing VCB blueprints

#![allow(clippy::upper_case_acronyms)]
use colored::Colorize;

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
            _                    => panic!(),
        }
    }
    fn is_wire(self) -> bool {
        match self {
            Trace::Gray|Trace::White|Trace::Red|Trace::Orange1|Trace::Orange2|Trace::Orange3|Trace::Yellow|Trace::Green1|Trace::Green2|Trace::Cyan1|Trace::Cyan2|Trace::Blue1|Trace::Blue2|Trace::Purple|Trace::Magenta|Trace::Pink|Trace::Read|Trace::Write => true,
            _ => false,
        }
    }
    fn should_connect(self, other: Trace) -> bool {
        other == Trace::Cross || self == other || self.is_wire() == other.is_wire()
    }
}
fn print_blueprint_data(data: &Vec<u8>, footer: &FooterInfo) 
{
    for y in 0..footer.height {
        for x in 0..footer.width {
            let i = (x + y*footer.width)*4;
            let p = "  ".on_truecolor(
                data[i],
                data[i+1],
                data[i+2],
                );
            print!("{}",p);
        }
        println!();
    }
}
#[derive(Debug)]
struct BoardElement{
    color: [u8; 4],
    kind: Trace,
    // every gate/trace has a unique id.
    // cross/empty has no id
    id: Option<usize>, 
}
impl BoardElement {
    fn new(color: &[u8]) -> Self {
        BoardElement {color: color.try_into().unwrap(), kind: Trace::from_color(color), id: None} 
    }
    fn print(&self) {
        print!("{}",match self.id {Some(t) => format!("{:>2}",t%100), None => format!("  ")}.on_truecolor(self.color[0],self.color[1],self.color[2]));
    }
}

#[derive(Debug)]
struct VcbBoard {
    elements: Vec<BoardElement>,
    width: usize,
    height: usize,
}
impl VcbBoard {
    fn new(data: Vec<u8>, width: usize, height: usize) -> Self {
        let mut elements: Vec<BoardElement> = Vec::with_capacity(width*height);
        for i in 0..width*height {
            elements.push(BoardElement::new(&data[i*4..i*4+4])); 
        }
        let mut board = VcbBoard{elements, width, height};
        board.print();
        let mut i = 0;
        for y in 0..height{
            for x in 0..width {
                if board.explore(x as i32, y as i32, 0, 0, i, None) {i+=1};
            }
        }
        board.print();
        
        board
    }
    fn explore(&mut self, mut x: i32, mut y: i32, dx: i32, dy: i32, i: usize, prev: Option<Trace>) -> bool {
        x+=dx;y+=dy;
        if !(0<=x && x<self.width as i32 && 0<=y && y<self.height as i32) {return false}
        let el = &mut self.elements[x as usize+y as usize*self.width];
        if match prev {Some(p) => !p.should_connect(el.kind),None => false} {return false}
        match el.kind {
            Trace::Empty => return false,
            Trace::Cross => {
                if (dx != 0 || dy != 0) {
                    //panic!("x:{x}, y:{y}, dx:{dx}, dy:{dy}, i:{i}");
                    self.explore(x,y,dx,dy,i,prev);
                } else {
                    //println!("x:{x}, y:{y}, dx:{dx}, dy:{dy}, i:{i}");
                }; 
                return false
            }
            _ => (),
        }
        match el.id {
            Some(id) => return false,
            None => el.id = Some(i),
        }
        let kind = el.kind;
        self.explore(x,y, 1, 0,i,Some(kind)); 
        self.explore(x,y,-1, 0,i,Some(kind)); 
        self.explore(x,y, 0, 1,i,Some(kind)); 
        self.explore(x,y, 0,-1,i,Some(kind)); 
        return true
    }
    fn print(&self) {
        println!("\nBoard:");
        for y in 0..self.height{
            for x in 0..self.width {
                let i = x+y*self.width;
                self.elements[i].print();
            }
            println!();
        }
    }
}
#[derive(Default)]
pub struct BlueprintParser {}
impl BlueprintParser {
    pub fn parse(&mut self, data: &str) {
        let bytes = base64::decode_config(data, base64::STANDARD).unwrap();

        let data_bytes = &bytes[..bytes.len()-Footer::SIZE];
        let footer_bytes: [u8; Footer::SIZE] = bytes[bytes.len()-Footer::SIZE..bytes.len()].try_into().unwrap();
        
        // TODO: this is easy but bad and non portable and does not consider endian etc...
        let footer;
        unsafe {footer = std::mem::transmute::<[u8; Footer::SIZE], Footer>(footer_bytes);}
        let footer = FooterInfo::new(footer); 

        let data = zstd::bulk::decompress(data_bytes, 9999999).unwrap_or_default();
        assert!(data.len() == footer.count*4);
        println!("{:#?}",data);
        for y in 0..footer.height {
            for x in 0..footer.width {
                let i = (x + y*footer.width)*4;
                let p = "  ".on_truecolor(
                    data[i],
                    data[i+1],
                    data[i+2],
                    );
                println!("{}: {:#?}",p, Trace::from_color(&[data[i],data[i+1],data[i+2],data[i+3]]));
            }
        }

        //for c in 0..footer.count {
        //    let i = c*4;
        //    println!("{:?}",(data[i],data[i+1],data[i+2],data[i+3]) );
        //}
        
        print_blueprint_data(&data, &footer);
        let board = VcbBoard::new(data, footer.width, footer.height);
        //println!("{:?}",board);
    }
}

