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
#[derive(Debug)]
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
    Cyan,
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
    fn from_color(color: (u8,u8,u8,u8)) -> Self {
        match color {
            (42,  53,  65,  255) => Trace::Gray,                       
            (159, 168, 174, 255) => Trace::White,
            (161, 85,  94,  255) => Trace::Red,
            (161, 108, 86,  255) => Trace::Orange1,
            (161, 133, 86,  255) => Trace::Orange2,
            (161, 152, 86,  255) => Trace::Orange3,
            (153, 161, 86,  255) => Trace::Yellow,
            (136, 161, 86,  255) => Trace::Green1,
            (108, 161, 86,  255) => Trace::Green2,
            (86,  161, 141, 255) => Trace::Cyan,
            (86,  147, 161, 255) => Trace::Cyan2,
            (86,  123, 161, 255) => Trace::Blue1,
            (86,  98,  161, 255) => Trace::Blue2,
            (102, 86,  161, 255) => Trace::Purple,
            (135, 86,  161, 255) => Trace::Magenta,
            (161, 85,  151, 255) => Trace::Pink,
            (77,  56,  62,  255) => Trace::Write,
            (0,   0,   0,   0)   => Trace::Empty,
            (102, 120, 142, 255) => Trace::Cross,
            (46,  71,  93,  255) => Trace::Read,
            (146, 255, 99,  255) => Trace::Buffer,
            (255, 198, 99,  255) => Trace::And,
            (99,  242, 255, 255) => Trace::Or,
            (174, 116, 255, 255) => Trace::Xor,
            (255, 98,  138, 255) => Trace::Not,
            (255, 162, 0,   255) => Trace::Nand,
            (48,  217, 255, 255) => Trace::Nor,
            (166, 0,   255, 255) => Trace::Xnor,
            (99,  255, 159, 255) => Trace::LatchOn,
            (56,  77,  71,  255) => Trace::LatchOff,
            (255, 0,   65,  255) => Trace::Clock,
            (255, 255, 255, 255) => Trace::Led,
            (58,  69,  81,  255) => Trace::Annotation,
            (140, 171, 161, 255) => Trace::Filler,                                                                            
            _                    => panic!(),
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
        println!("{:#?}",footer);

        // hopefully, input data isn't a zip bomb
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
                println!("{}: {:#?}",p, Trace::from_color((data[i],data[i+1],data[i+2],data[i+3])));
            }
        }

        for c in 0..footer.count {
            let i = c*4;
            println!("{:?}",(data[i],data[i+1],data[i+2],data[i+3]) );
        }
    }
}

