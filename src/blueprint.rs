// blueprint.rs: parsing VCB blueprints

#[allow(clippy::upper_case_acronyms)]
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
enum Trace {
    GrayTrace,
    WhiteTrace,
    RedTrace,
    Orange1Trace,

    Orange2Trace,
    Orange3Trace,
    YellowTrace,
    Green1Trace,

    Green2Trace,
    CyanTrace,
    Cyan2Trace,
    Blue1Trace,

    Blue2Trace,
    PurpleTrace,
    MagentaTrace,
    PinkTrace,

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
        let data = zstd::bulk::decompress(data_bytes, 9999999).unwrap_or(Vec::new());
        println!("{:#?}",data);
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

        for c in 0..footer.count {
            let i = c*4;
            println!("{:?}",(data[i],data[i+1],data[i+2],data[i+3]) );
        }
    }
}

