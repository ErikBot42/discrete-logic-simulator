/// All color constants used by vcb
#[rustfmt::skip]
mod vcb_colors {
    //                                               r,   g,   b,   w
    pub(crate) const COLOR_GRAY:       [u8; 4] =    [42,  53,  65,  255];
    pub(crate) const COLOR_WHITE:      [u8; 4] =    [159, 168, 174, 255];
    pub(crate) const COLOR_RED:        [u8; 4] =    [161, 85,  94,  255];
    pub(crate) const COLOR_ORANGE1:    [u8; 4] =    [161, 108, 86,  255];
    pub(crate) const COLOR_ORANGE2:    [u8; 4] =    [161, 133, 86,  255];
    pub(crate) const COLOR_ORANGE3:    [u8; 4] =    [161, 152, 86,  255];
    pub(crate) const COLOR_YELLOW:     [u8; 4] =    [153, 161, 86,  255];
    pub(crate) const COLOR_GREEN1:     [u8; 4] =    [136, 161, 86,  255];
    pub(crate) const COLOR_GREEN2:     [u8; 4] =    [108, 161, 86,  255];
    pub(crate) const COLOR_CYAN1:      [u8; 4] =    [86,  161, 141, 255];
    pub(crate) const COLOR_CYAN2:      [u8; 4] =    [86,  147, 161, 255];
    pub(crate) const COLOR_BLUE1:      [u8; 4] =    [86,  123, 161, 255];
    pub(crate) const COLOR_BLUE2:      [u8; 4] =    [86,  98,  161, 255];
    pub(crate) const COLOR_PURPLE:     [u8; 4] =    [102, 86,  161, 255];
    pub(crate) const COLOR_MAGENTA:    [u8; 4] =    [135, 86,  161, 255];
    pub(crate) const COLOR_PINK:       [u8; 4] =    [161, 85,  151, 255];
    pub(crate) const COLOR_WRITE:      [u8; 4] =    [77,  56,  62,  255];
    pub(crate) const COLOR_CROSS:      [u8; 4] =    [102, 120, 142, 255];
    pub(crate) const COLOR_READ:       [u8; 4] =    [46,  71,  93,  255];
    pub(crate) const COLOR_BUFFER:     [u8; 4] =    [146, 255, 99,  255];
    pub(crate) const COLOR_AND:        [u8; 4] =    [255, 198, 99,  255];
    pub(crate) const COLOR_OR:         [u8; 4] =    [99,  242, 255, 255];
    pub(crate) const COLOR_XOR:        [u8; 4] =    [174, 116, 255, 255];
    pub(crate) const COLOR_NOT:        [u8; 4] =    [255, 98,  138, 255];
    pub(crate) const COLOR_NAND:       [u8; 4] =    [255, 162, 0,   255];
    pub(crate) const COLOR_NOR:        [u8; 4] =    [48,  217, 255, 255];
    pub(crate) const COLOR_XNOR:       [u8; 4] =    [166, 0,   255, 255];
    pub(crate) const COLOR_LATCHON:    [u8; 4] =    [99,  255, 159, 255];
    pub(crate) const COLOR_LATCHOFF:   [u8; 4] =    [56,  77,  71,  255];
    pub(crate) const COLOR_CLOCK:      [u8; 4] =    [255, 0,   65,  255];
    pub(crate) const COLOR_LED:        [u8; 4] =    [255, 255, 255, 255];
    pub(crate) const COLOR_ANNOTATION: [u8; 4] =    [58,  69,  81,  255];
    pub(crate) const COLOR_FILLER:     [u8; 4] =    [140, 171, 161, 255];
    pub(crate) const COLOR_TUNNEL:     [u8; 4] =    [83,  85,  114, 255];
    pub(crate) const COLOR_MESH:       [u8; 4] =    [100, 106, 87,  255];
    pub(crate) const COLOR_WIRELESS_0: [u8; 4] =    [255, 0,   191, 255];
    pub(crate) const COLOR_WIRELESS_1: [u8; 4] =    [255, 0,   175, 255];
    pub(crate) const COLOR_WIRELESS_2: [u8; 4] =    [255, 0,   159, 255];
    pub(crate) const COLOR_WIRELESS_3: [u8; 4] =    [255, 0,   143, 255];
    pub(crate) const COLOR_TIMER:      [u8; 4] =    [255, 103, 0,   255];
    pub(crate) const COLOR_RANDOM:     [u8; 4] =    [229, 255, 0,   255];
    pub(crate) const COLOR_BREAK:      [u8; 4] =    [224, 0,   0,   255];
    pub(crate) const COLOR_BUS_RED:    [u8; 4] =    [122, 47,  36,  255];
    pub(crate) const COLOR_BUS_GREEN:  [u8; 4] =    [62,  122, 36,  255];
    pub(crate) const COLOR_BUS_BLUE:   [u8; 4] =    [36,  65,  122, 255];
    pub(crate) const COLOR_BUS_TEAL:   [u8; 4] =    [37,  98,  122, 255];
    pub(crate) const COLOR_BUS_PURPLE: [u8; 4] =    [122, 45,  102, 255];
    pub(crate) const COLOR_BUS_YELLOW: [u8; 4] =    [122, 112, 36,  255];
    
    pub(crate) const COLOR_EMPTY:      [u8; 4] = [0, 0, 0, 0];
    pub(crate) const COLOR_VMEM:       [u8; 4] = COLOR_LATCHOFF;
}

pub(crate) fn arbitrary_trace<'a>(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Trace> {
    use Trace::*;
    u.choose(&[
        Empty, Gray, White, Red, Orange1, Orange2, Orange3, Yellow, Green1, Green2, Cyan1, Cyan2,
        Blue1, Blue2, Purple, Magenta, Pink, Write, Cross, Read, Buffer, And, Or, Xor, Not, Nand,
        Nor, Xnor, LatchOn, LatchOff, Clock, Led, Annotation, Filler, Tunnel, Mesh, Wireless0,
        Wireless1, Wireless2, Wireless3, Timer, Random, Break, BusRed, BusGreen, BusBlue, BusTeal,
        BusPurple, BusYellow,
    ])
    .copied()
}

#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash)]
pub(crate) enum Trace {
    Empty = 0,
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
    Vmem,

    Tunnel,
    Mesh,
    Wireless0,
    Wireless1,
    Wireless2,
    Wireless3,
    Timer,
    Random,
    Break,
    BusRed,
    BusGreen,
    BusBlue,
    BusTeal,
    BusPurple,
    BusYellow,
}
impl Trace {
    #[cfg(feature = "render")]
    pub const VARIANTS: [Trace; 50] = [
        Trace::Empty,
        Trace::Gray,
        Trace::White,
        Trace::Red,
        Trace::Orange1,
        Trace::Orange2,
        Trace::Orange3,
        Trace::Yellow,
        Trace::Green1,
        Trace::Green2,
        Trace::Cyan1,
        Trace::Cyan2,
        Trace::Blue1,
        Trace::Blue2,
        Trace::Purple,
        Trace::Magenta,
        Trace::Pink,
        Trace::Write,
        Trace::Cross,
        Trace::Read,
        Trace::Buffer,
        Trace::And,
        Trace::Or,
        Trace::Xor,
        Trace::Not,
        Trace::Nand,
        Trace::Nor,
        Trace::Xnor,
        Trace::LatchOn,
        Trace::LatchOff,
        Trace::Clock,
        Trace::Led,
        Trace::Annotation,
        Trace::Filler,
        Trace::Vmem,
        Trace::Tunnel,
        Trace::Mesh,
        Trace::Wireless0,
        Trace::Wireless1,
        Trace::Wireless2,
        Trace::Wireless3,
        Trace::Timer,
        Trace::Random,
        Trace::Break,
        Trace::BusRed,
        Trace::BusGreen,
        Trace::BusBlue,
        Trace::BusTeal,
        Trace::BusPurple,
        Trace::BusYellow,
    ];
}
#[cfg(any(feature = "gif", feature = "print_sim", feature = "clip"))]
impl Trace {
    pub(crate) fn get_color(&self, state: bool) -> [u8; 4] {
        if state {
            self.to_color_on()
        } else {
            self.to_color_off()
        }
    }

    #[rustfmt::skip]
    pub(crate) fn to_color_raw(self) -> [u8; 4] {
        use vcb_colors::*;
        use Trace::*;
        match self {
            Gray       => COLOR_GRAY,
            White      => COLOR_WHITE,
            Red        => COLOR_RED,
            Orange1    => COLOR_ORANGE1,
            Orange2    => COLOR_ORANGE2,
            Orange3    => COLOR_ORANGE3,
            Yellow     => COLOR_YELLOW,
            Green1     => COLOR_GREEN1,
            Green2     => COLOR_GREEN2,
            Cyan1      => COLOR_CYAN1,
            Cyan2      => COLOR_CYAN2,
            Blue1      => COLOR_BLUE1,
            Blue2      => COLOR_BLUE2,
            Purple     => COLOR_PURPLE,
            Magenta    => COLOR_MAGENTA,
            Pink       => COLOR_PINK,
            Write      => COLOR_WRITE,
            Empty      => COLOR_EMPTY,
            Cross      => COLOR_CROSS,
            Read       => COLOR_READ,
            Buffer     => COLOR_BUFFER,
            And        => COLOR_AND,
            Or         => COLOR_OR,
            Xor        => COLOR_XOR,
            Not        => COLOR_NOT,
            Nand       => COLOR_NAND,
            Nor        => COLOR_NOR,
            Xnor       => COLOR_XNOR,
            LatchOn    => COLOR_LATCHON,
            LatchOff   => COLOR_LATCHOFF,
            Clock      => COLOR_CLOCK,
            Led        => COLOR_LED,
            Annotation => COLOR_ANNOTATION,
            Filler     => COLOR_FILLER,
            Vmem       => COLOR_VMEM,
            Tunnel     => COLOR_TUNNEL,
            Mesh       => COLOR_MESH,
            Wireless0  => COLOR_WIRELESS_0,
            Wireless1  => COLOR_WIRELESS_1,
            Wireless2  => COLOR_WIRELESS_2,
            Wireless3  => COLOR_WIRELESS_3,
            Timer      => COLOR_TIMER,
            Random     => COLOR_RANDOM,
            Break      => COLOR_BREAK,
            BusRed     => COLOR_BUS_RED,
            BusGreen   => COLOR_BUS_GREEN,
            BusBlue    => COLOR_BUS_BLUE,
            BusTeal    => COLOR_BUS_TEAL,
            BusPurple  => COLOR_BUS_PURPLE,
            BusYellow  => COLOR_BUS_YELLOW,
        }
    }
    pub(crate) fn to_color_on(self) -> [u8; 4] {
        match self {
            Trace::LatchOff => Trace::LatchOn,
            _ => self,
        }
        .to_color_raw()
    }
    pub(crate) fn to_color_off(self) -> [u8; 4] {
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
    pub(crate) fn is_gate(self) -> bool {
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
                | Trace::Vmem
        )
    }
    #[inline]
    pub(crate) fn is_logic(self) -> bool {
        self.is_wire() || self.is_gate()
    }
    ///// non logical
    //#[inline]
    pub(crate) fn is_passive(self) -> bool {
        !self.is_logic()
    }
}
impl Trace {
    // colors from file format
    #[rustfmt::skip]
    pub(crate) fn from_raw_color(color: [u8; 4]) -> Result<Self, [u8;4]> {
        use vcb_colors::*;
        use Trace::*;
        match color {
            COLOR_GRAY       => Ok(Gray),
            COLOR_WHITE      => Ok(White),
            COLOR_RED        => Ok(Red),
            COLOR_ORANGE1    => Ok(Orange1),
            COLOR_ORANGE2    => Ok(Orange2),
            COLOR_ORANGE3    => Ok(Orange3),
            COLOR_YELLOW     => Ok(Yellow),
            COLOR_GREEN1     => Ok(Green1),
            COLOR_GREEN2     => Ok(Green2),
            COLOR_CYAN1      => Ok(Cyan1),
            COLOR_CYAN2      => Ok(Cyan2),
            COLOR_BLUE1      => Ok(Blue1),
            COLOR_BLUE2      => Ok(Blue2),
            COLOR_PURPLE     => Ok(Purple),
            COLOR_MAGENTA    => Ok(Magenta),
            COLOR_PINK       => Ok(Pink),
            COLOR_WRITE      => Ok(Write),
            COLOR_EMPTY      => Ok(Empty),
            COLOR_CROSS      => Ok(Cross),
            COLOR_READ       => Ok(Read),
            COLOR_BUFFER     => Ok(Buffer),
            COLOR_AND        => Ok(And),
            COLOR_OR         => Ok(Or),
            COLOR_XOR        => Ok(Xor),
            COLOR_NOT        => Ok(Not),
            COLOR_NAND       => Ok(Nand),
            COLOR_NOR        => Ok(Nor),
            COLOR_XNOR       => Ok(Xnor),
            COLOR_LATCHON    => Ok(LatchOn),
            COLOR_LATCHOFF   => Ok(LatchOff),
            COLOR_CLOCK      => Ok(Clock),
            COLOR_LED        => Ok(Led),
            COLOR_ANNOTATION => Ok(Annotation),
            COLOR_FILLER     => Ok(Filler),

            COLOR_TUNNEL     => Ok(Tunnel),
            COLOR_MESH       => Ok(Mesh),
            COLOR_WIRELESS_0 => Ok(Wireless0),
            COLOR_WIRELESS_1 => Ok(Wireless1),
            COLOR_WIRELESS_2 => Ok(Wireless2),
            COLOR_WIRELESS_3 => Ok(Wireless3),
            COLOR_TIMER      => Ok(Timer),
            COLOR_RANDOM     => Ok(Random),
            COLOR_BREAK      => Ok(Break),
            COLOR_BUS_RED    => Ok(BusRed),
            COLOR_BUS_GREEN  => Ok(BusGreen),
            COLOR_BUS_BLUE   => Ok(BusBlue),
            COLOR_BUS_TEAL   => Ok(BusTeal),
            COLOR_BUS_PURPLE => Ok(BusPurple),
            COLOR_BUS_YELLOW => Ok(BusYellow),


            _ => Err(color),
            //_ => {dbg!(color); Ok(Gray)},
        }
    }

    ///// should this trace connect to other
    //#[inline]
    //pub(crate) fn will_connect(self, other: Self) -> bool {
    //    (self == other) || (self.is_wire() && other.is_wire())
    //}

    //#[inline]
    //pub(crate) fn to_gatetype_state(self) -> (GateType, bool) {
    //    //TODO: handle latch gatetype
    //    if self.is_wire() {
    //        (GateType::Cluster, false)
    //    } else {
    //        match self {
    //            Trace::Buffer | Trace::Or | Trace::Led => (GateType::Or, false),
    //            Trace::Not | Trace::Nor => (GateType::Nor, false),
    //            Trace::And => (GateType::And, false),
    //            Trace::Nand => (GateType::Nand, false),
    //            Trace::Xor => (GateType::Xor, false),
    //            Trace::Xnor => (GateType::Xnor, false),
    //            Trace::LatchOn => (GateType::Latch, true),
    //            Trace::LatchOff => (GateType::Latch, false),
    //            Trace::Vmem => (GateType::Interface(None), false),
    //            //_ => panic!("unsupported logic trace: {self:?}"),
    //            // ignore unsupported
    //            _ => (GateType::Cluster, false),
    //        }
    //    }
    //}
}
impl Trace {
    pub(crate) fn as_regular_emoji(self) -> &'static str {
        match self {
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
            | Trace::Pink => "⬛",
            Trace::Write => "  ",
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
            Trace::Vmem => "🔻",
            Trace::Clock => "🥞",
            Trace::Led => "🏁",
            Trace::Annotation => "🥚",
            Trace::Filler => "🌯",
            _ => unimplemented!("trace type not implemented"),
        }
    }

    pub(crate) fn as_discord_emoji(self) -> &'static str {
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
            Trace::LatchOff | Trace::Vmem => ":lt0:",
            Trace::Clock => "CLOCK",
            Trace::Led => ":led:",
            Trace::Annotation => ":non:",
            Trace::Filler => ":fil:",
            _ => unimplemented!("trace type missing"),
        }
    }
}
