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
    pub(crate) const COLOR_VMEM:       [u8; 4] = COLOR_LATCHOFF;
}
use super::*;
#[non_exhaustive]
#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash)]
pub(crate) enum Trace {
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
    Vmem,
}
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
            Trace::Vmem       => vcb_colors::COLOR_VMEM,
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
    // colors from file format
    #[rustfmt::skip]
    pub(crate) fn from_raw_color(color: [u8; 4]) -> Option<Self> {
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
    /// non logical
    #[inline]
    pub(crate) fn is_passive(self) -> bool {
        !self.is_logic()
    }

    /// should this trace connect to other
    #[inline]
    pub(crate) fn will_connect(self, other: Self) -> bool {
        (self == other) || (self.is_wire() && other.is_wire())
    }

    #[inline]
    pub(crate) fn to_gatetype_state(self) -> (GateType, bool) {
        //TODO: handle latch gatetype
        if self.is_wire() {
            (GateType::Cluster, false)
        } else {
            match self {
                Trace::Buffer | Trace::Or | Trace::Led => (GateType::Or, false),
                Trace::Not | Trace::Nor => (GateType::Nor, false),
                Trace::And => (GateType::And, false),
                Trace::Nand => (GateType::Nand, false),
                Trace::Xor => (GateType::Xor, false),
                Trace::Xnor => (GateType::Xnor, false),
                Trace::LatchOn => (GateType::Latch, true),
                Trace::LatchOff => (GateType::Latch, false),
                Trace::Vmem => (GateType::Interface(None), false),
                _ => panic!("unsupported logic trace: {self:?}"),
            }
        }
    }
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
        }
    }
}
