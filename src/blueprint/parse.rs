use super::*;
use json::JsonValue;

#[derive(Clone)]
pub enum VcbInput {
    BlueprintLegacy(String),
    Blueprint(String),
    WorldLegacy(String),
    World(String),
}

fn zstd_decompress(data: &[u8], num_traces: usize) -> std::io::Result<Vec<u8>> {
    const RGBA_SIZE: usize = 4;
    timed!(
        zstd::bulk::decompress(data, num_traces * RGBA_SIZE),
        "zstd decompress in: {:?}"
    )
}
fn base64_decode(data: &str) -> Result<Vec<u8>, base64::DecodeError> {
    timed!(
        BASE64_STANDARD.decode(data.trim()),
        "base64 decode in: {:?}"
    )
}

#[derive(Default)]
pub struct VcbParser {}
impl VcbParser {
    fn parse_legacy_blueprint(data: &str) -> anyhow::Result<VcbPlainBoard> {
        let bytes = base64_decode(data)?;
        let data_bytes = &bytes
            .get(..bytes.len() - BlueprintFooter::SIZE)
            .context("")?;
        let footer_bytes: [u8; BlueprintFooter::SIZE] = bytes
            .get(bytes.len() - BlueprintFooter::SIZE..bytes.len())
            .context("")?
            .try_into()?;
        let footer = BlueprintFooter::from_bytes(footer_bytes)?;
        if footer.layer != Layer::Logic {
            return Err(anyhow!("Blueprint layer not logic"));
        };
        let data = zstd_decompress(data_bytes, footer.count()?)?;
        if data.len() != footer.count()? * 4 {
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
            if bytes.is_empty() {
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
        let footer = BoardFooter::from_bytes(footer_bytes)?;
        let data = zstd_decompress(data_bytes, footer.count()?)?;
        VcbPlainBoard::from_color_data(&data, footer.width, footer.height)
    }
    fn parse_world(s: &str) -> anyhow::Result<VcbPlainBoard> {
        let parsed = json::parse(s)?;

        let vmem_settings = if let JsonValue::Array(vmem_settings) = &parsed["vmem_settings"] {
            let mut a: Vec<isize> = Vec::new();
            for v in vmem_settings {
                a.push(match v {
                    JsonValue::Number(number) => {
                        number.as_fixed_point_i64(0).unwrap().try_into().unwrap()
                    },
                    _ => return Err(anyhow!("invalid number format")),
                });
            }
            Ok(a)
        } else {
            Err(anyhow!("vmem_settings tag missing"))
        }?;
        let vmem_enabled = if let JsonValue::Boolean(vmem_enabled) = &parsed["is_vmem_enabled"] {
            Ok(*vmem_enabled)
        } else {
            Err(anyhow!("is_vmem_enabled tag missing"))
        }?;
        let vmem = VmemInfo::new(&vmem_settings, vmem_enabled)?;
        if let JsonValue::String(data) = &parsed["layers"][0] {
            let bytes = base64_decode(data)?;
            let (color_data, footer_bytes) = bytes.split_at(bytes.len() - BoardFooter::SIZE);
            let footer = BoardFooter::from_bytes(footer_bytes.try_into().context("")?)?;
            let data = zstd_decompress(color_data, footer.count()?)?;
            VcbPlainBoard::from_color_data_vmem(&data, footer.width, footer.height, vmem)
        } else {
            Err(anyhow!("layers tag missing"))
        }
    }

    fn parse(input: VcbInput) -> anyhow::Result<VcbPlainBoard> {
        Ok(match input {
            VcbInput::BlueprintLegacy(b) => Self::parse_legacy_blueprint(&b)?,
            VcbInput::Blueprint(b) => Self::parse_blueprint(&b)?,
            VcbInput::WorldLegacy(w) => Self::parse_legacy_world(&w)?,
            VcbInput::World(w) => Self::parse_world(&w)?,
        })
    }

    /// # Errors
    /// Returns Err if input could not be parsed
    pub fn parse_compile<T: LogicSim>(
        input: VcbInput,
        optimize: bool,
    ) -> anyhow::Result<VcbBoard<T>> {
        Ok(VcbBoard::new(Self::parse(input)?, optimize))
    }
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
        let mut n = || data.next().copied();
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
        let mut n = || data.next().copied();
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
#[derive(Debug)]
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
    fn from_bytes(bytes: [u8; Self::SIZE]) -> anyhow::Result<BoardFooterInfo> {
        let read = |i: usize| i32::from_le_bytes(from_fn(|k| bytes[k + (i * size_of::<i32>())]));
        BoardFooterInfo::new(
            &(Self {
                height_type: read(0),
                height: read(1),
                width_type: read(2),
                width: read(3),
                bytes_type: read(4),
                bytes: read(5),
            }),
        )
    }
}

/// Useable cleaned footer data.
#[derive(Debug)]
struct BoardFooterInfo {
    width: usize,
    height: usize,
}

impl BoardFooterInfo {
    fn new(footer: &BoardFooter) -> anyhow::Result<Self> {
        if usize::try_from(footer.bytes)?
            != usize::try_from(footer.height)?
                .checked_mul(usize::try_from(footer.width)?)
                .context("")?
                .checked_mul(size_of::<u32>())
                .context("")?
            || footer.width_type != 2
            || footer.bytes_type != 2
            || footer.height_type != 2
        {
            Err(anyhow!("Footer data invalid: {footer:?}"))
        } else {
            Ok(Self {
                width: footer.width.try_into()?,
                height: footer.height.try_into()?,
            })
        }
    }
    fn count(&self) -> anyhow::Result<usize> {
        self.width.checked_mul(self.height).context("")
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
    fn from_bytes(bytes: [u8; Self::SIZE]) -> anyhow::Result<BlueprintFooterInfo> {
        let read = |i: usize| i32::from_le_bytes(from_fn(|k| bytes[k + (i * size_of::<i32>())]));
        BlueprintFooterInfo::new(
            &(Self {
                height_type: read(0),
                height: read(1),
                width_type: read(2),
                width: read(3),
                bytes_type: read(4),
                bytes: read(5),
                layer_type: read(6),
                layer: read(7),
            }),
        )
    }
}

/// Useable cleaned footer data.
#[derive(Debug)]
struct BlueprintFooterInfo {
    width: usize,
    height: usize,
    layer: Layer,
}
impl BlueprintFooterInfo {
    fn new(footer: &BlueprintFooter) -> anyhow::Result<BlueprintFooterInfo> {
        if usize::try_from(footer.bytes)?
            != usize::try_from(footer.height)?
                .checked_mul(usize::try_from(footer.width)?)
                .context("")?
                .checked_mul(size_of::<u32>())
                .context("")?
            || footer.width_type != 2
            || footer.bytes_type != 2
            || footer.height_type != 2
        {
            Err(anyhow!("Footer data invalid: {footer:?}"))
        } else {
            Ok(BlueprintFooterInfo {
                width: footer.width.try_into()?,
                height: footer.height.try_into()?,
                layer: match footer.layer {
                    65_536 => Layer::Logic,
                    131_072 => Layer::On,
                    262_144 => Layer::Off,
                    _ => return Err(anyhow!("invalid footer layer")),
                },
            })
        }
    }
    fn count(&self) -> anyhow::Result<usize> {
        self.width.checked_mul(self.height).context("")
    }
}
#[derive(Debug, PartialEq, Eq)]
enum Layer {
    Logic,
    On,
    Off,
}
// bits, position xy, offset xy, size xy
#[derive(Debug)]
pub(crate) struct VmemInfoInner {
    bits: isize,
    position: (isize, isize), // should only be positive
    offset: (isize, isize),
    size: (isize, isize), // should only be positive
}
impl VmemInfoInner {
    // calc pos with given offset
    fn bit_pos_offset(&self, bit: isize, (ox, oy): (isize, isize)) -> Option<(isize, isize)> {
        let (x, y) = self.bit_pos(bit)?;
        Some((x.checked_add(ox)?, y.checked_add(oy)?))
    }

    // calc base pos of given vmem bit
    fn bit_pos(&self, bit: isize) -> Option<(isize, isize)> {
        Some((
            self.position
                .0
                .checked_sub(bit.checked_mul(self.offset.0)?)?,
            self.position
                .1
                .checked_sub(bit.checked_mul(self.offset.1)?)?,
        ))
    }
    fn new(a: [isize; 7]) -> anyhow::Result<Self> {
        let v = Self {
            bits: a[0],
            position: (a[1], a[2]),
            offset: (a[3], a[4]),
            size: (a[5], a[6]),
        };
        if v.position.0 < 0
            || v.position.1 < 0
            || v.size.0 < 0
            || v.size.1 < 0
            || v.bits < 0
            || (v.offset.0.abs() <= v.size.0.abs() && v.offset.1.abs() <= v.size.1.abs())
        {
            Err(anyhow!("Invalid vmem: {v:?}"))
        } else {
            Ok(v)
        }
    }
}
#[derive(Debug)]
pub(crate) struct VmemInfo {
    pub(crate) contents: VmemInfoInner,
    pub(crate) address: VmemInfoInner,
}
impl VmemInfo {
    fn new(a: &[isize], enabled: bool) -> anyhow::Result<Option<Self>> {
        Ok(if enabled {
            Some(Self {
                contents: VmemInfoInner::new(a.get(0..7).context("")?.try_into()?)?,
                address: VmemInfoInner::new(a.get(7..14).context("")?.try_into()?)?,
            })
        } else {
            None
        })
    }
}

/// Decoded blueprint or board
#[derive(Debug)]
#[non_exhaustive]
pub(crate) struct VcbPlainBoard {
    pub(crate) traces: Vec<Trace>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) vmem: Option<VmemInfo>,
}

#[derive(Debug)]
pub struct ArbitraryVcbPlainBoard {
    pub(crate) board: VcbPlainBoard,
}
impl<'a> arbitrary::Arbitrary<'a> for ArbitraryVcbPlainBoard {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let len: usize = std::cmp::max(u.arbitrary_len::<u8>()?, 1);
        let width: usize = u.int_in_range(1..=len)?;
        let height = len / width;

        let mut traces: Vec<Trace> = Vec::new();
        for _ in 0..(width * height) {
            traces.push(trace::arbitrary_trace(u)?);
        }

        Ok(ArbitraryVcbPlainBoard {
            board: VcbPlainBoard {
                traces,
                width,
                height,
                vmem: None,
            },
        })
    }
}

impl VcbPlainBoard {
    fn pos_to_index(&self, (x, y): (isize, isize)) -> Option<usize> {
        let x: usize = x.try_into().ok()?;
        let y: usize = y.try_into().ok()?;
        (x < self.width && y < self.height).then_some(x + y * self.width)
    }
    // Iterate origin of all vmem bits, contents then address
    //fn iter_vmem_bits(&self) {
    //    todo!()
    //}
    fn apply_vmem(mut self) -> anyhow::Result<Self> {
        match &self.vmem {
            None => Ok(self),
            Some(vmem) => {
                for vmem in [&vmem.contents, &vmem.address] {
                    for dy in 0..vmem.size.1 {
                        for bit in 0..vmem.bits {
                            for dx in 0..vmem.size.0 {
                                let index = vmem
                                    .bit_pos_offset(bit, (dx, dy))
                                    .and_then(|s| self.pos_to_index(s))
                                    .context("vmem position bounds")?;
                                let trace =
                                    self.traces.get_mut(index).context("vmem index bounds")?;
                                assert_ne!(*trace, Trace::Vmem, "implementation error");
                                *trace = Trace::Vmem;
                            }
                        }
                    }
                }
                Ok(self)
            },
        }
    }
    fn from_color_data(data: &[u8], width: usize, height: usize) -> anyhow::Result<Self> {
        Self::from_color_data_vmem(data, width, height, None)
    }
    fn from_color_data_vmem(
        data: &[u8],
        width: usize,
        height: usize,
        vmem: Option<VmemInfo>,
    ) -> anyhow::Result<Self> {
        timed!(
            {
                let traces = data
                    .array_chunks::<4>()
                    .copied()
                    .map(Trace::from_raw_color)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|c| anyhow::anyhow!("invalid color: {c:?}"))?;
                if traces.len() == width * height {
                    VcbPlainBoard {
                        traces,
                        width,
                        height,
                        vmem,
                    }
                    .apply_vmem()
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
