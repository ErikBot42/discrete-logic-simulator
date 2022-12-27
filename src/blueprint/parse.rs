
use super::*;

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
        let world_str: &json::JsonValue = &parsed["layers"][0];
        if let json::JsonValue::String(data) = world_str {
            let bytes = base64_decode(data)?;
            let (color_data, footer_bytes) = bytes.split_at(bytes.len() - BoardFooter::SIZE);
            let footer = BoardFooter::from_bytes(footer_bytes.try_into().context("")?)?;
            let data = zstd_decompress(color_data, footer.count()?)?;
            VcbPlainBoard::from_color_data(&data, footer.width, footer.height)
        } else {
            Err(anyhow!("json parsing went wrong"))
        }
    }

    /// # Errors
    /// Returns Err if input could not be parsed
    pub fn parse_compile<T: LogicSim>(
        input: VcbInput,
        optimize: bool,
    ) -> anyhow::Result<VcbBoard<T>> {
        let plain_board = match input {
            VcbInput::BlueprintLegacy(b) => Self::parse_legacy_blueprint(&b)?,
            VcbInput::Blueprint(b) => Self::parse_blueprint(&b)?,
            VcbInput::WorldLegacy(w) => Self::parse_legacy_world(&w)?,
            VcbInput::World(w) => Self::parse_world(&w)?,
        };
        Ok(VcbBoard::new(plain_board, optimize))
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
                    _ => Err(anyhow!("invalid footer layer"))?,
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

/// Decoded blueprint or board
#[derive(Debug)]
pub(crate) struct VcbPlainBoard {
    pub(crate) traces: Vec<Trace>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}
impl VcbPlainBoard {
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
