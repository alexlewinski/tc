//! Reader for nnc SQLite tensor checkpoints (the `.ckpt` files Draw Things stores models in).
//!
//! The format is defined by ccv/nnc (`ccv_nnc_tensor_io.c`, BSD-3) and s4nnc (`Store.swift`,
//! BSD-3); see docs/PROVENANCE.md. A checkpoint is a SQLite database with a table
//!
//! ```text
//! tensors(name TEXT, type INTEGER, format INTEGER, datatype INTEGER, dim BLOB, data BLOB)
//! ```
//!
//! * `type` = codec identifier << 32 | memory type; `datatype` = reserved << 32 | element type
//!   (nnc codes); `dim` = int32 dimensions; bit 32 of `format` marks a blob continued in
//!   `tensor_splits(name, part, data)`.
//! * Codec identifiers: 0 raw, 0x217 raw deflate, 0xf7217 fpzip, 0x8a1e8b 8-bit palettized
//!   ("q8p": u32 block size, then per block 256 palette entries of the element type followed by
//!   one u8 index per element). Bit 0x1000_0000 means the payload lives in an external store:
//!   the row then holds `u64 offset, u64 length` (q8p: `u32 block size, pad, u64 offset, u64
//!   length`).
//! * The external store is either a sidecar file `<checkpoint>-tensordata`, or data appended to
//!   the database itself ("trailer store"), whose start is the big-endian u32 at bytes 60..64.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use nnc::{DataType, HostTensor};
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags, OptionalExtension};

pub const CODEC_RAW: u32 = 0;
pub const CODEC_ZIP: u32 = 0x217;
pub const CODEC_FPZIP: u32 = 0xf7217;
pub const CODEC_EZM7: u32 = 0x511;
pub const CODEC_Q8P: u32 = 0x8a1e8b;
/// Payload stored in the external store.
pub const EXTERNAL_BIT: u32 = 0x1000_0000;
/// Modifier on i8x codecs (row-wise Hadamard rotation).
const HADAMARD_BIT: u32 = 0x0100_0000;
const SPLIT_FORMAT: i64 = 1 << 32;

/// Metadata of one stored tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct CkptTensorInfo {
    /// Codec identifier without the external-store bit.
    pub codec: u32,
    pub external: bool,
    /// Element type of the decoded tensor (and of q8p palettes).
    pub dtype: DataType,
    pub shape: Vec<usize>,
    /// nnc tensor format (1 = NCHW, 2 = NHWC, 4 = CHWN).
    pub format: i32,
    split: bool,
}

impl CkptTensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn codec_name(&self) -> String {
        let base = match self.codec & !HADAMARD_BIT {
            CODEC_RAW => "raw".to_string(),
            CODEC_ZIP => "zip".into(),
            CODEC_FPZIP => "fpzip".into(),
            CODEC_EZM7 => "ezm7".into(),
            0x8a1e4b => "q4p".into(),
            0x8a1e5b => "q5p".into(),
            0x8a1e6b => "q6p".into(),
            0x8a1e7b => "q7p".into(),
            CODEC_Q8P => "q8p".into(),
            0x8a1e9b => "i8x".into(),
            c @ 0x8a1eab..=0x8a1eb4 => format!("i8x-k({c:#x})"),
            c => format!("{c:#x}"),
        };
        if self.external {
            format!("{base}+external")
        } else {
            base
        }
    }
}

/// A decoded tensor: dense, or kept 8-bit palettized (the layout nnc's GPU kernels consume).
#[derive(Clone, Debug)]
pub enum CkptTensor {
    Dense(HostTensor),
    Palettized { data: Vec<u8>, dtype: DataType, shape: Vec<usize>, block: usize },
}

impl CkptTensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            CkptTensor::Dense(t) => t.shape(),
            CkptTensor::Palettized { shape, .. } => shape,
        }
    }

    /// Expand to a dense host tensor of `dtype`.
    pub fn to_host(&self, dtype: DataType) -> HostTensor {
        match self {
            CkptTensor::Dense(t) => {
                if t.dtype() == dtype {
                    t.clone()
                } else {
                    t.to_dtype(dtype)
                }
            }
            CkptTensor::Palettized { data, dtype: pdtype, shape, block } => {
                let t = depalettize8(data, *pdtype, shape, *block);
                if t.dtype() == dtype {
                    t
                } else {
                    t.to_dtype(dtype)
                }
            }
        }
    }
}

/// Expand q8p data (per block: 256 palette entries of `dtype`, then one u8 index per element).
pub fn depalettize8(data: &[u8], dtype: DataType, shape: &[usize], block: usize) -> HostTensor {
    let n: usize = shape.iter().product();
    let es = dtype.size();
    let mut out = vec![0u8; n * es];
    out.par_chunks_mut(block * es).enumerate().for_each(|(bi, dst)| {
        let src = &data[bi * (256 * es + block)..];
        let (palette, idx) = src.split_at(256 * es);
        for (i, d) in dst.chunks_exact_mut(es).enumerate() {
            let p = idx[i] as usize * es;
            d.copy_from_slice(&palette[p..p + es]);
        }
    });
    HostTensor::from_bytes(out, dtype, shape)
}

/// Decode ezm7 (f16 only): `i32 deflated size`, the deflated 5-bit exponents (one byte per
/// element), then one byte per element holding the sign bit and the top 7 mantissa bits.
pub fn ezm7_decode(payload: &[u8], n: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let zsize = i32::from_le_bytes(payload.get(0..4).ok_or_else(|| anyhow!("ezm7: empty payload"))?.try_into().unwrap());
    let zsize = usize::try_from(zsize).map_err(|_| anyhow!("ezm7: bad exponent size"))?;
    let rest = payload.get(4..).unwrap_or(&[]);
    if rest.len() < zsize + n {
        bail!("ezm7: payload too short");
    }
    let mut exponents = Vec::with_capacity(n);
    flate2::read::DeflateDecoder::new(&rest[..zsize]).take(n as u64).read_to_end(&mut exponents)?;
    if exponents.len() != n {
        bail!("ezm7: {} exponents, expected {n}", exponents.len());
    }
    let mut out = Vec::with_capacity(n * 2);
    for (e, m) in exponents.iter().zip(&rest[zsize..zsize + n]) {
        let m = *m as u16;
        let v = ((m >> 7) << 15) | (((*e as u16) & 0x1f) << 10) | ((m & 0x7f) << 3);
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// Re-encode the palettes of q8p data in another element type (indices are unchanged).
pub fn convert_palettes(data: &[u8], from: DataType, to: DataType, numel: usize, block: usize) -> Vec<u8> {
    let (fs, ts) = (from.size(), to.size());
    let blocks = numel.div_ceil(block);
    let mut out = Vec::with_capacity(q8p_size(numel, to, block));
    for b in 0..blocks {
        let src = &data[b * (256 * fs + block)..];
        let n = block.min(numel - b * block);
        let pal = HostTensor::from_bytes(src[..256 * fs].to_vec(), from, &[256]).to_dtype(to);
        debug_assert_eq!(pal.bytes().len(), 256 * ts);
        out.extend_from_slice(pal.bytes());
        out.extend_from_slice(&src[256 * fs..256 * fs + n]);
    }
    out
}

/// Size in bytes of q8p data for `numel` elements.
pub fn q8p_size(numel: usize, dtype: DataType, block: usize) -> usize {
    numel.div_ceil(block) * 256 * dtype.size() + numel
}

struct ExternalStore {
    path: PathBuf,
    base: u64,
    map: Mmap,
}

/// An open checkpoint.
pub struct CkptFile {
    path: PathBuf,
    conn: Mutex<Connection>,
    index: BTreeMap<String, CkptTensorInfo>,
    external: Option<ExternalStore>,
}

/// Whether `path` starts with the SQLite header (a nnc / Draw Things checkpoint).
pub fn is_ckpt(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 16];
    File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"SQLite format 3\0"
}

/// Offset of the trailer store (data appended after the SQLite pages), if any.
pub fn trailer_offset(path: &Path) -> Option<u64> {
    use std::io::Read;
    let mut header = [0u8; 64];
    let mut f = File::open(path).ok()?;
    f.read_exact(&mut header).ok()?;
    let offset = u32::from_be_bytes(header[60..64].try_into().unwrap()) as u64;
    let size = f.metadata().ok()?.len();
    (offset >= 100 && offset <= size).then_some(offset)
}

/// Path of the sidecar external store of a checkpoint (`<file>-tensordata`).
pub fn sidecar_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push("-tensordata");
    PathBuf::from(s)
}

fn sqlite_uri(path: &Path) -> String {
    let mut uri = String::from("file:");
    for b in path.to_string_lossy().bytes() {
        match b {
            b'?' | b'#' | b'%' => uri.push_str(&format!("%{b:02X}")),
            _ => uri.push(b as char),
        }
    }
    // Read-only and immutable: never create journal / WAL files next to the user's models.
    uri.push_str("?mode=ro&immutable=1");
    uri
}

impl CkptFile {
    pub fn open(path: &Path) -> Result<CkptFile> {
        if !is_ckpt(path) {
            bail!("{} is not a SQLite tensor checkpoint", path.display());
        }
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(sqlite_uri(path), flags).with_context(|| format!("opening {}", path.display()))?;
        let mut index = BTreeMap::new();
        {
            let mut stmt = conn
                .prepare("SELECT name, type, format, datatype, dim FROM tensors")
                .with_context(|| format!("{}: no tensors table", path.display()))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let name: String = row.get(0)?;
                let typ: i64 = row.get(1)?;
                let format: i64 = row.get(2)?;
                let datatype: i64 = row.get(3)?;
                let dim: Vec<u8> = row.get(4)?;
                let identifier = ((typ >> 32) & 0xffff_ffff) as u32;
                let raw_dtype = (datatype & 0xffff_ffff) as i32;
                let dtype = DataType::from_raw(raw_dtype)
                    .ok_or_else(|| anyhow!("{}: tensor {name} has unsupported datatype {raw_dtype:#x}", path.display()))?;
                let shape: Vec<usize> = dim
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .take_while(|d| *d > 0)
                    .map(|d| d as usize)
                    .collect();
                index.insert(
                    name,
                    CkptTensorInfo {
                        codec: identifier & !EXTERNAL_BIT,
                        external: identifier & EXTERNAL_BIT != 0,
                        dtype,
                        shape,
                        format: (format & 0xffff_ffff) as i32,
                        split: format & SPLIT_FORMAT != 0,
                    },
                );
            }
        }
        let external = if index.values().any(|i| i.external) {
            let (store_path, base) = match trailer_offset(path) {
                Some(off) => (path.to_path_buf(), off),
                None => (sidecar_path(path), 0),
            };
            let file = File::open(&store_path).with_context(|| {
                format!("{} keeps its weights in {}, which could not be opened", path.display(), store_path.display())
            })?;
            let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", store_path.display()))?;
            Some(ExternalStore { path: store_path, base, map })
        } else {
            None
        };
        Ok(CkptFile { path: path.to_path_buf(), conn: Mutex::new(conn), index, external })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The external store in use (sidecar file or the checkpoint itself), if any.
    pub fn external_store(&self) -> Option<&Path> {
        self.external.as_ref().map(|e| e.path.as_path())
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Option<&CkptTensorInfo> {
        self.index.get(name)
    }

    fn blob(&self, name: &str, split: bool) -> Result<Vec<u8>> {
        use std::io::Read;
        let conn = self.conn.lock().unwrap();
        let rowid: i64 = conn
            .query_row("SELECT rowid FROM tensors WHERE name = ?1", [name], |r| r.get(0))
            .optional()?
            .ok_or_else(|| anyhow!("tensor {name} not found"))?;
        // Incremental blob I/O: blobs can exceed SQLite's 1 GB value limit (e.g. embeddings).
        let mut data = Vec::new();
        conn.blob_open(rusqlite::DatabaseName::Main, "tensors", "data", rowid, true)?.read_to_end(&mut data)?;
        if split {
            let mut stmt = conn.prepare("SELECT rowid FROM tensor_splits WHERE name = ?1 ORDER BY part")?;
            let ids: Vec<i64> = stmt.query_map([name], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
            for id in ids {
                conn.blob_open(rusqlite::DatabaseName::Main, "tensor_splits", "data", id, true)?.read_to_end(&mut data)?;
            }
        }
        Ok(data)
    }

    fn external_bytes(&self, name: &str, offset: u64, length: u64) -> Result<&[u8]> {
        let ext = self.external.as_ref().ok_or_else(|| anyhow!("tensor {name}: no external store"))?;
        let start = ext.base.checked_add(offset).ok_or_else(|| anyhow!("tensor {name}: bad offset"))? as usize;
        let end = start.checked_add(length as usize).ok_or_else(|| anyhow!("tensor {name}: bad length"))?;
        if end > ext.map.len() {
            bail!(
                "tensor {name}: external data {start}..{end} is beyond the end of {} ({} bytes); the file may be incomplete",
                ext.path.display(),
                ext.map.len()
            );
        }
        Ok(&ext.map[start..end])
    }

    /// Read a tensor, keeping q8p tensors palettized.
    pub fn read_raw(&self, name: &str) -> Result<CkptTensor> {
        let info = self.index.get(name).ok_or_else(|| anyhow!("tensor {name} not found in {}", self.path.display()))?;
        let blob = self.blob(name, info.split)?;
        let n = info.numel();
        let u64_at = |b: &[u8], o: usize| -> Result<u64> {
            Ok(u64::from_le_bytes(b.get(o..o + 8).ok_or_else(|| anyhow!("tensor {name}: truncated header"))?.try_into().unwrap()))
        };
        if info.codec == CODEC_Q8P {
            let block = u32::from_le_bytes(blob.get(0..4).ok_or_else(|| anyhow!("tensor {name}: empty q8p data"))?.try_into().unwrap()) as usize;
            if block == 0 {
                bail!("tensor {name}: q8p block size 0");
            }
            let need = q8p_size(n, info.dtype, block);
            let data = if info.external {
                let (offset, length) = (u64_at(&blob, 8)?, u64_at(&blob, 16)?);
                self.external_bytes(name, offset, length)?.get(..need).map(|s| s.to_vec())
            } else {
                blob.get(4..4 + need).map(|s| s.to_vec())
            };
            let data = data.ok_or_else(|| anyhow!("tensor {name}: q8p data shorter than {need} bytes"))?;
            return Ok(CkptTensor::Palettized { data, dtype: info.dtype, shape: info.shape.clone(), block });
        }
        let payload: Vec<u8>;
        let payload_ref: &[u8] = if info.external {
            let (offset, length) = (u64_at(&blob, 0)?, u64_at(&blob, 8)?);
            self.external_bytes(name, offset, length)?
        } else {
            payload = blob;
            &payload
        };
        let wire = match info.dtype {
            DataType::F32 => tc_wire::WireType::F32,
            DataType::F16 => tc_wire::WireType::F16,
            DataType::BF16 => tc_wire::WireType::BF16,
            DataType::F64 => tc_wire::WireType::F64,
            DataType::I32 => tc_wire::WireType::I32,
            DataType::I64 => tc_wire::WireType::I64,
            DataType::U8 => tc_wire::WireType::U8,
        };
        let bytes = match info.codec {
            CODEC_RAW | CODEC_ZIP | CODEC_FPZIP => tc_wire::decode_payload(info.codec, payload_ref, wire, &info.shape)
                .with_context(|| format!("decoding tensor {name}"))?,
            CODEC_EZM7 if info.dtype == DataType::F16 => ezm7_decode(payload_ref, n).with_context(|| format!("decoding tensor {name}"))?,
            _ => bail!(
                "tensor {name} in {} uses the {} encoding, which is not supported (supported: raw, zip, fpzip, q8p)",
                self.path.display(),
                info.codec_name()
            ),
        };
        Ok(CkptTensor::Dense(HostTensor::from_bytes(bytes, info.dtype, &info.shape)))
    }

    /// Read a tensor as a dense host tensor of `dtype` (q8p tensors are expanded).
    pub fn read(&self, name: &str, dtype: DataType) -> Result<HostTensor> {
        Ok(self.read_raw(name)?.to_host(dtype))
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.read(name, DataType::F32)?.to_f32_vec())
    }
}

/// How a canonical tensor is assembled: the listed checkpoint tensors concatenated along their
/// first dimension, viewed as `shape`.
#[derive(Clone, Debug, PartialEq)]
pub struct Recipe {
    pub parts: Vec<String>,
    pub shape: Vec<usize>,
}

impl Recipe {
    pub fn direct(name: impl Into<String>, shape: &[usize]) -> Recipe {
        Recipe { parts: vec![name.into()], shape: shape.to_vec() }
    }
}

/// A checkpoint whose tensors are exposed under another (canonical) naming scheme.
pub struct MappedCkpt {
    pub file: CkptFile,
    recipes: BTreeMap<String, Recipe>,
}

impl MappedCkpt {
    /// Checks that every part exists and that the element counts add up.
    pub fn new(file: CkptFile, recipes: BTreeMap<String, Recipe>) -> Result<MappedCkpt> {
        for (name, r) in &recipes {
            let mut n = 0;
            for p in &r.parts {
                let info = file.info(p).ok_or_else(|| anyhow!("{}: missing tensor {p} (for {name})", file.path().display()))?;
                n += info.numel();
            }
            let want: usize = r.shape.iter().product();
            if n != want {
                bail!("{}: tensors {:?} have {n} elements, expected {:?} for {name}", file.path().display(), r.parts, r.shape);
            }
        }
        Ok(MappedCkpt { file, recipes })
    }

    /// Checkpoint tensors no recipe uses.
    pub fn unused(&self) -> Vec<String> {
        let used: std::collections::HashSet<&String> = self.recipes.values().flat_map(|r| r.parts.iter()).collect();
        self.file.keys().filter(|k| !used.contains(k)).cloned().collect()
    }

    pub fn recipe(&self, name: &str) -> Option<&Recipe> {
        self.recipes.get(name)
    }
}

impl crate::weights::TensorSource for MappedCkpt {
    fn names(&self) -> Vec<String> {
        self.recipes.keys().cloned().collect()
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.recipes.get(name).map(|r| r.shape.clone())
    }

    fn read(&self, name: &str, dtype: DataType) -> Result<HostTensor> {
        Ok(self.read_raw(name)?.to_host(dtype))
    }

    fn read_raw(&self, name: &str) -> Result<CkptTensor> {
        let r = self.recipes.get(name).ok_or_else(|| anyhow!("tensor {name} not found in {}", self.file.path().display()))?;
        let parts: Vec<CkptTensor> = r.parts.iter().map(|p| self.file.read_raw(p)).collect::<Result<_>>()?;
        if parts.len() == 1 {
            return Ok(match parts.into_iter().next().unwrap() {
                CkptTensor::Dense(t) => CkptTensor::Dense(t.reshape(&r.shape)),
                CkptTensor::Palettized { data, dtype, block, .. } => CkptTensor::Palettized { data, dtype, shape: r.shape.clone(), block },
            });
        }
        // Palettized parts whose sizes are whole blocks concatenate byte-wise.
        let first = match &parts[0] {
            CkptTensor::Palettized { dtype, block, .. } => Some((*dtype, *block)),
            _ => None,
        };
        let aligned = first.is_some()
            && parts.iter().all(|p| match p {
                CkptTensor::Palettized { dtype, block, shape, .. } => {
                    Some((*dtype, *block)) == first && shape.iter().product::<usize>() % *block == 0
                }
                _ => false,
            });
        if aligned {
            let (dtype, block) = first.unwrap();
            let mut data = Vec::new();
            for p in parts {
                if let CkptTensor::Palettized { data: d, .. } = p {
                    data.extend_from_slice(&d);
                }
            }
            return Ok(CkptTensor::Palettized { data, dtype, shape: r.shape.clone(), block });
        }
        let dtype = match &parts[0] {
            CkptTensor::Dense(t) => t.dtype(),
            CkptTensor::Palettized { dtype, .. } => *dtype,
        };
        let mut bytes = Vec::new();
        for p in &parts {
            bytes.extend_from_slice(p.to_host(dtype).bytes());
        }
        Ok(CkptTensor::Dense(HostTensor::from_bytes(bytes, dtype, &r.shape)))
    }
}

/// Writes nnc tensor checkpoints in the layout Draw Things uses (raw / zip / fpzip / q8p, inline
/// or with a `-tensordata` sidecar). Used to build test models.
pub struct CkptWriter {
    conn: Connection,
    external: Option<(PathBuf, Vec<u8>)>,
}

impl CkptWriter {
    pub fn new(path: &Path, sidecar: bool) -> Result<CkptWriter> {
        let _ = std::fs::remove_file(path);
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE tensors (name TEXT, type INTEGER, format INTEGER, datatype INTEGER, dim BLOB, data BLOB, PRIMARY KEY (name));
             CREATE TABLE tensor_splits (name TEXT, part INTEGER, data BLOB, PRIMARY KEY (name, part));",
        )?;
        Ok(CkptWriter { conn, external: sidecar.then(|| (sidecar_path(path), vec![])) })
    }

    pub(crate) fn insert(&self, name: &str, identifier: u32, format: i64, dtype: DataType, shape: &[usize], data: &[u8]) -> Result<()> {
        let mut dim = vec![0u8; 12 * 4];
        for (i, d) in shape.iter().enumerate() {
            dim[i * 4..i * 4 + 4].copy_from_slice(&(*d as i32).to_le_bytes());
        }
        self.conn
            .execute(
                "INSERT INTO tensors VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![name, ((identifier as i64) << 32) | 1, format, dtype.to_raw() as i64, dim, data],
            )
            .with_context(|| format!("writing {name}"))?;
        Ok(())
    }

    fn push_external(&mut self, bytes: &[u8]) -> u64 {
        let (_, buf) = self.external.as_mut().expect("no external store");
        // nnc aligns each payload to its chunk size.
        let aligned = buf.len().div_ceil(64) * 64;
        buf.resize(aligned, 0);
        buf.extend_from_slice(bytes);
        aligned as u64
    }

    pub fn dense(&mut self, name: &str, t: &HostTensor, codec: u32) -> Result<()> {
        let bytes = match codec {
            CODEC_RAW => t.bytes().to_vec(),
            CODEC_ZIP | CODEC_FPZIP => {
                let wt = tc_wire::WireTensor::from_f32(
                    &t.to_f32_vec(),
                    t.shape(),
                    if t.dtype() == DataType::F16 { tc_wire::WireType::F16 } else { tc_wire::WireType::F32 },
                );
                let c = if codec == CODEC_ZIP { tc_wire::Compression::Zip } else { tc_wire::Compression::FpzipOrZip };
                let enc = tc_wire::encode(&wt, c);
                if u32::from_le_bytes(enc[0..4].try_into().unwrap()) != codec {
                    bail!("{name}: data is not compressible with codec {codec:#x}");
                }
                enc[68..].to_vec()
            }
            _ => bail!("{name}: cannot write codec {codec:#x}"),
        };
        if self.external.is_some() {
            let off = self.push_external(&bytes);
            let mut row = off.to_le_bytes().to_vec();
            row.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            let id = if codec == CODEC_RAW { EXTERNAL_BIT } else { codec | EXTERNAL_BIT };
            self.insert(name, id, 1, t.dtype(), t.shape(), &row)
        } else {
            self.insert(name, codec, 1, t.dtype(), t.shape(), &bytes)
        }
    }

    pub fn q8p(&mut self, name: &str, t: &HostTensor, block: usize) -> Result<()> {
        let q = nnc::quant::palettize8_fast(t, block);
        if self.external.is_some() {
            let off = self.push_external(&q);
            let mut row = (block as u32).to_le_bytes().to_vec();
            row.extend_from_slice(&[0; 4]);
            row.extend_from_slice(&off.to_le_bytes());
            row.extend_from_slice(&(q.len() as u64).to_le_bytes());
            self.insert(name, CODEC_Q8P | EXTERNAL_BIT, 1, t.dtype(), t.shape(), &row)
        } else {
            let mut row = (block as u32).to_le_bytes().to_vec();
            row.extend_from_slice(&q);
            self.insert(name, CODEC_Q8P, 1, t.dtype(), t.shape(), &row)
        }
    }

    /// A raw tensor whose blob is split across `tensors` and `tensor_splits`.
    pub fn split(&mut self, name: &str, t: &HostTensor, parts: usize) -> Result<()> {
        let b = t.bytes();
        let chunk = b.len().div_ceil(parts + 1);
        self.insert(name, 0, 1 | SPLIT_FORMAT, t.dtype(), t.shape(), &b[..chunk])?;
        for (i, c) in b[chunk..].chunks(chunk).enumerate() {
            self.conn.execute("INSERT INTO tensor_splits VALUES (?1, ?2, ?3)", rusqlite::params![name, i as i64, c])?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        if let Some((p, buf)) = &self.external {
            std::fs::write(p, buf)?;
        }
        Ok(())
    }
}


#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Test helper: [`CkptWriter`] with panics instead of errors.
    pub struct Writer(pub CkptWriter);

    impl Writer {
        pub fn new(path: &Path, sidecar: bool) -> Writer {
            Writer(CkptWriter::new(path, sidecar).unwrap())
        }
        pub fn dense(&mut self, name: &str, t: &HostTensor, codec: u32) {
            self.0.dense(name, t, codec).unwrap()
        }
        pub fn q8p(&mut self, name: &str, t: &HostTensor, block: usize) {
            self.0.q8p(name, t, block).unwrap()
        }
        pub fn split(&mut self, name: &str, t: &HostTensor, parts: usize) {
            self.0.split(name, t, parts).unwrap()
        }
        fn insert(&self, name: &str, identifier: u32, format: i64, dtype: DataType, shape: &[usize], data: &[u8]) {
            self.0.insert(name, identifier, format, dtype, shape, data).unwrap()
        }
        pub fn finish(self) {
            self.0.finish().unwrap()
        }
    }

    pub fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tc-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    fn values(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.731).sin() * 0.05).collect()
    }

    #[test]
    fn reads_all_supported_codecs_inline_and_external() {
        for sidecar in [false, true] {
            let path = tmp(&format!("codecs_{sidecar}.ckpt"));
            let mut w = Writer::new(&path, sidecar);
            let a = HostTensor::from_f32(&values(3 * 5, 1.0), &[3, 5]).to_dtype(DataType::F16);
            let b = HostTensor::from_f32(&values(64 * 40, 2.0), &[64, 40]);
            let c = HostTensor::from_f32(&values(16 * 16, 3.0), &[16, 16]).to_dtype(DataType::F16);
            // Not a multiple of the block size: the last block is partial.
            let q = HostTensor::from_f32(&values(300 * 70, 4.0), &[300, 70]).to_dtype(DataType::F16);
            w.dense("raw", &a, CODEC_RAW);
            w.dense("zip", &b, CODEC_ZIP);
            w.dense("fpzip", &c, CODEC_FPZIP);
            w.q8p("q8p", &q, 4096);
            if !sidecar {
                w.split("split", &b, 3);
            }
            w.finish();

            let f = CkptFile::open(&path).unwrap();
            assert_eq!(f.external_store().is_some(), sidecar);
            assert_eq!(f.info("raw").unwrap().shape, vec![3, 5]);
            assert_eq!(f.read("raw", DataType::F16).unwrap().bytes(), a.bytes());
            assert_eq!(f.read("zip", DataType::F32).unwrap().bytes(), b.bytes());
            assert_eq!(f.read("fpzip", DataType::F16).unwrap().bytes(), c.bytes());
            if !sidecar {
                assert_eq!(f.read("split", DataType::F32).unwrap().bytes(), b.bytes());
            }
            let raw = f.read_raw("q8p").unwrap();
            assert!(matches!(raw, CkptTensor::Palettized { block: 4096, .. }));
            let got = raw.to_host(DataType::F32).to_f32_vec();
            let want = q.to_f32_vec();
            let max = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
            assert!(max < 2e-3, "q8p error {max}");
            // Same as nnc's own depalettization.
            let ours = depalettize8(match &raw { CkptTensor::Palettized { data, .. } => data, _ => unreachable!() }, DataType::F16, &[300, 70], 4096);
            let theirs = nnc::quant::depalettize(&nnc::quant::palettize8_fast(&q, 4096), DataType::F16, &[300, 70], 8, 4096);
            assert_eq!(ours.bytes(), theirs.bytes());
            assert!(f.read("missing", DataType::F32).is_err());
        }
    }

    #[test]
    fn mapped_concatenates_rows() {
        use crate::weights::TensorSource;
        let path = tmp("mapped.ckpt");
        let mut w = Writer::new(&path, false);
        let a = HostTensor::from_f32(&values(64 * 64, 1.0), &[64, 64]).to_dtype(DataType::F16);
        let b = HostTensor::from_f32(&values(64 * 64, 2.0), &[64, 64]).to_dtype(DataType::F16);
        w.q8p("a", &a, 1024);
        w.q8p("b", &b, 1024);
        w.dense("c", &a, CODEC_RAW);
        w.finish();
        let mut recipes = BTreeMap::new();
        recipes.insert("ab".to_string(), Recipe { parts: vec!["a".into(), "b".into()], shape: vec![128, 64] });
        recipes.insert("ac".to_string(), Recipe { parts: vec!["a".into(), "c".into()], shape: vec![128, 64] });
        recipes.insert("c1".to_string(), Recipe::direct("c", &[64, 64, 1, 1]));
        let m = MappedCkpt::new(CkptFile::open(&path).unwrap(), recipes.clone()).unwrap();
        assert_eq!(m.unused(), Vec::<String>::new());
        let ab = m.read_raw("ab").unwrap();
        assert!(matches!(ab, CkptTensor::Palettized { .. }));
        let ab = ab.to_host(DataType::F16);
        let qa = m.file.read("a", DataType::F16).unwrap();
        let qb = m.file.read("b", DataType::F16).unwrap();
        assert_eq!(&ab.bytes()[..qa.bytes().len()], qa.bytes());
        assert_eq!(&ab.bytes()[qa.bytes().len()..], qb.bytes());
        let ac = m.read("ac", DataType::F16).unwrap();
        assert_eq!(&ac.bytes()[qa.bytes().len()..], a.bytes());
        assert_eq!(m.shape("c1"), Some(vec![64, 64, 1, 1]));
        recipes.insert("bad".to_string(), Recipe::direct("c", &[3]));
        assert!(MappedCkpt::new(CkptFile::open(&path).unwrap(), recipes).is_err());
    }

    #[test]
    fn ezm7_drops_three_mantissa_bits() {
        use std::io::Write;
        let vals = [1.0f32, -0.333, 65504.0, 1e-5, 0.0, 2.75];
        let h: Vec<u16> = vals.iter().map(|v| half::f16::from_f32(*v).to_bits()).collect();
        let exps: Vec<u8> = h.iter().map(|b| ((b >> 10) & 0x1f) as u8).collect();
        let rest: Vec<u8> = h.iter().map(|b| (((b >> 15) << 7) | ((b & 0x3ff) >> 3)) as u8).collect();
        let mut e = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(&exps).unwrap();
        let z = e.finish().unwrap();
        let mut payload = (z.len() as i32).to_le_bytes().to_vec();
        payload.extend_from_slice(&z);
        payload.extend_from_slice(&rest);
        let out = ezm7_decode(&payload, vals.len()).unwrap();
        for (i, b) in h.iter().enumerate() {
            assert_eq!(u16::from_le_bytes([out[2 * i], out[2 * i + 1]]), b & !7);
        }
    }

    #[test]
    fn trailer_store_and_errors() {
        // Build a sidecar checkpoint, then append its data to the database (trailer store).
        let path = tmp("trailer_src.ckpt");
        let mut w = Writer::new(&path, true);
        let t = HostTensor::from_f32(&values(2048, 5.0), &[32, 64]).to_dtype(DataType::F16);
        w.dense("w", &t, CODEC_RAW);
        w.q8p("q", &t, 1024);
        w.finish();
        let mut db = std::fs::read(&path).unwrap();
        let ext = std::fs::read(sidecar_path(&path)).unwrap();
        let base = db.len().div_ceil(4096) * 4096;
        db.resize(base, 0);
        db[60..64].copy_from_slice(&(base as u32).to_be_bytes());
        db.extend_from_slice(&ext);
        let out = tmp("trailer.ckpt");
        std::fs::write(&out, &db).unwrap();
        let f = CkptFile::open(&out).unwrap();
        assert_eq!(f.external_store(), Some(out.as_path()));
        assert_eq!(f.read("w", DataType::F16).unwrap().bytes(), t.bytes());
        assert!(matches!(f.read_raw("q").unwrap(), CkptTensor::Palettized { .. }));

        // Missing sidecar: clear error.
        std::fs::remove_file(sidecar_path(&path)).unwrap();
        let err = CkptFile::open(&path).err().unwrap().to_string();
        assert!(err.contains("-tensordata"), "{err}");

        // Unsupported codec.
        let path = tmp("unsupported.ckpt");
        let w = Writer::new(&path, false);
        w.insert("x", 0x8a1e9b, 1, DataType::F16, &[4], &[0; 16]);
        w.finish();
        let f = CkptFile::open(&path).unwrap();
        let err = f.read("x", DataType::F32).unwrap_err().to_string();
        assert!(err.contains("i8x"), "{err}");
        assert!(!is_ckpt(Path::new("Cargo.toml")));
    }
}
