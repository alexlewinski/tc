use crate::sys;

/// Errors from nnc operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("nnc command {name} failed with code {code}")]
    Exec { name: String, code: i32 },
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Element type of a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    BF16,
    F64,
    I32,
    I64,
    U8,
}

impl DataType {
    pub fn to_raw(self) -> i32 {
        (match self {
            DataType::F32 => sys::CCV_32F,
            DataType::F16 => sys::CCV_16F,
            DataType::BF16 => sys::CCV_16BF,
            DataType::F64 => sys::CCV_64F,
            DataType::I32 => sys::CCV_32S,
            DataType::I64 => sys::CCV_64S,
            DataType::U8 => sys::CCV_8U,
        }) as i32
    }

    pub fn from_raw(raw: i32) -> Option<DataType> {
        let raw = (raw as u32) & 0xFF000;
        Some(match raw {
            x if x == sys::CCV_32F => DataType::F32,
            x if x == sys::CCV_16F => DataType::F16,
            x if x == sys::CCV_16BF => DataType::BF16,
            x if x == sys::CCV_64F => DataType::F64,
            x if x == sys::CCV_32S => DataType::I32,
            x if x == sys::CCV_64S => DataType::I64,
            x if x == sys::CCV_8U => DataType::U8,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            DataType::F32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::F64 | DataType::I64 => 8,
            DataType::U8 => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F64)
    }
}

/// Where a tensor lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Gpu(u32),
}

impl Device {
    pub(crate) fn to_raw_type(self) -> i32 {
        match self {
            Device::Cpu => sys::CCV_TENSOR_CPU_MEMORY as i32,
            Device::Gpu(id) => (sys::CCV_TENSOR_GPU_MEMORY as i32) | (((id as i32) & 0xfff) << 8),
        }
    }

    pub(crate) fn from_raw_type(ty: i32) -> Device {
        if ty & 0x3 == sys::CCV_TENSOR_GPU_MEMORY as i32 {
            Device::Gpu(((ty & 0xfff00) >> 8) as u32)
        } else {
            Device::Cpu
        }
    }

    pub fn is_gpu(self) -> bool {
        matches!(self, Device::Gpu(_))
    }
}

/// Memory layout tag used by convolution / normalization commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Nchw,
    Nhwc,
}

impl Format {
    pub(crate) fn to_raw(self) -> i32 {
        (match self {
            Format::Nchw => sys::CCV_TENSOR_FORMAT_NCHW,
            Format::Nhwc => sys::CCV_TENSOR_FORMAT_NHWC,
        }) as i32
    }
    pub(crate) fn from_raw(raw: i32) -> Format {
        if raw == sys::CCV_TENSOR_FORMAT_NCHW as i32 {
            Format::Nchw
        } else {
            Format::Nhwc
        }
    }
}

pub(crate) const MAX_DIM: usize = sys::CCV_NNC_MAX_DIM_ALLOC as usize;

pub(crate) fn params(
    device: Device,
    format: Format,
    dtype: DataType,
    shape: &[usize],
) -> sys::ccv_nnc_tensor_param_t {
    assert!(shape.len() <= MAX_DIM, "too many dimensions: {shape:?}");
    let mut p = sys::ccv_nnc_tensor_param_t {
        type_: device.to_raw_type(),
        format: format.to_raw(),
        datatype: dtype.to_raw(),
        ..Default::default()
    };
    for (i, d) in shape.iter().enumerate() {
        assert!(*d > 0, "zero sized dimension in {shape:?}");
        p.dim[i] = *d as i32;
    }
    p
}

pub(crate) fn shape_of(p: &sys::ccv_nnc_tensor_param_t) -> Vec<usize> {
    p.dim.iter().take_while(|d| **d > 0).map(|d| *d as usize).collect()
}
