use std::cell::Cell;
use std::fmt;
use std::rc::Rc;

use crate::host::HostTensor;
use crate::sys;
use crate::types::{params, shape_of, DataType, Device, Error, Format, Result, MAX_DIM};

pub(crate) struct GraphInner {
    pub(crate) raw: *mut sys::ccv_nnc_dynamic_graph_t,
    pub(crate) stream: Cell<*mut sys::ccv_nnc_stream_context_t>,
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        unsafe {
            let stream = self.stream.get();
            if !stream.is_null() {
                sys::ccv_nnc_stream_context_wait(stream);
                sys::ccv_nnc_stream_context_free(stream);
            }
            sys::ccv_nnc_dynamic_graph_free(self.raw);
        }
    }
}

/// A workspace that owns tensor variables. Not thread-safe; create one per worker thread.
#[derive(Clone)]
pub struct Graph {
    pub(crate) inner: Rc<GraphInner>,
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Graph({:p})", self.inner.raw)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Create a new inference graph (gradient tracking disabled).
    pub fn new() -> Graph {
        crate::init();
        let raw = unsafe { sys::ccv_nnc_dynamic_graph_new() };
        assert!(!raw.is_null());
        unsafe {
            sys::ccv_nnc_dynamic_graph_set_no_grad(raw, 1);
        }
        Graph { inner: Rc::new(GraphInner { raw, stream: Cell::new(std::ptr::null_mut()) }) }
    }

    /// Execute GPU work asynchronously on a dedicated stream. Host reads synchronize
    /// automatically.
    pub fn use_gpu_stream(&self) {
        if !self.inner.stream.get().is_null() || crate::gpu_count() == 0 {
            return;
        }
        let stream = unsafe { sys::ccv_nnc_stream_context_new(sys::CCV_STREAM_CONTEXT_GPU as i32) };
        self.inner.stream.set(stream);
    }

    /// Wait for all queued work to finish.
    pub fn synchronize(&self) {
        let stream = self.inner.stream.get();
        if !stream.is_null() {
            unsafe { sys::ccv_nnc_stream_context_wait(stream) };
        }
    }

    /// Release cached device memory that is no longer used.
    pub fn garbage_collect(&self) {
        self.synchronize();
        unsafe { sys::ccv_nnc_dynamic_graph_gc(self.inner.raw) };
    }

    pub(crate) fn stream(&self) -> *mut sys::ccv_nnc_stream_context_t {
        self.inner.stream.get()
    }

    fn wrap(&self, raw: sys::ccv_nnc_tensor_variable_t) -> Var {
        Var { inner: Rc::new(VarInner { graph: self.clone(), raw }) }
    }

    /// An uninitialized variable whose shape will be inferred by the op that writes it.
    pub(crate) fn auto_var(&self) -> Var {
        let p = sys::ccv_nnc_tensor_param_t::default();
        self.wrap(unsafe { sys::ccv_nnc_tensor_variable_new_impl(self.inner.raw, p) })
    }

    /// An uninitialized tensor variable with the given layout (memory is allocated lazily).
    pub fn empty(&self, device: Device, dtype: DataType, shape: &[usize]) -> Var {
        self.empty_with_format(device, Format::Nhwc, dtype, shape)
    }

    pub fn empty_with_format(&self, device: Device, format: Format, dtype: DataType, shape: &[usize]) -> Var {
        let p = params(device, format, dtype, shape);
        self.wrap(unsafe { sys::ccv_nnc_tensor_variable_new_impl(self.inner.raw, p) })
    }

    /// A tensor filled with a constant value.
    pub fn full(&self, device: Device, dtype: DataType, shape: &[usize], value: f32) -> Var {
        let v = self.empty(device, dtype, shape);
        v.fill(value);
        v
    }

    pub fn zeros(&self, device: Device, dtype: DataType, shape: &[usize]) -> Var {
        self.full(device, dtype, shape, 0.0)
    }

    /// Upload host data to a new variable on `device`.
    pub fn from_host(&self, host: &HostTensor, device: Device) -> Var {
        let cpu = self.empty(Device::Cpu, host.dtype(), host.shape());
        unsafe {
            let t = sys::ccv_nnc_tensor_from_variable_impl(self.inner.raw, cpu.raw(), std::ptr::null_mut());
            assert!(!t.is_null());
            let dst = (*t).data.ptr as *mut u8;
            std::ptr::copy_nonoverlapping(host.bytes().as_ptr(), dst, host.bytes().len());
        }
        if device == Device::Cpu {
            cpu
        } else {
            cpu.to_device(device)
        }
    }

    /// Convenience: upload f32 data, converting to `dtype` on the target device.
    pub fn from_f32(&self, data: &[f32], shape: &[usize], device: Device, dtype: DataType) -> Var {
        let host = HostTensor::from_f32(data, shape);
        let v = self.from_host(&host, device);
        if dtype == DataType::F32 {
            v
        } else {
            v.to_dtype(dtype)
        }
    }

    /// Run a command. `outputs` are pre-created variables (use [`Graph::auto_var`] for inferred shapes).
    pub(crate) fn exec(
        &self,
        cmd: sys::ccv_nnc_cmd_t,
        hint: sys::ccv_nnc_hint_t,
        inputs: &[Option<&Var>],
        outputs: &[Option<&Var>],
    ) -> Result<()> {
        let ins: Vec<sys::ccv_nnc_tensor_variable_t> =
            inputs.iter().map(|v| v.map(|v| v.raw()).unwrap_or(std::ptr::null_mut())).collect();
        let mut outs: Vec<sys::ccv_nnc_tensor_variable_t> =
            outputs.iter().map(|v| v.map(|v| v.raw()).unwrap_or(std::ptr::null_mut())).collect();
        let code = unsafe {
            sys::ccv_nnc_dynamic_graph_exec(
                self.inner.raw,
                cmd,
                hint,
                0,
                ins.as_ptr(),
                ins.len() as i32,
                outs.as_mut_ptr(),
                outs.len() as i32,
                0,
                self.stream(),
            )
        };
        if code != 0 {
            let name = unsafe { std::ffi::CStr::from_ptr(sys::ccv_nnc_cmd_name(cmd.cmd)) }
                .to_string_lossy()
                .into_owned();
            return Err(Error::Exec { name, code });
        }
        Ok(())
    }

    /// Run a command producing `n` new outputs with inferred shapes.
    pub(crate) fn exec_new(
        &self,
        cmd: sys::ccv_nnc_cmd_t,
        hint: sys::ccv_nnc_hint_t,
        inputs: &[Option<&Var>],
        n: usize,
    ) -> Result<Vec<Var>> {
        let outs: Vec<Var> = (0..n).map(|_| self.auto_var()).collect();
        let refs: Vec<Option<&Var>> = outs.iter().map(Some).collect();
        self.exec(cmd, hint, inputs, &refs)?;
        Ok(outs)
    }
}

pub(crate) struct VarInner {
    pub(crate) graph: Graph,
    pub(crate) raw: sys::ccv_nnc_tensor_variable_t,
}

impl Drop for VarInner {
    fn drop(&mut self) {
        unsafe { sys::ccv_nnc_tensor_variable_free(self.graph.inner.raw, self.raw) };
    }
}

/// A tensor variable living in a [`Graph`]. Cloning shares the same variable.
#[derive(Clone)]
pub struct Var {
    pub(crate) inner: Rc<VarInner>,
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Var({:?}, {:?}, {:?})", self.shape(), self.dtype(), self.device())
    }
}

impl Var {
    pub(crate) fn raw(&self) -> sys::ccv_nnc_tensor_variable_t {
        self.inner.raw
    }

    pub fn graph(&self) -> &Graph {
        &self.inner.graph
    }

    pub(crate) fn raw_params(&self) -> sys::ccv_nnc_tensor_param_t {
        unsafe { sys::ccv_nnc_tensor_variable_params(self.graph().inner.raw, self.raw()) }
    }

    pub fn shape(&self) -> Vec<usize> {
        shape_of(&self.raw_params())
    }

    pub fn dim(&self, i: usize) -> usize {
        self.shape()[i]
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn dtype(&self) -> DataType {
        DataType::from_raw(self.raw_params().datatype).expect("unknown data type")
    }

    pub fn device(&self) -> Device {
        Device::from_raw_type(self.raw_params().type_)
    }

    pub fn format(&self) -> Format {
        Format::from_raw(self.raw_params().format)
    }

    /// (offset, strides) if this variable is an alias with explicit strides.
    fn alias_info(&self) -> Option<([i32; MAX_DIM], [i32; MAX_DIM])> {
        let mut ofs = [0i32; MAX_DIM];
        let mut stride = [0i32; MAX_DIM];
        let r = unsafe {
            sys::ccv_nnc_tensor_variable_alias_params(
                self.graph().inner.raw,
                self.raw(),
                ofs.as_mut_ptr(),
                stride.as_mut_ptr(),
            )
        };
        if r == 0 {
            Some((ofs, stride))
        } else {
            None
        }
    }

    /// Element strides of this variable.
    pub fn strides(&self) -> Vec<usize> {
        let shape = self.shape();
        if let Some((_, stride)) = self.alias_info() {
            if stride[0] != 0 {
                return stride[..shape.len()].iter().map(|s| *s as usize).collect();
            }
        }
        contiguous_strides(&shape)
    }

    pub fn is_contiguous(&self) -> bool {
        let shape = self.shape();
        let strides = self.strides();
        let expected = contiguous_strides(&shape);
        // Leading dims of size 1 don't matter.
        shape
            .iter()
            .zip(strides.iter().zip(expected.iter()))
            .all(|(d, (s, e))| *d == 1 || s == e)
    }

    fn alias(&self, offset: &[usize], strides: Option<&[usize]>, shape: &[usize]) -> Var {
        let p = params(self.device(), self.format(), self.dtype(), shape);
        let mut ofs = [0i32; MAX_DIM];
        for (i, o) in offset.iter().enumerate() {
            ofs[i] = *o as i32;
        }
        let mut st = [0i32; MAX_DIM];
        if let Some(strides) = strides {
            for (i, s) in strides.iter().enumerate() {
                st[i] = *s as i32;
            }
        }
        let raw = unsafe {
            sys::ccv_nnc_tensor_variable_alias_new(self.graph().inner.raw, self.raw(), ofs.as_ptr(), st.as_ptr(), p)
        };
        self.graph().wrap(raw)
    }

    /// Reinterpret with a new shape (same number of elements). Copies if the variable is a
    /// non-contiguous view.
    pub fn reshape(&self, shape: &[usize]) -> Var {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.numel(), "reshape {:?} -> {:?}", self.shape(), shape);
        if !self.is_contiguous() {
            return self.contiguous().reshape(shape);
        }
        if shape == self.shape().as_slice() {
            return self.clone();
        }
        self.alias(&[0; MAX_DIM][..shape.len()], None, shape)
    }

    /// View of `len` elements starting at `start` along `axis`.
    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> Var {
        let shape = self.shape();
        assert!(start + len <= shape[axis], "narrow out of range");
        let strides = self.strides();
        let mut offset = vec![0usize; shape.len()];
        offset[axis] = start;
        let mut new_shape = shape.clone();
        new_shape[axis] = len;
        self.alias(&offset, Some(&strides), &new_shape)
    }

    /// View with permuted axes (no copy). Call [`Var::contiguous`] to materialize.
    pub fn permute(&self, axes: &[usize]) -> Var {
        let shape = self.shape();
        assert_eq!(axes.len(), shape.len());
        let strides = self.strides();
        let new_shape: Vec<usize> = axes.iter().map(|a| shape[*a]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|a| strides[*a]).collect();
        self.alias(&vec![0; shape.len()], Some(&new_strides), &new_shape)
    }

    /// Copy into a new contiguous variable if this is a strided view.
    pub fn contiguous(&self) -> Var {
        if self.is_contiguous() {
            return self.clone();
        }
        self.copy()
    }

    /// Always make a contiguous copy.
    pub fn copy(&self) -> Var {
        let out = self.graph().empty_with_format(self.device(), self.format(), self.dtype(), &self.shape());
        crate::ops::format_transform(self, &out);
        out
    }

    /// Copy `src` into this variable (shapes must match; either may be a view).
    pub fn assign(&self, src: &Var) {
        assert_eq!(self.shape(), src.shape(), "assign shape mismatch");
        crate::ops::format_transform(src, self);
    }

    /// Transfer to another device.
    pub fn to_device(&self, device: Device) -> Var {
        if self.device() == device {
            return self.clone();
        }
        let src = self.contiguous();
        let out = self.graph().empty_with_format(device, self.format(), self.dtype(), &self.shape());
        crate::ops::data_transfer(&src, &out);
        out
    }

    /// Convert element type.
    pub fn to_dtype(&self, dtype: DataType) -> Var {
        if self.dtype() == dtype {
            return self.clone();
        }
        let src = self.contiguous();
        let out = self.graph().empty_with_format(self.device(), self.format(), dtype, &self.shape());
        crate::ops::datatype_conversion(&src, &out);
        out
    }

    /// Copy back to the host (synchronizes).
    pub fn to_host(&self) -> HostTensor {
        let cpu = self.to_device(Device::Cpu).contiguous();
        let g = self.graph();
        g.synchronize();
        let shape = cpu.shape();
        let dtype = cpu.dtype();
        let n: usize = shape.iter().product::<usize>() * dtype.size();
        let mut bytes = vec![0u8; n];
        unsafe {
            let t = sys::ccv_nnc_tensor_from_variable_impl(g.inner.raw, cpu.raw(), std::ptr::null_mut());
            assert!(!t.is_null());
            std::ptr::copy_nonoverlapping((*t).data.ptr as *mut u8, bytes.as_mut_ptr(), n);
        }
        HostTensor::from_bytes(bytes, dtype, &shape)
    }

    /// Copy back to the host as f32.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.to_host().to_f32_vec()
    }

    /// Set every element to `value`.
    pub fn fill(&self, value: f32) {
        crate::ops::set(self, value);
    }
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
