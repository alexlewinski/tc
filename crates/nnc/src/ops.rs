//! Tensor operations. Every op executes eagerly on the device of its inputs.

use crate::graph::Var;
use crate::sys;
use crate::types::MAX_DIM;

fn base_params() -> sys::ccv_nnc_cmd_param_t {
    let mut p = sys::ccv_nnc_cmd_param_t::default();
    p.size.dim[0] = 1;
    p.size.dim[1] = 1;
    p.size.dim[2] = 1;
    p
}

fn cmd(id: u32, params: sys::ccv_nnc_cmd_param_t) -> sys::ccv_nnc_cmd_t {
    unsafe { sys::ccv_nnc_cmd(id, std::ptr::null_mut(), params, 0) }
}

fn no_hint() -> sys::ccv_nnc_hint_t {
    sys::ccv_nnc_hint_t::default()
}

fn axes_array(axes: &[usize]) -> [i32; MAX_DIM] {
    let mut a = [0i32; MAX_DIM];
    for (i, x) in axes.iter().enumerate() {
        a[i] = *x as i32;
    }
    a
}

fn run1(id: u32, params: sys::ccv_nnc_cmd_param_t, inputs: &[Option<&Var>]) -> Var {
    let g = inputs.iter().flatten().next().expect("at least one input").graph().clone();
    let mut outs = g.exec_new(cmd(id, params), no_hint(), inputs, 1).unwrap_or_else(|e| panic!("{e}"));
    outs.pop().unwrap()
}

fn run_into(id: u32, params: sys::ccv_nnc_cmd_param_t, inputs: &[Option<&Var>], outputs: &[Option<&Var>]) {
    let g = inputs.iter().chain(outputs.iter()).flatten().next().expect("a tensor").graph().clone();
    g.exec(cmd(id, params), no_hint(), inputs, outputs).unwrap_or_else(|e| panic!("{e}"));
}

pub(crate) fn format_transform(src: &Var, dst: &Var) {
    run_into(sys::CCV_NNC_FORMAT_TRANSFORM_FORWARD, base_params(), &[Some(src)], &[Some(dst)]);
}

pub(crate) fn data_transfer(src: &Var, dst: &Var) {
    run_into(sys::CCV_NNC_DATA_TRANSFER_FORWARD, base_params(), &[Some(src)], &[Some(dst)]);
}

pub(crate) fn datatype_conversion(src: &Var, dst: &Var) {
    run_into(sys::CCV_NNC_DATATYPE_CONVERSION_FORWARD, base_params(), &[Some(src)], &[Some(dst)]);
}

pub(crate) fn set(dst: &Var, value: f32) {
    let mut p = base_params();
    p.__bindgen_anon_1.blas.a = [value, 0.0, 0.0];
    run_into(sys::CCV_NNC_SET_FORWARD, p, &[], &[Some(dst)]);
}

/// Transpose specification for GEMM: swap these two axes of the operand.
pub type Transpose = Option<(usize, usize)>;

fn gemm_params(ta: Transpose, tb: Transpose) -> sys::ccv_nnc_cmd_param_t {
    let mut p = base_params();
    p.__bindgen_anon_1.blas.a = [1.0, 1.0, 1.0];
    if let Some((x, y)) = ta {
        p.__bindgen_anon_1.blas.transpose_a = [x as i32, y as i32];
    }
    if let Some((x, y)) = tb {
        p.__bindgen_anon_1.blas.transpose_b = [x as i32, y as i32];
    }
    p
}

/// Generalized (batched, broadcasting) matrix multiply.
pub fn matmul(a: &Var, b: &Var, ta: Transpose, tb: Transpose) -> Var {
    run1(sys::CCV_NNC_GEMM_FORWARD, gemm_params(ta, tb), &[Some(a), Some(b)])
}

/// `x @ w^T + bias`, where `w` has shape `[out, in]` (PyTorch Linear convention).
pub fn linear(x: &Var, w: &Var, bias: Option<&Var>) -> Var {
    assert_eq!(w.rank(), 2, "linear weight must be 2-D");
    run1(sys::CCV_NNC_GEMM_FORWARD, gemm_params(None, Some((0, 1))), &[Some(x), Some(w), bias])
}

fn broadcast_rank(a: &Var, b: &Var) -> (Var, Var) {
    let ra = a.rank();
    let rb = b.rank();
    if ra == rb {
        return (a.clone(), b.clone());
    }
    let pad = |v: &Var, r: usize| {
        let mut s = vec![1usize; r - v.rank()];
        s.extend(v.shape());
        v.reshape(&s)
    };
    if ra > rb {
        (a.clone(), pad(b, ra))
    } else {
        (pad(a, rb), b.clone())
    }
}

/// `p * a + q * b` with broadcasting.
pub fn add_scaled(a: &Var, b: &Var, p: f32, q: f32) -> Var {
    let (a, b) = broadcast_rank(a, b);
    let mut params = base_params();
    params.__bindgen_anon_1.blas.a = [p, q, 0.0];
    run1(sys::CCV_NNC_ADD_FORWARD, params, &[Some(&a), Some(&b)])
}

pub fn add(a: &Var, b: &Var) -> Var {
    add_scaled(a, b, 1.0, 1.0)
}

pub fn sub(a: &Var, b: &Var) -> Var {
    add_scaled(a, b, 1.0, -1.0)
}

/// Elementwise product with broadcasting, times `scale`.
pub fn mul_scaled(a: &Var, b: &Var, scale: f32) -> Var {
    let (a, b) = broadcast_rank(a, b);
    let mut params = base_params();
    params.__bindgen_anon_1.blas.a = [scale, 0.0, 0.0];
    run1(sys::CCV_NNC_MUL_FORWARD, params, &[Some(&a), Some(&b)])
}

pub fn mul(a: &Var, b: &Var) -> Var {
    mul_scaled(a, b, 1.0)
}

/// Elementwise division `a / b` (same shapes).
pub fn div(a: &Var, b: &Var) -> Var {
    run1(sys::CCV_NNC_EWDIV_FORWARD, base_params(), &[Some(a), Some(b)])
}

/// Multiply by a scalar.
pub fn scale(a: &Var, s: f32) -> Var {
    let mut params = base_params();
    params.__bindgen_anon_1.blas.a = [s, 0.0, 0.0];
    run1(sys::CCV_NNC_SCALAR_MUL_FORWARD, params, &[Some(a)])
}

/// Add a scalar to every element.
pub fn add_scalar(a: &Var, s: f32) -> Var {
    let ones = vec![1usize; a.rank()];
    let c = a.graph().full(a.device(), a.dtype(), &ones, s);
    add(a, &c)
}

/// Sum of same-shaped tensors.
pub fn sum(inputs: &[&Var]) -> Var {
    let ins: Vec<Option<&Var>> = inputs.iter().map(|v| Some(*v)).collect();
    run1(sys::CCV_NNC_EWSUM_FORWARD, base_params(), &ins)
}

pub fn exp(a: &Var) -> Var {
    run1(sys::CCV_NNC_EWEXP_FORWARD, base_params(), &[Some(a)])
}

pub fn sqrt(a: &Var) -> Var {
    run1(sys::CCV_NNC_EWSQRT_FORWARD, base_params(), &[Some(a)])
}

pub fn sin(a: &Var) -> Var {
    run1(sys::CCV_NNC_EWSIN_FORWARD, base_params(), &[Some(a)])
}

pub fn cos(a: &Var) -> Var {
    run1(sys::CCV_NNC_EWCOS_FORWARD, base_params(), &[Some(a)])
}

/// `x * sigmoid(x)`.
pub fn silu(a: &Var) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.swish.beta = 1.0;
    run1(sys::CCV_NNC_SWISH_FORWARD, p, &[Some(a)])
}

pub fn sigmoid(a: &Var) -> Var {
    run1(sys::CCV_NNC_SIGMOID_FORWARD, base_params(), &[Some(a)])
}

pub fn tanh(a: &Var) -> Var {
    run1(sys::CCV_NNC_TANH_FORWARD, base_params(), &[Some(a)])
}

pub fn gelu(a: &Var, approximate_tanh: bool) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.gelu.tanh = approximate_tanh as i32;
    run1(sys::CCV_NNC_GELU_FORWARD, p, &[Some(a)])
}

pub fn relu(a: &Var) -> Var {
    run1(sys::CCV_NNC_RELU_FORWARD, base_params(), &[Some(a)])
}

/// Softmax over the last axis.
pub fn softmax(a: &Var) -> Var {
    run1(sys::CCV_NNC_SOFTMAX_FORWARD, base_params(), &[Some(a)])
}

pub fn clamp(a: &Var, min: Option<f32>, max: Option<f32>) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.clamp.min = min.unwrap_or(f32::NAN);
    p.__bindgen_anon_1.clamp.max = max.unwrap_or(f32::NAN);
    run1(sys::CCV_NNC_CLAMP_FORWARD, p, &[Some(a)])
}

/// Layer normalization over `axes`. `scale`/`bias` are optional affine parameters.
pub fn layer_norm(x: &Var, axes: &[usize], eps: f32, scale_bias: Option<(&Var, &Var)>) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.lnorm.axis = axes_array(axes);
    p.__bindgen_anon_1.lnorm.count = axes.len() as i32;
    p.__bindgen_anon_1.lnorm.epsilon = eps;
    p.__bindgen_anon_1.lnorm.scale = 1.0;
    p.__bindgen_anon_1.lnorm.elementwise_affine = scale_bias.is_some() as i32;
    let g = x.graph().clone();
    let inputs: Vec<Option<&Var>> = match scale_bias {
        Some((s, b)) => vec![Some(x), Some(s), Some(b)],
        None => vec![Some(x)],
    };
    let outs = g
        .exec_new(cmd(sys::CCV_NNC_LAYER_NORM_FORWARD, p), no_hint(), &inputs, 3)
        .unwrap_or_else(|e| panic!("{e}"));
    outs.into_iter().next().unwrap()
}

/// RMS normalization over `axes` with optional scale.
pub fn rms_norm(x: &Var, axes: &[usize], eps: f32, scale: Option<&Var>) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.rmsnorm.axis = axes_array(axes);
    p.__bindgen_anon_1.rmsnorm.count = axes.len() as i32;
    p.__bindgen_anon_1.rmsnorm.epsilon = eps;
    p.__bindgen_anon_1.rmsnorm.scale = 1.0;
    p.__bindgen_anon_1.rmsnorm.elementwise_affine = scale.is_some() as i32;
    let g = x.graph().clone();
    let inputs: Vec<Option<&Var>> = match scale {
        Some(s) => vec![Some(x), Some(s)],
        None => vec![Some(x)],
    };
    let outs = g
        .exec_new(cmd(sys::CCV_NNC_RMSNORM_FORWARD, p), no_hint(), &inputs, 2)
        .unwrap_or_else(|e| panic!("{e}"));
    outs.into_iter().next().unwrap()
}

/// Group normalization. `group_axis` is the channel axis; statistics are computed over the
/// channel group plus `reduce_axes`.
pub fn group_norm(
    x: &Var,
    group_axis: usize,
    groups: usize,
    reduce_axes: &[usize],
    eps: f32,
    scale_bias: Option<(&Var, &Var)>,
) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.gnorm.group_axis = group_axis as i32;
    p.__bindgen_anon_1.gnorm.groups = groups as i32;
    p.__bindgen_anon_1.gnorm.reduce_axis = axes_array(reduce_axes);
    p.__bindgen_anon_1.gnorm.reduce_count = reduce_axes.len() as i32;
    p.__bindgen_anon_1.gnorm.epsilon = eps;
    p.__bindgen_anon_1.gnorm.elementwise_affine = scale_bias.is_some() as i32;
    let g = x.graph().clone();
    let inputs: Vec<Option<&Var>> = match scale_bias {
        Some((s, b)) => vec![Some(x), Some(s), Some(b)],
        None => vec![Some(x)],
    };
    let outs = g
        .exec_new(cmd(sys::CCV_NNC_GROUP_NORM_FORWARD, p), no_hint(), &inputs, 3)
        .unwrap_or_else(|e| panic!("{e}"));
    outs.into_iter().next().unwrap()
}

/// Scaled dot-product attention. `q`: `[B, Tq, Hq, D]`, `k`/`v`: `[B, Tk, Hk, D]` with
/// `Hq % Hk == 0` (grouped-query attention). Optional additive `mask` broadcastable to
/// `[B, Hq, Tq, Tk]`. Returns `[B, Tq, Hq, D]`.
pub fn scaled_dot_product_attention(
    q: &Var,
    k: &Var,
    v: &Var,
    scale: f32,
    is_causal: bool,
    mask: Option<&Var>,
) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.scaled_dot_product_attention.scale = scale;
    p.__bindgen_anon_1.scaled_dot_product_attention.is_causal = is_causal as i32;
    let g = q.graph().clone();
    let inputs: Vec<Option<&Var>> = match mask {
        Some(m) => vec![Some(q), Some(k), Some(v), Some(m)],
        None => vec![Some(q), Some(k), Some(v)],
    };
    let out = g.auto_var();
    g.exec(
        cmd(sys::CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, p),
        no_hint(),
        &inputs,
        &[Some(&out), None],
    )
    .unwrap_or_else(|e| panic!("{e}"));
    out
}

/// Complex multiplication of interleaved (real, imag) pairs along the last axis.
pub fn cmul(a: &Var, b: &Var) -> Var {
    let (a, b) = broadcast_rank(a, b);
    run1(sys::CCV_NNC_CMUL_FORWARD, base_params(), &[Some(&a), Some(&b)])
}

/// Gather rows of `table` (`[N, D]`) at `indices` (`[M]`, i32) -> `[M, D]`.
pub fn index_select(table: &Var, indices: &Var) -> Var {
    run1(sys::CCV_NNC_INDEX_SELECT_FORWARD, base_params(), &[Some(table), Some(indices)])
}

/// Swap two axes (materialized).
pub fn transpose(a: &Var, axis_a: usize, axis_b: usize) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.transpose.axis = [axis_a as i32, axis_b as i32];
    run1(sys::CCV_NNC_TRANSPOSE_FORWARD, p, &[Some(a)])
}

fn reduce(id: u32, a: &Var, axes: &[usize]) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.reduce.axis = axes_array(axes);
    p.__bindgen_anon_1.reduce.count = axes.len() as i32;
    run1(id, p, &[Some(a)])
}

/// Sum over `axes` (kept as size-1 dimensions).
pub fn reduce_sum(a: &Var, axes: &[usize]) -> Var {
    reduce(sys::CCV_NNC_REDUCE_SUM_FORWARD, a, axes)
}

/// Mean over `axes` (kept as size-1 dimensions).
pub fn reduce_mean(a: &Var, axes: &[usize]) -> Var {
    reduce(sys::CCV_NNC_REDUCE_MEAN_FORWARD, a, axes)
}

/// Concatenate along `axis`.
pub fn concat(inputs: &[&Var], axis: usize) -> Var {
    assert!(!inputs.is_empty());
    let first = inputs[0];
    let mut shape = first.shape();
    shape[axis] = inputs.iter().map(|v| v.dim(axis)).sum();
    let out = first.graph().empty_with_format(first.device(), first.format(), first.dtype(), &shape);
    let mut start = 0;
    for v in inputs {
        let n = v.dim(axis);
        out.narrow(axis, start, n).assign(v);
        start += n;
    }
    out
}

/// 2-D convolution on NHWC input `[N, H, W, Cin]`.
///
/// The weight layout depends on the device backend: CPU kernels take `[Cout, kh, kw, Cin/groups]`
/// tagged NHWC, GPU kernels take PyTorch's `[Cout, Cin/groups, kh, kw]` tagged NCHW. Both are
/// accepted here; the kernel size is read according to the weight's format.
pub fn conv2d(
    x: &Var,
    w: &Var,
    bias: Option<&Var>,
    stride: (usize, usize),
    padding: (usize, usize, usize, usize),
    groups: usize,
) -> Var {
    let ws = w.shape();
    assert_eq!(ws.len(), 4, "conv2d weight must be 4-D");
    let (kh, kw, cin) = match w.format() {
        crate::types::Format::Nchw => (ws[2], ws[3], ws[1]),
        crate::types::Format::Nhwc => (ws[1], ws[2], ws[3]),
    };
    let mut p = base_params();
    p.size.dim = [0; MAX_DIM];
    p.size.dim[0] = kh as i32;
    p.size.dim[1] = kw as i32;
    p.size.dim[2] = cin as i32;
    p.__bindgen_anon_1.convolution.count = ws[0] as i32;
    p.__bindgen_anon_1.convolution.groups = groups as i32;
    let mut dilation = [0; MAX_DIM];
    dilation[0] = 1;
    dilation[1] = 1;
    p.__bindgen_anon_1.convolution.dilation = dilation;
    let mut hint = no_hint();
    hint.stride.dim[0] = stride.0 as i32;
    hint.stride.dim[1] = stride.1 as i32;
    // padding: (top, bottom, left, right)
    hint.border.begin[0] = padding.0 as i32;
    hint.border.end[0] = padding.1 as i32;
    hint.border.begin[1] = padding.2 as i32;
    hint.border.end[1] = padding.3 as i32;
    let g = x.graph().clone();
    let mut outs = g
        .exec_new(cmd(sys::CCV_NNC_CONVOLUTION_FORWARD, p), hint, &[Some(x), Some(w), bias], 1)
        .unwrap_or_else(|e| panic!("{e}"));
    outs.pop().unwrap()
}

/// Nearest (or bilinear) upsampling of an NHWC tensor by integer-ish factors.
pub fn upsample(x: &Var, scale_h: f32, scale_w: f32, bilinear: bool) -> Var {
    let mut p = base_params();
    p.__bindgen_anon_1.upsample.type_ = bilinear as i32;
    p.__bindgen_anon_1.upsample.width_scale = scale_w;
    p.__bindgen_anon_1.upsample.height_scale = scale_h;
    p.__bindgen_anon_1.upsample.align_corners = 0;
    run1(sys::CCV_NNC_UPSAMPLE_FORWARD, p, &[Some(x)])
}

/// Elementwise `a * silu(b)`-style SwiGLU on two halves is model-specific; this computes
/// `silu(gate) * value` for same-shaped inputs.
pub fn silu_mul(gate: &Var, value: &Var) -> Var {
    mul(&silu(gate), value)
}
