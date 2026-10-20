# Vendored ccv / nnc

Source: https://github.com/liuliu/ccv at commit `69b1ed1f4e281520c29cd4b45b93c6fcdde8d2e8`
(only `lib/` and `COPYING` are vendored). License: BSD 3-Clause (see `COPYING`; a few
3rd-party pieces under `lib/3rdparty` carry their own permissive licenses).

Built by `crates/nnc-sys/build.rs` using ccv's own makefiles with a generated `config.mk`
(no `configure` auto-detection, so no optional GPL dependencies such as FFTW/GSL get linked).

## Local modifications

* `lib/nnc/mfa/kernels/{AttentionKernel,GEMMKernel}+Precompiled.inc` are stored gzipped
  (`*.inc.gz`) to stay under GitHub's file size limit; `build.rs` decompresses them.
* `lib/nnc/cmd/blas/cpu_sys/_ccv_nnc_gemm_cpu_sys.c`: the two static BLAS helper functions
  are wrapped in `#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)` so the file
  compiles on platforms without a CBLAS (they were only referenced inside such guards).
