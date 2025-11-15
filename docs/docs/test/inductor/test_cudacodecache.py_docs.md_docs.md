# Documentation: `docs/test/inductor/test_cudacodecache.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_cudacodecache.py_docs.md`
- **Size**: 6,760 bytes (6.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_cudacodecache.py`

## File Metadata

- **Path**: `test/inductor/test_cudacodecache.py`
- **Size**: 3,065 bytes (2.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import ctypes

import torch
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import CUDACodeCache
from torch._inductor.codegen.cuda.cuda_env import nvcc_exist
from torch._inductor.exc import CUDACompileError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.triton_utils import requires_cuda_and_triton


_SOURCE_CODE = r"""

#include <stdio.h>

__global__
void saxpy_device(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

extern "C" {

__attribute__((__visibility__("default")))
int saxpy(int n, float a, float *x, float *y) {
  // Perform SAXPY
  saxpy_device<<<(n+255)/256, 256>>>(n, a, x, y);
  return 0;
}

}
"""


class TestCUDACodeCache(InductorTestCase):
    @requires_cuda_and_triton
    def test_cuda_load(self):
        with fresh_cache():
            # Test both .o and .so compilation.
            (
                object_file_path,
                object_hash_key,
                source_code_path0,
            ) = CUDACodeCache.compile(_SOURCE_CODE, "o")
            dll_wrapper, so_hash_key, source_code_path1 = CUDACodeCache.load(
                _SOURCE_CODE, "so"
            )
            self.assertEqual(source_code_path0, source_code_path1)
            self.assertEqual(object_hash_key, so_hash_key)

            # Test load and call functions in .so.
            x = torch.rand(10).float().cuda()
            y = torch.rand(10).float().cuda()
            a = 5.0
            expected_y = a * x + y
            dll_wrapper.saxpy(
                ctypes.c_int(10),
                ctypes.c_float(a),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(y.data_ptr()),
            )
            torch.testing.assert_close(y, expected_y)

    @requires_cuda_and_triton
    def test_compilation_error(self):
        with fresh_cache():
            error_source_code = _SOURCE_CODE.replace("saxpy_device", "saxpy_wrong", 1)
            with self.assertRaises(CUDACompileError):
                CUDACodeCache.compile(error_source_code, "o")

    @requires_cuda_and_triton
    def test_async_compile(self):
        with fresh_cache():
            async_compile = AsyncCompile()
            compiled_res = async_compile.cuda(_SOURCE_CODE, "so")
            async_compile.wait(globals())

            # Test load and call functions in .so.
            x = torch.rand(5).float().cuda()
            y = torch.rand(5).float().cuda()
            a = 2.0
            expected_y = a * x + y
            compiled_res.result().saxpy(
                ctypes.c_int(5),
                ctypes.c_float(a),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(y.data_ptr()),
            )
            torch.testing.assert_close(y, expected_y)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if nvcc_exist():
        run_tests("cuda")

```



## High-Level Overview

_SOURCE_CODE = r"""#include <stdio.h>__global__void saxpy_device(int n, float a, float *x, float *y){  int i = blockIdx.x*blockDim.x + threadIdx.x;  if (i < n) y[i] = a*x[i] + y[i];}extern "C" {__attribute__((__visibility__("default")))int saxpy(int n, float a, float *x, float *y) {  // Perform SAXPY  saxpy_device<<<(n+255)/256, 256>>>(n, a, x, y);  return 0;}}

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCUDACodeCache`

**Functions defined**: `test_cuda_load`, `test_compilation_error`, `test_async_compile`

**Key imports**: ctypes, torch, AsyncCompile, CUDACodeCache, nvcc_exist, CUDACompileError, TestCase as InductorTestCase, fresh_cache, requires_cuda_and_triton, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `ctypes`
- `torch`
- `torch._inductor.async_compile`: AsyncCompile
- `torch._inductor.codecache`: CUDACodeCache
- `torch._inductor.codegen.cuda.cuda_env`: nvcc_exist
- `torch._inductor.exc`: CUDACompileError
- `torch._inductor.test_case`: TestCase as InductorTestCase
- `torch._inductor.utils`: fresh_cache
- `torch.testing._internal.triton_utils`: requires_cuda_and_triton


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_cudacodecache.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_cudacodecache.py_docs.md`
- **Keyword Index**: `test_cudacodecache.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/inductor/test_cudacodecache.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_split_cat_fx_passes.py_kw.md_docs.md`](./test_split_cat_fx_passes.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_cudacodecache.py_docs.md_docs.md`
- **Keyword Index**: `test_cudacodecache.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
