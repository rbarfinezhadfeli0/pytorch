# Documentation: `docs/test/inductor/test_cpp_wrapper_hipify.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_cpp_wrapper_hipify.py_docs.md`
- **Size**: 9,681 bytes (9.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_cpp_wrapper_hipify.py`

## File Metadata

- **Path**: `test/inductor/test_cpp_wrapper_hipify.py`
- **Size**: 5,913 bytes (5.77 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import torch
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor.codegen.common import get_device_op_overrides
from torch._inductor.test_case import run_tests, TestCase


TEST_CODES = [
    "CUresult code = EXPR;",
    "CUfunction kernel = nullptr;",
    "static CUfunction kernel = nullptr;",
    "CUdeviceptr var = reinterpret_cast<CUdeviceptr>(arg.data_ptr());",
    "at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal());",
    # Hipification should be idempotent, hipifying should be a no-op for already hipified files
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
]

HIP_CODES = [
    "hipError_t code = EXPR;",
    "hipFunction_t kernel = nullptr;",
    "static hipFunction_t kernel = nullptr;",
    "hipDeviceptr_t var = reinterpret_cast<hipDeviceptr_t>(arg.data_ptr());",
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
]


class TestCppWrapperHipify(TestCase):
    def test_hipify_basic_declaration(self) -> None:
        assert len(TEST_CODES) == len(HIP_CODES)
        for i in range(len(TEST_CODES)):
            result = maybe_hipify_code_wrapper(TEST_CODES[i], True)
            expected = HIP_CODES[i]
            self.assertEqual(result, expected)

    def test_hipify_aoti_driver_header(self) -> None:
        cuda_codegen = get_device_op_overrides("cuda")
        header = cuda_codegen.kernel_driver()
        expected = """
            #define CUDA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                hipError_t code = EXPR;                          \\
                const char *msg;                               \\
                hipError_t code_get_error = hipDrvGetErrorString(code, &msg); \\
                if (code_get_error != hipSuccess) {          \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string("invalid error code!"));   \\
                }                                              \\
                if (code != hipSuccess) {                    \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string(msg));                     \\
                }                                              \\
            } while (0);

            static inline hipFunction_t loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                hipModule_t mod;
                hipFunction_t func;
                CUDA_DRIVER_CHECK(hipModuleLoad(&mod, filePath.c_str()));
                CUDA_DRIVER_CHECK(hipModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(hipFuncSetAttribute(
                        func,
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline hipFunction_t loadKernel(const void* start, const std::string &funcName, uint32_t sharedMemBytes) {
                hipModule_t mod;
                hipFunction_t func;
                CUDA_DRIVER_CHECK(hipModuleLoadData(&mod, start));
                CUDA_DRIVER_CHECK(hipModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(hipFuncSetAttribute(
                        func,
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline void launchKernel(
                    hipFunction_t func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    hipStream_t stream) {
                CUDA_DRIVER_CHECK(hipModuleLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
        """
        if torch.version.hip is not None:
            # Adjusting the warp size to GPU supported wavefront size on AMD GPU
            prop = torch.cuda.get_device_properties(torch.cuda.current_device())
            expected = expected.replace(
                "32*numWarps", str(prop.warp_size) + "*numWarps"
            )
        result = maybe_hipify_code_wrapper(header, True)
        self.assertEqual(result.rstrip(), expected.rstrip())

    def test_hipify_cross_platform(self) -> None:
        assert len(TEST_CODES) == len(HIP_CODES)
        for i in range(len(TEST_CODES)):
            hip_result = maybe_hipify_code_wrapper(TEST_CODES[i], True)
            result = maybe_hipify_code_wrapper(TEST_CODES[i])
            if torch.version.hip is not None:
                self.assertEqual(result, hip_result)
            else:
                self.assertEqual(result, TEST_CODES[i])


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

expected = """            #define CUDA_DRIVER_CHECK(EXPR)                    \\            do {                                               \\                hipError_t code = EXPR;                          \\                const char *msg;                               \\                hipError_t code_get_error = hipDrvGetErrorString(code, &msg); \\                if (code_get_error != hipSuccess) {          \\                    throw std::runtime_error(                  \\                        std::string("CUDA driver error: ") +   \\                        std::string("invalid error code!"));   \\                }                                              \\                if (code != hipSuccess) {                    \\

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCppWrapperHipify`

**Functions defined**: `test_hipify_basic_declaration`, `test_hipify_aoti_driver_header`, `test_hipify_cross_platform`

**Key imports**: torch, maybe_hipify_code_wrapper, get_device_op_overrides, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.codegen.aoti_hipify_utils`: maybe_hipify_code_wrapper
- `torch._inductor.codegen.common`: get_device_op_overrides
- `torch._inductor.test_case`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_cpp_wrapper_hipify.py
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

- **File Documentation**: `test_cpp_wrapper_hipify.py_docs.md`
- **Keyword Index**: `test_cpp_wrapper_hipify.py_kw.md`
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
python docs/test/inductor/test_cpp_wrapper_hipify.py_docs.md
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

- **File Documentation**: `test_cpp_wrapper_hipify.py_docs.md_docs.md`
- **Keyword Index**: `test_cpp_wrapper_hipify.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
