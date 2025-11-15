# Documentation: `docs/torch/_inductor/codegen/rocm/ck_template.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/rocm/ck_template.py_docs.md`
- **Size**: 6,573 bytes (6.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/rocm/ck_template.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/rocm/ck_template.py`
- **Size**: 3,695 bytes (3.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any
from typing_extensions import override

import torch
from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate
from torch._inductor.ir import IRNode
from torch._inductor.utils import IndentedBuffer

from .rocm_template import ArgInfo


class CKTemplate(ROCmTemplate):
    """
    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic
    """

    _TORCH_DTYPE_TO_CK = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int32: "I32",
        torch.int8: "I8",
        torch.float8_e4m3fnuz: "F8",  # gfx94
        torch.float8_e4m3fn: "F8",  # gfx95
        torch.float8_e5m2fnuz: "BF8",  # gfx94
        torch.float8_e5m2: "BF8",  # gfx95
    }

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK headers

                #ifdef DEBUG_LOG
                #define DEBUG_LOG_TMP DEBUG_LOG
                #undef DEBUG_LOG
                #else
                #define DEBUG_LOG_TMP 0
                #endif
                #include "ck/ck.hpp"
                #undef DEBUG_LOG
                #define DEBUG_LOG DEBUG_LOG_TMP

                #include "ck/utility/data_type.hpp"
                #include "ck/library/utility/check_err.hpp"
                #include "ck/library/utility/device_memory.hpp"
                #include "ck/library/utility/fill.hpp"
                #include "ck/library/utility/host_tensor.hpp"
                #include "ck/library/utility/host_tensor_generator.hpp"
                #include "ck/library/utility/literals.hpp"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK globals

                template <ck::index_t... Is>
                using S = ck::Sequence<Is...>;

                template<typename... Ts>
                using Tuple = ck::Tuple<Ts...>;

                using PassThrough = ck::tensor_operation::element_wise::PassThrough;
                using Bilinear = ck::tensor_operation::element_wise::Bilinear;
                using Scale = ck::tensor_operation::element_wise::Scale;
                using ScaleAdd = ck::tensor_operation::element_wise::ScaleAdd;
                using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;

                // see "composable_kernel/include/ck/utility/data_type.hpp"
                using F8  = ck::f8_t;
                using BF8 = ck::bf8_t;
                using F16 = ck::half_t;
                using F32 = float;
                // using F64 = double;
                using BF16 = ck::bhalf_t;
                // using I32 = int32_t;
                // using I8 = int8_t;
                // using I4 = ck::int4_t;

                #if DEBUG_LOG
                static constexpr auto kDEBUG_LOG = 1;
                #else
                static constexpr auto kDEBUG_LOG = 0;
                #endif
            """
        )
        return res

    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._TORCH_DTYPE_TO_CK.get(node.get_dtype())}*)({ptr})"

    @override
    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return [ArgInfo("kBatch", "int32_t")]

    @override
    def get_runtime_arg_values(self, **kwargs: Any) -> list[Any]:
        """
        Helper method to retrieve runtime args from generate kwargs
        """
        return [kwargs[arg.name] for arg in self.get_runtime_arg_info()]

```



## High-Level Overview

"""    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic

This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CKTemplate`

**Functions defined**: `header`, `globals`, `torch_type_to_ck`, `get_runtime_arg_info`, `get_runtime_arg_values`

**Key imports**: Any, override, torch, ROCmTemplate, IRNode, IndentedBuffer, ArgInfo


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen/rocm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `typing_extensions`: override
- `torch`
- `torch._inductor.codegen.rocm.rocm_template`: ROCmTemplate
- `torch._inductor.ir`: IRNode
- `torch._inductor.utils`: IndentedBuffer
- `.rocm_template`: ArgInfo


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor/codegen/rocm`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ck_tile_universal_gemm_template.py_docs.md`](./ck_tile_universal_gemm_template.py_docs.md)
- [`rocm_benchmark_request.py_docs.md`](./rocm_benchmark_request.py_docs.md)
- [`rocm_template_buffer.py_docs.md`](./rocm_template_buffer.py_docs.md)
- [`ck_conv_template.py_docs.md`](./ck_conv_template.py_docs.md)
- [`rocm_template.py_docs.md`](./rocm_template.py_docs.md)
- [`ck_tile_template.py_docs.md`](./ck_tile_template.py_docs.md)
- [`rocm_cpp_scheduling.py_docs.md`](./rocm_cpp_scheduling.py_docs.md)
- [`rocm_utils.py_docs.md`](./rocm_utils.py_docs.md)


## Cross-References

- **File Documentation**: `ck_template.py_docs.md`
- **Keyword Index**: `ck_template.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen/rocm`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen/rocm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_inductor/codegen/rocm`):

- [`rocm_template.py_kw.md_docs.md`](./rocm_template.py_kw.md_docs.md)
- [`ck_tile_universal_gemm_template.py_docs.md_docs.md`](./ck_tile_universal_gemm_template.py_docs.md_docs.md)
- [`ck_tile_template.py_kw.md_docs.md`](./ck_tile_template.py_kw.md_docs.md)
- [`rocm_template_buffer.py_kw.md_docs.md`](./rocm_template_buffer.py_kw.md_docs.md)
- [`rocm_utils.py_kw.md_docs.md`](./rocm_utils.py_kw.md_docs.md)
- [`ck_universal_gemm_template.py_docs.md_docs.md`](./ck_universal_gemm_template.py_docs.md_docs.md)
- [`ck_conv_template.py_docs.md_docs.md`](./ck_conv_template.py_docs.md_docs.md)
- [`rocm_template.py_docs.md_docs.md`](./rocm_template.py_docs.md_docs.md)
- [`rocm_kernel.py_docs.md_docs.md`](./rocm_kernel.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ck_template.py_docs.md_docs.md`
- **Keyword Index**: `ck_template.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
