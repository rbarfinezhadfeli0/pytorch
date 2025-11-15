# Documentation: `docs/torch/_inductor/codegen/triton_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/triton_utils.py_docs.md`
- **Size**: 12,356 bytes (12.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/triton_utils.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/triton_utils.py`
- **Size**: 9,280 bytes (9.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any, Optional

import sympy

import torch
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config
from ..runtime.hints import AttrsDescriptorWrapper
from ..utils import _type_of, expr_fits_within_32bit, triton_version_uses_attrs_dict
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    KernelArgType,
    SizeArg,
    TensorArg,
    TMADescriptorArg,
    WorkspaceArg,
)


def should_unwrap_unspec_arg(name: str):
    if V.graph.is_unspec_arg(name):
        # Unwrap on all devices except CPU
        if V.graph.get_current_device_or_throw().type != "cpu":
            return True
        # Only unwrap on CPU if the input is not used as an output
        if name not in V.graph.mutated_buffers:
            return True
    return False


def signature_of(arg: KernelArgType, *, size_dtype: Optional[str]) -> str:
    if isinstance(arg, TensorArg):
        # TODO: Remove fp8 special handling when Triton supports PyTorch fp8 dtypes.
        # Related PR: https://github.com/triton-lang/triton/pull/2279/
        if arg.dtype == torch.float8_e4m3fn:
            typ = "*fp8e4nv"
        elif arg.dtype == torch.float8_e5m2:
            typ = "*fp8e5"
        elif arg.dtype == torch.float8_e4m3fnuz:
            typ = "*fp8e4b8"
        elif arg.dtype == torch.float8_e5m2fnuz:
            typ = "*fp8e5b16"
        else:
            typ = _type_of(arg.dtype)
        if should_unwrap_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_typ = typ.lstrip("*")
            if new_typ in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_typ
        else:
            return typ
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            if triton_version_uses_attrs_dict():
                # In newer versions of Triton, the signature includes "None" args
                # and their type is marked as "constexpr"
                return "constexpr"
            else:
                # In older versions of Triton...
                # From triton/runtime/jit.py
                # `None` is nullptr.  Implicitly convert to *i8.
                return "*i8"
        elif _arg_equals_1(arg) and triton_version_uses_attrs_dict():
            # In new versions of Triton, if we have an equal-to-1 arg that's marked as a constant,
            # it should be marked as "constexpr" in the signature.
            return "constexpr"
        elif isinstance(arg.expr, (float, sympy.Float)):
            return "fp32"
        elif isinstance(arg.expr, sympy.Symbol) and symbol_is_type(
            arg.expr, (SymT.UNBACKED_FLOAT)
        ):
            return "fp32"
        elif isinstance(arg.expr, bool):
            return "i1"

        # if this is a integer
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        elif size_dtype is None:
            # no hint: we'll see if we know that this is a 32-bit int, and guard if possible.
            int_max = torch.iinfo(torch.int32).max
            if expr_fits_within_32bit(arg.expr):
                V.graph.sizevars.check_leq(arg.expr, int_max)
                return "i32"
            else:
                return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    if isinstance(arg, WorkspaceArg):
        return _type_of(arg.dtype)
    if isinstance(arg, TMADescriptorArg):
        if arg.api_type == "experimental":
            return "nvTmaDesc"
        else:
            # https://github.com/triton-lang/triton/blob/9695baed9b46cf957e08b157bb4133f4a4b331c5/python/triton/runtime/jit.py#L360-L363
            assert arg.api_type == "stable"
            assert arg.block_shape is not None
            assert arg.dtype is not None
            inner = _type_of(arg.dtype)[1:]  # strip the `*`: *fp32 -> fp32
            return f"tensordesc<{inner}{list(arg.block_shape)}>"
    if isinstance(arg, ConstexprArg):
        return "constexpr"
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def non_constexpr_signature(signature):
    new_signature = []
    for arg in signature:
        if not isinstance(arg, ConstexprArg):
            new_signature.append(arg)

    return new_signature


def signature_to_meta(
    signature: list[KernelArgType],
    *,
    size_dtype: Optional[str],
    argdefs: list[ArgName],
    indices: Optional[list[int]] = None,
    is_template: bool = False,
) -> dict[str, str]:
    if indices is None:
        indices = list(range(len(signature)))

    def _decide_tl_dtype(arg):
        # Even if the ks0 symbol itself is within tl.int32 range, it's
        # risky to use tl.int32 dtype since we may have ks0*ks1 later
        # for kernels like torch.mean when dynamic shape is enabled.
        #
        # Check config.triton.use_block_ptr, since Triton block pointer
        # does not support 64bit indexing:
        # https://gist.github.com/shunting314/6a41c776171720ce4561f202dcde0ad6
        #
        # If the triton metadata is for a template, don't use tl.int64 index.
        # Templates like flex attention/decoding uses block pointers which
        # does not support 64 bit indexing.
        if (
            not config.triton.use_block_ptr
            and not is_template
            and isinstance(arg, SizeArg)
            and arg.name.startswith("ks")
        ):
            return "tl.int64"
        return size_dtype

    return {
        argdefs[i].name: signature_of(arg, size_dtype=_decide_tl_dtype(arg))
        for i, arg in zip(indices, signature)
    }


def is_unaligned_buffer(arg: TensorArg):
    buf_name = arg.buffer
    if buf_name in V.graph.unaligned_buffers:
        return True

    if buf_name in V.graph.graph_inputs:
        # See Note: [Input Alignment handling in Inductor]
        # For graph inputs that is not recorded in V.graph.unaligned_buffers,
        # we know for sure the tensor is aligned.
        return False

    if buf_name in V.graph.constants:
        # all constants are assumed to be aligned
        return False

    if V.graph.scheduler:
        layout = V.graph.scheduler.get_buffer_layout(buf_name)
    else:
        buffer = V.graph.try_get_buffer(buf_name)
        # output arg
        if not buffer:
            assert buf_name == V.kernel.output_node.name
            layout = V.kernel.output_node.layout
        else:
            layout = buffer.get_layout()

    if isinstance(layout, torch._inductor.ir.NonOwningLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False


def _arg_equals_1(arg: KernelArgType) -> bool:
    return (
        isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )


def equal_1_arg_indices(
    args: list[KernelArgType],
    *,
    indices: Optional[list[int]] = None,
) -> tuple[int, ...]:
    if indices is None:
        indices = list(range(len(args)))

    equal_to_1 = tuple(i for i, arg in zip(indices, args) if _arg_equals_1(arg))

    return equal_to_1


def config_of(
    args: list[KernelArgType],
    *,
    indices: Optional[list[int]] = None,
) -> Any:
    if indices is None:
        indices = list(range(len(args)))

    def is_aligned(x: KernelArgType, alignment: int, include_tensor: bool) -> bool:
        """
        Roughly follow triton code here:
        https://github.com/triton-lang/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        if isinstance(x, TensorArg):
            if include_tensor:
                offset_aligned = V.graph.sizevars.statically_known_multiple_of(
                    x.offset * x.dtype.itemsize,
                    alignment,  # type: ignore[arg-type]
                )
                return offset_aligned and not is_unaligned_buffer(x)
            else:
                return False
        if isinstance(x, SizeArg):
            # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
            # _maybe_evaluate_static...
            if x.name.startswith("load_seed_offset"):
                return False
            if x.expr is None:
                return False
            if isinstance(x.expr, float):
                return False
            return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)  # type: ignore[arg-type]
        if isinstance(x, WorkspaceArg):
            # We allocate the workspace ourselves, so it is always aligned
            return True
        if isinstance(x, (TMADescriptorArg, ConstexprArg)):
            return False
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    if config.triton.divisible_by_16:
        divisible_by_16 = tuple(
            i
            for i, arg in zip(indices, args)
            if is_aligned(arg, alignment=16, include_tensor=True)
        )
    else:
        divisible_by_16 = ()

    equal_to_1 = equal_1_arg_indices(args, indices=indices)

    # pyrefly: ignore [bad-argument-type]
    return AttrsDescriptorWrapper(divisible_by_16, equal_to_1)

```



## High-Level Overview


This Python file contains 0 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `should_unwrap_unspec_arg`, `signature_of`, `non_constexpr_signature`, `signature_to_meta`, `_decide_tl_dtype`, `is_unaligned_buffer`, `_arg_equals_1`, `equal_1_arg_indices`, `config_of`, `is_aligned`

**Key imports**: Any, Optional, sympy, torch, symbol_is_type, SymT, config, AttrsDescriptorWrapper, _type_of, expr_fits_within_32bit, triton_version_uses_attrs_dict, V


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, Optional
- `sympy`
- `torch`
- `torch.utils._sympy.symbol`: symbol_is_type, SymT
- `..`: config
- `..runtime.hints`: AttrsDescriptorWrapper
- `..utils`: _type_of, expr_fits_within_32bit, triton_version_uses_attrs_dict
- `..virtualized`: V


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/_inductor/codegen`):

- [`cpp_wrapper_mps.py_docs.md`](./cpp_wrapper_mps.py_docs.md)
- [`wrapper_fxir.py_docs.md`](./wrapper_fxir.py_docs.md)
- [`cpp_flex_attention_template.py_docs.md`](./cpp_flex_attention_template.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`simd_kernel_features.py_docs.md`](./simd_kernel_features.py_docs.md)
- [`block_analysis.py_docs.md`](./block_analysis.py_docs.md)
- [`cpp_wrapper_cpu_array_ref.py_docs.md`](./cpp_wrapper_cpu_array_ref.py_docs.md)
- [`cpp_bmm_template.py_docs.md`](./cpp_bmm_template.py_docs.md)
- [`python_wrapper_mtia.py_docs.md`](./python_wrapper_mtia.py_docs.md)
- [`cpp_template.py_docs.md`](./cpp_template.py_docs.md)


## Cross-References

- **File Documentation**: `triton_utils.py_docs.md`
- **Keyword Index**: `triton_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/_inductor/codegen`):

- [`wrapper_fxir.py_kw.md_docs.md`](./wrapper_fxir.py_kw.md_docs.md)
- [`simd.py_docs.md_docs.md`](./simd.py_docs.md_docs.md)
- [`mps_device_op_overrides.py_docs.md_docs.md`](./mps_device_op_overrides.py_docs.md_docs.md)
- [`simd_kernel_features.py_docs.md_docs.md`](./simd_kernel_features.py_docs.md_docs.md)
- [`segmented_tree.py_docs.md_docs.md`](./segmented_tree.py_docs.md_docs.md)
- [`wrapper.py_kw.md_docs.md`](./wrapper.py_kw.md_docs.md)
- [`mps.py_kw.md_docs.md`](./mps.py_kw.md_docs.md)
- [`cpu_device_op_overrides.py_kw.md_docs.md`](./cpu_device_op_overrides.py_kw.md_docs.md)
- [`cpp_gemm_template.py_kw.md_docs.md`](./cpp_gemm_template.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `triton_utils.py_docs.md_docs.md`
- **Keyword Index**: `triton_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
