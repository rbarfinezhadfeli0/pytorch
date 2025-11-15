# Documentation: `docs/torch/_inductor/codegen/cutedsl/cutedsl_op_overrides.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/cutedsl/cutedsl_op_overrides.py_docs.md`
- **Size**: 15,878 bytes (15.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/cutedsl/cutedsl_op_overrides.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/cutedsl/cutedsl_op_overrides.py`
- **Size**: 12,937 bytes (12.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""
CuteDSL-specific operation overrides for pointwise operations.

This module provides CuteDSL implementations of common operations used in
template kernels, particularly for flex attention modifications.
"""

import math
from typing import Optional, Union

import sympy

import torch
from torch._inductor.codegen.common import CSEVariable, OpOverrides
from torch._inductor.virtualized import OpsValue, V
from torch.utils._sympy.value_ranges import ValueRanges


CuteDSLArg = Union[CSEVariable, str]


def upcast_compute_type(dtype: torch.dtype) -> torch.dtype:
    """Maybe upcast [b]float16 to float32"""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


class CuteDSLOpOverrides(OpOverrides):
    """
    CuteDSL-specific operation overrides that generate code using CuteDSL syntax.

    CuteDSL TensorSSA objects have built-in operator overloads (__add__, __mul__, etc.)
    and math functions (cute.math.exp, cute.math.sqrt, etc.)
    """

    TORCH_TO_CUTE_DTYPE = {
        torch.float16: "cutlass.Float16",
        torch.bfloat16: "cutlass.BFloat16",
        torch.float32: "cutlass.Float32",
        torch.float64: "cutlass.Float64",
        torch.int8: "cutlass.Int8",
        torch.int16: "cutlass.Int16",
        torch.int32: "cutlass.Int32",
        torch.int64: "cutlass.Int64",
        torch.bool: "cutlass.Boolean",
        torch.float8_e4m3fn: "cutlass.Float8E4M3FN",
        torch.float8_e5m2: "cutlass.Float8E5M2",
    }

    # Math constants
    LOG2_E = 1.4426950408889634  # 1/ln(2) for converting natural exp to base-2 exp

    @staticmethod
    def _ensure_tensor_ssa(arg: CuteDSLArg, template_tensor: CuteDSLArg) -> str:
        """
        Convert scalar arguments to TensorSSA using cute.full_like if needed.

        Args:
            arg: The argument to check (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            template_tensor: A tensor argument to use as template for full_like

        Returns:
            String representation suitable for CuteDSL operations
        """
        if isinstance(arg, CSEVariable):
            return str(arg)

        if isinstance(arg, OpsValue) and isinstance(arg.value, CSEVariable):
            return str(arg.value)

        if isinstance(template_tensor, CSEVariable):
            return f"cute.full_like({template_tensor}, {arg})"

        return str(arg)

    @staticmethod
    def _extract_dtype_and_bounds(
        *args: CuteDSLArg,
    ) -> tuple[Optional[torch.dtype], ValueRanges[sympy.Expr]]:
        """Extract dtype and bounds from CSEVariable arguments."""
        for arg in args:
            if isinstance(arg, CSEVariable):
                return arg.dtype, arg.bounds
        return None, ValueRanges.unknown()

    @staticmethod
    def _apply_binary_op(a: CuteDSLArg, b: CuteDSLArg, op_format: str) -> CuteDSLArg:
        """
        Apply a binary operation with automatic scalar-to-tensor conversion.

        CuteDSL requires both operands to be TensorSSA objects for tensor operations.
        This helper automatically converts scalar arguments to TensorSSA using
        cute.full_like when at least one argument is a tensor (CSEVariable).

        Args:
            a: First operand (CSEVariable for tensors, str for scalars)
            b: Second operand (CSEVariable for tensors, str for scalars)
            op_format: Format string with {a} and {b} placeholders for the operation

        Returns:
            CSEVariable if at least one operand is a CSEVariable, otherwise string
        """
        tensor_arg = (
            a
            if isinstance(a, CSEVariable)
            else b
            if isinstance(b, CSEVariable)
            else None
        )
        if tensor_arg is not None:
            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(a, tensor_arg)
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(b, tensor_arg)
            result_expr = op_format.format(a=a_ssa, b=b_ssa)

            dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b)

            # Create and return CSEVariable using CSE generation for caching
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=bounds, dtype=dtype
            )

        return op_format.format(a=a, b=b)

    @staticmethod
    def _apply_unary_op(x: CuteDSLArg, op_format: str) -> CuteDSLArg:
        """
        Apply a unary operation, returning CSEVariable if input is CSEVariable.

        Args:
            x: Input operand (CSEVariable for tensors, str for scalars)
            op_format: Format string with {x} placeholder for the operation

        Returns:
            CSEVariable if input is a CSEVariable, otherwise string
        """
        if isinstance(x, CSEVariable):
            result_expr = op_format.format(x=str(x))
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=x.bounds, dtype=x.dtype
            )

        return op_format.format(x=x)

    @staticmethod
    def constant(value: Union[bool, float, int], dtype: torch.dtype) -> str:
        """Generate CuteDSL constant representation."""
        if value == float("-inf"):
            return "float('-inf')"
        elif value == float("inf"):
            return "float('inf')"
        elif math.isnan(value):
            return "float('nan')"
        return repr(value)

    @staticmethod
    def add(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} + {b})")

    @staticmethod
    def mul(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} * {b})")

    @staticmethod
    def sub(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} - {b})")

    @staticmethod
    def truediv(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} / {b})")

    @staticmethod
    def mod(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} % {b})")

    @staticmethod
    def remainder(a, b):
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} % {b})")

    @staticmethod
    def exp(x: CuteDSLArg) -> CuteDSLArg:
        """Exponential using CuteDSL cute.math.exp function."""
        return CuteDSLOpOverrides._apply_unary_op(
            x, f"cute.math.exp2({{x}} * {CuteDSLOpOverrides.LOG2_E})"
        )

    @staticmethod
    def sqrt(x: CuteDSLArg) -> CuteDSLArg:
        """Square root using CuteDSL cute.math.sqrt function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.sqrt({x})")

    @staticmethod
    def log(x: CuteDSLArg) -> CuteDSLArg:
        """Natural logarithm using CuteDSL cute.math.log function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.log({x})")

    @staticmethod
    def cos(x: CuteDSLArg) -> CuteDSLArg:
        """Cosine using CuteDSL cute.math.cos function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.cos({x})")

    @staticmethod
    def sin(x: CuteDSLArg) -> CuteDSLArg:
        """Sine using CuteDSL cute.math.sin function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.sin({x})")

    @staticmethod
    def erf(x: CuteDSLArg) -> CuteDSLArg:
        """Error function using CuteDSL cute.math.erf function."""
        return CuteDSLOpOverrides._apply_unary_op(x, "cute.math.erf({x})")

    @staticmethod
    def maximum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        raise NotImplementedError("TODO: maximum is not supported yet for TensorSSA")

    @staticmethod
    def minimum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        raise NotImplementedError("TODO: minimum is not supported yet for TensorSSA")

    @staticmethod
    def where(
        condition: CuteDSLArg,
        a: CuteDSLArg,
        b: CuteDSLArg,
    ) -> CuteDSLArg:
        """Conditional selection - handles both CSEVariable and string inputs."""
        # Find a tensor argument to use as template for full_like
        # Priority: use 'a' if it's a tensor, else use 'b', else condition
        tensor_arg = (
            a
            if isinstance(a, CSEVariable)
            else (
                b
                if isinstance(b, CSEVariable)
                else condition
                if isinstance(condition, CSEVariable)
                else None
            )
        )

        if tensor_arg is not None:
            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(a, tensor_arg)
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(b, tensor_arg)
            result_expr = f"cute.where({condition}, {a_ssa}, {b_ssa})"

            dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(
                a, b, condition
            )

            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=bounds, dtype=dtype
            )

        return f"cute.where({condition}, {a}, {b})"

    @staticmethod
    def pow(a: CuteDSLArg, b: CuteDSLArg):
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} ** {b})")

    @staticmethod
    def abs(x: CuteDSLArg) -> CuteDSLArg:
        """Absolute value using CuteDSL cute.math.abs function."""
        if isinstance(x, CSEVariable):
            x_dtype = x.dtype
        elif isinstance(x, OpsValue) and isinstance(x.value, CSEVariable):
            x_dtype = x.value.dtype
        else:
            x_dtype = torch.float32

        abs_op = (
            "mlir_math.absf"
            if x_dtype in (torch.float16, torch.bfloat16, torch.float32)
            else "mlir_math.absi"
        )
        return CuteDSLOpOverrides._apply_unary_op(
            # pyrefly: ignore [bad-argument-type]
            x,
            f"cute.TensorSSA({abs_op}({{x}}), {{x}}.shape, {{x}}.dtype)",
        )

    @staticmethod
    def neg(x: CuteDSLArg) -> CuteDSLArg:
        """Negation using CuteDSL TensorSSA __neg__ operator."""
        # TODO: See https://github.com/NVIDIA/cutlass/issues/2584
        return CuteDSLOpOverrides._apply_unary_op(
            x, "cute.TensorSSA(-{x}, {x}.shape, {x}.dtype)"
        )

    @staticmethod
    def to_dtype(
        x: CuteDSLArg, dtype: torch.dtype, src_dtype=None, use_compute_types=True
    ) -> CuteDSLArg:
        """Type conversion using CuteDSL TensorSSA.to(Type[Numeric]).

        Maps torch dtypes to cutlass.cute.typing numeric types and emits
        `{x}.to(cute.typing.<Type>)`.

        Raises NotImplementedError for unsigned integer and unsupported dtypes.
        """
        # Always convert up from bf16 and fp16 TODO on configuring
        dtype = upcast_compute_type(dtype)

        cute_type = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        if cute_type is None:
            raise NotImplementedError(
                f"CuteDSL dtype cast not implemented for torch dtype: {dtype}"
            )

        if isinstance(x, CSEVariable):
            result_expr = f"{str(x)}.to({cute_type})"
            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=x.bounds, dtype=dtype
            )

        return f"{x}.to({cute_type})"

    @staticmethod
    def tanh(x0: CuteDSLArg) -> CuteDSLArg:
        """Hyperbolic tangent using CuteDSL cute.math.tanh function."""
        return CuteDSLOpOverrides._apply_unary_op(x0, "cute.math.tanh({x})")

    # Logical operations
    @staticmethod
    def logical_and(x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} and {b})")

    @staticmethod
    def logical_or(x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} or {b})")

    @staticmethod
    def logical_not(a):
        """Logical NOT."""
        return CuteDSLOpOverrides._apply_unary_op(a, "({x} == 0)")

    # Comparison operations
    @staticmethod
    def eq(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.eq({a}, {b})")

    @staticmethod
    def ne(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.ne({a}, {b})")

    @staticmethod
    def lt(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.lt({a}, {b})")

    @staticmethod
    def le(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.le({a}, {b})")

    @staticmethod
    def gt(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.gt({a}, {b})")

    @staticmethod
    def ge(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.ge({a}, {b})")

```



## High-Level Overview

"""CuteDSL-specific operation overrides for pointwise operations.This module provides CuteDSL implementations of common operations used intemplate kernels, particularly for flex attention modifications.

This Python file contains 1 class(es) and 35 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CuteDSLOpOverrides`

**Functions defined**: `upcast_compute_type`, `_ensure_tensor_ssa`, `_extract_dtype_and_bounds`, `_apply_binary_op`, `_apply_unary_op`, `constant`, `add`, `mul`, `sub`, `truediv`, `mod`, `remainder`, `exp`, `sqrt`, `log`, `cos`, `sin`, `erf`, `maximum`, `minimum`

**Key imports**: math, Optional, Union, sympy, torch, CSEVariable, OpOverrides, OpsValue, V, ValueRanges


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen/cutedsl`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `typing`: Optional, Union
- `sympy`
- `torch`
- `torch._inductor.codegen.common`: CSEVariable, OpOverrides
- `torch._inductor.virtualized`: OpsValue, V
- `torch.utils._sympy.value_ranges`: ValueRanges


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

Files in the same folder (`torch/_inductor/codegen/cutedsl`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`cutedsl_kernel.py_docs.md`](./cutedsl_kernel.py_docs.md)
- [`cutedsl_template.py_docs.md`](./cutedsl_template.py_docs.md)
- [`_cutedsl_utils.py_docs.md`](./_cutedsl_utils.py_docs.md)
- [`cutedsl_scheduling.py_docs.md`](./cutedsl_scheduling.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)


## Cross-References

- **File Documentation**: `cutedsl_op_overrides.py_docs.md`
- **Keyword Index**: `cutedsl_op_overrides.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen/cutedsl`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen/cutedsl`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_inductor/codegen/cutedsl`):

- [`cutedsl_scheduling.py_docs.md_docs.md`](./cutedsl_scheduling.py_docs.md_docs.md)
- [`cutedsl_scheduling.py_kw.md_docs.md`](./cutedsl_scheduling.py_kw.md_docs.md)
- [`cutedsl_kernel.py_docs.md_docs.md`](./cutedsl_kernel.py_docs.md_docs.md)
- [`cutedsl_template.py_docs.md_docs.md`](./cutedsl_template.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`_cutedsl_utils.py_docs.md_docs.md`](./_cutedsl_utils.py_docs.md_docs.md)
- [`cutedsl_template.py_kw.md_docs.md`](./cutedsl_template.py_kw.md_docs.md)
- [`cutedsl_op_overrides.py_kw.md_docs.md`](./cutedsl_op_overrides.py_kw.md_docs.md)
- [`_cutedsl_utils.py_kw.md_docs.md`](./_cutedsl_utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cutedsl_op_overrides.py_docs.md_docs.md`
- **Keyword Index**: `cutedsl_op_overrides.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
