# Documentation: `functorch/dim/op_properties.py`

## File Metadata

- **Path**: `functorch/dim/op_properties.py`
- **Size**: 6,687 bytes (6.53 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch


# pointwise operators can go through a faster pathway

tensor_magic_methods = ["add", ""]
pointwise_magic_methods_with_reverse = (
    "add",
    "sub",
    "mul",
    "floordiv",
    "div",
    "truediv",
    "mod",
    "pow",
    "lshift",
    "rshift",
    "and",
    "or",
    "xor",
)
pointwise_magic_methods = (
    *(x for m in pointwise_magic_methods_with_reverse for x in (m, "r" + m)),
    "eq",
    "gt",
    "le",
    "lt",
    "ge",
    "gt",
    "ne",
    "neg",
    "pos",
    "abs",
    "invert",
    "iadd",
    "isub",
    "imul",
    "ifloordiv",
    "idiv",
    "itruediv",
    "imod",
    "ipow",
    "ilshift",
    "irshift",
    "iand",
    "ior",
    "ixor",
    "int",
    "long",
    "float",
    "complex",
)

pointwise_methods = (*(f"__{m}__" for m in pointwise_magic_methods),)

pointwise = (
    *(getattr(torch.Tensor, m) for m in pointwise_methods),
    torch.nn.functional.dropout,
    torch.where,
    torch.Tensor.abs,
    torch.abs,
    torch.Tensor.acos,
    torch.acos,
    torch.Tensor.acosh,
    torch.acosh,
    torch.Tensor.add,
    torch.add,
    torch.Tensor.addcdiv,
    torch.addcdiv,
    torch.Tensor.addcmul,
    torch.addcmul,
    torch.Tensor.addr,
    torch.addr,
    torch.Tensor.angle,
    torch.angle,
    torch.Tensor.asin,
    torch.asin,
    torch.Tensor.asinh,
    torch.asinh,
    torch.Tensor.atan,
    torch.atan,
    torch.Tensor.atan2,
    torch.atan2,
    torch.Tensor.atanh,
    torch.atanh,
    torch.Tensor.bitwise_and,
    torch.bitwise_and,
    torch.Tensor.bitwise_left_shift,
    torch.bitwise_left_shift,
    torch.Tensor.bitwise_not,
    torch.bitwise_not,
    torch.Tensor.bitwise_or,
    torch.bitwise_or,
    torch.Tensor.bitwise_right_shift,
    torch.bitwise_right_shift,
    torch.Tensor.bitwise_xor,
    torch.bitwise_xor,
    torch.Tensor.ceil,
    torch.ceil,
    torch.celu,
    torch.nn.functional.celu,
    torch.Tensor.clamp,
    torch.clamp,
    torch.Tensor.clamp_max,
    torch.clamp_max,
    torch.Tensor.clamp_min,
    torch.clamp_min,
    torch.Tensor.copysign,
    torch.copysign,
    torch.Tensor.cos,
    torch.cos,
    torch.Tensor.cosh,
    torch.cosh,
    torch.Tensor.deg2rad,
    torch.deg2rad,
    torch.Tensor.digamma,
    torch.digamma,
    torch.Tensor.div,
    torch.div,
    torch.dropout,
    torch.nn.functional.dropout,
    torch.nn.functional.elu,
    torch.Tensor.eq,
    torch.eq,
    torch.Tensor.erf,
    torch.erf,
    torch.Tensor.erfc,
    torch.erfc,
    torch.Tensor.erfinv,
    torch.erfinv,
    torch.Tensor.exp,
    torch.exp,
    torch.Tensor.exp2,
    torch.exp2,
    torch.Tensor.expm1,
    torch.expm1,
    torch.feature_dropout,
    torch.Tensor.float_power,
    torch.float_power,
    torch.Tensor.floor,
    torch.floor,
    torch.Tensor.floor_divide,
    torch.floor_divide,
    torch.Tensor.fmod,
    torch.fmod,
    torch.Tensor.frac,
    torch.frac,
    torch.Tensor.frexp,
    torch.frexp,
    torch.Tensor.gcd,
    torch.gcd,
    torch.Tensor.ge,
    torch.ge,
    torch.nn.functional.gelu,
    torch.nn.functional.glu,
    torch.Tensor.gt,
    torch.gt,
    torch.Tensor.hardshrink,
    torch.hardshrink,
    torch.nn.functional.hardshrink,
    torch.nn.functional.hardsigmoid,
    torch.nn.functional.hardswish,
    torch.nn.functional.hardtanh,
    torch.Tensor.heaviside,
    torch.heaviside,
    torch.Tensor.hypot,
    torch.hypot,
    torch.Tensor.i0,
    torch.i0,
    torch.Tensor.igamma,
    torch.igamma,
    torch.Tensor.igammac,
    torch.igammac,
    torch.Tensor.isclose,
    torch.isclose,
    torch.Tensor.isfinite,
    torch.isfinite,
    torch.Tensor.isinf,
    torch.isinf,
    torch.Tensor.isnan,
    torch.isnan,
    torch.Tensor.isneginf,
    torch.isneginf,
    torch.Tensor.isposinf,
    torch.isposinf,
    torch.Tensor.isreal,
    torch.isreal,
    torch.Tensor.kron,
    torch.kron,
    torch.Tensor.lcm,
    torch.lcm,
    torch.Tensor.ldexp,
    torch.ldexp,
    torch.Tensor.le,
    torch.le,
    torch.nn.functional.leaky_relu,
    torch.Tensor.lerp,
    torch.lerp,
    torch.Tensor.lgamma,
    torch.lgamma,
    torch.Tensor.log,
    torch.log,
    torch.Tensor.log10,
    torch.log10,
    torch.Tensor.log1p,
    torch.log1p,
    torch.Tensor.log2,
    torch.log2,
    torch.nn.functional.logsigmoid,
    torch.Tensor.logical_and,
    torch.logical_and,
    torch.Tensor.logical_not,
    torch.logical_not,
    torch.Tensor.logical_or,
    torch.logical_or,
    torch.Tensor.logical_xor,
    torch.logical_xor,
    torch.Tensor.logit,
    torch.logit,
    torch.Tensor.lt,
    torch.lt,
    torch.Tensor.maximum,
    torch.maximum,
    torch.Tensor.minimum,
    torch.minimum,
    torch.nn.functional.mish,
    torch.Tensor.mvlgamma,
    torch.mvlgamma,
    torch.Tensor.nan_to_num,
    torch.nan_to_num,
    torch.Tensor.ne,
    torch.ne,
    torch.Tensor.neg,
    torch.neg,
    torch.Tensor.nextafter,
    torch.nextafter,
    torch.Tensor.outer,
    torch.outer,
    torch.polar,
    torch.Tensor.polygamma,
    torch.polygamma,
    torch.Tensor.positive,
    torch.positive,
    torch.Tensor.pow,
    torch.pow,
    torch.Tensor.prelu,
    torch.prelu,
    torch.nn.functional.prelu,
    torch.Tensor.rad2deg,
    torch.rad2deg,
    torch.Tensor.reciprocal,
    torch.reciprocal,
    torch.Tensor.relu,
    torch.relu,
    torch.nn.functional.relu,
    torch.nn.functional.relu6,
    torch.Tensor.remainder,
    torch.remainder,
    torch.Tensor.round,
    torch.round,
    torch.rrelu,
    torch.nn.functional.rrelu,
    torch.Tensor.rsqrt,
    torch.rsqrt,
    torch.rsub,
    torch.selu,
    torch.nn.functional.selu,
    torch.Tensor.sgn,
    torch.sgn,
    torch.Tensor.sigmoid,
    torch.sigmoid,
    torch.nn.functional.sigmoid,
    torch.Tensor.sign,
    torch.sign,
    torch.Tensor.signbit,
    torch.signbit,
    torch.nn.functional.silu,
    torch.Tensor.sin,
    torch.sin,
    torch.Tensor.sinc,
    torch.sinc,
    torch.Tensor.sinh,
    torch.sinh,
    torch.nn.functional.softplus,
    torch.nn.functional.softshrink,
    torch.Tensor.sqrt,
    torch.sqrt,
    torch.Tensor.square,
    torch.square,
    torch.Tensor.sub,
    torch.sub,
    torch.Tensor.tan,
    torch.tan,
    torch.Tensor.tanh,
    torch.tanh,
    torch.nn.functional.tanh,
    torch.threshold,
    torch.nn.functional.threshold,
    torch.trapz,
    torch.Tensor.true_divide,
    torch.true_divide,
    torch.Tensor.trunc,
    torch.trunc,
    torch.Tensor.xlogy,
    torch.xlogy,
    torch.rand_like,
)

```



## High-Level Overview


This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `functorch/dim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`functorch/dim`):

- [`magic_trace.py_docs.md`](./magic_trace.py_docs.md)
- [`_wrap.py_docs.md`](./_wrap.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_dim_entry.py_docs.md`](./_dim_entry.py_docs.md)
- [`_py_inst_decoder.py_docs.md`](./_py_inst_decoder.py_docs.md)
- [`wrap_type.py_docs.md`](./wrap_type.py_docs.md)
- [`_enable_all_layers.py_docs.md`](./_enable_all_layers.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_tensor_info.py_docs.md`](./_tensor_info.py_docs.md)


## Cross-References

- **File Documentation**: `op_properties.py_docs.md`
- **Keyword Index**: `op_properties.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
