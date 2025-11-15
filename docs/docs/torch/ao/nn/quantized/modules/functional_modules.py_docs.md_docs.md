# Documentation: `docs/torch/ao/nn/quantized/modules/functional_modules.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/nn/quantized/modules/functional_modules.py_docs.md`
- **Size**: 12,509 bytes (12.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/nn/quantized/modules/functional_modules.py`

## File Metadata

- **Path**: `torch/ao/nn/quantized/modules/functional_modules.py`
- **Size**: 9,222 bytes (9.01 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch._ops import ops


__all__ = ["FloatFunctional", "FXFloatFunctional", "QFunctional"]


class FloatFunctional(torch.nn.Module):
    r"""State collector class for float operations.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self) -> None:
        super().__init__()
        self.activation_post_process = torch.nn.Identity()

    def forward(self, x):
        raise RuntimeError(
            "FloatFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.add(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.mul(x, y)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.mul(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        r = torch.cat(x, dim=dim)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``relu(torch.add(x,y))``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.matmul(x, y)
        r = self.activation_post_process(r)
        return r


class FXFloatFunctional(torch.nn.Module):
    r"""module to replace FloatFunctional module before FX graph mode quantization,
    since activation_post_process will be inserted in top level module directly

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def forward(self, x):
        raise RuntimeError(
            "FloatFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        return r

    r"""Operation equivalent to ``torch.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.add(x, y)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.mul(x, y)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.mul(x, y)
        return r

    r"""Operation equivalent to ``torch.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        r = torch.cat(x, dim=dim)
        return r

    r"""Operation equivalent to ``relu(torch.add(x,y))``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        return r

    r"""Operation equivalent to ``torch.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.matmul(x, y)
        return r


class QFunctional(torch.nn.Module):
    r"""Wrapper class for quantized operations.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> q_add = QFunctional()
        >>> # xdoctest: +SKIP
        >>> a = torch.quantize_per_tensor(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(a, b, 1.0, 0)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = 1.0
        self.zero_point = 0
        self.activation_post_process = torch.nn.Identity()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.scale = float(state_dict.pop(prefix + "scale"))
        self.zero_point = int(state_dict.pop(prefix + "zero_point"))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _get_name(self):
        return "QFunctional"

    def extra_repr(self):
        return f"scale={self.scale}, zero_point={self.zero_point}"

    def forward(self, x):
        raise RuntimeError(
            "Functional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.ops.quantized.add``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.add(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.add_scalar(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.mul(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.mul_scalar(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.ops.quantized.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        r = ops.quantized.cat(x, scale=self.scale, zero_point=self.zero_point, dim=dim)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.add_relu``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.add_relu(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.matmul(x, y, scale=self.scale, zero_point=self.zero_point)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        assert type(mod) is FloatFunctional, (
            "QFunctional.from_float expects an instance of FloatFunctional"
        )
        scale, zero_point = mod.activation_post_process.calculate_qparams()  # type: ignore[operator]
        new_mod = QFunctional()
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod

```



## High-Level Overview

r"""State collector class for float operations.    The instance of this class can be used instead of the ``torch.`` prefix for    some operations. See example usage below.    .. note::        This class does not provide a ``forward`` hook. Instead, you must use        one of the underlying functions (e.g. ``add``).    Examples::        >>> f_add = FloatFunctional()        >>> a = torch.tensor(3.0)        >>> b = torch.tensor(4.0)        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``    Valid operation names:        - add        - cat        - mul        - add_relu        - add_scalar        - mul_scalar

This Python file contains 9 class(es) and 31 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FloatFunctional`, `FXFloatFunctional`, `QFunctional`

**Functions defined**: `__init__`, `forward`, `add`, `add_scalar`, `mul`, `mul_scalar`, `cat`, `add_relu`, `matmul`, `forward`, `add`, `add_scalar`, `mul`, `mul_scalar`, `cat`, `add_relu`, `matmul`, `__init__`, `_save_to_state_dict`, `_load_from_state_dict`

**Key imports**: torch, Tensor, ops


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._ops`: ops


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`torch/ao/nn/quantized/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`embedding_ops.py_docs.md`](./embedding_ops.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)
- [`activation.py_docs.md`](./activation.py_docs.md)
- [`dropout.py_docs.md`](./dropout.py_docs.md)


## Cross-References

- **File Documentation**: `functional_modules.py_docs.md`
- **Keyword Index**: `functional_modules.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/nn/quantized/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/ao/nn/quantized/modules`):

- [`embedding_ops.py_kw.md_docs.md`](./embedding_ops.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`functional_modules.py_kw.md_docs.md`](./functional_modules.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`embedding_ops.py_docs.md_docs.md`](./embedding_ops.py_docs.md_docs.md)
- [`rnn.py_docs.md_docs.md`](./rnn.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`linear.py_docs.md_docs.md`](./linear.py_docs.md_docs.md)
- [`conv.py_kw.md_docs.md`](./conv.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `functional_modules.py_docs.md_docs.md`
- **Keyword Index**: `functional_modules.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
