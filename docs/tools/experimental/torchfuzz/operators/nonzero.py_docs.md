# Documentation: `tools/experimental/torchfuzz/operators/nonzero.py`

## File Metadata

- **Path**: `tools/experimental/torchfuzz/operators/nonzero.py`
- **Size**: 3,068 bytes (3.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
"""Nonzero operator implementation."""

from typing import Optional

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class NonzeroOperator(Operator):
    """Operator for finding nonzero elements in a tensor."""

    def __init__(self):
        super().__init__("nonzero")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nonzero"

    def can_produce(self, output_spec: Spec) -> bool:
        """Nonzero produces a tensor with shape (n_nonzero, n_dims).

        We can deterministically synthesize inputs to match any 2D int64 output
        shape (k, d) without data-dependent guards by constructing an input with
        exactly k non-zero elements and d dimensions.
        """
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype in [torch.int64, torch.long]
            and len(output_spec.size) == 2
        )

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        """Generate input spec for nonzero operation.

        The actual values will be synthesized in codegen to achieve the target size.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("NonzeroOperator can only produce TensorSpec outputs")

        # Provide a placeholder spec; codegen will ignore the actual input content
        # and synthesize a tensor with desired nonzero count and dimensionality.
        d = output_spec.size[1]
        input_spec = TensorSpec(
            size=tuple([1] * d) if d > 0 else (),
            stride=tuple([1] * d) if d > 0 else (),
            dtype=torch.bool,
        )
        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for nonzero using synthesized input to match target size.

        No data-dependent conditionals/guards. Constructs an input with exactly
        k = output_spec.size[0] non-zero elements and d = output_spec.size[1] dims,
        then calls torch.nonzero on it.
        """
        if len(input_names) != 1:
            raise ValueError("NonzeroOperator requires exactly one input")
        if not isinstance(output_spec, TensorSpec) or len(output_spec.size) != 2:
            raise ValueError("NonzeroOperator requires 2D TensorSpec output")
        k = output_spec.size[0]
        d = output_spec.size[1]
        # Construct concrete shape literal like (k, 1, 1, ...)
        shape_elems = [str(k)] + ["1"] * max(0, d - 1)
        shape_literal = (
            "(" + ", ".join(shape_elems) + ("," if d == 1 else "") + ")"
            if d > 0
            else "()"
        )
        return (
            f"_x_nz = torch.zeros({shape_literal}, dtype=torch.bool, device={input_names[0]}.device)\n"
            f"_x_nz_flat = _x_nz.reshape(-1)\n"
            f"_x_nz_flat[:{k}] = True\n"
            f"{output_name} = torch.nonzero(_x_nz)"
        )

```



## High-Level Overview

"""Nonzero operator implementation."""from typing import Optionalimport torchfrom torchfuzz.operators.base import Operatorfrom torchfuzz.tensor_fuzzer import Spec, TensorSpecclass NonzeroOperator(Operator):

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `NonzeroOperator`

**Functions defined**: `__init__`, `torch_op_name`, `can_produce`, `fuzz_inputs_specs`, `codegen`

**Key imports**: Optional, torch, Operator, Spec, TensorSpec


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/experimental/torchfuzz/operators`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torchfuzz.operators.base`: Operator
- `torchfuzz.tensor_fuzzer`: Spec, TensorSpec


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`tools/experimental/torchfuzz/operators`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`item.py_docs.md`](./item.py_docs.md)
- [`argsort.py_docs.md`](./argsort.py_docs.md)
- [`constant.py_docs.md`](./constant.py_docs.md)
- [`scalar_pointwise.py_docs.md`](./scalar_pointwise.py_docs.md)
- [`masked_select.py_docs.md`](./masked_select.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`tensor_pointwise.py_docs.md`](./tensor_pointwise.py_docs.md)
- [`gather.py_docs.md`](./gather.py_docs.md)


## Cross-References

- **File Documentation**: `nonzero.py_docs.md`
- **Keyword Index**: `nonzero.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
