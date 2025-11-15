# Documentation: `tools/experimental/torchfuzz/operators/unique.py`

## File Metadata

- **Path**: `tools/experimental/torchfuzz/operators/unique.py`
- **Size**: 2,319 bytes (2.26 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
"""Unique operator implementation."""

from typing import Optional

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class UniqueOperator(Operator):
    """Operator for finding unique elements in a tensor."""

    def __init__(self):
        super().__init__("unique")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.unique"

    def can_produce(self, output_spec: Spec) -> bool:
        """Unique can produce 1D tensor outputs of arbitrary length without guards.

        We will synthesize an input with exactly the desired number of unique
        elements so that torch.unique returns the target size deterministically.
        """
        return isinstance(output_spec, TensorSpec) and len(output_spec.size) == 1

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        """Generate input spec for unique operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("UniqueOperator can only produce TensorSpec outputs")

        # Input can be any tensor - unique will flatten and find unique values
        input_spec = TensorSpec(
            size=(2, 3),  # Fixed size for consistency
            stride=(3, 1),  # Contiguous
            dtype=output_spec.dtype,  # Match output dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for unique with deterministic target size input (no guards)."""
        if len(input_names) != 1:
            raise ValueError("UniqueOperator requires exactly one input")
        # Desired output length and target dtype
        desired_len = output_spec.size[0] if isinstance(output_spec, TensorSpec) else 0
        # Synthesize in a wide dtype (int64) to guarantee desired_len distinct values,
        # apply unique, then cast to the target dtype. No conditionals or guards.
        return (
            f"_inp_unique_wide = torch.arange({desired_len}, device={input_names[0]}.device, dtype=torch.int64)\n"
            f"_uniq_wide = torch.unique(_inp_unique_wide)\n"
            f"{output_name} = _uniq_wide.to({input_names[0]}.dtype)"
        )

```



## High-Level Overview

"""Unique operator implementation."""from typing import Optionalfrom torchfuzz.operators.base import Operatorfrom torchfuzz.tensor_fuzzer import Spec, TensorSpecclass UniqueOperator(Operator):

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `UniqueOperator`

**Functions defined**: `__init__`, `torch_op_name`, `can_produce`, `fuzz_inputs_specs`, `codegen`

**Key imports**: Optional, Operator, Spec, TensorSpec


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/experimental/torchfuzz/operators`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
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
- [`nonzero.py_docs.md`](./nonzero.py_docs.md)
- [`masked_select.py_docs.md`](./masked_select.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`tensor_pointwise.py_docs.md`](./tensor_pointwise.py_docs.md)
- [`gather.py_docs.md`](./gather.py_docs.md)


## Cross-References

- **File Documentation**: `unique.py_docs.md`
- **Keyword Index**: `unique.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
