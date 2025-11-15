# Documentation: `docs/tools/experimental/torchfuzz/operators/gather.py_docs.md`

## File Metadata

- **Path**: `docs/tools/experimental/torchfuzz/operators/gather.py_docs.md`
- **Size**: 6,659 bytes (6.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/experimental/torchfuzz/operators/gather.py`

## File Metadata

- **Path**: `tools/experimental/torchfuzz/operators/gather.py`
- **Size**: 3,954 bytes (3.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from typing import Optional

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class GatherOperator(Operator):
    """Operator for gathering values along an axis specified by dim using indices."""

    def __init__(self):
        super().__init__("gather")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.gather"

    def can_produce(self, output_spec: Spec) -> bool:
        """Gather can produce tensors of various shapes, but not 0-dimensional tensors."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # gather requires at least one dimension
        return len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Generate input specs for gather operation.

        torch.gather(input, dim, index) returns a tensor with:
        - output.shape == index.shape
        - output[i][j][k] = input[i][j][index[i][j][k]] (for dim=2 example)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GatherOperator can only produce TensorSpec outputs")

        # The output shape matches the index shape
        output_size = output_spec.size
        dim = 0  # Gather along dimension 0 for simplicity

        # Input tensor - create a shape that matches output except for the gather dimension
        # which can be any size >= max(indices) + 1
        # For simplicity, make input larger in the gather dimension
        if len(output_size) == 1:
            # Output is 1D
            input_size = (output_size[0] + 2,)
            input_stride = (1,)
        elif len(output_size) == 2:
            # Output is 2D, make input 2D with first dim larger
            input_size = (output_size[0] + 2, output_size[1])
            input_stride = (output_size[1], 1)  # Contiguous
        else:
            # For higher dimensions
            input_size = tuple(
                s + 2 if i == dim else s for i, s in enumerate(output_size)
            )
            # Contiguous stride
            input_stride = tuple(
                int(torch.tensor(input_size[i + 1 :]).prod().item())
                if i < len(input_size) - 1
                else 1
                for i in range(len(input_size))
            )

        input_tensor_spec = TensorSpec(
            size=input_size,
            stride=input_stride,
            dtype=output_spec.dtype,
        )

        # Index tensor - same shape as output, long dtype
        index_spec = TensorSpec(
            size=output_size,
            stride=tuple(
                int(torch.tensor(output_size[i + 1 :]).prod().item())
                if i < len(output_size) - 1
                else 1
                for i in range(len(output_size))
            ),
            dtype=torch.long,
        )

        return [input_tensor_spec, index_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for gather.

        Creates appropriate indices to gather from the input tensor.
        """
        if len(input_names) != 2:
            raise ValueError("GatherOperator requires exactly two inputs")
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GatherOperator requires TensorSpec output")

        # Determine dimension
        dim = 0  # Gather along dimension 0 for simplicity

        # Generate code that creates valid indices within the input tensor's dimension
        return (
            f"_input_size_{output_name} = {input_names[0]}.size({dim})\n"
            f"_index_{output_name} = torch.randint(0, _input_size_{output_name}, {output_spec.size}, device={input_names[0]}.device)\n"
            f"{output_name} = torch.gather({input_names[0]}, {dim}, _index_{output_name})"
        )

```



## High-Level Overview

"""Operator for gathering values along an axis specified by dim using indices."""    def __init__(self):        super().__init__("gather")    @property    def torch_op_name(self) -> Optional[str]:

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GatherOperator`

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
- [`nonzero.py_docs.md`](./nonzero.py_docs.md)
- [`masked_select.py_docs.md`](./masked_select.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`tensor_pointwise.py_docs.md`](./tensor_pointwise.py_docs.md)


## Cross-References

- **File Documentation**: `gather.py_docs.md`
- **Keyword Index**: `gather.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/experimental/torchfuzz/operators`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/experimental/torchfuzz/operators`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/tools/experimental/torchfuzz/operators`):

- [`constant.py_docs.md_docs.md`](./constant.py_docs.md_docs.md)
- [`tensor_pointwise.py_kw.md_docs.md`](./tensor_pointwise.py_kw.md_docs.md)
- [`layout.py_docs.md_docs.md`](./layout.py_docs.md_docs.md)
- [`unique.py_kw.md_docs.md`](./unique.py_kw.md_docs.md)
- [`scalar_pointwise.py_docs.md_docs.md`](./scalar_pointwise.py_docs.md_docs.md)
- [`registry.py_kw.md_docs.md`](./registry.py_kw.md_docs.md)
- [`matrix_multiply.py_kw.md_docs.md`](./matrix_multiply.py_kw.md_docs.md)
- [`gather.py_kw.md_docs.md`](./gather.py_kw.md_docs.md)
- [`item.py_kw.md_docs.md`](./item.py_kw.md_docs.md)
- [`layout.py_kw.md_docs.md`](./layout.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `gather.py_docs.md_docs.md`
- **Keyword Index**: `gather.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
