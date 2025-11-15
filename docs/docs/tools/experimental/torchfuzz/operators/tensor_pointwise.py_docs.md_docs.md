# Documentation: `docs/tools/experimental/torchfuzz/operators/tensor_pointwise.py_docs.md`

## File Metadata

- **Path**: `docs/tools/experimental/torchfuzz/operators/tensor_pointwise.py_docs.md`
- **Size**: 10,803 bytes (10.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/experimental/torchfuzz/operators/tensor_pointwise.py`

## File Metadata

- **Path**: `tools/experimental/torchfuzz/operators/tensor_pointwise.py`
- **Size**: 7,653 bytes (7.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
"""Tensor pointwise operator implementation."""

import random

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec
from torchfuzz.type_promotion import (
    get_dtype_map,
    get_dtype_name,
    get_promotion_table_for_strings,
)


class PointwiseOperator(Operator):
    """Base class for element-wise pointwise operations."""

    def __init__(self, name: str, symbol: str):
        super().__init__(name)
        self.symbol = symbol

    def can_produce(self, output_spec: Spec) -> bool:
        """Tensor pointwise operations can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Decompose tensor into input tensors for pointwise operation with type promotion."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )

        # Use shared type promotion table
        promotion_table = get_promotion_table_for_strings()

        # If num_inputs > 2, promote left-to-right (e.g. (((a op b) op c) op d))
        # For simplicity, we generate the first two with promotion, rest match output dtype
        dtype_str = get_dtype_name(output_spec.dtype)
        supported_types = promotion_table.get(dtype_str, [(dtype_str, dtype_str)])

        # Pick a random promotion pattern for the first two inputs
        if num_inputs >= 2:
            dtypes = list(random.choice(supported_types))
            # For >2 inputs, fill with output dtype
            while len(dtypes) < num_inputs:
                dtypes.append(dtype_str)
        else:
            dtypes = [dtype_str] * num_inputs

        # Convert dtype strings back to torch dtypes
        dtype_map = get_dtype_map()

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=dtype_map.get(dt, output_spec.dtype),
            )
            for dt in dtypes
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for pointwise operation."""
        if len(input_names) == 2:
            return f"{output_name} = {self.torch_op_name}({input_names[0]}, {input_names[1]})"
        else:
            # Chain operations using symbols for readability
            expr = f" {self.symbol} ".join(input_names)
            return f"{output_name} = {expr}"


class AddOperator(PointwiseOperator):
    """Operator for element-wise addition."""

    def __init__(self, weight: float = 1.0):
        super().__init__("add", "+")
        self.weight = float(weight)

    @property
    def torch_op_name(self) -> str:
        return "torch.add"


class MulOperator(PointwiseOperator):
    """Operator for element-wise multiplication."""

    def __init__(self):
        super().__init__("mul", "*")

    @property
    def torch_op_name(self) -> str:
        return "torch.mul"


class SubOperator(PointwiseOperator):
    """Operator for element-wise subtraction."""

    def __init__(self):
        super().__init__("sub", "-")

    @property
    def torch_op_name(self) -> str:
        return "torch.sub"


class DivOperator(PointwiseOperator):
    """Operator for element-wise division."""

    def __init__(self):
        super().__init__("div", "/")

    @property
    def torch_op_name(self) -> str:
        return "torch.div"


class ClampOperator(Operator):
    """Operator for torch.clamp (element-wise clamping)."""

    def __init__(self):
        super().__init__("clamp")

    @property
    def torch_op_name(self) -> str:
        return "torch.clamp"

    def can_produce(self, output_spec: Spec) -> bool:
        """Clamp can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for clamp operation.

        Clamp takes:
        - input tensor (same shape and dtype as output)
        - optional min value (scalar or None)
        - optional max value (scalar or None)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ClampOperator can only produce TensorSpec outputs")

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for clamp operation."""
        if len(input_names) != 1:
            raise ValueError("Clamp requires exactly 1 input tensor")

        input_name = input_names[0]

        # Generate random min and max values for clamping
        # We'll randomly decide whether to use min, max, or both
        use_min = random.random() > 0.33
        use_max = random.random() > 0.33

        if not use_min and not use_max:
            use_min = True

        args = [input_name]
        if use_min:
            args.append("min=-1.0")
        else:
            args.append("min=None")

        if use_max:
            args.append("max=1.0")
        else:
            args.append("max=None")

        return f"{output_name} = torch.clamp({', '.join(args)})"


class CumsumOperator(Operator):
    """Operator for torch.cumsum (cumulative sum along a dimension)."""

    def __init__(self):
        super().__init__("cumsum")

    @property
    def torch_op_name(self) -> str:
        return "torch.cumsum"

    def can_produce(self, output_spec: Spec) -> bool:
        """Cumsum can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        # Cumsum needs at least 1 dimension
        if isinstance(output_spec, TensorSpec) and len(output_spec.size) == 0:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input specs for cumsum operation.

        Cumsum takes an input tensor with same shape and dtype as output.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("CumsumOperator can only produce TensorSpec outputs")

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for cumsum operation."""
        if len(input_names) != 1:
            raise ValueError("Cumsum requires exactly 1 input tensor")

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("Output spec must be a TensorSpec")

        input_name = input_names[0]

        # Choose a random valid dimension
        num_dims = len(output_spec.size)
        if num_dims == 0:
            raise ValueError("Cumsum requires tensor with at least 1 dimension")

        # Pick a random dimension index
        dim = random.randint(0, num_dims - 1)

        return f"{output_name} = torch.cumsum({input_name}, dim={dim})"

```



## High-Level Overview

"""Tensor pointwise operator implementation."""import randomimport torchfrom torchfuzz.operators.base import Operatorfrom torchfuzz.tensor_fuzzer import Spec, TensorSpecfrom torchfuzz.type_promotion import (    get_dtype_map,    get_dtype_name,    get_promotion_table_for_strings,)class PointwiseOperator(Operator):

This Python file contains 8 class(es) and 22 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PointwiseOperator`, `AddOperator`, `MulOperator`, `SubOperator`, `DivOperator`, `ClampOperator`, `CumsumOperator`

**Functions defined**: `__init__`, `can_produce`, `fuzz_inputs_specs`, `codegen`, `__init__`, `torch_op_name`, `__init__`, `torch_op_name`, `__init__`, `torch_op_name`, `__init__`, `torch_op_name`, `__init__`, `torch_op_name`, `can_produce`, `fuzz_inputs_specs`, `codegen`, `__init__`, `torch_op_name`, `can_produce`

**Key imports**: random, torch, Operator, Spec, TensorSpec


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/experimental/torchfuzz/operators`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `random`
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
- [`gather.py_docs.md`](./gather.py_docs.md)


## Cross-References

- **File Documentation**: `tensor_pointwise.py_docs.md`
- **Keyword Index**: `tensor_pointwise.py_kw.md`
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

- **File Documentation**: `tensor_pointwise.py_docs.md_docs.md`
- **Keyword Index**: `tensor_pointwise.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
