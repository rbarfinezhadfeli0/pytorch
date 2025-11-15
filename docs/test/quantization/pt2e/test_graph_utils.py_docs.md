# Documentation: `test/quantization/pt2e/test_graph_utils.py`

## File Metadata

- **Path**: `test/quantization/pt2e/test_graph_utils.py`
- **Size**: 4,365 bytes (4.26 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]
import copy
import unittest

import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization.pt2e.graph_utils import (
    find_sequential_partitions,
    get_equivalent_types,
    update_equivalent_types_dict,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    raise_on_run_directly,
    TestCase,
)


class TestGraphUtils(TestCase):
    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_conv_bn_conv_relu(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                bn_out = self.bn1(self.conv1(x))
                relu_out = torch.nn.functional.relu(bn_out)
                return self.relu2(self.conv2(relu_out))

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(  # noqa: F841
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        self.assertEqual(len(fused_partitions), 1)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU]
        )
        self.assertEqual(len(fused_partitions), 1)

        def x():
            find_sequential_partitions(
                m,
                [
                    torch.nn.Conv2d,
                    torch.nn.BatchNorm2d,
                    torch.nn.ReLU,
                    torch.nn.functional.conv2d,
                ],
            )

        self.assertRaises(ValueError, x)

    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_conv_bn_relu(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                bn_out = self.bn1(x)
                return self.relu2(self.conv2(bn_out))

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(  # noqa: F841
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        self.assertEqual(len(fused_partitions), 0)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.BatchNorm2d, torch.nn.Conv2d]
        )
        self.assertEqual(len(fused_partitions), 1)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.BatchNorm2d, torch.nn.ReLU]
        )
        self.assertEqual(len(fused_partitions), 0)

    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on Windows")
    def test_customized_equivalet_types_dict(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return torch.nn.functional.relu6(self.conv(x))

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(  # noqa: F841
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        customized_equivalent_types = get_equivalent_types()
        customized_equivalent_types.append({torch.nn.ReLU6, torch.nn.functional.relu6})
        update_equivalent_types_dict(customized_equivalent_types)
        fused_partitions = find_sequential_partitions(
            m,
            [torch.nn.Conv2d, torch.nn.ReLU6],
        )
        self.assertEqual(len(fused_partitions), 1)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")

```



## High-Level Overview


This Python file contains 4 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestGraphUtils`, `M`, `M`, `M`

**Functions defined**: `test_conv_bn_conv_relu`, `__init__`, `forward`, `x`, `test_conv_bn_relu`, `__init__`, `forward`, `test_customized_equivalet_types_dict`, `__init__`, `forward`

**Key imports**: copy, unittest, torch, torch._dynamo as torchdynamo


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/pt2e`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `unittest`
- `torch`
- `torch._dynamo as torchdynamo`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/quantization/pt2e/test_graph_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/pt2e`):

- [`test_numeric_debugger.py_docs.md`](./test_numeric_debugger.py_docs.md)
- [`test_quantize_pt2e.py_docs.md`](./test_quantize_pt2e.py_docs.md)
- [`test_quantize_pt2e_qat.py_docs.md`](./test_quantize_pt2e_qat.py_docs.md)
- [`test_representation.py_docs.md`](./test_representation.py_docs.md)
- [`test_xnnpack_quantizer.py_docs.md`](./test_xnnpack_quantizer.py_docs.md)
- [`test_metadata_porting.py_docs.md`](./test_metadata_porting.py_docs.md)
- [`test_x86inductor_quantizer.py_docs.md`](./test_x86inductor_quantizer.py_docs.md)
- [`test_duplicate_dq.py_docs.md`](./test_duplicate_dq.py_docs.md)


## Cross-References

- **File Documentation**: `test_graph_utils.py_docs.md`
- **Keyword Index**: `test_graph_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
