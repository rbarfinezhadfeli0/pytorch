# Documentation: `docs/test/fx/test_shape_inference.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_shape_inference.py_docs.md`
- **Size**: 7,268 bytes (7.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_shape_inference.py`

## File Metadata

- **Path**: `test/fx/test_shape_inference.py`
- **Size**: 4,171 bytes (4.07 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

import copy
import unittest
from collections import defaultdict

import torch
import torch.fx as fx
from torch._dynamo.source import LocalSource
from torch.fx.experimental.shape_inference.infer_shape import infer_shape
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv


class TestShapeInference(unittest.TestCase):
    def test_infer_symbol_values(self):
        def mksym(shape_env, value, source, dynamic_dim) -> None:
            return shape_env.create_symintnode(
                shape_env.create_symbol(
                    value,
                    source=source,
                    dynamic_dim=dynamic_dim,
                ),
                hint=value,
                source=source,
            )

        shape_env = ShapeEnv()
        N = 8
        sample = {f"s{i}": 2 for i in range(N)}
        init_symints = [
            mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
            for k, v in sample.items()
        ]
        symints = copy.deepcopy(init_symints)
        symbol_to_idx_dict = {f"s{i}": i for i in range(N)}
        padding_constraints = defaultdict(list)

        # prepare constraints strings
        constraints = []
        constraints.append(
            "The size of tensor a (s1) must match the size of tensor b (1773) at non-singleton dimension 1)"
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, (s2//2) + 12] but got: [s0, 120]."
        )
        constraints.append("shape '[s0, -1, 32]' is invalid for input of size s0*s3")
        constraints.append(
            "a and b must have same reduction dim, but got [32*s0, s3] X [20, 15]."
        )
        constraints.append(
            "a and b must have same reduction dim, but got [s0, s4 + 1568] X [5728, 1024]."
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, 40] but got: [s0, s5]."
        )
        constraints.append(
            "shape '[s0, -1, 32]' is invalid for input of size s0*s6 + 1344*s0"
        )
        constraints.append(
            "shape '[-1, 47]' is invalid for input of size 32*s0*s6 + 1344*s0"
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, 47*s6] but got: [s0*s6, 47]."
        )
        constraints.append("Split sizes add up to 4258 but got the tensor's size of s7")

        for constraint in constraints:
            infer_symbol_values(
                symints,
                init_symints,
                symbol_to_idx_dict,
                padding_constraints,
                constraint,
            )

        self.assertEqual(symints[1], 1773)
        self.assertEqual(symints[2], 216)
        self.assertEqual(symints[3], 640)
        self.assertEqual(symints[4], 4160)
        self.assertEqual(symints[5], 40)
        self.assertEqual(symints[6], 160)
        self.assertEqual(symints[7], 4258)

    def test_infer_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w_1 = torch.empty([256, 328])
                self.b_1 = torch.empty([256])
                self.w_2 = torch.empty([328, 256])
                self.b_2 = torch.empty([328])

            def forward(self, x):
                l_1 = torch.nn.functional.linear(x, self.w_1, bias=self.b_1)
                s_1 = torch.sigmoid(l_1)
                l_2 = torch.nn.functional.linear(s_1, self.w_2, bias=self.b_2)
                t_1 = torch.tanh(l_2)
                return t_1

        def generate_graph_module(model):
            gm = fx.symbolic_trace(model)
            return gm

        m = TestModule()
        gm = generate_graph_module(m)
        input_tensors = [torch.randn(1, 1)]
        infer_shape(gm, input_tensors)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestShapeInference`, `TestModule`

**Functions defined**: `test_infer_symbol_values`, `mksym`, `test_infer_shape`, `__init__`, `forward`, `generate_graph_module`

**Key imports**: copy, unittest, defaultdict, torch, torch.fx as fx, LocalSource, infer_shape, DimDynamic, ShapeEnv


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `unittest`
- `collections`: defaultdict
- `torch`
- `torch.fx as fx`
- `torch._dynamo.source`: LocalSource
- `torch.fx.experimental.shape_inference.infer_shape`: infer_shape
- `torch.fx.experimental.symbolic_shapes`: DimDynamic, ShapeEnv


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

This is a test file. Run it with:

```bash
python test/fx/test_shape_inference.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/fx`):

- [`test_dynamism.py_docs.md`](./test_dynamism.py_docs.md)
- [`test_dce_pass.py_docs.md`](./test_dce_pass.py_docs.md)
- [`test_source_matcher_utils.py_docs.md`](./test_source_matcher_utils.py_docs.md)
- [`test_graph_pickler.py_docs.md`](./test_graph_pickler.py_docs.md)
- [`named_tup.py_docs.md`](./named_tup.py_docs.md)
- [`test_cse_pass.py_docs.md`](./test_cse_pass.py_docs.md)
- [`test_fx_traceback.py_docs.md`](./test_fx_traceback.py_docs.md)
- [`test_gradual_type.py_docs.md`](./test_gradual_type.py_docs.md)
- [`test_pass_infra.py_docs.md`](./test_pass_infra.py_docs.md)
- [`test_lazy_graph_module.py_docs.md`](./test_lazy_graph_module.py_docs.md)


## Cross-References

- **File Documentation**: `test_shape_inference.py_docs.md`
- **Keyword Index**: `test_shape_inference.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/fx`, which is part of the **testing infrastructure**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/fx/test_shape_inference.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/fx`):

- [`named_tup.py_kw.md_docs.md`](./named_tup.py_kw.md_docs.md)
- [`test_dynamism.py_kw.md_docs.md`](./test_dynamism.py_kw.md_docs.md)
- [`test_fx_traceback.py_docs.md_docs.md`](./test_fx_traceback.py_docs.md_docs.md)
- [`test_fx_xform_observer.py_docs.md_docs.md`](./test_fx_xform_observer.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_fx_xform_observer.py_kw.md_docs.md`](./test_fx_xform_observer.py_kw.md_docs.md)
- [`test_fx_node_hook.py_kw.md_docs.md`](./test_fx_node_hook.py_kw.md_docs.md)
- [`test_partitioner_order.py_docs.md_docs.md`](./test_partitioner_order.py_docs.md_docs.md)
- [`test_subgraph_rewriter.py_kw.md_docs.md`](./test_subgraph_rewriter.py_kw.md_docs.md)
- [`test_fx_split.py_docs.md_docs.md`](./test_fx_split.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_shape_inference.py_docs.md_docs.md`
- **Keyword Index**: `test_shape_inference.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
