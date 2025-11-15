# Documentation: `docs/test/distributed/tensor/experimental/test_tp_transform.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/tensor/experimental/test_tp_transform.py_docs.md`
- **Size**: 8,224 bytes (8.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/tensor/experimental/test_tp_transform.py`

## File Metadata

- **Path**: `test/distributed/tensor/experimental/test_tp_transform.py`
- **Size**: 5,602 bytes (5.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
from collections import defaultdict

import torch
from torch.distributed.tensor.experimental._tp_transform import (
    tensor_parallel_transformation,
)
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class MLPListModule(torch.nn.Module):
    """
    A dummy model with list of MLPs.
    """

    def __init__(self, num_mlps=3, bias=True):
        super().__init__()
        self.mlps = torch.nn.ModuleList()
        for _ in range(num_mlps):
            self.mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(6, 18),
                    torch.nn.ReLU(),
                    torch.nn.Linear(18, 6, bias=bias),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.chunk(x, 2, dim=1)[0]
        for mlp in self.mlps:
            x = mlp(x)
        return x + torch.ones_like(x)


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(3, 5)
        self.bn = torch.nn.BatchNorm1d(5)

    def forward(self, x):
        return self.bn(self.fc(x))


class TensorParallelTest(DTensorTestBase):
    def setUp(self) -> None:
        super().setUp()

    def assert_has_c10d_ops(
        self, gm: torch.fx.GraphModule, expected_ops_count: dict[str, int]
    ) -> None:
        actual_ops_count: dict[str, int] = defaultdict(int)
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if "c10d_functional" in str(node.target):
                    actual_ops_count[str(node.target)] += 1
        self.assertDictEqual(expected_ops_count, actual_ops_count)

    @with_comms
    def test_tp_transform_with_uncovered_op(self):
        model = DummyModel().to(device=self.device_type)
        inputs = (torch.randn(7, 3, requires_grad=False).to(device=self.device_type),)
        with torch.no_grad():
            res = model(*inputs)
            exported_program = torch.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            {"fc": ColwiseParallel},
        )
        tp_model = tp_exported_program.module()
        with torch.no_grad():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_gather to be inserted to distributed sharded fc results
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_gather_into_tensor.default": 1,
                "_c10d_functional.wait_tensor.default": 1,
            },
        )

    @with_comms
    def test_tp_transform_e2e(self):
        torch.manual_seed(0)
        model = MLPListModule(2).to(device=self.device_type)
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        parallel_strategies: dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
            "mlps.1.0": ColwiseParallel,
            "mlps.1.2": RowwiseParallel,
        }

        with torch.inference_mode():
            res = model(*inputs)
            exported_program = torch.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        tp_model = tp_exported_program.module()
        with torch.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_reduce to be inserted at the end of each MLP
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_reduce.default": 2,
                "_c10d_functional.wait_tensor.default": 2,
            },
        )

    @with_comms
    def test_tp_transform_no_bias(self):
        torch.manual_seed(0)
        model = MLPListModule(1, bias=False).to(device=self.device_type)
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        parallel_strategies: dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
        }

        with torch.inference_mode():
            res = model(*inputs)
            exported_program = torch.export.export(
                model, inputs, strict=True
            ).run_decompositions()
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        tp_model = tp_exported_program.module()
        with torch.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        self.assert_has_c10d_ops(
            tp_exported_program.graph_module,
            {
                "_c10d_functional.all_reduce.default": 1,
                "_c10d_functional.wait_tensor.default": 1,
            },
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""    A dummy model with list of MLPs.

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MLPListModule`, `DummyModel`, `TensorParallelTest`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `setUp`, `assert_has_c10d_ops`, `test_tp_transform_with_uncovered_op`, `test_tp_transform_e2e`, `test_tp_transform_no_bias`

**Key imports**: defaultdict, torch, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/tensor/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: defaultdict
- `torch`
- `torch.testing._internal.common_utils`: run_tests


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/tensor/experimental/test_tp_transform.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/tensor/experimental`):

- [`test_register_sharding.py_docs.md`](./test_register_sharding.py_docs.md)
- [`test_local_map.py_docs.md`](./test_local_map.py_docs.md)


## Cross-References

- **File Documentation**: `test_tp_transform.py_docs.md`
- **Keyword Index**: `test_tp_transform.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/tensor/experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/tensor/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/tensor/experimental/test_tp_transform.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/tensor/experimental`):

- [`test_local_map.py_kw.md_docs.md`](./test_local_map.py_kw.md_docs.md)
- [`test_register_sharding.py_docs.md_docs.md`](./test_register_sharding.py_docs.md_docs.md)
- [`test_register_sharding.py_kw.md_docs.md`](./test_register_sharding.py_kw.md_docs.md)
- [`test_local_map.py_docs.md_docs.md`](./test_local_map.py_docs.md_docs.md)
- [`test_tp_transform.py_kw.md_docs.md`](./test_tp_transform.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_tp_transform.py_docs.md_docs.md`
- **Keyword Index**: `test_tp_transform.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
