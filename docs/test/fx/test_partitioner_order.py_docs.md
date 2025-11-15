# Documentation: `test/fx/test_partitioner_order.py`

## File Metadata

- **Path**: `test/fx/test_partitioner_order.py`
- **Size**: 2,473 bytes (2.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

from collections.abc import Mapping

import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.testing._internal.common_utils import TestCase


class DummyDevOperatorSupport(OperatorSupport):
    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        return True


class DummyPartitioner(CapabilityBasedPartitioner):
    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__(
            graph_module,
            DummyDevOperatorSupport(),
            allows_single_node_partition=True,
        )


# original graph node order is: ['x', 'add', 'add_1', 'output']
class AddModule(torch.nn.Module):
    def forward(self, x):
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z


class TestPartitionerOrder(TestCase):
    # partitoner test to check graph node order remains the same with the original graph after partitioning
    def test_partitioner_graph_node_order(self):
        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        origin_node_order = [n.name for n in traced_m.graph.nodes]
        partions = DummyPartitioner(traced_m).propose_partitions()
        partion_nodes = [list(partition.nodes) for partition in partions]
        partition_node_order = [n.name for n in partion_nodes[0]]
        self.assertTrue(partition_node_order == origin_node_order)

    # partitoner test to check graph node order remains the same during multiple runs
    def test_partitioner_multiple_runs_order(self):
        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        partitions = DummyPartitioner(traced_m).propose_partitions()
        partition_nodes = [list(partition.nodes) for partition in partitions]
        node_order = [n.name for n in partition_nodes[0]]
        for _ in range(10):
            traced_m = torch.fx.symbolic_trace(m)
            new_partion = DummyPartitioner(traced_m).propose_partitions()
            new_partion_nodes = [list(partition.nodes) for partition in new_partion]
            new_node_order = [n.name for n in new_partion_nodes[0]]
            self.assertTrue(node_order == new_node_order)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 4 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DummyDevOperatorSupport`, `DummyPartitioner`, `AddModule`, `TestPartitionerOrder`

**Functions defined**: `is_node_supported`, `__init__`, `forward`, `test_partitioner_graph_node_order`, `test_partitioner_multiple_runs_order`

**Key imports**: Mapping, torch, CapabilityBasedPartitioner, OperatorSupport, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Mapping
- `torch`
- `torch.fx.passes.infra.partitioner`: CapabilityBasedPartitioner
- `torch.fx.passes.operator_support`: OperatorSupport
- `torch.testing._internal.common_utils`: TestCase


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
python test/fx/test_partitioner_order.py
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

- **File Documentation**: `test_partitioner_order.py_docs.md`
- **Keyword Index**: `test_partitioner_order.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
