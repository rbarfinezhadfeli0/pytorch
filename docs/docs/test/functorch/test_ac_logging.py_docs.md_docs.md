# Documentation: `docs/test/functorch/test_ac_logging.py_docs.md`

## File Metadata

- **Path**: `docs/test/functorch/test_ac_logging.py_docs.md`
- **Size**: 9,377 bytes (9.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/functorch/test_ac_logging.py`

## File Metadata

- **Path**: `test/functorch/test_ac_logging.py`
- **Size**: 6,506 bytes (6.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: functorch"]
from unittest.mock import MagicMock, patch

from torch._functorch._activation_checkpointing.ac_logging_utils import (
    create_activation_checkpointing_logging_structure_payload,
    create_joint_graph_edges,
    create_joint_graph_node_information,
    create_structured_trace_for_min_cut_info,
)
from torch.fx import Graph, Node
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAcLogging(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.graph: MagicMock = MagicMock(spec=Graph)
        self.node1: MagicMock = MagicMock(spec=Node)
        self.node2: MagicMock = MagicMock(spec=Node)

        self.node1.name = "node1"
        self.node1.target = "target1"
        self.node1.meta = {
            "tensor_meta": MagicMock(shape=(2, 2)),
            "stack_trace": "trace1",
        }
        self.node1.all_input_nodes = []

        self.node2.name = "node2"
        self.node2.target = "target2"
        self.node2.meta = {"tensor_meta": None, "stack_trace": "trace2"}
        self.node2.all_input_nodes = [self.node1]

        self.graph.nodes = [self.node1, self.node2]

        self.all_recomputable_banned_nodes: list[Node] = [self.node1]
        self.saved_node_idxs: list[int] = [0]
        self.recomputable_node_idxs: list[int] = []
        self.expected_runtime: int = 100
        self.memories_banned_nodes: list[int] = [50]
        self.normalized_memories_banned_nodes: list[float] = [0.10344827586206896]
        self.runtimes_banned_nodes: list[int] = [10]
        self.min_cut_saved_values: list[Node] = [self.node1]

    def test_create_joint_graph_node_information(self) -> None:
        recomputable_node_info: dict[str, int] = {"node1": 0}
        expected_output: dict[str, dict] = {
            "node1": {
                "index": 0,
                "name": "node1",
                "is_recomputable_candidate": True,
                "target": "target1",
                "shape": "(2, 2)",
                "input_arguments": [],
                "stack_trace": "trace1",
                "recomputable_candidate_info": {"recomputable_node_idx": 0},
            },
            "node2": {
                "index": 1,
                "name": "node2",
                "is_recomputable_candidate": False,
                "target": "target2",
                "shape": "[]",
                "input_arguments": ["node1"],
                "stack_trace": "trace2",
            },
        }
        result = create_joint_graph_node_information(self.graph, recomputable_node_info)
        self.assertEqual(result, expected_output)

    def test_create_joint_graph_edges(self) -> None:
        expected_edges: list[tuple[str, str]] = [("node1", "node2")]
        result = create_joint_graph_edges(self.graph)
        self.assertEqual(result, expected_edges)

    def test_create_activation_checkpointing_logging_structure_payload(self) -> None:
        input_joint_graph_node_information: dict[str, dict] = {
            "node1": {
                "index": 0,
                "name": "node1",
                "is_recomputable_candidate": True,
                "target": "target1",
                "shape": "(2, 2)",
                "input_arguments": [],
                "stack_trace": "trace1",
                "recomputable_candidate_info": {"recomputable_node_idx": 0},
            }
        }
        joint_graph_edges: list[tuple[str, str]] = [("node1", "node2")]
        expected_payload: dict[str, any] = {
            "Joint Graph Size": 2,
            "Joint Graph Edges": {"Total": 1, "Edges": joint_graph_edges},
            "Joint Graph Node Information": input_joint_graph_node_information,
            "Recomputable Banned Nodes Order": ["node1"],
            "Expected Runtime": self.expected_runtime,
            "Knapsack Saved Nodes": self.saved_node_idxs,
            "Knapsack Recomputed Nodes": self.recomputable_node_idxs,
            "Knapsack Input Memories": self.normalized_memories_banned_nodes,
            "Absolute Memories": self.memories_banned_nodes,
            "Knapsack Input Runtimes": self.runtimes_banned_nodes,
            "Min Cut Solution Saved Values": ["node1"],
        }
        result = create_activation_checkpointing_logging_structure_payload(
            joint_graph=self.graph,
            joint_graph_node_information=input_joint_graph_node_information,
            joint_graph_edges=joint_graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            expected_runtime=self.expected_runtime,
            saved_node_idxs=self.saved_node_idxs,
            recomputable_node_idxs=self.recomputable_node_idxs,
            memories_banned_nodes=self.memories_banned_nodes,
            normalized_memories_banned_nodes=self.normalized_memories_banned_nodes,
            runtimes_banned_nodes=self.runtimes_banned_nodes,
            min_cut_saved_values=self.min_cut_saved_values,
        )
        self.assertEqual(result, expected_payload)

    @patch(
        "torch._functorch._activation_checkpointing.ac_logging_utils.trace_structured"
    )
    @patch("json.dumps", return_value="mocked_payload")
    def test_create_structured_trace_for_min_cut_info(
        self, mock_json_dumps: MagicMock, mock_trace_structured: MagicMock
    ) -> None:
        create_structured_trace_for_min_cut_info(
            joint_graph=self.graph,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            saved_node_idxs=self.saved_node_idxs,
            recomputable_node_idxs=self.recomputable_node_idxs,
            expected_runtime=self.expected_runtime,
            memories_banned_nodes=self.memories_banned_nodes,
            normalized_memories_banned_nodes=self.normalized_memories_banned_nodes,
            runtimes_banned_nodes=self.runtimes_banned_nodes,
            min_cut_saved_values=self.min_cut_saved_values,
        )

        self.assertEqual(mock_trace_structured.call_count, 1)

        metadata_fn_result = mock_trace_structured.call_args[1]["metadata_fn"]()
        payload_fn_result = mock_trace_structured.call_args[1]["payload_fn"]()

        self.assertEqual(
            metadata_fn_result,
            {
                "name": "min_cut_information",
                "encoding": "json",
            },
        )
        self.assertEqual(payload_fn_result, "mocked_payload")

        mock_json_dumps.assert_called_once()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAcLogging`

**Functions defined**: `setUp`, `test_create_joint_graph_node_information`, `test_create_joint_graph_edges`, `test_create_activation_checkpointing_logging_structure_payload`, `test_create_structured_trace_for_min_cut_info`

**Key imports**: MagicMock, patch, Graph, Node, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest.mock`: MagicMock, patch
- `torch.fx`: Graph, Node
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/functorch/test_ac_logging.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/functorch`):

- [`test_vmap.py_docs.md`](./test_vmap.py_docs.md)
- [`test_rearrange.py_docs.md`](./test_rearrange.py_docs.md)
- [`test_aot_joint_with_descriptors.py_docs.md`](./test_aot_joint_with_descriptors.py_docs.md)
- [`functorch_additional_op_db.py_docs.md`](./functorch_additional_op_db.py_docs.md)
- [`xfail_suggester.py_docs.md`](./xfail_suggester.py_docs.md)
- [`discover_coverage.py_docs.md`](./discover_coverage.py_docs.md)
- [`test_eager_transforms.py_docs.md`](./test_eager_transforms.py_docs.md)
- [`test_ac.py_docs.md`](./test_ac.py_docs.md)
- [`common_utils.py_docs.md`](./common_utils.py_docs.md)
- [`test_logging.py_docs.md`](./test_logging.py_docs.md)


## Cross-References

- **File Documentation**: `test_ac_logging.py_docs.md`
- **Keyword Index**: `test_ac_logging.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/functorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python docs/test/functorch/test_ac_logging.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/functorch`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_aot_joint_with_descriptors.py_kw.md_docs.md`](./test_aot_joint_with_descriptors.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_eager_transforms.py_docs.md_docs.md`](./test_eager_transforms.py_docs.md_docs.md)
- [`functorch_additional_op_db.py_kw.md_docs.md`](./functorch_additional_op_db.py_kw.md_docs.md)
- [`test_ac_knapsack.py_docs.md_docs.md`](./test_ac_knapsack.py_docs.md_docs.md)
- [`common_utils.py_kw.md_docs.md`](./common_utils.py_kw.md_docs.md)
- [`test_logging.py_kw.md_docs.md`](./test_logging.py_kw.md_docs.md)
- [`test_rearrange.py_kw.md_docs.md`](./test_rearrange.py_kw.md_docs.md)
- [`test_dims.py_kw.md_docs.md`](./test_dims.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_ac_logging.py_docs.md_docs.md`
- **Keyword Index**: `test_ac_logging.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
