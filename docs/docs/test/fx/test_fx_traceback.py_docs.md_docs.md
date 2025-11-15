# Documentation: `docs/test/fx/test_fx_traceback.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_fx_traceback.py_docs.md`
- **Size**: 14,610 bytes (14.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_fx_traceback.py`

## File Metadata

- **Path**: `test/fx/test_fx_traceback.py`
- **Size**: 10,625 bytes (10.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

import torch
from torch._inductor.compile_fx import aot_export_module
from torch.export import default_decompositions
from torch.fx.traceback import get_graph_provenance_json, NodeSource, NodeSourceAction
from torch.testing._internal.common_utils import TestCase


CREATE_STR = NodeSourceAction.CREATE.name.lower()


class TestFXNodeSource(TestCase):
    def test_node_source(self):
        node_source = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        self.assertExpectedInline(
            node_source.print_readable().strip(),
            """(name=, pass_name=test_pass, action=create, graph_id=-1)""",
        )
        dummy_source_dict = {
            "name": "",
            "target": "",
            "pass_name": "test_pass",
            "action": CREATE_STR,
            "graph_id": -1,
            "from_node": [],
        }
        self.assertEqual(
            node_source.to_dict(),
            dummy_source_dict,
        )

        self.assertEqual(node_source, NodeSource._from_dict(node_source.to_dict()))

        # Dummy node
        node = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="add",
            op="call_function",
            target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
            args=(torch.tensor(3), torch.tensor(4)),
            kwargs={},
        )
        node.meta["from_node"] = [node_source]

        graph_id = id(node.graph)
        node_source = NodeSource(
            node=node, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        self.assertExpectedInline(
            node_source.print_readable().strip(),
            f"""\
(name=add, pass_name=test_pass, action=create, graph_id={graph_id})
    (name=, pass_name=test_pass, action=create, graph_id=-1)""",
        )
        self.assertEqual(
            node_source.to_dict(),
            {
                "name": "add",
                "target": "aten.add.Tensor",
                "pass_name": "test_pass",
                "action": CREATE_STR,
                "graph_id": graph_id,
                "from_node": [dummy_source_dict],
            },
        )

        # Test two node sources are same
        node_source1 = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        node_source2 = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        self.assertEqual(node_source1, node_source2)

        # Test hash function - equivalent objects should have same hash
        self.assertEqual(hash(node_source1), hash(node_source2))

        # Test two node sources are not same
        node_source3 = NodeSource(
            node=None, pass_name="test_pass_1", action=NodeSourceAction.CREATE
        )
        node_source4 = NodeSource(
            node=None, pass_name="test_pass_2", action=NodeSourceAction.CREATE
        )
        self.assertNotEqual(node_source3, node_source4)

        # Test hash function - different objects should have different hash
        self.assertNotEqual(hash(node_source3), hash(node_source4))

        # Test that equivalent NodeSource objects can be used in sets and dicts
        node_set = {node_source1, node_source2}
        self.assertEqual(len(node_set), 1)  # Should only contain one unique element

        node_dict = {node_source1: "value1", node_source2: "value2"}
        self.assertEqual(len(node_dict), 1)  # Should only contain one key
        self.assertEqual(node_dict[node_source1], "value2")  # Last value should win

        # Test with more complex NodeSource objects
        node_source_with_node = NodeSource(
            node=node, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        node_source_with_node_copy = NodeSource(
            node=node, pass_name="test_pass", action=NodeSourceAction.CREATE
        )

        # These should be equal and have same hash
        self.assertEqual(node_source_with_node, node_source_with_node_copy)
        self.assertEqual(hash(node_source_with_node), hash(node_source_with_node_copy))

        # Test with different actions
        node_source_replace = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.REPLACE
        )
        node_source_create = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.CREATE
        )

        # These should be different and have different hashes
        self.assertNotEqual(node_source_replace, node_source_create)
        self.assertNotEqual(hash(node_source_replace), hash(node_source_create))

    def test_graph_provenance(self):
        def check_node_source(node_source_dict, name, pass_name, action):
            self.assertEqual(node_source_dict["name"], name)
            self.assertEqual(node_source_dict["pass_name"], pass_name)
            self.assertEqual(node_source_dict["action"], action)

        def get_first_node_source_and_check(node_source_dict):
            """
            Get the first node source from the from_node list.
            """
            self.assertEqual(len(node_source_dict["from_node"]), 1)
            return node_source_dict["from_node"][0]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(16, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return (x,)

        model = Model()
        example_inputs = (torch.randn(8, 10),)
        ep = torch.export.export(model, example_inputs, strict=True)

        decomposed_ep = ep.run_decompositions(default_decompositions())
        # node decomposed from same ancestor node should have same from_node info
        for node in decomposed_ep.graph.nodes:
            if node.op not in {"placeholder", "output"}:
                assert "from_node" in node.meta

        node_name_to_from_node = {
            node.name: node.meta["from_node"]
            for node in decomposed_ep.graph.nodes
            if node.op not in {"placeholder", "output"}
        }
        same_ancestor_nodes = {
            "permute": "addmm",
            "addmm": "permute",
            "permute_1": "addmm_1",
            "addmm_1": "permute_1",
        }

        for node_name_1 in node_name_to_from_node:
            for node_name_2 in node_name_to_from_node:
                if node_name_2 in {
                    node_name_1,
                    same_ancestor_nodes.get(node_name_1),
                }:
                    self.assertEqual(
                        node_name_to_from_node[node_name_1],
                        node_name_to_from_node[node_name_2],
                    )
                    self.assertEqual(
                        [
                            NodeSource._from_dict(ns.to_dict())
                            for ns in node_name_to_from_node[node_name_1]
                        ],
                        node_name_to_from_node[node_name_2],
                    )
                else:
                    self.assertNotEqual(
                        node_name_to_from_node[node_name_1],
                        node_name_to_from_node[node_name_2],
                    )
                    self.assertNotEqual(
                        [
                            NodeSource._from_dict(ns.to_dict())
                            for ns in node_name_to_from_node[node_name_1]
                        ],
                        node_name_to_from_node[node_name_2],
                    )

        gm = ep.module()
        provenance = get_graph_provenance_json(gm.graph)
        self.assertEqual(
            set(provenance.keys()), {"relu", "linear", "sigmoid", "linear_1"}
        )

        # Check node "linear" is created from node "x" in PropagateUnbackedSymInts
        key_provenance = provenance["linear"][0]["from_node"]
        self.assertEqual(len(key_provenance), 1)
        key_provenance = key_provenance[0]
        check_node_source(
            key_provenance,
            "x",
            "Interpreter_PropagateUnbackedSymInts",
            CREATE_STR,
        )

        # Check node "x" is then created from another node "x" in FlattenInputOutputSignature
        key_provenance = get_first_node_source_and_check(key_provenance)
        check_node_source(
            key_provenance,
            "x",
            "Interpreter_DynamoGraphTransformer",
            CREATE_STR,
        )

        gm, graph_signature = aot_export_module(
            gm,
            example_inputs,
            trace_joint=False,
        )

        provenance = get_graph_provenance_json(gm.graph)

        self.assertEqual(
            set(provenance.keys()), {"t", "addmm", "relu", "t_1", "addmm_1", "sigmoid"}
        )
        for key in ["t", "addmm"]:
            # The node provenance hierarchy should be:
            # t -> linear -> x -> x
            #
            # x -> y means x is created from y

            key_provenance = provenance[key]
            self.assertEqual(len(key_provenance), 1)
            key_provenance = key_provenance[0]

            # Check node "t" and "addmm" is created from node "linear" in PropagateUnbackedSymInts
            check_node_source(
                key_provenance,
                "linear",
                "Interpreter_PropagateUnbackedSymInts",
                CREATE_STR,
            )

            # Check node "linear" is then created from node "x" in PropagateUnbackedSymInts
            key_provenance = get_first_node_source_and_check(key_provenance)[
                "from_node"
            ][0]
            check_node_source(
                key_provenance,
                "x",
                "Interpreter_PropagateUnbackedSymInts",
                CREATE_STR,
            )

            # Check node "x" is then created from another node "x" in FlattenInputOutputSignature
            key_provenance = get_first_node_source_and_check(key_provenance)
            check_node_source(
                key_provenance,
                "x",
                "Interpreter_DynamoGraphTransformer",
                CREATE_STR,
            )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview

"""(name=, pass_name=test_pass, action=create, graph_id=-1)""",        )        dummy_source_dict = {            "name": "",            "target": "",            "pass_name": "test_pass",            "action": CREATE_STR,            "graph_id": -1,            "from_node": [],        }        self.assertEqual(            node_source.to_dict(),            dummy_source_dict,        )        self.assertEqual(node_source, NodeSource._from_dict(node_source.to_dict()))        # Dummy node        node = torch.fx.Node(            graph=torch.fx.Graph(),            name="add",            op="call_function",            target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]            args=(torch.tensor(3), torch.tensor(4)),            kwargs={},        )        node.meta["from_node"] = [node_source]        graph_id = id(node.graph)        node_source = NodeSource(            node=node, pass_name="test_pass", action=NodeSourceAction.CREATE

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFXNodeSource`, `Model`

**Functions defined**: `test_node_source`, `test_graph_provenance`, `check_node_source`, `get_first_node_source_and_check`, `__init__`, `forward`

**Key imports**: torch, aot_export_module, default_decompositions, get_graph_provenance_json, NodeSource, NodeSourceAction, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.compile_fx`: aot_export_module
- `torch.export`: default_decompositions
- `torch.fx.traceback`: get_graph_provenance_json, NodeSource, NodeSourceAction
- `torch.testing._internal.common_utils`: TestCase


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/fx/test_fx_traceback.py
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
- [`test_gradual_type.py_docs.md`](./test_gradual_type.py_docs.md)
- [`test_pass_infra.py_docs.md`](./test_pass_infra.py_docs.md)
- [`test_lazy_graph_module.py_docs.md`](./test_lazy_graph_module.py_docs.md)


## Cross-References

- **File Documentation**: `test_fx_traceback.py_docs.md`
- **Keyword Index**: `test_fx_traceback.py_kw.md`
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

- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/fx/test_fx_traceback.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/fx`):

- [`named_tup.py_kw.md_docs.md`](./named_tup.py_kw.md_docs.md)
- [`test_dynamism.py_kw.md_docs.md`](./test_dynamism.py_kw.md_docs.md)
- [`test_fx_xform_observer.py_docs.md_docs.md`](./test_fx_xform_observer.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_fx_xform_observer.py_kw.md_docs.md`](./test_fx_xform_observer.py_kw.md_docs.md)
- [`test_fx_node_hook.py_kw.md_docs.md`](./test_fx_node_hook.py_kw.md_docs.md)
- [`test_partitioner_order.py_docs.md_docs.md`](./test_partitioner_order.py_docs.md_docs.md)
- [`test_subgraph_rewriter.py_kw.md_docs.md`](./test_subgraph_rewriter.py_kw.md_docs.md)
- [`test_fx_split.py_docs.md_docs.md`](./test_fx_split.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_fx_traceback.py_docs.md_docs.md`
- **Keyword Index**: `test_fx_traceback.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
