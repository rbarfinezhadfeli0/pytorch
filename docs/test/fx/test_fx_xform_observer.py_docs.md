# Documentation: `test/fx/test_fx_xform_observer.py`

## File Metadata

- **Path**: `test/fx/test_fx_xform_observer.py`
- **Size**: 7,533 bytes (7.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

import copy
import os
import tempfile

import torch
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.traceback import NodeSourceAction
from torch.testing._internal.common_utils import TestCase


class TestGraphTransformObserver(TestCase):
    def test_graph_transform_observer(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.neg(x)

        def replacement(x):
            return torch.relu(x)

        traced = symbolic_trace(M())

        log_url = tempfile.mkdtemp()

        with GraphTransformObserver(
            traced, "replace_neg_with_relu", log_url=log_url
        ) as ob:
            subgraph_rewriter.replace_pattern(traced, pattern, replacement)

            self.assertTrue("relu" in ob.created_nodes)
            self.assertTrue("neg" in ob.erased_nodes)

        current_pass_count = GraphTransformObserver.get_current_pass_count()

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_input_graph.dot",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_output_graph.dot",
                )
            )
        )

    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_graph_transform_observer_node_tracking(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.neg(x)

        def replacement(x):
            return torch.relu(x)

        def replacement2(x):
            return torch.cos(x)

        traced = symbolic_trace(M())

        def check_node_source(node_source, node_name, target, id, pass_name, action):
            self.assertEqual(node_source.name, node_name)
            self.assertEqual(node_source.target, target)
            self.assertEqual(node_source.pass_name, pass_name)
            self.assertEqual(node_source.graph_id, id)
            self.assertEqual(node_source.action, action)

        with GraphTransformObserver(traced, "replace_neg_with_relu") as ob:
            subgraph_rewriter.replace_pattern(traced, pattern, replacement)

            self.assertTrue("relu" in ob.created_nodes)
            self.assertTrue("neg" in ob.erased_nodes)

        self.assertEqual(len(traced._replace_hooks), 0)
        self.assertEqual(len(traced._create_node_hooks), 0)
        self.assertEqual(len(traced._erase_node_hooks), 0)
        self.assertEqual(len(traced._deepcopy_hooks), 0)

        for node in traced.graph.nodes:
            if node.name == "relu":
                from_node = node.meta["from_node"]
                self.assertTrue(len(from_node) == 1)
                check_node_source(
                    from_node[0],
                    "neg",
                    str(torch.neg),
                    id(traced.graph),
                    "replace_neg_with_relu",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )

        with GraphTransformObserver(traced, "replace_relu_with_cos") as ob:
            subgraph_rewriter.replace_pattern(traced, replacement, replacement2)

            self.assertTrue("cos" in ob.created_nodes)
            self.assertTrue("relu" in ob.erased_nodes)

        for node in traced.graph.nodes:
            if node.name == "cos":
                from_node = node.meta["from_node"]
                self.assertTrue(len(from_node) == 1)
                check_node_source(
                    from_node[0],
                    "relu",
                    str(torch.relu),
                    id(traced.graph),
                    "replace_relu_with_cos",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )
                check_node_source(
                    from_node[0].from_node[0],
                    "neg",
                    str(torch.neg),
                    id(traced.graph),
                    "replace_neg_with_relu",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )

        class SimpleLinearModel(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        model = SimpleLinearModel()
        gm = torch.export.export(model, (torch.rand(10),), strict=True).module()

        with GraphTransformObserver(gm, "test"):
            add_node = gm.graph.call_function(torch.ops.aten.add.default, (1, 1))
            neg_node = next(
                iter([node for node in gm.graph.nodes if node.name == "neg"])
            )
            neg_node.replace_all_uses_with(replace_with=add_node)

        from_node = add_node.meta["from_node"]
        self.assertTrue(len(from_node) == 1)
        check_node_source(
            from_node[0],
            "neg",
            str(torch.ops.aten.neg.default),
            id(gm.graph),
            "test",
            [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
        )

    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_graph_transform_observer_deepcopy(self):
        class SimpleLinearModel(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        model = SimpleLinearModel()
        gm = torch.export.export(model, (torch.rand(10),), strict=True).module()

        with GraphTransformObserver(gm, "test"):
            gm2 = copy.deepcopy(gm)

        nodes = [node.name for node in gm.graph.nodes]
        nodes2 = [node.name for node in gm2.graph.nodes]
        self.assertEqual(nodes, nodes2)

        # deepcopied graph modules should not have hooks after exiting
        # the context
        self.assertEqual(len(gm2._replace_hooks), 0)
        self.assertEqual(len(gm2._create_node_hooks), 0)
        self.assertEqual(len(gm2._erase_node_hooks), 0)
        self.assertEqual(len(gm2._deepcopy_hooks), 0)

    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_graph_transform_observer_replace(self):
        # the node should should not be duplicated
        class Model(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                z = y * 2
                w = y * 3
                return z, w

        model = Model()
        gm = symbolic_trace(model)

        with GraphTransformObserver(gm, "test"):
            for node in gm.graph.nodes:
                if node.name == "add":
                    new_node = gm.graph.call_function(
                        torch.ops.aten.add.Tensor, (node.args[0], node.args[1])
                    )
                    node.replace_all_uses_with(new_node)
                    new_node.name = "new_add"

        self.assertEqual(len(new_node.meta["from_node"]), 1)
        self.assertEqual(new_node.meta["from_node"][0].name, "add")
        self.assertEqual(new_node.meta["from_node"][0].pass_name, "test")


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 6 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestGraphTransformObserver`, `M`, `M`, `SimpleLinearModel`, `SimpleLinearModel`, `Model`

**Functions defined**: `test_graph_transform_observer`, `forward`, `pattern`, `replacement`, `test_graph_transform_observer_node_tracking`, `forward`, `pattern`, `replacement`, `replacement2`, `check_node_source`, `forward`, `test_graph_transform_observer_deepcopy`, `forward`, `test_graph_transform_observer_replace`, `forward`

**Key imports**: copy, os, tempfile, torch, subgraph_rewriter, symbolic_trace, GraphTransformObserver, NodeSourceAction, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `os`
- `tempfile`
- `torch`
- `torch.fx`: subgraph_rewriter, symbolic_trace
- `torch.fx.passes.graph_transform_observer`: GraphTransformObserver
- `torch.fx.traceback`: NodeSourceAction
- `torch.testing._internal.common_utils`: TestCase


## Code Patterns & Idioms

### Common Patterns

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
python test/fx/test_fx_xform_observer.py
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

- **File Documentation**: `test_fx_xform_observer.py_docs.md`
- **Keyword Index**: `test_fx_xform_observer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
