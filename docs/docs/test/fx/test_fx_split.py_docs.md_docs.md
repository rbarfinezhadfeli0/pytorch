# Documentation: `docs/test/fx/test_fx_split.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_fx_split.py_docs.md`
- **Size**: 14,110 bytes (13.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_fx_split.py`

## File Metadata

- **Path**: `test/fx/test_fx_split.py`
- **Size**: 10,675 bytes (10.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

import dataclasses
from collections import defaultdict

import torch
import torch.fx.passes.operator_support as op_support
import torch.fx.passes.splitter_base as splitter_base
from torch.fx.passes.split_utils import split_by_tags
from torch.testing._internal.common_utils import TestCase


@torch.jit.script
@dataclasses.dataclass
class DummyDataClass:
    a: int
    b: int
    c: int


@torch.fx.wrap
def wrapped_add(_dataclass, y):
    return _dataclass.c + y


class TestFXSplit(TestCase):
    def test_split_preserve_node_meta(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                x = x + x
                y = y * y
                return x - y

        gm = torch.fx.symbolic_trace(TestModule())
        for node in gm.graph.nodes:
            node.meta["name"] = node.name
            if node.name == "add":
                node.tag = "a"
            elif node.name == "mul":
                node.tag = "b"
            elif node.name == "sub":
                node.tag = "c"

        split_gm = split_by_tags(gm, ["a", "b", "c"])
        for m in split_gm.children():
            for n in m.graph.nodes:
                if n.op != "output":
                    self.assertIn("name", n.meta)
                    self.assertEqual(n.meta["name"], n.name)

        # Validate that metadata is copied correctly for graph placeholder nodes
        for node in split_gm.graph.nodes:
            if node.op == "placeholder":
                self.assertIn("name", node.meta)
                self.assertEqual(node.meta["name"], node.name)

    def test_dataclass_as_graph_entry(self):
        """
        Test that splitting works when the graph entry is a dataclass instance
        and a wrapped function is called with it, resulting in a call_function
        node with no input dependencies. This tests the edge case fixed in D81232435
        where call_function nodes with no dependencies should be handled properly
        in the starter_nodes() method.

        Graph visualization:
        y (input)    DummyDataClass(2,3,4) (no input deps, result as a call_function_node)
            \              /
             \            /
              wrapped_add
                  |
              z (output)
        """  # noqa: W605

        class TestModuleWithFunctionEntry(torch.nn.Module):
            def forward(self, y):
                # This creates a call_function node with no input dependencies
                dummy_data_class = DummyDataClass(2, 3, 4)
                z = wrapped_add(dummy_data_class, y)
                return z

        mod = TestModuleWithFunctionEntry()
        gm = torch.fx.symbolic_trace(mod)

        # Create custom operator support to mark wrapped_add as supported
        class CustomOpSupport(op_support.OperatorSupportBase):
            def is_node_supported(self, submodules, node) -> bool:
                return node.target == wrapped_add

        # Create a simple splitter to test the edge case
        class TestSplitter(splitter_base._SplitterBase):
            def __init__(self, module, sample_input, operator_support):
                settings = splitter_base._SplitterSettingBase()
                super().__init__(module, sample_input, operator_support, settings)

        # Create splitter instance - this tests the fix where call_function nodes
        # with no input dependencies are properly handled in starter_nodes()
        splitter = TestSplitter(
            module=gm,
            sample_input=[torch.randn(2, 3)],
            operator_support=CustomOpSupport(),
        )

        # This should not raise an exception (tests the fix from D81232435)
        # The fix allows call_function nodes with no dependencies as valid starter nodes
        split_result = splitter()

        # Verify the splitting worked correctly
        self.assertIsNotNone(split_result)

        # Test that the split module produces the same result as the original
        test_input = torch.randn(2, 3)
        original_result = mod(test_input)
        split_module_result = split_result(test_input)
        self.assertTrue(torch.equal(original_result, split_module_result))


class TestSplitByTags(TestCase):
    class TestModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(2, 3)
            self.linear2 = torch.nn.Linear(4, 5)
            self.linear3 = torch.nn.Linear(6, 7)
            self.linear4 = torch.nn.Linear(8, 6)

        def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
        ) -> torch.Tensor:
            v1 = self.linear1(x1)
            v2 = self.linear2(x2)
            v3 = self.linear3(x3)
            v4 = torch.cat([v1, v2, v3])
            return self.linear4(v4)

    @staticmethod
    def trace_and_tag(
        module: torch.nn.Module, tags: list[str]
    ) -> tuple[torch.fx.GraphModule, dict[str, list[str]]]:
        """
        Test simple gm consists of nodes with tag (only show call_module nodes here):
            linear1 - tag: "red"
            linear2 - tag: "blue"
            linear3, linear4 - tag: "green"

        At the beginning we have:
            gm:
                linear1
                linear2
                linear3
                linear4

        split_gm = split_by_tags(gm, tags)

        Then we have:
            split_gm:
                red:
                    linear1
                blue:
                    linear2
                green:
                    linear3
                    linear4
        """
        tag_node = defaultdict(list)
        gm: torch.fx.GraphModule = torch.fx.symbolic_trace(module)

        # Add tag to all nodes and build dictionary record tag to call_module nodes
        for node in gm.graph.nodes:
            if "linear1" in node.name:
                node.tag = tags[0]
                tag_node[tags[0]].append(node.name)
            elif "linear2" in node.name:
                node.tag = tags[1]
                tag_node[tags[1]].append(node.name)
            else:
                node.tag = tags[2]
                if node.op == "call_module":
                    tag_node[tags[2]].append(node.name)
        return gm, tag_node

    def test_split_by_tags(self) -> None:
        tags = ["red", "blue", "green"]
        module = TestSplitByTags.TestModule()
        gm, tag_node = TestSplitByTags.trace_and_tag(module, tags)
        split_gm, orig_to_split_fqn_mapping = split_by_tags(
            gm, tags, return_fqn_mapping=True
        )
        # Ensure split_gm has (and only has) ordered submodules named
        # red_0, blue_1, green_2
        for idx, (name, _) in enumerate(split_gm.named_children()):
            if idx < len(tags):
                self.assertTrue(
                    name == tags[idx],
                    f"split_gm has an incorrect submodule named {name}",
                )

        # Ensure each submodule has expected (ordered) call_module node(s).
        # For example, a submodule named split_gm.red_0 has (and only has) linear1;
        # split_gm.green_2 has (and only has) linear3 and linear4 with order
        sub_graph_idx = 0
        for sub_name, sub_graph_module in split_gm.named_children():
            node_idx = 0
            for node in sub_graph_module.graph.nodes:
                if node.op != "call_module":
                    continue
                self.assertTrue(
                    node.name == tag_node[f"{sub_name}"][node_idx],
                    # pyre-fixme[61]: `name` is undefined, or not always defined.
                    f"{sub_name} has incorrectly include {node.name}",
                )
                node_idx += 1
            sub_graph_idx += 1

        self.assertEqual(
            orig_to_split_fqn_mapping,
            {
                "linear1": "red.linear1",
                "linear2": "blue.linear2",
                "linear3": "green.linear3",
                "linear4": "green.linear4",
            },
            f"{orig_to_split_fqn_mapping=}",
        )


class TestSplitOutputType(TestCase):
    class TestModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            conv = self.conv(x)
            conv = conv * 0.5
            relu = self.relu(conv)
            return relu

    @staticmethod
    def trace_and_tag(
        module: torch.nn.Module, inputs: torch.Tensor, tags: list[str]
    ) -> tuple[torch.fx.GraphModule, dict[str, list[str]]]:
        """
        Test simple gm consists of nodes with tag (only show call_module nodes here):
            conv - tag: "red"
            mul - tag: "blue"
            relu - tag: "green"

        At the beginning we have:
            gm:
                conv
                mul
                relu

        split_gm = split_by_tags(gm, tags)

        Then we have:
            split_gm:
                red:
                    conv
                blue:
                    mul
                green:
                    relu
        """
        tag_node = defaultdict(list)
        gm: torch.fx.GraphModule = torch.export.export(
            module, (inputs,), strict=True
        ).module()
        # Add tag to all nodes and build dictionary record tag to call_module nodes
        for node in gm.graph.nodes:
            if "conv" in node.name:
                node.tag = tags[0]
                tag_node[tags[0]].append(node.name)
            elif "mul" in node.name:
                node.tag = tags[1]
                tag_node[tags[1]].append(node.name)
            else:
                node.tag = tags[2]
                if node.op == "call_module":
                    tag_node[tags[2]].append(node.name)
        return gm, tag_node

    def test_split_by_tags(self) -> None:
        tags = ["red", "blue", "green"]
        module = TestSplitOutputType.TestModule()

        inputs = torch.randn((1, 3, 224, 224))

        gm, _ = TestSplitOutputType.trace_and_tag(module, inputs, tags)
        split_gm, _ = split_by_tags(gm, tags, return_fqn_mapping=True)

        gm_output = module(inputs)
        split_gm_output = split_gm(inputs)

        self.assertTrue(type(gm_output) is type(split_gm_output))
        self.assertTrue(torch.equal(gm_output, split_gm_output))


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 11 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DummyDataClass`, `TestFXSplit`, `TestModule`, `TestModuleWithFunctionEntry`, `CustomOpSupport`, `TestSplitter`, `TestSplitByTags`, `TestModule`, `TestSplitOutputType`, `TestModule`

**Functions defined**: `wrapped_add`, `test_split_preserve_node_meta`, `forward`, `test_dataclass_as_graph_entry`, `forward`, `is_node_supported`, `__init__`, `__init__`, `forward`, `trace_and_tag`, `test_split_by_tags`, `__init__`, `forward`, `trace_and_tag`, `test_split_by_tags`

**Key imports**: dataclasses, defaultdict, torch, torch.fx.passes.operator_support as op_support, torch.fx.passes.splitter_base as splitter_base, split_by_tags, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `collections`: defaultdict
- `torch`
- `torch.fx.passes.operator_support as op_support`
- `torch.fx.passes.splitter_base as splitter_base`
- `torch.fx.passes.split_utils`: split_by_tags
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
python test/fx/test_fx_split.py
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

- **File Documentation**: `test_fx_split.py_docs.md`
- **Keyword Index**: `test_fx_split.py_kw.md`
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
python docs/test/fx/test_fx_split.py_docs.md
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


## Cross-References

- **File Documentation**: `test_fx_split.py_docs.md_docs.md`
- **Keyword Index**: `test_fx_split.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
