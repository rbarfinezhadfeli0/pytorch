# Documentation: `docs/test/fx/test_matcher_utils.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_matcher_utils.py_docs.md`
- **Size**: 14,437 bytes (14.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_matcher_utils.py`

## File Metadata

- **Path**: `test/fx/test_matcher_utils.py`
- **Size**: 10,823 bytes (10.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

import os
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.export import export
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import unittest

from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.jit_utils import JitTestCase


class WrapperModule(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class TestMatcher(JitTestCase):
    def test_subgraph_matcher_with_attributes(self):
        class LargeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._weight = torch.nn.Parameter(torch.ones(3, 3))
                self._bias = torch.nn.Parameter(torch.ones(3, 3))

            def forward(self, x):
                return torch.ops.aten.addmm.default(self._bias, x, self._weight)

        # Large Model graph:
        # opcode         name           target              args                 kwargs
        # -------------  -------------  ------------------  -------------------  --------
        # placeholder    x              x                   ()                   {}
        # get_attr       _bias          _bias               ()                   {}
        # get_attr       _weight        _weight             ()                   {}
        # call_function  addmm_default  aten.addmm.default  (_bias, x, _weight)  {}
        # output         output         output              (addmm_default,)     {}
        large_model_graph = symbolic_trace(LargeModel()).graph

        class PatternModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
                self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x):
                return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

        pattern_graph = torch.fx.symbolic_trace(PatternModel()).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(large_model_graph)
        self.assertEqual(len(match_result), 1)

    def test_subgraph_matcher_with_list(self):
        def original(x, y):
            return torch.ops.aten.view(x, [5, y.shape[0]])

        original_graph = torch.fx.symbolic_trace(original).graph

        def pattern(x, y, z):
            return torch.ops.aten.view(x, [z, y.shape[0]])

        pattern_graph = torch.fx.symbolic_trace(pattern).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 1)

    def test_subgraph_matcher_with_list_bad(self):
        def original(x, y):
            return torch.ops.aten._reshape_alias_copy.default(
                x, [1, y.shape[0]], [y.shape[1], y.shape[1]]
            )

        original_graph = torch.fx.symbolic_trace(original).graph

        def pattern(x, y, b):
            return torch.ops.aten._reshape_alias_copy.default(
                x, [b, y.shape[0], y.shape[1]], [y.shape[1]]
            )

        pattern_graph = torch.fx.symbolic_trace(pattern).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 0)

    def test_subgraph_matcher_ignore_literals(self):
        def original(x):
            return x + 1

        original_graph = make_fx(original)(torch.ones(3, 3)).graph
        original_graph.eliminate_dead_code()

        def pattern(x):
            return x + 2

        pattern_graph = make_fx(pattern)(torch.ones(4, 4)).graph
        pattern_graph.eliminate_dead_code()

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 0)

        subgraph_matcher = SubgraphMatcher(pattern_graph, ignore_literals=True)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 1)

    def test_variatic_arg_matching(self):
        inputs = (torch.randn(20, 16, 50, 32),)

        def maxpool(x, kernel_size, stride, padding, dilation):
            return torch.ops.aten.max_pool2d_with_indices.default(
                x, kernel_size, stride, padding, dilation
            )

        maxpool_graph = torch.fx.symbolic_trace(maxpool).graph

        maxpool_matcher = SubgraphMatcher(maxpool_graph)
        match_result = maxpool_matcher.match(maxpool_graph)
        self.assertEqual(len(match_result), 1)

        # Graph only contains "stride" argument
        maxpool_s = torch.nn.MaxPool2d(kernel_size=2, stride=1).eval()
        maxpool_s_graph = make_fx(maxpool_s)(*inputs).graph
        match_s_result = maxpool_matcher.match(maxpool_s_graph)
        self.assertEqual(len(match_s_result), 1)

        # Graph only contains "padding" argument
        maxpool_p = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        maxpool_p_graph = make_fx(maxpool_p)(*inputs).graph
        match_p_result = maxpool_matcher.match(maxpool_p_graph)
        self.assertEqual(len(match_p_result), 1)

        # Graph only contains "stride, padding" argument
        maxpool_sp = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        maxpool_sp_graph = make_fx(maxpool_sp)(*inputs).graph
        match_sp_result = maxpool_matcher.match(maxpool_sp_graph)
        self.assertEqual(len(match_sp_result), 1)

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_split_to_graph_and_name_node_map(self):
        """Testing the internal helper function for splitting the pattern graph"""
        from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
            _split_to_graph_and_name_node_map,
        )

        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu_mul_by_two = relu * 2
            return relu, relu_mul_by_two, {"conv": conv, "relu": relu}

        example_inputs = (
            torch.randn(1, 3, 3, 3) * 10,
            torch.randn(3, 3, 3, 3),
        )
        pattern_gm = export(
            WrapperModule(pattern), example_inputs, strict=True
        ).module()
        before_split_res = pattern_gm(*example_inputs)
        pattern_gm, _ = _split_to_graph_and_name_node_map(pattern_gm)
        after_split_res = pattern_gm(*example_inputs)
        self.assertEqual(before_split_res[0], after_split_res[0])
        self.assertEqual(before_split_res[1], after_split_res[1])

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_matcher_with_name_node_map_function(self):
        """Testing SubgraphMatcherWithNameNodeMap with function pattern"""

        def target_graph(x, weight):
            x = x * 2
            weight = weight * 3
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu2 = relu * 2
            return relu + relu2

        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu_mul_by_two = relu * 2
            return relu, relu_mul_by_two, {"conv": conv, "relu": relu}

        example_inputs = (
            torch.randn(1, 3, 3, 3) * 10,
            torch.randn(3, 3, 3, 3),
        )
        pattern_gm = export(
            WrapperModule(pattern), example_inputs, strict=True
        ).module()
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        target_gm = export(
            WrapperModule(target_graph), example_inputs, strict=True
        ).module()
        internal_matches = matcher.match(target_gm.graph)
        for internal_match in internal_matches:
            name_node_map = internal_match.name_node_map
            assert "conv" in name_node_map
            assert "relu" in name_node_map
            name_node_map["conv"].meta["custom_annotation"] = "annotation"
            # check if we correctly annotated the target graph module
            for n in target_gm.graph.nodes:
                if n == name_node_map["conv"]:
                    assert (
                        "custom_annotation" in n.meta
                        and n.meta["custom_annotation"] == "annotation"
                    )

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_matcher_with_name_node_map_module(self):
        """Testing SubgraphMatcherWithNameNodeMap with module pattern"""

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        class Pattern(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                linear = self.linear(x)
                # Note: we can't put "weight": self.linear.weight in dictionary since
                # nn.Parameter is not an allowed output type in dynamo
                return linear, {"linear": linear, "x": x}

        example_inputs = (torch.randn(3, 5),)
        pattern_gm = export(Pattern(), example_inputs, strict=True).module()
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        target_gm = export(M(), example_inputs, strict=True).module()
        internal_matches = matcher.match(target_gm.graph)
        for internal_match in internal_matches:
            name_node_map = internal_match.name_node_map
            assert "linear" in name_node_map
            assert "x" in name_node_map
            name_node_map["linear"].meta["custom_annotation"] = "annotation"
            # check if we correctly annotated the target graph module
            for n in target_gm.graph.nodes:
                if n == name_node_map["linear"]:
                    assert (
                        "custom_annotation" in n.meta
                        and n.meta["custom_annotation"] == "annotation"
                    )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 6 class(es) and 28 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WrapperModule`, `TestMatcher`, `LargeModel`, `PatternModel`, `M`, `Pattern`

**Functions defined**: `__init__`, `forward`, `test_subgraph_matcher_with_attributes`, `__init__`, `forward`, `__init__`, `forward`, `test_subgraph_matcher_with_list`, `original`, `pattern`, `test_subgraph_matcher_with_list_bad`, `original`, `pattern`, `test_subgraph_matcher_ignore_literals`, `original`, `pattern`, `test_variatic_arg_matching`, `maxpool`, `test_split_to_graph_and_name_node_map`, `pattern`

**Key imports**: os, sys, Callable, torch, torch.nn.functional as F, export, symbolic_trace, make_fx, unittest, SubgraphMatcher


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `collections.abc`: Callable
- `torch`
- `torch.nn.functional as F`
- `torch.export`: export
- `torch.fx`: symbolic_trace
- `torch.fx.experimental.proxy_tensor`: make_fx
- `unittest`
- `torch.fx.passes.utils.matcher_utils`: SubgraphMatcher
- `torch.testing._internal.common_utils`: IS_WINDOWS
- `torch.testing._internal.jit_utils`: JitTestCase


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
python test/fx/test_matcher_utils.py
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

- **File Documentation**: `test_matcher_utils.py_docs.md`
- **Keyword Index**: `test_matcher_utils.py_kw.md`
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/fx/test_matcher_utils.py_docs.md
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

- **File Documentation**: `test_matcher_utils.py_docs.md_docs.md`
- **Keyword Index**: `test_matcher_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
