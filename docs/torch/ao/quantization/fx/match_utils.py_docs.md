# Documentation: `torch/ao/quantization/fx/match_utils.py`

## File Metadata

- **Path**: `torch/ao/quantization/fx/match_utils.py`
- **Size**: 8,963 bytes (8.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import sys
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import MatchAllNode, Pattern
from torch.fx.graph import Graph, Node
from torch.nn.utils.parametrize import type_before_parametrizations

from .graph_module import _is_observed_standalone_module
from .quantize_handler import QuantizeHandler


__all__: list[str] = []

# TODO(future PR): the 1st argument is typed as `List[Node]`, but a better type
# would be a recursive `List[Union[Node, Tuple[Union[Node, ...]]]]`
_MatchResult = tuple[Node, list[Node], Pattern | None, QuantizeHandler]

_MatchResultWithQConfig = tuple[
    Node, list[Node], Pattern | None, QuantizeHandler, QConfigAny
]


# Note: The order of patterns is important! match function will take whatever is matched first, so we'll
# need to put the fusion patterns before single patterns. For example, add_relu should be registered come before relu.
# decorators are applied in the reverse order we see. Also when we match the nodes in the graph with these patterns,
# we'll start from the last node of the graph and traverse back.
def _is_match(modules, node, pattern, max_uses=sys.maxsize):
    """Matches a node in fx against a pattern"""
    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern
        if self_match is getattr:
            if len(pattern) != 2:
                raise AssertionError("Expecting getattr pattern to have two elements")
            arg_matches = []
    else:
        self_match = pattern
        arg_matches = []

    if isinstance(self_match, type) and issubclass(self_match, MatchAllNode):
        return True

    if node == pattern:
        return True

    if not isinstance(node, Node) or len(node.users) > max_uses:
        return False

    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != "call_module":
            return False
        if type_before_parametrizations(modules[node.target]) != self_match:
            return False
    elif callable(self_match):
        if node.op != "call_function" or node.target is not self_match:
            return False
        elif node.target is getattr:
            if node.args[1] != pattern[1]:
                return False
    elif isinstance(self_match, str):
        if node.op != "call_method" or node.target != self_match:
            return False
    elif node.target != self_match:
        return False

    if not arg_matches:
        return True

    if len(arg_matches) != len(node.args):
        return False

    return all(
        _is_match(modules, node, arg_match, max_uses=1)
        for node, arg_match in zip(node.args, arg_matches)
    )


def _find_matches(
    graph: Graph,
    modules: dict[str, torch.nn.Module],
    patterns: dict[Pattern, QuantizeHandler],
    root_node_getter_mapping: dict[Pattern, Callable],
    standalone_module_names: list[str] | None = None,
    standalone_module_classes: list[type] | None = None,
    custom_module_classes: list[Any] | None = None,
) -> dict[str, _MatchResult]:
    """
    Matches the nodes in the input graph to quantization patterns, and
    outputs the information needed to quantize them in future steps.

    Inputs:
      - graph: an fx.Graph object
      - modules: a mapping of fully qualified module name to instance,
          for example, {'foo': ModuleFoo, ...}
      - patterns: a mapping from a tuple of nodes in reverse order to
          uninitialized QuantizeHandler subclass.

    Outputs a map of
      node_name ->
        (node, matched_values, matched_pattern, QuantizeHandler instance,
         qconfig)

    For example, {
      'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,
                 <CopyNodeQuantizeHandler instance>, QConfig(...)),
      ...
    }
    """
    if custom_module_classes is None:
        custom_module_classes = []

    if standalone_module_classes is None:
        standalone_module_classes = []

    if standalone_module_names is None:
        standalone_module_names = []

    match_map: dict[str, _MatchResult] = {}
    all_matched: set[str] = set()

    def _recursive_record_node_in_match_map(
        last_node, match_map, node_pattern, matched_node_pattern, pattern, match_value
    ):
        if isinstance(node_pattern, Node):
            match_map[node_pattern.name] = (
                last_node,
                matched_node_pattern,
                pattern,
                match_value,
            )
        elif not isinstance(node_pattern, Iterable):
            return
        else:
            for n in node_pattern:
                _recursive_record_node_in_match_map(
                    last_node, match_map, n, matched_node_pattern, pattern, match_value
                )

    # TODO: 1. merge with fuse matcher 2. document the code
    def record_match(pattern, node, last_node, matched_node_pattern, match_map):
        if isinstance(pattern, tuple):
            s, *args = pattern
            is_single_arg = len(args) == 1
            current_node_pattern: list[Node] = []
            record_match(s, node, last_node, matched_node_pattern, match_map)
            if pattern[0] is not getattr:
                for subpattern, arg in zip(args, node.args):
                    record_match(subpattern, arg, node, current_node_pattern, match_map)
            if len(current_node_pattern) > 1:
                # current_node_pattern is  the node pattern we get from matching
                # the subpattern with arguments of the node
                # we use is_single_arg to recover the original structure of the pattern
                # if the original pattern has a single argument, we will have
                # (original_op, (original_arg, ...))
                # otherwise, we'll have a list of arguments
                # (original_op, arg0, arg1, arg2, ...)
                if is_single_arg:
                    matched_node_pattern.append(tuple(current_node_pattern))
                else:
                    matched_node_pattern.extend(list(current_node_pattern))
            else:
                matched_node_pattern.append(current_node_pattern[0])
        else:
            matched_node_pattern.append(node)

    for node in reversed(graph.nodes):
        if node.name not in match_map and node.name not in all_matched:
            for pattern, quantize_handler_cls in patterns.items():
                root_node_getter = root_node_getter_mapping.get(pattern)
                if _is_match(modules, node, pattern) and node.name not in match_map:
                    matched_node_pattern: list[Node] = []
                    record_match(pattern, node, node, matched_node_pattern, match_map)
                    quantize_handler = quantize_handler_cls(  # type: ignore[operator]
                        matched_node_pattern, modules, root_node_getter
                    )
                    last_node = node
                    # record the match for all nodes in the pattern
                    _recursive_record_node_in_match_map(
                        last_node,
                        match_map,
                        # we need to record all nodes in the matched pattern in the match_map
                        matched_node_pattern,
                        # this is a part of the value corresponding to the node
                        matched_node_pattern,
                        pattern,
                        quantize_handler,
                    )
                    break

    # add custom module instances to the match result
    if modules is None:
        raise AssertionError("modules must not be None")
    for node in graph.nodes:
        if (
            node.op == "call_module"
            and type(modules[node.target]) in custom_module_classes
        ):
            match_map[node.name] = (
                node,
                node,
                None,
                QuantizeHandler(node, modules, is_custom_module=True),
            )

    def is_standalone_module(node_target: str, modules: dict[str, torch.nn.Module]):
        if modules is None:
            raise AssertionError("modules must not be None")
        return (
            node_target in standalone_module_names
            or type(modules[node_target])  # type: ignore[operator]
            in standalone_module_classes  # type: ignore[operator]
        )

    # add standalone modules to the match
    for node in graph.nodes:
        if node.op == "call_module" and (
            is_standalone_module(node.target, modules)
            or _is_observed_standalone_module(modules[node.target])
        ):
            # add node to matched nodes
            match_map[node.name] = (
                node,
                node,
                None,
                QuantizeHandler(node, modules, is_standalone_module=True),
            )

    return match_map

```



## High-Level Overview

"""Matches a node in fx against a pattern"""    if isinstance(pattern, tuple):        self_match, *arg_matches = pattern        if self_match is getattr:            if len(pattern) != 2:                raise AssertionError("Expecting getattr pattern to have two elements")            arg_matches = []    else:        self_match = pattern        arg_matches = []    if isinstance(self_match, type) and issubclass(self_match, MatchAllNode):        return True    if node == pattern:        return True    if not isinstance(node, Node) or len(node.users) > max_uses:        return False

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_is_match`, `_find_matches`, `_recursive_record_node_in_match_map`, `record_match`, `is_standalone_module`

**Key imports**: sys, Callable, Iterable, Any, torch, QConfigAny, MatchAllNode, Pattern, Graph, Node, type_before_parametrizations, _is_observed_standalone_module, QuantizeHandler


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `collections.abc`: Callable, Iterable
- `typing`: Any
- `torch`
- `torch.ao.quantization.qconfig`: QConfigAny
- `torch.ao.quantization.utils`: MatchAllNode, Pattern
- `torch.fx.graph`: Graph, Node
- `torch.nn.utils.parametrize`: type_before_parametrizations
- `.graph_module`: _is_observed_standalone_module
- `.quantize_handler`: QuantizeHandler


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/ao/quantization/fx`):

- [`lstm_utils.py_docs.md`](./lstm_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_lower_to_native_backend.py_docs.md`](./_lower_to_native_backend.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`convert.py_docs.md`](./convert.py_docs.md)
- [`lower_to_fbgemm.py_docs.md`](./lower_to_fbgemm.py_docs.md)
- [`_equalize.py_docs.md`](./_equalize.py_docs.md)
- [`_decomposed.py_docs.md`](./_decomposed.py_docs.md)
- [`graph_module.py_docs.md`](./graph_module.py_docs.md)
- [`fuse.py_docs.md`](./fuse.py_docs.md)


## Cross-References

- **File Documentation**: `match_utils.py_docs.md`
- **Keyword Index**: `match_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
