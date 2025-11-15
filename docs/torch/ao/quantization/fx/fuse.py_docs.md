# Documentation: `torch/ao/quantization/fx/fuse.py`

## File Metadata

- **Path**: `torch/ao/quantization/fx/fuse.py`
- **Size**: 7,396 bytes (7.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import warnings
from collections.abc import Callable
from typing import Any

from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
)
from torch.ao.quantization.backend_config.utils import (
    get_fuser_method_mapping,
    get_fusion_pattern_to_extra_inputs_getter,
    get_fusion_pattern_to_root_node_getter,
)
from torch.ao.quantization.utils import NodePattern, Pattern
from torch.fx import GraphModule, map_arg, Node
from torch.fx.graph import Graph

from .custom_config import FuseCustomConfig
from .fuse_handler import _get_fusion_pattern_to_fuse_handler_cls, FuseHandler
from .match_utils import _is_match, MatchAllNode
from .pattern_utils import _sorted_patterns_dict


__all__ = [
    "fuse",
    # TODO: We should make this private in the future
    # This is currently needed for test_public_bindings for some reason
    "FuseHandler",
]


def fuse(
    model: GraphModule,
    is_qat: bool,
    fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()

    if isinstance(fuse_custom_config, dict):
        warnings.warn(
            "Passing a fuse_custom_config_dict to fuse is deprecated and will not be supported "
            "in a future version. Please pass in a FuseCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)

    if isinstance(backend_config, dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        backend_config = BackendConfig.from_dict(backend_config)

    named_modules = dict(model.named_modules())

    if backend_config is None:
        backend_config = get_native_backend_config()

    fusion_pattern_to_fuse_handler_cls = _sorted_patterns_dict(
        _get_fusion_pattern_to_fuse_handler_cls(backend_config)
    )
    fuser_method_mapping = get_fuser_method_mapping(backend_config)
    fusion_pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(
        backend_config
    )
    fusion_pattern_to_extra_inputs_getter = get_fusion_pattern_to_extra_inputs_getter(
        backend_config
    )

    # find fusion
    fusion_pairs = _find_matches(model, model.graph, fusion_pattern_to_fuse_handler_cls)
    # TODO: change this to inplace changes to graph, since we no longer construct
    # new GraphModule anymore
    fused_graph = Graph()
    env: dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    def default_root_node_getter(node_pattern):
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]

    for node in model.graph.nodes:
        (
            maybe_last_node,
            pattern,
            matched_node_pattern,
            obj,
            node_to_subpattern,
        ) = fusion_pairs.get(node.name, (None, None, None, None, None))
        # get the corresponding subpattern for the current node
        if node_to_subpattern is not None:
            node_subpattern = node_to_subpattern.get(node, None)
        else:
            node_subpattern = None
        if maybe_last_node is node:
            if obj is None:
                raise AssertionError(
                    "fuse handler object must not be None for matched root node"
                )
            root_node_getter = fusion_pattern_to_root_node_getter.get(
                pattern, default_root_node_getter
            )
            root_node = root_node_getter(matched_node_pattern)  # type: ignore[index]
            extra_inputs_getter = fusion_pattern_to_extra_inputs_getter.get(
                pattern, None
            )
            extra_inputs = []
            if extra_inputs_getter is not None:
                extra_inputs = extra_inputs_getter(matched_node_pattern)
            # TODO: add validation that root_node is a module and has the same type
            # as the root_module in the configuration
            env[node.name] = obj.fuse(
                load_arg,
                named_modules,
                fused_graph,
                root_node,
                extra_inputs,
                matched_node_pattern,  # type: ignore[arg-type]
                fuse_custom_config,
                fuser_method_mapping,
                is_qat,
            )
        elif maybe_last_node is None or node_subpattern is MatchAllNode:
            env[node.name] = fused_graph.node_copy(node, load_arg)
        # node matched in patterns and is not root is removed here

    model = GraphModule(model, fused_graph)
    return model


def _find_matches(
    root: GraphModule,
    graph: Graph,
    pattern_to_fuse_handler_cls: dict[Pattern, Callable],
) -> dict[str, tuple[Node, Pattern, NodePattern, FuseHandler, dict[Node, Any]]]:
    modules = dict(root.named_modules())
    # node name -> (root_node, match_value)
    match_map: dict[
        str, tuple[Node, Pattern, NodePattern, FuseHandler, dict[Node, Any]]
    ] = {}
    # a map from node to the matched subpattern
    node_to_subpattern: dict[Node, Any] = {}

    # TODO: dedup with quantization matching function in match_utils.py
    def apply_match(pattern, node, match, matched_node_pattern, node_to_subpattern):
        if isinstance(pattern, tuple):
            s, *args = pattern
            current_node_pattern: list[Node] = []
            apply_match(s, node, match, current_node_pattern, node_to_subpattern)
            for subpattern, arg in zip(args, node.args):
                apply_match(
                    subpattern, arg, match, current_node_pattern, node_to_subpattern
                )
            matched_node_pattern.append(tuple(current_node_pattern))
        else:
            # the first pattern matches will take precedence
            if node.name not in match_map:
                matched_node_pattern.append(node)
                # MatchAllNode here is actually MatchAllInputNode which should not
                # be added to match_map
                if pattern is not MatchAllNode:
                    node_to_subpattern[node] = pattern
                    root_node, pattern, handler = match
                    match_map[node.name] = (
                        root_node,
                        pattern,
                        matched_node_pattern,
                        handler,
                        node_to_subpattern,
                    )

    for node in reversed(graph.nodes):
        if node.name not in match_map:
            for pattern, fuse_handler_cls in pattern_to_fuse_handler_cls.items():
                matched_node_pattern: list[Node] = []
                if _is_match(modules, node, pattern):
                    apply_match(
                        pattern,
                        node,
                        (node, pattern, fuse_handler_cls(node)),
                        matched_node_pattern,
                        node_to_subpattern,
                    )
                    break

    return match_map

```



## High-Level Overview


This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `fuse`, `load_arg`, `default_root_node_getter`, `_find_matches`, `apply_match`

**Key imports**: warnings, Callable, Any, NodePattern, Pattern, GraphModule, map_arg, Node, Graph, FuseCustomConfig, _get_fusion_pattern_to_fuse_handler_cls, FuseHandler, _is_match, MatchAllNode, _sorted_patterns_dict


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `collections.abc`: Callable
- `typing`: Any
- `torch.ao.quantization.utils`: NodePattern, Pattern
- `torch.fx`: GraphModule, map_arg, Node
- `torch.fx.graph`: Graph
- `.custom_config`: FuseCustomConfig
- `.fuse_handler`: _get_fusion_pattern_to_fuse_handler_cls, FuseHandler
- `.match_utils`: _is_match, MatchAllNode
- `.pattern_utils`: _sorted_patterns_dict


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


## Cross-References

- **File Documentation**: `fuse.py_docs.md`
- **Keyword Index**: `fuse.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
