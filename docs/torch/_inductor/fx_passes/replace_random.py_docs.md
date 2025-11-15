# Documentation: `torch/_inductor/fx_passes/replace_random.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/replace_random.py`
- **Size**: 4,378 bytes (4.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import collections
import logging

import torch
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from ..virtualized import V


log = logging.getLogger(__name__)
patterns = PatternMatcherPass(subsystem="joint_graph_passes")
aten = torch.ops.aten


def replace_random_passes(gm: torch.fx.GraphModule):
    """Modify the given FX graph to use backend-native random ops"""
    if config.fallback_random:
        return 0

    count = patterns.apply(gm)
    with GraphTransformObserver(gm, "fuse_seed_creation_pass", "joint_graph_passes"):
        count += fuse_seed_creation_pass(gm.graph)

    return count


def fuse_seed_creation_pass(graph: torch.fx.Graph):
    """
    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
    device_seeds = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.seed).match(node):
            device_seeds[node.args[0]].append(node)

    if not device_seeds:
        return 0

    for device, seeds in device_seeds.items():
        with graph.inserting_before(seeds[0]):
            combined = graph.call_function(inductor_prims.seeds, (len(seeds), device))
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(seeds)], device=device, dtype=torch.int64
                )
                combined.meta["tensor_meta"] = _extract_tensor_metadata(
                    combined.meta["val"]
                )

        for idx, seed in enumerate(seeds):
            with graph.inserting_before(seed):
                new_seed = graph.call_function(
                    inductor_prims.lookup_seed, (combined, idx)
                )
            seed.replace_all_uses_with(new_seed)
            new_seed.meta.update(seed.meta)
            graph.erase_node(seed)

    return len(device_seeds)


def default_kwargs(device):
    return {}


def get_device(device):
    if device is not None:
        return device
    return torch.empty([]).device  # default device


# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.rand.default), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.rand.generator), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randn.default), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randn.generator), pass_dict=patterns)
def replace_random(
    match: Match,
    size,
    *,
    generator=None,
    dtype=None,
    device=None,
    layout=None,
    pin_memory=None,
):
    if generator is not None:
        return

    def replacement(size):
        result = inductor_prims.random(
            size, inductor_prims.seed(device), mode, **default_kwargs(device)
        )
        if dtype is not None:
            result = result.to(dtype)
        return result

    mode = {
        aten.rand: "rand",
        aten.randn: "randn",
    }[
        match.output_node().target.overloadpacket  # type: ignore[union-attr]
    ]  # type: ignore[union-attr]
    device = get_device(device)
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [size])


# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randint.low), pass_dict=patterns)
def replace_randint(
    match: Match,
    low,
    high,
    size,
    *,
    dtype=torch.int64,
    device=None,
    layout=None,
    pin_memory=None,
):
    def replacement(low, high, size):
        result = inductor_prims.randint(low, high, size, inductor_prims.seed(device))
        return result.to(dtype)

    device = get_device(device)
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [low, high, size])

```



## High-Level Overview

"""Modify the given FX graph to use backend-native random ops"""    if config.fallback_random:        return 0    count = patterns.apply(gm)    with GraphTransformObserver(gm, "fuse_seed_creation_pass", "joint_graph_passes"):        count += fuse_seed_creation_pass(gm.graph)    return countdef fuse_seed_creation_pass(graph: torch.fx.Graph):

This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `replace_random_passes`, `fuse_seed_creation_pass`, `default_kwargs`, `get_device`, `replace_random`, `replacement`, `replace_randint`, `replacement`

**Key imports**: collections, logging, torch, GraphTransformObserver, _extract_tensor_metadata, config, inductor_prims, V


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `logging`
- `torch`
- `torch.fx.passes.graph_transform_observer`: GraphTransformObserver
- `torch.fx.passes.shape_prop`: _extract_tensor_metadata
- `..`: config, inductor_prims
- `..virtualized`: V


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

Files in the same folder (`torch/_inductor/fx_passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fuse_attention.py_docs.md`](./fuse_attention.py_docs.md)
- [`efficient_conv_bn_eval.py_docs.md`](./efficient_conv_bn_eval.py_docs.md)
- [`bucketing.py_docs.md`](./bucketing.py_docs.md)
- [`numeric_utils.py_docs.md`](./numeric_utils.py_docs.md)
- [`dedupe_symint_uses.py_docs.md`](./dedupe_symint_uses.py_docs.md)
- [`post_grad.py_docs.md`](./post_grad.py_docs.md)
- [`joint_graph.py_docs.md`](./joint_graph.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)


## Cross-References

- **File Documentation**: `replace_random.py_docs.md`
- **Keyword Index**: `replace_random.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
