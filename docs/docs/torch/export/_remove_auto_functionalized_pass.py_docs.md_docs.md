# Documentation: `docs/torch/export/_remove_auto_functionalized_pass.py_docs.md`

## File Metadata

- **Path**: `docs/torch/export/_remove_auto_functionalized_pass.py_docs.md`
- **Size**: 4,733 bytes (4.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/export/_remove_auto_functionalized_pass.py`

## File Metadata

- **Path**: `torch/export/_remove_auto_functionalized_pass.py`
- **Size**: 1,951 bytes (1.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
from torch.export import ExportedProgram
from torch.fx import Graph


def remove_self_clone(graph: Graph) -> None:
    for node in graph.nodes:
        if node.target is torch.ops.aten.copy_.default and node.args[0] == node.args[1]:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)


def unsafe_remove_auto_functionalized_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    This pass removes an instances of the higher order op 'auto_functionalized',
    and modifies the calling EP inplace to have the original mutator op.
    This pass doesn't perform safety checks to make sure that this inplace mutation is safe.
    """

    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        for module in ep.graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in ep.graph.nodes:
                if (
                    node.op == "call_function" and node.target is auto_functionalized
                ) or (
                    node.op == "call_function" and node.target is auto_functionalized_v2
                ):
                    func = node.args[0]
                    assert isinstance(func, torch._ops.OpOverload)
                    # re-inplace everything
                    node.meta["only_clone_these_tensors"] = []
            decompose_auto_functionalized(ep.graph)
            remove_self_clone(ep.graph)
            ep.graph.eliminate_dead_code()

    return ep

```



## High-Level Overview

"""    This pass removes an instances of the higher order op 'auto_functionalized',    and modifies the calling EP inplace to have the original mutator op.    This pass doesn't perform safety checks to make sure that this inplace mutation is safe.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `remove_self_clone`, `unsafe_remove_auto_functionalized_pass`

**Key imports**: torch, decompose_auto_functionalized, ExportedProgram, Graph


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.fx_passes.post_grad`: decompose_auto_functionalized
- `torch.export`: ExportedProgram
- `torch.fx`: Graph


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

Files in the same folder (`torch/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`exported_program.py_docs.md`](./exported_program.py_docs.md)
- [`_wrapper_utils.py_docs.md`](./_wrapper_utils.py_docs.md)
- [`_unlift.py_docs.md`](./_unlift.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_swap.py_docs.md`](./_swap.py_docs.md)
- [`_tree_utils.py_docs.md`](./_tree_utils.py_docs.md)
- [`_safeguard.py_docs.md`](./_safeguard.py_docs.md)
- [`_remove_effect_tokens_pass.py_docs.md`](./_remove_effect_tokens_pass.py_docs.md)


## Cross-References

- **File Documentation**: `_remove_auto_functionalized_pass.py_docs.md`
- **Keyword Index**: `_remove_auto_functionalized_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/export`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/export`):

- [`custom_obj.py_kw.md_docs.md`](./custom_obj.py_kw.md_docs.md)
- [`_unlift.py_docs.md_docs.md`](./_unlift.py_docs.md_docs.md)
- [`_trace.py_kw.md_docs.md`](./_trace.py_kw.md_docs.md)
- [`_leakage_detection_utils.py_docs.md_docs.md`](./_leakage_detection_utils.py_docs.md_docs.md)
- [`_unlift.py_kw.md_docs.md`](./_unlift.py_kw.md_docs.md)
- [`_trace.py_docs.md_docs.md`](./_trace.py_docs.md_docs.md)
- [`_safeguard.py_kw.md_docs.md`](./_safeguard.py_kw.md_docs.md)
- [`custom_ops.py_docs.md_docs.md`](./custom_ops.py_docs.md_docs.md)
- [`graph_signature.py_kw.md_docs.md`](./graph_signature.py_kw.md_docs.md)
- [`_swap.py_docs.md_docs.md`](./_swap.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_remove_auto_functionalized_pass.py_docs.md_docs.md`
- **Keyword Index**: `_remove_auto_functionalized_pass.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
