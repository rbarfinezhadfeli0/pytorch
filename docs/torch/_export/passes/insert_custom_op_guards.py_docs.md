# Documentation: `torch/_export/passes/insert_custom_op_guards.py`

## File Metadata

- **Path**: `torch/_export/passes/insert_custom_op_guards.py`
- **Size**: 2,917 bytes (2.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import functools
from collections import defaultdict

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)
from torch._library.fake_profile import OpProfile, TensorMetadata


def insert_custom_op_guards(gm: torch.fx.GraphModule, ops_to_guard: set[str]) -> None:
    """
    This is used by draft_export to insert guards in front of calls to custom
    operators which have a generated fake kernel.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops_to_guard:
            with (
                _set_node_metadata_hook(
                    gm,
                    functools.partial(
                        _node_metadata_hook,
                        metadata={"stack_trace": node.meta.get("stack_trace")},
                    ),
                ),
                gm.graph.inserting_before(node),
            ):
                for arg in (*node.args, *node.kwargs.values()):
                    if isinstance(arg, torch.fx.Node) and isinstance(
                        arg.meta.get("val"), torch.Tensor
                    ):
                        val = arg.meta["val"]
                        gm.graph.call_function(
                            torch.ops.aten._assert_tensor_metadata.default,
                            args=(arg,),
                            kwargs={
                                "dtype": val.dtype,
                                "device": val.device,
                                "layout": val.layout,
                            },
                        )

    gm.recompile()


def get_op_profiles(
    gm: torch.fx.GraphModule, ops_to_guard: set[str]
) -> dict[str, set[OpProfile]]:
    """
    This is used by draft_export to get a list of custom operator profiles so
    that we can generate fake kernels.
    """

    def _get_op_profile(node: torch.fx.Node) -> OpProfile:
        args_profile = tuple(
            TensorMetadata.maybe_from_tensor(arg.meta.get("val"))
            if isinstance(arg, torch.fx.Node)
            else None
            for arg in (*node.args, *node.kwargs.values())
        )

        out_profile = None
        meta = node.meta.get("val")
        assert meta is not None
        if isinstance(meta, torch.Tensor):
            out_profile = TensorMetadata.maybe_from_tensor(meta)
        elif isinstance(meta, (list, tuple)):
            out_profile = tuple(TensorMetadata.maybe_from_tensor(m) for m in meta)  # type: ignore[assignment]
        assert out_profile is not None

        return OpProfile(args_profile, out_profile)  # type: ignore[arg-type]

    op_profiles: dict[str, set[OpProfile]] = defaultdict(set)

    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops_to_guard:
            op_profiles[str(node.target)].add(_get_op_profile(node))

    return op_profiles

```



## High-Level Overview

"""    This is used by draft_export to insert guards in front of calls to custom    operators which have a generated fake kernel.

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `insert_custom_op_guards`, `get_op_profiles`, `_get_op_profile`

**Key imports**: functools, defaultdict, torch, OpProfile, TensorMetadata


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `collections`: defaultdict
- `torch`
- `torch._library.fake_profile`: OpProfile, TensorMetadata


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_export/passes`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_node_metadata_hook.py_docs.md`](./_node_metadata_hook.py_docs.md)
- [`replace_set_grad_with_hop_pass.py_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md)
- [`functionalize_side_effectful_ops_pass.py_docs.md`](./functionalize_side_effectful_ops_pass.py_docs.md)
- [`constant_folding.py_docs.md`](./constant_folding.py_docs.md)
- [`replace_autocast_with_hop_pass.py_docs.md`](./replace_autocast_with_hop_pass.py_docs.md)
- [`add_runtime_assertions_for_constraints_pass.py_docs.md`](./add_runtime_assertions_for_constraints_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `insert_custom_op_guards.py_docs.md`
- **Keyword Index**: `insert_custom_op_guards.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
