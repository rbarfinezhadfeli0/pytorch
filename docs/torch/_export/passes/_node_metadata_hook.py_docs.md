# Documentation: `torch/_export/passes/_node_metadata_hook.py`

## File Metadata

- **Path**: `torch/_export/passes/_node_metadata_hook.py`
- **Size**: 3,379 bytes (3.30 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import contextlib
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.graph_module import GraphModule


_EMPTY_NN_MODULE_STACK_KEY = "_empty_nn_module_stack_from_metadata_hook"


def _node_metadata_hook(
    node: torch.fx.Node,
    metadata: Optional[dict[str, Any]] = None,
    fake_mode: Optional[FakeTensorMode] = None,
) -> None:
    """
    Hook for adding the appropriate metadata to nodes that are created during a
    pass using graph.create_node. An example of how to use it:

    ```
    with _set_node_metadata_hook(gm,
        functools.partial(_node_metadata_hook, metadata={"stack_trace": "file"})
    ):
        pass(gm)
    ```

    This hook should not work for all generic cases -- specifically it assumes
    that nodes being added are only call_function nodes, and copies over the
    first argument node's nn_module_stack.
    """
    # pyrefly: ignore [bad-assignment]
    fake_mode = fake_mode or contextlib.nullcontext()

    assert node.op == "call_function" and callable(node.target), (
        f"node: {node}, target: {node.target}"
    )

    if (
        isinstance(node.target, torch._ops.OpOverload)
        and len(node.target._schema.returns) == 0
    ):
        node.meta["val"] = None
    else:
        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.fx.Node, lambda arg: arg.meta["val"], (node.args, node.kwargs)
        )
        # pyrefly: ignore [bad-context-manager]
        with fake_mode, enable_python_dispatcher():
            fake_res = node.target(*fake_args, **fake_kwargs)
        node.meta["val"] = fake_res

    if metadata is not None:
        for k, v in metadata.items():
            node.meta[k] = v

    # Copy over metadata from argument nodes
    arg_meta = [
        arg.meta
        for arg in pytree.tree_flatten((node.args, node.kwargs))[0]
        if isinstance(arg, torch.fx.Node)
    ]
    if len(arg_meta) == 0:
        return
    arg_meta = arg_meta[0]

    node.meta["nn_module_stack"] = node.meta.get(
        "nn_module_stack",
        arg_meta.get(
            "nn_module_stack",
            {
                _EMPTY_NN_MODULE_STACK_KEY: (
                    _EMPTY_NN_MODULE_STACK_KEY,
                    _EMPTY_NN_MODULE_STACK_KEY,
                )
            },
        ),
    )

    node.meta["torch_fn"] = node.meta.get(
        "torch_fn",
        (
            f"{node.target.__name__}_0",
            # pyrefly: ignore [missing-attribute]
            f"{node.target.__class__.__name__}.{node.target.__name__}",
        ),
    )


@contextlib.contextmanager
def _set_node_metadata_hook(gm: torch.fx.GraphModule, f):
    """
    Takes a callable which will be called after we create a new node. The
    callable takes the newly created node as input and returns None.
    """
    assert callable(f), "node_metadata_hook must be a callable."

    # Add the hook to all submodules
    for m in gm.modules():
        if isinstance(m, GraphModule):
            m._register_create_node_hook(f)
    try:
        yield
    finally:
        # Restore hook for all submodules
        for m in gm.modules():
            if isinstance(m, GraphModule):
                m._unregister_create_node_hook(f)

```



## High-Level Overview

"""    Hook for adding the appropriate metadata to nodes that are created during a    pass using graph.create_node. An example of how to use it:    ```    with _set_node_metadata_hook(gm,        functools.partial(_node_metadata_hook, metadata={"stack_trace": "file"})    ):        pass(gm)    ```    This hook should not work for all generic cases -- specifically it assumes    that nodes being added are only call_function nodes, and copies over the    first argument node's nn_module_stack.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_node_metadata_hook`, `_set_node_metadata_hook`

**Key imports**: contextlib, Any, Optional, torch, torch.utils._pytree as pytree, enable_python_dispatcher, FakeTensorMode, GraphModule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `typing`: Any, Optional
- `torch`
- `torch.utils._pytree as pytree`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch.fx.graph_module`: GraphModule


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

Files in the same folder (`torch/_export/passes`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`replace_set_grad_with_hop_pass.py_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md)
- [`functionalize_side_effectful_ops_pass.py_docs.md`](./functionalize_side_effectful_ops_pass.py_docs.md)
- [`insert_custom_op_guards.py_docs.md`](./insert_custom_op_guards.py_docs.md)
- [`constant_folding.py_docs.md`](./constant_folding.py_docs.md)
- [`replace_autocast_with_hop_pass.py_docs.md`](./replace_autocast_with_hop_pass.py_docs.md)
- [`add_runtime_assertions_for_constraints_pass.py_docs.md`](./add_runtime_assertions_for_constraints_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `_node_metadata_hook.py_docs.md`
- **Keyword Index**: `_node_metadata_hook.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
