# Documentation: `docs/torch/_export/passes/collect_tracepoints_pass.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/passes/collect_tracepoints_pass.py_docs.md`
- **Size**: 9,765 bytes (9.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_export/passes/collect_tracepoints_pass.py`

## File Metadata

- **Path**: `torch/_export/passes/collect_tracepoints_pass.py`
- **Size**: 6,522 bytes (6.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
from torch.export.exported_program import ConstantArgument, TensorArgument
from torch.fx.passes.infra.pass_base import PassBase, PassResult


if TYPE_CHECKING:
    from torch.export.exported_program import ModuleCallSignature
    from torch.export.graph_signature import ExportGraphSignature


__all__ = ["CollectTracepointsPass"]


class CollectTracepointsPass(PassBase):
    """
    Performs constant folding and constant propagation.
    """

    def __init__(
        self, specs: dict[str, ModuleCallSignature], sig: ExportGraphSignature
    ) -> None:
        super().__init__()
        self.specs = specs
        self.sig = sig

    def call(self, gm: torch.fx.GraphModule) -> PassResult | None:
        def get_arg_spec(arg) -> TensorArgument | ConstantArgument:
            if isinstance(arg, torch.fx.Node):
                if isinstance(arg.meta.get("val"), torch.Tensor):
                    return TensorArgument(name=arg.name)
                else:
                    raise AssertionError(
                        "Symint input is not implemented yet for submodule call signature."
                    )
            else:
                return ConstantArgument(name="", value=arg)

        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            nn_module_stack = None
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    kind = node.kwargs["kind"]
                    if kind == "module_call_outputs":
                        nn_module_stack = node.meta["nn_module_stack"]
                    elif kind == "module_call_inputs":
                        nn_module_stack = None
                    else:
                        raise AssertionError(f"Unknown tracepoint kind: {kind}")
                elif node.meta["nn_module_stack"] == nn_module_stack:
                    node.meta["nn_module_stack"].popitem()
                else:
                    nn_module_stack = None
            nn_module_stack = None
            for node in reversed(module.graph.nodes):
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    kind = node.kwargs["kind"]
                    if kind == "module_call_inputs":
                        nn_module_stack = node.meta["nn_module_stack"]
                    elif kind == "module_call_outputs":
                        nn_module_stack = None
                    else:
                        raise AssertionError(f"Unknown tracepoint kind: {kind}")
                elif node.meta["nn_module_stack"] == nn_module_stack:
                    node.meta["nn_module_stack"].popitem()
                else:
                    nn_module_stack = None

        def copy_sig(sig) -> ModuleCallSignature:
            from torch.export.exported_program import ModuleCallSignature

            return ModuleCallSignature(
                inputs=[],
                outputs=[],
                in_spec=sig.in_spec,
                out_spec=sig.out_spec,
                forward_arg_names=None,
            )

        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    # There's some subtlety worth noting. Here fqn corresponds to
                    # the call name, whereas path corresponds to the module name.
                    # They are not necessarily the same! When a submodule is shared
                    # through different aliases, there are as many _export_tracepoint
                    # markers as there are aliases, since the shared submodule is
                    # wrapped once for each alias.
                    path = node.kwargs["path"]
                    fqn, _ = next(reversed(node.meta["nn_module_stack"].values()))

                    module_key = next(reversed(node.meta["nn_module_stack"]))
                    if "@" in module_key:
                        suffix = module_key.split("@")[-1]
                        path = f"{path}@{suffix}"

                        call_fqn = f"{fqn}@{suffix}"
                        if call_fqn not in self.specs:
                            self.specs[call_fqn] = copy_sig(self.specs[fqn])
                        fqn = call_fqn

                    kind = node.kwargs["kind"]
                    for i, arg in enumerate(node.args):
                        # We only update the signature of the alias used to call
                        # the submodule. Otherwise the signatures of all aliases
                        # would get conflated; the inputs/outputs of every call
                        # would be recorded in every other call as well.
                        if fqn == path:
                            if kind == "module_call_inputs":
                                self.specs[path].inputs.append(get_arg_spec(arg))
                            elif kind == "module_call_outputs":
                                self.specs[path].outputs.append(get_arg_spec(arg))
                            else:
                                raise AssertionError(f"Unknown tracepoint kind: {kind}")
                        if isinstance(arg, torch.fx.Node):
                            for user in node.users:
                                assert user.op == "call_function"
                                assert user.target is operator.getitem
                                assert isinstance(user.args[1], int)
                                if user.args[1] == i:
                                    user.replace_all_uses_with(arg)
                                    self.sig.replace_all_uses(user.name, arg.name)
                                    break
                    users = list(node.users)
                    for user in users:
                        assert len(user.users) == 0
                        gm.graph.erase_node(user)
                    gm.graph.erase_node(node)
            return PassResult(gm, True)

        return None

```



## High-Level Overview

"""    Performs constant folding and constant propagation.

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CollectTracepointsPass`

**Functions defined**: `__init__`, `call`, `get_arg_spec`, `copy_sig`

**Key imports**: annotations, operator, TYPE_CHECKING, torch, ConstantArgument, TensorArgument, PassBase, PassResult, ModuleCallSignature, ExportGraphSignature, ModuleCallSignature


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `operator`
- `typing`: TYPE_CHECKING
- `torch`
- `torch.export.exported_program`: ConstantArgument, TensorArgument
- `torch.fx.passes.infra.pass_base`: PassBase, PassResult
- `torch.export.graph_signature`: ExportGraphSignature


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
- [`_node_metadata_hook.py_docs.md`](./_node_metadata_hook.py_docs.md)
- [`replace_set_grad_with_hop_pass.py_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md)
- [`functionalize_side_effectful_ops_pass.py_docs.md`](./functionalize_side_effectful_ops_pass.py_docs.md)
- [`insert_custom_op_guards.py_docs.md`](./insert_custom_op_guards.py_docs.md)
- [`constant_folding.py_docs.md`](./constant_folding.py_docs.md)
- [`replace_autocast_with_hop_pass.py_docs.md`](./replace_autocast_with_hop_pass.py_docs.md)
- [`add_runtime_assertions_for_constraints_pass.py_docs.md`](./add_runtime_assertions_for_constraints_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `collect_tracepoints_pass.py_docs.md`
- **Keyword Index**: `collect_tracepoints_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_export/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/_export/passes`):

- [`replace_set_grad_with_hop_pass.py_docs.md_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md_docs.md)
- [`_node_metadata_hook.py_docs.md_docs.md`](./_node_metadata_hook.py_docs.md_docs.md)
- [`replace_view_ops_with_view_copy_ops_pass.py_kw.md_docs.md`](./replace_view_ops_with_view_copy_ops_pass.py_kw.md_docs.md)
- [`lift_constants_pass.py_kw.md_docs.md`](./lift_constants_pass.py_kw.md_docs.md)
- [`remove_runtime_assertions.py_kw.md_docs.md`](./remove_runtime_assertions.py_kw.md_docs.md)
- [`lift_constants_pass.py_docs.md_docs.md`](./lift_constants_pass.py_docs.md_docs.md)
- [`constant_folding.py_docs.md_docs.md`](./constant_folding.py_docs.md_docs.md)
- [`remove_runtime_assertions.py_docs.md_docs.md`](./remove_runtime_assertions.py_docs.md_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md)
- [`constant_folding.py_kw.md_docs.md`](./constant_folding.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `collect_tracepoints_pass.py_docs.md_docs.md`
- **Keyword Index**: `collect_tracepoints_pass.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
