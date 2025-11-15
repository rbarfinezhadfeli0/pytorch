# Documentation: `docs/torch/_export/passes/constant_folding.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/passes/constant_folding.py_docs.md`
- **Size**: 14,430 bytes (14.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_export/passes/constant_folding.py`

## File Metadata

- **Path**: `torch/_export/passes/constant_folding.py`
- **Size**: 11,303 bytes (11.04 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import collections
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree


aten = torch.ops.aten

# We would like to split modules into two subgraphs for runtime weight updates to work correctly.
# The use case and more information could be found at:
# https://docs.google.com/document/d/1inZC-8KarJ6gKB7G9egmYLx1V_dKX_apxon0w4zPC0Q/edit?usp=sharing
META_TAG = "MODULE_TYPE"
MODULE_TAG = "_MAIN_MODULE"
CONST_MODULE_TAG = "_CONST_MODULE"


def replace_node_with_constant(gm, node, constant, name=None):
    g = gm.graph

    if name:
        qualname = name
    else:
        if not hasattr(gm, "_frozen_param_count"):
            gm._frozen_param_count = 0
        i = gm._frozen_param_count

        while True:
            qualname = f"_frozen_param{i}"
            if not hasattr(gm, qualname):
                break
            i += 1

        gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


class ConstantFolder(torch.fx.Interpreter):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        skip_constructors: bool = False,
    ):
        super().__init__(gm)
        self.node_replacements: dict[torch.fx.Node, Any] = {}
        self.replaced_uses: dict[torch.fx.Node, int] = collections.Counter()
        self.unknown_value = object()
        self.skip_constructors: bool = skip_constructors

        # overwrite this to deallocate env values if their only remaining use
        # is the output
        self.user_to_last_uses = self.node_to_last_non_output_use()

    def is_impure(self, node: torch.fx.Node) -> bool:
        if (
            node.target is torch.ops.prims.convert_element_type.default
            and node.args[0].op == "get_attr"  # type: ignore[union-attr]
            and node.args[0].meta["val"].dtype == torch.int8  # type: ignore[union-attr]
            and node.args[1] == torch.bfloat16
        ):
            # For int8_weight -> dq -> bf16_weight
            return True
        if node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            torch.ops.pt2e_quant.dequantize_affine,
        ]:
            # For the pattern fp32_weight -> q -> dq
            # We only folding fp32_weight -> q
            # int8_weight and leave dq in graph to be fused
            return True
        return False

    def node_to_last_non_output_use(self):
        last_non_output_use = collections.defaultdict(list)
        seen_uses = set()
        output_node = next(iter(reversed(self.module.graph.nodes)))  # type: ignore[arg-type, union-attr]

        for node in reversed(self.module.graph.nodes):  # type: ignore[arg-type, union-attr]
            if node.target == "output":
                continue

            def add_use(inp):
                if inp in seen_uses:
                    return

                seen_uses.add(inp)
                last_non_output_use[node].append(inp)

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, add_use, (node.args, node.kwargs))

            # if this node is only used in output, we want to gc it right away
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)

        return last_non_output_use

    def run_node(self, node):
        if node.target == "output":
            # because we remove nodes from env on last non output use,
            # re-define them now or we'll get error in interpreter
            def set_env(arg):
                self.env[arg] = self.unknown_value

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, set_env, node.args)
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

        # We need to do this weird thing because in cases where flattened_inputs
        # contains a ScriptObject, equality checking results in a type error if
        # the types are different.
        if any(
            type(self.unknown_value) is type(input_) and self.unknown_value == input_
            for input_ in flattened_inputs
        ):
            return self.unknown_value

        # TODO - fix errors with this
        if (
            node.op == "call_function"
            and node.target is aten._efficientzerotensor.default
        ):
            return self.unknown_value

        # TODO - constant folding triton kernel returns the inputs -- fix this
        if (
            node.op == "call_function"
            and node.name == "triton_kernel_wrapper_functional_proxy"
        ):
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        # TODO - more complicated strategy
        if (
            self.skip_constructors
            and node.op != "get_attr"
            and not any(isinstance(e, torch.Tensor) for e in flattened_inputs)
        ):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return self.unknown_value

        out = super().run_node(node)

        if node.op != "get_attr" and isinstance(out, torch.Tensor):
            if out.device.type == "meta":
                return out

            if not self.insertable_tensor_check(out):
                return out

            if self.is_impure(node):
                return self.unknown_value

            self.add_node_replacement(node, out)

            flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)

            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue

                self.replaced_uses[n] += 1

            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)

        return out

    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    def run(self):  # type: ignore[override]
        env = {}
        for n in self.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            env[n] = self.unknown_value
        return super().run(initial_env=env)


def constant_fold(
    gm: torch.fx.GraphModule,
    constraint_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
):
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(gm, skip_constructors=True)
        cf.run()

        for node, constant in cf.node_replacements.items():
            if constraint_fn is not None and not constraint_fn(node):
                continue
            replace_node_with_constant(gm, node, constant)

        erased_params = []
        # Get all attr users by looking up the graph instead from node.users, because in this case
        # _tensor_constant0 and _tensor_constant0_1 are actually refereing to the same tensor.

        #     opcode         name                 target            args                         kwargs
        # -------------  -------------------  ----------------  ---------------------------  --------
        # placeholder    arg0_1               arg0              ()                           {}
        # get_attr       _tensor_constant0    state             ()                           {}
        # call_function  add                  aten.add.Tensor   (arg0_1, _tensor_constant0)  {}
        # get_attr       _tensor_constant0_1  state             ()                           {}
        # call_function  add_                 aten.add_.Tensor  (_tensor_constant0_1, 1)     {}
        # output         output               output            ([add],)                     {}

        get_attr_node_users = defaultdict(list)
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                get_attr_node_users[node.target].extend(node.users.keys())
        for node in gm.graph.find_nodes(op="get_attr"):
            if node.op == "get_attr" and len(get_attr_node_users[node.target]) == 0:
                if hasattr(gm, node.target):
                    delattr(gm, node.target)
                erased_params.append(node)
        for node in erased_params:
            gm.graph.erase_node(node)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()


def constant_graph_tag(gm: torch.fx.GraphModule) -> None:
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(gm, skip_constructors=True)
        cf.run()

        for node in gm.graph.nodes:
            if (
                node.op == "get_attr"
                or node in cf.node_replacements
                or node in cf.replaced_uses
            ):
                node.meta[META_TAG] = CONST_MODULE_TAG
            else:
                node.meta[META_TAG] = MODULE_TAG


def run_and_get_constant_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Construct a GraphModule which corresponds to the part which could be
    constant folded in provided gm.
    """

    constant_graph_tag(gm)
    # We rewrite the tags, if it's a constant being directly consumed, without
    # any folding opportunity, we keep it in main gm.
    for node in gm.graph.find_nodes(op="get_attr"):
        used_to_fold = False
        for u in node.users:
            if u.meta[META_TAG] == CONST_MODULE_TAG:
                used_to_fold = True
                break
        if not used_to_fold:
            node.meta[META_TAG] = MODULE_TAG

    new_graph = torch.fx.Graph()

    node_remapping: dict[torch.fx.Node, torch.fx.Node] = {}
    output_nodes = []
    for node in gm.graph.nodes:
        if node.meta[META_TAG] == MODULE_TAG:
            continue

        new_node = new_graph.node_copy(node, lambda x: node_remapping[x])
        node_remapping[node] = new_node

        for user in node.users:
            if user.meta[META_TAG] == MODULE_TAG:
                output_nodes.append(new_node)
                break

    new_graph.output(tuple(output_nodes))
    new_graph.lint()
    new_gm = torch.fx.GraphModule(gm, new_graph)

    return new_gm

```



## High-Level Overview


This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConstantFolder`

**Functions defined**: `replace_node_with_constant`, `__init__`, `is_impure`, `node_to_last_non_output_use`, `add_use`, `run_node`, `set_env`, `insertable_tensor_check`, `add_node_replacement`, `run`, `constant_fold`, `constant_graph_tag`, `run_and_get_constant_graph`

**Key imports**: collections, defaultdict, Callable, Any, Optional, torch, torch.utils._pytree as pytree


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `collections.abc`: Callable
- `typing`: Any, Optional
- `torch`
- `torch.utils._pytree as pytree`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
- [`insert_custom_op_guards.py_docs.md`](./insert_custom_op_guards.py_docs.md)
- [`replace_autocast_with_hop_pass.py_docs.md`](./replace_autocast_with_hop_pass.py_docs.md)
- [`add_runtime_assertions_for_constraints_pass.py_docs.md`](./add_runtime_assertions_for_constraints_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `constant_folding.py_docs.md`
- **Keyword Index**: `constant_folding.py_kw.md`
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

- May involve **JIT compilation** or compilation optimizations.
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
- [`remove_runtime_assertions.py_docs.md_docs.md`](./remove_runtime_assertions.py_docs.md_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md)
- [`constant_folding.py_kw.md_docs.md`](./constant_folding.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `constant_folding.py_docs.md_docs.md`
- **Keyword Index**: `constant_folding.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
