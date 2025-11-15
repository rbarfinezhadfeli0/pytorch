# Documentation: `docs/torch/_export/passes/add_runtime_assertions_for_constraints_pass.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/passes/add_runtime_assertions_for_constraints_pass.py_docs.md`
- **Size**: 13,704 bytes (13.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_export/passes/add_runtime_assertions_for_constraints_pass.py`

## File Metadata

- **Path**: `torch/_export/passes/add_runtime_assertions_for_constraints_pass.py`
- **Size**: 10,226 bytes (9.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import math
import operator
import traceback
from functools import partial
from typing import NamedTuple, TYPE_CHECKING

import sympy

import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["InputDim"]


class InputDim(NamedTuple):
    input_name: str
    dim: int


def _convert_to_int(val):
    # Convert simple sympy Integers into concrete int
    if val in (sympy.oo, int_oo):
        return math.inf
    if val in (-sympy.oo, -int_oo):
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)
    raise RuntimeError("Export constraints cannot be non-integer expressions")


def _convert_range_to_int(range: ValueRanges):
    assert isinstance(range, ValueRanges)
    min_val = _convert_to_int(range.lower)
    max_val = _convert_to_int(range.upper)
    return min_val, max_val


class _AddRuntimeAssertionsForInlineConstraintsPass(PassBase):
    def __init__(
        self,
        range_constraints: dict[sympy.Symbol, ValueRanges],
    ):
        super().__init__()
        self.range_constraints: dict[sympy.Symbol, ValueRanges] = range_constraints
        self._asserts_generated_unbacked_symbols: set[sympy.Symbol] = set()
        self.counter = 0

    def _assert_range_constraint(self, node, lower, upper, assert_msg):
        last_node = node
        if lower > -math.inf:
            last_node = self._insert_assert_async(
                last_node, operator.ge, node, lower, assert_msg
            )

        if upper < math.inf:
            last_node = self._insert_assert_async(
                last_node, operator.le, node, upper, assert_msg
            )

    def _insert_assert_async(self, last_node, op, lower, upper, assert_msg):
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
        self.counter += 1
        graph = last_node.graph
        with graph.inserting_after(last_node):
            cmp = graph.call_function(op, (lower, upper), {})
        with graph.inserting_after(cmp):
            cmp_tensor = graph.call_function(
                torch.ops.aten.scalar_tensor.default, (cmp,), {}
            )
        with graph.inserting_after(cmp_tensor):
            assert_async = graph.call_function(
                torch.ops.aten._assert_async.msg,
                (cmp_tensor, assert_msg),
                {},
            )
        return assert_async

    def call(self, graph_module) -> PassResult:
        self.existing_inline_assertions = _get_existing_inline_assertions(
            graph_module, self.range_constraints
        )

        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if "val" not in node.meta:
                    continue

                val = node.meta["val"]
                # In general, we may have to deal the case such as: ret[1].shape[0].
                # We need first find out what symbols require assertion, then we need to follow the path
                # from ret to the symbol, construct the proxies along the way and construct the messages
                # piece-wise at the same time.
                #
                # We use post-order traversal to collect all the proxies callbacks needed, construct
                # the error message callbacks, and at the top-level traversal tree we execute all the callbacks.
                # We need the callbacks because, in order to call the function to create a proxy for shape[0], we
                # need the proxy for shape, which further requires the proxy for ret[1], etc.

                def add_assertions(val):
                    call_backs: list[Callable] = []
                    messages: list[str] = []
                    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                        symbol = val.node.expr
                        if symbol in self.existing_inline_assertions:
                            return call_backs, messages
                        if isinstance(symbol, sympy.Symbol) and free_unbacked_symbols(
                            symbol
                        ):
                            if symbol in self._asserts_generated_unbacked_symbols:
                                return call_backs, messages
                            # We only care about unbacked symints for these inline
                            # constraints, which are prefixed with 'u'
                            constraint = self.range_constraints[symbol]
                            min_val, max_val = _convert_range_to_int(constraint)
                            assert_msg = f" is outside of inline constraint [{min_val}, {max_val}]."
                            call_backs.append(
                                partial(
                                    self._assert_range_constraint,
                                    lower=min_val,
                                    upper=max_val,
                                )
                            )
                            messages.append(assert_msg)
                            self._asserts_generated_unbacked_symbols.add(symbol)

                    elif isinstance(val, torch.Tensor):
                        for i, sym in enumerate(val.shape):
                            cbs, msgs = add_assertions(sym)
                            for cb, msg in zip(cbs, msgs):

                                def sym_size_cb(node, assert_msg, dim):
                                    with node.graph.inserting_after(node):
                                        dim_node = module.graph.call_function(
                                            torch.ops.aten.sym_size.int,
                                            (node, dim),
                                            {},
                                        )
                                    cb(node=dim_node, assert_msg=assert_msg)

                                call_backs.append(partial(sym_size_cb, dim=i))
                                messages.append(f".shape[{i}]" + msg)
                    return call_backs, messages

                callbacks, messages = add_assertions(val)
                for cb, msg in zip(callbacks, messages):
                    cb(node=node, assert_msg=f"{node}" + msg)

            module.recompile()

        # Sometimes this pass would return a wrong graph where we have mismatched
        # node names in signature. Before we fix it, let's just skip it.
        if (
            self.counter == 0
            and type(self) is _AddRuntimeAssertionsForInlineConstraintsPass
        ):
            return PassResult(graph_module, False)

        # Populate the stack trace with dummy vals to respect IR
        for node in graph_module.graph.nodes:
            if not node.meta.get("stack_trace", None) and node.op not in [
                "placeholder",
                "output",
            ]:
                node.meta["stack_trace"] = "".join(traceback.format_stack(limit=1))
        return PassResult(graph_module, True)


def _get_existing_inline_assertions(
    graph_module: torch.fx.GraphModule,
    range_constraints: dict[sympy.Symbol, ValueRanges],
) -> dict[sympy.Symbol, ValueRanges]:
    existing_inline_assertions: dict[sympy.Symbol, ValueRanges] = {}

    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        # Find all the existing inline assertions. They will look something like:
        # %_local_scalar_dense = call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%arg1_1,), kwargs = {})
        # %ge = call_function[target=operator.ge](args = (%_local_scalar_dense, 0), kwargs = {})
        # %_assert_scalar = call_function[target=torch.ops.aten._assert_scalar.default](args = (%scalar_tensor, "..."), kwargs = {})
        for node in module.graph.nodes:
            if node.target != torch.ops.aten._assert_scalar.default:
                continue

            compare_arg = node.args[0]
            if not (
                isinstance(compare_arg, torch.fx.Node)
                and compare_arg.op == "call_function"
                and compare_arg.target in (operator.le, operator.ge)
                and len(compare_arg.args) == 2
            ):
                continue

            compare_op = compare_arg.target
            lhs, rhs = compare_arg.args

            def maybe_get_symint(x):
                if (
                    isinstance(x, torch.fx.Node)
                    and "val" in x.meta
                    and isinstance(x.meta["val"], torch.SymInt)
                ):
                    return x.meta["val"].node.expr
                return x

            lhs = maybe_get_symint(lhs)
            rhs = maybe_get_symint(rhs)

            if compare_op is operator.ge:
                lhs, rhs = rhs, lhs

            if isinstance(lhs, sympy.Symbol) and isinstance(rhs, int):
                symint = lhs
                scalar = rhs
            elif isinstance(rhs, sympy.Symbol) and isinstance(lhs, int):
                symint = rhs
                scalar = lhs
            else:
                continue

            if symint not in range_constraints:
                raise RuntimeError(
                    f"Unable to find symint {symint} in {range_constraints}"
                )

            previous_range = existing_inline_assertions.get(
                symint, ValueRanges(-math.inf, math.inf)
            )

            if symint is lhs:
                bounds = ValueRanges(-math.inf, scalar)
            else:
                bounds = ValueRanges(scalar, math.inf)
            existing_inline_assertions[symint] = previous_range & bounds

    return existing_inline_assertions

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `InputDim`, `_AddRuntimeAssertionsForInlineConstraintsPass`

**Functions defined**: `_convert_to_int`, `_convert_range_to_int`, `__init__`, `_assert_range_constraint`, `_insert_assert_async`, `call`, `add_assertions`, `sym_size_cb`, `_get_existing_inline_assertions`, `maybe_get_symint`

**Key imports**: math, operator, traceback, partial, NamedTuple, TYPE_CHECKING, sympy, torch, torch.fx, free_unbacked_symbols, PassBase, PassResult


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `operator`
- `traceback`
- `functools`: partial
- `typing`: NamedTuple, TYPE_CHECKING
- `sympy`
- `torch`
- `torch.fx`
- `torch.fx.experimental.symbolic_shapes`: free_unbacked_symbols
- `torch.fx.passes.infra.pass_base`: PassBase, PassResult
- `torch.utils._sympy.numbers`: int_oo
- `torch.utils._sympy.value_ranges`: ValueRanges
- `collections.abc`: Callable


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
- [`constant_folding.py_docs.md`](./constant_folding.py_docs.md)
- [`replace_autocast_with_hop_pass.py_docs.md`](./replace_autocast_with_hop_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `add_runtime_assertions_for_constraints_pass.py_docs.md`
- **Keyword Index**: `add_runtime_assertions_for_constraints_pass.py_kw.md`
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
- [`constant_folding.py_docs.md_docs.md`](./constant_folding.py_docs.md_docs.md)
- [`remove_runtime_assertions.py_docs.md_docs.md`](./remove_runtime_assertions.py_docs.md_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md)
- [`constant_folding.py_kw.md_docs.md`](./constant_folding.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `add_runtime_assertions_for_constraints_pass.py_docs.md_docs.md`
- **Keyword Index**: `add_runtime_assertions_for_constraints_pass.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
