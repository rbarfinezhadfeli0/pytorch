# Documentation: `torch/_inductor/bounds.py`

## File Metadata

- **Path**: `torch/_inductor/bounds.py`
- **Size**: 9,717 bytes (9.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import logging
import operator
from collections.abc import Callable
from functools import partial
from typing import Any, Optional, Union

import sympy
from sympy import Expr

import torch
from torch.utils._sympy.value_ranges import (
    bound_sympy,
    SymPyValueRangeAnalysis,
    ValueRanges,
)

from ..utils._sympy.functions import PowByNatural
from ..utils._sympy.numbers import int_oo
from .loop_body import InterpreterShim, LoopBody, LoopBodyBlock
from .ops_handler import DefaultHandler, ReductionType, StoreMode
from .utils import cache_on_self, dominated_nodes
from .virtualized import V


log = logging.getLogger(__name__)


class BoundVars:
    """
    Performs Value Range Analysis on LoopBody's fx graph by calling BoundVars.run()
    It exposes the ranges of the nodes in the `bounds` variable

    Note. A current limitation of this analysis is that it just works on a per-loop basis.
    We should be able to propagate the bounds between across the whole graph. This may benefit
    the case a bounded variable is returned by a kernel and fed into another.
    """

    def __init__(self, loop_body: LoopBody) -> None:
        def upper_bound(v: Union[Expr, int]) -> int:
            return bound_sympy(v).upper if isinstance(v, Expr) else v

        self.loop_body = loop_body
        self.replacement_vals = {
            k: ValueRanges[Expr](0, upper_bound(v) - 1)
            for k, v in loop_body.var_ranges.items()
        }
        # avoid computing these values, pessimistically assume that they are unbounded
        self.unbounded_vars = dominated_nodes(
            node
            for node in self.loop_body.get_nodes()
            if node.target in ["load", "reduction", operator.getitem]
            or "masked_subblock" in node.target
        )
        # To access this variable call `get_bounds()`
        self._bounds: dict[torch.fx.Node, ValueRanges[Expr]] = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loop_body={self.loop_body},\n "
            f"replacement_vals={self.replacement_vals}, \n"
            f"unbounded_vars={self.unbounded_vars}, \n"
            f"_bounds={self._bounds})"
        )

    @cache_on_self
    def get_bounds(self) -> dict[torch.fx.Node, ValueRanges[Expr]]:
        submodules = self.swap_submodules(self.loop_body.submodules)

        # Initialize the environment with the unbounded variables
        for node in self.unbounded_vars:
            # we need to evaluate masked_subblock to recurse, and we need to set indirect values
            if not isinstance(node.target, str) or (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self._bounds[node] = ValueRanges[Expr].unknown()

        with V.set_ops_handler(ValueRangeAnalysis()):
            interpreter = InterpreterShim(self.loop_body.root_block.graph, submodules)
            log.debug("get_bounds:\n%s", self.loop_body.root_block.graph)
            interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
        return self._bounds

    def swap_submodules(
        self, submodules: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., ValueRanges[Expr]]]:
        result: dict[str, Callable[..., ValueRanges[Expr]]] = {}
        for key in submodules:
            if key == "get_index":
                result[key] = self.get_index
            elif "masked_subblock" in key:
                subblock = self.loop_body.subblocks[key]
                # The result within the lambda will reference to the final
                # set of modules at the end of the for-loop as it stores a reference to it

                # bind subblock in a function because python lambdas close over by reference
                # moving the lambda out of make_fn would close over the reference to subblock,
                # so all lambdas would have the same subblock reference that is the final
                # subblock in the loop
                def make_fn(
                    subblock: LoopBodyBlock,
                ) -> Callable[[Any, Any], ValueRanges[Expr]]:
                    return lambda mask, value: self.masked_subblock(
                        subblock, self._bounds, mask, value, result
                    )

                result[key] = make_fn(subblock)
            elif "set_indirect" in key:
                idx = int(key[len("set_indirect") :])
                var = self.loop_body.indirect_vars[idx]
                indirect = partial(self.set_indirect, var)
                result[key] = indirect
            else:
                assert "scan" in key
                result[key] = submodules[key]

        return result

    def masked_subblock(
        self,
        subblock: LoopBodyBlock,
        env: dict[torch.fx.Node, ValueRanges[Expr]],
        mask: Any,
        value: Any,
        submodules: dict[str, Callable[..., Any]],
    ) -> ValueRanges[Expr]:
        interp = InterpreterShim(subblock.graph, submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        assert len(output) == 1
        # dont bother unioning with value since the load from buffer will be
        # pessimistically assumed to be inf anyway
        return interp.env[output[0]]

    def set_indirect(self, old: Expr, new: ValueRanges[Expr]) -> ValueRanges[Expr]:
        assert isinstance(new, ValueRanges)
        self.replacement_vals[old] = new
        return new

    def get_index(self, name: str) -> ValueRanges[Expr]:
        expr = self.loop_body.indexing_exprs[name]
        bound = self.replacement_vals.get(expr)
        if bound is None:
            bound = bound_sympy(expr, self.replacement_vals)
        # The following assertion is true at the time of this writing
        # We don't assert is as to not execute bound_sympy when bound is not None
        # assert bound is None or bound == bound_sympy(expr, self.replacement_vals)
        self.replacement_vals[name] = bound
        return bound


class ValueRangeAnalysis(SymPyValueRangeAnalysis, DefaultHandler):
    def __init__(self) -> None:
        self.name = "ValueRangeAnalysis"
        boolean_operators = (
            "xor",
            "logical_and",
            "logical_or",
            "logical_not",
        )
        for op in boolean_operators:
            setattr(self, op, self.bool_handler)

    @staticmethod
    def bool_handler(*args: Any, **kwargs: Any) -> ValueRanges[Any]:
        # just assuming bools can have both values
        return ValueRanges(sympy.false, sympy.true)  # type: ignore[arg-type]

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # many ops are unlikely to show up in optimizable indexing compute,
        # so we dont have full coverage
        return ValueRanges.unknown()

    def load(self, name: str, index: sympy.Expr) -> ValueRanges[Any]:
        return ValueRanges.unknown()

    def store(
        self, name: str, index: sympy.Expr, value: Any, mode: StoreMode = None
    ) -> None:
        return

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Any,
    ) -> ValueRanges[Any]:
        return ValueRanges.unknown()

    @classmethod
    def index_expr(cls, index: Any, dtype: torch.dtype) -> ValueRanges[Any]:
        assert isinstance(index, ValueRanges)
        return cls.to_dtype(index, dtype)

    @staticmethod
    def to_dtype(
        x: Any,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = True,
    ) -> ValueRanges[Any]:
        x = ValueRanges.wrap(x)

        if dtype == torch.bool:
            if x.is_singleton():
                return ValueRanges.wrap(x.lower != 0)
            elif x.is_bool:
                return x
            elif 0 not in x:
                return ValueRanges.wrap(sympy.true)
            else:
                return ValueRanges(sympy.false, sympy.true)

        def cast(x: Any, dtype: torch.dtype) -> sympy.Expr:
            # dtype is int or float
            if dtype.is_floating_point:
                return sympy.Float(x)
            else:
                if x in (int_oo, -int_oo):
                    return x
                try:
                    return sympy.Integer(x)
                except TypeError:
                    # inf cannot be cast to Integer
                    return x

        if x.is_bool:
            if x.is_singleton():
                val = 1 if x.lower else 0
                return ValueRanges.wrap(cast(val, dtype))
            else:
                return ValueRanges(cast(0, dtype), cast(1, dtype))
        else:
            # int to float or float to int
            return ValueRanges(cast(x.lower, dtype), cast(x.upper, dtype))

    @staticmethod
    def square(x: Any) -> ValueRanges[Any]:
        return ValueRanges.convex_min_zero_map(x, lambda y: PowByNatural(y, 2))

    @staticmethod
    def neg(x: Any) -> ValueRanges[Any]:
        return ValueRanges.decreasing_map(x, operator.neg)

    # TODO: this is slightly inaccurate because truncdiv operates at integer
    # precision, but we're going through float truediv which means we can
    # potentially lose precision on the bounds
    @classmethod
    def truncdiv(cls, a: Any, b: Any) -> ValueRanges[Any]:
        x = cls.truediv(a, b)
        if x == ValueRanges.unknown():
            return x

        return cls.trunc(x)

    @classmethod
    def sub(cls, a: Any, b: Any) -> ValueRanges[Any]:
        return cls.add(a, cls.neg(b))

```



## High-Level Overview

"""    Performs Value Range Analysis on LoopBody's fx graph by calling BoundVars.run()    It exposes the ranges of the nodes in the `bounds` variable    Note. A current limitation of this analysis is that it just works on a per-loop basis.    We should be able to propagate the bounds between across the whole graph. This may benefit    the case a bounded variable is returned by a kernel and fed into another.

This Python file contains 2 class(es) and 22 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BoundVars`, `ValueRangeAnalysis`

**Functions defined**: `__init__`, `upper_bound`, `__repr__`, `get_bounds`, `swap_submodules`, `make_fn`, `masked_subblock`, `set_indirect`, `get_index`, `__init__`, `bool_handler`, `_default`, `load`, `store`, `reduction`, `index_expr`, `to_dtype`, `cast`, `square`, `neg`

**Key imports**: logging, operator, Callable, partial, Any, Optional, Union, sympy, Expr, torch, PowByNatural, int_oo


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `operator`
- `collections.abc`: Callable
- `functools`: partial
- `typing`: Any, Optional, Union
- `sympy`
- `torch`
- `..utils._sympy.functions`: PowByNatural
- `..utils._sympy.numbers`: int_oo
- `.loop_body`: InterpreterShim, LoopBody, LoopBodyBlock
- `.ops_handler`: DefaultHandler, ReductionType, StoreMode
- `.utils`: cache_on_self, dominated_nodes
- `.virtualized`: V


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `bounds.py_docs.md`
- **Keyword Index**: `bounds.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
