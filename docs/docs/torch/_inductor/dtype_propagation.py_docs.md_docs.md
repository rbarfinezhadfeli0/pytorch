# Documentation: `docs/torch/_inductor/dtype_propagation.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/dtype_propagation.py_docs.md`
- **Size**: 15,346 bytes (14.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/dtype_propagation.py`

## File Metadata

- **Path**: `torch/_inductor/dtype_propagation.py`
- **Size**: 12,027 bytes (11.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools
from collections.abc import Callable, Sequence
from typing import Any, Optional, Protocol, TYPE_CHECKING, TypeVar, Union

import sympy

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, type_to_dtype
from torch.utils._ordered_set import OrderedSet

from .ops_handler import OP_NAMES, OpsHandler
from .utils import upcast_compute_type
from .virtualized import OpsValue, V


T = TypeVar("T")


class DTypeVar(Protocol):
    @property
    def dtype(self) -> torch.dtype: ...


DTypeArg = Union[DTypeVar, torch.types.Number, str, OpsValue]


# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.cache
def get_promoted_dtype(
    *args: Sequence[tuple[torch.dtype, bool]],
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND] = None,
):
    def construct_input(inp):
        if inp[1]:
            return torch.empty([], dtype=inp[0])
        else:
            return torch.empty([1], dtype=inp[0])

    inps = [construct_input(arg) for arg in args]
    _, dtype = torch._prims_common.elementwise_dtypes(
        *inps,
        type_promotion_kind=(
            type_promotion_kind
            if type_promotion_kind
            else ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        ),
    )
    return dtype


def promote_types(
    args: Sequence[DTypeArg],
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND] = None,
):
    dtype_prop_candidates = []

    # pyrefly: ignore [bad-assignment]
    for arg in args:
        assert not isinstance(arg, str)
        if isinstance(arg, OpsValue):
            arg = arg.value
            assert isinstance(arg, torch._prims_common.Number) or hasattr(arg, "dtype")

        if isinstance(arg, torch._prims_common.Number):
            dtype_prop_candidates.append((type_to_dtype(type(arg)), True))
            continue

        # pyrefly: ignore [missing-attribute]
        dtype_prop_candidates.append((arg.dtype, getattr(arg, "is_scalar", False)))

    dtype = get_promoted_dtype(
        *dtype_prop_candidates,
        type_promotion_kind=type_promotion_kind,
    )

    return dtype


class DtypePropagationOpsHandler:
    """
    Propagate dtype from args to output
    """

    # Singleton DtypePropagationOpsHandler, because we meta program over a number of op rules.
    # Those are only defined after other inductor state has run.

    _instance: Optional["DtypePropagationOpsHandler"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for op, rule in torch._inductor.utils.op_dtype_propagation_rules.items():
            fn = (
                functools.partial(self.return_dtype, dtype=rule.override_return_dtype)
                if rule.override_return_dtype
                else functools.partial(
                    self.op_dtype_rule, type_promotion_kind=rule.type_promotion_kind
                )
            )
            setattr(self, op, fn)

        # Set pointwise operation rules
        for op in torch._inductor.codegen.common.pointwise_overrides_data.values():
            if not hasattr(self, op.name):
                setattr(
                    self,
                    op.name,
                    functools.partial(
                        self.op_dtype_rule, type_promotion_kind=op.type_promotion_kind
                    ),
                )

        # Set boolean operation rules
        for op in torch._inductor.utils.boolean_ops():
            if not hasattr(self, op):
                setattr(
                    self, op, functools.partial(self.return_dtype, dtype=torch.bool)
                )

        unimplemented_ops = OP_NAMES - OrderedSet(dir(self))
        torch._check(
            len(unimplemented_ops) == 0,
            lambda: f"Unimplemented dtype rule for ops: {unimplemented_ops}",
        )

    # metaprogrammed in __init__

    @staticmethod
    def op_dtype_rule(
        *args: DTypeArg, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
    ) -> torch.dtype:
        return promote_types(args, type_promotion_kind=type_promotion_kind)

    @staticmethod
    def return_dtype(*args: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    # op rules

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> torch.dtype:
        return upcast_compute_type(dtype)

    @staticmethod
    def load_seed(name: str, offset: int) -> torch.dtype:
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def randint64(seed: int, offset: int, low: int, high: int) -> torch.dtype:
        return torch.int64

    @staticmethod
    def masked(
        mask: DTypeArg, body: Callable[[], DTypeArg], other: DTypeArg
    ) -> torch.dtype:
        from .loop_body import LoopBodyBlock

        assert isinstance(body, LoopBodyBlock), "body must be a LoopBodyBlock"
        # TODO - we avoid calling this in codegen, needs work for non codegen use cases
        loads = body.graph.find_nodes(op="call_method", target="load")
        if len(loads) <= 1:
            return promote_types([other])

        return upcast_compute_type(V.graph.get_dtype(loads[-1].args[1]))

    @staticmethod
    def where(a: DTypeArg, b: DTypeArg, c: DTypeArg) -> torch.dtype:
        return promote_types([b, c])

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> torch.dtype:
        # TODO - TODO - rationalize index_expr. The dtype is not always used and we are inconsistent about int32 or int64
        # in lowerings. cpp just uses the dtype
        if dtype not in (torch.int32, torch.int64) or not hasattr(
            V.kernel, "index_dtype"
        ):
            return upcast_compute_type(dtype)

        return V.kernel.get_index_dtype_as_torch_dtype()

    @staticmethod
    def to_dtype(
        x: DTypeArg,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types=True,
    ) -> torch.dtype:
        return upcast_compute_type(dtype) if use_compute_types else dtype

    @staticmethod
    def to_dtype_bitcast(
        x: DTypeArg, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> torch.dtype:
        return upcast_compute_type(dtype)

    @staticmethod
    def gelu(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def mul(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def truediv(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def pow(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def mod(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def indirect_indexing(
        x: DTypeArg, size: int, check: bool = True, wrap_neg: bool = True
    ) -> torch.dtype:
        return torch.int64

    @staticmethod
    def randn(seed: int, offset: int) -> torch.dtype:
        return torch.float

    @staticmethod
    def rand(seed: int, offset: int) -> torch.dtype:
        return torch.float

    @staticmethod
    def store_reduction(name: str, index, value: DTypeArg) -> None:
        return None

    @staticmethod
    def reduction(
        dtype: torch.dtype, src_dtype: torch.dtype, reduction_type: str, value: DTypeArg
    ) -> torch.dtype:
        return dtype

    @staticmethod
    def store(name: str, index, value: DTypeArg, mode: Optional[str] = None) -> None:
        return None

    @staticmethod
    def partial_accumulate(
        name: str,
        reduction_type: str,
        value: DTypeArg,
        extra_meta: dict[str, Any],
    ) -> None:
        return None

    @staticmethod
    def load(name: str, index) -> torch.dtype:
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def floor(x: DTypeArg) -> torch.dtype:
        return promote_types(
            [x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def ceil_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def int_truediv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types(
            [x, y], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def scan(
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[[tuple[T, ...], tuple[T, ...]], tuple[T, ...]],
        values: tuple[T, ...],
    ) -> tuple[torch.dtype, ...]:
        return dtypes

    @staticmethod
    def fmod(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def round_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def identity(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def frexp(x: DTypeArg) -> tuple[torch.dtype, torch.dtype]:
        # TODO - need to handle multiple outputs
        return (promote_types([x]), torch.int32)

    @staticmethod
    def sort(
        dtypes: tuple[torch.dtype, ...],
        values: tuple[T, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[torch.dtype, ...]:
        return dtypes

    @staticmethod
    def trunc(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def bucketize(
        values: DTypeArg,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: DTypeArg,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[tuple[str, sympy.Expr]] = None,
        sorter_indices: Optional[T] = None,
    ) -> torch.dtype:
        return indexing_dtype

    @staticmethod
    def rshift(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def round(x: DTypeArg) -> torch.dtype:
        return promote_types(
            [x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def trunc_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def floor_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def truncdiv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def floordiv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def halide_clamp(value, size, check):
        # TODO - way of registering dtype for op in backend
        return torch.int32

    @staticmethod
    def dot(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        # triton tl.dot out_dtype is tl.float32 by default.
        return torch.float32

    @staticmethod
    def inline_asm_elementwise(
        *inputs, asm, constraints=None, dtype=torch.float32, is_pure=True, pack=1
    ):
        return dtype

    @staticmethod
    def lshift(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def check_bounds(
        expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        return None

    def output(self, *args: DTypeArg) -> None:
        raise AssertionError(
            f"{type(self).__name__}: ops.output should not appear here"
        )

    def placeholder(self, index: int) -> torch.dtype:
        raise AssertionError(
            f"{type(self).__name__}: ops.placeholder should not appear here"
        )

    @staticmethod
    def device_assert_async(cond, msg: str) -> None:
        return None


if TYPE_CHECKING:

    class _typecheck_DtypePropagation(DtypePropagationOpsHandler, OpsHandler[Any]):
        pass  # mypy will error if we got any of the signatures wrong

```



## High-Level Overview


This Python file contains 3 class(es) and 54 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DTypeVar`, `DtypePropagationOpsHandler`, `_typecheck_DtypePropagation`

**Functions defined**: `dtype`, `get_promoted_dtype`, `construct_input`, `promote_types`, `__new__`, `__init__`, `op_dtype_rule`, `return_dtype`, `constant`, `load_seed`, `randint64`, `masked`, `where`, `index_expr`, `to_dtype`, `to_dtype_bitcast`, `gelu`, `mul`, `truediv`, `pow`

**Key imports**: functools, Callable, Sequence, Any, Optional, Protocol, TYPE_CHECKING, TypeVar, Union, sympy, torch, ELEMENTWISE_TYPE_PROMOTION_KIND, type_to_dtype, OrderedSet, OP_NAMES, OpsHandler, upcast_compute_type, OpsValue, V


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `collections.abc`: Callable, Sequence
- `typing`: Any, Optional, Protocol, TYPE_CHECKING, TypeVar, Union
- `sympy`
- `torch`
- `torch._prims_common`: ELEMENTWISE_TYPE_PROMOTION_KIND, type_to_dtype
- `torch.utils._ordered_set`: OrderedSet
- `.ops_handler`: OP_NAMES, OpsHandler
- `.utils`: upcast_compute_type
- `.virtualized`: OpsValue, V
- `.loop_body`: LoopBodyBlock


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

- **File Documentation**: `dtype_propagation.py_docs.md`
- **Keyword Index**: `dtype_propagation.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `dtype_propagation.py_docs.md_docs.md`
- **Keyword Index**: `dtype_propagation.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
