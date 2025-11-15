# Documentation: `docs/torch/_higher_order_ops/effects.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_higher_order_ops/effects.py_docs.md`
- **Size**: 13,683 bytes (13.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_higher_order_ops/effects.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/effects.py`
- **Size**: 9,855 bytes (9.62 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.torchbind import call_torchbind
from torch._library.custom_ops import CustomOpDef
from torch._library.effects import EffectType
from torch._library.utils import RegistrationHandle
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


_op_identifier = Union[
    str,
    "torch._ops.OpOverload",
    "torch._library.custom_ops.CustomOpDef",
    "torch._ops.HigherOrderOperator",
]
OpType = Union["torch._ops.HigherOrderOperator", "torch._ops.OpOverload"]

_EffectType = EffectType


def _get_op_qualname(op: _op_identifier) -> str:
    """Convert an op identifier to a qualified string key."""
    if isinstance(op, torch._ops.OpOverload):
        return op._name
    elif isinstance(op, torch._ops.HigherOrderOperator):
        return f"{op.namespace}::{op.name()}"
    elif isinstance(op, CustomOpDef):
        return op._qualname
    elif isinstance(op, str):
        return op

    raise ValueError(f"Invalid operator input {op}")


def _register_effectful_op(
    op: _op_identifier, effect: Optional[EffectType]
) -> RegistrationHandle:
    qualname = _get_op_qualname(op)
    entry = torch._library.simple_registry.singleton.find(qualname)
    handle = entry.effect.register(effect)
    return handle


def _get_effect(op: _op_identifier) -> Optional[_EffectType]:
    qualname = _get_op_qualname(op)
    entry = torch._library.simple_registry.singleton.find(qualname)
    return entry.effect.effect


_register_effectful_op("aten::_print", _EffectType.ORDERED)
_register_effectful_op("aten::_async_error", _EffectType.ORDERED)
_register_effectful_op("profiler::_record_function_exit._RecordFunction", None)
_register_effectful_op(call_torchbind, _EffectType.ORDERED)


class WithEffects(HigherOrderOperator):
    """
    with_effects(token, op, args, kwargs) -> (new_token, op_results)

    This HOP helps ensure ordering between side effectful ops like prints or ops
    using torchbind objects. This is needed to ensure a traced graph from
    AOTAutograd is functional so that future optimization passes do not reorder
    these operators. This is done through threading "effect tokens" through the
    graph to enforce data dependence between side effectful ops.

    The tokens are basically dummy values (torch.tensor([])). We create a token
    per "effect type", which are enumerated in the _EffectType enum.
    """

    def __init__(self) -> None:
        super().__init__("with_effects")

    def __call__(
        self,
        token,
        op: OpType,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        assert isinstance(op, (torch._ops.HigherOrderOperator, torch._ops.OpOverload))
        assert not has_aliasing(op), "Ops with aliasing is not supported"
        assert has_effects(op)
        assert isinstance(kwargs, dict)
        return super().__call__(token, op, *args, **kwargs)


with_effects = WithEffects()


def has_aliasing(op: OpType):
    # NOT FOR PUBLIC USE
    if isinstance(op, torch._ops.HigherOrderOperator):
        return not _get_effect(op)

    for arg in op._schema.arguments:
        if arg.alias_info is not None:
            return True
    for arg in op._schema.returns:
        if arg.alias_info is not None:
            return True
    return False


def has_effects(op) -> bool:
    # Skip over the profiler's RecordFunction as they should not show up in the graph
    _skip_ops = {torch.ops.profiler._record_function_exit._RecordFunction}
    if op in _skip_ops:
        return False

    return (
        isinstance(op, (torch._ops.HigherOrderOperator, torch._ops.OpOverload))
        and not has_aliasing(op)
        and _get_effect(op) is not None
    )


def new_token_tensor() -> torch.Tensor:
    return torch.tensor([])


@with_effects.py_impl(DispatchKey.CompositeExplicitAutograd)
def with_effects_dense(
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    out = op(*args, **kwargs)
    new_token = new_token_tensor()
    # [NOTE: with_effects return type]
    # Note that we should only do *out for tuple type, but not list type.
    # This is to match the schema of the op.
    # For tuple output, the length of schema output is the same as the length of out.
    # For list output, the length of schema output is 1 (e.g. Tensor[]) regardless of the
    # length of the list.
    if isinstance(out, tuple):
        return (new_token, *out)
    return (new_token, out)


@with_effects.py_impl(FakeTensorMode)
def with_effects_fake(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    with mode:
        result = with_effects_dense(token, op, *args, **kwargs)
        return result


@with_effects.py_impl(ProxyTorchDispatchMode)
def with_effects_proxy(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    with disable_proxy_modes_tracing():
        out = with_effects(token, op, *args, **kwargs)

    proxy_token = mode.tracer.unwrap_proxy(token)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    from torch.fx.node import has_side_effect

    # To avoid the being DCEed by graph.eliminate_dead_code if they.
    # don't have output or their outputs are not used.
    has_side_effect(op)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        with_effects,
        (proxy_token, op, *proxy_args),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


with_effects.fallthrough(DispatchKey.AutogradCPU)
with_effects.fallthrough(DispatchKey.AutogradCUDA)


def _get_schema(op, args) -> torch.FunctionSchema:
    if isinstance(op, torch._ops.OpOverload):
        return op._schema
    elif op == call_torchbind:
        return getattr(args[0], args[1]).schema
    else:
        raise RuntimeError(f"Unable to get schema for op {op}")


def handle_effects(
    allow_token_discovery: bool,
    tokens: dict[_EffectType, torch.Tensor],
    op: OpType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Args:
        allow_token_discovery: Whether or not we are discovering tokens. If this
        is true, we will create a token for every side effect type seen that
        does not have a token assigned yet.  If this is false, the tokens
        should've all been created ahead of time, so we will error if there is
        no token mapping to every effect type.

        tokens: Map of effect type to tokens. This is to chain operators of the
        same effects together so that they do not get reordered in later
        optimization passes.
    """

    # Get a token. We can't do `tokens.get(op, torch.tensor([]))` because
    # this will create an empty tensor during proxy mode tracing if the token
    # doesn't exist. But the tokens should always exist during proxy mode tracing.
    key = _get_effect(op)
    assert key is not None
    if key not in tokens:
        assert allow_token_discovery, (
            f"Could not find a token for effect {key} which came from the function {op}"
        )
        proxy_tensor_mode = torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.PROXY
        )
        if proxy_tensor_mode is not None:
            # If we discovered a new token during tracing, we are in backward.
            # Then we patch the graph, adding additional tangents_token as input to the joint graph.
            tracer = proxy_tensor_mode.tracer

            from torch.fx.experimental.proxy_tensor import (
                disable_proxy_modes_tracing,
                track_tensor_tree,
            )

            with disable_proxy_modes_tracing():
                token_tensor = new_token_tensor()

            token_proxy = proxy_tensor_mode.tracer.create_proxy(
                "placeholder", "tangents_token", (), {}, name="tangents_token"
            )
            track_tensor_tree(token_tensor, token_proxy, constant=None, tracer=tracer)

            tokens[key] = token_tensor
        else:
            tokens[key] = new_token_tensor()

    token = tokens[key]

    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI()

    unwrapped_token = ctx.unwrap_tensors([token])[0]
    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    with ctx.redispatch_to_next():
        (new_token, *unwrapped_outs) = with_effects(
            unwrapped_token, op, *unwrapped_args, **unwrapped_kwargs
        )

    schema = _get_schema(op, unwrapped_args)
    if len(schema.returns) == 0:
        assert unwrapped_outs[0] is None
        unwrapped_outs = None  # type: ignore[assignment]
    elif len(schema.returns) == 1:
        assert len(unwrapped_outs) == 1
        unwrapped_outs = unwrapped_outs[0]
    else:
        assert len(unwrapped_outs) == len(schema.returns)

    # Add the newly created token into the tokens map for a following call to
    # use this token.
    wrapped_token = ctx.wrap_tensors(new_token)
    assert isinstance(wrapped_token, torch.Tensor)
    tokens[key] = wrapped_token

    # pyrefly: ignore [bad-argument-type]
    return ctx.wrap_tensors(unwrapped_outs)

```



## High-Level Overview

"""Convert an op identifier to a qualified string key."""    if isinstance(op, torch._ops.OpOverload):        return op._name    elif isinstance(op, torch._ops.HigherOrderOperator):        return f"{op.namespace}::{op.name()}"    elif isinstance(op, CustomOpDef):        return op._qualname    elif isinstance(op, str):        return op    raise ValueError(f"Invalid operator input {op}")def _register_effectful_op(    op: _op_identifier, effect: Optional[EffectType]) -> RegistrationHandle:    qualname = _get_op_qualname(op)    entry = torch._library.simple_registry.singleton.find(qualname)    handle = entry.effect.register(effect)

This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WithEffects`

**Functions defined**: `_get_op_qualname`, `_register_effectful_op`, `_get_effect`, `__init__`, `__call__`, `has_aliasing`, `has_effects`, `new_token_tensor`, `with_effects_dense`, `with_effects_fake`, `with_effects_proxy`, `_get_schema`, `handle_effects`

**Key imports**: Any, Optional, Union, torch, torch.utils._pytree as pytree, DispatchKey, call_torchbind, CustomOpDef, EffectType, RegistrationHandle, HigherOrderOperator, FakeTensorMode


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, Optional, Union
- `torch`
- `torch.utils._pytree as pytree`
- `torch._C`: DispatchKey
- `torch._higher_order_ops.torchbind`: call_torchbind
- `torch._library.custom_ops`: CustomOpDef
- `torch._library.effects`: EffectType
- `torch._library.utils`: RegistrationHandle
- `torch._ops`: HigherOrderOperator
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch.fx.node`: has_side_effect
- `torch._subclasses.functional_tensor`: PythonFunctionalizeAPI


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/_higher_order_ops`):

- [`associative_scan.py_docs.md`](./associative_scan.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`foreach_map.py_docs.md`](./foreach_map.py_docs.md)
- [`strict_mode.py_docs.md`](./strict_mode.py_docs.md)
- [`torchbind.py_docs.md`](./torchbind.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`run_const_graph.py_docs.md`](./run_const_graph.py_docs.md)
- [`_invoke_quant.py_docs.md`](./_invoke_quant.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `effects.py_docs.md`
- **Keyword Index**: `effects.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_higher_order_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/_higher_order_ops`):

- [`schema.py_docs.md_docs.md`](./schema.py_docs.md_docs.md)
- [`run_const_graph.py_docs.md_docs.md`](./run_const_graph.py_docs.md_docs.md)
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`partitioner.py_docs.md_docs.md`](./partitioner.py_docs.md_docs.md)
- [`strict_mode.py_docs.md_docs.md`](./strict_mode.py_docs.md_docs.md)
- [`out_dtype.py_kw.md_docs.md`](./out_dtype.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`while_loop.py_kw.md_docs.md`](./while_loop.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`invoke_subgraph.py_docs.md_docs.md`](./invoke_subgraph.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `effects.py_docs.md_docs.md`
- **Keyword Index**: `effects.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
