# Documentation: `docs/torch/fx/operator_schemas.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/operator_schemas.py_docs.md`
- **Size**: 25,290 bytes (24.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/operator_schemas.py`

## File Metadata

- **Path**: `torch/fx/operator_schemas.py`
- **Size**: 21,904 bytes (21.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import enum
import inspect
import numbers
import types
import typing
import warnings
from collections.abc import Callable
from typing import Any, cast, NamedTuple, Optional, TYPE_CHECKING

import torch
from torch._jit_internal import boolean_dispatched
from torch._ops import OpOverload, OpOverloadPacket

from ._compatibility import compatibility


if TYPE_CHECKING:
    from .node import Argument

__all__ = [
    "ArgsKwargsPair",
    "check_for_mutable_operation",
    "get_signature_for_torch_op",
    "create_type_hint",
    "type_matches",
    "normalize_function",
    "normalize_module",
]


@compatibility(is_backward_compatible=False)
class ArgsKwargsPair(NamedTuple):
    """
    Simple named tuple for wrapping args/kwargs pairs.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


_manual_overrides: dict[Callable, list[inspect.Signature]] = {}


def _nonzero_schemas():
    signatures = []

    def nonzero(self):
        pass

    signatures.append(inspect.signature(nonzero))

    def nonzero(self, *, as_tuple: bool):  # type: ignore[no-redef]
        pass

    signatures.append(inspect.signature(nonzero))

    return signatures


_manual_overrides[torch.nonzero] = _nonzero_schemas()


class _FakeGlobalNamespace:
    def __getattr__(self, name):
        if name == "torch":
            return torch
        raise RuntimeError("Expected a torch namespace lookup")


_type_eval_globals = {
    "Tensor": torch.Tensor,
    "Device": torch.device,
    "Layout": torch.layout,
    "number": numbers.Number,
    "Future": torch.jit.Future,
    "AnyEnumType": enum.Enum,
    "QScheme": torch.qscheme,
    "__torch__": _FakeGlobalNamespace(),
    "NoneType": type(None),
    "Storage": torch.UntypedStorage,
    "t": typing.TypeVar("t"),
    "PyObject": Any,
}
for k in dir(typing):
    _type_eval_globals[k] = getattr(typing, k)


def _torchscript_type_to_python_type(ts_type: "torch._C.JitType") -> Any:
    """
    Convert a TorchScript type to a Python type (including subtypes) via
    eval'ing the annotation_str. _type_eval_globals sets up expressions
    like "List" and "Future" to map to actual types (typing.List and jit.Future)
    """
    return eval(ts_type.annotation_str, _type_eval_globals)


def _torchscript_schema_to_signature_impl(
    ts_schema: torch._C.FunctionSchema,
) -> inspect.Signature:
    from inspect import Parameter

    parameters: list[Parameter] = []
    for arg in ts_schema.arguments:
        arg_type = _torchscript_type_to_python_type(arg.type)
        default = arg.default_value if arg.has_default_value() else Parameter.empty
        # TODO: Figure out if this is safe. It seems like when generating the type signatures for
        # PythonArgParser, we emit signatures with `input` instead of `self` as the first tensor
        # argument name. Downstream, if someone converts that positional argument to a keyword
        # argument, the name mismatch will break things, so here we're going to normalize the
        # name to "input"
        name = arg.name if arg.name != "self" else "input"
        kind = (
            Parameter.KEYWORD_ONLY
            if arg.kwarg_only
            else Parameter.POSITIONAL_OR_KEYWORD
        )
        # "from" is a keyword therefore it must be a POSITIONAL_ONLY argument
        if name == "from":
            assert kind == Parameter.POSITIONAL_OR_KEYWORD
            # ParameterKind type is internal implementation detail to inspec package
            # which makes it hard to do type annotation
            kind = Parameter.POSITIONAL_ONLY  # type: ignore[assignment]
            # This renders all previous arguments to positional only

            for idx, p in enumerate(parameters):
                assert p.kind == Parameter.POSITIONAL_OR_KEYWORD
                parameters[idx] = Parameter(
                    name=p.name,
                    kind=Parameter.POSITIONAL_ONLY,
                    default=p.default,
                    annotation=p.annotation,
                )

        parameters.append(
            Parameter(name=name, kind=kind, default=default, annotation=arg_type)
        )
    return_types = [
        _torchscript_type_to_python_type(ret.type) for ret in ts_schema.returns
    ]
    if len(return_types) == 0:
        return_type = None
    elif len(return_types) == 1:
        return_type = return_types[0]
    else:
        return_type = tuple(return_types)

    return inspect.Signature(parameters, return_annotation=return_type)


_SCHEMA_TO_SIGNATURE_CACHE: dict[tuple[str, str], inspect.Signature] = {}


def _torchscript_schema_to_signature(
    ts_schema: torch._C.FunctionSchema,
) -> inspect.Signature:
    # Cached as it's called in the hot path of FakeTensor dispatch
    cache_key = ts_schema.name, ts_schema.overload_name
    cache_val = _SCHEMA_TO_SIGNATURE_CACHE.get(cache_key)
    if cache_val is not None:
        return cache_val

    res = _torchscript_schema_to_signature_impl(ts_schema)
    _SCHEMA_TO_SIGNATURE_CACHE[cache_key] = res
    return res


@compatibility(is_backward_compatible=False)
def check_for_mutable_operation(
    target: Callable, args: tuple["Argument", ...], kwargs: dict[str, "Argument"]
):
    signatures, schemas = get_signature_for_torch_op(target, return_schemas=True)

    if signatures and schemas:
        matched_schemas = []

        # Iterate through all of the schema until we find one that matches
        # If one matches, populate `new_args_and_kwargs` with the new args/kwargs
        # values. If none matches, `new_args_and_kwargs` will be None
        for candidate_signature, schema in zip(signatures, schemas):
            try:
                candidate_signature.bind(*args, **kwargs)
                matched_schemas.append((candidate_signature, schema))
            except TypeError:
                continue

        def throw_if_mutable(schema):
            if schema.is_mutable:
                raise RuntimeError(
                    f"Tried to trace mutable operation {schema}. FX only supports functional "
                    f"code, so operations that mutate operands in-place (e.g. via `out` arguments) "
                    f"are not supported"
                )

        if len(matched_schemas) == 0:
            # Did not match any schema. Cannot check for mutation
            pass
        elif len(matched_schemas) == 1:
            # Matched exactly one schema, unambiguous
            _, schema_to_check = matched_schemas[0]
            throw_if_mutable(schema_to_check)
        else:
            # Ambiguous schema match. Since mutability checking is best effort,
            # do nothing.
            pass


@compatibility(is_backward_compatible=False)
def get_signature_for_torch_op(op: Callable, return_schemas: bool = False):
    """
    Given an operator on the `torch` namespace, return a list of `inspect.Signature`
    objects corresponding to the overloads of that op.. May return `None` if a signature
    could not be retrieved.

    Args:
        op (Callable): An operator on the `torch` namespace to look up a signature for

    Returns:
        Optional[List[inspect.Signature]]: A list of signatures for the overloads of this
            operator, or None if the operator signatures could not be retrieved. If
            return_schemas=True, returns a tuple containing the optional Python signatures
            and the optional TorchScript Function signature
    """
    if isinstance(op, OpOverload):
        schemas = [op._schema]
    elif isinstance(op, OpOverloadPacket):
        schemas = [getattr(op, overload)._schema for overload in op.overloads()]
    else:
        override = _manual_overrides.get(op)
        if override:
            return (override, None) if return_schemas else None

        aten_fn = torch.jit._builtins._find_builtin(op)

        if aten_fn is None:
            return (None, None) if return_schemas else None
        schemas = torch._C._jit_get_schemas_for_operator(aten_fn)

    signatures = [_torchscript_schema_to_signature(schema) for schema in schemas]
    return (signatures, schemas) if return_schemas else signatures


@compatibility(is_backward_compatible=False)
def create_type_hint(x):
    """
    Produces a type hint for the given argument.

    The :func:`create_type_hint` looks for a type hint compatible with the input argument `x`.

    If `x` is a `list` or `tuple`, it looks for an object in the list whose type is a superclass
    of the rest, and uses that as `base_type` for the `List` or `Tuple` to be returned.
    If no such object is found, it defaults to `List[Any]`.

    If `x` is neither a `list` nor a `tuple`, it returns `x`.
    """
    try:
        if isinstance(x, (list, tuple)):
            # todo(chilli): Figure out the right way for mypy to handle this
            if isinstance(x, list):

                def ret_type(x):
                    return list[x]  # type: ignore[valid-type]

            else:

                def ret_type(x):
                    return tuple[x, ...]  # type: ignore[valid-type]

            if len(x) == 0:
                return ret_type(Any)
            base_type = x[0]
            for t in x:
                if issubclass(t, base_type):
                    continue
                elif issubclass(base_type, t):
                    base_type = t
                else:
                    return ret_type(Any)
            return ret_type(base_type)
    except Exception:
        # We tried to create a type hint for list but failed.
        warnings.warn(
            f"We were not able to successfully create type hint from the type {x}"
        )
    return x


@compatibility(is_backward_compatible=False)
def type_matches(signature_type: Any, argument_type: Any):
    sig_origin_type = getattr(signature_type, "__origin__", signature_type)

    if signature_type is argument_type:
        return True

    # Union types in signature. Given type needs to match one of the
    # contained types in the Union
    if sig_origin_type is typing.Union and signature_type != argument_type:
        sig_contained = signature_type.__args__
        return any(type_matches(c, argument_type) for c in sig_contained)

    if getattr(signature_type, "__origin__", None) is list:
        sig_el_type = signature_type.__args__[0]

        # int can be promoted to list[int]
        if argument_type is int and sig_el_type is int:
            return True

        if not inspect.isclass(sig_el_type):
            warnings.warn(
                f"Does not support nested parametric types, got {signature_type}. Please file a bug."
            )
            return False
        if getattr(argument_type, "__origin__", None) is list:
            return issubclass(argument_type.__args__[0], sig_el_type)

        def is_homogeneous_tuple(t):
            if getattr(t, "__origin__", None) is not tuple:
                return False
            contained = t.__args__
            if t.__args__ == ((),):  # Tuple[()].__args__ == ((),) for some reason
                return True
            return all((c is Ellipsis) or issubclass(c, sig_el_type) for c in contained)

        # Tuple[T] is accepted for List[T] parameters
        return is_homogeneous_tuple(argument_type)

    # Dtype is an int in schemas
    if signature_type is int and argument_type is torch.dtype:
        return True

    if signature_type is numbers.Number and argument_type in {int, float}:
        return True
    if inspect.isclass(argument_type) and inspect.isclass(signature_type):
        return issubclass(argument_type, signature_type)

    return False


@compatibility(is_backward_compatible=False)
def normalize_function(
    target: Callable,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    arg_types: Optional[tuple[Any]] = None,
    kwarg_types: Optional[dict[str, Any]] = None,
    normalize_to_only_use_kwargs: bool = False,
) -> Optional[ArgsKwargsPair]:
    """
    Returns normalized arguments to PyTorch functions. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs). Does not support modules.

    May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

    Args:
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
        kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
    if kwargs is None:
        kwargs = {}
    new_args_and_kwargs = None
    if (
        not isinstance(target, types.BuiltinFunctionType)
        and not (isinstance(target, (OpOverloadPacket, OpOverload)))
        and hasattr(target, "_op")
    ):
        # ExecuTorch's EdgeOpOverload are a wrapper around PyTorch's OpOverload,
        # so we can unwrap it here to get its schema
        # Can't import EdgeOpOverload directly because of a circular dependency,
        # so checking for "_op" existing is the next best thing.
        target = target._op

    # Repeat the condition after checking for the inner _op field.
    if not isinstance(target, types.BuiltinFunctionType) and not (
        isinstance(target, (OpOverloadPacket, OpOverload))
    ):
        target_for_analysis = target
        if target in boolean_dispatched:
            # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
            # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
            # branches of the dispatch have exactly the same signature. If they do, use the `true`
            # branch signature for analysis. Otherwise, leave this un-normalized
            assert not isinstance(target, str)
            dispatched = boolean_dispatched[target]
            if_true, if_false = dispatched["if_true"], dispatched["if_false"]
            if (
                inspect.signature(if_true).parameters
                != inspect.signature(if_false).parameters
            ):
                return None
            target_for_analysis = if_true

        assert callable(target_for_analysis)
        sig = inspect.signature(inspect.unwrap(target_for_analysis))
        new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
            sig, args, kwargs, normalize_to_only_use_kwargs
        )
    else:
        assert callable(target)
        torch_op_schemas = get_signature_for_torch_op(target)
        matched_schemas = []
        if torch_op_schemas:
            # Iterate through all of the schema until we find one that matches
            # If one matches, populate `new_args_and_kwargs` with the new args/kwargs
            # values. If none matches, `new_args_and_kwargs` will be None
            for candidate_signature in torch_op_schemas:
                try:
                    candidate_signature.bind(*args, **kwargs)
                    matched_schemas.append(candidate_signature)
                except TypeError:
                    continue

            if len(matched_schemas) == 0:
                # Did not match any schema. Cannot normalize
                pass
            elif len(matched_schemas) == 1:
                # Matched exactly one schema, unambiguous
                new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
                    matched_schemas[0], args, kwargs, normalize_to_only_use_kwargs
                )
            else:
                if arg_types is not None or kwarg_types is not None:
                    arg_types = arg_types if arg_types else cast(tuple[Any], ())
                    kwarg_types = kwarg_types if kwarg_types else {}
                    for candidate_signature in torch_op_schemas:
                        sig_matches = True
                        try:
                            bound_types = candidate_signature.bind(
                                *arg_types, **kwarg_types
                            )
                            for arg_name, arg_type in bound_types.arguments.items():
                                param = candidate_signature.parameters[arg_name]
                                sig_matches = sig_matches and type_matches(
                                    param.annotation, arg_type
                                )
                        except TypeError:
                            sig_matches = False
                        if sig_matches:
                            new_args_and_kwargs = (
                                _args_kwargs_to_normalized_args_kwargs(
                                    candidate_signature,
                                    args,
                                    kwargs,
                                    normalize_to_only_use_kwargs,
                                )
                            )
                            break
                else:
                    # Matched more than one schema. In this situation, the caller must provide the types of
                    # the arguments of the overload they expect.
                    schema_printouts = "\n".join(
                        str(schema) for schema in matched_schemas
                    )
                    raise RuntimeError(
                        f"Tried to normalize arguments to {torch.typename(target)} but "
                        f"the schema match was ambiguous! Please provide argument types to "
                        f"the normalize_arguments() call. Available schemas:\n{schema_printouts}"
                    )

    return new_args_and_kwargs


@compatibility(is_backward_compatible=False)
def normalize_module(
    root: torch.nn.Module,
    target: str,
    args: tuple[Any],
    kwargs: Optional[dict[str, Any]] = None,
    normalize_to_only_use_kwargs: bool = False,
) -> Optional[ArgsKwargsPair]:
    """
    Returns normalized arguments to PyTorch modules. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (*args, **kwargs).

    Args:
        root (nn.Module): root module upon which we query modules
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
    try:
        submod = root.get_submodule(target)
    except AttributeError as e:
        raise RuntimeError(
            f"Tried to normalize node with target {target} but root did not "
            f"have that target!"
        ) from e
    if hasattr(submod.__class__, "__name__"):
        classname = submod.__class__.__name__
        if getattr(torch.nn, classname, None) == submod.__class__:
            sig = inspect.signature(inspect.unwrap(submod.forward))
            if kwargs is None:
                kwargs = {}
            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
                sig, args, kwargs, normalize_to_only_use_kwargs
            )
            return new_args_and_kwargs
    return None


def _args_kwargs_to_normalized_args_kwargs(
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    normalize_to_only_use_kwargs: bool,
) -> Optional[ArgsKwargsPair]:
    """
    Given a call target, args, and kwargs, return the arguments normalized into
    an ArgsKwargsPair, or None if the type signature is not supported by
    this normalization.

    Args:

        sig (inspect.Signature): Signature object for the target
        args (Tuple): Arguments that appear at the callsite for `target`
        kwargs (Dict): Keyword arguments that appear at the callsite for `target`
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Optional[ArgsKwargsPair]: Normalized args and kwargs for `target`, or `None` if
            this target is not supported.
    """

    # Don't currently support positional-only
    # or varargs (*args, **kwargs) signatures
    supported_parameter_types = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
        # Add an exception for one signature, which is common for random/uniform, i.e.:
        # Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None
        # `from` is Python keyword and as such functions with that signature should have
        # positional-only args, but at the same time they could be dispatched as kwargs
        if list(sig.parameters.keys()) != ["input", "from", "to", "generator"]:
            return None

    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    new_kwargs: dict[str, Any] = {}
    new_args: list[Any] = []
    for i, param in enumerate(sig.parameters):
        if not normalize_to_only_use_kwargs and i < len(args):
            new_args.append(bound_args.arguments[param])
        else:
            new_kwargs[param] = bound_args.arguments[param]

    return ArgsKwargsPair(tuple(new_args), new_kwargs)

```



## High-Level Overview

"""    Simple named tuple for wrapping args/kwargs pairs.

This Python file contains 3 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ArgsKwargsPair`, `_FakeGlobalNamespace`

**Functions defined**: `_nonzero_schemas`, `nonzero`, `nonzero`, `__getattr__`, `_torchscript_type_to_python_type`, `_torchscript_schema_to_signature_impl`, `_torchscript_schema_to_signature`, `check_for_mutable_operation`, `throw_if_mutable`, `get_signature_for_torch_op`, `create_type_hint`, `ret_type`, `ret_type`, `type_matches`, `is_homogeneous_tuple`, `normalize_function`, `normalize_module`, `_args_kwargs_to_normalized_args_kwargs`

**Key imports**: enum, inspect, numbers, types, typing, warnings, Callable, Any, cast, NamedTuple, Optional, TYPE_CHECKING, torch, boolean_dispatched


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `enum`
- `inspect`
- `numbers`
- `types`
- `typing`
- `warnings`
- `collections.abc`: Callable
- `torch`
- `torch._jit_internal`: boolean_dispatched
- `torch._ops`: OpOverload, OpOverloadPacket
- `._compatibility`: compatibility
- `.node`: Argument
- `EdgeOpOverload directly because of a circular dependency,`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/fx`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`tensor_type.py_docs.md`](./tensor_type.py_docs.md)
- [`traceback.py_docs.md`](./traceback.py_docs.md)
- [`_symbolic_trace.py_docs.md`](./_symbolic_trace.py_docs.md)
- [`graph.py_docs.md`](./graph.py_docs.md)
- [`node.py_docs.md`](./node.py_docs.md)
- [`annotate.py_docs.md`](./annotate.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`subgraph_rewriter.py_docs.md`](./subgraph_rewriter.py_docs.md)


## Cross-References

- **File Documentation**: `operator_schemas.py_docs.md`
- **Keyword Index**: `operator_schemas.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/fx`):

- [`annotate.py_kw.md_docs.md`](./annotate.py_kw.md_docs.md)
- [`_compatibility.py_docs.md_docs.md`](./_compatibility.py_docs.md_docs.md)
- [`tensor_type.py_kw.md_docs.md`](./tensor_type.py_kw.md_docs.md)
- [`_graph_pickler.py_kw.md_docs.md`](./_graph_pickler.py_kw.md_docs.md)
- [`_compatibility.py_kw.md_docs.md`](./_compatibility.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`interpreter.py_kw.md_docs.md`](./interpreter.py_kw.md_docs.md)
- [`subgraph_rewriter.py_docs.md_docs.md`](./subgraph_rewriter.py_docs.md_docs.md)
- [`node.py_docs.md_docs.md`](./node.py_docs.md_docs.md)
- [`graph_module.py_docs.md_docs.md`](./graph_module.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `operator_schemas.py_docs.md_docs.md`
- **Keyword Index**: `operator_schemas.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
