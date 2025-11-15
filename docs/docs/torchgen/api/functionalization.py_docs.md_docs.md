# Documentation: `docs/torchgen/api/functionalization.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/api/functionalization.py_docs.md`
- **Size**: 10,246 bytes (10.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/api/functionalization.py`

## File Metadata

- **Path**: `torchgen/api/functionalization.py`
- **Size**: 7,815 bytes (7.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from torchgen.api import dispatcher
from torchgen.api.types import (
    BaseCppType,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    longT,
    NamedCType,
    tensorT,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsViewGroup,
)


# This file describes the translation of JIT schema to API's used
# when creating `ViewMeta` specializations that are used by the functionalization pass.
# These API's mostly follow the dispatcher API, with one difference:
# - While the forward function just directly calls into the at::_ops API
#   (following the dispatcher convention), the logic here for the reverse function
#   is responsible for generating both the call-site, and the declarations
#   (which are implemented manually in the at::functionalization::impl namespace).

# Define some specific lambda input arguments.
base_binding = Binding(
    name="base",
    nctype=NamedCType(name="base", type=ConstRefCType(BaseCType(tensorT))),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)

has_symbolic_inputs_binding = Binding(
    name="has_symbolic_inputs",
    nctype=NamedCType(name="has_symbolic_inputs", type=BaseCType(boolT)),
    argument=Argument(
        name="has_symbolic_inputs",
        type=BaseType(BaseTy.bool),
        default=None,
        annotation=None,
    ),
    default=None,
)
mutated_view_binding = Binding(
    name="mutated_view",
    nctype=NamedCType(name="mutated_view", type=ConstRefCType(BaseCType(tensorT))),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)
out_index_binding = Binding(
    name="out_index",
    nctype=NamedCType(name="out_index", type=BaseCType(longT)),
    argument=Argument(
        name="out_index", type=BaseType(BaseTy.int), default=None, annotation=None
    ),
    default=None,
)
reapply_views_binding = Binding(
    name="reapply_views",
    nctype=NamedCType(name="reapply_views", type=BaseCType(boolT)),
    argument=Argument(
        name="reapply_views", type=BaseType(BaseTy.bool), default=None, annotation=None
    ),
    default=None,
)

InverseReturnModeT = BaseCppType("at::functionalization", "InverseReturnMode")
inverse_return_mode_binding = Binding(
    name="inverse_return_mode",
    nctype=NamedCType(name="inverse_return_mode", type=BaseCType(InverseReturnModeT)),
    argument=Argument(
        name="inverse_return_mode",
        # NB: not actually a bool but it doesn't matter because this isn't used
        type=BaseType(BaseTy.bool),
        default=None,
        annotation=None,
    ),
    default=None,
)


# Name of the `ViewMeta` specialization class created.
def classname(func: FunctionSchema, with_namespace: bool = False) -> str:
    namespace = "at::functionalization::" if with_namespace else ""
    return f"{namespace}{func.name.unambiguous_name()}_ViewMeta"


# Name of the operation called inside the `forward`/`reverse` implementations.
def name(
    g: NativeFunctionsViewGroup,
    *,
    is_reverse: bool,
    include_namespace: bool,
    reapply_views: bool | None = None,
) -> str:
    if reapply_views is None:
        # reapply_views is only important for the fwd lambda,
        # since we always plumb the runtime "reapply_views" argument into the reverse function.
        assert is_reverse
    if is_reverse:
        return reverse_name(g.view, include_namespace)
    # in the forward case, we just directly call into the at::_ops API (so we always need the namespace)
    assert include_namespace
    assert g.view_copy is not None
    api_name = (
        g.view.func.name.unambiguous_name()
        if reapply_views
        else g.view_copy.func.name.unambiguous_name()
    )
    return f"at::_ops::{api_name}::call"


def reverse_name(f: NativeFunction, include_namespace: bool) -> str:
    # for the reverse: we plumb the "reapply_views" flag into that function and support
    # both copy and non-copy variants. (We could avoid doing that, but that would require
    # writing out twice as many view inverse functions).
    api_name = f.func.name.unambiguous_name()
    # in the reverse case, we codegen both the call-sites (which need the full namespace) and the declarations (which don't)
    if include_namespace:
        return f"at::functionalization::FunctionalInverses::{api_name}_inverse"
    else:
        return f"{api_name}_inverse"


def returns_type(func: FunctionSchema) -> CType:
    # Assertion: all view ops return tensor-like outputs
    assert len(func.returns) >= 1
    for ret in func.returns:
        assert ret.type.is_tensor_like()
    # However, the return type of the lambda is always an individual tensor.
    # For multi-tensor outputs, each tensor needs to be tracked individually.
    return BaseCType(tensorT)


# Checks whether `func` might return more than one value.
def is_multi_output(func: FunctionSchema) -> bool:
    return len(func.returns) > 1 or (
        len(func.returns) == 1 and func.returns[0].type.is_list_like() is not None
    )


# `ViewMeta` specialization constructor parameters.
def base_ctor_arguments(func: FunctionSchema) -> list[Binding]:
    # All specializations are parematerized by `has_symbolic_inputs` flag.
    arguments = [has_symbolic_inputs_binding]

    # If `func` might return more than 1 value, we also parameterize this specialization
    # with the output index.
    if is_multi_output(func):
        arguments.append(out_index_binding)

    return arguments


# `ViewMeta` specialized class' constructor arguments.
#
# Values needed specifically by this specialization, that the base class does not need.
# Same as the class' attributes, but non-owning.
def extra_ctor_arguments(func: FunctionSchema) -> list[Binding]:
    return attributes(func, owning=False)


# `ViewMeta` specialized class' non-static member data.
#
# Essential data for calling the instance's `forward` and `reverse functions. You can
# think of them as values that should be captured from the functionalization kernel.
def attributes(func: FunctionSchema, owning: bool = True) -> list[Binding]:
    args = func.arguments.flat_all
    assert args[0].type == BaseType(BaseTy.Tensor)
    return [
        reapply_views_binding,
        inverse_return_mode_binding,
        *[dispatcher.argument(a, remove_non_owning_ref_types=owning) for a in args[1:]],
    ]


def op_arguments(func: FunctionSchema, is_reverse: bool) -> list[Binding]:
    args = func.arguments.flat_all
    assert args[0].type == BaseType(BaseTy.Tensor)
    non_self_args = args[1:]
    # The forward lambda calls the at::_ops API, while the reverse lambda calls the view inverse API.
    # Both of these follow the dispatcher API.
    non_self_bindings = [dispatcher.argument(a) for a in non_self_args]
    if not is_reverse:
        # the forward lambda swaps out the original tensor argument with the lambd arg "base"
        return [base_binding] + non_self_bindings
    else:
        # the reverse lambda does the same, but with an additional "mutated_view" arg
        # additionally, we have a calling convention: for view ops that return multiple tensor outputs
        # their corresponding view_inverse function takes in an additional index argument.
        if is_multi_output(func):
            return [
                base_binding,
                mutated_view_binding,
                inverse_return_mode_binding,
                out_index_binding,
            ] + non_self_bindings
        else:
            return [
                base_binding,
                mutated_view_binding,
                inverse_return_mode_binding,
            ] + non_self_bindings

```



## High-Level Overview


This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `classname`, `name`, `reverse_name`, `returns_type`, `is_multi_output`, `base_ctor_arguments`, `extra_ctor_arguments`, `attributes`, `op_arguments`

**Key imports**: annotations, dispatcher


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/api`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `torchgen.api`: dispatcher


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

Files in the same folder (`torchgen/api`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ufunc.py_docs.md`](./ufunc.py_docs.md)
- [`meta.py_docs.md`](./meta.py_docs.md)
- [`autograd.py_docs.md`](./autograd.py_docs.md)
- [`structured.py_docs.md`](./structured.py_docs.md)
- [`python.py_docs.md`](./python.py_docs.md)
- [`unboxing.py_docs.md`](./unboxing.py_docs.md)
- [`translate.py_docs.md`](./translate.py_docs.md)
- [`native.py_docs.md`](./native.py_docs.md)


## Cross-References

- **File Documentation**: `functionalization.py_docs.md`
- **Keyword Index**: `functionalization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen/api`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torchgen/api`):

- [`meta.py_docs.md_docs.md`](./meta.py_docs.md_docs.md)
- [`translate.py_docs.md_docs.md`](./translate.py_docs.md_docs.md)
- [`unboxing.py_docs.md_docs.md`](./unboxing.py_docs.md_docs.md)
- [`ufunc.py_docs.md_docs.md`](./ufunc.py_docs.md_docs.md)
- [`cpp.py_docs.md_docs.md`](./cpp.py_docs.md_docs.md)
- [`dispatcher.py_docs.md_docs.md`](./dispatcher.py_docs.md_docs.md)
- [`autograd.py_kw.md_docs.md`](./autograd.py_kw.md_docs.md)
- [`translate.py_kw.md_docs.md`](./translate.py_kw.md_docs.md)
- [`native.py_kw.md_docs.md`](./native.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `functionalization.py_docs.md_docs.md`
- **Keyword Index**: `functionalization.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
