# Documentation: `torchgen/gen_vmap_plumbing.py`

## File Metadata

- **Path**: `torchgen/gen_vmap_plumbing.py`
- **Size**: 9,391 bytes (9.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    SchemaKind,
    Type,
)
from torchgen.utils import mapMaybe


if TYPE_CHECKING:
    from collections.abc import Sequence


def is_tensor(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.Tensor


def is_optional_tensor(typ: Type) -> bool:
    return isinstance(typ, OptionalType) and is_tensor(typ.elem)


def is_tensor_list(typ: Type) -> bool:
    return isinstance(typ, ListType) and is_tensor(typ.elem)


def unwrap_tensor(name: str, cur_level_var: str) -> list[str]:
    result = f"""\
    auto [{name}_value, {name}_bdim] = unwrapTensorAtLevel({name}, {cur_level_var});"""
    return textwrap.dedent(result).split("\n")


def unwrap_optional_tensor(name: str, cur_level_var: str) -> list[str]:
    result = f"""\
    std::optional<Tensor> {name}_value;
    std::optional<int64_t> {name}_bdim;
    if ({name}) {{
        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), {cur_level_var});
    }}"""
    return textwrap.dedent(result).split("\n")


def gen_unwraps(
    flat_arguments: Sequence[Argument], cur_level_var: str
) -> tuple[str, list[str]]:
    arg_names = [a.name for a in flat_arguments]
    arg_types = [a.type for a in flat_arguments]

    tensors = [name for typ, name in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [
        name for typ, name in zip(arg_types, arg_names) if is_optional_tensor(typ)
    ]

    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor, cur_level_var)

    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor, cur_level_var)
    unwrap_code = "\n".join(unwraps)

    unwrapped_arg_list = []
    for arg in arg_names:
        if arg in tensors or arg in optional_tensors:
            unwrapped_arg_list += [f"{arg}_value", f"{arg}_bdim"]
        else:
            unwrapped_arg_list.append(arg)
    return unwrap_code, unwrapped_arg_list


def gen_case_where_all_bdims_are_none(
    outer_sig: DispatcherSignature, schema: FunctionSchema, cur_level_var: str
) -> str:
    conditions = []
    flat_args = schema.arguments.flat_all
    for arg in flat_args:
        if not arg.type.is_tensor_like():
            continue
        conditions.append(f"!isBatchedAtLevel({arg.name}, {cur_level_var})")

    sig = DispatcherSignature.from_schema(schema)
    translated_args = ", ".join(
        e.expr for e in translate(outer_sig.arguments(), sig.arguments())
    )
    return f"""\
if ({" && ".join(conditions)}) {{
  return at::_ops::{sig.func.name.unambiguous_name()}::call({translated_args});
}}"""


def gen_returns(
    returns: tuple[Return, ...], cur_level_var: str, results_var: str
) -> str:
    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret.type):
            wrapped_returns.append(
                f"makeBatched(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})"
            )
            idx += 2
        elif is_tensor_list(ret.type):
            wrapped_returns.append(
                f"makeBatchedVector(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})"
            )
            idx += 2
        else:
            wrapped_returns.append(f"std::get<{idx}>({results_var})")
            idx += 1
    if len(wrapped_returns) == 1:
        result = f"return {wrapped_returns[0]};"
    else:
        result = f"return std::make_tuple({', '.join(wrapped_returns)});"
    return result


def accepts_at_least_one_tensor_input(schema: FunctionSchema) -> bool:
    return any(a.type.is_tensor_like() for a in schema.arguments.flat_all)


def is_mutated_arg(argument: Argument) -> bool:
    return argument.annotation is not None and argument.annotation.is_write


def gen_vmap_inplace_plumbing(native_function: NativeFunction) -> str | None:
    # Assumptions:
    # - only one argument is being modified in-place
    # - the argument that is being modified in-place is the first argument
    # - all returns are either Tensor, tuple of Tensor, or TensorList
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    # Check assumptions. If these are invalid we return None
    # and punt the work to handle them to the future.
    assert schema.kind() == SchemaKind.inplace
    if not is_mutated_arg(schema.arguments.flat_all[0]):
        return None
    if len([arg for arg in schema.arguments.flat_all if is_mutated_arg(arg)]) != 1:
        return None

    # Only support cases where all returns are Tensors or vector<Tensor>
    if len(returns) == 0:
        return None
    if not all(is_tensor(ret.type) or is_tensor_list(ret.type) for ret in returns):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None

    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)

    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + "_generated_plumbing")} {{
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  batch_rule({", ".join(unwrapped_arg_list)});
  return {schema.arguments.flat_all[0].name};
}}"""


def gen_vmap_plumbing_no_returns(native_function: NativeFunction) -> str:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)

    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + "_generated_plumbing")} {{
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  batch_rule({", ".join(unwrapped_arg_list)});
}}"""


def gen_vmap_plumbing(native_function: NativeFunction) -> str | None:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    # Only support cases where all returns are Tensors or vector<Tensor>
    if not accepts_at_least_one_tensor_input(schema):
        return None
    if len(returns) == 0:
        return gen_vmap_plumbing_no_returns(native_function)
    return_symint_overrides = [
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
    ]
    if (
        not all(ret.type.is_tensor_like() for ret in returns)
        and schema.name.unambiguous_name() not in return_symint_overrides
    ):
        return None
    # in-place views need special handling
    if "inplace_view" in native_function.tags:
        return None

    if schema.kind() == SchemaKind.inplace:
        return gen_vmap_inplace_plumbing(native_function)

    # Don't support these (mutable, out, scratch)
    if schema.kind() != SchemaKind.functional:
        return None

    results_var = "results"
    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)

    wrapped_returns = gen_returns(returns, cur_level_var, results_var)
    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + "_generated_plumbing")} {{
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  auto {results_var} = batch_rule({", ".join(unwrapped_arg_list)});
  {wrapped_returns}
}}"""


@dataclass(frozen=True)
class ComputeBatchRulePlumbing:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        result = gen_vmap_plumbing(f)
        return result


def gen_all_vmap_plumbing(native_functions: Sequence[NativeFunction]) -> str:
    body = "\n".join(list(mapMaybe(ComputeBatchRulePlumbing(), native_functions)))
    return f"""
#pragma once
#include <ATen/Operators.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at {{ namespace functorch {{

{body}

}}}} // namespace at::functorch
"""

```



## High-Level Overview

result = f"""\

This Python file contains 2 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ComputeBatchRulePlumbing`

**Functions defined**: `is_tensor`, `is_optional_tensor`, `is_tensor_list`, `unwrap_tensor`, `unwrap_optional_tensor`, `gen_unwraps`, `gen_case_where_all_bdims_are_none`, `gen_returns`, `accepts_at_least_one_tensor_input`, `is_mutated_arg`, `gen_vmap_inplace_plumbing`, `gen_vmap_plumbing_no_returns`, `gen_vmap_plumbing`, `__call__`, `gen_all_vmap_plumbing`

**Key imports**: annotations, textwrap, dataclass, TYPE_CHECKING, translate, DispatcherSignature, method_with_native_function, mapMaybe, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `textwrap`
- `dataclasses`: dataclass
- `typing`: TYPE_CHECKING
- `torchgen.api.translate`: translate
- `torchgen.api.types`: DispatcherSignature
- `torchgen.context`: method_with_native_function
- `torchgen.utils`: mapMaybe
- `collections.abc`: Sequence


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

Files in the same folder (`torchgen`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`gen_backend_stubs.py_docs.md`](./gen_backend_stubs.py_docs.md)
- [`local.py_docs.md`](./local.py_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`model.py_docs.md`](./model.py_docs.md)
- [`yaml_utils.py_docs.md`](./yaml_utils.py_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`gen_schema_utils.py_docs.md`](./gen_schema_utils.py_docs.md)
- [`gen.py_docs.md`](./gen.py_docs.md)


## Cross-References

- **File Documentation**: `gen_vmap_plumbing.py_docs.md`
- **Keyword Index**: `gen_vmap_plumbing.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
