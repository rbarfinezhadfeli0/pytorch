# Documentation: `torchgen/gen.py`

## File Metadata

- **Path**: `torchgen/gen.py`
- **Size**: 114,318 bytes (111.64 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import argparse
import functools
import json
import keyword
import os
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never

import yaml

import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
    Binding,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    NamedCType,
    NativeSignature,
    SpecialArgName,
)
from torchgen.context import (
    method_with_native_function,
    native_function_manager,
    with_native_function,
    with_native_function_and_indices,
)
from torchgen.gen_aoti_c_shim import (
    gen_aoti_c_shim_files,
    gen_static_dispatch_backend_call_signature,
)
from torchgen.gen_functionalization_type import (
    gen_functionalization_definition,
    gen_functionalization_registration,
    gen_functionalization_view_inverse_declaration,
    gen_functionalization_view_meta_classes_decl,
    gen_functionalization_view_meta_classes_impl,
    GenCompositeViewCopyKernel,
)
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseOperatorName,
    DEFAULT_KERNEL_NAMESPACE,
    dispatch_device_map,
    DispatchKey,
    FRAGMENT_NAMESPACES,
    FunctionSchema,
    is_cuda_dispatch_key,
    is_generic_dispatch_key,
    is_ufunc_dispatch_key,
    is_xpu_dispatch_key,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    OperatorName,
    OptionalType,
    SchemaKind,
    SelfArgument,
    STRUCTURED_DISPATCH_KEYS,
    TensorOptionsArguments,
    Type,
    Variant,
    ViewSchemaKind,
)
from torchgen.native_function_generation import (
    add_generated_native_functions,
    gen_composite_functional_kernel,
    gen_composite_out_kernel,
    pre_group_native_functions,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
    concatMap,
    context,
    FileManager,
    make_file_manager,
    mapMaybe,
    NamespaceHelper,
    Target,
)
from torchgen.yaml_utils import YamlDumper, YamlLoader


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Optional


T = TypeVar("T")

# Welcome to the ATen code generator v2!  The ATen code generator is
# responsible for parsing native_functions.yaml and then generating
# various generated files (e.g., TypeDefault.cpp) based on the operators
# defined in this file.  This means that the code generator knows how to
# parse function schema, and then translate this into various C++ types
# and boilerplate code.
#
# Some things to know about this file when you modify it:
#
# - This file has STRICT mypy typechecking.  Typecheck it with
#   `mypy --config mypy-strict.ini` in the root source directory
#
# - Most of the heavy lifting lives in external modules:
#   - 'model' has the data model for native_functions.yaml.  The classes
#     in those file represent what you see when you look at
#     a native_functions.yaml
#   - 'api' has conversions for how to translate JIT schema into
#     the various C++ APIs that the codegen interacts with.  There
#     are in fact THREE different C++ APIs: the public C++ API,
#     the dispatcher API, and the legacy dispatcher API.  See each
#     of these respective files for more information

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         HELPER FUNCTIONS
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# A custom loader for YAML to let us also keep track of line numbers
# of each entry in the YAML file
class LineLoader(YamlLoader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        # Add 1 so line numbering starts at 1
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple("ParsedYaml", ["native_functions", "backend_indices"])


_GLOBAL_PARSE_NATIVE_YAML_CACHE: dict[str, ParsedYaml] = {}
_GLOBAL_PARSE_TAGS_YAML_CACHE: dict[str, set[str]] = {}


def file_manager_from_dispatch_key(
    dispatch_key: DispatchKey,
    device_fms: dict[str, FileManager],
    default_fm: FileManager,
) -> FileManager:
    fm = device_fms.get(
        next(
            (
                device
                for check, device in dispatch_device_map.items()
                if check(dispatch_key)
            ),
            "",
        ),
        default_fm,
    )
    return fm


def parse_native_yaml_struct(
    es: object,
    valid_tags: set[str],
    ignore_keys: set[DispatchKey] | None = None,
    path: str = "<stdin>",
    skip_native_fns_gen: bool = False,
) -> ParsedYaml:
    assert isinstance(es, list)
    rs: list[NativeFunction] = []
    bs: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    for e in es:
        assert isinstance(e, dict), f"expected to be dict: {e}"
        assert isinstance(e.get("__line__"), int), e
        loc = Location(path, e["__line__"])
        funcs = e.get("func")
        assert funcs is not None, f"missed 'func' in {e}"
        with context(lambda: f"in {loc}:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
            rs.append(func)
            BackendIndex.grow_index(bs, m)
    error_check_native_functions(rs)
    # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
    indices: dict[DispatchKey, BackendIndex] = defaultdict(
        lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined,
            use_out_as_primary=True,
            external=False,
            device_guard=False,
            # I'm actually not sure about this; undefined could be hit on
            # empty TensorList, hypothetically that could have sizes in it
            index={},
        )
    )
    if not skip_native_fns_gen:
        add_generated_native_functions(rs, bs)
    for k, v in bs.items():
        # All structured in-tree operators are implemented in terms of their out operator.
        indices[k] = BackendIndex(
            dispatch_key=k,
            use_out_as_primary=True,
            external=False,
            # Only cuda-like devices in tree require device guards
            device_guard=is_cuda_dispatch_key(k) or is_xpu_dispatch_key(k),
            index=v,
        )
    return ParsedYaml(rs, indices)


def parse_tags_yaml_struct(es: object, path: str = "<stdin>") -> set[str]:
    assert isinstance(es, list)
    rs: set[str] = set()
    for e in es:
        assert isinstance(e.get("__line__"), int), e
        loc = Location(path, e["__line__"])
        tags = e.get("tag")
        with context(lambda: f"in {loc}:\n  {tags}"):
            e_i = e.copy()
            name = e_i.pop("tag")
            desc = e_i.pop("desc", "")
            # ensure that each tag has a non-empty description
            assert desc != ""
            rs.add(name)
    return rs


@functools.cache
def parse_tags_yaml(path: str) -> set[str]:
    global _GLOBAL_PARSE_TAGS_YAML_CACHE
    if path not in _GLOBAL_PARSE_TAGS_YAML_CACHE:
        with open(path) as f:
            es = yaml.load(f, Loader=LineLoader)
            _GLOBAL_PARSE_TAGS_YAML_CACHE[path] = parse_tags_yaml_struct(es, path=path)

    return _GLOBAL_PARSE_TAGS_YAML_CACHE[path]


def parse_native_yaml(
    path: str,
    tags_yaml_path: str,
    ignore_keys: set[DispatchKey] | None = None,
    *,
    skip_native_fns_gen: bool = False,
    loaded_yaml: object | None = None,
) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tags_yaml_path)

        # if a loaded yaml is provided, use that instead of reading from path
        if loaded_yaml is None:
            with open(path) as f:
                es = yaml.load(f, Loader=LineLoader)
        else:
            es = loaded_yaml

        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(
            es,
            valid_tags,
            ignore_keys,
            path=path,
            skip_native_fns_gen=skip_native_fns_gen,
        )

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]


# Some assertions are already performed during parsing, but those are only within a single NativeFunction.
# Assertions here are meant to be performed across NativeFunctions.
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    func_map: dict[OperatorName, NativeFunction] = {}
    base_func_map: dict[BaseOperatorName, list[NativeFunction]] = defaultdict(list)
    for f in funcs:
        func_map[f.func.name] = f
        base_func_map[f.func.name.name].append(f)
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map.get(f.structured_delegate)
            assert delegate_func is not None, (
                f"{f.func.name} is marked as a structured_delegate pointing to "
                f"{f.structured_delegate}, but {f.structured_delegate} is missing."
            )
            assert delegate_func.structured, (
                f"{f.func.name} is marked as a structured_delegate pointing to "
                f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. "
                f"Consider adding 'structured=True' to the delegated operator"
            )

        # Check for reserved Python keywords
        PYTHON_RESERVED_KEYWORDS = set(keyword.kwlist)
        # List of pre-existing operators that are known to have reserved keywords
        # Exclusion list is used to suppress the assertion for these operators
        EXCLUSION_LIST = {
            ("_has_compatible_shallow_copy_type", "from"),
            ("random_.from", "from"),
            ("uniform_", "from"),
        }

        for arg in f.func.arguments.flat_all:
            if arg.name in PYTHON_RESERVED_KEYWORDS:
                if (str(f.func.name), arg.name) not in EXCLUSION_LIST:
                    raise AssertionError(
                        f"Argument name '{arg.name}' in function '{f.func.name}' is a reserved Python keyword."
                    )
        # See Note [resize_ in Functionalization]
        # resize_() is technically an inplace view op (and therefore needs the tag),
        # but it would be overkill to add a true "view" variant of resize.
        # Instead, resize_() gets special treatment in functionalization,
        # and we have a resize() op that is non-aliasing + functional.
        if (
            "inplace_view" in f.tags
            and str(f.func.name) != "resize_"
            and str(f.func.name) != "resize_as_"
            and str(f.func.name.name) != "set_"
        ):
            base_name = f.func.name.name
            assert base_name.inplace, (
                f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming "
                "convention for inplace ops - the codegen expects the base name to have a trailing underscore. "
            )
            out_of_place_base_name = BaseOperatorName(
                base_name.base, False, base_name.dunder_method
            )
            assert len(base_func_map[out_of_place_base_name]) > 0, (
                f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding "
                f"out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one. "
            )


def cpp_string(s: str) -> str:
    """Convert a python string into a c++ string literal"""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\a", "\\a")
    s = s.replace("\b", "\\b")
    s = s.replace("\f", "\\f")
    s = s.replace("\n", "\\n")
    s = s.replace("\v", "\\v")
    s = s.replace("\t", "\\t")
    return f'"{s}"'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Most functions in this section are curried: they consist of a function
# that takes some parameters (e.g., what is to be generated) which itself
# returns a function that actually maps NativeFunction to the code
# to be generated.  This pattern makes it convenient to use map, concatMap
# and similar functional combinators.


def static_dispatch_keys(backends: list[BackendIndex]) -> list[DispatchKey]:
    if len(backends) == 0:
        return []
    else:
        return [backend.dispatch_key for backend in backends] + [
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.CompositeImplicitAutogradNestedTensor,
            DispatchKey.CompositeExplicitAutograd,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        ]


def get_static_dispatch_backend(
    f: NativeFunction, backend_index: BackendIndex
) -> DispatchKey | None:
    if f.structured_delegate is not None or backend_index.has_kernel(f):
        # TODO: for ops with structured_delegate it should check the dispatch table of
        # the out variant instead. For now, these structured ops all have CPU/CUDA kernels
        # so we always dispatch to the `backend`, but this could be wrong when we
        # migrate math/default_backend ops to use structured delegate.
        return backend_index.dispatch_key
    elif f.has_composite_explicit_autograd_kernel:
        return DispatchKey.CompositeExplicitAutograd
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return DispatchKey.CompositeExplicitAutogradNonFunctional
    elif f.has_composite_implicit_autograd_kernel:
        return DispatchKey.CompositeImplicitAutograd
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return DispatchKey.CompositeImplicitAutogradNestedTensor
    return None


def static_dispatch_ops_header(
    f: NativeFunction, backend_index: list[BackendIndex]
) -> str | None:
    if backend_index is None or f.manual_kernel_registration:
        return None

    output = []
    for index in backend_index:
        dispatch_key = get_static_dispatch_backend(f, index)
        if dispatch_key is not None:
            output.append(
                f"#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>"
            )
    return "\n".join(output)


def static_dispatch_extra_headers(backends: list[BackendIndex]) -> list[str]:
    return [
        f"#include <ATen/{dispatch_key}Functions.h>"
        for dispatch_key in static_dispatch_keys(backends)
    ]


# Translates arguments of `sig` to CppSignature bindings.
# Note that we have a special case for `memory_format` argument and this case is not covered by
# tools.codegen.api.translate() yet as its application is limited to static dispatch.
def translate_args(
    sig: CppSignature | DispatcherSignature,
    cpp_sig: CppSignature,
) -> str:
    # Adds SpecialArgName.possibly_redundant_memory_format NamedCType for memory_format bindings
    def add_spl_memory_format_binding(input_bindings: list[Binding]) -> list[Binding]:
        output_bindings: list[Binding] = []
        for binding in input_bindings:
            if binding.name == "memory_format":
                spl_mem_format_binding = Binding(
                    nctype=NamedCType(
                        SpecialArgName.possibly_redundant_memory_format,
                        binding.nctype.type,
                    ),
                    name=binding.name,
                    default=binding.default,
                    argument=binding.argument,
                )
                output_bindings.append(spl_mem_format_binding)
            else:
                output_bindings.append(binding)
        return output_bindings

    src_bindings = list(sig.arguments())
    goal_bindings = list(cpp_sig.arguments())
    # When last argument of CPP signature has SpecialArgName.possibly_redundant_memory_format NCType,
    # get memory_format bindings of dispatcher signature to have the same NCType as well
    for arg in goal_bindings:
        if arg.nctype.name == SpecialArgName.possibly_redundant_memory_format:
            src_bindings = add_spl_memory_format_binding(src_bindings)
            break
    exprs = translate(src_bindings, goal_bindings)
    return ", ".join(a.expr for a in exprs)


def generate_static_dispatch_backend_call(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_index: BackendIndex,
) -> str:
    cpp_sig = gen_static_dispatch_backend_call_signature(sig, f)
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    backend_metadata = backend_index.get_kernel(f)
    kernel_ns = (
        backend_metadata.cpp_namespace
        if backend_metadata and backend_metadata.cpp_namespace
        else DEFAULT_KERNEL_NAMESPACE
    )
    ns = kernel_ns.replace("::native", "")
    return f"return {ns}::{backend_index.dispatch_key.lower()}::{name}({exprs});"


def generate_static_dispatch_fallback_call(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_indices: list[BackendIndex],
) -> str:
    cpp_sigs = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    ns = DEFAULT_KERNEL_NAMESPACE.replace("::native", "")
    if f.has_composite_explicit_autograd_kernel:
        return f"return {ns}::{DispatchKey.CompositeExplicitAutograd.lower()}::{name}({exprs});"
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return f"return {ns}::{DispatchKey.CompositeExplicitAutogradNonFunctional.lower()}::{name}({exprs});"
    elif f.has_composite_implicit_autograd_kernel:
        return f"return {ns}::{DispatchKey.CompositeImplicitAutograd.lower()}::{name}({exprs});"
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return f"return {ns}::{DispatchKey.CompositeImplicitAutogradNestedTensor.lower()}::{name}({exprs});"
    else:
        return f"""TORCH_CHECK(false, "Static dispatch does not support {name} for\
{", ".join([str(index.dispatch_key) for index in backend_indices])} ");"""


def static_dispatch(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_indices: list[BackendIndex],
) -> str:
    """
    For a given `NativeFunction`, find out the corresponding backend and dispatch to it. If more than one
    backends exist, fallback to static dispatch by determining dispatch key from inputs.
    Arguments:
        sig: A CppSignature or DispatcherSignature for this native function we want to use.
        f: NativeFunction to generate static dispatch.
        backend_indices: All available backends.
    Return:
        C++ code to call backend-specific functions, e.g., "return at::cpu::add(self, other, scale);"
    """
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ""

    keys = [
        b
        for b in backend_indices
        if b.has_kernel(f)
        or (
            f.structured_delegate is not None
            and b.dispatch_key in STRUCTURED_DISPATCH_KEYS
        )
    ]
    if len(keys) == 1:
        return generate_static_dispatch_backend_call(sig, f, keys[0])
    elif len(keys) == 0:
        return generate_static_dispatch_fallback_call(sig, f, backend_indices)

    native_tensor_args = [
        a.name
        for a in sig.arguments()
        if isinstance(a.argument, SelfArgument)
        or isinstance(a.argument, Argument)
        and a.argument.type.is_tensor_like()
    ]
    tensor_args = ", ".join(native_tensor_args)
    tensor_opts = f.func.arguments.tensor_options

    stmts = []
    subexprs: list[str] = []
    if tensor_opts is not None:
        subexprs.append(
            "DispatchKeySet(c10::computeDispatchKey(dtype, layout, device))"
        )
    if tensor_args != "":
        subexprs.append(f"c10::detail::multi_dispatch_key_set({tensor_args})")
    stmts.append(f"""DispatchKeySet _dk_set = {" | ".join(subexprs)};""")
    stmts.append("DispatchKey _dk = c10::highestPriorityBackendTypeId(_dk_set);")

    dispatch_code = []
    for index in keys:
        dispatch_code.append(f"""case DispatchKey::{index.dispatch_key}:""")
        dispatch_code.append(
            f"""\t{generate_static_dispatch_backend_call(sig, f, index)};"""
        )

    fallback = generate_static_dispatch_fallback_call(sig, f, backend_indices)
    connector = "\n\t\t"

    return f"""
    {connector.join(stmts)}
    switch (_dk) {{
        {connector.join(dispatch_code)}
        default:
            {fallback}
    }}
    """


# Generates RegisterSchema.cpp.  Depending on the selector, either
# all schemas are registered, or only some are (in the case of
# selective build)
@dataclass(frozen=True)
class RegisterSchema:
    selector: SelectiveBuilder
    known_tags: dict[str, int] = field(default_factory=dict)

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if not self.selector.is_native_function_selected(f):
            return None
        tags = "{" + ", ".join(f"at::Tag::{tag}" for tag in sorted(f.tags)) + "}"
        if tags == "{}":
            return f"m.def({cpp_string(str(f.func))}, {{}});\n"
        maybe_tags = ""
        if tags not in self.known_tags:
            idx = len(self.known_tags)
            self.known_tags[tags] = idx
            maybe_tags = f"const std::vector<at::Tag> tags_{idx} = {tags};\n"
        return f"{maybe_tags}m.def({cpp_string(str(f.func))}, tags_{self.known_tags[tags]});\n"


# Generates Operators.h and Operators.cpp.
# These provide macros that, given an operator and overload name, allow users
# to access an "un-overloaded" function version of the operator. This
# is useful for extension writers who want to (1) want to decltype the operator
# and (2) don't want to worry about method-only operators.
@dataclass(frozen=True)
class ComputeOperators:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: list[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig = DispatcherSignature.from_schema(f.func)
        name = f.func.name.unambiguous_name()

        if self.target is Target.DECLARATION:
            # Note [The ATen Operators API]
            # The ATen Operators API lives in the at::_ops namespace, and contains compile-time
            # metadata about each operator + entry points into the Dispatcher.
            # The C++ function, method, and redispatch API's are all implemented as wrappers
            # into various bits of the structs defined here.
            #
            # Important characteristics about the Operators API:
            # (1) It follows the Dispatcher API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Overload names are disambiguated.
            #     This is helpful for pytorch extenders who would like to decltype() an aten operator,
            #     that has overloads, e.g. decltype(at::_ops::mul_Tensor::call)
            # (3) No argument defaulting is allowed.
            #     This is more of an implementation detail to avoid #include cycles,
            #     since TensorBody.h (which defines the Tensor class) needs to include this file.
            # (4) manual_cpp_bindings and faithful names are not included in the API.
            #     This applies to stuff like __dispatch__is_complex(), and add_outf().
            #     These aren't "real aten ops", they're just additional functions provided by the C++ API.
            #     They're implemented as wrappers in Functions.h that call into the actual operators
            #     defined here, i.e. at::_ops::is_complex::call() and at::_ops::add_out::call().
            #     This means that ATEN_OP(is_complex) will not fastpath, and will go through the dispatcher.
            return f"""
struct TORCH_API {name} {{
  using schema = {sig.type()};
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::{f.func.name.name}";
  static constexpr const char* overload_name = "{f.func.name.overload_name}";
  static constexpr const char* schema_str = {cpp_string(str(f.func))};
  static {sig.defn(name="call", is_redispatching_fn=False)};
  static {sig.defn(name="redispatch", is_redispatching_fn=True)};
}};"""

        elif self.target is Target.DEFINITION:
            defns = f"""
// aten::{f.func}
static C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow({name}::name, {name}::overload_name)
      .typed<{name}::schema>();
}}
"""
            for is_redispatching_fn in [False, True]:
                if is_redispatching_fn:
                    dispatcher_exprs_str = ", ".join(
                        ["dispatchKeySet"] + [a.name for a in sig.arguments()]
                    )
                    method_base = "redispatch"
                else:
                    dispatcher_exprs_str = ", ".join([a.name for a in sig.arguments()])
                    method_base = "call"

                dispatcher_call = method_base
                method_name = f"{name}::{method_base}"

                fn_body = f"""
    static auto op = create_{name}_typed_handle();
    return op.{dispatcher_call}({dispatcher_exprs_str});"""

                if (
                    not is_redispatching_fn
                    and len(self.static_dispatch_backend_indices) > 0
                ):
                    # call() should go through static dispatch
                    fn_body = static_dispatch(
                        sig, f, backend_indices=self.static_dispatch_backend_indices
                    )
                defns += f"""
// aten::{f.func}
{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{
    {fn_body}
}}
"""
            return defns
        else:
            assert_never(self.target)


# Generates Functions.h, which provides the functional public C++ API,
# and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeFunction:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )
        has_symint = f.func.has_symint()

        result = ""
        for sig in sig_group.signatures():
            # See Note [The ATen Operators API]
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ", ".join([e.expr for e in exprs])

            if sig.symint:
                intlike_t = "c10::SymInt"
            else:
                intlike_t = "int64_t"

            if Variant.function in f.variants:
                result += f"""
// aten::{f.func}
inline {sig.decl()} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}"""

            # The template function can be used from template situations
            # where you want to switch between the symint or not version
            # depending on a template argument
            #
            # NB: we ALWAYS generate this even for methods.  But we put it in
            # this header so it can take advantage of per-op headers
            if has_symint:
                result += f"""
namespace symint {{
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, {intlike_t}>>>
  {sig.decl(suppress_symint_suffix=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
  }}
}}
"""
        return result


# Generates TensorBody.h. This file provides the object-oriented (method-based)
# public C++ API, and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: list[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert f.func.arguments.self_arg is not None

        sig_group = CppSignatureGroup.from_native_function(
            f, method=True, fallback_binding=f.manual_cpp_binding
        )

        if self.target is Target.DECLARATION:
            result = ""
            for sig in sig_group.signatures():
                result += f"{sig.decl()} const;\n"
            return result

        if self.target is not Target.DEFINITION:
            assert_never(self.target)

        result = ""

        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ", ".join([e.expr for e in exprs])

            result += f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""

        return result


# Generates RedispatchFunctions.h.
# This is similar to the C++ API defined in Functions.h, but provides access
# to the dispatcher's redispatch API.
@dataclass(frozen=True)
class ComputeRedispatchFunction:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        # We unconditionally generate function variants of the redispatch API.
        # This is mainly because we can namespace functions separately, but not methods,
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )

        result = ""
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ", ".join(["dispatchKeySet"] + [a.expr for a in exprs])

            result += f"""
// aten::{f.func}
inline {sig.decl(is_redispatching_fn=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::redispatch({exprs_str});
}}
"""

        return result


# Generates ATenOpList.cpp, a runtime accessible list of all aten
# operators.
# TODO: This was historically used to help some JIT interop code
# figure out whether or not to treat aten namespace'd operators
# one way or another, we should reevaluate if this is actually needed.
@with_native_function
def compute_aten_op(f: NativeFunction) -> str:
    return f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},'


# Generates MetaFunctions.h
def compute_meta_function_declaration(g: NativeFunctionsGroup) -> str | None:
    if not g.structured:
        return None
    with native_function_manager(g.out):
        name = meta.name(g)
        args = structured.meta_arguments(g)
        args_str = ", ".join(a.decl() for a in args)
        parent_class = g.out.structured_inherits
        if parent_class is None:
            parent_class = "at::impl::MetaBase"
        meta_return = "void"
        precomputed = g.out.precomputed if g.structured else None

        if precomputed:
            # Generate the template declaration with one bool parameter for each
            # precomputed element. Each parameter is true if the corresponding (in
            # terms of position) precomputed element has been set.
            precomputed_values = [*precomputed.replace.values(), precomputed.add]
            precomputed_elements = [
                elem for replace_list in precomputed_values for elem in replace_list
            ]
            precomputed_template_parameters = [
                elem.name.upper() for elem in precomputed_elements
            ]
            precomputed_template_params_str = ", ".join(
                f"bool {param} = false" for param in precomputed_template_parameters
            )
            precompute_template_decl = f"template <{precomputed_template_params_str}>"

            # Generate a string containing declarations of all precomputed elements.
            precomputed_elements_with_cpp_types = [
                structured.argument_type(elem, binds=elem.name)
                for elem in precomputed_elements
            ]

            precomputed_elements_decl = ";\n".join(
                f"{elem.cpp_type(strip_ref=True)} {elem.name}"
                for elem in precomputed_elements_with_cpp_types
            )

            # Generate "setter" methods for each precomputed element. Each method will return
            # a new instance of precompute_out with the template parameter that corresponds to
            # the member set by the method to true (to indicate that it has been set).
            setter_methods = []
            for i, elem in enumerate(precomputed_elements):
                # Generate the signature. The return type will be the same
                # as the type of `this` but with the template parameter
                # corresponding to the element set by this method set to true.
                # The assert generated below will ensure that this template
                # parameter is false on the type of `this`.
                return_ty_templates = ", ".join(
                    precomputed_template_parameters[:i]
                    + ["true"]
                    + precomputed_template_parameters[i + 1 :]
                )
                return_ty = f"precompute_out<{return_ty_templates}>"
                elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(
                    strip_ref=True
                )
                signature = f"{return_ty} set_{elem.name}({elem_cpp_ty} value)"

                # Generate an assert which checks that the
                # template parameter corresponding to the precomputed
                # element that is set by this method is false on the
                # class corresponding to the object that `this` points to.
                # This ensures that each element can be set only once.
                assert_msg = f'"{elem.name} already set"'
                assert_stmt = f"static_assert({precomputed_template_parameters[i]} == false, {assert_msg});"

                # Generate the new object construction block. All state
                # except the element that this method sets is copied from the
                # object that `this` points to. The value for the element that
                # the method sets is taken from a method parameter.
                construction_stmts = []
                construction_stmts.append(f"{return_ty} ret;")

                for j, elem in enumerate(precomputed_elements):
                    if i == j:
                        construction_stmts.append(f"ret.{elem.name} = value;")
                    else:
                        construction_stmts.append(
                            f"ret.{elem.name} = this->{elem.name};"
                        )

                construction_stmts.append("return ret;")
                construction_block = "\n".join(construction_stmts)

                setter_methods.append(
                    f"""
                    {signature} {{
                        {assert_stmt}
                        {construction_block}
                    }}
                """
                )
            setter_methods_decl = "\n".join(setter_methods)

            # Meta should return an instance of the struct containing the precomputed elements.
            meta_return_template_params = ", ".join(
                ["true"] * len(precomputed_template_parameters)
            )
            # This typedef (actually a using statement) is needed so that TORCH_META_FUNC can reuse the return
            # type (which has a variable number of template parameters).
            meta_return_typedef = f"using meta_return_ty = precompute_out <{meta_return_template_params}>;"
            meta_return = "meta_return_ty"
            precomputed_decl = f"""
                {precompute_template_decl}
                struct TORCH_API precompute_out {{
                    {setter_methods_decl}
                    {precomputed_elements_decl};
            }};"""
        else:
            meta_return_typedef = ""
            precomputed_decl = ""

        return f"""\
struct TORCH_API structured_{name} : public {parent_class} {{
    {precomputed_decl}
    {meta_return_typedef}
    {meta_return} meta({args_str});
}};
"""


def needs_backend_select(f: NativeFunction, selector: SelectiveBuilder) -> bool:
    name = str(f.func.name.name)
    if name.endswith("_like") or name.startswith("new_"):
        return False
    if f.func.arguments.tensor_options is None:
        return False
    return selector.is_native_function_selected(f)


# Generates RegisterBackendSelect.cpp, a series of kernels which provide
# specialized computation of dispatch key for operator signatures which cannot
# be easily done automatically using templating.
@dataclass(frozen=True)
class ComputeBackendSelect:
    target: Literal[Target.DEFINITION, Target.REGISTRATION]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if not needs_backend_select(f, self.selector):
            return None

        name = native.name(f.func)
        # BackendSelect can go to Meta, so it must preserve symints
        native_sig = NativeSignature(f.func, symint=True)

        native_tensor_args = [
            a
            for a in native_sig.arguments()
            if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
        ]

        dispatcher_sig = DispatcherSignature.from_schema(f.func)

        sig: NativeSignature | DispatcherSignature
        sig = dispatcher_sig
        dispatcher_exprs = dispatcher_sig.exprs()
        dispatch_key = "c10::computeDispatchKey(dtype, layout, device)"

        if self.target is Target.DEFINITION:
            # I don't think there's actually a good reason to generate
            # these two cases differently
            # The first case could probably be improved though- it calls computeDispatchKeySet(),
            # which looks at TLS dispatch keys- there should not be any by the time we reach backend select.
            if native_tensor_args:
                assert f.func.arguments.has_tensor_arg()
                tensor_args = ", ".join(a.name for a in native_tensor_args)
                compute_dk = f"""\
DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});
DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
DispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);"""
            else:
                assert not f.func.arguments.has_tensor_arg()
                compute_dk = (
                    f"DispatchKeySet _dk = c10::DispatchKeySet({dispatch_key});"
                )
            return f"""\
// aten::{f.func}
C10_ALWAYS_INLINE
{sig.defn(name)} {{
  {compute_dk}
  return at::_ops::{f.func.name.unambiguous_name()}::redispatch(
      _dk, {", ".join(a.expr for a in dispatcher_exprs)});
}}
"""
        elif self.target is Target.REGISTRATION:
            return f"""m.impl("aten::{f.func.name}", TORCH_FN({name}));"""
        else:
            assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       YAML CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def format_yaml(data: object) -> str:
    # Ignore alias in Dumper
    YamlDumper.ignore_aliases = lambda self, data: True  # type: ignore[assignment]

    # Support serializing OrderedDict
    def dict_representer(dumper: Any, data: Any) -> Any:
        return dumper.represent_dict(data.items())

    YamlDumper.add_representer(OrderedDict, dict_representer)  # type: ignore[no-untyped-call]
    # Some yaml parsers (e.g. Haskell's) don't understand line breaks.
    # width=1e9 turns off optional line breaks and improves
    # the portability of the outputted yaml.
    return yaml.dump(data, default_flow_style=False, Dumper=YamlDumper, width=1e9)  # type: ignore[no-any-return, call-overload]


# For some reason, some defaults we write to YAML are written as native
# YAML objects, rather than doing them uniformly as strings.  This
# function detects those cases and converts them into native Python
# objects.
def pythonify_default(s: str) -> object:
    if s == "true":
        return True
    elif s == "false":
        return False

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


# What is a dynamic type?  Over time, the semantic meaning of
# dynamic type has degraded to meaninglessness (in the old days,
# it captured dtype-ness of types, but that has gone away with
# the removal of TH).  These days, it's mostly the same thing as
# the C++ API argument type, except that Tensor and Tensor?
# arguments simply present as Tensor.
#
# TODO: Get rid of dynamic_type, after getting tools/autograd
# to use the new codegen framework
def dynamic_type(t: Type) -> str:
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
    # Note we don't use t.is_tensor_like() here because it would
    # also include Tensor[]
    if str(t) == "Tensor":
        return "at::Tensor"
    # This is a legacy concept, so never report SymInt
    return cpp.argumenttype_type(
        t, mutable=False, binds="__placeholder__", symint=False
    ).cpp_type()


def compute_method_of_yaml(variants: set[Variant]) -> list[str]:
    # This is written out explicitly to ensure that Tensor and
    # namespace are put into the list in the right order
    method_of = ["Type"]
    if Variant.method in variants:
        method_of.append("Tensor")
    if Variant.function in variants:
        method_of.append("namespace")
    return method_of


def compute_returns_yaml(
    f: NativeFunction,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    # Note [name and field_name]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # To understand name_to_field_name, we must first talk about this
    # schema:
    #
    #   lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
    #
    # There is something very odd about this schema: it is an out
    # variant of the function (that is to say, it will convert into
    # at::lstsq_out() in the C++ API), but the names of the output
    # return arguments don't match the keyword argument names of
    # the inputs.  It TURNS OUT that in this situation, the historical
    # Declarations.yaml we want to output is this (abbreviated to
    # only show relevant fields):
    #
    #   arguments:
    #     ...
    #   - field_name: solution
    #     name: X
    #   - field_name: QR
    #     name: qr
    #     ...
    #
    #   returns:
    #   - field_name: solution
    #     name: X
    #   - field_name: QR
    #     name: qr
    #
    # The name of the return fields is stored in 'field_name', and the
    # name of the arguments is stored in 'name'.  So when we process
    # arguments, we need a way to get at the corresponding return.  At
    # the moment, this is most conveniently done by constructing a
    # mapping from name (the argument concept) to field_name (the
    # return concept) while processing return arguments, since we don't
    # directly maintain this correspondence in the modeling of function
    # schema itself.
    #
    # See also https://github.com/pytorch/pytorch/issues/43114
    name_to_field_name: dict[str, str] = {}

    # Compute the returns field of the YAML entry
    names = cpp.return_names(f)
    returns = []
    for i, (r, name) in enumerate(zip(f.func.returns, names)):
        ret = {
            "dynamic_type": dynamic_type(r.type),
            "name": name,
            # legacy, report ints
            "type": cpp.return_type(r, symint=False).cpp_type(),
        }

        if r.name:
            # See Note [name and field_name]
            ret["field_name"] = r.name
            if f.func.is_out_fn():
                name_to_field_name[f.func.arguments.out[i].name] = r.name

        returns.append(ret)

    return returns, name_to_field_name


# arguments in yaml roughly corresponds to the public C++ API
def compute_cpp_argument_yaml(
    cpp_a: Binding,
    *,
    schema_order: bool,
    kwarg_only_set: set[str],
    out_arg_set: set[str],
    name_to_field_name: dict[str, str],
) -> object:
    if isinstance(cpp_a.argument, TensorOptionsArguments):
        arg: dict[str, object] = {
            "annotation": None,
            "dynamic_type": "at::TensorOptions",
            "is_nullable": False,
            "name": cpp_a.name,
            "type": cpp_a.type,
            "kwarg_only": True,
        }
        if cpp_a.default is not None:
            arg["default"] = cpp_a.default
        return arg
    elif isinstance(cpp_a.argument, SelfArgument):
        raise AssertionError
    elif isinstance(cpp_a.argument, Argument):
        return compute_argument_yaml(
            cpp_a.argument,
            schema_order=schema_order,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )


def compute_argument_yaml(
    a: Argument,
    *,
    schema_order: bool,
    kwarg_only_set: set[str],
    out_arg_set: set[str],
    name_to_field_name: dict[str, str],
) -> object:
    arg: dict[str, object] = {
        "annotation": str(a.annotation) if a.annotation else None,
        "dynamic_type": dynamic_type(a.type),
        "is_nullable": a.type.is_nullable(),
        "name": a.name,
        # legacy, report ints
        "type": cpp.argument_type(a, binds="__placeholder__", symint=False).cpp_type(),
    }
    if a.default is not None:
        arg["default"] = pythonify_default(
            cpp.default_expr(a.default, a.type, symint=False)
        )
    if a.name in kwarg_only_set:
        arg["kwarg_only"] = True
    if a.name in out_arg_set:
        arg["output"] = True
        arg["allocate"] = True
        # See Note [name and field_name]
        if a.name in name_to_field_name:
            arg["field_name"] = name_to_field_name[a.name]
    # Historically, booleans don't get their size recorded, because it
    # is already built into the cpp type (e.g., std::array<bool, 4>)
    l = a.type.is_list_like()
    if l is not None and l.size is not None and str(l.elem) != "bool":
        arg["size"] = l.size
    return arg


@with_native_function
def compute_declaration_yaml(f: NativeFunction) -> object:
    returns, name_to_field_name = compute_returns_yaml(f)

    # These sets are used to conveniently test if an argument is a
    # kwarg-only or out argument
    kwarg_only_set = {a.name for a in f.func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in f.func.arguments.out}

    sig_group = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    cpp_args = sig_group.signature.arguments()
    arguments = [
        compute_cpp_argument_yaml(
            cpp_a,
            schema_order=False,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for cpp_a in cpp_args
    ]

    schema_order_jit_arguments = list(f.func.schema_order_arguments())

    schema_order_arguments = [
        compute_argument_yaml(
            a,
            schema_order=True,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for a in schema_order_jit_arguments
    ]

    cpp_schema_order_types = [
        # NB: method here doesn't matter
        r.type
        for a in schema_order_jit_arguments
        for r in cpp.argument(
            a,
            method=False,
            cpp_no_default_args=set(),
            faithful=False,
            symint=False,
            has_tensor_options=False,
        )
    ]

    # legacy, report ints
    cpp_returns = cpp.returns_type(f.func.returns, symint=False).cpp_type()
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"

    is_factory_method = (
        any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args)
        and Variant.method not in f.variants
    )

    return OrderedDict(
        [
            ("name", cpp.name(f.func)),
            ("operator_name", str(f.func.name.name)),
            ("overload_name", str(f.func.name.overload_name)),
            ("manual_kernel_registration", f.manual_kernel_registrati
```



## High-Level Overview


This Python file contains 9 class(es) and 65 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LineLoader`, `RegisterSchema`, `ComputeOperators`, `ComputeFunction`, `ComputeTensorMethod`, `ComputeRedispatchFunction`, `ComputeBackendSelect`

**Functions defined**: `construct_mapping`, `file_manager_from_dispatch_key`, `parse_native_yaml_struct`, `parse_tags_yaml_struct`, `parse_tags_yaml`, `parse_native_yaml`, `error_check_native_functions`, `cpp_string`, `static_dispatch_keys`, `get_static_dispatch_backend`, `static_dispatch_ops_header`, `static_dispatch_extra_headers`, `translate_args`, `add_spl_memory_format_binding`, `generate_static_dispatch_backend_call`, `generate_static_dispatch_fallback_call`, `static_dispatch`, `__call__`, `__call__`, `__call__`

**Key imports**: annotations, argparse, functools, json, keyword, os, defaultdict, namedtuple, OrderedDict, dataclass, field, Path, Any, Literal, TYPE_CHECKING, TypeVar


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `functools`
- `json`
- `keyword`
- `os`
- `collections`: defaultdict, namedtuple, OrderedDict
- `dataclasses`: dataclass, field
- `pathlib`: Path
- `typing`: Any, Literal, TYPE_CHECKING, TypeVar
- `typing_extensions`: assert_never
- `yaml`
- `torchgen.api.dispatcher as dispatcher`
- `torchgen.api.meta as meta`
- `torchgen.api.native as native`
- `torchgen.api.structured as structured`
- `torchgen.dest as dest`
- `torchgen.api`: cpp
- `torchgen.api.translate`: translate
- `torchgen.gen_vmap_plumbing`: gen_all_vmap_plumbing
- `torchgen.selective_build.selector`: SelectiveBuilder
- `torchgen.yaml_utils`: YamlDumper, YamlLoader
- `collections.abc`: Callable, Sequence
- `torchgen.model`: dispatch_keys


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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


## Cross-References

- **File Documentation**: `gen.py_docs.md`
- **Keyword Index**: `gen.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
