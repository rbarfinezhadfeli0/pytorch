# Documentation: `torch/fx/graph.py`

## File Metadata

- **Path**: `torch/fx/graph.py`
- **Size**: 87,086 bytes (85.04 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import builtins
import contextlib
import copy
import enum
import functools
import inspect
import keyword
import math
import os
import pprint
import re
import typing
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._C import _fx_map_arg as map_arg, _NodeIter
from torch.utils._dtype_abbrs import dtype_abbrs

from . import _pytree as fx_pytree
from ._compatibility import compatibility
from .immutable_collections import immutable_dict
from .node import _get_qualified_name, _type_repr, Argument, Node, Target


__all__ = ["PythonCode", "CodeGen", "Graph"]

if TYPE_CHECKING:
    from ._symbolic_trace import Tracer  # noqa: F401
    from .graph_module import GraphModule  # noqa: F401


# Mapping of builtins to their `typing` equivalent.
# (PEP585: See D68459095 test plan)
_origin_type_map = {
    list: typing.List,  # noqa: UP006
    dict: typing.Dict,  # noqa: UP006
    set: typing.Set,  # noqa: UP006
    frozenset: typing.FrozenSet,  # noqa: UP006
    tuple: typing.Tuple,  # noqa: UP006
}

_legal_ops = dict.fromkeys(
    ["call_function", "call_method", "get_attr", "call_module", "placeholder", "output"]
)


# Signature for functions thattransforms the body (`list[str]`) of the
# generated code
TransformCodeFunc = Callable[[list[str]], list[str]]


class _CustomBuiltin(NamedTuple):
    """Additional objs that we add to every graph's globals.

    The repr() for some standard library objects is not valid Python code without
    an import. For common objects of this sort, we bundle them in the globals of
    every FX graph.
    """

    # How to import this object from the standard library.
    import_str: str
    # The actual object, produced from that import string.
    obj: Any


# Combined dict of disallowed variable names so we can check with one lookup
_illegal_names = {k: object() for k in keyword.kwlist}
_illegal_names.update(builtins.__dict__)  # can't shadow a builtin name

_custom_builtins: dict[str, _CustomBuiltin] = {}


def _register_custom_builtin(name: str, import_str: str, obj: Any):
    _custom_builtins[name] = _CustomBuiltin(import_str, obj)
    _illegal_names[name] = obj


_register_custom_builtin("inf", "from math import inf", math.inf)
_register_custom_builtin("nan", "from math import nan", math.nan)
_register_custom_builtin("NoneType", "NoneType = type(None)", type(None))
_register_custom_builtin("torch", "import torch", torch)
_register_custom_builtin("device", "from torch import device", torch.device)
_register_custom_builtin("fx_pytree", "import torch.fx._pytree as fx_pytree", fx_pytree)
_register_custom_builtin("pytree", "import torch.utils._pytree as pytree", pytree)


def _is_magic(x: str) -> bool:
    return x.startswith("__") and x.endswith("__")


def _snake_case(s: str) -> str:
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    return _snake_case_sub(s).lower()


# Replace occurrences where a lowercase letter is followed by an uppercase letter
_snake_case_sub = functools.partial(re.compile(r"(?<=[a-z])([A-Z])").sub, r"_\1")

# Find chars that can't be in a Python identifier
_illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")

# Combined check for variable names:
# 1) Checks name is not empty
# 2) Checks first character is not a digit
# 3) Checks name has no illegal characters (_illegal_char_regex)
# 3) Splits off the number suffix (if present)
_name_regex = re.compile(r"^([a-zA-Z_][0-9a-zA-Z_]*?)(?:_(\d+))?$")

# starts with torch but does not start with torch._dynamo. or torch._inductor.
_torch_but_not_dynamo = re.compile(
    r"^torch(?:\.(?!_dynamo\.|_inductor\.)[^.]+)*$"
).fullmatch


def _is_from_torch(obj: Any) -> bool:
    module_name = getattr(obj, "__module__", None)
    if module_name is not None:
        return _torch_but_not_dynamo(module_name) is not None

    name = getattr(obj, "__name__", None)
    # exclude torch because torch.torch.torch.torch works. idk mang
    if name is not None and name != "torch":
        for guess in [torch, torch.nn.functional]:
            if getattr(guess, name, None) is obj:
                return True

    return False


class _Namespace:
    """A context for associating names uniquely with objects.

    The following invariants are enforced:
    - Each object gets a single name.
    - Each name is unique within a given namespace.
    - Names generated do not shadow builtins, unless the object is indeed that builtin.
    """

    def __init__(self):
        self._obj_to_name: dict[Any, str] = {}
        self._used_names: set[str] = set()
        self._base_count: dict[str, int] = {}

    def create_name(self, candidate: str, obj: Optional[Any]) -> str:
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
        if obj is not None and obj in self._obj_to_name:
            return self._obj_to_name[obj]

        # optimistically check if candidate is already a valid name
        match = _name_regex.match(candidate)
        if match is None:
            # delete all characters that are illegal in a Python identifier
            candidate = _illegal_char_regex.sub("_", candidate)

            if not candidate:
                candidate = "_unnamed"

            if candidate[0].isdigit():
                candidate = f"_{candidate}"

            match = _name_regex.match(candidate)
            assert match is not None

        base, num = match.group(1, 2)
        if num is None or candidate in self._used_names:
            num = self._base_count.get(candidate, 0)
            if _illegal_names.get(candidate, obj) is not obj:
                num += 1
                candidate = f"{base}_{num}"
                # assume illegal names don't end in _\d so no need to check again
        else:
            num = int(num)

        while candidate in self._used_names:
            num += 1
            candidate = f"{base}_{num}"

        self._used_names.add(candidate)
        self._base_count[base] = num
        if obj is not None:
            self._obj_to_name[obj] = candidate
        return candidate

    def associate_name_with_obj(self, name: str, obj: Any):
        """Associate a unique name with an object.

        Neither `name` nor `obj` should be associated already.
        """
        maybe_existing = self._obj_to_name.setdefault(obj, name)
        assert maybe_existing is name, "obj is already associated"

    def _rename_object(self, obj: Any, name: str):
        assert obj in self._obj_to_name
        self._obj_to_name[obj] = name
        self._used_names.add(name)


@compatibility(is_backward_compatible=True)
@dataclass
class PythonCode:
    """
    Represents all the information necessary to exec or save a graph as Python code.
    """

    # Python source code for the forward function definition.
    src: str
    # Values in global scope during execution of `src_def`.
    globals: dict[str, Any]
    # Optional mapping from the forward function's line number to
    # node index. Line number starts at the prologue (i.e. forward()).
    _lineno_map: Optional[dict[int, Optional[int]]]
    # The line number of prologue in fn_code
    _prologue_start: int = 0


def _format_target(base: str, target: str) -> str:
    elems = target.split(".")
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f"{r}.{e}"
    return r


class _InsertPoint:
    def __init__(self, graph, new_insert):
        self.graph = graph
        self.orig_insert, graph._insert = graph._insert, new_insert

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        self.graph._insert = self.orig_insert


class _node_list:
    def __init__(self, graph: "Graph", direction: Literal["_prev", "_next"] = "_next"):
        assert direction in ("_next", "_prev")
        self.graph = graph
        self.direction = direction

    def __len__(self):
        return self.graph._len

    def __iter__(self):
        return _NodeIter(self.graph._root, self.direction == "_prev")

    def __reversed__(self):
        return _node_list(self.graph, "_next" if self.direction == "_prev" else "_prev")


class _PyTreeInfo(NamedTuple):
    """
    Contains extra info stored when we're using Pytrees
    """

    orig_args: list[str]
    in_spec: pytree.TreeSpec
    out_spec: Optional[pytree.TreeSpec]


@dataclass(frozen=True)
class _ParsedStackTrace:
    """
    Represents the top-most frame of a parsed stack trace
    """

    file: str
    lineno: str
    name: str
    code: str

    def get_summary_str(self):
        return f"File: {self.file}:{self.lineno} in {self.name}, code: {self.code}"


# get File:lineno code from stack_trace
def _parse_stack_trace(
    stack_trace: str, filter_fn: Optional[Callable[[str, str, str], bool]] = None
):
    if stack_trace is None:
        return None
    pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")
    lines = stack_trace.strip().split("\n")
    # stacktrace should have innermost frame last, so we
    # iterate backwards to find the first line that starts
    # with 'File '
    for idx in range(len(lines) - 2, -1, -1):
        line = lines[idx].strip()
        matches = pattern.match(line)
        if matches:
            file = matches.group(1)
            lineno = matches.group(2)
            name = matches.group(3)
            # next line should be the code
            code = lines[idx + 1].strip()
            if filter_fn and not filter_fn(file, name, code):
                continue
            return _ParsedStackTrace(file, lineno, name, code)
    return None


@compatibility(is_backward_compatible=False)
class CodeGen:
    # This is an override hook so we can customize the SymNode printer.
    _sym_repr: Callable[["torch.types.PySymType"], str] = lambda x: repr(x)

    def __init__(self):
        self._body_transformer: Optional[TransformCodeFunc] = None
        self._func_name: str = "forward"

    def _format_multiline_args(self, args: list[str]) -> str:
        """Helper to format function arguments in expanded multiline format."""
        return "".join(self._format_single_arg(arg) for arg in args)

    def _format_single_arg(self, arg: str) -> str:
        """Helper to format a single argument with optional comment."""
        if "#" in arg:
            arg_part, comment_part = arg.split("#", 1)
            return f"    {arg_part.rstrip()},  # {comment_part.lstrip()}\n"
        else:
            return f"    {arg},\n"

    def _get_delimiters(self, container) -> tuple[str, str]:
        """Helper to get opening and closing delimiters for containers."""
        return ("(", ")") if isinstance(container, tuple) else ("[", "]")

    def _format_multiline_container(self, items, descs=None, prefix="") -> str:
        """Helper to format containers (lists/tuples) in multiline format."""
        ldelim, rdelim = self._get_delimiters(items)
        desc_trailers = self._get_desc_trailers(items, descs)

        return (
            f"{prefix}{ldelim}\n"
            + "".join(
                f"    {item},{trailer}\n" for item, trailer in zip(items, desc_trailers)
            )
            + f"{rdelim}"
        )

    def _get_desc_trailers(self, items, descs):
        """Helper to generate description trailers for items."""
        if descs is None:
            return [""] * len(items)
        return [f"  # {desc}" for desc in descs]

    def _call_method_with_signature_check(self, method, *args, **kwargs):
        """Helper to call a method with optional parameters based on signature."""
        sig = inspect.signature(method)
        # Filter kwargs to only include parameters that exist in the method signature
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return method(*args, **filtered_kwargs)

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        """
        Given the free variables and a return annotation, generates the beginning of the FX function.
        By default, `gen_fn_def(['a', 'b'], '') == 'def {self._func_name}(a, b):'`
        """
        # If the original function didn't have self as its first argument, we
        # would have added it.
        if len(free_vars) == 0 or free_vars[0] != "self":
            free_vars.insert(0, "self")

        if expanded_def:
            args_formatted = self._format_multiline_args(free_vars)
            return (
                f"def {self._func_name}(\n{args_formatted}){maybe_return_annotation}:"
            )
        else:
            return f"def {self._func_name}({', '.join(free_vars)}){maybe_return_annotation}:"

    def generate_output(
        self, output_args: Argument, *, descs: Optional[Any] = None
    ) -> str:
        """
        Given the output arguments, generates the return statement of the FX function.
        Note: The returned statement should not be indented.
        """
        if descs is not None and isinstance(output_args, (list, tuple)):
            return self._format_multiline_container(output_args, descs, "return ")
        else:
            return f"return {repr(output_args)}"

    def process_inputs(self, *args: Any) -> Any:
        """
        Transforms the inputs so that the graph can take them as arguments, as
        non-default codegen may result in the inputs to the function being
        different from the inputs to the graph.

        If the graph was directly runnable, this invariant should hold true
        `f.graph.process_outputs(f.graph(*f.graph.process_inputs(*inputs))) == f(*inputs)`
        """
        return args

    def process_outputs(self, outputs: Any) -> Any:
        """
        Transforms the outputs of the graph to be identical to the codegen.

        See ``process_inputs`` for more details.
        """
        return outputs

    def additional_globals(self) -> list[tuple[str, Any]]:
        """
        If your codegen uses extra global values, add tuples of (identifier,reference to the value) here.
        For example, return ['List', typing.List] if you need ``List`` in the global context.
        """
        return []

    def _gen_python_code(
        self,
        nodes,
        root_module: str,
        namespace: _Namespace,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        # Render each argument on its own line
        expanded_def: bool = False,
        record_func: bool = False,
    ) -> PythonCode:
        free_vars: list[str] = []
        body: list[str] = []
        globals_: dict[str, Any] = {}
        wrapped_fns: dict[str, None] = {}

        # Wrap string in list to pass by reference
        maybe_return_annotation: list[str] = [""]
        include_stride = include_stride or (
            os.environ.get("FX_GRAPH_SHOW_STRIDE", "0") == "1"
        )
        include_device = include_device or (
            os.environ.get("FX_GRAPH_SHOW_DEVICE", "0") == "1"
        )
        include_meta = os.environ.get("FX_GRAPH_SHOW_META", "0") == "1"

        def add_global(name_hint: str, obj: Any):
            """Add an obj to be tracked as a global.

            We call this for names that reference objects external to the
            Graph, like functions or types.

            Returns: the global name that should be used to reference 'obj' in generated source.
            """
            if (
                _is_from_torch(obj) and obj != torch.device
            ):  # to support registering torch.device
                # HACK: workaround for how torch custom ops are registered. We
                # can't import them like normal modules so they must retain their
                # fully qualified name.
                return _get_qualified_name(obj)

            # normalize the name hint to get a proper identifier
            global_name = namespace.create_name(name_hint, obj)

            if global_name in globals_:
                assert globals_[global_name] == obj
                return global_name
            globals_[global_name] = obj
            return global_name

        # Pre-fill the globals table with registered builtins.
        for name, (_, obj) in _custom_builtins.items():
            add_global(name, obj)

        def type_repr(o: Any):
            if o == ():
                # Empty tuple is used for empty tuple type annotation Tuple[()]
                return "()"

            typename = _type_repr(o)

            if origin_type := getattr(o, "__origin__", None):
                # list[...], typing.List[...], TensorType[...]

                if isinstance(o, typing._GenericAlias):  # type: ignore[attr-defined]
                    # This is a generic pre-PEP585 type, e.g. typing.List[torch.Tensor]
                    origin_type = _origin_type_map.get(origin_type, origin_type)

                origin_typename = add_global(_type_repr(origin_type), origin_type)

                if hasattr(o, "__args__") and o.__args__:
                    args = [type_repr(arg) for arg in o.__args__]
                    return f"{origin_typename}[{','.join(args)}]"
                else:
                    return origin_typename

            # Common case: this is a regular module name like 'foo.bar.baz'
            return add_global(typename, o)

        if colored:
            red = _color_fns["red"]
            dim_green = _color_fns["dim_green"]
            dim = _color_fns["dim"]
            dim_blue = _color_fns["dim_blue"]
            blue = _color_fns["blue"]
        else:
            red = _identity
            dim_green = _identity
            dim = _identity
            dim_blue = _identity
            blue = _identity

        def _get_repr(arg: Any) -> str:
            if isinstance(arg, Node):  # first because common
                return repr(arg)
            elif isinstance(arg, tuple) and hasattr(arg, "_fields"):
                # Handle NamedTuples (if it has `_fields`) via add_global.
                qualified_name = _get_qualified_name(type(arg))
                global_name = add_global(qualified_name, type(arg))
                return f"{global_name}{repr(tuple(arg))}"
            elif isinstance(
                arg, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
            ):
                qualified_name = _get_qualified_name(arg)
                global_name = add_global(qualified_name, arg)
                return f"{global_name}"
            elif isinstance(arg, enum.Enum):
                cls = arg.__class__
                clsname = add_global(cls.__name__, cls)
                return f"{clsname}.{arg.name}"
            elif isinstance(arg, torch.Tensor):
                size = list(arg.size())
                dtype = str(arg.dtype).split(".")[-1]
                return f"torch.Tensor(size={size}, dtype={dtype})"
            elif isinstance(arg, tuple):
                if len(arg) == 1:
                    return f"({_get_repr(arg[0])},)"
                else:
                    return "(" + ", ".join(_get_repr(a) for a in arg) + ")"
            elif isinstance(arg, list):
                return "[" + ", ".join(_get_repr(a) for a in arg) + "]"
            elif isinstance(arg, slice):
                return f"slice({_get_repr(arg.start)}, {_get_repr(arg.stop)}, {_get_repr(arg.step)})"
            else:
                return blue(repr(arg))

        def _format_args(
            args: tuple[Argument, ...], kwargs: dict[str, Argument]
        ) -> str:
            res = [_get_repr(a) for a in args]
            res.extend([f"{k} = {_get_repr(v)}" for k, v in kwargs.items()])
            return ", ".join(res)

        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use: dict[Node, Node] = {}
        user_to_last_uses: dict[Node, list[Node]] = {}

        def register_last_uses(n: Node, user: Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            for input_node in node._input_nodes:
                register_last_uses(input_node, node)

        def delete_unused_values(user: Node):
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
            if user.op == "placeholder":
                return
            if user.op == "output":
                body.append("\n")
                return
            nodes_to_delete = user_to_last_uses.get(user, [])

            if len(user.users.keys()) == 0:
                # This node is not used by any others. however it's also not
                # removed by DCE since side-effect. We want to free it's outputs
                # right after its execution done to save memory.
                nodes_to_delete.append(user)

            if len(nodes_to_delete):
                to_delete_str = " = ".join(
                    [repr(n) for n in nodes_to_delete] + ["None"]
                )
                body.append(f";  {dim(to_delete_str)}\n")
            else:
                body.append("\n")

        prev_summary_str = None

        def append_stacktrace_summary(node: Node):
            """
            Append a summary of the stacktrace to the generated code. This is
            useful for debugging.
            """
            nonlocal prev_summary_str

            if node.op not in {"placeholder", "output"}:
                annotation_str = ""
                annotation = node.meta.get("custom", {})
                if annotation:
                    annotation_str = f" Annotation: {annotation}"

                stack_trace_str = "No stacktrace found for following nodes"
                if stack_trace := node.stack_trace:
                    if parsed_stack_trace := _parse_stack_trace(stack_trace):
                        stack_trace_str = parsed_stack_trace.get_summary_str()

                summary_str = f"\n{dim(f'#{annotation_str} {stack_trace_str}')}\n"

                if summary_str != prev_summary_str:
                    prev_summary_str = summary_str
                    body.append(summary_str)

        def stringify_shape(shape: Iterable) -> str:
            return f"[{', '.join([str(x) for x in shape])}]"

        def emit_node(node: Node):
            maybe_type_annotation = (
                "" if node.type is None else f" : {type_repr(node.type)}"
            )
            maybe_comment = ""

            if verbose:
                # override annotation with more detailed information
                try:
                    from torch.distributed.tensor._api import DTensor, DTensorSpec

                    dtensorspec_format_shard_order_str = (
                        DTensorSpec.format_shard_order_str
                    )
                except ModuleNotFoundError:
                    DTensor = None  # type: ignore[assignment,misc]
                    dtensorspec_format_shard_order_str = None
                from torch.fx.experimental.proxy_tensor import py_sym_types
                from torch.fx.passes.shape_prop import TensorMetadata

                meta_val = node.meta.get(
                    "val",
                    node.meta.get("tensor_meta", node.meta.get("example_value", None)),
                )

                def _tensor_annotation(t: torch.Tensor) -> str:
                    stride = stringify_shape(t.stride()) if include_stride else ""
                    device = f"{t.device}" if include_device else ""
                    return (
                        f"{red(dtype_abbrs[t.dtype])}"
                        f"{blue(stringify_shape(t.shape))}"
                        f"{dim_blue(stride)}"
                        f"{dim_green(device)}"
                    )

                # use string as annotation, to make it valid python code
                if isinstance(meta_val, torch.Tensor) and meta_val.layout not in (
                    torch.sparse_csc,
                    torch.sparse_csr,
                ):
                    # Fake tensors cause tests to wobble, so do not custom print them.
                    is_plain = type(meta_val) is torch.Tensor or isinstance(
                        meta_val, torch._subclasses.FakeTensor
                    )
                    core = _tensor_annotation(meta_val)
                    if is_plain:
                        maybe_type_annotation = f': "{core}"'
                    elif type(meta_val) is DTensor:
                        assert dtensorspec_format_shard_order_str is not None
                        dtensor_meta = dtensorspec_format_shard_order_str(
                            meta_val._spec.placements,  # type: ignore[attr-defined]
                            meta_val._spec.shard_order,  # type: ignore[attr-defined]
                        )
                        cls = meta_val.__class__.__name__
                        maybe_type_annotation = (
                            f': "{cls}({core}, {dim_green(dtensor_meta)})"'
                        )
                    else:
                        cls = meta_val.__class__.__name__
                        maybe_type_annotation = f': "{cls}({core})"'

                elif isinstance(meta_val, py_sym_types):
                    val_str = CodeGen._sym_repr(meta_val)
                    maybe_type_annotation = f': "Sym({val_str})"'

                elif isinstance(meta_val, TensorMetadata):
                    maybe_type_annotation = f': "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}"'

            desc = None
            if expanded_def:
                desc = node.meta.get("desc", None)
                if desc is not None and node.op == "placeholder":
                    maybe_comment += f"  # {desc}"
                # output is handled specially

            if include_meta and hasattr(node, "meta") and node.meta:
                body.append('"""\n')
                for k, v in node.meta.items():
                    # use str over repr since repr is susceptible to sympy
                    # errors such as "cannot determine truth value of Relational"
                    # Pretty print the high-level dict with str() for values
                    body.append(
                        f"{k}: {pprint.pformat(str(v), width=80, compact=True)}\n"
                    )
                body.append('"""\n')

            if node.op == "placeholder":
                assert isinstance(node.target, str)
                maybe_default_arg = (
                    "" if not node.args else f" = {_get_repr(node.args[0])}"
                )
                free_vars.append(
                    f"{node.target}{maybe_type_annotation}{maybe_default_arg}{maybe_comment}"
                )
                raw_name = node.target.replace("*", "")
                if raw_name != repr(node):
                    body.append(f"{repr(node)} = {raw_name}\n")
                return
            elif node.op == "call_method":
                assert isinstance(node.target, str)
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.target)}"
                    f"({_format_args(node.args[1:], node.kwargs)})"
                )
                return
            elif node.op == "call_function":
                assert callable(node.target)
                # pretty print operators
                if (
                    getattr(node.target, "__module__", "") == "_operator"
                    and node.target.__name__ in magic_methods
                ):
                    assert isinstance(node.args, tuple)
                    body.append(
                        f"{repr(node)}{maybe_type_annotation} = "
                        f"{magic_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))}"
                    )
                    return

                # pretty print inplace operators; required for jit.script to work properly
                # not currently supported in normal FX graphs, but generated by torchdynamo
                if (
                    getattr(node.target, "__module__", "") == "_operator"
                    and node.target.__name__ in inplace_methods
                ):
                    body.append(
                        f"{inplace_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))};  "
                        f"{repr(node)}{maybe_type_annotation} = {_get_repr(node.args[0])}"
                    )
                    return

                qualified_name = _get_qualified_name(node.target)
                global_name = add_global(qualified_name, node.target)
                # special case for getattr: node.args could be 2-argument or 3-argument
                # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
                if (
                    global_name == "getattr"
                    and isinstance(node.args, tuple)
                    and isinstance(node.args[1], str)
                    and node.args[1].isidentifier()
                    and len(node.args) == 2
                ):
                    body.append(
                        f"{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.args[1])}"
                    )
                    return
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})"
                )
                if node.meta.get("is_wrapped", False):
                    wrapped_fns.setdefault(global_name)
                return
            elif node.op == "call_module":
                assert isinstance(node.target, str)
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = "
                    f"{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})"
                )
                return
            elif node.op == "get_attr":
                assert isinstance(node.target, str)
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}"
                )
                return
            elif node.op == "output":
                if node.type is not None:
                    maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                body.append(
                    self._call_method_with_signature_check(
                        self.generate_output,
                        node.args[0],
                        descs=desc if expanded_def else None,
                    )
                )
                return
            raise NotImplementedError(f"node: {node.op} {node.target}")

        if record_func:
            body.append(
                "_rf = torch._C._profiler._RecordFunctionFast('## ENTER_GRAPH_PLACEHOLDER_KEY ##'); _rf.__enter__()\n"
            )
        for i, node in enumerate(nodes):
            # NOTE: emit_node does not emit a string with newline. It depends
            # on delete_unused_values to append one
            if verbose:
                append_stacktrace_summary(node)
            # emit a counter comment to keep track of
            # node index, which will be deleted later
            # after going through _body_transformer
            body.append(f"# COUNTER: {i}\n")
            do_record = record_func and node.op in (
                "call_function",
                "call_method",
                "call_module",
            )
            if do_record:
                # The double hash ## convention is used by post-processing to find the fx markers
                body.append(
                    f"_rf_{node.name} = torch._C._profiler._RecordFunctionFast('## {i} ##'); _rf_{node.name}.__enter__()\n"
                )
            emit_node(node)
            delete_unused_values(node)
            if do_record:
                body.append(f"_rf_{node.name}.__exit__(None, None, None)\n")
        if record_func:
            body.append("_rf.__exit__(None, None, None)\n")

        if len(body) == 0:
            # If the Graph has no non-placeholder nodes, no lines for the body
            # have been emitted. To continue to have valid Python code, emit a
            # single pass statement
            body.append("pass\n")

        if len(wrapped_fns) > 0:
            wrap_name = add_global("wrap", torch.fx.wrap)
            wrap_stmts = "\n".join([f'{wrap_name}("{name}")' for name in wrapped_fns])
        else:
            wrap_stmts = ""

        if self._body_transformer:
            body = self._body_transformer(body)

        for name, value in self.additional_globals():
            add_global(name, value)

        prologue = self._call_method_with_signature_check(
            self.gen_fn_def,
            free_vars,
            maybe_return_annotation[0],
            expanded_def=expanded_def,
        )

        # remove counter and generate lineno to node index mapping
        lineno_map: dict[int, Optional[int]] = {}
        prologue_len = prologue.count("\n") + 1
        new_lines: list[str] = []
        cur_idx = None
        for line in "".join(body).split("\n"):
            counter = _counter_regexp.search(line)
            if counter is not None:
                cur_idx = int(counter.group(1))
            else:
                lineno_map[len(new_lines) + prologue_len] = cur_idx
                new_lines.append(line)

        code = "\n".join(new_lines).lstrip("\n")
        code = "\n".join("    " + line for line in code.split("\n"))

        fn_code = f"""
{wrap_stmts}

{prologue}
{code}"""
        # The +4 accounts for the empty lines before prologue in fn_code
        prologue_start = wrap_stmts.count("\n") + 4
        return PythonCode(
            fn_code,
            globals_,
            _lineno_map=lineno_map,
            _prologue_start=prologue_start,
        )


# Ideally, we'd like to refactor all of the pytree logic into this codegen
# class. Unfortunately, there are 3 areas we currently need extra logic in FX.
# 1. In the initial symbolic trace, the pytree logic is tied up with `concrete_args`.
# 2. In the FX graph, we need to access 2 attributes - in_spec and out_spec.
#    Since we can't access .graph within the FX forward, we need to copy the attribute to the module.
# 3. We currently can't register the pytree imports with `add_global` - not sure why.
class _BoxedCodeGen(CodeGen):
    """
    CodeGen subclass that generates code using the "boxed" calling convention.

    The boxed calling convention takes a single list argument and clears it
    after extracting the arguments, which allows for early deallocation of
    input tensors.
    """

    def gen_fn_def(
        self, free_vars, maybe_return_annotation, *, expanded_def: bool = False
    ):
        """
        Generate function definition for boxed calling convention.

        Instead of taking individual arguments, the generated function takes
        a single 'args_list' parameter, extracts placeholder values from it,
        and clears the list.
        """
        # Generate the function signature with args_list parameter
        fn_def = f"def {self._func_name}(self, args_list){maybe_return_annotation}:"

        if free_vars:
            # This is horribly manual but we don't get the "raw" free vars
            # without a bigger refactor.
            placeholder_vars = [
                v.split(":")[0].split("=")[0].strip() for v in free_vars if v != "self"
            ]

            if placeholder_vars:
                fn_def += "\n    args_iter = iter(args_list)"
                for var in placeholder_vars:
                    fn_def += f"\n    {var} = next(args_iter)"
                fn_def += "\n    args_list.clear()"

        return fn_def


class _PyTreeCodeGen(CodeGen):
    def __init__(self, pytree_info: _PyTreeInfo):
        super().__init__()
        self.pytree_info: _PyTreeInfo = pytree_info

    def process_inputs(self, *inputs: Any) -> Any:
        flat_args = pytree.arg_tree_leaves(*inputs)
        return flat_args

    def process_outputs(self, out: Any) -> Any:
        if self.pytree_info is None or self.pytree_info.out_spec is None:
            return out
        if not isinstance(out, (list, tuple)):
            out = [out]
        assert self.pytree_info.out_spec is not None
        return pytree.tree_unflatten(out, self.pytree_info.out_spec)

    def _format_annotations(self, free_vars: list[str], expanded_def: bool) -> str:
        """Helper to format annotations for variables in pytree codegen."""
        if not free_vars:
            return ""

        has_annotation = [x for x in free_vars if ":" in x]
        if not has_annotation:
            return ""

        if expanded_def:
            return "\n    " + "\n    ".join(has_annotation)
        else:
            return "\n    " + "".join(x + "; " for x in has_annotation) + "\n"

    def gen_var_bindings(self, fn_args, free_vars, expanded_def) -> str:
        in_spec = self.pytree_info.in_spec
        # when kwargs is present, in_spec is tuple(args, kwargs)
        has_args_kwargs_tuple = (
            in_spec.type is tuple
            and in_spec.num_children == 2
            and in_spec.child(0).type is tuple
            and in_spec.child(1).type is dict
        )
        fn_kwargs = "{}"
        fn_signature = f"[{', '.join(fn_args)}], self._in_spec"
        if has_args_kwargs_tuple:
            count_args = in_spec.child(0).num_children
            fn_args = self.pytree_info.orig_args[:count_args]
            fn_kwargs = (
                "{"
                + ", ".join(
                    f"'{k}':{v}"
                    for k, v in zip(
                        in_spec.child(1).context,
                        self.pytree_info.orig_args[count_args:],
                    )
                )
                + "}"
            )
            fn_signature = f"([{', '.join(fn_args)}], {fn_kwargs}), self._in_spec"

        # in Python, `var1: annotation1, var2: annotation2 = function_call()` is invalid.
        # we need to split it to two lines:
        # one for annotation: `var1: annotation1; var2: annotation2;` (note the semicolon)
        # one for code: `var1, var2, = function_call()`
        without_annotation = [x.split(":")[0].split("#")[0] for x in free_vars]
        bindings = self._format_annotations(free_vars, expanded_def)
        bindings += f"""
    {", ".join(without_annotation)}, = fx_pytree.tree_flatten_spec({fn_signature})"""
        return bindings

    def gen_fn_def(
        self, free_vars, maybe_return_annotation, *, expanded_def: bool = False
    ):
        # Given a user function/model:
        #   myargs = [myargs0, myargs1]
        #   mykwargs = {'mykwargs0': ..., 'mykwargs1': ...}
        #   def forward(self, mypos, *myargs, mykey=None, **mykwargs):
        #
        # The generated code flattens all keywords into positional arguments for `forward()`
        #   e.g forward(self, mypos, myargs0, myargs1, mykey, mykwargs0, mykwargs1):
        #
        # Within `forward`, `tree_flatten_spec``still parses args and kwargs separately
        #   e.g. tree_flatten_spec(([mypos, myargs0, myargs1],
        #                           {'mykey':mykey, 'mykwargs0':mykwargs0, 'mykwargs1':mykwargs1}),
        #                          self._in_spec)
        #
        # If the user function/model does not have keywords, the dict is suppressed from tree_flatten_spec
        #   e.g. tree_flatten_spec([mypos, myargs0, myargs1]), self._in_spec)
        if self.pytree_info is None:
            return super().gen_fn_def(
                free_vars, maybe_return_annotation, expanded_def=expanded_def
            )

        fn_args = self.pytree_info.orig_args
        has_orig_self = (fn_args[0] == "self") if len(fn_args) > 0 else False
        if has_orig_self:
            free_vars.insert(0, "self")
        fn_definition = super().gen_fn_def(
            fn_args[:], maybe_return_annotation, expanded_def=expanded_def
        )

        if len(free_vars) > 0:  # pytree has placeholders in it
            fn_definition += self.gen_var_bindings(fn_args, free_vars, expanded_def)
        return fn_definition

    def generate_output(self, output_args, *, descs: Optional[Any] = None):
        if self.pytree_info and self.pytree_info.out_spec:
            if descs is not None and isinstance(output_args, (list, tuple)):
                return (
                    self._format_multiline_container(
                        output_args, descs, "return pytree.tree_unflatten("
                    )
                    + ", self._out_spec)"
                )
            else:
                return (
                    f"return pytree.tree_unflatten({repr(output_args)}, self._out_spec)"
                )
        else:
            return super().generate_output(output_args, descs=descs)


class _ExportCodeGen(_PyTreeCodeGen):
    def __init__(
        self,
        pytree_info: _PyTreeInfo,
        in_shuffle_graph: "GraphModule",
        out_shuffle_graph: "GraphModule",
        tree_leaf_names: list[str],
        root: Optional[torch.nn.Module],
    ):
        super().__init__(pytree_info)
        self.in_shuffle_graph = in_shuffle_graph
        self.out_shuffle_graph = out_shuffle_graph
        self.tree_leaf_names = tree_leaf_names
        self.root = root

    def process_inputs(self, *inputs: Any) -> Any:
        flat_args = super().process_inputs(*inputs)
        if self.root is not None:
            flat_args = (self.root, *flat_args)
        self.flat_args = flat_args
        return self.in_shuffle_graph(*flat_args)

    def process_outputs(self, out: Any) -> Any:
        flat_outs = self.out_shuffle_graph(*self.flat_args, *out)
        del self.flat_args
        ret = super().process_outputs(flat_outs)
        return ret

    def gen_fn_def(self, *args, **kwargs) -> str:
        fn_def = super().gen_fn_def(*args, **kwargs)
        return fn_def

    def gen_var_bindings(self, fn_args, free_vars, expanded_def) -> str:
        without_annotation = [x.split(":")[0].split("#")[0] for x in free_vars]
        fn_signature: str = f"{', '.join(fn_args)}"
        if self.root is not None:
            fn_signature = f"self, {fn_signature}"
        return f"""
    {", ".join(self.tree_leaf_names)}, = pytree.tree_leaves(({fn_signature},))
    {", ".join(without_annotation)}, = self._in_shuffle_graph({", ".join(self.tree_leaf_names)})"""

    def generate_output(self, output_args, *args, **kwargs) -> str:
        output = f"self._out_shuffle_graph({', '.join(self.tree_leaf_names)}, {', '.join([str(a) for a in output_args])})"
        return f"return pytree.tree_unflatten({output}, self._out_spec)"


class _FindNodesLookupTable:
    """
    Side table for the graph for the purpose of doing fast queries
    """

    def __init__(self):
        self.table: dict[tuple[str, Optional[Target]], dict[Node, None]] = defaultdict(
            dict
        )

    def _key(self, node) -> tuple[str, Optional[Target]]:
        return (node.op, node.target if node.op == "call_function" else None)

    def __contains__(self, node) -> bool:
        return node in self.table[self._key(node)]

    def insert(self, node: Node) -> None:
        self.table[self._key(node)][node] = None

    def remove(self, node: Node) -> None:
        self.table[self._key(node)].pop(node)

    def find_nodes(self, *, op: str, target: Optional["Target"] = None):
        if op == "call_function":
            assert target is not None
            return [*self.table[(op, target)].keys()]

        if target is None:
            return [*self.table[(op, None)].keys()]

        # op is call_method, get_attr, call_module
        return [node for node in self.table[(op, None)] if node.target == target]


@compatibility(is_backward_compatible=True)
class Graph:
    """
    ``Graph`` is the main data structure used in the FX Intermediate Representation.
    It consists of a series of ``Node`` s, each representing callsites (or other
    syntactic constructs). The list of ``Node`` s, taken together, constitute a
    valid Python function.

    For example, the following code

    .. code-block:: python

        import torch
        import torch.fx


        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return torch.topk(
                    torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3
                )


        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

    Will produce the following Graph::

        print(gm.graph)

    .. code-block:: text

        graph(x):
            %linear_weight : [num_users=1] = self.linear.weight
            %add_1 : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
            %linear_1 : [num_users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
            %relu_1 : [num_users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
            %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
            %topk_1 : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
            return topk_1

    For the semantics of operations represented in the ``Graph``, please see :class:`Node`.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        owning_module: Optional["GraphModule"] = None,
        tracer_cls: Optional[type["Tracer"]] = None,
        tracer_extras: Optional[dict[str, Any]] = None,
    ):
        """
        Construct an empty Graph.
        """
        self._root: Node = Node(self, "", "root", "", (), {})
        self._used_names: dict[str, int] = {}  # base name -> number
        self._insert = self._root.prepend
        self._len = 0
        self._graph_namespace = _Namespace()
        self._owning_module = owning_module
        self._tracer_cls = tracer_cls
        self._tracer_extras = tracer_extras
        self._codegen = CodeGen()
        self._co_fields: dict[str, Any] = {}
        self._find_nodes_lookup_table = _FindNodesLookupTable()

    @property
    def owning_module(self):
        return self._owning_module

    @owning_module.setter
    def owning_module(self, mod: Optional["GraphModule"]):
        self._owning_module = mod

    @property
    def nodes(self) -> _node_list:
        """
        Get the list of Nodes that constitute this Graph.

        Note that this ``Node`` list representation is a doubly-linked list. Mutations
        during iteration (e.g. delete a Node, add a Node) are safe.

        Returns:

            A doubly-linked list of Nodes. Note that ``reversed`` can be called on
            this list to switch iteration order.
        """
        return _node_list(self)

    @compatibility(is_backward_compatible=False)
    def output_node(self) -> Node:
        output_node = next(iter(reversed(self.nodes)))
        assert output_node.op == "output"
        return output_node

    @compatibility(is_backward_compatible=False)
    def find_nodes(
        self, *, op: str, target: Optional["Target"] = None, sort: bool = True
    ):
        """
        Allows for fast query of nodes

        Args:

            op (str): the name of the operation

            target (Optional[Target]): the target of the node. For call_function,
                the target is required. For other ops, the target is optional.

            sort (bool): whether to return nodes in the order they appear on
                         on the graph.

        Returns:

            Iterable of nodes with the requested op and target.
        """
        node_list = self._find_nodes_lookup_table.find_nodes(op=op, target=target)
        if sort:
            return sorted(node_list)
        return node_list

    @compatibility(is_backward_compatible=True)
    def graph_copy(
        self, g: "Graph", val_map: dict[Node, Node], return_output_node=False
    ) -> "Optional[Argument]":
        """
        Copy all nodes from a given graph into ``self``.

        Args:

            g (Graph): The source graph from which to copy Nodes.

            val_map (Dict[Node, Node]): a dictionary that will be populated with a mapping
                from nodes in ``g`` to nodes in ``self``. Note that ``val_map`` can be passed
                in with values in it already to override copying of certain values.

        Returns:

            The value in ``self`` that is now equivalent to the output value in ``g``,
            if ``g`` had an ``output`` node. ``None`` otherwise.
        """
        for node in g.nodes:
            if node i
```



## High-Level Overview


This Python file contains 16 class(es) and 109 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_CustomBuiltin`, `_Namespace`, `PythonCode`, `_InsertPoint`, `_node_list`, `_PyTreeInfo`, `_ParsedStackTrace`, `CodeGen`, `_BoxedCodeGen`, `_PyTreeCodeGen`, `_ExportCodeGen`, `_FindNodesLookupTable`, `Graph`, `MyModule`

**Functions defined**: `_register_custom_builtin`, `_is_magic`, `_snake_case`, `_is_from_torch`, `__init__`, `create_name`, `associate_name_with_obj`, `_rename_object`, `_format_target`, `__init__`, `__enter__`, `__exit__`, `__init__`, `__len__`, `__iter__`, `__reversed__`, `get_summary_str`, `_parse_stack_trace`, `__init__`, `_format_multiline_args`

**Key imports**: builtins, contextlib, copy, enum, functools, inspect, keyword, math, os, pprint


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `builtins`
- `contextlib`
- `copy`
- `enum`
- `functools`
- `inspect`
- `keyword`
- `math`
- `os`
- `pprint`
- `re`
- `typing`
- `warnings`
- `collections`: defaultdict
- `collections.abc`: Callable, Iterable, Iterator
- `dataclasses`: dataclass
- `torch`
- `torch.utils._pytree as pytree`
- `torch._C`: _fx_map_arg as map_arg, _NodeIter
- `torch.utils._dtype_abbrs`: dtype_abbrs
- `.`: _pytree as fx_pytree
- `._compatibility`: compatibility
- `.immutable_collections`: immutable_dict
- `.node`: _get_qualified_name, _type_repr, Argument, Node, Target
- `._symbolic_trace`: Tracer  
- `.graph_module`: GraphModule  
- `this object from the standard library.`
- `that`: string


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/fx`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`tensor_type.py_docs.md`](./tensor_type.py_docs.md)
- [`traceback.py_docs.md`](./traceback.py_docs.md)
- [`_symbolic_trace.py_docs.md`](./_symbolic_trace.py_docs.md)
- [`node.py_docs.md`](./node.py_docs.md)
- [`annotate.py_docs.md`](./annotate.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`subgraph_rewriter.py_docs.md`](./subgraph_rewriter.py_docs.md)


## Cross-References

- **File Documentation**: `graph.py_docs.md`
- **Keyword Index**: `graph.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
