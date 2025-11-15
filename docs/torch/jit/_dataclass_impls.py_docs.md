# Documentation: `torch/jit/_dataclass_impls.py`

## File Metadata

- **Path**: `torch/jit/_dataclass_impls.py`
- **Size**: 6,681 bytes (6.52 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Functions for synthesizing magic methods for JIT-compiled dataclasses
import ast
import dataclasses
import inspect
import os
from collections.abc import Callable
from functools import partial

from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext


def _get_fake_filename(cls, method_name):
    return os.path.join(FAKE_FILENAME_PREFIX, cls.__name__, method_name)


def compose_fn(cls, name: str, body_lines: list[str], signature: str) -> ParsedDef:
    body = "\n".join(f"  {b}" for b in body_lines)
    decl = f"def {name}{signature}:\n{body}"

    # Parse the function declaration
    try:
        py_ast = ast.parse(decl)
    except SyntaxError as e:
        # This should only happen if there's some unforeseeable change
        # in the dataclasses module that makes our synthesized code fail
        raise RuntimeError(
            f"TorchScript failed to synthesize dataclass method '{name}' for class '{cls.__name__}'. "
            "Please file a bug report at <https://github.com/pytorch/pytorch/issues>"
        ) from e
    fake_filename = _get_fake_filename(cls, name)
    # Parse the function
    return ParsedDef(
        py_ast,
        ctx=SourceContext(
            source=decl, filename=fake_filename, file_lineno=0, leading_whitespace_len=0
        ),
        source=decl,
        filename=fake_filename,
        file_lineno=0,
    )


def synthesize__init__(cls) -> ParsedDef:
    # Supporting default factories in the way that people expect would sort of require us to
    # allow compiling lambda functions, which is not currently supported.
    if any(
        field.default_factory is not dataclasses.MISSING
        for field in dataclasses.fields(cls)
    ):
        raise NotImplementedError(
            "Default factory initializers are not supported in TorchScript dataclasses"
        )

    # Simply read off the generated __init__ signature from CPython's implementation. It'll be
    # almost correct except for InitVar annotations, which we need to handle specially.
    signature = inspect.signature(cls.__init__)

    # Handle InitVars if needed (only works on Python 3.8+, when a `type` attribute was added to InitVar);
    # see CPython commit here https://github.com/python/cpython/commit/01ee12ba35a333e8a6a25c4153c4a21838e9585c
    init_vars: list[str] = []
    params = []
    for name, param in signature.parameters.items():
        ann = param.annotation

        if isinstance(ann, dataclasses.InitVar):
            # The TorchScript interpreter can't handle InitVar annotations, so we unwrap the underlying type here
            init_vars.append(name)
            params.append(param.replace(annotation=ann.type))  # type: ignore[attr-defined]
        else:
            params.append(param)

    signature = signature.replace(parameters=params)

    body = [
        # Assign all attributes to self
        f"self.{field.name} = {field.name}"
        for field in dataclasses.fields(cls)
        if field.init and field.name not in init_vars
    ]
    # Call user's impl of __post_init__ if it exists
    if hasattr(cls, "__post_init__"):
        body.append("self.__post_init__(" + ", ".join(init_vars) + ")")

    return compose_fn(cls, "__init__", body or ["pass"], signature=str(signature))


# This is a placeholder at the moment since the TorchScript interpreter doesn't call __repr__
def synthesize__repr__(cls) -> ParsedDef:
    return compose_fn(
        cls,
        "__repr__",
        [
            f"return '{cls.__name__}("
            + ", ".join(
                [
                    f"{field.name}=self.{field.name}"
                    for field in dataclasses.fields(cls)
                    if field.repr
                ]
            )
            + ")'"
        ],
        signature="(self) -> str",
    )


def synthesize__hash__(cls) -> ParsedDef:
    return compose_fn(
        cls,
        "__hash__",
        [
            # This is just a placeholder to prevent compilation from failing; this won't even get called at
            # all right now because the TorchScript interpreter doesn't call custom __hash__ implementations
            "raise NotImplementedError('__hash__ is not supported for dataclasses in TorchScript')"
        ],
        signature="(self) -> int",
    )


# Implementation for __eq__ and __ne__
def synthesize_equality(cls, name: str, converse: str) -> ParsedDef:
    return synthesize_comparison(
        cls,
        name,
        allow_eq=True,
        raise_on_none=False,
        inner=[f"if val1 {converse} val2: return False"],
    )


def synthesize_inequality(cls, name: str, op: str, allow_eq: bool) -> ParsedDef:
    return synthesize_comparison(
        cls,
        name,
        allow_eq,
        raise_on_none=True,
        inner=[
            f"if val1 {op} val2: return True",
            f"elif val2 {op} val1: return False",
        ],
    )


def synthesize_comparison(
    cls, name: str, allow_eq: bool, raise_on_none: bool, inner: list[str]
) -> ParsedDef:
    body = []
    for field in dataclasses.fields(cls):
        if not field.compare:
            continue

        body.extend(
            [
                f"val1 = self.{field.name}",
                f"val2 = other.{field.name}",
            ]
        )
        body.extend(
            inner
            if not is_optional(field.type)
            else [
                # Type refinement for optional fields; we need this to avoid type errors from the interpreter
                "if val1 is not None and val2 is not None:",
                *["  " + line for line in inner],
                "elif (val1 is None) != (val2 is None):",
                f"  raise TypeError('Cannot compare {cls.__name__} with None')"
                if raise_on_none
                else "  return False",
            ]
        )

    body.append(f"return {allow_eq}")
    return compose_fn(
        cls, name, body, signature=f"(self, other: {cls.__name__}) -> bool"
    )


DATACLASS_MAGIC_METHODS: dict[str, Callable] = {
    "__init__": synthesize__init__,
    "__repr__": synthesize__repr__,
    "__hash__": synthesize__hash__,
    "__eq__": partial(synthesize_equality, name="__eq__", converse="!="),
    "__ne__": partial(synthesize_equality, name="__ne__", converse="=="),
    "__lt__": partial(synthesize_inequality, name="__lt__", op="<", allow_eq=False),
    "__le__": partial(synthesize_inequality, name="__le__", op="<", allow_eq=True),
    "__gt__": partial(synthesize_inequality, name="__gt__", op=">", allow_eq=False),
    "__ge__": partial(synthesize_inequality, name="__ge__", op=">", allow_eq=True),
}

```



## High-Level Overview


This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_fake_filename`, `compose_fn`, `synthesize__init__`, `synthesize__repr__`, `synthesize__hash__`, `synthesize_equality`, `synthesize_inequality`, `synthesize_comparison`

**Key imports**: ast, dataclasses, inspect, os, Callable, partial, FAKE_FILENAME_PREFIX, is_optional, ParsedDef, SourceContext


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `ast`
- `dataclasses`
- `inspect`
- `os`
- `collections.abc`: Callable
- `functools`: partial
- `torch._jit_internal`: FAKE_FILENAME_PREFIX, is_optional
- `torch._sources`: ParsedDef, SourceContext


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/jit`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_decompositions.py_docs.md`](./_decompositions.py_docs.md)
- [`quantized.py_docs.md`](./quantized.py_docs.md)
- [`frontend.py_docs.md`](./frontend.py_docs.md)
- [`_builtins.py_docs.md`](./_builtins.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)
- [`_state.py_docs.md`](./_state.py_docs.md)
- [`_await.py_docs.md`](./_await.py_docs.md)


## Cross-References

- **File Documentation**: `_dataclass_impls.py_docs.md`
- **Keyword Index**: `_dataclass_impls.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
