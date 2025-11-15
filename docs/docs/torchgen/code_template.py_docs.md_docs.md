# Documentation: `docs/torchgen/code_template.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/code_template.py_docs.md`
- **Size**: 5,753 bytes (5.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/code_template.py`

## File Metadata

- **Path**: `torchgen/code_template.py`
- **Size**: 3,211 bytes (3.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import itertools
import re
import textwrap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# match $identifier or ${identifier} and replace with value in env
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block substitution by indenting to that depth and putting each element
# of the list on its own line
# if the identifier is on a line starting with non-whitespace and a list
# then it is comma separated ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.


class CodeTemplate:
    substitution_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    substitution = re.compile(substitution_str, re.MULTILINE)

    pattern: str
    filename: str

    @staticmethod
    def from_file(filename: str) -> CodeTemplate:
        with open(filename) as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str = "") -> None:
        self.pattern = pattern
        self.filename = filename

    def substitute(
        self, env: Mapping[str, object] | None = None, **kwargs: object
    ) -> str:
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            assert env is not None
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            content = "\n".join(
                itertools.chain.from_iterable(str(e).splitlines() for e in v)
            )
            content = textwrap.indent(content, prefix=indent)
            # Remove trailing whitespace on each line
            return "\n".join(map(str.rstrip, content.splitlines())).rstrip()

        def replace(match: re.Match[str]) -> str:
            indent = match.group(1)
            key = match.group(2)
            comma_before = ""
            comma_after = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ", "
                    key = key[1:]
                if key[-1] == ",":
                    comma_after = ", "
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ", ".join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)

        return self.substitution.sub(replace, self.pattern)


if __name__ == "__main__":
    c = CodeTemplate(
        """\
    int foo($args) {

        $bar
            $bar
        $a+$b
    }
    int commatest(int a${,stuff})
    int notest(int a${,empty,})
    """
    )
    print(
        c.substitute(
            args=["hi", 8],
            bar=["what", 7],
            a=3,
            b=4,
            stuff=["things...", "others"],
            empty=[],
        )
    )

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CodeTemplate`

**Functions defined**: `from_file`, `__init__`, `substitute`, `lookup`, `indent_lines`, `replace`

**Key imports**: annotations, itertools, re, textwrap, TYPE_CHECKING, Mapping, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `itertools`
- `re`
- `textwrap`
- `typing`: TYPE_CHECKING
- `collections.abc`: Mapping, Sequence


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

- **File Documentation**: `code_template.py_docs.md`
- **Keyword Index**: `code_template.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torchgen`):

- [`gen_functionalization_type.py_docs.md_docs.md`](./gen_functionalization_type.py_docs.md_docs.md)
- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`context.py_kw.md_docs.md`](./context.py_kw.md_docs.md)
- [`native_function_generation.py_kw.md_docs.md`](./native_function_generation.py_kw.md_docs.md)
- [`gen_schema_utils.py_docs.md_docs.md`](./gen_schema_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`gen_aoti_c_shim.py_docs.md_docs.md`](./gen_aoti_c_shim.py_docs.md_docs.md)
- [`local.py_docs.md_docs.md`](./local.py_docs.md_docs.md)
- [`gen.py_kw.md_docs.md`](./gen.py_kw.md_docs.md)
- [`gen_aoti_c_shim.py_kw.md_docs.md`](./gen_aoti_c_shim.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `code_template.py_docs.md_docs.md`
- **Keyword Index**: `code_template.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
