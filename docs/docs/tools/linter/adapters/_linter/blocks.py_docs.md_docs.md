# Documentation: `docs/tools/linter/adapters/_linter/blocks.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/_linter/blocks.py_docs.md`
- **Size**: 6,279 bytes (6.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/_linter/blocks.py`

## File Metadata

- **Path**: `tools/linter/adapters/_linter/blocks.py`
- **Size**: 3,720 bytes (3.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import token
from typing import NamedTuple, TYPE_CHECKING

from . import EMPTY_TOKENS, ParseError
from .block import Block


if TYPE_CHECKING:
    from collections.abc import Sequence
    from tokenize import TokenInfo


class BlocksResult(NamedTuple):
    blocks: list[Block]
    errors: dict[str, str]


def blocks(tokens: Sequence[TokenInfo]) -> BlocksResult:
    blocks: list[Block] = []
    indent_to_dedent = _make_indent_dict(tokens)
    errors: dict[str, str] = {}

    def starts_block(t: TokenInfo) -> bool:
        return t.type == token.NAME and t.string in ("class", "def")

    it = (i for i, t in enumerate(tokens) if starts_block(t))
    blocks = [_make_block(tokens, i, indent_to_dedent, errors) for i in it]

    for i, parent in enumerate(blocks):
        for j in range(i + 1, len(blocks)):
            if parent.contains(child := blocks[j]):
                child.parent = i
                parent.children.append(j)
            else:
                break

    for i, b in enumerate(blocks):
        b.index = i
        parents = [b]
        while (p := parents[-1].parent) is not None:
            parents.append(blocks[p])
        parents = parents[1:]

        b.is_local = not all(p.is_class for p in parents)
        b.is_method = not b.is_class and bool(parents) and parents[0].is_class

    _add_full_names(blocks, [b for b in blocks if b.parent is None])
    return BlocksResult(blocks, errors)


def _make_indent_dict(tokens: Sequence[TokenInfo]) -> dict[int, int]:
    dedents = dict[int, int]()
    stack = list[int]()

    for i, t in enumerate(tokens):
        if t.type == token.INDENT:
            stack.append(i)
        elif t.type == token.DEDENT:
            dedents[stack.pop()] = i

    return dedents


def _docstring(tokens: Sequence[TokenInfo], start: int) -> str:
    for i in range(start + 1, len(tokens)):
        tk = tokens[i]
        if tk.type == token.STRING:
            return tk.string
        if tk.type not in EMPTY_TOKENS:
            return ""
    return ""


def _add_full_names(
    blocks: Sequence[Block], children: Sequence[Block], prefix: str = ""
) -> None:
    # Would be trivial except that there can be duplicate names at any level
    dupes: dict[str, list[Block]] = {}
    for b in children:
        dupes.setdefault(b.name, []).append(b)

    for dl in dupes.values():
        for i, b in enumerate(dl):
            suffix = f"[{i + 1}]" if len(dl) > 1 else ""
            b.full_name = prefix + b.name + suffix

    for b in children:
        if kids := [blocks[i] for i in b.children]:
            _add_full_names(blocks, kids, b.full_name + ".")


def _make_block(
    tokens: Sequence[TokenInfo],
    begin: int,
    indent_to_dedent: dict[int, int],
    errors: dict[str, str],
) -> Block:
    def next_token(start: int, token_type: int, error: str) -> int:
        for i in range(start, len(tokens)):
            if tokens[i].type == token_type:
                return i
        raise ParseError(tokens[-1], error)

    t = tokens[begin]
    category = Block.Category[t.string.upper()]
    indent = -1
    dedent = -1
    docstring = ""
    name = "(not found)"
    try:
        ni = next_token(begin + 1, token.NAME, "Definition but no name")
        name = tokens[ni].string
        indent = next_token(ni + 1, token.INDENT, "Definition but no indent")
        dedent = indent_to_dedent[indent]
        docstring = _docstring(tokens, indent)
    except ParseError as e:
        errors[t.line] = " ".join(e.args)

    return Block(
        begin=begin,
        category=category,
        dedent=dedent,
        docstring=docstring,
        indent=indent,
        name=name,
        tokens=tokens,
    )

```



## High-Level Overview


This Python file contains 4 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BlocksResult`

**Functions defined**: `blocks`, `starts_block`, `_make_indent_dict`, `_docstring`, `_add_full_names`, `_make_block`, `next_token`

**Key imports**: annotations, token, NamedTuple, TYPE_CHECKING, EMPTY_TOKENS, ParseError, Block, Sequence, TokenInfo


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters/_linter`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `token`
- `typing`: NamedTuple, TYPE_CHECKING
- `.`: EMPTY_TOKENS, ParseError
- `.block`: Block
- `collections.abc`: Sequence
- `tokenize`: TokenInfo


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`tools/linter/adapters/_linter`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`messages.py_docs.md`](./messages.py_docs.md)
- [`sets.py_docs.md`](./sets.py_docs.md)
- [`block.py_docs.md`](./block.py_docs.md)
- [`argument_parser.py_docs.md`](./argument_parser.py_docs.md)
- [`python_file.py_docs.md`](./python_file.py_docs.md)
- [`bracket_pairs.py_docs.md`](./bracket_pairs.py_docs.md)
- [`file_linter.py_docs.md`](./file_linter.py_docs.md)


## Cross-References

- **File Documentation**: `blocks.py_docs.md`
- **Keyword Index**: `blocks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/linter/adapters/_linter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/linter/adapters/_linter`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/tools/linter/adapters/_linter`):

- [`argument_parser.py_kw.md_docs.md`](./argument_parser.py_kw.md_docs.md)
- [`messages.py_docs.md_docs.md`](./messages.py_docs.md_docs.md)
- [`blocks.py_kw.md_docs.md`](./blocks.py_kw.md_docs.md)
- [`block.py_docs.md_docs.md`](./block.py_docs.md_docs.md)
- [`sets.py_kw.md_docs.md`](./sets.py_kw.md_docs.md)
- [`argument_parser.py_docs.md_docs.md`](./argument_parser.py_docs.md_docs.md)
- [`block.py_kw.md_docs.md`](./block.py_kw.md_docs.md)
- [`sets.py_docs.md_docs.md`](./sets.py_docs.md_docs.md)
- [`messages.py_kw.md_docs.md`](./messages.py_kw.md_docs.md)
- [`file_linter.py_kw.md_docs.md`](./file_linter.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `blocks.py_docs.md_docs.md`
- **Keyword Index**: `blocks.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
