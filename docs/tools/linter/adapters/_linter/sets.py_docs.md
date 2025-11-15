# Documentation: `tools/linter/adapters/_linter/sets.py`

## File Metadata

- **Path**: `tools/linter/adapters/_linter/sets.py`
- **Size**: 2,308 bytes (2.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import dataclasses as dc
import token
from functools import cached_property
from typing import TYPE_CHECKING

from . import EMPTY_TOKENS
from .bracket_pairs import bracket_pairs


if TYPE_CHECKING:
    from tokenize import TokenInfo


@dc.dataclass
class LineWithSets:
    """A logical line of Python tokens, terminated by a NEWLINE or the end of file"""

    tokens: list[TokenInfo]

    @cached_property
    def sets(self) -> list[TokenInfo]:
        """A list of tokens which use the built-in set symbol"""
        return [t for i, t in enumerate(self.tokens) if self.is_set(i)]

    @cached_property
    def braced_sets(self) -> list[list[TokenInfo]]:
        """A list of lists of tokens, each representing a braced set, like {1}"""
        return [
            self.tokens[b : e + 1]
            for b, e in self.bracket_pairs.items()
            if self.is_braced_set(b, e)
        ]

    @cached_property
    def bracket_pairs(self) -> dict[int, int]:
        return bracket_pairs(self.tokens)

    def is_set(self, i: int) -> bool:
        t = self.tokens[i]
        after = i < len(self.tokens) - 1 and self.tokens[i + 1]
        if t.string == "Set" and t.type == token.NAME:
            # pyrefly: ignore [bad-return]
            return after and after.string == "[" and after.type == token.OP
        return (
            (t.string == "set" and t.type == token.NAME)
            and not (i and self.tokens[i - 1].string in ("def", "."))
            and not (after and after.string == "=" and after.type == token.OP)
        )

    def is_braced_set(self, begin: int, end: int) -> bool:
        if (
            begin + 1 == end
            or self.tokens[begin].string != "{"
            or begin
            and self.tokens[begin - 1].string == "in"  # skip `x in {1, 2, 3}`
        ):
            return False

        i = begin + 1
        empty = True
        while i < end:
            t = self.tokens[i]
            if t.type == token.OP and t.string in (":", "**"):
                return False
            if brace_end := self.bracket_pairs.get(i):
                # Skip to the end of a subexpression
                i = brace_end
            elif t.type not in EMPTY_TOKENS:
                empty = False
            i += 1
        return not empty

```



## High-Level Overview

"""A logical line of Python tokens, terminated by a NEWLINE or the end of file"""    tokens: list[TokenInfo]    @cached_property    def sets(self) -> list[TokenInfo]:

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LineWithSets`

**Functions defined**: `sets`, `braced_sets`, `bracket_pairs`, `is_set`, `is_braced_set`

**Key imports**: annotations, dataclasses as dc, token, cached_property, TYPE_CHECKING, EMPTY_TOKENS, bracket_pairs, TokenInfo


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters/_linter`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `dataclasses as dc`
- `token`
- `functools`: cached_property
- `typing`: TYPE_CHECKING
- `.`: EMPTY_TOKENS
- `.bracket_pairs`: bracket_pairs
- `tokenize`: TokenInfo


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`tools/linter/adapters/_linter`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`messages.py_docs.md`](./messages.py_docs.md)
- [`block.py_docs.md`](./block.py_docs.md)
- [`argument_parser.py_docs.md`](./argument_parser.py_docs.md)
- [`python_file.py_docs.md`](./python_file.py_docs.md)
- [`blocks.py_docs.md`](./blocks.py_docs.md)
- [`bracket_pairs.py_docs.md`](./bracket_pairs.py_docs.md)
- [`file_linter.py_docs.md`](./file_linter.py_docs.md)


## Cross-References

- **File Documentation**: `sets.py_docs.md`
- **Keyword Index**: `sets.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
