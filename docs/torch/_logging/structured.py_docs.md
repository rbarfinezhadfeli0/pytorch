# Documentation: `torch/_logging/structured.py`

## File Metadata

- **Path**: `torch/_logging/structured.py`
- **Size**: 2,922 bytes (2.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Utilities for converting data types into structured JSON for dumping.
"""

import inspect
import os
import traceback
from collections.abc import Sequence
from typing import Any, Optional

import torch._logging._internal


INTERN_TABLE: dict[str, int] = {}


DUMPED_FILES: set[str] = set()


def intern_string(s: Optional[str]) -> int:
    if s is None:
        return -1

    r = INTERN_TABLE.get(s)
    if r is None:
        r = len(INTERN_TABLE)
        INTERN_TABLE[s] = r
        torch._logging._internal.trace_structured(
            "str", lambda: (s, r), suppress_context=True
        )
    return r


def dump_file(filename: str) -> None:
    if "eval_with_key" not in filename:
        return
    if filename in DUMPED_FILES:
        return
    DUMPED_FILES.add(filename)
    from torch.fx.graph_module import _loader

    torch._logging._internal.trace_structured(
        "dump_file",
        metadata_fn=lambda: {
            "name": filename,
        },
        payload_fn=lambda: _loader.get_source(filename),
    )


def from_traceback(tb: Sequence[traceback.FrameSummary]) -> list[dict[str, Any]]:
    # dict naming convention here coincides with
    # python/combined_traceback.cpp
    r = [
        {
            "line": frame.lineno,
            "name": frame.name,
            "filename": intern_string(frame.filename),
            "loc": frame.line,
        }
        for frame in tb
    ]
    return r


def get_user_stack(num_frames: int) -> list[dict[str, Any]]:
    from torch._guards import TracingContext
    from torch.utils._traceback import CapturedTraceback

    user_tb = TracingContext.extract_stack()
    if user_tb:
        return from_traceback(user_tb[-1 * num_frames :])

    tb = CapturedTraceback.extract().summary()

    # Filter out frames that are within the torch/ codebase
    torch_filepath = os.path.dirname(inspect.getfile(torch)) + os.path.sep
    for i, frame in enumerate(reversed(tb)):
        if torch_filepath not in frame.filename:
            # Only display `num_frames` frames in the traceback
            filtered_tb = tb[len(tb) - i - num_frames : len(tb) - i]
            return from_traceback(filtered_tb)

    return from_traceback(tb[-1 * num_frames :])


def get_framework_stack(
    num_frames: int = 25, cpp: bool = False
) -> list[dict[str, Any]]:
    """
    Returns the traceback for the user stack and the framework stack
    """
    from torch.fx.experimental.symbolic_shapes import uninteresting_files
    from torch.utils._traceback import CapturedTraceback

    tb = CapturedTraceback.extract(cpp=cpp).summary()
    tb = [
        frame
        for frame in tb
        if (
            (
                frame.filename.endswith(".py")
                and frame.filename not in uninteresting_files()
            )
            or ("at::" in frame.name or "torch::" in frame.name)
        )
    ]

    return from_traceback(tb[-1 * num_frames :])

```



## High-Level Overview

"""Utilities for converting data types into structured JSON for dumping.

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `intern_string`, `dump_file`, `from_traceback`, `get_user_stack`, `get_framework_stack`

**Key imports**: inspect, os, traceback, Sequence, Any, Optional, torch._logging._internal, _loader, TracingContext, CapturedTraceback, uninteresting_files


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_logging`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `inspect`
- `os`
- `traceback`
- `collections.abc`: Sequence
- `typing`: Any, Optional
- `torch._logging._internal`
- `torch.fx.graph_module`: _loader
- `torch._guards`: TracingContext
- `torch.utils._traceback`: CapturedTraceback
- `torch.fx.experimental.symbolic_shapes`: uninteresting_files


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

Files in the same folder (`torch/_logging`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_registrations.py_docs.md`](./_registrations.py_docs.md)
- [`_internal.py_docs.md`](./_internal.py_docs.md)
- [`scribe.py_docs.md`](./scribe.py_docs.md)


## Cross-References

- **File Documentation**: `structured.py_docs.md`
- **Keyword Index**: `structured.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
