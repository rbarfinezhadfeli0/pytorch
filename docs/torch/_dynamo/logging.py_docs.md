# Documentation: `torch/_dynamo/logging.py`

## File Metadata

- **Path**: `torch/_dynamo/logging.py`
- **Size**: 2,215 bytes (2.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Logging utilities for Dynamo and Inductor.

This module provides specialized logging functionality including:
- Step-based logging that prepends step numbers to log messages
- Progress bar management for compilation phases
- Centralized logger management for Dynamo and Inductor components

The logging system helps track the progress of compilation phases and provides structured
logging output for debugging and monitoring.
"""

import itertools
import logging
from collections.abc import Callable
from typing import Any

from torch.hub import _Faketqdm, tqdm


# Disable progress bar by default, not in dynamo config because otherwise get a circular import
disable_progress = True


# Return all loggers that torchdynamo/torchinductor is responsible for
def get_loggers() -> list[logging.Logger]:
    return [
        logging.getLogger("torch.fx.experimental.symbolic_shapes"),
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch._inductor"),
    ]


# Creates a logging function that logs a message with a step # prepended.
# get_step_logger should be lazily called (i.e. at runtime, not at module-load time)
# so that step numbers are initialized properly. e.g.:

# @functools.cache
# def _step_logger():
#     return get_step_logger(logging.getLogger(...))

# def fn():
#     _step_logger()(logging.INFO, "msg")

_step_counter = itertools.count(1)

# Update num_steps if more phases are added: Dynamo, AOT, Backend
# This is very inductor centric
# _inductor.utils.has_triton() gives a circular import error here

if not disable_progress:
    try:
        import triton  # noqa: F401

        num_steps = 3
    except ImportError:
        num_steps = 2
    pbar = tqdm(total=num_steps, desc="torch.compile()", delay=0)


def get_step_logger(logger: logging.Logger) -> Callable[..., None]:
    if not disable_progress:
        pbar.update(1)
        if not isinstance(pbar, _Faketqdm):
            pbar.set_postfix_str(f"{logger.name}")

    step = next(_step_counter)

    def log(level: int, msg: str, **kwargs: Any) -> None:
        if "stacklevel" not in kwargs:
            kwargs["stacklevel"] = 2
        logger.log(level, "Step %s: %s", step, msg, **kwargs)

    return log

```



## High-Level Overview

"""Logging utilities for Dynamo and Inductor.This module provides specialized logging functionality including:- Step-based logging that prepends step numbers to log messages- Progress bar management for compilation phases- Centralized logger management for Dynamo and Inductor componentsThe logging system helps track the progress of compilation phases and provides structuredlogging output for debugging and monitoring.

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_loggers`, `_step_logger`, `fn`, `get_step_logger`, `log`

**Key imports**: itertools, logging, Callable, Any, _Faketqdm, tqdm, disable_progress , error here, triton  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `logging`
- `collections.abc`: Callable
- `typing`: Any
- `torch.hub`: _Faketqdm, tqdm
- `disable_progress `
- `error here`
- `triton  `


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `logging.py_docs.md`
- **Keyword Index**: `logging.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
