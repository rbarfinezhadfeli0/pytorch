# Documentation: `docs/torch/distributed/c10d_logger.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/c10d_logger.py_docs.md`
- **Size**: 6,017 bytes (5.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/c10d_logger.py`

## File Metadata

- **Path**: `torch/distributed/c10d_logger.py`
- **Size**: 3,185 bytes (3.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.distributed as dist
from torch.distributed.logging_handlers import _log_handlers
from torch.monitor import _WaitCounter


__all__: list[str] = []

_DEFAULT_DESTINATION = "default"


def _get_or_create_logger(destination: str = _DEFAULT_DESTINATION) -> logging.Logger:
    logging_handler, log_handler_name = _get_logging_handler(destination)
    logger = logging.getLogger(f"c10d-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str = _DEFAULT_DESTINATION,
) -> tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = f"{type(log_handler).__name__}-{destination}"
    return (log_handler, log_handler_name)


# pyrefly: ignore [unknown-name]
global _c10d_logger
_c10d_logger = _get_or_create_logger()


def _get_msg_dict(func_name, *args, **kwargs) -> dict[str, Any]:
    if dist.is_initialized():
        group = kwargs.get("group") or kwargs.get("process_group")
        msg_dict = {
            "func_name": f"{func_name}",
            "pg_name": f"{dist._get_process_group_name(kwargs.get('pg'))}",  # type: ignore[arg-type]
            "backend": f"{dist.get_backend(group)}",
            "world_size": f"{dist.get_world_size()}",
            "group_size": f"{dist.get_world_size(group)}",
            "global_rank": f"{dist.get_rank()}",
            "local_rank": f"{dist.get_rank(group)}",
        }
        if msg_dict["backend"] == "nccl":
            nccl_version = torch.cuda.nccl.version()
            msg_dict["nccl_version"] = ".".join(str(v) for v in nccl_version)
    else:
        msg_dict = {
            "func_name": f"{func_name}",
        }
    return msg_dict


_T = TypeVar("_T")
_P = ParamSpec("_P")


def _exception_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return func(*args, **kwargs)
        except Exception as error:
            msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)
            msg_dict["error"] = f"{error}"
            _c10d_logger.debug(msg_dict)
            raise

    return wrapper


def _time_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        with _WaitCounter(f"pytorch.wait_counter.c10d.{func.__name__}").guard():
            func_return = func(*args, **kwargs)
        return func_return

    return wrapper

```



## High-Level Overview


This Python file contains 0 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_or_create_logger`, `_get_logging_handler`, `_get_msg_dict`, `_exception_logger`, `wrapper`, `_time_logger`, `wrapper`

**Key imports**: functools, logging, Callable, Any, TypeVar, ParamSpec, torch, torch.distributed as dist, _log_handlers, _WaitCounter


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `logging`
- `collections.abc`: Callable
- `typing`: Any, TypeVar
- `typing_extensions`: ParamSpec
- `torch`
- `torch.distributed as dist`
- `torch.distributed.logging_handlers`: _log_handlers
- `torch.monitor`: _WaitCounter


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_mesh_layout.py_docs.md`](./_mesh_layout.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`CONTRIBUTING.md_docs.md`](./CONTRIBUTING.md_docs.md)
- [`_functional_collectives_impl.py_docs.md`](./_functional_collectives_impl.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)


## Cross-References

- **File Documentation**: `c10d_logger.py_docs.md`
- **Keyword Index**: `c10d_logger.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/distributed`):

- [`_mesh_layout.py_docs.md_docs.md`](./_mesh_layout.py_docs.md_docs.md)
- [`run.py_docs.md_docs.md`](./run.py_docs.md_docs.md)
- [`device_mesh.py_docs.md_docs.md`](./device_mesh.py_docs.md_docs.md)
- [`_composable_state.py_docs.md_docs.md`](./_composable_state.py_docs.md_docs.md)
- [`run.py_kw.md_docs.md`](./run.py_kw.md_docs.md)
- [`_dist2.py_kw.md_docs.md`](./_dist2.py_kw.md_docs.md)
- [`_state_dict_utils.py_kw.md_docs.md`](./_state_dict_utils.py_kw.md_docs.md)
- [`rendezvous.py_kw.md_docs.md`](./rendezvous.py_kw.md_docs.md)
- [`rendezvous.py_docs.md_docs.md`](./rendezvous.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `c10d_logger.py_docs.md_docs.md`
- **Keyword Index**: `c10d_logger.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
