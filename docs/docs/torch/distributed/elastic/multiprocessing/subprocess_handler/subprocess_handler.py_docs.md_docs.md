# Documentation: `docs/torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py_docs.md`
- **Size**: 5,401 bytes (5.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py`

## File Metadata

- **Path**: `torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py`
- **Size**: 2,719 bytes (2.66 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import signal
import sys
from subprocess import Popen
from typing import Any, Optional

from torch.numa.binding import maybe_wrap_command_args_with_numa_binding, NumaOptions


__all__ = ["SubprocessHandler"]

IS_WINDOWS = sys.platform == "win32"


def _get_default_signal() -> signal.Signals:
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821
    else:
        return signal.SIGTERM


class SubprocessHandler:
    """
    Convenience wrapper around python's ``subprocess.Popen``. Keeps track of
    meta-objects associated to the process (e.g. stdout and stderr redirect fds).
    """

    def __init__(
        self,
        entrypoint: str,
        args: tuple,
        env: dict[str, str],
        stdout: Optional[str],
        stderr: Optional[str],
        local_rank_id: int,
        numa_options: Optional[NumaOptions],
    ):
        self._stdout = open(stdout, "w") if stdout else None
        self._stderr = open(stderr, "w") if stderr else None
        # inherit parent environment vars
        env_vars = os.environ.copy()
        env_vars.update(env)

        args_str = (entrypoint, *[str(e) for e in args])
        args_str = maybe_wrap_command_args_with_numa_binding(
            args_str,
            gpu_index=local_rank_id,
            numa_options=numa_options,
        )

        self.local_rank_id = local_rank_id

        self.proc: Popen = self._popen(args_str, env_vars)

    def _popen(self, args: tuple, env: dict[str, str]) -> Popen:
        kwargs: dict[str, Any] = {}
        if not IS_WINDOWS:
            kwargs["start_new_session"] = True

        return Popen(
            # pyre-fixme[6]: Expected `Union[typing.Sequence[Union[_PathLike[bytes],
            #  _PathLike[str], bytes, str]], bytes, str]` for 1st param but got
            #  `Tuple[str, *Tuple[Any, ...]]`.
            args=args,
            env=env,
            stdout=self._stdout,
            stderr=self._stderr,
            **kwargs,
        )

    def close(self, death_sig: Optional[signal.Signals] = None) -> None:
        if not death_sig:
            death_sig = _get_default_signal()
        if IS_WINDOWS:
            self.proc.send_signal(death_sig)
        else:
            os.killpg(self.proc.pid, death_sig)
        if self._stdout:
            self._stdout.close()
        if self._stderr:
            self._stderr.close()

```



## High-Level Overview

"""Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""    if IS_WINDOWS:        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821    else:        return signal.SIGTERMclass SubprocessHandler:

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SubprocessHandler`

**Functions defined**: `_get_default_signal`, `__init__`, `_popen`, `close`

**Key imports**: os, signal, sys, Popen, Any, Optional, maybe_wrap_command_args_with_numa_binding, NumaOptions


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/multiprocessing/subprocess_handler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `signal`
- `sys`
- `subprocess`: Popen
- `typing`: Any, Optional
- `torch.numa.binding`: maybe_wrap_command_args_with_numa_binding, NumaOptions


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/elastic/multiprocessing/subprocess_handler`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`handlers.py_docs.md`](./handlers.py_docs.md)


## Cross-References

- **File Documentation**: `subprocess_handler.py_docs.md`
- **Keyword Index**: `subprocess_handler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/elastic/multiprocessing/subprocess_handler`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/elastic/multiprocessing/subprocess_handler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/elastic/multiprocessing/subprocess_handler`):

- [`handlers.py_kw.md_docs.md`](./handlers.py_kw.md_docs.md)
- [`subprocess_handler.py_kw.md_docs.md`](./subprocess_handler.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`handlers.py_docs.md_docs.md`](./handlers.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `subprocess_handler.py_docs.md_docs.md`
- **Keyword Index**: `subprocess_handler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
