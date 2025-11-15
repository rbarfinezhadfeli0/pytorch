# Documentation: `torch/testing/_internal/distributed/checkpoint_utils.py`

## File Metadata

- **Path**: `torch/testing/_internal/distributed/checkpoint_utils.py`
- **Size**: 6,187 bytes (6.04 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates

import io
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from functools import wraps
from typing import Any, cast, IO, Optional

# introduced as collections.abc.Buffer in Python 3.12
from typing_extensions import Buffer

import torch.distributed as dist
from torch.distributed.checkpoint._extension import (
    ExtensionRegistry,
    StreamTransformExtension,
)


class Rot13Example(StreamTransformExtension):
    """
    This is an example stream transform extension which just does rot13 on each
    alphanumeric character of the stream.  It is mainly intended as a demonstration
    and for testing; there isn't a production use case for this.
    """

    def __init__(self, chunk_size: int = io.DEFAULT_BUFFER_SIZE) -> None:
        super().__init__()
        self._chunk_size = chunk_size

    @staticmethod
    def from_descriptor(version: str) -> "Rot13Example":
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension {version=}")
        return Rot13Example()

    @staticmethod
    def registry_name() -> str:
        return "stream.rot13"

    def get_descriptor(self) -> str:
        return f"{self.registry_name()}/1"

    @staticmethod
    def _rot13bytes(b: Buffer, count: int) -> None:
        b = memoryview(b)
        for i in range(count):
            ch = b[i]
            if ch >= ord("A") and ch <= ord("Z"):
                ch += ord("a") - ord("A")
            elif ch >= ord("a") and ch <= ord("z"):
                ch += ord("A") - ord("a")
            b[i] = ch

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        class Writer(io.RawIOBase):
            def __init__(self, output: IO[bytes]) -> None:
                self.output = output

            def writeable(self) -> bool:
                return True

            def write(self, b: Buffer) -> Optional[int]:
                # Don't mutate the input
                chunk = bytearray(b)
                Rot13Example._rot13bytes(chunk, len(chunk))
                return self.output.write(chunk)

            def flush(self) -> None:
                self.output.flush()

        return cast(IO[bytes], Writer(output))

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        class Reader(io.RawIOBase):
            def __init__(self, input: IO[bytes]) -> None:
                self.input = input

            def readable(self) -> bool:
                return True

            def readinto(self, b: Buffer) -> Optional[int]:
                if hasattr(self.input, "readinto"):
                    count = self.input.readinto(b)
                else:
                    # It's possible self.input is an IO[bytes] with no readinto method.
                    # In that case, we emulate with a read and copy.  In practice,
                    # all of the current concrete extensions have readinto.
                    view = memoryview(b)
                    r = self.input.read(len(view))
                    if r is None:
                        count = None
                    else:
                        count = len(r)
                        view[:count] = r
                if count == 0 or count is None:
                    return count

                Rot13Example._rot13bytes(b, count)
                return count

            def seekable(self) -> bool:
                return self.input.seekable()

            def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
                return self.input.seek(offset, whence)

            def tell(self) -> int:
                return self.input.tell()

        return cast(IO[bytes], Reader(input))


def get_test_extension_registry() -> ExtensionRegistry:
    registry = ExtensionRegistry()
    registry.register(Rot13Example)
    return registry


def with_temp_dir(
    func: Optional[Callable] = None,
) -> Optional[Callable]:
    """
    Wrapper to initialize temp directory for distributed checkpoint.
    """
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: tuple[object], **kwargs: dict[str, Any]) -> None:
        if dist.is_initialized():
            # Only create temp_dir when rank is 0
            if dist.get_rank() == 0:
                temp_dir = tempfile.mkdtemp()
                print(f"Using temp directory: {temp_dir}")
            else:
                temp_dir = ""
            object_list = [temp_dir]

            # Broadcast temp_dir to all the other ranks
            os.sync()
            dist.broadcast_object_list(object_list)
            self.temp_dir = object_list[0]
            os.sync()
        else:
            temp_dir = tempfile.mkdtemp()
            print(f"No process group initialized, using temp directory: {temp_dir}")
            self.temp_dir = temp_dir

        try:
            func(self, *args, **kwargs)
        finally:
            if dist.is_initialized() and dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            else:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    return wrapper


def with_checkpoint_logging(
    func: Optional[Callable] = None,
    logger_name: str = "torch.distributed.checkpoint",
    level: int = logging.INFO,
) -> Optional[Callable]:
    """
    Wrapper to configure checkpoint logging for distributed tests.

    Args:
        func: The test function to wrap
        logger_name: Name of the logger to configure (default: 'torch.distributed.checkpoint')
        level: Logging level to set (default: logging.INFO)
    """
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: tuple[object], **kwargs: dict[str, Any]) -> None:
        # Get the logger and store original level
        target_logger = logging.getLogger(logger_name)
        original_level = target_logger.level

        # Set the desired logging level
        target_logger.setLevel(level)

        try:
            func(self, *args, **kwargs)
        finally:
            # Restore original logging level
            target_logger.setLevel(original_level)

    return wrapper

```



## High-Level Overview

"""    This is an example stream transform extension which just does rot13 on each    alphanumeric character of the stream.  It is mainly intended as a demonstration    and for testing; there isn't a production use case for this.

This Python file contains 3 class(es) and 22 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Rot13Example`, `Writer`, `Reader`

**Functions defined**: `__init__`, `from_descriptor`, `registry_name`, `get_descriptor`, `_rot13bytes`, `transform_to`, `__init__`, `writeable`, `write`, `flush`, `transform_from`, `__init__`, `readable`, `readinto`, `seekable`, `seek`, `tell`, `get_test_extension_registry`, `with_temp_dir`, `wrapper`

**Key imports**: io, logging, os, shutil, tempfile, Callable, wraps, Any, cast, IO, Optional, Buffer, torch.distributed as dist


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `logging`
- `os`
- `shutil`
- `tempfile`
- `collections.abc`: Callable
- `functools`: wraps
- `typing`: Any, cast, IO, Optional
- `typing_extensions`: Buffer
- `torch.distributed as dist`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python torch/testing/_internal/distributed/checkpoint_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ddp_under_dist_autograd_test.py_docs.md`](./ddp_under_dist_autograd_test.py_docs.md)
- [`fake_pg.py_docs.md`](./fake_pg.py_docs.md)
- [`multi_threaded_pg.py_docs.md`](./multi_threaded_pg.py_docs.md)
- [`common_state_dict.py_docs.md`](./common_state_dict.py_docs.md)
- [`distributed_utils.py_docs.md`](./distributed_utils.py_docs.md)
- [`rpc_utils.py_docs.md`](./rpc_utils.py_docs.md)
- [`distributed_test.py_docs.md`](./distributed_test.py_docs.md)


## Cross-References

- **File Documentation**: `checkpoint_utils.py_docs.md`
- **Keyword Index**: `checkpoint_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
