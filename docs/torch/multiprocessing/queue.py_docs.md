# Documentation: `torch/multiprocessing/queue.py`

## File Metadata

- **Path**: `torch/multiprocessing/queue.py`
- **Size**: 1,477 bytes (1.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import io
import multiprocessing.queues
import pickle
from multiprocessing.reduction import ForkingPickler


class ConnectionWrapper:
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler for object serialization."""

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if "conn" in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'conn'")


class Queue(multiprocessing.queues.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
        self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
            self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        super()._make_methods()  # type: ignore[misc]

```



## High-Level Overview

"""Proxy class for _multiprocessing.Connection which uses ForkingPickler for object serialization."""    def __init__(self, conn):        self.conn = conn    def send(self, obj):        buf = io.BytesIO()        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)        self.send_bytes(buf.getvalue())    def recv(self):        buf = self.recv_bytes()        return pickle.loads(buf)    def __getattr__(self, name):        if "conn" in self.__dict__:            return getattr(self.conn, name)        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'conn'")class Queue(multiprocessing.queues.Queue):    def __init__(self, *args, **kwargs):        super().__init__(*args, **kwargs)        self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)        self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)        self._send = self._writer.send        self._recv = self._reader.recvclass SimpleQueue(multiprocessing.queues.SimpleQueue):    def _make_methods(self):        if not isinstance(self._reader, ConnectionWrapper):            self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)            self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)        super()._make_methods()  # type: ignore[misc]

This Python file contains 4 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConnectionWrapper`, `Queue`, `SimpleQueue`

**Functions defined**: `__init__`, `send`, `recv`, `__getattr__`, `__init__`, `_make_methods`

**Key imports**: io, multiprocessing.queues, pickle, ForkingPickler


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/multiprocessing`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `multiprocessing.queues`
- `pickle`
- `multiprocessing.reduction`: ForkingPickler


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/multiprocessing`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`pool.py_docs.md`](./pool.py_docs.md)
- [`_atfork.py_docs.md`](./_atfork.py_docs.md)
- [`spawn.py_docs.md`](./spawn.py_docs.md)
- [`cuda_multiprocessing.md_docs.md`](./cuda_multiprocessing.md_docs.md)
- [`reductions.py_docs.md`](./reductions.py_docs.md)


## Cross-References

- **File Documentation**: `queue.py_docs.md`
- **Keyword Index**: `queue.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
