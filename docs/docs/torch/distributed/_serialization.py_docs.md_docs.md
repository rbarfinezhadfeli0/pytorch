# Documentation: `docs/torch/distributed/_serialization.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/_serialization.py_docs.md`
- **Size**: 7,466 bytes (7.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/_serialization.py`

## File Metadata

- **Path**: `torch/distributed/_serialization.py`
- **Size**: 4,584 bytes (4.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import pickle
from dataclasses import dataclass
from io import BufferedIOBase
from typing import Any

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch.serialization import _load, _save, DEFAULT_PROTOCOL, MAP_LOCATION


__all__: list[str] = []


@dataclass
class _Entry:
    key: str
    is_storage: bool
    length: int


_weights_only_unpickler._add_safe_globals([_Entry])


class _PseudoZipFile:
    def __init__(self) -> None:
        self.records: dict[str, tuple[object, int]] = {}

    def write_record(self, key: str, data: object, length: int) -> None:
        self.records[key] = (data, length)

    def write_to(self, f: BufferedIOBase) -> None:
        entries = []
        for key, (data, length) in self.records.items():
            entries.append(
                _Entry(
                    key=key,
                    is_storage=isinstance(data, torch.UntypedStorage),
                    length=length,
                )
            )

        pickle.dump(entries, f, protocol=DEFAULT_PROTOCOL)

        for data, _ in self.records.values():
            if isinstance(data, bytes):
                f.write(data)
            elif isinstance(data, str):
                f.write(data.encode("utf-8"))
            elif isinstance(data, torch.UntypedStorage):
                data._write_file(f, False, False, 1)
            else:
                raise TypeError(f"unknown type: {type(data)}")

    def read_from(self, f: BufferedIOBase) -> None:
        entries = _weights_only_unpickler.load(f)

        for entry in entries:
            data = f.read(entry.length)
            if entry.is_storage:
                if entry.length == 0:
                    storage = torch.UntypedStorage(0)
                else:
                    storage = torch.frombuffer(
                        data,
                        dtype=torch.uint8,
                    ).untyped_storage()

                self.records[entry.key] = (
                    storage,
                    entry.length,
                )
            else:
                self.records[entry.key] = (data, entry.length)

    def has_record(self, key: str) -> bool:
        return key in self.records

    def get_record(self, key: str) -> object:
        return self.records[key][0]

    def get_storage_from_record(
        self, key: str, _length: int, _type: int
    ) -> torch.Tensor:
        return torch.tensor(self.records[key][0], dtype=torch.uint8)

    def serialization_id(self) -> str:
        return "torchft"


def _streaming_save(
    obj: object,
    f: BufferedIOBase,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
) -> None:
    """
    Save the object to a file-like object in a streaming fashion compatible with
    network sockets.

    This behaves similarly to :func:`torch.save` with a few notable differences:

    * A non-seekable file like object can be used when loading.
    * No forwards/backwards compatibility is provided for the serialization
      format. This is only intended to be used with a single version of PyTorch
      with transient storage (i.e. sockets or temp files).
    * mmap is not supported

    See :func:`torch.save` for more details on specific arguments.
    """

    zip_file = _PseudoZipFile()
    _save(
        obj,
        zip_file=zip_file,
        pickle_module=pickle_module,
        pickle_protocol=pickle_protocol,
        _disable_byteorder_record=False,
    )
    zip_file.write_to(f)


def _streaming_load(
    f: BufferedIOBase,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = True,
    **pickle_load_args: Any,
) -> object:
    """
    Load the object from a file-like object in a streaming fashion compatible with
    network sockets.

    See :func:`_streaming_save` for more details about the streaming behavior.

    See :func:`torch.load` for more details on specific arguments.
    """
    if weights_only:
        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when explicit pickle_module is specified"
            )
        pickle_module = _weights_only_unpickler
    else:
        if pickle_module is None:
            pickle_module = pickle

    if "encoding" not in pickle_load_args:
        pickle_load_args["encoding"] = "utf-8"

    zip_file = _PseudoZipFile()
    zip_file.read_from(f)
    return _load(
        zip_file=zip_file,
        map_location=map_location,
        pickle_module=pickle_module,
        **pickle_load_args,
    )

```



## High-Level Overview


This Python file contains 3 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_Entry`, `_PseudoZipFile`

**Functions defined**: `__init__`, `write_record`, `write_to`, `read_from`, `has_record`, `get_record`, `get_storage_from_record`, `serialization_id`, `_streaming_save`, `_streaming_load`

**Key imports**: pickle, dataclass, BufferedIOBase, Any, torch, torch._weights_only_unpickler as _weights_only_unpickler, _load, _save, DEFAULT_PROTOCOL, MAP_LOCATION


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `pickle`
- `dataclasses`: dataclass
- `io`: BufferedIOBase
- `typing`: Any
- `torch`
- `torch._weights_only_unpickler as _weights_only_unpickler`
- `torch.serialization`: _load, _save, DEFAULT_PROTOCOL, MAP_LOCATION


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

Files in the same folder (`torch/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_mesh_layout.py_docs.md`](./_mesh_layout.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`c10d_logger.py_docs.md`](./c10d_logger.py_docs.md)
- [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`CONTRIBUTING.md_docs.md`](./CONTRIBUTING.md_docs.md)
- [`_functional_collectives_impl.py_docs.md`](./_functional_collectives_impl.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)


## Cross-References

- **File Documentation**: `_serialization.py_docs.md`
- **Keyword Index**: `_serialization.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

- **File Documentation**: `_serialization.py_docs.md_docs.md`
- **Keyword Index**: `_serialization.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
