# Documentation: `docs/torch/distributed/checkpoint/_extension.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/_extension.py_docs.md`
- **Size**: 11,293 bytes (11.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/_extension.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_extension.py`
- **Size**: 7,790 bytes (7.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates

import abc
import io
from collections.abc import Sequence
from typing import cast, IO, Optional

# introduced as collections.abc.Buffer in Python 3.12
from typing_extensions import Buffer

from torch._utils import try_import


# NOTE: everything in this file is experimental, and subject to
# change.  Feedback and bug fixes are always welcome.

pyzstd_module_name = "pyzstd"
pyzstd = try_import(pyzstd_module_name)
zstandard_module_name = "zstandard"
zstandard = try_import(zstandard_module_name)


__all__ = [
    "Extension",
    "StreamTransformExtension",
    "ZStandard",
    "ExtensionRegistry",
]


class Extension(abc.ABC):
    """
    Extensions provide modular additions to functionality within distributed checkpointing,
    which affect the layout or format of the written artifacts.  Extensions may be
    built into pytorch, or provided externally.

    When writing, the caller provides a list of extension instances of the appropriate
    type.  Each extension can output a descriptor which is used to reconstitute the
    extension at read-time.
    """

    @staticmethod
    @abc.abstractmethod
    def registry_name() -> str:
        """
        See ExtensionRegistry.from_descriptor_list
        """

    @staticmethod
    @abc.abstractmethod
    def from_descriptor(version: str) -> "Extension":
        """
        See ExtensionRegistry.from_descriptor_list
        """

    @abc.abstractmethod
    def get_descriptor(self) -> str:
        """
        Return descriptor name to be included in metadata.  The form should be
        "extension_name[@local-domain][/version]".
        """


class StreamTransformExtension(Extension):
    """
    An extension which performs transformation on a byte stream, such as compression
    or encryption.

    Implementations should try to be memory friendly and performant.  For example, don't
    read the whole input, then transform it, and write it back.  If at all possible, do it in
    chunks.  But, don't read/transform/write one byte at a time, either.
    """

    @abc.abstractmethod
    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        """
        Takes a writeable output stream, and generates a new stream which implements the
        output transform.  Input data written to the returned stream will be transformed
        and written to the `output` argument stream.
        """

    @abc.abstractmethod
    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        """
        Takes a readable input stream, and generates a new stream which implements the
        input transform.  When the returned stream is read, data will be read from the
        'input' stream, transformed, and returned.
        """


class ZStandard(StreamTransformExtension):
    @staticmethod
    def is_available() -> bool:
        return zstandard is not None or pyzstd is not None

    @staticmethod
    # pyrefly: ignore [bad-override]
    def from_descriptor(version: str) -> "ZStandard":
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension {version=}")
        if not ZStandard.is_available():
            raise ValueError(
                f"Stream with ZStandard compression cannot be processed because "
                f"no module named '{zstandard_module_name}' or '{pyzstd_module_name}'"
            )
        return ZStandard()

    @staticmethod
    def registry_name() -> str:
        return "stream.zstd"

    def __init__(self) -> None:
        super().__init__()
        if not ZStandard.is_available():
            raise ValueError(
                f"ZStandard extension is unavailable because no module named '{zstandard_module_name}' or '{pyzstd_module_name}'"
            )

    def get_descriptor(self) -> str:
        return f"{self.registry_name()}/1"

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        if zstandard is not None:
            compressor = zstandard.ZstdCompressor()  # type: ignore[union-attr]
            return compressor.stream_writer(output)

        class Writer(io.RawIOBase):
            def __init__(self, output: IO[bytes]) -> None:
                self.output = output
                self.compressor = pyzstd.ZstdCompressor()  # type: ignore[union-attr]

            def writeable(self) -> bool:
                return True

            def write(self, b: Buffer) -> Optional[int]:
                outdata = self.compressor.compress(b)
                if outdata:
                    self.output.write(outdata)
                return len(memoryview(b))

            def flush(self) -> None:
                outdata = self.compressor.flush()
                if outdata:
                    self.output.write(outdata)
                self.output.flush()

        return cast(IO[bytes], Writer(output))

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        if zstandard is not None:
            decompressor = zstandard.ZstdDecompressor()  # type: ignore[union-attr]
            return decompressor.stream_reader(input)

        class Reader(io.RawIOBase):
            def __init__(self, input: IO[bytes]) -> None:
                self.input = input
                self.decompressor = pyzstd.EndlessZstdDecompressor()  # type: ignore[union-attr]

            def readable(self) -> bool:
                return True

            def readinto(self, b: Buffer) -> Optional[int]:
                # This needs to read enough so it can decompress
                # something so the output doesn't look like EOF.  This
                # means reading at least one block.  The max block
                # size is 128KB, so we read that plus some
                # overhead to be sure.

                if self.decompressor.needs_input:
                    indata = self.input.read((128 + 6) * 1024)
                else:
                    indata = b""

                bview = memoryview(b)
                blen = len(bview)
                outdata = self.decompressor.decompress(indata, blen)
                if outdata is None:
                    return None

                count = len(outdata)
                bview[:count] = outdata
                return count

            def seekable(self) -> bool:
                return False

        return cast(IO[bytes], Reader(input))


class ExtensionRegistry:
    def __init__(self) -> None:
        # Populate default registry contents
        self.extensions: dict[str, type[Extension]] = {
            cls.registry_name(): cls for cls in (ZStandard,)
        }

    def register(self, cls: type[Extension]) -> None:
        self.extensions[cls.registry_name()] = cls

    def from_descriptor_list(self, descriptors: Sequence[str]) -> Sequence[Extension]:
        """
        Given a seuquence of descriptor strings as returned by
        Extension.get_descriptor at save time, creates a sequence of
        Extension instances.  The name[@local-domain] preceding the
        version number is used to look up an implementation class in
        the registry, and the version is passed to the class's
        from_descriptor static method.  If the registry contains no
        match, this will throw ValueError.  If the from_descriptor
        method raises an exception, that will pass through to the
        caller.
        """

        def from_descriptor(desc: str) -> Extension:
            name, _, version = desc.partition("/")
            if version is None:
                version = 0
            ext = self.extensions.get(name)
            if not ext:
                raise ValueError(f"Unknown extension {name=}")
            # pyrefly: ignore [bad-argument-type]
            return ext.from_descriptor(version)

        return [from_descriptor(desc) for desc in descriptors]

```



## High-Level Overview

"""    Extensions provide modular additions to functionality within distributed checkpointing,    which affect the layout or format of the written artifacts.  Extensions may be    built into pytorch, or provided externally.    When writing, the caller provides a list of extension instances of the appropriate    type.  Each extension can output a descriptor which is used to reconstitute the    extension at read-time.

This Python file contains 7 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Extension`, `StreamTransformExtension`, `ZStandard`, `Writer`, `Reader`, `ExtensionRegistry`

**Functions defined**: `registry_name`, `from_descriptor`, `get_descriptor`, `transform_to`, `transform_from`, `is_available`, `from_descriptor`, `registry_name`, `__init__`, `get_descriptor`, `transform_to`, `__init__`, `writeable`, `write`, `flush`, `transform_from`, `__init__`, `readable`, `readinto`, `seekable`

**Key imports**: abc, io, Sequence, cast, IO, Optional, Buffer, try_import


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `io`
- `collections.abc`: Sequence
- `typing`: cast, IO, Optional
- `typing_extensions`: Buffer
- `torch._utils`: try_import


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

Files in the same folder (`torch/distributed/checkpoint`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`filesystem.py_docs.md`](./filesystem.py_docs.md)
- [`_consolidate_hf_safetensors.py_docs.md`](./_consolidate_hf_safetensors.py_docs.md)
- [`hf_storage.py_docs.md`](./hf_storage.py_docs.md)
- [`state_dict_loader.py_docs.md`](./state_dict_loader.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`_storage_utils.py_docs.md`](./_storage_utils.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_async_process_executor.py_docs.md`](./_async_process_executor.py_docs.md)
- [`resharding.py_docs.md`](./resharding.py_docs.md)


## Cross-References

- **File Documentation**: `_extension.py_docs.md`
- **Keyword Index**: `_extension.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/checkpoint`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

Files in the same folder (`docs/torch/distributed/checkpoint`):

- [`storage.py_docs.md_docs.md`](./storage.py_docs.md_docs.md)
- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`_async_process_executor.py_kw.md_docs.md`](./_async_process_executor.py_kw.md_docs.md)
- [`stateful.py_kw.md_docs.md`](./stateful.py_kw.md_docs.md)
- [`state_dict_loader.py_kw.md_docs.md`](./state_dict_loader.py_kw.md_docs.md)
- [`_async_executor.py_kw.md_docs.md`](./_async_executor.py_kw.md_docs.md)
- [`_state_dict_stager.py_kw.md_docs.md`](./_state_dict_stager.py_kw.md_docs.md)
- [`_extension.py_kw.md_docs.md`](./_extension.py_kw.md_docs.md)
- [`resharding.py_docs.md_docs.md`](./resharding.py_docs.md_docs.md)
- [`format_utils.py_docs.md_docs.md`](./format_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_extension.py_docs.md_docs.md`
- **Keyword Index**: `_extension.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
