# Documentation: `docs/torch/utils/data/datapipes/iter/routeddecoder.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/iter/routeddecoder.py_docs.md`
- **Size**: 6,370 bytes (6.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/iter/routeddecoder.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/iter/routeddecoder.py`
- **Size**: 2,731 bytes (2.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Callable, Iterable, Iterator, Sized
from io import BufferedIOBase
from typing import Any

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _deprecation_warning
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
    Decoder,
    extension_extract_fn,
    imagehandler as decoder_imagehandler,
)


__all__ = ["RoutedDecoderIterDataPipe"]


@functional_datapipe("routed_decode")
class RoutedDecoderIterDataPipe(IterDataPipe[tuple[str, Any]]):
    r"""
    Decodes binary streams from input DataPipe, yields pathname and decoded data in a tuple.

    (functional name: ``routed_decode``)

    Args:
        datapipe: Iterable datapipe that provides pathname and binary stream in tuples
        handlers: Optional user defined decoder handlers. If ``None``, basic and image decoder
            handlers will be set as default. If multiple handles are provided, the priority
            order follows the order of handlers (the first handler has the top priority)
        key_fn: Function for decoder to extract key from pathname to dispatch handlers.
            Default is set to extract file extension from pathname

    Note:
        When ``key_fn`` is specified returning anything other than extension, the default
        handler will not work and users need to specify custom handler. Custom handler
        could use regex to determine the eligibility to handle data.
    """

    def __init__(
        self,
        datapipe: Iterable[tuple[str, BufferedIOBase]],
        *handlers: Callable,
        key_fn: Callable = extension_extract_fn,
    ) -> None:
        super().__init__()
        self.datapipe: Iterable[tuple[str, BufferedIOBase]] = datapipe
        if not handlers:
            handlers = (decoder_basichandlers, decoder_imagehandler("torch"))
        self.decoder = Decoder(*handlers, key_fn=key_fn)
        _deprecation_warning(
            type(self).__name__,
            deprecation_version="1.12",
            removal_version="1.13",
            old_functional_name="routed_decode",
        )

    def add_handler(self, *handler: Callable) -> None:
        self.decoder.add_handler(*handler)

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for data in self.datapipe:
            pathname = data[0]
            result = self.decoder(data)
            yield (pathname, result[pathname])

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

```



## High-Level Overview

r"""    Decodes binary streams from input DataPipe, yields pathname and decoded data in a tuple.    (functional name: ``routed_decode``)    Args:        datapipe: Iterable datapipe that provides pathname and binary stream in tuples        handlers: Optional user defined decoder handlers. If ``None``, basic and image decoder            handlers will be set as default. If multiple handles are provided, the priority            order follows the order of handlers (the first handler has the top priority)        key_fn: Function for decoder to extract key from pathname to dispatch handlers.            Default is set to extract file extension from pathname    Note:        When ``key_fn`` is specified returning anything other than extension, the default        handler will not work and users need to specify custom handler. Custom handler        could use regex to determine the eligibility to handle data.

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RoutedDecoderIterDataPipe`

**Functions defined**: `__init__`, `add_handler`, `__iter__`, `__len__`

**Key imports**: Callable, Iterable, Iterator, Sized, BufferedIOBase, Any, functional_datapipe, IterDataPipe, _deprecation_warning


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable, Iterable, Iterator, Sized
- `io`: BufferedIOBase
- `typing`: Any
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.datapipe`: IterDataPipe
- `torch.utils.data.datapipes.utils.common`: _deprecation_warning


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/utils/data/datapipes/iter`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`combining.py_docs.md`](./combining.py_docs.md)
- [`callable.py_docs.md`](./callable.py_docs.md)
- [`filelister.py_docs.md`](./filelister.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`grouping.py_docs.md`](./grouping.py_docs.md)
- [`selecting.py_docs.md`](./selecting.py_docs.md)
- [`sharding.py_docs.md`](./sharding.py_docs.md)
- [`streamreader.py_docs.md`](./streamreader.py_docs.md)


## Cross-References

- **File Documentation**: `routeddecoder.py_docs.md`
- **Keyword Index**: `routeddecoder.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/data/datapipes/iter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/utils/data/datapipes/iter`):

- [`combining.py_docs.md_docs.md`](./combining.py_docs.md_docs.md)
- [`selecting.py_docs.md_docs.md`](./selecting.py_docs.md_docs.md)
- [`sharding.py_kw.md_docs.md`](./sharding.py_kw.md_docs.md)
- [`filelister.py_kw.md_docs.md`](./filelister.py_kw.md_docs.md)
- [`fileopener.py_kw.md_docs.md`](./fileopener.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`selecting.py_kw.md_docs.md`](./selecting.py_kw.md_docs.md)
- [`grouping.py_kw.md_docs.md`](./grouping.py_kw.md_docs.md)
- [`filelister.py_docs.md_docs.md`](./filelister.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `routeddecoder.py_docs.md_docs.md`
- **Keyword Index**: `routeddecoder.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
