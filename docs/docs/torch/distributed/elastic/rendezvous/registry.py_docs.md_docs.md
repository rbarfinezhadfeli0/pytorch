# Documentation: `docs/torch/distributed/elastic/rendezvous/registry.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/elastic/rendezvous/registry.py_docs.md`
- **Size**: 5,950 bytes (5.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/elastic/rendezvous/registry.py`

## File Metadata

- **Path**: `torch/distributed/elastic/rendezvous/registry.py`
- **Size**: 2,922 bytes (2.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from importlib.metadata import entry_points

from .api import (
    rendezvous_handler_registry as handler_registry,
    RendezvousHandler,
    RendezvousParameters,
)
from .dynamic_rendezvous import create_handler


log = logging.getLogger(__name__)

__all__ = ["get_rendezvous_handler"]


def _create_static_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import static_tcp_rendezvous

    return static_tcp_rendezvous.create_rdzv_handler(params)


def _create_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import etcd_rendezvous

    return etcd_rendezvous.create_rdzv_handler(params)


def _create_etcd_v2_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .etcd_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)


def _create_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .c10d_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)


def _register_default_handlers() -> None:
    handler_registry.register("etcd", _create_etcd_handler)
    handler_registry.register("etcd-v2", _create_etcd_v2_handler)
    handler_registry.register("c10d", _create_c10d_handler)
    handler_registry.register("static", _create_static_handler)


def _register_out_of_tree_handlers() -> None:
    discovered_handler_generators = entry_points(group="torchrun.handlers")

    for handler_generator in discovered_handler_generators:
        try:
            get_handler = discovered_handler_generators[handler_generator.name].load()
            handler_registry.register(handler_generator.name, get_handler())
        except Exception:
            log.warning(
                "Exception while registering out of tree plugin %s: ",
                handler_generator.name,
                exc_info=True,
            )


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    """
    Obtain a reference to a :py:class`RendezvousHandler`.

    Custom rendezvous handlers can be registered by

    ::

      from torch.distributed.elastic.rendezvous import rendezvous_handler_registry
      from torch.distributed.elastic.rendezvous.registry import get_rendezvous_handler


      def create_my_rdzv(params: RendezvousParameters):
          return MyCustomRdzv(params)


      rendezvous_handler_registry.register("my_rdzv_backend_name", create_my_rdzv)

      my_rdzv_handler = get_rendezvous_handler(
          "my_rdzv_backend_name", RendezvousParameters
      )
    """
    return handler_registry.create_handler(params)

```



## High-Level Overview


This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_create_static_handler`, `_create_etcd_handler`, `_create_etcd_v2_handler`, `_create_c10d_handler`, `_register_default_handlers`, `_register_out_of_tree_handlers`, `get_rendezvous_handler`, `create_my_rdzv`

**Key imports**: logging, entry_points, create_handler, static_tcp_rendezvous, etcd_rendezvous, create_backend, create_backend, rendezvous_handler_registry, get_rendezvous_handler


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/rendezvous`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `importlib.metadata`: entry_points
- `.dynamic_rendezvous`: create_handler
- `.`: static_tcp_rendezvous
- `.etcd_rendezvous_backend`: create_backend
- `.c10d_rendezvous_backend`: create_backend
- `torch.distributed.elastic.rendezvous`: rendezvous_handler_registry
- `torch.distributed.elastic.rendezvous.registry`: get_rendezvous_handler


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`torch/distributed/elastic/rendezvous`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`etcd_rendezvous_backend.py_docs.md`](./etcd_rendezvous_backend.py_docs.md)
- [`dynamic_rendezvous.py_docs.md`](./dynamic_rendezvous.py_docs.md)
- [`etcd_server.py_docs.md`](./etcd_server.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`etcd_rendezvous.py_docs.md`](./etcd_rendezvous.py_docs.md)
- [`etcd_store.py_docs.md`](./etcd_store.py_docs.md)
- [`c10d_rendezvous_backend.py_docs.md`](./c10d_rendezvous_backend.py_docs.md)


## Cross-References

- **File Documentation**: `registry.py_docs.md`
- **Keyword Index**: `registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/elastic/rendezvous`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/elastic/rendezvous`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`docs/torch/distributed/elastic/rendezvous`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`etcd_rendezvous_backend.py_kw.md_docs.md`](./etcd_rendezvous_backend.py_kw.md_docs.md)
- [`etcd_server.py_kw.md_docs.md`](./etcd_server.py_kw.md_docs.md)
- [`registry.py_kw.md_docs.md`](./registry.py_kw.md_docs.md)
- [`_etcd_stub.py_docs.md_docs.md`](./_etcd_stub.py_docs.md_docs.md)
- [`c10d_rendezvous_backend.py_kw.md_docs.md`](./c10d_rendezvous_backend.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`etcd_server.py_docs.md_docs.md`](./etcd_server.py_docs.md_docs.md)
- [`_etcd_stub.py_kw.md_docs.md`](./_etcd_stub.py_kw.md_docs.md)
- [`dynamic_rendezvous.py_kw.md_docs.md`](./dynamic_rendezvous.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `registry.py_docs.md_docs.md`
- **Keyword Index**: `registry.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
