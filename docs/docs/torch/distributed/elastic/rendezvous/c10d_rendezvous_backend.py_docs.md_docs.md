# Documentation: `docs/torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py_docs.md`
- **Size**: 14,146 bytes (13.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py`

## File Metadata

- **Path**: `torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py`
- **Size**: 10,760 bytes (10.51 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
import logging
import os
import tempfile
from base64 import b64decode, b64encode
from datetime import timedelta
from typing import Any, cast, Optional

from torch.distributed import FileStore, Store, TCPStore
from torch.distributed.elastic.events import construct_and_record_rdzv_event, NodeState

from .api import (
    RendezvousConnectionError,
    RendezvousError,
    RendezvousParameters,
    RendezvousStateError,
)
from .dynamic_rendezvous import RendezvousBackend, Token
from .utils import _matches_machine_hostname, parse_rendezvous_endpoint


logger = logging.getLogger(__name__)

# default port for the TCP store
DEFAULT_PORT = 29400


class C10dRendezvousBackend(RendezvousBackend):
    """Represents a C10d-backed rendezvous backend.

    Args:
        store:
            The :py:class:`torch.distributed.Store` instance to use to
            communicate with the C10d store.
        run_id:
            The run id of the rendezvous.
    """

    # See the explanation in the __init__ method.
    _NULL_SENTINEL = "Y2FuaW1hZGFt"

    _store: Store
    _key: str

    def __init__(self, store: Store, run_id: str) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._store = store

        self._key = "torch.rendezvous." + run_id

        # The read operation of a store blocks the caller until the specified
        # key becomes available. This behavior makes it tricky to use a store
        # as a regular key-value dictionary.
        #
        # As a workaround we initially set a sentinel value as the rendezvous
        # state. Whenever this value gets returned we treat it as a None.
        self._call_store("compare_set", self._key, "", self._NULL_SENTINEL)

    @property
    def name(self) -> str:
        """See base class."""
        return "c10d"

    def get_state(self) -> Optional[tuple[bytes, Token]]:
        """See base class."""
        base64_state: bytes = self._call_store("get", self._key)

        return self._decode_state(base64_state)

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[tuple[bytes, Token, bool]]:
        """See base class."""
        base64_state_str: str = b64encode(state).decode()

        if token:
            # Shortcut if we know for sure that the token is not valid.
            if not isinstance(token, bytes):
                result = self.get_state()
                if result is not None:
                    return *result, False
                return None

            token = token.decode()
        else:
            token = self._NULL_SENTINEL

        base64_state: bytes = self._call_store(
            "compare_set", self._key, token, base64_state_str
        )

        state_token_pair = self._decode_state(base64_state)
        if state_token_pair is None:
            return None

        new_state, new_token = state_token_pair

        # C10d Store's compare_set method does not offer an easy way to find out
        # whether our write attempt was successful. As a brute-force solution we
        # perform a bitwise comparison of our local state and the remote state.
        return new_state, new_token, new_state == state

    def _call_store(self, store_op: str, *args, **kwargs) -> Any:
        try:
            return getattr(self._store, store_op)(*args, **kwargs)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to the C10d store has failed. See inner exception for details."
            ) from exc

    def _decode_state(self, base64_state: bytes) -> Optional[tuple[bytes, Token]]:
        if base64_state == self._NULL_SENTINEL.encode():
            return None

        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        return state, base64_state


def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=DEFAULT_PORT)

    cfg_is_host = params.get_as_bool("is_host")
    # If the user has explicitly specified whether our process should host the
    # the store, respect it.
    if cfg_is_host is not None:
        is_host = cfg_is_host
    # Otherwise try to determine whether we are the host based on our hostname
    # and IP address.
    else:
        is_host = _matches_machine_hostname(host)

    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # In specific cases we attempt to instantiate the store twice. For details
    # see the explanation in the except clause below.
    for is_server in [is_host, False]:
        try:
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                multi_tenant=True,
                timeout=timedelta(seconds=read_timeout),
            )

            if is_server:
                msg = f"Process {os.getpid()} hosts the TCP store for the C10d rendezvous backend."
                construct_and_record_rdzv_event(
                    run_id=params.run_id, message=msg, node_state=NodeState.INIT
                )
                logger.info(msg)

            break
        except (ValueError, RuntimeError, TimeoutError) as exc:
            # If we heuristically inferred the value of is_host as True and our
            # first attempt to instantiate the TCP store has failed, try it one
            # more time with is_host set to False. As an edge case there can be
            # more than one process that is part of the same rendezvous on this
            # machine and only one of them will eventually host the store.

            if not is_server or cfg_is_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the C10d store has failed. See inner exception for details."
                ) from exc

    return store  # type: ignore[possibly-undefined]


def _create_file_store(params: RendezvousParameters) -> FileStore:
    # If a user specifies an endpoint, we treat it as a path to a file.
    if params.endpoint:
        path = params.endpoint
    else:
        try:
            # The temporary file is readable and writable only by the user of
            # this process.
            _, path = tempfile.mkstemp()
        except OSError as exc:
            raise RendezvousError(
                "The file creation for C10d store has failed. See inner exception for details."
            ) from exc

    try:
        store = FileStore(path)
    except (ValueError, RuntimeError) as exc:
        raise RendezvousConnectionError(
            "The connection to the C10d store has failed. See inner exception for details."
        ) from exc

    return store


def create_backend(params: RendezvousParameters) -> tuple[C10dRendezvousBackend, Store]:
    """Create a new :py:class:`C10dRendezvousBackend` from the specified parameters.

    +--------------+-----------------------------------------------------------+
    | Parameter    | Description                                               |
    +==============+===========================================================+
    | store_type   | The type of the C10d store. The currently supported types |
    |              | are "tcp" and "file" which correspond to                  |
    |              | :py:class:`torch.distributed.TCPStore` and                |
    |              | :py:class:`torch.distributed.FileStore`, respectively.    |
    |              | Defaults to "tcp".                                        |
    +--------------+-----------------------------------------------------------+
    | read_timeout | The read timeout, in seconds, for store operations.       |
    |              | Defaults to 60 seconds.                                   |
    |              |                                                           |
    |              | Note this only applies to                                 |
    |              | :py:class:`torch.distributed.TCPStore`. It is not relevant|
    |              | to :py:class:`torch.distributed.FileStore` which does not |
    |              | take in timeout as a parameter.                           |
    +--------------+-----------------------------------------------------------+
    | is_host      | A boolean value indicating whether this backend instance  |
    |              | will host the C10d store. If not specified it will be     |
    |              | inferred heuristically by matching the hostname or the IP |
    |              | address of this machine against the specified rendezvous  |
    |              | endpoint. Defaults to ``None``.                           |
    |              |                                                           |
    |              | Note that this configuration option only applies to       |
    |              | :py:class:`torch.distributed.TCPStore`. In normal         |
    |              | circumstances you can safely skip it; the only time when  |
    |              | it is needed is if its value cannot be correctly          |
    |              | determined (e.g. the rendezvous endpoint has a CNAME as   |
    |              | the hostname or does not match the FQDN of the machine).  |
    +--------------+-----------------------------------------------------------+
    """
    # As of today we only support TCPStore and FileStore. Other store types do
    # not have the required functionality (e.g. compare_set) yet.
    store_type = params.get("store_type", "tcp").strip().lower()
    store: Store

    try:
        if store_type == "file":
            store = _create_file_store(params)
        elif store_type == "tcp":
            store = _create_tcp_store(params)
        else:
            raise ValueError(
                "Invalid store type given. Currently only supports file and tcp."
            )

        backend = C10dRendezvousBackend(store, params.run_id)

    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise

    return backend, store

```



## High-Level Overview

"""Represents a C10d-backed rendezvous backend.    Args:        store:            The :py:class:`torch.distributed.Store` instance to use to            communicate with the C10d store.        run_id:            The run id of the rendezvous.

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `C10dRendezvousBackend`

**Functions defined**: `__init__`, `name`, `get_state`, `set_state`, `_call_store`, `_decode_state`, `_create_tcp_store`, `_create_file_store`, `create_backend`

**Key imports**: binascii, logging, os, tempfile, b64decode, b64encode, timedelta, Any, cast, Optional, FileStore, Store, TCPStore, construct_and_record_rdzv_event, NodeState, RendezvousBackend, Token


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/rendezvous`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `binascii`
- `logging`
- `os`
- `tempfile`
- `base64`: b64decode, b64encode
- `datetime`: timedelta
- `typing`: Any, cast, Optional
- `torch.distributed`: FileStore, Store, TCPStore
- `torch.distributed.elastic.events`: construct_and_record_rdzv_event, NodeState
- `.dynamic_rendezvous`: RendezvousBackend, Token
- `.utils`: _matches_machine_hostname, parse_rendezvous_endpoint


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`dynamic_rendezvous.py_docs.md`](./dynamic_rendezvous.py_docs.md)
- [`etcd_server.py_docs.md`](./etcd_server.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`etcd_rendezvous.py_docs.md`](./etcd_rendezvous.py_docs.md)
- [`etcd_store.py_docs.md`](./etcd_store.py_docs.md)


## Cross-References

- **File Documentation**: `c10d_rendezvous_backend.py_docs.md`
- **Keyword Index**: `c10d_rendezvous_backend.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
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

- **File Documentation**: `c10d_rendezvous_backend.py_docs.md_docs.md`
- **Keyword Index**: `c10d_rendezvous_backend.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
