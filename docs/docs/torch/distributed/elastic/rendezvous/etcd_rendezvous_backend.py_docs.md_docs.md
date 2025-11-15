# Documentation: `docs/torch/distributed/elastic/rendezvous/etcd_rendezvous_backend.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/elastic/rendezvous/etcd_rendezvous_backend.py_docs.md`
- **Size**: 10,889 bytes (10.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/elastic/rendezvous/etcd_rendezvous_backend.py`

## File Metadata

- **Path**: `torch/distributed/elastic/rendezvous/etcd_rendezvous_backend.py`
- **Size**: 7,373 bytes (7.20 KB)
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
from base64 import b64decode, b64encode
from typing import cast, Optional

import urllib3.exceptions  # type: ignore[import]


try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    from . import _etcd_stub as etcd

from torch.distributed import Store

from .api import RendezvousConnectionError, RendezvousParameters, RendezvousStateError
from .dynamic_rendezvous import RendezvousBackend, Token
from .etcd_store import EtcdStore
from .utils import parse_rendezvous_endpoint


class EtcdRendezvousBackend(RendezvousBackend):
    """Represents an etcd-based rendezvous backend.

    Args:
        client:
            The ``etcd.Client`` instance to use to communicate with etcd.
        run_id:
            The run id of the rendezvous.
        key_prefix:
            The path under which to store the rendezvous state in etcd.
        ttl:
            The TTL of the rendezvous state. If not specified, defaults to two hours.
    """

    _DEFAULT_TTL = 7200  # 2 hours

    _client: etcd.Client
    _key: str
    _ttl: int

    def __init__(
        self,
        client: etcd.Client,
        run_id: str,
        key_prefix: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._client = client

        if key_prefix:
            self._key = key_prefix + "/" + run_id
        else:
            self._key = run_id

        if ttl and ttl > 0:
            self._ttl = ttl
        else:
            self._ttl = self._DEFAULT_TTL

    @property
    def name(self) -> str:
        """See base class."""
        return "etcd-v2"

    def get_state(self) -> Optional[tuple[bytes, Token]]:
        """See base class."""
        try:
            result = self._client.read(self._key)
        except etcd.EtcdKeyNotFound:
            return None
        except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        return self._decode_state(result)

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[tuple[bytes, Token, bool]]:
        """See base class."""
        base64_state = b64encode(state).decode()

        kwargs = {}

        def get_state():
            result = self.get_state()
            if result is not None:
                return *result, False
            return None

        if token:
            try:
                token = int(token)
            except ValueError:
                return get_state()

        if token:
            kwargs["prevIndex"] = token
        else:
            kwargs["prevExist"] = False

        try:
            result = self._client.write(self._key, base64_state, self._ttl, **kwargs)
        except (etcd.EtcdAlreadyExist, etcd.EtcdCompareFailed):
            result = None
        except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        if result is None:
            return get_state()

        tmp = *self._decode_state(result), True
        return tmp

    def _decode_state(self, result: etcd.EtcdResult) -> tuple[bytes, Token]:
        # pyrefly: ignore [missing-attribute]
        base64_state = result.value.encode()

        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        # pyrefly: ignore [missing-attribute]
        return state, result.modifiedIndex


def _create_etcd_client(params: RendezvousParameters) -> etcd.Client:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=2379)

    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # The communication protocol
    protocol = params.get("protocol", "http").strip().lower()
    if protocol != "http" and protocol != "https":
        raise ValueError("The protocol must be HTTP or HTTPS.")

    # The SSL client certificate
    ssl_cert = params.get("ssl_cert")
    if ssl_cert:
        ssl_cert_key = params.get("ssl_cert_key")
        if ssl_cert_key:
            # The etcd client expects the certificate key as the second element
            # of the `cert` tuple.
            ssl_cert = (ssl_cert, ssl_cert_key)

    # The root certificate
    ca_cert = params.get("ca_cert")

    try:
        return etcd.Client(
            host,
            port,
            read_timeout=read_timeout,
            protocol=protocol,
            cert=ssl_cert,
            ca_cert=ca_cert,
            allow_reconnect=True,
        )
    except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
        raise RendezvousConnectionError(
            "The connection to etcd has failed. See inner exception for details."
        ) from exc


def create_backend(params: RendezvousParameters) -> tuple[EtcdRendezvousBackend, Store]:
    """Create a new :py:class:`EtcdRendezvousBackend` from the specified parameters.

    +--------------+-----------------------------------------------------------+
    | Parameter    | Description                                               |
    +==============+===========================================================+
    | read_timeout | The read timeout, in seconds, for etcd operations.        |
    |              | Defaults to 60 seconds.                                   |
    +--------------+-----------------------------------------------------------+
    | protocol     | The protocol to use to communicate with etcd. Valid       |
    |              | values are "http" and "https". Defaults to "http".        |
    +--------------+-----------------------------------------------------------+
    | ssl_cert     | The path to the SSL client certificate to use along with  |
    |              | HTTPS. Defaults to ``None``.                              |
    +--------------+-----------------------------------------------------------+
    | ssl_cert_key | The path to the private key of the SSL client certificate |
    |              | to use along with HTTPS. Defaults to ``None``.            |
    +--------------+-----------------------------------------------------------+
    | ca_cert      | The path to the rool SSL authority certificate. Defaults  |
    |              | to ``None``.                                              |
    +--------------+-----------------------------------------------------------+
    """
    client = _create_etcd_client(params)

    backend = EtcdRendezvousBackend(
        client, params.run_id, key_prefix="/torch/elastic/rendezvous"
    )

    store = EtcdStore(client, "/torch/elastic/store")

    return backend, store

```



## High-Level Overview

"""Represents an etcd-based rendezvous backend.    Args:        client:            The ``etcd.Client`` instance to use to communicate with etcd.        run_id:            The run id of the rendezvous.        key_prefix:            The path under which to store the rendezvous state in etcd.        ttl:            The TTL of the rendezvous state. If not specified, defaults to two hours.

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EtcdRendezvousBackend`

**Functions defined**: `__init__`, `name`, `get_state`, `set_state`, `get_state`, `_decode_state`, `_create_etcd_client`, `create_backend`

**Key imports**: binascii, b64decode, b64encode, cast, Optional, urllib3.exceptions  , etcd  , _etcd_stub as etcd, Store, RendezvousConnectionError, RendezvousParameters, RendezvousStateError, RendezvousBackend, Token, EtcdStore


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/rendezvous`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `binascii`
- `base64`: b64decode, b64encode
- `typing`: cast, Optional
- `urllib3.exceptions  `
- `etcd  `
- `.`: _etcd_stub as etcd
- `torch.distributed`: Store
- `.api`: RendezvousConnectionError, RendezvousParameters, RendezvousStateError
- `.dynamic_rendezvous`: RendezvousBackend, Token
- `.etcd_store`: EtcdStore
- `.utils`: parse_rendezvous_endpoint


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
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`dynamic_rendezvous.py_docs.md`](./dynamic_rendezvous.py_docs.md)
- [`etcd_server.py_docs.md`](./etcd_server.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`etcd_rendezvous.py_docs.md`](./etcd_rendezvous.py_docs.md)
- [`etcd_store.py_docs.md`](./etcd_store.py_docs.md)
- [`c10d_rendezvous_backend.py_docs.md`](./c10d_rendezvous_backend.py_docs.md)


## Cross-References

- **File Documentation**: `etcd_rendezvous_backend.py_docs.md`
- **Keyword Index**: `etcd_rendezvous_backend.py_kw.md`
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

- **File Documentation**: `etcd_rendezvous_backend.py_docs.md_docs.md`
- **Keyword Index**: `etcd_rendezvous_backend.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
