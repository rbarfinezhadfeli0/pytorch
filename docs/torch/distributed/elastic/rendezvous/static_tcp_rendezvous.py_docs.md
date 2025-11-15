# Documentation: `torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py`

## File Metadata

- **Path**: `torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py`
- **Size**: 3,665 bytes (3.58 KB)
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

import datetime
import logging
from typing import cast, Optional

from torch.distributed import PrefixStore, Store, TCPStore
from torch.distributed.elastic.rendezvous import (
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStoreInfo,
)
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint


__all__ = ["StaticTCPRendezvous", "create_rdzv_handler"]

logger = logging.getLogger(__name__)

_default_timeout_seconds = 600


class StaticTCPRendezvous(RendezvousHandler):
    """
    Static rendezvous that is a wrapper around the TCPStore.

    Creates TCPStore based on the input parameters with the
    listener on the agent with group_rank=0
    """

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        run_id: str,
        timeout: int,
    ):
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.timeout = datetime.timedelta(seconds=timeout)
        self._store: Optional[Store] = None

    def get_backend(self) -> str:
        return "static"

    @property
    def use_agent_store(self) -> bool:
        return True

    def next_rendezvous(self) -> RendezvousInfo:
        logger.info("Creating TCPStore as the c10d::Store implementation")
        is_master = self.rank == 0
        if not self._store:
            self._store = TCPStore(  # type: ignore[call-arg]
                self.master_addr,
                self.master_port,
                self.world_size,
                is_master,
                self.timeout,
                multi_tenant=True,
            )
        store = PrefixStore(self.run_id, self._store)
        # TCPStore server instance is used by trainer code
        bootstrap_store_info = RendezvousStoreInfo(self.master_addr, self.master_port)
        return RendezvousInfo(
            store,
            self.rank,
            self.world_size,
            bootstrap_store_info,
        )

    def is_closed(self):
        return False

    def set_closed(self):
        pass

    def num_nodes_waiting(self):
        return 0

    def get_run_id(self) -> str:
        return self.run_id

    def shutdown(self) -> bool:
        return True


def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    if "rank" not in params.config:
        raise ValueError(
            "rank is absent in RendezvousParameters."
            "Try add --node-rank to the cmd request"
        )
    endpoint = params.endpoint.strip()
    if not endpoint:
        raise ValueError(
            "endpoint is absent in RendezvousParameters"
            "Try add --master-port and --master-addr to the cmd request"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, -1)
    if master_port == -1:
        raise ValueError(
            f"Port is absent in endpoint: {endpoint}. Try launching with --master-port"
        )
    world_size = params.max_nodes
    rank = cast(int, params.config.get("rank"))
    run_id = params.run_id
    if "timeout" in params.config:
        timeout = int(params.config["timeout"])
    else:
        timeout = _default_timeout_seconds

    return StaticTCPRendezvous(
        master_addr, master_port, rank, world_size, run_id, timeout
    )

```



## High-Level Overview

"""    Static rendezvous that is a wrapper around the TCPStore.    Creates TCPStore based on the input parameters with the    listener on the agent with group_rank=0

This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StaticTCPRendezvous`

**Functions defined**: `__init__`, `get_backend`, `use_agent_store`, `next_rendezvous`, `is_closed`, `set_closed`, `num_nodes_waiting`, `get_run_id`, `shutdown`, `create_rdzv_handler`

**Key imports**: datetime, logging, cast, Optional, PrefixStore, Store, TCPStore, parse_rendezvous_endpoint


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/rendezvous`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `datetime`
- `logging`
- `typing`: cast, Optional
- `torch.distributed`: PrefixStore, Store, TCPStore
- `torch.distributed.elastic.rendezvous.utils`: parse_rendezvous_endpoint


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
- [`c10d_rendezvous_backend.py_docs.md`](./c10d_rendezvous_backend.py_docs.md)


## Cross-References

- **File Documentation**: `static_tcp_rendezvous.py_docs.md`
- **Keyword Index**: `static_tcp_rendezvous.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
