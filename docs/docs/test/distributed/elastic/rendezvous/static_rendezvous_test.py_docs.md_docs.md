# Documentation: `docs/test/distributed/elastic/rendezvous/static_rendezvous_test.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/elastic/rendezvous/static_rendezvous_test.py_docs.md`
- **Size**: 6,180 bytes (6.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/elastic/rendezvous/static_rendezvous_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/rendezvous/static_rendezvous_test.py`
- **Size**: 3,188 bytes (3.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from contextlib import closing

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.static_tcp_rendezvous import (
    create_rdzv_handler,
)
from torch.distributed.elastic.utils import get_socket_with_port


class StaticTCPRendezvousTest(unittest.TestCase):
    def test_missing_port(self):
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="localhost",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_empty_endpoint(self):
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_ipv6_addr(self):
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:90",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_ipv6_addr_localhost(self):
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="[::1]:90",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_get_backend(self):
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="localhost:123",
            run_id="test",
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            rank=0,
        )

        static_rdzv = create_rdzv_handler(rdzv_params)
        self.assertEqual("static", static_rdzv.get_backend())

    def test_static_rdzv_multiple_calls(self):
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        master_addr = "localhost"

        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint=f"{master_addr}:{master_port}",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
            rank=0,
        )
        rdzv_handler = create_rdzv_handler(rdzv_params)

        # Call rendezvous two times
        rdzv_info = rdzv_handler.next_rendezvous()
        self.assertIsNotNone(rdzv_info.store)
        self.assertEqual(0, rdzv_info.rank)
        self.assertEqual(1, rdzv_info.world_size)

        rdzv_info = rdzv_handler.next_rendezvous()
        self.assertIsNotNone(rdzv_info.store)
        self.assertEqual(0, rdzv_info.rank)
        self.assertEqual(1, rdzv_info.world_size)

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StaticTCPRendezvousTest`

**Functions defined**: `test_missing_port`, `test_empty_endpoint`, `test_ipv6_addr`, `test_ipv6_addr_localhost`, `test_get_backend`, `test_static_rdzv_multiple_calls`

**Key imports**: unittest, closing, RendezvousParameters, get_socket_with_port


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/rendezvous`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `contextlib`: closing
- `torch.distributed.elastic.rendezvous`: RendezvousParameters
- `torch.distributed.elastic.utils`: get_socket_with_port


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/distributed/elastic/rendezvous/static_rendezvous_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/rendezvous`):

- [`etcd_rendezvous_backend_test.py_docs.md`](./etcd_rendezvous_backend_test.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`dynamic_rendezvous_test.py_docs.md`](./dynamic_rendezvous_test.py_docs.md)
- [`api_test.py_docs.md`](./api_test.py_docs.md)
- [`utils_test.py_docs.md`](./utils_test.py_docs.md)
- [`c10d_rendezvous_backend_test.py_docs.md`](./c10d_rendezvous_backend_test.py_docs.md)
- [`etcd_server_test.py_docs.md`](./etcd_server_test.py_docs.md)
- [`etcd_rendezvous_test.py_docs.md`](./etcd_rendezvous_test.py_docs.md)
- [`rendezvous_backend_test.py_docs.md`](./rendezvous_backend_test.py_docs.md)


## Cross-References

- **File Documentation**: `static_rendezvous_test.py_docs.md`
- **Keyword Index**: `static_rendezvous_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/elastic/rendezvous`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/elastic/rendezvous`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python docs/test/distributed/elastic/rendezvous/static_rendezvous_test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/elastic/rendezvous`):

- [`etcd_server_test.py_kw.md_docs.md`](./etcd_server_test.py_kw.md_docs.md)
- [`utils_test.py_docs.md_docs.md`](./utils_test.py_docs.md_docs.md)
- [`etcd_rendezvous_test.py_docs.md_docs.md`](./etcd_rendezvous_test.py_docs.md_docs.md)
- [`out_of_tree_rendezvous_test.py_kw.md_docs.md`](./out_of_tree_rendezvous_test.py_kw.md_docs.md)
- [`out_of_tree_rendezvous_test.py_docs.md_docs.md`](./out_of_tree_rendezvous_test.py_docs.md_docs.md)
- [`dynamic_rendezvous_test.py_kw.md_docs.md`](./dynamic_rendezvous_test.py_kw.md_docs.md)
- [`rendezvous_backend_test.py_docs.md_docs.md`](./rendezvous_backend_test.py_docs.md_docs.md)
- [`static_rendezvous_test.py_kw.md_docs.md`](./static_rendezvous_test.py_kw.md_docs.md)
- [`etcd_rendezvous_test.py_kw.md_docs.md`](./etcd_rendezvous_test.py_kw.md_docs.md)
- [`etcd_server_test.py_docs.md_docs.md`](./etcd_server_test.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `static_rendezvous_test.py_docs.md_docs.md`
- **Keyword Index**: `static_rendezvous_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
