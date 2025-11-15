# Documentation: `docs/test/distributed/elastic/utils/distributed_test.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/elastic/utils/distributed_test.py_docs.md`
- **Size**: 8,710 bytes (8.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/elastic/utils/distributed_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/utils/distributed_test.py`
- **Size**: 6,013 bytes (5.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import socket
import sys
import unittest
from contextlib import closing

from torch.distributed import DistNetworkError, DistStoreError
from torch.distributed.elastic.utils.distributed import (
    create_c10d_store,
    get_socket_with_port,
)
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skipIfRocm,
    TEST_WITH_TSAN,
    TestCase,
)


def _create_c10d_store_mp(is_server, server_addr, port, world_size, wait_for_workers):
    store = create_c10d_store(
        is_server,
        server_addr,
        port,
        world_size,
        wait_for_workers=wait_for_workers,
        timeout=2,
    )
    if store is None:
        raise AssertionError

    store.set(f"test_key/{os.getpid()}", b"test_value")


if IS_WINDOWS or IS_MACOS:
    print("tests incompatible with tsan or asan", file=sys.stderr)
    sys.exit(0)


class DistributedUtilTest(TestCase):
    def test_create_store_single_server(self):
        store = create_c10d_store(is_server=True, server_addr=socket.gethostname())
        self.assertIsNotNone(store)

    def test_create_store_no_port_multi(self):
        with self.assertRaises(ValueError):
            create_c10d_store(
                is_server=True, server_addr=socket.gethostname(), world_size=2
            )

    @unittest.skipIf(TEST_WITH_TSAN, "test incompatible with tsan")
    def test_create_store_multi(self):
        world_size = 3
        wait_for_workers = False
        localhost = socket.gethostname()

        # start the server on the main process using an available port
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )

        # worker processes will use the port that was assigned to the server
        server_port = store.port

        worker0 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size, wait_for_workers),
        )
        worker1 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size, wait_for_workers),
        )

        worker0.start()
        worker1.start()

        worker0.join()
        worker1.join()

        # check test_key/pid == "test_value"
        self.assertEqual(
            "test_value", store.get(f"test_key/{worker0.pid}").decode("UTF-8")
        )
        self.assertEqual(
            "test_value", store.get(f"test_key/{worker1.pid}").decode("UTF-8")
        )

        self.assertEqual(0, worker0.exitcode)
        self.assertEqual(0, worker1.exitcode)

    def test_create_store_timeout_on_server(self):
        with self.assertRaises(DistStoreError):
            # use any available port (port 0) since timeout is expected
            create_c10d_store(
                is_server=True,
                server_addr=socket.gethostname(),
                server_port=0,
                world_size=2,
                timeout=1,
            )

    def test_create_store_timeout_on_worker(self):
        with self.assertRaises(DistNetworkError):
            # use any available port (port 0) since timeout is expected
            create_c10d_store(
                is_server=False,
                server_addr=socket.gethostname(),
                server_port=0,
                world_size=2,
                timeout=1,
            )

    def test_create_store_with_libuv_support(self):
        world_size = 1
        wait_for_workers = False
        localhost = socket.gethostname()

        os.environ["USE_LIBUV"] = "0"
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )
        self.assertFalse(store.libuvBackend)
        del os.environ["USE_LIBUV"]
        assert "USE_LIBUV" not in os.environ

        # libuv backend is enabled by default
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )
        self.assertTrue(store.libuvBackend)

    def test_port_already_in_use_on_server(self):
        # try to create the TCPStore server twice on the same port
        # the second should fail due to a port conflict
        # first store binds onto a free port
        # try creating the second store on the port that the first store binded to
        server_addr = socket.gethostname()
        pick_free_port = 0
        store1 = create_c10d_store(
            is_server=True,
            server_addr=server_addr,
            server_port=pick_free_port,
            timeout=1,
        )
        with self.assertRaises(DistNetworkError):
            create_c10d_store(
                is_server=True, server_addr=server_addr, server_port=store1.port
            )

    @skipIfRocm
    def test_port_already_in_use_on_worker(self):
        sock = get_socket_with_port()
        with closing(sock):
            port = sock.getsockname()[1]
            # on the worker port conflict shouldn't matter, it should just timeout
            # since we never created a server
            with self.assertRaises(DistNetworkError):
                create_c10d_store(
                    is_server=False,
                    server_addr=socket.gethostname(),
                    server_port=port,
                    timeout=1,
                )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DistributedUtilTest`

**Functions defined**: `_create_c10d_store_mp`, `test_create_store_single_server`, `test_create_store_no_port_multi`, `test_create_store_multi`, `test_create_store_timeout_on_server`, `test_create_store_timeout_on_worker`, `test_create_store_with_libuv_support`, `test_port_already_in_use_on_server`, `test_port_already_in_use_on_worker`

**Key imports**: multiprocessing as mp, os, socket, sys, unittest, closing, DistNetworkError, DistStoreError


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/utils`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `multiprocessing as mp`
- `os`
- `socket`
- `sys`
- `unittest`
- `contextlib`: closing
- `torch.distributed`: DistNetworkError, DistStoreError


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
python test/distributed/elastic/utils/distributed_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/utils`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`logging_test.py_docs.md`](./logging_test.py_docs.md)
- [`util_test.py_docs.md`](./util_test.py_docs.md)


## Cross-References

- **File Documentation**: `distributed_test.py_docs.md`
- **Keyword Index**: `distributed_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/elastic/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/elastic/utils`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/elastic/utils/distributed_test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/elastic/utils`):

- [`logging_test.py_kw.md_docs.md`](./logging_test.py_kw.md_docs.md)
- [`util_test.py_docs.md_docs.md`](./util_test.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`logging_test.py_docs.md_docs.md`](./logging_test.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`distributed_test.py_kw.md_docs.md`](./distributed_test.py_kw.md_docs.md)
- [`util_test.py_kw.md_docs.md`](./util_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `distributed_test.py_docs.md_docs.md`
- **Keyword Index**: `distributed_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
