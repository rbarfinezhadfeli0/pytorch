# Documentation: `docs/test/distributed/elastic/rendezvous/c10d_rendezvous_backend_test.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/elastic/rendezvous/c10d_rendezvous_backend_test.py_docs.md`
- **Size**: 14,372 bytes (14.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/elastic/rendezvous/c10d_rendezvous_backend_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/rendezvous/c10d_rendezvous_backend_test.py`
- **Size**: 10,454 bytes (10.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from base64 import b64encode
from collections.abc import Callable
from datetime import timedelta
from typing import cast, ClassVar
from unittest import mock, TestCase

from rendezvous_backend_test import RendezvousBackendTestMixin

from torch.distributed import FileStore, TCPStore
from torch.distributed.elastic.rendezvous import (
    RendezvousConnectionError,
    RendezvousError,
    RendezvousParameters,
)
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import (
    C10dRendezvousBackend,
    create_backend,
)
from torch.distributed.elastic.utils.distributed import get_free_port


class TCPStoreBackendTest(TestCase, RendezvousBackendTestMixin):
    _store: ClassVar[TCPStore]

    @classmethod
    def setUpClass(cls) -> None:
        cls._store = TCPStore("localhost", 0, is_master=True)  # type: ignore[call-arg]

    def setUp(self) -> None:
        # Make sure we have a clean slate.
        self._store.delete_key("torch.rendezvous.dummy_run_id")

        self._backend = C10dRendezvousBackend(self._store, "dummy_run_id")

    def _corrupt_state(self) -> None:
        self._store.set("torch.rendezvous.dummy_run_id", "non_base64")


class FileStoreBackendTest(TestCase, RendezvousBackendTestMixin):
    _store: ClassVar[FileStore]

    def setUp(self) -> None:
        _, path = tempfile.mkstemp()
        self._path = path

        # Currently, filestore doesn't implement a delete_key method, so a new
        # filestore has to be initialized for every test in order to have a
        # clean slate.
        self._store = FileStore(path)
        self._backend = C10dRendezvousBackend(self._store, "dummy_run_id")

    def tearDown(self) -> None:
        os.remove(self._path)

    def _corrupt_state(self) -> None:
        self._store.set("torch.rendezvous.dummy_run_id", "non_base64")


class CreateBackendTest(TestCase):
    def setUp(self) -> None:
        # For testing, the default parameters used are for tcp. If a test
        # uses parameters for file store, we set the self._params to
        # self._params_filestore.

        port = get_free_port()
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint=f"localhost:{port}",
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            is_host="true",
            store_type="tCp",
            read_timeout="10",
        )

        _, tmp_path = tempfile.mkstemp()

        # Parameters for filestore testing.
        self._params_filestore = RendezvousParameters(
            backend="dummy_backend",
            endpoint=tmp_path,
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            store_type="fIlE",
        )
        self._expected_endpoint_file = tmp_path
        self._expected_temp_dir = tempfile.gettempdir()

        self._expected_endpoint_host = "localhost"
        self._expected_endpoint_port = port
        self._expected_store_type = TCPStore
        self._expected_read_timeout = timedelta(seconds=10)

    def tearDown(self) -> None:
        os.remove(self._expected_endpoint_file)

    def _run_test_with_store(self, store_type: str, test_to_run: Callable):
        """
        Use this function to specify the store type to use in a test. If
        not used, the test will default to TCPStore.
        """
        if store_type == "file":
            self._params = self._params_filestore
            self._expected_store_type = FileStore
            self._expected_read_timeout = timedelta(seconds=300)

        test_to_run()

    def _assert_create_backend_returns_backend(self) -> None:
        backend, store = create_backend(self._params)

        self.assertEqual(backend.name, "c10d")

        self.assertIsInstance(store, self._expected_store_type)

        typecast_store = cast(self._expected_store_type, store)
        self.assertEqual(typecast_store.timeout, self._expected_read_timeout)  # type: ignore[attr-defined]
        if self._expected_store_type == TCPStore:
            self.assertEqual(typecast_store.host, self._expected_endpoint_host)  # type: ignore[attr-defined]
            self.assertEqual(typecast_store.port, self._expected_endpoint_port)  # type: ignore[attr-defined]
        if self._expected_store_type == FileStore:
            if self._params.endpoint:
                self.assertEqual(typecast_store.path, self._expected_endpoint_file)  # type: ignore[attr-defined]
            else:
                self.assertTrue(typecast_store.path.startswith(self._expected_temp_dir))  # type: ignore[attr-defined]

        backend.set_state(b"dummy_state")

        state = store.get("torch.rendezvous." + self._params.run_id)

        self.assertEqual(state, b64encode(b"dummy_state"))

    def test_create_backend_returns_backend(self) -> None:
        for store_type in ["tcp", "file"]:
            with self.subTest(store_type=store_type):
                self._run_test_with_store(
                    store_type, self._assert_create_backend_returns_backend
                )

    def test_create_backend_returns_backend_if_is_host_is_false(self) -> None:
        TCPStore(  # type: ignore[call-arg]
            self._expected_endpoint_host, self._expected_endpoint_port, is_master=True
        )

        self._params.config["is_host"] = "false"

        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_is_host_is_not_specified(self) -> None:
        del self._params.config["is_host"]

        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_is_host_is_not_specified_and_store_already_exists(
        self,
    ) -> None:
        TCPStore(  # type: ignore[call-arg]
            self._expected_endpoint_host, self._expected_endpoint_port, is_master=True
        )

        del self._params.config["is_host"]

        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_endpoint_port_is_not_specified(
        self,
    ) -> None:
        # patch default port and pass endpoint with no port specified
        with mock.patch(
            "torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.DEFAULT_PORT",
            self._expected_endpoint_port,
        ):
            self._params.endpoint = self._expected_endpoint_host

            self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_endpoint_file_is_not_specified(
        self,
    ) -> None:
        self._params_filestore.endpoint = ""

        self._run_test_with_store("file", self._assert_create_backend_returns_backend)

    def test_create_backend_returns_backend_if_store_type_is_not_specified(
        self,
    ) -> None:
        del self._params.config["store_type"]

        self._expected_store_type = TCPStore
        if not self._params.get("read_timeout"):
            self._expected_read_timeout = timedelta(seconds=60)

        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_read_timeout_is_not_specified(
        self,
    ) -> None:
        del self._params.config["read_timeout"]

        self._expected_read_timeout = timedelta(seconds=60)

        self._assert_create_backend_returns_backend()

    def test_create_backend_raises_error_if_store_is_unreachable(self) -> None:
        self._params.config["is_host"] = "false"
        self._params.config["read_timeout"] = "2"

        with self.assertRaisesRegex(
            RendezvousConnectionError,
            r"^The connection to the C10d store has failed. See inner exception for details.$",
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_endpoint_is_invalid(self) -> None:
        for is_host in [True, False]:
            with self.subTest(is_host=is_host):
                self._params.config["is_host"] = str(is_host)

                self._params.endpoint = "dummy_endpoint"

                with self.assertRaisesRegex(
                    RendezvousConnectionError,
                    r"^The connection to the C10d store has failed. See inner exception for "
                    r"details.$",
                ):
                    create_backend(self._params)

    def test_create_backend_raises_error_if_store_type_is_invalid(self) -> None:
        self._params.config["store_type"] = "dummy_store_type"

        with self.assertRaisesRegex(
            ValueError,
            r"^Invalid store type given. Currently only supports file and tcp.$",
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_read_timeout_is_invalid(self) -> None:
        for read_timeout in ["0", "-10"]:
            with self.subTest(read_timeout=read_timeout):
                self._params.config["read_timeout"] = read_timeout

                with self.assertRaisesRegex(
                    ValueError, r"^The read timeout must be a positive integer.$"
                ):
                    create_backend(self._params)

    @mock.patch("tempfile.mkstemp")
    def test_create_backend_raises_error_if_tempfile_creation_fails(
        self, tempfile_mock
    ) -> None:
        tempfile_mock.side_effect = OSError("test error")
        # Set the endpoint to empty so it defaults to creating a temp file
        self._params_filestore.endpoint = ""
        with self.assertRaisesRegex(
            RendezvousError,
            r"The file creation for C10d store has failed. See inner exception for details.",
        ):
            create_backend(self._params_filestore)

    @mock.patch(
        "torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.FileStore"
    )
    def test_create_backend_raises_error_if_file_path_is_invalid(
        self, filestore_mock
    ) -> None:
        filestore_mock.side_effect = RuntimeError("test error")
        self._params_filestore.endpoint = "bad file path"
        with self.assertRaisesRegex(
            RendezvousConnectionError,
            r"^The connection to the C10d store has failed. See inner exception for "
            r"details.$",
        ):
            create_backend(self._params_filestore)

```



## High-Level Overview


This Python file contains 3 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TCPStoreBackendTest`, `FileStoreBackendTest`, `CreateBackendTest`

**Functions defined**: `setUpClass`, `setUp`, `_corrupt_state`, `setUp`, `tearDown`, `_corrupt_state`, `setUp`, `tearDown`, `_run_test_with_store`, `_assert_create_backend_returns_backend`, `test_create_backend_returns_backend`, `test_create_backend_returns_backend_if_is_host_is_false`, `test_create_backend_returns_backend_if_is_host_is_not_specified`, `test_create_backend_returns_backend_if_is_host_is_not_specified_and_store_already_exists`, `test_create_backend_returns_backend_if_endpoint_port_is_not_specified`, `test_create_backend_returns_backend_if_endpoint_file_is_not_specified`, `test_create_backend_returns_backend_if_store_type_is_not_specified`, `test_create_backend_returns_backend_if_read_timeout_is_not_specified`, `test_create_backend_raises_error_if_store_is_unreachable`, `test_create_backend_raises_error_if_endpoint_is_invalid`

**Key imports**: os, tempfile, b64encode, Callable, timedelta, cast, ClassVar, mock, TestCase, RendezvousBackendTestMixin, FileStore, TCPStore, get_free_port


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/rendezvous`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `tempfile`
- `base64`: b64encode
- `collections.abc`: Callable
- `datetime`: timedelta
- `typing`: cast, ClassVar
- `unittest`: mock, TestCase
- `rendezvous_backend_test`: RendezvousBackendTestMixin
- `torch.distributed`: FileStore, TCPStore
- `torch.distributed.elastic.utils.distributed`: get_free_port


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
python test/distributed/elastic/rendezvous/c10d_rendezvous_backend_test.py
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
- [`etcd_server_test.py_docs.md`](./etcd_server_test.py_docs.md)
- [`etcd_rendezvous_test.py_docs.md`](./etcd_rendezvous_test.py_docs.md)
- [`rendezvous_backend_test.py_docs.md`](./rendezvous_backend_test.py_docs.md)
- [`static_rendezvous_test.py_docs.md`](./static_rendezvous_test.py_docs.md)


## Cross-References

- **File Documentation**: `c10d_rendezvous_backend_test.py_docs.md`
- **Keyword Index**: `c10d_rendezvous_backend_test.py_kw.md`
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
python docs/test/distributed/elastic/rendezvous/c10d_rendezvous_backend_test.py_docs.md
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

- **File Documentation**: `c10d_rendezvous_backend_test.py_docs.md_docs.md`
- **Keyword Index**: `c10d_rendezvous_backend_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
