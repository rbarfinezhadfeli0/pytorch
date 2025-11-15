# Documentation: `test/distributed/elastic/test_control_plane.py`

## File Metadata

- **Path**: `test/distributed/elastic/test_control_plane.py`
- **Size**: 7,583 bytes (7.41 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import json
import os
import pickle
import socket
import tempfile
from contextlib import contextmanager

from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool

from torch.distributed.elastic.control_plane import (
    TORCH_WORKER_SERVER_SOCKET,
    worker_main,
)
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase


class UnixHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")

        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class UnixHTTPConnectionPool(HTTPConnectionPool):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")

        self.socket_path = socket_path

    def _new_conn(self):
        return UnixHTTPConnection(self.socket_path)


@contextmanager
def local_worker_server() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = os.path.join(tmpdir, "socket.sock")
        os.environ[TORCH_WORKER_SERVER_SOCKET] = socket_path

        with worker_main():
            pool = UnixHTTPConnectionPool(socket_path)
            yield pool


class WorkerServerTest(TestCase):
    def test_worker_server(self) -> None:
        with local_worker_server() as pool:
            resp = pool.request("GET", "/")
            self.assertEqual(resp.status, 200)
            self.assertEqual(
                resp.data,
                b"<h1>torch.distributed.WorkerServer</h1>\n"
                b'<a href="'
                b"/handler/"
                b'">Handler names</a>\n',
            )

            resp = pool.request("POST", "/handler/ping")
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.data, b"pong")

            resp = pool.request("GET", "/handler/")
            self.assertEqual(resp.status, 200)
            self.assertIn("ping", json.loads(resp.data))

            resp = pool.request("POST", "/handler/nonexistent")
            self.assertEqual(resp.status, 404)
            self.assertIn(b"Handler nonexistent not found:", resp.data)

    @requires_cuda
    def test_dump_nccl_trace_pickle(self) -> None:
        with local_worker_server() as pool:
            resp = pool.request("POST", "/handler/dump_nccl_trace_pickle")
            self.assertEqual(resp.status, 200)
            out = pickle.loads(resp.data)
            self.assertIsInstance(out, dict)
            self.assertIn("version", out)

    @requires_cuda
    def test_dump_nccl_trace_pickle_with_params(self) -> None:
        with local_worker_server() as pool:
            # bad key - not lower case
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includeCollectives=true"
            )
            self.assertEqual(resp.status, 400)
            # unknown key
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?unknownkey=true"
            )
            self.assertEqual(resp.status, 400)
            # bad value - not a bool
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=notabool"
            )
            self.assertEqual(resp.status, 400)
            # bad value - value not lowercase
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=True"
            )
            self.assertEqual(resp.status, 400)
            # good key and value
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=true"
            )
            self.assertEqual(resp.status, 200)
            # multiple good keys and values
            resp = pool.request(
                "POST",
                "/handler/dump_nccl_trace_pickle?includecollectives=true&includestacktraces=false&onlyactive=true",
            )
            self.assertEqual(resp.status, 200)

    @requires_cuda
    def test_dump_nccl_trace_pickle_with_json(self) -> None:
        with local_worker_server() as pool:
            # bad key - not lower case
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includeCollectives=true"
            )
            self.assertEqual(resp.status, 400)
            # unknown key
            resp = pool.request("POST", "/handler/dump_nccl_trace_json?unknownkey=true")
            self.assertEqual(resp.status, 400)
            # bad value - not a bool
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=notabool"
            )
            self.assertEqual(resp.status, 400)
            # bad value - value not lowercase
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=True"
            )
            self.assertEqual(resp.status, 400)
            # good key and value
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=true"
            )
            self.assertEqual(resp.status, 200)
            # multiple good keys and values
            resp = pool.request(
                "POST",
                "/handler/dump_nccl_trace_json?includecollectives=true&onlyactive=true",
            )
            self.assertEqual(resp.status, 200)

    def test_tcp(self) -> None:
        import requests

        from torch._C._distributed_c10d import _WorkerServer

        server = _WorkerServer("", 1234)
        out = requests.get("http://localhost:1234/handler/")
        self.assertEqual(out.status_code, 200)

        server.shutdown()

    def test_dump_traceback(self) -> None:
        with local_worker_server() as pool:
            resp = pool.request("POST", "/handler/dump_traceback")
            self.assertEqual(resp.status, 200)
            self.assertIn(b"in test_dump_traceback\n", resp.data)

    def test_run_handler(self) -> None:
        from torch._C._distributed_c10d import _get_handler, _Request, _Response

        handler = _get_handler("ping")

        class Request(_Request):
            def __init__(self) -> None:
                _Request.__init__(self)

            def body(self) -> bytes:
                return b"dummy"

            def params(self) -> dict[str, str]:
                return {}

        class Response(_Response):
            def __init__(self) -> None:
                _Response.__init__(self)

            def set_content(self, content: str, content_type: str) -> None:
                self.content = content
                self.content_type = content_type

            def set_status(self, status: int) -> None:
                self.status = status

        req = Request()
        resp = Response()

        handler(req, resp)

        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.content, "pong")
        self.assertEqual(resp.content_type, "text/plain")

    def test_get_handler_nonexistant(self) -> None:
        from torch._C._distributed_c10d import _get_handler

        with self.assertRaisesRegex(ValueError, "Failed to find handler nonexistent"):
            _get_handler("nonexistent")

    def test_get_handler_names(self) -> None:
        from torch._C._distributed_c10d import _get_handler_names

        names = _get_handler_names()
        self.assertIn("ping", names)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `UnixHTTPConnection`, `UnixHTTPConnectionPool`, `WorkerServerTest`, `Request`, `Response`

**Functions defined**: `__init__`, `connect`, `__init__`, `_new_conn`, `local_worker_server`, `test_worker_server`, `test_dump_nccl_trace_pickle`, `test_dump_nccl_trace_pickle_with_params`, `test_dump_nccl_trace_pickle_with_json`, `test_tcp`, `test_dump_traceback`, `test_run_handler`, `__init__`, `body`, `params`, `__init__`, `set_content`, `set_status`, `test_get_handler_nonexistant`, `test_get_handler_names`

**Key imports**: json, os, pickle, socket, tempfile, contextmanager, HTTPConnection, HTTPConnectionPool, requires_cuda, run_tests, TestCase, requests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `os`
- `pickle`
- `socket`
- `tempfile`
- `contextlib`: contextmanager
- `urllib3.connection`: HTTPConnection
- `urllib3.connectionpool`: HTTPConnectionPool
- `torch.testing._internal.common_utils`: requires_cuda, run_tests, TestCase
- `requests`
- `torch._C._distributed_c10d`: _WorkerServer


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/elastic/test_control_plane.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic`):



## Cross-References

- **File Documentation**: `test_control_plane.py_docs.md`
- **Keyword Index**: `test_control_plane.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
