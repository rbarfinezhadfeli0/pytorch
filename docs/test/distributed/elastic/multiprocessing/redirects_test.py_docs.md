# Documentation: `test/distributed/elastic/multiprocessing/redirects_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/multiprocessing/redirects_test.py`
- **Size**: 4,749 bytes (4.64 KB)
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
import ctypes
import os
import shutil
import sys
import tempfile
import unittest

from torch.distributed.elastic.multiprocessing.redirects import (
    redirect,
    redirect_stderr,
    redirect_stdout,
)


libc = ctypes.CDLL("libc.so.6")
c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")


class RedirectsTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_redirect_invalid_std(self):
        with self.assertRaises(ValueError):
            with redirect("stdfoo", os.path.join(self.test_dir, "stdfoo.log")):
                pass

    def test_redirect_stdout(self):
        stdout_log = os.path.join(self.test_dir, "stdout.log")

        # printing to stdout before redirect should go to console not stdout.log
        print("foo first from python")
        libc.printf(b"foo first from c\n")
        os.system("echo foo first from cmd")

        with redirect_stdout(stdout_log):
            print("foo from python")
            libc.printf(b"foo from c\n")
            os.system("echo foo from cmd")

        # make sure stdout is restored
        print("foo again from python")
        libc.printf(b"foo again from c\n")
        os.system("echo foo again from cmd")

        with open(stdout_log) as f:
            # since we print from python, c, cmd -> the stream is not ordered
            # do a set comparison
            lines = set(f.readlines())
            self.assertEqual(
                {"foo from python\n", "foo from c\n", "foo from cmd\n"}, lines
            )

    def test_redirect_stderr(self):
        stderr_log = os.path.join(self.test_dir, "stderr.log")

        print("bar first from python")
        libc.fprintf(c_stderr, b"bar first from c\n")
        os.system("echo bar first from cmd 1>&2")

        with redirect_stderr(stderr_log):
            print("bar from python", file=sys.stderr)
            libc.fprintf(c_stderr, b"bar from c\n")
            os.system("echo bar from cmd 1>&2")

        print("bar again from python")
        libc.fprintf(c_stderr, b"bar again from c\n")
        os.system("echo bar again from cmd 1>&2")

        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"bar from python\n", "bar from c\n", "bar from cmd\n"}, lines
            )

    def test_redirect_both(self):
        stdout_log = os.path.join(self.test_dir, "stdout.log")
        stderr_log = os.path.join(self.test_dir, "stderr.log")

        print("first stdout from python")
        libc.printf(b"first stdout from c\n")

        print("first stderr from python", file=sys.stderr)
        libc.fprintf(c_stderr, b"first stderr from c\n")

        with redirect_stdout(stdout_log), redirect_stderr(stderr_log):
            print("redir stdout from python")
            print("redir stderr from python", file=sys.stderr)
            libc.printf(b"redir stdout from c\n")
            libc.fprintf(c_stderr, b"redir stderr from c\n")

        print("again stdout from python")
        libc.fprintf(c_stderr, b"again stderr from c\n")

        with open(stdout_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"redir stdout from python\n", "redir stdout from c\n"}, lines
            )

        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"redir stderr from python\n", "redir stderr from c\n"}, lines
            )

    def _redirect_large_buffer(self, print_fn, num_lines=500_000):
        stdout_log = os.path.join(self.test_dir, "stdout.log")

        with redirect_stdout(stdout_log):
            for i in range(num_lines):
                print_fn(i)

        with open(stdout_log) as fp:
            actual = {int(line.split(":")[1]) for line in fp}
            expected = set(range(num_lines))
            self.assertSetEqual(expected, actual)

    def test_redirect_large_buffer_py(self):
        def py_print(i):
            print(f"py:{i}")

        self._redirect_large_buffer(py_print)

    def test_redirect_large_buffer_c(self):
        def c_print(i):
            libc.printf(bytes(f"c:{i}\n", "utf-8"))

        self._redirect_large_buffer(c_print)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RedirectsTest`

**Functions defined**: `setUp`, `tearDown`, `test_redirect_invalid_std`, `test_redirect_stdout`, `test_redirect_stderr`, `test_redirect_both`, `_redirect_large_buffer`, `test_redirect_large_buffer_py`, `py_print`, `test_redirect_large_buffer_c`, `c_print`

**Key imports**: ctypes, os, shutil, sys, tempfile, unittest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/multiprocessing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `ctypes`
- `os`
- `shutil`
- `sys`
- `tempfile`
- `unittest`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/elastic/multiprocessing/redirects_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/multiprocessing`):

- [`api_test.py_docs.md`](./api_test.py_docs.md)
- [`tail_log_test.py_docs.md`](./tail_log_test.py_docs.md)
- [`test_api.py_docs.md`](./test_api.py_docs.md)


## Cross-References

- **File Documentation**: `redirects_test.py_docs.md`
- **Keyword Index**: `redirects_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
