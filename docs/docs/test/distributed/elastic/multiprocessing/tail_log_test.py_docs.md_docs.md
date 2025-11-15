# Documentation: `docs/test/distributed/elastic/multiprocessing/tail_log_test.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/elastic/multiprocessing/tail_log_test.py_docs.md`
- **Size**: 11,821 bytes (11.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/elastic/multiprocessing/tail_log_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/multiprocessing/tail_log_test.py`
- **Size**: 8,842 bytes (8.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from concurrent.futures import wait
from concurrent.futures._base import ALL_COMPLETED
from concurrent.futures.thread import ThreadPoolExecutor
from unittest import mock

from torch.distributed.elastic.multiprocessing.tail_log import TailLog


def write(max: int, sleep: float, file: str):
    with open(file, "w") as fp:
        for i in range(max):
            print(i, file=fp, flush=True)
            time.sleep(sleep)


class TailLogTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")
        self.threadpool = ThreadPoolExecutor()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tail(self):
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into an IOString
        and validate that all lines are accounted for.
        """
        nprocs = 32
        max = 1000
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = io.StringIO()
        tail = TailLog(
            name="writer", log_files=log_files, dst=dst, interval_sec=interval_sec
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)

        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)

        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()

        dst.seek(0)
        actual: dict[int, set[int]] = {}

        for line in dst.readlines():
            header, num = line.split(":")
            nums = actual.setdefault(header, set())
            nums.add(int(num))

        self.assertEqual(nprocs, len(actual))
        self.assertEqual(
            {f"[writer{i}]": set(range(max)) for i in range(nprocs)}, actual
        )
        self.assertTrue(tail.stopped())

    def test_tail_write_to_dst_file(self):
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into a temp file
        and validate that all lines are accounted for.
        """
        nprocs = 32
        max = 1000
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = os.path.join(self.test_dir, "tailed_stdout.log")
        dst_file = open(dst, "w", buffering=1)
        tail = TailLog(
            name="writer", log_files=log_files, dst=dst_file, interval_sec=interval_sec
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)

        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)

        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()
        dst_file.close()

        actual: dict[int, set[int]] = {}
        with open(dst) as read_dst_file:
            for line in read_dst_file:
                header, num = line.split(":")
                nums = actual.setdefault(header, set())
                nums.add(int(num))

        self.assertEqual(nprocs, len(actual))
        self.assertEqual(
            {f"[writer{i}]": set(range(max)) for i in range(nprocs)}, actual
        )
        self.assertTrue(tail.stopped())

    def test_tail_with_custom_prefix(self):
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into an IOString
        and validate that all lines are accounted for.
        """
        nprocs = 3
        max = 10
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = io.StringIO()
        log_line_prefixes = {n: f"[worker{n}][{n}]:" for n in range(nprocs)}
        tail = TailLog(
            "writer",
            log_files,
            dst,
            interval_sec=interval_sec,
            log_line_prefixes=log_line_prefixes,
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)
        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)
        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()
        dst.seek(0)

        headers: set[str] = set()
        for line in dst.readlines():
            header, _ = line.split(":")
            headers.add(header)
        self.assertEqual(nprocs, len(headers))
        for i in range(nprocs):
            self.assertIn(f"[worker{i}][{i}]", headers)
        self.assertTrue(tail.stopped())

    def test_tail_with_custom_filter(self):
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into an IOString
        and validate that all lines are accounted for.
        """
        nprocs = 3
        max = 20
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = io.StringIO()
        tail = TailLog(
            "writer",
            log_files,
            dst,
            interval_sec=interval_sec,
            log_line_filter=lambda line: "2" in line,  # only print lines containing '2'
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)
        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)
        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()
        dst.seek(0)

        actual: dict[int, set[int]] = {}
        for line in dst.readlines():
            header, num = line.split(":")
            nums = actual.setdefault(header, set())
            nums.add(int(num))
        self.assertEqual(nprocs, len(actual))
        self.assertEqual({f"[writer{i}]": {2, 12} for i in range(nprocs)}, actual)
        self.assertTrue(tail.stopped())

    def test_tail_no_files(self):
        """
        Ensures that the log tail can gracefully handle no log files
        in which case it does nothing.
        """
        tail = TailLog("writer", log_files={}, dst=sys.stdout).start()
        self.assertFalse(tail.stopped())
        tail.stop()
        self.assertTrue(tail.stopped())

    def test_tail_logfile_never_generates(self):
        """
        Ensures that we properly shutdown the threadpool
        even when the logfile never generates.
        """

        tail = TailLog("writer", log_files={0: "foobar.log"}, dst=sys.stdout).start()
        tail.stop()
        self.assertTrue(tail.stopped())
        self.assertTrue(tail._threadpool._shutdown)

    @mock.patch("torch.distributed.elastic.multiprocessing.tail_log.logger")
    def test_tail_logfile_error_in_tail_fn(self, mock_logger):
        """
        Ensures that when there is an error in the tail_fn (the one that runs in the
        threadpool), it is dealt with and raised properly.
        """

        # try giving tail log a directory (should fail with an IsADirectoryError
        tail = TailLog("writer", log_files={0: self.test_dir}, dst=sys.stdout).start()
        tail.stop()

        mock_logger.exception.assert_called_once()

```



## High-Level Overview

"""        writer() writes 0 - max (on number on each line) to a log file.        Run nprocs such writers and tail the log files into an IOString        and validate that all lines are accounted for.

This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TailLogTest`

**Functions defined**: `write`, `setUp`, `tearDown`, `test_tail`, `test_tail_write_to_dst_file`, `test_tail_with_custom_prefix`, `test_tail_with_custom_filter`, `test_tail_no_files`, `test_tail_logfile_never_generates`, `test_tail_logfile_error_in_tail_fn`

**Key imports**: io, os, shutil, sys, tempfile, time, unittest, wait, ALL_COMPLETED, ThreadPoolExecutor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/multiprocessing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `os`
- `shutil`
- `sys`
- `tempfile`
- `time`
- `unittest`
- `concurrent.futures`: wait
- `concurrent.futures._base`: ALL_COMPLETED
- `concurrent.futures.thread`: ThreadPoolExecutor
- `torch.distributed.elastic.multiprocessing.tail_log`: TailLog


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/elastic/multiprocessing/tail_log_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/multiprocessing`):

- [`api_test.py_docs.md`](./api_test.py_docs.md)
- [`redirects_test.py_docs.md`](./redirects_test.py_docs.md)
- [`test_api.py_docs.md`](./test_api.py_docs.md)


## Cross-References

- **File Documentation**: `tail_log_test.py_docs.md`
- **Keyword Index**: `tail_log_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/elastic/multiprocessing`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/elastic/multiprocessing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/distributed/elastic/multiprocessing/tail_log_test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/elastic/multiprocessing`):

- [`redirects_test.py_kw.md_docs.md`](./redirects_test.py_kw.md_docs.md)
- [`test_api.py_docs.md_docs.md`](./test_api.py_docs.md_docs.md)
- [`test_api.py_kw.md_docs.md`](./test_api.py_kw.md_docs.md)
- [`tail_log_test.py_kw.md_docs.md`](./tail_log_test.py_kw.md_docs.md)
- [`redirects_test.py_docs.md_docs.md`](./redirects_test.py_docs.md_docs.md)
- [`api_test.py_kw.md_docs.md`](./api_test.py_kw.md_docs.md)
- [`api_test.py_docs.md_docs.md`](./api_test.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `tail_log_test.py_docs.md_docs.md`
- **Keyword Index**: `tail_log_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
