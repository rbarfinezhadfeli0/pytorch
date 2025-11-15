# Documentation: `test/distributed/elastic/utils/util_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/utils/util_test.py`
- **Size**: 8,415 bytes (8.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from multiprocessing.pool import ThreadPool
from unittest import mock

import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util
from torch.distributed.elastic.utils.logging import get_logger
from torch.testing._internal.common_utils import run_tests, TestCase


class MockStore:
    _TEST_TIMEOUT = 1234

    def __init__(self) -> None:
        self.ops = []

    def set_timeout(self, timeout: float) -> None:
        self.ops.append(("set_timeout", timeout))

    @property
    def timeout(self) -> datetime.timedelta:
        self.ops.append(("timeout",))

        return datetime.timedelta(seconds=self._TEST_TIMEOUT)

    def set(self, key: str, value: str) -> None:
        self.ops.append(("set", key, value))

    def get(self, key: str) -> str:
        self.ops.append(("get", key))
        return "value"

    def multi_get(self, keys: list[str]) -> list[str]:
        self.ops.append(("multi_get", keys))
        return ["value"] * len(keys)

    def add(self, key: str, val: int) -> int:
        self.ops.append(("add", key, val))
        return 3

    def wait(self, keys: list[str]) -> None:
        self.ops.append(("wait", keys))


class StoreUtilTest(TestCase):
    def test_get_all_rank_0(self):
        world_size = 3

        store = MockStore()

        store_util.get_all(store, 0, "test/store", world_size)

        self.assertListEqual(
            store.ops,
            [
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
                ("wait", ["test/store/finished/last_member"]),
            ],
        )

    def test_get_all_rank_n(self):
        store = MockStore()
        world_size = 3
        store_util.get_all(store, 1, "test/store", world_size)

        self.assertListEqual(
            store.ops,
            [
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
            ],
        )

    def test_synchronize(self):
        store = MockStore()

        data = b"data0"
        store_util.synchronize(store, data, 0, 3, key_prefix="test/store")

        self.assertListEqual(
            store.ops,
            [
                ("timeout",),
                ("set_timeout", datetime.timedelta(seconds=300)),
                ("set", "test/store0", data),
                ("multi_get", ["test/store0", "test/store1", "test/store2"]),
                ("add", "test/store/finished/num_members", 1),
                ("set", "test/store/finished/last_member", "<val_ignored>"),
                ("wait", ["test/store/finished/last_member"]),
                ("set_timeout", datetime.timedelta(seconds=store._TEST_TIMEOUT)),
            ],
        )

    def test_synchronize_hash_store(self) -> None:
        N = 4

        store = dist.HashStore()

        def f(i: int):
            return store_util.synchronize(
                store, f"data{i}", i, N, key_prefix="test/store"
            )

        with ThreadPool(N) as pool:
            out = pool.map(f, range(N))

        self.assertListEqual(out, [[f"data{i}".encode() for i in range(N)]] * N)

    def test_barrier(self):
        store = MockStore()

        store_util.barrier(store, 3, key_prefix="test/store")

        self.assertListEqual(
            store.ops,
            [
                ("timeout",),
                ("set_timeout", datetime.timedelta(seconds=300)),
                ("add", "test/store/num_members", 1),
                ("set", "test/store/last_member", "<val_ignored>"),
                ("wait", ["test/store/last_member"]),
                ("set_timeout", datetime.timedelta(seconds=store._TEST_TIMEOUT)),
            ],
        )

    def test_barrier_timeout_rank_tracing(self):
        N = 3

        store = dist.HashStore()

        def run_barrier_for_rank(i: int):
            try:
                store_util.barrier(
                    store,
                    N,
                    key_prefix="test/store",
                    barrier_timeout=0.1,
                    rank=i,
                    rank_tracing_decoder=lambda x: f"Rank {x} host",
                    trace_timeout=0.01,
                )
            except Exception as e:
                return str(e)
            return ""

        with ThreadPool(N - 1) as pool:
            outputs: list[str] = pool.map(run_barrier_for_rank, range(N - 1))

        self.assertTrue(any("missing_ranks=[Rank " in msg for msg in outputs))

        self.assertTrue(
            any(
                "check rank 0 (Rank 0 host) for missing rank info" in msg
                for msg in outputs
            )
        )

    def test_barrier_timeout_operations(self):
        import torch

        DistStoreError = torch._C._DistStoreError

        N = 3
        store = MockStore()

        # rank 0
        with mock.patch.object(store, "wait") as wait_mock:
            wait_mock.side_effect = [DistStoreError("test"), None, None]

            with self.assertRaises(DistStoreError):
                store_util.barrier(
                    store,
                    N,
                    key_prefix="test/store",
                    barrier_timeout=1,
                    rank=0,
                    rank_tracing_decoder=lambda x: f"Rank {x} host",
                    trace_timeout=0.1,
                )

            self.assertListEqual(
                store.ops,
                [
                    ("timeout",),
                    ("set_timeout", datetime.timedelta(seconds=1)),
                    ("add", "test/store/num_members", 1),
                    ("set", "test/store/last_member", "<val_ignored>"),
                    # wait for last member is mocked
                    ("set", "test/store0/TRACE", "<val_ignored>"),
                    # wait for each rank is mocked
                    ("set", "test/store/TRACING_GATE", "<val_ignored>"),
                ],
            )

        # rank 1
        with mock.patch.object(store, "wait") as wait_mock:
            store.ops = []

            wait_mock.side_effect = [
                DistStoreError("test"),
                None,
            ]

            with self.assertRaises(DistStoreError):
                store_util.barrier(
                    store,
                    N,
                    key_prefix="test/store",
                    barrier_timeout=1,
                    rank=1,
                    rank_tracing_decoder=lambda x: f"Rank {x} host",
                    trace_timeout=0.1,
                )

            self.assertListEqual(
                store.ops,
                [
                    ("timeout",),
                    ("set_timeout", datetime.timedelta(seconds=1)),
                    ("add", "test/store/num_members", 1),
                    ("set", "test/store/last_member", "<val_ignored>"),
                    ("set", "test/store1/TRACE", "<val_ignored>"),
                    # wait for gate is mocked
                ],
            )

    def test_barrier_hash_store(self) -> None:
        N = 4

        store = dist.HashStore()

        def f(i: int):
            store_util.barrier(store, N, key_prefix="test/store")

        with ThreadPool(N) as pool:
            out = pool.map(f, range(N))

        self.assertEqual(out, [None] * N)


class UtilTest(TestCase):
    def test_get_logger_different(self):
        logger1 = get_logger("name1")
        logger2 = get_logger("name2")
        self.assertNotEqual(logger1.name, logger2.name)

    def test_get_logger(self):
        logger1 = get_logger()
        self.assertEqual(__name__, logger1.name)

    def test_get_logger_none(self):
        logger1 = get_logger(None)
        self.assertEqual(__name__, logger1.name)

    def test_get_logger_custom_name(self):
        logger1 = get_logger("test.module")
        self.assertEqual("test.module", logger1.name)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 23 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MockStore`, `StoreUtilTest`, `UtilTest`

**Functions defined**: `__init__`, `set_timeout`, `timeout`, `set`, `get`, `multi_get`, `add`, `wait`, `test_get_all_rank_0`, `test_get_all_rank_n`, `test_synchronize`, `test_synchronize_hash_store`, `f`, `test_barrier`, `test_barrier_timeout_rank_tracing`, `run_barrier_for_rank`, `test_barrier_timeout_operations`, `test_barrier_hash_store`, `f`, `test_get_logger_different`

**Key imports**: datetime, ThreadPool, mock, torch.distributed as dist, torch.distributed.elastic.utils.store as store_util, get_logger, run_tests, TestCase, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/utils`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `datetime`
- `multiprocessing.pool`: ThreadPool
- `unittest`: mock
- `torch.distributed as dist`
- `torch.distributed.elastic.utils.store as store_util`
- `torch.distributed.elastic.utils.logging`: get_logger
- `torch.testing._internal.common_utils`: run_tests, TestCase
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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
python test/distributed/elastic/utils/util_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/utils`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`logging_test.py_docs.md`](./logging_test.py_docs.md)
- [`distributed_test.py_docs.md`](./distributed_test.py_docs.md)


## Cross-References

- **File Documentation**: `util_test.py_docs.md`
- **Keyword Index**: `util_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
