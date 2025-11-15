# Documentation: `test/distributed/test_control_collectives.py`

## File Metadata

- **Path**: `test/distributed/test_control_collectives.py`
- **Size**: 7,398 bytes (7.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

from datetime import timedelta
from multiprocessing.pool import ThreadPool

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


# simple example of user code that takes the base class ControlCollectives
# and executes multiple different collectives
def simple_user_func(collectives: dist._ControlCollectives, rank: int) -> int:
    timeout = timedelta(seconds=10)
    # first a barrier
    collectives.barrier("1", timeout, True)
    # then an all_sum
    out = collectives.all_sum("2", rank, timeout)
    return out


class TestCollectives(TestCase):
    def test_barrier(self) -> None:
        store = dist.HashStore()

        world_size = 2

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            collectives.barrier("foo", timedelta(seconds=10), True)

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_broadcast(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            if rank == 2:
                collectives.broadcast_send("foo", b"data", timeout)
            else:
                out = collectives.broadcast_recv("foo", timeout)
                self.assertEqual(out, b"data")

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_gather(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            if rank == 2:
                out = collectives.gather_recv("foo", str(rank), timeout)
                self.assertEqual(out, [b"0", b"1", b"2", b"3"])
            else:
                collectives.gather_send("foo", str(rank), timeout)

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_scatter(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            if rank == 2:
                out = collectives.scatter_send(
                    "foo", [str(i) for i in range(world_size)], timeout
                )
            else:
                out = collectives.scatter_recv("foo", timeout)
            self.assertEqual(out, str(rank).encode())

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_all_sum(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            out = collectives.all_sum("foo", rank, timeout)
            self.assertEqual(out, sum(range(world_size)))

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_broadcast_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(Exception, "Wait timeout"):
            collectives.broadcast_recv("foo", timeout)

    def test_gather_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "gather failed -- missing ranks: 0, 2, 3"
        ):
            collectives.gather_recv("foo", "data", timeout)

    def test_scatter_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(Exception, "Wait timeout"):
            collectives.scatter_recv("foo", timeout)

    def test_all_gather_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "all_gather failed -- missing ranks: 0, 2, 3"
        ):
            collectives.all_gather("foo", "data", timeout)

    def test_barrier_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "barrier failed -- missing ranks: 0, 2, 3"
        ):
            collectives.barrier("foo", timeout, True)

    def test_all_sum_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "barrier failed -- missing ranks: 0, 2, 3"
        ):
            collectives.all_sum("foo", 1, timeout)

    def test_unique(self) -> None:
        store = dist.HashStore()

        collectives = dist._StoreCollectives(store, 1, 1)
        collectives.broadcast_send("foo", "bar")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.broadcast_send("foo", "bar")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.broadcast_recv("foo")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.gather_send("foo", "bar")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.gather_recv("foo", "asdf")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.scatter_send("foo", ["asdf"])

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.scatter_recv("foo")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.all_gather("foo", "bar")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.all_sum("foo", 2)

    def test_simple_user_func(self) -> None:
        store = dist.HashStore()
        world_size = 4

        def f(rank: int) -> None:
            # user need to create child collectives
            # but simple_user_func do not need to be changed for different child collectives
            store_collectives = dist._StoreCollectives(store, rank, world_size)
            out = simple_user_func(store_collectives, rank)
            self.assertEqual(out, sum(range(world_size)))

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))


if __name__ == "__main__":
    assert not torch.cuda._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCollectives`

**Functions defined**: `simple_user_func`, `test_barrier`, `f`, `test_broadcast`, `f`, `test_gather`, `f`, `test_scatter`, `f`, `test_all_sum`, `f`, `test_broadcast_timeout`, `test_gather_timeout`, `test_scatter_timeout`, `test_all_gather_timeout`, `test_barrier_timeout`, `test_all_sum_timeout`, `test_unique`, `test_simple_user_func`, `f`

**Key imports**: timedelta, ThreadPool, torch, torch.distributed as dist, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `datetime`: timedelta
- `multiprocessing.pool`: ThreadPool
- `torch`
- `torch.distributed as dist`
- `torch.testing._internal.common_utils`: run_tests, TestCase


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
python test/distributed/test_control_collectives.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed`):

- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`test_c10d_logger.py_docs.md`](./test_c10d_logger.py_docs.md)
- [`test_dist2.py_docs.md`](./test_dist2.py_docs.md)
- [`test_c10d_functional_native.py_docs.md`](./test_c10d_functional_native.py_docs.md)
- [`test_c10d_object_collectives.py_docs.md`](./test_c10d_object_collectives.py_docs.md)
- [`test_c10d_spawn_ucc.py_docs.md`](./test_c10d_spawn_ucc.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_control_collectives.py_docs.md`
- **Keyword Index**: `test_control_collectives.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
