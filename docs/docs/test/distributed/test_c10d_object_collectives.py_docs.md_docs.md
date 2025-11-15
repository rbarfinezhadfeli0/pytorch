# Documentation: `docs/test/distributed/test_c10d_object_collectives.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/test_c10d_object_collectives.py_docs.md`
- **Size**: 7,664 bytes (7.48 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/test_c10d_object_collectives.py`

## File Metadata

- **Path**: `test/distributed/test_c10d_object_collectives.py`
- **Size**: 4,546 bytes (4.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import DistributedTestBase, TEST_SKIPS
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfHpu,
    TEST_WITH_DEV_DBG_ASAN,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
device_count = torch.accelerator.device_count()


def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if device_type != "cpu" and device_count < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.pg = self.create_pg(device=device_type)
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapper


class TestObjectCollectives(DistributedTestBase):
    @with_comms()
    def test_all_gather_object(self):
        output = [None] * dist.get_world_size()
        dist.all_gather_object(object_list=output, obj=self.rank)

        for i, v in enumerate(output):
            self.assertEqual(i, v, f"rank: {self.rank}")

    @with_comms()
    def test_gather_object(self):
        output = [None] * dist.get_world_size() if self.rank == 0 else None
        dist.gather_object(obj=self.rank, object_gather_list=output)

        if self.rank == 0:
            for i, v in enumerate(output):
                self.assertEqual(i, v, f"rank: {self.rank}")

    @skipIfHpu
    @with_comms()
    def test_send_recv_object_list(self):
        val = 99 if self.rank == 0 else None
        object_list = [val] * dist.get_world_size()
        if self.rank == 0:
            dist.send_object_list(object_list, 1)
        if self.rank == 1:
            dist.recv_object_list(object_list, 0)

        if self.rank < 2:
            self.assertEqual(99, object_list[0])
        else:
            self.assertEqual(None, object_list[0])

    @with_comms()
    def test_broadcast_object_list(self):
        val = 99 if self.rank == 0 else None
        object_list = [val] * dist.get_world_size()
        # TODO test with broadcast_object_list's device argument
        dist.broadcast_object_list(object_list=object_list)

        self.assertEqual(99, object_list[0])

    @with_comms()
    def test_scatter_object_list(self):
        input_list = list(range(dist.get_world_size())) if self.rank == 0 else None
        output_list = [None]
        dist.scatter_object_list(
            scatter_object_output_list=output_list, scatter_object_input_list=input_list
        )

        self.assertEqual(self.rank, output_list[0])

    # Test Object Collectives With Sub Pg

    def setup_sub_pg(self):
        rank = dist.get_rank()
        base_rank = rank - (rank % 2)
        ranks = [base_rank, base_rank + 1]
        my_pg = dist.new_group(ranks, use_local_synchronization=True)
        return rank, ranks, my_pg

    @with_comms()
    def test_subpg_scatter_object(self):
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None]
        dist.scatter_object_list(out_list, ranks, src=ranks[0], group=my_pg)
        self.assertEqual(rank, out_list[0])

    @with_comms()
    def test_subpg_all_gather_object(self):
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None] * len(ranks)
        dist.all_gather_object(out_list, rank, group=my_pg)
        self.assertEqual(ranks, out_list)

    @with_comms()
    def test_subpg_gather_object(self):
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None] * len(ranks) if rank == ranks[0] else None
        dist.gather_object(rank, out_list, dst=ranks[0], group=my_pg)
        if rank == ranks[0]:
            self.assertEqual(ranks, out_list)

    @with_comms()
    def test_subpg_broadcast_object(self):
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None]
        if rank == ranks[0]:
            out_list[0] = rank
        dist.broadcast_object_list(out_list, src=ranks[0], group=my_pg)
        self.assertEqual(ranks[0], out_list[0])


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestObjectCollectives`

**Functions defined**: `with_comms`, `wrapper`, `test_all_gather_object`, `test_gather_object`, `test_send_recv_object_list`, `test_broadcast_object_list`, `test_scatter_object_list`, `setup_sub_pg`, `test_subpg_scatter_object`, `test_subpg_all_gather_object`, `test_subpg_gather_object`, `test_subpg_broadcast_object`

**Key imports**: sys, partial, wraps, torch, torch.distributed as dist, DistributedTestBase, TEST_SKIPS


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `functools`: partial, wraps
- `torch`
- `torch.distributed as dist`
- `torch.testing._internal.common_distributed`: DistributedTestBase, TEST_SKIPS


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
python test/distributed/test_c10d_object_collectives.py
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
- [`test_c10d_spawn_ucc.py_docs.md`](./test_c10d_spawn_ucc.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_c10d_object_collectives.py_docs.md`
- **Keyword Index**: `test_c10d_object_collectives.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/test_c10d_object_collectives.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed`):

- [`test_run.py_kw.md_docs.md`](./test_run.py_kw.md_docs.md)
- [`test_inductor_collectives.py_docs.md_docs.md`](./test_inductor_collectives.py_docs.md_docs.md)
- [`test_control_collectives.py_kw.md_docs.md`](./test_control_collectives.py_kw.md_docs.md)
- [`test_c10d_gloo.py_docs.md_docs.md`](./test_c10d_gloo.py_docs.md_docs.md)
- [`test_collective_utils.py_kw.md_docs.md`](./test_collective_utils.py_kw.md_docs.md)
- [`test_data_parallel.py_kw.md_docs.md`](./test_data_parallel.py_kw.md_docs.md)
- [`test_overlap_bucketing_unit.py_kw.md_docs.md`](./test_overlap_bucketing_unit.py_kw.md_docs.md)
- [`test_c10d_nccl.py_kw.md_docs.md`](./test_c10d_nccl.py_kw.md_docs.md)
- [`test_multi_threaded_pg.py_docs.md_docs.md`](./test_multi_threaded_pg.py_docs.md_docs.md)
- [`argparse_util_test.py_kw.md_docs.md`](./argparse_util_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_c10d_object_collectives.py_docs.md_docs.md`
- **Keyword Index**: `test_c10d_object_collectives.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
