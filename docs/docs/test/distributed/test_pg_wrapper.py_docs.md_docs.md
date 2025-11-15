# Documentation: `docs/test/distributed/test_pg_wrapper.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/test_pg_wrapper.py_docs.md`
- **Size**: 21,604 bytes (21.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/test_pg_wrapper.py`

## File Metadata

- **Path**: `test/distributed/test_pg_wrapper.py`
- **Size**: 17,972 bytes (17.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import os
import sys
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as c10d
from torch._C._distributed_c10d import _ProcessGroupWrapper


if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from test_c10d_common import LOOPBACK

from torch.testing._internal.common_distributed import (
    create_device,
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    with_dist_debug_levels,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


class AbstractProcessGroupWrapperTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _validate_error(self, exception, op_type, rank, tensor, verify_diff=True):
        err = str(exception)
        self.assertTrue(
            op_type in err, f"Got {err} but expected {op_type} to be in error."
        )
        # User doesn't call barrier with tensor.
        if op_type != "BARRIER":
            self.assertTrue(
                f"{list(tensor.shape)}" in err,
                f"Did not find shapes {list(tensor.shape)} in error {err}",
            )
            # For CUDA, only assert on device type, not index
            if "cuda" in str(tensor.device):
                self.assertTrue(
                    "cuda" in err, f"Did not find cuda device in error {err}"
                )
            else:
                self.assertTrue(
                    str(tensor.device) in err,
                    f"Did not find tensor device {str(tensor.device)} in error {err}",
                )
            # C++ and python type strings are not exactly the same.
            if "float" in str(tensor.dtype):
                self.assertTrue("Float" in err, "Expected Float type")
            elif "int" in str(tensor.dtype):
                self.assertTrue("Long" in err, "Expected Long type")
            else:
                self.fail(f"Unexpected dtype {str(tensor.dtype)} for error {err}")

            # Ensure sequence number is logged in error
            self.assertTrue("SequenceNumber" in err)
            # Ensure info about how collectives diff is in the error.
            if verify_diff:
                self.assertTrue(
                    "Collectives differ in the following" in err, f"Got error {err}"
                )

    def _test_collective_hang(self, wrapper_pg, use_cuda=False):
        # All ranks besides 1 call allreduce and wrapper_pg should detect a hang
        # and report an issue with rank 1.
        faulty_rank = 1
        if self.rank != faulty_rank:
            tensor = torch.randn(20, 10)
            if use_cuda:
                tensor = tensor.to(self.rank)

            if self.rank == 0:
                # Rank 0 reports faulty ranks
                err = f"Ranks {faulty_rank} failed to pass monitoredBarrier"
            else:
                err = "Please check rank 0 logs for faulty rank"

            # Gloo can sometimes throw the following error if a rank exits early
            # before rank 0 calls into the allreduce.
            err += "|Connection closed by peer|Connection reset by peer"
            with self.assertRaisesRegex(RuntimeError, err):
                wrapper_pg.allreduce([tensor])

    def _test_collectives_op_mismatch(self, wrapper_pg, use_cuda=False):
        tensor = torch.randn(20, 10)
        if use_cuda:
            tensor = tensor.to(self.rank)
        works = []
        # Run a few successful collectives
        for _ in range(500):
            work = wrapper_pg.allreduce([tensor])
            works.append(work)

        for w in works:
            w.wait()

        # Simulate mismatch: allreduce vs reduce.
        # Error including info about inconsistent collective, rank, tensor
        # shape, device, and dtype should be raised.
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.allreduce([tensor])
            else:
                wrapper_pg.reduce([tensor])
        self._validate_error(
            exception=cm.exception,
            op_type="ALLREDUCE" if self.rank == 0 else "REDUCE",
            rank=self.rank,
            tensor=tensor,
        )

        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.reduce([tensor])
            else:
                wrapper_pg.barrier()
        self._validate_error(
            exception=cm.exception,
            op_type="REDUCE" if self.rank == 0 else "BARRIER",
            rank=self.rank,
            tensor=tensor,
        )

        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.broadcast(tensor, 0)
            else:
                output_tensors = [
                    torch.zeros_like(tensor) for _ in range(self.world_size)
                ]
                wrapper_pg.allgather([output_tensors], [tensor])
        self._validate_error(
            exception=cm.exception,
            op_type="BROADCAST" if self.rank == 0 else "ALLGATHER",
            rank=self.rank,
            tensor=tensor,
        )

    def _test_collective_shape_mismatch(self, wrapper_pg, use_cuda=False):
        wrapper_pg.barrier()
        dim = 2 if self.rank == 0 else 10
        tensor = torch.randn(20, dim)
        if use_cuda:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            wrapper_pg.allreduce([tensor])
        self._validate_error(
            exception=cm.exception,
            op_type="ALLREDUCE",
            rank=self.rank,
            tensor=tensor,
        )

        # Check errors are raised when dimensionality of shapes is different
        tensor = torch.randn(20, 10, 2) if self.rank == 0 else torch.randn(20, 10)
        if use_cuda:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            wrapper_pg.allreduce([tensor])
        self._validate_error(
            exception=cm.exception,
            op_type="ALLREDUCE",
            rank=self.rank,
            tensor=tensor,
        )

        # Check shape errors with scatter
        input = [
            torch.tensor(
                [self.rank] if self.rank == 0 else [self.rank, self.rank],
                device=self.rank if use_cuda else "cpu",
            )
            for _ in range(self.world_size)
        ]
        outputs = [
            torch.tensor(
                [-1] if self.rank == 0 else [-1, -1],
                device=self.rank if use_cuda else "cpu",
            )
            for _ in range(self.world_size)
        ]
        root_rank = 0
        opts = c10d.ScatterOptions()
        opts.rootRank = root_rank
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == root_rank:
                wrapper_pg.scatter([outputs[self.rank]], [input], opts).wait()
            else:
                wrapper_pg.scatter([outputs[self.rank]], [], opts).wait()
        self._validate_error(
            exception=cm.exception,
            op_type="SCATTER",
            rank=self.rank,
            tensor=outputs[self.rank],
        )


# ASAN is not safe since we are spawning processes.
if not TEST_WITH_DEV_DBG_ASAN:

    @requires_gloo()
    @requires_nccl()
    class ProcessGroupNCCLWrapperTest(AbstractProcessGroupWrapperTest):
        def setUp(self):
            super(AbstractProcessGroupWrapperTest, self).setUp()
            self._spawn_processes()
            # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
            # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        @property
        def world_size(self) -> int:
            return 2

        def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size,
                store=store,
                timeout=timedelta(seconds=timeout),
            )
            if with_new_group:
                pg = c10d.new_group(backend="nccl", timeout=timedelta(seconds=timeout))
            else:
                _pg = c10d.ProcessGroupNCCL(
                    store,
                    self.rank,
                    self.world_size,
                    timeout=timedelta(seconds=timeout),
                )
                pg = c10d._create_process_group_wrapper(
                    _pg,
                    "unused",
                    store,
                    self.rank,
                    self.world_size,
                    timeout=timeout,
                )
            return pg

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        def test_collective_hang(self):
            pg = self._create_wrapper_pg(timeout=2.0)
            self._test_collective_hang(pg)

        # NOTE: these tests are separated by debug level instead of combined into
        # one due to https://github.com/pytorch/pytorch/issues/55967, they can be
        # combined after that is resolved.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_collectives_op_mismatch_debug_mode(self):
            pg = self._create_wrapper_pg(with_new_group=True)
            self._test_collectives_op_mismatch(pg, use_cuda=True)
            self._test_nccl_only_op_mismatch(pg)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["OFF"])
        def test_collectives_op_mismatch(self):
            pg = self._create_wrapper_pg(with_new_group=False)
            self._test_collectives_op_mismatch(pg, use_cuda=True)
            self._test_nccl_only_op_mismatch(pg)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_collective_shape_mismatch_debug_mode_detail(self):
            pg = self._create_wrapper_pg(with_new_group=True)
            self._test_collective_shape_mismatch(pg, use_cuda=True)
            self._test_nccl_only_shape_mismatch(pg)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["OFF"])
        def test_collective_shape_mismatch_debug_mode_off(self):
            pg = self._create_wrapper_pg(with_new_group=False)
            self._test_collective_shape_mismatch(pg, use_cuda=True)
            self._test_nccl_only_shape_mismatch(pg)

        def _test_nccl_only_op_mismatch(self, wrapper_pg):
            device = f"cuda:{self.rank}"
            with self.assertRaisesRegex(RuntimeError, ".*") as cm:
                output = torch.zeros(4 + self.rank, device=device)
                input = torch.ones(4 * self.world_size, device=device)
                if self.rank == 0:
                    wrapper_pg._allgather_base(output, input).wait()
                else:
                    wrapper_pg._reduce_scatter_base(output, input).wait()

            op_type = "ALLGATHER_BASE" if self.rank == 0 else "REDUCE_SCATTER_BASE"
            self._validate_error(
                exception=cm.exception,
                op_type=op_type,
                rank=self.rank,
                tensor=input,
            )

        def _test_nccl_only_shape_mismatch(self, wrapper_pg):
            device = f"cuda:{self.rank}"
            with self.assertRaisesRegex(RuntimeError, ".*") as cm:
                output = torch.zeros(4 + self.rank, device=device)
                input = torch.ones(4 * (self.world_size + 1), device=device)

                wrapper_pg._reduce_scatter_base(output, input).wait()
            self._validate_error(
                exception=cm.exception,
                op_type="REDUCE_SCATTER_BASE",
                rank=self.rank,
                tensor=input,
                verify_diff=False,
            )
            with self.assertRaisesRegex(RuntimeError, ".*") as cm:
                output = torch.zeros(4, device=device)
                input = torch.ones((4 + self.rank) * self.world_size, device=device)

                wrapper_pg._reduce_scatter_base(output, input).wait()
            self._validate_error(
                exception=cm.exception,
                op_type="REDUCE_SCATTER_BASE",
                rank=self.rank,
                tensor=input,
                verify_diff=False,
            )

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_coalescing_manager_debug_mode_detail(self):
            """
            Tests that coalescing manager w/TORCH_DISTRIBUTED_DEBUG
            does not crash: https://github.com/pytorch/pytorch/issues/109520
            """
            torch.cuda.set_device(self.rank)
            pg = self._create_wrapper_pg(with_new_group=True)
            dev = torch.cuda.current_device()
            pg._start_coalescing(torch.device(dev))
            pg.allreduce([torch.ones(1, device=dev)])
            pg._end_coalescing(torch.device(dev))

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        @patch("torch.distributed.distributed_c10d._GLOO_AVAILABLE", False)
        def test_debug_level_detail_no_gloo(self):
            with self.assertRaisesRegex(
                AssertionError, "ProcessGroupWrapper unsupported without GLOO backend"
            ):
                self._create_wrapper_pg()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @patch("torch.distributed.distributed_c10d._GLOO_AVAILABLE", False)
        def test_new_group_no_gloo(self):
            def patched_isinstance(obj, clazz):
                if clazz is _ProcessGroupWrapper:
                    raise NameError
                else:
                    return isinstance(obj, clazz)

            with patch(
                "torch.distributed.distributed_c10d.isinstance",
                side_effect=patched_isinstance,
            ):
                self._create_wrapper_pg(with_new_group=True)
                # nothing to assert, isinstance(pg, _ProcessGroupWrapper)
                # should never be invoked since it is proceeded by
                # _GLOO_AVAILABLE check, this test will fail on
                # an unexpected NameError if not.


@requires_gloo()
class ProcessGroupGlooWrapperTest(AbstractProcessGroupWrapperTest):
    def opts(self, threads=2, timeout=10.0):
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = timeout
        opts._devices = [create_device(interface=LOOPBACK)]
        opts._threads = threads
        return opts

    def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )
        if with_new_group:
            pg = c10d.new_group(backend="gloo")
        else:
            _pg = c10d.ProcessGroupGloo(
                store, self.rank, self.world_size, self.opts(timeout=timeout)
            )
            pg = c10d._create_process_group_wrapper(
                _pg,
                "unused",
                store,
                self.rank,
                self.world_size,
                timeout=timeout,
            )
        return pg

    def test_collective_hang(self):
        pg = self._create_wrapper_pg(timeout=2.0)
        self._test_collective_hang(pg)

    # NOTE: these tests are separated by debug level instead of combined into
    # one due to https://github.com/pytorch/pytorch/issues/55967, they can be
    # combined after that is resolved.
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg)

    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch_debug_mode_off(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_cuda_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch_cuda(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_cuda_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch_cuda(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg, use_cuda=True)


if __name__ == "__main__":
    assert not torch.cuda._initialized, (
        "test_pg_wrapper must not have initialized CUDA context on main process"
    )

    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 30 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AbstractProcessGroupWrapperTest`, `ProcessGroupNCCLWrapperTest`, `ProcessGroupGlooWrapperTest`

**Functions defined**: `setUp`, `_validate_error`, `_test_collective_hang`, `_test_collectives_op_mismatch`, `_test_collective_shape_mismatch`, `setUp`, `world_size`, `_create_wrapper_pg`, `test_collective_hang`, `test_collectives_op_mismatch_debug_mode`, `test_collectives_op_mismatch`, `test_collective_shape_mismatch_debug_mode_detail`, `test_collective_shape_mismatch_debug_mode_off`, `_test_nccl_only_op_mismatch`, `_test_nccl_only_shape_mismatch`, `test_coalescing_manager_debug_mode_detail`, `test_debug_level_detail_no_gloo`, `test_new_group_no_gloo`, `patched_isinstance`, `opts`

**Key imports**: os, sys, timedelta, patch, torch, torch.distributed as c10d, _ProcessGroupWrapper, LOOPBACK, run_tests, TEST_WITH_DEV_DBG_ASAN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `datetime`: timedelta
- `unittest.mock`: patch
- `torch`
- `torch.distributed as c10d`
- `torch._C._distributed_c10d`: _ProcessGroupWrapper
- `test_c10d_common`: LOOPBACK
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


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
python test/distributed/test_pg_wrapper.py
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

- **File Documentation**: `test_pg_wrapper.py_docs.md`
- **Keyword Index**: `test_pg_wrapper.py_kw.md`
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
python docs/test/distributed/test_pg_wrapper.py_docs.md
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

- **File Documentation**: `test_pg_wrapper.py_docs.md_docs.md`
- **Keyword Index**: `test_pg_wrapper.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
