# Documentation: `test/distributed/test_c10d_nccl.py`

## File Metadata

- **Path**: `test/distributed/test_c10d_nccl.py`
- **Size**: 255,058 bytes (249.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import copy
import json
import logging
import os
import pickle
import random
import re
import signal
import sys
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import auto, Enum
from itertools import chain, product
from unittest import mock, SkipTest

import torch
import torch.distributed as c10d
import torch.distributed._functional_collectives as _functional_collectives
from torch.distributed.distributed_c10d import SHRINK_ABORT as NCCL_SHRINK_ABORT


if not c10d.is_available() or not c10d.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


import test_c10d_common
from test_c10d_common import (
    ConvNet,
    DoubleGpuNet,
    FFTModel,
    gpus_for_rank,
    ModuleForDdpCommHook,
)

import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch._C._distributed_c10d import ErrorType, OpType, WorkResult
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_cuda import _get_torch_rocm_version, TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    get_required_world_size,
    get_timeout,
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_multicast_support,
    requires_nccl,
    requires_nccl_shrink,
    requires_nccl_version,
    requires_world_size,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
    sm_is_or_higher_than,
    TEST_SKIPS,
    with_dist_debug_levels,
    with_nccl_blocking_wait,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_SANDCASTLE,
    MI300_ARCH,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    runOnRocmArch,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

BFLOAT16_AVAILABLE = torch.cuda.is_available() and (
    torch.version.cuda is not None or torch.version.hip is not None
)


_start_time = time.time()
_logger = logging.getLogger(__name__)


def _ts():
    return time.time() - _start_time


def configure(level=logging.INFO, force=False):
    try:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            force=force,
        )
    except TypeError:
        logging.basicConfig(
            level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
        )


def log_test_info(rank, message):
    _logger.info("[%7.3fs][Rank %s] %s", _ts(), rank, message)


def log_test_success(rank, message):
    _logger.info("[%7.3fs][Rank %s] ✅ %s", _ts(), rank, message)


def log_test_validation(rank, message):
    _logger.info("[%7.3fs][Rank %s] ✓ %s", _ts(), rank, message)


def log_test_warning(rank, message):
    _logger.warning("[%7.3fs][Rank %s] ⚠️ %s", _ts(), rank, message)


def log_test_error(rank, message):
    _logger.error("[%7.3fs][Rank %s] ✗ %s", _ts(), rank, message)


_log_configure = configure


_log_configure(level=logging.INFO, force=True)


class RendezvousEnvTest(TestCase):
    @retry_on_connect_failures
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def test_common_errors(self):
        vars = {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(common.find_free_port()),
        }

        class Env:
            def __init__(self, vars):
                self.env_patcher = mock.patch.dict(os.environ, vars, clear=True)

            def __enter__(self):
                self.env_patcher.start()

            def __exit__(self, type, value, traceback):
                self.env_patcher.stop()

        def without(d, key):
            d = d.copy()
            d.pop(key)
            return d

        def withouts(d, keys):
            d = d.copy()
            for key in keys:
                d.pop(key)
            return d

        with Env(without(vars, "WORLD_SIZE")):
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="nccl", world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            with self.assertRaisesRegex(ValueError, "RANK expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="nccl", rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            c10d.init_process_group(backend="nccl", rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(vars):
            c10d.init_process_group(backend="nccl")
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "MASTER_ADDR")):
            self.assertEqual(None, os.environ.get("MASTER_ADDR"))
            with self.assertRaisesRegex(ValueError, "MASTER_ADDR expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "MASTER_PORT")):
            self.assertEqual(None, os.environ.get("MASTER_PORT"))
            with self.assertRaisesRegex(ValueError, "MASTER_PORT expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "WORLD_SIZE")):
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            gen = c10d.rendezvous(f"env://?world_size={1}")
            _, _, size = next(gen)
            self.assertEqual(size, 1)

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            gen = c10d.rendezvous(f"env://?rank={0}")
            _, rank, _ = next(gen)
            self.assertEqual(rank, 0)

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            gen = c10d.rendezvous(f"env://?rank={0}&world_size={1}")
            _, rank, size = next(gen)
            self.assertEqual(rank, 0)
            self.assertEqual(size, 1)


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    @requires_nccl()
    @retry_on_connect_failures
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def test_default_store_timeout_nccl(self):
        self._test_default_store_timeout("nccl")


class ProcessGroupNCCLNoGPUTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        super().setUp()
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        pass

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(TEST_CUDA, "GPUs are available, skipping test")
    def test_init_no_gpus(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        with self.assertRaisesRegex(
            ValueError, "ProcessGroupNCCL is only supported with GPUs, no GPUs found!"
        ):
            c10d.ProcessGroupNCCL(store, self.rank, self.world_size)


class ProcessGroupNCCLInitTest(MultiProcessTestCase):
    device_type = "cuda"

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        dm = torch.get_device_module(self.device_type)
        return dm.device_count()

    @property
    def device(self):
        return torch.device(self.device_type, self.rank % self.world_size)

    # A helper with the must-needed init args for test infra.
    # kwargs can be filled in by individual init tests.
    def _init_process_group(self, **kwargs):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            **kwargs,
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_init_wo_backend_str(self):
        self._init_process_group(device_id=self.device)
        x = torch.empty(1, device=self.device)
        c10d.all_reduce(x)

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_scalable_init(self):
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = "1"
        self._init_process_group(device_id=self.device)
        x = torch.empty(1, device=self.device)
        c10d.all_reduce(x)
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = "0"


class ProcessGroupNCCLGroupTest(MultiProcessTestCase):
    def _create_process_group_nccl(self, store, opts, device_id=None):
        # create nccl processgroup with opts
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=opts,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    def setUp(self):
        super().setUp()

        # These tests are expected to throw SIGABRT(6);
        # But if we are in Sandcastle, `skip_but_pass_in_sandcastle` would return 0.
        TEST_NAN_ASSERT_RETURN = 0 if IS_SANDCASTLE else signal.SIGABRT
        self.special_return_code_checks = {
            self.test_nan_assert_float16.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float32.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float64.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_bfloat16.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float8_e4m3fn.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float8_e5m2.__wrapped__: TEST_NAN_ASSERT_RETURN,
        }

        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # self.num_gpus = torch.cuda.device_count()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return get_required_world_size(self, 2)

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "nccl")

    @property
    def destroy_pg_upon_exit(self) -> bool:
        # This TestCase focuses on creation, destroy and abort of PG's. So it
        # does not need auto-destroy upon exit.
        return False

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 1 GPU")
    @skip_if_lt_x_gpu(1)
    def test_nccl_dist_backend_error(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())

        # Both rank 0 and 1 will use the same CUDA device resulting in ncclInvalidUsage
        with self.assertRaises(dist.DistBackendError) as cm:
            dist.broadcast(torch.tensor([1, 2, 3]).cuda(), 0)
        self.assertTrue(isinstance(cm.exception, dist.DistError))

        self.assertIsInstance(cm.exception, RuntimeError)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_abort_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)

        def abortpg():
            c10d.distributed_c10d._get_default_group()._get_backend(
                torch.device(device)
            ).abort()

        # Initialize DDP to ensure "destroy_process_group" will not call
        # ProcessGroupNCCL destructor since DDP holds a reference to process group.
        # Run a single iteration of DDP to initialize state.
        model = DistributedDataParallel(
            torch.nn.Linear(10, 10).to(device), device_ids=[device]
        )
        model(t).sum().backward()

        # Now simulate collective getting stuck and abort gets us unstuck
        if self.rank == 0:
            dist.all_reduce(t)

            # Schedule thread before we get stuck to abort pg.
            thread = threading.Thread(target=abortpg)
            thread.start()

            # We would get stuck here due to d2h if we didn't abort.
            t.cpu()

            thread.join()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("eager_init", [True, False])
    def test_close_pg(self, eager_init: bool):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)

        # Destroy pg and validate pg is no longer valid
        dist.destroy_process_group()
        with self.assertRaises(ValueError):
            dist.all_reduce(t)

    @requires_nccl()
    @skip_if_rocm_multiprocess
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_restart_pg(self):
        # Note: restart test passes steadily only for blocking mode for now.
        # TODO: expand this test to non-blocking mode
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        # initialize pg for the first time
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        t0 = torch.rand(10, 10, device=device)
        # First allreduce to lazy initialize default pg
        dist.all_reduce(t0)
        torch.cuda.synchronize()
        # Destroy pg
        dist.destroy_process_group()

        # we need a new Store for the new PG, achieving it by adding prefix
        new_store = c10d.PrefixStore("2nd", store)

        # re-initialize pg
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=new_store,
        )
        t1 = torch.rand(5, 5, device=device)
        dist.all_reduce(t1)
        torch.cuda.synchronize()
        dist.destroy_process_group()
        # validate default pg is no longer valid
        with self.assertRaises(ValueError):
            dist.all_reduce(t1)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_cuda_event_cache_mthd_race(self):
        # This unit test is to test the case when the collective is launched in
        # a side thread and the thread dies before the cache has been fully recycled.
        # More details can be found in this issue: https://github.com/pytorch/pytorch/issues/143470.

        # initiate collectives here
        def init_collective_task(t):
            dist.all_reduce(t)
            dist.all_reduce(t)
            dist.all_reduce(t)

        os.environ["TORCH_NCCL_CUDA_EVENT_CACHE"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)
        dist.all_reduce(t)
        dist.all_reduce(t)
        side_thread = threading.Thread(target=init_collective_task, args=(t,))
        side_thread.start()
        side_thread.join()
        torch.cuda.synchronize()

        # reset ENV
        os.environ["TORCH_NCCL_CUDA_EVENT_CACHE"] = "0"

    CUDA_12_AND_ABOVE = torch.cuda.is_available() and (
        torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 12
    )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        # skip for cu126 as well due to https://github.com/pytorch/pytorch/issues/153479
        not (TEST_MULTIGPU and CUDA_12_AND_ABOVE),
        "NCCL test requires 2+ GPUs and Device side assert could cause unexpected errors in lower versions of CUDA",
    )
    @parametrize(
        "type",
        [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    )
    @skip_if_rocm_multiprocess
    def test_nan_assert(self, type):
        # Expecting a device-side error when NaN is detected
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        backend = pg._get_backend(torch.device("cuda"))

        device = self.rank_to_GPU[self.rank][0]
        # Cover different buffer sizes
        if type == torch.float64:
            size = (1024,)  # 1K elements
        elif type == torch.float32:
            size = (1024, 1024)  # 1M elements
        elif type == torch.float16:
            size = (1024, 1024, 1024)  # 1G elements
        else:
            size = (1,)  # 1 element

        # Note: currently we cannot fill values into a FP8 tensor, thus we
        # create the NaN tensor in float32 type and cast it to FP8
        if type == torch.float8_e4m3fn or type == torch.float8_e5m2:
            init_type = torch.float32
        else:
            init_type = type

        nan_tensor = torch.zeros(*size, dtype=init_type, device=device)
        # randomly pick an nan element
        index = tuple([random.randrange(size[i]) for i in range(len(size))])
        nan_tensor[index] = float("nan")
        if init_type != type:
            # Now cast to the targeted dtype
            nan_tensor = nan_tensor.to(type)

        output = torch.empty(self.world_size, *size, dtype=type, device=device)

        # confirm enable/disable flag works
        backend._set_enable_nan_check(False)
        # Note: using all-gather here bc some NCCL/SM version does not support
        # FP8 reduction
        # temporarily skip due to https://github.com/pytorch/pytorch/issues/153479
        # pg._allgather_base(output, nan_tensor)

        backend._set_enable_nan_check(True)
        try:
            pg._allgather_base(output, nan_tensor)
        except Exception:
            sys.exit(signal.SIGABRT)

        dist.destroy_process_group()

        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nan_rank_filter(self):
        # Putting NaN at recv buffer, program should not fail as NaN checker
        # should not check on receive buffer
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank:d}")
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        t = torch.ones(3, 4, dtype=torch.bfloat16, device=device)
        if self.rank != 0:
            # Putting NaN at recv buffer
            t[1, 1] = float("nan")
        # Against broadcast
        c10d.broadcast(t, 0)
        # Against P2P
        if self.rank == 0:
            c10d.send(t, 1)
        elif self.rank == 1:
            c10d.recv(t, 0)
        c10d.destroy_process_group()
        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nan_check(self):
        # Not expecting an error, NaN check should not make legit code fail
        device = torch.device(f"cuda:{self.rank:d}")
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        x = torch.ones((10,), device=device) * self.rank
        t = torch.ones(3, 4, device=device)
        c10d.broadcast(x, src=0)
        c10d.all_reduce(t)
        c10d.barrier()
        c10d.destroy_process_group()
        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    def _helper_test_extra_cuda_context_by_nvml(self):
        """
        A helper for `test_extra_cuda_context`, if pynvml is available.
        pynvml provides python bindings for NVIDIA NVML functionalities.
        Here we are interested in: nvmlDeviceGetComputeRunningProcesses
        """
        import pynvml

        pynvml.nvmlInit()

        device = torch.device(f"cuda:{self.rank:d}")
        x = torch.empty((1,), device=device)
        work = c10d.all_reduce(x, async_op=True)

        # Wait for non-0 ranks to garbage collect Work -- this is the latest
        # point where extra CUDA context can be created
        if self.rank == 0:
            time.sleep(5)
        del work
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        nprocs = len(processes)

        # A barrier for non-0 ranks
        c10d.all_reduce(x)
        torch.cuda.synchronize(device)
        c10d.destroy_process_group()
        self.assertLessEqual(
            nprocs,
            1,
            f"Found {nprocs} processes creating contexts on {device}, expecting 1 at most",
        )

    def _helper_test_extra_cuda_context_by_memory(self):
        """
        A helper for `test_extra_cuda_context`, if pynvml is NOT available.
        If extra context is created, it would manifest into device 0's memory usage.
        """
        device = torch.device(f"cuda:{self.rank:d}")
        x = torch.empty((1,), device=device)
        # Rank 0 takes a snapshot before collective -- this snapshot should have
        # included rank 0's own context.
        if self.rank == 0:
            free, total = torch.cuda.mem_get_info(device)
            used_before = float(total - free)

        work = c10d.all_reduce(x, async_op=True)

        # Wait for non-0 ranks to garbage collect Work -- this is the latest
        # point where extra CUDA context can be created
        if self.rank == 0:
            time.sleep(5)
            free, total = torch.cuda.mem_get_info(device)
            used_after = float(total - free)
        del work

        # A barrier for non-0 ranks
        c10d.all_reduce(x)
        torch.cuda.synchronize(device)
        c10d.destroy_process_group()
        if self.rank == 0:
            # If non-0 rank creates a context on device 0, this assert would
            # fail because one context takes about 1 GB -- much more than the
            # tensor size created in this test.
            self.assertTrue(
                # Bump the heuristic from 1.5 to 1.7 due to
                # https://github.com/pytorch/pytorch/issues/153122
                used_after < used_before * 1.7,
                f"{device} used {used_after} bytes after collective, "
                f"70% more than the status before ({used_before} bytes). "
                f"Extra CUDA context may have been created.",
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_extra_cuda_context(self):
        # Check if non-0 ranks would create extra CUDA context on device 0
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank:d}")
        c10d.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )
        try:
            self._helper_test_extra_cuda_context_by_nvml()
        except ModuleNotFoundError:
            self._helper_test_extra_cuda_context_by_memory()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_extra_cuda_context_sync_ops(self):
        # Loop a bunch of sync ops and see if any of them creates extra context.
        # Requires nvml to check number of processes resident on a device.
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:
            self.skipTest("pynvml not available")

        # Check if non-0 ranks would create extra CUDA context on device 0
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank:d}")
        c10d.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )

        x = torch.empty((1,), device=device)
        y = torch.empty((self.world_size,), device=device)

        c10d.all_reduce(x)
        c10d.reduce(x, dst=0)
        c10d.broadcast(x, src=0)
        c10d.all_gather_into_tensor(y, x)
        c10d.reduce_scatter_tensor(x, y)
        c10d.barrier()

        # Wait a bit for remote processes to touch my device
        if self.rank == 0:
            time.sleep(5)

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        nprocs = len(processes)

        # Don't exit till rank 0 is done with the nvml detection
        c10d.barrier()
        c10d.destroy_process_group()
        self.assertLessEqual(
            nprocs,
            1,
            f"Found {nprocs} processes creating contexts on {device}, expecting 1 at most",
        )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_destruct_before_terminate_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)
        # force destruction before terminating comms, destructor would terminate comms
        del pg

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_abort_in_destroy_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)

        # Destroy pg and validate pg is NOT in working condition since
        # we have shutdown comms
        dist.destroy_process_group()
        with self.assertRaises(dist.DistBackendError):
            pg.allreduce([t])

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_abort_in_destroy_multi_pgs(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        t1 = torch.rand(10, 10, device=device)
        t2 = torch.rand(10, 10, device=device)
        new_pg1.allreduce(t1).wait()
        new_pg2.allreduce(t2).wait()
        backend = pg._get_backend(torch.device(device))
        # default PG's backend should have a split count of 0 because
        # it's not eager initialized
        self.assertEqual(backend.comm_split_count(), 0)
        # shutdown all NCCL PGs in one shot
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_abort_in_destroy_mixed_empty_pgs(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        # PG1 is an PG without comms initialized, since we don't call collective on it
        new_pg1 = c10d.new_group([0, 1])  # noqa: F841
        new_pg2 = c10d.new_group([0, 1])
        t2 = torch.rand(10, 10, device=device)

        new_pg2.allreduce(t2).wait()
        backend = pg._get_backend(torch.device(device))
        # default PG's backend should have a split count of 0
        self.assertEqual(backend.comm_split_count(), 0)
        # shutdown all NCCL PGs in one shot
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_file_store_check(self):
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
        # FileStore check() would be executed
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "0"

        # self.file_name is created using "delete=False"
        # e.g., self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )
        pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(pg.rank(), self.rank)
        self.assertEqual(pg.size(), self.world_size)
        # give enough time for check() to be executed multiple times
        time.sleep(2)
        dist.destroy_process_group()

    def _check_nccl_timeout(self, expected_timeout):
        pg = dist.distributed_c10d._get_default_group()
        options = pg._get_backend(torch.device(f"cuda:{self.rank}")).options
        self.assertEqual(options._timeout, expected_timeout)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def test_init_process_group_nccl_timeout(self):
        # nccl is handled 'specially' inside init_process_group and its options class is different from the options
        # used by the other PG's.  There are specific edge cases for nccl that need to be tested.

        store = c10d.FileStore(self.file_name, self.world_size)
        base_opts = dict(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )

        # test the default value coming from the `init_process_group` kwarg default
        dist.init_process_group(**base_opts)
        self._check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()

        # test that `kwarg` timeout takes effect
        new_timeout = timedelta(seconds=123)
        dist.init_process_group(**base_opts, timeout=new_timeout)
        self._check_nccl_timeout(new_timeout)
        dist.destroy_process_group()

        # test that timeout value provided via `pg_options` kwarg is ignored and issues warning,
        # 'timeout' kwarg (or its kwdefault) taking precedence
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        with warnings.catch_warnings(record=True):
            dist.init_process_group(**base_opts, pg_options=opts)
            # TODO(whc) i verified that we are indeed emitting this warning, and i can't figure out why i can't catch it.
            # self.assertEqual(len(w), 1)
            # self.assertTrue("pg_options._timeout was specified" in str(w[-1].message))
        self._check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()

        # test that timeout value provided via `pg_options` kwarg is ignored and issues warning,
        # 'timeout' kwarg taking precedence
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        dist.init_process_group(
            **base_opts, pg_options=opts, timeout=timedelta(seconds=1240)
        )
        self._check_nccl_timeout(timedelta(seconds=1240))
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("backend", [None, "nccl"])
    def test_set_nccl_pg_timeout(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = dict(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=123),
        )
        dist.init_process_group(**opts)
        pg = dist.distributed_c10d._get_default_group()
        pg.allreduce(torch.rand(10).cuda(self.rank))
        self._check_nccl_timeout(timedelta(seconds=123))
        pg._get_backend(torch.device(f"cuda:{self.rank}"))._set_default_timeout(
            timedelta(seconds=23)
        )
        self._check_nccl_timeout(timedelta(seconds=23))
        pg.allreduce(torch.rand(10).cuda(self.rank))
        c10d.distributed_c10d._set_pg_timeout(timedelta(seconds=252), pg)
        self._check_nccl_timeout(timedelta(seconds=252))

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("backend", [None, "nccl"])
    def test_extend_nccl_pg_timeout(self, backend):
        torch.cuda.set_device(self.rank)
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = dict(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=123),
        )
        dist.init_process_group(**opts)
        pg = dist.distributed_c10d._get_default_group()
        bankend = pg._get_backend(torch.device(f"cuda:{self.rank}"))
        w = pg.allreduce(torch.rand(10).cuda(self.rank))
        self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=123)))
        w.wait()
        bankend._set_default_timeout(timedelta(seconds=3))
        if self.rank == 0:
            # Ideally we want to sleep for a very long time, but this is not
            # feasible in unit test. So this is only a very tiny case.
            time.sleep(5)
            pg.allreduce(torch.rand(10).cuda(self.rank))
            time.sleep(5)
            pg.allreduce(torch.rand(5).cuda(self.rank))
            w = pg.allreduce(torch.rand(10).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=3)))
            w.wait()
        else:
            dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
                timedelta(seconds=10)
            )
            w1 = pg.allreduce(torch.rand(10).cuda(self.rank))
            w2 = pg.allreduce(torch.rand(5).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w1, timedelta(seconds=13)))
            self.assertTrue(bankend._verify_work_timeout(w2, timedelta(seconds=13)))
            w1.wait()
            dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
                timedelta(seconds=5)
            )
            # Since we are not block wait so use a sync here to leave enough time
            # for watchdog to reset first timeout extension.
            torch.cuda.synchronize(torch.device(f"cuda:{self.rank}"))
            w = pg.allreduce(torch.rand(10).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=8)))
            w.wait()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("eager_init", [True, False])
    def test_new_group(self, eager_init: bool):
        # Test the optimization of new groups that contain all world
        # ranks use the "transparent" `ncclCommSplit` optimization.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )
        ng = c10d.new_group()
        tensor = torch.tensor([self.rank], device=device)
        dist.broadcast(tensor, 0)
        dist.broadcast(tensor, 0, group=ng)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.nccl.version()[-1] == "x", "NCCL test not for NCCLX"
    )
    def test_comm_split_subgroup(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).cuda(device)
        original_tensor = tensor.clone()
        ng = c10d.new_group([0])

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        if self.rank == 0:
            dist.broadcast(tensor, 0, group=ng)

        # no additional comm split happens after a collective.
        self.assertEqual(backend.comm_split_count(), 1)
        self.assertEqual(tensor, original_tensor)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_comm_eager_init_subgroup(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        # default PG comm is not initialized yet
        pg = self._create_process_group_nccl(store, self.opts())
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend._is_initialized(), False)
        # create a subgroup eagerly
        new_group = c10d.new_group([0, 1], device_id=device)
        tensor = torch.full((1,), self.rank).cuda(device)
        dist.broadcast(tensor, 0, group=new_group)
        # the default group should stay lazy
        self.assertEqual(backend._is_initialized(), False)
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_comm_split_group(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).cuda(device)
        # Create subgroup between ranks 0, 1
        subg_ranks = [0, 1]
        ng1 = c10d.split_group(pg, [subg_ranks])
        backend1 = ng1._get_backend(torch.device(device))

        # check basic options are the same between parent and child
        self.assertEqual(backend.options._timeout, backend1.options._timeout)
        self.assertEqual(
            backend.options.is_high_priority_stream,
            backend1.options.is_high_priority_stream,
        )
        self.assertEqual(ng1.group_desc, "default_pg:split:0")

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        # dist.get_process_group_ranks returns the global ranks in the subgroup.
        self.assertEqual(
            dist.get_process_group_ranks(ng1),
            subg_ranks if self.rank in subg_ranks else [],
        )

        # is part of ng1; otherwise, -1
        if dist.get_rank(ng1) >= 0:
            dist.broadcast(tensor, dist.get_global_rank(ng1, 0), group=ng1)
            self.assertEqual(tensor, torch.full((1,), 0))

        ng2 = c10d.split_group(pg, [subg_ranks])
        self.assertEqual(ng2.group_desc, "default_pg:split:1")
        self.assertEqual(backend.comm_split_count(), 2)

        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_comm_split_group_mixed_backend(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        # pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        # create nccl processgroup with opts
        c10d.init_process_group(
            "cpu:gloo,cuda:nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=self.opts(),
            device_id=device,
        )
        pg = c10d.distributed_c10d._get_default_group()
        backend = pg._get_backend(torch.device(device))

        cuda_tensor = torch.full((1,), self.rank).cuda(device)
        cpu_tensor = torch.full((1,), self.rank)
        # Create subgroup between ranks 0, 1
        subg_ranks = [0, 1]
        ng1 = c10d.split_group(pg, [subg_ranks])
        backend1 = ng1._get_backend(torch.device(device))

        # check basic options are the same between parent and child
        self.assertEqual(backend.options._timeout, backend1.options._timeout)
        self.assertEqual(
            backend.options.is_high_priority_stream,
            backend1.options.is_high_priority_stream,
        )
        self.assertEqual(ng1.group_desc, "default_pg:split:0")

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        # dist.get_process_group_ranks returns the global ranks in the subgroup.
        self.assertEqual(
            dist.get_process_group_ranks(ng1),
            subg_ranks if self.rank in subg_ranks else [],
        )

        # is part of ng1; otherwise, -1
        if dist.get_rank(ng1) >= 0:
            dist.broadcast(cuda_tensor, dist.get_global_rank(ng1, 0), group=ng1)
            self.assertEqual(cuda_tensor, torch.full((1,), 0))
            dist.broadcast(cpu_tensor, dist.get_global_rank(ng1, 0), group=ng1)
            self.assertEqual(cpu_tensor, torch.full((1,), 0))

        ng2 = c10d.split_group(pg, [subg_ranks])
        self.assertEqual(ng2.group_desc, "default_pg:split:1")
        self.assertEqual(backend.comm_split_count(), 2)

        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_init(self):
        # Test creating a pg using nonblocking mode but not eagerly
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = self.rank_to_GPU[self.rank][0]
        pg = self._create_process_group_nccl(store, self.opts())
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 0)
        reduce_tensor = torch.rand(10, 10, device=device)
        # Run an allreduce, which should trigger a comm init for pg
        pg.allreduce(reduce_tensor).wait()
        new_pg = c10d.new_group()
        # even after pg's collective call, new pg's comm is not initialized until its own collectcive calls
        self.assertEqual(backend.comm_split_count(), 0)
        broadcast_tensor = torch.tensor([self.rank]).cuda(device)
        new_pg.broadcast(broadcast_tensor, 0).wait()
        self.assertEqual(backend.comm_split_count(), 0)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_with_eager_init(self):
        # Test creating a pg eagerly with nonblocking mode when
        # we've passed a specific device_id to init_process_group.
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        # bound device to trigger eager init mode
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 0)
        reduce_tensor = torch.rand(10, 10, device=device)
        # Run an allreduce, comm should have already started initilizaing,
        # but allreduce is issued to CUDA STREAM only after the initialization is a success
        pg.allreduce(reduce_tensor).wait()
        new_pg = c10d.new_group()
        # new pg's comm is initialized eagerly
        self.assertEqual(backend.comm_split_count(), 1)
        broadcast_tensor = torch.tensor([self.rank]).cuda(device)
        new_pg.broadcast(broadcast_tensor, 0).wait()
        self.assertEqual(backend.comm_split_count(), 1)
        dist.destroy_process_group()

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_p2p(self):
        # Test creating a pg using nonblocking mode but not eagerly
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = self.rank_to_GPU[self.rank][0]
        self._create_process_group_nccl(store, self.opts())
        # Generate the same tensor
        send_tensor = torch.ones(10, 10, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)
        dist.destroy_process_group()

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("eager_init", [True, False])
    def test_subgroup_p2p(self, eager_init: bool):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        c10d
```



## High-Level Overview


This Python file contains 32 class(es) and 320 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RendezvousEnvTest`, `Env`, `TimeoutTest`, `ProcessGroupNCCLNoGPUTest`, `ProcessGroupNCCLInitTest`, `ProcessGroupNCCLGroupTest`, `DistributedDataParallelTest`, `ForwardReturnValueModule`, `FindUnusedParametersModule`, `MultipleOutputModule`, `NoGradModule`, `TestModel`, `AcceptsParam`, `WorkHookTest`, `NcclErrorHandlingTest`, `NcclUserBufferRegistrationTest`, `CommTest`, `SetDeviceMethod`, `NcclProcessGroupWithDispatchedCollectivesTests`, `LargeCommTest`

**Functions defined**: `_ts`, `configure`, `log_test_info`, `log_test_success`, `log_test_validation`, `log_test_warning`, `log_test_error`, `test_common_errors`, `__init__`, `__enter__`, `__exit__`, `without`, `withouts`, `test_default_store_timeout_nccl`, `setUp`, `tearDown`, `test_init_no_gpus`, `setUp`, `tearDown`, `world_size`

**Key imports**: copy, json, logging, os, pickle, random, re, signal, sys, tempfile


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `json`
- `logging`
- `os`
- `pickle`
- `random`
- `re`
- `signal`
- `sys`
- `tempfile`
- `threading`
- `time`
- `warnings`
- `contextlib`: contextmanager
- `datetime`: datetime, timedelta
- `enum`: auto, Enum
- `itertools`: chain, product
- `unittest`: mock, SkipTest
- `torch`
- `torch.distributed as c10d`
- `torch.distributed._functional_collectives as _functional_collectives`
- `torch.distributed.distributed_c10d`: SHRINK_ABORT as NCCL_SHRINK_ABORT
- `test_c10d_common`
- `torch.distributed as dist`
- `torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default`
- `torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD`
- `torch.nn.functional as F`
- `torch.testing._internal.common_utils as common`
- `torch._C._distributed_c10d`: ErrorType, OpType, WorkResult


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/test_c10d_nccl.py
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

- **File Documentation**: `test_c10d_nccl.py_docs.md`
- **Keyword Index**: `test_c10d_nccl.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
