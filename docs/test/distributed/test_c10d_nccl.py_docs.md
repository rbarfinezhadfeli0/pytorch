# Documentation: test_c10d_nccl.py

## File Metadata
- **Path**: `test/distributed/test_c10d_nccl.py`
- **Size**: 255033 bytes
- **Lines**: 6366
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )
        send_tensor = torch.ones(10, 10, device=device)
        group = dist.new_group()
        if self.rank == 0:
            dist.send(send_tensor, 1, group=group)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0, group=group)
            self.assertEqual(send_tensor, recv_tensor)
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_get_uid(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        from torch.distributed.distributed_c10d import _get_process_group_uid

        self.assertEqual(_get_process_group_uid(pg), 0)
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(_get_process_group_uid(pg_2), 1)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_set_process_group_desc(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg_default = self._create_process_group_nccl(
            store, self.opts(), device_id=device
        )
        self.assertEqual(pg_default.group_desc, "default_pg")
        pg_1 = c10d.new_group([0, 1], group_desc="test_purpose")
        self.assertEqual(pg_1.group_desc, "test_purpose")
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(pg_2.group_desc, "undefined")

    @requires_nccl_shrink()
    @requires_world_size(2)
    def test_shrink_group_basic(self):
        """Test basic shrink_group functionality."""
        self._perform_shrink_test([1], "Basic shrink test")

    @requires_nccl_shrink()
    @requires_world_size(2)
    def test_shrink_group_validation(self):
        """Test input validation in shrink_group."""
        device, pg = self._setup_shrink_test("validation")

        def _test_invalid_input(ranks, description, expected_exception):
            """Helper to test invalid inputs."""
            try:
                c10d.shrink_group(ranks)
                self.fail(f"Expected {expected_exception.__name__} for {description}")
            except expected_exception:
                log_test_validation(self.rank, f"✓ {description}")
            except Exception:
                if expected_exception is Exception:  # Accept any exception
                    log_test_validation(self.rank, f"✓ {description}")
                else:
                    raise

        # Test cases
        _test_invalid_input([], "Empty exclusion list", ValueError)
        if self.world_size > 1:
            _test_invalid_input([0, 0, 1], "Duplicate ranks", Exception)
        _test_invalid_input([self.world_size + 1], "Out of bounds rank", Exception)

        log_test_success(self.rank, "All validation tests passed")
        dist.destroy_process_group()

    @requires_nccl_shrink()
    @requires_world_size(2)
    def test_shrink_group_backend_properties(self):
        """Test that backend properties are preserved after shrinking."""

        test_name = "Backend Properties Test"
        ranks_to_exclude = [0]

        # Reuse _setup_shrink_test for complete setup (device, environment, and process group)
        device, pg = self._setup_shrink_test("backend_properties")

        # Follow _perform_shrink_test pattern from here
        log_test_info(self.rank, f"{test_name} (world_size={self.world_size})")

        is_excluded = self.rank in ranks_to_exclude
        log_test_info(
            self.rank,
            f"Excluding ranks: {ranks_to_exclude}, am_excluded: {is_excluded}",
        )

        # Store original backend property values (not references) before shrinking
        original_timeout = None
        original_high_priority = None
        if not is_excluded:
            original_backend = pg._get_backend(device)
            original_timeout = original_backend.options._timeout
            original_high_priority = original_backend.options.is_high_priority_stream
            log_test_info(
                self.rank,
                f"Storing original backend properties: timeout={original_timeout}, high_priority={original_high_priority}",
            )

        if is_excluded:
            log_test_info(
                self.rank,
                f"Excluded rank {self.rank} - setup complete, skipping shrink operation",
            )
            dist.destroy_process_group()  # hang without it
            return

        # Only non-excluded ranks proceed with shrink (same as _perform_shrink_test)
        log_test_info(self.rank, "Non-excluded rank calling shrink_group")
        shrunk_pg = c10d.shrink_group(ranks_to_exclude)

        # Reuse _validate_shrunk_group helper (same as _perform_shrink_test)
        expected_size = self.world_size - len(ranks_to_exclude)
        _ = self._validate_shrunk_group(shrunk_pg, expected_size, test_name)

        # Add custom backend properties validation
        new_backend = shrunk_pg._get_backend(device)
        log_test_info(self.rank, "Validating backend properties are preserved")

        new_timeout = new_backend.options._timeout
        new_high_priority = new_backend.options.is_high_priority_stream

        log_test_info(
            self.rank,
            f"Timeout comparison - original: {original_timeout}, new: {new_timeout}",
        )
        self.assertEqual(
            original_timeout, new_timeout, f"{test_name}: timeout not preserved"
        )

        log_test_info(
            self.rank,
            f"High priority stream comparison - original: {original_high_priority}, new: {new_high_priority}",
        )
        self.assertEqual(
            original_high_priority,
            new_high_priority,
            f"{test_name}: high_priority_stream not preserved",
        )

        log_test_validation(
            self.rank, f"{test_name}: Backend properties preserved successfully"
        )
        log_test_success(
            self.rank, f"{test_name} successful (shrink + backend validation)"
        )

        # Cleanup (same as _perform_shrink_test)
        dist.destroy_process_group()

    @requires_nccl_shrink()
    @requires_world_size(2)
    def test_shrink_group_multiple_comms(self):
        """Test shrink_group with multiple communicators and subgroup invalidation."""

        device, pg = self._setup_shrink_test("multiple_comms")

        # Create subgroup [0, 1] and test shrinking it
        subgroup = c10d.new_group([0, 1])
        if self.rank <= 1:
            # Shrink subgroup: exclude rank 1
            if self.rank == 0:  # Only rank 0 remains
                shrunk_subgroup = c10d.shrink_group([1], group=subgroup)
                self.assertEqual(shrunk_subgroup.size(), 1)
                # Test communication on shrunk subgroup
                tensor = torch.full((1,), self.rank).cuda(device)
                c10d.all_reduce(tensor, group=shrunk_subgroup)
                self.assertEqual(tensor.item(), 0)  # Only rank 0
                log_test_success(self.rank, "Subgroup shrinking successful")

        dist.barrier()  # Sync before default group test

        # Shrink default group: exclude last rank
        ranks_to_exclude = [self.world_size - 1]
        if self.rank not in ranks_to_exclude:
            shrunk_default = c10d.shrink_group(ranks_to_exclude)
            expected_size = self.world_size - 1
            self.assertEqual(shrunk_default.size(), expected_size)

            # Test collective on shrunk default group
            tensor = torch.full((1,), self.rank).cuda(device)
            c10d.all_reduce(tensor, group=shrunk_default)
            expected_sum = sum(
                range(self.world_size - 1)
            )  # 0 + 1 + ... + (world_size-2)
            self.assertEqual(tensor.item(), expected_sum)
            log_test_success(self.rank, "Default group shrinking successful")

            # Note: After shrinking default group, the old subgroup is invalid
            # due to global rank reassignment

        dist.destroy_process_group()

    def _test_shrink_group_with_flag(self, shrink_flag, flag_name, rank_to_exclude):
        """Helper method to test shrink_group with a specific flag."""
        if self.world_size < 2:
            log_test_info(self.rank, f"Skipping (needs ≥2 GPUs, got {self.world_size})")
            return
        ranks_to_exclude = [rank_to_exclude]
        log_test_info(self.rank, f"Using {flag_name} flag (value: {shrink_flag})")
        if flag_name == "NCCL_SHRINK_ABORT":
            log_test_info(
                self.rank,
                "ABORT flag will terminate ongoing operations before shrinking",
            )

        self._perform_shrink_test(
            ranks_to_exclude, f"{flag_name} flag test", shrink_flags=shrink_flag
        )

    @requires_nccl_shrink()
    @requires_world_size(2)
    def test_shrink_group_flags(self):
        """Test shrink_group with different shrink flags."""
        # Test ABORT flags
        log_test_info(self.rank, "Testing NCCL_SHRINK_ABORT flag")
        self._test_shrink_group_with_flag(NCCL_SHRINK_ABORT, "NCCL_SHRINK_ABORT", 1)

    @requires_nccl_shrink()
    @requires_world_size(2)
    def test_shrink_group_nccl_config(self):
        """Verify that passing NCCL config via pg_options influences the shrunk group's backend options."""
        device, pg = self._setup_shrink_test("config")
        if self.rank == self.world_size - 1:
            # excluded rank should not call shrink_group
            dist.destroy_process_group()
            return

        # Prepare pg_options with NCCL config overrides
        # Capture parent's current backend options to ensure we can prove override vs inherit
        parent_backend = pg._get_backend(torch.device("cuda"))
        parent_hp = parent_backend.options.is_high_priority_stream
        parent_blocking = parent_backend.options.config.blocking

        # Choose overrides that differ from the parent (flip where possible)
        override_hp = not parent_hp
        if parent_blocking in (0, 1):
            override_blocking = 1 - parent_blocking
        else:
            # If undefined or unexpected, set to 1 which is a concrete value
            override_blocking = 1

        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = override_hp
        opts.config.blocking = override_blocking

        shrunk_pg = c10d.shrink_group([self.world_size - 1], pg_options=opts)

        # Validate backend options propagated
        backend = shrunk_pg._get_backend(torch.device("cuda"))
        # is_high_priority_stream should exactly match our override and differ from parent
        self.assertEqual(backend.options.is_high_priority_stream, override_hp)
        self.assertNotEqual(backend.options.is_high_priority_stream, parent_hp)
        # config is a struct; check representative field and difference from parent when meaningful
        self.assertEqual(backend.options.config.blocking, override_blocking)
        if parent_blocking in (0, 1):
            self.assertNotEqual(backend.options.config.blocking, parent_blocking)

        dist.destroy_process_group()

    @requires_nccl_shrink()
    @requires_world_size(2)
    def test_shrink_group_performance(self):
        """Test shrink_group performance and regression detection."""
        import time

        ranks_to_exclude = self._get_default_ranks_to_exclude()
        is_excluded = self.rank in ranks_to_exclude

        if not ranks_to_exclude:
            log_test_info(self.rank, "Skipping performance test (world_size=1)")
            return

        log_test_info(self.rank, f"Performance test with {self.world_size} processes")
        device, pg = self._setup_shrink_test("performance")

        if not is_excluded:
            log_test_info(self.rank, "Measuring shrink_group performance")
            start_time = time.time()
            shrunk_pg = c10d.shrink_group(ranks_to_exclude)
            end_time = time.time()

            elapsed_time = end_time - start_time
            log_test_info(self.rank, f"shrink_group: {elapsed_time:.3f}s")

            # Regression check: should complete within reasonable time
            self.assertLess(
                elapsed_time,
                30.0,
                f"shrink_group took {elapsed_time:.3f}s, possible regression",
            )

            # Test collective performance
            expected_size = self.world_size - len(ranks_to_exclude)
            self._validate_shrunk_group(shrunk_pg, expected_size, "performance")

            collective_start = time.time()
            _ = self._test_collective_on_shrunk_group(
                shrunk_pg, device, ranks_to_exclude, "performance"
            )
            collective_time = time.time() - collective_start

            log_test_info(self.rank, f"all_reduce: {collective_time:.3f}s")
            log_test_success(self.rank, "Performance test passed")
        else:
            log_test_info(self.rank, "Excluded rank - waiting")

        dist.destroy_process_group()

    @requires_nccl_shrink()
    @requires_world_size(4)
    def test_shrink_group_multiple_exclusions(self):
        """Test shrink_group with multiple ranks excluded at once."""
        # Scale exclusions with world size
        ranks_to_exclude = list(range(2, self.world_size, 2))  # Every other rank from 2

        self._perform_shrink_test(ranks_to_exclude, "Multiple exclusions test")

    @requires_nccl_shrink()
    @requires_world_size(3)
    def test_shrink_group_multiple_iterations(self):
        """Test multiple shrink operations in sequence."""
        log_test_info(
            self.rank,
            f"Starting test_shrink_group_multiple_iterations with world_size={self.world_size}",
        )

        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        _ = self._create_process_group_nccl(store, self.opts(), device_id=device)

        # Track current effective world size throughout shrinking operations
        current_world_size = self.world_size
        log_test_info(self.rank, f"Initial world_size: {current_world_size}")

        # First shrinking: exclude the last rank(s)
        first_exclusion = [self.world_size - 1]
        if self.world_size >= 6:
            first_exclusion.append(
                self.world_size - 2
            )  # Exclude last two ranks for larger sizes

        log_test_info(self.rank, f"First shrinking: excluding ranks {first_exclusion}")

        if self.rank not in first_exclusion:
            # Only non-excluded ranks should call shrink_group
            first_pg = c10d.shrink_group(first_exclusion)
            self.assertIsNotNone(first_pg)
            # IMPORTANT: Update world size after first shrinking
            current_world_size = first_pg.size()
            expected_first_size = self.world_size - len(first_exclusion)
            log_test_info(
                self.rank,
                f"After first shrinking: world_size {self.world_size} -> {current_world_size}",
            )
            self.assertEqual(first_pg.size(), expected_first_size)

            # Second shrinking: exclude another rank from the remaining group
            # Choose a rank that's in the middle range
            if current_world_size >= 3:
                second_exclusion = [
                    current_world_size - 1
                ]  # Exclude the new "last" rank
                log_test_info(
                    self.rank,
                    f"Second shrinking from group of size {current_world_size}: excluding ranks {second_exclusion}",
                )

                if self.rank not in second_exclusion:
                    # Only non-excluded ranks should call shrink_group for second iteration
                    second_pg = c10d.shrink_group(second_exclusion, group=first_pg)
                    self.assertIsNotNone(second_pg)
                    # IMPORTANT: Update world size after second shrinking
                    final_world_size = second_pg.size()
                    expected_final_size = current_world_size - len(second_exclusion)
                    log_test_info(
                        self.rank,
                        f"After second shrinking: world_size {current_world_size} -> {final_world_size}",
                    )
                    self.assertEqual(second_pg.size(), expected_final_size)

                    # Test collective on final group
                    tensor = torch.full((1,), self.rank).cuda(device)
                    log_test_info(
                        self.rank,
                        f"Performing all_reduce on final group (size {final_world_size}) with tensor: {tensor.item()}",
                    )
                    c10d.all_reduce(tensor, group=second_pg)
                    log_test_info(
                        self.rank,
                        f"Final all_reduce completed, result: {tensor.item()}",
                    )

                    # Calculate expected sum of remaining ranks
                    all_excluded = set(first_exclusion + second_exclusion)
                    remaining_ranks = [
                        r for r in range(self.world_size) if r not in all_excluded
                    ]
                    expected_sum = sum(remaining_ranks)
                    log_test_info(
                        self.rank,
                        f"Remaining ranks: {remaining_ranks}, expected sum: {expected_sum}, actual: {tensor.item()}",
                    )
                    self.assertEqual(tensor.item(), expected_sum)
                    log_test_info(self.rank, "Final verification passed")
                else:
                    log_test_info(
                        self.rank,
                        "This rank excluded in second shrinking, not calling shrink_group",
                    )
            else:
                log_test_info(
                    self.rank, "Skipping second shrinking (remaining group too small)"
                )
        else:
            log_test_info(
                self.rank,
                "This rank excluded in first shrinking, not calling shrink_group",
            )

        log_test_info(self.rank, "Destroying process group")
        dist.destroy_process_group()
        log_test_info(self.rank, "test_shrink_group_multiple_iterations completed")

    # Helper methods for optimized shrink group tests
    def _setup_shrink_test(self, test_suffix, world_size=None, warmup=True):
        """Common setup for shrink group tests."""
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        world_size = world_size or self.world_size
        store = c10d.FileStore(self.file_name + f"_{test_suffix}", world_size)
        device = torch.device(f"cuda:{self.rank}")
        c10d.init_process_group(
            "nccl",
            world_size=world_size,
            rank=self.rank,
            store=store,
            pg_options=self.opts(),
            device_id=device,
        )
        pg = c10d.distributed_c10d._get_default_group()

        if warmup:
            c10d.all_reduce(torch.ones(1).cuda(device), group=pg)

        return device, pg

    def _validate_shrunk_group(self, shrunk_pg, expected_size, test_name=""):
        """Validate properties of a shrunk process group."""
        self.assertIsNotNone(shrunk_pg, f"{test_name}: shrunk_pg should not be None")
        actual_size = shrunk_pg.size()
        self.assertEqual(
            actual_size, expected_size, f"{test_name}: group size mismatch"
        )

        new_rank = shrunk_pg.rank()
        self.assertTrue(
            0 <= new_rank < expected_size, f"{test_name}: invalid new rank {new_rank}"
        )

        log_test_info(
            self.rank,
            f"{test_name}: world_size {self.world_size} -> {actual_size}, rank {self.rank} -> {new_rank}",
        )
        return new_rank

    def _test_collective_on_shrunk_group(
        self, shrunk_pg, device, ranks_to_exclude, test_name=""
    ):
        """Test collective communication on shrunk group and verify correctness."""
        test_tensor = torch.full((1,), self.rank, device=device, dtype=torch.float32)
        c10d.all_reduce(test_tensor, group=shrunk_pg)

        result = test_tensor.item()
        expected_sum = sum(
            r for r in range(self.world_size) if r not in ranks_to_exclude
        )

        self.assertEqual(
            result, expected_sum, f"{test_name}: collective result mismatch"
        )
        log_test_info(
            self.rank, f"{test_name}: collective passed ({result} == {expected_sum})"
        )
        return result

    def _perform_shrink_test(
        self, ranks_to_exclude, test_name, shrink_flags=0, with_collective=True
    ):
        """Complete shrink test flow: setup, shrink, validate, test collective, cleanup.

        Consistent API: All ranks perform setup to initialize distributed environment.
        ONLY non-excluded ranks call shrink_group() for both default and non-default groups.
        Excluded ranks perform setup, then exit without calling shrink_group() or waiting.
        """
        log_test_info(self.rank, f"{test_name} (world_size={self.world_size})")

        is_excluded = self.rank in ranks_to_exclude
        log_test_info(
            self.rank,
            f"Excluding ranks: {ranks_to_exclude}, am_excluded: {is_excluded}",
        )

        # All ranks (including excluded ones) perform setup to initialize distributed environment
        device, pg = self._setup_shrink_test(test_name.lower().replace(" ", "_"))
        is_default_group = pg == c10d.distributed_c10d._get_default_group()

        if is_excluded:
            log_test_info(
                self.rank,
                f"Excluded rank {self.rank} - setup complete, skipping shrink operation",
            )
            if shrink_flags & NCCL_SHRINK_ABORT:
                log_test_info(self.rank, f"Using abort for excluded rank {self.rank}")
                pg._get_backend(torch.device(device)).abort()
                log_test_info(
                    self.rank, f"cleanup resources for excluded rank {self.rank}"
                )
                dist.destroy_process_group()
                log_test_info(self.rank, f"Excluded rank {self.rank} - exit")
            else:
                log_test_info(
                    self.rank, f"Using regular destroy for excluded rank {self.rank}"
                )
                dist.destroy_process_group()
            return None

        # Only non-excluded ranks proceed with shrink
        log_test_info(
            self.rank,
            f"Non-excluded rank calling shrink_group (default_group={is_default_group})",
        )
        shrunk_pg = c10d.shrink_group(ranks_to_exclude, shrink_flags=shrink_flags)
        log_test_info(
            self.rank,
            f"Non-excluded rank calling shrink_group (default_group={is_default_group}) done",
        )

        # Non-excluded ranks: validate and test the new group
        expected_size = self.world_size - len(ranks_to_exclude)
        _ = self._validate_shrunk_group(shrunk_pg, expected_size, test_name)

        if with_collective:
            _ = self._test_collective_on_shrunk_group(
                shrunk_pg, device, ranks_to_exclude, test_name
            )
            log_test_success(self.rank, f"{test_name} successful (shrink + collective)")
        else:
            log_test_success(self.rank, f"{test_name} successful (shrink only)")

        dist.destroy_process_group()
        return shrunk_pg

    def _get_default_ranks_to_exclude(self):
        """Get default ranks to exclude based on world size."""
        if self.world_size <= 1:
            return []
        return [self.world_size - 1]  # Exclude last rank by default

    @requires_nccl_shrink()
    @requires_world_size(3)
    def test_shrink_group_vs_abort_reinit_performance(self):
        """Compare performance of shrink_group vs traditional abort+reinit (simplified for reliability)."""
        log_test_info(self.rank, "=== TEST 1: abort+reinit ===")

        device, pg1 = self._setup_shrink_test("_perf_reinit")
        torch.cuda.synchronize(device)

        # Test 1: Traditional abort + reinit
        start_time = time.perf_counter()
        dist.destroy_process_group()

        device, new_pg = self._setup_shrink_test("perf_shrink_test1")
        reinit_time = time.perf_counter() - start_time

        # Test collective with original rank values for fair comparison (non-blocking mode)
        test_tensor = torch.full((1,), self.rank, device=device, dtype=torch.float32)
        work = c10d.all_reduce(test_tensor, group=new_pg, async_op=True)
        work.wait()

        torch.cuda.synchronize(device)

        # Verify correctness
        expected_sum = sum(r for r in range(self.world_size))
        self.assertEqual(test_tensor.item(), expected_sum, "Reinit collective failed")

        log_test_info(self.rank, f"abort+reinit: {reinit_time:.4f}s")
        dist.destroy_process_group(new_pg)

        # Test 2: shrink_group with NCCL_SHRINK_ABORT
        log_test_info(self.rank, "=== TEST 2: shrink_group ===")

        ranks_to_exclude = [self.world_size - 1]
        is_excluded = self.rank in ranks_to_exclude
        log_test_info(
            self.rank,
            f"Excluding ranks: {ranks_to_exclude}, am_excluded: {is_excluded}",
        )

        device, pg1 = self._setup_shrink_test("perf_shrink_test2")  # Unique suffix

        shrink_time = 0
        if not is_excluded:
            torch.cuda.synchronize(device)  # Ensure accurate timing
            start_time = time.perf_counter()
            shrunk_pg = c10d.shrink_group(
                ranks_to_exclude, shrink_flags=NCCL_SHRINK_ABORT
            )
            c10d.all_reduce(torch.ones(1).cuda(device), group=shrunk_pg)
            shrink_time = time.perf_counter() - start_time

            # Test collective communication on shrunk group (non-blocking mode)
            test_tensor = torch.full(
                (1,), self.rank, device=device, dtype=torch.float32
            )
            work = c10d.all_reduce(test_tensor, group=shrunk_pg, async_op=True)
            work.wait()

            # Verify correctness
            expected_sum = sum(
                r for r in range(self.world_size) if r not in ranks_to_exclude
            )
            self.assertEqual(
                test_tensor.item(),
                expected_sum,
                "shrink_test: collective result mismatch",
            )

            torch.cuda.synchronize(device)  # Ensure operations complete
            log_test_info(self.rank, f"shrink_group: {shrink_time:.4f}s")
            dist.destroy_process_group()
        else:
            log_test_info(self.rank, "Excluded from shrink test - exiting immediately")
            dist.destroy_process_group()
            return

        # Performance analysis (only for participating ranks)
        if shrink_time > 0 and reinit_time > 0:
            speedup = reinit_time / shrink_time
            time_saved = reinit_time - shrink_time

            log_test_info(self.rank, "=== PERFORMANCE RESULTS ===")
            log_test_info(self.rank, f"shrink_group:  {shrink_time:.4f}s")
            log_test_info(self.rank, f"abort+reinit:  {reinit_time:.4f}s")
            log_test_info(self.rank, f"time_saved:    {time_saved:+.4f}s")
            log_test_info(self.rank, f"speedup:       {speedup:.2f}x")

            if speedup > 1.1:
                log_test_success(self.rank, "shrink_group significantly faster")
            elif speedup > 0.9:
                log_test_info(self.rank, "≈ comparable performance")
            else:
                log_test_warning(self.rank, "abort+reinit faster")

        log_test_info(self.rank, "Performance test completed")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_deterministic_mode_no_break(self):
        torch.use_deterministic_algorithms(True)
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        self._create_process_group_nccl(store, self.opts(), device_id=device)
        tensor = torch.empty(10, 10, device=device)
        dist.all_reduce(tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_init_with_idx(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device_idx = self.rank
        dist.init_process_group(
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device_idx,
        )
        dist.all_reduce(torch.empty(1, device=torch.device("cuda", device_idx)))

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_block_current_stream(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)

        t = torch.rand(10, device=device)
        work = pg.allreduce(t)
        work.block_current_stream()

        torch.cuda.current_stream().synchronize()
        work.wait()
        torch.cuda.synchronize()


class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def _get_process_group(self):
        store = self._get_store()
        c10d.init_process_group(
            "nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    def _test_nccl_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        process_group = self._get_process_group()
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_propagate_error_reason(self):
        # Need to use TORCH_NCCL_BLOCKING_WAIT and not ASYNC_ERROR_HANDLING,
        # otherwise process will be taken down and we can't check for errors.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        # Need to disable TORCH_NCCL_DUMP_ON_TIMEOUT otherwise this test times out
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        # provide sufficient timeout to initialize NCCL comm.
        pg = c10d.ProcessGroupNCCL(
            store, self.rank, self.world_size, timeout=timedelta(seconds=15)
        )
        pg_gloo = c10d.ProcessGroupGloo(store, self.rank, self.world_size)
        pg.barrier().wait(timedelta(seconds=5))
        # Simulate stuckness in rank 0.
        if self.rank == 0:
            pg_gloo.barrier().wait()
        inp = torch.ones(1).cuda(self.rank)

        if self.rank != 0:
            # Time out due to rank 0 not calling into allreduce.
            with self.assertRaises(dist.DistBackendError):
                pg.allreduce([inp]).wait(timedelta(seconds=5))

            # Now when nonzero rank attempts to use communicator, original failure reason should be logged.
            try:
                pg.allreduce([torch.ones(2).cuda(self.rank)]).wait()
            except dist.DistBackendError as e:
                self.assertTrue("aborted" in str(e))
            else:
                self.fail("Expected error to be raised!")

            # Unblock rank 0
            pg_gloo.barrier().wait()

        # TODO: We can also test that if rank 0 attempts to use the communicator,
        # then we should error out with the info that it was aborted due to
        # timeout on another rank. Although this would only be the case after
        # the watchdog has run on the rank, and there is no reliable way
        # to confirm it has run.

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_multi_device_ids_not_allowed(self):
        int_devices = list(range(torch.cuda.device_count()))
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_device_ids_None(self):
        self._test_nccl_backend(None, None)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_empty_device_ids(self):
        # This tests the backward compatibility of accepting an empty list as `device_ids`,
        # although we no longer document this in favor of the default value of `None`,
        # which is consistent with multi-device modules and CPU modules.
        self._test_nccl_backend(None, [])

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_multi_device_module_device_ids_None(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_integer_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_torch_device_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_2gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(8)
    def test_nccl_backend_4gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_ddp_multi_device_module_config(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]

        self.assertTrue(len(gpus) >= 2, "expecting at least 2 gpus per process")

        process_group = self._get_process_group()

        gpus = gpus[:2]
        model = DoubleGpuNet(gpus)

        with self.assertRaisesRegex(
            ValueError,
            "DistributedDataParallel device_ids and output_device arguments only work with "
            "single-device/multiple-device GPU modules or CPU modules",
        ):
            DistributedDataParallel(
                model, output_device=gpus[1], process_group=process_group
            )

        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            DistributedDataParallel(model, device_ids=gpus, process_group=process_group)

        with self.assertRaisesRegex(
            ValueError, "input module must be on the same type of devices"
        ):
            model.fc1 = model.fc1.cpu()
            DistributedDataParallel(model, process_group=process_group)

        model = model.cpu()
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            DistributedDataParallel(model, device_ids=gpus, process_group=process_group)

    def _test_fp16(self, gradient_as_bucket_view=False):
        process_group = self._get_process_group()

        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).cuda(gpus[0]).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[gpus[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Input 2**15, so that the gradients will overflow with a
        # world_size of 2, unless we normalize the gradient by the
        # world_size before the reduction
        input = torch.tensor([[2**15]]).cuda(gpus[0]).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(any(torch.isinf(p.grad).any() for p in ddp_model.parameters()))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16(self):
        self._test_fp16()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_grad_is_view(self):
        self._test_fp16(gradient_as_bucket_view=True)

    def _test_arbitrary_forward_return_value(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class ForwardReturnValueModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x, fn):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # The first softmax does NOT include fc3 in its autograd graph
                # whereas the second softmax DOES. If we pass only the first
                # tensor we see in the output to the reducer, it marks the
                # gradient for fc3 as ready (because it doesn't show up). If
                # downstream uses of this return value choose to differentiate
                # against the second output tensor, it would still receive a
                # gradient and a callback for this tensor, resulting in a crash.
                return fn(
                    F.softmax(x, dim=1),
                    F.softmax(self.fc3(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            ForwardReturnValueModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Always run "backward" to ensure the reducer is called by autograd.
        # If we don't correctly capture the output tensors from the return value,
        # the reducer won't see a hook for the unused parameter, and throw an error.
        # The correct capture is what we're testing in this function.
        def test(box, unbox):
            output = model(input, fn=box)
            loss = criterion(unbox(output), target)
            loss.backward()

        # Test with identity return value
        test(
            box=lambda x, y: (x, y),
            unbox=lambda obj: obj[1],
        )

        # Test with list return value
        test(
            box=lambda x, y: ["foo", x, "bar", y],
            unbox=lambda obj: obj[3],
        )

        # Test with tuple return value
        test(
            box=lambda x, y: ("foo", x, "bar", y),
            unbox=lambda obj: obj[3],
        )

        # Test with dict return value
        test(
            box=lambda x, y: {"foo": "bar", "a": x, "b": y},
            unbox=lambda obj: obj["b"],
        )

        # Test with list with dict return value
        test(
            box=lambda x, y: ["foo", "bar", {"a": x, "b": y}],
            unbox=lambda obj: obj[2]["b"],
        )

        # Test with dict with list return value
        test(
            box=lambda x, y: {"foo": "bar", "list": [0, x, 1, y]},
            unbox=lambda obj: obj["list"][3],
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value(self):
        self._test_arbitrary_forward_return_value()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value_grad_is_view(self):
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_with_lazy_parameters(self):
        process_group = self._get_process_group()
        with self.assertRaisesRegex(
            RuntimeError, "Modules with uninitialized parameters"
        ):
            DistributedDataParallel(
                torch.nn.LazyLinear(10), process_group=process_group
            )

    def _test_find_unused_parameters_kwarg(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )
        process_group = c10d.distributed_c10d._get_default_group()

        class FindUnusedParametersModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # Return the fc3 module so that the caller can invoke it
                # outside of the forward function. While this is bad practice,
                # we can use it to trigger a reducer error.
                return (F.softmax(x, dim=1), self.fc3)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        ddp_model = None

        def test_find_unused_parameters(
            find_unused_parameters, test_default=False, gradient_as_bucket_view=False
        ):
            if test_default:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            else:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    find_unused_parameters=find_unused_parameters,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            nonlocal ddp_model
            ddp_model = model

            output, fc3 = model(input)
            output = fc3(output)
            loss = criterion(output, target)
            loss.backward()

        # First test that finding unused params under these conditions is to
        # trigger an error when `backward` is called (because fc3 is an unused
        # parameter and will therefore be marked ready twice).
        try:
            test_find_unused_parameters(
                True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.assertTrue(
                str(ex).startswith(
                    "Expected to mark a variable ready only once.",
                )
            )
            unused_index = 2
            unused_index_str = f"Parameter at index {unused_index}"
            model = ddp_model.module
            for module_name, module in model.named_modules():
                if module == model.fc3:
                    for parameter_name, _ in module.named_parameters(recurse=False):
                        unused_fqn = f"{module_name}.{parameter_name}"
                        # Only one such parameter in model.fc3, since bias=False
                        break

            if dist.get_debug_level() != dist.DebugLevel.OFF:
                unused_index_str += f" with name {unused_fqn}"

            self.assertTrue(unused_index_str in str(ex))
        else:
            self.fail("Expected exception")

        dist.barrier(process_group)

        # Then test that the default behavior can be overridden by setting
        # `find_unused_parameters=False`.
        try:
            test_find_unused_parameters(
                False, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail(f"Unexpected exception: {ex}")

        # Test find_unused_parameters defaults to False
        try:
            test_find_unused_parameters(
                True, test_default=True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail(f"Unexpected exception: {ex}")

    # TODO: Combine the following tests once https://github.com/pytorch/pytorch/issues/55967
    # is resolved.
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_debug_detail(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_debug_info(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_debug_off(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_detail(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_info(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_off(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    def _test_multiple_outputs_multiple_backward(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class MultipleOutputModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                def define_module():
                    return nn.Sequential(
                        nn.Linear(2, 10, bias=False),
                        nn.ReLU(),
                        nn.Linear(10, 4, bias=False),
                        nn.ReLU(),
                    )

                self.module0 = define_module()
                self.module1 = define_module()

            def forward(self, x):
                return (
                    F.softmax(self.module0(x), dim=1),
                    F.softmax(self.module1(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            MultipleOutputModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Compute loss and gradients for both outputs
        output1, output2 = model(input)
        loss1 = criterion(output1, target)
        loss1.backward()
        loss2 = criterion(output2, target)
        loss2.backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        self._te

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 27 class(es): RendezvousEnvTest, Env, TimeoutTest, ProcessGroupNCCLNoGPUTest, ProcessGroupNCCLInitTest, ProcessGroupNCCLGroupTest, ForwardReturnValueModule, FindUnusedParametersModule, MultipleOutputModule, NoGradModule, TestModel, AcceptsParam, WorkHookTest, NcclErrorHandlingTest, NcclUserBufferRegistrationTest, CommTest, SetDeviceMethod, LargeCommTest, SparseCollective, ToyModel

### Functions
This file defines 320 function(s): _ts, configure, log_test_info, log_test_success, log_test_validation, log_test_warning, log_test_error, test_common_errors, __init__, __enter__, __exit__, without, withouts, test_default_store_timeout_nccl, setUp, tearDown, test_init_no_gpus, setUp, tearDown, world_size, device, _init_process_group, test_init_wo_backend_str, test_scalable_init, _create_process_group_nccl, opts, setUp, tearDown, world_size, rank_to_GPU


## Key Components

The file contains 18513 words across 6366 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 255033 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
