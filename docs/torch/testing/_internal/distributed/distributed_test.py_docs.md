# Documentation: distributed_test.py

## File Metadata
- **Path**: `torch/testing/_internal/distributed/distributed_test.py`
- **Size**: 438301 bytes
- **Lines**: 10423
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-defs

import copy
import itertools
import json
import math
import operator
import os
import random
import re
import sys
import tempfile
import time
import unittest
from collections import defaultdict, namedtuple, OrderedDict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from typing import Any, NamedTuple, Union

import numpy as np

import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
import torch.distributed.algorithms.model_averaging.utils as model_averaging_utils
import torch.distributed.optim.post_localSGD_optimizer as post_localSGD_optimizer
import torch.nn as nn
import torch.nn.functional as F
from torch._utils_internal import (
    TEST_MASTER_ADDR as MASTER_ADDR,
    TEST_MASTER_PORT as MASTER_PORT,
)
from torch.autograd import DeviceType
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
    post_localSGD_hook as post_localSGD,
    powerSGD_hook as powerSGD,
    quantization as quantization_hooks,
)
from torch.distributed.distributed_c10d import (
    _get_default_group,
    _get_pg_config,
    get_world_size,
)
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.utils import (
    _sync_module_states,
    _verify_param_shape_across_processes,
)
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars, _MixedPrecision
from torch.profiler import ExecutionTraceObserver, ProfilerActivity
from torch.testing._internal.common_distributed import (
    captured_output,
    cleanup_temp_dir,
    DistTestCases,
    init_multigpu_helper,
    initialize_temp_directories,
    MultiProcessTestCase,
    nccl_skip_if_lt_x_gpu,
    require_n_gpus_for_nccl_backend,
    requires_nccl_version,
    simple_sparse_reduce_tests,
    skip_if_lt_x_gpu,
    skip_if_no_gpu,
    skip_if_odd_worldsize,
    skip_if_rocm_multiprocess,
    skip_if_small_worldsize,
    TEST_SKIPS,
    verify_ddp_error_logged,
    with_dist_debug_levels,
    with_nccl_blocking_wait,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.data.distributed import DistributedSampler


try:
    import torchvision

    HAS_TORCHVISION = True
except Exception:  # Covering both ImportError and RuntimeError
    HAS_TORCHVISION = False

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class NetWithBuffers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(10, 10, bias=False)
        self.b = nn.Linear(10, 1, bias=False)
        self.register_buffer("buffer", torch.randn(1, 2))

    def forward(self, x):
        self.buffer.add_(1)
        return self.b(self.a(x))


class Foo:
    def __init__(self, x):
        # Can be tensor or int
        self.x = x

    def __eq__(self, other):
        def eq(value, other):
            if isinstance(value, torch.Tensor):
                return torch.equal(value, other)
            return value == other

        for attr, value in self.__dict__.items():
            other_value = other.__dict__[attr]
            if not eq(value, other_value):
                return False
        return True


f = Foo(10)
f.bar = 1


# Defer instantiation until the seed is set so that randn() returns the same
# values in all processes.
def create_collectives_object_test_list():
    return [
        {"key1": 3, "key2": 4, "key3": {"nested": True}},
        f,
        Foo(torch.randn(3, 3)),
        "foo",
        [1, 2, True, "string", [4, 5, "nested"]],
    ]


# Allowlist of distributed backends where profiling collectives is supported.
PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.NCCL,
    dist.Backend.GLOO,
    dist.Backend.MPI,
    dist.Backend.UCC,
]

# Allowlist of distributed backends where profiling is supported with use_cuda=True
CUDA_PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.GLOO,
    dist.Backend.MPI,
    dist.Backend.NCCL,
    dist.Backend.UCC,
]

# Allowlist of distributed backends where profiling is supported for p2p ops
SEND_RECV_PROFILING_SUPPORTED_BACKENDS = [
    dist.Backend.MPI,
    dist.Backend.GLOO,
    dist.Backend.NCCL,
    dist.Backend.UCC,
]

# Dummy NamedTuple data structures to test DDP support for NamedTuple types.
EXPECTED_FIELDS = ("a", "b")
TestNamedTupleInput_0 = namedtuple("NamedTuple", EXPECTED_FIELDS)


class TestNamedTupleInput_1(NamedTuple):
    a: torch.tensor
    b: torch.tensor


skipIfNoTorchVision = skip_but_pass_in_sandcastle_if(
    not HAS_TORCHVISION, "no torchvision"
)

BACKEND = os.environ["BACKEND"]
INIT_METHOD = os.getenv("INIT_METHOD", "env://")

DEFAULT_TIMEOUT = 300
CUSTOMIZED_TIMEOUT = {"test_DistributedDataParallel": 500}


def get_profiling_event(event_name, profiler, dedup_gpu_user_annotation=False):
    event_list = (
        profiler.events()
        if isinstance(profiler, torch.profiler.profile)
        else profiler.function_events
    )
    return [
        event
        for event in event_list
        if (
            (event.name.endswith(event_name) or event.name.startswith(event_name))
            and (not dedup_gpu_user_annotation or event.device_type != DeviceType.CUDA)
        )
    ]


def get_profiler_nccl_meta(prof):
    """Torch profiler includes nccl metadata in an inserted operator called "record_param_comms"
    We will need to test metadata obtained from profiler here"""
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".json") as tf:
        tf.close()
        trace_file = tf.name

        prof.export_chrome_trace(trace_file)
        with open(trace_file) as f:
            events = json.load(f)["traceEvents"]
        print(f"Trace saved to {trace_file}")

        return [e for e in events if e.get("name") == "record_param_comms"]


# Base error message substring on unfinished reductions.
ddp_prev_reduction_unfinished_str = (
    "Expected to have finished reduction in the prior iteration"
)
# Error message substring when find_unused_parameters=True has not been passed
ddp_recommend_find_unused_params_str = (
    "passing the keyword argument `find_unused_parameters=True`"
)
# Error message substring when find_unused_parameters=True is enabled
ddp_find_unused_params_enabled_str = "Since `find_unused_parameters=True` is enabled"
# Error message substring for possibility of not all model outputs being used
# in loss computation
ddp_outputs_not_used_in_loss_str = (
    "`forward` function outputs participate in calculating loss"
)
# Error message substring suggesting to use TORCH_DISTRIBUTED_DEBUG
ddp_suggest_debug_mode_str = (
    "set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL"
)


class DDPUnevenTestInput(NamedTuple):
    name: str
    model: nn.Module
    inp: Union[torch.tensor, tuple]
    sync_interval: int
    throw_on_early_termination: bool = False
    hook: Callable = None
    state: Any = None


class _FC2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        )

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class LargeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1000, 2000, bias=False)
        self.fc2 = nn.Linear(2000, 500, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Task(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        return self.p + x


class BatchNormNet(nn.Module):
    def __init__(self, affine=True):
        super().__init__()
        self.fc1 = nn.Linear(2, 40, bias=False)
        self.bn = nn.BatchNorm1d(4, affine=affine)
        self.fc2 = nn.Linear(40, 4, bias=False)

    def forward(self, x):
        x = torch.reshape(self.fc1(x), (-1, 4, 10))
        x = self.bn(x)
        x = torch.reshape(x, (-1, 40))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class UnusedParamTwoLinLayerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(10, 10, bias=False)
        self.b = nn.Linear(10, 10, bias=False)
        self.c = nn.Linear(5, 5, bias=False)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)


class DictOutputModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = UnusedParamTwoLinLayerNet()

    def forward(self, x):
        predictions = self.module(x)
        loss = (predictions[0] + predictions[1]).sum()
        return {
            "predictions": predictions,
            "loss": loss,
        }


class TwoLinLayerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(10, 10, bias=False)
        self.b = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)


class EmbeddingNetDifferentParams(nn.Module):
    """
    A module containing an embedding with different dimension or different # of
    parameters depending on the rank.
    """

    def __init__(self, rank, diff_num_params=False):
        super().__init__()
        embedding_dim = 500 if diff_num_params or rank == 0 else 50
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        self.lin = nn.Linear(embedding_dim, 1)
        if diff_num_params:
            self.lin2 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        return self.lin(x)


class ControlFlowToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(10, 10, bias=False)
        self.lin2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        # Second layer is used dependent on input x.
        use_second_layer = torch.equal(x, torch.ones(20, 10, device=x.device))
        if use_second_layer:
            return self.lin2(F.relu(self.lin1(x)))
        else:
            return F.relu(self.lin1(x))


def get_timeout(test_id):
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        return DEFAULT_TIMEOUT


default_pg_timeout = 60

CUSTOM_PG_TIMEOUT = {
    # This test runs slowly and needs additional time to complete, otherwise can
    # be taken down by TORCH_NCCL_ASYNC_ERROR_HANDLING
    "test_ddp_uneven_inputs": 300,
    # This test has a short timeout since it tests being taken down by
    # TORCH_NCCL_ASYNC_ERROR_HANDLING which we want to happen quickly.
    "test_ddp_model_diff_across_ranks": 5,
    # This test has a short timeout since it tests being taken down by
    # TORCH_NCCL_ASYNC_ERROR_HANDLING which we want to happen quickly.
    "test_ddp_has_finalized": 5,
}


def require_backend_is_available(backends):
    def check(backend):
        if backend == dist.Backend.GLOO:
            return dist.is_gloo_available()
        if backend == dist.Backend.NCCL:
            return dist.is_nccl_available()
        if backend == dist.Backend.MPI:
            return dist.is_mpi_available()
        if backend == dist.Backend.UCC:
            return dist.is_ucc_available()
        if backend in DistTestCases.backend_feature["plugin"]:
            return True
        return False

    if BACKEND not in backends:
        return skip_but_pass_in_sandcastle(
            f"Test requires backend {BACKEND} to be one of {backends}"
        )

    if not check(dist.Backend(BACKEND)):
        return skip_but_pass_in_sandcastle(
            f"Test requires backend {BACKEND} to be available"
        )
    return lambda func: func


def require_world_size(world_size):
    if int(os.environ["WORLD_SIZE"]) < world_size:
        return skip_but_pass_in_sandcastle(
            f"Test requires world size of {world_size:d}"
        )
    return lambda func: func


def require_exact_world_size(world_size):
    if int(os.environ["WORLD_SIZE"]) != world_size:
        return skip_but_pass_in_sandcastle(
            f"Test requires an exact world size of {world_size:d}"
        )
    return lambda func: func


@contextmanager
def _lock():
    TEMP_DIR = os.environ["TEMP_DIR"]
    lockfile = os.path.join(TEMP_DIR, "lockfile")
    with open(lockfile, "w") as lf:
        try:
            if sys.platform == "win32":
                msvcrt.locking(lf.fileno(), msvcrt.LK_RLCK, 1)
                yield
            else:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            if sys.platform == "win32":
                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            lf.close()


@contextmanager
def _rank_temp_file():
    if dist.get_rank() == 0:
        fd, name = tempfile.mkstemp()
        os.close(fd)
    else:
        name = None
    object_list = [name]
    dist.broadcast_object_list(object_list)
    name = object_list[0]
    try:
        yield name
    finally:
        if dist.get_rank() == 0:
            os.remove(name)


def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, size, size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, size, size, dtype=dtype).fill_(value).cuda(device_id)


def _build_multidim_tensor(dim, dim_size, value=None, dtype=torch.float):
    if value is None:
        value = dim
    return torch.empty(size=[dim_size for _ in range(dim)], dtype=dtype).fill_(value)


def _create_autograd_profiler():
    return torch.autograd.profiler.profile(record_shapes=True)


def _create_torch_profiler():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
    )


class Barrier:
    barrier_id = 0

    @classmethod
    def init(cls):
        cls.barrier_id = 0
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))

    @classmethod
    def sync(cls, wait_for=None, timeout=10):
        if wait_for is None:
            wait_for = dist.get_world_size()
        cls.barrier_id += 1
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        pid = str(os.getpid())
        barrier_file = os.path.join(barrier_dir, pid)
        with _lock():
            with open(barrier_file, "w") as f:
                f.write(str(cls.barrier_id))

        start_time = time.time()
        while True:
            arrived = 0
            with _lock():
                for f_name in os.listdir(barrier_dir):
                    with open(os.path.join(barrier_dir, f_name)) as f:
                        data = f.read()
                        if int(data) >= cls.barrier_id:
                            arrived += 1
            if arrived == wait_for:
                break

            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            time.sleep(0.1)


class TestDistBackend(MultiProcessTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        # Not setting MASTER_PORT and get a random free port
        super().setUpClass()

    def setUp(self):
        super().setUp()
        # initialize temp directories
        initialize_temp_directories()
        # initialize Barrier
        Barrier.init()
        # Skip return code checking for following tests as they are expected to
        # crash a process due to TORCH_NCCL_ASYNC_ERROR_HANDLING.
        self.skip_return_code_checks = [self.test_ddp_has_finalized.__wrapped__]

    def tearDown(self):
        cleanup_temp_dir()
        super().tearDown()

    @property
    def init_method(self):
        return f"{FILE_SCHEMA}{self.file_name}"

    @property
    def destroy_pg_upon_exit(self) -> bool:
        # Overriding base test class: do not auto destroy PG upon exit.
        return False

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe, **kwargs):
        if BACKEND == "nccl" and not torch.cuda.is_available():
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        if torch.cuda.is_available() and torch.cuda.device_count() < int(
            self.world_size
        ):
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        try:
            pg_timeout_seconds = CUSTOM_PG_TIMEOUT.get(test_name, default_pg_timeout)
            timeout = timedelta(seconds=pg_timeout_seconds)
            dist.init_process_group(
                init_method=self.init_method,
                backend=BACKEND,
                world_size=int(self.world_size),
                rank=self.rank,
                timeout=timeout,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        self._barrier()

        self.run_test(test_name, pipe)
        self._barrier()
        dist.destroy_process_group()
        sys.exit(0)

    # Needed since MultiProcessTestCase assumes a world_size of 4, but we
    # run these tests under other various world_sizes.
    @property
    def world_size(self):
        return os.environ["WORLD_SIZE"]


class DistributedTest:
    class _DistTestBase:
        def _barrier(self, *args, **kwargs):
            Barrier.sync(*args, **kwargs)

        def _init_group_test(self, **kwargs):
            group = [1, 2]
            group_id = dist.new_group(group, **kwargs)
            rank = dist.get_rank()
            if rank not in group:
                return ([], None, rank)

            return (group, group_id, rank)

        def _init_full_group_test(self, **kwargs):
            group = list(range(dist.get_world_size()))
            group_id = dist.new_group(**kwargs)
            rank = dist.get_rank()
            return (group, group_id, rank)

        def _init_global_test(self):
            group = list(range(dist.get_world_size()))
            group_id = dist.group.WORLD
            rank = dist.get_rank()
            return (group, group_id, rank)

        def _verify_buffers_equal(self, m1, m2):
            # verify buffers across models
            m1_buf_dict = dict(m1.module.named_buffers())
            for name, buf in m2.module.named_buffers():
                self.assertEqual(buf, m1_buf_dict[name])

            # Verify buffers across ranks.
            m1_buffers = list(m1.buffers())
            m2_buffers = list(m2.buffers())
            for buf1, buf2 in zip(m1_buffers, m2_buffers, strict=True):
                gathered_bufs = [
                    torch.empty_like(buf1) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_bufs, buf1)
                gathered_bufs_m2 = [
                    torch.empty_like(buf2) for _ in range(dist.get_world_size())
                ]
                for b in gathered_bufs:
                    self.assertEqual(b, buf1)
                dist.all_gather(gathered_bufs_m2, buf2)
                for b in gathered_bufs_m2:
                    self.assertEqual(b, buf2)

        def _sanity_check_profiler_nccl_meta(self, nccl_meta_events):
            """Torch profiler includes nccl metadata in an inserted operator called "record_param_comms"
            We test for basic fields in this profiler event that correspond to the nccl communication
            collectives"""
            per_coll_meta = defaultdict(list)
            for e in nccl_meta_events:
                args = e.get("args", {})
                collname = args.get("Collective name", "")
                self.assertNotEqual(collname, "")
                self.assertNotEqual(args.get("dtype", ""), "")

                per_coll_meta[collname].append(args)
                if collname == "wait":
                    continue

                self.assertEqual(args["Process Group Description"], "default_pg")
                self.assertNotEqual(args["Process Group Ranks"], "")

                self.assertGreaterEqual(args.get("In msg nelems", -1), 0)
                self.assertGreaterEqual(args.get("Out msg nelems", -1), 0)
                self.assertGreaterEqual(args.get("Group size", -1), 0)
                self.assertGreaterEqual(args.get("Global rank start", -1), 0)
                self.assertGreaterEqual(args.get("Global rank stride", -1), 0)

            # print(per_coll_meta)
            return per_coll_meta

        def test_dump_DDP_relevant_env_vars(self):
            with captured_output() as (out, _):
                _dump_DDP_relevant_env_vars()
                lines = out.getvalue().splitlines()

            def format_line(var):
                return f"env:{var}={os.environ.get(var, 'N/A')}"

            # Check relevant env vars
            vars = [
                "MASTER_ADDR",
                "MASTER_PORT",
                "WORLD_SIZE",
                "NCCL_TOPO_DUMP_FILE",  # N/A
                "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            ]
            for var in vars:
                line = format_line(var)
                self.assertIn(line, lines)
            # Check irrelevant env vars
            vars = [
                "xxx",
                "yyy",
                "zzz",
            ]
            for var in vars:
                line = format_line(var)
                self.assertNotIn(line, lines)

        # GET RANK
        def test_get_rank(self):
            test_dir = os.path.join(os.environ["TEMP_DIR"], "test_dir")
            pid = str(os.getpid())
            num_processes = dist.get_world_size()
            with open(os.path.join(test_dir, pid), "w") as f:
                f.write(str(dist.get_rank()))

            self._barrier()

            all_ranks = set()
            for f_name in os.listdir(test_dir):
                with open(os.path.join(test_dir, f_name)) as f:
                    all_ranks.add(int(f.read()))
            self.assertEqual(len(all_ranks), num_processes)

            self._barrier()

            if dist.get_rank() == 0:
                for f_name in os.listdir(test_dir):
                    os.unlink(os.path.join(test_dir, f_name))

            self._barrier()

        def test_get_backend(self):
            if dist.get_world_size() > 2:
                group = [1, 2]
            else:
                group = [0, 1]
            group_id = dist.new_group(group)
            backend_str = BACKEND.lower()
            self.assertEqual(dist.get_backend(), backend_str)
            if dist.get_rank() in group:
                self.assertEqual(dist.get_backend(group_id), backend_str)
            else:
                with self.assertRaisesRegex(
                    ValueError, "Invalid process group specified"
                ):
                    dist.get_backend(group_id)

        def test_Backend_enum_class(self):
            # test parsing
            backend = BACKEND.lower()
            self.assertEqual(dist.Backend(BACKEND.upper()), backend)
            self.assertEqual(dist.Backend(BACKEND), backend)
            with self.assertRaises(ValueError):
                dist.Backend(None)
            with self.assertRaises(ValueError):
                dist.Backend(3)
            with self.assertRaises(ValueError):
                dist.Backend(["gloo"])

        # Test destroy
        def test_destroy_group(self):
            if dist.get_world_size() > 2:
                group = [1, 2]
            else:
                group = [0, 1]
            group_id = dist.new_group(group)
            self._barrier()
            dist.destroy_process_group(group_id)

        # Test get rank and size of group
        def test_get_rank_size_group(self):
            if dist.get_world_size() > 2:
                group = [1, 2]
            else:
                group = [0, 1]
            group_id = dist.new_group(group)
            if dist.get_rank() in group:
                self.assertEqual(dist.get_world_size(group_id), 2)
                self.assertTrue(dist.get_rank(group_id) in list(range(2)))
            else:
                self.assertEqual(dist.get_world_size(group_id), -1)
                self.assertEqual(dist.get_rank(group_id), -1)

        # Test destroy full groups
        def test_destroy_full_group(self):
            _, group_id, _ = self._init_full_group_test()
            self._barrier()
            dist.destroy_process_group(group_id)

        # Test get rank and size of full group
        def test_get_rank_size_full_group(self):
            _, group_id, _ = self._init_full_group_test()
            self.assertEqual(dist.get_world_size(group_id), dist.get_world_size())
            self.assertEqual(dist.get_rank(group_id), dist.get_rank())

        def _test_barrier_timeout(self, group_id, timeout):
            local_rank = dist.get_rank(group_id)

            # Only execute barrier on rank == 0, causing it to timeout
            if local_rank == 0:
                expected_time = time.time() + timeout.total_seconds()
                # In debug mode, we execute a monitored_barrier before the
                # collective, so assert on that.
                if dist.get_debug_level() == dist.DebugLevel.DETAIL:
                    exception_ctx = self.assertRaisesRegex(
                        Exception, "failed to pass monitoredBarrier"
                    )
                else:
                    exception_ctx = self.assertRaisesRegex(
                        Exception, " (Timed out|closed|timeout) "
                    )
                with exception_ctx:
                    dist.barrier(group_id)
                self.assertGreaterAlmostEqual(time.time(), expected_time, delta=0.1)

        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports timeouts"
        )
        @skip_but_pass_in_sandcastle_if(
            not INIT_METHOD.startswith("file://"),
            "Requires file:// initialization method. "
            + "Both tcp:// and env:// rely on the TCP store for which "
            "reinitialization has proven racy.",
        )
        def test_barrier_timeout_global(self):
            dist.destroy_process_group()

            # Explicitly pass world size to the barrier because we've
            # just destroyed any state in torch.distributed.
            self._barrier(wait_for=int(os.environ["WORLD_SIZE"]))

            # Reinitialize global process group
            timeout = timedelta(seconds=1)
            dist.init_process_group(
                init_method=INIT_METHOD,
                backend=BACKEND,
                world_size=int(os.environ["WORLD_SIZE"]),
                rank=self.rank,
                timeout=timeout,
            )
            self._test_barrier_timeout(dist.group.WORLD, timeout)

        @skip_if_small_worldsize
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports timeouts"
        )
        def test_barrier_timeout_group(self):
            timeout = timedelta(seconds=5)
            _, group_id, _ = self._init_group_test(timeout=timeout)
            if group_id is not None:
                self._test_barrier_timeout(group_id, timeout)

        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports timeouts"
        )
        def test_barrier_timeout_full_group(self):
            timeout = timedelta(seconds=1)
            _, group_id, _ = self._init_full_group_test(timeout=timeout)
            if group_id is not None:
                self._test_barrier_timeout(group_id, timeout)

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @require_world_size(4)
        @skip_if_lt_x_gpu(2)
        def test_new_subgroups(self):
            subgroup_size = 2
            cur_subgroup, subgroups = dist.new_subgroups(subgroup_size)

            world_size = dist.get_world_size()
            self.assertEqual(cur_subgroup.size(), subgroup_size)
            self.assertEqual(len(subgroups), world_size / subgroup_size)
            self.assertFalse(dist._rank_not_in_group(cur_subgroup))

            for subgroup in subgroups:
                dist.destroy_process_group(subgroup)

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @require_exact_world_size(4)
        def test_new_subgroups_with_group_param(self):
            # Initialize global test environment
            self._init_global_test()
            # Set up GPU devices for each rank
            init_multigpu_helper(dist.get_world_size(), BACKEND)
            # Create two subgroups: one with ranks [0,2] and another with ranks [1,3]
            cur_subgroup, subgroups = dist.new_subgroups_by_enumeration(
                ranks_per_subgroup_list=[[0, 2], [1, 3]]
            )

            # Further divide the current subgroup into sub-subgroups of size 1
            cur_sub_subgroup, sub_subgroups = dist.new_subgroups(
                group_size=1, group=cur_subgroup
            )
            # Verify we have 2 sub-subgroups (one for each rank in the original subgroup)
            self.assertEqual(len(sub_subgroups), 2)
            # Verify the current process's sub-subgroup has size 1
            self.assertEqual(cur_sub_subgroup.size(), 1)
            # Verify the current process is in its assigned sub-subgroup
            self.assertFalse(dist._rank_not_in_group(group=cur_sub_subgroup))

            # Clean up by destroying all created process groups
            for sub_subgroup in sub_subgroups:
                dist.destroy_process_group(sub_subgroup)

            for subgroup in subgroups:
                dist.destroy_process_group(subgroup)

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @skip_if_no_gpu
        def test_new_subgroups_group_size_exceeds_world_size(self):
            with self.assertRaisesRegex(ValueError, "must not exceed"):
                dist.new_subgroups(100)

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @require_world_size(4)
        @skip_if_lt_x_gpu(4)
        def test_new_subgroups_world_size_not_divisible_by_group_size(self):
            expected_msg = f"The world size ({dist.get_world_size()}) must be divisible by 'group_size=3'"
            with self.assertRaisesRegex(
                ValueError,
                re.escape(expected_msg),
            ):
                dist.new_subgroups(3)

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @require_world_size(4)
        @skip_if_lt_x_gpu(4)
        def test_new_subgroups_by_enumeration(self):
            _group, _group_id, rank = self._init_global_test()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]
            cur_subgroup, subgroups = dist.new_subgroups_by_enumeration(
                ranks_per_subgroup_list=[[0, 2], [1, 3]]
            )
            if device_id >= 4:
                self.assertIsNone(cur_subgroup)
            else:
                self.assertEqual(cur_subgroup.size(), 2)
                self.assertEqual(len(subgroups), 2)
                if device_id == 0 or device_id == 2:
                    self.assertEqual(cur_subgroup, subgroups[0])
                else:
                    self.assertEqual(cur_subgroup, subgroups[1])

            for subgroup in subgroups:
                dist.destroy_process_group(subgroup)

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @require_world_size(4)
        @skip_if_lt_x_gpu(4)
        def test_new_subgroups_by_enumeration_input_rank_exceeds_world_size(self):
            _group, group_id, _rank = self._init_global_test()
            init_multigpu_helper(dist.get_world_size(), BACKEND)
            world_size = get_world_size(group_id)

            with self.assertRaisesRegex(
                ValueError,
                "The new group's rank should be within the world_size set by init_process_group",
            ):
                dist.new_subgroups_by_enumeration(
                    ranks_per_subgroup_list=[[0, 1], [world_size, 2]]
                )

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @skip_if_no_gpu
        def test_new_subgroups_by_enumeration_negative_input_rank(self):
            self._init_global_test()

            with self.assertRaisesRegex(
                ValueError,
                "The new group's rank should be within the world_size set by init_process_group",
            ):
                dist.new_subgroups_by_enumeration(
                    ranks_per_subgroup_list=[[-1, -2], [-3, -4]]
                )

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @require_world_size(4)
        @skip_if_lt_x_gpu(4)
        def test_new_subgroups_overlap_not_allowed(self):
            with self.assertRaisesRegex(
                ValueError, "Rank 1 has appeared in both subgroup"
            ):
                dist.new_subgroups_by_enumeration(
                    ranks_per_subgroup_list=[[0], [1, 2], [1, 3]]
                )

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @skip_if_lt_x_gpu(2)
        def test_average_parameters(self):
            rank = dist.get_rank()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]

            model = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Linear(1, 5, bias=False),
            ).cuda(device_id)
            # Test global model averaging
            for p in model.parameters():
                p.data = torch.ones_like(p.data)
            model_averaging_utils.average_parameters(
                params=model.parameters(), process_group=None
            )
            # Every element will be the same as the input.
            for p in model.parameters():
                self.assertEqual(p.data, torch.ones_like(p.data))

            # Test partial model averaging
            for p in model.parameters():
                p.data = torch.ones_like(p.data) * rank
            group_nccl = dist.new_group(ranks=[0, 1], backend="nccl")
            model_averaging_utils.average_parameters(
                params=model.parameters(), process_group=group_nccl
            )
            if not dist._rank_not_in_group(group_nccl):
                # Every element on device 0 or 1 should be the average of 0 and 1, i.e., 0.5.
                for p in model.parameters():
                    self.assertEqual(p.data, torch.ones_like(p.data) * 0.5)
            else:
                # Every element on device not in the subgroup should remain the same.
                for p in model.parameters():
                    self.assertEqual(p.data, torch.ones_like(p.data) * rank)

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @skip_if_lt_x_gpu(2)
        def test_periodic_model_averager(self):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]

            model = nn.Linear(1, 5, bias=False).cuda(device_id)
            param = next(model.parameters())
            tensor = torch.ones_like(param.data) * rank
            expected_avg_tensor = (
                torch.ones_like(param.data) * sum(range(world_size)) / world_size
            )
            period = 4
            for warmup_steps in [12, 13, 14, 15]:
                averager = averagers.PeriodicModelAverager(
                    period=period, warmup_steps=warmup_steps
                )
                for step in range(20):
                    # Reset the parameters at every step.
                    param.data = copy.deepcopy(tensor)
                    for params in model.parameters():
                        # mock grad
                        params.grad = torch.ones_like(param.data)
                    averager.average_parameters(model.parameters())
                    if step >= warmup_steps and (step - warmup_steps) % period == 0:
                        self.assertEqual(param.data, expected_avg_tensor)
                    else:
                        # No model averaging, so the parameters are not updated.
                        self.assertEqual(param.data, tensor)

        @skip_if_lt_x_gpu(2)
        def test_periodic_model_averager_param_group(self):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]

            model = nn.Linear(1, 5, bias=False).cuda(device_id)
            param = next(model.parameters())
            opt = torch.optim.SGD(model.parameters(), lr=0.1)

            period = 4
            for warmup_steps in [12, 13, 14, 15]:
                averager = averagers.PeriodicModelAverager(
                    period=period, warmup_steps=warmup_steps
                )
                for step in range(20):
                    # Reset the parameters at every step.
                    for param_group in opt.param_groups:
                        for params in param_group["params"]:
                            # mock grad
                            params.grad = torch.ones_like(param.data) * rank
                            params.data = torch.ones_like(param.data) * rank
                    averager.average_parameters(opt.param_groups)
                    if step >= warmup_steps and (step - warmup_steps) % period == 0:
                        for param_group in opt.param_groups:
                            for params in param_group["params"]:
                                if params.grad is None:
                                    continue
                                self.assertEqual(
                                    param.data,
                                    torch.ones_like(param.data)
                                    * sum(range(world_size))
                                    / world_size,
                                )
                    else:
                        # No model averaging, so the parameters are not updated.
                        for param_group in opt.param_groups:
                            for params in param_group["params"]:
                                if params.grad is None:
                                    continue
                                self.assertEqual(
                                    param.data, torch.ones_like(param.data) * rank
                                )

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @skip_if_lt_x_gpu(2)
        def test_1_level_hierarchical_model_averager_equivalent_to_periodic_model_averager(
            self,
        ):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]

            model = nn.Linear(1, 5, bias=False).cuda(device_id)
            param = next(model.parameters())
            tensor = torch.ones_like(param.data) * rank
            expected_avg_tensor = (
                torch.ones_like(param.data) * sum(range(world_size)) / world_size
            )
            period = 4
            for warmup_steps in [12, 13, 14, 15]:
                averager = hierarchicalSGD.HierarchicalModelAverager(
                    # Run the global averaging at a period of 4,
                    # which is equivalent to the above periodic model averaging test case.
                    period_group_size_dict=OrderedDict([(period, world_size)]),
                    warmup_steps=warmup_steps,
                )

                averager = averagers.PeriodicModelAverager(
                    period=period, warmup_steps=warmup_steps
                )
                for step in range(20):
                    # Reset the parameters at every step.
                    param.data = copy.deepcopy(tensor)
                    for params in model.parameters():
                        # mock grad
                        params.grad = torch.ones_like(param.data)
                    averager.average_parameters(model.parameters())
                    if step >= warmup_steps and (step - warmup_steps) % period == 0:
                        self.assertEqual(param.data, expected_avg_tensor)
                    else:
                        # No model averaging, so the parameters are not updated.
                        self.assertEqual(param.data, tensor)

        @skip_but_pass_in_sandcastle_if(
            BACKEND not in DistTestCases.backend_feature["subgroup"],
            f"The {BACKEND} backend does not support creating subgroups on CUDA devices",
        )
        @require_exact_world_size(4)
        @skip_if_lt_x_gpu(4)
        def test_3_level_hierarchical_model_averager(self):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]

            model = nn.Linear(1, 5, bias=False).cuda(device_id)
            param = next(model.parameters())
            tensor = torch.ones_like(param.data) * rank
            # Set up such a hierarchical model averaging as follows:
            # after the first 10 warmup steps,
            # run model averaging every 2 steps within each subgroup of size 2,
            # run model averaging every 4 steps within each subgroup of size 3,
            # and run the global model averaging every 8 steps.
            # If there is a conflict in model averaging at a step, only run the highest-level model averaging.
            warmup_steps = 10
            subgroup_size1 = 2
            subgroup_avg_period1 = 2
            subgroup_size2 = 4
            subgroup_avg_period2 = 4
            global_avg_period = 8
            period_group_size_dict = OrderedDict(
                [
                    (subgroup_avg_period1, subgroup_size1),
                    (subgroup_avg_period2, subgroup_size2),
                    (global_avg_period, world_size),
                ]
            )
            averager = hierarchicalSGD.HierarchicalModelAverager(
                period_group_size_dict=period_group_size_dict, warmup_steps=warmup_steps
            )
            self.assertEqual(dist.get_pg_count(), len(period_group_size_dict))

            subgroup1 = averager.period_process_group_dict[subgroup_avg_period1]
            subgroup2 = averager.period_process_group_dict[subgroup_avg_period2]
            real_group_ranks_res1 = _get_pg_config(subgroup1)["ranks"]
            real_group_ranks_res2 = _get_pg_config(subgroup2)["ranks"]

            expect_group_ranks_res1 = (
                rank // subgroup_size1 * subgroup_size1
                + np.array(list(range(subgroup_size1)))
            ).tolist()
            expect_group_ranks_res2 = (
                rank // subgroup_size2 * subgroup_size2
                + np.array(list(range(subgroup_size2)))
            ).tolist()
            self.assertEqual(real_group_ranks_res1, expect_group_ranks_res1)
            self.assertEqual(real_group_ranks_res2, expect_group_ranks_res2)

            expected_avg_tensor_within_subgroup1 = (
                torch.ones_like(param.data)
                * sum(real_group_ranks_res1)
                / subgroup_size1
            )
            expected_avg_tensor_within_subgroup2 = (
                torch.ones_like(param.data)
                * sum(real_group_ranks_res2)
                / subgroup_size2
            )
            expected_global_avg_tensor = (
                torch.ones_like(param.data) * sum(range(world_size)) / world_size
            )
            for step in range(25):
                # Reset the parameters at every step.
                param.data = copy.deepcopy(tensor)
                for params in model.parameters():
                    # mock grad
                    params.grad = torch.ones_like(param.data)
                averager.average_parameters(model.parameters())
                if step == 16 or step == 24:
                    # Run global model averaging when `step` can be divided by 8.
                    self.assertEqual(param.data, expected_global_avg_tensor)
                elif step == 12 or step == 20:
                    # Run model averaging within subgroup when `step` can be divided by 4 but not by 8.
                    self.assertEqual(param.data, expected_avg_tensor_within_subgroup2)
                elif step == 10 or step == 14 or step == 18 or step == 22:
                    # Run model averaging within subgroup when `step` can be divided by 2 but not by 4 or 8.
                    self.assertEqual(param.data, expected_avg_tensor_within_subgroup1)
                else:
                    # No model averaging, so the parameters are not updated.
                    self.assertEqual(param.data, tensor)

        # Coalescing manager (sync mode)
        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl" or IS_FBCODE or IS_SANDCASTLE,
            "Coalescing manager currently tests with NCCL only; internal test flaky",
        )
        def test_coalescing_manager(self):
            self._barrier()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            num_colls = 2
            size_per_coll = 8
            small_tensors = [
                torch.ones(size_per_coll, device=device_id) for _ in range(num_colls)
            ]

            with dist._coalescing_manager():
                for i in range(num_colls):
                    dist.all_reduce(small_tensors[i])

            big_tensor = torch.ones(num_colls * size_per_coll, device=device_id)
            dist.all_reduce(big_tensor)

            for i in range(num_colls):
                self.assertEqual(
                    small_tensors[i],
                    big_tensor[i * size_per_coll : (i + 1) * size_per_coll],
                )

            self._barrier()

        # Coalescing manager (async mode)
        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl" or IS_FBCODE or IS_SANDCASTLE,
            "Coalescing manager currently tests with NCCL only; internal test flaky",
        )
        def test_coalescing_manager_async(self):
            self._barrier()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            num_colls = 2
            size_per_coll = 8
            small_tensors = [
                torch.ones(size_per_coll, device=device_id) for _ in range(num_colls)
            ]

            with dist._coalescing_manager(async_ops=True) as cm:
                for i in range(num_colls):
                    dist.all_reduce(small_tensors[i])
            cm.wait()

            big_tensor = torch.ones(num_colls * size_per_coll, device=device_id)
            dist.all_reduce(big_tensor)

            for i in range(num_colls):
                self.assertEqual(
                    small_tensors[i],
                    big_tensor[i * size_per_coll : (i + 1) * size_per_coll],
                )

            self._barrier()

        # NCCL Batch SEND RECV
        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_nccl(self):
            self._barrier()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            p2p_op_list = []
            recv_tensors = [None for _ in range(world_size)]
            expected_tensors = [None for _ in range(world_size)]

            for val in ["1", "0"]:
                os.environ["TORCH_NCCL_BLOCKING_WAIT"] = val
                for src in range(world_size):
                    send_tensor = _build_tensor(rank + 1, device_id=device_id).fill_(
                        src
                    )
                    recv_tensors[src] = _build_tensor(
                        src + 1, value=-1, device_id=device_id
                    ).fill_(-1)
                    expected_tensors[src] = _build_tensor(
                        src + 1, value=-1, device_id=device_id
                    ).fill_(rank)
                    recv_op = dist.P2POp(dist.irecv, recv_tensors[src], src)
                    p2p_op_list.append(recv_op)
                    send_op = dist.P2POp(dist.isend, send_tensor, src)
                    p2p_op_list.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()

                for src in range(world_size):
                    self.assertEqual(recv_tensors[src], expected_tensors[src])

            self._barrier()

        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_ring_exchange_nccl(self):
            self._barrier()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)

            send_tensor = _build_tensor(world_size, device_id=device_id)
            recv_tensor = _build_tensor(world_size, value=-1, device_id=device_id)
            send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
            recv_op = dist.P2POp(
                dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size
            )
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()

            self._barrier()

        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_self_nccl(self):
            self._barrier()
            # Ensure the process group has been fully initialized (needed by
            # the first sub-group batch_isend_irecv call)
            dist.barrier()
            rank = dist.get_rank()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]
            p2p_op_list = []

            if rank == 0:
                send_tensor = _build_tensor(rank + 1, device_id=device_id)
                recv_tensor = _build_tensor(rank + 1, value=-1, device_id=device_id)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, 0)
                p2p_op_list.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()

            self._barrier()

        @skip_if_no_gpu
        @skip_if_small_worldsize
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_no_rank_zero_nccl(self):
            self._barrier()
            # Ensure the process group has been fully initialized (needed by
            # the first sub-group batch_isend_irecv call)
            dist.barrier()
            rank = dist.get_rank()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            p2p_op_list = []

            if rank == 1:
                peer = 2
            elif rank == 2:
                peer = 1

            if rank in [1, 2]:
                send_tensor = _build_tensor(rank + 1, device_id=device_id)
                recv_tensor = _build_tensor(peer + 1, value=-1, device_id=device_id)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, peer)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, peer)
                p2p_op_list.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()

            self._barrier()

        # GLOO Batch SEND RECV CPU
        @skip_but_pass_in_sandcastle_if(BACKEND != "gloo", "GLOO Batch Send Recv CPU")
        def test_batch_isend_irecv_gloo(self):
            self._barrier()
            rank = dist.get_rank()
            p2p_op_list = []

            for src in range(dist.get_world_size()):
                if src == rank:
                    continue
                send_tensor = _build_tensor(rank + 1)
                recv_tensor = _build_tensor(src + 1, value=-1)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, src)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, src)
                p2p_op_list.append(send_op)

            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()

            self._barrier()

        # GLOO Batch SEND RECV CPU with provided tags
        @skip_but_pass_in_sandcastle_if(BACKEND != "gloo", "GLOO Batch Send Recv CPU")
        def test_batch_isend_irecv_gloo_tags(self):
            self._barrier()
            rank = dist.get_rank()
            p2p_op_list = []

            for src in range(dist.get_world_size()):
                if src == rank:
                    continue
                send_tensor = _build_tensor(rank + 1)
                recv_tensor = _build_tensor(src + 1, value=-1)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, src, tag=src)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, src, tag=rank)
                p2p_op_list.append(send_op)

            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()

            self._barrier()

        # NCCL Batch SEND RECV Op Error
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_op_err(self):
            self._barrier()
            rank = dist.get_rank()
            if rank == 0:
                rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
                device_id = rank_to_GPU[rank][0]
                with self.assertRaisesRegex(ValueError, "^Invalid ``op``"):
                    send_tensor = _build_tensor(rank + 1, device_id=device_id)
                    send_op = dist.P2POp(dist.broadcast, send_tensor, 1)
                    dist.batch_isend_irecv([send_op])

        # NCCL Batch SEND RECV p2p_op_list Error
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_op_list_err(self):
            self._barrier()
            rank = dist.get_rank()
            if rank == 0:
                with self.assertRaisesRegex(ValueError, "^Invalid ``p2p_op_list``"):
                    dist.batch_isend_irecv([1, 2])

        # NCCL Batch SEND RECV Mixed Backend Error
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Batch Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_batch_isend_irecv_mixed_backend_err(self):
            self._barrier()
            rank = dist.get_rank()
            init_multigpu_helper(dist.get_world_size(), BACKEND)
            group_gloo = dist.new_group(ranks=[0, 1], backend="gloo")
            group_nccl = dist.new_group(ranks=[0, 1], backend="nccl")
            if rank == 0:
                with self.assertRaisesRegex(
                    ValueError, "All ops need to use the same group"
                ):
                    send_tensor = _build_tensor(rank + 1)
                    send_op_gloo = dist.P2POp(dist.isend, send_tensor, 1, group_gloo)
                    send_op_nccl = dist.P2POp(dist.isend, send_tensor, 1, group_nccl)
                    dist.batch_isend_irecv([send_op_gloo, send_op_nccl])

        # NCCL SEND RECV
        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def _test_send_recv_nccl(self, profiler_ctx=None):
            # TODO: now that nccl send/recv is supported, there does not seem to
            # be a need to have nccl send/recv be tested separately.
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)

            tensor = _build_tensor(rank + 1, device_id=device_id)
            profiler_cls = profiler_ctx if profiler_ctx is not None else nullcontext()
            with profiler_cls as prof:
                for src in range(world_size):
                    if src == rank:
                        # Send mode
                        for dst in range(world_size):
                            if dst == rank:
                                continue
                            dist.send(tensor, dst)
                    else:
                        # Recv mode
                        expected_tensor = _build_tensor(src + 1)
                        output_tensor = _build_tensor(
                            src + 1, value=-1, device_id=device_id
                        )
                        dist.recv(output_tensor, src)
                        self.assertEqual(output_tensor, expected_tensor)

                self._barrier()

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    for event_name in [f"{backend}:send", f"{backend}:recv"]:
                        events = get_profiling_event(
                            event_name, prof, dedup_gpu_user_annotation=True
                        )
                        self.assertTrue(events)
                        # Event order is not deterministic, so simply assert their shape
                        # is found in the following list.
                        expected_shapes = [
                            [[rank + 1] * 3] for rank in range(dist.get_world_size())
                        ]
                        for event in events:
                            self.assertTrue(event.input_shapes in expected_shapes)

        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_send_recv_nccl(self):
            self._test_send_recv_nccl()

        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        def test_send_recv_nccl_autograd_profiler(self):
            profiler_ctx = torch.autograd.profiler.profile(record_shapes=True)
            self._test_send_recv_nccl(profiler_ctx)

        @skip_if_no_gpu
        @skip_but_pass_in_sandcastle_if(BACKEND != "nccl", "NCCL Send Recv Only")
        @requires_nccl_version((2, 7, 0), "Need NCCL 2.7+ for send/recv")
        @skip_but_pass_in_sandcastle_if(IS_FBCODE, "Kineto in fbcode causes hang")
        @skip_but_pass_in_sandcastle_if(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124",
        )
        def test_send_recv_nccl_torch_profiler(self):
            profiler_ctx = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            )
            self._test_send_recv_nccl(profiler_ctx)

        # SEND RECV
        def _test_send_recv(self, profiler_ctx):
            rank = dist.get_rank()
            send_size = rank + 1
            tensor = _build_tensor(send_size)
            ctx = profiler_ctx if profiler_ctx is not None else nullcontext()
            with ctx as prof:
                for src in range(dist.get_world_size()):
                    if src == rank:
                        # Send mode
                        for dst in range(dist.get_world_size()):
                            if dst == rank:
                                continue
                            dist.send(tensor, dst)
                    else:
                        # Recv mode
                        recv_size = src + 1
                        expected_tensor = _build_tensor(recv_size)
                        output_tensor = _build_tensor(recv_size, value=-1)
                        dist.recv(output_tensor, src)
                        self.assertEqual(output_tensor, expected_tensor)

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    for event_name in [f"{backend}:send", f"{backend}:recv"]:
                        events = get_profiling_event(event_name, prof)
                        # Each rank sends/recvs from all other ranks.
                        event_count = sum(e.count for e in events)
                        expected_event_count = dist.get_world_size() - 1
                        self.assertEqual(event_count, expected_event_count)
                        # Event order is not deterministic, so simply assert their shape
                        # is found in the following list.
                        expected_shapes = [
                            [[rank + 1] * 3] for rank in range(dist.get_world_size())
                        ]
                        for event in events:
                            self.assertTrue(event.is_async)
                            self.assertTrue(event.input_shapes in expected_shapes)

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl send/recv tested by test_send_recv_nccl"
        )
        def test_send_recv(self):
            self._test_send_recv(profiler_ctx=None)

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl"
        )
        def test_send_recv_autograd_profiler(self):
            autograd_profiler_ctx = _create_autograd_profiler()
            self._test_send_recv(profiler_ctx=autograd_profiler_ctx)

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl"
        )
        @skip_but_pass_in_sandcastle_if(IS_FBCODE, "Kineto in fbcode causes hang")
        @skip_but_pass_in_sandcastle_if(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124",
        )
        def test_send_recv_torch_profiler(self):
            torch_profiler_ctx = _create_torch_profiler()
            return self._test_send_recv(profiler_ctx=torch_profiler_ctx)

        # SEND RECV ANY SOURCE
        def _test_send_recv_any_source(self, profiler_ctx):
            rank = dist.get_rank()
            send_recv_size = 10
            tensor = _build_tensor(send_recv_size, value=rank)
            recv_ranks = []
            irecv_ranks = []

            ctx = profiler_ctx if profiler_ctx is not None else nullcontext()
            with ctx as prof:
                for dst in range(dist.get_world_size()):
                    if dst == rank:
                        # Recv mode
                        for dst in range(dist.get_world_size()):
                            if dst == rank:
                                continue

                            for recv in ["recv", "irecv"]:
                                output_tensor = _build_tensor(send_recv_size, value=-1)

                                if recv == "recv":
                                    sender = dist.recv(output_tensor)
                                    recv_ranks.append(sender)
                                elif recv == "irecv":
                                    work = dist.irecv(output_tensor)
                                    work.wait()
                                    sender = work._source_rank()
                                    irecv_ranks.append(sender)

                                # Assert the scalar value "sender" that should be
                                # equal to the rank of the sender is equal to all
                                # values in the received tensor.
                                self.assertTrue(output_tensor.eq(sender).all())
                    else:
                        # Send mode
                        dist.send(tensor, dst)  # recv
                        dist.send(tensor, dst)  # irecv

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    for event_name in [f"{backend}:send", f"{backend}:recvAnySource"]:
                        events = get_profiling_event(event_name, prof)
                        # Each rank sends/recvs from other rank twice.
                        self.assertEqual(
                            sum(event.count for event in events),
                            2 * (dist.get_world_size() - 1),
                        )
                        for event in events:
                            self.assertTrue(event.is_async)
                            self.assertEqual(event.input_shapes, [[send_recv_size] * 3])

                # Each rank would have 2 * (world_size - 1) sends, verify that
                # globally we receive the same amount on the other end.
                recv_ranks_tensor = torch.cat(
                    (torch.tensor(recv_ranks), torch.tensor(irecv_ranks)), 0
                )
                global_recv_ranks = [
                    torch.empty_like(recv_ranks_tensor)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(global_recv_ranks, recv_ranks_tensor)
                global_recv_ranks_list = []
                for tensor in global_recv_ranks:
                    global_recv_ranks_list += tensor.tolist()

                from itertools import groupby

                global_recv_ranks_list.sort()
                frequency = [
                    len(list(group)) for key, group in groupby(global_recv_ranks_list)
                ]
                self.assertEqual(dist.get_world_size(), len(frequency))
                self.assertEqual(
                    [2 * (dist.get_world_size() - 1)] * dist.get_world_size(), frequency
                )
                self._barrier()

        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["sendrecv anysource"],
            f"{BACKEND} does not support send/recv from any source",
        )
        def test_send_recv_any_source(self):
            self._test_send_recv_any_source(profiler_ctx=None)

        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["sendrecv anysource"],
            f"{BACKEND} does not support send/recv from any source",
        )
        def test_send_recv_any_source_autograd_profiler(self):
            autograd_profiler_ctx = _create_autograd_profiler()
            self._test_send_recv_any_source(profiler_ctx=autograd_profiler_ctx)

        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["sendrecv anysource"],
            f"{BACKEND} does not support send/recv from any source",
        )
        @skip_but_pass_in_sandcastle_if(IS_FBCODE, "Kineto in fbcode code causes hang")
        @skip_but_pass_in_sandcastle_if(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124",
        )
        def test_send_recv_any_source_torch_profiler(self):
            torch_profiler_ctx = _create_torch_profiler()
            return self._test_send_recv_any_source(profiler_ctx=torch_profiler_ctx)

        # SEND RECV WITH TAG
        def _test_send_recv_with_tag(self, profiler_ctx):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            send_recv_size = 10
            tensor = _build_tensor(send_recv_size, value=rank)
            ctx = profiler_ctx if profiler_ctx is not None else nullcontext()
            with ctx as prof:
                for dst in range(world_size):
                    if dst == rank:
                        # Recv mode
                        for src in range(world_size):
                            if src == rank:
                                continue
                            output_tensor = _build_tensor(send_recv_size, value=-1)
                            dist.recv(output_tensor, src, tag=src)
                            self.assertTrue(output_tensor.eq(src).all())
                    else:
                        # Send mode
                        dist.send(tensor, dst, tag=rank)

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    for event_name in [f"{backend}:send", f"{backend}:recv"]:
                        events = get_profiling_event(event_name, prof)
                        # Each rank sends/recvs from all other ranks
                        event_count = sum(e.count for e in events)
                        expected_event_count = dist.get_world_size() - 1
                        self.assertEqual(event_count, expected_event_count)
                        for event in events:
                            self.assertTrue(event.is_async)
                            self.assertEqual(event.name, event_name)
                            self.assertEqual(event.input_shapes, [[send_recv_size] * 3])

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl"
        )
        def test_send_recv_with_tag(self):
            self._test_send_recv_with_tag(profiler_ctx=None)

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl"
        )
        def test_send_recv_with_tag_autograd_profiler(self):
            autograd_profiler_ctx = _create_autograd_profiler()
            return self._test_send_recv_with_tag(profiler_ctx=autograd_profiler_ctx)

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "NCCL send/recv tested by test_send_recv_nccl"
        )
        @skip_but_pass_in_sandcastle_if(IS_FBCODE, "Kineto in fbcode code causes hang")
        @skip_but_pass_in_sandcastle_if(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124",
        )
        def test_send_recv_with_tag_torch_profiler(self):
            torch_profiler_ctx = _create_torch_profiler()
            return self._test_send_recv_with_tag(profiler_ctx=torch_profiler_ctx)

        # ISEND
        def _test_isend(self, profiler_ctx):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            ctx = profiler_ctx if profiler_ctx is not None else nullcontext()
            with ctx as prof:
                if rank == 0:
                    requests = [
                        dist.isend(_build_tensor(dest, 10), dest)
                        for dest in range(1, world_size)
                    ]
                    for request in requests:
                        request.wait()
                        self.assertTrue(request.is_completed())
                else:
                    tensor = _build_tensor(rank, -1)
                    dist.recv(tensor, 0)
                    self.assertEqual(tensor, _build_tensor(rank, 10))

                self._barrier()

            if profiler_ctx is not None:
                backend = dist.get_backend()
                if backend in SEND_RECV_PROFILING_SUPPORTED_BACKENDS:
                    expected_event_name = (
                        f"{backend}:send" if rank == 0 else f"{backend}:recv"
                    )
                    events = get_profiling_event(expected_event_name, prof)
                    event_count = sum(e.count for e in events)
                    expected_count = dist.get_world_size() - 1 if rank == 0 else 1
                    self.assertEqual(expected_count, event_count)
                    # Event ordering is not guaranteed, so simply ensure the shapes are
                    # found in the following map.
                    expected_shapes = {
                        r: [[r] * 3] for r in range(1, dist.get_world_size())
                    }
                    for event in events:
                        self.assertTrue(event.is_async)
                        self.assertEqual(event.name, expected_event_name)
                        if rank == 0:
                            self.assertTrue(
                                event.input_shapes in expected_shapes.values()
                            )
                        else:
                            self.assertEqual(event.input_shapes, expected_shapes[rank])

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support isend"
        )
        def test_isend(self):
            self._test_isend(profiler_ctx=None)

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support isend"
        )
        def test_isend_autograd_profiler(self):
            autograd_profiler_ctx = _create_autograd_profiler()
            self._test_isend(profiler_ctx=autograd_profiler_ctx)

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support isend"
        )
        @skip_but_pass_in_sandcastle_if(IS_FBCODE, "Kineto in fbcode code causes hang")
        @skip_but_pass_in_sandcastle_if(
            IS_MACOS or IS_WINDOWS,
            "torch.profiler not enabled for mac/windows: https://github.com/pytorch/pytorch/pull/56124",
        )
        def test_isend_torch_profiler(self):
            torch_profiler_ctx = _create_torch_profiler()
            self._test_isend(profiler_ctx=torch_profiler_ctx)

        # IRECV
        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support irecv"
        )
        def test_irecv(self):
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            if rank == 0:
                expected_tensors = [
                    _build_tensor(src, -1) for src in range(1, world_size)
                ]
                requests = [
                    dist.irecv(expected_tensors[src - 1], src)
                    for src in range(1, world_size)
                ]

                for src in range(1, world_size):
                    requests[src - 1].wait()
                    self.assertTrue(requests[src - 1].is_completed())
                    self.assertEqual(expected_tensors[src - 1], _build_tensor(src, 10))
            else:
                tensor = _build_tensor(rank, 10)
                dist.send(tensor, 0)

            self._barrier()

        # BROADCAST
        def _test_broadcast_helper(
            self,
            group,
            group_id,
            rank,
            cuda=False,
            rank_to_GPU=None,
            with_options=False,
        ):
            for dtype, value, requires_cuda in [
                (torch.float, -1e-10, False),
                (torch.double, -1e-100, False),
                (torch.half, -0.1, True),
                (torch.int8, -2, False),
                (torch.uint8, 129, False),
                (torch.int, -1e5, False),
                (torch.long, -1e15, False),
            ]:
                if requires_cuda and not cuda:
                    continue
                for src in group:
                    expected_tensor = _build_tensor(src + 1, value, dtype)
                    if cuda:
                        expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
                    if rank == src:
                        if with_options:
                            opts = dist.BroadcastOptions()
                            opts.rootTensor = 0
                            opts.rootRank = src
                            self.call_dist_op(
                                ":broadcast",
                                True,
                                group_id.broadcast,
                                [expected_tensor],
                                opts,
                            )
                        else:
                            self.call_dist_op(
                                ":broadcast",
                                False,
                                dist.broadcast,
                                expected_tensor,
                                src,
                                group_id,
                            )
                    else:
                        tensor = _build_tensor(src + 1, -1, dtype)
                        if cuda:
                            tensor = tensor.cuda(rank_to_GPU[rank][0])
                        if with_options:
                            opts = dist.BroadcastOptions()
                            opts.rootTensor = 0
                            opts.rootRank = src
                            self.call_dist_op(
                                ":broadcast", True, group_id.broadcast, [tensor], opts
                            )
                        else:
                            self.call_dist_op(
                                ":broadcast",
                                False,
                                dist.broadcast,
                                tensor,
                                src,
                                group_id,
                            )
                        self.assertEqual(tensor.size(), expected_tensor.size())
                        self.assertEqual(
                            tensor.ne(expected_tensor).max(), torch.tensor(False)
                        )

            self._barrier()

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        def test_broadcast(self):
            group, group_id, rank = self._init_global_test()
            self._test_broadcast_helper(group, group_id, rank)

        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo" and BACKEND != "nccl",
            "Only Gloo and Nccl backend supports CUDA allReduce",
        )
        @skip_if_no_gpu
        def test_broadcast_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            self._test_broadcast_helper(group, group_id, rank, True, rank_to_GPU)

        @skip_if_small_worldsize
        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        def test_broadcast_group(self):
            group, group_id, rank = self._init_group_test()
            self._test_broadcast_helper(group, group_id, rank)

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        def test_broadcast_full_group(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_broadcast_helper(group, group_id, rank)

        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl",
            "Only NCCL backend supports high priority stream",
        )
        @skip_if_no_gpu
        def test_nccl_high_priority_stream(self):
            group, _, rank = self._init_global_test()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)

            new_port = str(MASTER_PORT + 1)
            os.environ["MASTER_PORT"] = new_port
            gen_iterator = dist.rendezvous("env://", rank, dist.get_world_size())
            store, rank, size = next(gen_iterator)
            store = dist.PrefixStore(new_port, store)

            opts = dist.ProcessGroupNCCL.Options()
            opts.is_high_priority_stream = False
            group_id = dist.ProcessGroupNCCL(store, rank, size, opts)

            self._test_broadcast_helper(group, group_id, rank, True, rank_to_GPU, True)

        # REDUCE
        def _test_reduce_helper(
            self,
            group,
            group_id,
            rank,
            op,
            master_value,
            worker_value,
            expected_value,
            cuda=False,
            rank_to_GPU=None,
        ):
            for src in group:
                tensor = _build_tensor(src + 1).fill_(
                    master_value if rank == src else worker_value
                )
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                self.call_dist_op(
                    ":reduce",
                    False,
                    dist.reduce,
                    tensor,
                    src,
                    op,
                    group_id,
                    tensor_shapes=[tensor.shape],
                )
                if rank == src:
                    self.assertEqual(tensor, _build_tensor(src + 1, expected_value))

            self._barrier()

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_sum(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only Nccl supports CUDA reduce"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        @skip_if_no_gpu
        def test_reduce_sum_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + 10 * (len(group) - 1),
                True,
                rank_to_GPU,
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_product(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce(operator.mul, [10] * (len(group) - 1), 2),
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_min(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_max(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        @skip_if_small_worldsize
        def test_reduce_group_sum(self):
            group, group_id, rank = self._init_group_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        @skip_if_small_worldsize
        def test_reduce_group_product(self):
            group, group_id, rank = self._init_group_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce(operator.mul, [10] * (len(group) - 1), 2),
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        @skip_if_small_worldsize
        def test_reduce_group_min(self):
            group, group_id, rank = self._init_group_test()
            self._test_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        @skip_if_small_worldsize
        def test_reduce_group_max(self):
            group, group_id, rank = self._init_group_test()
            self._test_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_full_group_sum(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_full_group_product(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_reduce_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.PRODUCT,
                2,
                10,
                reduce(operator.mul, [10] * (len(group) - 1), 2),
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_full_group_min(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MIN, 1010, 1, 1
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_full_group_max(self):
            group, group_id, rank = self._init_full_group_test()
            self._test_reduce_helper(
                group, group_id, rank, dist.ReduceOp.MAX, -1, 10, 10
            )

        # REDUCE TWICE
        def _test_reduce_twice_helper(
            self,
            group,
            group_id,
            rank,
            op,
            master_value,
            worker_value,
            expected_value,
            cuda=False,
            rank_to_GPU=None,
        ):
            for src in group:
                tensors = [
                    _build_tensor(src + 1).fill_(
                        master_value if rank == src else worker_value
                    )
                    for i in range(2)
                ]
                if cuda:
                    for i in range(2):
                        tensors[i] = tensors[i].cuda(rank_to_GPU[rank][0])
                self.call_dist_op(
                    ":reduce",
                    False,
                    dist.reduce,
                    tensors[0],
                    src,
                    op,
                    group_id,
                    secondary_op_call=lambda: dist.reduce(
                        tensors[1], src, op, group_id
                    ),
                    tensor_shapes=[tensors[0].shape],
                )
                if rank == src:
                    for tensor in tensors:
                        self.assertEqual(tensor, _build_tensor(src + 1, expected_value))

            self._barrier()

        @skip_but_pass_in_sandcastle_if(
            BACKEND == "nccl", "Nccl does not support CPU tensors"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        def test_reduce_sum_twice(self):
            group, group_id, rank = self._init_global_test()
            self._test_reduce_twice_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + (10 * (len(group) - 1)),
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only Nccl supports CUDA reduce"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        @skip_if_no_gpu
        def test_reduce_sum_cuda_twice(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]
            torch.cuda.set_device(device_id)
            self._test_reduce_twice_helper(
                group,
                group_id,
                rank,
                dist.ReduceOp.SUM,
                2,
                10,
                2 + 10 * (len(group) - 1),
                True,
                rank_to_GPU,
            )

        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only Nccl supports reduce_scatter_v"
        )
        @skip_but_pass_in_sandcastle_if(
            BACKEND in DistTestCases.skip_collective["reduce"],
            f"{BACKEND} does not support reduce",
        )
        @skip_if_no_gpu
        def test_reduce_scatter_v_cuda(self):
            self._barrier()
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
            device_id = rank_to_GPU[rank][0]

            input_split_sizes = [src + 1 for src in group]
            start_len = sum(input_split_sizes[:rank])
            end_len = start_len + input_split_sizes[rank]
            sum_len = sum(input_split_sizes)
            master_value = 2
            worker_value = 10

            for async_val in [True, False]:
                tensor = _build_tensor(sum_len, worker_value, device_id=device_id)
                tensor[start_len:end_len].fill_(master_value)
                out_tensor = (
                    torch.empty(
                        input_split_sizes[rank], sum_len, sum_len, dtype=torch.float
                    )
                    .fill_(-1)
                    .cuda(device_id)
                )

                req = dist.reduce_scatter(
                    out_tensor,
                    list(torch.split(tensor, input_split_sizes)),
                    dist.ReduceOp.SUM,
                    group_id,
                    async_val,
                )
                if async_val:
                    req.wait()

                expected_value = 2 + (10 * (len(group) - 1))
                expected_tensor = torch.empty(
                    input_split_sizes[rank], sum_len, sum_len, dtype=torch.float
                )
                expected_tensor = expected_tensor.fill_(expected_value).cuda(device_id)

                self.assertEqual(out_tensor, expected_tensor)
            self._barrier()

        # Test reduce_scatter_tensor accepting single tensor as input
        def _reduce_scatter_tensor_helper(
            self, tensor_out, tensor_in, group_id, rank, cuda=True, rank_to_GPU=None
        ):
            if cuda:
                tensor_in = tensor_in.cuda(rank_to_GPU[rank][0])
                tensor_out = tensor_out.cuda(rank_to_GPU[rank][0])
            tensor_shapes = [tensor_out.shape]
            self.call_dist_op(
                ":reduce_scatter_tensor",
                False,
                dist.reduce_scatter_tensor,
                tensor_out,
                tensor_in,
                dist.ReduceOp.SUM,
                group_id,
                False,
                expect_event=False,
                tensor_shapes=tensor_shapes,
            )
            return tensor_out

        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only Nccl supports CUDA reduce_scatter_tensor"
        )
        @skip_if_no_gpu
        def test_reduce_scatter_tensor_cuda(self):
            group, group_id, rank = self._init_global_test()
            rank_to_GPU = init_multigpu_helper(dist.ge

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 56 class(es): NetWithBuffers, Foo, TestNamedTupleInput_1, DDPUnevenTestInput, _FC2, Net, LargeNet, Task, BatchNormNet, UnusedParamTwoLinLayerNet, DictOutputModule, TwoLinLayerNet, EmbeddingNetDifferentParams, ControlFlowToyModel, Barrier, TestDistBackend, DistributedTest, _DistTestBase, ToyModel, Model

### Functions
This file defines 512 function(s): __init__, forward, __init__, __eq__, eq, create_collectives_object_test_list, get_profiling_event, get_profiler_nccl_meta, __init__, forward, __init__, forward, __init__, forward, __init__, forward, __init__, forward, __init__, forward, __init__, forward, __init__, forward, __init__, forward, __init__, forward, get_timeout, require_backend_is_available


## Key Components

The file contains 29210 words across 10423 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 438301 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
