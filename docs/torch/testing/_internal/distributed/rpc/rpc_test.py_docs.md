# Documentation: rpc_test.py

## File Metadata
- **Path**: `torch/testing/_internal/distributed/rpc/rpc_test.py`
- **Size**: 225880 bytes
- **Lines**: 6312
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-defs

import concurrent.futures
import contextlib
import json
import operator
import os
import sys
import threading
import time
from collections import namedtuple
from functools import partial
from threading import Event, Lock
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.autograd.profiler_legacy import profile as _profile
from torch.distributed.rpc import (
    _get_debug_info,
    _rref_context_get_debug_info,
    RRef,
    WorkerInfo,
)
from torch.distributed.rpc.api import _thread_local_var, _use_rpc_pickler, _wait_all
from torch.distributed.rpc.internal import (
    _build_rpc_profiling_key,
    _internal_rpc_pickler,
    PythonUDF,
    RPCExecMode,
)
from torch.futures import Future
from torch.testing._internal.common_distributed import (
    captured_output,
    skip_if_lt_x_gpu,
    tp_transports,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    IS_MACOS,
    load_tests,
    skip_but_pass_in_sandcastle_if,
    TemporaryFileName,
)
from torch.testing._internal.dist_utils import (
    dist_init,
    get_function_event,
    initialize_pg,
    wait_until_node_failure,
    wait_until_owners_and_forks_on_rank,
    wait_until_pending_futures_and_users_flushed,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


def foo_add():
    return torch.add(torch.ones(1), torch.ones(1))


def udf_with_torch_ops(device=-1, use_record_function=False):
    device_ctx = contextlib.nullcontext() if device == -1 else torch.cuda.device(device)
    record_function_ctx = (
        torch.autograd.profiler.record_function("##forward##")
        if use_record_function
        else contextlib.nullcontext()
    )
    with device_ctx, record_function_ctx:
        t1, t2 = torch.ones(1), torch.ones(1)
        t = torch.add(t1, t2)
        t = torch.mul(t, t)
        t = t.relu()
        t = t.sigmoid()


# Events (operator invocations) that are expected to be ran as part of the above
# function.
EXPECTED_REMOTE_EVENTS = [
    "aten::ones",
    "aten::ones",
    "aten::add",
    "aten::mul",
    "aten::relu",
    "aten::clamp_min",
    "aten::sigmoid",
]

# Remote operations are prefixed with the following string for RPC profiling.
REMOTE_OP_STR = "#remote_op: "


VALUE_FUTURE = concurrent.futures.Future()
DONE_FUTURE = concurrent.futures.Future()

FIFTY_MIL_CYCLES = 50000000

_rpc_barrier_count = 0


def _increment_count():
    global _rpc_barrier_count
    _rpc_barrier_count += 1


def _reset_count():
    global _rpc_barrier_count
    _rpc_barrier_count = 0


class StubRpcAgent:
    def __init__(self, world_size):
        self.world_size = world_size

    def get_worker_infos(self):
        return {
            WorkerInfo(name=worker_name(rank), id=rank)
            for rank in range(self.world_size)
        }


def _stub_construct_rpc_backend_options_handler(**kwargs):
    return mock.Mock()  # RpcBackendOptions.


def _stub_init_rpc_backend_handler(store, name, rank, world_size, rpc_backend_options):
    return StubRpcAgent(world_size=world_size)


def set_value(value):
    VALUE_FUTURE.set_result(value)


def wait_for_value_future():
    return VALUE_FUTURE.result()


def set_and_check_done(value):
    VALUE_FUTURE.set_result(value)
    return DONE_FUTURE.result()


# it is used to test python user defined function over rpc
# classes and functions are used to test python user defined class and
# methods over rpc
TensorClass = namedtuple("TensorClass", ["tensors"])


class MyPickleClass:
    def __init__(self) -> None:
        self.t = None

    def __getstate__(self):
        (pickled_python_udf, tensors) = _internal_rpc_pickler.serialize(
            PythonUDF(my_tensor_function, (torch.ones(2, 2), torch.ones(2, 2)), None)
        )
        return (pickled_python_udf, tensors)

    def __setstate__(self, obj):
        python_udf = _internal_rpc_pickler.deserialize(obj[0], obj[1])
        result = python_udf.func(python_udf.args[0], python_udf.args[1])
        self.t = result

    def set(self, val):
        self.t = val


class SlowPickleClass:
    def __init__(self, t):
        self.t = t

    def __getstate__(self):
        time.sleep(self.t)
        return (self.t,)

    def __setstate__(self, obj):
        self.t = obj[0]
        time.sleep(self.t)


class MyClass:
    def __init__(self, a, delay=False):
        self.a = a
        # delay initialization to simulate errors if specified
        if delay:
            time.sleep(2)

    def my_instance_method(self, b):
        return self.a + b

    @classmethod
    def my_class_method(cls, d, e):
        return d + e

    @staticmethod
    def my_static_method(f):
        return f > 10

    def increment_value(self, increment):
        self.a += increment

    def get_value(self):
        return self.a

    def my_slow_method(self, my_tensor_arg):
        time.sleep(5)
        return torch.add(self.a, my_tensor_arg)


def _call_method_on_rref(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def get_rref_list(values):
    return [RRef(MyClass(a)) for a in values]


def add_rref_to_value(rref, value):
    return rref.to_here() + value


def run_nested_pickle(pickle_cls_instance, tensor):
    return pickle_cls_instance.t + tensor


def build_sparse_tensor(coalesce=False):
    i = [[0, 1, 1], [2, 0, 2]]
    v = [3, 4, 5]
    tensor = torch.sparse_coo_tensor(i, v, (2, 3))
    if coalesce:
        tensor = tensor.coalesce()
    return tensor


def build_complex_tensors():
    a = torch.ones(3, 3)
    b = [a, a]
    c = [b, b]
    d = [a, b]
    e = {a: d}
    return [a, b, c, d, e]


def non_cont_test(t_view, t_cont):
    if t_view.is_contiguous():
        raise Exception("t_view is contiguous!")  # noqa: TRY002
    if not t_cont.is_contiguous():
        raise Exception("t_cont is not contiguous!")  # noqa: TRY002
    if not torch.equal(t_view, t_cont):
        raise Exception("t_view is not equal to t_cont!")  # noqa: TRY002
    return t_view


def my_function(a, b, c):
    return a + b + c


def my_tensor_function(a, b):
    return a + b


def my_container_sum(a):
    result = a[0]
    for tensor in a[1:]:
        result += tensor
    return result


def my_sleep_func(seconds=1):
    time.sleep(seconds)
    return torch.mul(torch.tensor(1), torch.tensor(1))


def my_complex_tensor_function(list_input, tensor_class_input, dict_input):
    res = list_input[0]
    for t in list_input:
        res += t
    for v in dict_input.values():
        res += v
    complex_tensors = tensor_class_input.tensors
    return (res, complex_tensors[0], complex_tensors[1], complex_tensors[2])


def my_rref_function(rref_a, rref_b):
    return rref_a.to_here() + rref_b.to_here()


def delayed_add(a, b, seconds=0.05):
    time.sleep(seconds)
    return a + b


def identity(a):
    return a


def no_result():
    print("do nothing")


def raise_or_inc(value):
    if value.numel() == 2:
        raise ValueError("Expected error")
    return value + 1


def nested_rpc(dst):
    return rpc.rpc_sync(dst, torch.add, args=(torch.ones(2, 2), 1))


def nested_rpc_sparse(dst):
    return rpc.rpc_sync(
        dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor())
    )


def multi_layer_nested_async_rpc(dst, world_size, ttl):
    # this method returns immediately without blocking the callee, but will
    # generate additional requests.
    if ttl > 0:
        current_dst = worker_name(dst)
        next_dst = (dst + 1) % world_size
        rpc.rpc_async(
            current_dst,
            multi_layer_nested_async_rpc,
            args=(next_dst, world_size, ttl - 1),
        )
        return 0


def nested_rref(dst):
    return (
        rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1)),
        rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 2)),
    )


def nested_rref_sparse(dst):
    return (
        rpc.remote(dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor())),
        rpc.remote(dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor())),
    )


def nested_remote(dst):
    rref = rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 3))
    return rref.to_here()


def nested_remote_sparse(dst):
    rref = rpc.remote(
        dst, torch.add, args=(build_sparse_tensor(), build_sparse_tensor())
    )
    return rref.to_here()


def rref_forward_chain(dst, world_size, rref, ttl):
    if ttl > 0:
        current_dst = worker_name(dst)
        next_dst = (dst + 1) % world_size
        ret_rref = rpc.remote(
            current_dst, rref_forward_chain, args=(next_dst, world_size, rref, ttl - 1)
        )
        return [ret_rref]
    else:
        return rref.to_here()


def rpc_return_rref(dst):
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))


def light_rpc():
    return 0


def heavy_rpc(tensor):
    for i in range(1, 100):
        tensor *= i
        tensor /= i + 1
    return 0


def heavy_rpc_sparse(tensor):
    for i in range(1, 100):
        tensor *= i
        tensor = tensor / (i + 1)
    return 0


@torch.jit.script
def heavy_rpc_torchscript(tensor):
    for i in range(1, 100):
        tensor *= i
        tensor /= i + 1
    return 0


@torch.jit.script
def my_script_func(tensor):
    return torch.add(tensor, tensor)


expected_err = "Expected error"


# Note that it needs to inherit from Exception, not BaseException. See comment
# in rpc/internal.py
class CustomException(Exception):
    def __init__(self, bool, msg):
        self.bool = bool
        super().__init__(msg)


def raise_func():
    raise ValueError(expected_err)


def custom_raise_func():
    raise CustomException(True, "foo")


@torch.jit.script
def raise_func_script(expected_err: str) -> torch.Tensor:
    raise ValueError(expected_err)


expected_err_escape = (
    "\nFirst line of error \n next line of error \n last line of error"
)


def raise_func_escape():
    raise ValueError(expected_err_escape)


global_rref = None


def set_global_rref(rref):
    global global_rref
    global_rref = rref


def clear_global_rref():
    global global_rref
    global_rref = None


def check_rref_confirmed(rref):
    return rref.confirmed_by_owner()


def get_rref_debug_info():
    return _rref_context_get_debug_info()


def add_use_future_cb(to, x, y, z):
    out = concurrent.futures.Future()

    def callback(fut):
        out.set_result(fut.wait() + z)

    fut = rpc.rpc_async(to, torch.add, args=(x, y))
    fut.then(callback)
    return out.result()


def get_events_from_profile(profile_rref):
    return profile_rref.local_value().process_global_function_events


def add_use_future_set_result(to, x, y, z):
    out = torch.futures.Future()
    fut = rpc.rpc_async(to, torch.add, args=(x, y))
    fut.then(lambda fut: out.set_result(fut.wait() + z))
    return out.wait()


def add_use_future_nested_cb(to, x, y, z):
    out = torch.futures.Future()

    def callback(fut1):
        fut2 = rpc.rpc_async(to, torch.add, args=(fut1.wait(), z))
        fut2.then(lambda fut2: out.set_result(fut2.wait()))

    fut1 = rpc.rpc_async(to, torch.add, args=(x, y))
    fut1.then(callback)
    return out.wait()


def fail_on_fut(fut):
    pass


@rpc.functions.async_execution
def async_raise_func():
    raise RuntimeError("Expected error")


@rpc.functions.async_execution
def async_wrong_type():
    return torch.zeros(2, 2)


@rpc.functions.async_execution
def async_add(to, x, y):
    return rpc.rpc_async(to, torch.add, args=(x, y))


def slow_add(x, y, device="cpu"):
    time.sleep(1)
    x = x.to(device)
    y = y.to(device)
    return torch.add(x, y).cpu()


@rpc.functions.async_execution
def slow_async_add(to, x, y, device="cpu"):
    return rpc.rpc_async(to, slow_add, args=(x, y, device))


@rpc.functions.async_execution
def async_add_with_future_ctor(to, x, y, z):
    fut = torch.futures.Future()
    rpc.rpc_async(to, torch.add, args=(x, y)).then(
        lambda fut1: fut.set_result(fut1.wait() + z)
    )
    return fut


@rpc.functions.async_execution
def async_add_chained(to, x, y, z):
    return rpc.rpc_async(to, torch.add, args=(x, y)).then(lambda fut: fut.wait() + z)


@rpc.functions.async_execution
def async_add_chained_multi(to, x, num, step):
    fut = rpc.rpc_async(to, torch.add, args=(x, 0))
    for _ in range(num):
        fut = fut.then(lambda fut: fut.wait() + step)
    return fut


@rpc.functions.async_execution
def async_add_nested(to, x, y, z):
    return rpc.rpc_async(to, async_add, args=(to, x, y)).then(
        lambda fut: fut.wait() + z
    )


@rpc.functions.async_execution
def async_add_multi_fanout(to, x, num, step):
    futs = []
    for i in range(num):
        if i == 0:
            futs.append(rpc.rpc_async(to, torch.add, args=(x, step)))
        else:
            futs.append(rpc.rpc_async(to, torch.add, args=(0, step)))

    # TODO: use torch.futures.collect_all
    lock = Lock()
    state = {"cnt": 0, "ret": torch.zeros_like(x)}
    ret_future = torch.futures.Future()

    def inc_and_set(fut):
        with lock:
            state["cnt"] += 1
            state["ret"] += fut.wait()
            if state["cnt"] >= len(futs):
                ret_future.set_result(state["ret"])

    for fut in futs:
        fut.then(inc_and_set)

    return ret_future


@rpc.functions.async_execution
def async_cuda_sleep_and_set_to_one(t):
    device = t.device
    original_stream = torch.cuda.current_stream(device)
    new_stream = torch.cuda.Stream(device)
    new_stream.wait_stream(original_stream)
    with torch.cuda.stream(new_stream):
        torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
        t.fill_(1)
        fut = Future(devices=[device])
        fut.set_result(t)
        return fut


@rpc.functions.async_execution
def async_cuda_nested_add(to, x, y, z):
    def cb(fut):
        torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
        return fut.value() + z

    return rpc.rpc_async(to, torch.add, args=(x, y)).then(cb)


# A custom Python class that contains a tensor, needed to see if we correctly
# use the Python pickler to extract tensors from non-IValue-convertible types.
class TensorWrapper:
    __slots__ = ("tensor", "lock", "event", "thread")

    def __init__(self, t):
        self.tensor = t
        # Add one non-picklable field, to ensure it's ignored/skipped.
        self.lock = Lock()
        self.event = torch.cuda.Event(enable_timing=True)
        self.thread = threading.Thread()
        self.thread.start()

    def increase(self, v):
        with self.lock:
            self.tensor += v

    def sum(self):
        with self.lock:
            self.event.record()
            return self.tensor.sum()


class AsyncExecutionClass:
    @staticmethod
    @rpc.functions.async_execution
    def static_async_add(to, x, y, z):
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
        )

    @classmethod
    @rpc.functions.async_execution
    def class_async_add(cls, to, x, y, z):
        ret_fut = torch.futures.Future()
        rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: ret_fut.set_result(fut.wait() + z)
        )
        return ret_fut

    @rpc.functions.async_execution
    def bound_async_add(self, to, x, y, z):
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
        )


def return_future():
    return torch.futures.Future()


class FooBackendOptions(rpc.RpcBackendOptions):
    def __init__(self, init_method):
        # Must call the __init__ of the superclass (and do so directly,
        # without using super()) because... pybind.
        rpc.RpcBackendOptions.__init__(self)
        self.init_method = init_method


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127


class MyEmbeddingBagModel(torch.nn.Module):
    def __init__(self, sparse):
        super().__init__()
        self.eb = torch.nn.EmbeddingBag(10, 10, sparse=sparse)

    def forward(self, x):
        return self.eb(x)


class MyParameterServer:
    def __init__(self, trainers):
        self.lock = Lock()
        self.trainers = trainers
        self.iteration = 0
        self.updates = 0
        self.futures = []
        self.total = None
        self.gradient = None

    @staticmethod
    def get_gradient(rref):
        return rref.local_value().gradient

    @staticmethod
    @rpc.functions.async_execution
    def average(rref, riteration, tensor):
        self = rref.local_value()
        fut = torch.futures.Future()
        with self.lock:
            if riteration > self.iteration:
                self.iteration = riteration
                self.updates = 0
                self.futures.clear()
            self.futures.append(fut)
            if self.total is None:
                self.total = tensor
            else:
                self.total += tensor
            self.updates += 1
            if self.trainers == self.updates:
                self.gradient = self.total / float(self.trainers)
                for fut in self.futures:
                    result = self.total / float(self.trainers)
                    fut.set_result(result)
        return fut


class MyConvNetForMNIST(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(4608, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(device)
        self.device = device

    def forward(self, x, is_rref=False):
        x = x.to_here() if is_rref else x
        with torch.cuda.stream(torch.cuda.current_stream(self.device)):
            # intentionally adding delay to current CUDA stream
            torch.cuda._sleep(10 * FIFTY_MIL_CYCLES)
            return self.net(x)

    def __getstate__(self):
        # return an empty dict to avoid inspecting the model contents on the
        # owner
        return {}


class RpcTestCommon:
    def _run_func_in_mode(self, to, fn, mode, args=None, kwargs=None):
        if mode == RPCExecMode.SYNC:
            return rpc.rpc_sync(to, fn, args=args, kwargs=kwargs)
        elif mode == RPCExecMode.ASYNC:
            return rpc.rpc_async(to, fn, args=args, kwargs=kwargs).wait()
        elif mode == RPCExecMode.REMOTE:
            return rpc.remote(to, fn, args=args, kwargs=kwargs).to_here()

    def _self_py_udf_remote(self, worker_info, x, y, z):
        rref = rpc.remote(worker_info, my_function, args=(x, y, z))
        self.assertEqual(rref.to_here(), x + y + z)

    def _self_remote_rref_as_rpc_arg(self, dst, x, y, z):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(x, y, z))
        fut = rpc.rpc_async(dst, add_rref_to_value, args=(rref, x))
        ret = rpc.rpc_sync(dst, add_rref_to_value, args=(rref, x + y))
        self.assertEqual(ret, x + y + z + x + y)
        self.assertEqual(fut.wait(), x + y + z + x)

    def _self_remote_rref_as_remote_arg(self, dst, x, y, z):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(x, y, z))
        ret_rref = rpc.remote(dst, add_rref_to_value, args=(rref, x))
        self.assertEqual(ret_rref.to_here(), x + y + z + x)

    def _world_size_one(self, a, b):
        if self.rank == 0:
            rpc.init_rpc(
                name="me",
                backend=self.rpc_backend,
                rank=0,
                world_size=1,
                rpc_backend_options=self.rpc_backend_options,
            )

            def _rpc_sync(x, y):
                expect = x * 2
                result = rpc.rpc_sync("me", my_tensor_function, args=(x, y))
                self.assertEqual(expect, result)

            def _rpc_async(x, y):
                expect = x * 2
                result = rpc.rpc_async("me", my_tensor_function, args=(x, y)).wait()
                self.assertEqual(expect, result)

            def _remote(x, y):
                expect = x * 2
                result = rpc.remote("me", my_tensor_function, args=(x, y)).to_here()
                self.assertEqual(expect, result)

            _rpc_sync(a, b)
            _rpc_async(a, b)
            _remote(a, b)

            rpc.shutdown()

    def _multi_rpc(self, sparse):
        dst_rank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            if sparse:
                x = build_sparse_tensor() * n
                y = build_sparse_tensor() * n
            else:
                x = torch.ones(2, 2)
                y = torch.ones(2, 2)
            ret = rpc.rpc_sync(
                worker_name(dst_rank),
                torch.add,
                args=(x, y),
            )
            self.assertEqual(ret, x * 2)

    def _run_uneven_workload(self, f, x, num_repeat=30):
        # worker0 drives and waits for worker1 and worker2
        # throughout the test.
        if self.rank == 0:
            self.assertTrue(self.world_size >= 3)

            # Phase 1: Only worker1 has workload.
            dst = "worker1"
            futs = []
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, f, args=(x,))
                futs.append(fut)

            for fut in torch.futures.collect_all(futs).wait():
                self.assertEqual(fut.wait(), 0)

            # Phase 2: Only worker2 has workload.
            # If join is not correctly implemented,
            # worker2 should be closed by now.
            dst = "worker2"
            futs = []
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, f, args=(x,))
                futs.append(fut)

            for val in torch.futures.wait_all(futs):
                self.assertEqual(val, 0)

    def _wait_all_workers(self, f, x):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        self._run_uneven_workload(f, x)

        # worker0 calls this at the end after waiting for RPC responses.
        # worker1/2 calls this immediately and has some works after it.
        # worker3 calls this immediately and has no more work.
        rpc.api._wait_all_workers()

        # Wait before proceeding to shutdown to ensure worker0 RPCs make
        # it through to other workers.
        dist.barrier()
        rpc.shutdown(graceful=False)

    def _wait_all_workers_twice(self, f, x):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        self._run_uneven_workload(f, x)

        # worker0 calls this at the end after waiting for RPC responses.
        # worker1/2 calls this immediately and has some works after it.
        # worker3 calls this immediately and has no more work.
        rpc.api._wait_all_workers()
        rpc.api._wait_all_workers()

        # Wait before proceeding to shutdown to ensure worker0 RPCs make
        # it through to other workers.
        dist.barrier()
        rpc.shutdown(graceful=False)

    def _nested_rpc(self, f, expected):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            f,
            args=(worker_name(self.rank),),
        )
        self.assertEqual(ret, expected)

    def _stress_test_rpc(self, f, repeat=1000, args=()):
        n = self.rank + 1
        dst_rank = n % self.world_size
        futs = []
        tik = time.time()
        for _ in range(repeat):
            fut = rpc.rpc_async(worker_name(dst_rank), f, args=args)
            futs.append(fut)

        for val in torch.futures.wait_all(futs):
            self.assertEqual(val, 0)
        tok = time.time()
        print(
            f"Rank {self.rank} finished testing {repeat} times in {tok - tik} seconds."
        )

    def _builtin_remote_ret(self, x, y, expected):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            worker_name(dst_rank),
            torch.add,
            args=(x, y),
        )
        self.assertEqual(rref.to_here(), expected)

    def _builtin_remote_self(self, x, y, expected):
        rref = rpc.remote(
            worker_name(self.rank),
            torch.add,
            args=(x, y),
        )
        self.assertEqual(rref.local_value(), expected)

    def _test_multi_remote_call(
        self, fn, sparse, args_fn=lambda x, y: (), kwargs_fn=lambda x, y: {}
    ):
        m = 10
        n = self.rank + 1
        dst_rank = n % self.world_size
        rrefs = []
        expected = []
        for i in range(m):
            n = n + i
            rrefs.append(
                rpc.remote(
                    worker_name(dst_rank),
                    fn,
                    args=args_fn(n, sparse),
                    kwargs=kwargs_fn(n, sparse),
                )
            )
            expected.append(fn(*args_fn(n, sparse), **kwargs_fn(n, sparse)))

        for i in range(m):
            self.assertEqual(rrefs[i].to_here(), expected[i])

    def _py_rref_args(self, a, b, x, y, expected):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(worker_name(dst_rank), torch.add, args=(a, b))
        rref_b = rpc.remote(worker_name(dst_rank), torch.add, args=(x, y))
        rref_c = rpc.remote(
            worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), expected)

    def _py_rref_args_user_share(self, a, b, c, x, y, z, expected):
        n = self.rank + 1
        owner_rank = n % self.world_size
        user_rank = (n + 1) % self.world_size
        rref_a = rpc.remote(worker_name(owner_rank), my_function, args=(a, b, c))
        rref_b = rpc.remote(worker_name(owner_rank), my_function, args=(x, y, z))
        rref_c = rpc.remote(
            worker_name(user_rank), my_rref_function, args=(rref_a, rref_b)
        )
        self.assertEqual(rref_c.to_here(), expected)

    def _py_rpc_rref_args(self, a, b, c, x, y, z, expected):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(worker_name(dst_rank), my_function, args=(a, b, c))
        rref_b = rpc.remote(worker_name(dst_rank), my_function, args=(x, y, z))

        c = rpc.rpc_sync(worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b))
        self.assertEqual(c, expected)

    def _nested_remote(self, f, expected):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size

        rref = rpc.remote(
            worker_name(dst_rank1),
            f,
            args=(worker_name(dst_rank2),),
        )
        self.assertEqual(rref.to_here(), expected)

    def _nested_rref(self, f, expected1, expected2):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref_of_rrefs = rpc.remote(
            worker_name(dst_rank1),
            f,
            args=(worker_name(dst_rank2),),
        )

        # Say C has 2 OwnerRRefs.
        # B has 2 UserRRefs to those 2 OwnerRRefs, respectively.
        # This call is effectively A asking B to share its 2 UserRRefs.
        rrefs = rref_of_rrefs.to_here()

        self.assertEqual(len(rrefs), 2)
        self.assertEqual(rrefs[0].to_here(), expected1)
        self.assertEqual(rrefs[1].to_here(), expected2)

    def _nested_rref_stress(self, f, expected1, expected2):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        all_rrefs = [
            rpc.remote(
                worker_name(dst_rank1),
                f,
                args=(worker_name(dst_rank2),),
            )
            for _ in range(20)
        ]

        for i in range(20):
            rref_of_rrefs = all_rrefs[i]
            rrefs = rref_of_rrefs.to_here()
            self.assertEqual(len(rrefs), 2)
            self.assertEqual(rrefs[0].to_here(), expected1)
            self.assertEqual(rrefs[1].to_here(), expected2)

    def _trainer_func(self, rref, sparse):
        m = MyEmbeddingBagModel(sparse=sparse)
        loss_fn = nn.MSELoss()
        for i in range(10):
            outputs = m(torch.rand(10, 10).long())
            loss_fn(outputs, torch.rand(10, 10)).backward()
            gradient = next(iter(m.parameters())).grad
            fut = rref.rpc_async().average(rref, i, gradient)
            gradient = fut.wait()
            if gradient.is_sparse:
                gradient = gradient.to_dense().double()
            ps_gradient = rref.rpc_sync().get_gradient(rref)
            if ps_gradient.is_sparse:
                ps_gradient = ps_gradient.to_dense().double()
            self.assertTrue(torch.equal(gradient, ps_gradient))

    def _my_parameter_server(self, sparse):
        ps_rref = RRef(MyParameterServer(self.world_size - 1))
        futures = [
            rpc.rpc_async(
                worker_name((self.rank + index) % self.world_size),
                self._trainer_func,
                args=(ps_rref, sparse),
            )
            for index in range(1, self.world_size)
        ]
        torch.futures.wait_all(futures)

    def _test_cuda_future_extraction(self, wrapper, unwrapper, sparse_tensor):
        # We check proper CUDA stream synchronization by adding to the tensor
        # in one stream to get the expected value, and reading it from another stream.
        future = Future(devices=["cuda:0"])
        with torch.cuda.device("cuda:0"):
            stream = torch.cuda.Stream()
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                if sparse_tensor:
                    tensor = build_sparse_tensor().to("cuda:0")
                    add_tensor = build_sparse_tensor().to("cuda:0")
                    expected_tensor = (tensor + add_tensor).coalesce()
                else:
                    tensor = torch.zeros((100,), device="cuda:0")
                    add_tensor = torch.ones((100,), device="cuda:0")
                    expected_tensor = tensor + add_tensor
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor += add_tensor
                if sparse_tensor:
                    tensor = tensor.coalesce()
                future.set_result(wrapper(tensor))
            with torch.cuda.stream(another_stream):
                tensor = unwrapper(future.wait())
                if sparse_tensor:
                    self.assertTrue(
                        torch.eq(tensor.indices(), expected_tensor.indices())
                        .all()
                        .item()
                    )
                    self.assertTrue(
                        torch.eq(tensor.values(), expected_tensor.values()).all().item()
                    )
                    self.assertEqual(tensor.size(), expected_tensor.size())
                else:
                    self.assertTrue(torch.eq(tensor, expected_tensor).all().item())


class RpcTest(RpcAgentTestFixture, RpcTestCommon):
    @dist_init
    def test_worker_id(self):
        n = self.rank + 1
        peer_rank = n % self.world_size
        self_worker_info = rpc.get_worker_info()
        peer_worker_info = rpc.get_worker_info(worker_name(peer_rank))

        self.assertEqual(self_worker_info.name, worker_name(self.rank))
        self.assertEqual(peer_worker_info.name, worker_name(peer_rank))

        with self.assertRaisesRegex(RuntimeError, "could not find destination"):
            rpc.get_worker_info("WorkerUnknown")

    @dist_init
    def test_get_worker_infos(self):
        worker_infos = rpc.api._get_current_rpc_agent().get_worker_infos()

        worker_names = {worker_info.name for worker_info in worker_infos}
        expected_worker_names = {worker_name(rank) for rank in range(self.world_size)}
        self.assertEqual(worker_names, expected_worker_names)

        worker_ids = {worker_info.id for worker_info in worker_infos}
        expected_worker_ids = set(range(self.world_size))
        self.assertEqual(worker_ids, expected_worker_ids)

    @dist_init
    def test_self_add(self):
        self_worker_info = rpc.get_worker_info()
        fut = rpc.rpc_async(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        ret = rpc.rpc_sync(self_worker_info, torch.add, args=(torch.ones(2, 2), 1))
        self.assertEqual(fut.wait(), torch.ones(2, 2) + 1)
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    @dist_init
    def test_send_to_rank(self):
        dst_rank = (self.rank + 1) % self.world_size

        # Test dense tensor
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            ret = self._run_func_in_mode(
                dst_rank, torch.add, exec_mode, args=(torch.ones(2, 2), 1)
            )
            self.assertEqual(ret, torch.ones(2, 2) + 1)

        # Test invalid ranks
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(RuntimeError):
                self._run_func_in_mode(
                    self.world_size + 1,
                    torch.add,
                    exec_mode,
                    args=(torch.ones(2, 2), 1),
                )

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(RuntimeError):
                self._run_func_in_mode(
                    -1, torch.add, exec_mode, args=(torch.ones(2, 2), 1)
                )

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(ValueError):
                self._run_func_in_mode(
                    dst_rank + 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1)
                )

        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            with self.assertRaises(ValueError):
                self._run_func_in_mode(
                    dst_rank - 0.5, torch.add, exec_mode, args=(torch.ones(2, 2), 1)
                )

    @dist_init
    def test_self_py_udf_remote(self):
        self._self_py_udf_remote(rpc.get_worker_info(), torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_rpc_arg(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_rpc_arg(dst, torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_self_rpc_arg(self):
        self._self_remote_rref_as_rpc_arg(rpc.get_worker_info(), torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_remote_arg(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_remote_arg(dst, torch.ones(2, 2), 1, 3)

    @dist_init
    def test_self_remote_rref_as_self_remote_arg(self):
        self._self_remote_rref_as_remote_arg(
            rpc.get_worker_info(), torch.ones(2, 2), 1, 3
        )

    @dist_init
    def test_rref_proxy_non_exist(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))
        msg = "has no attribute 'non_exist'"
        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_sync().non_exist()

        with self.assertRaisesRegex(AttributeError, msg):
            rref.rpc_async().non_exist().wait()

        with self.assertRaisesRegex(AttributeError, msg):
            rref.remote().non_exist()

    def _test_rref_proxy_tensor(self, dst):
        rref = rpc.remote(dst, my_function, args=(torch.ones(2, 2), 1, 3))

        expected = torch.ones(2, 2) + 1 + 3
        self.assertEqual(expected.size(), rref.rpc_sync().size())
        self.assertEqual(expected + 1, rref.rpc_async().add(1).wait())
        self.assertEqual(expected.view(1, 4), rref.remote().view(1, 4).to_here())

    @dist_init
    def test_rref_proxy_tensor(self):
        self._test_rref_proxy_tensor(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_tensor_self(self):
        self._test_rref_proxy_tensor(rpc.get_worker_info())

    @dist_init
    def test_rref_proxy_reuse(self):
        rref = rpc.remote(
            worker_name((self.rank + 1) % self.world_size),
            my_function,
            args=(torch.ones(2, 2), 1, 3),
        )
        expected = torch.ones(2, 2) + 1 + 3

        proxy_rpc_sync = rref.rpc_sync()
        proxy_rpc_async = rref.rpc_async()
        proxy_remote = rref.remote()

        self.assertEqual(expected.size(), proxy_rpc_sync.size())
        self.assertEqual(expected + 1, proxy_rpc_sync.add(1))
        self.assertEqual(expected.view(1, 4), proxy_rpc_sync.view(1, 4))

        self.assertEqual(expected.size(), proxy_rpc_async.size().wait())
        self.assertEqual(expected + 3, proxy_rpc_async.add(3).wait())
        self.assertEqual(expected.view(4, 1), proxy_rpc_async.view(4, 1).wait())

        self.assertEqual(expected.size(), proxy_remote.size().to_here())
        self.assertEqual(expected + 5, proxy_remote.add(5).to_here())
        self.assertEqual(expected.view(-1), proxy_remote.view(-1).to_here())

    def _test_rref_proxy_class(self, dst):
        rref = rpc.remote(dst, MyClass, args=(7,))
        expected = MyClass(7)
        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())

        expected.increment_value(3)
        self.assertEqual(None, rref.rpc_sync().increment_value(1))
        self.assertEqual(None, rref.rpc_async().increment_value(1).wait())
        self.assertEqual(None, rref.remote().increment_value(1).to_here())

        self.assertEqual(expected.get_value(), rref.rpc_sync().get_value())
        self.assertEqual(expected.get_value(), rref.rpc_async().get_value().wait())
        self.assertEqual(expected.get_value(), rref.remote().get_value().to_here())

        self.assertEqual(
            expected.my_instance_method(2), rref.rpc_sync().my_instance_method(2)
        )
        self.assertEqual(
            expected.my_instance_method(3),
            rref.rpc_async().my_instance_method(3).wait(),
        )
        self.assertEqual(
            expected.my_instance_method(4),
            rref.remote().my_instance_method(4).to_here(),
        )

        self.assertEqual(
            expected.my_static_method(9), rref.rpc_sync().my_static_method(9)
        )
        self.assertEqual(
            expected.my_static_method(10), rref.rpc_async().my_static_method(10).wait()
        )
        self.assertEqual(
            expected.my_static_method(11), rref.remote().my_static_method(11).to_here()
        )

        self.assertEqual(
            expected.my_class_method(2, torch.zeros(2, 2)),
            rref.rpc_sync().my_class_method(2, torch.zeros(2, 2)),
        )
        self.assertEqual(
            expected.my_class_method(2, torch.ones(3, 3)),
            rref.rpc_async().my_class_method(2, torch.ones(3, 3)).wait(),
        )
        self.assertEqual(
            expected.my_class_method(2, torch.ones(4, 4)),
            rref.remote().my_class_method(2, torch.ones(4, 4)).to_here(),
        )

    @dist_init
    def test_rref_proxy_class(self):
        self._test_rref_proxy_class(worker_name((self.rank + 1) % self.world_size))

    @dist_init
    def test_rref_proxy_class_self(self):
        self._test_rref_proxy_class(rpc.get_worker_info())

    @mock.patch.object(torch.distributed.autograd, "_init")
    @mock.patch.object(torch.distributed.rpc.api, "_set_and_start_rpc_agent")
    @dist_init(setup_rpc=False)
    def test_register_rpc_backend_and_set_and_start_rpc_backend(
        self, mock_rpc_agent, mock_dist_autograd_init
    ):
        backend_name = "stub_backend"

        backend = rpc.backend_registry.register_backend(
            backend_name,
            _stub_construct_rpc_backend_options_handler,
            _stub_init_rpc_backend_handler,
        )

        with self.assertRaisesRegex(
            RuntimeError, "^RPC backend .+: already registered$"
        ):
            backend = rpc.backend_registry.register_backend(
                backend_name,
                _stub_construct_rpc_backend_options_handler,
                _stub_init_rpc_backend_handler,
            )

        rpc.init_rpc(
            name="worker1",
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

    @dist_init(setup_rpc=False)
    def test_duplicate_name(self):
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            store, _, _ = next(
                torch.distributed.rendezvous(
                    self.init_method, rank=self.rank, world_size=self.world_size
                )
            )
            rpc._init_rpc_backend(
                backend=self.rpc_backend,
                store=store,
                name="duplicate_name",
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_duplicate_name_2(self):
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            rpc.init_rpc(
                name=worker_name(self.rank % (self.world_size - 1)),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

    @dist_init(setup_rpc=False)
    def test_reinit(self):
        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        initialize_pg(self.file_init_method, self.rank, self.world_size)
        # Wait for all init to complete.
        dist.barrier()

        # TODO: with TCP init, rank 0 raises Address already in use because
        # rank 0 is the start daemon and the store is created before checking if
        # RPC is already initialized in init_rpc.
        if os.environ.get("RPC_INIT_WITH_TCP", None) == "1" and self.rank == 0:
            expected_reinit_err = "Address already in use"
        else:
            expected_reinit_err = "is already initialized"

        with self.assertRaisesRegex(RuntimeError, expected_reinit_err):
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_pg_init_no_rpc_init(self):
        dist.init_process_group(
            backend="gloo",
            init_method=self.file_init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(3, 4)

            def forward(self, x):
                return self.lin(x)

        model = MyModel()
        model.train()
        model = torch.nn.parallel.DistributedDataParallel(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "Current RPC agent is not set! Did you initialize the RPC framework",
        ):
            [RRef(param) for param in model.parameters()]

    def test_world_size_one(self):
        self._world_size_one(torch.ones(2, 2), torch.ones(2, 2))

    @dist_init(setup_rpc=False)
    def test_invalid_names(self):
        worker_id = 0
        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            WorkerInfo("abc*", worker_id)

        with self.assertRaisesRegex(RuntimeError, "Worker name must match"):
            WorkerInfo(" ", worker_id)

        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            WorkerInfo("", worker_id)

        # If the number in the message does not match, it is likely that the
        # value of MAX_NAME_LEN in RPC WorkerInfo has changed.
        with self.assertRaisesRegex(RuntimeError, "shorter than 128"):
            WorkerInfo("".join(["a" for i in range(500)]), worker_id)

    # Test that WorkerInfo can be pickled and sent in RPC call
    @dist_init
    def test_worker_info_pickle(self):
        dst_rank = (self.rank + 1) % self.world_size
        worker_info = rpc.api.get_worker_info()
        ret = rpc.rpc_sync(worker_name(dst_rank), identity, args=(worker_info,))
        self.assertEqual(ret, worker_info)

    @dist_init
    def test_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @staticmethod
    def return_callee_id():
        return rpc.get_worker_info().id

    @dist_init
    def test_int_callee(self):
        dst_rank = (self.rank + 1) % self.world_size
        ret = rpc.rpc_sync(dst_rank, RpcTest.return_callee_id)
        self.assertEqual(ret, dst_rank)

    @dist_init
    def test_add_with_id(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        workder_info = rpc.get_worker_info(worker_name(dst_rank))

        ret = rpc.rpc_sync(
            workder_info, torch.add, args=(torch.ones(n, n), torch.ones(n, n))
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)

    @dist_init
    def test_scalar_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), n))
        self.assertEqual(ret, (torch.ones(n, n) + n))

    @dist_init
    def test_async_add(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(fut.wait(), torch.ones(n, n) * 2)

    @dist_init
    def test_nonzero(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        x = torch.ones(self.world_size, self.world_size)
        x[self.rank][self.rank] = 0
        ret = rpc.rpc_sync(worker_name(dst_rank), torch.nonzero, args=(x,))
        self.assertEqual(ret, x.nonzero())

    @dist_init
    def test_multi_rpc(self):
        self._multi_rpc(False)

    @dist_init
    def test_future_wait_twice(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        futs = [rpc.rpc_async(dst, raise_func) for _ in range(20)]

        with self.assertRaisesRegex(ValueError, "Expected error"):
            torch.futures.wait_all(futs)

        for fut in futs:
            with self.assertRaisesRegex(ValueError, "Expected error"):
                fut.wait()

    @dist_init(setup_rpc=False)
    def test_wait_all_workers_timeout(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)

        rpc.init_rpc(
            name=worker_name(self.rank),
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        og_func = rpc.api._wait_all_workers

        def wait_all_workers_sleep(timeout):
            rpc.api._all_gather(SlowPickleClass(0.5), timeout=timeout)

        rpc.api._wait_all_workers = wait_all_workers_sleep

        try:
            with self.assertRaisesRegex(RuntimeError, ""):
                rpc.shutdown(graceful=True, timeout=0.01)
        finally:
            rpc.api._wait_all_workers = og_func
        dist.barrier()

    def test_wait_all_workers_dense(self):
        self._wait_all_workers(heavy_rpc, torch.ones(100, 100))

    def test_wait_all_workers_twice_dense(self):
        self._wait_all_workers_twice(heavy_rpc, torch.ones(100, 100))

    @dist_init
    def test_all_gather(self):
        info = rpc.get_worker_info()
        results = rpc.api._all_gather(info.id)
        expected = {}
        for info in rpc._get_current_rpc_agent().get_worker_infos():
            expected[info.name] = info.id

        self.assertEqual(expected, results)

    @dist_init
    def test_all_gather_timeout(self):
        rpc._set_rpc_timeout(0.1)

        if self.rank == 0:
            with self.assertRaisesRegex(
                RuntimeError, "timed out in _all_gather after 0\\.10 seconds"
            ):
                rpc.api._all_gather(SlowPickleClass(0.5))
        else:
            expected_error = self.get_timeout_error_regex()
            with self.assertRaisesRegex(RuntimeError, expected_error):
                rpc.api._all_gather(SlowPickleClass(0.5))

    def _test_barrier_helper(self, info, names, multi_threaded=False):
        names = sorted(names)
        leader = names[0]
        rpc.rpc_sync(leader, _reset_count)
        if not multi_threaded and info.name == leader:
            self.assertEqual(_rpc_barrier_count, 0)
        rpc.api._barrier(names)
        rpc.rpc_sync(leader, _increment_count)
        rpc.api._barrier(names)
        if not multi_threaded and info.name == leader:
            self.assertEqual(_rpc_barrier_count, len(names))

    @dist_init
    def test_rpc_barrier_all(self):
        # Test rpc barrier when called with full list of workers
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        names = [worker.name for worker in all_worker_info]
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_subset(self):
        # Test rpc barrier when processes are called with different subsets of the full list
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        if info.id % 2:
            names = [worker.name for worker in all_worker_info if worker.id % 2]
        else:
            names = [worker.name for worker in all_worker_info if not worker.id % 2]
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_partial_subset(self):
        # Test rpc barrier when some processes are not involved in the barrier
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        if info.id % 2:
            names = [worker.name for worker in all_worker_info if worker.id % 2]
        else:
            names = [f"worker{info.id}"]
        self._test_barrier_helper(info, names)

    @dist_init
    def test_rpc_barrier_multithreaded(self):
        # This tests validates the implementation of barrier when multiple threads call into it
        # We only need to check that it does not hang in this case
        info = rpc.get_worker_info()
        all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
        names = [worker.name for worker in all_worker_info]
        threads = []
        for _ in range(3):
            th = threading.Thread(
                target=self._test_barrier_helper, args=(info, names, True)
            )
            threads.append(th)
            th.start()
        for th in threads:
            th.join()

    @dist_init
    def test_graceful_shutdown_with_uneven_workload(self):
        """Test graceful termination."""
        self._run_uneven_workload(heavy_rpc, torch.ones(100, 100))

    @dist_init(setup_rpc=False)
    def test_shutdown_followed_by_rpc(self):
        # Initialize RPC.
        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            backend=self.rpc_backend,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=self.rpc_backend_options,
        )

        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            torch.add,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, torch.ones(n, n) * 2)
        rpc.shutdown()

        with self.assertRaisesRegex(RuntimeError, "^RPC has not been initialized"):
            rpc.rpc_sync(
                worker_name(dst_rank),
                torch.add,
                args=(torch.ones(n, n), torch.ones(n, n)),
            )

    @dist_init
    def test_expected_src(self):
        dst_rank = (self.rank + 1) % self.world_size
        expected_src_rank = (self.rank - 1) % self.world_size
        rpc.rpc_sync(worker_name(dst_rank), set_value, args=(self.rank,))
        value = VALUE_FUTURE.result()
        self.assertEqual(value, expected_src_rank)

    @dist_init
    def test_py_built_in(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), min, args=(n, n + 1, n + 2))
        self.assertEqual(ret, min(n, n + 1, n + 2))

    @dist_init
    def test_py_user_defined(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        self.assertEqual(ret, my_function(n, n + 1, n + 2))

    def test_build_rpc_profiling_key(self):
        # Tests that the name that shows up as an Event in profiling RPCs has all
        # the necessary information.
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            rpc_profiling_key = _build_rpc_profiling_key(
                exec_mode, "foo", "worker0", "worker1"
            )
            self.assertIn(exec_mode.value, rpc_profiling_key)
            self.assertIn("foo", rpc_profiling_key)
            self.assertIn("worker0", rpc_profiling_key)
            self.assertIn("worker1", rpc_profiling_key)

    def check_profiling_info(
        self, self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode
    ):
        self.assertTrue(self_worker_name in rpc_event.name)
        self.assertTrue(dst_worker_name in rpc_event.name)
        if isinstance(func, torch.jit.ScriptFunction):
            self.assertTrue(torch._jit_internal._qualified_name(func) in rpc_event.name)
        else:
            self.assertTrue(func.__name__ in rpc_event.name)
        self.assertTrue(rpc_exec_mode.value in rpc_event.name)
        self.assertEqual(rpc_event.count, 1)

    @dist_init
    def test_profiler_rpc_record_shapes(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        t1, t2 = torch.ones(100), torch.ones(100)
        with _profile(record_shapes=True) as prof:
            rpc.rpc_sync(dst_worker, torch.add, args=(t1, t2))

        function_events = prof.function_events
        remote_events = [event for event in function_events if event.is_remote]
        remote_add_event = next(
            event for event in remote_events if "aten::add" in event.name
        )
        remote_add_input_shapes = remote_add_event.input_shapes
        # Run profiler on equivalent local op and validate shapes are the same.
        with _profile(record_shapes=True) as prof:
            torch.add(t1, t2)

        local_function_events = prof.function_events
        local_add_event = next(
            event for event in local_function_events if "aten::add" in event.name
        )
        local_add_input_shapes = local_add_event.input_shapes
        self.assertEqual(remote_add_input_shapes, local_add_input_shapes)

    @dist_init
    def test_profiler_rpc_memory(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        with _profile(profile_memory=True) as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            fut.wait()

        function_events = p.function_events
        event_cpu_mem_usages = {event.cpu_memory_usage for event in function_events}
        # if cpu_memory_usage was not propagated over the wire, this set would
        # only contain 0 (indicates no memory being profiled)
        self.assertNotEqual({0}, event_cpu_mem_usages)
        # No memory profiled if profile_memory=False
        with _profile(profile_memory=False) as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            fut.wait()

        function_events = p.function_events
        event_cpu_mem_usages = {event.cpu_memory_usage for event in function_events}
        self.assertEqual({0}, event_cpu_mem_usages)

    @dist_init
    def test_profiler_export_trace(self):
        if self.rank != 1:
            return
        dst = (self.rank + 1) % self.world_size
        dst_worker = worker_name(dst)
        with _profile() as p:
            fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
            fut.wait()

        with TemporaryFileName() as fname:
            path = fname
            p.export_chrome_trace(path)
            with open(path) as f:
                trace = json.load(f)
                event_names = [event["name"] for event in trace]
                for expected_event_name in EXPECTED_REMOTE_EVENTS + [
                    RPCExecMode.ASYNC.value
                ]:
                    event_exists = any(
                        expected_event_name in event_name for event_name in event_names
                    )
                    self.assertTrue(event_exists)

    @dist_init
    def test_profiler_rpc_key_names(self):
        # tests that remote events are properly prefixed with the RPC profiling key.
        if self.rank != 1:
            return

        # Spawn multiple threads that send RPCs to ensure keys are correctly
        # prefixed when there are multiple RPCs being created/in flight at the
        # same time.
        dst_ranks = [rank for rank in range(self.world_size) if rank != self.rank]

        def rpc_with_profiling(dst_worker):
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                fut.wait()

            events = prof.function_events
            remote_event_names = {
                event.name: event for event in events if event.is_remote
            }
            rpc_profiling_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC,
                udf_with_torch_ops.__qualname__,
                worker_name(self.rank),
                dst_worker,
            )

            remote_event_name_set = set(EXPECTED_REMOTE_EVENTS)
            for name, event in remote_event_names.items():
                # Ensure that we have the expected key as part of the remote
                # event.
                self.assertTrue(name.startswith(rpc_profiling_key))
                self.assertTrue(event.is_remote)
                self.assertTrue(event.node_id == rpc.get_worker_info(dst_worker).id)
                # Ensure that the remote event name also contains the operator.
                operator_name_substr = name[len(rpc_profiling_key) :]
                # Note: we don't assert that every remote event needs to be
                # in the above set, the set is just a representative set of
                # what we expect to see. The profiler can change and add more
                # events, but we should always expect to see this representative
                # set.
                matching_event = {
                    remote_event_name
                    for remote_event_name in remote_event_name_set
                    if remote_event_name in operator_name_substr
                }
                remote_event_name_set -= matching_event

            # The set should be empty, otherwise its contained elements did
            # not show up in the remote profiler output.
            self.assertTrue(
                remote_event_name_set == set(),
                f"Expected {remote_event_name_set} to be included in remote profiler output.",
            )

        for dst in dst_ranks:
            dst_worker = worker_name(dst)
            num_parallel_rpcs = 2
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_parallel_rpcs
            ) as executor:
                futs = [
                    executor.submit(rpc_with_profiling, dst_worker)
                    for _ in range(num_parallel_rpcs)
                ]
                # Wait for workers to finish test
                for fut in futs:
                    fut.result()

    def _run_test_profiler_remote_events_profiled(self):
        # Tests that we can successfully invoke the profiler on a remote node,
        # and collect the remote events back in the local profiler.
        if self.rank != 1:
            return

        dst_ranks = [rank for rank in range(self.world_size) if rank != self.rank]
        for dst in dst_ranks:
            dst_worker = worker_name(dst)
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=())
                fut.wait()

            events = prof.function_events

            rpc_event = get_function_event(events, RPCExecMode.ASYNC.value)
            self.check_profiling_info(
                worker_name(self.rank),
                dst_worker,
                udf_with_torch_ops,
                rpc_event,
                RPCExecMode.ASYNC,
            )

            remote_events = {event.name: event for event in events if event.is_remote}
            rpc_profiling_key = _build_rpc_profiling_key(
                RPCExecMode.ASYNC,
                udf_with_torch_ops.__qualname__,
                worker_name(self.rank),
                worker_name(dst),
            )

            for expected_remote_event_name in EXPECTED_REMOTE_EVENTS:
                expected_key = (
                    rpc_profiling_key + REMOTE_OP_STR + expected_remote_event_name
                )
                self.assertTrue(expected_key in remote_events)
                remote_event = remote_events[expected_key]
                # Remote event should have a node ID corresponding to the worker
                # it ran on.
                self.assertEqual(remote_event.node_id, dst)

            # Validate order remote events show up in profiling output.
            def convert_remote_to_local(event_name):
                remote_op_key = rpc_profiling_key + REMOTE_OP_STR
                return event_name[event_name.find(remote_op_key) + len(remote_op_key) :]

            remote_events_list = [
                convert_remote_to_local(event.name)
                for event in events
                if convert_remote_to_local(event.name) in EXPECTED_REMOTE_EVENTS
            ]
            self.assertEqual(
                set(remote_events_list),
                set(EXPECTED_REMOTE_EVENTS),
                f"Mismatch between profiled events: {set(remote_events_list)} and expected events: {set(EXPECTED_REMOTE_EVENTS)}",
            )

    @dist_init
    def test_profiler_remote_events_profiled(self):
        self._run_test_profiler_remote_events_profiled()

    @dist_init
    def test_profiler_remote_events_profiled_single_threaded(self):
        self._run_test_profiler_remote_events_profiled()

    def run_profiling_workload(self, dst):
        fut = rpc.rpc_async(
            worker_name(dst),
            torch.mul,
            args=(
                torch.tensor(1.0, requires_grad=True),
                torch.tensor(1.0, requires_grad=True),
            ),
        )
        fut.wait()

    def _run_rpc_profiling_async_function(self, device="cpu"):
        if self.rank != 1:
            return

        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        x = torch.ones(2)
        y = torch.ones(2)
        with _profile() as prof:
            ret = rpc.rpc_async(
                dst1, slow_async_add, args=(dst2, x, y, device), timeout=20
            )
            ret.wait()

        function_events = prof.function_events
        # slow_async_add resulted in an RPC from dst1 -> dst2, so this should be
        # recorded.
        key_prefix = _build_rpc_profiling_key(
            RPCExecMode.ASYNC, slow_async_add.__qualname__, worker_name(self.rank), dst1
        )

        nested_rpc_key_prefix = _build_rpc_profiling_key(
            RPCExecMode.ASYNC, slow_add.__qualname__, dst1, dst2
        )
        expected_key = key_prefix + REMOTE_OP_STR + nested_rpc_key_prefix
        remote_events = [event for event in function_events if event.is_remote]
        rpc_remote_event = [
            event for event in remote_events if event.name == expected_key
        ]
        self.assertEqual(1, len(rpc_remote_event))
        rpc_remote_event = rpc_remote_event[0]
        self.assertEqual(rpc_remote_event.node_id, (self.rank + 1) % self.world_size)
        # slow_async_add's RPC does an add on dst2, which should be reflected as well.
        remote_add_key = (
            expected_key + REMOTE_OP_STR + torch.jit._builtins._find_builtin(torch.add)
        )
        remote_add_event = [
            event for event in remote_events if event.name == remote_add_key
        ]
        self.assertEqual(1, len(remote_add_event))
        remote_add_event = remote_add_event[0]
        # Validate that node_id is dst2.
        self.assertEqual(remote_add_event.node_id, (self.rank + 2) % self.world_size)

    @dist_init
    def test_rpc_profiling_async_function(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        self._run_rpc_profiling_async_function()
        if torch.cuda.is_available():
            dist.barrier()
            self._run_rpc_profiling_async_function(device="cuda:0")

    @dist_init
    def test_rpc_profiling_async_function_single_threaded(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        self._run_rpc_profiling_async_function()
        if torch.cuda.is_available():
            dist.barrier()
            self._run_rpc_profiling_async_function(device="cuda:0")

    @dist_init
    def test_rpc_profiling_remote_record_function(self):
        # test that functions run over RPC with record_function show the expected
        # profiled block.
        if self.rank != 1:
            return
        dst_ranks = [i for i in range(self.world_size) if i != self.rank]
        for dst_rank in dst_ranks:
            dst_worker = worker_name(dst_rank)
            with _profile() as prof:
                fut = rpc.rpc_async(dst_worker, udf_with_torch_ops, args=(-1, True))
                fut.wait()

            function_events = prof.function_events
            record_function_remote_event = [
                evt for evt in function_events if "##forward##" in evt.name
            ]
            self.assertEqual(1, len(record_function_remote_event))
            record_function_remote_event = record_function_remote_event[0]
            self.assertEqual(record_function_remote_event.node_id, dst_rank)
            # cpu_children only returns direct children, so here we get all
            # children recursively.

            def get_cpu_children(event):
                if not event.cpu_children:
                    return []
                cpu_children = event.cpu_children
                for e in event.cpu_children:
                    cpu_children.extend(get_cpu_children(e))
                return cpu_children

            remote_children = get_cpu_children(record_function_remote_event)
            # Get local children and verify parity.
            with _profile() as prof:
                udf_with_torch_ops(-1, True)

            local_function_events = prof.function_events
            local_record_function_event = next(
                evt for evt in local_function_events if "##forward##" in evt.name
            )
            local_children = get_cpu_children(local_record_function_event)
            local_children_names = [evt.name for evt in local_children]

            REMOTE_OP_STR = "#remote_op: "

            def convert_remote_to_local(event_name):
                remote_op_key = REMOTE_OP_STR
                return event_name[event_name.find(remote_op_key) + len(remote_op_key) :]

            for evt in remote_children:
                local_name = convert_remote_to_local(evt.name)
                self.assertTrue(local_name in local_children_names)

    def validate_profiling_workload(self, dst, prof):
        def convert_remote_to_local(event_name):
            return event_name[event_name.find(REMOTE_OP_STR) + len(REMOTE_OP_STR) :]

        events = prof.function_events
        remote_events = {
            convert_remote_to_local(event.name): event
            for event in events
            if event.is_remote
        }
        self.assertTrue("aten::mul" in remote_events)
        remote_mul_event = remote_events["aten::mul"]
        self.assertEqual(remote_mul_event.node_id, dst)
        self.check_profiling_info(
            worker_name(self.rank),
            worker_name(dst),
            torch.mul,
            remote_mul_event,
            RPCExecMode.ASYNC,
        )

    def _run_test_profiler_with_autograd_context(self):
        dst = (self.rank + 1) % self.world_size
        if self.rank == 1:
            # Cases where we can double wrap messages with profiling information and autograd info.
            with dist_autograd.context(), _profile() as prof:
                self.run_profiling_workload(dst)

            self.validate_profiling_workload(dst, prof)

            # Ensure that flipped order of ctx managers results in events being
            # recorded as expected.
            with _profile() as prof, dist_autograd.context():
                self.run_profiling_workload(dst)

            self.validate_profiling_workload(dst, prof)

    @dist_init
    def test_profiler_with_autograd_context_single_threaded(self):
        self._run_test_profiler_with_autograd_context()

    @dist_init
    def test_profiler_with_autograd_context(self):
        self._run_test_profiler_with_autograd_context()

    def _profiler_test_with_rpc(
        self,
        rpc_exec_mode,
        func,
        args,
        use_record_function=False,
        dst=None,
        kineto_profile=False,
    ):
        dst = dst if dst is not None else (self.rank + 1) % self.world_size

        # only run profiler on rank 1.
        p = _profile if not kineto_profile else torch.profiler.profile  # kineto
        if self.rank == 1:
            with p() as prof:
                record_function_ctx_mgr = (
                    contextlib.nullcontext()
                    if not use_record_function
                    else torch.autograd.profiler.record_function("foo")
                )
                with record_function_ctx_mgr:
                    if rpc_exec_mode == RPCExecMode.SYNC:
                        rpc.rpc_sync(worker_name(dst), func, args=args)
                    elif rpc_exec_mode == RPCExecMode.ASYNC:
                        fut = rpc.rpc_async(worker_name(dst), func, args=args)
                        if kineto_profile:
                            # Ensure multiple async RPCs don't cause issues.
                            # Would have raised
                            # "RuntimeError: Cannot call
                            # RemoteProfilerManager::setCurrentKey when current
                            # key is already set." error if RPC profiling was
                            # not disabled properly for kineto.
                            fut2 = rpc.rpc_async(worker_name(dst), func, args=args)
                            fut2.wait()
                        fut.wait()
                    else:
                        self.assertTrue(rpc_exec_mode == RPCExecMode.REMOTE)
                        rref = rpc.remote(worker_name(dst), func, args=args)
                        rref.to_here()
                        # To avoid flakiness, wait for the RRef to be profiled. This
                        # means that we received the acknowledgement of successful
                        # creation on the owner and ran the callbacks responsible
                        # for recording the profiling event.
                        rref._get_profiling_future().wait()

            events = prof.function_events if not kineto_profile else prof.events()
            if kineto_profile:
                # RPC profiling is disabled so there should be no rpc related
                # events.
                with self.assertRaises(IndexError):
                    get_function_event(events, rpc_exec_mode.value)

                return

            rpc_event = get_function_event(events, rpc_exec_mode.value)
            # verify Node ID for this rpc event.
            self.assertEqual(rpc_event.node_id, self.rank)
            # Ensure recording of remote events.
            remote_events = {event for event in events if event.node_id == dst} - {
                rpc_event
            }
            self.assertGreaterEqual(len(remote_events), 1)
            for remote_event in remote_events:
                self.assertEqual(remote_event.node_id, dst)

            if use_record_function:
                scope_event = get_function_event(events, "foo")
                # Since RPC call is within the scope, its CPU interval should be
                # contained within foo's interval.
                self.assertLessEqual(
                    scope_event.time_range.start, rpc_event.time_range.start
                )
                self.assertGreaterEqual(
                    scope_event.time_range.end, rpc_event.time_range.end
                )
            # the sender, dest worker, function run, and type of RPC should all
            # be recorded.
            self_worker_name = worker_name(self.rank)
            dst_worker_name = worker_name(dst)
            self.check_profiling_info(
                self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode
            )
            if use_record_function:
                # verify order by ensuring that the outer context comes
                # before the rpc event.
                foo_event_ix = next(
                    i for i, event in enumerate(events) if "foo" in event.name
                )
                rpc_event_idx = next(
                    i
                    for i, event in enumerate(events)
                    if rpc_exec_mode.value in event.name
                )
                self.assertLess(foo_event_ix, rpc_event_idx)

    def _run_test_profiler_with_sync_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.SYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, my_sleep_func, args=(1,), use_record_function=True
        )

    @dist_init
    def test_profiler_with_sync_rpc_udf(self):
        self._run_test_profiler_with_sync_rpc_udf()

    @dist_init
    def test_profiler_with_sync_rpc_udf_single_threaded(self):
        self._run_test_profiler_with_sync_rpc_udf()

    def _run_test_profiler_with_sync_rpc_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_sync_rpc_builtin(self):
        self._run_test_profiler_with_sync_rpc_builtin()

    @dist_init
    def test_profiler_with_sync_rpc_builtin_single_threaded(self):
        self._run_test_profiler_with_sync_rpc_builtin()

    def _run_test_profiler_with_async_rpc_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.ASYNC, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_sleep_func, args=(1,), use_record_function=True
        )
        # Test to ensure that kineto profiler enabled in RPC does not enable
        # RPC profiling (it is unsupported) and does not result in issues.
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_sleep_func, args=(1,), kineto_profile=True
        )

    @dist_init
    def test_profiler_with_async_rpc_udf(self):
        self._run_test_profiler_with_async_rpc_udf()

    @dist_init
    def test_profiler_with_async_rpc_udf_single_threaded(self):
        self._run_test_profiler_with_async_rpc_udf()

    def _run_test_profiler_with_async_rpc_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_async_rpc_builtin(self):
        self._run_test_profiler_with_async_rpc_builtin()

    @dist_init
    def test_profiler_with_async_rpc_builtin_single_threaded(self):
        self._run_test_profiler_with_async_rpc_builtin()

    def _run_test_profiler_with_remote_udf(self):
        self._profiler_test_with_rpc(RPCExecMode.REMOTE, my_sleep_func, args=(1,))
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_sleep_func, args=(1,), use_record_function=True
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_sleep_func, args=(1,), dst=self.rank
        )

    @dist_init
    def test_profiler_with_remote_udf(self):
        self._run_test_profiler_with_remote_udf()

    @dist_init
    def test_profiler_with_remote_udf_single_threaded(self):
        self._run_test_profiler_with_remote_udf()

    def _run_test_profiler_with_remote_builtin(self):
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, torch.mul, args=(torch.ones(1), torch.ones(1))
        )
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            use_record_function=True,
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            torch.mul,
            args=(torch.ones(1), torch.ones(1)),
            dst=self.rank,
        )

    @dist_init
    def test_profiler_with_remote_builtin(self):
        self._run_test_profiler_with_remote_builtin()

    @dist_init
    def test_profiler_with_remote_builtin_single_threaded(self):
        self._run_test_profiler_with_remote_builtin()

    def _run_test_profiler_with_script_async_rpc(self):
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.ASYNC,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_script_async_rpc(self):
        self._run_test_profiler_with_script_async_rpc()

    @dist_init
    def test_profiler_with_script_async_rpc_single_threaded(self):
        self._run_test_profiler_with_script_async_rpc()

    def _run_test_profiler_with_script_sync_rpc(self):
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.SYNC,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )

    @dist_init
    def test_profiler_with_script_sync_rpc(self):
        self._run_test_profiler_with_script_sync_rpc()

    @dist_init
    def test_profiler_with_script_sync_rpc_single_threaded(self):
        self._run_test_profiler_with_script_sync_rpc()

    def _run_test_profiler_with_script_remote_rpc(self):
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),)
        )
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE,
            my_script_func,
            args=(torch.tensor(1),),
            use_record_function=True,
        )
        # test remote to self
        self._profiler_test_with_rpc(
            RPCExecMode.REMOTE, my_script_func, args=(torch.tensor(1),), dst=self.rank
        )

    @dist_init
    def test_profiler_with_script_remote_rpc(self):
        self._run_test_profiler_with_script_remote_rpc()

    @dist_init
    def test_profiler_with_script_remote_rpc_single_threaded(self):
        self._run_test_profiler_with_script_remote_rpc()

    def _assert_top_level_events(
        self, process_global_events, expected_top_level_event_names
    ):
        top_level_event_names = []
        for thread_local_events in process_global_events:
            # Get top-level events from all events happened on a thread.
            last_end_time = 0
            for event in thread_local_events:
                event_name = event.name
                time_range = event.time_range
                if time_range.start > last_end_time:
                    top_level_event_names.append(event_name)
                    last_end_time = time_range.end
        top_level_event_names = sorted(top_level_event_names)
        expected_top_level_event_names = sorted(expected_top_level_event_names)
        self.assertEqual(
            top_level_event_names,
            expected_top_level_event_names,
            f"Expected events {expected_top_level_event_names}, but got {top_level_event_names}",
        )

    @dist_init
    def test_server_process_global_profiler(self):
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker_name = worker_name(dst_rank)

        x = torch.tensor(1)
        y = torch.tensor(2)

        outer_profile_rref = rpc.remote(
            dst_worker_name, rpc._server_process_global_profile
        )
        outer_profile_rref.rpc_sync().__enter__()
        rpc.rpc_sync(dst_worker_name, torch.add, (x, y))
        inner_profile_rref = rpc.remote(
            dst_worker_name, rpc._server_process_global_profile
        )
        inner_profile_rref.rpc_sync().__enter__()
        rpc.rpc_sync(dst_worker_name, torch.sub, (x, y))
        inner_profile_rref.rpc_sync().__exit__(None, None, None)
        outer_profile_rref.rpc_sync().__exit__(None, None, None)

        inner_events = rpc.rpc_sync(
            dst_worker_name, get_events_from_profile, (inner_profile_rref,)
        )
        expected_inner_events = ["aten::sub"]
        expected_outer_events = expected_inner_events + ["aten::add"]

        self._assert_top_level_events(inner_events, expected_inner_events)
        outer_events = rpc.rpc_sync(
            dst_worker_name, get_events_from_profile, (outer_profile_rref,)
        )
        self._assert_top_level_events(outer_events, expected_outer_events)

        inner_profile_rref.rpc_sync().key_averages()
        outer_profile_rref.rpc_sync().key_averages()

    @dist_init
    def test_async_record_function_double_end_callbacks(self):
        num_sleep_seconds = 1
        if self.rank == 1:
            # Validate that calling the function twice results in an error.
            with _profile():
                with torch.autograd.profiler.record_function("foo") as rf:
                    fut = rpc.rpc_async(
                        worker_name(0), my_sleep_func, args=(num_sleep_seconds,)
                    )
                    rf._call_end_callbacks_on_future(fut)
                    with self.assertRaisesRegex(
                        RuntimeError, "can only be called once."
                    ):
                        rf._call_end_callbacks_on_future(fut)
                fut.wait()

    @dist_init
    def test_async_record_function_legacy(self):
        # Test the legacy _record_function ops work
        # Note: These exist for backward compatibility with TorchScript
        num_sleep_seconds = 1
        if self.rank == 1:
            with _profile():
                try:
                    handle = torch.ops.profiler._record_function_enter("foo", None)
                    fut = rpc.rpc_async(
                        worker_name(0), my_sleep_func, args=(num_sleep_seconds,)
                    )
                    torch.ops.profiler._call_end_callbacks_on_jit_fut(handle, fut)
                finally:
                    torch.ops.profiler._record_function_exit(handle)

                fut.wait()

    @dist_init
    def test_async_record_function_cbs_jit_call(self):
        if self.rank == 1:
            with _profile() as pf:
                key = _build_rpc_profiling_key(
                    RPCExecMode.ASYNC,
                    torch._jit_internal._qualified_name(my_script_func),
                    "worker1",
                    "worker0",
                )
                with torch.autograd.profiler.record_function(key) as rf:
                    fut = rpc.rpc_async(
                        worker_name(0), my_script_func, args=(torch.tensor(1),)
                    )
                    # Intentionally calling record_function internals
                    fut = torch.ops.profiler._call_end_callbacks_on_jit_fut(
                        rf.record, fut
                    )
                result = fut.wait()
                # Validate that the profiling future returns the same value as the RPC
                # future.
                expected = torch.add(torch.tensor(1), torch.tensor(1))
                self.assertEqual(result, expected)
            events = pf.function_events
            rpc_event = get_function_event(
                events, torch._jit_internal._qualified_name(my_script_func)
            )
            self.assertTrue(
                torch._jit_internal._qualified_name(my_script_func) in rpc_event.name
            )

    @dist_init
    def test_py_class_constructor(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), MyClass, args=(n,))
        self.assertEqual(ret.a, n)

    @dist_init
    def test_py_class_instance_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass(2).my_instance_method, args=(n,)
        )
        self.assertEqual(ret, MyClass(2).my_instance_method(n))

    @dist_init
    def test_py_class_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass.my_class_method, args=(n, n + 1)
        )
        self.assertEqual(ret, MyClass.my_class_method(n, n + 1))

    @dist_init
    def test_py_class_static_method(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank), MyClass.my_static_method, args=(n + 10,)
        )
        self.assertEqual(ret, MyClass.my_static_method(n + 10))

    @dist_init
    def test_py_multi_async_call(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        dst_worker_info = rpc.get_worker_info(worker_name(dst_rank))
        fut1 = rpc.rpc_async(dst_worker_info, MyClass.my_static_method, args=(n + 10,))
        fut2 = rpc.rpc_async(dst_worker_info, min, args=(n, n + 1, n + 2))
        self.assertEqual(fut1.wait(), MyClass.my_static_method(n + 10))
        self.assertEqual(fut2.wait(), min(n, n + 1, n + 2))

    @dist_init
    def test_py_no_return_result(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), no_result)
        self.assertEqual(ret, no_result())

    @dist_init
    def test_py_tensors(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            my_tensor_function,
            args=(torch.ones(n, n), torch.ones(n, n)),
        )
        self.assertEqual(ret, my_tensor_function(torch.ones(n, n), torch.ones(n, n)))

    @dist_init
    def test_py_tensors_multi_async_call(self):
        futs = []
        n = self.rank + 1
        dst_rank = n % self.world_size
        for i in range(100):
            fut = rpc.rpc_async(
                worker_name(dst_rank),
                my_tensor_function,
                args=(torch.ones(i, i), torch.ones(i, i)),
            )
            futs.append(fut)

        for j, val in enumerate(torch.futures.wait_all(futs)):
            self.assertEqual(
                val, my_tensor_function(torch.ones(j, j), torch.ones(j, j))
            )

    @dist_init
    def test_py_tensors_in_container(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        a = [torch.ones(n, n), torch.ones(n, n)]
        b = TensorClass(build_complex_tensors())
        c = {"foo": torch.ones(n, n), "bar": torch.ones(n, n)}
        ret = rpc.rpc_sync(
            worker_name(dst_rank), my_complex_tensor_function, args=(a, b, c)
        )
        self.assertEqual(ret, my_complex_tensor_function(a, b, c))

    @dist_init
    def test_py_nested_pickle(self):
        n = self.rank + 1
        dst_rank = n % self.world_size

        ret = rpc.rpc_sync(
            worker_name(dst_rank),
            run_nested_pickle,
            args=(MyPickleClass(), torch.ones(2, 2)),
        )

        m = MyPickleClass()
        m.set(my_tensor_function(torch.ones(2, 2), torch.ones(2, 2)))
        self.assertEqual(ret, run_nested_pickle(m, torch.ones(2, 2)))

    @dist_init
    def test_py_function_exception(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        with self.assertRaises(TypeError):
            rpc.rpc_sync(worker_name(dst_rank), no_result, args=(10,))

    @dist_init
    def test_py_raise_in_user_func(self):
        with captured_output() as (_, err):
            # This barrier prevents a race condition where the main thread has
            # not entered the context manager when the remote function runs.
            initialize_pg(self.file_init_method, self.rank, self.world_size)
            dist.barrier()
            n = self.rank + 1
            dst_rank = n % self.world_size
            fut = rpc.rpc_async(worker_name(dst_rank), raise_func)
            with self.assertRaisesRegex(ValueError, expected_err):
                fut.wait()
            # This barrier prevents a race condition where the main thread exits
            # context manager before the remote function has ran.
            dist.barrier()

        # Validate that trainers log errors when running functions.
        stderr_lines = err.getvalue()
        self.assertTrue(expected_err in stderr_lines)

    @dist_init
    def test_py_raise_in_user_func_escaped_str(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        fut = rpc.rpc_async(worker_name(dst_rank), raise_func_escape)
        try:
            fut.wait()
        except ValueError as e:
            msg = str(e)
            # Ensure newlines are unescaped to provide a better repr of error.
            self.assertEqual(msg, msg.encode("utf-8").decode("unicode_escape"))
        else:
            self.assertTrue(False, "expected raise_func_escape to raise ValueError.")

    @dist_init
    def test_nested_rpc(self):
        self._nested_rpc(nested_rpc, torch.ones(2, 2) + 1)

    @dist_init
    def test_stress_light_rpc(self):
        self._stress_test_rpc(light_rpc)

    @dist_init
    def test_stress_heavy_rpc(self):
        self._stress_test_rpc(heavy_rpc, repeat=20, args=(torch.ones(100, 100),))

    @dist_init
    def test_stress_heavy_rpc_torchscript(self):
        self._stress_test_rpc(
            heavy_rpc_torchscript, repeat=20, args=(torch.ones(100, 100),)
        )

    @dist_init
    def test_builtin_remote_ret(self):
        self._builtin_remote_ret(
            torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2) * 2
        )

    @dist_init
    def test_builtin_remote_self(self):
        self._builtin_remote_self(
            torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2) * 2
        )

    @staticmethod
    def _multi_args_fn(n, sparse=False):
        if sparse:
            return (build_sparse_tensor(), build_sparse_tensor())
        else:
            return (torch.ones(n, n), torch.ones(n, n))

    @dist_init
    def test_multi_builtin_remote_ret(self):
        self._test_multi_remote_call(torch.add, False, args_fn=RpcTest._multi_args_fn)

    @dist_init
    def test_py_udf_remote(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(
            worker_name(dst_rank),
            my_function,
            kwargs={"a": n, "b": n + 1, "c": n + 2},
        )
        self.assertEqual(rref.to_here(), my_function(n, n + 1, n + 2))

    @staticmethod
    def _multi_kwargs_fn(n, sparse=False):
        if sparse:
            return {
                "a": build_sparse_tensor(),
                "b": build_sparse_tensor(),
                "c": build_sparse_tensor(),
            }
        else:
            return {"a": torch.ones(n, n), "b": torch.ones(n, n), "c": torch.ones(n, n)}

    @dist_init
    def test_multi_py_udf_remote(self):
        self._test_multi_remote_call(
            my_function, False, kwargs_fn=RpcTest._multi_kwargs_fn
        )

    @dist_init
    def test_py_rref_args(self):
        self._py_rref_args(
            torch.ones(2, 2), 1, torch.ones(2, 2), 2, torch.ones(2, 2) * 2 + 3
        )

    @dist_init
    def test_py_rref_args_user_share(self):
        self._py_rref_args_user_share(
            torch.ones(2, 2), 1, 2, torch.ones(2, 2), 3, 4, torch.ones(2, 2) * 2 + 10
        )

    @dist_init
    def test_py_rpc_rref_args(self):
        self._py_rpc_rref_args(
            torch.ones(2, 2), 1, 2, torch.ones(2, 2), 3, 4, torch.ones(2, 2) * 2 + 10
        )

    @dist_init
    def test_nested_remote(self):
        self._nested_remote(nested_remote, torch.ones(2, 2) + 3)

    @dist_init
    def test_nested_rref(self):
        self._nested_rref(nested_rref, torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)

    @dist_init
    def test_nested_rref_stress(self):
        self._nested_rref_stress(
            nested_rref, torch.ones(2, 2) + 1, torch.ones(2, 2) + 2
        )

    @dist_init
    def test_multi_layer_nested_async_rpc(self):
        # This test will exit right away, but there will be a chain of async
        # RPCs. The termination algorithm should detect those messages properly.
        # Otherwise, some peer could exit early, leaving others to timeout
        # errors or connection closed errors.
        ttl = 20
        n = self.rank + 1
        dst_rank = n % self.world_size

        multi_layer_nested_async_rpc(dst_rank, self.world_size, ttl)

    @dist_init
    def test_remote_with_exception(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        # check ref to other workers
        rref = rpc.remote(worker_name(dst_rank), raise_func)
        with self.assertRaises(ValueError):
            rref.to_here()
        # check ref to itself
        rref = rpc.remote(worker_name(self.rank), no_result, args=(10,))
        with self.assertRaises(TypeError):
            rref.to_here()

    @dist_init
    def test_rpc_return_rref(self):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref = rpc.rpc_sync(
            worker_name(dst_rank1),
            rpc_return_rref,
            args=(worker_name(dst_rank2),),
        )
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_rref_forward_chain(self):
        ttl = 8
        n = self.rank + 1
        dst_rank = n % self.world_size

        rref = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(n, n), 1))

        ret_rref = rref_forward_chain(dst_rank, self.world_size, rref, ttl)

        for _ in range(ttl):
            self.assertEqual(len(ret_rref), 1)
            ret_rref = ret_rref[0].to_here()

        ret = ret_rref
        self.assertEqual(ret, torch.add(torch.ones(n, n), 1))

    @dist_init
    def test_local_rref_no_fork(self):
        local_rref = RRef(35)
        self.assertEqual(local_rref.local_value(), 35)

    @dist_init
    def test_local_value_not_on_owner(self):
        # ensure that an error message is thrown if a user tries to call
        # local_value() on a non-owning node.
        next_rank = (self.rank + 1) % self.world_size
        rref = rpc.remote(
            worker_name(next_rank), torch.add, args=(torch.ones(1), torch.ones(1))
        )
        with self.assertRaisesRegex(
            RuntimeError,
            (
                rf"For UserRRef\(rref_id=GloballyUniqueId\(created_on={self.rank}, local_id=0\), "
                rf"fork_id=GloballyUniqueId\(created_on={self.rank}, local_id=1\)\), "
                r"can't call localValue\(\) on user "
                rf"WorkerInfo\(id={self.rank}, name={worker_name(self.rank)}\). "
                rf"Call it on owner WorkerInfo\(id={next_rank}, name={worker_name(next_rank)}\)"
            ),
        ):
            rref.local_value()

    @dist_init
    def test_return_local_rrefs(self):
        n = self.rank + 1
        dst_rank = n % self.world_size

        rref_list = rpc.rpc_sync(
            worker_name(dst_rank), get_rref_list, args=([1, 2, 3],)
        )

        for rref in rref_list:
            rpc.rpc_sync(
                rref.owner(),
                _call_method_on_rref,
                args=(MyClass.increment_value, rref, 10),
            )

        rets = [
            rpc.rpc_sync(
                rref.owner(), _call_method_on_rref, args=(MyClass.get_value, rref)
            )
            for rref in rref_list
        ]

        self.assertEqual(rets, [11, 12, 13])

    @dist_init
    def _test_rref_type(self, blocking):
        def launched_rpc(events):
            expected_name = f"rpc_{RPCExecMode.ASYNC.value}#_rref_typeof_on_owner"
            return any(e.name.startswith(expected_name) for e in events)

        dst = worker_name((self.rank + 1) % self.world_size)
        rref = rpc.remote(dst, torch.add, args=(torch.ones(2), 1))

        with _profile() as p:
            t = rref._get_type(blocking=blocking)
            if not blocking:
                t = t.wait()

        self.assertTrue(launched_rpc(p.function_events))
        expected_type = type(torch.ones(2))
        self.assertEqual(t, expected_type)

        futs = []

        def verify(fut):
            self.assertEqual(fut.value(), expected_type)

        with _profile() as p:
            for _ in range(10):
                t = rref._get_type(blocking=blocking)
                if not blocking:
 

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 18 class(es): StubRpcAgent, MyPickleClass, SlowPickleClass, MyClass, CustomException, TensorWrapper, AsyncExecutionClass, FooBackendOptions, MyEmbeddingBagModel, MyParameterServer, MyConvNetForMNIST, RpcTestCommon, RpcTest, MyModel, TestPickler, CudaRpcTest, TensorPipeAgentRpcTest, TensorPipeAgentCudaRpcTest

### Functions
This file defines 562 function(s): foo_add, udf_with_torch_ops, _increment_count, _reset_count, __init__, get_worker_infos, _stub_construct_rpc_backend_options_handler, _stub_init_rpc_backend_handler, set_value, wait_for_value_future, set_and_check_done, __init__, __getstate__, __setstate__, set, __init__, __getstate__, __setstate__, __init__, my_instance_method, my_class_method, my_static_method, increment_value, get_value, my_slow_method, _call_method_on_rref, get_rref_list, add_rref_to_value, run_nested_pickle, build_sparse_tensor


## Key Components

The file contains 16293 words across 6312 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 225880 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
