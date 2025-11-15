# Documentation: `torch/testing/_internal/distributed/rpc/rpc_test.py`

## File Metadata

- **Path**: `torch/testing/_internal/distributed/rpc/rpc_test.py`
- **Size**: 225,880 bytes (220.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
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

    def _test_barrier_helper(self,
```



## High-Level Overview


This Python file contains 20 class(es) and 562 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StubRpcAgent`, `MyPickleClass`, `SlowPickleClass`, `MyClass`, `CustomException`, `TensorWrapper`, `AsyncExecutionClass`, `FooBackendOptions`, `MyEmbeddingBagModel`, `MyParameterServer`, `MyConvNetForMNIST`, `RpcTestCommon`, `RpcTest`, `MyModel`, `TestPickler`, `CudaRpcTest`, `TensorPipeAgentRpcTest`, `TensorPipeAgentCudaRpcTest`

**Functions defined**: `foo_add`, `udf_with_torch_ops`, `_increment_count`, `_reset_count`, `__init__`, `get_worker_infos`, `_stub_construct_rpc_backend_options_handler`, `_stub_init_rpc_backend_handler`, `set_value`, `wait_for_value_future`, `set_and_check_done`, `__init__`, `__getstate__`, `__setstate__`, `set`, `__init__`, `__getstate__`, `__setstate__`, `__init__`, `my_instance_method`

**Key imports**: concurrent.futures, contextlib, json, operator, os, sys, threading, time, namedtuple, partial


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `concurrent.futures`
- `contextlib`
- `json`
- `operator`
- `os`
- `sys`
- `threading`
- `time`
- `collections`: namedtuple
- `functools`: partial
- `unittest`: mock
- `torch`
- `torch.distributed as dist`
- `torch.distributed.autograd as dist_autograd`
- `torch.distributed.rpc as rpc`
- `torch.nn as nn`
- `torch.autograd.profiler_legacy`: profile as _profile
- `torch.distributed.rpc.api`: _thread_local_var, _use_rpc_pickler, _wait_all
- `torch.futures`: Future
- `torch.distributed.rpc.api as api`
- `torch.distributed.rpc.internal as internal`
- `datetime`: timedelta


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python torch/testing/_internal/distributed/rpc/rpc_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal/distributed/rpc`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`faulty_agent_rpc_test.py_docs.md`](./faulty_agent_rpc_test.py_docs.md)
- [`dist_autograd_test.py_docs.md`](./dist_autograd_test.py_docs.md)
- [`rpc_agent_test_fixture.py_docs.md`](./rpc_agent_test_fixture.py_docs.md)
- [`dist_optimizer_test.py_docs.md`](./dist_optimizer_test.py_docs.md)
- [`faulty_rpc_agent_test_fixture.py_docs.md`](./faulty_rpc_agent_test_fixture.py_docs.md)
- [`tensorpipe_rpc_agent_test_fixture.py_docs.md`](./tensorpipe_rpc_agent_test_fixture.py_docs.md)


## Cross-References

- **File Documentation**: `rpc_test.py_docs.md`
- **Keyword Index**: `rpc_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
