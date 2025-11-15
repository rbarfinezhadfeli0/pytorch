# Documentation: `docs/test/profiler/test_profiler.py_docs.md`

## File Metadata

- **Path**: `docs/test/profiler/test_profiler.py_docs.md`
- **Size**: 54,463 bytes (53.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/profiler/test_profiler.py`

## File Metadata

- **Path**: `test/profiler/test_profiler.py`
- **Size**: 128,715 bytes (125.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: profiler"]
# ruff: noqa: F841

import collections
import gc
import json
import mmap
import os
import pickle
import random
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from unittest.mock import patch

import expecttest

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall
from torch._inductor.utils import is_big_gpu
from torch.autograd.profiler import KinetoStepTracker, profile as _profile
from torch.autograd.profiler_legacy import profile as _profile_legacy
from torch.profiler import (
    _utils,
    DeviceType,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch.profiler._pattern_matcher import (
    Conv2dBiasFollowedByBatchNorm2dPattern,
    ExtraCUDACopyPattern,
    ForLoopIndexingPattern,
    FP32MatMulPattern,
    GradNotSetToNonePattern,
    MatMulDimInFP16Pattern,
    NamePattern,
    OptimizerSingleTensorPattern,
    Pattern,
    report_all_anti_patterns,
    SynchronizedDataLoaderPattern,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_JETSON,
    IS_LINUX,
    IS_WINDOWS,
    parametrize,
    run_tests,
    serialTest,
    skipIfTorchDynamo,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_CUDA,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TEST_XPU,
    TestCase,
)


if TYPE_CHECKING:
    from torch.autograd.profiler_util import FunctionEvent


# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerCUDA(TestCase):
    def test_mem_leak(self):
        """Checks that there's no memory leak when using profiler with CUDA"""
        t = torch.rand(1, 1).cuda()
        p = psutil.Process()
        last_rss = collections.deque(maxlen=5)
        for _ in range(10):
            with _profile(use_cuda=True):
                for _ in range(1024):
                    t = torch.mm(t, t)

            gc.collect()
            torch.cuda.empty_cache()
            last_rss.append(p.memory_info().rss)

        # with CUDA events leaking the increase in memory was ~7 MB between
        # profiler invocations above
        is_increasing = all(
            last_rss[idx] > last_rss[idx - 1] for idx in range(1, len(last_rss))
        )
        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])
        self.assertTrue(
            not (is_increasing and max_diff > 100 * 1024),
            msg=f"memory usage is increasing, {str(last_rss)}",
        )

    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        # Only testing that emit_nvtx runs when
        # record_shapes option is enabled.
        with torch.autograd.profiler.emit_nvtx(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_cudagraph_profiling_workaround(self):
        import subprocess

        # repro taken from #75504
        # Launch in a separate process to catch hanging/illegal memory errors
        # and to make sure CUPTI isn't already initialized.
        p = subprocess.check_call(
            [
                sys.executable,
                "-c",
                """
import os
import torch
from torch.profiler import ProfilerActivity, profile

def add_one(in_: torch.Tensor):
    return in_ + 1

sample_arg = torch.zeros(10, device="cuda").requires_grad_(True)

# add this before cuda graphs are created
torch.profiler._utils._init_for_cuda_graphs()

add_one_graphed = torch.cuda.graphs.make_graphed_callables(add_one, sample_args=(sample_arg,))
zeros = torch.zeros(10, device="cuda")
out = add_one_graphed(zeros)
assert out[0] == 1

with profile(activities=[ProfilerActivity.CPU]):
    add_one_graphed(zeros)

with profile(activities=[ProfilerActivity.CUDA]):
    add_one_graphed(zeros)
""",
            ],
            universal_newlines=True,
            timeout=60,
        )

        # ^ this will throw an exception if the script fails.


@unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
class TestProfilerITT(TestCase):
    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        # Only testing that emit_itt runs when
        # record_shapes option is enabled.
        with torch.autograd.profiler.emit_itt(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()


@instantiate_parametrized_tests
class TestProfiler(TestCase):
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    def test_source(self):
        """Checks that source code attribution works for eager, TS and autograd mode"""
        # avoid automatic inlining
        prev_opt = torch._C._get_graph_executor_optimize()
        torch._C._set_graph_executor_optimize(False)

        @torch.jit.script
        def ts_method_2(x, y):
            return torch.matmul(x, y)

        @torch.jit.script
        def ts_method_1(x, y, z):
            a = x + z
            w = ts_method_2(x, y) + a
            return w.sum()

        class DummyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )

            def forward(self, x):
                return self.conv(x)

        mod = DummyModule()

        def call_module(x):
            return mod(x)

        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            w = ts_method_1(x, y, z)
            v = 2 * w
            v.backward()
            a = torch.randn(2, 3, 2, 2, requires_grad=True)
            b = call_module(a)
            c = b.sum()
            c.backward()

        for e in p.function_events:
            if "aten::add" in e.name or "AddBackward" in e.name:
                self.assertTrue(any("test_profiler" in entry for entry in e.stack))
                self.assertTrue(
                    any(
                        (
                            "test_source" in entry
                            or "ts_method_1" in entry
                            or "ts_method_2" in entry
                        )
                        for entry in e.stack
                    )
                )

        if kineto_available():
            with TemporaryFileName(mode="w+") as fname:
                p.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]

                def extract(pattern: str):
                    matches = [e for e in events if re.search(pattern, e["name"])]
                    self.assertEqual(
                        len(matches), 1, repr([e["name"] for e in matches])
                    )
                    return matches[0]

                module_event = extract(r"DummyModule_0")
                wrapper_event = extract(r"call_module")
                self.assertEqual(
                    module_event["args"]["Python parent id"],
                    wrapper_event["args"]["Python id"],
                )

        torch._C._set_graph_executor_optimize(prev_opt)

    @parametrize(
        "name,thread_spec",
        {
            "basic": ((False, False),),
            "multiple_preexisting": ((False, False),) * 2,
            "open_in_scope": ((True, False),),
            "close_in_scope": ((False, True),),
            "complex": (
                # Large number of background threads
                (False, False),
                (False, False),
                (False, False),
                (False, False),
                # some of which finish during profiling
                (False, True),
                (False, True),
                # And the profiled section is also multithreaded
                (True, False),
                (True, True),
            ),
        }.items(),
        name_fn=lambda name, thread_spec: name,
    )
    @serialTest()
    @parametrize("work_in_main_thread", [True, False])
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_source_multithreaded(self, name, thread_spec, work_in_main_thread):
        """Test various threading configurations.

        `thread_spec` is a Tuple[Tuple[bool, bool], ...] where each pair is a
        thread. The first bool indicates if the thread should be started under
        the profiler context and the second is if it should be joined under the
        profiler context.
        """

        timeout = 15
        num_threads = len(thread_spec) + 1  # Main thread
        start_barrier = threading.Barrier(num_threads, timeout=timeout)
        end_barrier = threading.Barrier(num_threads, timeout=timeout)

        class Task(threading.Thread):
            def __init__(self) -> None:
                self._end_gate = threading.Event()
                super().__init__(daemon=True)
                self.start()
                self.finished = False

            def run(self):
                self._run(self._end_gate)

            def release(self):
                self._end_gate.set()

            @staticmethod
            def _run(end_gate=None):
                def known_preexisting_function():
                    start_barrier.wait()

                # Fixed point that we can use to test capture of functions
                # which are already running when profiling is enabled.
                known_preexisting_function()

                model = torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                )

                def invoked_during_run():
                    pass

                invoked_during_run()

                _ = model(torch.rand(4, 10))
                end_barrier.wait()

                if end_gate is not None:
                    end_gate.wait(timeout=timeout)

        threads = {}

        def add_threads(context: bool):
            for idx, (start_under_profiler, _) in enumerate(thread_spec):
                if start_under_profiler == context:
                    assert idx not in threads
                    threads[idx] = Task()

        def join_threads(context: bool):
            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                if end_under_profiler == context:
                    threads[idx].release()

            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                t = threads[idx]
                if end_under_profiler == context:
                    t.join(timeout=timeout)

        try:
            add_threads(False)
            with torch.profiler.profile(with_stack=True) as prof:
                # Threads added while the profiler are running will not be observed
                # since there is no way to hook into Python's thread start call to
                # register the observer. These are here purely to verify safety.
                add_threads(True)

                if work_in_main_thread:
                    Task._run()
                else:
                    start_barrier.wait()
                    end_barrier.wait()

                join_threads(True)
            join_threads(False)

        finally:
            # It is very important that we clean up everything because the
            # Python tracer will detect ALL active threads. (Even orphans from
            # prior failed tests.) If we don't clean up properly we can
            # contaminate subsequent tests.
            start_barrier.abort()
            end_barrier.abort()
            for t in threads.values():
                t.release()

            for t in threads.values():
                t.join(timeout=timeout)

            for t in threads.values():
                self.assertFalse(t.is_alive())

        roots = prof.profiler.kineto_results.experimental_event_tree()
        nodes = [
            node
            for node in _utils.traverse_dfs(roots)
            if isinstance(node.extra_fields, _ExtraFields_PyCall)
        ]
        tid_counts = collections.Counter([node.start_tid for node in nodes])

        prior_threads = sum(
            not start_under_profiler for start_under_profiler, _ in thread_spec
        )
        expected_threads = prior_threads + 1
        self.assertEqual(
            len(tid_counts), expected_threads, f"{expected_threads}, {tid_counts}"
        )
        self.assertEqual(len(nodes), sum(tid_counts.values()))

        # Profiler uses uint64_t max as a placeholder until TID can be determined.
        no_tid = 2**64 - 1
        self.assertFalse(no_tid in tid_counts)

        worker_threads = prior_threads + (1 if work_in_main_thread else 0)

        observed_preexisting = [
            node.start_tid
            for node in nodes
            if "known_preexisting_function" in node.name
        ]
        self.assertEqual(len(observed_preexisting), worker_threads)
        self.assertEqual(len(observed_preexisting), len(set(observed_preexisting)))

        observed_during_run = [
            node.start_tid for node in nodes if "invoked_during_run" in node.name
        ]
        self.assertEqual(len(observed_during_run), worker_threads)
        self.assertEqual(len(observed_during_run), len(set(observed_during_run)))

    def payload(self, use_cuda=False):
        x = torch.randn(10, 10)
        if use_cuda:
            x = x.cuda()
        y = torch.randn(10, 10)
        if use_cuda:
            y = y.cuda()
        z = torch.mm(x, y)
        z = z + y
        if use_cuda:
            z = z.cpu()

    def _check_stats(self, profiler_stats):
        self.assertGreater(profiler_stats.profiling_window_duration_sec, 0)
        self.assertGreater(profiler_stats.number_of_events, 0)
        self.assertGreater(profiler_stats.profiler_prepare_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_enable_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_disable_call_duration_us, 0)
        self.assertGreater(profiler_stats.parse_kineto_call_duration_us, 0)
        self.assertGreater(
            profiler_stats.function_events_build_tree_call_duration_us, 0
        )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_kineto(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with _profile(use_cuda=use_cuda, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        # rerun to avoid initial start overhead
        with _profile(use_cuda=use_cuda, use_kineto=True) as p:
            self.payload(use_cuda=use_cuda)

        self.assertTrue("aten::mm" in str(p))

        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
            row_limit=-1,
        )
        # print(output)
        found_gemm = False
        found_memcpy = False
        found_mm = False
        for e in p.function_events:
            if "aten::mm" in e.name:
                found_mm = True
            if "gemm" in e.name.lower() or "Cijk" in e.name:
                found_gemm = True
            if "memcpy" in e.name.lower() or "__amd_rocclr_copyBuffer" in e.name:
                found_memcpy = True
        if use_cuda:
            self.assertTrue(found_gemm)
            self.assertTrue(found_memcpy)
        else:
            self.assertTrue(found_mm)
        self._check_stats(p._stats)
        # p.export_chrome_trace("/tmp/test_trace.json")

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not TEST_MULTIGPU, "Multiple GPUs needed")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_kineto_multigpu(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for gpu_id in [0, 1]:
                x = torch.randn(10, 10).cuda(gpu_id)
                y = torch.randn(10, 10).cuda(gpu_id)
                z = x.matmul(y)

        found_gemm_0 = False
        found_gemm_1 = False
        found_cuda = False
        for evt in prof.events():
            if "gemm" in evt.name.lower() and evt.device_type == DeviceType.CUDA:
                if evt.device_index == 0:
                    found_gemm_0 = True
                elif evt.device_index == 1:
                    found_gemm_1 = True
            if "cuda" in evt.name.lower() and evt.device_type == DeviceType.CPU:
                found_cuda = True

        self.assertTrue(found_gemm_0)
        self.assertTrue(found_gemm_1)
        self.assertTrue(found_cuda)
        self._check_stats(prof._stats())

    def test_memory_profiler(self):
        def run_profiler(tensor_creation_fn):
            # collecting allocs / deallocs
            with _profile(
                profile_memory=True,
                record_shapes=True,
                use_kineto=kineto_available(),
            ) as prof:
                x = None
                with record_function("test_user_scope_alloc"):
                    x = tensor_creation_fn()
                with record_function("test_user_scope_dealloc"):
                    del x
            return prof.key_averages(group_by_input_shape=True)

        def check_metrics(stats, metric, allocs=None, deallocs=None):
            stat_metrics = {}
            # print(stats)
            for stat in stats:
                stat_metrics[stat.key] = getattr(stat, metric)
            # print(stat_metrics)
            if allocs is not None:
                for alloc_fn in allocs:
                    self.assertTrue(alloc_fn in stat_metrics)
                    self.assertGreater(
                        stat_metrics[alloc_fn], 0, f"alloc_fn = {alloc_fn}"
                    )
            if deallocs is not None:
                for dealloc_fn in deallocs:
                    self.assertTrue(dealloc_fn in stat_metrics)
                    self.assertLess(
                        stat_metrics[dealloc_fn], 0, f"alloc_fn = {dealloc_fn}"
                    )

        def create_cpu_tensor():
            return torch.rand(10, 10)

        def create_cuda_tensor():
            return torch.rand(10, 10).cuda()

        def create_xpu_tensor():
            return torch.rand(10, 10).xpu()

        def create_mkldnn_tensor():
            return torch.rand(10, 10, dtype=torch.float32).to_mkldnn()

        stats = run_profiler(create_cpu_tensor)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::empty",
                "aten::rand",
                "test_user_scope_alloc",
            ],
            deallocs=[
                "test_user_scope_dealloc",
            ],
        )

        if kineto_available():
            with TemporaryFileName(mode="w+") as fname:
                with profile(profile_memory=True) as prof:
                    x = None
                    with record_function("test_user_scope_alloc"):
                        x = create_cpu_tensor()
                    with record_function("test_user_scope_dealloc"):
                        del x
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    trace = json.load(f)
                    assert "traceEvents" in trace
                    events = trace["traceEvents"]
                    found_memory_events = False
                    for evt in events:
                        assert "name" in evt
                        if evt["name"] == "[memory]":
                            found_memory_events = True
                            assert "args" in evt
                            assert "Addr" in evt["args"]
                            assert "Device Type" in evt["args"]
                            assert "Device Id" in evt["args"]
                            assert "Bytes" in evt["args"]

                            # Memory should be an instantaneous event.
                            assert "dur" not in evt["args"]
                            assert "cat" not in evt["args"]
                    assert found_memory_events

        if torch.cuda.is_available():
            create_cuda_tensor()
            stats = run_profiler(create_cuda_tensor)
            check_metrics(
                stats,
                "device_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ],
            )

        if torch.xpu.is_available():
            create_xpu_tensor()
            stats = run_profiler(create_xpu_tensor)
            check_metrics(
                stats,
                "device_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ],
            )

        if torch.backends.mkldnn.is_available():
            create_mkldnn_tensor()
            stats = run_profiler(create_mkldnn_tensor)
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::rand",
                    "aten::empty",
                    "aten::to_mkldnn",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )

        # check top-level memory events
        with _profile(profile_memory=True, use_kineto=kineto_available()) as prof:
            x = torch.rand(10, 10)
            del x
            if torch.cuda.is_available():
                y = torch.rand(10, 10).cuda()
                del y
            elif torch.xpu.is_available():
                y = torch.rand(10, 10).to("xpu")
                del y
            gc.collect()
        stats = prof.key_averages(group_by_input_shape=True)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=["aten::rand", "aten::empty"],
            deallocs=["[memory]"],
        )
        if torch.cuda.is_available():
            check_metrics(stats, "device_memory_usage", deallocs=["[memory]"])
        elif torch.xpu.is_available():
            check_metrics(stats, "device_memory_usage", deallocs=["[memory]"])

    @unittest.skipIf(
        IS_JETSON, "Jetson has a guard against OOM since host and gpu memory are shared"
    )
    def test_oom_tracing(self):
        def run_profiler(tensor_creation_fn):
            with _profile(profile_memory=True, record_shapes=True) as prof:
                with self.assertRaisesRegex(RuntimeError, ".*[tT]ried to allocate.*"):
                    x = tensor_creation_fn()
                return prof

        def create_cuda_tensor_oom():
            device = torch.device("cuda:0")
            return torch.empty(
                1024, 1024, 1024, 1024, dtype=torch.float32, device=device
            )

        def check_trace(fname):
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                self.assertTrue("traceEvents" in trace)
                events = trace["traceEvents"]
                found_out_of_memory_events = False
                for evt in events:
                    self.assertTrue("name" in evt)
                    if evt["name"] == "[OutOfMemory]":
                        found_out_of_memory_events = True
                        self.assertTrue("args" in evt)
                        self.assertTrue("Device Type" in evt["args"])
                        self.assertTrue("Device Id" in evt["args"])
                        self.assertTrue("Bytes" in evt["args"])

                        # Memory should be an instantaneous event.
                        self.assertTrue("dur" not in evt["args"])
                        self.assertTrue("cat" not in evt["args"])
                self.assertTrue(found_out_of_memory_events)

        if torch.cuda.is_available():
            with TemporaryFileName(mode="w+") as fname:
                prof = run_profiler(create_cuda_tensor_oom)
                check_trace(fname)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_module_hierarchy(self):
        class A(nn.Module):
            def my_new_method(self, x):
                return x * 3

            def forward_impl_(self, x, y):
                return self.my_new_method(x) + y

            def forward(self, x, y):
                y = y - 2
                return self.forward_impl_(x, y)

        class B(nn.Module):
            def forward(self, x):
                return x + 2

        class C(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.A0 = A()
                self.B0 = B()

            def call_b(self, x):
                return self.B0.forward(x)

            def forward(self, x, y):
                return self.A0.forward(x, y) + self.call_b(x)

        model = C()
        model = torch.jit.script(model)
        input_a = torch.rand(128, 128)
        input_b = torch.rand(128, 128)
        op_to_module_hierarchy = {}
        op_to_module_hierarchy["aten::sub"] = ["TOP(C)::forward.A0(A)::forward."]
        op_to_module_hierarchy["aten::mul"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method."
        ]
        op_to_module_hierarchy["aten::add"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.",
            "TOP(C)::forward.SELF(C)::call_b.B0(B)::forward.",
            "TOP(C)::forward.",
        ]
        with TemporaryFileName(mode="w+") as fname:
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                with_modules=True,
            ) as prof:
                model(input_a, input_b)
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                assert "traceEvents" in trace
                events = trace["traceEvents"]
                found_memory_events = False
                for evt in events:
                    assert "name" in evt
                    if "args" in evt:
                        op_name = evt["name"]
                        if "Module Hierarchy" in evt["args"]:
                            hierarchy = evt["args"]["Module Hierarchy"]
                            if op_name in op_to_module_hierarchy:
                                assert hierarchy in op_to_module_hierarchy[op_name]

    def test_high_level_trace(self):
        """Checks that python side high level events are recorded."""

        class RepeatedDataset(torch.utils.data.Dataset):
            def __init__(self, N, D_in, D_out):
                self.N = N
                self.x = torch.randn(N, D_in)
                self.y = torch.randn(N, D_out)

            def __len__(self):
                return self.N

            def __getitem__(self, idx):
                return self.x, self.y

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)

            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        class CustomSGD(torch.optim.SGD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        def train():
            for data in dataloader:
                x, y = data[0], data[1]
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        N, D_in, H, D_out = 8, 10, 5, 2
        model = TwoLayerNet(D_in, H, D_out)
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        ds = RepeatedDataset(N, D_in, D_out)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

        try:
            train()
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")

        # Create multiple instances, expect each func is hooked only one time.
        # Nested wrappers(repeated patching) will make following test fail.
        optimizer_duplicate = torch.optim.SGD(model.parameters(), lr=1e-4)
        dataloader_duplicate = torch.utils.data.DataLoader(ds, batch_size=1)

        def judge(expected_event_count, prof):
            actual_event_count = {}
            for e in prof.function_events:
                if "#" in e.name:
                    key = e.name
                    if key in expected_event_count:
                        actual_event_count[key] = (
                            actual_event_count.setdefault(key, 0) + 1
                        )
            for key, count in expected_event_count.items():
                self.assertTrue(
                    (key in actual_event_count) and (count == actual_event_count[key])
                )

        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            # "+1" because the final iteration will enter __next__ but skip the loop body.
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#SGD.step": N,
            "Optimizer.zero_grad#SGD.zero_grad": N,
        }
        judge(expected_event_count, prof)

        # Test on pickle/unpickle. Expect to work in multi-processing.
        optimizer = pickle.loads(pickle.dumps(optimizer))
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        judge(expected_event_count, prof)

        # Test on customized optimizer.
        optimizer = CustomSGD(model.parameters(), lr=1e-4)
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#CustomSGD.step": N,
            "Optimizer.zero_grad#CustomSGD.zero_grad": N,
        }
        judge(expected_event_count, prof)

    def test_flops(self):
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        nested_tensor = torch.nested.nested_tensor(
            [torch.randn((2, 5)), torch.randn((3, 5))], layout=torch.jagged
        )
        with _profile(
            record_shapes=True, with_flops=True, use_kineto=kineto_available()
        ) as prof:
            model(inputs)
            # test that nested tensor won't cause exception during flop compute
            nested_tensor = nested_tensor + nested_tensor
        profiler_output = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
        self.assertRegex(profiler_output, "Total M?FLOPs")
        if not (kineto_available() and torch.cuda.is_available()):
            return

        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_flops=True,
        ) as kineto_profiler:
            model(inputs)
        profiler_output = kineto_profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1
        )
        self.assertRegex(profiler_output, "Total M?FLOPs")

    def test_override_time_units(self):
        US_IN_SECOND = 1000.0 * 1000.0
        US_IN_MS = 1000.0

        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        with _profile() as prof:
            model(inputs)

        profiler_output = prof.key_averages().table(time_unit="s")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_s = f"{event.cpu_time / US_IN_SECOND:.3f}s"
            cpu_time_total_str_s = f"{event.cpu_time_total / US_IN_SECOND:.3f}s"
            self.assertTrue(cpu_time_str_s in profiler_output)
            self.assertTrue(cpu_time_total_str_s in profiler_output)

        profiler_output = prof.key_averages().table(time_unit="ms")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_ms = f"{event.cpu_time / US_IN_MS:.3f}ms"
            cpu_time_total_str_ms = f"{event.cpu_time_total / US_IN_MS:.3f}ms"
            self.assertTrue(cpu_time_str_ms in profiler_output)
            self.assertTrue(cpu_time_total_str_ms in profiler_output)

        profiler_output = prof.key_averages().table(time_unit="us")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_us = f"{event.cpu_time:.3f}us"
            cpu_time_total_str_us = f"{event.cpu_time_total:.3f}us"
            self.assertTrue(cpu_time_str_us in profiler_output)
            self.assertTrue(cpu_time_total_str_us in profiler_output)

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    def test_kineto_profiler_api(self):
        called_num = [0]

        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with profile(activities=supported_activities()):
            self.payload(use_cuda=use_cuda)

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
                row_limit=-1,
            )
            # print(output)
            # p.export_chrome_trace("/tmp/test_trace_" + str(called_num[0]) + ".json")
            called_num[0] += 1

        initial_step = KinetoStepTracker.current_step()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=trace_handler,
        ) as p:
            for _ in range(8):
                self.payload(use_cuda=use_cuda)
                p.step()

        self.assertEqual(called_num[0], 2)
        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 8)

        # case without schedule
        with profile(activities=supported_activities()) as p:
            self.payload(use_cuda=use_cuda)
            self.payload(use_cuda=use_cuda)
        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
            row_limit=-1,
        )
        # print(output)

        test_schedule = torch.profiler.schedule(
            skip_first=3, wait=2, warmup=1, active=4, repeat=2
        )
        test_schedule_expected_outputs = [
            # skip first 3
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # ----
            # repeat No. 1 begin
            # wait 2
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 1 end
            # ---
            # repeat No. 2 begin
            # wait 2
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 2 end
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    def test_kineto_profiler_multiple_steppers(self):
        niters = 8
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        net = SimpleNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        opt.zero_grad()
        inputs = torch.rand(10)

        with profile(activities=supported_activities()):
            self.payload(use_cuda=use_cuda)

        def optimizer_step():
            """This simulates a step() hook in the optimizer"""
            KinetoStepTracker.increment_step("yet_another_step")

        initial_step = KinetoStepTracker.current_step()

        def run_batch():
            out = net(inputs)
            loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
            loss.backward()
            opt.step()
            # Manually call the hook. TODO: Remove this once we add the
            # profiler step hooks in the Optimizer class that will get triggered above.
            # See https://github.com/pytorch/pytorch/issues/88446
            optimizer_step()

        for _ in range(niters):
            run_batch()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        ) as p:
            for _ in range(niters):
                run_batch()
                p.step()

        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 2 * niters)

    def test_export_stacks(self):
        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)
            z = z + y

        with TemporaryFileName(mode="w+") as fname:
            p.export_stacks(fname)
            with open(fname) as f:
                lines = f.readlines()
            assert len(lines) > 0, "Empty stacks file"
            for line in lines:
                is_int = False
                try:
                    assert int(line.split(" ")[-1]) > 0, "Invalid stacks record"
                    is_int = True
                except ValueError:
                    pass
                assert is_int, "Invalid stacks record"

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_tensorboard_trace_handler(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with _profile(use_cuda=use_cuda, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname),
            ) as p:
                for _ in range(18):
                    self.payload(use_cuda=use_cuda)
                    p.step()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-4].isdigit() and int(parts[-4]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-3:], ["pt", "trace", "json"])
                file_num += 1
            self.assertEqual(file_num, 3)

        # test case for gzip file format
        with TemporaryDirectoryName() as dname:
            p = profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dname, use_gzip=True
                ),
            )
            p.start()
            for _ in range(18):
                self.payload(use_cuda=use_cuda)
                p.step()
            p.stop()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-5].isdigit() and int(parts[-5]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-4:], ["pt", "trace", "json", "gz"])
                file_num += 1
            self.assertEqual(file_num, 3)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_profiler_metadata(self):
        t1, t2 = torch.ones(1), torch.ones(1)
        with profile() as prof:
            torch.add(t1, t2)
            prof.add_metadata("test_key1", "test_value1")
            prof.add_metadata_json("test_key2", "[1,2,3]")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                assert "test_key1" in trace
                assert trace["test_key1"] == "test_value1"
                assert "test_key2" in trace
                assert trace["test_key2"] == [1, 2, 3]

    def _test_profiler_tracing(self, use_kineto):
        with _profile(use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1), torch.ones(1)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # read the trace and expect valid json
            # if the JSON generated by export_chrome_trace is not valid, this will throw and fail the test.
            with open(fname) as f:
                json.load(f)

        # test empty trace
        with _profile(use_kineto=use_kineto) as prof:
            pass
        # saving an empty trace
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            if use_kineto:
                with open(fname) as f:
                    contents = json.load(f)
                    # Some builds may not have logger observer
                    # so skip if not
                    if "WARNING" in contents:
                        found_empty_warning = False
                        for warning in contents["WARNING"]:
                            if "No Valid Trace Events" in warning:
                                found_empty_warning = True
                        self.assertTrue(found_empty_warning)

        # Same test but for cuda.
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        if not use_cuda:
            return

        device = torch.device("cuda:0")
        with _profile(use_cuda=True, use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # Now validate the json
            with open(fname) as f:
                json.load(f)

    def test_profiler_tracing(self):
        self._test_profiler_tracing(False)
        if kineto_available():
            self._test_profiler_tracing(True)

    def test_profiler_op_event_args(self):
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        with _profile(record_shapes=True) as prof:
            a = torch.ones((64, 32), dtype=torch.float32)
            c = torch.cat([a, a]).sin()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                for e in op_events:
                    args = e["args"]
                    if e["name"] == "aten::ones":
                        self.assertEqual(
                            args["Input type"],
                            ["ScalarList", "Scalar", "", "", "Scalar"],
                        )
                        self.assertEqual(
                            args["Concrete Inputs"], ["[64, 32]", "6", "", "", "False"]
                        )

                    if e["name"] == "aten::cat":
                        self.assertEqual(args["Input Dims"], [[[64, 32], [64, 32]], []])
                        self.assertEqual(args["Input type"], ["TensorList", "Scalar"])

                    # check that each op has record function id
                    self.assertGreaterEqual(
                        args.get("Record function id", -1),
                        0,
                        f"Failed finding record funciont for op = {e}",
                    )

    def test_profiler_strides(self):
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        base_tensor = torch.randn(1024, dtype=torch.float32)
        a = base_tensor.as_strided((16, 16), (17, 1), 0)
        b = base_tensor.as_strided((16, 16), (25, 2), 272)
        with _profile(record_shapes=True) as prof:
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/profiler`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/profiler`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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
- Contains **benchmarking** code or performance tests.

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
python docs/test/profiler/test_profiler.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/profiler`):

- [`test_record_function.py_kw.md_docs.md`](./test_record_function.py_kw.md_docs.md)
- [`profiler_utils_mock_events.json_docs.md_docs.md`](./profiler_utils_mock_events.json_docs.md_docs.md)
- [`test_profiler.py_kw.md_docs.md`](./test_profiler.py_kw.md_docs.md)
- [`test_torch_tidy.py_kw.md_docs.md`](./test_torch_tidy.py_kw.md_docs.md)
- [`test_memory_profiler.py_kw.md_docs.md`](./test_memory_profiler.py_kw.md_docs.md)
- [`test_cpp_thread.cpp_docs.md_docs.md`](./test_cpp_thread.cpp_docs.md_docs.md)
- [`test_profiler_tree.py_docs.md_docs.md`](./test_profiler_tree.py_docs.md_docs.md)
- [`test_kineto.py_docs.md_docs.md`](./test_kineto.py_docs.md_docs.md)
- [`test_execution_trace.py_kw.md_docs.md`](./test_execution_trace.py_kw.md_docs.md)
- [`test_cpp_thread.py_kw.md_docs.md`](./test_cpp_thread.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_profiler.py_docs.md_docs.md`
- **Keyword Index**: `test_profiler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
