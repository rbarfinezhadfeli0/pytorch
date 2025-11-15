# Documentation: test_cuda.py

## File Metadata
- **Path**: `test/test_cuda.py`
- **Size**: 293745 bytes
- **Lines**: 7588
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: cuda"]
# ruff: noqa: F841

import contextlib
import ctypes
import gc
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import product
from random import randint

import psutil

import torch
import torch.cuda
import torch.nn as nn
from torch import inf, nan
from torch.cuda._memory_viz import (
    _profile_to_snapshot,
    profile_plot,
    segment_plot,
    trace_plot,
)
from torch.testing._internal.autocast_test_lists import AutocastTestLists, TestAutocast
from torch.testing._internal.common_cuda import (
    _create_scaling_case,
    SM70OrLater,
    TEST_CUDNN,
    TEST_MULTIGPU,
    tf32_on_and_off,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipCUDAIf,
)
from torch.testing._internal.common_optimizers import (
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    optims,
    TensorTracker,
)
from torch.testing._internal.common_utils import (
    cuda_python_error_check,
    EXPANDABLE_SEGMENTS,
    freeze_rng_state,
    gcIfJetson,
    get_cycles_per_ms,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_JETSON,
    IS_LINUX,
    IS_SANDCASTLE,
    IS_WINDOWS,
    IS_X86,
    load_tests,
    MI300_ARCH,
    parametrize,
    recover_orig_fp32_precision,
    run_tests,
    serialTest,
    setBlasBackendsToDefaultFinally,
    skipCUDAMemoryLeakCheckIf,
    skipCUDANonDefaultStreamIf,
    skipIfRocm,
    skipIfRocmArch,
    slowTest,
    subtest,
    TemporaryFileName,
    TEST_CUDA,
    TEST_CUDA_GRAPH,
    TEST_CUDA_PYTHON_BINDINGS,
    TEST_NUMPY,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.utils._triton import has_triton
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.viz._cycles import observe_tensor_cycles


requiresCppContext = unittest.skipUnless(
    IS_X86 and IS_LINUX, "cpp contexts are x86 linux only"
)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

try:
    import torchvision.models  # noqa: F401
    from torchvision.models import resnet18  # noqa: F401

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)
TEST_LARGE_TENSOR = TEST_CUDA
TEST_MEDIUM_TENSOR = TEST_CUDA
TEST_BF16 = False
TEST_PYNVML = (
    torch.cuda._HAS_PYNVML and not IS_JETSON
)  # nvml not fully supported on jetson
if TEST_CUDA:
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9
    TEST_MEDIUM_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 6e9
    TEST_BF16 = torch.cuda.is_bf16_supported()

_cycles_per_ms = None


@unittest.skipIf(not TEST_CUDA, "CUDA not available, skipping tests")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCuda(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    FIFTY_MIL_CYCLES = 50000000

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    @property
    def expandable_segments(self):
        return EXPANDABLE_SEGMENTS

    def test_pinned_memory_with_cudaregister(self):
        try:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
            )
            t = torch.ones(20)
            self.assertFalse(t.is_pinned())
            try:
                pinned_t = torch.ones(1 << 21).pin_memory()
                self.assertTrue(pinned_t.is_pinned())
                pinned_t = torch.ones(1 << 24).pin_memory()
                self.assertTrue(pinned_t.is_pinned())
            except RuntimeError as e:
                # Some GPUs don't support same address space on host and device side
                pass
        finally:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )

    def test_pinned_memory_with_cudaregister_multithread(self):
        num_threads = 4
        threads = [
            threading.Thread(target=self.test_pinned_memory_with_cudaregister)
            for t in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    @serialTest()
    def test_host_memory_stats(self):
        # Helper functions
        def empty_stats():
            return {
                "allocated_bytes.allocated": 0,
                "allocated_bytes.current": 0,
                "allocated_bytes.freed": 0,
                "allocated_bytes.peak": 0,
                "allocations.allocated": 0,
                "allocations.current": 0,
                "allocations.freed": 0,
                "allocations.peak": 0,
                "host_alloc_time.count": 0,
                "host_free_time.count": 0,
                "num_host_alloc": 0,
                "num_host_free": 0,
                "active_bytes.allocated": 0,
                "active_bytes.current": 0,
                "active_bytes.freed": 0,
                "active_bytes.peak": 0,
                "active_requests.allocated": 0,
                "active_requests.current": 0,
                "active_requests.freed": 0,
                "active_requests.peak": 0,
            }

        def check_stats(expected):
            stats = torch.cuda.host_memory_stats()
            for k, v in expected.items():
                if v != stats[k]:
                    print(f"key: {k}, expected: {v}, stats: {stats[k]}")
                self.assertEqual(v, stats[k])

        # Setup the test cleanly
        alloc1 = 10
        alloc1_aligned = 16
        alloc2 = 20
        alloc2_aligned = 32
        expected = empty_stats()

        # Reset any lingering state
        gc.collect()
        torch._C._host_emptyCache()

        # Check that stats are empty
        check_stats(expected)

        # Make first allocation and check stats
        t1 = torch.ones(alloc1 * 1024, pin_memory=True)
        self.assertTrue(t1.is_pinned())
        for prefix in ["active_requests", "allocations"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += 1

        allocation_size1 = alloc1_aligned * 1024 * 4
        for prefix in ["allocated_bytes", "active_bytes"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += allocation_size1

        expected["num_host_alloc"] += 1
        expected["host_alloc_time.count"] += 1

        check_stats(expected)

        # Make second allocation and check stats
        t2 = torch.ones(alloc2 * 1024, pin_memory=True)
        self.assertTrue(t2.is_pinned())
        for prefix in ["active_requests", "allocations"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += 1

        allocation_size2 = alloc2_aligned * 1024 * 4
        for prefix in ["allocated_bytes", "active_bytes"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += allocation_size2

        expected["num_host_alloc"] += 1
        expected["host_alloc_time.count"] += 1

        check_stats(expected)

        # Empty cache and check stats
        torch._C._host_emptyCache()

        check_stats(expected)

        # Finally, check the reset of peak and accumulated stats
        torch.cuda.reset_peak_host_memory_stats()
        torch.cuda.reset_accumulated_host_memory_stats()

        expected = empty_stats()

    def test_pinned_memory_empty_cache(self):
        try:
            for alloc_settings in (True, False):
                torch.cuda.memory._set_allocator_settings(
                    f"pinned_use_cuda_host_register:{alloc_settings}"
                )
                try:
                    t = torch.ones(1024 * 1024, pin_memory=True)
                    self.assertTrue(t.is_pinned())
                    del t
                    torch._C._host_emptyCache()
                except RuntimeError as e:
                    # Some GPUs don't support same address space on host and device side
                    pass
        finally:
            torch.cuda.memory._set_allocator_settings(
                "pinned_use_cuda_host_register:False"
            )

    def test_pinned_memory_use_background_threads(self):
        script = """
import torch

torch.cuda.memory._set_allocator_settings(
    f"pinned_use_background_threads:True"
)
t = torch.ones(1024 * 1024, pin_memory=True)
print(t.is_pinned())
"""
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True)
        self.assertEqual(proc.returncode, 0)

    def test_cudart_register(self):
        t = torch.ones(20)
        self.assertFalse(t.is_pinned())
        cudart = torch.cuda.cudart()
        r = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        self.assertEqual(r, 0)
        self.assertTrue(t.is_pinned())
        r = cudart.cudaHostUnregister(t.data_ptr())
        self.assertEqual(r, 0)
        self.assertFalse(t.is_pinned())

    def test_memory_allocation(self):
        gc.collect()
        torch.cuda.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            prev = torch.cuda.memory_allocated()
            mem = torch.cuda.caching_allocator_alloc(size)
            self.assertGreater(torch.cuda.memory_allocated(), prev)
        finally:
            if mem is not None:
                torch.cuda.caching_allocator_delete(mem)
                self.assertEqual(torch.cuda.memory_allocated(), prev)

    def test_memory_stats(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        prev_allocated = torch.accelerator.memory_allocated()
        prev_reserved = torch.accelerator.memory_reserved()
        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        self.assertEqual(prev_allocated, prev_max_allocated)
        self.assertEqual(prev_reserved, prev_max_reserved)
        # Activate 1kB memory
        prev_active_current = torch.accelerator.memory_stats()[
            "active_bytes.all.current"
        ]
        tmp = torch.randn(256, device="cuda")
        # Detect if the current active memory is 1kB
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            1024 + prev_active_current,
        )
        self.assertEqual(torch.accelerator.memory_stats()["active_bytes.all.freed"], 0)
        del tmp
        gc.collect()
        torch.accelerator.empty_cache()
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            prev_active_current,
        )
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.freed"], 1024
        )
        torch.accelerator.reset_peak_memory_stats()
        self.assertEqual(torch.accelerator.max_memory_allocated(), prev_max_allocated)
        self.assertEqual(torch.accelerator.max_memory_reserved(), prev_max_reserved)

    def test_check_error(self):
        # Assert this call doesn't raise.
        torch.cuda.check_error(0)

        with self.assertRaisesRegex(
            torch.cuda.CudaError, "out of memory|hipErrorOutOfMemory"
        ):
            torch.cuda.check_error(2)

    def test_cuda_get_device_name(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        device_name_None = torch.cuda.get_device_name(None)
        self.assertEqual(current_device_name, device_name_None)

        # Testing the behaviour for No argument
        device_name_no_argument = torch.cuda.get_device_name()
        self.assertEqual(current_device_name, device_name_no_argument)

    def test_cuda_get_device_capability(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_capability = torch.cuda.get_device_capability(current_device)
        device_capability_None = torch.cuda.get_device_capability(None)
        self.assertEqual(current_device_capability, device_capability_None)

        # Testing the behaviour for No argument
        device_capability_no_argument = torch.cuda.get_device_capability()
        self.assertEqual(current_device_capability, device_capability_no_argument)

    def test_cuda_get_device_properties(self):
        # Testing the behaviour with None as an argument
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        device_properties_None = torch.cuda.get_device_properties(None)
        self.assertEqual(current_device_properties, device_properties_None)

        # Testing the behaviour for No argument
        device_properties_no_argument = torch.cuda.get_device_properties()
        self.assertEqual(current_device_properties, device_properties_no_argument)

    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_out_of_memory(self):
        tensor = torch.zeros(1024, device="cuda")

        oom_regex = (
            "would exceed allowed memory"
            if TEST_CUDAMALLOCASYNC
            else f"Tried to allocate 800000000.00 GiB. GPU {tensor.device.index} has a total capacity of"
        )
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            torch.empty(1024 * 1024 * 1024 * 800000000, dtype=torch.int8, device="cuda")

        with self.assertRaisesRegex(
            RuntimeError, "Tried to allocate more than 1EB memory"
        ):
            torch.empty(
                1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device="cuda"
            )

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC or IS_JETSON, "Segmentation fault (core dumped)"
    )
    @serialTest()
    def test_out_of_memory_retry(self):
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        oom_regex = (
            "would exceed allowed memory"
            if TEST_CUDAMALLOCASYNC
            else "Tried to allocate"
        )
        size = int(total_memory * 0.5)
        a = torch.empty(size, dtype=torch.int8, device="cuda")
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            b = torch.empty(size, dtype=torch.int8, device="cuda")
        del a
        b = torch.empty(size, dtype=torch.int8, device="cuda")
        del b
        # We used a lot of memory here, clean up so we don't affect other tests too much
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @serialTest()
    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_set_per_process_memory_fraction(self):
        orig = torch.cuda.get_per_process_memory_fraction(0)
        torch.cuda.reset_peak_memory_stats(0)
        try:
            # test invalid fraction value.
            with self.assertRaisesRegex(TypeError, "Invalid type"):
                torch.cuda.set_per_process_memory_fraction(1)
            with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
                torch.cuda.set_per_process_memory_fraction(-0.1)
            with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
                torch.cuda.set_per_process_memory_fraction(2.0)

            tensor = torch.zeros(1024, device="cuda")
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(0.5, 0)

            # test 0.499 allocation is ok.
            application = int(total_memory * 0.499) - torch.cuda.max_memory_reserved()
            tmp_tensor = torch.empty(application, dtype=torch.int8, device="cuda")
            del tmp_tensor
            torch.cuda.empty_cache()

            application = int(total_memory * 0.5)
            # it will get OOM when try to allocate more than half memory.
            oom_regex = (
                "would exceed allowed memory"
                if TEST_CUDAMALLOCASYNC
                else "out of memory"
            )
            with self.assertRaisesRegex(RuntimeError, oom_regex):
                torch.empty(application, dtype=torch.int8, device="cuda")

            # ensure out of memory error doesn't disturb subsequent kernel
            tensor.fill_(1)
            self.assertTrue((tensor == 1).all())
        finally:
            torch.cuda.set_per_process_memory_fraction(orig, 0)

    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    @serialTest()
    def test_get_per_process_memory_fraction(self):
        # get the initial memory fraction
        init_fraction = torch.cuda.get_per_process_memory_fraction()

        # set and get the limiting cases
        torch.cuda.set_per_process_memory_fraction(1.0)
        self.assertEqual(torch.cuda.get_per_process_memory_fraction(), 1.0)
        torch.cuda.set_per_process_memory_fraction(0.0)
        self.assertEqual(torch.cuda.get_per_process_memory_fraction(), 0.0)

        # test a few random cases
        for val in torch.rand(3):
            torch.cuda.set_per_process_memory_fraction(float(val))
            self.assertEqual(torch.cuda.get_per_process_memory_fraction(), float(val))

        # restore the initial memory fraction
        torch.cuda.set_per_process_memory_fraction(init_fraction)

    def test_uuid(self):
        uuid = torch.cuda.get_device_properties(0).uuid
        self.assertEqual(len(str(uuid)), 36)  # xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        self.assertEqual(len(uuid.bytes), 16)

    def test_copy_non_blocking(self):
        def _test_copy_non_blocking(a, b):
            event = torch.cuda.Event()
            a.copy_(b, non_blocking=True)
            event.record()
            event.synchronize()
            self.assertEqual(a, b)

        # 10MB copies
        x = torch.ones(10000000, dtype=torch.uint8).cuda()
        y = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        y = torch.ones(10000000, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

        # Test the case where the pinned data_ptr is not equal to the storage data_ptr.
        x_base = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        x = x_base[1:]
        self.assertTrue(x.is_pinned())
        self.assertTrue(x_base.is_pinned())
        self.assertNotEqual(x_base.data_ptr(), x.data_ptr())
        self.assertEqual(x_base.storage().data_ptr(), x.storage().data_ptr())
        y = torch.ones(10000000 - 1, dtype=torch.uint8).cuda()
        _test_copy_non_blocking(x, y)

    def test_copy_non_blocking_type_conversion(self):
        a = torch.ones(1, device="cuda")
        b = torch.zeros(1, device="cpu", pin_memory=True)
        c = torch.empty(1, device="cuda", dtype=torch.long)
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        b.copy_(a, non_blocking=True)
        c.copy_(b, non_blocking=True)
        self.assertEqual(a, c, exact_dtype=False)

    @serialTest()
    def test_to_non_blocking(self):
        stream = torch.cuda.current_stream()

        def _test_to_non_blocking(a, non_blocking, dst):
            torch.cuda.synchronize()
            # Pushes an 0.1 second spin to stream so if the copy is non blocking,
            # stream will almost surely be active when we query().
            torch.cuda._sleep(int(100 * get_cycles_per_ms()))
            b = a.to(device=dst, non_blocking=non_blocking)
            self.assertEqual(stream.query(), not non_blocking)
            stream.synchronize()
            self.assertEqual(a, b)
            self.assertTrue(b.is_pinned() == (non_blocking and dst == "cpu"))

        for dst, try_non_blocking in product(("cuda", "cpu"), (True, False)):
            # Creates source on the opposite device from destination.
            src = torch.randn(
                1000000,
                device="cuda" if dst == "cpu" else "cpu",
                pin_memory=dst == "cuda",
            )
            _test_to_non_blocking(src, try_non_blocking, dst)

    def test_to_cpu_blocking_by_default(self):
        src = torch.randn(1000000, device="cuda")
        torch.cuda.synchronize()
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        dst = src.to(device="cpu")
        self.assertEqual(torch.cuda.current_stream().query(), True)
        self.assertEqual(src, dst)
        self.assertFalse(dst.is_pinned())

    def test_serialization_array_with_storage(self):
        x = torch.randn(5, 5).cuda()
        y = torch.IntTensor(2, 5).fill_(0).cuda()
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, atol=0, rtol=0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], atol=0, rtol=0)
        self.assertTrue(isinstance(q_copy[0], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[1], torch.cuda.IntTensor))
        self.assertTrue(isinstance(q_copy[2], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[3], torch.storage.TypedStorage))
        self.assertTrue(isinstance(q_copy[3]._untyped_storage, torch.UntypedStorage))
        q_copy[1].fill_(10)
        self.assertEqual(q_copy[3], torch.cuda.IntStorage(10).fill_(10))

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Does not work in fbcode yet")
    @setBlasBackendsToDefaultFinally
    def test_preferred_blas_library_settings(self):
        def _check_default():
            default = torch.backends.cuda.preferred_blas_library()
            if torch.version.cuda:
                # CUDA logic is easy, it's always cublas
                self.assertTrue(default == torch._C._BlasBackend.Cublas)
            else:
                # ROCm logic is less so, it's cublaslt for some Instinct, cublas for all else
                gcn_arch = str(
                    torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
                )
                if gcn_arch in ["gfx90a", "gfx942", "gfx950"]:
                    self.assertTrue(default == torch._C._BlasBackend.Cublaslt)
                else:
                    self.assertTrue(default == torch._C._BlasBackend.Cublas)

        _check_default()
        # "Default" can be set but is immediately reset internally to the actual default value.
        self.assertTrue(
            torch.backends.cuda.preferred_blas_library("default")
            != torch._C._BlasBackend.Default
        )
        _check_default()
        self.assertTrue(
            torch.backends.cuda.preferred_blas_library("cublas")
            == torch._C._BlasBackend.Cublas
        )
        self.assertTrue(
            torch.backends.cuda.preferred_blas_library("hipblas")
            == torch._C._BlasBackend.Cublas
        )
        # check bad strings
        with self.assertRaisesRegex(
            RuntimeError,
            "Unknown input value. Choose from: default, cublas, hipblas, cublaslt, hipblaslt, ck.",
        ):
            torch.backends.cuda.preferred_blas_library("unknown")
        # check bad input type
        with self.assertRaisesRegex(RuntimeError, "Unknown input value type."):
            torch.backends.cuda.preferred_blas_library(1.0)
        # check env var override
        custom_envs = [
            {"TORCH_BLAS_PREFER_CUBLASLT": "1"},
            {"TORCH_BLAS_PREFER_HIPBLASLT": "1"},
        ]
        test_script = "import torch;print(torch.backends.cuda.preferred_blas_library())"
        for env_config in custom_envs:
            env = os.environ.copy()
            for key, value in env_config.items():
                env[key] = value
            r = (
                subprocess.check_output([sys.executable, "-c", test_script], env=env)
                .decode("ascii")
                .strip()
            )
            self.assertEqual("_BlasBackend.Cublaslt", r)

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled for async")
    @setBlasBackendsToDefaultFinally
    def test_cublas_workspace_explicit_allocation(self):
        torch.backends.cuda.preferred_blas_library("cublas")
        a = torch.randn(7, 7, device="cuda", requires_grad=False)
        if torch.version.hip:
            default_workspace_size = 1024 * 32 * 1024  # :1024:32  32MiB
            # different size (128 MiB) expected on MI300 GPU
            gcn_arch = str(
                torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
            )
            if "gfx94" in gcn_arch or "gfx95" in gcn_arch:
                default_workspace_size = 1024 * 128 * 1024  # :1024:128
        else:
            default_workspace_size = (
                4096 * 2 * 1024 + 16 * 8 * 1024
            )  # :4096:2:16:8  8MiB
            # different size (32 MiB) expected on Hopper GPU
            if torch.cuda.get_device_capability() == (9, 0):
                default_workspace_size = 4096 * 8 * 1024

        def check_workspace_size(inp):
            torch._C._cuda_clearCublasWorkspaces()
            start = torch.cuda.memory_stats()["active_bytes.all.allocated"]
            with torch.no_grad():
                torch.matmul(inp, inp)
            finish = torch.cuda.memory_stats()["active_bytes.all.allocated"]
            return finish - start

        # check default
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ""
        self.assertTrue(abs(check_workspace_size(a) - default_workspace_size) < 524288)

        # check default with bad user config
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "-1"
        self.assertTrue(abs(check_workspace_size(a) - default_workspace_size) < 524288)

        # check valid config
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":128:8:64:16:32:32"
        self.assertTrue(abs(check_workspace_size(a) - (3072 * 1024)) < 524288)

        torch._C._cuda_clearCublasWorkspaces()

    def test_cublas_allow_tf32_get_set(self):
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        if skip_tf32_cublas:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            return

        orig = torch.backends.cuda.matmul.allow_tf32
        self.assertEqual(torch._C._get_cublas_allow_tf32(), orig)
        torch.backends.cuda.matmul.allow_tf32 = not orig
        self.assertEqual(torch._C._get_cublas_allow_tf32(), not orig)
        torch.backends.cuda.matmul.allow_tf32 = orig

    def test_float32_matmul_precision_get_set(self):
        orig = torch.get_float32_matmul_precision()
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        # this is really just checking that the environment variable is respected during testing
        # and not overwritten by another function that doesn't revert it to the initial value
        if not skip_tf32_cublas:
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        else:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

        for p in ("medium", "high"):
            torch.set_float32_matmul_precision(p)
            self.assertEqual(torch.get_float32_matmul_precision(), p)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision(orig)

    def test_cublas_allow_fp16_reduced_precision_reduction_get_set(self):
        orig = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        orig_splitk = (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction_split_k
        )
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (orig, orig_splitk),
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not orig
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (not orig, True),
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            False,
            False,
        )
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (False, False),
        )
        with self.assertRaisesRegex(RuntimeError, "allow_splitk=False"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                True,
                False,
            )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            orig,
            orig_splitk,
        )

    def test_cublas_allow_bf16_reduced_precision_reduction_get_set(self):
        orig = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        orig_splitk = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k
        )
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            (orig, orig_splitk),
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not orig
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            (not orig, True),
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            False,
            False,
        )
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            (False, False),
        )
        with self.assertRaisesRegex(RuntimeError, "allow_splitk=False"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                True,
                False,
            )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            orig,
            orig_splitk,
        )

    def test_cublas_allow_fp16_accumulation_get_set(self):
        orig = torch.backends.cuda.matmul.allow_fp16_accumulation
        self.assertEqual(torch._C._get_cublas_allow_fp16_accumulation(), orig)
        torch.backends.cuda.matmul.allow_fp16_accumulation = not orig
        self.assertEqual(torch._C._get_cublas_allow_fp16_accumulation(), not orig)
        torch.backends.cuda.matmul.allow_fp16_accumulation = orig

    def test_cudnn_allow_tf32_get_set(self):
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=False
        ):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            self.assertTrue(torch.backends.cudnn.allow_tf32)

    @recover_orig_fp32_precision
    def test_fp32_precision_with_tf32(self):
        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            benchmark_limit=None,
            deterministic=None,
            allow_tf32=True,
            fp32_precision="none",
        ):
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "tf32")
            self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "tf32")

        with torch.backends.cudnn.flags(
            enabled=None,
            benchmark=None,
            benchmark_limit=None,
            deterministic=None,
            allow_tf32=False,
            fp32_precision="none",
        ):
            self.assertEqual(torch.backends.cudnn.conv.fp32_precision, "none")
            self.assertEqual(torch.backends.cudnn.rnn.fp32_precision, "none")

    @recover_orig_fp32_precision
    def test_fp32_precision_with_float32_matmul_precision(self):
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "ieee")
        torch.set_float32_matmul_precision("high")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        torch.set_float32_matmul_precision("medium")
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")

    @recover_orig_fp32_precision
    def test_invalid_status_for_legacy_api(self):
        torch.backends.cudnn.conv.fp32_precision = "none"
        torch.backends.cudnn.rnn.fp32_precision = "tf32"
        with self.assertRaisesRegex(RuntimeError, "mix of the legacy and new APIs"):
            print(torch.backends.cudnn.allow_tf32)

        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        with self.assertRaisesRegex(RuntimeError, "mix of the legacy and new APIs"):
            print(torch.get_float32_matmul_precision())

        if not TEST_WITH_ROCM:
            with self.assertRaisesRegex(RuntimeError, "mix of the legacy and new APIs"):
                print(torch.backends.cuda.matmul.allow_tf32)

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().double(), torch.cuda.DoubleTensor)
        self.assertIsInstance(x.cuda().float(), torch.cuda.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu(), torch.FloatTensor)
        self.assertIsInstance(x.cuda().float().cpu().int(), torch.IntTensor)

        y = x.storage()
        self.assertIsInstance(y.float(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().double(), torch.cuda.DoubleStorage)
        self.assertIsInstance(y.cuda().float(), torch.cuda.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu(), torch.FloatStorage)
        self.assertIsInstance(y.cuda().float().cpu().int(), torch.IntStorage)

    @unittest.skip("was disabled due to not enough memory, but actually it always fail")
    def test_arithmetic_large_tensor(self):
        x = torch.empty(2**30, device="cuda")

        x.fill_(1)
        self.assertEqual(x.sum(), 2**30)

        x += 1
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x -= 0.5
        self.assertEqual(x.sum(), 2**29)

        x.fill_(1)
        x *= 2
        self.assertEqual(x.sum(), 2**31)

        x.fill_(1)
        x /= 2
        self.assertEqual(x.sum(), 2**29)

    def test_gather_bool(self):
        t = torch.tensor([[False, True], [True, True]], device="cuda")
        self.assertEqual(
            torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]], device="cuda")),
            torch.tensor([[False, False], [True, True]], device="cuda"),
        )

    def test_torch_manual_seed_seeds_cuda_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().cuda()
            torch.cuda.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.cuda.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch.cuda.initial_seed(), 2)

    def test_specify_improper_device_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "tempfile.pt")
            with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
                torch.save(
                    [torch.nn.Parameter(torch.randn(10, 10))],
                    fname,
                    _use_new_zipfile_serialization=True,
                )
                torch.load(fname, "cuda0")

    def test_get_device_index(self):
        from torch.cuda._utils import _get_device_index

        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            _get_device_index("cuda0", optional=True)

        with self.assertRaisesRegex(ValueError, "Expected a cuda device"):
            cpu_device = torch.device("cpu")
            _get_device_index(cpu_device, optional=True)

    def test_serialization_array_with_empty(self):
        x = [torch.randn(4, 4).cuda(), torch.cuda.FloatTensor()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), original.get_device())

    @skipCUDANonDefaultStreamIf(True)
    def test_streams(self):
        default_stream = torch.cuda.current_stream()
        user_stream = torch.cuda.Stream()
        self.assertEqual(torch.cuda.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        self.assertEqual(default_stream.cuda_stream, 0)
        self.assertNotEqual(user_stream.cuda_stream, 0)
        with torch.cuda.stream(user_stream):
            self.assertEqual(torch.cuda.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        tensor1 = torch.ByteTensor(5).pin_memory()
        default_stream.synchronize()
        self.assertTrue(default_stream.query())

    def test_stream_event_repr(self):
        s = torch.cuda.current_stream()
        self.assertTrue("torch.cuda.Stream" in s.__repr__())
        e = torch.cuda.Event()
        self.assertTrue("torch.cuda.Event" in e.__repr__())
        s.record_event(e)
        self.assertTrue("torch.cuda.Event" in e.__repr__())

    def test_cuda_stream_protocol(self):
        stream = torch.cuda.Stream()

        self.assertTrue(hasattr(stream, "__cuda_stream__"))

        result = stream.__cuda_stream__()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 0)  # Protocol version
        self.assertEqual(result[1], stream.cuda_stream)  # Stream handle

        external_stream = torch.cuda.ExternalStream(stream.cuda_stream)
        external_result = external_stream.__cuda_stream__()

        self.assertEqual(external_result[0], 0)
        self.assertEqual(external_result[1], external_stream.cuda_stream)

    def test_events(self):
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event(enable_timing=True)
        self.assertTrue(event.query())
        start_event = torch.cuda.Event(enable_timing=True)
        stream.record_event(start_event)
        torch.cuda._sleep(int(50 * get_cycles_per_ms()))
        stream.record_event(event)
        self.assertFalse(event.query())
        event.synchronize()
        self.assertTrue(event.query())
        self.assertGreater(start_event.elapsed_time(event), 0)

        event = torch.cuda.Event(enable_timing=True)
        self.assertEqual(event.cuda_event, 0)
        self.assertEqual(event.event_id, 0)

        event.record()
        self.assertNotEqual(event.cuda_event, 0)
        self.assertNotEqual(event.event_id, 0)
        self.assertEqual(event.cuda_event, event.event_id)

    def test_events_elapsedtime(self):
        event1 = torch.cuda.Event(enable_timing=False)
        event2 = torch.cuda.Event(enable_timing=False)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

        event1 = torch.cuda.Event(enable_timing=True)
        event2 = torch.cuda.Event(enable_timing=True)
        with self.assertRaisesRegex(
            ValueError, "Both events must be recorded before calculating elapsed time"
        ):
            event1.elapsed_time(event2)

        # check default value of enable_timing: False
        event1 = torch.cuda.Event()
        event2 = torch.cuda.Event()
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

    def test_generic_stream_event(self):
        stream = torch.Stream("cuda")
        self.assertEqual(stream.device_index, torch.cuda.current_device())
        cuda_stream = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        self.assertIsInstance(cuda_stream, torch.Stream)
        self.assertTrue(issubclass(type(cuda_stream), torch.Stream))
        self.assertTrue(torch.Stream in type(cuda_stream).mro())
        self.assertEqual(stream.stream_id, cuda_stream.stream_id)
        self.assertNotEqual(stream.stream_id, torch.cuda.current_stream().stream_id)

        event1 = torch.Event("cuda", enable_timing=True)
        event2 = torch.Event("cuda", enable_timing=True)
        self.assertEqual(event1.event_id, 0)
        a = torch.randn(1000)
        b = torch.randn(1000)
        with torch.cuda.stream(cuda_stream):
            a_cuda = a.to("cuda", non_blocking=True)
            b_cuda = b.to("cuda", non_blocking=True)
            self.assertEqual(stream.stream_id, torch.cuda.current_stream().stream_id)
        event1.record(stream)
        event1.synchronize()
        self.assertTrue(event1.query())
        c_cuda = a_cuda + b_cuda
        event2.record()
        event2.synchronize()
        self.assertTrue(event2.query())
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertEqual(c_cuda.cpu(), a + b)
        self.assertTrue(event1.elapsed_time(event2) > 0)
        cuda_event = torch.cuda.Event()
        self.assertIsInstance(cuda_event, torch.Event)
        self.assertTrue(issubclass(type(cuda_event), torch.Event))
        self.assertTrue(torch.Event in type(cuda_event).mro())

    def test_stream_compatibility(self):
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s1.stream_id)
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s2.stream_id)
        with self.assertRaisesRegex(
            RuntimeError, "Device index value .* is out of index range"
        ):
            torch.accelerator.current_stream(torch.accelerator.device_count())

    def test_record_stream(self):
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
        result = torch.cuda.FloatTensor(t.size())
        stream = torch.cuda.Stream()
        ptr = [None]

        # Performs the CPU->GPU copy in a background stream
        def perform_copy():
            with torch.cuda.stream(stream):
                tmp = t.cuda(non_blocking=True)
                ptr[0] = tmp.data_ptr()
            torch.cuda.current_stream().wait_stream(stream)
            tmp.record_stream(torch.cuda.current_stream())
            torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            result.copy_(tmp)

        perform_copy()
        with torch.cuda.stream(stream):
            tmp2 = torch.cuda.FloatTensor(t.size())
            tmp2.zero_()
            self.assertNotEqual(
                tmp2.data_ptr(), ptr[0], msg="allocation reused to soon"
            )

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        if not TEST_CUDAMALLOCASYNC:
            # In the native allocator, we expect "tmp"'s side-stream-tagged block will be reused
            # in that side stream after result.copy_(tmp) in the main stream finishes.
            torch.cuda.current_stream().synchronize()
            with torch.cuda.stream(stream):
                tmp3 = torch.cuda.FloatTensor(t.size())
                self.assertEqual(tmp3.data_ptr(), ptr[0], msg="allocation not reused")

    def test_record_stream_on_shifted_view(self):
        # See issue #27366

        # This test detects unexpected block reallocation. For reliable test,
        # the stream to allocate tensors is isolated. The allocator will not
        # reuse free blocks which were allocated from another stream.
        stream_alloc = torch.cuda.Stream()
        with torch.cuda.stream(stream_alloc):
            base = torch.cuda.FloatTensor([10, 10])

        # Record another stream on a shifted view tensor.
        view = base[5:]
        self.assertTrue(view.storage_offset() > 0)

        stream_record = torch.cuda.Stream()
        with torch.cuda.stream(stream_record):
            torch.cuda._busy_wait_for_flag()

        view.record_stream(stream_record)

        # Delete those tensors to make the block free soon.
        data_ptr = base.data_ptr()
        del base, view

        # A new tensor should not be allocated to the block above.
        stream_alloc.synchronize()

        with torch.cuda.stream(stream_alloc):
            try_realloc = torch.cuda.FloatTensor([10, 10])
        torch.cuda._clear_flag()

        self.assertNotEqual(try_realloc.data_ptr(), data_ptr)

    def test_device_context_manager(self):
        prev_device = torch.cuda.current_device()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.cuda.current_device(), prev_device)
        self.assertEqual(torch.cuda.current_device(), prev_device)
        with torch.accelerator.device_index(0):
            self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.cuda.current_device(), prev_device)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_multi_device_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.cuda.set_device(src_device)
        with torch.accelerator.device_index(dst_device):
            self.assertEqual(torch.cuda.current_device(), 1)
        self.assertEqual(torch.cuda.current_device(), src_device)

    def test_stream_context_manager(self):
        prev_stream = torch.cuda.current_stream()
        with torch.cuda.Stream() as stream:
            self.assertEqual(stream, torch.cuda.current_stream())
        self.assertEqual(prev_stream, torch.cuda.current_stream())

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_multi_device_stream_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.cuda.set_device(src_device)
        src_prev_stream = torch.cuda.current_stream(src_device)
        dst_prev_stream = torch.cuda.current_stream(dst_device)
        with torch.cuda.Stream(dst_device) as dst_stream:
            self.assertEqual(dst_device, torch.cuda.current_device())
            self.assertEqual(dst_stream, torch.cuda.current_stream())
            self.assertEqual(src_prev_stream, torch.cuda.current_stream(src_device))
        self.assertEqual(src_device, torch.cuda.current_device())
        self.assertEqual(src_prev_stream, torch.cuda.current_stream())
        self.assertEqual(dst_prev_stream, torch.cuda.current_stream(dst_device))

    def test_noncontiguous_pinned_memory(self):
        # See issue #3266
        x = torch.arange(0, 10).view((2, 5))
        self.assertEqual(x.t(), x.t().pin_memory())

    def test_caching_pinned_memory(self):
        cycles_per_ms = get_cycles_per_ms()

        # check that allocations are reused after deletion
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t.data_ptr(), ptr, msg="allocation not reused")

        # check that the allocation is not reused if it's in-use by a copy
        gpu_tensor = torch.cuda.FloatTensor([0])
        torch.cuda._sleep(int(1000 * cycles_per_ms))  # delay the copy by 1s
        gpu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, msg="allocation reused too soon")
        self.assertEqual(list(gpu_tensor), [1])

    def test_caching_allocator_record_stream_oom(self):
        """allocations delayed by a record_stream call should still be freed on
        an out-of-memory in cuda_malloc_retry. see issue #19219"""
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            y = torch.zeros(40 * 1024 * 1024, device="cuda")

        for _ in range(100):
            x = torch.empty(40 * 1024 * 1024, device="cuda")
            with torch.cuda.stream(stream):
                y += x
            # delays reuse of `x` until after all operations in `stream`
            x.record_stream(stream)
            del x

        # we've made a mess by allocating up to the device capacity. free any
        # cached blocks in case it affects future tests.
        torch.cuda.empty_cache()

    # Tests for historic illegal memory access, see #17040.
    def test_reduction_gpu_memory_accessing(self):
        x = torch.ones(512, 8, dtype=torch.float32, device="cuda")
        torch.sum(x, 0)

    def test_sum_fp16(self):
        x = torch.zeros(10, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(), 0)

        x = torch.ones(65504, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(), 65504)
        self.assertEqual(x.sum(dtype=torch.float32), 65504)

        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum(dtype=torch.float32), 65536)

        a = torch.zeros(1203611).bernoulli_(0.0005)
        x = a.to(device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum().item(), a.sum().item())

        a = torch.zeros(100, 121, 80).bernoulli_(0.0005)
        x = a.to(device="cuda", dtype=torch.float16)
        self.assertEqual(x.sum((0, 2)).float().cpu(), a.sum((0, 2)))

    def test_mean_fp16(self):
        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.mean(), 1)

        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        self.assertEqual(x.mean(dtype=torch.float32), 1)

    def test_prod_large(self):
        # tests global reduction (should_global_reduce = true) in case of non-zero identity element
        x = torch.ones(240000, device="cuda", dtype=torch.float32)
        self.assertEqual(x.prod(), 1)

        # test for complex types. Note 240k is divisible by 4
        for dtype in [torch.cfloat, torch.cdouble]:
            x = torch.ones(240000, device="cuda", dtype=dtype) * (0 + 1j)
            self.assertEqual(x.prod(), 1)

    def test_multinomial_ext(self):
        # Test two corner cases from older PyTorch (Issue #4858)
        freqs = torch.cuda.FloatTensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.03178183361887932,
                0.027680952101945877,
                0.033176131546497345,
                0.046052902936935425,
                0.07742464542388916,
                0.11543981730937958,
                0.14148041605949402,
                0.15784293413162231,
                0.13180233538150787,
                0.08271478116512299,
                0.049702685326337814,
                0.027557924389839172,
                0.018125897273421288,
                0.011851548217236996,
                0.010252203792333603,
                0.007422595750540495,
                0.005372154992073774,
                0.0045109698548913,
                0.0036087757907807827,
                0.0035267581697553396,
                0.0018864056328311563,
                0.0024605290964245796,
                0.0022964938543736935,
                0.0018453967059031129,
                0.0010662291897460818,
                0.0009842115687206388,
                0.00045109697384759784,
                0.0007791675161570311,
                0.00020504408166743815,
                0.00020504408166743815,
                0.00020504408166743815,
                0.00012302644609007984,
                0.0,
                0.00012302644609007984,
                4.100881778867915e-05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        torch.cuda.manual_seed(11042)
        sample = torch.multinomial(freqs, 1000, True)
        self.assertNotEqual(freqs[sample].min(), 0)

        p = torch.zeros(3421, 2, device="cuda", dtype=torch.float)
        p[:, 1] = 1
        torch.cuda.manual_seed(5214)
        r = torch.multinomial(p, 1)
        self.assertNotEqual(r.min().item(), 0)

        # test corner case from Issue #13867
        torch.cuda.manual_seed(33)
        probs = torch.randn(1000000, device="cuda").clamp(min=0) * 3e-5
        samples = probs.multinomial(1000000, replacement=True)
        self.assertGreater(probs[samples].min().item(), 0)

    def _spawn_test_multinomial_invalid_probs_cuda(self, probs):
        import subprocess

        try:
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"""\
import sys
import torch
from torch import inf, nan
try:
    with torch.random.fork_rng(devices=[0]):
        torch.multinomial(torch.tensor({probs}).to('cuda'), 2, replacement=True)
        torch.cuda.synchronize()
    sys.exit(-1) # Should not be reached
except RuntimeError as e:
    sys.exit(-2)
""",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            out, err = p.communicate(timeout=10)
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        expected_messages = [
            "device-side assert triggered",  # CUDA
            "Assertion",  # CUDA
            "HSA_STATUS_ERROR_EXCEPTION",  # ROCm
            "Device-side assertion",  # ROCm
        ]
        self.assertTrue(any(msg in out or msg in err for msg in expected_messages))

    @slowTest
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    def test_multinomial_invalid_probs_cuda(self):
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, -1.0, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, inf, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, -inf, 1.0])
        self._spawn_test_multinomial_invalid_probs_cuda([1.0, 1.0, nan])

    @staticmethod
    def _mute_init():
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stderr.fileno())

    def _spawn_method(self, method, arg):
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(1, initializer=self._mute_init) as pool:
            errors = pool.map(method, [arg])
            for e in errors:
                if "device-side assert triggered" not in str(e):
                    self.fail(e)
                if e.error_code != 710:  # cudaErrorAssert == 710
                    self.fail(e)

    @staticmethod
    def _test_index_bounds_cuda(idx):
        x = torch.arange(10, device="cuda")
        try:
            y = x[torch.tensor([idx])]
            return f"x[torch.tensor([{idx})]={y}"
        except RuntimeError as err:
            return err

    @slowTest
    @skipIfRocm
    def test_index_out_of_bounds_exception_cuda(self):
        test_method = TestCuda._test_index_bounds_cuda
        # Test in-bound access works fine
        self.assertEqual(
            test_method(1), "x[torch.tensor([1)]=tensor([1], device='cuda:0')"
        )
        # Test that indexing out of bounds causes assert
        self._spawn_method(test_method, 11)

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @serialTest()
    def test_huge_index(self):
        src = torch.empty(15000000, 45, device="cuda", dtype=torch.long).random_(
            0, 2**22
        )
        idx = torch.randperm(src.shape[0], device="cuda")
        res = src[idx]
        res_cpu = src.cpu()[idx.cpu()]
        self.assertEqual(res.cpu(), res_cpu)

    def test_randint_randomness_for_large_range(self) -> None:
        # For large ranges, randint generation is slightly different. This lead to a subtle bug where some Philox
        # offsets were not calculated correctly, resulting in reused random states.
        # See https://github.com/pytorch/pytorch/issues/125224
        size = 1_000_000
        high = 6_000_000_000  # Keep this above 2**32

        def run(dev: torch.device) -> int:
            # Measure how many unique numbers are generated in 2 consecutive calls to randint. If random states are
            # reused, this will yield fewer unique numbers.
            gen = torch.Generator(device=dev)
            gen.manual_seed(0)
            t1 = torch.randint(
                0, high, [size], device=dev, generator=gen, dtype=torch.int64
            )
            t2 = torch.randint(
                0, high, [size], device=dev, generator=gen, dtype=torch.int64
            )
            return torch.stack([t1, t2]).unique().shape[0]

        # Use CPU as reference. The results should not deviate too much.
        self.assertTrue(
            abs(run(torch.device("cuda")) - run(torch.device("cpu"))) < 10_000
        )

    @largeTensorTest("20GB", "cuda")
    @serialTest()
    def test_randint_generation_for_large_numel(self) -> None:
        numel = 2**31 + 1
        s = torch.randint(2, (numel,), device="cuda", dtype=torch.int8).sum()
        self.assertTrue(s > 0, "expected randint in [0, 1] to generate nonzero values")

    @parametrize("dtype", [torch.float32, torch.double])
    def test_random_no_reused_random_states(self, dtype: torch.dtype) -> None:
        # Test if random states do not overlap between consecutive rand/randn calls.
        # See https://github.com/pytorch/pytorch/issues/125224

        def run(func, dev: torch.device, dtype: torch.dtype) -> int:
            # Measure how many unique numbers are generated in 2 consecutive calls. If random states are
            # reused, this will yield fewer unique numbers.
            size = 1000000
            gen = torch.Generator(device=dev)
            gen.manual_seed(0)
            t1 = func((size,), device=dev, generator=gen, dtype=dtype)
            t2 = func((size,), device=dev, generator=gen, dtype=dtype)
            return torch.stack([t1, t2]).unique().shape[0]

        # Use CPU as reference. The results should not deviate too much.
        for func in [torch.rand, torch.randn]:
            deviation = abs(
                run(func, torch.device("cuda"), dtype)
                - run(func, torch.device("cpu"), dtype)
            )
            self.assertTrue(deviation < 50_000, deviation)

    def test_min_max_inits(self):
        # Testing if THC_reduceAll received the correct index initialization.
        # This affects the result of THC_reduceAll operations at extreme values
        x = torch.cuda.ByteTensor([0])
        y = torch.cuda.ByteTensor([255])
        expected = torch.cuda.LongTensor([0])[0]

        _, v = x.max(dim=0)
        self.assertEqual(v, expected)

        _, v = y.min(dim=0)
        self.assertEqual(v, expected)

    def test_nvtx(self):
        # Just making sure we can see the symbols
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()
        range_handle = torch.cuda.nvtx.range_start("range_start")
        torch.cuda.nvtx.range_end(range_handle)

    def test_bincount_ext(self):
        # ensure CUDA code coverage
        input_size = (100000,)
        w = torch.randn(input_size, dtype=torch.double, device="cuda")
        w_cpu = w.cpu()
        # test shared memory impl
        t = torch.randint(50, input_size, dtype=torch.int8, device="cuda")
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))
        # test global memory impl
        #   see `CUDAHistogramMemoryType` in SummaryOps.cu
        #   50000 * sizeof(int64_t) == 390 KiB, which should exceed smem of any known GPU
        t = torch.randint(50000, input_size, dtype=torch.int64, device="cuda")
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.zeros([10], dtype=torch.int32, device="cuda")
        # 35488 * 65536 as int32 would cause overflow to negative value
        # giving negative bin offset
        t[0] = 35488
        counted = t.bincount(minlength=65536)
        self.assertEqual(torch.sum(counted), 10)

    def test_tiny_half_norm_(self):
        a = torch.arange(25).cuda().float()
        a /= 100000000
        b = a.half()
        self.assertGreater(b.norm().item(), 0)

    def test_norm_type_conversion(self):
        a = torch.ones(65536).cuda().half()
        self.assertEqual(a.norm(p=0, dtype=torch.float32), 65536)

    def test_cuda_memory_leak_detection_propagates_errors(self):
        with self.assertRaisesRegex(
            RuntimeError, r"The size of tensor a \(3\) must match"
        ):
            with self.assertLeaksNoCudaTensors():
                x = torch.randn(3, 1, device="cuda")
                y = torch.randn(2, 1, device="cuda")
                x + y

    @unittest.skipIf(not TEST_MEDIUM_TENSOR, "not enough memory")
    @serialTest()
    def test_cuda_kernel_loop_overflow(self):
        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        x = torch.randn(1, 1, 1, 2**30 + 1, dtype=torch.float16, device="cuda")
        expected = x[0, 0, 0, 2**30]
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        torch.cuda.synchronize()
        self.assertEqual(y[0, 0, 0, 2**30], expected)

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @gcIfJetson
    @serialTest()
    def test_cuda_kernel_loop_overflow_large(self):
        # Make sure input.numel() > INT_MAX is handled:
        x = torch.randn(1, 1, 1, 2**31, dtype=torch.float16, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "integer out of range"):
            y = torch.nn.functional.avg_pool2d(x, kernel_size=1)

        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        x = torch.randn(1, 1, 1, 2**31 - 1, dtype=torch.float16, device="cuda")
        expected = x[0, 0, 0, 2**31 - 2]
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        torch.cuda.synchronize()
        self.assertEqual(y[0, 0, 0, 2**31 - 2], expected)

    # this might create a reference cycle on self...
    def _make_multiply_in_stream(self):
        class MultiplyInStream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, val):
                ctx.val = val
                ctx.stream = torch.cuda.current_stream()
                return x * val

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(torch.cuda.current_stream(), ctx.stream)
                # delays the operation in the background stream
                torch.cuda._sleep(1000 * 5000)
                return grad * ctx.val, None

        return MultiplyInStream

    @skipCUDANonDefaultStreamIf(True)
    def test_streaming_backwards_sync(self):
        default_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()

        MultiplyInStream = self._make_multiply_in_stream()

        # Tests using grads outside the backward() stream context
        # See "Stream semantics of backward passes" on https://pytorch.org/docs/stable/notes/cuda.html
        x = torch.randn(5, 5, device="cuda", requires_grad=True)
        with torch.cuda.stream(stream):
            stream.wait_stream(default_stream)
            output = MultiplyInStream.apply(x, 2)
            output.sum().backward()
        # sync needed
        default_stream.wait_stream(stream)
        self.assertEqual(x.grad, torch.ones_like(x) * 2)
        self.assertEqual(torch.cuda.current_stream(), default_stream)

        # Tests that using grads in the same stream context as backward()
        # is safe regardless what streams bwd ops ran on
        bwd_ambient_stream = torch.cuda.Stream()
        x = torch.randn(5, 5, device="cuda", requires_grad=True)
        with torch.cuda.stream(stream):
            stream.wait_stream(default_stream)
            output = MultiplyInStream.apply(x, 3)
        with torch.cuda.stream(bwd_ambient_stream):
            bwd_ambient_stream.wait_stream(stream)
            output.sum().backward()
            # x was first used on "stream" so its AccumulateGrad leaf should run on "stream".
            # The end of backward() should have synced "bwd_ambient_stream" with "stream"
            # so it should be safe to use x.grad here without any syncs.
            self.assertEqual(x.grad, torch.ones_like(x) * 3)
            self.assertEqual(torch.cuda.current_stream(), bwd_ambient_stream)

    def test_streaming_backwards_multiple_streams(self):
        MultiplyInStream = self._make_multiply_in_stream()

        class StreamModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.event = torch.cuda.Event()
                self.stream0 = torch.cuda.Stream()
                self.stream1 = torch.cuda.Stream()

            def forward(self, x, x_first_use_on_ambient):
                if x_first_use_on_ambient:
                    x0 = x.clone()
                self.stream0.wait_stream(torch.cuda.current_stream())
                self.stream1.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream0):
                    if not x_first_use_on_ambient:
                        x0 = x.clone()
                    y0 = MultiplyInStream.apply(x0, 2)
                    self.event.record(stream=torch.cuda.current_stream())

                with torch.cuda.stream(self.stream1):
                    y1 = MultiplyInStream.apply(x, 3)
                    self.stream1.wait_event(self.event)
                    return y0 + y1

        stream = torch.cuda.Stream()

        for x_first_use_on_ambient in (True, False):
            # the out_of_place=False, iters=1 case stresses if proper syncs are inserted
            # when grads are initially None and stolen by backward ops.
            for out_of_place, iters in ((True, 1), (False, 1), (False, 5)):
                with torch.cuda.stream(stream):
                    x = torch.randn(5, 5, device="cuda", requires_grad=True)
                    model = StreamModel().cuda()
                    x.register_hook(
                        lambda grad: self.assertEqual(
                            torch.cuda.current_stream(),
                            stream if x_first_use_on_ambient else model.stream0,
                        )
                    )
                    for p in model.parameters():
                        self.assertTrue(p.grad is None)
                    for _ in range(iters):
                        loss = model(x, x_first_use_on_ambient).sum()
                        if out_of_place:
                            x_grad = torch.autograd.grad((loss,), (x,))[0]
                        else:
                            loss.backward()
                # See "Stream semantics of backward passes" on https://pytorch.org/docs/stable/notes/cuda.html
                torch.cuda.current_stream().wait_stream(stream)

                if out_of_place:
                    self.assertEqual(x_grad, torch.ones_like(x) * 5 * iters)
                else:
                    self.assertEqual(x.grad, torch.ones_like(x) * 5 * iters)

    def test_streaming_backwards_sync_graph_root(self):
        # This function tests if bwd ops running on a side stream properly sync with the GraphRoot.
        # The potential bug it targets is a race condition. The test uses multiple trials and
        # torch.cuda._sleep such that if the race condition exists, the test will almost certainly fail,
        # but there's a chance it may spuriously pass. Passing does not guarantee the backend is bug-free,
        # but failure does guarantee there is a bug.
        fwd_bwd_op_stream = torch.cuda.Stream()
        bwd_ambient_stream = torch.cuda.Stream()
        # We need these streams to be different otherwise the test is meaningless.
        self.assertTrue(fwd_bwd_op_stream != bwd_ambient_stream)

        size = int(1e3)

        a = torch.full((size,), 2.0, device="cuda", requires_grad=True)
        b = torch.full((size,), 3.0, device="cuda", requires_grad=True)

        # I don't think we need any manual record_streams below.
        # a and b remain in scope for the entire test.
        # c and grad remain in scope for each iteration, and there's a full sync between iterations.
        for trial in range(5):
            torch.cuda.synchronize()
            a.grad = b.grad = None
            with torch.cuda.stream(fwd_bwd_op_stream):
                c = a * b

            with torch.cuda.stream(bwd_ambient_stream):
                torch.cuda.synchronize()
                # Long-running dummy kernel on bwd_ambient_stream delays filling of grad
                torch.cuda._sleep(int(50 * get_cycles_per_ms()))
                # Fills grad on bwd_ambient_stream
                grad = torch.full((size,), float(trial + 1), device="cuda")

                # Bwd ops still run on fwd_bwd_ops_stream, so the following will likely fail if
                # bwd ops don't sync with bwd_ambient_stream before consuming grad.
                torch.autograd.backward(tensors=c, grad_tensors=grad)

                # See https://github.com/pytorch/pytorch/issues/47028
                # assertEquals below run on bwd_ambient_stream, so this test may also fail
                # if backward() fails to sync with bwd_ambient_stream at the end.
                # Synchronizing here works around the issue until a proper fix can be made.
                torch.cuda.synchronize()
                with torch.no_grad():
                    self.assertEqual(a.grad, grad * b)
                    self.assertEqual(b.grad, grad * a)

    def test_streaming_backwards_callback(self):
        # Tests if autograd callbacks sync properly with respect to leaf streams and
        # the user-facing stream surrounding backward(). If it fails, first suspect is
        # sync logic where  "final_callbacks_" are called in torch/csrc/autograd/engine.cpp
        MultiplyInStream = self._make_multiply_in_stream()

        size = int(1e3)
        a = torch.full((size,), 1, device="cuda", dtype=torch.float, requires_grad=True)
        b = torch.full((size,), 1, device="cuda", dtype=torch.float, requires_grad=True)

        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        stash = []

        # sets up a nontrivial structure of leaf streams
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            c = MultiplyInStream.apply(a, 2)

        s1.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s1):
            d = MultiplyInStream.apply(b, 3)
            s1.wait_stream(s0)
            e = c * d

            def clone_leaf_grads():
                stash.append(a.grad.clone())
                stash.append(b.grad.clone())

            # Use a hook on e to install the callback
            e.register_hook(
                lambda grad: torch.autograd.Variable._execution_engine.queue_callback(
                    clone_leaf_grads
                )
            )

        s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            e.sum().backward()
            # The autograd engine should sync s2 with all leaf streams then run the callback clone_leaf_grads on s2.
            # If those things happened properly, checking the values of the cloned grads on s2 should be safe:
            self.assertEqual(stash[0], torch.full_like(a, 6))
            self.assertEqual(stash[1], torch.full_like(a, 6))

    @unittest.skipIf(
        TEST_WITH_ROCM,
        "In ROCm, kernel asserts are disabled due to performance overhead",
    )
    def test_fixed_cuda_assert_async(self):
        with self.assertRaisesRegex(
            RuntimeError, "Boolean value of Tensor with no values is ambiguous"
        ):
            torch._assert_async(torch.tensor([], device="cuda"))
        with self.assertRaisesRegex(
            RuntimeError,
            "Boolean value of Tensor with more than one value is ambiguous",
        ):
            torch._assert_async(torch.tensor([0, 0], device="cuda"))

        torch._assert_async(torch.tensor(1, device="cuda"))
        torch._assert_async(torch.tensor(0.1, device="cuda"))
        torch._assert_async(torch.tensor(-0.1, device="cuda"))
        torch._assert_async(torch.tensor(True, device="cuda"))
        torch._assert_async(torch.tensor(0 + 0.1j, device="cuda"))

        fail_stmts = [
            "torch._assert_async(torch.tensor(0, device='cuda'))",
            "torch._assert_async(torch.tensor(0.0, device='cuda'))",
            "torch._assert_async(torch.tensor(False, device='cuda'))",
            "torch._assert_async(torch.tensor(0 + 0j, device='cuda'))",
        ]

        import subprocess

        for stmt in fail_stmts:
            with self.subTest(stmt=stmt):
                r = subprocess.call(
                    [
                        sys.executable,
                        "-c",
                        f"""\
import torch

{stmt}
torch.cuda.synchronize()
""",
                    ]
                )
                self.assertTrue(r != 0)

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "FAIL")
    def test_cublas_multiple_threads_same_device(self):
        # Note, these parameters should be very carefully tuned
        # Too small number makes it hard for the racing condition
        # to happen, while too large number sometimes cause hang
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 100

        weight = torch.ones((size, size), device="cuda")
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            my_stream = torch.cuda.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.cuda.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.cuda.stream(my_stream):
                for _ in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS is its own stream,
                    #          but actually now races with its gemm.
                    results[t] = torch.mm(results[t], weight)
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device="cuda")

            threads = [
                threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    # Test is flaky on Windows (https://github.com/pytorch/pytorch/issues/57401)
    @unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows (see issue 57401)")
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    @skipIfRocm
    def test_cudnn_multiple_threads_same_device(self):
        # This function is intended to test the lazy creation and reuse of per-thread
        # cudnn handles on each device in aten/src/ATen/cudnn/Handles.cpp.
        # Failure here likely indicates something wrong with that logic.
        weight = torch.ones((1, 1, 2, 2), device="cuda")

        results = {}

        num_threads = 2
        trials = 3
        test_iters = 1000
        barrier = threading.Barrier(num_threads)

        with torch.backends.cudnn.flags(enabled=True):

            def _worker(t):
                my_stream = torch.cuda.Stream()
                # Hard sync so we don't need to worry about creating and using tensors
                # across streams or the fact that default streams are thread-local.
                # Those issues are not the target of this test.
                torch.cuda.synchronize()
                # Line up threads to increase likelihood of race conditions.
                barrier.wait()
                with torch.cuda.stream(my_stream):
                    for _ in range(test_iters):
                        # If all threads are sharing the same cudnn handle,
                        # the following sequence may occur:
                        # thread 0 calls setCuDNNStreamToCurrent()
                        # thread 1 calls setCuDNNStreamToCurrent()
                        # thread 0 launches its raw convolution, which it thinks is in
                        #          its own stream, but is actually in thread 1's stream.
                        # thread 0 enqueues its div_, which IS is its own stream,
                        #          but now races with its convolution.
                        results[t] = torch.nn.functional.conv2d(
                            results[t], weight, padding=0
                        )
                        results[t].div_(4.0)
                torch.cuda.synchronize()

            for _ in range(trials):
                for t in range(num_threads):
                    results[t] = torch.ones((1, 1, 2048, 2048), device="cuda")

                threads = [
                    threading.Thread(target=_worker, args=(t,))
                    for t in range(num_threads)
                ]

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                for t in range(num_threads):
                    self.assertEqual(
                        results[t].sum().item(),
                        (2048 - test_iters) * (2048 - test_iters),
                    )

    def test_cusparse_multiple_threads_same_device(self):
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 500

        def ones_sparse(size):
            a = torch.arange(size, device="cuda")
            indices = torch.cartesian_prod(a, a).t()
            values = torch.ones(size * size, device="cuda")
            return torch.sparse_coo_tensor(indices, values)

        weight = ones_sparse(size)
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            my_stream = torch.cuda.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.cuda.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.cuda.stream(my_stream):
                for _ in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS is its own stream,
                    #          but actually now races with its gemm.
                    results[t] = weight.mm(results[t])
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device="cuda")

            threads = [
                threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @serialTest()
    def test_max_large_axis(self):
        x = torch.zeros(2**32, device="cuda", dtype=torch.int8)
        x[-1] = 1
        val, idx = x.max(0)
        self.assertEqual(val, 1)
        self.assertEqual(idx, x.shape[0] - 1)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_to_numpy(self):
        self.assertRaises(TypeError, lambda: torch.empty(1, device="cuda").numpy())

    def test_graph_is_current_stream_capturing(self):
        self.assertFalse(torch.cuda.is_current_stream_capturing())

        if TEST_CUDA and (not TEST_WITH_ROCM):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                g = torch.cuda.CUDAGraph()
                self.assertFalse(torch.cuda.is_current_stream_capturing())
                g.capture_begin()
                self.assertTrue(torch.cuda.is_current_stream_capturing())
                g.capture_end()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_simple(self):
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            a = torch.full((1000,), 1, device="cuda")
            g = torch.cuda.CUDAGraph()
            torch.cuda.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graphsafe_set_get_rng_state(self):
        # Define a function to create generator states, with optional graph registration
        def create_states(generator):
            """Initializes generator states and registers them with a CUDA graph if provided."""
            # Ensure the CUDA generator is initialized
            torch.rand(1, device="cuda")
            generator.manual_seed(0)

            # Save the current state of the generator
            old_state = generator.graphsafe_get_state()
            # Create and save a cloned state of the generator
            new_state = generator.clone_state()
            # Return the original generator and its two states
            return generator, old_state, new_state

        def register_states_to_graph(generator_state, graph):
            _, old_state, new_state = generator_state
            graph.register_generator_state(old_state)
            graph.register_generator_state(new_state)

        # Define a function to perform specific RNG actions using the generator's states
        def perform_random_generation_steps(generator_state):
            generator, old_state, new_state = generator_state
            random_values = []

            # Generate random numbers with the new generator state
            generator.graphsafe_set_state(new_state)
            random_values.append(torch.rand(5, device="cuda", generator=generator))

            # Generate random numbers twice with the old generator state
            generator.graphsafe_set_state(old_state)
            random_values.extend(
                [torch.rand(5, device="cuda", generator=generator) for _ in range(2)]
            )

            return random_values

        # Define a function to retrieve the final offsets of the original and new generator states
        def get_final_offsets_of_states(generator_state):
            _, old_state, new_state = generator_state
            old_state_offset = old_state.get_offset()
            new_state_offset = new_state.get_offset()
            return old_state_offset, new_state_offset

        # Set up and test a new CUDA generator
        generator = torch.Generator(device="cuda")
        generator_state = create_states(generator)

        # Set up and test the default CUDA generator with a CUDA Graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        default_generator = torch.cuda.default_generators[0]
        default_generator_state = create_states(default_generator)
        register_states_to_graph(default_generator_state, g)

        # Perform random number generation within a CUDA graph
        with torch.cuda.stream(s):
            g.capture_begin()
            graphed_random_values = perform_random_generation_steps(
                default_generator_state
            )
            g.capture_end()

        # Synchronize the streams and replay the graph
        torch.cuda.current_stream().wait_stream(s)
        for _ in range(3):
            random_values = perform_random_generation_steps(generator_state)
            g.replay()
            offset = get_final_offsets_of_states(generator_state)
            graph_offset = get_final_offsets_of_states(default_generator_state)

            # Compare the final offsets of states for both generators to ensure consistency
            self.assertEqual(offset, graph_offset)
            # Compare the states generated outside and inside the graph
            self.assertEqual(random_values, graphed_random_values)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_memory_stats_of_multiple_generators_and_graphs(self):
        # Function to clear CUDA cache and collect garbage
        def clear_cuda_cache():
            gc.collect()
            torch.cuda.empty_cache()

        # Executes a simple graph task which includes capturing and executing a random number generation within a CUDA graph.
        def simple_graph_task(graph):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                graph.capture_begin()
                torch.rand(1, device="cuda")
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            graph.replay()  # Replays the captured operations

        def get_memory_stats():
            stats = torch.cuda.memory_stats()
            num_blocks = stats["active.all.current"]
            total_size = stats["active_bytes.all.current"]
            return num_blocks, total_size

        def test(num_graphs, num_generators):
            baseline = get_memory_stats()
            baseline_num_blocks, baseline_total_size = baseline

            # Allocate CUDA graphs
            graphs = [torch.cuda.CUDAGraph() for _ in range(num_graphs)]

            # Allocate and manage generator states
            default_generator = torch.cuda.default_generators[0]
            generators = [default_generator.graphsafe_get_state()]

            # Starts from 1 as one state is already added
            for _ in range(1, num_generators):
                generators.append(default_generator.clone_state())

            for graph in graphs:
                for generator_state in generators:
                    graph.register_generator_state(generator_state)
                simple_graph_task(graph)

            # Assert conditions after graph tasks
            num_blocks, total_size = get_memory_stats()
            # The allocated blocks should only be proportional to the number of generators
            expected_blocks_diff = 2 * num_generators
            expected_size_diff = 2 * 512 * num_generators  # Each block's size is 512

            self.assertEqual(
                (num_blocks - baseline_num_blocks),
                expected_blocks_diff,
                "Unexpected number of active blocks.",
            )
            self.assertEqual(
                (total_size - baseline_total_size),
                expected_size_diff,
                "Unexpected total memory size.",
            )

            # Cleanup graphs and clear CUDA cache
            while graphs:
                graph = graphs.pop()
                del graph
            clear_cuda_cache()

            # Assert that memory stats return to baseline after cleanup
            self.assertEqual(
                get_memory_stats(),
                baseline,
                "Memory stats do not match baseline after cleanup.",
            )

        # Running the test function with different parameters
        test(1, 1)
        test(3, 2)
        test(10, 20)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_reset_recapture(self):
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            a = torch.full((1000,), 1, device="cuda")
            g = torch.cuda.CUDAGraph()
            torch.cuda.empty_cache()
            g.capture_begin()
            b = a
            for _ in range(10):
                b = b + 1
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()

        self.assertEqual(b.sum().item(), 11000.0)

        g.reset()

        with torch.cuda.stream(s):
            g.capture_begin()
            b.fill_(2.0)
            for _ in range(10):
                b = b + 2
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        g.replay()
        self.assertEqual(b.sum().item(), 22000.0)

        g.reset()
        del g

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_debugdump(self):
        torch.cuda.empty_cache()
        x = torch.randn(10240000, device="cuda")
        y = torch.rand_like(x)
        g = torch.cuda.CUDAGraph()
        g.enable_debug_mode()
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            g.capture_begin()
            z = x + y
            with torch.cuda.stream(s1):
                s1.wait_stream(s0)
                z + y
            s0.wait_stream(s1)
            g.capture_end()
        s0.synchronize()
        torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as tempdir:
            g.debug_dump(os.path.join(tempdir, "out_multi_stream.dot"))

    @unittest.skipIf(
        not TEST_CUDA_GRAPH or TEST_WITH_ROCM,
        "CUDA >= 11.0 required for external events in cuda graphs. rocm does not support external events",
    )
    def test_graph_timing(self):
        torch.cuda.empty_cache()
        x = torch.randn(10240000, device="cuda")
        y = torch.rand_like(x)
        g = torch.cuda.CUDAGraph()
        start_event = torch.cuda.Event(enable_timing=True, external=True)
        end_event = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(g):
            start_event.record()
            z = x + y
            end_event.record()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(start_event.elapsed_time(end_event) > 0)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_error(self):
        # We need to run this test in a separate thread as the error we trigger
        # puts the cuda context in a bad state
        script = """
import torch

g = torch.cuda.CUDAGraph()
try:
    g.capture_begin()
except RuntimeError as e:
    if "CUDA graphs must be captured on a non-default stream." in str(e):
        exit(0)
    else:
        exit(1)
exit(2)
"""
        try:
            subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                self.assertTrue(
                    False,
                    "Error raise by starting capture without a stream is not the expected one",
                )
            elif e.returncode == 2:
                self.assertTrue(
                    False,
                    "Error raised by starting capture without a stream was not caught",
                )

    @unittest.skipIf(
        (not TEST_CUDA) or TEST_WITH_ROCM,
        "CUDA >= 11.0 required for graphs",
    )
    def test_graph_warn_if_has_zero_nodes(self):
        with warnings.catch_warnings(record=True) as caught:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                g.capture_begin()
                g.capture_end()
        self.assertTrue(
            any("The CUDA Graph is empty" in str(w.message) for w in caught)
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_graph_capture_oom(self):
        oom_regex = (
            "would exceed allowed memory" if TEST_CUDAMALLOCASYNC else "out of memory"
        )
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                torch.zeros(2**40, device="cuda")

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @serialTest()
    @setBlasBackendsToDefaultFinally
    def test_repeat_graph_capture_cublas_workspace_memory(self):
        torch.backends.cuda.preferred_blas_library("cublas")
        (x, y, z) = 1024, 512, 64
        a = torch.rand((x, y), device="cuda")
        b = torch.rand((y, z), device="cuda")

        # warmup
        torch.mm(a, b)

        free_bytes_before, total_bytes = torch.cuda.mem_get_info()
        used_gb_before = (total_bytes - free_bytes_before) / 1e9

        for _ in range(100):
            torch_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(torch_graph):
                torch.mm(a, b)
            torch_graph.replay()

        free_bytes_after, _ = torch.cuda.mem_get_info()
        used_gb_after = (total_bytes - free_bytes_after) / 1e9

        self.assertFalse(used_gb_before + 0.1 < used_gb_after)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_rng_functional(self):
        ops_with_kwargs = (
            (torch.nn.functional.dropout, {"p": 0.1}),
            (torch.nn.functional.rrelu, {"training": True}),
        )
        size = 10000

        def run(op, kwargs):
            a = torch.randn((size,), device="cuda", dtype=torch.float)

            # Control
            torch.cuda.manual_seed(5)
            eager_out = a
            for _ in range(6):
                eager_out = op(eager_out, **kwargs)

            graph_in = a.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda.manual_seed(5)

                g = torch.cuda.CUDAGraph()
                torch.cuda.empty_cache()
                g.capture_begin()
                graph_out = graph_in
                for _ in range(2):
                    graph_out = op(graph_out, **kwargs)
                g.capture_end()
            torch.cuda.current_stream().wait_stream(stream)

            # Runs a graphed->eager->graphed sequence of RNG ops.
            # replay() plays 2 invocations of the op, so the sequence has 6
            # invocations total, matching Control.
            # replay() reads from graph_in and writes to graph_out.
            g.replay()
            out = op(graph_out, **kwargs)
            out = op(out, **kwargs)
            graph_in.copy_(out)
            g.replay()

            # If replay() updated RNG state correctly, graph_out
            # should now hold data equal to eager_out.
            try:
                self.assertEqual(eager_out, graph_out)
            except Exception as e:
                raise RuntimeError("Failed on ", op) from e

            # Do the same operations varying seeds
            seeds = [6, 128, 9999]

            for seed in seeds:
                torch.cuda.manual_seed(seed)
                graph_in.copy_(a)
                for _ in range(3):
                    g.replay()

                # If the random seed was not updated then the graph would
                # generate the same output as in previous check.
                try:
                    self.assertNotEqual(eager_out, graph_out)
                except Exception as e:
                    raise RuntimeError("Failed on ", op) from e

                # Now repeat the same operations in non-graphed mode.
                torch.cuda.manual_seed(seed)
                for _ in range(3):
                    eager_out.copy_(a)
                    eager_out = op(eager_out, **kwargs)
                    eager_out = op(eager_out, **kwargs)

                # In the end, graph_out and 

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 25 class(es): TestCuda, MultiplyInStream, StreamModel, MLP1, MLP2, ParameterlessModule, ParameterlessModule, MyFunction, MyModule, TestCudaMallocAsync, MyModel, TestBlockStateAbsorption, TestMemPool, TestCudaOptims, TestGDS, TestCudaAutocast, MyMM, MyMM, MyMM, Model

### Functions
This file defines 360 function(s): setUp, tearDown, expandable_segments, test_pinned_memory_with_cudaregister, test_pinned_memory_with_cudaregister_multithread, test_host_memory_stats, empty_stats, check_stats, test_pinned_memory_empty_cache, test_pinned_memory_use_background_threads, test_cudart_register, test_memory_allocation, test_memory_stats, test_check_error, test_cuda_get_device_name, test_cuda_get_device_capability, test_cuda_get_device_properties, test_out_of_memory, test_out_of_memory_retry, test_set_per_process_memory_fraction, test_get_per_process_memory_fraction, test_uuid, test_copy_non_blocking, _test_copy_non_blocking, test_copy_non_blocking_type_conversion, test_to_non_blocking, _test_to_non_blocking, test_to_cpu_blocking_by_default, test_serialization_array_with_storage, test_preferred_blas_library_settings


## Key Components

The file contains 22340 words across 7588 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 293745 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
