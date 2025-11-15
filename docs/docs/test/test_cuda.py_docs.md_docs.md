# Documentation: `docs/test/test_cuda.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_cuda.py_docs.md`
- **Size**: 54,532 bytes (53.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_cuda.py`

## File Metadata

- **Path**: `test/test_cuda.py`
- **Size**: 293,745 bytes (286.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
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
        self.assertEqual(list(gpu_tensor
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_cuda.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_cuda.py_docs.md_docs.md`
- **Keyword Index**: `test_cuda.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
