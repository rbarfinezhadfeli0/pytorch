# Documentation: `docs/test/inductor/test_codecache.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_codecache.py_docs.md`
- **Size**: 54,703 bytes (53.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_codecache.py`

## File Metadata

- **Path**: `test/inductor/test_codecache.py`
- **Size**: 121,637 bytes (118.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import functools
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import contextmanager
from typing import Optional, Union
from typing_extensions import override
from unittest import mock

import torch
from torch._dynamo import reset
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._inductor import config, metrics
from torch._inductor.codecache import (
    BypassFxGraphCache,
    cuda_compile_command,
    CUDACodeCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    PyCodeCache,
    TensorMetadata,
    TensorMetadataAndValues,
)
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    CustomGraphPass,
    CustomPartitionerFn,
    get_hash_for_files,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.mock_cache import global_stats, PatchCaches, Stats
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches, fresh_cache
from torch._library import capture_triton
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.testing._internal.common_cuda import (
    SM80OrLater,
    TEST_MULTIGPU,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_SANDCASTLE,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    HAS_MULTIGPU,
    HAS_TRITON,
    HAS_XPU_AND_TRITON,
    patch_inductor_backend,
    requires_gpu,
    requires_triton,
)
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
)


try:
    from . import custom_inductor_config
except ImportError:
    import custom_inductor_config


if HAS_TRITON:
    import triton  # @manual

    from torch.testing._internal.triton_utils import add_kernel, sub_kernel

torch._dynamo.config.fake_tensor_cache_enabled = True
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True


class LogCaptureHandler(logging.Handler):
    def __init__(self, level):
        super().__init__(level)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextmanager
def capture_logs(log_name, log_level):
    try:
        logger = logging.getLogger(log_name)
        old_level = logger.level
        handler = logging.Handler()
        logger.setLevel(log_level)
        log_records = []

        def emit(record):
            log_records.append(record)

        handler.emit = emit
        logger.addHandler(handler)

        yield log_records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


class MyModelConv2d(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        torch._dynamo.graph_break()
        x = self.conv2(x)
        return x


class TestPyCodeCache(TestCase):
    def test_linemaps_empty(self):
        src = """import torch"""
        (key, path) = PyCodeCache.write(src, "")
        # Load with an empty linemap
        PyCodeCache.load_by_key_path(key, path, linemap=[])
        stack_frames = PyCodeCache.stack_frames_for_code(path, 0)
        self.assertEqual(stack_frames, None)

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_editable_cached_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = tmpdir

            step1 = textwrap.dedent(
                """
                import glob
                import os
                import torch
                import warnings
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                cache_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]
                pyfiles = glob.glob(os.path.join(cache_dir, "**", "*.py"), recursive=True)
                print(pyfiles[0])
                """
            )
            wrapper_path = (
                subprocess.check_output([sys.executable, "-c", step1], env=env)
                .decode()
                .strip()
            )

            step2 = textwrap.dedent(
                """
                import torch
                import warnings
                from torch._dynamo.utils import counters
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                print(counters["inductor"]["fxgraph_cache_hit"])
                """
            )
            hit = (
                subprocess.check_output([sys.executable, "-c", step2], env=env)
                .decode()
                .strip()
            )
            # XPU have extra lines, so get the last line, refer https://github.com/intel/torch-xpu-ops/issues/2261
            if torch.xpu.is_available():
                wrapper_path = wrapper_path.splitlines()[-1]
                hit = hit.splitlines()[-1]
            self.assertEqual(hit, "1")

            with open(wrapper_path) as f:
                src = f.read()
            with open(wrapper_path, "w") as f:
                f.write(
                    src.replace(
                        "def call(self, args):",
                        "def call(self, args):\n        print('debug')",
                    )
                )

            step3 = textwrap.dedent(
                """
                import torch
                import warnings
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                """
            )
            out = subprocess.check_output(
                [sys.executable, "-c", step3], env=env
            ).decode()
            self.assertIn("debug", out)


@instantiate_parametrized_tests
class TestFxGraphCache(TestCase):
    device_type = GPU_TYPE

    def setUp(self):
        super().setUp()
        counters.clear()
        DynamoCache.clear()
        PrecompileContext.clear()
        AOTAutogradCache.clear()
        PatchCaches.setUp()
        CacheArtifactManager.clear()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        PatchCaches.tearDown()

    def reset(self):
        AOTAutogradCache.clear()
        DynamoCache.clear()
        PrecompileContext.clear()
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"compile_threads": 1})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @parametrize("bundle_triton", (False, True))
    @parametrize("use_static_cuda_launcher", (False, True))
    @parametrize("grad", (False, True))
    def test_cache_load_function(
        self, device, dtype, dynamic, bundle_triton, use_static_cuda_launcher, grad
    ):
        """
        Verify that we can populate and load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")
        if use_static_cuda_launcher and not (device == "cuda" and bundle_triton):
            raise unittest.SkipTest(
                "Static cuda launcher requires cuda and triton bundling"
            )
        if use_static_cuda_launcher and TEST_WITH_ROCM:
            raise unittest.SkipTest("Static cuda launcher doesn't work with ROCM")

        grad_multiplier = 2 if grad else 1

        def fn(x, y):
            yy = y @ y
            return x * 2 + yy.view(25)

        a_orig = torch.rand(25, dtype=dtype, device=device)
        b_orig = torch.rand(5, 5, dtype=dtype, device=device)

        with config.patch(
            bundle_triton_into_fx_graph_cache=bundle_triton,
            use_static_cuda_launcher=use_static_cuda_launcher,
        ):
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            # A first call should miss in the cache.
            eager_result = fn(a1, b1)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            if grad:
                eager_result.sum().backward()
                compiled_result.sum().backward()
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 1
            )
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

            # we expect:
            #  .ttir
            #  .ttgir
            #  .llir
            #  .ptx (cuda) or .spv (xpu)
            #  .json
            #  __grp__.*.json
            # optionally, we can also get
            #  .cubin (CUDA only)
            #  .source (new versions of triton only, triton-lang/triton#6992)

            # to avoid depending on the device and triton version, just assert that
            # we have at least 6 kernels.
            save_and_read_min_artifact_count = 6
            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                self.assertEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"], 0
                )
                if use_static_cuda_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier if device == "cuda" else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"], 0
                    )

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean triton kernels
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            eager_result = fn(a1, b1)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            if grad:
                eager_result.sum().backward()
                compiled_result.sum().backward()
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_lookup_write_file"], grad_multiplier * 1
            )

            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                if use_static_cuda_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier if device == "cuda" else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"],
                        grad_multiplier if device == "cuda" else 0,
                    )

            self.reset()

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            eager_result = fn(a1, b1)
            if grad:
                eager_result.sum().backward()
            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                compiled_result = compiled_fn(a2, b2)
                if grad:
                    compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            if grad:
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)

            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 2
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_lookup_write_file"], grad_multiplier * 1
            )

            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count * 2,
                )
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                if use_static_cuda_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier * 2 if device == "cuda" else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"],
                        grad_multiplier if device == "cuda" else 0,
                    )

    @requires_triton()
    @config.patch({"fx_graph_remote_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @parametrize("bundle_triton", (False, True))
    @parametrize("use_static_cuda_launcher", (False, True))
    @config.patch(
        {"compile_threads": 1}
    )  # Can't check globalStats if there are workers
    def test_remote_cache_load_function(
        self, device, dtype, dynamic, bundle_triton, use_static_cuda_launcher
    ):
        from unittest.mock import patch

        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if (
            device == "cuda"
            and torch.version.hip is None
            and dtype == torch.bfloat16
            and not SM80OrLater
        ):
            raise unittest.SkipTest("requires SM80 or later")
        if use_static_cuda_launcher and not (device == "cuda" and bundle_triton):
            raise unittest.SkipTest(
                "Static cuda launcher requires cuda and triton bundling"
            )

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)

        with (
            config.patch(
                {
                    "fx_graph_remote_cache": True,
                    "bundle_triton_into_fx_graph_cache": bundle_triton,
                    "use_static_cuda_launcher": use_static_cuda_launcher,
                }
            ),
            patch.dict(os.environ),
            PatchCaches(),
        ):
            os.environ.pop("TRITON_CACHE_MANAGER", None)
            for _ in range(4):
                with fresh_cache():
                    compiled_fn = torch.compile(fn, dynamic=dynamic)
                    self.assertEqual(fn(a, b), compiled_fn(a, b))
                reset()

            self.assertEqual(global_stats.fx_graph, Stats(1, 3, 1))

            with torch.compiler.config.patch({"cache_key_tag": "test"}), fresh_cache():
                compiled_fn = torch.compile(fn, dynamic=dynamic)
                self.assertEqual(fn(a, b), compiled_fn(a, b))

            self.assertEqual(global_stats.fx_graph, Stats(2, 3, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.fx_graph.cache.keys():
            self.assertRegex(k, r"pt2:fx-graph-v1::[0-9a-z]{52}:c[0-9]+")

    @requires_triton()
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    def test_cache_hot_load(self, device, dtype, dynamic):
        """
        Verify that we can populate and hot load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device)
        b = torch.rand(100, 100, dtype=dtype, device=device)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 1 if device == GPU_TYPE else 0

        self.assertEqual(len(cache_info.inductor_artifacts), 1)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.inductor_artifacts), 1)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_triton()
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @torch._dynamo.config.patch(
        {
            "caching_precompile": True,
        }
    )
    @parametrize("dynamic", (False, True))
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_cache_hot_load_caching_precompile(self, device, dtype, dynamic):
        """
        Verify that we can populate and hot load functions from the cache.
        """

        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)
        b = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 1)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 2 if device == GPU_TYPE else 0
        self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)
        self.assertEqual(len(cache_info.precompile_artifacts), 1)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            # With caching precompile, we have to re torch.compile the function
            # to trigger cache lookup
            compiled_fn = torch.compile(fn, dynamic=dynamic)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 2)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 0)
        self.reset()
        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache(), torch.compiler.set_stance("fail_on_recompile"):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)
            self.assertEqual(len(cache_info.precompile_artifacts), 1)

            # With caching precompile, we have to re torch.compile the function
            # to trigger cache lookup
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 2)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 1)

    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
        }
    )
    def test_cache_hot_load_repeat(self):
        def fn(x, y):
            return x @ y.sin()

        compiled_fn = torch.compile(fn, dynamic=False)

        a = torch.randn(4, 4)
        b = torch.randn(4, 4)

        a2 = torch.randn(4, 8)
        b2 = torch.randn(8, 4)

        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())
        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.reset()

        with fresh_cache():
            torch.compiler.load_cache_artifacts(artifact_bytes)
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())

        self.reset()

        with fresh_cache():
            eager_result = fn(a2, b2)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertTrue(torch.compiler._cache.CacheArtifactManager.need_serialize())

    @torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    def test_cache_hot_load_pgo(self):
        """
        Verify that we can populate and hot load functions from the cache with pgo.
        """

        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        @torch.compile(backend=backend, fullgraph=True)
        def f(x):
            return x * 2

        # Record artifacts
        with torch.compiler.config.patch(job_id=self.id()), fresh_cache():
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
            self.assertEqual(backend.frame_count, 2)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), 0)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
        self.assertEqual(len(cache_info.pgo_artifacts), 2)

        self.reset()
        backend.clear()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with torch.compiler.config.patch({"job_id": self.id()}), fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), 0)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
            self.assertEqual(len(cache_info.pgo_artifacts), 2)

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(backend.frame_count, 1)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    def test_cache_hot_load_pgo_swap_file_names(self):
        """
        Verify that we can populate and hot load functions from the cache with pgo
        with file name swapping
        """

        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        @torch.compile(backend=backend, fullgraph=True)
        def f(x):
            return x * 2

        # Record artifacts
        with mock.patch(
            "torch._utils_internal.get_mast_job_name_version", return_value=("foo", 5)
        ):
            with fresh_cache():
                f(torch.randn(2, 3))
                f(torch.randn(2, 4))
                self.assertEqual(backend.frame_count, 2)

            artifacts = torch.compiler.save_cache_artifacts()

            self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.pgo_artifacts), 2)

        self.reset()
        backend.clear()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with (
            mock.patch(
                "torch._utils_internal.get_mast_job_name_version",
                return_value=("bar", 10),
            ),
            fresh_cache(),
        ):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.pgo_artifacts), 2)

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(backend.frame_count, 1)

    def test_cache_hot_load_empty(self):
        self.assertIsNone(torch.compiler.save_cache_artifacts())

    def test_cache_hot_load_generic(self):
        class CacheStub:
            def __init__(self):
                self.cache = {}

            def lookup(self, key):
                content = self.cache.get(key)
                if content is None:
                    return None

                CacheArtifactManager.record_artifact(
                    ArbitraryCacheArtifact.type(), key, content
                )
                return content

            def save(self, key, content):
                self.cache[key] = content
                CacheArtifactManager.record_artifact(
                    ArbitraryCacheArtifact.type(), key, content
                )

            def clear(self):
                self.cache.clear()

        cache_stub = CacheStub()

        @CacheArtifactFactory.register
        class ArbitraryCacheArtifact(CacheArtifact):
            @override
            def populate_cache(self) -> None:
                cache_stub.cache[self.key] = self.content.decode()

            @override
            @staticmethod
            def type() -> str:
                return "test"

            @override
            @staticmethod
            def encode(content: str) -> bytes:
                return content.encode()

        test_cache = {"1": "foo", "2": "bar", "foo": "bar"}

        for k, v in test_cache.items():
            cache_stub.save(k, v)

        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.test_artifacts), 3)

        cache_stub.clear()
        CacheArtifactManager.clear()

        cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
        self.assertEqual(len(cache_info.test_artifacts), 3)
        self.assertEqual(cache_stub.cache, test_cache)

        CacheArtifactManager.clear()
        cache_stub.lookup("foo")
        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        _, cache_info = artifacts
        self.assertEqual(len(cache_info.test_artifacts), 1)

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.float64))
    @parametrize("dynamic", (False, True))
    def test_cache_load_model(self, device, dtype, dynamic):
        """
        Verify that we can populate and load models from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn(mod, x):
            mod.zero_grad()
            mod(x).sum().backward()
            return [p.grad for p in mod.parameters()]

        compiled_fn = torch.compile(fn, dynamic=dynamic)

        mod = MyModelConv2d().to(device=device, dtype=dtype)
        inp = torch.randn(2, 3, 16, 32, device=device, dtype=dtype)

        # The first call should see all cache misses.
        counters.clear()
        grads1 = compiled_fn(mod, inp)
        self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # The second should see all hits. (First reset so in-memory guards
        # don't prevent compilation).
        counters.clear()
        self.reset()
        grads2 = compiled_fn(mod, inp)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

        # And the results should be the same.
        self.assertEqual(grads1, grads2)

    @largeTensorTest("64GB", device=GPU_TYPE, inductor=True)
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE,))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_cache_load_with_guards_int32_bounds(self, device, dtype):
        """
        Test caching the same graph, but under conditions that introduce guards
        for tensor sizes < int32.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires CUDA SM80 or later")

        def fn(x, y):
            return (x + x, y + y)

        compiled_fn = torch.compile(fn, dynamic=True)

        # Iterate over different shapes, varying whether the total
        # size is below or above int32. For each combination, we expect
        # different guards around whether the symbolic sizes do or do
        # not exceed int32.
        shapes = (
            ((5, 6), (7, 8)),
            ((5, 6), (47000, 47001)),
            ((47000, 47001), (5, 6)),
        )
        for a_shape, b_shape in shapes:
            a = torch.rand(a_shape, device=device, dtype=dtype)
            b = torch.rand(b_shape, device=device, dtype=dtype)

            # AVOID a dynamo reset here. We expect guards to have been
            # added that will be violated with the new shape. We should
            # see a recompilation (along with a cache miss).
            counters.clear()
            res1 = compiled_fn(a, b)
            self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # A second call should hit. (Reset here to force compilation).
            counters.clear()
            self.reset()
            res2 = compiled_fn(a, b)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.assertEqual(res1, res2)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_cache_load_with_guards_static_bounds(self, device, dtype):
        """
        Test caching the same graph, but under conditions that introduce guards
        for static bounds.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        # See lowering; for all of the pooling operators, we always guard and
        # make the height/width static.
        def fn(x):
            return torch.nn.functional.adaptive_avg_pool2d(x, [5, 7])

        compiled_fn = torch.compile(fn, dynamic=True)

        # Iterate over different input shapes. Each new shape should cause
        # a cache miss.
        shapes = ((1, 64, 8, 9), (1, 64, 9, 10), (1, 64, 10, 11))
        for shape in shapes:
            x = torch.rand(shape, device=device, dtype=dtype)

            # AVOID a dynamo reset here. For each cache hit, we expect guards
            # to have been added that will be violated with each new shape.
            # We should see a recompilation (along with a cache miss).
            counters.clear()
            res1 = compiled_fn(x)
            self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # A second call should hit.
            counters.clear()
            self.reset()
            res2 = compiled_fn(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.assertEqual(res1, res2)

    @config.patch("fx_graph_cache", True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch("fx_graph_remote_cache", False)
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @requires_cuda_and_triton
    def test_no_arguments_tensor_device_guards(self):
        """
        Usually, when there are example inputs, the device index of the inputs
        is sufficient to make sure we don't cache hit with the results from different
        cuda devices.
        When the input has no arguments, we still need to have the cuda
        device index in the cache key.
        """

        @torch.compile
        def f():
            y = torch.randn(3, device="cuda")
            return (y,)

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:0"))
        self.reset()
        # Should not cache hit with device guard
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:1"))

    @config.patch("fx_graph_cache", True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch("fx_graph_remote_cache", False)
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @requires_cuda_and_triton
    def test_tensor_device_guards_cpu_tensor(self):
        """
        CPU tensor arguments should still cache hit
        """

        @torch.compile
        def f(x):
            return x.sin()

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            result = f(torch.randn(3, device="cpu"))
            self.assertEqual(result.device, torch.device("cpu"))

        self.reset()
        # Should not cache hit with device guard
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            result = f(torch.randn(3, device="cpu"))
            self.assertEqual(result.device, torch.device("cpu"))

        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_constant_handling(self, device):
        """
        Test that different constants are recognized correctly.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn1(x):
            return x + torch.tensor(list(range(12)), device=device)

        def fn2(x):
            return x + torch.tensor(list(range(1, 13)), device=device)

        a = torch.rand(12, device=device)

        compiled_fn1 = torch.compile(fn1)
        compiled_fn2 = torch.compile(fn2)

        # A call to fn1 should miss in the cache.
        self.assertEqual(fn1(a), compiled_fn1(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A call to fn2 should also miss (the constant is different)
        self.assertEqual(fn2(a), compiled_fn2(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("variant", ("v1", "v2"))
    def test_auto_functionalized_caching(self, variant):
        if variant == "v1":
            patch = torch._inductor.config.patch(enable_auto_functionalized_v2=False)
        else:
            assert variant == "v2"
            patch = torch._inductor.config.patch(enable_auto_functionalized_v2=True)

        @torch.library.custom_op("mylib::sin_inplace", mutates_args=["x"])
        def sin_inplace(x: torch.Tensor) -> None:
            x.sin_()

        @torch.library.custom_op("mylib::cos_inplace", mutates_args=["x"])
        def cos_inplace(x: torch.Tensor) -> None:
            x.cos_()

        @torch.compile(fullgraph=True)
        def fn(x, op):
            y = torch.empty_like(x)
            op(y)
            return y

        x = torch.randn(3)

        with patch:
            # A first call should miss in the cache.
            fn(x, sin_inplace)
            self.reset()
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()
            fn(x, sin_inplace)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

            # A third call with different operator should have a cache miss
            self.reset()
            fn(x, cos_inplace)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_gpu_and_triton
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @with_tf32_off
    def test_flex_attention_caching(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        block_mask = create_block_mask(
            lambda b, h, q, kv: q >= kv, None, None, 512, 512
        )

        def score_mod(score, b, h, q, kv):
            return score + (q - kv)

        def fn(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        def score_mod2(score, b, h, q, kv):
            return score

        def fn2(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod2, block_mask=block_mask)

        a, b, c = (torch.randn(1, 4, 512, 64).to(GPU_TYPE) for _ in range(3))
        compiled_fn = torch.compile(fn)
        compiled_fn2 = torch.compile(fn2)

        atol, rtol = 1e-4, 1e-4

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b, c), compiled_fn(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self.reset()
        self.assertEqual(fn(a, b, c), compiled_fn(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

        # A third call with different score_mod should have a cache miss
        self.reset()
        self.assertEqual(fn2(a, b, c), compiled_fn2(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("bundle_triton", (False, True))
    def test_higher_order_op_bypass(self, bundle_triton):
        """
        Verify that we bypass the cache when we have a higher order ops
        and that bundler start/end works with a cache bypass.
        """

        def fn(x):
            def true_fn(x: torch.Tensor):
                return x.cos()

            def false_fn(x: torch.Tensor):
                return x.sin()

            return torch.cond(x.shape[0], true_fn, false_fn, (x,))

        with config.patch(
            bundle_triton_into_fx_graph_cache=bundle_triton,
        ):
            compiled_fn = torch.compile(fn, dynamic=True, fullgraph=True)

            x = torch.randn(4, 4, device=GPU_TYPE)
            compiled_fn(x)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("bundle_triton", (False, 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/inductor/test_codecache.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_split_cat_fx_passes.py_kw.md_docs.md`](./test_split_cat_fx_passes.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_codecache.py_docs.md_docs.md`
- **Keyword Index**: `test_codecache.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
