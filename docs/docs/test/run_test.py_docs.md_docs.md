# Documentation: `docs/test/run_test.py_docs.md`

## File Metadata

- **Path**: `docs/test/run_test.py_docs.md`
- **Size**: 54,246 bytes (52.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/run_test.py`

## File Metadata

- **Path**: `test/run_test.py`
- **Size**: 76,464 bytes (74.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

import argparse
import contextlib
import copy
import glob
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast, NamedTuple, Optional, Union

import torch
import torch.distributed as dist
from torch.multiprocessing import current_process, get_context
from torch.testing._internal.common_utils import (
    get_report_path,
    IS_CI,
    IS_MACOS,
    retry_shell,
    set_cwd,
    shell,
    TEST_CUDA,
    TEST_SAVE_XML,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW_GRADCHECK,
    TEST_XPU,
)


# using tools/ to optimize test run.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_CLASS_TIMES_FILE,
    TEST_TIMES_FILE,
)
from tools.stats.upload_metrics import add_global_metric, emit_metric
from tools.testing.discover_tests import (
    CPP_TEST_PATH,
    CPP_TEST_PREFIX,
    CPP_TESTS_DIR,
    parse_test_module,
    TESTS,
)
from tools.testing.do_target_determination_for_s3 import import_results
from tools.testing.target_determination.gen_artifact import gen_ci_artifact
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    gen_additional_test_failures_file,
)
from tools.testing.target_determination.heuristics.utils import get_pr_number
from tools.testing.test_run import TestRun
from tools.testing.test_selections import (
    calculate_shards,
    get_test_case_configs,
    NUM_PROCS,
    ShardedTest,
    THRESHOLD,
)


try:
    from tools.testing.upload_artifacts import (
        parse_xml_and_upload_json,
        upload_adhoc_failure_json,
        zip_and_upload_artifacts,
    )
except ImportError:
    # some imports in those files might fail, e.g., boto3 not installed. These
    # functions are only needed under specific circumstances (CI) so we can
    # define dummy functions here.
    def parse_xml_and_upload_json():
        pass

    def zip_and_upload_artifacts(*args, **kwargs):
        pass

    def upload_adhoc_failure_json(*args, **kwargs):
        pass


# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))


HAVE_TEST_SELECTION_TOOLS = True
TEST_CONFIG = os.getenv("TEST_CONFIG", "")
BUILD_ENVIRONMENT = os.getenv("BUILD_ENVIRONMENT", "")
RERUN_DISABLED_TESTS = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"
DISTRIBUTED_TEST_PREFIX = "distributed"
INDUCTOR_TEST_PREFIX = "inductor"
IS_SLOW = "slow" in TEST_CONFIG or "slow" in BUILD_ENVIRONMENT
IS_S390X = platform.machine() == "s390x"


# Note [ROCm parallel CI testing]
# https://github.com/pytorch/pytorch/pull/85770 added file-granularity parallel testing.
# In .ci/pytorch/test.sh, TEST_CONFIG == "default", CUDA and HIP_VISIBLE_DEVICES is set to 0.
# This results in multiple test files sharing the same GPU.
# This should be a supported use case for ROCm, but it exposed issues in the kernel driver resulting in hangs.
# See https://github.com/pytorch/pytorch/issues/90940.
#
# Further, ROCm self-hosted runners have up to 4 GPUs.
# Device visibility was set to 0 to match CUDA test behavior, but this was wasting available GPU resources.
# Assigning each Pool worker their own dedicated GPU avoids the ROCm oversubscription issues.
# This should also result in better overall wall clock time since all GPUs can be utilized.
def maybe_set_hip_visible_devies():
    # Special handling of ROCm GHA runners for parallel (file granularity) tests.
    if torch.version.hip:
        p = current_process()
        if p.name != "MainProcess":
            # this is a Process from a parallel Pool, not the MainProcess
            os.environ["HIP_VISIBLE_DEVICES"] = str(p._identity[0] % NUM_PROCS)


def strtobool(s):
    return s.lower() not in {"", "0", "false", "off"}


class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super().__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))


FSDP_TEST = [test for test in TESTS if test.startswith("distributed/fsdp")]

WINDOWS_BLOCKLIST = [
    "distributed/nn/jit/test_instantiator",
    "distributed/rpc/test_faulty_agent",
    "distributed/rpc/test_tensorpipe_agent",
    "distributed/rpc/test_share_memory",
    "distributed/rpc/cuda/test_tensorpipe_agent",
    "distributed/pipeline/sync/skip/test_api",
    "distributed/pipeline/sync/skip/test_gpipe",
    "distributed/pipeline/sync/skip/test_inspect_skip_layout",
    "distributed/pipeline/sync/skip/test_leak",
    "distributed/pipeline/sync/skip/test_portal",
    "distributed/pipeline/sync/skip/test_stash_pop",
    "distributed/pipeline/sync/skip/test_tracker",
    "distributed/pipeline/sync/skip/test_verify_skippables",
    "distributed/pipeline/sync/test_balance",
    "distributed/pipeline/sync/test_bugs",
    "distributed/pipeline/sync/test_checkpoint",
    "distributed/pipeline/sync/test_copy",
    "distributed/pipeline/sync/test_deferred_batch_norm",
    "distributed/pipeline/sync/test_dependency",
    "distributed/pipeline/sync/test_inplace",
    "distributed/pipeline/sync/test_microbatch",
    "distributed/pipeline/sync/test_phony",
    "distributed/pipeline/sync/test_pipe",
    "distributed/pipeline/sync/test_pipeline",
    "distributed/pipeline/sync/test_stream",
    "distributed/pipeline/sync/test_transparency",
    "distributed/pipeline/sync/test_worker",
    "distributed/elastic/agent/server/test/api_test",
    "distributed/elastic/multiprocessing/api_test",
    "distributed/_shard/checkpoint/test_checkpoint"
    "distributed/_shard/checkpoint/test_file_system_checkpoint"
    "distributed/_shard/sharding_spec/test_sharding_spec",
    "distributed/_shard/sharding_plan/test_sharding_plan",
    "distributed/_shard/sharded_tensor/test_sharded_tensor",
    "distributed/_shard/sharded_tensor/test_sharded_tensor_reshard",
    "distributed/_shard/sharded_tensor/ops/test_embedding",
    "distributed/_shard/sharded_tensor/ops/test_embedding_bag",
    "distributed/_shard/sharded_tensor/ops/test_binary_cmp",
    "distributed/_shard/sharded_tensor/ops/test_init",
    "distributed/_shard/sharded_optim/test_sharded_optim",
] + FSDP_TEST

ROCM_BLOCKLIST = [
    "distributed/rpc/test_faulty_agent",
    "distributed/rpc/test_tensorpipe_agent",
    "distributed/rpc/test_share_memory",
    "distributed/rpc/cuda/test_tensorpipe_agent",
    "inductor/test_max_autotune",  # taking excessive time, many tests >30 min
    "test_determination",
    "test_jit_legacy",
    "test_cuda_nvml_based_avail",
    "test_jit_cuda_fuser",
    "test_openreg",
]

S390X_BLOCKLIST = [
    # these tests fail due to various reasons
    "dynamo/test_misc",
    "inductor/test_cpu_repro",
    "inductor/test_cpu_select_algorithm",
    "inductor/test_torchinductor_codegen_dynamic_shapes",
    "lazy/test_meta_kernel",
    "onnx/test_utility_funs",
    "profiler/test_profiler",
    "test_jit",
    "dynamo/test_utils",
    "test_nn",
    # these tests run long and fail in addition to that
    "dynamo/test_dynamic_shapes",
    "test_quantization",
    "inductor/test_torchinductor",
    "inductor/test_torchinductor_dynamic_shapes",
    "inductor/test_torchinductor_opinfo",
    # these tests fail when cuda is not available
    "inductor/test_aot_inductor",
    "inductor/test_best_config",
    "inductor/test_cudacodecache",
    "inductor/test_inductor_utils",
    "inductor/test_inplacing_pass",
    "inductor/test_kernel_benchmark",
    "inductor/test_max_autotune",
    "inductor/test_move_constructors_to_cuda",
    "inductor/test_multi_kernel",
    "inductor/test_pattern_matcher",
    "inductor/test_perf",
    "inductor/test_select_algorithm",
    "inductor/test_snode_runtime",
    "inductor/test_triton_wrapper",
    # these tests fail when mkldnn is not available
    "inductor/test_custom_post_grad_passes",
    "inductor/test_mkldnn_pattern_matcher",
    "test_metal",
    # lacks quantization support
    "onnx/test_models_quantized_onnxruntime",
    "onnx/test_pytorch_onnx_onnxruntime",
    # sysctl -n hw.memsize is not available
    "test_mps",
    # https://github.com/pytorch/pytorch/issues/102078
    "test_decomp",
    # https://github.com/pytorch/pytorch/issues/146698
    "test_model_exports_to_core_aten",
    # runs very long, skip for now
    "inductor/test_layout_optim",
    "test_fx",
    # some false errors
    "doctests",
    # new failures to investigate and fix
    "test_tensorboard",
    # onnx + protobuf failure, see
    # https://github.com/protocolbuffers/protobuf/issues/22104
    "dynamo/test_backends",
    "dynamo/test_modules",
    "inductor/test_config",
    "test_public_bindings",
    "test_testing",
    # depend on z3-solver
    "fx/test_z3_gradual_types",
    "test_proxy_tensor",
    "test_openreg",
]

XPU_BLOCKLIST = [
    "test_autograd",
    "profiler/test_memory_profiler",
    "test_openreg",
]

XPU_TEST = [
    "test_xpu",
]

# The tests inside these files should never be run in parallel with each other
RUN_PARALLEL_BLOCKLIST = [
    "test_extension_utils",
    "test_cpp_extensions_jit",
    "test_cpp_extensions_stream_and_event",
    "test_cpp_extensions_mtia_backend",
    "test_jit_disabled",
    "test_mobile_optimizer",
    "test_multiprocessing",
    "test_multiprocessing_spawn",
    "test_namedtuple_return_api",
    "test_openreg",
    "test_overrides",
    "test_show_pickle",
    "test_tensorexpr",
    "test_cuda_primary_ctx",
    "test_cuda_trace",
    "inductor/test_benchmark_fusion",
    "test_cuda_nvml_based_avail",
    # temporarily sets a global config
    "test_autograd_fallback",
    "inductor/test_compiler_bisector",
    "test_privateuseone_python_backend",
] + FSDP_TEST

# Test files that should always be run serially with other test files,
# but it's okay if the tests inside them are run in parallel with each other.
CI_SERIAL_LIST = [
    "test_nn",
    "test_fake_tensor",
    "test_cpp_api_parity",
    "test_reductions",
    "test_fx_backends",
    "test_cpp_extensions_jit",
    "test_torch",
    "test_tensor_creation_ops",
    "test_dispatch",
    "test_python_dispatch",  # torch.library creation and deletion must be serialized
    "test_spectral_ops",  # Cause CUDA illegal memory access https://github.com/pytorch/pytorch/issues/88916
    "nn/test_pooling",
    "nn/test_convolution",  # Doesn't respect set_per_process_memory_fraction, results in OOM for other tests in slow gradcheck
    "distributions/test_distributions",
    "test_fx",  # gets SIGKILL
    "functorch/test_memory_efficient_fusion",  # Cause CUDA OOM on ROCm
    "test_utils",  # OOM
    "test_sort_and_select",  # OOM
    "test_backward_compatible_arguments",  # OOM
    "test_autocast",  # OOM
    "test_native_mha",  # OOM
    "test_module_hooks",  # OOM
    "inductor/test_max_autotune",
    "inductor/test_cutlass_backend",  # slow due to many nvcc compilation steps,
    "inductor/test_flex_attention",  # OOM
]
# A subset of onnx tests that cannot run in parallel due to high memory usage.
ONNX_SERIAL_LIST = [
    "onnx/test_models",
    "onnx/test_models_quantized_onnxruntime",
    "onnx/test_models_onnxruntime",
    "onnx/test_custom_ops",
    "onnx/test_utility_funs",
]

# A subset of our TEST list that validates PyTorch's ops, modules, and autograd function as expected
CORE_TEST_LIST = [
    "test_autograd",
    "test_autograd_fallback",
    "test_modules",
    "test_nn",
    "test_ops",
    "test_ops_gradients",
    "test_ops_fwd_gradients",
    "test_ops_jit",
    "test_torch",
]


# if a test file takes longer than 5 min, we add it to TARGET_DET_LIST
SLOW_TEST_THRESHOLD = 300

DISTRIBUTED_TESTS_CONFIG = {}


if dist.is_available():
    num_gpus = torch.cuda.device_count()
    DISTRIBUTED_TESTS_CONFIG["test"] = {"WORLD_SIZE": "1"}
    if not TEST_WITH_ROCM and dist.is_mpi_available():
        DISTRIBUTED_TESTS_CONFIG["mpi"] = {
            "WORLD_SIZE": "3",
        }
    if dist.is_nccl_available() and num_gpus > 0:
        DISTRIBUTED_TESTS_CONFIG["nccl"] = {
            "WORLD_SIZE": f"{num_gpus}",
        }
    if dist.is_gloo_available():
        DISTRIBUTED_TESTS_CONFIG["gloo"] = {
            # TODO: retire testing gloo with CUDA
            "WORLD_SIZE": f"{num_gpus if num_gpus > 0 else 3}",
        }
    del num_gpus
    # Test with UCC backend is deprecated.
    # See https://github.com/pytorch/pytorch/pull/137161
    # if dist.is_ucc_available():
    #     DISTRIBUTED_TESTS_CONFIG["ucc"] = {
    #         "WORLD_SIZE": f"{torch.cuda.device_count()}",
    #         "UCX_TLS": "tcp,cuda",
    #         "UCC_TLS": "nccl,ucp,cuda",
    #         "UCC_TL_UCP_TUNE": "cuda:0",  # don't use UCP TL on CUDA as it is not well supported
    #         "UCC_EC_CUDA_USE_COOPERATIVE_LAUNCH": "n",  # CI nodes (M60) fail if it is on
    #     }

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if n.startswith("SIG") and "_" not in n
}

CPP_EXTENSIONS_ERROR = """
Ninja (https://ninja-build.org) is required for some of the C++ extensions
tests, but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`. Alternatively, disable said tests with
`run_test.py --exclude test_cpp_extensions_aot_ninja test_cpp_extensions_jit`.
"""

PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))

JIT_EXECUTOR_TESTS = [
    "test_jit_profiling",
    "test_jit_legacy",
    "test_jit_fuser_legacy",
]

INDUCTOR_TESTS = [test for test in TESTS if test.startswith(INDUCTOR_TEST_PREFIX)]
DISTRIBUTED_TESTS = [test for test in TESTS if test.startswith(DISTRIBUTED_TEST_PREFIX)]
TORCH_EXPORT_TESTS = [test for test in TESTS if test.startswith("export")]
AOT_DISPATCH_TESTS = [
    test for test in TESTS if test.startswith("functorch/test_aotdispatch")
]
FUNCTORCH_TESTS = [test for test in TESTS if test.startswith("functorch")]
DYNAMO_CORE_TESTS = [test for test in TESTS if test.startswith("dynamo")]
ONNX_TESTS = [test for test in TESTS if test.startswith("onnx")]
QUANTIZATION_TESTS = [test for test in TESTS if test.startswith("test_quantization")]


def _is_cpp_test(test):
    # Note: tests underneath cpp_extensions are different from other cpp tests
    # in that they utilize the usual python test infrastructure.
    return test.startswith(CPP_TEST_PREFIX) and not test.startswith("cpp_extensions")


CPP_TESTS = [test for test in TESTS if _is_cpp_test(test)]

TESTS_REQUIRING_LAPACK = [
    "distributions/test_constraints",
    "distributions/test_distributions",
]

# These are just the slowest ones, this isn't an exhaustive list.
TESTS_NOT_USING_GRADCHECK = [
    # Note that you should use skipIfSlowGradcheckEnv if you do not wish to
    # skip all the tests in that file, e.g. test_mps
    "doctests",
    "test_meta",
    "test_hub",
    "test_fx",
    "test_decomp",
    "test_cpp_extensions_jit",
    "test_jit",
    "test_matmul_cuda",
    "test_ops",
    "test_ops_jit",
    "dynamo/test_recompile_ux",
    "inductor/test_compiled_optimizers",
    "inductor/test_cutlass_backend",
    "inductor/test_max_autotune",
    "inductor/test_select_algorithm",
    "inductor/test_smoke",
    "test_quantization",
]


def print_to_stderr(message):
    print(message, file=sys.stderr)


def get_executable_command(options, disable_coverage=False, is_cpp_test=False):
    if options.coverage and not disable_coverage:
        if not is_cpp_test:
            executable = ["coverage", "run", "--parallel-mode", "--source=torch"]
        else:
            # TODO: C++ with coverage is not yet supported
            executable = []
    else:
        if not is_cpp_test:
            executable = [sys.executable, "-bb"]
        else:
            executable = ["pytest"]

    return executable


def run_test(
    test_module: ShardedTest,
    test_directory,
    options,
    launcher_cmd=None,
    extra_unittest_args=None,
    env=None,
    print_log=True,
) -> int:
    scribe_token = os.getenv("SCRIBE_GRAPHQL_ACCESS_TOKEN", "")
    if scribe_token:
        print_to_stderr("SCRIBE_GRAPHQL_ACCESS_TOKEN is set")
    else:
        print_to_stderr("SCRIBE_GRAPHQL_ACCESS_TOKEN is NOT set")

    env = env or os.environ.copy()
    maybe_set_hip_visible_devies()
    unittest_args = options.additional_args.copy()
    test_file = test_module.name
    stepcurrent_key = test_file

    is_distributed_test = test_file.startswith(DISTRIBUTED_TEST_PREFIX)
    is_cpp_test = _is_cpp_test(test_file)
    # NB: Rerun disabled tests depends on pytest-flakefinder and it doesn't work with
    # pytest-cpp atm. We also don't have support to disable C++ test yet, so it's ok
    # to just return successfully here
    if is_cpp_test and RERUN_DISABLED_TESTS:
        print_to_stderr(
            "Skipping C++ tests when running under RERUN_DISABLED_TESTS mode"
        )
        return 0

    if is_cpp_test:
        stepcurrent_key = f"{test_file}_{os.urandom(8).hex()}"
    else:
        unittest_args.extend(
            [
                f"--shard-id={test_module.shard}",
                f"--num-shards={test_module.num_shards}",
            ]
        )
        stepcurrent_key = f"{test_file}_{test_module.shard}_{os.urandom(8).hex()}"

    if options.verbose:
        unittest_args.append(f"-{'v' * options.verbose}")  # in case of pytest

    if test_file in RUN_PARALLEL_BLOCKLIST:
        unittest_args = [
            arg for arg in unittest_args if not arg.startswith("--run-parallel")
        ]

    if extra_unittest_args:
        assert isinstance(extra_unittest_args, list)
        unittest_args.extend(extra_unittest_args)

    # If using pytest, replace -f with equivalent -x
    if options.pytest:
        unittest_args.extend(
            get_pytest_args(
                options,
                is_cpp_test=is_cpp_test,
                is_distributed_test=is_distributed_test,
            )
        )
        unittest_args.extend(test_module.get_pytest_args())
        replacement = {"-f": "-x", "-dist=loadfile": "--dist=loadfile"}
        unittest_args = [replacement.get(arg, arg) for arg in unittest_args]

    if options.showlocals:
        if options.pytest:
            unittest_args.extend(["--showlocals", "--tb=long", "--color=yes"])
        else:
            unittest_args.append("--locals")

    # NB: These features are not available for C++ tests, but there is little incentive
    # to implement it because we have never seen a flaky C++ test before.
    if IS_CI and not is_cpp_test:
        ci_args = ["--import-slow-tests", "--import-disabled-tests"]
        if RERUN_DISABLED_TESTS:
            ci_args.append("--rerun-disabled-tests")
        # use the downloaded test cases configuration, not supported in pytest
        unittest_args.extend(ci_args)

    if test_file in PYTEST_SKIP_RETRIES:
        if not options.pytest:
            raise RuntimeError(
                "A test running without pytest cannot skip retries using "
                "the PYTEST_SKIP_RETRIES set."
            )
        unittest_args = [arg for arg in unittest_args if "--reruns" not in arg]

    # Extra arguments are not supported with pytest
    executable = get_executable_command(options, is_cpp_test=is_cpp_test)
    if not executable:
        # If there is no eligible executable returning here, it means an unsupported
        # case such as coverage for C++ test. So just returning ok makes sense
        return 0

    if is_cpp_test:
        # C++ tests are not the regular test directory
        if CPP_TESTS_DIR:
            cpp_test = os.path.join(
                CPP_TESTS_DIR,
                test_file.replace(f"{CPP_TEST_PREFIX}/", ""),
            )
        else:
            cpp_test = os.path.join(
                Path(test_directory).parent,
                CPP_TEST_PATH,
                test_file.replace(f"{CPP_TEST_PREFIX}/", ""),
            )

        argv = [
            cpp_test if sys.platform != "win32" else cpp_test + ".exe"
        ] + unittest_args
    else:
        # Can't call `python -m unittest test_*` here because it doesn't run code
        # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
        argv = [test_file + ".py"] + unittest_args

    os.makedirs(REPO_ROOT / "test" / "test-reports", exist_ok=True)
    if options.pipe_logs:
        log_fd, log_path = tempfile.mkstemp(
            dir=REPO_ROOT / "test" / "test-reports",
            prefix=f"{sanitize_file_name(str(test_module))}_",
            suffix="_toprint.log",
        )
        os.close(log_fd)

    command = (launcher_cmd or []) + executable + argv
    should_retry = (
        "--subprocess" not in command
        and not RERUN_DISABLED_TESTS
        and not is_cpp_test
        and "-n" not in command
    )
    timeout = (
        None
        if not options.enable_timeout
        else THRESHOLD * 6
        if IS_SLOW
        else THRESHOLD * 3
        if should_retry
        and isinstance(test_module, ShardedTest)
        and test_module.time is not None
        else THRESHOLD * 3
        if is_cpp_test
        else None
    )
    print_to_stderr(f"Executing {command} ... [{datetime.now()}]")

    with ExitStack() as stack:
        output = None
        if options.pipe_logs:
            output = stack.enter_context(open(log_path, "w"))

        if should_retry:
            ret_code, was_rerun = run_test_retries(
                command,
                test_directory,
                env,
                timeout,
                stepcurrent_key,
                output,
                options.continue_through_error,
                test_file,
                options,
            )
        else:
            command.extend([f"--sc={stepcurrent_key}", "--print-items"])
            ret_code, was_rerun = retry_shell(
                command,
                test_directory,
                stdout=output,
                stderr=output,
                env=env,
                timeout=timeout,
                retries=0,
            )

            # Pytest return code 5 means no test is collected. Exit code 4 is
            # returned when the binary is not a C++ test executable, but 4 can
            # also be returned if the file fails before running any tests. All
            # binary files under build/bin that are not C++ test at the time of
            # this writing have been excluded and new ones should be added to
            # the list of exclusions in tools/testing/discover_tests.py
            ret_code = 0 if ret_code == 5 else ret_code

    if options.pipe_logs and print_log:
        handle_log_file(
            test_module, log_path, failed=(ret_code != 0), was_rerun=was_rerun
        )
    return ret_code


def install_cpp_extensions(extensions_dir, env=os.environ):
    # Wipe the build folder, if it exists already
    build_dir = os.path.join(extensions_dir, "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    # Build the test cpp extensions modules
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        ".",
        "--root",
        "./install",
    ]
    return_code = shell(cmd, cwd=extensions_dir, env=env)
    if return_code != 0:
        return None, return_code

    # Get the site-packages directory prepared for PYTHONPATH
    platlib_path = sysconfig.get_paths()["platlib"]
    platlib_rel = os.path.relpath(
        platlib_path, os.path.splitdrive(platlib_path)[0] + os.sep
    )
    install_directory = os.path.join(extensions_dir, "install", platlib_rel)

    assert install_directory, "install_directory must not be empty"
    return install_directory, 0


@contextlib.contextmanager
def extend_python_path(install_directories):
    python_path = os.environ.get("PYTHONPATH", "")
    try:
        os.environ["PYTHONPATH"] = os.pathsep.join(install_directories + [python_path])
        yield
    finally:
        os.environ["PYTHONPATH"] = python_path


def try_set_cpp_stack_traces(env, command, set=True):
    # Print full c++ stack traces during retries
    env = env or {}
    env["TORCH_SHOW_CPP_STACKTRACES"] = "1" if set else "0"
    return env


def run_test_retries(
    command,
    test_directory,
    env,
    timeout,
    stepcurrent_key,
    output,
    continue_through_error,
    test_file,
    options,
):
    # Run the test with -x to stop at first failure.  Rerun the test by itself.
    # If it succeeds, move on to the rest of the tests in a new process.  If it
    # still fails, see below
    #
    # If continue through error is not set, then we fail fast.
    #
    # If continue through error is set, then we skip that test, and keep going.
    # Basically if the same test fails 3 times in a row, skip the test on the
    # next run, but still fail in the end. I take advantage of the value saved
    # in stepcurrent to keep track of the most recently run test (which is the
    # one that failed if there was a failure).

    def print_to_file(s):
        print(s, file=output, flush=True)

    num_failures = defaultdict(int)

    def read_pytest_cache(key: str) -> Any:
        cache_file = (
            REPO_ROOT / ".pytest_cache/v/cache/stepcurrent" / stepcurrent_key / key
        )
        try:
            with open(cache_file) as f:
                return f.read()
        except FileNotFoundError:
            return None

    print_items = ["--print-items"]
    sc_command = f"--sc={stepcurrent_key}"
    while True:
        ret_code, _ = retry_shell(
            command + [sc_command] + print_items,
            test_directory,
            stdout=output,
            stderr=output,
            env=env,
            timeout=timeout,
            retries=0,  # no retries here, we do it ourselves, this is because it handles timeout exceptions well
        )
        ret_code = 0 if ret_code == 5 else ret_code
        if ret_code == 0 and not sc_command.startswith("--rs="):
            break  # Got to the end of the test suite successfully
        signal_name = f" ({SIGNALS_TO_NAMES_DICT[-ret_code]})" if ret_code < 0 else ""
        print_to_file(f"Got exit code {ret_code}{signal_name}")

        # Read what just failed/ran
        try:
            current_failure = read_pytest_cache("lastrun")
            if current_failure is None:
                raise FileNotFoundError
            if current_failure == "null":
                current_failure = f"'{test_file}'"
        except FileNotFoundError:
            print_to_file(
                "No stepcurrent file found. Either pytest didn't get to run (e.g. import error)"
                + " or file got deleted (contact dev infra)"
            )
            break

        env = try_set_cpp_stack_traces(env, command, set=False)
        if ret_code != 0:
            num_failures[current_failure] += 1

        if ret_code == 0:
            # Rerunning the previously failing test succeeded, so now we can
            # skip it and move on
            sc_command = f"--scs={stepcurrent_key}"
            print_to_file(
                "Test succeeeded in new process, continuing with the rest of the tests"
            )
        elif num_failures[current_failure] >= 3:
            # This is for log classifier so it can prioritize consistently
            # failing tests instead of reruns. [1:-1] to remove quotes
            print_to_file(f"FAILED CONSISTENTLY: {current_failure[1:-1]}")
            if (
                read_pytest_cache("made_failing_xml") == "false"
                and IS_CI
                and options.upload_artifacts_while_running
            ):
                upload_adhoc_failure_json(test_file, current_failure[1:-1])

            if not continue_through_error:
                print_to_file("Stopping at first consistent failure")
                break
            sc_command = f"--scs={stepcurrent_key}"
            print_to_file(
                "Test failed consistently, "
                "continuing with the rest of the tests due to continue-through-error being set"
            )
        else:
            env = try_set_cpp_stack_traces(env, command, set=True)
            sc_command = f"--rs={stepcurrent_key}"
            print_to_file("Retrying single test...")
        print_items = []  # do not continue printing them, massive waste of space

    consistent_failures = [x[1:-1] for x in num_failures if num_failures[x] >= 3]
    flaky_failures = [x[1:-1] for x in num_failures if 0 < num_failures[x] < 3]
    if len(flaky_failures) > 0:
        print_to_file(
            "The following tests failed and then succeeded when run in a new process"
            + f"{flaky_failures}",
        )
    if len(consistent_failures) > 0:
        print_to_file(f"The following tests failed consistently: {consistent_failures}")
        return 1, True
    return ret_code, any(x > 0 for x in num_failures.values())


def run_test_with_subprocess(test_module, test_directory, options):
    return run_test(
        test_module, test_directory, options, extra_unittest_args=["--subprocess"]
    )


def _test_cpp_extensions_aot(test_directory, options, use_ninja):
    if use_ninja:
        try:
            from torch.utils import cpp_extension

            cpp_extension.verify_ninja_availability()
        except RuntimeError:
            print_to_stderr(CPP_EXTENSIONS_ERROR)
            return 1

    # Wipe the build folder, if it exists already
    cpp_extensions_test_dir = os.path.join(test_directory, "cpp_extensions")
    cpp_extensions_test_build_dir = os.path.join(cpp_extensions_test_dir, "build")
    if os.path.exists(cpp_extensions_test_build_dir):
        shutil.rmtree(cpp_extensions_test_build_dir)

    # Build the test cpp extensions modules
    shell_env = os.environ.copy()
    shell_env["USE_NINJA"] = str(1 if use_ninja else 0)
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        ".",
        "--root",
        "./install",
    ]
    wheel_cmd = [sys.executable, "-m", "build", "--wheel", "--no-isolation"]
    return_code = shell(install_cmd, cwd=cpp_extensions_test_dir, env=shell_env)
    if return_code != 0:
        return return_code
    if sys.platform != "win32":
        exts_to_build = [
            (install_cmd, "no_python_abi_suffix_test"),
        ]
        if TEST_CUDA or TEST_XPU:
            exts_to_build.append((wheel_cmd, "python_agnostic_extension"))
        if TEST_CUDA:
            exts_to_build.append((install_cmd, "libtorch_agnostic_extension"))
        for cmd, extension_dir in exts_to_build:
            return_code = shell(
                cmd,
                cwd=os.path.join(cpp_extensions_test_dir, extension_dir),
                env=shell_env,
            )
            if return_code != 0:
                return return_code

    from shutil import copyfile

    os.environ["USE_NINJA"] = shell_env["USE_NINJA"]
    test_module = "test_cpp_extensions_aot" + ("_ninja" if use_ninja else "_no_ninja")
    copyfile(
        test_directory + "/test_cpp_extensions_aot.py",
        test_directory + "/" + test_module + ".py",
    )

    try:
        cpp_extensions = os.path.join(test_directory, "cpp_extensions")
        install_directories = []
        # install directory is the one that is named site-packages
        for root, directories, _ in os.walk(os.path.join(cpp_extensions, "install")):
            for directory in directories:
                if "-packages" in directory:
                    install_directories.append(os.path.join(root, directory))

        for root, directories, _ in os.walk(
            os.path.join(cpp_extensions, "libtorch_agnostic_extension", "install")
        ):
            for directory in directories:
                if "-packages" in directory:
                    install_directories.append(os.path.join(root, directory))

        with extend_python_path(install_directories):
            return run_test(ShardedTest(test_module, 1, 1), test_directory, options)
    finally:
        if os.path.exists(test_directory + "/" + test_module + ".py"):
            os.remove(test_directory + "/" + test_module + ".py")
        os.environ.pop("USE_NINJA")


def test_cpp_extensions_aot_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=True)


def test_cpp_extensions_aot_no_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=False)


def test_autoload_enable(test_module, test_directory, options):
    return _test_autoload(test_directory, options, enable=True)


def test_autoload_disable(test_module, test_directory, options):
    return _test_autoload(test_directory, options, enable=False)


def _test_autoload(test_directory, options, enable=True):
    cpp_extensions_test_dir = os.path.join(test_directory, "cpp_extensions")
    install_directory, return_code = install_cpp_extensions(cpp_extensions_test_dir)
    if return_code != 0:
        return return_code

    try:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = str(int(enable))
        with extend_python_path([install_directory]):
            cmd = [sys.executable, "test_autoload.py"]
            return_code = shell(cmd, cwd=test_directory, env=os.environ)
            return return_code
    finally:
        os.environ.pop("TORCH_DEVICE_BACKEND_AUTOLOAD")


# test_openreg is designed to run all tests under torch_openreg, which
# is an torch backend similar to CUDA or MPS and implemented by using
# third-party accelerator integration mechanism. Therefore, if all the
# tests under torch_openreg are passing, it can means that the mechanism
# mentioned above is working as expected.
def test_openreg(test_module, test_directory, options):
    openreg_dir = os.path.join(
        test_directory, "cpp_extensions", "open_registration_extension", "torch_openreg"
    )
    install_dir, return_code = install_cpp_extensions(openreg_dir)
    if return_code != 0:
        return return_code

    with extend_python_path([install_dir]):
        cmd = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            os.path.join(openreg_dir, "tests"),
            "-v",
        ]
        return shell(cmd, cwd=test_directory, env=os.environ)


def test_distributed(test_module, test_directory, options):
    mpi_available = shutil.which("mpiexec")
    if options.verbose and not mpi_available:
        print_to_stderr("MPI not available -- MPI backend tests will be skipped")

    config = DISTRIBUTED_TESTS_CONFIG
    for backend, env_vars in config.items():
        if sys.platform == "win32" and backend != "gloo":
            continue
        if backend == "mpi" and not mpi_available:
            continue
        for with_init_file in {True, False}:
            if sys.platform == "win32" and not with_init_file:
                continue
            tmp_dir = tempfile.mkdtemp()
            init_method = "file" if with_init_file else "env"
            if options.verbose:
                with_init = f"with {init_method} init_method"
                print_to_stderr(
                    f"Running distributed tests for the {backend} backend {with_init}"
                )
            old_environ = dict(os.environ)
            os.environ["TEMP_DIR"] = tmp_dir
            os.environ["BACKEND"] = backend
            os.environ.update(env_vars)
            report_tag = f"dist-{backend}" if backend != "test" else ""
            report_tag += f"-init-{init_method}"
            os.environ["TEST_REPORT_SOURCE_OVERRIDE"] = report_tag
            try:
                os.mkdir(os.path.join(tmp_dir, "barrier"))
                os.mkdir(os.path.join(tmp_dir, "test_dir"))
                if backend == "mpi":
                    # test mpiexec for --noprefix option
                    with open(os.devnull, "w") as devnull:
                        allowrunasroot_opt = (
                            "--allow-run-as-root"
                            if subprocess.call(
                                'mpiexec --allow-run-as-root -n 1 bash -c ""',
                                shell=True,
                                stdout=devnull,
                                stderr=subprocess.STDOUT,
                            )
                            == 0
                            else ""
                        )
                        noprefix_opt = (
                            "--noprefix"
                            if subprocess.call(
                                f'mpiexec {allowrunasroot_opt} -n 1 --noprefix bash -c ""',
                                shell=True,
                                stdout=devnull,
                                stderr=subprocess.STDOUT,
                            )
                            == 0
                            else ""
                        )

                    mpiexec = ["mpiexec", "-n", "3", noprefix_opt, allowrunasroot_opt]

                    return_code = run_test(
                        test_module, test_directory, options, launcher_cmd=mpiexec
                    )
                else:
                    return_code = run_test(
                        test_module,
                        test_directory,
                        options,
                        extra_unittest_args=["--subprocess"],
                    )
                if return_code != 0:
                    return return_code
            finally:
                shutil.rmtree(tmp_dir)
                os.environ.clear()
                os.environ.update(old_environ)
    return 0


def run_doctests(test_module, test_directory, options):
    """
    Assumes the incoming test module is called doctest, and simply executes the
    xdoctest runner on the torch library itself.
    """
    import xdoctest

    pkgpath = Path(torch.__file__).parent

    exclude_module_list = ["torch._vendor.*"]
    enabled = {
        # TODO: expose these options to the user
        # For now disable all feature-conditional tests
        # 'lapack': 'auto',
        # 'cuda': 'auto',
        # 'cuda1': 'auto',
        # 'qengine': 'auto',
        "lapack": 0,
        "cuda": 0,
        "cuda1": 0,
        "qengine": 0,
        "autograd_profiler": 0,
        "cpp_ext": 0,
        "monitor": 0,
        "onnx": "auto",
    }

    # Resolve "auto" based on a test to determine if the feature is available.
    if enabled["cuda"] == "auto" and torch.cuda.is_available():
        enabled["cuda"] = True

    if (
        enabled["cuda1"] == "auto"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        enabled["cuda1"] = True

    if enabled["lapack"] == "auto" and torch._C.has_lapack:
        enabled["lapack"] = True

    if enabled["qengine"] == "auto":
        try:
            # Is there a better check if quantization is enabled?
            import torch.ao.nn.quantized as nnq  # NOQA: F401

            torch.backends.quantized.engine = "qnnpack"
            torch.backends.quantized.engine = "fbgemm"
        except (ImportError, RuntimeError):
            ...
        else:
            enabled["qengine"] = True

    if enabled["onnx"] == "auto":
        try:
            import onnx  # NOQA: F401
            import onnxruntime  # NOQA: F401
            import onnxscript  # NOQA: F401
        except ImportError:
            exclude_module_list.append("torch.onnx.*")
            enabled["onnx"] = False
        else:
            enabled["onnx"] = True

    # Set doctest environment variables
    if enabled["cuda"]:
        os.environ["TORCH_DOCTEST_CUDA"] = "1"

    if enabled["cuda1"]:
        os.environ["TORCH_DOCTEST_CUDA1"] = "1"

    if enabled["lapack"]:
        os.environ["TORCH_DOCTEST_LAPACK"] = "1"

    if enabled["qengine"]:
        os.environ["TORCH_DOCTEST_QENGINE"] = "1"

    if enabled["autograd_profiler"]:
        os.environ["TORCH_DOCTEST_AUTOGRAD_PROFILER"] = "1"

    if enabled["cpp_ext"]:
        os.environ["TORCH_DOCTEST_CPP_EXT"] = "1"

    if enabled["monitor"]:
        os.environ["TORCH_DOCTEST_MONITOR"] = "1"

    if enabled["onnx"]:
        os.environ["TORCH_DOCTEST_ONNX"] = "1"

    if torch.mps.is_available():
        os.environ["TORCH_DOCTEST_MPS"] = "1"

    if torch.distributed.is_available():
        os.environ["TORCH_DOCTEST_DISTRIBUTED"] = "1"

    if 0:
        # TODO: could try to enable some of these
        os.environ["TORCH_DOCTEST_QUANTIZED_DYNAMIC"] = "1"
        os.environ["TORCH_DOCTEST_ANOMALY"] = "1"
        os.environ["TORCH_DOCTEST_AUTOGRAD"] = "1"
        os.environ["TORCH_DOCTEST_HUB"] = "1"
        os.environ["TORCH_DOCTEST_DATALOADER"] = "1"
        os.environ["TORCH_DOCTEST_FUTURES"] = "1"

    pkgpath = os.path.dirname(torch.__file__)

    xdoctest_config = {
        "global_exec": r"\n".join(
            [
                "from torch import nn",
                "import torch.nn.functional as F",
                "import torch",
            ]
        ),
        "analysis": "static",  # set to "auto" to test doctests in compiled modules
        "style": "google",
        "options": "+IGNORE_WHITESPACE",
    }
    xdoctest_verbose = max(1, options.verbose)
    run_summary = xdoctest.runner.doctest_module(
        os.fspath(pkgpath),
        config=xdoctest_config,
        verbose=xdoctest_verbose,
        command=options.xdoctest_command,
        argv=[],
        exclude=exclude_module_list,
    )
    result = 1 if run_summary.get("n_failed", 0) else 0
    return result


def sanitize_file_name(file: str):
    return file.replace("\\", ".").replace("/", ".").replace(" ", "_")


def handle_log_file(
    test: ShardedTest, file_path: str, failed: bool, was_rerun: bool
) -> None:
    test = str(test)
    with open(file_path, errors="ignore") as f:
        full_text = f.read()

    new_file = "test/test-reports/" + sanitize_file_name(
        f"{test}_{os.urandom(8).hex()}_.log"
    )
    os.rename(file_path, REPO_ROOT / new_file)

    if not failed and not was_rerun and "=== RERUNS ===" not in full_text:
        # If success + no retries (idk how else to check for test level retries
        # other than reparse xml), print only what tests ran
        print_to_stderr(
            f"\n{test} was successful, full logs can be found in artifacts with path {new_file}"
        )
        for line in full_text.splitlines():
            if re.search("Running .* items in this shard:", line):
                print_to_stderr(line.rstrip())
        print_to_stderr("")
        return

    # otherwise: print entire file
    print_to_stderr(f"\nPRINTING LOG FILE of {test} ({new_file})")
    print_to_stderr(full_text)
    print_to_stderr(f"FINISHED PRINTING LOG FILE of {test} ({new_file})\n")


def get_pytest_args(options, is_cpp_test=False, is_distributed_test=False):
    if is_distributed_test:
        # Distributed tests do not support rerun, see https://github.com/pytorch/pytorch/issues/162978
        rerun_options = ["-x", "--reruns=0"]
    elif RERUN_DISABLED_TESTS:
        # ASAN tests are too slow, so running them x50 will cause the jobs to timeout after
        # 3+ hours. So, let's opt for less number of reruns. We need at least 150 instances of the
        # test every 2 weeks to satisfy the SQL query (15 x 14 = 210).
        count = 15 if TEST_WITH_ASAN else 50
        # When under rerun-disabled-tests mode, run the same tests multiple times to determine their
        # flakiness status. Default to 50 re-runs
        rerun_options = ["--flake-finder", f"--flake-runs={count}"]
    else:
        # When under the normal mode, retry a failed test 2 more times. -x means stop at the first
        # failure
        rerun_options = ["-x", "--reruns=2"]

    pytest_args = [
        "-vv",
        "-rfEX",
    ]
    if not is_cpp_test:
        # C++ tests need to be run with pytest directly, not via python
        # We have a custom pytest shard that conflicts with the normal plugin
        pytest_args.extend(["-p", "no:xdist", "--use-pytest"])
    else:
        # Use pytext-dist to run C++ tests in parallel as running them sequentially using run_test
        # is much slower than running them directly
        pytest_args.extend(["-n", str(NUM_PROCS)])

        if TEST_SAVE_XML:
            # Add the option to generate XML test report here as C++ tests
            # won't go into common_utils
            test_report_path = get_report_path(pytest=True)
            pytest_args.extend(["--junit-xml-reruns", test_report_path])

    if options.pytest_k_expr:
        pytest_args.extend(["-k", options.pytest_k_expr])

    pytest_args.extend(rerun_options)
    return pytest_args


def run_ci_sanity_check(test: ShardedTest, test_directory, options):
    assert test.name == "test_ci_sanity_check_fail", (
        f"This handler only works for test_ci_sanity_check_fail, got {test.name}"
    )
    ret_code = run_test(test, test_directory, options, print_log=False)
    # This test should fail
    if ret_code != 1:
        return 1
    test_reports_dir = str(REPO_ROOT / "test/test-reports")
    # Delete the log files and xmls generated by the test
    for file in glob.glob(f"{test_reports_dir}/{test.name}*.log"):
        os.remove(file)
    for dirname in glob.glob(f"{test_reports_dir}/**/{test.name}"):
        shutil.rmtree(dirname)
    return 0


CUSTOM_HANDLERS = {
    "test_cuda_primary_ctx": run_test_with_subprocess,
    "test_cuda_nvml_based_avail": run_test_with_subprocess,
    "test_cuda_trace": run_test_with_subprocess,
    "test_cpp_extensions_aot_no_ninja": test_cpp_extensions_aot_no_ninja,
    "test_cpp_extensions_aot_ninja": test_cpp_extensions_aot_ninja,
    "distributed/test_distributed_spawn": test_distributed,
    "distributed/algorithms/quantization/test_quantization": test_distributed,
    "distributed/test_c10d_nccl": run_test_with_subprocess,
    "distributed/test_c10d_gloo": run_test_with_subprocess,
    "distributed/test_c10d_ucc": run_test_with_subprocess,
    "distributed/test_c10d_common": run_test_with_subprocess,
    "distributed/test_c10d_spawn_gloo": run_test_with_subprocess,
    "distributed/test_c10d_spawn_nccl": run_test_with_subprocess,
    "distributed/test_c10d_spawn_ucc": run_test_with_subprocess,
    "distributed/test_store": run_test_with_subprocess,
    "distributed/test_pg_wrapper": run_test_with_subprocess,
    "distributed/rpc/test_faulty_agent": run_test_with_subprocess,
    "distributed/rpc/test_tensorpipe_agent": run_test_with_subprocess,
    "distributed/rpc/test_share_memory": run_test_with_subprocess,
    "distributed/rpc/cuda/test_tensorpipe_agent": run_test_with_subprocess,
    "doctests": run_doctests,
    "test_ci_sanity_check_fail": run_ci_sanity_check,
    "test_autoload_enable": test_autoload_enable,
    "test_autoload_disable": test_autoload_disable,
    "test_openreg": test_openreg,
}


PYTEST_SKIP_RETRIES = {"test_public_bindings"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PyTorch unit test suite",
        epilog="where TESTS is any of: {}".format(", ".join(TESTS)),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print verbose information and test-by-test results",
    )
    parser.add_argument(
        "--showlocals",
        action=argparse.BooleanOptionalAction,
        default=strtobool(os.environ.get("TEST_SHOWLOCALS", "False")),
        help="Show local variables in tracebacks (default: True)",
    )
    parser.add_argument("--jit", "--jit", action="store_true", help="run all jit tests")
    parser.add_argument(
        "--distributed-tests",
        "--distributed-tests",
        action="store_true",
        help="Run all distributed tests",
    )
    parser.add_argument(
        "--include-dynamo-core-tests",
        "--include-dynamo-core-tests",
        action="store_true",
        help=(
            "If this flag is present, we will only run dynamo tests. "
            "If this flag is not present, we will run all tests "
            "(including dynamo tests)."
        ),
    )
    parser.add_argument(
        "--functorch",
        "--functorch",
        action="store_true",
        help=(
            "If this flag is present, we will only run functorch tests. "
            "If this flag is not present, we will run all tests "
            "(including functorch tests)."
        ),
    )
    parser.add_argument(
        "--einops",
        "--einops",
        action="store_true",
        help=(
            "If this flag is present, we will only run einops tests. "
            "If this flag is not present, we will run all tests "
            "(including einops tests)."
        ),
    )
    parser.add_argument(
        "--mps",
        "--mps",
        action="store_true",
        help=("If this flag is present, we will only run test_mps and test_metal"),
    )
    parser.add_argument(
        "--xpu",
        "--xpu",
        action="store_true",
        help=("If this flag is present, we will run xpu tests except XPU_BLOCK_LIST"),
    )
    parser.add_argument(
        "--cpp",
        "--cpp",
        action="store_true",
        help=
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
- **Database**: May involve SQL - watch for injection vulnerabilities

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/run_test.py_docs.md
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

- **File Documentation**: `run_test.py_docs.md_docs.md`
- **Keyword Index**: `run_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
