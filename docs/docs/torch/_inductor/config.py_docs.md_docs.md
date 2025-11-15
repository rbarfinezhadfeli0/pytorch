# Documentation: `docs/torch/_inductor/config.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/config.py_docs.md`
- **Size**: 53,699 bytes (52.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file handles **configuration or setup**.

## Original Source

```markdown
# Documentation: `torch/_inductor/config.py`

## File Metadata

- **Path**: `torch/_inductor/config.py`
- **Size**: 89,105 bytes (87.02 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file handles **configuration or setup**.

## Original Source

```python
import os
import sys
from collections.abc import Callable
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

import torch
import torch._inductor.custom_graph_pass
from torch._environment import is_fbcode
from torch.utils._config_module import Config, get_tristate_env, install_config_module


if TYPE_CHECKING:
    from torch._inductor.choices import InductorChoices

inplace_padding = os.environ.get("TORCHINDUCTOR_INPLACE_PADDING", "1") == "1"
can_inplace_pad_graph_input = False  # ease testing


def fx_graph_remote_cache_default() -> Optional[bool]:
    return get_tristate_env("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE")


def vec_isa_ok_default() -> Optional[bool]:
    if os.environ.get("TORCHINDUCTOR_VEC_ISA_OK") == "1":
        return True
    if os.environ.get("TORCHINDUCTOR_VEC_ISA_OK") == "0":
        return False
    return None


def autotune_remote_cache_default() -> Optional[bool]:
    return get_tristate_env("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE")


def bundled_autotune_remote_cache_default() -> Optional[bool]:
    return get_tristate_env("TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE")


def bundle_triton_into_fx_graph_cache_default() -> Optional[bool]:
    return get_tristate_env(
        "TORCHINDUCTOR_BUNDLE_TRITON_INTO_FX_GRAPH_CACHE",
        True if not is_fbcode() else None,
    )


def static_cuda_launcher_default() -> bool:
    STATIC_CUDA_LAUNCHER_VERSION = 2

    if "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER" in os.environ:
        return os.environ.get("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER") == "1"
    elif is_fbcode():
        version = torch._utils_internal.justknobs_getval_int(
            "pytorch/inductor:static_cuda_launcher_version"
        )
        return version <= STATIC_CUDA_LAUNCHER_VERSION
    else:
        # Default true in OSS
        return True


def prologue_fusion_enabled() -> bool:
    ENABLE_PROLOGUE_FUSION_VERSION = 0

    if "TORCHINDUCTOR_PROLOGUE_FUSION" in os.environ:
        return os.environ.get("TORCHINDUCTOR_PROLOGUE_FUSION") == "1"
    elif is_fbcode():
        jk_name = "pytorch/inductor:prologue_fusion_version"
        version = torch._utils_internal.justknobs_getval_int(jk_name)
        return version <= ENABLE_PROLOGUE_FUSION_VERSION
    else:
        return True


# Enable auto_functionalized_v2 (enabled by default)
enable_auto_functionalized_v2 = (
    os.environ.get("TORCHDYNAMO_AUTO_FUNCTIONALIZED_V2", "1") == "1"
)

# add some debug printouts
debug = False

# Whether to disable a progress bar for autotuning
disable_progress = True

# Whether to enable printing the source code for each future
verbose_progress = False

# Configurable compile worker logging path for subproc_pool
worker_log_path = (
    "/logs/dedicated_log_torch_compile_worker_rank" if is_fbcode() else None
)

# precompilation timeout
precompilation_timeout_seconds: int = 60 * 60

# use fx aot graph codegen cache
fx_graph_cache: bool = Config(
    justknob="pytorch/remote_cache:enable_local_fx_graph_cache",
    env_name_default="TORCHINDUCTOR_FX_GRAPH_CACHE_DEFAULT",
    env_name_force="TORCHINDUCTOR_FX_GRAPH_CACHE",
    default=True,
)

remote_gemm_autotune_cache: bool = False

# use remote fx aot graph codegen cache
# False: Disables the cache
# True: Enables the cache
# None: Not set -- Off for OSS, JustKnobs based for internal
fx_graph_remote_cache: Optional[bool] = fx_graph_remote_cache_default()

# should we bundle triton caching into fx graph cache
bundle_triton_into_fx_graph_cache: Optional[bool] = (
    bundle_triton_into_fx_graph_cache_default()
)

non_blocking_remote_cache_write: bool = Config(
    justknob="pytorch/remote_cache:enable_non_blocking_remote_cache_write_v2",
    env_name_force="TORCHINDUCTOR_NON_BLOCKING_REMOTE_CACHE_WRITE",
    default=True,
)

# Enable autotune local cache.
#
# See bundled_autotune_remote_cache for the effect this flag has on the bundled
# remote cache.
autotune_local_cache: bool = True

# Enable autotune remote cache.
#
# Enables/disables the autotune remote cache regardless of the state of
# autotune_local_cache. If both local and remote are enabled then on write both
# are written and on read local is checked first and only on a cache miss is
# remote read.
#
# False: Disables the cache
# True: Enables the cache
# None: Not set -- Off for OSS, JustKnobs based for internal
autotune_remote_cache: Optional[bool] = autotune_remote_cache_default()

# Enable bundled autotune cache.
#
# Enables/disables the bundled autotune cache regardless of the state of
# autotune_remote_cache. However it does depend on the local cache for local
# state management - as a result if the local cache is disabled this will also
# disable the bundled autotune cache.
#
# False: Disables the cache
# True: Enables the cache (requires autotune_local_cache)
# None: Not set -- Off for OSS, JustKnobs based for internal
bundled_autotune_remote_cache: Optional[bool] = bundled_autotune_remote_cache_default()

# See torch.compiler.config.force_disable_caches
force_disable_caches: bool = Config(alias="torch.compiler.config.force_disable_caches")

# Unsafe way to skip dynamic shape guards to get faster cache load
unsafe_skip_cache_dynamic_shape_guards: bool = False

# Unsafe way to mark non torch functions as safe to cache
# dictionary is from function name -> cache key
# Any function name in the dictionary will be allowed to be cacheable
# by AOTAutogradCache and FxGraphCache.
# changing the cache key value will change the resulting
# FXGraphCache key.
# Example usage:
# torch._inductor.config.unsafe_marked_cacheable_functions = {
# 'torch.ops.my_function' : torch.__version__
# }
# The above example causes the custom op torch.ops.my_function to be cacheable,
# and for cache keys to be keyed by the current torch version
unsafe_marked_cacheable_functions: dict[str, str] = {}

# sleep in inductor for testing
sleep_sec_TESTING_ONLY: Optional[int] = None

# The default layout constraint for user-defined triton kernels.
# See "The default layout constraint for custom operators" for options.
triton_kernel_default_layout_constraint: Literal[
    "needs_fixed_stride_order", "flexible_layout"
] = "needs_fixed_stride_order"

# use cpp wrapper instead of python wrapper
# incompatible with disable_cpp_codegen
cpp_wrapper: bool = os.environ.get("TORCHINDUCTOR_CPP_WRAPPER", "0") == "1"

# controls whether to compile entry and kernel separately for cpp_wrapper mode.
# turn on this option to compile entry and kernel separately and minimize compile time of the entry part.
# see https://github.com/pytorch/pytorch/pull/148773
# Note: compiling entry and kernel separately may have a non-negligible impact on the performance.
# see https://github.com/pytorch/pytorch/issues/156037
cpp_wrapper_build_separate: bool = (
    os.environ.get("TORCHINDUCTOR_CPP_WRAPPER_BUILD_SEPARATE", "0") == "1"
)

fx_wrapper: bool = os.environ.get("TORCHINDUCTOR_FX_WRAPPER", "0") == "1"

# Controls automatic precompiling of common include files for codecache.CppCodeCache
# (i.e. for cpp_wrapper mode and for cpp kernels on CPU).  AOTI header precompiling is
# controlled by a separate flag.
cpp_cache_precompile_headers: bool = not is_fbcode()

online_softmax = os.environ.get("TORCHINDUCTOR_ONLINE_SOFTMAX", "1") == "1"

# dead code elimination
dce = False

# assume weight tensors are fixed size
static_weight_shapes = True

# put correctness assertions in generated code
size_asserts = os.environ.get("TORCHINDUCTOR_SIZE_ASSERTS", "1") == "1"
nan_asserts = os.environ.get("TORCHINDUCTOR_NAN_ASSERTS") == "1"
runtime_triton_nan_asserts = (
    os.environ.get("TORCHINDUCTOR_RUNTIME_TRITON_NAN_ASSERTS") == "1"
)
scalar_asserts = os.environ.get("TORCHINDUCTOR_SCALAR_ASSERTS", "1") == "1"

# Disable by default in fbcode
alignment_asserts = (
    os.environ.get("TORCHINDUCTOR_ALIGNMENT_ASSERTS", "0" if is_fbcode() else "1")
    == "1"
)

# enable loop reordering based on input orders
pick_loop_orders = True

# reuse a kernel input as the output
inplace_buffers = True

# reuse a buffer for an unrelated purpose
allow_buffer_reuse = True

# Enable pooled allocations for non-output tensors
memory_planning = os.environ.get("TORCHINDUCTOR_MEMORY_PLANNING", "0") == "1"

# Enable to allow using ftz variant of exponenet instruction in triton codegen.
use_fast_math = os.environ.get("TORCHINDUCTOR_USE_FAST_MATH") == "1"

# How to organize memory under memory_planning=True:
# - "none": do not try to pool storage, just reuse
# - "intermediates": all non-outputs share storage, outputs each get unique storage
# - "outputs": two pools, one for intermediates (freed on return) and one for outputs
# - "combined": a single pool for both intermediates and outputs
memory_pool: Literal["none", "intermediates", "outputs", "combined"] = os.environ.get(
    "TORCHINDUCTOR_MEMORY_POOL", "intermediates"
)  # type: ignore[assignment]

# codegen benchmark harness
benchmark_harness = True

# fuse pointwise into templates epilogues
epilogue_fusion = True

# fuse pointwise into template prologues
prologue_fusion = prologue_fusion_enabled()

# do epilogue fusions before other fusions
epilogue_fusion_first = False

# enable pattern match+replace optimizations
pattern_matcher = True

# set to True to enable the back-to-back GEMM pass
b2b_gemm_pass = False

# register custom graph optimization pass hook. so far, pre/post passes are
# only applied before/after pattern_matcher in post_grad_passes.
#
# Implement CustomGraphPass to allow Inductor to graph compiled artifacts
# to which your custom passes have been applied:
post_grad_custom_pre_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None
post_grad_custom_post_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None

# Allow users to pass in custom partition function
custom_partitioner_fn: torch._inductor.custom_graph_pass.CustomPartitionerFnType = None

# Registers a custom joint graph pass.
joint_custom_pre_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None
joint_custom_post_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None

# Registers a custom pregrad pass. Note that the pre-grad IR is 1.
# non-functional, 2. non-normalized, and 3. prone to change. Ideally we should
# use post-grad passes.
pre_grad_custom_pass: Optional[Callable[[torch.fx.graph.Graph], None]] = None

# Registers a custom pass to be run right before fusion in Inductor scheduler.
# WARNING: Inductor scheduler IR is at prototype stage and subject to change,
# hence custom IR passes built on top of it might break in the future.
_pre_fusion_custom_pass: Optional[
    Callable[
        [list["torch._inductor.scheduler.BaseSchedulerNode"]],
        list["torch._inductor.scheduler.BaseSchedulerNode"],
    ]
] = None

# Registers a custom pass to be run right after fusion in Inductor scheduler.
# WARNING: Inductor scheduler IR is at prototype stage and subject to change,
# hence custom IR passes built on top of it might break in the future.
_post_fusion_custom_pass: Optional[
    Callable[
        [list["torch._inductor.scheduler.BaseSchedulerNode"]],
        list["torch._inductor.scheduler.BaseSchedulerNode"],
    ]
] = None

# Deprecated
split_cat_fx_passes = True

# Optimize conv-batchnorm if batchnorm is in eval mode. Slightly reduces numerical stability.
efficient_conv_bn_eval_fx_passes = False

# Enable predispatch aten IR for export
is_predispatch = False

# Deprecated
group_fusion = False

# Deprecated
batch_fusion = True

# Pre grad fusion and options in order, set to empty dict to disable fusion.
# Call `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions()` to see available fusions.
# batch fusion options:
# batch_linear
# batch_linear_lhs
# batch_layernorm
# batch_tanh
# batch_relu
# batch_sigmoid

# split cat fusion options:
# normalization_pass
# remove_split_with_size_one_pass
# merge_getitem_cat_pass
# merge_stack_tahn_unbind
# merge_splits_pass
# mutate_cat_pass
# split_cat_pass
pre_grad_fusion_options: dict[str, dict[str, Any]] = {}

# Post grad fusion and options, set to empty dict to disable fusion.
# Call `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions(False)` to see available fusions.
post_grad_fusion_options: dict[str, dict[str, Any]] = {}

# enable reordering pass for improving memory locality
reorder_for_locality = True

# Scale down Rn_BLOCK for better occupancy
dynamic_scale_rblock = os.environ.get("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", "1") == "1"

# this forces fusion for int_mm with mul. Needed when you want to avoid realizing the int32
# but the mul gets fused with other pointwise ops instead.
force_fuse_int_mm_with_mul = False

# DEPRECATED. This setting is ignored.
use_mixed_mm = True

# enable runtime numeric check for pre/post grad fx passes
# floating point provides limited accuracy (about 7 decimal digits for single precision
# floating point numbers,about 16 decimal digits for double precision floating point numbers)
# according to PyTorch documentation.
# https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations
fx_passes_numeric_check: dict[str, Any] = {
    "pre_grad": False,
    "precision": 1e-4,
    "num_iterations": 1,
    "requires_optimizer": True,
}

# DEPRECATED. This setting is ignored.
mixed_mm_choice: Literal["default", "triton", "aten", "heuristic"] = "heuristic"

# enable reordering pass for increasing overlap between compute and communication
reorder_for_compute_comm_overlap = False

# passes (in execution order) for increasing overlap between compute and communication
# for built-in passes, use string name; for user-defined passes, pass in the function handle
# WARNING: Inductor scheduler IR is at prototype stage and subject to change,
# hence custom IR passes built on top of it might break in the future.
#
# See aten_distributed_optimizations, it is recommended way for distributed optimizations.
#
# Recommended configuration for reorder_for_compute_comm_overlap_passes:
# [
#     "reorder_communication_preserving_peak_memory",
#     "sink_waits_iterative",
#     "reorder_communication_preserving_peak_memory",
# ]
reorder_for_compute_comm_overlap_passes: list[
    Union[
        str,
        Callable[
            [list["torch._inductor.scheduler.BaseSchedulerNode"]],
            list["torch._inductor.scheduler.BaseSchedulerNode"],
        ],
    ]
] = []

# Maximum number of positions to advance a given collective, unlimited by default
reorder_prefetch_limit: Optional[int] = None

# enable operator reordering for peak memory optimization
reorder_for_peak_memory = True
reorder_for_peak_memory_debug = False

# In some cases, when all the nodes that can be scheduled are quite large,
# it is beneficial to switch the scheduling strategy. So instead of using
# size as the criterion, we choose a node that can unlock more nodes to
# become schedulable by analyzing their successor nodes. The default value
# is zero, which turns off this optimization.
size_threshold_for_succ_based_strategy: int = 0


bucket_all_gathers_fx: Literal["none", "all", "only_fsdp"] = "none"
# By default torch._inductor.fx_passes.bucketing.bucket_size_determinator is used
bucket_all_gathers_fx_bucket_size_determinator: Optional[Callable[[int], int]] = None

bucket_reduce_scatters_fx: Literal["none", "all"] = "none"
# By default torch._inductor.fx_passes.bucketing.bucket_size_determinator is used
bucket_reduce_scatters_fx_bucket_size_determinator: Optional[Callable[[int], int]] = (
    None
)

# runtime estimation function for ops
# for built-in estimation function, pass in "default"; for user-defined estimation function, pass in the function handle
estimate_op_runtime = "default"

runtime_estimations_mms_benchmark: bool = False

# unit: GB/s, uni-directional P2P bandwidth per card
# default value is NVLink
intra_node_bw = 300

# unit: GB/s, uni-directional P2P bandwidth per node
# default value is InfiniBand
inter_node_bw = 25

# use Inductor's experimental benchmarker (runtime/benchmarking.py)
# to benchmark kernels during autotuning, otherwise fall back to
# Triton's `do_bench`. the experimental benchmarker may produce
# results that are not consistent with `do_bench`'s results
use_experimental_benchmarker: bool = Config(
    default=True,
    env_name_force="TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER",
    justknob="pytorch/inductor:use_experimental_benchmarker",
)

# Enable distributed autotuning. When this is enabled we will distribute the
# autotuning across distributed ranks in the same program group - so instead of
# each rank autotuning every kernel they only autotune 1/world size kernels and
# then share the results.
distributed_max_autotune_gemm = (
    os.environ.get("TORCHINDUCTOR_DISTRIBUTED_MAX_AUTOTUNE_GEMM") == "1"
)

# enable slow autotuning passes to select algorithms
max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"

# enable slow autotuning passes to select pointwise/reductions algorithms
max_autotune_pointwise = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE") == "1"

# enable slow autotuning passes to select gemm algorithms
max_autotune_gemm = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM") == "1"

# Modifies the number of autotuning choices displayed, set to None for all
autotune_num_choices_displayed: Optional[int] = 10

# Report the autotune choices and their benchmark results. Default is True.
max_autotune_report_choices_stats = (
    os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS", "1") == "1"
)

# Prune configs that require more shared memory than the hardware limit
max_autotune_prune_choices_based_on_shared_mem = (
    os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM", "1")
    == "1"
)

# Disable triton from trying to initialize and detect devices on the host
triton_disable_device_detection = (
    os.environ.get("TORCHINDUCTOR_TRITON_DISABLE_DEVICE_DETECTION", "0") == "1"
)

# enable inductor graph partition to allow multiple inductor graphs for the same dynamo graph
graph_partition: bool = (
    os.environ.get("TORCHINDUCTOR_GRAPH_PARTITION", "1" if not is_fbcode() else "0")
    == "1"
)

# register ops upon which inductor should partition the graph. name format should be
# "namespace::kernel_name" (e.g., aten::mm) for op overload packet, or
# "namespace::kernel_name.overload" (e.g., aten::mm.default).
custom_should_partition_ops: list[str] = []

# whether template autotuning should allow flexible layouts if possible (e.g. only extern choices)
max_autotune_allow_flexible_layouts: bool = False

# force cublas and triton to use the same precision; cublas supports TF32 for matmul operations
# when m, n, k are multiples of 16, 16, 8, whereas triton supports TF32 for matmul operations
# for any combinations of m, n, k, regardless of their alignment. setting this flag will ensure
# that triton does not use TF32 wherever cublas would not use TF32
# DEPRECATED. cuBLAS no longer has the above alignment requirements. will remove in the future.
force_same_precision: bool = Config(
    justknob="pytorch/compiler:force_same_precision",
    env_name_force="TORCHINDUCTOR_FORCE_SAME_PRECISION",
    default=False,
)

# Size hints for multi-kernel dispatch.
# A reasonable default value of this config would be [64, 256, 4096]
# TODO: @bobrenjc93 to roll this out to a few internal models to ensure this works
# as expected before turning it on for everyone.
multi_kernel_hints: list[int] = []

# Specify candidate backends for gemm autotune.
# Possible choices are combinations of: ATen, Triton, CUTLASS, CK, CKTILE, CPP.
# ATen: default Pytorch ATen kernels.
# Triton: Triton templates defined in torch inductor (AMD and NVidia GPUs).
# CUTLASS: Cutlass templates and kernels (NVidia GPUs only).
# CK: Composable Kernel templates and kernels (AMD Instinct GPUs only).
# CKTILE: Composable Kernel templates and kernels, new API (AMD Instinct GPUs only).
# CPP: CPP templates and kernels for CPU.
max_autotune_gemm_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON,CPP"
).upper()


# As above, specify candidate backends for conv autotune.
# NB: in some cases for 1x1 convs we emit as matmul,
# which will use the backends of `max_autotune_gemm_backends`
max_autotune_conv_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS", "ATEN,TRITON"
).upper()


# Specify the size of the search space for GEMM autotuning.
# DEFAULT     - balance between compile time overhead and performance
# EXHAUSTIVE  - maximize performance
max_autotune_gemm_search_space: Literal["DEFAULT", "EXHAUSTIVE"] = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "DEFAULT"
).upper()  # type: ignore[assignment]

# Specify the size of the search space for flex attention autotuning.
# DEFAULT     - balance between compile time overhead and performance
# EXHAUSTIVE  - maximize performance
max_autotune_flex_search_space: Literal["DEFAULT", "EXHAUSTIVE"] = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_FLEX_SEARCH_SPACE", "DEFAULT"
).upper()  # type: ignore[assignment]


# Fall back to ATen for all ops by default, except those nodes that users explicitly
# annotated with regional inductor compile. Please read torch.fx.passes.regional_inductor
# on to explicitly annotate. This is currently only used by inductor lite mode.
# Different from default inductor mode that fuses all nodes, this config enables an
# opt-in mode that only fuse for user-specified nodes. The motivation is to provide
# guaranteed numeric correctness and give full control to users.
fallback_by_default: bool = False


# This config allows selective decomposition of certain operators in the graph.
# Currently the only use case is to patch the same-name config in functorch, for
# inductor lite mode. See more details in [Note: Selective Decomposition]
selective_decompose: bool = False


# Use dead code elimination
use_dce: bool = True


# Use fx graph passes
use_pre_grad_passes: bool = True
use_joint_graph_passes: bool = True
use_post_grad_passes: bool = True


cutedsl_enable_autotuning: bool = (
    os.environ.get("CUTEDSL_ENABLE_AUTOTUNING", "0") == "1"
)

# DEPRECATED. This setting is ignored.
autotune_fallback_to_aten = False

# the value used as a fallback for the unbacked SymInts
# that can appear in the input shapes (e.g., in autotuning)
unbacked_symint_fallback = 8192

# DEPRECATED. This setting is ignored.
search_autotune_cache = False

save_args = os.environ.get("TORCHINDUCTOR_SAVE_ARGS") == "1"

# We will disable creating subprocess for autotuning if this is False
autotune_in_subproc = os.environ.get("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC") == "1"

# The following three timeouts are applicable if autotune_in_subproc is True:

# Max time that a valid benchmark result may take during autotuning
max_autotune_subproc_result_timeout_seconds = 60.0
# DEPRECATED. This setting is ignored.
max_autotune_subproc_graceful_timeout_seconds = 0.0
# DEPRECATED. This setting is ignored.
max_autotune_subproc_terminate_timeout_seconds = 0.0

# If autotuning in subprocess, whether to use multiple devices
autotune_multi_device = os.environ.get("TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE") == "1"

coordinate_descent_tuning = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING") == "1"
)
coordinate_descent_check_all_directions = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS") == "1"
)
coordinate_descent_search_radius = int(
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS", "1")
)

# AutoHeuristic is a framework that allows one to collect data from autotuning, use the data to learn a heuristic, and
# generate the learned heuristic to code which is shipped with the compiler
# Specify a list of comma separated optimizations to collect data for
autoheuristic_collect = os.environ.get("TORCHINDUCTOR_AUTOHEURISTIC_COLLECT", "")
# Specify a list of comma separated optimizations to use learned heuristics for
autoheuristic_use = os.environ.get("TORCHINDUCTOR_AUTOHEURISTIC_USE", "mixed_mm")

# If set to 1, will run a JIT post compile hook if one is set.
run_jit_post_compile_hook = (
    os.environ.get("TORCHINDUCTOR_RUN_JIT_POST_COMPILE_HOOK", "0") == "1"
)


def run_autoheuristic(name: str) -> bool:
    return collect_autoheuristic(name) or use_autoheuristic(name)


def collect_autoheuristic(name: str) -> bool:
    return name in torch._inductor.config.autoheuristic_collect.split(",")


def use_autoheuristic(name: str) -> bool:
    return name in torch._inductor.config.autoheuristic_use.split(",")


# If set to "DEFAULT", this will use the default log path specified in autoheuristic.py.
# If set to another path, autoheuristic will instead log results to the given path.
autoheuristic_log_path = os.environ.get(
    "TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH", "DEFAULT"
)

# Disabled by default on ROCm, opt-in if model utilises NHWC convolutions
layout_opt_default = "1" if not torch.version.hip else "0"
layout_optimization = (
    os.environ.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION", layout_opt_default) == "1"
)

force_layout_optimization = os.environ.get("TORCHINDUCTOR_FORCE_LAYOUT_OPT", "0") == "1"


# Whether to keep the output strides the same as eager after layout optimization.
keep_output_stride = os.environ.get("TORCHINDUCTOR_KEEP_OUTPUT_STRIDE", "1") == "1"

# Enabling this will let compiler print warning messages if a generated triton
# kernel has inputs with mixed layouts.  This is helpful for perf debugging
# since kernel with mixed layout inputs may run much slower then one whose inputs
# have uniform layouts.
warn_mix_layout = os.environ.get("TORCHINDUCTOR_WARN_MIX_LAYOUT") == "1"

# control store vs recompute heuristic
# For fanouts, rematerialization can lead to exponential blowup. So, have
# smaller threshold
realize_reads_threshold = 4
realize_opcount_threshold = 30

# Threshold to prevent excessive accumulation of ops in one buffer during lowering
realize_acc_reads_threshold = 8
realize_acc_reads_size_threshold: Optional[int] = (
    None  # TODO(xuanzh): harden this to make it non optional
)

# fallback to eager for random/dropout, this is slow but useful for debugging
fallback_random = False

# fallback embedding_bag_byte_unpack to eager
fallback_embedding_bag_byte_unpack = False

# automatically create fallbacks when encountering an unhandled op
implicit_fallbacks = True
assume_unaligned_fallback_output = (
    os.environ.get("TORCHINDUCTOR_ASSUME_UNALIGNED_FALLBACK_OUTPUT") == "1"
)

# Custom InductorChoices callable to use (can be a class or functools.partial with kwargs)
inductor_choices_class: Optional[Callable[[], "InductorChoices"]] = None

# fuse even in cases without common reads
aggressive_fusion = False

# For each fused kernel in the wrapper, comment with the nodes that get fused.
# Useful for debugging fusion.
debug_fusion: bool = os.environ.get("TORCHINDUCTOR_DEBUG_FUSION") == "1"
benchmark_fusion: bool = os.environ.get("TORCHINDUCTOR_BENCHMARK_FUSION") == "1"
enabled_metric_tables = os.environ.get("TORCHINDUCTOR_ENABLED_METRIC_TABLES", "")
loop_ordering_after_fusion: bool = (
    os.environ.get(
        "TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION", "0" if is_fbcode() else "1"
    )
    == "1"
)


# When trying to fuse two nodes, one with:
# a[contiguous_writes] = fn(...)
# and another node:
# b[contiguous_writes] = a[discontiguous_reads]
# If b is unary, and we can figure out an inverse formula for
# discontiguous writes, invert b as :
# b[inverse(discontiguous_writes)] = a[contiguous_reads]
# so that the nodes can fuse. for more details: https://gist.github.com/eellison/6f9f4a7ec10a860150b15b719f9285a9
loop_index_inversion_in_fusion: bool = True

# If fusing two nodes only save less then score_fusion_memory_threshold memory,
# we should not bother fusing the nodes.
#
# This is especially helpful to resolve https://github.com/pytorch/pytorch/issues/133242
# Previously we fuse two nodes because of common read of a scalar tensor.
# If we skip it, the loop ordering after fusion mechanism kicks in and can
# brings more savings.
#
# For the cases loop ordering after fusion does not help, we don't lose much.
score_fusion_memory_threshold = 10

# For Triton Templates, select fastest of best template + epilogue vs best template + separate epilogue kernel
benchmark_epilogue_fusion = (
    os.environ.get("TORCHINDUCTOR_BENCHMARK_EPILOGUE_FUSION", "1") == "1"
)

# Take how many of the top triton kernels to benchmark epilogue
max_epilogue_benchmarked_choices = 1

# how many nodes to allow into a single fusion
max_fusion_size = 64

# how many nodes to attempt pairwise fusion with in a buffer group
max_fusion_buffer_group_pairwise_attempts = 64

# maximum number of unique input/output buffers allowed in fused kernels.
# The check is disabled if set to None.
max_fusion_unique_io_buffers: Optional[int] = None

# max number of inputs to generate cat as a pointwise op with masked loads
max_pointwise_cat_inputs = 8

# force concat to be generated as a pointwise op with masked loads
force_pointwise_cat = False

# replace small reductions with pointwise, disable with `= 1`
unroll_reductions_threshold = 8

# Add extra comments to output code (causes compile cache misses)
comment_origin = False

# Convert 1x1 convs into matmuls
conv_1x1_as_mm = False

# For reductions with a small output size (usually 1, e.g. x.sum()) there is not enough
# parallelism to saturate the GPU.  We have two ways of handling this, either `split_reductions`
# or `triton.cooperative_reductions` which are mutually exclusive.
#   split_reductions: uses multiple kernels to gain more parallelism
#   triton.cooperative_reductions: uses cross thread-block synchronization to gain more parallelism
# enabling both of these will implicitly disable split_reductions
split_reductions = os.getenv("TORCHINDUCTOR_SPLIT_REDUCTIONS", "1") == "1"

# A deterministic mode that skips any on device benchmarking in Inductor
# if we know they affect numerics.  WARNING: Expect perf hit in this mode.
deterministic = os.getenv("TORCHINDUCTOR_DETERMINISTIC") == "1"

# When we do split reduction, this number control the minimum value for
# num_split. Too small num_split make the split reduction less efficient.
# It's a much bigger problem when we compile a dynamic shape kernel with
# non-representative inputs.
min_num_split = int(os.environ.get("TORCHINDUCTOR_MIN_NUM_SPLIT", 0))

benchmark_kernel = os.environ.get("TORCHINDUCTOR_BENCHMARK_KERNEL", "0") == "1"

# Enable constant and index_expr folding
constant_and_index_propagation = True

# we always add constants into graph.constants without
# performing any constant-inlining optimization
always_keep_tensor_constants = False

# assert that indirect indexing does not read / write out of bounds
assert_indirect_indexing = True

# compute CSE bounds on variables that do not appear in the FX graph
compute_all_bounds = False

# enable the combo kernel that combines data-independent kernels (additional
# to foreach kernels) into a single one (Experimental)
combo_kernels = False
# benchmark combo kernels and only allow ones with perf gains
benchmark_combo_kernel = False
# combo_kernel autotuning options: 0 - disable, 1 - enable except for foreach,
# 2 - enable for all
combo_kernels_autotune = 1
# Enable masking for combining kernels of mixed sizes: 0 - disable, 1 - enable
# for all except for foreach, 2 - enable for all
combo_kernel_allow_mixed_sizes = 1
# Enable dynamic shapes for foreach kernels
combo_kernel_foreach_dynamic_shapes = True
# Maximum number of arguments (read/write buffers) allowed in a combo kernel
combo_kernel_max_num_args = 250

# constant folding on the joint graph
joint_graph_constant_folding = True

# Enable indirect_indexing asserts for decompositions and lowerings
debug_index_asserts = False

# Mode to emulate PyTorch eager numerics when doing lower precision compute
# (fp16, bf16).  PyTorch eager computes bf16/fp16 by upcasting inputs to fp32
# and downcasting after.  When two low precision operators are fused together,
# Inductor will elide the downcast-upcast pairs (effectively a precision
# truncation) that would occur between these two operators.  Typically,
# Inductor's behavior should be closer to fp64 ref numerics.  However, with
# this knob you can ensure the downcast-upcast are preserved so that you can
# emulate the eager numerics.
emulate_precision_casts = (
    os.environ.get("TORCHINDUCTOR_EMULATE_PRECISION_CASTS", "0") == "1"
)

# x / y in Triton is lowered to div.full which is approx
# PyTorch eager uses the equivalent of Triton's div_rn, which can
# come at a performance penalty
emulate_divison_rounding = (
    os.environ.get("TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING", "0") == "1"
)

# warnings intended for PyTorch developers, disable for point releases
is_nightly_or_source = "dev" in torch.__version__ or "git" in torch.__version__
developer_warnings = is_fbcode() or is_nightly_or_source

# This pattern matches a special usage of scatter
# 1. It's applied to a constant tensor
# 2. The index tensor has size 1 in the scatter dimension
# Such pattern generates a sparse matrix when the const tensor is all-zero.
# We can lower this pattern to a pointwise kernel for more fusion opportunities
# and saving memory footprint.
optimize_scatter_upon_const_tensor = (
    os.environ.get("TORCHINDUCTOR_OPTIMIZE_SCATTER_UPON_CONST_TENSOR", "1") == "1"
)

# options in caffe2/torch/_inductor/fx_passes/pre_grad.py
add_pre_grad_passes: Optional[str] = None
remove_pre_grad_passes: Optional[str] = None


# The multiprocessing start method to use for inductor workers in the codecache.
def decide_worker_start_method() -> str:
    if "TORCHINDUCTOR_WORKER_START" in os.environ:
        start_method = os.environ["TORCHINDUCTOR_WORKER_START"]
    else:
        start_method = "subprocess"
    assert start_method in (
        "subprocess",
        "fork",
        "spawn",
    ), f"Invalid start method: {start_method}"
    return start_method


worker_start_method: str = decide_worker_start_method()

# Threshold to decide if a kernel has small memory access in bytes
# Default value is 16 MB which is arbitrarily selected.
small_memory_access_threshold: int = 16777216

# Whether to log from subprocess workers that are launched.
worker_suppress_logging: bool = Config(
    justknob="pytorch/compiler:worker_suppress_logging",
    env_name_force="TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING",
    default=True,
)

# Log per-operation runtime estimates for TLParse analysis.
log_tlparse: bool = Config(
    env_name_force="LOG_TLPARSE",
    default=False,
)

# Flags to turn on all_reduce fusion. These 2 flags should be automatically turned
# on by DDP and should not be set by the users.
_fuse_ddp_communication = False
_fuse_ddp_bucket_size = 25

# Flag to control which fusion passes to apply. Functions in the list will
# be applied in order. There are two different different fusion passes
# --"fuse_ddp_with_concat_op" and "fuse_ddp_with_coalesced_op". The default
# one is "fuse_ddp_with_concat_op". Users can also change this to a customized
# fusion function.
#
# The fusion currently does not support multiple DDP with different PG or
# data type. This feature will be added in the future PRs.
#
# "schedule_comm_wait" is used to delay the wait ops to maximize comm/comp
# overlapping. At this moment, this pass performs better than
# reorder_for_compute_comm_overlap_passes but we will add the logic of
# "schedule_comm_wait" in the future and remove the one here.
_fuse_ddp_communication_passes: list[Union[Callable[..., None], str]] = [
    "fuse_ddp_with_concat_op",
    "schedule_comm_wait",
]

_micro_pipeline_tp: bool = False


class _collective:
    auto_select: bool = False
    one_shot_all_reduce_threshold_bytes: int = 128 * 1024


class aten_distributed_optimizations:
    """Configuration for distributed optimization passes on ATen FX graphs."""

    # Enable overlap scheduling pass
    enable_overlap_scheduling: bool = False

    # Enable overlap-preserving collective bucketing
    collective_bucketing: Optional[bool] = None

    # Insert ordering dependencies to preserve overlap relationships. This should only be used if
    # compiling with inductor, or for subsequent passes before removing the ops prior to execution
    insert_overlap_deps: Optional[bool] = None

    # Maximum compute node prefetch distance for overlap scheduling
    max_compute_pre_fetch: Optional[int] = None

    # Custom runtime estimation function for ops
    # For user-defined estimation function, pass in the function handle
    # None means use default estimations
    # TODO - need estimated and profile based version
    custom_runtime_estimation: Optional[Callable[[torch.fx.Node], Optional[float]]] = (
        None
    )

    # Method for estimating collective runtime
    # "analytical": Use bandwidth formulas (default)
    # "benchmark": Use CUDA events with power-of-2 rounding and interpolation
    collective_estimator: Literal["analytical", "benchmark"] = "analytical"


def parallel_compile_enabled_internally() -> bool:
    """
    TODO: Remove when parallel compiled is fully enabled internally. For rollout, use a
    knob to enable / disable. The justknob should not be performed at import, however.
    So for fbcode, we assign compile_threads to 'None' below and initialize lazily in
    async_compile.py.
    """
    ENABLE_PARALLEL_COMPILE_VERSION = 1

    jk_name = "pytorch/inductor:enable_parallel_compile_version"
    version = torch._utils_internal.justknobs_getval_int(jk_name)
    return ENABLE_PARALLEL_COMPILE_VERSION >= version


def decide_compile_threads() -> int:
    """
    Here are the precedence to decide compile_threads
    1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.  One may want to disable async compiling by
       setting this to 1 to make pdb happy.
    2. Set to 1 if it's win32 platform
    3. decide by the number of CPU cores
    """
    import logging

    # Defined locally so install_config_module doesn't try to parse
    # as a config option.
    log = logging.getLogger(__name__)

    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        compile_threads = int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
        log.info("compile_threads set to %d via env", compile_threads)
    elif sys.platform == "win32":
        compile_threads = 1
        log.info("compile_threads set to 1 for win32")
    elif is_fbcode() and not parallel_compile_enabled_internally():
        compile_threads = 1
        log.info("compile_threads set to 1 in fbcode")
    else:
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        assert cpu_count
        compile_threads = min(32, cpu_count)
        log.info("compile_threads set to %d", compile_threads)

    return compile_threads


# TODO: Set directly after internal rollout.
compile_threads: Optional[int] = None if is_fbcode() else decide_compile_threads()

# Whether to quiesce the Triton-compile subprocess pool at the end of each compilation.
quiesce_async_compile_pool: bool = Config(
    justknob="pytorch/inductor:quiesce_async_compile_pool",
    env_name_force="TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL",
    default=False,
)

# Time in seconds to wait before quiescing
quiesce_async_compile_time: int = Config(
    default=60,
)

# Whether or not to enable statically launching CUDA kernels
# compiled by triton (instead of using triton's own launcher)
use_static_cuda_launcher: bool = static_cuda_launcher_default()

# Attempt to statically launch user defined triton kernels
# Requires use_static_cuda_launcher
static_launch_user_defined_triton_kernels: bool = Config(
    justknob="pytorch/inductor:static_launch_user_defined_triton_kernels",
    env_name_force="TORCHINDUCTOR_STATIC_LAUNCH_USER_DEFINED_TRITON_KERNELS",
    default=False,
)

# Raise error if we bypass the launcher
strict_static_cuda_launcher: bool = (
    os.environ.get("TORCHINDUCTOR_STRICT_STATIC_CUDA_LAUNCHER", "0") == "1"
)

# gemm autotuning global cache dir
global_cache_dir: Optional[str]
if is_fbcode():
    try:
        from libfb.py import parutil

        if __package__:
            global_cache_dir = parutil.get_dir_path(
                os.path.join(__package__.replace(".", os.sep), "fb/cache")
            )
        else:
            global_cache_dir = parutil.get_dir_path("fb/cache")
    except (ValueError, ImportError):
        global_cache_dir = None

else:
    global_cache_dir = None

# If kernel is fused, the name is generated from the origin node op names
# for larger kernels limit this
kernel_name_max_ops = 10

# Pad input tensors of matmul/bmm/addmm to leverage Tensor Cores in NVIDIA GPUs
shape_padding = os.environ.get("TORCHINDUCTOR_SHAPE_PADDING", "1") == "1"

# Control if we will do padding for pointwise/reductions
comprehensive_padding = (
    os.environ.get("TORCHINDUCTOR_COMPREHENSIVE_PADDING", "1") == "1"
)
pad_channels_last = False

# Control if we will do padding on dynamic shapes
pad_dynamic_shapes = False

# Disable comprehensive padding on the CPU
disable_padding_cpu = True

# Control if we will expand the dimension of pointwise nodes to fuse
expand_dimension_for_pointwise_nodes = False

# The width of comprehensive padding, in bytes.
# CUDA max memory transaction size is 128 bytes for a warp.
padding_alignment_bytes = 128

# Threshold on the minimum stride that will be padded.
#
# Don't align a too small stride since that causes too much memory increase.
# Pad too small stride may also cause perf loss. We may result in many tiny data blocks
# with gaps in between. That causes less coalesced GPU memory access!
#
# Initially we pick 320 as the threshold since for alignment=16,
# that results in at most 5% memory cost.
#
# But later on we raise the threshold to 1024 to avoid interfere with persistent reduction.
# Let's say an inner reduction has a row size 513. Inductor will generate
# persistent reduction code.
# If we do padding, the strides are not contiguous any more. Inductor
# uses a much smaller threshold for persistent reduction in this case and
# generates potentially worse non-persistent reduction code.
#
# This change turns HF AllenaiLongformerBase amp training from a loss of 1.09x to a win of 1.05x.
# (baseline: 71.09ms, padding w/o this change: 77.38ms, padding with this change: 67.77ms)
padding_stride_threshold = 1024

# Enable padding outputs, even if they would not be padded in eager mode.
# By default, we use the same strides as eager mode.
pad_outputs = False

# Whether to treat output of the backward graph as user visible.
# For user visible outputs, inductor will make sure the stride matches with eager.
bw_outputs_user_visible = True

# Whether to always use shape padding if it is enabled and possible
force_shape_pad: bool = False

# Fx-based linear/matmul/bmm + permute/transpose vertical fusion
permute_fusion = os.environ.get("TORCHINDUCTOR_PERMUTE_FUSION", "0") == "1"

# Mark the wrapper call in PyTorch profiler
profiler_mark_wrapper_call = False

# Generate hook calls to torch._inductor.hooks.run_intermediate_hooks for
# every intermediate for which we can correlate it with an intermediate
# from the original FX graph
generate_intermediate_hooks = False

# Populate traceback field on IRNode; good for debugging why origin_node is
# not populated, or finding out where an IRNode was constructed
debug_ir_traceback = False

# used for debugging to make sure config is properly set
_raise_error_for_testing = False

_profile_var = os.environ.get("TORCHINDUCTOR_PROFILE", "")
profile_bandwidth = _profile_var != ""
profile_bandwidth_regex = "" if _profile_var == "1" else _profile_var
# Specify a file where we print out the profiling results.
# None means we do not dump results to a file.
profile_bandwidth_output: Optional[str] = os.environ.get(
    "TORCHINDUCTOR_PROFILE_OUTPUT", None
)
# Switch to do_bench_using_profiling to exclude the CPU overheads
profile_bandwidth_with_do_bench_using_profiling = (
    os.environ.get("TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING") == "1"
)


# TODO: remove later
# incompatible with cpp_wrapper
disable_cpp_codegen = False


# Freezing will attempt to inline weights as constants in optimization
# and run constant folding and other optimizations on them. After freezing, weights
# can no longer be updated.
freezing: bool = os.environ.get("TORCHINDUCTOR_FREEZING", "0") == "1"

# Make freezing invalidate the eager Parameters of nn modules, to avoid memory overhead
# of potentially keeping multiple copies of weights.
freezing_discard_parameters: bool = False

# decompose some memory bound matmul/bmm to mul
decompose_mem_bound_mm: bool = False

# assume_aligned_inputs means that we assume that inputs will be aligned; we generate
# code using this assumption, and clone tensors before use if they aren't aligned.
# In the common case, most inputs will be aligned.
assume_aligned_inputs: bool = False

# assume_32bit_indexing means that we assume 32-bit indexing is always safe; we always
# use 32-bit indices regardless of tensor sizes. If assume_32bit_indexing contradicts
# with example inputs we throw. This is useful when all dynamic shapes are unbacked and
# you know you only operate with 32-bit sizes.
assume_32bit_indexing: bool = False

# For the user-written Triton kernels compiled with the model, ignore the unsupported
# arguments passed to the @triton.autotune in the user's code; this is unsafe, as
# ignoring the unsupported args may lead to unexpected autotuning behavior: don't
# set unless you know what you're doing.
unsafe_ignore_unsupported_triton_autotune_args: bool = False

# When True, we will check in scheduler.py _codegen that there are no "loops"
# in the call stack; that is to say, the same frame multiple times.  This
# ensures that a cProfile trace to this frame will be a straight line without
# any cycles. Incompatible with cpp_wrapper.
check_stack_no_cycles_TESTING_ONLY: bool = False

# When True, complex_memory_overlap always reports True
always_complex_memory_overlap_TESTING_ONLY: bool = False

# enable linear binary folding
enable_linear_binary_folding = (
    os.environ.get("TORCHINDUCTOR_ENABLE_LINEAR_BINARY_FOLDING", "0") == "1"
)


# Adds NVTX annotations around training phases
annotate_training: bool = os.environ.get("TORCHINDUCTOR_ANNOTATE_TRAINING", "0") == "1"

# Enable caching codegen of triton templates.
enable_caching_generated_triton_templates: bool = True

# Lookup table for overriding autotune configs based on hash of Triton source code
autotune_lookup_table: dict[str, dict[str, Any]] = {}

file_lock_timeout: int = int(os.environ.get("TORCHINDUCTOR_FILE_LOCK_TIMEOUT", "600"))

enable_autograd_for_aot: bool = False


def get_worker_log_path() -> Optional[str]:
    log_loc = None
    if is_fbcode():
        mast_job_name = os.environ.get("MAST_HPC_JOB_NAME", None)
        global_rank = os.environ.get("ROLE_RANK", "0")

        if mast_job_name is not None:
            log_loc = f"/logs/dedicated_log_torch_compile_worker_rank{global_rank}"

    return log_loc


torchinductor_worker_logpath: str = Config(
    env_name_force="TORCHINDUCTOR_WORKER_LOGPATH",
    default="",
)


# config specific to codegen/cpp.py
class cpp:
    """
    Settings for cpp backend.
    This class provides a centralized location for managing cpp backend settings.
    """

    # set to torch.get_num_threads()
    threads = -1

    # Do not generate loops when the condition doesn't hold, like:
    # for(long i0=4096; i0<4096; i0+=1)
    no_redundant_loops = (
        os.environ.get("TORCHINDUCTOR_CPP_NO_REDUNDANT_LOOPS", "1") == "1"
    )

    # Assume number of threads is dynamic, don't specialize thread number.
    # Kernels don't recompile on thread number changes with this flag on.
    # For single-threaded workload, turning it on would incur a slight
    # performance degradation.
    dynamic_threads = os.environ.get("TORCHINDUCTOR_CPP_DYNAMIC_THREADS", "0") == "1"

    simdlen: Optional[int] = None
    min_chunk_size = int(os.environ.get("TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE", "512"))

    cxx: tuple[None, str] = (
        None,  # download gcc12 from conda-forge if conda is installed
        os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "g++"),
    )  # type: ignore[assignment]

    # Allow kernel performance profiling via PyTorch profiler
    enable_kernel_profile = (
        os.environ.get("TORCHINDUCTOR_CPP_ENABLE_KERNEL_PROFILE", "0") == "1"
    )

    # enable weight prepacking to get a better performance; may lead to large memory footprint
    weight_prepack = os.environ.get("TORCHINDUCTOR_CPP_WEIGHT_PREPACK", "1") == "1"

    # Inject a bug into our relu implementation; useful for testing our repro
    # extraction and minification functionality.
    # Valid values: "compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY: Optional[str] = None
    inject_log1p_bug_TESTING_ONLY: Optional[str] = None

    # If None, autodetect whether or not AVX512/AVX2 can be used.  Otherwise,
    # force usage as specified, without testing. Default None.
    vec_isa_ok: Optional[bool] = get_tristate_env("TORCHINDUCTOR_VEC_ISA_OK")

    # similar to config.triton.descriptive_names
    descriptive_names: Literal["torch", "original_aten", "inductor_node"] = (
        "original_aten"
    )

    # how many nodes to allow into a single horizontal fusion
    max_horizontal_fusion_size = int(
        os.environ.get("TORCHINDUCTOR_CPP_MAX_HORIZONTAL_FUSION_SIZE", "16")
    )

    # Make scatter_reduce fallback when reduce is sum to avoid performance regr
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `config.py_docs.md_docs.md`
- **Keyword Index**: `config.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
