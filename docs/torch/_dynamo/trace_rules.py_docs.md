# Documentation: `torch/_dynamo/trace_rules.py`

## File Metadata

- **Path**: `torch/_dynamo/trace_rules.py`
- **Size**: 157,150 bytes (153.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Contains **unit tests** using Python testing frameworks.

## Original Source

```python
"""
Tracing rules and policies for TorchDynamo compilation decisions.

This module defines the rules that govern what code TorchDynamo should trace and compile
versus what should be executed eagerly. It contains functions and classes that determine:

- Which modules, functions, and objects should be skipped during tracing
- Which parts of the code should cause graph breaks
- How to handle different Python libraries and third-party packages
- Rules for determining when to inline functions vs calling them eagerly

Key components:
- Skip rules: Functions that return True if an object should be skipped during tracing
- Inlining rules: Policies for when to inline function calls during compilation
- Library-specific handling: Special cases for popular Python packages
- Performance heuristics: Rules that balance compilation overhead vs runtime benefits

These rules are critical for TorchDynamo's ability to automatically determine
compilation boundaries and optimize PyTorch programs effectively.
"""

import abc
import builtins
import copy
import dataclasses
import functools
import importlib
import inspect
import linecache
import operator
import os
import random
import re
import sys
import traceback
import types
import unittest
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast, Optional, Union

import torch
import torch._inductor.test_operators
import torch.distributed
import torch.utils._content_store
from torch._environment import is_fbcode
from torch.utils import _config_module

from . import config
from .resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX
from .utils import (
    getfile,
    hashable,
    is_lru_cache_wrapped_function,
    NP_SUPPORTED_MODULES,
    unwrap_if_wrapper,
)
from .variables import (
    BuiltinVariable,
    FunctionalCallVariable,
    FunctorchHigherOrderVariable,
    LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable,
    NestedUserFunctionVariable,
    PolyfilledFunctionVariable,
    ReparametrizeModuleCallVariable,
    SkipFunctionVariable,
    TorchInGraphFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .variables.base import VariableTracker


np: Optional[types.ModuleType] = None
try:
    import numpy as np
except ModuleNotFoundError:
    pass


"""
A note on skip/inline rules:

Dynamo consults this file to determine whether function should be inlined or skipped.

A skip applies at the frame boundary, meaning dynamo either triggers a graph break
at the beginning of the frame or attempts to trace/inline the whole frame. When skipping
a frame, recursively called frames are still traced by dynamo unless also skipped.

Skipfiles (skipped at the file level instead of function level) still apply on a
frame-by-frame boundary as dynamo traces, but apply to all functions in that file.

@skip is a helper decorator that can be applied to your function to cause it to be
included here.

Dynamo skip/inline rules & priorities are defined as follows:
* Inline is the default behavior and will be used unless explicitly skipped.
* Dynamo has two SKIPLIST: BUILTIN_SKIPLIST and THIRDPARTY_SKIPLIST.
    * BUILTIN_SKIPLIST contains builtin python modules, such as abc, collections, etc.
    * THIRDPARTY_SKIPLIST contains common third party libraries, such as numpy, pandas, etc.
* Functions in these two SKIPLISTs are always skipped, except:
    * They have explicitly defined rule in `manual_torch_name_rule_map`;
    * The corresponding python module has been put into MOD_INLINELIST.
* PyTorch(torch) is in the BUILTIN_SKIPLIST by default, but there are many cases
    where we want inline the functions under torch namespace.
    We should specify inline for the functions in `manual_torch_name_rule_map` or
    put the corresponding python module into MOD_INLINELIST to make dynamo inline them.
* If you call functions under skipped modules/files, Dynamo will wrap these functions
    as SkipFunctionVariable. There are a few functions(e.g, collections.OrderedDict) that
    we have special handling at SkipFunctionVariable.call_function.

Overall: *_INLINELIST has precedence over *_SKIPLIST has precedence over DEFAULT (inline)

To figure out what the behavior is, check the following list in order:
* `manual_torch_name_rule_map` (Inline if YES)
* MOD_INLINELIST (Inline if YES)
* BUILTIN_SKIPLIST & THIRDPARTY_SKIPLIST (Skip if YES)
* MOD_SKIPLIST (Skip if YES)
* Inline by default

In general, if you want to force inline a function or module, please consider adding
the function's python module to MOD_INLINELIST first.
Use the `manual_torch_name_rule_map` only when there are other functions under the same module that
you don't want to inline them.
"""

"""
Map of function objects to their tracing rules (Dynamo variables).
* TorchInGraphFunctionVariable: The functions should be put into the FX graph or can be constant folded. E.g.,
  - torch.add: should be put into the FX graph.
  - torch.is_floating_point: constant folded.
* SkipFunctionVariable: The objects should be skipped from tracing.
* UserFunctionVariable: The functions should be inlined.

For developers: If you add/remove a torch level API, it may trigger failures from
test/dynamo/test_trace_rules.py:test_torch_name_rule_map_updated. To fix the failures:
If you are adding a new torch level API or Dynamo implementation:
* Add the name with the corresponding tracing rule to this map
  if you are adding a new in graph function or Dynamo implementation for an existing function.
* Remove the object name from test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names if it's there.

If you are removing an existing torch level API:
* Remove the entry represented the API from this map or test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names
  depends on where it is.


"""
manual_torch_name_rule_map: dict[
    str,
    Union[
        type[TorchInGraphFunctionVariable],
        type[SkipFunctionVariable],
        type[UserFunctionVariable],
    ],
] = {
    "torch.onnx.is_in_onnx_export": TorchInGraphFunctionVariable,
    "torch.onnx.operators.shape_as_tensor": TorchInGraphFunctionVariable,
    "torch.overrides.is_tensor_like": TorchInGraphFunctionVariable,
    "torch.jit.is_scripting": TorchInGraphFunctionVariable,
    "torch.jit.is_tracing": TorchInGraphFunctionVariable,
    "torch.jit.annotate": TorchInGraphFunctionVariable,
    "torch.distributed.is_available": TorchInGraphFunctionVariable,
    "torch.distributed.is_initialized": TorchInGraphFunctionVariable,
    "torch.distributed.get_rank": TorchInGraphFunctionVariable,
    "torch.distributed.get_world_size": TorchInGraphFunctionVariable,
    "torch.distributed.tensor._api.DTensor#from_local": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._get_group_size_by_name": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._resolve_group_name_by_ranks_and_tag": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._get_group_tag": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d.get_process_group_ranks": TorchInGraphFunctionVariable,
    "torch._utils.is_compiling": TorchInGraphFunctionVariable,
    "torch.fx._symbolic_trace.is_fx_tracing": TorchInGraphFunctionVariable,
    "torch.fx._symbolic_trace.is_fx_symbolic_tracing": TorchInGraphFunctionVariable,
    "torch._dynamo.external_utils.is_compiling": TorchInGraphFunctionVariable,
    "torch._dynamo.utils._disable_side_effect_safety_checks_for_current_subtracer": UserFunctionVariable,
    "torch.compiler.is_compiling": TorchInGraphFunctionVariable,
    "torch.compiler.is_dynamo_compiling": TorchInGraphFunctionVariable,
    "torch.compiler.is_exporting": TorchInGraphFunctionVariable,
    "torch._C._to_dlpack": SkipFunctionVariable,
    "torch.to_dlpack": SkipFunctionVariable,
    "torch._check": TorchInGraphFunctionVariable,
    # We graph break on RNG state setters or getters like
    # `torch.get_rng_state` or `torch.set_rng_state`. These functions
    # are not aten operations and therefore they are completely ignored
    # by the AOT dispatcher. As a result, the AOT graph does not have
    # these setter or getter functions, producing an incorrect graph
    # when it comes to rng states.
    "torch.default_generator#get_state": SkipFunctionVariable,
    "torch._C.Generator#get_state": SkipFunctionVariable,
    "torch.get_rng_state": SkipFunctionVariable,
    "torch.cuda.get_rng_state": SkipFunctionVariable,
    "torch.default_generator#set_state": SkipFunctionVariable,
    "torch._C.Generator#set_state": SkipFunctionVariable,
    "torch.set_rng_state": SkipFunctionVariable,
    "torch.cuda.set_rng_state": SkipFunctionVariable,
    # https://github.com/pytorch/pytorch/issues/107187
    "torch.manual_seed": SkipFunctionVariable,
    # https://github.com/pytorch/pytorch/issues/93501
    "torch.nn.utils.rnn.pack_padded_sequence": SkipFunctionVariable,
    "torch.nn.Parameter": TorchInGraphFunctionVariable,
    "torch.nn.Buffer": TorchInGraphFunctionVariable,
    "torch._nested_tensor_from_mask": SkipFunctionVariable,
    "torch.nested._internal.nested_tensor.nested_from_padded": TorchInGraphFunctionVariable,
    "torch.nested.nested_tensor_from_jagged": UserFunctionVariable,
    "torch.nested.nested_tensor_from_padded": UserFunctionVariable,
    # torch.fx map utils
    "torch.fx.node.map_aggregate": UserFunctionVariable,
    "torch.fx.node.map_arg": UserFunctionVariable,
    "torch.fx.immutable_collections._no_mutation": UserFunctionVariable,
    "torch.fx.immutable_collections._immutable_list_flatten": UserFunctionVariable,
    "torch.fx.immutable_collections._immutable_list_unflatten": UserFunctionVariable,
    "torch.fx.immutable_collections._immutable_dict_flatten": UserFunctionVariable,
    "torch.fx.immutable_collections._immutable_dict_unflatten": UserFunctionVariable,
    # symbol operators implemented in Python
    "torch.sym_not": TorchInGraphFunctionVariable,
    "torch.sym_float": TorchInGraphFunctionVariable,
    "torch.sym_int": TorchInGraphFunctionVariable,
    "torch.sym_max": TorchInGraphFunctionVariable,
    "torch.sym_min": TorchInGraphFunctionVariable,
    "torch.sym_sqrt": TorchInGraphFunctionVariable,
    "torch.sym_ite": TorchInGraphFunctionVariable,
    "torch.sym_sum": TorchInGraphFunctionVariable,
    "torch.sym_fresh_size": UserFunctionVariable,
    "torch.Tensor#_make_wrapper_subclass": SkipFunctionVariable,
    "torch.Tensor#__init__": SkipFunctionVariable,
    "torch.Tensor#split": TorchInGraphFunctionVariable,
    "torch.cuda.set_device": SkipFunctionVariable,
    "torch.cuda.current_device": TorchInGraphFunctionVariable,
    "torch._C.autocast_decrement_nesting": SkipFunctionVariable,
    "torch._C.autocast_increment_nesting": SkipFunctionVariable,
    "torch.autograd.grad": SkipFunctionVariable,
    "torch.autograd.backward": SkipFunctionVariable,
    "torch._C.clear_autocast_cache": SkipFunctionVariable,
    "torch.distributions.constraints.is_dependent": SkipFunctionVariable,
    "torch.jit.isinstance": SkipFunctionVariable,
    "torch._C.set_anomaly_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_cache_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_cpu_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_cpu_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_gpu_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_ipu_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_ipu_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_xla_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_xla_enabled": SkipFunctionVariable,
    "torch.resize_as_": SkipFunctionVariable,
    "torch._functorch.predispatch._add_batch_dim": TorchInGraphFunctionVariable,
    "torch._functorch.predispatch._remove_batch_dim": TorchInGraphFunctionVariable,
    "torch.resize_as_sparse_": SkipFunctionVariable,
    "torch.get_default_device": TorchInGraphFunctionVariable,
    # functorch/vmap
    "torch._functorch.vmap._check_int_or_none": UserFunctionVariable,
    "torch._functorch.vmap._check_out_dims_is_int_or_int_pytree": UserFunctionVariable,
    "torch._functorch.vmap._check_randomness_arg": UserFunctionVariable,
    "torch._functorch.vmap._chunked_vmap": UserFunctionVariable,
    "torch._functorch.vmap._concat_chunked_outputs": UserFunctionVariable,
    "torch._functorch.vmap._create_batched_inputs": UserFunctionVariable,
    "torch._functorch.vmap._flat_vmap": UserFunctionVariable,
    "torch._functorch.vmap._flatten_chunks_output": UserFunctionVariable,
    "torch._functorch.vmap._get_chunked_inputs": UserFunctionVariable,
    "torch._functorch.vmap._get_name": UserFunctionVariable,
    "torch._functorch.vmap._maybe_remove_batch_dim": UserFunctionVariable,
    "torch._functorch.vmap._num_outputs": UserFunctionVariable,
    "torch._functorch.vmap._process_batched_inputs": UserFunctionVariable,
    "torch._functorch.vmap._unwrap_batched": UserFunctionVariable,
    "torch._functorch.vmap._validate_and_get_batch_size": UserFunctionVariable,
    "torch._functorch.vmap.doesnt_support_saved_tensors_hooks": UserFunctionVariable,
    "torch._functorch.vmap.get_chunk_sizes": UserFunctionVariable,
    # lazy_load_decompositions uses a lock that is not supported yet in dynamo
    # "torch._functorch.vmap.lazy_load_decompositions": UserFunctionVariable,
    "torch._functorch.vmap.restore_vmap": UserFunctionVariable,
    "torch._functorch.apis.vmap": UserFunctionVariable,
    "torch._functorch.vmap.unwrap_batched": UserFunctionVariable,
    "torch._functorch.vmap.vmap_impl": FunctorchHigherOrderVariable,
    "torch._functorch.vmap.wrap_batched": UserFunctionVariable,
    # functorch/grad
    "torch._functorch.eager_transforms.grad_impl": FunctorchHigherOrderVariable,
    "torch._functorch.apis.grad_and_value": UserFunctionVariable,
    "torch._functorch.eager_transforms._as_tuple": UserFunctionVariable,
    "torch._functorch.eager_transforms._check_unique_non_empty": UserFunctionVariable,
    "torch._functorch.eager_transforms._create_differentiable": UserFunctionVariable,
    "torch._functorch.eager_transforms._slice_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms._undo_create_differentiable": UserFunctionVariable,
    "torch._functorch.eager_transforms._validate_and_wrap_argnum": UserFunctionVariable,
    "torch._functorch.eager_transforms._validate_and_wrap_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms._wrap_all_tensors": UserFunctionVariable,
    "torch._functorch.eager_transforms._wrap_tensor_for_grad": UserFunctionVariable,
    # functorch/jacrev
    "torch._functorch.eager_transforms.jacrev": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms.error_if_complex": UserFunctionVariable,
    "torch._functorch.eager_transforms._chunked_standard_basis_for_": UserFunctionVariable,
    "torch._functorch.eager_transforms._safe_zero_index": UserFunctionVariable,
    # functorch/vjp
    "torch._functorch.eager_transforms.vjp": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._vjp_with_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_non_empty_tensor_output": UserFunctionVariable,
    # functorch/jvp
    "torch._functorch.eager_transforms._jvp_with_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms.jvp": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._replace_args": UserFunctionVariable,
    "torch._functorch.eager_transforms.safe_unpack_dual": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_non_empty_list_of_tensors": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_output_is_tensor_or_tensors": UserFunctionVariable,
    "torch.autograd.forward_ad.enter_dual_level": UserFunctionVariable,
    "torch.autograd.forward_ad.exit_dual_level": UserFunctionVariable,
    "torch.autograd.forward_ad.make_dual": UserFunctionVariable,
    "torch.autograd.forward_ad.unpack_dual": UserFunctionVariable,
    # functorch/linearize
    "torch._functorch.eager_transforms.linearize": FunctorchHigherOrderVariable,
    # functorch/jacfwd
    "torch._functorch.eager_transforms.jacfwd": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._construct_standard_basis_for": UserFunctionVariable,
    "torch._functorch.eager_transforms.safe_unflatten": UserFunctionVariable,
    # functorch/hessian
    "torch._functorch.eager_transforms.hessian": FunctorchHigherOrderVariable,
    # functional_call
    "torch._functorch.functional_call.functional_call": FunctionalCallVariable,
    "torch.nn.utils.stateless._groupby_tensor": TorchInGraphFunctionVariable,
    "torch.nn.utils.stateless._reparametrize_module": ReparametrizeModuleCallVariable,
    # functorch/deprecated
    "torch._functorch.deprecated.jvp": UserFunctionVariable,
    "torch._functorch.deprecated.hessian": UserFunctionVariable,
    "torch._functorch.deprecated.jacfwd": UserFunctionVariable,
    "torch._functorch.deprecated.jacrev": UserFunctionVariable,
    "torch._functorch.deprecated.grad": UserFunctionVariable,
    "torch._functorch.deprecated.grad_and_value": UserFunctionVariable,
    "torch._functorch.deprecated.vjp": UserFunctionVariable,
    # functorch/C++ bindings
    "torch._C._functorch._wrap_for_grad": TorchInGraphFunctionVariable,
    "torch._C._functorch._unwrap_for_grad": TorchInGraphFunctionVariable,
    "torch._C._functorch._unwrap_batched": TorchInGraphFunctionVariable,
    "torch._C._functorch.current_level": TorchInGraphFunctionVariable,
    "torch._C._functorch.maybe_current_level": TorchInGraphFunctionVariable,
    "torch._C._functorch.is_batchedtensor": TorchInGraphFunctionVariable,
    "torch._C._functorch.peek_interpreter_stack": TorchInGraphFunctionVariable,
    "torch._C._functorch.unwrap_if_dead": TorchInGraphFunctionVariable,
    "torch._functorch.predispatch._vmap_increment_nesting": TorchInGraphFunctionVariable,
    "torch._functorch.predispatch._vmap_decrement_nesting": TorchInGraphFunctionVariable,
    # everything else
    "torch._functorch.pyfunctorch.coerce_cinterpreter": TorchInGraphFunctionVariable,
    "torch._higher_order_ops.triton_kernel_wrap.do_prune_configs": UserFunctionVariable,
    "torch._higher_order_ops.foreach_map.foreach_map": UserFunctionVariable,
    "torch._constrain_as_size": UserFunctionVariable,
    "torch._tensor._convert": UserFunctionVariable,
    "torch.jit._unwrap_optional": UserFunctionVariable,
    "torch.backends.mha.get_fastpath_enabled": UserFunctionVariable,
    "torch._dynamo.dont_skip_tracing": UserFunctionVariable,
    "torch._dynamo.mark_static": UserFunctionVariable,
    "torch._dynamo.nonstrict_trace": UserFunctionVariable,
    "torch._dynamo.patch_dynamo_config": UserFunctionVariable,
    "torch._dynamo.error_on_graph_break": UserFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.guard_size_oblivious": TorchInGraphFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.guard_or_true": TorchInGraphFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.guard_or_false": TorchInGraphFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.statically_known_true": TorchInGraphFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.statically_known_false": TorchInGraphFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.sym_and": TorchInGraphFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.sym_or": TorchInGraphFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.guard_scalar": TorchInGraphFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.has_static_value": TorchInGraphFunctionVariable,
    "torch.cuda._get_device_properties": TorchInGraphFunctionVariable,
    "torch.utils.hooks.BackwardHook": TorchInGraphFunctionVariable,
    "torch.set_default_device": UserFunctionVariable,
    "torch.sparse_bsc_tensor": SkipFunctionVariable,
    "torch.sparse_bsr_tensor": SkipFunctionVariable,
    "torch.sparse_csc_tensor": SkipFunctionVariable,
    "torch.sparse_csr_tensor": SkipFunctionVariable,
    "torch.sparse_compressed_tensor": SkipFunctionVariable,
    "torch._C._autograd._unsafe_set_version_counter": TorchInGraphFunctionVariable,
    "torch.xpu.get_rng_state": SkipFunctionVariable,
    "torch.xpu.set_rng_state": SkipFunctionVariable,
    # avoid skipping user defined modules in distributed unit tests
    "torch/testing/_internal/common_fsdp.py#forward": UserFunctionVariable,
    f"torch/testing/_internal/common_fsdp.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    "torch/testing/_internal/distributed/_tensor/common_dtensor.py#forward": UserFunctionVariable,
    f"torch/testing/_internal/distributed/_tensor/common_dtensor.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    "torch/testing/_internal/common_distributed.py#forward": UserFunctionVariable,
    f"torch/testing/_internal/common_distributed.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
}


# In graph functions (including constant folding) that are C bindings
torch_c_binding_in_graph_functions = dict.fromkeys(
    [
        "math.acos",
        "math.acosh",
        "math.asin",
        "math.asinh",
        "math.atan",
        "math.atan2",
        "math.atanh",
        "math.ceil",
        "math.comb",
        "math.copysign",
        "math.cos",
        "math.cosh",
        "math.degrees",
        "math.dist",
        "math.erf",
        "math.erfc",
        "math.exp",
        "math.expm1",
        "math.fabs",
        "math.factorial",
        "math.floor",
        "math.fmod",
        "math.frexp",
        "math.fsum",
        "math.gamma",
        "math.gcd",
        "math.hypot",
        "math.isclose",
        "math.isfinite",
        "math.isinf",
        "math.isnan",
        "math.isqrt",
        "math.lcm",
        "math.ldexp",
        "math.lgamma",
        "math.log",
        "math.log10",
        "math.log1p",
        "math.log2",
        "math.modf",
        "math.nextafter",
        "math.perm",
        "math.pow",
        "math.prod",
        "math.radians",
        "math.remainder",
        "math.sin",
        "math.sinh",
        "math.tan",
        "math.tanh",
        "math.trunc",
        "math.ulp",
        "torch._adaptive_avg_pool2d",
        "torch._adaptive_avg_pool3d",
        "torch._add_batch_dim",
        "torch._add_relu_",
        "torch._add_relu",
        "torch._addmm_activation",
        "torch._aminmax",
        "torch._amp_foreach_non_finite_check_and_unscale_",
        "torch._amp_update_scale_",
        "torch._assert_async",
        "torch._assert_tensor_metadata",
        "torch._batch_norm_impl_index",
        "torch._C._accelerator_getAccelerator",
        "torch._C._accelerator_getDeviceIndex",
        "torch._C._accelerator_getStream",
        "torch._C._accelerator_setAllocatorSettings",
        "torch._C._accelerator_setStream",
        "torch._C._accelerator_synchronizeDevice",
        "torch._C._activate_gpu_trace",
        "torch._C._add_cached_tensor",
        "torch._C._add_docstr",
        "torch._C._are_functorch_transforms_active",
        "torch._C._autograd_init",
        "torch._C._awaitable_nowait",
        "torch._C._awaitable_wait",
        "torch._C._awaitable",
        "torch._C._backport_for_mobile_from_buffer_to_buffer",
        "torch._C._backport_for_mobile_from_buffer",
        "torch._C._backport_for_mobile_to_buffer",
        "torch._C._backport_for_mobile",
        "torch._C._broadcast_coalesced",
        "torch._C._broadcast_out",
        "torch._C._broadcast",
        "torch._C._c10d_init",
        "torch._C._calculate_package_version_based_on_upgraders",
        "torch._C._can_use_flash_attention",
        "torch._C._can_use_mem_efficient_attention",
        "torch._C._can_use_cudnn_attention",
        "torch._C._check_onnx_proto",
        "torch._C._check_sparse_tensor_invariants",
        "torch._C._collect_all",
        "torch._C._commit_update",
        "torch._C._compile_graph_to_code_table",
        "torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata",
        "torch._C._construct_storage_from_data_pointer",
        "torch._C._conv_determine_backend_memory_format",
        "torch._C._cpu._is_avx2_supported",
        "torch._C._cpu._is_avx512_supported",
        "torch._C._cpu._is_avx512_vnni_supported",
        "torch._C._cpu._is_avx512_bf16_supported",
        "torch._C._cpu._is_amx_tile_supported",
        "torch._C._cpu._is_amx_fp16_supported",
        "torch._C._cpu._init_amx",
        "torch._C._crash_if_aten_asan",
        "torch._C._crash_if_csrc_asan",
        "torch._C._crash_if_csrc_ubsan",
        "torch._C._crash_if_debug_asserts_fail",
        "torch._C._crash_if_vptr_ubsan",
        "torch._C._create_function_from_graph",
        "torch._C._create_function_from_trace_with_dict",
        "torch._C._create_function_from_trace",
        "torch._C._create_graph_by_tracing",
        "torch._C._create_module_with_type",
        "torch._C._create_object_with_type",
        "torch._C._cuda_attach_out_of_memory_observer",
        "torch._C._cuda_beginAllocateCurrentStreamToPool",
        "torch._C._cuda_canDeviceAccessPeer",
        "torch._C._cuda_changeCurrentAllocator",
        "torch._C._cuda_checkPoolLiveAllocations",
        "torch._C._cuda_clearCublasWorkspaces",
        "torch._C._cuda_cudaCachingAllocator_raw_alloc",
        "torch._C._cuda_cudaCachingAllocator_raw_delete",
        "torch._C._cuda_cudaHostAllocator",
        "torch._C._cuda_customAllocator",
        "torch._C._cuda_emptyCache",
        "torch._C._cuda_endAllocateToPool",
        "torch._C._cuda_exchangeDevice",
        "torch._C._cuda_get_conv_benchmark_empty_cache",
        "torch._C._cuda_get_cudnn_benchmark_limit",
        "torch._C._cuda_get_sync_debug_mode",
        "torch._C._cuda_getAllocator",
        "torch._C._cuda_getAllocatorBackend",
        "torch._C._cuda_getArchFlags",
        "torch._C._cuda_getCheckpointState",
        "torch._C._cuda_getCompiledVersion",
        "torch._C._cuda_getCurrentBlasHandle",
        "torch._C._cuda_getCurrentRawStream",
        "torch._C._cuda_getCurrentStream",
        "torch._C._cuda_getDefaultStream",
        "torch._C._cuda_getDevice",
        "torch._C._cuda_getDeviceCount",
        "torch._C._cuda_hasPrimaryContext",
        "torch._C._cuda_hostMemoryStats",
        "torch._C._cuda_init",
        "torch._C._cuda_ipc_collect",
        "torch._C._cuda_isCurrentStreamCapturing",
        "torch._C._cuda_isHistoryEnabled",
        "torch._C._cuda_isInBadFork",
        "torch._C._cuda_jiterator_compile_and_launch_kernel",
        "torch._C._cuda_lock_mutex",
        "torch._C._cuda_maybeExchangeDevice",
        "torch._C._cuda_memorySnapshot",
        "torch._C._cuda_memoryStats",
        "torch._C._cuda_record_memory_history_legacy",
        "torch._C._cuda_record_memory_history",
        "torch._C._cuda_releasePool",
        "torch._C._cuda_resetAccumulatedHostMemoryStats",
        "torch._C._cuda_resetAccumulatedMemoryStats",
        "torch._C._cuda_resetPeakHostMemoryStats",
        "torch._C._cuda_resetPeakMemoryStats",
        "torch._C._cuda_set_cudnn_benchmark_limit",
        "torch._C._cuda_set_sync_debug_mode",
        "torch._C._cuda_setCheckpointPoolState",
        "torch._C._cuda_setDevice",
        "torch._C._cuda_setMemoryFraction",
        "torch._C._cuda_setStream",
        "torch._C._cuda_sleep",
        "torch._C._cuda_synchronize",
        "torch._C._cuda_unlock_mutex",
        "torch._C._cudnn_set_conv_benchmark_empty_cache",
        "torch._C._cudnn.getCompileVersion",
        "torch._C._cudnn.getRuntimeVersion",
        "torch._C._cudnn.getVersionInt",
        "torch._C._current_autograd_node",
        "torch._C._current_graph_task_execution_order",
        "torch._C._current_graph_task_id",
        "torch._C._cxx_flags",
        "torch._C._debug_get_fusion_group_inlining",
        "torch._C._debug_only_are_vmap_fallback_warnings_enabled",
        "torch._C._debug_only_display_vmap_fallback_warnings",
        "torch._C._debug_set_autodiff_subgraph_inlining",
        "torch._C._debug_set_fusion_group_inlining",
        "torch._C._demangle",
        "torch._C._disabled_torch_dispatch_impl",
        "torch._C._dispatch_call_boxed",
        "torch._C._dispatch_check_all_invariants",
        "torch._C._dispatch_check_invariants",
        "torch._C._dispatch_dump_table",
        "torch._C._dispatch_dump",
        "torch._C._dispatch_find_dangling_impls",
        "torch._C._dispatch_find_schema_or_throw",
        "torch._C._dispatch_get_all_op_names",
        "torch._C._dispatch_get_backend_keyset_from_autograd",
        "torch._C._dispatch_get_registrations_for_dispatch_key",
        "torch._C._dispatch_has_backend_fallback",
        "torch._C._dispatch_has_computed_kernel_for_dispatch_key",
        "torch._C._dispatch_has_kernel_for_any_dispatch_key",
        "torch._C._dispatch_has_kernel_for_dispatch_key",
        "torch._C._dispatch_has_kernel",
        "torch._C._dispatch_is_alias_key",
        "torch._C._dispatch_is_included_in_alias",
        "torch._C._dispatch_isTensorSubclassLike",
        "torch._C._dispatch_key_for_device",
        "torch._C._dispatch_key_name",
        "torch._C._dispatch_key_parse",
        "torch._C._dispatch_key_set",
        "torch._C._dispatch_keys",
        "torch._C._dispatch_keyset_full_after",
        "torch._C._dispatch_keyset_full",
        "torch._C._dispatch_keyset_to_string",
        "torch._C._dispatch_library",
        "torch._C._dispatch_num_backends",
        "torch._C._dispatch_print_registrations_for_dispatch_key",
        "torch._C._dispatch_pystub",
        "torch._C._dispatch_set_report_error_callback",
        "torch._C._dispatch_tls_is_dispatch_key_excluded",
        "torch._C._dispatch_tls_is_dispatch_key_included",
        "torch._C._dispatch_tls_local_exclude_set",
        "torch._C._dispatch_tls_local_include_set",
        "torch._C._dispatch_tls_set_dispatch_key_excluded",
        "torch._C._dispatch_tls_set_dispatch_key_included",
        "torch._C._dist_autograd_init",
        "torch._C._dump_local_tls_set",
        "torch._C._dump_upgraders_map",
        "torch._C._enable_mobile_interface_call_export",
        "torch._C._enter_dual_level",
        "torch._C._error_if_any_worker_fails",
        "torch._C._exit_dual_level",
        "torch._C._export_operator_list",
        "torch._C._export_opnames",
        "torch._C._faulty_agent_init",
        "torch._C._fft.fft_fft",
        "torch._C._fft.fft_fft2",
        "torch._C._fft.fft_fftfreq",
        "torch._C._fft.fft_fftn",
        "torch._C._fft.fft_fftshift",
        "torch._C._fft.fft_hfft",
        "torch._C._fft.fft_hfft2",
        "torch._C._fft.fft_hfftn",
        "torch._C._fft.fft_ifft",
        "torch._C._fft.fft_ifft2",
        "torch._C._fft.fft_ifftn",
        "torch._C._fft.fft_ifftshift",
        "torch._C._fft.fft_ihfft",
        "torch._C._fft.fft_ihfft2",
        "torch._C._fft.fft_ihfftn",
        "torch._C._fft.fft_irfft",
        "torch._C._fft.fft_irfft2",
        "torch._C._fft.fft_irfftn",
        "torch._C._fft.fft_rfft",
        "torch._C._fft.fft_rfft2",
        "torch._C._fft.fft_rfftfreq",
        "torch._C._fft.fft_rfftn",
        "torch._C._free_And_Remove_DeleterFn",
        "torch._C._freeze_module",
        "torch._C._from_dlpack",
        "torch._C._functionality_to_backend_keys",
        "torch._C._functionalization_reapply_views_tls",
        "torch._C._fuse_to_static_module",
        "torch._C._gather_out",
        "torch._C._gather",
        "torch._C._generate_upgraders_graph",
        "torch._C._get_autograd_fallback_mode",
        "torch._C._get_backcompat_broadcast_warn",
        "torch._C._get_backcompat_keepdim_warn",
        "torch._C._get_blas_preferred_backend",
        "torch._C._get_caught_jit_exception_class_name",
        "torch._C._get_caught_jit_exception_original_msg",
        "torch._C._get_constant_bool_symnode",
        "torch._C._get_cpp_backtrace",
        "torch._C._get_cpu_capability",
        "torch._C._get_cublas_allow_bf16_reduced_precision_reduction",
        "torch._C._get_cublas_allow_fp16_reduced_precision_reduction",
        "torch._C._get_cublas_allow_tf32",
        "torch._C._get_cudnn_allow_tf32",
        "torch._C._get_cudnn_benchmark",
        "torch._C._get_miopen_immediate",
        "torch._C._get_cudnn_deterministic",
        "torch._C._get_cudnn_enabled",
        "torch._C._get_custom_class_python_wrapper",
        "torch._C._get_default_device",
        "torch._C._get_deterministic_algorithms_warn_only",
        "torch._C._get_deterministic_algorithms",
        "torch._C._get_deterministic_fill_uninitialized_memory",
        "torch._C._get_dispatch_mode",
        "torch._C._get_dispatch_stack_at",
        "torch._C._get_file_format",
        "torch._C._get_flash_sdp_enabled",
        "torch._C._get_float32_matmul_precision",
        "torch._C._get_function_stack_at",
        "torch._C._get_graph_executor_optimize",
        "torch._C._get_linalg_preferred_backend",
        "torch._C._get_rocm_fa_preferred_backend",
        "torch._C._get_math_sdp_enabled",
        "torch._C._get_math_sdp_allow_fp16_bf16_reduction",
        "torch._C._get_max_operator_version",
        "torch._C._get_mem_efficient_sdp_enabled",
        "torch._C._get_mkldnn_enabled",
        "torch._C._get_cudnn_sdp_enabled",
        "torch._C._get_overrideable_sdp_enabled",
        "torch._C._set_sdp_use_cudnn",
        "torch._C._get_mobile_model_contained_types_from_buffer",
        "torch._C._get_mobile_model_contained_types",
        "torch._C._get_model_bytecode_version_from_buffer",
        "torch._C._get_model_bytecode_version",
        "torch._C._get_model_extra_files_from_buffer",
        "torch._C._get_model_extra_files",
        "torch._C._get_model_ops_and_info_from_buffer",
        "torch._C._get_model_ops_and_info",
        "torch._C._get_module_info_from_flatbuffer",
        "torch._C._get_nnpack_enabled",
        "torch._C._get_obj_in_tls",
        "torch._C._get_operation_overload",
        "torch._C._get_operator_version_map",
        "torch._C._get_privateuse1_backend_name",
        "torch._C._get_qengine",
        "torch._C._get_schema",
        "torch._C._get_sm_carveout_experimental",
        "torch._C._get_nested_int",
        "torch._C._get_tensor_metadata",
        "torch._C._get_tracing_state",
        "torch._C._get_upgrader_ranges",
        "torch._C._get_upgraders_entry_map",
        "torch._C._get_upgraders_map_size",
        "torch._C._get_value_trace",
        "torch._C._get_version_calculator_flag",
        "torch._C._get_warnAlways",
        "torch._C._graph_pool_handle",
        "torch._C._group_tensors_by_device_and_dtype",
        "torch._C._hack_do_not_use_clone_module_with_class",
        "torch._C._has_distributed",
        "torch._C._has_Standard_Deleter",
        "torch._C._has_storage",
        "torch._C._has_tensorexpr_cpp_tests",
        "torch._C._run_tensorexpr_cpp_tests",
        "torch._C._has_torch_function_unary",
        "torch._C._has_torch_function_variadic",
        "torch._C._has_torch_function",
        "torch._C._import_ir_module_from_package",
        "torch._C._increment_version",
        "torch._C._infer_size",
        "torch._C._init_names",
        "torch._C._initExtension",
        "torch._C._is_alias_of",
        "torch._C._is_any_autocast_enabled",
        "torch._C._is_cached_tensor",
        "torch._C._is_flash_attention_available",
        "torch._C._is_fwd_grad_enabled",
        "torch._C._is_key_in_tls",
        "torch._C._is_multithreading_enabled",
        "torch._C._is_torch_function_enabled",
        "torch._C._is_torch_function_mode_enabled",
        "torch._C._is_torch_function_all_disabled",
        "torch._C._is_tracing",
        "torch._C._is_view_replay_enabled",
        "torch._C._is_xnnpack_enabled",
        "torch._C._itt.is_available",
        "torch._C._itt.mark",
        "torch._C._itt.rangePop",
        "torch._C._itt.rangePush",
        "torch._C._ivalue_debug_python_object",
        "torch._C._ivalue_tags_match",
        "torch._C._jit_assert_is_instance",
        "torch._C._jit_can_fuse_on_cpu_legacy",
        "torch._C._jit_can_fuse_on_cpu",
        "torch._C._jit_can_fuse_on_gpu",
        "torch._C._jit_cat_wo_conditionals",
        "torch._C._jit_check_alias_annotation",
        "torch._C._jit_clear_class_registry",
        "torch._C._jit_debug_fuser_num_cached_kernel_specs",
        "torch._C._jit_debug_module_iterators",
        "torch._C._jit_decay_packed_param_input_types",
        "torch._C._jit_decomposition_graph_for_node",
        "torch._C._jit_differentiate",
        "torch._C._jit_erase_non_input_shape_information",
        "torch._C._jit_flatten",
        "torch._C._jit_fuser_get_fused_kernel_code",
        "torch._C._jit_get_all_schemas",
        "torch._C._jit_get_custom_class_schemas",
        "torch._C._jit_get_emit_hooks",
        "torch._C._jit_get_inline_everything_mode",
        "torch._C._jit_get_logging_option",
        "torch._C._jit_get_num_profiled_runs",
        "torch._C._jit_get_operation",
        "torch._C._jit_get_schemas_for_operator",
        "torch._C._jit_get_te_cuda_pointwise_block_count",
        "torch._C._jit_get_te_cuda_pointwise_block_size",
        "torch._C._jit_get_te_cuda_pointwise_loop_levels",
        "torch._C._jit_get_te_generate_block_code",
        "torch._C._jit_get_te_must_use_llvm_cpu",
        "torch._C._jit_get_tracer_state_warn",
        "torch._C._jit_has_cpp_tests",
        "torch._C._jit_init",
        "torch._C._jit_interpret_graph",
        "torch._C._jit_is_onnx_log_enabled",
        "torch._C._jit_is_script_object",
        "torch._C._jit_llga_enabled",
        "torch._C._jit_nvfuser_can_be_enabled",
        "torch._C._jit_nvfuser_clear_comparison_callback",
        "torch._C._jit_nvfuser_enabled",
        "torch._C._jit_nvfuser_horizontal_mode",
        "torch._C._jit_nvfuser_set_comparison_callback",
        "torch._C._jit_nvfuser_single_node_mode",
        "torch._C._jit_object_is_non_holding",
        "torch._C._jit_onnx_convert_pattern_from_subblock",
        "torch._C._jit_onnx_create_full_scope_name",
        "torch._C._jit_onnx_list_model_parameters",
        "torch._C._jit_onnx_log",
        "torch._C._jit_opt_conditionals",
        "torch._C._jit_override_can_fuse_on_cpu_legacy",
        "torch._C._jit_override_can_fuse_on_cpu",
        "torch._C._jit_override_can_fuse_on_gpu",
        "torch._C._jit_pass_autocast",
        "torch._C._jit_pass_batch_mm",
        "torch._C._jit_pass_canonicalize_graph_fuser_ops",
        "torch._C._jit_pass_canonicalize",
        "torch._C._jit_pass_complete_shape_analysis",
        "torch._C._jit_pass_concat_frozen_linear",
        "torch._C._jit_pass_constant_loop_unrolling",
        "torch._C._jit_pass_constant_pooling",
        "torch._C._jit_pass_constant_propagation_immutable_types",
        "torch._C._jit_pass_constant_propagation",
        "torch._C._jit_pass_convert_frozen_ops_to_mkldnn",
        "torch._C._jit_pass_create_autodiff_subgraphs",
        "torch._C._jit_pass_create_functional_graphs",
        "torch._C._jit_pass_cse",
        "torch._C._jit_pass_custom_pattern_based_rewrite_graph",
        "torch._C._jit_pass_custom_pattern_based_rewrite",
        "torch._C._jit_pass_dbr_quant_remove_redundant_aliases",
        "torch._C._jit_pass_dce_allow_deleting_nodes_with_side_effects",
        "torch._C._jit_pass_dce",
        "torch._C._jit_pass_decompose_ops",
        "torch._C._jit_pass_dedup_module_uses",
        "torch._C._jit_pass_erase_number_types",
        "torch._C._jit_pass_erase_shape_information",
        "torch._C._jit_pass_filter_non_tensor_arguments",
        "torch._C._jit_pass_fixup_onnx_controlflow_node",
        "torch._C._jit_pass_fold_convbn",
        "torch._C._jit_pass_fold_frozen_conv_add_or_sub",
        "torch._C._jit_pass_fold_frozen_conv_bn",
        "torch._C._jit_pass_fold_frozen_conv_mul_or_div",
        "torch._C._jit_pass_fold_frozen_linear_bn",
        "torch._C._jit_pass_fold_prepacking_ops",
        "torch._C._jit_pass_functional_to_inplace_activation",
        "torch._C._jit_pass_fuse_add_relu",
        "torch._C._jit_pass_fuse_addmm",
        "torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv",
        "torch._C._jit_pass_fuse_frozen_conv_add_relu",
        "torch._C._jit_pass_fuse_linear",
        "torch._C._jit_pass_fuse_quantized_add_relu",
        "torch._C._jit_pass_fuse_tensorexprs",
        "torch._C._jit_pass_fuse",
        "torch._C._jit_pass_inline_fork_wait",
        "torch._C._jit_pass_inline_functional_graphs",
        "torch._C._jit_pass_inline",
        "torch._C._jit_pass_inplace_to_functional_activation",
        "torch._C._jit_pass_insert_observer_method_for_ondevice_ptq",
        "torch._C._jit_pass_insert_observers",
        "torch._C._jit_pass_insert_prepack_unpack",
        "torch._C._jit_pass_insert_prepacked_ops",
        "torch._C._jit_pass_insert_quant_dequant_for_ondevice_ptq",
        "torch._C._jit_pass_insert_quant_dequant",
        "torch._C._jit_pass_integer_value_refinement",
        "torch._C._jit_pass_lint",
        "torch._C._jit_pass_loop_unrolling",
        "torch._C._jit_pass_lower_all_tuples",
        "torch._C._jit_pass_lower_graph",
        "torch._C._jit_pass_metal_fold_prepacking_ops",
        "torch._C._jit_pass_metal_fuse_clamp_w_prepacked_conv",
        "torch._C._jit_pass_metal_insert_prepacked_ops",
        "torch._C._jit_pass_metal_optimize_for_mobile",
        "torch._C._jit_pass_onnx_assign_output_shape",
        "torch._C._jit_pass_onnx_assign_scoped_names_for_node_and_value",
        "torch._C._jit_pass_onnx_autograd_function_process",
        "torch._C._jit_pass_onnx_block",
        "torch._C._jit_pass_onnx_cast_all_constant_to_floating",
        "torch._C._jit_pass_onnx_clear_scope_records",
        "torch._C._jit_pass_onnx_constant_fold",
        "torch._C._jit_pass_onnx_deduplicate_initializers",
        "torch._C._jit_pass_onnx_eliminate_unused_items",
        "torch._C._jit_pass_onnx_eval_peephole",
        "torch._C._jit_pass_onnx_function_extraction",
        "torch._C._jit_pass_onnx_function_substitution",
        "torch._C._jit_pass_onnx_graph_shape_type_inference",
        "torch._C._jit_pass_onnx_lint",
        "torch._C._jit_pass_onnx_node_shape_type_inference",
        "torch._C._jit_pass_onnx_peephole",
        "torch._C._jit_pass_onnx_preprocess_caffe2",
        "torch._C._jit_pass_onnx_preprocess",
        "torch._C._jit_pass_onnx_quantization_insert_permutes",
        "torch._C._jit_pass_onnx_remove_inplace_ops_for_onnx",
        "torch._C._jit_pass_onnx_remove_print",
        "torch._C._jit_pass_onnx_scalar_type_analysis",
        "torch._C._jit_pass_onnx_set_dynamic_input_shape",
        "torch._C._jit_pass_onnx_track_scope_attributes",
        "torch._C._jit_pass_onnx_unpack_quantized_weights",
        "torch._C._jit_pass_onnx",
        "torch._C._jit_pass_optimize_for_inference",
        "torch._C._jit_pass_optimize_for_mobile",
        "torch._C._jit_pass_optimize_frozen_graph",
        "torch._C._jit_pass_pattern_based_rewrite",
        "torch._C._jit_pass_peephole_list_idioms",
        "torch._C._jit_pass_peephole",
        "torch._C._jit_pass_prepare_division_for_onnx",
        "torch._C._jit_pass_propagate_device",
        "torch._C._jit_pass_propagate_dtype",
        "torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute",
        "torch._C._jit_pass_propagate_shapes_on_graph",
        "torch._C._jit_pass_quant_finalize_for_ondevice_ptq",
        "torch._C._jit_pass_quant_finalize",
        "torch._C._jit_pass_quant_fusion",
        "torch._C._jit_pass_refine_integer_values",
        "torch._C._jit_pass_refine_tuple_types",
        "torch._C._jit_pass_remove_dropout",
        "torch._C._jit_pass_remove_expands",
        "torch._C._jit_pass_remove_inplace_ops",
        "torch._C._jit_pass_remove_mutation",
        "torch._C._jit_pass_replace_old_ops_with_upgraders",
        "torch._C._jit_pass_replicate_dequantize",
        "torch._C._jit_pass_run_decompositions",
        "torch._C._jit_pass_specialize_autogradzero",
        "torch._C._jit_pass_swap_functional_linear",
        "torch._C._jit_pass_transform_conv1d_to_conv2d",
        "torch._C._jit_pass_transpose_frozen_linear",
        "torch._C._jit_pass_vulkan_fold_prepacking_ops",
        "torch._C._jit_pass_vulkan_fuse_clamp_w_prepacked_conv",
        "torch._C._jit_pass_vulkan_insert_prepacked_ops",
        "torch._C._jit_pass_vulkan_optimize_for_mobile",
        "torch._C._jit_register_decomposition_for_schema",
        "torch._C._jit_register_shape_compute_graph_for_node",
        "torch._C._jit_resolve_packet",
        "torch._C._jit_run_cpp_tests",
        "torch._C._jit_script_class_compile",
        "torch._C._jit_script_compile_overload",
        "torch._C._jit_script_compile",
        "torch._C._jit_script_interface_compile",
        "torch._C._jit_set_autocast_mode",
        "torch._C._jit_set_bailout_depth",
        "torch._C._jit_set_emit_hooks",
        "torch._C._jit_set_fusion_strategy",
        "torch._C._jit_set_inline_everything_mode",
        "torch._C._jit_set_llga_enabled",
        "torch._C._jit_set_logging_option",
        "torch._C._jit_set_logging_stream",
        "torch._C._jit_set_num_profiled_runs",
        "torch._C._jit_set_nvfuser_enabled",
        "torch._C._jit_set_nvfuser_guard_mode",
        "torch._C._jit_set_nvfuser_horizontal_mode",
        "torch._C._jit_set_nvfuser_single_node_mode",
        "torch._C._jit_set_nvfuser_skip_node_kind",
        "torch._C._jit_set_onnx_log_enabled",
        "torch._C._jit_set_onnx_log_output_stream",
        "torch._C._jit_set_profiling_executor",
        "torch._C._jit_set_profiling_mode",
        "torch._C._jit_set_symbolic_shapes_test_mode",
        "torch._C._jit_set_te_cuda_pointwise_block_count",
        "torch._C._jit_set_te_cuda_pointwise_block_size",
        "torch._C._jit_set_te_cuda_pointwise_loop_levels",
        "torch._C._jit_set_te_generate_block_code",
        "torch._C._jit_set_te_must_use_llvm_cpu",
        "torch._C._jit_set_texpr_dynamic_shape_enabled",
        "torch._C._jit_set_texpr_fuser_enabled",
        "torch._C._jit_set_texpr_reductions_enabled",
        "torch._C._jit_set_tracer_state_warn",
        "torch._C._jit_set_utf8_decoding_ignore",
        "torch._C._jit_shape_compute_graph_for_node",
        "torch._C._jit_symbolic_shapes_test_mode_enabled",
        "torch._C._jit_texpr_dynamic_shape_enabled",
        "torch._C._jit_texpr_fallback_allowed",
        "torch._C._jit_texpr_fuser_enabled",
        "torch._C._jit_texpr_reductions_enabled",
        "torch._C._jit_texpr_set_fallback_allowed",
        "torch._C._jit_to_backend_selective",
        "torch._C._jit_to_backend",
        "torch._C._jit_to_static_module",
        "torch._C._jit_trace_graph",
        "torch._C._jit_trace_module",
        "torch._C._jit_tree_views.FalseLiteral",
        "torch._C._jit_tree_views.NoneLiteral",
        "torch._C._jit_tree_views.TrueLiteral",
        "torch._C._jit_try_infer_type",
        "torch._C._jit_unflatten",
        "torch._C._last_executed_optimized_graph",
        "torch._C._len_torch_dispatch_stack",
        "torch._C._len_torch_function_stack",
        "torch._C._linalg._linalg_eigvals",
        "torch._C._linalg.linalg_cholesky_ex",
        "torch._C._linalg.linalg_cholesky",
        "torch._C._linalg.linalg_cond",
        "torch._C._linalg.linalg_cross",
        "torch._C._linalg.linalg_det",
        "torch._C._linalg.linalg_diagonal",
        "torch._C._linalg.linalg_eig",
        "torch._C._linalg.linalg_eigh",
        "torch._C._linalg.linalg_eigvals",
        "torch._C._linalg.linalg_eigvalsh",
        "torch._C._linalg.linalg_householder_product",
        "torch._C._linalg.linalg_inv_ex",
        "torch._C._linalg.linalg_inv",
        "torch._C._linalg.linalg_ldl_factor_ex",
        "torch._C._linalg.linalg_ldl_factor",
        "torch._C._linalg.linalg_ldl_solve",
        "torch._C._linalg.linalg_lstsq",
        "torch._C._linalg.linalg_lu_factor_ex",
        "torch._C._linalg.linalg_lu_factor",
        "torch._C._linalg.linalg_lu_solve",
        "torch._C._linalg.linalg_lu",
        "torch._C._linalg.linalg_matmul",
        "torch._C._linalg.linalg_matrix_exp",
        "torch._C._linalg.linalg_matrix_norm",
        "torch._C._linalg.linalg_matrix_power",
        "torch._C._linalg.linalg_matrix_rank",
        "torch._C._linalg.linalg_multi_dot",
        "torch._C._linalg.linalg_norm",
        "torch._C._linalg.linalg_pinv",
        "torch._C._linalg.linalg_qr",
        "torch._C._linalg.linalg_slogdet",
        "torch._C._linalg.linalg_solve_ex",
        "torch._C._linalg.linalg_solve_triangular",
        "torch._C._linalg.linalg_solve",
        "torch._C._linalg.linalg_svd",
        "torch._C._linalg.linalg_svdvals",
        "torch._C._linalg.linalg_tensorinv",
        "torch._C._linalg.linalg_tensorsolve",
        "torch._C._linalg.linalg_vander",
        "torch._C._linalg.linalg_vecdot",
        "torch._C._linalg.linalg_vector_norm",
        "torch._C._llvm_enabled",
        "torch._C._load_for_lite_interpreter_from_buffer",
        "torch._C._load_for_lite_interpreter",
        "torch._C._load_jit_module_from_bytes",
        "torch._C._load_jit_module_from_file",
        "torch._C._load_mobile_module_from_bytes",
        "torch._C._load_mobile_module_from_file",
        "torch._C._log_api_usage_metadata",
        "torch._C._log_api_usage_once",
        "torch._C._logging_set_logger",
        "torch._C._meta_in_tls_dispatch_include",
        "torch._C._mps_acquireEvent",
        "torch._C._mps_currentAllocatedMemory",
 
```



## High-Level Overview

"""Tracing rules and policies for TorchDynamo compilation decisions.This module defines the rules that govern what code TorchDynamo should trace and compileversus what should be executed eagerly. It contains functions and classes that determine:- Which modules, functions, and objects should be skipped during tracing- Which parts of the code should cause graph breaks- How to handle different Python libraries and third-party packages- Rules for determining when to inline functions vs calling them eagerlyKey components:- Skip rules: Functions that return True if an object should be skipped during tracing- Inlining rules: Policies for when to inline function calls during compilation- Library-specific handling: Special cases for popular Python packages- Performance heuristics: Rules that balance compilation overhead vs runtime benefitsThese rules are critical for TorchDynamo's ability to automatically determinecompilation boundaries and optimize PyTorch programs effectively.

This Python file contains 3 class(es) and 53 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FunctionIdSet`, `SkipResult`, `FunctionInfo`

**Functions defined**: `get_torch_obj_rule_map`, `_load_obj_from_str`, `load_object`, `get_tensor_method`, `is_aten_op_or_tensor_method`, `__init__`, `__call__`, `get_name`, `add`, `remove`, `__contains__`, `_allowed_callable_ids`, `_disallowed_callable_ids`, `_nonstrict_trace_callable_ids`, `_builtin_function_ids`, `_polyfilled_function_ids`, `_numpy_function_ids`, `is_supported`, `_builtin_constant_ids`, `add_module_init_func`

**Key imports**: abc, builtins, copy, dataclasses, functools, importlib, inspect, linecache, operator, os


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `builtins`
- `copy`
- `dataclasses`
- `functools`
- `importlib`
- `inspect`
- `linecache`
- `operator`
- `os`
- `random`
- `re`
- `sys`
- `traceback`
- `types`
- `unittest`
- `collections`: defaultdict
- `collections.abc`: Callable
- `pathlib`: Path
- `typing`: Any, cast, Optional, Union
- `torch`
- `torch._inductor.test_operators`
- `torch.distributed`
- `torch.utils._content_store`
- `torch._environment`: is_fbcode
- `torch.utils`: _config_module
- `.`: config
- `.resume_execution`: TORCH_DYNAMO_RESUME_IN_PREFIX
- `.variables.base`: VariableTracker
- `numpy as np`


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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `trace_rules.py_docs.md`
- **Keyword Index**: `trace_rules.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
