# Documentation: `docs/build_variables.bzl_docs.md`

## File Metadata

- **Path**: `docs/build_variables.bzl_docs.md`
- **Size**: 52,405 bytes (51.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `build_variables.bzl`

## File Metadata

- **Path**: `build_variables.bzl`
- **Size**: 76,237 bytes (74.45 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This is a source file (.bzl) that is part of the PyTorch project.

## Original Source

```
# WARNING: the contents of this file must BOTH be valid Starlark (for Buck and

# Bazel) as well as valid Python (for our cmake build).  This means that
# load() directives are not allowed (as they are not recognized by Python).
# If you want to fix this, figure out how run this file from cmake with a proper
# Starlark interpreter as part of the default OSS build process.  If you need
# some nontrivial Starlark features, make a separate bzl file (remember that

# bzl files are not exported via ShipIt by default, so you may also need to
# update PyTorch's ShipIt config)

# This is duplicated in caffe2/CMakeLists.txt for now and not yet used in buck
GENERATED_LAZY_TS_CPP = [
    "lazy/generated/LazyNativeFunctions.cpp",
    "lazy/generated/RegisterAutogradLazy.cpp",
    "lazy/generated/RegisterLazy.cpp",
]

def libtorch_generated_sources(gencode_pattern):
    return [gencode_pattern.format(name) for name in [
        "torch/csrc/autograd/generated/Functions.cpp",
        "torch/csrc/autograd/generated/VariableType_0.cpp",
        "torch/csrc/autograd/generated/VariableType_1.cpp",
        "torch/csrc/autograd/generated/VariableType_2.cpp",
        "torch/csrc/autograd/generated/VariableType_3.cpp",
        "torch/csrc/autograd/generated/VariableType_4.cpp",
        "torch/csrc/autograd/generated/ViewFuncs.cpp",
        "torch/csrc/autograd/generated/TraceType_0.cpp",
        "torch/csrc/autograd/generated/TraceType_1.cpp",
        "torch/csrc/autograd/generated/TraceType_2.cpp",
        "torch/csrc/autograd/generated/TraceType_3.cpp",
        "torch/csrc/autograd/generated/TraceType_4.cpp",
        "torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp",
        "torch/csrc/autograd/generated/ADInplaceOrViewType_1.cpp",
    ]]

# copied from https://github.com/pytorch/pytorch/blob/f99a693cd9ff7a9b5fdc71357dac66b8192786d3/aten/src/ATen/core/CMakeLists.txt
jit_core_headers = [
    "torch/csrc/Export.h",
    "torch/csrc/jit/frontend/source_range.h",
    "torch/csrc/jit/serialization/callstack_debug_info_serialization.h",
    "torch/csrc/jit/serialization/source_range_serialization.h",
    "torch/csrc/jit/frontend/lexer.h",
    "torch/csrc/jit/frontend/strtod.h",
    "torch/csrc/jit/frontend/parser_constants.h",
    "torch/csrc/jit/frontend/function_schema_parser.h",
    "torch/csrc/jit/frontend/parse_string_literal.h",
    "torch/csrc/jit/frontend/schema_type_parser.h",
    "torch/csrc/jit/frontend/error_report.h",
    "torch/csrc/jit/frontend/tree.h",
    "torch/csrc/stable/library.h",
    "torch/custom_class.h",
    "torch/custom_class_detail.h",
    "torch/library.h",
]

jit_core_sources = [
    "torch/csrc/jit/frontend/error_report.cpp",
    "torch/csrc/jit/frontend/function_schema_parser.cpp",
    "torch/csrc/jit/frontend/lexer.cpp",
    "torch/csrc/jit/frontend/schema_type_parser.cpp",
    "torch/csrc/jit/frontend/strtod.cpp",
    "torch/csrc/jit/frontend/source_range.cpp",
]

# copied from https://github.com/pytorch/pytorch/blob/0bde610c14b92d351b968a0228df29e92442b1cc/torch/CMakeLists.txt
# There are some common files used in both internal lite-interpreter and full-jit. Making a separate
# list for the shared files.

core_sources_common = [
    # This needs to belong here because it defines the first non-inline virtual
    # function, which matters for AutogradMetaInterface's vtable.
    "torch/csrc/autograd/autograd_meta.cpp",
    "torch/csrc/autograd/forward_grad.cpp",
    "torch/csrc/jit/frontend/edit_distance.cpp",
    "torch/csrc/jit/mobile/compatibility/runtime_compatibility.cpp",
    "torch/csrc/jit/mobile/type_parser.cpp",
    "torch/csrc/jit/operator_upgraders/version_map.cpp",
    "torch/csrc/jit/runtime/instruction.cpp",
    "torch/csrc/jit/runtime/jit_exception.cpp",
    "torch/csrc/jit/runtime/operator.cpp",
    "torch/csrc/jit/mobile/register_ops_common_utils.cpp",
    "torch/csrc/jit/runtime/print_handler.cpp",
    "torch/csrc/jit/runtime/slice_indices_adjust.cpp",
    "torch/csrc/jit/runtime/register_ops_utils.cpp",
    "torch/csrc/jit/runtime/vararg_functions.cpp",
    "torch/csrc/jit/mobile/promoted_prim_ops.cpp",
    "torch/csrc/jit/mobile/prim_ops_registery.cpp",
    "torch/csrc/profiler/util.cpp",
]

torch_unpickler_common = [
    "torch/csrc/jit/serialization/import_read.cpp",
    "torch/csrc/jit/serialization/pickler_helper.cpp",
    "torch/csrc/jit/serialization/unpickler.cpp",
]

libtorch_sources_common = sorted(core_sources_common + torch_unpickler_common)

# The profilers are not needed in the lite interpreter build.
libtorch_profiler_sources = [
    "torch/csrc/autograd/profiler_legacy.cpp",
    "torch/csrc/autograd/profiler_kineto.cpp",
    "torch/csrc/profiler/collection.cpp",
    "torch/csrc/profiler/data_flow.cpp",
    "torch/csrc/profiler/kineto_shim.cpp",
    "torch/csrc/mtia/profiler/MTIAMemoryProfiler.cpp",
    "torch/csrc/profiler/kineto_client_interface.cpp",
    "torch/csrc/profiler/orchestration/observer.cpp",
    "torch/csrc/profiler/orchestration/python_tracer.cpp",
    "torch/csrc/profiler/standalone/execution_trace_observer.cpp",
    "torch/csrc/profiler/standalone/itt_observer.cpp",
    "torch/csrc/profiler/standalone/nvtx_observer.cpp",
    "torch/csrc/profiler/standalone/privateuse1_observer.cpp",
    "torch/csrc/profiler/stubs/base.cpp",
    "torch/csrc/profiler/orchestration/vulkan.cpp",
    "torch/csrc/profiler/perf.cpp",
    "torch/csrc/monitor/counters.cpp",
    "torch/csrc/monitor/events.cpp",
]

libtorch_edge_profiler_sources = libtorch_profiler_sources + [
    "torch/csrc/jit/mobile/profiler_edge.cpp",
]

core_trainer_sources = [
    "torch/csrc/autograd/anomaly_mode.cpp",
    "torch/csrc/autograd/autograd.cpp",
    "torch/csrc/autograd/autograd_not_implemented_fallback.cpp",
    "torch/csrc/autograd/cpp_hook.cpp",
    "torch/csrc/autograd/custom_function.cpp",
    "torch/csrc/autograd/variable_info.cpp",
    "torch/csrc/autograd/engine.cpp",
    "torch/csrc/autograd/function.cpp",
    "torch/csrc/autograd/input_metadata.cpp",
    "torch/csrc/autograd/functions/accumulate_grad.cpp",
    "torch/csrc/autograd/functions/basic_ops.cpp",
    "torch/csrc/autograd/functions/tensor.cpp",
    "torch/csrc/autograd/functions/utils.cpp",
    "torch/csrc/autograd/input_buffer.cpp",
    "torch/csrc/autograd/record_function_ops.cpp",
    "torch/csrc/autograd/saved_variable.cpp",
    "torch/csrc/autograd/variable.cpp",
    "torch/csrc/autograd/utils/warnings.cpp",
    "torch/csrc/autograd/jit_decomp_interface.cpp",
    "torch/csrc/dynamo/compiled_autograd.cpp",
    "torch/csrc/jit/frontend/name_mangler.cpp",
    "torch/csrc/jit/ir/type_hashing.cpp",
    "torch/csrc/jit/serialization/pickler.cpp",
    "torch/csrc/jit/serialization/type_name_uniquer.cpp",
]

torch_mobile_core = [
    # backend_debug_info.cpp provides
    # __torch__.torch.classes.backend.BackendDebugInfo class
    # This should not be needed eventually.
    # TODO: Remove this dependency
    "torch/csrc/jit/backends/backend_debug_info.cpp",
    "torch/csrc/jit/mobile/compatibility/model_compatibility.cpp",
    "torch/csrc/jit/mobile/function.cpp",
    "torch/csrc/jit/mobile/import.cpp",
    "torch/csrc/jit/mobile/flatbuffer_loader.cpp",
    "torch/csrc/jit/mobile/interpreter.cpp",
    "torch/csrc/jit/mobile/module.cpp",
    "torch/csrc/jit/mobile/observer.cpp",
    "torch/csrc/jit/mobile/parse_bytecode.cpp",
    "torch/csrc/jit/mobile/parse_operators.cpp",
    "torch/csrc/jit/mobile/quantization.cpp",
    "torch/csrc/jit/mobile/upgrader_mobile.cpp",
    "torch/csrc/jit/runtime/register_prim_ops.cpp",
    "torch/csrc/jit/runtime/register_special_ops.cpp",
]

core_sources_full_mobile_no_backend_interface_xplat = [
    "torch/csrc/jit/api/function_impl.cpp",
    "torch/csrc/jit/api/module.cpp",
    "torch/csrc/jit/api/object.cpp",
    "torch/csrc/jit/backends/backend_debug_handler.cpp",
    "torch/csrc/jit/backends/backend_detail.cpp",
    "torch/csrc/jit/backends/backend_resolver.cpp",
    "torch/csrc/jit/codegen/fuser/codegen.cpp",
    "torch/csrc/jit/codegen/fuser/compiler.cpp",
    "torch/csrc/jit/codegen/fuser/executor.cpp",
    "torch/csrc/jit/codegen/fuser/fallback.cpp",
    "torch/csrc/jit/codegen/fuser/interface.cpp",
    "torch/csrc/jit/codegen/fuser/kernel_cache.cpp",
    "torch/csrc/jit/frontend/builtin_functions.cpp",
    "torch/csrc/jit/frontend/versioned_symbols.cpp",
    "torch/csrc/jit/frontend/canonicalize_modified_loop.cpp",
    "torch/csrc/jit/frontend/convert_to_ssa.cpp",
    "torch/csrc/jit/frontend/exit_transforms.cpp",
    "torch/csrc/jit/frontend/inline_loop_condition.cpp",
    "torch/csrc/jit/frontend/ir_emitter.cpp",
    "torch/csrc/jit/frontend/parser.cpp",
    "torch/csrc/jit/frontend/schema_matching.cpp",
    "torch/csrc/jit/frontend/script_type_parser.cpp",
    "torch/csrc/jit/frontend/sugared_value.cpp",
    "torch/csrc/jit/frontend/tracer.cpp",
    "torch/csrc/jit/ir/alias_analysis.cpp",
    "torch/csrc/jit/ir/attributes.cpp",
    "torch/csrc/jit/ir/constants.cpp",
    "torch/csrc/jit/ir/ir.cpp",
    "torch/csrc/jit/ir/irparser.cpp",
    "torch/csrc/jit/ir/node_hashing.cpp",
    "torch/csrc/jit/ir/scope.cpp",
    "torch/csrc/jit/ir/subgraph_matcher.cpp",
    "torch/csrc/jit/ir/graph_utils.cpp",
    "torch/csrc/jit/jit_log.cpp",
    "torch/csrc/jit/jit_opt_limit.cpp",
    "torch/csrc/jit/mobile/nnc/aot_compiler.cpp",
    "torch/csrc/jit/mobile/nnc/backend.cpp",
    "torch/csrc/jit/mobile/nnc/context.cpp",
    "torch/csrc/jit/mobile/nnc/registry.cpp",
    "torch/csrc/jit/operator_upgraders/utils.cpp",
    "torch/csrc/jit/operator_upgraders/upgraders.cpp",
    "torch/csrc/jit/operator_upgraders/upgraders_entry.cpp",
    "torch/csrc/jit/passes/add_if_then_else.cpp",
    "torch/csrc/jit/passes/annotate_warns.cpp",
    "torch/csrc/jit/passes/bailout_graph.cpp",
    "torch/csrc/jit/passes/check_strict_fusion.cpp",
    "torch/csrc/jit/passes/batch_mm.cpp",
    "torch/csrc/jit/passes/canonicalize.cpp",
    "torch/csrc/jit/passes/canonicalize_graph_fuser_ops.cpp",
    "torch/csrc/jit/passes/clear_profiling.cpp",
    "torch/csrc/jit/passes/clear_undefinedness.cpp",
    "torch/csrc/jit/passes/common_subexpression_elimination.cpp",
    "torch/csrc/jit/passes/concat_opt.cpp",
    "torch/csrc/jit/passes/constant_pooling.cpp",
    "torch/csrc/jit/passes/constant_propagation.cpp",
    "torch/csrc/jit/passes/restore_mutation.cpp",
    "torch/csrc/jit/passes/create_autodiff_subgraphs.cpp",
    "torch/csrc/jit/passes/dead_code_elimination.cpp",
    "torch/csrc/jit/passes/eliminate_no_ops.cpp",
    "torch/csrc/jit/passes/remove_redundant_profiles.cpp",
    "torch/csrc/jit/passes/remove_exceptions.cpp",
    "torch/csrc/jit/passes/decompose_ops.cpp",
    "torch/csrc/jit/passes/dtype_analysis.cpp",
    "torch/csrc/jit/passes/device_type_analysis.cpp",
    "torch/csrc/jit/passes/erase_number_types.cpp",
    "torch/csrc/jit/passes/fixup_trace_scope_blocks.cpp",
    "torch/csrc/jit/passes/freeze_module.cpp",
    "torch/csrc/jit/passes/fuse_linear.cpp",
    "torch/csrc/jit/passes/fuse_relu.cpp",
    "torch/csrc/jit/passes/graph_fuser.cpp",
    "torch/csrc/jit/passes/graph_rewrite_helper.cpp",
    "torch/csrc/jit/passes/guard_elimination.cpp",
    "torch/csrc/jit/passes/hoist_conv_packed_params.cpp",
    "torch/csrc/jit/passes/inline_autodiff_subgraphs.cpp",
    "torch/csrc/jit/passes/inline_forked_closures.cpp",
    "torch/csrc/jit/passes/inline_fork_wait.cpp",
    "torch/csrc/jit/passes/inliner.cpp",
    "torch/csrc/jit/passes/inplace_check.cpp",
    "torch/csrc/jit/passes/insert_guards.cpp",
    "torch/csrc/jit/passes/lift_closures.cpp",
    "torch/csrc/jit/passes/liveness.cpp",
    "torch/csrc/jit/passes/loop_unrolling.cpp",
    "torch/csrc/jit/passes/lower_grad_of.cpp",
    "torch/csrc/jit/passes/lower_tuples.cpp",
    "torch/csrc/jit/passes/normalize_ops.cpp",
    "torch/csrc/jit/passes/peephole_dict_idioms.cpp",
    "torch/csrc/jit/passes/peephole_list_idioms.cpp",
    "torch/csrc/jit/passes/value_refinement_utils.cpp",
    "torch/csrc/jit/passes/peephole_alias_sensitive.cpp",
    "torch/csrc/jit/passes/pass_manager.cpp",
    "torch/csrc/jit/passes/peephole.cpp",
    "torch/csrc/jit/passes/peephole_non_tensor.cpp",
    "torch/csrc/jit/passes/create_functional_graphs.cpp",
    "torch/csrc/jit/passes/refine_tuple_types.cpp",
    "torch/csrc/jit/passes/remove_mutation.cpp",
    "torch/csrc/jit/passes/prepack_folding.cpp",
    "torch/csrc/jit/passes/fold_conv_bn.cpp",
    "torch/csrc/jit/passes/fold_linear_bn.cpp",
    "torch/csrc/jit/passes/dbr_quantization/remove_redundant_aliases.cpp",
    "torch/csrc/jit/passes/frozen_concat_linear.cpp",
    "torch/csrc/jit/passes/frozen_conv_add_relu_fusion.cpp",
    "torch/csrc/jit/passes/frozen_conv_folding.cpp",
    "torch/csrc/jit/passes/frozen_linear_folding.cpp",
    "torch/csrc/jit/passes/frozen_linear_transpose.cpp",
    "torch/csrc/jit/passes/frozen_ops_to_mkldnn.cpp",
    "torch/csrc/jit/passes/frozen_graph_optimizations.cpp",
    "torch/csrc/jit/passes/remove_expands.cpp",
    "torch/csrc/jit/passes/remove_dropout.cpp",
    "torch/csrc/jit/passes/requires_grad_analysis.cpp",
    "torch/csrc/jit/passes/shape_analysis.cpp",
    "torch/csrc/jit/passes/integer_value_refinement.cpp",
    "torch/csrc/jit/passes/replacement_of_old_operators.cpp",
    "torch/csrc/jit/passes/symbolic_shape_analysis.cpp",
    "torch/csrc/jit/passes/symbolic_shape_cache.cpp",
    "torch/csrc/jit/passes/symbolic_shape_runtime_fusion.cpp",
    "torch/csrc/jit/passes/specialize_autogradzero.cpp",
    "torch/csrc/jit/passes/update_differentiable_graph_requires_grad.cpp",
    "torch/csrc/jit/passes/variadic_ops.cpp",
    "torch/csrc/jit/passes/subgraph_rewrite.cpp",
    "torch/csrc/jit/passes/tensorexpr_fuser.cpp",
    "torch/csrc/jit/passes/utils/memory_dag.cpp",
    "torch/csrc/jit/passes/utils/subgraph_utils.cpp",
    "torch/csrc/jit/passes/utils/optimization_utils.cpp",
    "torch/csrc/jit/passes/utils/op_registry.cpp",
    "torch/csrc/jit/passes/mkldnn_rewrite.cpp",
    "torch/csrc/jit/passes/xnnpack_rewrite.cpp",
    "torch/csrc/jit/passes/vulkan_rewrite.cpp",
    "torch/csrc/jit/passes/metal_rewrite.cpp",
    "torch/csrc/jit/passes/quantization/helper.cpp",
    "torch/csrc/jit/passes/quantization/quantization_type.cpp",
    "torch/csrc/jit/passes/quantization/insert_observers.cpp",
    "torch/csrc/jit/passes/quantization/insert_quant_dequant.cpp",
    "torch/csrc/jit/passes/quantization/dedup_module_uses.cpp",
    "torch/csrc/jit/passes/quantization/finalize.cpp",
    "torch/csrc/jit/passes/quantization/fusion_passes.cpp",
    "torch/csrc/jit/passes/quantization/register_packed_params.cpp",
    "torch/csrc/jit/python/update_graph_executor_opt.cpp",
    "torch/csrc/jit/python/utf8_decoding_ignore.cpp",
    "torch/csrc/jit/runtime/argument_spec.cpp",
    "torch/csrc/jit/runtime/autodiff.cpp",
    "torch/csrc/jit/runtime/graph_executor.cpp",
    "torch/csrc/jit/runtime/interpreter/frame.cpp",
    "torch/csrc/jit/runtime/interpreter/preprocess_graph.cpp",
    "torch/csrc/jit/runtime/interpreter.cpp",
    "torch/csrc/jit/runtime/logging.cpp",
    "torch/csrc/jit/runtime/simple_graph_executor_impl.cpp",
    "torch/csrc/jit/runtime/profiling_graph_executor_impl.cpp",
    "torch/csrc/jit/runtime/profiling_record.cpp",
    "torch/csrc/jit/runtime/script_profile.cpp",
    "torch/csrc/jit/runtime/symbolic_script.cpp",
    "torch/csrc/jit/runtime/symbolic_shape_registry.cpp",
    "torch/csrc/jit/runtime/decomposition_registry.cpp",
    "torch/csrc/jit/runtime/decomposition_registry_util.cpp",
    "torch/csrc/jit/runtime/serialized_shape_function_registry.cpp",
    "torch/csrc/jit/runtime/symbolic_shape_registry_util.cpp",
    "torch/csrc/jit/runtime/jit_trace.cpp",
    "torch/csrc/jit/serialization/callstack_debug_info_serialization.cpp",
    "torch/csrc/jit/serialization/import.cpp",
    "torch/csrc/jit/serialization/import_export_helpers.cpp",
    "torch/csrc/jit/serialization/import_source.cpp",
    "torch/csrc/jit/serialization/pickle.cpp",
    "torch/csrc/jit/serialization/python_print.cpp",
    "torch/csrc/jit/serialization/source_range_serialization.cpp",
    "torch/csrc/jit/tensorexpr/block_codegen.cpp",
    "torch/csrc/jit/tensorexpr/bounds_inference.cpp",
    "torch/csrc/jit/tensorexpr/bounds_overlap.cpp",
    "torch/csrc/jit/tensorexpr/codegen.cpp",
    "torch/csrc/jit/tensorexpr/cpp_codegen.cpp",
    "torch/csrc/jit/tensorexpr/eval.cpp",
    "torch/csrc/jit/tensorexpr/expr.cpp",
    "torch/csrc/jit/tensorexpr/external_functions_core.cpp",
    "torch/csrc/jit/tensorexpr/external_functions_registry.cpp",
    "torch/csrc/jit/tensorexpr/graph_opt.cpp",
    "torch/csrc/jit/tensorexpr/hash_provider.cpp",
    "torch/csrc/jit/tensorexpr/intrinsic_symbols.cpp",
    "torch/csrc/jit/tensorexpr/ir.cpp",
    "torch/csrc/jit/tensorexpr/ir_cloner.cpp",
    "torch/csrc/jit/tensorexpr/ir_mutator.cpp",
    "torch/csrc/jit/tensorexpr/ir_printer.cpp",
    "torch/csrc/jit/tensorexpr/ir_simplifier.cpp",
    "torch/csrc/jit/tensorexpr/ir_verifier.cpp",
    "torch/csrc/jit/tensorexpr/ir_visitor.cpp",
    "torch/csrc/jit/tensorexpr/kernel.cpp",
    "torch/csrc/jit/tensorexpr/llvm_codegen.cpp",
    "torch/csrc/jit/tensorexpr/llvm_jit.cpp",
    "torch/csrc/jit/tensorexpr/loopnest.cpp",
    "torch/csrc/jit/tensorexpr/loopnest_randomization.cpp",
    "torch/csrc/jit/tensorexpr/lowerings.cpp",
    "torch/csrc/jit/tensorexpr/mem_dependency_checker.cpp",
    "torch/csrc/jit/tensorexpr/operators/conv2d.cpp",
    "torch/csrc/jit/tensorexpr/operators/matmul.cpp",
    "torch/csrc/jit/tensorexpr/operators/misc.cpp",
    "torch/csrc/jit/tensorexpr/operators/norm.cpp",
    "torch/csrc/jit/tensorexpr/operators/pointwise.cpp",
    "torch/csrc/jit/tensorexpr/operators/quantization.cpp",
    "torch/csrc/jit/tensorexpr/operators/reduction.cpp",
    "torch/csrc/jit/tensorexpr/operators/softmax.cpp",
    "torch/csrc/jit/tensorexpr/reduction.cpp",
    "torch/csrc/jit/tensorexpr/registerizer.cpp",
    "torch/csrc/jit/tensorexpr/tensor.cpp",
    "torch/csrc/jit/tensorexpr/types.cpp",
    "torch/csrc/jit/tensorexpr/unique_name_manager.cpp",
    "torch/csrc/jit/testing/file_check.cpp",
    "torch/csrc/profiler/unwind/unwind.cpp",
    "torch/csrc/profiler/unwind/unwind_fb.cpp",
    "torch/csrc/profiler/combined_traceback.cpp",
    "torch/csrc/jit/testing/hooks_for_testing.cpp",
    "torch/csrc/utils/cpp_stacktraces.cpp",
    "torch/csrc/utils/schema_info.cpp",
    "torch/csrc/utils/tensor_flatten.cpp",
    "torch/csrc/utils/variadic.cpp",
]

core_sources_full_mobile_no_backend_interface = core_sources_full_mobile_no_backend_interface_xplat + [
    # backend_debug_info.cpp provides
    # __torch__.torch.classes.backend.BackendDebugInfo class
    # This should not be needed eventually.
    # TODO: Remove this dependency
    "torch/csrc/jit/backends/backend_debug_info.cpp",
    "torch/csrc/jit/mobile/compatibility/model_compatibility.cpp",
    "torch/csrc/jit/mobile/function.cpp",
    "torch/csrc/jit/mobile/import.cpp",
    "torch/csrc/jit/mobile/flatbuffer_loader.cpp",
    "torch/csrc/jit/mobile/interpreter.cpp",
    "torch/csrc/jit/mobile/module.cpp",
    "torch/csrc/jit/mobile/observer.cpp",
    "torch/csrc/jit/mobile/parse_bytecode.cpp",
    "torch/csrc/jit/mobile/parse_operators.cpp",
    "torch/csrc/jit/mobile/quantization.cpp",
    "torch/csrc/jit/mobile/upgrader_mobile.cpp",
]

core_sources_full_mobile = core_sources_full_mobile_no_backend_interface + [
    "torch/csrc/jit/backends/backend_debug_info.cpp",
    "torch/csrc/jit/backends/backend_interface.cpp",
]

core_sources_full = core_sources_full_mobile + [
    "torch/csrc/jit/runtime/static/fusion.cpp",
    "torch/csrc/jit/runtime/static/generated_ops.cpp",
    "torch/csrc/jit/runtime/static/impl.cpp",
    "torch/csrc/jit/runtime/static/memory_planner.cpp",
    "torch/csrc/jit/runtime/static/native_ops.cpp",
    "torch/csrc/jit/runtime/static/ops.cpp",
    "torch/csrc/jit/runtime/static/passes.cpp",
    "torch/csrc/jit/runtime/static/te_wrapper.cpp",
    "torch/csrc/jit/tensorexpr/external_functions.cpp",
    "torch/csrc/jit/tensorexpr/external_functions_codegen.cpp",
]

lazy_tensor_core_sources = [
    "torch/csrc/lazy/backend/backend_device.cpp",
    "torch/csrc/lazy/backend/backend_interface.cpp",
    "torch/csrc/lazy/backend/lowering_context.cpp",
    "torch/csrc/lazy/core/config.cpp",
    "torch/csrc/lazy/core/debug_util.cpp",
    "torch/csrc/lazy/core/hash.cpp",
    "torch/csrc/lazy/core/helpers.cpp",
    "torch/csrc/lazy/core/ir.cpp",
    "torch/csrc/lazy/core/ir_dump_util.cpp",
    "torch/csrc/lazy/core/ir_metadata.cpp",
    "torch/csrc/lazy/core/ir_util.cpp",
    "torch/csrc/lazy/core/lazy_graph_executor.cpp",
    "torch/csrc/lazy/core/metrics.cpp",
    "torch/csrc/lazy/core/multi_wait.cpp",
    "torch/csrc/lazy/core/ops/arithmetic_ir_ops.cpp",
    "torch/csrc/lazy/core/ops/utils.cpp",
    "torch/csrc/lazy/core/permutation_util.cpp",
    "torch/csrc/lazy/core/shape.cpp",
    "torch/csrc/lazy/core/shape_inference.cpp",
    "torch/csrc/lazy/core/tensor.cpp",
    "torch/csrc/lazy/core/tensor_impl.cpp",
    "torch/csrc/lazy/core/tensor_util.cpp",
    "torch/csrc/lazy/core/thread_pool.cpp",
    "torch/csrc/lazy/core/trie.cpp",
]

# We can't build all of the ts backend under certain build configurations, e.g. mobile,
# since it depends on things like autograd, meta functions, which may be disabled
lazy_tensor_ts_sources = [
    "torch/csrc/lazy/ts_backend/dynamic_ir.cpp",
    "torch/csrc/lazy/ts_backend/config.cpp",
    "torch/csrc/lazy/ts_backend/ops/device_data.cpp",
    "torch/csrc/lazy/ts_backend/ops/generic.cpp",
    "torch/csrc/lazy/ts_backend/tensor_aten_ops.cpp",
    "torch/csrc/lazy/ts_backend/ts_autograd_functions.cpp",
    "torch/csrc/lazy/ts_backend/ts_backend_impl.cpp",
    "torch/csrc/lazy/ts_backend/ts_eager_fallback.cpp",
    "torch/csrc/lazy/ts_backend/ts_lowering_context.cpp",
    "torch/csrc/lazy/ts_backend/ts_native_functions.cpp",
    "torch/csrc/lazy/ts_backend/ts_node.cpp",
    "torch/csrc/lazy/ts_backend/ts_node_lowering.cpp",
]

lazy_tensor_core_python_sources = [
    "torch/csrc/lazy/python/init.cpp",
    "torch/csrc/lazy/python/python_util.cpp",
]

inductor_core_resources = [
    "torch/csrc/inductor/aoti_package/model_package_loader.cpp",
    "torch/csrc/inductor/aoti_runner/model_container_runner.cpp",
    "torch/csrc/inductor/aoti_runner/model_container_runner_cpu.cpp",
    "torch/csrc/inductor/aoti_torch/shim_common.cpp",
    "torch/csrc/inductor/aoti_torch/shim_cpu.cpp",
    "torch/csrc/inductor/aoti_torch/tensor_converter.cpp",
    "torch/csrc/inductor/aoti_torch/mkldnn_tensor.cpp",
    "torch/csrc/inductor/aoti_torch/oss_proxy_executor.cpp",
    "torch/csrc/inductor/inductor_ops.cpp",
    "torch/csrc/jit/serialization/pickle.cpp",
    "torch/csrc/shim_common.cpp",
]

libtorch_core_sources = sorted(
    core_sources_common +
    torch_unpickler_common +
    core_sources_full +
    core_trainer_sources +
    inductor_core_resources +
    libtorch_profiler_sources +
    lazy_tensor_core_sources,
)

# These files are the only ones that are supported on Windows.
libtorch_distributed_base_sources = [
    "torch/csrc/distributed/c10d/Backoff.cpp",
    "torch/csrc/distributed/c10d/Backend.cpp",
    "torch/csrc/distributed/c10d/FileStore.cpp",
    "torch/csrc/distributed/c10d/FlightRecorder.cpp",
    "torch/csrc/distributed/c10d/Functional.cpp",
    "torch/csrc/distributed/c10d/GlooDeviceFactory.cpp",
    "torch/csrc/distributed/c10d/GroupRegistry.cpp",
    "torch/csrc/distributed/c10d/Ops.cpp",
    "torch/csrc/distributed/c10d/ParamCommsUtils.cpp",
    "torch/csrc/distributed/c10d/PrefixStore.cpp",
    "torch/csrc/distributed/c10d/ProcessGroup.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupGloo.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupMPI.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupWrapper.cpp",
    "torch/csrc/distributed/c10d/Store.cpp",
    "torch/csrc/distributed/c10d/TCPStore.cpp",
    "torch/csrc/distributed/c10d/TCPStoreBackend.cpp",
    "torch/csrc/distributed/c10d/TCPStoreLibUvBackend.cpp",
    "torch/csrc/distributed/c10d/Types.cpp",
    "torch/csrc/distributed/c10d/Utils.cpp",
    "torch/csrc/distributed/c10d/Work.cpp",
    "torch/csrc/distributed/c10d/comm.cpp",
    "torch/csrc/distributed/c10d/control_collectives/StoreCollectives.cpp",
    "torch/csrc/distributed/c10d/control_plane/Handlers.cpp",
    "torch/csrc/distributed/c10d/control_plane/WorkerServer.cpp",
    "torch/csrc/distributed/c10d/cuda/StreamBlock.cpp",
    "torch/csrc/distributed/c10d/debug.cpp",
    "torch/csrc/distributed/c10d/default_comm_hooks.cpp",
    "torch/csrc/distributed/c10d/logger.cpp",
    "torch/csrc/distributed/c10d/logging.cpp",
    "torch/csrc/distributed/c10d/quantization/quantization.cpp",
    "torch/csrc/distributed/c10d/reducer.cpp",
    "torch/csrc/distributed/c10d/sequence_num.cpp",
    "torch/csrc/distributed/c10d/socket.cpp",
    "torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.cpp",
    "torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp",
]

# These files are only supported on Linux (and others) but not on Windows.
libtorch_distributed_extra_sources = [
    "torch/csrc/distributed/autograd/autograd.cpp",
    "torch/csrc/distributed/autograd/utils.cpp",
    "torch/csrc/distributed/autograd/context/container.cpp",
    "torch/csrc/distributed/autograd/context/context.cpp",
    "torch/csrc/distributed/autograd/engine/dist_engine.cpp",
    "torch/csrc/distributed/autograd/functions/recvrpc_backward.cpp",
    "torch/csrc/distributed/autograd/functions/sendrpc_backward.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.cpp",
    "torch/csrc/distributed/autograd/rpc_messages/rref_backward_resp.cpp",
    "torch/csrc/distributed/c10d/HashStore.cpp",
    "torch/csrc/distributed/rpc/agent_utils.cpp",
    "torch/csrc/distributed/rpc/message.cpp",
    "torch/csrc/distributed/rpc/profiler/remote_profiler_manager.cpp",
    "torch/csrc/distributed/rpc/profiler/server_process_global_profiler.cpp",
    "torch/csrc/distributed/rpc/python_call.cpp",
    "torch/csrc/distributed/rpc/python_remote_call.cpp",
    "torch/csrc/distributed/rpc/python_resp.cpp",
    "torch/csrc/distributed/rpc/request_callback.cpp",
    "torch/csrc/distributed/rpc/request_callback_no_python.cpp",
    "torch/csrc/distributed/rpc/rpc_agent.cpp",
    "torch/csrc/distributed/rpc/rref_context.cpp",
    "torch/csrc/distributed/rpc/rref_impl.cpp",
    "torch/csrc/distributed/rpc/rref_proto.cpp",
    "torch/csrc/distributed/rpc/script_call.cpp",
    "torch/csrc/distributed/rpc/script_remote_call.cpp",
    "torch/csrc/distributed/rpc/script_resp.cpp",
    "torch/csrc/distributed/rpc/tensorpipe_agent.cpp",
    "torch/csrc/distributed/rpc/tensorpipe_utils.cpp",
    "torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.cpp",
    "torch/csrc/distributed/rpc/torchscript_functions.cpp",
    "torch/csrc/distributed/rpc/types.cpp",
    "torch/csrc/distributed/rpc/utils.cpp",
]

libtorch_distributed_sources = libtorch_distributed_base_sources + libtorch_distributed_extra_sources

jit_sources_full = [
    "torch/csrc/jit/codegen/cuda/interface.cpp",
    "torch/csrc/jit/passes/lower_graph.cpp",
    "torch/csrc/jit/runtime/register_c10_ops.cpp",
    "torch/csrc/jit/runtime/register_prim_ops.cpp",
    "torch/csrc/jit/runtime/register_prim_ops_fulljit.cpp",
    "torch/csrc/jit/runtime/register_special_ops.cpp",
    "torch/csrc/jit/passes/remove_inplace_ops.cpp",
    "torch/csrc/jit/passes/utils/check_alias_annotation.cpp",
    "torch/csrc/jit/passes/autocast.cpp",
]

libtorch_core_jit_sources = sorted(jit_sources_full)


libtorch_nativert_sources = [
    "torch/nativert/ModelRunner.cpp",
    "torch/nativert/graph/Graph.cpp",
    "torch/nativert/graph/GraphPasses.cpp",
    "torch/nativert/graph/GraphSignature.cpp",
    "torch/nativert/graph/Serialization.cpp",
    "torch/nativert/graph/TensorMeta.cpp",
    "torch/nativert/graph/GraphUtils.cpp",
    "torch/nativert/executor/DelegateExecutor.cpp",
    "torch/nativert/executor/Placement.cpp",
    "torch/nativert/executor/ExecutionPlanner.cpp",
    "torch/nativert/executor/ExecutionFrame.cpp",
    "torch/nativert/executor/Executor.cpp",
    "torch/nativert/executor/GraphExecutorBase.cpp",
    "torch/nativert/executor/ConstantFolder.cpp",
    "torch/nativert/executor/OpKernel.cpp",
    "torch/nativert/executor/PlacementUtils.cpp",
    "torch/nativert/executor/SerialGraphExecutor.cpp",
    "torch/nativert/executor/Weights.cpp",
    "torch/nativert/executor/memory/FunctionSchema.cpp",
    "torch/nativert/common/FileUtil.cpp",
    "torch/nativert/detail/ITree.cpp",
    "torch/nativert/kernels/C10Kernel.cpp",
    "torch/nativert/kernels/AutoFunctionalizeKernel.cpp",
    "torch/nativert/kernels/HigherOrderKernel.cpp",
    "torch/nativert/executor/memory/GreedyBySize.cpp",
    "torch/nativert/executor/memory/Bump.cpp",
    "torch/nativert/executor/ParallelGraphExecutor.cpp",
    "torch/nativert/kernels/CallTorchBindKernel.cpp",
    "torch/nativert/kernels/KernelFactory.cpp",
    "torch/nativert/kernels/PrimKernelRegistry.cpp",
    "torch/nativert/executor/memory/DisjointStorageGroups.cpp",
    "torch/nativert/executor/memory/AliasAnalyzer.cpp",
    "torch/nativert/executor/memory/LayoutPlanner.cpp",
    "torch/nativert/executor/memory/LayoutManager.cpp",
    "torch/nativert/kernels/KernelRegistry.cpp",
    "torch/nativert/kernels/NativeKernels.cpp",
    "torch/nativert/kernels/GeneratedStaticDispatchKernels.cpp",
    "torch/nativert/kernels/GeneratedNativeStaticDispatchKernels.cpp",
    "torch/nativert/graph/passes/SubgraphRewriter.cpp",
    "torch/nativert/graph/passes/pass_manager/GraphPasses.cpp",
    "torch/nativert/graph/passes/pass_manager/PassManager.cpp",
    "torch/nativert/kernels/KernelHandlerRegistry.cpp",
    "torch/nativert/kernels/TritonKernel.cpp",
    "torch/nativert/executor/triton/CpuTritonKernelManager.cpp",
    "torch/nativert/executor/AOTInductorDelegateExecutor.cpp",
    "torch/nativert/kernels/ETCallDelegateKernel.cpp",
]

libtorch_nativert_cuda_sources = [
    "torch/nativert/executor/triton/CudaTritonKernelManager.cpp",
    "torch/nativert/executor/AOTInductorModelContainerCudaShim.cpp",
]

torch_mobile_tracer_sources = [
    "torch/csrc/jit/mobile/model_tracer/tracer.cpp",
    "torch/csrc/jit/mobile/model_tracer/TensorUtils.cpp",
    "torch/csrc/jit/mobile/model_tracer/TracerRunner.cpp",
    "torch/csrc/jit/mobile/model_tracer/MobileModelRunner.cpp",
    "torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.cpp",
    "torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.cpp",
    "torch/csrc/jit/mobile/model_tracer/CustomClassTracer.cpp",
    "torch/csrc/jit/mobile/model_tracer/BuildFeatureTracer.cpp",
]

libtorch_lite_eager_symbolication = [
    "torch/csrc/jit/frontend/source_range.cpp",
    "torch/csrc/jit/ir/scope.cpp",
    "torch/csrc/jit/mobile/debug_info.cpp",
    "torch/csrc/jit/serialization/callstack_debug_info_serialization.cpp",
    "torch/csrc/jit/serialization/source_range_serialization.cpp",
    # Later we can split serialization and deserialization logic
    # to have better separation within build and only build relevant parts.
    "torch/csrc/jit/serialization/pickle.cpp",
    "torch/csrc/jit/serialization/pickler_helper.cpp",
    "torch/csrc/jit/serialization/pickler.cpp",
    "torch/csrc/jit/serialization/unpickler.cpp",
]

# TODO: core_trainer_sources is not necessary for libtorch lite
libtorch_lite_cmake_sources = sorted(
    core_trainer_sources +
    core_sources_common +
    torch_unpickler_common +
    torch_mobile_core,
)

libtorch_cmake_sources = libtorch_core_sources + libtorch_core_jit_sources + libtorch_nativert_sources

libtorch_extra_sources = libtorch_core_jit_sources + [
    "torch/csrc/autograd/TraceTypeManual.cpp",
    "torch/csrc/autograd/VariableTypeManual.cpp",
    "torch/csrc/autograd/FunctionsManual.cpp",
    "torch/csrc/jit/api/module_save.cpp",
    "torch/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp",
    "torch/csrc/jit/mobile/compatibility/backport.cpp",
    "torch/csrc/jit/mobile/compatibility/backport_manager.cpp",
    "torch/csrc/jit/mobile/compatibility/model_compatibility.cpp",
    # To be included for eager symbolication in lite interpreter
    # when it is built in libtorch
    "torch/csrc/jit/mobile/debug_info.cpp",
    "torch/csrc/jit/mobile/function.cpp",
    "torch/csrc/jit/mobile/flatbuffer_loader.cpp",
    "torch/csrc/jit/mobile/import.cpp",
    "torch/csrc/jit/mobile/import_data.cpp",
    "torch/csrc/jit/mobile/interpreter.cpp",
    "torch/csrc/jit/mobile/module.cpp",
    "torch/csrc/jit/mobile/observer.cpp",
    "torch/csrc/jit/mobile/parse_bytecode.cpp",
    "torch/csrc/jit/mobile/parse_operators.cpp",
    "torch/csrc/jit/mobile/quantization.cpp",
    "torch/csrc/jit/mobile/train/export_data.cpp",
    "torch/csrc/jit/mobile/train/optim/sgd.cpp",
    "torch/csrc/jit/mobile/train/random.cpp",
    "torch/csrc/jit/mobile/train/sequential.cpp",
    "torch/csrc/jit/mobile/upgrader_mobile.cpp",
    "torch/csrc/jit/serialization/onnx.cpp",
    "torch/csrc/jit/serialization/export.cpp",
    "torch/csrc/jit/serialization/export_bytecode.cpp",
    "torch/csrc/jit/serialization/export_module.cpp",
    "torch/csrc/jit/serialization/flatbuffer_serializer.cpp",
    "torch/csrc/utils/byte_order.cpp",
    "torch/csrc/utils/out_types.cpp",
]

def libtorch_sources(gencode_pattern = ":generate-code[{}]"):
    return (
        libtorch_generated_sources(gencode_pattern) + libtorch_core_sources + libtorch_distributed_sources + libtorch_extra_sources + libtorch_nativert_sources
    )

libtorch_cuda_core_sources = [
    "torch/csrc/CudaIPCTypes.cpp",
    "torch/csrc/cuda/comm.cpp",
    "torch/csrc/cuda/memory_snapshot.cpp",
    "torch/csrc/cuda/CUDAPluggableAllocator.cpp",
    "torch/csrc/inductor/aoti_runner/model_container_runner_cuda.cpp",
    "torch/csrc/inductor/aoti_torch/shim_cuda.cpp",
    "torch/csrc/jit/codegen/fuser/cuda/fused_kernel.cpp",
    "torch/csrc/profiler/stubs/cuda.cpp",
    "torch/csrc/autograd/functions/comm.cpp",
    "torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp",
    "torch/csrc/jit/tensorexpr/cuda_codegen.cpp",
    "torch/csrc/jit/runtime/register_cuda_ops.cpp",
]

# These files are the only ones that are supported on Windows.
libtorch_cuda_distributed_base_sources = [
    "torch/csrc/distributed/c10d/reducer_cuda.cpp",
]

# These files are only supported on Linux (and others) but not on Windows.
libtorch_cuda_distributed_extra_sources = [
    "torch/csrc/distributed/c10d/FlightRecorderCuda.cpp",
    "torch/csrc/distributed/c10d/NCCLUtils.cpp",
    "torch/csrc/distributed/c10d/NanCheck.cu",
    "torch/csrc/distributed/c10d/ProcessGroupGlooCuda.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupUCC.cpp",
    "torch/csrc/distributed/c10d/UCCTracing.cpp",
    "torch/csrc/distributed/c10d/UCCUtils.cpp",
    "torch/csrc/distributed/c10d/cuda/AsyncMM.cu",
    "torch/csrc/distributed/c10d/cuda/CUDAEventCache.cpp",
    "torch/csrc/distributed/c10d/cuda/utils.cpp",
    "torch/csrc/distributed/c10d/cuda/StreamBlock.cu",
    "torch/csrc/distributed/c10d/quantization/quantization_gpu.cu",
    "torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu",
    "torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu",
    "torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp",
    "torch/csrc/distributed/c10d/symm_mem/CudaDMAConnectivity.cpp",
    "torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu",
    "torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp",
    "torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cu",
    "torch/csrc/distributed/c10d/symm_mem/cuda_mem_pool.cpp",
    "torch/csrc/distributed/rpc/tensorpipe_cuda.cpp",
]

libtorch_nvshmem_sources = [
    "torch/csrc/distributed/c10d/cuda/utils.cpp",
    "torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp",
    "torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cu",
    "torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cu",
]

libtorch_cuda_distributed_sources = libtorch_cuda_distributed_base_sources + libtorch_cuda_distributed_extra_sources

libtorch_cuda_sources = libtorch_cuda_core_sources + libtorch_cuda_distributed_sources + [
    "torch/csrc/cuda/nccl.cpp",
] + libtorch_nativert_cuda_sources

torch_cpp_srcs = [
    "torch/csrc/api/src/cuda.cpp",  # this just forwards stuff, no real CUDA
    "torch/csrc/api/src/data/datasets/mnist.cpp",
    "torch/csrc/api/src/data/samplers/distributed.cpp",
    "torch/csrc/api/src/data/samplers/random.cpp",
    "torch/csrc/api/src/data/samplers/sequential.cpp",
    "torch/csrc/api/src/data/samplers/stream.cpp",
    "torch/csrc/api/src/enum.cpp",
    "torch/csrc/api/src/imethod.cpp",
    "torch/csrc/api/src/jit.cpp",
    "torch/csrc/api/src/mps.cpp",
    "torch/csrc/api/src/serialize.cpp",
    "torch/csrc/api/src/nn/init.cpp",
    "torch/csrc/api/src/nn/module.cpp",
    "torch/csrc/api/src/nn/modules/_functions.cpp",
    "torch/csrc/api/src/nn/modules/activation.cpp",
    "torch/csrc/api/src/nn/modules/adaptive.cpp",
    "torch/csrc/api/src/nn/modules/batchnorm.cpp",
    "torch/csrc/api/src/nn/modules/normalization.cpp",
    "torch/csrc/api/src/nn/modules/instancenorm.cpp",
    "torch/csrc/api/src/nn/modules/conv.cpp",
    "torch/csrc/api/src/nn/modules/dropout.cpp",
    "torch/csrc/api/src/nn/modules/distance.cpp",
    "torch/csrc/api/src/nn/modules/embedding.cpp",
    "torch/csrc/api/src/nn/modules/fold.cpp",
    "torch/csrc/api/src/nn/modules/linear.cpp",
    "torch/csrc/api/src/nn/modules/loss.cpp",
    "torch/csrc/api/src/nn/modules/padding.cpp",
    "torch/csrc/api/src/nn/modules/pixelshuffle.cpp",
    "torch/csrc/api/src/nn/modules/pooling.cpp",
    "torch/csrc/api/src/nn/modules/rnn.cpp",
    "torch/csrc/api/src/nn/modules/upsampling.cpp",
    "torch/csrc/api/src/nn/modules/transformer.cpp",
    "torch/csrc/api/src/nn/modules/container/functional.cpp",
    "torch/csrc/api/src/nn/options/activation.cpp",
    "torch/csrc/api/src/nn/options/adaptive.cpp",
    "torch/csrc/api/src/nn/options/batchnorm.cpp",
    "torch/csrc/api/src/nn/options/conv.cpp",
    "torch/csrc/api/src/nn/options/dropout.cpp",
    "torch/csrc/api/src/nn/options/instancenorm.cpp",
    "torch/csrc/api/src/nn/options/linear.cpp",
    "torch/csrc/api/src/nn/options/normalization.cpp",
    "torch/csrc/api/src/nn/options/embedding.cpp",
    "torch/csrc/api/src/nn/options/padding.cpp",
    "torch/csrc/api/src/nn/options/pooling.cpp",
    "torch/csrc/api/src/nn/options/rnn.cpp",
    "torch/csrc/api/src/nn/options/vision.cpp",
    "torch/csrc/api/src/nn/options/transformer.cpp",
    "torch/csrc/api/src/optim/adagrad.cpp",
    "torch/csrc/api/src/optim/adam.cpp",
    "torch/csrc/api/src/optim/adamw.cpp",
    "torch/csrc/api/src/optim/lbfgs.cpp",
    "torch/csrc/api/src/optim/optimizer.cpp",
    "torch/csrc/api/src/optim/rmsprop.cpp",
    "torch/csrc/api/src/optim/serialize.cpp",
    "torch/csrc/api/src/optim/sgd.cpp",
    "torch/csrc/api/src/optim/schedulers/lr_scheduler.cpp",
    "torch/csrc/api/src/optim/schedulers/reduce_on_plateau_scheduler.cpp",
    "torch/csrc/api/src/optim/schedulers/step_lr.cpp",
    "torch/csrc/api/src/serialize/input-archive.cpp",
    "torch/csrc/api/src/serialize/output-archive.cpp",
    "torch/csrc/api/src/xpu.cpp",
]

libtorch_python_cuda_core_sources = [
    "torch/csrc/cuda/Event.cpp",
    "torch/csrc/cuda/Module.cpp",
    "torch/csrc/cuda/python_comm.cpp",
    "torch/csrc/cuda/Stream.cpp",
    "torch/csrc/cuda/Graph.cpp",
    "torch/csrc/cuda/MemPool.cpp",
    "torch/csrc/cuda/GreenContext.cpp",
    "torch/csrc/cuda/shared/cudart.cpp",
    "torch/csrc/cuda/shared/nvtx.cpp",
    "torch/csrc/cuda/utils.cpp",
    "torch/csrc/cuda/GdsFile.cpp",
]

libtorch_python_cuda_sources = libtorch_python_cuda_core_sources + [
    "torch/csrc/cuda/python_nccl.cpp",
    "torch/csrc/cuda/shared/cudnn.cpp",
    "torch/csrc/cuda/shared/cusparselt.cpp",
]

libtorch_python_xpu_sources = [
    "torch/csrc/xpu/Event.cpp",
    "torch/csrc/xpu/Module.cpp",
    "torch/csrc/xpu/Stream.cpp",
    "torch/csrc/inductor/aoti_runner/model_container_runner_xpu.cpp",
    "torch/csrc/inductor/aoti_torch/shim_xpu.cpp",
]

libtorch_xpu_sources = libtorch_python_xpu_sources

libtorch_python_core_sources = [
    "torch/csrc/DataLoader.cpp",
    "torch/csrc/DeviceAccelerator.cpp",
    "torch/csrc/Device.cpp",
    "torch/csrc/Dtype.cpp",
    "torch/csrc/DynamicTypes.cpp",
    "torch/csrc/Exceptions.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Layout.cpp",
    "torch/csrc/MemoryFormat.cpp",
    "torch/csrc/QScheme.cpp",
    "torch/csrc/Module.cpp",
    "torch/csrc/PyInterpreter.cpp",
    "torch/csrc/PyInterpreterHooks.cpp",
    "torch/csrc/python_dimname.cpp",
    "torch/csrc/Size.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/StorageMethods.cpp",
    "torch/csrc/StorageSharing.cpp",
    "torch/csrc/Stream.cpp",
    "torch/csrc/Event.cpp",
    "torch/csrc/TypeInfo.cpp",
    "torch/csrc/acc/Module.cpp",
    "torch/csrc/api/src/python/init.cpp",
    "torch/csrc/autograd/functions/init.cpp",
    "torch/csrc/autograd/init.cpp",
    "torch/csrc/autograd/profiler_python.cpp",
    "torch/csrc/autograd/python_anomaly_mode.cpp",
    "torch/csrc/autograd/python_saved_variable_hooks.cpp",
    "torch/csrc/autograd/python_cpp_function.cpp",
    "torch/csrc/autograd/python_engine.cpp",
    "torch/csrc/autograd/python_function.cpp",
    "torch/csrc/autograd/python_hook.cpp",
    "torch/csrc/autograd/python_legacy_variable.cpp",
    "torch/csrc/autograd/python_nested_functions_manual.cpp",
    "torch/csrc/autograd/python_torch_functions_manual.cpp",
    "torch/csrc/autograd/python_variable.cpp",
    "torch/csrc/autograd/python_variable_indexing.cpp",
    "torch/csrc/distributed/python_placement.cpp",
    "torch/csrc/dynamo/python_compiled_autograd.cpp",
    "torch/csrc/dynamo/cache_entry.cpp",
    "torch/csrc/dynamo/cpp_shim.cpp",
    "torch/csrc/dynamo/cpython_defs.c",
    "torch/csrc/dynamo/eval_frame.c",
    "torch/csrc/dynamo/eval_frame_cpp.cpp",
    "torch/csrc/dynamo/extra_state.cpp",
    "torch/csrc/dynamo/framelocals_mapping.cpp",
    "torch/csrc/dynamo/guards.cpp",
    "torch/csrc/dynamo/utils.cpp",
    "torch/csrc/dynamo/init.cpp",
    "torch/csrc/dynamo/stackref_bridge.c",
    "torch/csrc/functorch/init.cpp",
    "torch/csrc/fx/node.cpp",
    "torch/csrc/mps/Module.cpp",
    "torch/csrc/mtia/Module.cpp",
    "torch/csrc/export/pybind.cpp",
    "torch/csrc/export/upgrader.cpp",
    "torch/csrc/export/example_upgraders.cpp",
    "torch/csrc/inductor/aoti_package/pybind.cpp",
    "torch/csrc/inductor/aoti_runner/pybind.cpp",
    "torch/csrc/inductor/aoti_eager/kernel_holder.cpp",
    "torch/csrc/inductor/aoti_eager/kernel_meta_info.cpp",
    "torch/csrc/inductor/resize_storage_bytes.cpp",
    "torch/csrc/inductor/static_cuda_launcher.cpp",
    "torch/csrc/jit/backends/backend_init.cpp",
    "torch/csrc/jit/python/init.cpp",
    "torch/csrc/jit/passes/onnx.cpp",
    "torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.cpp",
    "torch/csrc/jit/passes/onnx/deduplicate_initializers.cpp",
    "torch/csrc/jit/passes/onnx/eval_peephole.cpp",
    "torch/csrc/jit/passes/onnx/constant_fold.cpp",
    "torch/csrc/jit/passes/onnx/constant_map.cpp",
    "torch/csrc/jit/passes/onnx/eliminate_unused_items.cpp",
    "torch/csrc/jit/passes/onnx/fixup_onnx_controlflow.cpp",
    "torch/csrc/jit/passes/onnx/list_model_parameters.cpp",
    "torch/csrc/jit/passes/onnx/function_substitution.cpp",
    "torch/csrc/jit/passes/onnx/helper.cpp",
    "torch/csrc/jit/passes/onnx/peephole.cpp",
    "torch/csrc/jit/passes/onnx/preprocess_for_onnx.cpp",
    "torch/csrc/jit/passes/onnx/prepare_division_for_onnx.cpp",
    "torch/csrc/jit/passes/onnx/scalar_type_analysis.cpp",
    "torch/csrc/jit/passes/onnx/unpack_quantized_weights.cpp",
    "torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.cpp",
    "torch/csrc/jit/passes/onnx/shape_type_inference.cpp",
    "torch/csrc/jit/passes/onnx/function_extraction.cpp",
    "torch/csrc/jit/passes/onnx/onnx_log.cpp",
    "torch/csrc/jit/passes/onnx/naming.cpp",
    "torch/csrc/jit/python/pybind_utils.cpp",
    "torch/csrc/jit/passes/onnx/pattern_conversion/autograd_function_process.cpp",
    "torch/csrc/jit/passes/onnx/pattern_conversion/common.cpp",
    "torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.cpp",
    "torch/csrc/jit/passes/onnx/pattern_conversion/pattern_conversion.cpp",
    "torch/csrc/jit/python/python_arg_flatten.cpp",
    "torch/csrc/jit/python/python_custom_class.cpp",
    "torch/csrc/jit/python/python_dict.cpp",
    "torch/csrc/jit/python/python_interpreter.cpp",
    "torch/csrc/jit/python/python_ir.cpp",
    "torch/csrc/jit/python/python_list.cpp",
    "torch/csrc/jit/python/python_tracer.cpp",
    "torch/csrc/jit/python/script_init.cpp",
    "torch/csrc/jit/frontend/concrete_module_type.cpp",
    "torch/csrc/jit/frontend/tree_views.cpp",
    "torch/csrc/jit/python/python_sugared_value.cpp",
    "torch/csrc/jit/python/python_tree_views.cpp",
    "torch/csrc/jit/runtime/static/init.cpp",
    "torch/csrc/jit/tensorexpr/tensorexpr_init.cpp",
    "torch/csrc/monitor/python_init.cpp",
    "torch/csrc/multiprocessing/init.cpp",
    "torch/csrc/onnx/init.cpp",
    "torch/csrc/profiler/python/init.cpp",
    "torch/csrc/profiler/python/combined_traceback.cpp",
    "torch/csrc/serialization.cpp",
    "torch/csrc/tensor/python_tensor.cpp",
    "torch/csrc/utils/init.cpp",
    "torch/csrc/utils/throughput_benchmark.cpp",
    "torch/csrc/utils.cpp",
    "torch/csrc/utils/device_lazy_init.cpp",
    "torch/csrc/utils/invalid_arguments.cpp",
    "torch/csrc/utils/nested.cpp",
    "torch/csrc/utils/object_ptr.cpp",
    "torch/csrc/utils/python_arg_parser.cpp",
    "torch/csrc/utils/python_dispatch.cpp",
    "torch/csrc/utils/python_symnode.cpp",
    "torch/csrc/utils/pybind.cpp",
    "torch/csrc/utils/pyobject_preservation.cpp",
    "torch/csrc/utils/structseq.cpp",
    "torch/csrc/utils/tensor_apply.cpp",
    "torch/csrc/utils/tensor_dtypes.cpp",
    "torch/csrc/utils/tensor_layouts.cpp",
    "torch/csrc/utils/tensor_memoryformats.cpp",
    "torch/csrc/utils/tensor_qschemes.cpp",
    "torch/csrc/utils/tensor_list.cpp",
    "torch/csrc/utils/tensor_new.cpp",
    "torch/csrc/utils/tensor_numpy.cpp",
    "torch/csrc/utils/tensor_types.cpp",
    "torch/csrc/utils/disable_torch_function.cpp",
    "torch/csrc/utils/verbose.cpp",
    "torch/csrc/cpu/Module.cpp",
    "torch/csrc/functionalization/Module.cpp",
    "torch/csrc/instruction_counter/Module.cpp",
    "torch/nativert/python/Bindings.cpp",
] + lazy_tensor_core_python_sources

libtorch_python_distributed_core_sources = [
    "torch/csrc/distributed/c10d/init.cpp",
    "torch/csrc/distributed/c10d/python_comm_hook.cpp",
    "torch/csrc/distributed/c10d/python_callback_work.cpp",
]

libtorch_python_distributed_sources = libtorch_python_distributed_core_sources + [
    "torch/csrc/distributed/autograd/init.cpp",
    "torch/csrc/distributed/rpc/init.cpp",
    "torch/csrc/distributed/rpc/py_rref.cpp",
    "torch/csrc/distributed/rpc/python_functions.cpp",
    "torch/csrc/distributed/rpc/python_rpc_handler.cpp",
    "torch/csrc/distributed/rpc/request_callback_impl.cpp",
    "torch/csrc/distributed/rpc/testing/init.cpp",
    "torch/csrc/distributed/rpc/unpickled_python_call.cpp",
    "torch/csrc/distributed/rpc/unpickled_python_remote_call.cpp",
    "torch/csrc/jit/runtime/register_distributed_ops.cpp",
    "torch/csrc/distributed/c10d/control_plane/PythonHandlers.cpp",
]

def glob_libtorch_python_sources(gencode_pattern = ":generate-code[{}]"):
    _libtorch_python_sources = [gencode_pattern.format(name) for name in [
        "torch/csrc/autograd/generated/python_functions_0.cpp",
        "torch/csrc/autograd/generated/python_functions_1.cpp",
        "torch/csrc/autograd/generated/python_functions_2.cpp",
        "torch/csrc/autograd/generated/python_functions_3.cpp",
        "torch/csrc/autograd/generated/python_functions_4.cpp",
        "torch/csrc/autograd/generated/python_nested_functions.cpp",
        "torch/csrc/autograd/generated/python_nn_functions.cpp",
        "torch/csrc/autograd/generated/python_fft_functions.cpp",
        "torch/csrc/autograd/generated/python_linalg_functions.cpp",
        "torch/csrc/autograd/generated/python_enum_tag.cpp",
        "torch/csrc/autograd/generated/python_return_types.cpp",
        "torch/csrc/autograd/generated/python_sparse_functions.cpp",
        "torch/csrc/autograd/generated/python_special_functions.cpp",
        "torch/csrc/autograd/generated/python_torch_functions_0.cpp",
        "torch/csrc/autograd/generated/python_torch_functions_1.cpp",
        "torch/csrc/autograd/generated/python_torch_functions_2.cpp",
        "torch/csrc/autograd/generated/python_variable_methods.cpp",
        "torch/csrc/functionalization/generated/ViewMetaClassesPythonBinding.cpp",
    ]]

    _libtorch_python_sources.exten
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs`):

- [`Makefile_docs.md`](./Makefile_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`requirements.txt_docs.md`](./requirements.txt_docs.md)
- [`libtorch.rst_docs.md`](./libtorch.rst_docs.md)
- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`generate_repo_docs.py_kw.md_docs.md`](./generate_repo_docs.py_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`pt_template_srcs.bzl_kw.md_docs.md`](./pt_template_srcs.bzl_kw.md_docs.md)
- [`CLAUDE.md_docs.md_docs.md`](./CLAUDE.md_docs.md_docs.md)
- [`setup.py_kw.md_docs.md`](./setup.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `build_variables.bzl_docs.md_docs.md`
- **Keyword Index**: `build_variables.bzl_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
