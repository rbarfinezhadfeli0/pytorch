# Documentation: `docs/torch/csrc/jit/python/init.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/python/init.cpp_docs.md`
- **Size**: 53,850 bytes (52.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/python/init.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/python/init.cpp`
- **Size**: 95,169 bytes (92.94 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <pybind11/pytypes.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/schema_info.h>

#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_init.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
// #include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))
#include <torch/csrc/jit/codegen/onednn/interface.h>
#endif
#include <c10/core/SymNodeImpl.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/schema_type_parser.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/autocast.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/create_functional_graphs.h>
#include <torch/csrc/jit/passes/dbr_quantization/remove_redundant_aliases.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/device_type_analysis.h>
#include <torch/csrc/jit/passes/dtype_analysis.h>
#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_concat_linear.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/frozen_linear_folding.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/integer_value_refinement.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/metal_rewrite.h>
#include <torch/csrc/jit/passes/mobile_optimizer_type.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/quantization/dedup_module_uses.h>
#include <torch/csrc/jit/passes/quantization/finalize.h>
#include <torch/csrc/jit/passes/quantization/fusion_passes.h>
#include <torch/csrc/jit/passes/quantization/insert_observers.h>
#include <torch/csrc/jit/passes/quantization/insert_quant_dequant.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>
#include <torch/csrc/jit/passes/refine_tuple_types.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/check_alias_annotation.h>
#include <torch/csrc/jit/passes/vulkan_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/python/init.h>
#include <torch/csrc/jit/python/opaque_obj.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/jit/python/python_custom_class.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/python/python_tree_views.h>
#include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/decomposition_registry.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/print_handler.h>
#include <torch/csrc/jit/runtime/static/init.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/tensorexpr_init.h>
#include <torch/csrc/utils/cpp_stacktraces.h>

#include <c10/macros/Export.h>
#include <c10/util/irange.h>
#include <c10/util/signal_handler.h>
#include <caffe2/serialize/inline_container.h>

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>

#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace torch::jit {

using c10::AliasInfo;
using c10::Argument;
using c10::FunctionSchema;
using c10::SchemaArgType;
using c10::SchemaArgument;
using c10::SymNode;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using torch::utils::SchemaInfo;

namespace {

using autograd::variable_list;

bool loadPythonClasses() {
  // Leaving this code here, because it will likely be useful at some point
  // PyObject *jit_module = PyImport_ImportModule("torch.jit");
  // THPUtils_assert(jit_module, "class loader couldn't access "
  //"torch.jit module");
  // PyObject *jit_dict = PyModule_GetDict(jit_module);

  return true;
}

static bool opAllowsNumbersAsTensors(c10::Symbol symbol) {
  return symbol.is_prims() || symbol.is_nvprims() ||
      (symbol.is_aten() &&
       torch::should_allow_numbers_as_tensors(symbol.toUnqualString()));
}

std::optional<IValue> toTypeInferredIValueOptional(py::handle input) {
  // Errors need to be caught here because toTypeInferredIValue errors out
  // on various object types, but we want it to work with all types.
  try {
    return toTypeInferredIValue(input);
  } catch (const c10::Error& e) {
    return std::nullopt;
  }
}
} // anonymous namespace

#if defined(BUILDING_TESTS) && !defined(USE_ROCM)
// NOLINTNEXTLINE(misc-use-internal-linkage)
TORCH_API void runJITCPPTests();
#endif

void initJITBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto jit = m.def_submodule("_jit");

  // This is a static object, so we must leak the Python object
  // "release()" is used here to preserve 1 refcount on the
  // object, preventing it from ever being de-allocated by CPython.
  static py::handle exc =
      py::exception<JITException>(m, "JITException").release();

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const JITException& e) {
      // special handling of JITException, to set its python class name and msg
      py::gil_scoped_acquire acquire;
      const auto& className = e.getPythonClassName();
      const auto& originalMsg = e.getOriginalMsg();
      JITException::setCaughtOriginalMsg(originalMsg.value_or(""));
      JITException::setCaughtPythonClassName(className.value_or(""));
      // If we still had the py::exception<JITException> object, we could
      // just call it. But we must get a handle to leak it and there is no
      // way I can find to re-create it from the handle. So setting the
      // exception manually
      PyErr_SetString(exc.ptr(), e.what());
    }
  });

  m.def(
      "_get_caught_jit_exception_class_name",
      JITException::getCaughtPythonClassName);
  m.def(
      "_get_caught_jit_exception_original_msg",
      JITException::getCaughtOriginalMsg);

  py::class_<python::IODescriptor> iodescriptor(
      m,
      "IODescriptor"); // NOLINT(bugprone-unused-raii)

  m.def("_jit_init", loadPythonClasses)
      .def(
          "_jit_debug_fuser_num_cached_kernel_specs",
          torch::jit::fuser::debugNumCachedKernelSpecs)
      .def("_jit_pass_lower_all_tuples", LowerAllTuples)
      .def(
          "_new_symbolic_shape_symbol",
          []() { return c10::ShapeSymbol::newSymbol().value(); })
      .def(
          "_jit_shape_compute_graph_for_node",
          [](Node* n) -> std::optional<std::shared_ptr<Graph>> {
            if (!n->maybeSchema()) {
              return std::nullopt;
            }
            return shapeComputeGraphForSchema(n->schema());
          })
      .def(
          "_jit_decomposition_graph_for_node",
          [](Node* n) -> std::optional<std::shared_ptr<Graph>> {
            if (!n->maybeSchema()) {
              return std::nullopt;
            }
            return GetDecomposition(n->schema());
          })
      .def("_jit_pass_run_decompositions", RunDecompositions)
      // using Node* here instead of Schema because looking up the schema
      // and passing it in from Python will have a different pointer than the
      // schema that is globally used for caching
      .def(
          "_jit_register_shape_compute_graph_for_node",
          [](Node* n, std::shared_ptr<Graph>& graph) {
            if (n->maybeSchema()) {
              const FunctionSchema& schema = n->schema();
              RegisterShapeComputeGraphForSchema(schema, graph);
            } else {
              TORCH_INTERNAL_ASSERT(false, "Expected schema", n);
            }
          })
      .def(
          "_jit_register_decomposition_for_schema",
          [](const FunctionSchema& s, std::shared_ptr<Graph>& graph) {
            // because this is invoked by python, the function schema *
            // becomes different, and we need to find and reuse the
            // one that is used for caching
            auto op =
                findOperatorFor(c10::OperatorName(s.name(), s.overload_name()));
            RegisterDecomposition(op->schema(), graph);
          })
      .def("_jit_pass_propagate_shapes_on_graph", PropagateShapesOnGraph)
      .def(
          "_jit_pass_propagate_shapes_on_graph_and_build_compute",
          [](std::shared_ptr<Graph>& graph) {
            return PropagateShapesAndBuildLargeShapeComputeGraph(
                graph, *graph->nodes().begin(), *graph->nodes().end());
          })
      .def(
          "_jit_pass_propagate_shapes_on_graph_and_build_compute",
          [](std::shared_ptr<Graph>& graph, Node* beg) {
            return PropagateShapesAndBuildLargeShapeComputeGraph(
                graph, beg, *graph->nodes().end());
          })
      .def(
          "_jit_pass_propagate_shapes_on_graph_and_build_compute",
          PropagateShapesAndBuildLargeShapeComputeGraph)
      .def("_jit_pass_integer_value_refinement", RefineIntegerValues)
      .def(
          "_jit_set_symbolic_shapes_test_mode",
          &setSymbolicShapeAnalysisTestMode)
      .def(
          "_jit_symbolic_shapes_test_mode_enabled",
          &symbolicShapeAnalysisTestModeEnabled)
      .def("_jit_pass_autocast", Autocast)
      .def("_jit_set_autocast_mode", &setAutocastMode)
      .def("_jit_pass_fuse", FuseGraph)
      .def(
          "_jit_pass_replace_old_ops_with_upgraders",
          [](std::shared_ptr<Graph>& g) {
            return ReplaceOldOperatorsWithUpgraders(g);
          })
      .def(
          "_jit_pass_dce",
          [](std::shared_ptr<Graph>& g) {
            return EliminateDeadCode(g->block()); // overload resolution
          })
      .def(
          "_jit_pass_dce_graph",
          [](std::shared_ptr<Graph>& g) { return EliminateDeadCode(g); })
      .def(
          "_jit_pass_dce_allow_deleting_nodes_with_side_effects",
          [](std::shared_ptr<Graph>& g) {
            return EliminateDeadCode(
                g->block(),
                true,
                DCESideEffectPolicy::
                    ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS); // overload
                                                             // resolution
          })
      .def(
          "_jit_pass_cse",
          [](std::shared_ptr<Graph>& g) {
            return EliminateCommonSubexpression(g); // overload resolution
          })
      .def(
          "_jit_pass_fuse_quantized_add_relu",
          [](std::shared_ptr<Graph>& g) {
            return FuseQuantizedAddRelu(g); // overload resolution
          })
      .def(
          "_jit_pass_insert_observers",
          [](Module& module,
             const std::string& method_name,
             const py::dict& qconfig_dict,
             bool inplace,
             int quant_type_int) {
            auto dict = py::cast<std::unordered_map<
                std::string,
                std::optional<std::tuple<Module, Module>>>>(qconfig_dict);
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return InsertObservers(
                module, method_name, dict, inplace, quant_type);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("qconfig_dict"),
          py::arg("inplace"),
          py::arg("quant_type_int") = 1)
      .def(
          "_jit_pass_insert_observer_method_for_ondevice_ptq",
          [](Module& module,
             const std::string& method_name,
             const py::dict& qconfig_dict,
             bool inplace,
             int quant_type_int) {
            auto dict = py::cast<std::unordered_map<
                std::string,
                std::optional<std::tuple<Module, Module>>>>(qconfig_dict);
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return InsertObserversForOnDevicePTQ(
                module, method_name, dict, inplace, quant_type);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("qconfig_dict"),
          py::arg("inplace"),
          py::arg("quant_type_int") = 1)
      .def(
          "_jit_pass_insert_quant_dequant",
          [](Module& module,
             const std::string& method_name,
             bool inplace,
             bool debug,
             int quant_type_int) {
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return InsertQuantDeQuant(
                module, method_name, inplace, debug, quant_type);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("inplace"),
          py::arg("debug"),
          py::arg("quant_type_int") = 1)
      .def(
          "_jit_pass_insert_quant_dequant_for_ondevice_ptq",
          [](Module& module,
             const std::string& method_name,
             bool inplace,
             bool debug,
             int quant_type_int) {
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return InsertQuantDeQuantOnDevicePTQ(
                module, method_name, inplace, debug, quant_type);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("inplace"),
          py::arg("debug"),
          py::arg("quant_type_int") = 1)
      .def(
          "_jit_pass_insert_prepack_unpack",
          [](std::shared_ptr<Graph>& g) { return InsertPrepackUnpack(g); })
      .def(
          "_jit_pass_insert_prepack_unpack",
          [](Module& module) { return InsertPrepackUnpack(module); })
      .def(
          "_jit_pass_quant_fusion",
          [](std::shared_ptr<Graph>& g) { return QuantFusion(g); })
      .def(
          "_jit_pass_fold_convbn",
          [](Module& module) { return FoldConvBatchNorm(module); })
      .def(
          "_jit_pass_dbr_quant_remove_redundant_aliases",
          [](Module& module) { return DBRQuantRemoveRedundantAliases(module); })
      .def(
          "_freeze_module",
          [](Module& module,
             std::vector<std::string>& preservedAttrs,
             bool freezeInterfaces,
             bool preserveParameters) {
            return freeze_module(
                module, preservedAttrs, freezeInterfaces, preserveParameters);
          },
          py::arg("module"),
          py::arg("preservedAttrs") = std::vector<std::string>(),
          py::arg("freezeInterfaces") = true,
          py::arg("preserveParameters") = false)
      .def("_jit_pass_concat_frozen_linear", &FrozenConcatLinear)
      .def("_jit_pass_fold_frozen_conv_bn", &FoldFrozenConvBatchnorm)
      .def("_jit_pass_fold_frozen_conv_add_or_sub", &FoldFrozenConvAddOrSub)
      .def("_jit_pass_fold_frozen_conv_mul_or_div", &FoldFrozenConvMulOrDiv)
      .def("_jit_pass_fold_frozen_linear_bn", &FoldFrozenLinearBatchnorm)
      .def("_jit_pass_convert_frozen_ops_to_mkldnn", &ConvertFrozenOpsToMKLDNN)
      .def("_jit_pass_fuse_frozen_conv_add_relu", &FuseFrozenConvAddRelu)
      .def("_jit_pass_transpose_frozen_linear", &FrozenLinearTranspose)
      .def("_jit_pass_optimize_frozen_graph", &OptimizeFrozenGraph)
      .def(
          "_jit_pass_optimize_for_inference",
          [](Module& module, const std::vector<std::string>& other_methods) {
            optimize_for_inference(module, other_methods);
          },
          py::arg("module"),
          py::arg("other_methods") = std::vector<std::string>())
      .def("_jit_pass_fuse_linear", &FuseLinear)
      .def(
          "_jit_pass_fuse_add_relu",
          [](std::shared_ptr<Graph>& graph) { FuseAddRelu(graph); })
      .def("_jit_pass_dedup_module_uses", &DedupModuleUses)
      .def("_jit_pass_replicate_dequantize", &ReplicateDeQuant)
      .def(
          "_jit_pass_swap_functional_linear",
          [](std::shared_ptr<Graph>& graph) { SwapFunctionalLinear(graph); })
      .def(
          "_jit_pass_swap_functional_linear",
          [](Module& module) { SwapFunctionalLinear(module); })
      .def(
          "_jit_pass_quant_finalize",
          [](Module& module,
             int quant_type_int,
             const std::vector<std::string>& preserved_attrs) {
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return Finalize(module, quant_type, preserved_attrs);
          },
          py::arg("module"),
          py::arg("quant_type_int") = 1,
          py::arg("preserved_attrs") = std::vector<std::string>())
      .def(
          "_jit_pass_quant_finalize_for_ondevice_ptq",
          [](Module& module,
             int quant_type_int,
             const std::string& method_name) {
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return FinalizeOnDevicePTQ(module, quant_type, method_name);
          },
          py::arg("module"),
          py::arg("quant_type_int") = 1,
          py::arg("preserved_attrs") = std::vector<std::string>())
      .def(
          "_jit_pass_pattern_based_rewrite",
          [](const Module& m) { return PatternBasedRewrite(m); })
      .def(
          "_jit_pass_custom_pattern_based_rewrite",
          [](const std::string& pattern,
             const std::string& fused_node_name,
             const Module& m) {
            SubgraphRewriter subgraph_rewriter;
            subgraph_rewriter.RegisterRewritePattern(pattern, fused_node_name);
            subgraph_rewriter.runOnModule(m);
          })
      .def(
          "_jit_pass_custom_pattern_based_rewrite_graph",
          [](const std::string& pattern,
             const std::string& fused_node_name,
             std::shared_ptr<Graph> g,
             const std::vector<std::pair<std::string, std::string>>&
                 value_name_pairs) {
            SubgraphRewriter subgraph_rewriter;
            subgraph_rewriter.RegisterRewritePattern(
                pattern, fused_node_name, value_name_pairs);
            subgraph_rewriter.runOnGraph(g);
          },
          py::arg("pattern"),
          py::arg("fused_node_name"),
          py::arg("g"),
          py::arg("value_name_pairs") =
              std::vector<std::pair<std::string, std::string>>())
      .def("_jit_pass_constant_pooling", ConstantPooling)
      // RemoveInplaceOps is used by CoreML so it must be removed with care.
      .def("_jit_pass_propagate_dtype", DtypePropagation)
      .def("_jit_pass_propagate_device", DeviceTypePropagation)
      .def(
          "_jit_pass_remove_inplace_ops",
          [](const std::shared_ptr<Graph>& g) { return RemoveInplaceOps(g); })
      .def(
          "_jit_pass_create_functional_graphs",
          [](std::shared_ptr<Graph>& g) { return CreateFunctionalGraphs(g); })
      .def(
          "_jit_pass_remove_mutation",
          [](std::shared_ptr<Graph>& g) {
            RemoveListMutation(g);
            return RemoveTensorMutation(g);
          })
      .def(
          "_jit_pass_functional_to_inplace_activation",
          [](std::shared_ptr<Graph>& g) {
            return FunctionalToInplaceActivation(g);
          })
      .def(
          "_jit_pass_inplace_to_functional_activation",
          [](std::shared_ptr<Graph>& g) {
            return InplaceToFunctionalActivation(g);
          })
      .def(
          "_jit_pass_inline_functional_graphs",
          [](std::shared_ptr<Graph>& g) { return InlineFunctionalGraphs(g); })
      .def(
          "_jit_pass_peephole",
          [](const std::shared_ptr<Graph>& g, bool disable_shape_peepholes) {
            return PeepholeOptimize(g, disable_shape_peepholes);
          },
          py::arg("graph"),
          py::arg("disable_shape_peepholes") = false)
      .def(
          "_jit_pass_peephole_list_idioms",
          [](const std::shared_ptr<Graph>& g, bool refine_list_len) {
            return PeepholeOptimizeListIdioms(g, refine_list_len);
          },
          py::arg("graph"),
          py::arg("refine_list_len") = false)
      .def(
          "_jit_pass_refine_integer_values",
          [](std::shared_ptr<Graph>& g) { return RefineIntegerValues(g); })
      .def(
          "_jit_pass_fuse_addmm",
          [](std::shared_ptr<Graph>& g) { return FuseAddMM(g); })
      .def(
          "_jit_pass_canonicalize",
          [](const std::shared_ptr<Graph>& g, bool keep_unique_names = true) {
            return Canonicalize(g, keep_unique_names);
          },
          py::arg("graph"),
          py::arg("keep_unique_names") = true)
      .def("_jit_pass_lint", LintGraph)
      .def(
          "_jit_pass_complete_shape_analysis",
          [](const std::shared_ptr<Graph>& graph,
             const py::tuple& inputs,
             bool with_grad) {
            ArgumentSpecCreator arg_spec_creator(*graph);
            Stack stack;
            stack.reserve(inputs.size()); // captures?
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            ArgumentSpec spec = arg_spec_creator.create(with_grad, stack);
            arg_spec_creator.specializeTypes(*graph, spec);
            // We only get partial specialization from the arg_spec_creator, but
            // we want full shape specialization. The alternative would be to
            // have a "complete type inference" function in ArguemntSpecCreator.
            auto g_inputs = graph->inputs();
            for (const auto i : c10::irange(inputs.size())) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            PropagateInputShapes(graph);
          })
      .def(
          "_jit_interpret_graph",
          [](std::shared_ptr<Graph>& graph, const py::tuple& inputs) {
            Stack stack;
            stack.reserve(inputs.size()); // captures?
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            auto g_inputs = graph->inputs();
            for (const auto i : c10::irange(inputs.size())) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            Code code(graph, "<on-demand-func>");
            InterpreterState(code).run(stack);
            return createPyObjectForStack(std::move(stack));
          },
          py::doc(
              "Interpret a JIT graph with given inputs without running any optimization passes on it"))
      .def(
          "_jit_trace_graph",
          [](std::shared_ptr<Graph>& graph, const py::tuple& inputs) {
            Stack stack;
            stack.reserve(inputs.size()); // captures?
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            auto g_inputs = graph->inputs();
            for (const auto i : c10::irange(inputs.size())) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            return TraceGraph(graph, stack);
          })
      .def(
          "_jit_trace_module",
          [](Module& model, const py::tuple& inputs) {
            auto graph = model.get_method("forward").graph();
            Stack stack;
            stack.reserve(inputs.size() + 1); // captures?
            push(stack, model._ivalue());
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            auto traced = TraceGraph(graph, stack);
            GRAPH_DUMP("Traced Graph", traced);

            // the easiest way to replace a graph in a module is
            // to remove all the nodes in the original graph
            // clone everything from the traced one
            graph->block()->clear();
            graph->block()->cloneFrom(traced->block(), nullptr);
            GRAPH_DUMP("Copied Graph", graph);
          })
      .def("_jit_pass_remove_expands", RemoveExpands)
      .def("_jit_pass_erase_number_types", EraseNumberTypes)
      .def("_jit_pass_inline_fork_wait", InlineForkWait)
      .def("_jit_pass_inline", Inline)
      .def(
          "_jit_pass_lower_graph",
          [](std::shared_ptr<Graph>& graph, const Module& self) {
            return LowerGraph(*graph, self._ivalue());
          })
      .def("_jit_pass_loop_unrolling", UnrollLoops)
      .def("_jit_pass_constant_loop_unrolling", UnrollConstantLoops)
      .def(
          "_jit_pass_constant_propagation_immutable_types",
          [](std::shared_ptr<Graph>& g) {
            return ConstantPropagationImmutableTypes(g);
          })
      .def(
          "_jit_pass_constant_propagation",
          [](std::shared_ptr<Graph>& g) { return ConstantPropagation(g); },
          py::arg("graph"))
      .def("_jit_pass_erase_shape_information", EraseShapeInformation)
      .def(
          "_jit_object_is_non_holding",
          [](Node& n) {
            return toIValue(n.output())->toObject()->is_weak_compilation_ref();
          })
      .def(
          "_jit_erase_non_input_shape_information",
          [](std::shared_ptr<Graph>& g) {
            std::vector<TypePtr> input_types;
            for (Value* v : g->inputs()) {
              if (auto tt = v->type()->cast<TensorType>()) {
                input_types.emplace_back(tt);
              } else {
                input_types.emplace_back(nullptr);
              }
            }
            EraseShapeInformation(g);
            for (size_t i = 0; i < input_types.size(); ++i) {
              if (input_types[i]) {
                g->inputs().at(i)->setType(input_types[i]);
              }
            }
          })
      .def(
          "_jit_pass_create_autodiff_subgraphs",
          [](const std::shared_ptr<Graph>& graph, const py::object& threshold) {
            if (threshold.is_none()) {
              CreateAutodiffSubgraphs(graph);
            } else {
              CreateAutodiffSubgraphs(graph, py::cast<int>(threshold));
            }
          },
          py::arg("graph"),
          py::arg("threshold") = py::none())
#if defined(BUILDING_TESTS) && !defined(USE_ROCM)
      .def(
          "_jit_run_cpp_tests",
          []() {
            // We have to release the GIL inside this method, because if we
            // happen to initialize the autograd engine in these tests, the
            // newly spawned worker threads will try to initialize their
            // PyThreadState*, and they need the GIL for this.
            pybind11::gil_scoped_release _no_gil;
            return runJITCPPTests();
          })
      .def("_jit_has_cpp_tests", []() { return true; })
      .def("_has_tensorexpr_cpp_tests", []() { return true; })
#else
      .def("_jit_run_cpp_tests", []() { throw std::exception(); })
      .def("_jit_has_cpp_tests", []() { return false; })
      .def("_run_tensorexpr_cpp_tests", []() { throw std::exception(); })
      .def("_has_tensorexpr_cpp_tests", []() { return false; })
#endif
      .def(
          "_jit_flatten",
          [](py::handle& obj) {
            auto res = python::flatten(obj);
            return std::make_pair(res.vars, res.desc);
          })
      .def(
          "_jit_unflatten",
          [](const autograd::variable_list& vars, python::IODescriptor& desc) {
            return py::reinterpret_steal<py::object>(
                python::unflatten(vars, desc));
          })
      .def("_jit_pass_canonicalize_graph_fuser_ops", CanonicalizeOps)
      .def("_jit_pass_decompose_ops", DecomposeOps)
      .def("_jit_pass_specialize_autogradzero", specializeAutogradZero)
      .def("_jit_override_can_fuse_on_cpu", &overrideCanFuseOnCPU)
      .def("_jit_override_can_fuse_on_gpu", &overrideCanFuseOnGPU)
      .def("_jit_can_fuse_on_cpu", &canFuseOnCPU)
      .def("_jit_can_fuse_on_gpu", &canFuseOnGPU)
      .def("_jit_can_fuse_on_cpu_legacy", &canFuseOnCPULegacy)
      .def("_jit_override_can_fuse_on_cpu_legacy", &overrideCanFuseOnCPULegacy)
      .def(
          "_jit_differentiate",
          [](Graph& g) {
            // the python binding slightly differs in semantics
            // it makes a copy of the input Graph, and works on that
            // jit::differentiate mutates the input Graph
            auto g_clone = g.copy();
            return differentiate(g_clone);
          })
      .def(
          "_jit_check_alias_annotation",
          [](const std::shared_ptr<Graph>& g,
             const py::tuple& args,
             const std::string& unqualified_op_name) {
            auto stack = toTraceableStack(args);
            checkAliasAnnotation(g, std::move(stack), unqualified_op_name);
          })
#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))
      .def("_jit_set_llga_enabled", &RegisterLlgaFuseGraph::setEnabled)
      .def("_jit_llga_enabled", &RegisterLlgaFuseGraph::isEnabled)
#else
      .def("_jit_set_llga_enabled", [](bool flag) { return false; })
      .def("_jit_llga_enabled", []() { return false; })
#endif
      .def(
          "_jit_set_tracer_state_warn",
          [](bool new_warn) {
            jit::tracer::getTracerStateWarnMode() = new_warn;
          })
      .def(
          "_jit_get_tracer_state_warn",
          []() {
            bool current_tracer_warn = jit::tracer::getTracerStateWarnMode();
            return current_tracer_warn;
          })
      .def(
          "_jit_set_nvfuser_skip_node_kind",
          [](const std::string& op_name, bool flip = true) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_skip_node_kind is deprecated and a no-op");
          })
      .def(
          "_jit_set_nvfuser_enabled",
          [](bool) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_can_be_enabled",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_can_be_enabled is deprecated and a no-op");
          })
      .def(
          "_jit_set_nvfuser_single_node_mode",
          [](bool) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_single_node_mode is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_single_node_mode",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_single_node_mode is deprecated and a no-op");
          })
      .def(
          "_jit_set_nvfuser_horizontal_mode",
          [](bool) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_horizontal_mode is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_horizontal_mode",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_horizontal_mode is deprecated and a no-op");
          })
      .def(
          "_jit_set_nvfuser_guard_mode",
          [](bool) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_guard_mode is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_enabled",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_enabled is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_set_comparison_callback",
          [](bool, py::function) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_set_comparison_callback is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_clear_comparison_callback",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_clear_comparison_callback is deprecated and a no-op");
          })
      .def(
          "_jit_set_profiling_mode",
          [](bool profiling_flag) {
            bool oldState = getProfilingMode();
            getProfilingMode() = profiling_flag;
            return oldState;
          })
      .def(
          "_jit_set_profiling_executor",
          [](bool profiling_flag) {
            bool oldState = getExecutorMode();
            getExecutorMode() = profiling_flag;
            return oldState;
          })
      .def(
          "_jit_set_num_profiled_runs",
          [](size_t num) {
            size_t old_num = getNumProfiledRuns();
            getNumProfiledRuns() = num;
            return old_num;
          })
      .def(
          "_jit_get_num_profiled_runs",
          [] {
            // pybind can't automatically bind to atomic size_t
            size_t num_runs = getNumProfiledRuns();
            return num_runs;
          })
      .def(
          "_jit_set_bailout_depth",
          [](size_t depth) {
            TORCH_WARN(
                "Use _jit_set_fusion_strategy, bailout depth is deprecated. Setting to (STATIC, ",
                depth,
                ")");
            size_t old_depth = getBailoutDepth();
            FusionStrategy strat = {{FusionBehavior::STATIC, depth}};
            setFusionStrategy(strat);
            return old_depth;
          })
      .def(
          "_jit_set_fusion_strategy",
          [](const std::vector<std::pair<std::string, size_t>>& strategy) {
            FusionStrategy vec_conv;
            for (const auto& pair : strategy) {
              if (pair.first == "STATIC") {
                vec_conv.emplace_back(FusionBehavior::STATIC, pair.second);
              } else if (pair.first == "DYNAMIC") {
                vec_conv.emplace_back(FusionBehavior::DYNAMIC, pair.second);
              } else {
                TORCH_INTERNAL_ASSERT(
                    false,
                    "FusionBehavior only supported 'STATIC' or 'DYNAMIC', got: ",
                    pair.first);
              }
            }
            auto old_strategy = getFusionStrategy();
            auto strat =
                fmap(old_strategy, [](std::pair<FusionBehavior, size_t> behav) {
                  return std::pair<std::string, size_t>(
                      behav.first == FusionBehavior::STATIC ? "STATIC"
                                                            : "DYNAMIC",
                      behav.second);
                });
            setFusionStrategy(vec_conv);
            return strat;
          })
      .def(
          "_jit_set_inline_everything_mode",
          [](bool enabled) { getInlineEverythingMode() = enabled; })
      .def(
          "_jit_get_inline_everything_mode",
          []() { return getInlineEverythingMode(); })
      .def(
          "_jit_get_logging_option",
          []() { return ::torch::jit::get_jit_logging_levels(); })
      .def(
          "_jit_set_logging_option",
          [](std::string loggingOption) -> void {
            ::torch::jit::set_jit_logging_levels(std::move(loggingOption));
          })
      .def(
          "_jit_set_logging_stream",
          [](const std::string& stream_name) -> void {
            if (stream_name == "stdout") {
              ::torch::jit::set_jit_logging_output_stream(std::cout);
            } else if (stream_name == "stderr") {
              ::torch::jit::set_jit_logging_output_stream(std::cerr);
            } else {
              std::cerr << "ERROR: only `stdout` and `stderr`"
                        << "are supported as output options" << '\n';
            }
          })
      .def(
          "_storage_id",
          [](const at::Tensor& ten) -> int64_t {
            return reinterpret_cast<int64_t>(
                ten.storage().unsafeGetStorageImpl());
          })
      .def(
          "_jit_try_infer_type",
          [](py::object obj) -> InferredType {
            return tryToInferType(std::move(obj));
          })
      .def(
          "_jit_get_te_cuda_pointwise_loop_levels",
          []() -> int {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseLoopLevels();
          })
      .def(
          "_jit_set_te_cuda_pointwise_loop_levels",
          [](int level) {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseLoopLevels() = level;
          })
      .def(
          "_jit_get_te_cuda_pointwise_block_count",
          []() -> int {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseBlockCount();
          })
      .def(
          "_jit_set_te_cuda_pointwise_block_count",
          [](int block_count) {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseBlockCount() = block_count;
          })
      .def(
          "_jit_get_te_cuda_pointwise_block_size",
          []() -> int {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseBlockSize();
          })
      .def(
          "_jit_set_te_cuda_pointwise_block_size",
          [](int block_size) {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseBlockSize() = block_size;
          })
      .def("_jit_set_texpr_fuser_enabled", &setTensorExprFuserEnabled)
      .def("_jit_texpr_fuser_enabled", &tensorExprFuserEnabled)
      .def("_jit_texpr_fallback_allowed", &tensorexpr::fallbackAllowed)
      .def("_jit_texpr_set_fallback_allowed", &tensorexpr::setFallbackAllowed)
      .def("_jit_set_texpr_reductions_enabled", &setTexprReductionsEnabled)
      .def(
          "_jit_set_texpr_dynamic_shape_enabled",
          &setTensorExprDynamicShapeFusionEnabled)
      .def(
          "_jit_texpr_dynamic_shape_enabled",
          &tensorExprDynamicShapeFusionEnabled)
      .def("_jit_texpr_reductions_enabled", &texprReductionsEnabled)
      .def(
          "_jit_set_te_generate_block_code",
          [](bool gen_block_code) {
            using namespace torch::jit::tensorexpr;
            return getTEGenerateBlockCode() = gen_block_code;
          })
      .def(
          "_jit_get_te_generate_block_code",
          []() -> bool {
            using namespace torch::jit::tensorexpr;
            return getTEGenerateBlockCode();
          })
      .def(
          "_jit_get_te_must_use_llvm_cpu",
          []() -> bool {
            using namespace torch::jit::tensorexpr;
            return getTEMustUseLLVMOnCPU();
          })
      .def(
          "_jit_set_te_must_use_llvm_cpu",
          [](bool use_llvm) {
            using namespace torch::jit::tensorexpr;
            getTEMustUseLLVMOnCPU() = use_llvm;
          })
      .def(
          "_jit_cat_wo_conditionals",
          [](bool optimize_cat) {
            using namespace torch::jit::tensorexpr;
            getCatWoConditionals() = optimize_cat;
          })
      .def(
          "_jit_opt_conditionals",
          [](bool opt_conds) {
            using namespace torch::jit::tensorexpr;
            getOptConditionals() = opt_conds;
          })
      .def(
          "_llvm_enabled",
          []() {
#ifdef TORCH_ENABLE_LLVM
            return true;
#else
            return false;
#endif
          })
      .def(
          "_jit_pass_fuse_tensorexprs",
          [](std::shared_ptr<Graph>& g) {
            FuseTensorExprs(g);
            RemoveTensorTypeSpecializations(g);
          })
      .def(
          "_jit_fuser_get_fused_kernel_code",
          [](Graph& g, const std::vector<at::Tensor>& inps) {
            return debugGetFusedKernelCode(g, inps);
          })
      .def(
          "_jit_pass_remove_dropout",
          [](script::Module& module) { return removeDropout(module); })
      .def(
          "_jit_pass_refine_tuple_types",
          [](std::shared_ptr<Graph>& graph) { return RefineTupleTypes(graph); })
      .def(
          "_jit_pass_transform_conv1d_to_conv2d",
          [](std::shared_ptr<Graph>& graph) {
            return transformConv1dToConv2d(graph);
          })
      .def(
          "_jit_pass_transform_conv1d_to_conv2d",
          [](script::Module& module) {
            return transformConv1dToConv2d(module);
          })
      .def(
          "_jit_pass_insert_prepacked_ops",
          [](std::shared_ptr<Graph>& graph) {
            return insertPrePackedOps(graph);
          })
      .def(
          "_jit_pass_insert_prepacked_ops",
          [](script::Module& module) { return insertPrePackedOps(module); })
      .def(
          "_jit_pass_fuse_clamp_w_prepacked_linear_conv",
          [](script::Module& module) {
            return fusePrePackedLinearConvWithClamp(module);
          })
      .def(
          "_jit_pass_fold_prepacking_ops",
          [](script::Module& module) { return FoldPrePackingOps(module); })
      .def(
          "_jit_pass_optimize_for_mobile",
          [](script::Module& module,
             std::set<MobileOptimizerType>& optimization_blocklist,
             std::vector<std::string>& preserved_methods) {
            return optimizeForMobile(
                module, optimization_blocklist, preserved_methods);
          })
      .def(
          "_hack_do_not_use_clone_module_with_class",
          [](script::Module& module,
             std::vector<std::string>& ignored_methods,
             std::vector<std::string>& ignored_attributes) {
            const bool inplace = false;
            const std::unordered_set<std::string> ignored_methods_set(
                ignored_methods.begin(), ignored_methods.end());
            const std::unordered_set<std::string> ignored_attributes_set(
                ignored_attributes.begin(), ignored_attributes.end());
            return module.clone(
                inplace, ignored_methods_set, ignored_attributes_set);
          })
      .def(
          "_jit_pass_vulkan_insert_prepacked_ops",
          [](std::shared_ptr<Graph>& graph) {
            return vulkanInsertPrePackedOps(graph);
          })
      .def(
          "_jit_pass_vulkan_insert_prepacked_ops",
          [](script::Module& module) {
            return vulkanInsertPrePackedOps(module);
          })
      .def(
          "_jit_pass_vulkan_fuse_clamp_w_prepacked_conv",
          [](script::Module& module) {
            return vulkanFusePrePackedConvWithClamp(module);
          })
      .def(
          "_jit_pass_vulkan_fold_prepacking_ops",
          [](script::Module& module) {
            return vulkanFoldPrePackingOps(module);
          })
      .def(
          "_jit_pass_vulkan_optimize_for_mobile",
          [](script::Module& module,
             std::set<MobileOptimizerType>& optimization_blocklist,
             std::vector<std::string>& preserved_methods) {
            return vulkanOptimizeForMobile(
                module, optimization_blocklist, preserved_methods);
          })
      .def(
          "_jit_pass_metal_insert_prepacked_ops",
          [](std::shared_ptr<Graph>& graph) {
            return metalInsertPrePackedOps(graph);
          })
      .def(
          "_jit_pass_metal_insert_prepacked_ops",
          [](script::Module& module) {
            return metalInsertPrePackedOps(module);
          })
      .def(
          "_jit_pass_metal_fuse_clamp_w_prepacked_conv",
          [](script::Module& module) {
            return metalFusePrePackedConvWithClamp(module);
          })
      .def(
          "_jit_pass_metal_fold_prepacking_ops",
          [](script::Module& module) { return metalFoldPrePackingOps(module); })
      .def(
          "_jit_pass_metal_optimize_for_mobile",
          [](script::Module& module,
             std::vector<std::string>& preserved_methods) {
            return metalOptimizeForMobile(module, preserved_methods);
          })
      .def(
          "_jit_pass_filter_non_tensor_arguments",
          [](std::map<std::string, IValue> params) {
            std::map<std::string, at::Tensor> retval;
            for (auto& kv : params) {
              if (kv.second.isTensor()) {
                retval[kv.first] = std::move(kv.second).toTensor();
              }
            }
            return retval;
          })
      .def("_jit_pass_batch_mm", BatchMM)
      .def(
          "_jit_decay_packed_param_input_types",
          [](Graph& g) {
            for (Value* i : g.inputs()) {
              if (i->type() ==
                      getCustomClass(
                          "__torch__.torch.classes.quantized.Conv2dPackedParamsBase") ||
                  i->type() ==
                      getCustomClass(
                          "__torch__.torch.classes.quantized.Conv3dPackedParamsBase") ||
                  i->type() ==
                      getCustomClass(
                          "__torch__.torch.classes.quantized.LinearPackedParamsBase")) {
                // Dummy CompleteTensorType to appease ONNX validator.
                i->setType(TensorType::create(
                    at::kQInt8,
                    c10::kCPU,
                    std::vector<int64_t>{1},
                    std::vector<int64_t>{1},
                    std::nullopt));
              }
            }
          })
      .def("_jit_set_utf8_decoding_ignore", &setUTF8DecodingIgnore);

  // NB: This isn't actually used for regular PyTorch symbolic tracing;
  // XLA is what needs this
#define SYMNODE_UNARY(n) .def(#n, [](const c10::SymNode& a) { return a->n(); })
#define SYMNODE_BINARY(n) \
  .def(#n, [](const c10::SymNode& a, const c10::SymNode& b) { return a->n(b); })
#define SYMNODE_SIZES_STRIDES(n)                \
  .def(                                         \
      #n,                                       \
      [](const c10::SymNode& a,                 \
         c10::ArrayRef<c10::SymNode> sizes,     \
         c10::ArrayRef<c10::SymNode> strides) { \
        return a->n(sizes, strides);            \
      })
  auto symnode_class =
      py::class_<c10::SymNodeImpl, c10::SymNode>(m, "_SymNode")
      // clang-format off
      // These DO NOT install magic methods; the SymInt/SymFloat wrapper in
      // Python is responsible for this
      SYMNODE_UNARY(clone)
      SYMNODE_UNARY(is_int)
      SYMNODE_UNARY(is_float)
      SYMNODE_UNARY(is_bool)
      SYMNODE_UNARY(bool_)
      SYMNODE_UNARY(int_)
      SYMNODE_UNARY(sym_float)
      SYMNODE_BINARY(add)
      SYMNODE_BINARY(sub)
      SYMNODE_BINARY(mul)
      SYMNODE_BINARY(truediv)
      SYMNODE_BINARY(int_truediv)
      SYMNODE_BINARY(float_truediv)
      SYMNODE_BINARY(pow)
      SYMNODE_BINARY(float_pow)
      SYMNODE_BINARY(pow_by_natural)
      SYMNODE_BINARY(floordiv)
      SYMNODE_BINARY(int_floordiv)
      SYMNODE_BINARY(mod)
      SYMNODE_BINARY(eq)
      SYMNODE_BINARY(ne)
      SYMNODE_BINARY(gt)
      SYMNODE_BINARY(lt)
      SYMNODE_BINARY(le)
      SYMNODE_BINARY(ge)
      SYMNODE_BINARY(sym_min)
      SYMNODE_BINARY(sym_max)
      SYMNODE_BINARY(sym_and)
      SYMNODE_BINARY(sym_or)
      SYMNODE_UNARY(sym_not)
      SYMNODE_UNARY(ceil)
      SYMNODE_UNARY(floor)
      SYMNODE_UNARY(neg)
      SYMNODE_SIZES_STRIDES(is_contiguous)
      SYMNODE_SIZES_STRIDES(is_channels_last_contiguous_2d)
      SYMNODE_SIZES_STRIDES(is_channels_last_contiguous_3d)
      SYMNODE_SIZES_STRIDES(is_
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/python`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/python`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/python`):

- [`opaque_obj.h_kw.md_docs.md`](./opaque_obj.h_kw.md_docs.md)
- [`script_init.h_docs.md_docs.md`](./script_init.h_docs.md_docs.md)
- [`python_tree_views.cpp_docs.md_docs.md`](./python_tree_views.cpp_docs.md_docs.md)
- [`python_dict.cpp_docs.md_docs.md`](./python_dict.cpp_docs.md_docs.md)
- [`python_tree_views.h_docs.md_docs.md`](./python_tree_views.h_docs.md_docs.md)
- [`opaque_obj.h_docs.md_docs.md`](./opaque_obj.h_docs.md_docs.md)
- [`python_custom_class.cpp_docs.md_docs.md`](./python_custom_class.cpp_docs.md_docs.md)
- [`python_tracer.cpp_kw.md_docs.md`](./python_tracer.cpp_kw.md_docs.md)
- [`python_interpreter.cpp_kw.md_docs.md`](./python_interpreter.cpp_kw.md_docs.md)
- [`python_tracer.cpp_docs.md_docs.md`](./python_tracer.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `init.cpp_docs.md_docs.md`
- **Keyword Index**: `init.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
