# Index: `torch/_inductor/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `torch/_inductor/`

## Subfolders

- [`analysis/`](./analysis/index.md) - analysis module
- [`autoheuristic/`](./autoheuristic/index.md) - autoheuristic module
- [`codegen/`](./codegen/index.md) - codegen module
- [`compile_worker/`](./compile_worker/index.md) - compile_worker module
- [`fx_passes/`](./fx_passes/index.md) - fx_passes module
- [`kernel/`](./kernel/index.md) - kernel module
- [`lookup_table/`](./lookup_table/index.md) - lookup_table module
- [`package/`](./package/index.md) - package module
- [`runtime/`](./runtime/index.md) - runtime module
- [`template_heuristics/`](./template_heuristics/index.md) - template_heuristics module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`__autotune_main__.py`](../../../torch/_inductor/__autotune_main__.py) | Source code | [docs](./__autotune_main__.py_docs.md) | [keywords](./__autotune_main__.py_kw.md) |
| [`__init__.py`](../../../torch/_inductor/__init__.py) | Package initialization | [docs](./__init__.py_docs.md) | [keywords](./__init__.py_kw.md) |
| [`analyze_preserves_zero_mask.py`](../../../torch/_inductor/analyze_preserves_zero_mask.py) | Source code | [docs](./analyze_preserves_zero_mask.py_docs.md) | [keywords](./analyze_preserves_zero_mask.py_kw.md) |
| [`aoti_eager.py`](../../../torch/_inductor/aoti_eager.py) | Source code | [docs](./aoti_eager.py_docs.md) | [keywords](./aoti_eager.py_kw.md) |
| [`async_compile.py`](../../../torch/_inductor/async_compile.py) | Source code | [docs](./async_compile.py_docs.md) | [keywords](./async_compile.py_kw.md) |
| [`augmented_graph_helper.py`](../../../torch/_inductor/augmented_graph_helper.py) | Source code | [docs](./augmented_graph_helper.py_docs.md) | [keywords](./augmented_graph_helper.py_kw.md) |
| [`autotune_process.py`](../../../torch/_inductor/autotune_process.py) | Source code | [docs](./autotune_process.py_docs.md) | [keywords](./autotune_process.py_kw.md) |
| [`await_utils.py`](../../../torch/_inductor/await_utils.py) | Source code | [docs](./await_utils.py_docs.md) | [keywords](./await_utils.py_kw.md) |
| [`bounds.py`](../../../torch/_inductor/bounds.py) | Source code | [docs](./bounds.py_docs.md) | [keywords](./bounds.py_kw.md) |
| [`cache.py`](../../../torch/_inductor/cache.py) | Source code | [docs](./cache.py_docs.md) | [keywords](./cache.py_kw.md) |
| [`choices.py`](../../../torch/_inductor/choices.py) | Source code | [docs](./choices.py_docs.md) | [keywords](./choices.py_kw.md) |
| [`codecache.py`](../../../torch/_inductor/codecache.py) | Source code | [docs](./codecache.py_docs.md) | [keywords](./codecache.py_kw.md) |
| [`comm_analysis.py`](../../../torch/_inductor/comm_analysis.py) | Source code | [docs](./comm_analysis.py_docs.md) | [keywords](./comm_analysis.py_kw.md) |
| [`comm_lowering.py`](../../../torch/_inductor/comm_lowering.py) | Source code | [docs](./comm_lowering.py_docs.md) | [keywords](./comm_lowering.py_kw.md) |
| [`comms.py`](../../../torch/_inductor/comms.py) | Source code | [docs](./comms.py_docs.md) | [keywords](./comms.py_kw.md) |
| [`comms_debug.py`](../../../torch/_inductor/comms_debug.py) | Source code | [docs](./comms_debug.py_docs.md) | [keywords](./comms_debug.py_kw.md) |
| [`compile_fx.py`](../../../torch/_inductor/compile_fx.py) | Source code | [docs](./compile_fx.py_docs.md) | [keywords](./compile_fx.py_kw.md) |
| [`compile_fx_async.py`](../../../torch/_inductor/compile_fx_async.py) | Source code | [docs](./compile_fx_async.py_docs.md) | [keywords](./compile_fx_async.py_kw.md) |
| [`compile_fx_ext.py`](../../../torch/_inductor/compile_fx_ext.py) | Source code | [docs](./compile_fx_ext.py_docs.md) | [keywords](./compile_fx_ext.py_kw.md) |
| [`compile_fx_subproc.py`](../../../torch/_inductor/compile_fx_subproc.py) | Source code | [docs](./compile_fx_subproc.py_docs.md) | [keywords](./compile_fx_subproc.py_kw.md) |
| [`compiler_bisector.py`](../../../torch/_inductor/compiler_bisector.py) | Source code | [docs](./compiler_bisector.py_docs.md) | [keywords](./compiler_bisector.py_kw.md) |
| [`config.py`](../../../torch/_inductor/config.py) | Source code | [docs](./config.py_docs.md) | [keywords](./config.py_kw.md) |
| [`config_comms.py`](../../../torch/_inductor/config_comms.py) | Source code | [docs](./config_comms.py_docs.md) | [keywords](./config_comms.py_kw.md) |
| [`constant_folding.py`](../../../torch/_inductor/constant_folding.py) | Source code | [docs](./constant_folding.py_docs.md) | [keywords](./constant_folding.py_kw.md) |
| [`cpp_builder.py`](../../../torch/_inductor/cpp_builder.py) | Source code | [docs](./cpp_builder.py_docs.md) | [keywords](./cpp_builder.py_kw.md) |
| [`cpu_vec_isa.py`](../../../torch/_inductor/cpu_vec_isa.py) | Source code | [docs](./cpu_vec_isa.py_docs.md) | [keywords](./cpu_vec_isa.py_kw.md) |
| [`cudagraph_trees.py`](../../../torch/_inductor/cudagraph_trees.py) | Source code | [docs](./cudagraph_trees.py_docs.md) | [keywords](./cudagraph_trees.py_kw.md) |
| [`cudagraph_utils.py`](../../../torch/_inductor/cudagraph_utils.py) | Source code | [docs](./cudagraph_utils.py_docs.md) | [keywords](./cudagraph_utils.py_kw.md) |
| [`custom_graph_pass.py`](../../../torch/_inductor/custom_graph_pass.py) | Source code | [docs](./custom_graph_pass.py_docs.md) | [keywords](./custom_graph_pass.py_kw.md) |
| [`debug.py`](../../../torch/_inductor/debug.py) | Source code | [docs](./debug.py_docs.md) | [keywords](./debug.py_kw.md) |
| [`decomposition.py`](../../../torch/_inductor/decomposition.py) | Source code | [docs](./decomposition.py_docs.md) | [keywords](./decomposition.py_kw.md) |
| [`dependencies.py`](../../../torch/_inductor/dependencies.py) | Source code | [docs](./dependencies.py_docs.md) | [keywords](./dependencies.py_kw.md) |
| [`distributed_autotune.py`](../../../torch/_inductor/distributed_autotune.py) | Source code | [docs](./distributed_autotune.py_docs.md) | [keywords](./distributed_autotune.py_kw.md) |
| [`dtype_propagation.py`](../../../torch/_inductor/dtype_propagation.py) | Source code | [docs](./dtype_propagation.py_docs.md) | [keywords](./dtype_propagation.py_kw.md) |
| [`exc.py`](../../../torch/_inductor/exc.py) | Source code | [docs](./exc.py_docs.md) | [keywords](./exc.py_kw.md) |
| [`extern_node_serializer.py`](../../../torch/_inductor/extern_node_serializer.py) | Source code | [docs](./extern_node_serializer.py_docs.md) | [keywords](./extern_node_serializer.py_kw.md) |
| [`freezing.py`](../../../torch/_inductor/freezing.py) | Source code | [docs](./freezing.py_docs.md) | [keywords](./freezing.py_kw.md) |
| [`freezing_utils.py`](../../../torch/_inductor/freezing_utils.py) | Source code | [docs](./freezing_utils.py_docs.md) | [keywords](./freezing_utils.py_kw.md) |
| [`fuzzer.py`](../../../torch/_inductor/fuzzer.py) | Source code | [docs](./fuzzer.py_docs.md) | [keywords](./fuzzer.py_kw.md) |
| [`fx_utils.py`](../../../torch/_inductor/fx_utils.py) | Source code | [docs](./fx_utils.py_docs.md) | [keywords](./fx_utils.py_kw.md) |
| [`graph.py`](../../../torch/_inductor/graph.py) | Source code | [docs](./graph.py_docs.md) | [keywords](./graph.py_kw.md) |
| [`hooks.py`](../../../torch/_inductor/hooks.py) | Source code | [docs](./hooks.py_docs.md) | [keywords](./hooks.py_kw.md) |
| [`index_propagation.py`](../../../torch/_inductor/index_propagation.py) | Source code | [docs](./index_propagation.py_docs.md) | [keywords](./index_propagation.py_kw.md) |
| [`inductor_prims.py`](../../../torch/_inductor/inductor_prims.py) | Source code | [docs](./inductor_prims.py_docs.md) | [keywords](./inductor_prims.py_kw.md) |
| [`invert_expr_analysis.py`](../../../torch/_inductor/invert_expr_analysis.py) | Source code | [docs](./invert_expr_analysis.py_docs.md) | [keywords](./invert_expr_analysis.py_kw.md) |
| [`ir.py`](../../../torch/_inductor/ir.py) | Source code | [docs](./ir.py_docs.md) | [keywords](./ir.py_kw.md) |
| [`jagged_lowerings.py`](../../../torch/_inductor/jagged_lowerings.py) | Source code | [docs](./jagged_lowerings.py_docs.md) | [keywords](./jagged_lowerings.py_kw.md) |
| [`kernel_inputs.py`](../../../torch/_inductor/kernel_inputs.py) | Source code | [docs](./kernel_inputs.py_docs.md) | [keywords](./kernel_inputs.py_kw.md) |
| [`kernel_template_choice.py`](../../../torch/_inductor/kernel_template_choice.py) | Source code | [docs](./kernel_template_choice.py_docs.md) | [keywords](./kernel_template_choice.py_kw.md) |
| [`loop_body.py`](../../../torch/_inductor/loop_body.py) | Source code | [docs](./loop_body.py_docs.md) | [keywords](./loop_body.py_kw.md) |
| [`lowering.py`](../../../torch/_inductor/lowering.py) | Source code | [docs](./lowering.py_docs.md) | [keywords](./lowering.py_kw.md) |
| [`memory.py`](../../../torch/_inductor/memory.py) | Source code | [docs](./memory.py_docs.md) | [keywords](./memory.py_kw.md) |
| [`metrics.py`](../../../torch/_inductor/metrics.py) | Source code | [docs](./metrics.py_docs.md) | [keywords](./metrics.py_kw.md) |
| [`mkldnn_ir.py`](../../../torch/_inductor/mkldnn_ir.py) | Source code | [docs](./mkldnn_ir.py_docs.md) | [keywords](./mkldnn_ir.py_kw.md) |
| [`mkldnn_lowerings.py`](../../../torch/_inductor/mkldnn_lowerings.py) | Source code | [docs](./mkldnn_lowerings.py_docs.md) | [keywords](./mkldnn_lowerings.py_kw.md) |
| [`mock_cache.py`](../../../torch/_inductor/mock_cache.py) | Source code | [docs](./mock_cache.py_docs.md) | [keywords](./mock_cache.py_kw.md) |
| [`ops_handler.py`](../../../torch/_inductor/ops_handler.py) | Source code | [docs](./ops_handler.py_docs.md) | [keywords](./ops_handler.py_kw.md) |
| [`optimize_indexing.py`](../../../torch/_inductor/optimize_indexing.py) | Source code | [docs](./optimize_indexing.py_docs.md) | [keywords](./optimize_indexing.py_kw.md) |
| [`output_code.py`](../../../torch/_inductor/output_code.py) | Source code | [docs](./output_code.py_docs.md) | [keywords](./output_code.py_kw.md) |
| [`pattern_matcher.py`](../../../torch/_inductor/pattern_matcher.py) | Source code | [docs](./pattern_matcher.py_docs.md) | [keywords](./pattern_matcher.py_kw.md) |
| [`quantized_lowerings.py`](../../../torch/_inductor/quantized_lowerings.py) | Source code | [docs](./quantized_lowerings.py_docs.md) | [keywords](./quantized_lowerings.py_kw.md) |
| [`remote_cache.py`](../../../torch/_inductor/remote_cache.py) | Source code | [docs](./remote_cache.py_docs.md) | [keywords](./remote_cache.py_kw.md) |
| [`remote_gemm_autotune_cache.py`](../../../torch/_inductor/remote_gemm_autotune_cache.py) | Source code | [docs](./remote_gemm_autotune_cache.py_docs.md) | [keywords](./remote_gemm_autotune_cache.py_kw.md) |
| [`rocm_multiarch_utils.py`](../../../torch/_inductor/rocm_multiarch_utils.py) | Source code | [docs](./rocm_multiarch_utils.py_docs.md) | [keywords](./rocm_multiarch_utils.py_kw.md) |
| [`scheduler.py`](../../../torch/_inductor/scheduler.py) | Source code | [docs](./scheduler.py_docs.md) | [keywords](./scheduler.py_kw.md) |
| [`select_algorithm.py`](../../../torch/_inductor/select_algorithm.py) | Source code | [docs](./select_algorithm.py_docs.md) | [keywords](./select_algorithm.py_kw.md) |
| [`shape_propagation.py`](../../../torch/_inductor/shape_propagation.py) | Source code | [docs](./shape_propagation.py_docs.md) | [keywords](./shape_propagation.py_kw.md) |
| [`sizevars.py`](../../../torch/_inductor/sizevars.py) | Source code | [docs](./sizevars.py_docs.md) | [keywords](./sizevars.py_kw.md) |
| [`standalone_compile.py`](../../../torch/_inductor/standalone_compile.py) | Source code | [docs](./standalone_compile.py_docs.md) | [keywords](./standalone_compile.py_kw.md) |
| [`subgraph_lowering.py`](../../../torch/_inductor/subgraph_lowering.py) | Source code | [docs](./subgraph_lowering.py_docs.md) | [keywords](./subgraph_lowering.py_kw.md) |
| [`test_case.py`](../../../torch/_inductor/test_case.py) | Test file | [docs](./test_case.py_docs.md) | [keywords](./test_case.py_kw.md) |
| [`test_operators.py`](../../../torch/_inductor/test_operators.py) | Test file | [docs](./test_operators.py_docs.md) | [keywords](./test_operators.py_kw.md) |
| [`tiling_utils.py`](../../../torch/_inductor/tiling_utils.py) | Source code | [docs](./tiling_utils.py_docs.md) | [keywords](./tiling_utils.py_kw.md) |
| [`triton_bundler.py`](../../../torch/_inductor/triton_bundler.py) | Source code | [docs](./triton_bundler.py_docs.md) | [keywords](./triton_bundler.py_kw.md) |
| [`utils.py`](../../../torch/_inductor/utils.py) | Source code | [docs](./utils.py_docs.md) | [keywords](./utils.py_kw.md) |
| [`virtualized.py`](../../../torch/_inductor/virtualized.py) | Source code | [docs](./virtualized.py_docs.md) | [keywords](./virtualized.py_kw.md) |
| [`wrapper_benchmark.py`](../../../torch/_inductor/wrapper_benchmark.py) | Source code | [docs](./wrapper_benchmark.py_docs.md) | [keywords](./wrapper_benchmark.py_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
