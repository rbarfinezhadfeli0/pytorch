# Index: `torch/_dynamo/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `torch/_dynamo/`

## Subfolders

- [`backends/`](./backends/index.md) - backends module
- [`polyfills/`](./polyfills/index.md) - polyfills module
- [`repro/`](./repro/index.md) - repro module
- [`variables/`](./variables/index.md) - variables module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`__init__.py`](../../../torch/_dynamo/__init__.py) | Package initialization | [docs](./__init__.py_docs.md) | [keywords](./__init__.py_kw.md) |
| [`_trace_wrapped_higher_order_op.py`](../../../torch/_dynamo/_trace_wrapped_higher_order_op.py) | Source code | [docs](./_trace_wrapped_higher_order_op.py_docs.md) | [keywords](./_trace_wrapped_higher_order_op.py_kw.md) |
| [`aot_compile.py`](../../../torch/_dynamo/aot_compile.py) | Source code | [docs](./aot_compile.py_docs.md) | [keywords](./aot_compile.py_kw.md) |
| [`aot_compile_types.py`](../../../torch/_dynamo/aot_compile_types.py) | Source code | [docs](./aot_compile_types.py_docs.md) | [keywords](./aot_compile_types.py_kw.md) |
| [`bytecode_analysis.py`](../../../torch/_dynamo/bytecode_analysis.py) | Source code | [docs](./bytecode_analysis.py_docs.md) | [keywords](./bytecode_analysis.py_kw.md) |
| [`bytecode_transformation.py`](../../../torch/_dynamo/bytecode_transformation.py) | Source code | [docs](./bytecode_transformation.py_docs.md) | [keywords](./bytecode_transformation.py_kw.md) |
| [`cache_size.py`](../../../torch/_dynamo/cache_size.py) | Source code | [docs](./cache_size.py_docs.md) | [keywords](./cache_size.py_kw.md) |
| [`callback.py`](../../../torch/_dynamo/callback.py) | Source code | [docs](./callback.py_docs.md) | [keywords](./callback.py_kw.md) |
| [`code_context.py`](../../../torch/_dynamo/code_context.py) | Source code | [docs](./code_context.py_docs.md) | [keywords](./code_context.py_kw.md) |
| [`codegen.py`](../../../torch/_dynamo/codegen.py) | Source code | [docs](./codegen.py_docs.md) | [keywords](./codegen.py_kw.md) |
| [`compiled_autograd.py`](../../../torch/_dynamo/compiled_autograd.py) | Source code | [docs](./compiled_autograd.py_docs.md) | [keywords](./compiled_autograd.py_kw.md) |
| [`comptime.py`](../../../torch/_dynamo/comptime.py) | Source code | [docs](./comptime.py_docs.md) | [keywords](./comptime.py_kw.md) |
| [`config.py`](../../../torch/_dynamo/config.py) | Source code | [docs](./config.py_docs.md) | [keywords](./config.py_kw.md) |
| [`convert_frame.py`](../../../torch/_dynamo/convert_frame.py) | Source code | [docs](./convert_frame.py_docs.md) | [keywords](./convert_frame.py_kw.md) |
| [`create_parameter_op.py`](../../../torch/_dynamo/create_parameter_op.py) | Source code | [docs](./create_parameter_op.py_docs.md) | [keywords](./create_parameter_op.py_kw.md) |
| [`current_scope_id.py`](../../../torch/_dynamo/current_scope_id.py) | Source code | [docs](./current_scope_id.py_docs.md) | [keywords](./current_scope_id.py_kw.md) |
| [`debug_utils.py`](../../../torch/_dynamo/debug_utils.py) | Source code | [docs](./debug_utils.py_docs.md) | [keywords](./debug_utils.py_kw.md) |
| [`decorators.py`](../../../torch/_dynamo/decorators.py) | Source code | [docs](./decorators.py_docs.md) | [keywords](./decorators.py_kw.md) |
| [`device_interface.py`](../../../torch/_dynamo/device_interface.py) | Source code | [docs](./device_interface.py_docs.md) | [keywords](./device_interface.py_kw.md) |
| [`distributed.py`](../../../torch/_dynamo/distributed.py) | Source code | [docs](./distributed.py_docs.md) | [keywords](./distributed.py_kw.md) |
| [`eval_frame.py`](../../../torch/_dynamo/eval_frame.py) | Source code | [docs](./eval_frame.py_docs.md) | [keywords](./eval_frame.py_kw.md) |
| [`exc.py`](../../../torch/_dynamo/exc.py) | Source code | [docs](./exc.py_docs.md) | [keywords](./exc.py_kw.md) |
| [`external_utils.py`](../../../torch/_dynamo/external_utils.py) | Source code | [docs](./external_utils.py_docs.md) | [keywords](./external_utils.py_kw.md) |
| [`funcname_cache.py`](../../../torch/_dynamo/funcname_cache.py) | Source code | [docs](./funcname_cache.py_docs.md) | [keywords](./funcname_cache.py_kw.md) |
| [`functional_export.py`](../../../torch/_dynamo/functional_export.py) | Source code | [docs](./functional_export.py_docs.md) | [keywords](./functional_export.py_kw.md) |
| [`graph_break_hints.py`](../../../torch/_dynamo/graph_break_hints.py) | Source code | [docs](./graph_break_hints.py_docs.md) | [keywords](./graph_break_hints.py_kw.md) |
| [`graph_break_registry.json`](../../../torch/_dynamo/graph_break_registry.json) | Source code | [docs](./graph_break_registry.json_docs.md) | [keywords](./graph_break_registry.json_kw.md) |
| [`graph_bytecode_inputs.py`](../../../torch/_dynamo/graph_bytecode_inputs.py) | Source code | [docs](./graph_bytecode_inputs.py_docs.md) | [keywords](./graph_bytecode_inputs.py_kw.md) |
| [`graph_deduplication.py`](../../../torch/_dynamo/graph_deduplication.py) | Source code | [docs](./graph_deduplication.py_docs.md) | [keywords](./graph_deduplication.py_kw.md) |
| [`graph_region_tracker.py`](../../../torch/_dynamo/graph_region_tracker.py) | Source code | [docs](./graph_region_tracker.py_docs.md) | [keywords](./graph_region_tracker.py_kw.md) |
| [`graph_utils.py`](../../../torch/_dynamo/graph_utils.py) | Source code | [docs](./graph_utils.py_docs.md) | [keywords](./graph_utils.py_kw.md) |
| [`guards.py`](../../../torch/_dynamo/guards.py) | Source code | [docs](./guards.py_docs.md) | [keywords](./guards.py_kw.md) |
| [`hooks.py`](../../../torch/_dynamo/hooks.py) | Source code | [docs](./hooks.py_docs.md) | [keywords](./hooks.py_kw.md) |
| [`logging.py`](../../../torch/_dynamo/logging.py) | Source code | [docs](./logging.py_docs.md) | [keywords](./logging.py_kw.md) |
| [`metrics_context.py`](../../../torch/_dynamo/metrics_context.py) | Source code | [docs](./metrics_context.py_docs.md) | [keywords](./metrics_context.py_kw.md) |
| [`mutation_guard.py`](../../../torch/_dynamo/mutation_guard.py) | Source code | [docs](./mutation_guard.py_docs.md) | [keywords](./mutation_guard.py_kw.md) |
| [`output_graph.py`](../../../torch/_dynamo/output_graph.py) | Source code | [docs](./output_graph.py_docs.md) | [keywords](./output_graph.py_kw.md) |
| [`package.py`](../../../torch/_dynamo/package.py) | Source code | [docs](./package.py_docs.md) | [keywords](./package.py_kw.md) |
| [`pgo.py`](../../../torch/_dynamo/pgo.py) | Source code | [docs](./pgo.py_docs.md) | [keywords](./pgo.py_kw.md) |
| [`precompile_context.py`](../../../torch/_dynamo/precompile_context.py) | Source code | [docs](./precompile_context.py_docs.md) | [keywords](./precompile_context.py_kw.md) |
| [`profiler.py`](../../../torch/_dynamo/profiler.py) | Source code | [docs](./profiler.py_docs.md) | [keywords](./profiler.py_kw.md) |
| [`replay_record.py`](../../../torch/_dynamo/replay_record.py) | Source code | [docs](./replay_record.py_docs.md) | [keywords](./replay_record.py_kw.md) |
| [`resume_execution.py`](../../../torch/_dynamo/resume_execution.py) | Source code | [docs](./resume_execution.py_docs.md) | [keywords](./resume_execution.py_kw.md) |
| [`side_effects.py`](../../../torch/_dynamo/side_effects.py) | Source code | [docs](./side_effects.py_docs.md) | [keywords](./side_effects.py_kw.md) |
| [`source.py`](../../../torch/_dynamo/source.py) | Source code | [docs](./source.py_docs.md) | [keywords](./source.py_kw.md) |
| [`symbolic_convert.py`](../../../torch/_dynamo/symbolic_convert.py) | Source code | [docs](./symbolic_convert.py_docs.md) | [keywords](./symbolic_convert.py_kw.md) |
| [`tensor_version_op.py`](../../../torch/_dynamo/tensor_version_op.py) | Source code | [docs](./tensor_version_op.py_docs.md) | [keywords](./tensor_version_op.py_kw.md) |
| [`test_case.py`](../../../torch/_dynamo/test_case.py) | Test file | [docs](./test_case.py_docs.md) | [keywords](./test_case.py_kw.md) |
| [`test_dont_skip_tracing_functions.py`](../../../torch/_dynamo/test_dont_skip_tracing_functions.py) | Test file | [docs](./test_dont_skip_tracing_functions.py_docs.md) | [keywords](./test_dont_skip_tracing_functions.py_kw.md) |
| [`test_minifier_common.py`](../../../torch/_dynamo/test_minifier_common.py) | Test file | [docs](./test_minifier_common.py_docs.md) | [keywords](./test_minifier_common.py_kw.md) |
| [`testing.py`](../../../torch/_dynamo/testing.py) | Source code | [docs](./testing.py_docs.md) | [keywords](./testing.py_kw.md) |
| [`trace_rules.py`](../../../torch/_dynamo/trace_rules.py) | Source code | [docs](./trace_rules.py_docs.md) | [keywords](./trace_rules.py_kw.md) |
| [`types.py`](../../../torch/_dynamo/types.py) | Source code | [docs](./types.py_docs.md) | [keywords](./types.py_kw.md) |
| [`utils.py`](../../../torch/_dynamo/utils.py) | Source code | [docs](./utils.py_docs.md) | [keywords](./utils.py_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
