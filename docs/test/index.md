# Index: `test/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `test/`

## Subfolders

- [`ao/`](./ao/index.md) - ao module
- [`autograd/`](./autograd/index.md) - autograd module
- [`backends/`](./backends/index.md) - backends module
- [`benchmark_utils/`](./benchmark_utils/index.md) - benchmark_utils module
- [`compiled_autograd_skips/`](./compiled_autograd_skips/index.md) - compiled_autograd_skips module
- [`cpp/`](./cpp/index.md) - cpp module
- [`cpp_api_parity/`](./cpp_api_parity/index.md) - cpp_api_parity module
- [`cpp_extensions/`](./cpp_extensions/index.md) - cpp_extensions module
- [`custom_backend/`](./custom_backend/index.md) - custom_backend module
- [`custom_operator/`](./custom_operator/index.md) - custom_operator module
- [`distributed/`](./distributed/index.md) - distributed module
- [`distributions/`](./distributions/index.md) - distributions module
- [`dynamo/`](./dynamo/index.md) - dynamo module
- [`dynamo_expected_failures/`](./dynamo_expected_failures/index.md) - dynamo_expected_failures module
- [`dynamo_skips/`](./dynamo_skips/index.md) - dynamo_skips module
- [`error_messages/`](./error_messages/index.md) - error_messages module
- [`expect/`](./expect/index.md) - expect module
- [`export/`](./export/index.md) - export module
- [`forward_backward_compatibility/`](./forward_backward_compatibility/index.md) - forward_backward_compatibility module
- [`functorch/`](./functorch/index.md) - functorch module
- [`fx/`](./fx/index.md) - fx module
- [`higher_order_ops/`](./higher_order_ops/index.md) - higher_order_ops module
- [`inductor/`](./inductor/index.md) - inductor module
- [`inductor_expected_failures/`](./inductor_expected_failures/index.md) - inductor_expected_failures module
- [`inductor_skips/`](./inductor_skips/index.md) - inductor_skips module
- [`jit/`](./jit/index.md) - jit module
- [`jit_hooks/`](./jit_hooks/index.md) - jit_hooks module
- [`lazy/`](./lazy/index.md) - lazy module
- [`mobile/`](./mobile/index.md) - mobile module
- [`nn/`](./nn/index.md) - nn module
- [`onnx/`](./onnx/index.md) - onnx module
- [`optim/`](./optim/index.md) - optim module
- [`package/`](./package/index.md) - package module
- [`profiler/`](./profiler/index.md) - profiler module
- [`quantization/`](./quantization/index.md) - quantization module
- [`scripts/`](./scripts/index.md) - Utility scripts
- [`strobelight/`](./strobelight/index.md) - strobelight module
- [`test_img/`](./test_img/index.md) - test_img module
- [`torch_np/`](./torch_np/index.md) - torch_np module
- [`typing/`](./typing/index.md) - typing module
- [`xpu/`](./xpu/index.md) - xpu module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`HowToWriteTestsUsingFileCheck.md`](../../test/HowToWriteTestsUsingFileCheck.md) | Documentation | [docs](./HowToWriteTestsUsingFileCheck.md_docs.md) | [keywords](./HowToWriteTestsUsingFileCheck.md_kw.md) |
| [`_test_bazel.py`](../../test/_test_bazel.py) | Source code | [docs](./_test_bazel.py_docs.md) | [keywords](./_test_bazel.py_kw.md) |
| [`allowlist_for_publicAPI.json`](../../test/allowlist_for_publicAPI.json) | Source code | [docs](./allowlist_for_publicAPI.json_docs.md) | [keywords](./allowlist_for_publicAPI.json_kw.md) |
| [`bench_mps_ops.py`](../../test/bench_mps_ops.py) | Source code | [docs](./bench_mps_ops.py_docs.md) | [keywords](./bench_mps_ops.py_kw.md) |
| [`conftest.py`](../../test/conftest.py) | Source code | [docs](./conftest.py_docs.md) | [keywords](./conftest.py_kw.md) |
| [`create_dummy_torchscript_model.py`](../../test/create_dummy_torchscript_model.py) | Source code | [docs](./create_dummy_torchscript_model.py_docs.md) | [keywords](./create_dummy_torchscript_model.py_kw.md) |
| [`linear.py`](../../test/linear.py) | Source code | [docs](./linear.py_docs.md) | [keywords](./linear.py_kw.md) |
| [`load_torchscript_model.py`](../../test/load_torchscript_model.py) | Source code | [docs](./load_torchscript_model.py_docs.md) | [keywords](./load_torchscript_model.py_kw.md) |
| [`minioptest_failures_dict.json`](../../test/minioptest_failures_dict.json) | Source code | [docs](./minioptest_failures_dict.json_docs.md) | [keywords](./minioptest_failures_dict.json_kw.md) |
| [`mkl_verbose.py`](../../test/mkl_verbose.py) | Source code | [docs](./mkl_verbose.py_docs.md) | [keywords](./mkl_verbose.py_kw.md) |
| [`mkldnn_verbose.py`](../../test/mkldnn_verbose.py) | Source code | [docs](./mkldnn_verbose.py_docs.md) | [keywords](./mkldnn_verbose.py_kw.md) |
| [`pytest_shard_custom.py`](../../test/pytest_shard_custom.py) | Source code | [docs](./pytest_shard_custom.py_docs.md) | [keywords](./pytest_shard_custom.py_kw.md) |
| [`run_doctests.sh`](../../test/run_doctests.sh) | Source code | [docs](./run_doctests.sh_docs.md) | [keywords](./run_doctests.sh_kw.md) |
| [`run_test.py`](../../test/run_test.py) | Source code | [docs](./run_test.py_docs.md) | [keywords](./run_test.py_kw.md) |
| [`simulate_nccl_errors.py`](../../test/simulate_nccl_errors.py) | Source code | [docs](./simulate_nccl_errors.py_docs.md) | [keywords](./simulate_nccl_errors.py_kw.md) |
| [`slow_tests.json`](../../test/slow_tests.json) | Source code | [docs](./slow_tests.json_docs.md) | [keywords](./slow_tests.json_kw.md) |
| [`test_accelerator.py`](../../test/test_accelerator.py) | Test file | [docs](./test_accelerator.py_docs.md) | [keywords](./test_accelerator.py_kw.md) |
| [`test_ao_sparsity.py`](../../test/test_ao_sparsity.py) | Test file | [docs](./test_ao_sparsity.py_docs.md) | [keywords](./test_ao_sparsity.py_kw.md) |
| [`test_appending_byte_serializer.py`](../../test/test_appending_byte_serializer.py) | Test file | [docs](./test_appending_byte_serializer.py_docs.md) | [keywords](./test_appending_byte_serializer.py_kw.md) |
| [`test_as_strided.py`](../../test/test_as_strided.py) | Test file | [docs](./test_as_strided.py_docs.md) | [keywords](./test_as_strided.py_kw.md) |
| [`test_autocast.py`](../../test/test_autocast.py) | Test file | [docs](./test_autocast.py_docs.md) | [keywords](./test_autocast.py_kw.md) |
| [`test_autograd.py`](../../test/test_autograd.py) | Test file | [docs](./test_autograd.py_docs.md) | [keywords](./test_autograd.py_kw.md) |
| [`test_autograd_fallback.py`](../../test/test_autograd_fallback.py) | Test file | [docs](./test_autograd_fallback.py_docs.md) | [keywords](./test_autograd_fallback.py_kw.md) |
| [`test_autoload.py`](../../test/test_autoload.py) | Test file | [docs](./test_autoload.py_docs.md) | [keywords](./test_autoload.py_kw.md) |
| [`test_binary_ufuncs.py`](../../test/test_binary_ufuncs.py) | Test file | [docs](./test_binary_ufuncs.py_docs.md) | [keywords](./test_binary_ufuncs.py_kw.md) |
| [`test_bundled_images.py`](../../test/test_bundled_images.py) | Test file | [docs](./test_bundled_images.py_docs.md) | [keywords](./test_bundled_images.py_kw.md) |
| [`test_bundled_inputs.py`](../../test/test_bundled_inputs.py) | Test file | [docs](./test_bundled_inputs.py_docs.md) | [keywords](./test_bundled_inputs.py_kw.md) |
| [`test_ci_sanity_check_fail.py`](../../test/test_ci_sanity_check_fail.py) | Test file | [docs](./test_ci_sanity_check_fail.py_docs.md) | [keywords](./test_ci_sanity_check_fail.py_kw.md) |
| [`test_comparison_utils.py`](../../test/test_comparison_utils.py) | Test file | [docs](./test_comparison_utils.py_docs.md) | [keywords](./test_comparison_utils.py_kw.md) |
| [`test_compile_benchmark_util.py`](../../test/test_compile_benchmark_util.py) | Test file | [docs](./test_compile_benchmark_util.py_docs.md) | [keywords](./test_compile_benchmark_util.py_kw.md) |
| [`test_complex.py`](../../test/test_complex.py) | Test file | [docs](./test_complex.py_docs.md) | [keywords](./test_complex.py_kw.md) |
| [`test_content_store.py`](../../test/test_content_store.py) | Test file | [docs](./test_content_store.py_docs.md) | [keywords](./test_content_store.py_kw.md) |
| [`test_cpp_api_parity.py`](../../test/test_cpp_api_parity.py) | Test file | [docs](./test_cpp_api_parity.py_docs.md) | [keywords](./test_cpp_api_parity.py_kw.md) |
| [`test_cpp_extensions_aot.py`](../../test/test_cpp_extensions_aot.py) | Test file | [docs](./test_cpp_extensions_aot.py_docs.md) | [keywords](./test_cpp_extensions_aot.py_kw.md) |
| [`test_cpp_extensions_jit.py`](../../test/test_cpp_extensions_jit.py) | Test file | [docs](./test_cpp_extensions_jit.py_docs.md) | [keywords](./test_cpp_extensions_jit.py_kw.md) |
| [`test_cpp_extensions_mtia_backend.py`](../../test/test_cpp_extensions_mtia_backend.py) | Test file | [docs](./test_cpp_extensions_mtia_backend.py_docs.md) | [keywords](./test_cpp_extensions_mtia_backend.py_kw.md) |
| [`test_cpp_extensions_stream_and_event.py`](../../test/test_cpp_extensions_stream_and_event.py) | Test file | [docs](./test_cpp_extensions_stream_and_event.py_docs.md) | [keywords](./test_cpp_extensions_stream_and_event.py_kw.md) |
| [`test_cuda.py`](../../test/test_cuda.py) | Test file | [docs](./test_cuda.py_docs.md) | [keywords](./test_cuda.py_kw.md) |
| [`test_cuda_expandable_segments.py`](../../test/test_cuda_expandable_segments.py) | Test file | [docs](./test_cuda_expandable_segments.py_docs.md) | [keywords](./test_cuda_expandable_segments.py_kw.md) |
| [`test_cuda_multigpu.py`](../../test/test_cuda_multigpu.py) | Test file | [docs](./test_cuda_multigpu.py_docs.md) | [keywords](./test_cuda_multigpu.py_kw.md) |
| [`test_cuda_nvml_based_avail.py`](../../test/test_cuda_nvml_based_avail.py) | Test file | [docs](./test_cuda_nvml_based_avail.py_docs.md) | [keywords](./test_cuda_nvml_based_avail.py_kw.md) |
| [`test_cuda_primary_ctx.py`](../../test/test_cuda_primary_ctx.py) | Test file | [docs](./test_cuda_primary_ctx.py_docs.md) | [keywords](./test_cuda_primary_ctx.py_kw.md) |
| [`test_cuda_sanitizer.py`](../../test/test_cuda_sanitizer.py) | Test file | [docs](./test_cuda_sanitizer.py_docs.md) | [keywords](./test_cuda_sanitizer.py_kw.md) |
| [`test_cuda_trace.py`](../../test/test_cuda_trace.py) | Test file | [docs](./test_cuda_trace.py_docs.md) | [keywords](./test_cuda_trace.py_kw.md) |
| [`test_custom_ops.py`](../../test/test_custom_ops.py) | Test file | [docs](./test_custom_ops.py_docs.md) | [keywords](./test_custom_ops.py_kw.md) |
| [`test_dataloader.py`](../../test/test_dataloader.py) | Test file | [docs](./test_dataloader.py_docs.md) | [keywords](./test_dataloader.py_kw.md) |
| [`test_datapipe.py`](../../test/test_datapipe.py) | Test file | [docs](./test_datapipe.py_docs.md) | [keywords](./test_datapipe.py_kw.md) |
| [`test_decomp.py`](../../test/test_decomp.py) | Test file | [docs](./test_decomp.py_docs.md) | [keywords](./test_decomp.py_kw.md) |
| [`test_determination.py`](../../test/test_determination.py) | Test file | [docs](./test_determination.py_docs.md) | [keywords](./test_determination.py_kw.md) |
| [`test_dispatch.py`](../../test/test_dispatch.py) | Test file | [docs](./test_dispatch.py_docs.md) | [keywords](./test_dispatch.py_kw.md) |
| [`test_dlpack.py`](../../test/test_dlpack.py) | Test file | [docs](./test_dlpack.py_docs.md) | [keywords](./test_dlpack.py_kw.md) |
| [`test_dynamic_shapes.py`](../../test/test_dynamic_shapes.py) | Test file | [docs](./test_dynamic_shapes.py_docs.md) | [keywords](./test_dynamic_shapes.py_kw.md) |
| [`test_expanded_weights.py`](../../test/test_expanded_weights.py) | Test file | [docs](./test_expanded_weights.py_docs.md) | [keywords](./test_expanded_weights.py_kw.md) |
| [`test_extension_utils.py`](../../test/test_extension_utils.py) | Test file | [docs](./test_extension_utils.py_docs.md) | [keywords](./test_extension_utils.py_kw.md) |
| [`test_fake_tensor.py`](../../test/test_fake_tensor.py) | Test file | [docs](./test_fake_tensor.py_docs.md) | [keywords](./test_fake_tensor.py_kw.md) |
| [`test_file_check.py`](../../test/test_file_check.py) | Test file | [docs](./test_file_check.py_docs.md) | [keywords](./test_file_check.py_kw.md) |
| [`test_flop_counter.py`](../../test/test_flop_counter.py) | Test file | [docs](./test_flop_counter.py_docs.md) | [keywords](./test_flop_counter.py_kw.md) |
| [`test_foreach.py`](../../test/test_foreach.py) | Test file | [docs](./test_foreach.py_docs.md) | [keywords](./test_foreach.py_kw.md) |
| [`test_function_schema.py`](../../test/test_function_schema.py) | Test file | [docs](./test_function_schema.py_docs.md) | [keywords](./test_function_schema.py_kw.md) |
| [`test_functional_autograd_benchmark.py`](../../test/test_functional_autograd_benchmark.py) | Test file | [docs](./test_functional_autograd_benchmark.py_docs.md) | [keywords](./test_functional_autograd_benchmark.py_kw.md) |
| [`test_functional_optim.py`](../../test/test_functional_optim.py) | Test file | [docs](./test_functional_optim.py_docs.md) | [keywords](./test_functional_optim.py_kw.md) |
| [`test_functionalization.py`](../../test/test_functionalization.py) | Test file | [docs](./test_functionalization.py_docs.md) | [keywords](./test_functionalization.py_kw.md) |
| [`test_functionalization_of_rng_ops.py`](../../test/test_functionalization_of_rng_ops.py) | Test file | [docs](./test_functionalization_of_rng_ops.py_docs.md) | [keywords](./test_functionalization_of_rng_ops.py_kw.md) |
| [`test_futures.py`](../../test/test_futures.py) | Test file | [docs](./test_futures.py_docs.md) | [keywords](./test_futures.py_kw.md) |
| [`test_fx.py`](../../test/test_fx.py) | Test file | [docs](./test_fx.py_docs.md) | [keywords](./test_fx.py_kw.md) |
| [`test_fx_experimental.py`](../../test/test_fx_experimental.py) | Test file | [docs](./test_fx_experimental.py_docs.md) | [keywords](./test_fx_experimental.py_kw.md) |
| [`test_fx_passes.py`](../../test/test_fx_passes.py) | Test file | [docs](./test_fx_passes.py_docs.md) | [keywords](./test_fx_passes.py_kw.md) |
| [`test_fx_reinplace_pass.py`](../../test/test_fx_reinplace_pass.py) | Test file | [docs](./test_fx_reinplace_pass.py_docs.md) | [keywords](./test_fx_reinplace_pass.py_kw.md) |
| [`test_hop_infra.py`](../../test/test_hop_infra.py) | Test file | [docs](./test_hop_infra.py_docs.md) | [keywords](./test_hop_infra.py_kw.md) |
| [`test_hub.py`](../../test/test_hub.py) | Test file | [docs](./test_hub.py_docs.md) | [keywords](./test_hub.py_kw.md) |
| [`test_import_stats.py`](../../test/test_import_stats.py) | Test file | [docs](./test_import_stats.py_docs.md) | [keywords](./test_import_stats.py_kw.md) |
| [`test_indexing.py`](../../test/test_indexing.py) | Test file | [docs](./test_indexing.py_docs.md) | [keywords](./test_indexing.py_kw.md) |
| [`test_itt.py`](../../test/test_itt.py) | Test file | [docs](./test_itt.py_docs.md) | [keywords](./test_itt.py_kw.md) |
| [`test_jit.py`](../../test/test_jit.py) | Test file | [docs](./test_jit.py_docs.md) | [keywords](./test_jit.py_kw.md) |
| [`test_jit_autocast.py`](../../test/test_jit_autocast.py) | Test file | [docs](./test_jit_autocast.py_docs.md) | [keywords](./test_jit_autocast.py_kw.md) |
| [`test_jit_disabled.py`](../../test/test_jit_disabled.py) | Test file | [docs](./test_jit_disabled.py_docs.md) | [keywords](./test_jit_disabled.py_kw.md) |
| [`test_jit_fuser.py`](../../test/test_jit_fuser.py) | Test file | [docs](./test_jit_fuser.py_docs.md) | [keywords](./test_jit_fuser.py_kw.md) |
| [`test_jit_fuser_legacy.py`](../../test/test_jit_fuser_legacy.py) | Test file | [docs](./test_jit_fuser_legacy.py_docs.md) | [keywords](./test_jit_fuser_legacy.py_kw.md) |
| [`test_jit_fuser_te.py`](../../test/test_jit_fuser_te.py) | Test file | [docs](./test_jit_fuser_te.py_docs.md) | [keywords](./test_jit_fuser_te.py_kw.md) |
| [`test_jit_legacy.py`](../../test/test_jit_legacy.py) | Test file | [docs](./test_jit_legacy.py_docs.md) | [keywords](./test_jit_legacy.py_kw.md) |
| [`test_jit_llga_fuser.py`](../../test/test_jit_llga_fuser.py) | Test file | [docs](./test_jit_llga_fuser.py_docs.md) | [keywords](./test_jit_llga_fuser.py_kw.md) |
| [`test_jit_profiling.py`](../../test/test_jit_profiling.py) | Test file | [docs](./test_jit_profiling.py_docs.md) | [keywords](./test_jit_profiling.py_kw.md) |
| [`test_jit_simple.py`](../../test/test_jit_simple.py) | Test file | [docs](./test_jit_simple.py_docs.md) | [keywords](./test_jit_simple.py_kw.md) |
| [`test_jit_string.py`](../../test/test_jit_string.py) | Test file | [docs](./test_jit_string.py_docs.md) | [keywords](./test_jit_string.py_kw.md) |
| [`test_jiterator.py`](../../test/test_jiterator.py) | Test file | [docs](./test_jiterator.py_docs.md) | [keywords](./test_jiterator.py_kw.md) |
| [`test_kernel_launch_checks.py`](../../test/test_kernel_launch_checks.py) | Test file | [docs](./test_kernel_launch_checks.py_docs.md) | [keywords](./test_kernel_launch_checks.py_kw.md) |
| [`test_legacy_vmap.py`](../../test/test_legacy_vmap.py) | Test file | [docs](./test_legacy_vmap.py_docs.md) | [keywords](./test_legacy_vmap.py_kw.md) |
| [`test_license.py`](../../test/test_license.py) | Test file | [docs](./test_license.py_docs.md) | [keywords](./test_license.py_kw.md) |
| [`test_linalg.py`](../../test/test_linalg.py) | Test file | [docs](./test_linalg.py_docs.md) | [keywords](./test_linalg.py_kw.md) |
| [`test_logging.py`](../../test/test_logging.py) | Test file | [docs](./test_logging.py_docs.md) | [keywords](./test_logging.py_kw.md) |
| [`test_masked.py`](../../test/test_masked.py) | Test file | [docs](./test_masked.py_docs.md) | [keywords](./test_masked.py_kw.md) |
| [`test_maskedtensor.py`](../../test/test_maskedtensor.py) | Test file | [docs](./test_maskedtensor.py_docs.md) | [keywords](./test_maskedtensor.py_kw.md) |
| [`test_matmul_cuda.py`](../../test/test_matmul_cuda.py) | Test file | [docs](./test_matmul_cuda.py_docs.md) | [keywords](./test_matmul_cuda.py_kw.md) |
| [`test_meta.py`](../../test/test_meta.py) | Test file | [docs](./test_meta.py_docs.md) | [keywords](./test_meta.py_kw.md) |
| [`test_metal.py`](../../test/test_metal.py) | Test file | [docs](./test_metal.py_docs.md) | [keywords](./test_metal.py_kw.md) |
| [`test_mkl_verbose.py`](../../test/test_mkl_verbose.py) | Test file | [docs](./test_mkl_verbose.py_docs.md) | [keywords](./test_mkl_verbose.py_kw.md) |
| [`test_mkldnn.py`](../../test/test_mkldnn.py) | Test file | [docs](./test_mkldnn.py_docs.md) | [keywords](./test_mkldnn.py_kw.md) |
| [`test_mkldnn_fusion.py`](../../test/test_mkldnn_fusion.py) | Test file | [docs](./test_mkldnn_fusion.py_docs.md) | [keywords](./test_mkldnn_fusion.py_kw.md) |
| [`test_mkldnn_verbose.py`](../../test/test_mkldnn_verbose.py) | Test file | [docs](./test_mkldnn_verbose.py_docs.md) | [keywords](./test_mkldnn_verbose.py_kw.md) |
| [`test_mobile_optimizer.py`](../../test/test_mobile_optimizer.py) | Test file | [docs](./test_mobile_optimizer.py_docs.md) | [keywords](./test_mobile_optimizer.py_kw.md) |
| [`test_model_exports_to_core_aten.py`](../../test/test_model_exports_to_core_aten.py) | Test file | [docs](./test_model_exports_to_core_aten.py_docs.md) | [keywords](./test_model_exports_to_core_aten.py_kw.md) |
| [`test_module_tracker.py`](../../test/test_module_tracker.py) | Test file | [docs](./test_module_tracker.py_docs.md) | [keywords](./test_module_tracker.py_kw.md) |
| [`test_modules.py`](../../test/test_modules.py) | Test file | [docs](./test_modules.py_docs.md) | [keywords](./test_modules.py_kw.md) |
| [`test_monitor.py`](../../test/test_monitor.py) | Test file | [docs](./test_monitor.py_docs.md) | [keywords](./test_monitor.py_kw.md) |
| [`test_mps.py`](../../test/test_mps.py) | Test file | [docs](./test_mps.py_docs.md) | [keywords](./test_mps.py_kw.md) |
| [`test_multiprocessing.py`](../../test/test_multiprocessing.py) | Test file | [docs](./test_multiprocessing.py_docs.md) | [keywords](./test_multiprocessing.py_kw.md) |
| [`test_multiprocessing_spawn.py`](../../test/test_multiprocessing_spawn.py) | Test file | [docs](./test_multiprocessing_spawn.py_docs.md) | [keywords](./test_multiprocessing_spawn.py_kw.md) |
| [`test_namedtensor.py`](../../test/test_namedtensor.py) | Test file | [docs](./test_namedtensor.py_docs.md) | [keywords](./test_namedtensor.py_kw.md) |
| [`test_namedtuple_return_api.py`](../../test/test_namedtuple_return_api.py) | Test file | [docs](./test_namedtuple_return_api.py_docs.md) | [keywords](./test_namedtuple_return_api.py_kw.md) |
| [`test_native_functions.py`](../../test/test_native_functions.py) | Test file | [docs](./test_native_functions.py_docs.md) | [keywords](./test_native_functions.py_kw.md) |
| [`test_native_mha.py`](../../test/test_native_mha.py) | Test file | [docs](./test_native_mha.py_docs.md) | [keywords](./test_native_mha.py_kw.md) |
| [`test_nestedtensor.py`](../../test/test_nestedtensor.py) | Test file | [docs](./test_nestedtensor.py_docs.md) | [keywords](./test_nestedtensor.py_kw.md) |
| [`test_nn.py`](../../test/test_nn.py) | Test file | [docs](./test_nn.py_docs.md) | [keywords](./test_nn.py_kw.md) |
| [`test_nnapi.py`](../../test/test_nnapi.py) | Test file | [docs](./test_nnapi.py_docs.md) | [keywords](./test_nnapi.py_kw.md) |
| [`test_numa_binding.py`](../../test/test_numa_binding.py) | Test file | [docs](./test_numa_binding.py_docs.md) | [keywords](./test_numa_binding.py_kw.md) |
| [`test_numba_integration.py`](../../test/test_numba_integration.py) | Test file | [docs](./test_numba_integration.py_docs.md) | [keywords](./test_numba_integration.py_kw.md) |
| [`test_numpy_interop.py`](../../test/test_numpy_interop.py) | Test file | [docs](./test_numpy_interop.py_docs.md) | [keywords](./test_numpy_interop.py_kw.md) |
| [`test_opaque_obj.py`](../../test/test_opaque_obj.py) | Test file | [docs](./test_opaque_obj.py_docs.md) | [keywords](./test_opaque_obj.py_kw.md) |
| [`test_opaque_obj_v2.py`](../../test/test_opaque_obj_v2.py) | Test file | [docs](./test_opaque_obj_v2.py_docs.md) | [keywords](./test_opaque_obj_v2.py_kw.md) |
| [`test_openmp.py`](../../test/test_openmp.py) | Test file | [docs](./test_openmp.py_docs.md) | [keywords](./test_openmp.py_kw.md) |
| [`test_ops.py`](../../test/test_ops.py) | Test file | [docs](./test_ops.py_docs.md) | [keywords](./test_ops.py_kw.md) |
| [`test_ops_fwd_gradients.py`](../../test/test_ops_fwd_gradients.py) | Test file | [docs](./test_ops_fwd_gradients.py_docs.md) | [keywords](./test_ops_fwd_gradients.py_kw.md) |
| [`test_ops_gradients.py`](../../test/test_ops_gradients.py) | Test file | [docs](./test_ops_gradients.py_docs.md) | [keywords](./test_ops_gradients.py_kw.md) |
| [`test_ops_jit.py`](../../test/test_ops_jit.py) | Test file | [docs](./test_ops_jit.py_docs.md) | [keywords](./test_ops_jit.py_kw.md) |
| [`test_optim.py`](../../test/test_optim.py) | Test file | [docs](./test_optim.py_docs.md) | [keywords](./test_optim.py_kw.md) |
| [`test_out_dtype_op.py`](../../test/test_out_dtype_op.py) | Test file | [docs](./test_out_dtype_op.py_docs.md) | [keywords](./test_out_dtype_op.py_kw.md) |
| [`test_overrides.py`](../../test/test_overrides.py) | Test file | [docs](./test_overrides.py_docs.md) | [keywords](./test_overrides.py_kw.md) |
| [`test_package.py`](../../test/test_package.py) | Test file | [docs](./test_package.py_docs.md) | [keywords](./test_package.py_kw.md) |
| [`test_per_overload_api.py`](../../test/test_per_overload_api.py) | Test file | [docs](./test_per_overload_api.py_docs.md) | [keywords](./test_per_overload_api.py_kw.md) |
| [`test_prims.py`](../../test/test_prims.py) | Test file | [docs](./test_prims.py_docs.md) | [keywords](./test_prims.py_kw.md) |
| [`test_privateuseone_python_backend.py`](../../test/test_privateuseone_python_backend.py) | Test file | [docs](./test_privateuseone_python_backend.py_docs.md) | [keywords](./test_privateuseone_python_backend.py_kw.md) |
| [`test_proxy_tensor.py`](../../test/test_proxy_tensor.py) | Test file | [docs](./test_proxy_tensor.py_docs.md) | [keywords](./test_proxy_tensor.py_kw.md) |
| [`test_pruning_op.py`](../../test/test_pruning_op.py) | Test file | [docs](./test_pruning_op.py_docs.md) | [keywords](./test_pruning_op.py_kw.md) |
| [`test_public_bindings.py`](../../test/test_public_bindings.py) | Test file | [docs](./test_public_bindings.py_docs.md) | [keywords](./test_public_bindings.py_kw.md) |
| [`test_python_dispatch.py`](../../test/test_python_dispatch.py) | Test file | [docs](./test_python_dispatch.py_docs.md) | [keywords](./test_python_dispatch.py_kw.md) |
| [`test_pytree.py`](../../test/test_pytree.py) | Test file | [docs](./test_pytree.py_docs.md) | [keywords](./test_pytree.py_kw.md) |
| [`test_quantization.py`](../../test/test_quantization.py) | Test file | [docs](./test_quantization.py_docs.md) | [keywords](./test_quantization.py_kw.md) |
| [`test_reductions.py`](../../test/test_reductions.py) | Test file | [docs](./test_reductions.py_docs.md) | [keywords](./test_reductions.py_kw.md) |
| [`test_rename_privateuse1_to_existing_device.py`](../../test/test_rename_privateuse1_to_existing_device.py) | Test file | [docs](./test_rename_privateuse1_to_existing_device.py_docs.md) | [keywords](./test_rename_privateuse1_to_existing_device.py_kw.md) |
| [`test_scaled_matmul_cuda.py`](../../test/test_scaled_matmul_cuda.py) | Test file | [docs](./test_scaled_matmul_cuda.py_docs.md) | [keywords](./test_scaled_matmul_cuda.py_kw.md) |
| [`test_scatter_gather_ops.py`](../../test/test_scatter_gather_ops.py) | Test file | [docs](./test_scatter_gather_ops.py_docs.md) | [keywords](./test_scatter_gather_ops.py_kw.md) |
| [`test_schema_check.py`](../../test/test_schema_check.py) | Test file | [docs](./test_schema_check.py_docs.md) | [keywords](./test_schema_check.py_kw.md) |
| [`test_segment_reductions.py`](../../test/test_segment_reductions.py) | Test file | [docs](./test_segment_reductions.py_docs.md) | [keywords](./test_segment_reductions.py_kw.md) |
| [`test_serialization.py`](../../test/test_serialization.py) | Test file | [docs](./test_serialization.py_docs.md) | [keywords](./test_serialization.py_kw.md) |
| [`test_set_default_mobile_cpu_allocator.py`](../../test/test_set_default_mobile_cpu_allocator.py) | Test file | [docs](./test_set_default_mobile_cpu_allocator.py_docs.md) | [keywords](./test_set_default_mobile_cpu_allocator.py_kw.md) |
| [`test_shape_ops.py`](../../test/test_shape_ops.py) | Test file | [docs](./test_shape_ops.py_docs.md) | [keywords](./test_shape_ops.py_kw.md) |
| [`test_show_pickle.py`](../../test/test_show_pickle.py) | Test file | [docs](./test_show_pickle.py_docs.md) | [keywords](./test_show_pickle.py_kw.md) |
| [`test_sort_and_select.py`](../../test/test_sort_and_select.py) | Test file | [docs](./test_sort_and_select.py_docs.md) | [keywords](./test_sort_and_select.py_kw.md) |
| [`test_sparse.py`](../../test/test_sparse.py) | Test file | [docs](./test_sparse.py_docs.md) | [keywords](./test_sparse.py_kw.md) |
| [`test_sparse_csr.py`](../../test/test_sparse_csr.py) | Test file | [docs](./test_sparse_csr.py_docs.md) | [keywords](./test_sparse_csr.py_kw.md) |
| [`test_sparse_semi_structured.py`](../../test/test_sparse_semi_structured.py) | Test file | [docs](./test_sparse_semi_structured.py_docs.md) | [keywords](./test_sparse_semi_structured.py_kw.md) |
| [`test_spectral_ops.py`](../../test/test_spectral_ops.py) | Test file | [docs](./test_spectral_ops.py_docs.md) | [keywords](./test_spectral_ops.py_kw.md) |
| [`test_stateless.py`](../../test/test_stateless.py) | Test file | [docs](./test_stateless.py_docs.md) | [keywords](./test_stateless.py_kw.md) |
| [`test_static_runtime.py`](../../test/test_static_runtime.py) | Test file | [docs](./test_static_runtime.py_docs.md) | [keywords](./test_static_runtime.py_kw.md) |
| [`test_subclass.py`](../../test/test_subclass.py) | Test file | [docs](./test_subclass.py_docs.md) | [keywords](./test_subclass.py_kw.md) |
| [`test_sympy_utils.py`](../../test/test_sympy_utils.py) | Test file | [docs](./test_sympy_utils.py_docs.md) | [keywords](./test_sympy_utils.py_kw.md) |
| [`test_tensor_creation_ops.py`](../../test/test_tensor_creation_ops.py) | Test file | [docs](./test_tensor_creation_ops.py_docs.md) | [keywords](./test_tensor_creation_ops.py_kw.md) |
| [`test_tensorboard.py`](../../test/test_tensorboard.py) | Test file | [docs](./test_tensorboard.py_docs.md) | [keywords](./test_tensorboard.py_kw.md) |
| [`test_tensorexpr.py`](../../test/test_tensorexpr.py) | Test file | [docs](./test_tensorexpr.py_docs.md) | [keywords](./test_tensorexpr.py_kw.md) |
| [`test_tensorexpr_pybind.py`](../../test/test_tensorexpr_pybind.py) | Test file | [docs](./test_tensorexpr_pybind.py_docs.md) | [keywords](./test_tensorexpr_pybind.py_kw.md) |
| [`test_testing.py`](../../test/test_testing.py) | Test file | [docs](./test_testing.py_docs.md) | [keywords](./test_testing.py_kw.md) |
| [`test_throughput_benchmark.py`](../../test/test_throughput_benchmark.py) | Test file | [docs](./test_throughput_benchmark.py_docs.md) | [keywords](./test_throughput_benchmark.py_kw.md) |
| [`test_torch.py`](../../test/test_torch.py) | Test file | [docs](./test_torch.py_docs.md) | [keywords](./test_torch.py_kw.md) |
| [`test_torchfuzz_repros.py`](../../test/test_torchfuzz_repros.py) | Test file | [docs](./test_torchfuzz_repros.py_docs.md) | [keywords](./test_torchfuzz_repros.py_kw.md) |
| [`test_transformers.py`](../../test/test_transformers.py) | Test file | [docs](./test_transformers.py_docs.md) | [keywords](./test_transformers.py_kw.md) |
| [`test_type_hints.py`](../../test/test_type_hints.py) | Test file | [docs](./test_type_hints.py_docs.md) | [keywords](./test_type_hints.py_kw.md) |
| [`test_type_info.py`](../../test/test_type_info.py) | Test file | [docs](./test_type_info.py_docs.md) | [keywords](./test_type_info.py_kw.md) |
| [`test_type_promotion.py`](../../test/test_type_promotion.py) | Test file | [docs](./test_type_promotion.py_docs.md) | [keywords](./test_type_promotion.py_kw.md) |
| [`test_typing.py`](../../test/test_typing.py) | Test file | [docs](./test_typing.py_docs.md) | [keywords](./test_typing.py_kw.md) |
| [`test_unary_ufuncs.py`](../../test/test_unary_ufuncs.py) | Test file | [docs](./test_unary_ufuncs.py_docs.md) | [keywords](./test_unary_ufuncs.py_kw.md) |
| [`test_utils.py`](../../test/test_utils.py) | Test file | [docs](./test_utils.py_docs.md) | [keywords](./test_utils.py_kw.md) |
| [`test_utils_config_module.py`](../../test/test_utils_config_module.py) | Test file | [docs](./test_utils_config_module.py_docs.md) | [keywords](./test_utils_config_module.py_kw.md) |
| [`test_utils_filelock.py`](../../test/test_utils_filelock.py) | Test file | [docs](./test_utils_filelock.py_docs.md) | [keywords](./test_utils_filelock.py_kw.md) |
| [`test_varlen_attention.py`](../../test/test_varlen_attention.py) | Test file | [docs](./test_varlen_attention.py_docs.md) | [keywords](./test_varlen_attention.py_kw.md) |
| [`test_view_ops.py`](../../test/test_view_ops.py) | Test file | [docs](./test_view_ops.py_docs.md) | [keywords](./test_view_ops.py_kw.md) |
| [`test_vulkan.py`](../../test/test_vulkan.py) | Test file | [docs](./test_vulkan.py_docs.md) | [keywords](./test_vulkan.py_kw.md) |
| [`test_weak.py`](../../test/test_weak.py) | Test file | [docs](./test_weak.py_docs.md) | [keywords](./test_weak.py_kw.md) |
| [`test_xnnpack_integration.py`](../../test/test_xnnpack_integration.py) | Test file | [docs](./test_xnnpack_integration.py_docs.md) | [keywords](./test_xnnpack_integration.py_kw.md) |
| [`test_xpu.py`](../../test/test_xpu.py) | Test file | [docs](./test_xpu.py_docs.md) | [keywords](./test_xpu.py_kw.md) |
| [`test_xpu_expandable_segments.py`](../../test/test_xpu_expandable_segments.py) | Test file | [docs](./test_xpu_expandable_segments.py_docs.md) | [keywords](./test_xpu_expandable_segments.py_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
