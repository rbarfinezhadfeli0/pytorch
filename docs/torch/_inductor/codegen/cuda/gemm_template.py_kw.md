# Keyword Index: `torch/_inductor/codegen/cuda/gemm_template.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cuda/gemm_template.py](../../../../../torch/_inductor/codegen/cuda/gemm_template.py)
- **Documentation**: [`gemm_template.py_docs.md`](./gemm_template.py_docs.md)
- **Folder**: `torch/_inductor/codegen/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CUTLASS2xGemmTemplate`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`CUTLASS3xGemmTemplate`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`CUTLASSGemmTemplate`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`Element`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`below`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)

### Functions

- **`CUTLASS_BACKEND_DISABLE_CHECKS`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`CUTLASS_DEBUG_TRACE_LEVEL`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`GENERATE_STANDALONE_RUNNER`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`__init__`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_add_cutlass_gemm_choices`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_alignment_match`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_are_inputs_layout_compatible`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_define_gemm_instance`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_dtype_match`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_get_extra_inputs_and_names`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_get_supported_ops`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_get_template`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_get_template_args`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_has_tma_epilogue`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_render_evt`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_set_bias_layout_and_alignment`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_shape_match`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`_update_arg_names_for_test_call_statement`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`add_cutlass_gemm_choices`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`clone_with_transposed_stride`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`cutlass_layout`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`filter_op`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`fix_op_layout`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`flip_cutlass_layout`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`gemm_mode`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`gen_ops`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`global_filter_ops`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`header`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`layout_match`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`render`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`render_gemm_arguments`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`set_alignment`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`set_layout`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`should_swap_XW`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`supports_epilogue_fusion`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`swap_XW`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`test_call_statement`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)

### Imports

- **`.`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`...`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`...config`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`...ir`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`...utils`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`...virtualized`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`..common`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`.cuda_kernel`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`.cuda_template`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`.cutlass_lib_extensions`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`.cutlass_lib_extensions.evt_extensions`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`.cutlass_python_evt`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`.cutlass_utils`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`ABC`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`Any`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`BaseSchedulerNode`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`CUDATemplateKernel`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`CUTLASSTemplate`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`CutlassEVTCodegen`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`IndentedBuffer`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`PythonWrapperCodegen`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`TensorMeta`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`V`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`abc`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`clear_on_fresh_cache`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`copy`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`create_example_tensors`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`create_inputs_key`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`cuda`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`cutlass_library.gemm_operation`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`cutlass_library.library`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`cutlass_utils`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`dynamo_timed`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`enum`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`functools`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`gemm_operation_extensions`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`ir`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`is_dynamic`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`logging`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`maybe_fetch_ops`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`re`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`time`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch._inductor.autotune_process`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch._inductor.codegen.cuda.cutlass_cache`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch._inductor.codegen.wrapper`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch._inductor.runtime.runtime_utils`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch._inductor.scheduler`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch._inductor.select_algorithm`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch._inductor.utils`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`torch.utils._pytree`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)
- **`typing`**: [gemm_template.py_docs.md](./gemm_template.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
