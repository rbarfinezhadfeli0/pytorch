# Keyword Index: `torch/_higher_order_ops/triton_kernel_wrap.py`

## File Information

- **Original File**: [torch/_higher_order_ops/triton_kernel_wrap.py](../../../torch/_higher_order_ops/triton_kernel_wrap.py)
- **Documentation**: [`triton_kernel_wrap.py_docs.md`](./triton_kernel_wrap.py_docs.md)
- **Folder**: `torch/_higher_order_ops`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Autotuner`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`Intermediate`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`JITFunction`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`KernelSideTable`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`MemoizeWithCycleCheck`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`Op`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`Param`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`TraceableTritonKernelWrapper`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`TracingTritonHOPifier`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`TritonHOPifier`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`TritonKernelWrapperFunctional`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`TritonKernelWrapperMutation`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`that`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)

### Functions

- **`_`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`__call__`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`__getitem__`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`__init__`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`__post_init__`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`_get_specialization`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`_native_specialize_impl`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`add_constant_args`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`add_kernel`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`analyze_kernel_mutations`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`call_HOP`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`call_getitem`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`call_grid`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`call_run`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`call_triton_kernel`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`call_user_defined_fn`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`check_grid`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`create_tma_experimental_metadata`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`create_tma_stable_metadata`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`defaults_ok`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`do_prune_configs`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`fake`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`generate_ttir`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`get_constant_args`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`get_kernel`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`get_mutated_tensors`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`get_signature_value`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`get_tensor_names`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`get_tma_stores`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`get_value`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`identify_mutated_tensors`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`init_variable`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`is_callable`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`is_graphable`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`is_stable_tensor_descriptor_arg`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`is_tensor_like_arg`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`maybe_unpack_configs`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`maybe_unpack_heuristic_result`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`maybe_unpack_tma_experimental_metadata`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`maybe_unpack_tma_stable_metadata`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`mlir_to_functions`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`raise_unsupported`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`reindex`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`reset`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`reset_table`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`run`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`specialize_symbolic`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`store_non_graphable_args`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`trace_triton_kernel_wrapper`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton_kernel_wrapper_functional_dense`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton_kernel_wrapper_functional_fake_tensor_mode`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton_kernel_wrapper_functional_functionalize`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton_kernel_wrapper_functional_proxy_torch_dispatch_mode`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton_kernel_wrapper_mutation_dense`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton_kernel_wrapper_mutation_fake_tensor_mode`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton_kernel_wrapper_mutation_functionalize`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`ttir_to_functions`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`wrap_user_defined_obj`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)

### Imports

- **`ASTSource`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`Any`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`AttrsDescriptor`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`Autotuner`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`BaseBackend`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`BaseFunctionalizeAPI`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`Callable`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`ConstantVariable`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`DispatchKey`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`FakeTensor`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`FakeTensorMode`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`HigherOrderOperator`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`InstructionTranslator`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`IntLikeType`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`JITFunction`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`Never`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`Proxy`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`SymInt`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`TensorDescriptor`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`TritonKernelVariable`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`_CachedTorchDispatchMode`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`autotune`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`clone_preserve_strides`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`collections`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`collections.abc`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`copy`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`dataclasses`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`defaultdict`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`find_paths_if`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`functools`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`guard_scalar`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`has_triton`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`has_triton_tensor_descriptor_host_tma`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`inspect`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`is_wrap_triton_enabled`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`itertools`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`logging`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`native_specialize_impl`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`operator`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`redirect_to_mode`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`specialize_impl`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`sympy`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`threading`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._C`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._dynamo`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._dynamo.symbolic_convert`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._dynamo.variables.constant`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._dynamo.variables.functions`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._higher_order_ops.utils`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._inductor.codegen.wrapper`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._inductor.ir`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._inductor.utils`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._library.triton`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._ops`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._prims_common`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch._subclasses.functional_tensor`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch.fx`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch.fx.proxy`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch.types`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch.utils._pytree`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch.utils._triton`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`torch.utils.checkpoint`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton._C.libtriton.ir`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton._utils`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton.backends`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton.backends.compiler`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton.compiler.compiler`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton.runtime.autotuner`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton.runtime.jit`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton.tools.experimental_descriptor`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`triton.tools.tensor_descriptor`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`typing`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`typing_extensions`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)
- **`user_defined_kernel_grid_fn_code`**: [triton_kernel_wrap.py_docs.md](./triton_kernel_wrap.py_docs.md)


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
