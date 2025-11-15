# Keyword Index: `test/distributed/tensor/debug/test_debug_mode.py`

## File Information

- **Original File**: [test/distributed/tensor/debug/test_debug_mode.py](../../../../../test/distributed/tensor/debug/test_debug_mode.py)
- **Documentation**: [`test_debug_mode.py_docs.md`](./test_debug_mode.py_docs.md)
- **Folder**: `test/distributed/tensor/debug`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Bar`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`DummyTorchDispatchMode1`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`DummyTorchDispatchMode2`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`Foo`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`TestDTensorDebugMode`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)

### Functions

- **`__init__`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`__torch_dispatch__`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`call_triton`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`f`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`forward`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`mm`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`setUp`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`tearDown`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_check_hash_mismatches`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_check_structure_mismatches`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_check_triton_hash_mismatches`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_compile`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_debug_mode_backward`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_debug_mode_densor_redistribution_trace`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_debug_mode_einsum`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_debug_mode_higher_order_cond`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_debug_mode_mm`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_debug_string_inside_context`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_fake_tensor`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_nested_debug_mode`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_nn_module`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_pretty_print_dtensor_make_fx`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_real_tensor`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_tensor_attributes`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`test_triton_kernel_logs`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)

### Imports

- **`CompileCounterWithBackend`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`FakeStore`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`FakeTensorMode`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`GPU_TYPE`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`ShardOrderEntry`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`TorchDispatchMode`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`add_kernel_autotuned`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`contextlib`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`has_triton_package`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`make_fx`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch._dynamo.testing`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.distributed`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.distributed.tensor`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.testing._internal.triton_utils`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.utils._debug_mode`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.utils._python_dispatch`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`torch.utils._triton`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`triton`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)
- **`unittest`**: [test_debug_mode.py_docs.md](./test_debug_mode.py_docs.md)


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
