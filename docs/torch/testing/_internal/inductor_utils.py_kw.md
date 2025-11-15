# Keyword Index: `torch/testing/_internal/inductor_utils.py`

## File Information

- **Original File**: [torch/testing/_internal/inductor_utils.py](../../../../torch/testing/_internal/inductor_utils.py)
- **Documentation**: [`inductor_utils.py_docs.md`](./inductor_utils.py_docs.md)
- **Folder**: `torch/testing/_internal`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MockGraphHandler`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`to`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)

### Functions

- **`__init__`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`_amax_to_scale`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`_check_has_dynamic_shape`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`_quantize_blockwise`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`_quantize_rowwise`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`_quantize_tensorwise`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`_to_fp8_saturated`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`call`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`clone_preserve_strides_offset`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`decorate_fn`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`dummy_graph`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`get_dtype`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`get_func_call`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`get_kernel_launch`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`inner`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`maybe_skip_size_asserts`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`patch_inductor_backend`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`requires_cuda_with_enough_memory`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`skipDeviceIf`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`skip_windows_ci`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`test_cpu`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)

### Imports

- **`CalledProcessError`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`ConfigModule`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`CppCodeCache`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`CustomGraphModulePass`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`GraphLowering`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`PythonWrapperCodegen`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`contextlib`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`functools`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`has_helion`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`has_pallas`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`has_triton`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`logging`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`make_fx`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`os`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`re`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`shape_env_from_inputs`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`subprocess`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`sys`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.async_compile`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.codecache`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.codegen.common`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.codegen.wrapper`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.compile_fx`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.custom_graph_pass`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.graph`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.sizevars`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch._inductor.utils`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch.testing._internal.common_device_type`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch.testing._internal.common_utils`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch.utils._config_module`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch.utils._helion`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch.utils._pallas`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`torch.utils._triton`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`triton`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)
- **`unittest`**: [inductor_utils.py_docs.md](./inductor_utils.py_docs.md)


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
