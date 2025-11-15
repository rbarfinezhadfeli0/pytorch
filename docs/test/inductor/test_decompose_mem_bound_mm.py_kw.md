# Keyword Index: `test/inductor/test_decompose_mem_bound_mm.py`

## File Information

- **Original File**: [test/inductor/test_decompose_mem_bound_mm.py](../../../test/inductor/test_decompose_mem_bound_mm.py)
- **Documentation**: [`test_decompose_mem_bound_mm.py_docs.md`](./test_decompose_mem_bound_mm.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MyModule`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`MyModule2`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`MyModule3`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`TestDecomposeAddMM`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`TestDecomposeMemMM`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)

### Functions

- **`__init__`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`compare_dict_tensors`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`compare_gradients`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`compare_parameters`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`compare_pred`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`foo`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`forward`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`setup_tolerance`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_check_device`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_decompose_bmm`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_decompose_bmm_cpu`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_decompose_linear`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_decompose_linear_mixed_precision`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_decompose_mm`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_decompose_mm_cpu`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_decompose_mm_mixed_precision`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_dynamic_shape`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_dynamic_shape_decompose_addmm`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`test_realize_input`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)

### Imports

- **`FileCheck`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`GPU_TYPE`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`check_device`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`counters`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`logging`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`requires_gpu`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`run_and_get_code`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`run_tests`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch._dynamo.utils`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch._inductor`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch._inductor.fx_passes.decompose_mem_bound_mm`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch._inductor.test_case`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch._inductor.utils`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch.testing`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`torch.testing._internal.triton_utils`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)
- **`unittest`**: [test_decompose_mem_bound_mm.py_docs.md](./test_decompose_mem_bound_mm.py_docs.md)


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
