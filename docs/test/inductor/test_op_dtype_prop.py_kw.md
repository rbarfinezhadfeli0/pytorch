# Keyword Index: `test/inductor/test_op_dtype_prop.py`

## File Information

- **Original File**: [test/inductor/test_op_dtype_prop.py](../../../test/inductor/test_op_dtype_prop.py)
- **Documentation**: [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestCase`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)

### Functions

- **`fn`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`func`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`run`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_any`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_assoc_scan`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_binary_math_mixed_precision`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_codegen_upcast_to_fp32`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_constant`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_downcast_div_mod`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_dtype_aware_codegen`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_low_precision_reduction`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_op_dtype_propagation`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_op_dtype_support`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`test_upcast_rank_0_cpu`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)

### Imports

- **`FileCheck`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`GPU_TYPE`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`HAS_GPU`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`OpDtypeSupport`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`TestCase`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`associative_scan`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`config`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`disable_cache_limit`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`get_signature_for_torch_op`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`importlib`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`instantiate_device_type_tests`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`lowerings`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`op_db`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`ops`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`os`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`parametrize`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`re`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`run_and_get_code`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`run_tests`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`sys`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch._dynamo.utils`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch._higher_order_ops.associative_scan`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch._inductor`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch._inductor.codegen.triton`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch._inductor.lowering`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch._inductor.test_case`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch._inductor.utils`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch.fx.operator_schemas`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch.testing`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch.testing._internal.common_device_type`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch.testing._internal.common_methods_invocations`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_op_dtype_prop.py_docs.md](./test_op_dtype_prop.py_docs.md)


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
