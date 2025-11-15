# Keyword Index: `torch/fx/passes/_tensorify_python_scalars.py`

## File Information

- **Original File**: [torch/fx/passes/_tensorify_python_scalars.py](../../../../torch/fx/passes/_tensorify_python_scalars.py)
- **Documentation**: [`_tensorify_python_scalars.py_docs.md`](./_tensorify_python_scalars.py_docs.md)
- **Folder**: `torch/fx/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_sympy_interp`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`tensorify_python_scalars`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)

### Imports

- **`Any`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`BooleanAtom`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`FakeTensor`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`GraphModule`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`Integer`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`MetaProxy`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`TensorReferenceAnalysis`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`TensorifyScalarRestartAnalysis`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`TensorifyState`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`__future__`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`_get_sym_val`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`_run_sympy_handler`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`annotations`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`fake_tensor`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`get_computation_dtype`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`get_metrics_context`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`justknobs_check`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`lazy_format_graph_code`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`logging`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`os`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`symbol_is_type`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`sympy`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`sympy.logic.boolalg`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch._dynamo.exc`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch._dynamo.symbolic_convert`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch._dynamo.utils`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch._prims_common`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch._subclasses`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch._utils_internal`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.fx`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.fx._utils`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.fx.graph_module`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.fx.passes.runtime_assert`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.fx.proxy`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.utils._sympy.interp`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.utils._sympy.reference`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`torch.utils._sympy.symbol`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)
- **`typing`**: [_tensorify_python_scalars.py_docs.md](./_tensorify_python_scalars.py_docs.md)


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
