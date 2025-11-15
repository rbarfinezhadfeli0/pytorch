# Keyword Index: `test/test_xnnpack_integration.py`

## File Information

- **Original File**: [test/test_xnnpack_integration.py](../../test/test_xnnpack_integration.py)
- **Documentation**: [`test_xnnpack_integration.py_docs.md`](./test_xnnpack_integration.py_docs.md)
- **Folder**: `test`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Conv1D`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`Conv2D`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`Conv2DPrePacked`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`Conv2DT`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`Conv2DTPrePacked`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`DecomposedLinearAddmm`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`DecomposedLinearMatmul`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`DecomposedLinearMatmulAdd`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`Linear`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`LinearNoBias`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`LinearPrePacked`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`M`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`MFusionAntiPattern`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`MFusionAntiPatternParamMinMax`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`MPrePacked`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`Net`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`TestXNNPACKConv1dTransformPass`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`TestXNNPACKOps`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`TestXNNPACKRewritePass`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`TestXNNPACKSerDes`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)

### Functions

- **`__init__`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`forward`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`test_combined_model`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`test_conv1d_basic`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`test_conv1d_with_relu_fc`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`test_conv2d`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`test_conv2d_transpose`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`test_decomposed_linear`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`test_linear`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`test_linear_1d_input`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`validate_transform_conv1d_to_conv2d`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`validate_transformed_module`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)

### Imports

- **`FileCheck`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`assume`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`functional`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`hypothesis`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`io`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`itertools`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`optimize_for_mobile`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`torch`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`torch.backends.xnnpack`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`torch.nn`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`torch.testing`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`torch.testing._internal.hypothesis_utils`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`torch.utils.mobile_optimizer`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)
- **`unittest`**: [test_xnnpack_integration.py_docs.md](./test_xnnpack_integration.py_docs.md)


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
