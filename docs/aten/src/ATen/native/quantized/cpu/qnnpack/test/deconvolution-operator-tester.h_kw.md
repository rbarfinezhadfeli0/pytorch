# Keyword Index: `aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution-operator-tester.h`

## File Information

- **Original File**: [aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution-operator-tester.h](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution-operator-tester.h)
- **Documentation**: [`deconvolution-operator-tester.h_docs.md`](./deconvolution-operator-tester.h_docs.md)
- **Folder**: `aten/src/ATen/native/quantized/cpu/qnnpack/test`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`DeconvolutionOperatorTester`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)

### Functions

- **`adjustmentHeight`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`adjustmentWidth`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`batchSize`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`dilatedKernelHeight`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`dilatedKernelWidth`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`dilationHeight`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`dilationWidth`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`groupInputChannels`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`groupOutputChannels`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`groups`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`inputHeight`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`inputPixelStride`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`inputWidth`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`iterations`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`kernelHeight`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`kernelWidth`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`outputHeight`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`outputPixelStride`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`outputWidth`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`paddingHeight`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`paddingWidth`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`per_channel`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`qmax`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`qmin`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`strideHeight`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`strideWidth`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`testQ8`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)

### Includes

- **`algorithm`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`cassert`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`cmath`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`cstddef`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`cstdlib`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`functional`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`memory`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`pytorch_qnnpack.h`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`qnnpack_func.h`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`random`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`test_utils.h`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)
- **`vector`**: [deconvolution-operator-tester.h_docs.md](./deconvolution-operator-tester.h_docs.md)


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
