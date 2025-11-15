# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution.cc_docs.md`
- **Size**: 15,254 bytes (14.90 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution.cc`
- **Size**: 12,604 bytes (12.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "deconvolution-operator-tester.h"

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, zero_batch,
  DeconvolutionOperatorTester()
    .inputSize(5, 5)
    .kernelSize(1, 1)
    .groupInputChannels(2)
    .groupOutputChannels(2)
    .iterations(1)
    .batchSize(0)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x1,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x1_with_qmin,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmin(128)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x1_with_qmax,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmax(128)
      .iterations(3)
)

_STATIC_TEST(DECONVOLUTION_OP, 1x1_with_input_stride,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .inputPixelStride(28)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_TEST(DECONVOLUTION_OP, 1x1_with_output_stride,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .outputPixelStride(29)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x1_with_batch,
  DeconvolutionOperatorTester()
      .inputSize(13, 14)
      .kernelSize(1, 1)
      .batchSize(3)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, grouped_1x1,
  DeconvolutionOperatorTester()
      .inputSize(24, 25)
      .kernelSize(1, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x3,
  DeconvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, grouped_1x3,
  DeconvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x1,
  DeconvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, grouped_3x1,
  DeconvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3,
  DeconvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_TEST(DECONVOLUTION_OP, 3x3_with_input_stride,
  DeconvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .inputPixelStride(22)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_TEST(DECONVOLUTION_OP, 3x3_with_output_stride,
  DeconvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .outputPixelStride(23)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3_with_batch,
  DeconvolutionOperatorTester()
      .inputSize(10, 9)
      .padding(1)
      .kernelSize(3, 3)
      .batchSize(3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, grouped_3x3,
  DeconvolutionOperatorTester()
      .inputSize(10, 11)
      .padding(1)
      .kernelSize(3, 3)
      .groups(2)
      .groupInputChannels(14)
      .groupOutputChannels(13)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3s2,
  DeconvolutionOperatorTester()
      .inputSize(19, 21)
      .padding(1)
      .kernelSize(3, 3)
      .stride(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3s1x2,
  DeconvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .stride(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3s2x1,
  DeconvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .stride(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3d2,
  DeconvolutionOperatorTester()
      .inputSize(13, 14)
      .padding(2)
      .kernelSize(3, 3)
      .dilation(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3d1x2,
  DeconvolutionOperatorTester()
      .inputSize(14, 15)
      .padding(1, 2)
      .kernelSize(3, 3)
      .dilation(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3d2x1,
  DeconvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
)


_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, zero_batch_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(5, 5)
      .kernelSize(1, 1)
      .groupInputChannels(2)
      .groupOutputChannels(2)
      .iterations(1)
      .per_channel(true)
      .batchSize(0)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x1_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x1_with_qmin_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmin(128)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x1_with_qmax_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .qmax(128)
      .iterations(3)
      .per_channel(true)
)

_STATIC_TEST(DECONVOLUTION_OP, 1x1_with_input_stride_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .inputPixelStride(28)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_TEST(DECONVOLUTION_OP, 1x1_with_output_stride_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(27, 29)
      .kernelSize(1, 1)
      .outputPixelStride(29)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x1_with_batch_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(13, 14)
      .kernelSize(1, 1)
      .batchSize(3)
      .groupInputChannels(23)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, grouped_1x1_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(24, 25)
      .kernelSize(1, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 1x3_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, grouped_1x3_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(20, 19)
      .paddingWidth(1)
      .kernelSize(1, 3)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x1_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, grouped_3x1_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(19, 20)
      .paddingHeight(1)
      .kernelSize(3, 1)
      .groups(2)
      .groupInputChannels(17)
      .groupOutputChannels(15)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_TEST(DECONVOLUTION_OP, 3x3_with_input_stride_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .inputPixelStride(22)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_TEST(DECONVOLUTION_OP, 3x3_with_output_stride_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(13, 12)
      .padding(1)
      .kernelSize(3, 3)
      .outputPixelStride(23)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3_with_batch_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(10, 9)
      .padding(1)
      .kernelSize(3, 3)
      .batchSize(3)
      .groupInputChannels(15)
      .groupOutputChannels(17)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, grouped_3x3_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(10, 11)
      .padding(1)
      .kernelSize(3, 3)
      .groups(2)
      .groupInputChannels(14)
      .groupOutputChannels(13)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3s2_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(19, 21)
      .padding(1)
      .kernelSize(3, 3)
      .stride(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3s1x2_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .stride(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3s2x1_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(13, 13)
      .padding(1)
      .kernelSize(3, 3)
      .stride(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3d2_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(13, 14)
      .padding(2)
      .kernelSize(3, 3)
      .dilation(2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3d1x2_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(14, 15)
      .padding(1, 2)
      .kernelSize(3, 3)
      .dilation(1, 2)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

_STATIC_AND_RUNTIME_TEST(DECONVOLUTION_OP, 3x3d2x1_per_channel,
  DeconvolutionOperatorTester()
      .inputSize(15, 14)
      .padding(2, 1)
      .kernelSize(3, 3)
      .dilation(2, 1)
      .groupInputChannels(27)
      .groupOutputChannels(19)
      .iterations(3)
      .per_channel(true)
)

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `deconvolution-operator-tester.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution.cc
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/test`):

- [`avgpool-microkernel-tester.h_docs.md`](./avgpool-microkernel-tester.h_docs.md)
- [`tanh.cc_docs.md`](./tanh.cc_docs.md)
- [`average-pooling-operator-tester.h_docs.md`](./average-pooling-operator-tester.h_docs.md)
- [`u8lut32norm.cc_docs.md`](./u8lut32norm.cc_docs.md)
- [`lut-norm-microkernel-tester.h_docs.md`](./lut-norm-microkernel-tester.h_docs.md)
- [`softargmax.cc_docs.md`](./softargmax.cc_docs.md)
- [`hardsigmoid-operator-tester.h_docs.md`](./hardsigmoid-operator-tester.h_docs.md)
- [`q8avgpool.cc_docs.md`](./q8avgpool.cc_docs.md)
- [`global-average-pooling.cc_docs.md`](./global-average-pooling.cc_docs.md)
- [`channel-shuffle-operator-tester.h_docs.md`](./channel-shuffle-operator-tester.h_docs.md)


## Cross-References

- **File Documentation**: `deconvolution.cc_docs.md`
- **Keyword Index**: `deconvolution.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/aten/src/ATen/native/quantized/cpu/qnnpack/test/deconvolution.cc_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/test`):

- [`leaky-relu.cc_kw.md_docs.md`](./leaky-relu.cc_kw.md_docs.md)
- [`sgemm.cc_kw.md_docs.md`](./sgemm.cc_kw.md_docs.md)
- [`softargmax-operator-tester.h_kw.md_docs.md`](./softargmax-operator-tester.h_kw.md_docs.md)
- [`maxpool-microkernel-tester.h_kw.md_docs.md`](./maxpool-microkernel-tester.h_kw.md_docs.md)
- [`rmax-microkernel-tester.h_kw.md_docs.md`](./rmax-microkernel-tester.h_kw.md_docs.md)
- [`add-operator-tester.h_kw.md_docs.md`](./add-operator-tester.h_kw.md_docs.md)
- [`tanh-operator-tester.h_docs.md_docs.md`](./tanh-operator-tester.h_docs.md_docs.md)
- [`channel-shuffle.cc_docs.md_docs.md`](./channel-shuffle.cc_docs.md_docs.md)
- [`q8vadd.cc_kw.md_docs.md`](./q8vadd.cc_kw.md_docs.md)
- [`global-average-pooling.cc_docs.md_docs.md`](./global-average-pooling.cc_docs.md_docs.md)


## Cross-References

- **File Documentation**: `deconvolution.cc_docs.md_docs.md`
- **Keyword Index**: `deconvolution.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
