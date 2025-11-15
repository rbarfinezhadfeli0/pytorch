# Documentation: `aten/src/ATen/native/metal/mpscnn/MPSCNNUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/metal/mpscnn/MPSCNNUtils.h`
- **Size**: 2,996 bytes (2.93 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string>

// This is a utility macro that can be used to throw an exception when a Metal
// API function produces a NSError. The exception will contain a message with
// useful info extracted from the NSError.
#define METAL_THROW_IF_ERROR(error, preamble)                                    \
  do {                                                                           \
    if C10_LIKELY(error) {                                                       \
      throw c10::Error(                                                          \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},                 \
          c10::str(                                                              \
              preamble,                                                          \
              " Error details: ",                                                \
              " Localized_description: ", error.localizedDescription.UTF8String, \
              " Domain: ", error.domain.UTF8String,                              \
              " Code: ", error.code,                                             \
              " User Info: ", error.userInfo.description.UTF8String));           \
    }                                                                            \
  } while (false)

namespace at::native::metal::mpscnn {

struct LaunchParams {
  MTLSize threadsPerThreadgroup;
  MTLSize threadgroupsPerGrid;
  MTLSize threadsPerGrid; // iOS 11.0
};

API_AVAILABLE(ios(11.0), macos(10.13))
LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,
    MPSImage* im);

API_AVAILABLE(ios(11.0), macos(10.13))
LaunchParams spatialPointwiseKernelLaunchParams(
    id<MTLComputePipelineState> pipeline,
    NSUInteger numberOfImages,
    NSUInteger featureChannels,
    NSUInteger height,
    NSUInteger width);

API_AVAILABLE(ios(11.0), macos(10.13))
static inline std::string kernelFor(
    MPSImage* image,
    const std::string& arrayKernel,
    const std::string& nonArrayKernel) {
  if (image.featureChannels > 4 || image.numberOfImages > 1) {
    return arrayKernel;
  }
  return nonArrayKernel;
}

static inline int computeMPSAlignOffset(int kernel, int pad) {
  // To set the offset, we can just match the top-left pixel (in the input
  // image, with negative values for padding) that we look at. For 3x3s1p1, we
  // look at the (-1, -1) pixel in the original impl. For 3x3s1p0, we look at
  // (0, 0) pixel. For 3x3s1p2, look at (-2, -2) MPSCNN always looks at
  // (-floor(kernel_size - 1 / 2), -floor(kernel_size - 1 / 2)) Thus, we just
  // need to match this up.

  // For 3x3s1p1, offset should be (0, 0)
  // For 3x3s1p0, offset should be (1, 1)
  // For 3x3s1p2, offset should be (-1, -1)
  const int mps_offset = kernel / 2;
  const int pt_offset = pad;
  return mps_offset - pt_offset;
}

} // namespace at::native::metal::mpscnn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `LaunchParams`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/metal/mpscnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `string`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/metal/mpscnn`):

- [`MPSCNNClampOp.h_docs.md`](./MPSCNNClampOp.h_docs.md)
- [`MPSCNNFullyConnectedOp.h_docs.md`](./MPSCNNFullyConnectedOp.h_docs.md)
- [`MPSCNNNeuronOp.h_docs.md`](./MPSCNNNeuronOp.h_docs.md)
- [`MPSImageUtils.h_docs.md`](./MPSImageUtils.h_docs.md)
- [`MPSCNNConvOp.h_docs.md`](./MPSCNNConvOp.h_docs.md)
- [`MPSImageWrapper.h_docs.md`](./MPSImageWrapper.h_docs.md)
- [`MPSCNNOp.h_docs.md`](./MPSCNNOp.h_docs.md)
- [`MPSImage+Tensor.h_docs.md`](./MPSImage+Tensor.h_docs.md)


## Cross-References

- **File Documentation**: `MPSCNNUtils.h_docs.md`
- **Keyword Index**: `MPSCNNUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
