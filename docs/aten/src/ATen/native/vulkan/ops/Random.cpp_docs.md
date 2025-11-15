# Documentation: `aten/src/ATen/native/vulkan/ops/Random.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Random.cpp`
- **Size**: 4,439 bytes (4.33 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ArrayRef.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/library.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

#ifdef USE_VULKAN_API

static Tensor& uniform_(
    Tensor& self,
    const double from,
    const double to,
    const std::optional<at::Generator> /* not implemented */) {
  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self);

  const struct Block final {
    uvec3 extents;
    float from;
    float to;
  } block{v_self.extents(), static_cast<float>(from), static_cast<float>(to)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      // shader_descriptor,
      VK_KERNEL(uniform_),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self;
}

static Tensor rand_like(
    const at::Tensor& input_arg,
    const std::optional<c10::ScalarType> /* not implemented */,
    const std::optional<c10::Layout> /* not implemented */,
    const std::optional<c10::Device> /* not implemented */,
    const std::optional<bool> /* not implemented */,
    const std::optional<c10::MemoryFormat> /* not implemented */) {
  // Returns a tensor with the same size as input that is filled with random
  // numbers from a uniform distribution on the interval [0,1). To match the CPU
  // implementation, we simplify the range to [0,1] and tolerate the small
  // chance of 1 being sampled.
  return input_arg.detach().clone().uniform_(0.0, 1.0);
}

static Tensor& normal_(
    Tensor& self,
    const double mean,
    const double std,
    const std::optional<at::Generator> /* not implemented */) {
  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  TORCH_CHECK(std >= 0, "Vulkan: Standard deviation (std) can be negative.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self);

  const struct Block final {
    uvec3 extents;
    float mean;
    float std;
  } block{v_self.extents(), static_cast<float>(mean), static_cast<float>(std)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      // shader_descriptor,
      VK_KERNEL(normal_),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self;
}

static Tensor randn_like(
    const at::Tensor& input_arg,
    const std::optional<c10::ScalarType> /* not implemented */,
    const std::optional<c10::Layout> /* not implemented */,
    const std::optional<c10::Device> /* not implemented */,
    const std::optional<bool> /* not implemented */,
    const std::optional<c10::MemoryFormat> /* not implemented */) {
  // Returns a tensor with the same size as input that is filled with random
  // numbers from a normal distribution with mean 0 and standard deviation 1.
  return input_arg.detach().clone().normal_(0.0, 1.0);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::uniform_"), TORCH_FN(uniform_));
  m.impl(TORCH_SELECTIVE_NAME("aten::rand_like"), TORCH_FN(rand_like));
  m.impl(TORCH_SELECTIVE_NAME("aten::normal_"), TORCH_FN(normal_));
  m.impl(TORCH_SELECTIVE_NAME("aten::randn_like"), TORCH_FN(randn_like));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `ops`, `api`, `native`, `at`

**Classes/Structs**: `Block`, `Block`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ArrayRef.h`
- `ATen/CPUGeneratorImpl.h`
- `ATen/native/vulkan/ops/Common.h`
- `ATen/native/vulkan/ops/QuantizedFunctions.h`
- `torch/library.h`
- `vector`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/vulkan/ops`):

- [`Convert.h_docs.md`](./Convert.h_docs.md)
- [`Batchnorm.cpp_docs.md`](./Batchnorm.cpp_docs.md)
- [`Slice.cpp_docs.md`](./Slice.cpp_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)
- [`Shape.cpp_docs.md`](./Shape.cpp_docs.md)
- [`Mean.cpp_docs.md`](./Mean.cpp_docs.md)
- [`UnaryOp.cpp_docs.md`](./UnaryOp.cpp_docs.md)
- [`Permute.cpp_docs.md`](./Permute.cpp_docs.md)
- [`Unsqueeze.cpp_docs.md`](./Unsqueeze.cpp_docs.md)
- [`Stack.cpp_docs.md`](./Stack.cpp_docs.md)


## Cross-References

- **File Documentation**: `Random.cpp_docs.md`
- **Keyword Index**: `Random.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
