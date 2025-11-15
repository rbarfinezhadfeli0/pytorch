# Documentation: `aten/src/ATen/native/vulkan/ops/UnaryOp.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/UnaryOp.cpp`
- **Size**: 3,613 bytes (3.53 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ArrayRef.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/library.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
using namespace api::utils;

Tensor unary_op(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& unary_op_(Tensor& self_arg, const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor exp(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(exp));
}

Tensor& exp_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(exp_inplace));
}

Tensor sqrt(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(sqrt));
}

Tensor& sqrt_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(sqrt_inplace));
}

Tensor log(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(log));
}

Tensor& log_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(log_inplace));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::exp"), TORCH_FN(exp));
  m.impl(TORCH_SELECTIVE_NAME("aten::exp_"), TORCH_FN(exp_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt"), TORCH_FN(sqrt));
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt_"), TORCH_FN(sqrt_));
  m.impl(TORCH_SELECTIVE_NAME("aten::log"), TORCH_FN(log));
  m.impl(TORCH_SELECTIVE_NAME("aten::log_"), TORCH_FN(log_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

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
- [`Permute.cpp_docs.md`](./Permute.cpp_docs.md)
- [`Unsqueeze.cpp_docs.md`](./Unsqueeze.cpp_docs.md)
- [`Stack.cpp_docs.md`](./Stack.cpp_docs.md)


## Cross-References

- **File Documentation**: `UnaryOp.cpp_docs.md`
- **Keyword Index**: `UnaryOp.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
