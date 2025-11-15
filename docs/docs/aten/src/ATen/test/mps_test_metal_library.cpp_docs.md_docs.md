# Documentation: `docs/aten/src/ATen/test/mps_test_metal_library.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/mps_test_metal_library.cpp_docs.md`
- **Size**: 6,473 bytes (6.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/mps_test_metal_library.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/mps_test_metal_library.cpp`
- **Size**: 3,698 bytes (3.61 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <stdexcept>
#include <torch/torch.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/MetalShaderLibrary.h>

using namespace at::native::mps;
TEST(MPSTestMetalLibrary, ShaderCreation) {
   DynamicMetalShaderLibrary lib("// Empty library");
   ASSERT_EQ(lib.getFunctionNames().size(), 0);
}

TEST(MPSTestMetalLibrary, SyntaxErrorThrows) {
  ASSERT_THROW(new DynamicMetalShaderLibrary("printf(x);"), c10::Error);
}

TEST(MPSTestMetalLibrary, ArangeShader) {
  auto y = torch::arange(10.0, at::device(at::kMPS));
  auto x = torch::empty(10, at::device(at::kMPS));
  DynamicMetalShaderLibrary lib(R"MTL(
  kernel void foo(device float* t, uint idx [[thread_position_in_grid]]) {
    t[idx] = idx;
  }
  )MTL");
  auto func = lib.getKernelFunction("foo");
  func->runCommandBlock([&] {
     func->startEncoding();
     func->setArg(0, x);
     func->dispatch(x.numel());
  });
  ASSERT_TRUE((x==y).all().item().toBool());
}

TEST(MPSTestMetalLibrary, ArangeWithArgsShader) {
  const auto size = 10;
  const float start = .25;
  const float step = .4;
  auto x = torch::empty(size, at::device(at::kMPS));
  auto y = torch::arange(start, start + size * step, step, at::device(at::kMPS));
  ASSERT_EQ(x.numel(), y.numel());
  DynamicMetalShaderLibrary lib(R"MTL(
  kernel void foo(device float* t, constant float& start, constant float& step, uint idx [[thread_position_in_grid]]) {
    t[idx] = start + idx * step;
  }
  )MTL");
  auto func = lib.getKernelFunction("foo");
  func->runCommandBlock([&] {
     func->startEncoding();
     func->setArg(0, x);
     func->setArg(1, start);
     func->setArg(2, step);
     func->dispatch(x.numel());
  });
  ASSERT_TRUE((x==y).all().item().toBool());
}

TEST(MPSTestMetalLibrary, Arange2DShader) {
  const auto size = 16;
  auto x = torch::empty({size, size}, at::device(at::kMPS));
  DynamicMetalShaderLibrary lib(R"MTL(
  kernel void full(device float* t, constant ulong2& strides, uint2 idx [[thread_position_in_grid]]) {
    t[idx.x*strides.x + idx.y*strides.y] = idx.x + 33.0 * idx.y;
  }
  )MTL");
  auto func = lib.getKernelFunction("full");
  func->runCommandBlock([&] {
     func->startEncoding();
     func->setArg(0, x);
     func->setArg(1, x.strides());
     func->dispatch({static_cast<uint64_t>(x.size(0)), static_cast<uint64_t>(x.size(1))});
  });
  ASSERT_EQ(x.sum().item().to<int>(), 65280);
}

TEST(MPSTestMetalLibrary, ArgumentBuffers) {
  constexpr auto nbuffers = 64;
  const auto size = 32;
  std::vector<at::Tensor> ibuffers;
  std::vector<void *> ibuffers_gpu_ptrs;
  for([[maybe_unused]] auto idx: c10::irange(nbuffers)) {
    ibuffers.push_back(torch::rand({size}, at::device(at::kMPS)));
    ibuffers_gpu_ptrs.push_back(get_tensor_gpu_address(ibuffers.back()));
  }
  auto output = torch::empty({size}, at::device(at::kMPS));
  DynamicMetalShaderLibrary lib(R"MTL(
  constant constexpr auto nbuffers = 64;
  struct Inputs {
    metal::array<device float *, nbuffers> args;
  };

  kernel void sum_all(device float* output, constant Inputs& inputs, uint idx [[thread_position_in_grid]]) {
    output[idx] = 0;
    for(auto i = 0; i < nbuffers; ++i) {
      output[idx] += inputs.args[i][idx];
    }
  }
  )MTL");
  auto func = lib.getKernelFunction("sum_all");
  func->runCommandBlock([&] {
     func->startEncoding();
     func->setArg(0, output);
     func->setArg(1, ibuffers_gpu_ptrs);
     func->dispatch(size);
  });
  // Compute sum of all 64 input tensors
  auto result = torch::zeros({size}, at::device(at::kMPS));
  for(auto buf: ibuffers) {
    result += buf;
  }
  ASSERT_EQ(result.sum().item().to<float>(), output.sum().item().to<float>());
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Inputs`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `stdexcept`
- `torch/torch.h`
- `ATen/mps/MPSStream.h`
- `ATen/mps/MPSProfiler.h`
- `ATen/native/mps/MetalShaderLibrary.h`


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

This is a test file. Run it with:

```bash
python aten/src/ATen/test/mps_test_metal_library.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `mps_test_metal_library.cpp_docs.md`
- **Keyword Index**: `mps_test_metal_library.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/aten/src/ATen/test/mps_test_metal_library.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mps_test_metal_library.cpp_docs.md_docs.md`
- **Keyword Index**: `mps_test_metal_library.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
