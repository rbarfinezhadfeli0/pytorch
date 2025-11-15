# Documentation: `docs/test/cpp/api/support.h_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/support.h_docs.md`
- **Size**: 8,308 bytes (8.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/support.h`

## File Metadata

- **Path**: `test/cpp/api/support.h`
- **Size**: 5,781 bytes (5.65 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```c
#pragma once

#include <test/cpp/common/support.h>

#include <gtest/gtest.h>

#include <ATen/TensorIndexing.h>
#include <c10/util/Exception.h>
#include <torch/nn/cloneable.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <string>
#include <utility>

namespace torch {
namespace test {

// Lets you use a container without making a new class,
// for experimental implementations
class SimpleContainer : public nn::Cloneable<SimpleContainer> {
 public:
  void reset() override {}

  template <typename ModuleHolder>
  ModuleHolder add(
      ModuleHolder module_holder,
      std::string name = std::string()) {
    return Module::register_module(std::move(name), module_holder);
  }
};

struct SeedingFixture : public ::testing::Test {
  SeedingFixture() {
    torch::manual_seed(0);
  }
};

struct WarningCapture : public WarningHandler {
  WarningCapture() : prev_(WarningUtils::get_warning_handler()) {
    WarningUtils::set_warning_handler(this);
  }

  ~WarningCapture() override {
    WarningUtils::set_warning_handler(prev_);
  }

  const std::vector<std::string>& messages() {
    return messages_;
  }

  std::string str() {
    return c10::Join("\n", messages_);
  }

  void process(const c10::Warning& warning) override {
    messages_.push_back(warning.msg());
  }

 private:
  WarningHandler* prev_;
  std::vector<std::string> messages_;
};

inline bool pointer_equal(at::Tensor first, at::Tensor second) {
  return first.data_ptr() == second.data_ptr();
}

// This mirrors the `isinstance(x, torch.Tensor) and isinstance(y,
// torch.Tensor)` branch in `TestCase.assertEqual` in
// torch/testing/_internal/common_utils.py
inline void assert_tensor_equal(
    at::Tensor a,
    at::Tensor b,
    bool allow_inf = false) {
  ASSERT_TRUE(a.sizes() == b.sizes());
  if (a.numel() > 0) {
    if (a.device().type() == torch::kCPU &&
        (a.scalar_type() == torch::kFloat16 ||
         a.scalar_type() == torch::kBFloat16)) {
      // CPU half and bfloat16 tensors don't have the methods we need below
      a = a.to(torch::kFloat32);
    }
    if (a.device().type() == torch::kCUDA &&
        a.scalar_type() == torch::kBFloat16) {
      // CUDA bfloat16 tensors don't have the methods we need below
      a = a.to(torch::kFloat32);
    }
    b = b.to(a);

    if ((a.scalar_type() == torch::kBool) !=
        (b.scalar_type() == torch::kBool)) {
      TORCH_CHECK(false, "Was expecting both tensors to be bool type.");
    } else {
      if (a.scalar_type() == torch::kBool && b.scalar_type() == torch::kBool) {
        // we want to respect precision but as bool doesn't support subtraction,
        // boolean tensor has to be converted to int
        a = a.to(torch::kInt);
        b = b.to(torch::kInt);
      }

      auto diff = a - b;
      if (a.is_floating_point()) {
        // check that NaNs are in the same locations
        auto nan_mask = torch::isnan(a);
        ASSERT_TRUE(torch::equal(nan_mask, torch::isnan(b)));
        diff.index_put_({nan_mask}, 0);
        // inf check if allow_inf=true
        if (allow_inf) {
          auto inf_mask = torch::isinf(a);
          auto inf_sign = inf_mask.sign();
          ASSERT_TRUE(torch::equal(inf_sign, torch::isinf(b).sign()));
          diff.index_put_({inf_mask}, 0);
        }
      }
      // TODO: implement abs on CharTensor (int8)
      if (diff.is_signed() && diff.scalar_type() != torch::kInt8) {
        diff = diff.abs();
      }
      auto max_err = diff.max().item<double>();
      ASSERT_LE(max_err, 1e-5);
    }
  }
}

// This mirrors the `isinstance(x, torch.Tensor) and isinstance(y,
// torch.Tensor)` branch in `TestCase.assertNotEqual` in
// torch/testing/_internal/common_utils.py
inline void assert_tensor_not_equal(at::Tensor x, at::Tensor y) {
  if (x.sizes() != y.sizes()) {
    return;
  }
  ASSERT_GT(x.numel(), 0);
  y = y.type_as(x);
  y = x.is_cuda() ? y.to({torch::kCUDA, x.get_device()}) : y.cpu();
  auto nan_mask = x != x;
  if (torch::equal(nan_mask, y != y)) {
    auto diff = x - y;
    if (diff.is_signed()) {
      diff = diff.abs();
    }
    diff.index_put_({nan_mask}, 0);
    // Use `item()` to work around:
    // https://github.com/pytorch/pytorch/issues/22301
    auto max_err = diff.max().item<double>();
    ASSERT_GE(max_err, 1e-5);
  }
}

inline int count_substr_occurrences(
    const std::string& str,
    const std::string& substr) {
  int count = 0;
  size_t pos = str.find(substr);

  while (pos != std::string::npos) {
    count++;
    pos = str.find(substr, pos + substr.size());
  }

  return count;
}

// A RAII, thread local (!) guard that changes default dtype upon
// construction, and sets it back to the original dtype upon destruction.
//
// Usage of this guard is synchronized across threads, so that at any given
// time, only one guard can take effect.
struct AutoDefaultDtypeMode {
  static std::mutex default_dtype_mutex;

  AutoDefaultDtypeMode(c10::ScalarType default_dtype)
      : prev_default_dtype(
            torch::typeMetaToScalarType(torch::get_default_dtype())) {
    default_dtype_mutex.lock();
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(default_dtype));
  }
  ~AutoDefaultDtypeMode() {
    default_dtype_mutex.unlock();
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(prev_default_dtype));
  }
  c10::ScalarType prev_default_dtype;
};

inline void assert_tensor_creation_meta(
    torch::Tensor& x,
    torch::autograd::CreationMeta creation_meta) {
  auto autograd_meta = x.unsafeGetTensorImpl()->autograd_meta();
  TORCH_CHECK(autograd_meta);
  auto view_meta =
      static_cast<torch::autograd::DifferentiableViewMeta*>(autograd_meta);
  TORCH_CHECK(view_meta->has_bw_view());
  ASSERT_EQ(view_meta->get_creation_meta(), creation_meta);
}
} // namespace test
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `test`, `torch`

**Classes/Structs**: `SimpleContainer`, `SeedingFixture`, `WarningCapture`, `AutoDefaultDtypeMode`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `test/cpp/common/support.h`
- `gtest/gtest.h`
- `ATen/TensorIndexing.h`
- `c10/util/Exception.h`
- `torch/nn/cloneable.h`
- `torch/types.h`
- `torch/utils.h`
- `string`
- `utility`


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
python test/cpp/api/support.h
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/api`):

- [`fft.cpp_docs.md`](./fft.cpp_docs.md)
- [`tensor_options.cpp_docs.md`](./tensor_options.cpp_docs.md)
- [`any.cpp_docs.md`](./any.cpp_docs.md)
- [`torch_include.cpp_docs.md`](./torch_include.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`jit.cpp_docs.md`](./jit.cpp_docs.md)
- [`nn_utils.cpp_docs.md`](./nn_utils.cpp_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`meta_tensor.cpp_docs.md`](./meta_tensor.cpp_docs.md)
- [`nested_int.cpp_docs.md`](./nested_int.cpp_docs.md)


## Cross-References

- **File Documentation**: `support.h_docs.md`
- **Keyword Index**: `support.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/cpp/api/support.h_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/api`):

- [`init_baseline.py_kw.md_docs.md`](./init_baseline.py_kw.md_docs.md)
- [`support.cpp_kw.md_docs.md`](./support.cpp_kw.md_docs.md)
- [`memory.cpp_docs.md_docs.md`](./memory.cpp_docs.md_docs.md)
- [`parallel_benchmark.cpp_docs.md_docs.md`](./parallel_benchmark.cpp_docs.md_docs.md)
- [`dataloader.cpp_docs.md_docs.md`](./dataloader.cpp_docs.md_docs.md)
- [`moduledict.cpp_kw.md_docs.md`](./moduledict.cpp_kw.md_docs.md)
- [`support.h_kw.md_docs.md`](./support.h_kw.md_docs.md)
- [`ordered_dict.cpp_docs.md_docs.md`](./ordered_dict.cpp_docs.md_docs.md)
- [`functional.cpp_docs.md_docs.md`](./functional.cpp_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)


## Cross-References

- **File Documentation**: `support.h_docs.md_docs.md`
- **Keyword Index**: `support.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
