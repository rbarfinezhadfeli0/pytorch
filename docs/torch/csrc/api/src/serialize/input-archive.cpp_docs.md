# Documentation: `torch/csrc/api/src/serialize/input-archive.cpp`

## File Metadata

- **Path**: `torch/csrc/api/src/serialize/input-archive.cpp`
- **Size**: 4,724 bytes (4.61 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/serialize/input-archive.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <c10/util/Exception.h>
#include <caffe2/serialize/read_adapter_interface.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>

#include <istream>
#include <memory>
#include <string>
#include <utility>

namespace torch::serialize {

InputArchive::InputArchive()
    : module_("Module", std::make_shared<jit::CompilationUnit>()) {}

void InputArchive::read(const std::string& key, c10::IValue& ivalue) {
  ivalue = module_.attr(key);
}

bool InputArchive::try_read(const std::string& key, c10::IValue& ivalue) {
  if (!module_.hasattr(key)) {
    return false;
  }
  ivalue = module_.attr(key);
  return true;
}

bool InputArchive::try_read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  if (!module_.hasattr(key)) {
    return false;
  }
  auto iv = module_.attr(key);
  if (!iv.isTensor()) {
    return false;
  }
  auto read_tensor = iv.toTensor();
  // clang-format on
  if (tensor.defined()) {
    torch::NoGradGuard guard;
    if (tensor.device() != read_tensor.device()) {
      tensor.set_data(read_tensor);
    } else {
      tensor.set_(read_tensor);
    }
  } else {
    tensor = std::move(read_tensor);
  }
  return true;
}

void InputArchive::read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  TORCH_CHECK(
      try_read(key, tensor, is_buffer),
      "No such serialized tensor '",
      hierarchy_prefix_,
      key,
      "'");
}

bool InputArchive::try_read(const std::string& key, InputArchive& archive) {
  if (!module_.hasattr(key)) {
    return false;
  }
  auto iv = module_.attr(key);
  if (!iv.isModule()) {
    return false;
  }
  archive.module_ = iv.toModule();
  archive.hierarchy_prefix_ = hierarchy_prefix_ + key + ".";
  return true;
}

void InputArchive::read(const std::string& key, InputArchive& archive) {
  TORCH_CHECK(
      try_read(key, archive),
      "No such serialized submodule: '",
      hierarchy_prefix_,
      key,
      "'");
}

void InputArchive::load_from(
    const std::string& filename,
    std::optional<torch::Device> device /*= std::nullopt*/) {
  module_ = torch::jit::load(filename, device);
}

void InputArchive::load_from(
    std::istream& stream,
    std::optional<torch::Device> device /*= std::nullopt*/) {
  module_ = torch::jit::load(stream, device);
}

void InputArchive::load_from(
    const char* data,
    size_t size,
    std::optional<torch::Device> device /*= std::nullopt*/) {
  using caffe2::serialize::ReadAdapterInterface;
  class OurAdapter : public ReadAdapterInterface {
   public:
    OurAdapter(const char* data, size_t size) : data_(data), size_(size) {}
    size_t size() const override {
      return size_;
    }
    size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
        const override {
      (void)what;
      if (pos >= size_) {
        return 0;
      }
      size_t nread = std::min(static_cast<size_t>(pos) + n, size_) - pos;
      memcpy(buf, data_ + pos, nread);
      return nread;
    }

   private:
    const char* data_;
    size_t size_;
  };
  module_ = torch::jit::load(std::make_unique<OurAdapter>(data, size), device);
}

void InputArchive::load_from(
    const std::function<size_t(uint64_t, void*, size_t)>& read_func,
    const std::function<size_t(void)>& size_func,
    std::optional<torch::Device> device /*= std::nullopt*/) {
  using caffe2::serialize::ReadAdapterInterface;
  class OurAdapter : public ReadAdapterInterface {
   public:
    OurAdapter(
        const std::function<size_t(uint64_t, void*, size_t)>& read_func,
        const std::function<size_t(void)>& size_func)
        : read_func_(read_func), size_func_(size_func) {}
    size_t size() const override {
      return size_func_();
    }
    size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
        const override {
      (void)what;
      return read_func_(pos, buf, n);
    }

   private:
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::function<size_t(uint64_t, void*, size_t)>& read_func_;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::function<size_t(void)>& size_func_;
  };
  module_ = torch::jit::load(
      std::make_unique<OurAdapter>(read_func, size_func), device);
}

std::vector<std::string> InputArchive::keys() {
  std::vector<std::string> all_keys;
  all_keys.reserve(module_.named_attributes(/*recurse=*/false).size());

  for (const torch::jit::NameValue& s :
       module_.named_attributes(/*recurse=*/false)) {
    all_keys.push_back(s.name);
  }

  return all_keys;
}

} // namespace torch::serialize

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `OurAdapter`, `OurAdapter`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/src/serialize`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/serialize/input-archive.h`
- `torch/types.h`
- `torch/utils.h`
- `c10/util/Exception.h`
- `caffe2/serialize/read_adapter_interface.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/serialization/import.h`
- `istream`
- `memory`
- `string`
- `utility`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/csrc/api/src/serialize`):

- [`output-archive.cpp_docs.md`](./output-archive.cpp_docs.md)


## Cross-References

- **File Documentation**: `input-archive.cpp_docs.md`
- **Keyword Index**: `input-archive.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
