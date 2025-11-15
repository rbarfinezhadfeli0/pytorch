# Documentation: `torch/csrc/api/src/data/datasets/mnist.cpp`

## File Metadata

- **Path**: `torch/csrc/api/src/data/datasets/mnist.cpp`
- **Size**: 3,689 bytes (3.60 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/data/datasets/mnist.h>

#include <torch/data/example.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <fstream>
#include <string>

namespace torch::data::datasets {
namespace {
constexpr uint32_t kTrainSize = 60000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kImageMagicNumber = 2051;
constexpr uint32_t kTargetMagicNumber = 2049;
constexpr uint32_t kImageRows = 28;
constexpr uint32_t kImageColumns = 28;
constexpr const char* kTrainImagesFilename = "train-images-idx3-ubyte";
constexpr const char* kTrainTargetsFilename = "train-labels-idx1-ubyte";
constexpr const char* kTestImagesFilename = "t10k-images-idx3-ubyte";
constexpr const char* kTestTargetsFilename = "t10k-labels-idx1-ubyte";

bool check_is_little_endian() {
  const uint32_t word = 1;
  return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

constexpr uint32_t flip_endianness(uint32_t value) {
  return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) |
      ((value & 0xff0000u) >> 8u) | ((value & 0xff000000u) >> 24u);
}

uint32_t read_int32(std::ifstream& stream) {
  static const bool is_little_endian = check_is_little_endian();
  uint32_t value = 0;
  AT_ASSERT(stream.read(reinterpret_cast<char*>(&value), sizeof value));
  return is_little_endian ? flip_endianness(value) : value;
}

uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
  const auto value = read_int32(stream);
  TORCH_CHECK(
      value == expected,
      "Expected to read number ",
      expected,
      " but found ",
      value,
      " instead");
  return value;
}

std::string join_paths(std::string head, const std::string& tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}

Tensor read_images(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  std::ifstream images(path, std::ios::binary);
  TORCH_CHECK(images, "Error opening images file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  // From http://yann.lecun.com/exdb/mnist/
  expect_int32(images, kImageMagicNumber);
  expect_int32(images, count);
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);

  auto tensor =
      torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
  return tensor.to(torch::kFloat32).div_(255);
}

Tensor read_targets(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
  std::ifstream targets(path, std::ios::binary);
  TORCH_CHECK(targets, "Error opening targets file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  expect_int32(targets, kTargetMagicNumber);
  expect_int32(targets, count);

  auto tensor = torch::empty(count, torch::kByte);
  targets.read(reinterpret_cast<char*>(tensor.data_ptr()), count);
  return tensor.to(torch::kInt64);
}
} // namespace

MNIST::MNIST(const std::string& root, Mode mode)
    : images_(read_images(root, mode == Mode::kTrain)),
      targets_(read_targets(root, mode == Mode::kTrain)) {}

Example<> MNIST::get(size_t index) {
  return {
      images_[static_cast<int64_t>(index)],
      targets_[static_cast<int64_t>(index)]};
}

std::optional<size_t> MNIST::size() const {
  return images_.size(0);
}

bool MNIST::is_train() const noexcept {
  return images_.size(0) == kTrainSize;
}

const Tensor& MNIST::images() const {
  return images_;
}

const Tensor& MNIST::targets() const {
  return targets_;
}

} // namespace torch::data::datasets

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `MNIST`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/src/data/datasets`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/data/datasets/mnist.h`
- `torch/data/example.h`
- `torch/types.h`
- `c10/util/Exception.h`
- `cstddef`
- `fstream`
- `string`


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

Files in the same folder (`torch/csrc/api/src/data/datasets`):



## Cross-References

- **File Documentation**: `mnist.cpp_docs.md`
- **Keyword Index**: `mnist.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
