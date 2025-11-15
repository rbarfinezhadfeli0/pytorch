# Documentation: `c10/test/core/AllocatorConfig_test.cpp`

## File Metadata

- **Path**: `c10/test/core/AllocatorConfig_test.cpp`
- **Size**: 5,439 bytes (5.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. This file handles **configuration or setup**.

## Original Source

```cpp
#include <c10/core/AllocatorConfig.h>

#include <gtest/gtest.h>

using namespace c10::CachingAllocator;
constexpr size_t kMB = 1024 * 1024ul;

struct ExtendedAllocatorConfig {
  static ExtendedAllocatorConfig& instance() {
    static ExtendedAllocatorConfig instance;
    return instance;
  }

  // Returns the device-specific option value in bytes.
  static size_t device_specific_option() {
    return instance().device_specific_option_;
  }

  static const std::unordered_set<std::string>& getKeys() {
    return keys_;
  }

  void parseArgs(const std::string& env) {
    // Parse device-specific options from the environment variable
    ConfigTokenizer tokenizer(env);
    for (size_t i = 0; i < tokenizer.size(); i++) {
      const auto& key = tokenizer[i];
      if (key == "device_specific_option_mb") {
        tokenizer.checkToken(++i, ":");
        device_specific_option_ = tokenizer.toSizeT(++i) * kMB;
      } else {
        i = tokenizer.skipKey(i);
      }

      if (i + 1 < tokenizer.size()) {
        tokenizer.checkToken(++i, ",");
      }
    }
  }

 private:
  // Device-specific option, e.g., memory limit for a specific device.
  std::atomic<size_t> device_specific_option_{0};
  inline static std::unordered_set<std::string> keys_{
      "device_specific_option_mb"};
};

REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK(ExtendedAllocatorConfig)

TEST(AllocatorConfigTest, allocator_config_test) {
  std::string env =
      "max_split_size_mb:40,"
      "max_non_split_rounding_mb:30,"
      "garbage_collection_threshold:0.5,"
      "roundup_power2_divisions:[64:8,128:2,256:4,512:2,1024:4,>:1],"
      "expandable_segments:True,"
      "pinned_use_background_threads:True,"
      "device_specific_option_mb:64";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::max_split_size(), 40 * kMB);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::max_non_split_rounding_size(), 30 * kMB);
  EXPECT_EQ(AcceleratorAllocatorConfig::garbage_collection_threshold(), 0.5);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(32 * kMB), 8);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(64 * kMB), 8);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(128 * kMB), 2);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(256 * kMB), 4);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(512 * kMB), 2);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(1024 * kMB), 4);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(2048 * kMB), 1);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(4096 * kMB), 1);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(8192 * kMB), 1);
  EXPECT_EQ(AcceleratorAllocatorConfig::use_expandable_segments(), true);
  EXPECT_EQ(AcceleratorAllocatorConfig::pinned_use_background_threads(), true);
  EXPECT_EQ(ExtendedAllocatorConfig::device_specific_option(), 64 * kMB);

  env =
      "max_split_size_mb:20,"
      "max_non_split_rounding_mb:40,"
      "garbage_collection_threshold:0.8";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::max_split_size(), 20 * kMB);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::max_non_split_rounding_size(), 40 * kMB);
  EXPECT_EQ(AcceleratorAllocatorConfig::garbage_collection_threshold(), 0.8);

  // roundup_power2_divisions knob array syntax
  env = "roundup_power2_divisions:[128:8,256:16,512:1,2048:8,>:2]";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(64 * kMB), 8);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(128 * kMB), 8);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(256 * kMB), 16);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(512 * kMB), 1);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(1024 * kMB), 0);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(2048 * kMB), 8);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(4096 * kMB), 2);

  // roundup_power2_divisions single value syntax for backward compatibility
  env = "roundup_power2_divisions:4";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(64 * kMB), 4);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(256 * kMB), 4);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(2048 * kMB), 4);

  env = "expandable_segments:False,";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::use_expandable_segments(), false);

  env = "pinned_use_background_threads:False";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::pinned_use_background_threads(), false);

  env = "foo:123,bar:456";
  ASSERT_THROW(c10::CachingAllocator::setAllocatorSettings(env), c10::Error);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `ExtendedAllocatorConfig`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/AllocatorConfig.h`
- `gtest/gtest.h`


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
python c10/test/core/AllocatorConfig_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`c10/test/core`):

- [`Scalar_test.cpp_docs.md`](./Scalar_test.cpp_docs.md)
- [`DeviceGuard_test.cpp_docs.md`](./DeviceGuard_test.cpp_docs.md)
- [`Device_test.cpp_docs.md`](./Device_test.cpp_docs.md)
- [`CompileTimeFunctionPointer_test.cpp_docs.md`](./CompileTimeFunctionPointer_test.cpp_docs.md)
- [`DispatchKeySet_test.cpp_docs.md`](./DispatchKeySet_test.cpp_docs.md)
- [`StreamGuard_test.cpp_docs.md`](./StreamGuard_test.cpp_docs.md)
- [`SymInt_test.cpp_docs.md`](./SymInt_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `AllocatorConfig_test.cpp_docs.md`
- **Keyword Index**: `AllocatorConfig_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
