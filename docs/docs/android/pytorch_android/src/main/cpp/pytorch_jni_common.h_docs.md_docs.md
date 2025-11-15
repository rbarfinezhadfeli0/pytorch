# Documentation: `docs/android/pytorch_android/src/main/cpp/pytorch_jni_common.h_docs.md`

## File Metadata

- **Path**: `docs/android/pytorch_android/src/main/cpp/pytorch_jni_common.h_docs.md`
- **Size**: 5,919 bytes (5.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `android/pytorch_android/src/main/cpp/pytorch_jni_common.h`

## File Metadata

- **Path**: `android/pytorch_android/src/main/cpp/pytorch_jni_common.h`
- **Size**: 3,585 bytes (3.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/FunctionRef.h>
#include <fbjni/fbjni.h>
#include <torch/csrc/api/include/torch/types.h>
#include "caffe2/serialize/read_adapter_interface.h"

#include "cmake_macros.h"

#ifdef __ANDROID__
#include <android/log.h>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-jni", __VA_ARGS__)
#define ALOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, "pytorch-jni", __VA_ARGS__)
#endif

#if defined(TRACE_ENABLED) && defined(__ANDROID__)
#include <android/trace.h>
#include <dlfcn.h>
#endif

namespace pytorch_jni {

constexpr static int kDeviceCPU = 1;
constexpr static int kDeviceVulkan = 2;

c10::DeviceType deviceJniCodeToDeviceType(jint deviceJniCode);

class Trace {
 public:
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
  typedef void* (*fp_ATrace_beginSection)(const char* sectionName);
  typedef void* (*fp_ATrace_endSection)(void);

  static fp_ATrace_beginSection ATrace_beginSection;
  static fp_ATrace_endSection ATrace_endSection;
#endif

  static void ensureInit() {
    if (!Trace::is_initialized_) {
      init();
      Trace::is_initialized_ = true;
    }
  }

  static void beginSection(const char* name) {
    Trace::ensureInit();
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
    ATrace_beginSection(name);
#endif
  }

  static void endSection() {
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
    ATrace_endSection();
#endif
  }

  Trace(const char* name) {
    ensureInit();
    beginSection(name);
  }

  ~Trace() {
    endSection();
  }

 private:
  static void init();
  static bool is_initialized_;
};

class MemoryReadAdapter final : public caffe2::serialize::ReadAdapterInterface {
 public:
  explicit MemoryReadAdapter(const void* data, off_t size)
      : data_(data), size_(size){};

  size_t size() const override {
    return size_;
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override {
    memcpy(buf, (int8_t*)(data_) + pos, n);
    return n;
  }

  ~MemoryReadAdapter() {}

 private:
  const void* data_;
  off_t size_;
};

class JIValue : public facebook::jni::JavaClass<JIValue> {
  using DictCallback = c10::function_ref<facebook::jni::local_ref<JIValue>(
      c10::Dict<c10::IValue, c10::IValue>)>;

 public:
  constexpr static const char* kJavaDescriptor = "Lorg/pytorch/IValue;";

  constexpr static int kTypeCodeNull = 1;

  constexpr static int kTypeCodeTensor = 2;
  constexpr static int kTypeCodeBool = 3;
  constexpr static int kTypeCodeLong = 4;
  constexpr static int kTypeCodeDouble = 5;
  constexpr static int kTypeCodeString = 6;

  constexpr static int kTypeCodeTuple = 7;
  constexpr static int kTypeCodeBoolList = 8;
  constexpr static int kTypeCodeLongList = 9;
  constexpr static int kTypeCodeDoubleList = 10;
  constexpr static int kTypeCodeTensorList = 11;
  constexpr static int kTypeCodeList = 12;

  constexpr static int kTypeCodeDictStringKey = 13;
  constexpr static int kTypeCodeDictLongKey = 14;

  static facebook::jni::local_ref<JIValue> newJIValueFromAtIValue(
      const at::IValue& ivalue,
      DictCallback stringDictCallback = newJIValueFromStringDict,
      DictCallback intDictCallback = newJIValueFromIntDict);

  static at::IValue JIValueToAtIValue(
      facebook::jni::alias_ref<JIValue> jivalue);

 private:
  static facebook::jni::local_ref<JIValue> newJIValueFromStringDict(
      c10::Dict<c10::IValue, c10::IValue>);
  static facebook::jni::local_ref<JIValue> newJIValueFromIntDict(
      c10::Dict<c10::IValue, c10::IValue>);
};

void common_registerNatives();
} // namespace pytorch_jni

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `pytorch_jni`

**Classes/Structs**: `Trace`, `MemoryReadAdapter`, `JIValue`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `android/pytorch_android/src/main/cpp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/FunctionRef.h`
- `fbjni/fbjni.h`
- `torch/csrc/api/include/torch/types.h`
- `caffe2/serialize/read_adapter_interface.h`
- `cmake_macros.h`
- `android/log.h`
- `android/trace.h`
- `dlfcn.h`


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

Files in the same folder (`android/pytorch_android/src/main/cpp`):

- [`pytorch_jni_jit.cpp_docs.md`](./pytorch_jni_jit.cpp_docs.md)
- [`cmake_macros.h_docs.md`](./cmake_macros.h_docs.md)
- [`pytorch_jni_common.cpp_docs.md`](./pytorch_jni_common.cpp_docs.md)
- [`pytorch_jni_lite.cpp_docs.md`](./pytorch_jni_lite.cpp_docs.md)


## Cross-References

- **File Documentation**: `pytorch_jni_common.h_docs.md`
- **Keyword Index**: `pytorch_jni_common.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/android/pytorch_android/src/main/cpp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/android/pytorch_android/src/main/cpp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/android/pytorch_android/src/main/cpp`):

- [`pytorch_jni_common.h_kw.md_docs.md`](./pytorch_jni_common.h_kw.md_docs.md)
- [`pytorch_jni_common.cpp_docs.md_docs.md`](./pytorch_jni_common.cpp_docs.md_docs.md)
- [`cmake_macros.h_docs.md_docs.md`](./cmake_macros.h_docs.md_docs.md)
- [`pytorch_jni_jit.cpp_kw.md_docs.md`](./pytorch_jni_jit.cpp_kw.md_docs.md)
- [`cmake_macros.h_kw.md_docs.md`](./cmake_macros.h_kw.md_docs.md)
- [`pytorch_jni_common.cpp_kw.md_docs.md`](./pytorch_jni_common.cpp_kw.md_docs.md)
- [`pytorch_jni_lite.cpp_docs.md_docs.md`](./pytorch_jni_lite.cpp_docs.md_docs.md)
- [`pytorch_jni_jit.cpp_docs.md_docs.md`](./pytorch_jni_jit.cpp_docs.md_docs.md)
- [`pytorch_jni_lite.cpp_kw.md_docs.md`](./pytorch_jni_lite.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `pytorch_jni_common.h_docs.md_docs.md`
- **Keyword Index**: `pytorch_jni_common.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
