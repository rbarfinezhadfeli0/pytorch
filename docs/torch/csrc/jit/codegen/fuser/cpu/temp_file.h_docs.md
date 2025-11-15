# Documentation: `torch/csrc/jit/codegen/fuser/cpu/temp_file.h`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/fuser/cpu/temp_file.h`
- **Size**: 2,891 bytes (2.82 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>

#ifdef _WIN32
#include <WinError.h>
#include <c10/util/Unicode.h>
#include <c10/util/win32-headers.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdio.h>
#include <sys/stat.h>
#include <random>
#else
#include <unistd.h>
#endif

#include <string>
#include <vector>

namespace torch::jit::fuser::cpu {

#ifdef _MSC_VER
inline int wmkstemps(wchar_t* tmpl, int suffix_len) {
  int len;
  wchar_t* name;
  int fd = -1;
  int save_errno = errno;

  len = wcslen(tmpl);
  if (len < 6 + suffix_len ||
      wcsncmp(&tmpl[len - 6 - suffix_len], L"XXXXXX", 6)) {
    return -1;
  }

  name = &tmpl[len - 6 - suffix_len];

  std::random_device rd;
  do {
    for (unsigned i = 0; i < 6; ++i) {
      name[i] = "abcdefghijklmnopqrstuvwxyz0123456789"[rd() % 36];
    }

    fd = _wopen(tmpl, _O_RDWR | _O_CREAT | _O_EXCL, _S_IWRITE | _S_IREAD);
  } while (errno == EEXIST);

  if (fd >= 0) {
    errno = save_errno;
    return fd;
  } else {
    return -1;
  }
}
#endif

struct TempFile {
  AT_DISALLOW_COPY_AND_ASSIGN(TempFile);

  TempFile(const std::string& t, int suffix) {
#ifdef _MSC_VER
    auto wt = c10::u8u16(t);
    std::vector<wchar_t> tt(wt.c_str(), wt.c_str() + wt.size() + 1);
    int fd = wmkstemps(tt.data(), suffix);
    AT_ASSERT(fd != -1);
    file_ = _wfdopen(fd, L"r+");
    auto wname = std::wstring(tt.begin(), tt.end() - 1);
    name_ = c10::u16u8(wname);
#else
    // mkstemps edits its first argument in places
    // so we make a copy of the string here, including null terminator
    std::vector<char> tt(t.c_str(), t.c_str() + t.size() + 1);
    int fd = mkstemps(tt.data(), suffix);
    AT_ASSERT(fd != -1);
    file_ = fdopen(fd, "r+");
    // - 1 because tt.size() includes the null terminator,
    // but std::string does not expect one
    name_ = std::string(tt.begin(), tt.end() - 1);
#endif
  }

  const std::string& name() const {
    return name_;
  }

  void sync() {
    fflush(file_);
  }

  void write(const std::string& str) {
    size_t result = fwrite(str.c_str(), 1, str.size(), file_);
    AT_ASSERT(str.size() == result);
  }

#ifdef _MSC_VER
  void close() {
    if (file_ != nullptr) {
      fclose(file_);
    }
    file_ = nullptr;
  }
#endif

  FILE* file() {
    return file_;
  }

  ~TempFile() {
#ifdef _MSC_VER
    if (file_ != nullptr) {
      fclose(file_);
    }
    auto wname = c10::u8u16(name_);
    if (!wname.empty() && _waccess(wname.c_str(), 0) != -1) {
      _wunlink(wname.c_str());
    }
#else
    if (file_ != nullptr) {
      // unlink first to ensure another mkstemps doesn't
      // race between close and unlink
      unlink(name_.c_str());
      fclose(file_);
    }
#endif
  }

 private:
  FILE* file_ = nullptr;
  std::string name_;
};

} // namespace torch::jit::fuser::cpu

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TempFile`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/fuser/cpu`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/Utils.h`
- `c10/util/Exception.h`
- `torch/csrc/Export.h`
- `WinError.h`
- `c10/util/Unicode.h`
- `c10/util/win32-headers.h`
- `fcntl.h`
- `io.h`
- `process.h`
- `stdio.h`
- `sys/stat.h`
- `random`
- `unistd.h`
- `string`
- `vector`


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

Files in the same folder (`torch/csrc/jit/codegen/fuser/cpu`):

- [`fused_kernel.cpp_docs.md`](./fused_kernel.cpp_docs.md)
- [`fused_kernel.h_docs.md`](./fused_kernel.h_docs.md)
- [`resource_strings.h_docs.md`](./resource_strings.h_docs.md)


## Cross-References

- **File Documentation**: `temp_file.h_docs.md`
- **Keyword Index**: `temp_file.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
