# Documentation: `aten/src/ATen/DynamicLibrary.cpp`

## File Metadata

- **Path**: `aten/src/ATen/DynamicLibrary.cpp`
- **Size**: 2,698 bytes (2.63 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/Exception.h>
#include <c10/util/Unicode.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/Utils.h>

#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#else
#include <c10/util/win32-headers.h>
#endif

namespace at {


#ifndef C10_MOBILE
#ifndef _WIN32

// Unix

static void* checkDL(void* x) {
  if (!x) {
    TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen or dlsym: ", dlerror());
  }

  return x;
}
DynamicLibrary::DynamicLibrary(const char* name, const char* alt_name, bool leak_handle_): leak_handle(leak_handle_), handle(dlopen(name, RTLD_LOCAL | RTLD_NOW)) {
  if (!handle) {
    if (alt_name) {
      handle = dlopen(alt_name, RTLD_LOCAL | RTLD_NOW);
      if (!handle) {
        TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen for library ", name, "and ", alt_name);
      }
    } else {
      TORCH_CHECK_WITH(DynamicLibraryError, false, "Error in dlopen: ", dlerror());
    }
  }
}

void* DynamicLibrary::sym(const char* name) {
  AT_ASSERT(handle);
  return checkDL(dlsym(handle, name));
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle || leak_handle) {
    return;
  }
  dlclose(handle);
}

#else

// Windows

DynamicLibrary::DynamicLibrary(const char* name, const char* alt_name, bool leak_handle_): leak_handle(leak_handle_) {
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  HMODULE theModule;
  bool reload = true;
  auto wname = c10::u8u16(name);
  // Check if LOAD_LIBRARY_SEARCH_DEFAULT_DIRS is supported
  if (GetProcAddress(GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") != NULL) {
    theModule = LoadLibraryExW(
        wname.c_str(),
        NULL,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (theModule != NULL || (GetLastError() != ERROR_MOD_NOT_FOUND)) {
      reload = false;
    }
  }

  if (reload) {
    theModule = LoadLibraryW(wname.c_str());
  }

  if (theModule) {
    handle = theModule;
  } else {
    char buf[256];
    DWORD dw = GetLastError();
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  buf, (sizeof(buf) / sizeof(char)), NULL);
    TORCH_CHECK_WITH(DynamicLibraryError, false, "error in LoadLibrary for ", name, ". WinError ", dw, ": ", buf);
  }
}

void* DynamicLibrary::sym(const char* name) {
  AT_ASSERT(handle);
  FARPROC procAddress = GetProcAddress((HMODULE)handle, name);
  if (!procAddress) {
    TORCH_CHECK_WITH(DynamicLibraryError, false, "error in GetProcAddress");
  }
  return (void*)procAddress;
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle || leak_handle) {
    return;
  }
  FreeLibrary((HMODULE)handle);
}

#endif
#endif

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `c10/util/Unicode.h`
- `ATen/DynamicLibrary.h`
- `ATen/Utils.h`
- `dlfcn.h`
- `libgen.h`
- `c10/util/win32-headers.h`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `DynamicLibrary.cpp_docs.md`
- **Keyword Index**: `DynamicLibrary.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
