# Documentation: `cmake/Modules/FindSanitizer.cmake`

## File Metadata

- **Path**: `cmake/Modules/FindSanitizer.cmake`
- **Size**: 4,275 bytes (4.17 KB)
- **Type**: CMake Build Script
- **Extension**: `.cmake`

## File Purpose

This is a cmake build script that is part of the PyTorch project.

## Original Source

```cmake
# Find sanitizers
#
# This module sets the following targets:
#  Sanitizer::address
#  Sanitizer::thread
#  Sanitizer::undefined
#  Sanitizer::leak
#  Sanitizer::memory
include_guard(GLOBAL)

option(UBSAN_FLAGS "additional UBSAN flags" OFF)

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

set(_source_code
    [==[
  #include <stdio.h>
  int main() {
  printf("hello world!");
  return 0;
  }
  ]==])

include(CMakePushCheckState)
cmake_push_check_state(RESET)
foreach(sanitizer_name IN ITEMS address thread undefined leak memory)
  if(TARGET Sanitizer::${sanitizer_name})
    continue()
  endif()

  set(CMAKE_REQUIRED_FLAGS
      "-fsanitize=${sanitizer_name};-fno-omit-frame-pointer")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_C_COMPILER_ID STREQUAL
                                              "MSVC")
    if(sanitizer_name STREQUAL "address")
      set(CMAKE_REQUIRED_FLAGS "/fsanitize=${sanitizer_name}")
    else()
      continue()
    endif()
  endif()
  if(sanitizer_name STREQUAL "address")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL
                                                 "Clang")
      list(APPEND CMAKE_REQUIRED_FLAGS "-shared-libasan")
    endif()
  endif()
  if(sanitizer_name STREQUAL "undefined" AND UBSAN_FLAGS)
    list(APPEND CMAKE_REQUIRED_FLAGS "${UBSAN_FLAGS}")
  endif()
  if(sanitizer_name STREQUAL "memory")
    list(APPEND CMAKE_REQUIRED_FLAGS "-fsanitize-memory-track-origins=2")
  endif()

  set(CMAKE_REQUIRED_QUIET ON)
  set(_run_res 0)
  include(CheckCSourceRuns)
  include(CheckCXXSourceRuns)
  foreach(lang IN LISTS languages)
    if(lang STREQUAL C)
      check_c_source_runs("${_source_code}"
                        __${lang}_${sanitizer_name}_res)
      if(__${lang}_${sanitizer_name}_res)
        set(_run_res 1)
      endif()
    endif()
    if(lang STREQUAL CXX)
      check_cxx_source_runs("${_source_code}"
                        __${lang}_${sanitizer_name}_res)
      if(__${lang}_${sanitizer_name}_res)
        set(_run_res 1)
      endif()
    endif()
  endforeach()
  if(_run_res)
    add_library(Sanitizer::${sanitizer_name} INTERFACE IMPORTED GLOBAL)
    target_compile_options(
      Sanitizer::${sanitizer_name}
      INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
    )
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND NOT CMAKE_C_COMPILER_ID
                                                     STREQUAL "MSVC")
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
      )
    else()
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:/INCREMENTAL:NO>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>>:/INCREMENTAL:NO>
      )
    endif()

    if(sanitizer_name STREQUAL "address")
      target_compile_definitions(
        Sanitizer::${sanitizer_name}
        INTERFACE
          $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:_GLIBCXX_SANITIZE_VECTOR>
          $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:_GLIBCXX_SANITIZE_STD_ALLOCATOR>
      )
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>,$<CXX_COMPILER_ID:GNU>>:-lasan>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>,$<C_COMPILER_ID:GNU>>:-lasan>
      )
    endif()
    if(sanitizer_name STREQUAL "undefined")
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>,$<CXX_COMPILER_ID:GNU>>:-lubsan>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>,$<C_COMPILER_ID:GNU>>:-lubsan>
      )
    endif()
  endif()
endforeach()

cmake_pop_check_state()

```



## High-Level Overview

This file is part of the PyTorch framework located at `cmake/Modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `cmake/Modules`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`cmake/Modules`):

- [`FindOpenTelemetryApi.cmake_docs.md`](./FindOpenTelemetryApi.cmake_docs.md)
- [`FindOpenMP.cmake_docs.md`](./FindOpenMP.cmake_docs.md)
- [`FindGloo.cmake_docs.md`](./FindGloo.cmake_docs.md)
- [`FindCUSPARSELT.cmake_docs.md`](./FindCUSPARSELT.cmake_docs.md)
- [`FindAPL.cmake_docs.md`](./FindAPL.cmake_docs.md)
- [`FindSYCLToolkit.cmake_docs.md`](./FindSYCLToolkit.cmake_docs.md)
- [`FindBLIS.cmake_docs.md`](./FindBLIS.cmake_docs.md)
- [`FindNCCL.cmake_docs.md`](./FindNCCL.cmake_docs.md)
- [`Findpybind11.cmake_docs.md`](./Findpybind11.cmake_docs.md)


## Cross-References

- **File Documentation**: `FindSanitizer.cmake_docs.md`
- **Keyword Index**: `FindSanitizer.cmake_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
