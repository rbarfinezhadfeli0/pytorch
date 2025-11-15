# Documentation: `cmake/Modules/FindNCCL.cmake`

## File Metadata

- **Path**: `cmake/Modules/FindNCCL.cmake`
- **Size**: 3,693 bytes (3.61 KB)
- **Type**: CMake Build Script
- **Extension**: `.cmake`

## File Purpose

This is a cmake build script that is part of the PyTorch project.

## Original Source

```cmake
# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/facebookarchive/caffe2/issues/1601

set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL headers")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL libraries")
set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

if ($ENV{NCCL_ROOT_DIR})
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
endif()
list(APPEND NCCL_ROOT $ENV{NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${NCCL_INCLUDE_DIR})

if (USE_STATIC_NCCL)
  MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  SET(NCCL_LIBNAME "nccl_static")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(NCCL_LIBNAME "nccl")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_library(NCCL_LIBRARIES
  NAMES ${NCCL_LIBNAME}
  HINTS ${NCCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if(NCCL_FOUND)  # obtaining NCCL version and some sanity checks
  set (NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
  message (STATUS "Determining NCCL version from ${NCCL_HEADER_FILE}...")
  set (OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  list (APPEND CMAKE_REQUIRED_INCLUDES ${NCCL_INCLUDE_DIRS})
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(NCCL_VERSION_CODE nccl.h NCCL_VERSION_DEFINED)

  # this condition check only works for non static NCCL linking
  if (NCCL_VERSION_DEFINED AND NOT USE_STATIC_NCCL)
    set(file "${PROJECT_BINARY_DIR}/detect_nccl_version.cc")
    file(WRITE ${file} "
      #include <iostream>
      #include <nccl.h>
      int main()
      {
        std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH << std::endl;
        int x;
        ncclGetVersion(&x);
        return x == NCCL_VERSION_CODE;
      }
")
    try_run(NCCL_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
          RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_HEADER
          CMAKE_FLAGS  "-DINCLUDE_DIRECTORIES=${NCCL_INCLUDE_DIRS}"
          LINK_LIBRARIES ${NCCL_LIBRARIES})
    if (NOT NCCL_VERSION_MATCHED)
      message(FATAL_ERROR "Found NCCL header version and library version do not match! \
(include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES}) Please set NCCL_INCLUDE_DIR and NCCL_LIB_DIR manually.")
    endif()
    message(STATUS "NCCL version: ${NCCL_VERSION_FROM_HEADER}")
  endif ()

  set (CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()

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
- Implements or uses **caching** mechanisms.
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
- [`FindSanitizer.cmake_docs.md`](./FindSanitizer.cmake_docs.md)
- [`FindOpenMP.cmake_docs.md`](./FindOpenMP.cmake_docs.md)
- [`FindGloo.cmake_docs.md`](./FindGloo.cmake_docs.md)
- [`FindCUSPARSELT.cmake_docs.md`](./FindCUSPARSELT.cmake_docs.md)
- [`FindAPL.cmake_docs.md`](./FindAPL.cmake_docs.md)
- [`FindSYCLToolkit.cmake_docs.md`](./FindSYCLToolkit.cmake_docs.md)
- [`FindBLIS.cmake_docs.md`](./FindBLIS.cmake_docs.md)
- [`Findpybind11.cmake_docs.md`](./Findpybind11.cmake_docs.md)


## Cross-References

- **File Documentation**: `FindNCCL.cmake_docs.md`
- **Keyword Index**: `FindNCCL.cmake_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
