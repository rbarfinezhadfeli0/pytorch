# Documentation: `cmake/Modules_CUDA_fix/FindCUDNN.cmake`

## File Metadata

- **Path**: `cmake/Modules_CUDA_fix/FindCUDNN.cmake`
- **Size**: 3,085 bytes (3.01 KB)
- **Type**: CMake Build Script
- **Extension**: `.cmake`

## File Purpose

This is a cmake build script that is part of the PyTorch project.

## Original Source

```cmake
# Find the CUDNN libraries
#
# The following variables are optionally searched for defaults
#  CUDNN_ROOT: Base directory where CUDNN is found
#  CUDNN_INCLUDE_DIR: Directory where CUDNN header is searched for
#  CUDNN_LIBRARY: Directory where CUDNN library is searched for
#  CUDNN_STATIC: Are we looking for a static library? (default: no)
#
# The following are set after configuration is done:
#  CUDNN_FOUND
#  CUDNN_INCLUDE_PATH
#  CUDNN_LIBRARY_PATH
#

include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} CACHE PATH "Folder containing NVIDIA cuDNN")
if (DEFINED $ENV{CUDNN_ROOT_DIR})
  message(WARNING "CUDNN_ROOT_DIR is deprecated. Please set CUDNN_ROOT instead.")
endif()
list(APPEND CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})

# Compatible layer for CMake <3.12. CUDNN_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${CUDNN_ROOT})

set(CUDNN_INCLUDE_DIR $ENV{CUDNN_INCLUDE_DIR} CACHE PATH "Folder containing NVIDIA cuDNN header files")

find_path(CUDNN_INCLUDE_PATH cudnn.h
  HINTS ${CUDNN_INCLUDE_DIR}
  PATH_SUFFIXES cuda/include cuda include)

option(CUDNN_STATIC "Look for static CUDNN" OFF)
if (CUDNN_STATIC)
  set(CUDNN_LIBNAME "libcudnn_static.a")
else()
  set(CUDNN_LIBNAME "cudnn")
endif()

set(CUDNN_LIBRARY $ENV{CUDNN_LIBRARY} CACHE PATH "Path to the cudnn library file (e.g., libcudnn.so)")
if (CUDNN_LIBRARY MATCHES ".*cudnn_static.a" AND NOT CUDNN_STATIC)
  message(WARNING "CUDNN_LIBRARY points to a static library (${CUDNN_LIBRARY}) but CUDNN_STATIC is OFF.")
endif()

find_library(CUDNN_LIBRARY_PATH ${CUDNN_LIBNAME}
  PATHS ${CUDNN_LIBRARY}
  PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARY_PATH CUDNN_INCLUDE_PATH)

if(CUDNN_FOUND)
  # Get cuDNN version
  if(EXISTS ${CUDNN_INCLUDE_PATH}/cudnn_version.h)
    file(READ ${CUDNN_INCLUDE_PATH}/cudnn_version.h CUDNN_HEADER_CONTENTS)
  else()
    file(READ ${CUDNN_INCLUDE_PATH}/cudnn.h CUDNN_HEADER_CONTENTS)
  endif()
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
               CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
               CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
               CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
               CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
  # Assemble cuDNN version
  if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "?")
  else()
    set(CUDNN_VERSION
        "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  endif()
endif()

mark_as_advanced(CUDNN_ROOT CUDNN_INCLUDE_DIR CUDNN_LIBRARY CUDNN_VERSION)

```



## High-Level Overview

This file is part of the PyTorch framework located at `cmake/Modules_CUDA_fix`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `cmake/Modules_CUDA_fix`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`cmake/Modules_CUDA_fix`):

- [`FindCUDA.cmake_docs.md`](./FindCUDA.cmake_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)


## Cross-References

- **File Documentation**: `FindCUDNN.cmake_docs.md`
- **Keyword Index**: `FindCUDNN.cmake_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
