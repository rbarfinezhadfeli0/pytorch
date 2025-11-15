# Documentation: `cmake/Modules/FindSYCLToolkit.cmake`

## File Metadata

- **Path**: `cmake/Modules/FindSYCLToolkit.cmake`
- **Size**: 4,547 bytes (4.44 KB)
- **Type**: CMake Build Script
- **Extension**: `.cmake`

## File Purpose

This is a cmake build script that is part of the PyTorch project.

## Original Source

```cmake
# This will define the following variables:
# SYCL_FOUND               : True if the system has the SYCL library.
# SYCL_INCLUDE_DIR         : Include directories needed to use SYCL.
# SYCL_LIBRARY_DIR         : The path to the SYCL library.
# SYCL_LIBRARY             : SYCL library fullname.
# SYCL_COMPILER_VERSION    : SYCL compiler version.

include(FindPackageHandleStandardArgs)

set(SYCL_ROOT "")
if(DEFINED ENV{SYCL_ROOT})
  set(SYCL_ROOT $ENV{SYCL_ROOT})
elseif(DEFINED ENV{CMPLR_ROOT})
  set(SYCL_ROOT $ENV{CMPLR_ROOT})
else()
  # Use the default path to ensure proper linking with torch::xpurt when the user is working with libtorch.
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(SYCL_ROOT "/opt/intel/oneapi/compiler/latest")
  elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(SYCL_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest")
  endif()
  if(NOT EXISTS ${SYCL_ROOT})
    set(SYCL_ROOT "")
  endif()
endif()

string(COMPARE EQUAL "${SYCL_ROOT}" "" nosyclfound)
if(nosyclfound)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library not set!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

# Find SYCL compiler executable.
find_program(
  SYCL_COMPILER
  NAMES icx
  PATHS "${SYCL_ROOT}"
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )

function(parse_sycl_compiler_version version_number)
  # Execute the SYCL compiler with the --version flag to match the version string.
  execute_process(COMMAND ${SYCL_COMPILER} --version OUTPUT_VARIABLE SYCL_VERSION_STRING)
  string(REGEX REPLACE "Intel\\(R\\) (.*) Compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)" "\\2"
               SYCL_VERSION_STRING_MATCH ${SYCL_VERSION_STRING})
  string(REPLACE "." ";" SYCL_VERSION_LIST ${SYCL_VERSION_STRING_MATCH})
  # Split the version number list into major, minor, and patch components.
  list(GET SYCL_VERSION_LIST 0 VERSION_MAJOR)
  list(GET SYCL_VERSION_LIST 1 VERSION_MINOR)
  list(GET SYCL_VERSION_LIST 2 VERSION_PATCH)
  # Calculate the version number in the format XXXXYYZZ, using the formula (major * 10000 + minor * 100 + patch).
  math(EXPR VERSION_NUMBER_MATCH "${VERSION_MAJOR} * 10000 + ${VERSION_MINOR} * 100 + ${VERSION_PATCH}")
  set(${version_number} "${VERSION_NUMBER_MATCH}" PARENT_SCOPE)
endfunction()

if(SYCL_COMPILER)
  parse_sycl_compiler_version(SYCL_COMPILER_VERSION)
endif()

if(NOT SYCL_COMPILER_VERSION)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "Cannot parse sycl compiler version to get SYCL_COMPILER_VERSION!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

# Find include path from binary.
find_file(
  SYCL_INCLUDE_DIR
  NAMES include
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

# Find include/sycl path from include path.
find_file(
  SYCL_INCLUDE_SYCL_DIR
  NAMES sycl
  HINTS ${SYCL_ROOT}/include/
  NO_DEFAULT_PATH
  )

# Due to the unrecognized compilation option `-fsycl` in other compiler.
list(APPEND SYCL_INCLUDE_DIR ${SYCL_INCLUDE_SYCL_DIR})

# Find library directory from binary.
find_file(
  SYCL_LIBRARY_DIR
  NAMES lib lib64
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

# Define the old version of SYCL toolkit that is compatible with the current version of PyTorch.
set(PYTORCH_2_5_SYCL_TOOLKIT_VERSION 20249999)

# By default, we use libsycl.so on Linux and sycl.lib on Windows as the SYCL library name.
if (SYCL_COMPILER_VERSION VERSION_LESS_EQUAL PYTORCH_2_5_SYCL_TOOLKIT_VERSION)
  # Don't use if(WIN32) here since this requires cmake>=3.25 and file is installed
  # and used by other projects.
  # See: https://cmake.org/cmake/help/v3.25/variable/LINUX.html
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    # On Windows, the SYCL library is named sycl7.lib until PYTORCH_2_5_SYCL_TOOLKIT_VERSION.
    # sycl.lib is supported in the later version.
    set(sycl_lib_suffix "7")
  endif()
endif()

# Find SYCL library fullname.
find_library(
  SYCL_LIBRARY
  NAMES "sycl${sycl_lib_suffix}"
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

# Find OpenCL library fullname, which is a dependency of oneDNN.
find_library(
  OCL_LIBRARY
  NAMES OpenCL
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT SYCL_LIBRARY) OR (NOT OCL_LIBRARY))
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library is incomplete!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

find_package_handle_standard_args(
  SYCL
  FOUND_VAR SYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}"
  VERSION_VAR SYCL_COMPILER_VERSION
  )

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
- [`FindBLIS.cmake_docs.md`](./FindBLIS.cmake_docs.md)
- [`FindNCCL.cmake_docs.md`](./FindNCCL.cmake_docs.md)
- [`Findpybind11.cmake_docs.md`](./Findpybind11.cmake_docs.md)


## Cross-References

- **File Documentation**: `FindSYCLToolkit.cmake_docs.md`
- **Keyword Index**: `FindSYCLToolkit.cmake_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
