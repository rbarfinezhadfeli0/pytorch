# Documentation: `test/cpp/aoti_abi_check/CMakeLists.txt`

## File Metadata

- **Path**: `test/cpp/aoti_abi_check/CMakeLists.txt`
- **Size**: 3,747 bytes (3.66 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```
# Skip on windows
if(WIN32)
  return()
endif()

set(AOTI_ABI_CHECK_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_abi_check)

# Build the cpp gtest binary containing the cpp-only tests.
set(AOTI_ABI_CHECK_TEST_SRCS
  ${AOTI_ABI_CHECK_TEST_ROOT}/main.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_cast.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_devicetype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dispatch.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dispatch_v2.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dtype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_exception.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_headeronlyarrayref.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_macros.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_math.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_metaprogramming.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_rand.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_scalartype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_typelist.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_typetraits.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec_half.cpp
)

# The below are tests that require CPU_CAPABILITY setup
# You may think test_vec.cpp needs to be in there, but it does not.
set(AOTI_ABI_CHECK_VEC_TEST_SRCS
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec_half.cpp
)

add_executable(test_aoti_abi_check
  ${AOTI_ABI_CHECK_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aoti_abi_check PRIVATE USE_GTEST)

# WARNING: DO NOT LINK torch!!!
# The purpose is to check if the used aten/c10 headers are written in a header-only way
target_link_libraries(test_aoti_abi_check PRIVATE gtest_main sleef)
target_include_directories(test_aoti_abi_check PRIVATE ${ATen_CPU_INCLUDE})
if(NOT USE_SYSTEM_SLEEF)
  target_include_directories(test_aoti_abi_check PRIVATE ${CMAKE_BINARY_DIR}/include)
endif()

# Disable unused-variable warnings for variables that are only used to test compilation
target_compile_options_if_supported(test_aoti_abi_check -Wno-unused-variable)
target_compile_options_if_supported(test_aoti_abi_check -Wno-unused-but-set-variable)
# Add -Wno-dangling-pointer for GCC 13
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  target_compile_options_if_supported(test_aoti_abi_check -Wno-dangling-pointer)
endif()

foreach(test_src ${AOTI_ABI_CHECK_VEC_TEST_SRCS})
  foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
    get_filename_component(test_name ${test_src} NAME_WE)
    list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
    list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
    separate_arguments(FLAGS UNIX_COMMAND "${FLAGS}")
    add_executable(${test_name}_${CPU_CAPABILITY} "${test_src}")

    target_link_libraries(${test_name}_${CPU_CAPABILITY} PRIVATE gtest_main sleef)
    target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE ${ATen_CPU_INCLUDE})
    if(NOT USE_SYSTEM_SLEEF)
      target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE ${CMAKE_BINARY_DIR}/include)
    endif()

    # Define CPU_CAPABILITY and CPU_CAPABILITY_XXX macros for conditional compilation
    target_compile_definitions(${test_name}_${CPU_CAPABILITY} PRIVATE CPU_CAPABILITY=${CPU_CAPABILITY} CPU_CAPABILITY_${CPU_CAPABILITY})
    target_compile_options(${test_name}_${CPU_CAPABILITY} PRIVATE ${FLAGS})
    target_compile_options_if_supported(${test_name}_${CPU_CAPABILITY} -Wno-unused-variable)
    target_compile_options_if_supported(${test_name}_${CPU_CAPABILITY} -Wno-unused-but-set-variable)
  endforeach()
endforeach()

if(INSTALL_TEST)
  install(TARGETS test_aoti_abi_check DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aoti_abi_check> DESTINATION bin OPTIONAL)
  endif()
endif()

```



## High-Level Overview

This file is part of the PyTorch framework located at `test/cpp/aoti_abi_check`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/aoti_abi_check`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python test/cpp/aoti_abi_check/CMakeLists.txt
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/aoti_abi_check`):

- [`test_metaprogramming.cpp_docs.md`](./test_metaprogramming.cpp_docs.md)
- [`test_headeronlyarrayref.cpp_docs.md`](./test_headeronlyarrayref.cpp_docs.md)
- [`test_cast.cpp_docs.md`](./test_cast.cpp_docs.md)
- [`test_scalartype.cpp_docs.md`](./test_scalartype.cpp_docs.md)
- [`test_typetraits.cpp_docs.md`](./test_typetraits.cpp_docs.md)
- [`test_dtype.cpp_docs.md`](./test_dtype.cpp_docs.md)
- [`test_math.cpp_docs.md`](./test_math.cpp_docs.md)
- [`test_dispatch.cpp_docs.md`](./test_dispatch.cpp_docs.md)
- [`test_exception.cpp_docs.md`](./test_exception.cpp_docs.md)


## Cross-References

- **File Documentation**: `CMakeLists.txt_docs.md`
- **Keyword Index**: `CMakeLists.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
