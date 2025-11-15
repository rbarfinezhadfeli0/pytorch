# Documentation: `cmake/ProtoBufPatch.cmake`

## File Metadata

- **Path**: `cmake/ProtoBufPatch.cmake`
- **Size**: 3,650 bytes (3.56 KB)
- **Type**: CMake Build Script
- **Extension**: `.cmake`

## File Purpose

This is a cmake build script that is part of the PyTorch project.

## Original Source

```cmake
# CMake file to replace the string contents in ONNX, Caffe, and Caffe2 proto.
# Usage example:
#   cmake -DFILENAME=caffe2.pb.h -DLOCAL_PROTOBUF=ON -P ProtoBufPatch.cmake

file(READ ${FILENAME} content)

if(NOT SYSTEM_PROTOBUF)
  # protobuf-3.6.0 pattern
  string(
    REPLACE
    "::google::protobuf::internal::GetEmptyStringAlreadyInited"
    "GetEmptyStringAlreadyInited"
    content
    "${content}")

  # protobuf-3.8.0+ pattern
  string(
    REPLACE
    "::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited"
    "GetEmptyStringAlreadyInited"
    content
    "${content}")

  string(
    REPLACE
    "PROTOBUF_CONSTEXPR"
    ""
    content
    "${content}")

  # https://github.com/protocolbuffers/protobuf/commit/0400cca3236de1ca303af38bf81eab332d042b7c
  # changes PROTOBUF_CONSTEXPR to constexpr, which breaks windows
  # build.
  if(MSVC)
    string(
      REGEX REPLACE
      "static constexpr ([^ ]+) ([^ ]+) ="
      "static \\1 const \\2 ="
      content
      "${content}")
  endif()

  foreach(ns ${NAMESPACES})
    # Insert "const ::std::string& GetEmptyStringAlreadyInited();" within
    # the namespace and make sure we only do it once in the file. Unfortunately
    # using string(REPLACE ...) doesn't work because it will replace at all
    # locations and there might be multiple declarations of the namespace
    # depending on how the proto is structured.
    set(search "namespace ${ns} {")
    string(LENGTH "${search}" search_len)
    string(FIND "${content}" "${search}" pos)
    if(${pos} GREATER -1)
      math(EXPR pos "${pos}+${search_len}")
      string(SUBSTRING "${content}" 0 ${pos} content_pre)
      string(SUBSTRING "${content}" ${pos} -1 content_post)
      string(
        CONCAT
        content
        "${content_pre}"
        " const ::std::string& GetEmptyStringAlreadyInited(); "
        "${content_post}")
    endif()
  endforeach()

  # The moving constructor is defined in the header file, which will cause
  # a link error that claims that the vftable is not found. Luckily, we
  # could move the definition into the source file to solve the problem.
  list(LENGTH NAMESPACES ns_count)
  if("${FILENAME}" MATCHES ".pb.h" AND ns_count EQUAL 1)
    string(REPLACE ".pb.h" ".pb.cc" SOURCE_FILENAME ${FILENAME})
    file(READ ${SOURCE_FILENAME} content_cc_origin)

    string(REGEX MATCHALL "([a-zA-Z_]+)\\([a-zA-Z_]+&& from\\) noexcept[^}]*}" content_cc "${content}")
    string(REGEX REPLACE "};" "}\n" content_cc "${content_cc}")
    string(REGEX REPLACE "([a-zA-Z_]+)\\([a-zA-Z_]+&& from\\) noexcept" "  \\1::\\1(\\1&& from) noexcept" content_cc "${content_cc}")
    set(content_cc "${content_cc_origin}\nnamespace ${NAMESPACES} {\n#if LANG_CXX11\n${content_cc}\n#endif\n}")

    string(REGEX REPLACE "([a-zA-Z_]+)\\([a-zA-Z_]+&& from\\) noexcept([^}]*)}" "\\1(\\1&& from) noexcept;" content "${content}")

    file(WRITE ${SOURCE_FILENAME} "${content_cc}")
  endif()
endif(NOT SYSTEM_PROTOBUF)

# constexpr int TensorBoundShape_DimType_DimType_ARRAYSIZE = TensorBoundShape_DimType_DimType_MAX + 1;
# throws
# error: more than one operator "+" matches these operands:
#     built-in operator "arithmetic + arithmetic"
#     function "c10::operator+(int, c10::BFloat16)"
#     function "c10::operator+(c10::BFloat16, int)"
#     function "c10::operator+(int, c10::Half)"
#     function "c10::operator+(c10::Half, int)"
#   operand types are: const caffe2::ExternalDataProto_SourceType + int
string(
  REGEX REPLACE
  "constexpr ([^ ]+) ([^ ]+_ARRAYSIZE) = ([^ ]+_MAX) \\+ 1;"
  "constexpr \\1 \\2 = static_cast<\\1>(\\3) + 1;"
  content
  "${content}")

file(WRITE ${FILENAME} "${content}")

```



## High-Level Overview

This file is part of the PyTorch framework located at `cmake`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `cmake`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`cmake`):

- [`Metal.cmake_docs.md`](./Metal.cmake_docs.md)
- [`Codegen.cmake_docs.md`](./Codegen.cmake_docs.md)
- [`prioritized_text.txt_docs.md`](./prioritized_text.txt_docs.md)
- [`DebugHelper.cmake_docs.md`](./DebugHelper.cmake_docs.md)
- [`VulkanDependencies.cmake_docs.md`](./VulkanDependencies.cmake_docs.md)
- [`Summary.cmake_docs.md`](./Summary.cmake_docs.md)
- [`BuildVariables.cmake_docs.md`](./BuildVariables.cmake_docs.md)
- [`ProtoBuf.cmake_docs.md`](./ProtoBuf.cmake_docs.md)
- [`VulkanCodegen.cmake_docs.md`](./VulkanCodegen.cmake_docs.md)


## Cross-References

- **File Documentation**: `ProtoBufPatch.cmake_docs.md`
- **Keyword Index**: `ProtoBufPatch.cmake_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
