# Documentation: `aten/src/ATen/native/transformers/hip/flash_attn/ck/CMakeLists.txt`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/hip/flash_attn/ck/CMakeLists.txt`
- **Size**: 4,472 bytes (4.37 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This is a source file (.txt) that is part of the PyTorch project.

## Original Source

```
# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api fwd --receipt 4 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api fwd_splitkv --receipt 4 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_splitkv_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD_SPLITKV kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api fwd_appendkv --receipt 4 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_appendkv_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD_APPENDKV kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api bwd --receipt 4 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of BWD kernels via Python.")
endif()

# Generate the files for both fwd, fwd_splitkv, fwd_appendkv, and bwd
execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api fwd --receipt 4 --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api fwd_splitkv --receipt 4 --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD_SPLITKV kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api fwd_appendkv --receipt 4 --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD_APPENDKV kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api bwd --receipt 4 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate BWD kernels.")
endif()

# Change make_kernel to make_kernel_pt for fwd
execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd pass")
endif()

execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_splitkv_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd_splitkv pass")
endif()

execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_appendkv_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd appendkv pass")
endif()

# Change make_kernel to make_kernel_pt for bwd
execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the bwd pass")
endif()

# Change file extensions to .hip
execute_process(COMMAND bash -c "for file in ${CMAKE_CURRENT_LIST_DIR}/*.cpp; do mv -- \"$file\" \"\${file%.cpp}.hip\"; done"
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change the generated instances extensions from .cpp to .hpp")
endif()

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/transformers/hip/flash_attn/ck`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/hip/flash_attn/ck`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`aten/src/ATen/native/transformers/hip/flash_attn/ck`):

- [`add_make_kernel_pt.sh_docs.md`](./add_make_kernel_pt.sh_docs.md)
- [`launch_kernel_pt.hpp_docs.md`](./launch_kernel_pt.hpp_docs.md)
- [`me_ck_api.h_docs.md`](./me_ck_api.h_docs.md)


## Cross-References

- **File Documentation**: `CMakeLists.txt_docs.md`
- **Keyword Index**: `CMakeLists.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
