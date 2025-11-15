# Documentation: `torch/csrc/export/pt2_archive_constants.h`

## File Metadata

- **Path**: `torch/csrc/export/pt2_archive_constants.h`
- **Size**: 4,673 bytes (4.56 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <array>
#include <string_view>

namespace torch::_export::archive_spec {

#define FORALL_CONSTANTS(DO)                                                   \
  DO(ARCHIVE_ROOT_NAME, "package")                                             \
  /* Archive format */                                                         \
  DO(ARCHIVE_FORMAT_PATH, "archive_format")                                    \
  DO(ARCHIVE_FORMAT_VALUE, "pt2")                                              \
  /* Archive version */                                                        \
  DO(ARCHIVE_VERSION_PATH, "archive_version")                                  \
  DO(ARCHIVE_VERSION_VALUE, "0") /* Sep.4.2024: This is the initial version of \
                                    the PT2 Archive Spec */                    \
  /*                                                                           \
   * ######## Note on updating ARCHIVE_VERSION_VALUE ########                  \
   * When there is a BC breaking change to the PT2 Archive Spec,               \
   * e.g. deleting a folder, or changing the naming convention of the          \
   * following fields it would require bumping the ARCHIVE_VERSION_VALUE       \
   * Archive reader would need corresponding changes to support loading both   \
   * the current and older versions of the PT2 Archive.                        \
   */                                                                          \
  /* Model definitions */                                                      \
  DO(MODELS_DIR, "models/")                                                    \
  DO(MODELS_FILENAME_FORMAT, "models/{}.json") /* {model_name} */              \
  /* AOTInductor artifacts */                                                  \
  DO(AOTINDUCTOR_DIR, "data/aotinductor/")                                     \
  /* MTIA artifacts */                                                         \
  DO(MTIA_DIR, "data/mtia")                                                    \
  /* weights, including parameters and buffers */                              \
  DO(WEIGHTS_DIR, "data/weights/")                                             \
  DO(WEIGHT_FILENAME_PREFIX, "weight_")                                        \
  DO(WEIGHTS_PARAM_CONFIG_FORMAT, "data/weights/{}_model_param_config.json")   \
  DO(WEIGHTS_CONFIG_FILENAME_FORMAT, "data/weights/{}_weights_config.json")    \
  /* constants, including tensor_constants, non-persistent buffers and script  \
   * objects */                                                                \
  DO(CONSTANTS_DIR, "data/constants/")                                         \
  DO(CONSTANTS_PARAM_CONFIG_FORMAT,                                            \
     "data/constants/{}_model_constants_config.json")                          \
  DO(CONSTANTS_CONFIG_FILENAME_FORMAT,                                         \
     "data/constants/{}_constants_config.json")                                \
  DO(TENSOR_CONSTANT_FILENAME_PREFIX, "tensor_")                               \
  DO(CUSTOM_OBJ_FILENAME_PREFIX, "custom_obj_")                                \
  /* example inputs */                                                         \
  DO(SAMPLE_INPUTS_DIR, "data/sample_inputs/")                                 \
  DO(SAMPLE_INPUTS_FILENAME_FORMAT,                                            \
     "data/sample_inputs/{}.pt") /* {model_name} */                            \
  /* ExecuTorch artifacts, including PTE files */                              \
  DO(EXECUTORCH_DIR, "data/executorch/")                                       \
  /* extra folder */                                                           \
  DO(EXTRA_DIR, "extra/")                                                      \
  DO(MODULE_INFO_PATH, "extra/module_info.json")                               \
  /* xl_model_weights, this folder is used for storing per-feature-weights for \
   * remote net data in this folder is consume by Predictor, and is not        \
   * intended to be used by Sigmoid */                                         \
  DO(XL_MODEL_WEIGHTS_DIR, "xl_model_weights/")                                \
  DO(XL_MODEL_WEIGHTS_PARAM_CONFIG_PATH, "xl_model_weights/model_param_config")

#define DEFINE_GLOBAL(NAME, VALUE) \
  inline constexpr std::string_view NAME = VALUE;
FORALL_CONSTANTS(DEFINE_GLOBAL)
#undef DEFINE_GLOBAL

#define DEFINE_ENTRY(NAME, VALUE) std::pair(#NAME, VALUE),
inline constexpr std::array kAllConstants{FORALL_CONSTANTS(DEFINE_ENTRY)};
#undef DEFINE_ENTRY

#undef FORALL_CONSTANTS
} // namespace torch::_export::archive_spec

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `array`
- `string_view`


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

Files in the same folder (`torch/csrc/export`):

- [`pybind.cpp_docs.md`](./pybind.cpp_docs.md)
- [`example_upgraders.h_docs.md`](./example_upgraders.h_docs.md)
- [`example_upgraders.cpp_docs.md`](./example_upgraders.cpp_docs.md)
- [`upgrader.h_docs.md`](./upgrader.h_docs.md)
- [`pybind.h_docs.md`](./pybind.h_docs.md)
- [`upgrader.cpp_docs.md`](./upgrader.cpp_docs.md)


## Cross-References

- **File Documentation**: `pt2_archive_constants.h_docs.md`
- **Keyword Index**: `pt2_archive_constants.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
