# Documentation: `docs/torch/csrc/distributed/c10d/ParamCommsUtils.hpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/ParamCommsUtils.hpp_docs.md`
- **Size**: 11,266 bytes (11.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/ParamCommsUtils.hpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/ParamCommsUtils.hpp`
- **Size**: 8,751 bytes (8.55 KB)
- **Type**: C++ Header File
- **Extension**: `.hpp`

## File Purpose

This is a c++ header file that is part of the PyTorch project.

## Original Source

```cpp
#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>
#include <c10/macros/Macros.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <string>
#include <vector>

namespace torch {

class TORCH_API ParamCommsDebugInfo : public c10::DebugInfoBase {
 public:
  ParamCommsDebugInfo() = default;
  ParamCommsDebugInfo(
      std::tuple<std::string, std::string> pgName,
      int rank,
      std::string&& collName,
      int64_t inNelems,
      int64_t outNelems,
      at::ScalarType dType,
      std::vector<int64_t> inSplitSizes,
      std::vector<int64_t> outSplitSizes,
      int globalRankStart,
      int globalRankStride,
      int worldSize);

  ~ParamCommsDebugInfo() override = default;

  const std::string getProcessGroupName() const {
    return std::get<0>(pgName_);
  }

  const std::string getProcessGroupDesc() const {
    return std::get<1>(pgName_);
  }

  int getRank() const {
    return rank_;
  }

  int getWorldSize() const {
    return worldSize_;
  }

  int getGlobalRankStart() const {
    return globalRankStart_;
  }

  int getGlobalRankStride() const {
    return globalRankStride_;
  }

  const std::string getCollectiveName() const {
    return collectiveName_;
  }

  int64_t getInMessageNelems() const {
    return inMessageNelems_;
  }

  int64_t getOutMessageNelems() const {
    return outMessageNelems_;
  }

  at::ScalarType getDType() const {
    return dType_;
  }

  const std::vector<int64_t>& getInputSplitSizes() const {
    return inputSplitSizes_;
  }

  const std::vector<int64_t>& getOutputSplitSizes() const {
    return outputSplitSizes_;
  }

  const std::vector<int64_t>& getGroupRanks() const {
    return groupRanks_;
  }

 private:
  std::tuple<std::string, std::string> pgName_; // <group_name, group_desc>
  int rank_{};
  int worldSize_{};
  std::string collectiveName_;
  int64_t inMessageNelems_{};
  int64_t outMessageNelems_{};
  at::ScalarType dType_ = at::kByte;
  std::vector<int64_t> inputSplitSizes_;
  std::vector<int64_t> outputSplitSizes_;
  int globalRankStart_{};
  int globalRankStride_{};
  std::vector<int64_t> groupRanks_;
};

#define RECORD_PARAM_COMMS(                                                    \
    seq,                                                                       \
    pgName,                                                                    \
    rank,                                                                      \
    collName,                                                                  \
    inNelems,                                                                  \
    outNelems,                                                                 \
    dType,                                                                     \
    inSplitSizes,                                                              \
    outSplitSizes,                                                             \
    globalRankStart,                                                           \
    globalRankStride,                                                          \
    worldSize)                                                                 \
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>(          \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inNelems,                                                                \
      outNelems,                                                               \
      dType,                                                                   \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize);                                                              \
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  std::initializer_list<const c10::IValue> paramList = {                       \
      seq,                                                                     \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize};                                                              \
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);                     \
  RECORD_FUNCTION(at::kParamCommsCallName, paramInputs);

#define RECORD_PARAM_COMMS_DATA(                                               \
    seq,                                                                       \
    pgName,                                                                    \
    InputTensors,                                                              \
    OutputTensors,                                                             \
    rank,                                                                      \
    collName,                                                                  \
    inNelems,                                                                  \
    outNelems,                                                                 \
    dType,                                                                     \
    inSplitSizes,                                                              \
    outSplitSizes,                                                             \
    globalRankStart,                                                           \
    globalRankStride,                                                          \
    worldSize)                                                                 \
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>(          \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inNelems,                                                                \
      outNelems,                                                               \
      dType,                                                                   \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize);                                                              \
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  std::initializer_list<const c10::IValue> paramList = {                       \
      c10::IValue(InputTensors),                                               \
      seq,                                                                     \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize};                                                              \
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);                     \
  RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(                                         \
      at::kParamCommsCallName,                                                 \
      paramInputs,                                                             \
      std::vector<c10::IValue>(1, c10::IValue(OutputTensors)));
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `ATen/record_function.h`
- `c10/macros/Macros.h`
- `c10/util/ThreadLocalDebugInfo.h`
- `string`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/csrc/distributed/c10d`):

- [`Utils.hpp_docs.md`](./Utils.hpp_docs.md)
- [`Ops.cpp_docs.md`](./Ops.cpp_docs.md)
- [`Store.hpp_docs.md`](./Store.hpp_docs.md)
- [`WinSockUtils.hpp_docs.md`](./WinSockUtils.hpp_docs.md)
- [`FakeProcessGroup.hpp_docs.md`](./FakeProcessGroup.hpp_docs.md)
- [`Work.cpp_docs.md`](./Work.cpp_docs.md)
- [`PrefixStore.hpp_docs.md`](./PrefixStore.hpp_docs.md)
- [`PyProcessGroup.hpp_docs.md`](./PyProcessGroup.hpp_docs.md)
- [`debug.h_docs.md`](./debug.h_docs.md)
- [`exception.h_docs.md`](./exception.h_docs.md)


## Cross-References

- **File Documentation**: `ParamCommsUtils.hpp_docs.md`
- **Keyword Index**: `ParamCommsUtils.hpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/csrc/distributed/c10d`):

- [`ProcessGroupWrapper.cpp_docs.md_docs.md`](./ProcessGroupWrapper.cpp_docs.md_docs.md)
- [`c10d.h_kw.md_docs.md`](./c10d.h_kw.md_docs.md)
- [`TCPStoreLibUvBackend.cpp_kw.md_docs.md`](./TCPStoreLibUvBackend.cpp_kw.md_docs.md)
- [`ProcessGroupGlooCuda.cpp_docs.md_docs.md`](./ProcessGroupGlooCuda.cpp_docs.md_docs.md)
- [`NanCheck.cu_docs.md_docs.md`](./NanCheck.cu_docs.md_docs.md)
- [`python_callback_work.hpp_kw.md_docs.md`](./python_callback_work.hpp_kw.md_docs.md)
- [`sequence_num.hpp_kw.md_docs.md`](./sequence_num.hpp_kw.md_docs.md)
- [`Functional.hpp_kw.md_docs.md`](./Functional.hpp_kw.md_docs.md)
- [`TCPStoreBackend.cpp_kw.md_docs.md`](./TCPStoreBackend.cpp_kw.md_docs.md)
- [`ProcessGroupUCC.cpp_kw.md_docs.md`](./ProcessGroupUCC.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ParamCommsUtils.hpp_docs.md_docs.md`
- **Keyword Index**: `ParamCommsUtils.hpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
