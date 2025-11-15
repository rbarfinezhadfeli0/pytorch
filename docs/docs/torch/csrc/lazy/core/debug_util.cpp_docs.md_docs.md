# Documentation: `docs/torch/csrc/lazy/core/debug_util.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/core/debug_util.cpp_docs.md`
- **Size**: 8,071 bytes (7.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/core/debug_util.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/core/debug_util.cpp`
- **Size**: 5,489 bytes (5.36 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/debug_util.h>

#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/unique.h>

#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace torch::lazy {
namespace {

std::string GetEnvString(const char* name, const std::string& defval) {
  const auto env = c10::utils::get_env(name);
  return env.value_or(defval);
}

DebugUtil::GraphFormat DefaultGraphFormat() {
  std::string fmt_str = GetEnvString("LTC_SAVE_TENSORS_FMT", "text");
  if (fmt_str == "text") {
    return DebugUtil::GraphFormat::kText;
  } else if (fmt_str == "backend") {
    return DebugUtil::GraphFormat::kBackend;
  } else if (fmt_str == "dot") {
    return DebugUtil::GraphFormat::kDot;
  }
  LOG(ERROR) << "Invalid save graph format: " << fmt_str;
  return DebugUtil::GraphFormat::kText;
}

std::unordered_set<std::string>* LoadExperiments() {
  std::unique_ptr<std::unordered_set<std::string>> xset =
      std::make_unique<std::unordered_set<std::string>>();
  std::string experiments = GetEnvString("LTC_EXPERIMENTAL", "");
  std::vector<std::string> experiment_list =
      torch::lazy::StrSplit(experiments, ':');
  for (auto& name : experiment_list) {
    xset->insert(name);
  }
  return xset.release();
}

} // namespace

static std::vector<SourceLocation> NoPythonFrames() {
  SourceLocation dummy_loc;
  dummy_loc.file = "No Python Frames";
  return {dummy_loc};
}

std::function<std::vector<SourceLocation>()>& GetPythonFramesFunction() {
  static std::function<std::vector<SourceLocation>()> func_ = NoPythonFrames;
  return func_;
}

DebugUtil::GraphFormat DebugUtil::GetDefaultGraphFormat() {
  static GraphFormat format = DefaultGraphFormat();
  return format;
}

std::string GetFirstUserFrameInPython() {
  std::string empty;
  if (!torch::lazy::GetPythonFramesFunction()) {
    return empty;
  }

  auto frames = torch::lazy::GetPythonFramesFunction()();

  for (auto i = frames.size(); i > 0; i--) {
    auto& loc = frames[i - 1];
    if (loc.file.find("site-packages") == std::string::npos) {
      std::stringstream ss;
      ss << loc.file << " " << loc.function << " " << loc.line;
      return ss.str();
    }
  }
  return empty;
}

std::string DebugUtil::GetTensorsGraphInfo(
    c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
    const std::vector<size_t>* indices,
    GraphFormat format) {
  std::vector<const torch::lazy::Node*> root_nodes;
  std::vector<torch::lazy::Value> root_values;
  std::vector<torch::lazy::hash_t> root_hashes;
  torch::lazy::Unique<torch::lazy::BackendDevice> unique_device;
  if (indices != nullptr) {
    for (auto index : *indices) {
      const torch::lazy::LazyTensorPtr& tensor = tensors[index];
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor->GetDevice());
      }
    }
  } else {
    for (auto& tensor : tensors) {
      torch::lazy::Value ir_value = tensor->CurrentIrValue();
      if (ir_value) {
        root_nodes.push_back(ir_value.node.get());
        root_hashes.push_back(ir_value.hash());
        root_values.push_back(std::move(ir_value));
        unique_device.set(tensor->GetDevice());
      }
    }
  }
  std::stringstream ss;
  // Call into a function pointer that may backed by python or empty depending
  // on runtime
  std::vector<SourceLocation> frames = GetPythonFramesFunction()();
  ss << "Python Stacktrace:\n";
  for (auto& location : frames) {
    ss << "  " << location.function << " (" << location.file << ":"
       << location.line << ")\n";
  }
  ss << "\nHashes: (";
  for (const auto i : c10::irange(root_hashes.size())) {
    if (i > 0) {
      ss << ", ";
    }
    ss << torch::lazy::HashToString(root_hashes[i]);
  }
  ss << ")\n";

  std::string graph_str;
  if (format == GraphFormat::kText) {
    graph_str = torch::lazy::DumpUtil::ToText(root_nodes);
  } else if (format == GraphFormat::kDot) {
    graph_str = torch::lazy::DumpUtil::ToDot(root_nodes);
  } else if (format == GraphFormat::kBackend) {
    graph_str = torch::lazy::DumpUtil::ToBackend(
        root_values,
        unique_device ? *unique_device : torch::lazy::BackendDevice());
  } else {
    LOG(ERROR) << "Invalid graph format: " << format;
  }
  ss << "\n## BEGIN_GRAPH\n" << graph_str << "\n## END_GRAPH\n\n";
  return ss.str();
}

void DebugUtil::SaveTensorsGraphInfo(
    const char* name,
    c10::ArrayRef<torch::lazy::LazyTensorPtr> tensors,
    const std::vector<size_t>* indices,
    GraphFormat format) {
  static const std::string save_file =
      GetEnvString("LTC_SAVE_TENSORS_FILE", "");
  if (!save_file.empty()) {
    static std::mutex lock;
    std::string info = GetTensorsGraphInfo(tensors, indices, format);
    std::lock_guard<std::mutex> guard(lock);
    std::ofstream graph_file(save_file, std::ios_base::app);
    graph_file << "[" << name << "]\n" << info << "\n";
  }
}

bool DebugUtil::ExperimentEnabled(const std::string& name) {
  static const std::unordered_set<std::string>* xset = LoadExperiments();
  return xset->find(name) != xset->end();
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `static`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/env.h`
- `c10/util/irange.h`
- `torch/csrc/lazy/core/debug_util.h`
- `torch/csrc/lazy/backend/backend_device.h`
- `torch/csrc/lazy/core/helpers.h`
- `torch/csrc/lazy/core/ir.h`
- `torch/csrc/lazy/core/ir_dump_util.h`
- `torch/csrc/lazy/core/ir_util.h`
- `torch/csrc/lazy/core/unique.h`
- `fstream`
- `mutex`
- `sstream`
- `unordered_set`


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

Files in the same folder (`torch/csrc/lazy/core`):

- [`hash.cpp_docs.md`](./hash.cpp_docs.md)
- [`shape_inference.cpp_docs.md`](./shape_inference.cpp_docs.md)
- [`tensor_impl.h_docs.md`](./tensor_impl.h_docs.md)
- [`helpers.h_docs.md`](./helpers.h_docs.md)
- [`tensor_impl.cpp_docs.md`](./tensor_impl.cpp_docs.md)
- [`ir_metadata.cpp_docs.md`](./ir_metadata.cpp_docs.md)
- [`ir_metadata.h_docs.md`](./ir_metadata.h_docs.md)
- [`trie.cpp_docs.md`](./trie.cpp_docs.md)
- [`cache.h_docs.md`](./cache.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)


## Cross-References

- **File Documentation**: `debug_util.cpp_docs.md`
- **Keyword Index**: `debug_util.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/csrc/lazy/core`):

- [`helpers.cpp_docs.md_docs.md`](./helpers.cpp_docs.md_docs.md)
- [`tensor_util.h_kw.md_docs.md`](./tensor_util.h_kw.md_docs.md)
- [`permutation_util.h_kw.md_docs.md`](./permutation_util.h_kw.md_docs.md)
- [`ir_util.cpp_kw.md_docs.md`](./ir_util.cpp_kw.md_docs.md)
- [`shape_inference.h_kw.md_docs.md`](./shape_inference.h_kw.md_docs.md)
- [`ir_builder.h_docs.md_docs.md`](./ir_builder.h_docs.md_docs.md)
- [`shape_inference.cpp_kw.md_docs.md`](./shape_inference.cpp_kw.md_docs.md)
- [`hash.h_kw.md_docs.md`](./hash.h_kw.md_docs.md)
- [`multi_wait.cpp_kw.md_docs.md`](./multi_wait.cpp_kw.md_docs.md)
- [`lazy_graph_executor.cpp_docs.md_docs.md`](./lazy_graph_executor.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `debug_util.cpp_docs.md_docs.md`
- **Keyword Index**: `debug_util.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
