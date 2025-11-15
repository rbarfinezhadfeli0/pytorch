# Documentation: `torch/csrc/profiler/unwind/fast_symbolizer.h`

## File Metadata

- **Path**: `torch/csrc/profiler/unwind/fast_symbolizer.h`
- **Size**: 3,303 bytes (3.23 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <fmt/format.h>
#include <sys/types.h>
#include <torch/csrc/profiler/unwind/debug_info.h>
#include <torch/csrc/profiler/unwind/line_number_program.h>
#include <torch/csrc/profiler/unwind/sections.h>
#include <torch/csrc/profiler/unwind/unwind.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <memory>
#include <unordered_map>

namespace torch::unwind {

#define UNWIND_WARN(w, ...)                   \
  do {                                        \
    w.emplace_back(fmt::format(__VA_ARGS__)); \
    LOG_INFO("WARNING: {}\n", w.back());      \
  } while (0);

struct FastSymbolizer {
  FastSymbolizer() = default;
  Frame symbolize(const std::string& library, uint64_t offset) {
    LOG_INFO("symbolizing {} + 0x{:x}\n", library, offset);
    Frame frame;
    frame.funcname = "??";
    frame.filename = library;
    frame.lineno = offset;
    auto s = getOrCreateSections(library);
    if (auto e = s->findSubprogramName(offset)) {
      frame.funcname = *e;
    } else {
      UNWIND_WARN(
          warnings_,
          "failed to find subprogram name for {} 0x{:x}",
          library,
          offset);
    }
    if (auto e = findLine(s, offset)) {
      frame.filename = e->first;
      frame.lineno = e->second;
    } else {
      UNWIND_WARN(
          warnings_, "failed to find file/line for {} 0x{:x}", library, offset);
    }
    return frame;
  }
  const std::vector<std::string>& warnings() {
    return warnings_;
  }

 private:
  void parseDebugInfo(Sections* s) {
    uint64_t offset = 0;
    while (offset < s->debug_info.size) {
      DebugInfo info(*s);
      info.parse(offset);
      if (auto lnp_offset = info.lineNumberProgramOffset()) {
        for (auto r : info.ranges()) {
          s->addDebugInfoRange(r.first, r.second, line_number_programs_.size());
        }
        line_number_programs_.emplace_back(
            std::make_unique<LineNumberProgram>(*s, *lnp_offset));
      }
      offset = info.nextOffset();
    }
  }
  Sections* getOrCreateSections(const std::string& library) {
    auto it = libraries_.find(library);
    if (it == libraries_.end()) {
      it = libraries_.insert({library, std::make_unique<Sections>()}).first;
      try {
        Sections* s = it->second.get();
        s->parse(library.c_str());
        parseDebugInfo(s);
      } catch (UnwindError& err) {
        UNWIND_WARN(
            warnings_, "failed to parse library {}: {}", library, err.what());
      }
    }
    return it->second.get();
  }
  std::optional<std::pair<std::string, int64_t>> findLine(
      Sections* s,
      uint64_t offset) {
    if (auto idx = s->findDebugInfoOffset(offset)) {
      auto r = line_number_programs_.at(*idx).get();
      try {
        r->parse();
      } catch (UnwindError& err) {
        UNWIND_WARN(
            warnings_,
            "failed to read line number program [{:x}] {}",
            r->offset(),
            err.what());
      }
      if (auto e = r->find(offset)) {
        return std::make_pair(r->filename(e->file), e->line);
      }
    }
    return std::nullopt;
  }
  std::unordered_map<std::string, std::unique_ptr<Sections>> libraries_;
  std::vector<std::unique_ptr<LineNumberProgram>> line_number_programs_;
  std::vector<std::string> warnings_;
};

} // namespace torch::unwind

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `FastSymbolizer`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler/unwind`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `fmt/format.h`
- `sys/types.h`
- `torch/csrc/profiler/unwind/debug_info.h`
- `torch/csrc/profiler/unwind/line_number_program.h`
- `torch/csrc/profiler/unwind/sections.h`
- `torch/csrc/profiler/unwind/unwind.h`
- `torch/csrc/profiler/unwind/unwind_error.h`
- `memory`
- `unordered_map`


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

Files in the same folder (`torch/csrc/profiler/unwind`):

- [`unwind_fb.cpp_docs.md`](./unwind_fb.cpp_docs.md)
- [`unwind.cpp_docs.md`](./unwind.cpp_docs.md)
- [`dwarf_symbolize_enums.h_docs.md`](./dwarf_symbolize_enums.h_docs.md)
- [`fde.h_docs.md`](./fde.h_docs.md)
- [`sections.h_docs.md`](./sections.h_docs.md)
- [`unwind.h_docs.md`](./unwind.h_docs.md)
- [`debug_info.h_docs.md`](./debug_info.h_docs.md)
- [`action.h_docs.md`](./action.h_docs.md)
- [`lexer.h_docs.md`](./lexer.h_docs.md)
- [`unwind_error.h_docs.md`](./unwind_error.h_docs.md)


## Cross-References

- **File Documentation**: `fast_symbolizer.h_docs.md`
- **Keyword Index**: `fast_symbolizer.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
