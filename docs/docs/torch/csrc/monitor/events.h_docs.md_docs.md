# Documentation: `docs/torch/csrc/monitor/events.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/monitor/events.h_docs.md`
- **Size**: 4,837 bytes (4.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/monitor/events.h`

## File Metadata

- **Path**: `torch/csrc/monitor/events.h`
- **Size**: 2,682 bytes (2.62 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

#include <c10/macros/Macros.h>
#include <variant>

namespace torch::monitor {

// data_value_t is the type for Event data values.
using data_value_t = std::variant<std::string, double, int64_t, bool>;

// Event represents a single event that can be logged out to an external
// tracker. This does acquire a lock on logging so should be used relatively
// infrequently to avoid performance issues.
struct TORCH_API Event {
  // name is the name of the event. This is a static string that's used to
  // differentiate between event types for programmatic access. The type should
  // be in the format of a fully qualified Python-style class name.
  // Ex: torch.monitor.MonitorEvent
  std::string name;

  // timestamp is a timestamp relative to the Unix epoch time.
  std::chrono::system_clock::time_point timestamp;

  // data contains rich information about the event. The contents are event
  // specific so you should check the type to ensure it's what you expect before
  // accessing the data.
  //
  // NOTE: these events are not versioned and it's up to the consumer of the
  // events to check the fields to ensure backwards compatibility.
  std::unordered_map<std::string, data_value_t> data;
};

inline bool operator==(const Event& lhs, const Event& rhs) {
  return lhs.name == rhs.name && lhs.timestamp == rhs.timestamp &&
      lhs.data == rhs.data;
}

// EventHandler represents an abstract event handler that can be registered to
// capture events. Every time an event is logged every handler will be called
// with the events contents.
//
// NOTE: The handlers should avoid any IO, blocking calls or heavy computation
// as this may block the main thread and cause performance issues.
class TORCH_API EventHandler {
 public:
  virtual ~EventHandler() = default;

  // handle needs to be implemented to handle the events. This may be called
  // from multiple threads so needs to be thread safe.
  virtual void handle(const Event& e) = 0;
};

// logEvent calls each registered event handler with the event. This method can
// be called from concurrently from multiple threads.
TORCH_API void logEvent(const Event& e);

// registerEventHandler registers an EventHandler so it receives any logged
// events. Typically an EventHandler will be registered during program
// setup and unregistered at the end.
TORCH_API void registerEventHandler(std::shared_ptr<EventHandler> p);

// unregisterEventHandler unregisters the event handler pointed to by the
// shared_ptr.
TORCH_API void unregisterEventHandler(const std::shared_ptr<EventHandler>& p);

} // namespace torch::monitor

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `name`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/monitor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `chrono`
- `memory`
- `string`
- `unordered_map`
- `c10/macros/Macros.h`
- `variant`


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

Files in the same folder (`torch/csrc/monitor`):

- [`counters.cpp_docs.md`](./counters.cpp_docs.md)
- [`python_init.cpp_docs.md`](./python_init.cpp_docs.md)
- [`python_init.h_docs.md`](./python_init.h_docs.md)
- [`counters.h_docs.md`](./counters.h_docs.md)
- [`events.cpp_docs.md`](./events.cpp_docs.md)


## Cross-References

- **File Documentation**: `events.h_docs.md`
- **Keyword Index**: `events.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/monitor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/monitor`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/monitor`):

- [`counters.h_docs.md_docs.md`](./counters.h_docs.md_docs.md)
- [`counters.cpp_docs.md_docs.md`](./counters.cpp_docs.md_docs.md)
- [`counters.cpp_kw.md_docs.md`](./counters.cpp_kw.md_docs.md)
- [`events.h_kw.md_docs.md`](./events.h_kw.md_docs.md)
- [`counters.h_kw.md_docs.md`](./counters.h_kw.md_docs.md)
- [`events.cpp_docs.md_docs.md`](./events.cpp_docs.md_docs.md)
- [`python_init.cpp_kw.md_docs.md`](./python_init.cpp_kw.md_docs.md)
- [`python_init.h_kw.md_docs.md`](./python_init.h_kw.md_docs.md)
- [`python_init.h_docs.md_docs.md`](./python_init.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `events.h_docs.md_docs.md`
- **Keyword Index**: `events.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
