# Documentation: `docs/torch/csrc/distributed/c10d/GroupRegistry.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/GroupRegistry.cpp_docs.md`
- **Size**: 5,377 bytes (5.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/GroupRegistry.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/GroupRegistry.cpp`
- **Size**: 2,890 bytes (2.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

#include <torch/csrc/distributed/c10d/RankLocal.hpp>

namespace {

// Each rank operates on a different `c10d::ProcessGroup` instance for the same
// logical process group. Use `RankLocal<GroupRegistry>::get()` to ensure each
// rank gets a unique registry.
class GroupRegistry {
 public:
  void register_group(
      const std::string& group_name,
      c10::intrusive_ptr<c10d::ProcessGroup> group) {
    std::unique_lock write_lock(lock_);
    auto [_, inserted] = registry_.try_emplace(group_name, std::move(group));
    TORCH_CHECK(
        inserted,
        "A process group is already registered under the name",
        group_name);
  }

  c10::intrusive_ptr<c10d::ProcessGroup> resolve_group(
      const std::string& group_name) {
    std::shared_lock read_lock(lock_);
    auto it = registry_.find(group_name);
    TORCH_CHECK(
        it != registry_.end(),
        "Could not resolve the process group registered under the name ",
        group_name);

    auto group = it->second.lock();
    TORCH_CHECK(
        group != nullptr,
        "Process group registered under the name ",
        group_name,
        " has already been destroyed.");
    return group;
  }

  void unregister_group(const std::string& group_name) {
    std::unique_lock write_lock(lock_);
    registry_.erase(group_name);
  }

  void unregister_all_groups() {
    std::unique_lock write_lock(lock_);
    registry_.clear();
  }

 private:
  std::map<std::string, c10::weak_intrusive_ptr<c10d::ProcessGroup>> registry_;
  std::shared_mutex lock_;
};

} // namespace

namespace c10d {

static bool thread_isolation_mode = false;
static GroupRegistry process_registry;

void set_thread_isolation_mode(bool enable) {
  thread_isolation_mode = enable;
}

bool get_thread_isolation_mode() {
  return thread_isolation_mode;
}

void register_process_group(
    const std::string& group_name,
    const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().register_group(group_name, group);
  } else {
    process_registry.register_group(group_name, group);
  }
}

c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& group_name) {
  if (thread_isolation_mode) {
    return RankLocal<::GroupRegistry>::get().resolve_group(group_name);
  } else {
    return process_registry.resolve_group(group_name);
  }
}

void unregister_process_group(const std::string& group_name) {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().unregister_group(group_name);
  } else {
    process_registry.unregister_group(group_name);
  }
}

void unregister_all_process_groups() {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().unregister_all_groups();
  } else {
    process_registry.unregister_all_groups();
  }
}

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `c10d`

**Classes/Structs**: `GroupRegistry`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/GroupRegistry.hpp`
- `torch/csrc/distributed/c10d/RankLocal.hpp`


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

- **File Documentation**: `GroupRegistry.cpp_docs.md`
- **Keyword Index**: `GroupRegistry.cpp_kw.md`
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

- **File Documentation**: `GroupRegistry.cpp_docs.md_docs.md`
- **Keyword Index**: `GroupRegistry.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
