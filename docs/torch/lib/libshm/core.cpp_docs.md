# Documentation: `torch/lib/libshm/core.cpp`

## File Metadata

- **Path**: `torch/lib/libshm/core.cpp`
- **Size**: 4,416 bytes (4.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <array>
#include <cstring>
#include <string>
#include <unordered_map>
#include <c10/util/error.h>

#include <libshm/err.h>
#include <libshm/libshm.h>
#include <libshm/socket.h>

static std::unordered_map<std::string, ClientSocket> managers;
static std::string manager_executable_path;

static AllocInfo get_alloc_info(const char* filename) {
  AllocInfo info = {};
  info.pid = getpid();
  info.free = false;
  size_t len = strlen(filename);
  TORCH_CHECK(len < sizeof(info.filename), "MapAllocatorContext_filename too long");
  memcpy(info.filename, filename, len + 1);
  return info;
}

static void start_manager() {
  std::array<int, 2> pipe_ends;
  SYSCHECK_ERR_RETURN_NEG1(pipe(pipe_ends.data()));

  pid_t pid = -1;
  SYSCHECK_ERR_RETURN_NEG1(pid = fork());
  if (!pid) {
    SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[0]));
    SYSCHECK_ERR_RETURN_NEG1(dup2(pipe_ends[1], 1)); // Replace stdout
    SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[1]));
    execl(manager_executable_path.c_str(), "torch_shm_manager", NULL);

    std::string msg("ERROR: execl failed: ");
    msg += c10::utils::str_error(errno);
    msg += '\n';
    auto res = write(1, msg.c_str(), msg.size());
    (void)res;

    exit(1);
  }
  SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[1]));

  constexpr auto MAX_BUFFER_SIZE = 1000;
  std::array<char, MAX_BUFFER_SIZE> buffer;
  std::string handle;
  while (handle.empty() || handle.back() != '\n') {
    const auto bytes_read = read(pipe_ends[0], buffer.data(), buffer.size());
    SYSCHECK_ERR_RETURN_NEG1(bytes_read);
    if (bytes_read == 0) {
      break;
    }
    handle.append(buffer.data(), bytes_read);
  }
  SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[0]));

  TORCH_CHECK(!handle.empty(), "no response from torch_shm_manager at \"", manager_executable_path, "\"");

  handle.pop_back(); // remove \n
  TORCH_CHECK(
      handle.rfind("ERROR: ", 0) != 0,
      "torch_shm_manager at \"",
      manager_executable_path,
      "\": ",
      handle.substr(7));

  ClientSocket manager{handle};
  managers.emplace(std::move(handle), std::move(manager));
}

static ClientSocket& get_manager_socket(const std::string& manager_handle) {
  auto it = managers.find(manager_handle);
  if (it == managers.end()) {
    auto socket = ClientSocket(manager_handle);
    auto result = managers.emplace(manager_handle, std::move(socket));
    return result.first->second;
  } else {
    return it->second;
  }
}

void libshm_init(const char* manager_exec_path) {
  manager_executable_path = std::string(manager_exec_path);
}

THManagedMapAllocatorInit::THManagedMapAllocatorInit(
    const char* manager_handle,
    const char* filename)
    : manager_handle_(manager_handle ? manager_handle : "") {
  // TODO: unlock GIL when contacting the manager
  try {
    ClientSocket* socket = nullptr;
    if (!manager_handle_.empty()) {
      socket = &get_manager_socket(manager_handle_);
    } else {
      if (managers.empty()) {
        start_manager();
      }
      const auto& manager = managers.begin();
      manager_handle_ = manager->first;
      socket = &manager->second;
    }
    AllocInfo info = get_alloc_info(filename);
    socket->register_allocation(info);
  } catch (std::exception& e) {
    TORCH_CHECK(false, e.what());
  }
}

THManagedMapAllocator::THManagedMapAllocator(
    const char* manager_handle,
    const char* filename,
    int flags,
    size_t size)
    : THManagedMapAllocatorInit(manager_handle, filename),
      at::RefcountedMapAllocator(filename, flags, size) {}

void THManagedMapAllocator::close() {
  if (closed_)
    return;
  AllocInfo info = get_alloc_info(filename());
  info.free = true;
  ClientSocket& socket = get_manager_socket(manager_handle_);
  at::RefcountedMapAllocator::close();
  socket.register_deallocation(info);
}

static void deleteTHManagedMapAllocator(void* ptr) {
  delete static_cast<THManagedMapAllocator*>(ptr);
}

at::DataPtr THManagedMapAllocator::makeDataPtr(
    const char* manager_handle,
    const char* filename,
    int flags,
    size_t size) {
  auto* context =
      new THManagedMapAllocator(manager_handle, filename, flags, size);
  return {
      context->data(),
      context,
      &deleteTHManagedMapAllocator,
      at::DeviceType::CPU};
}

THManagedMapAllocator* THManagedMapAllocator::fromDataPtr(
    const at::DataPtr& dptr) {
  return dptr.cast_context<THManagedMapAllocator>(&deleteTHManagedMapAllocator);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/lib/libshm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `array`
- `cstring`
- `string`
- `unordered_map`
- `c10/util/error.h`
- `libshm/err.h`
- `libshm/libshm.h`
- `libshm/socket.h`


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

Files in the same folder (`torch/lib/libshm`):

- [`manager.cpp_docs.md`](./manager.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`libshm.h_docs.md`](./libshm.h_docs.md)
- [`socket.h_docs.md`](./socket.h_docs.md)
- [`err.h_docs.md`](./err.h_docs.md)
- [`alloc_info.h_docs.md`](./alloc_info.h_docs.md)


## Cross-References

- **File Documentation**: `core.cpp_docs.md`
- **Keyword Index**: `core.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
