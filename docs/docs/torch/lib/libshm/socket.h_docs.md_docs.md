# Documentation: `docs/torch/lib/libshm/socket.h_docs.md`

## File Metadata

- **Path**: `docs/torch/lib/libshm/socket.h_docs.md`
- **Size**: 6,657 bytes (6.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/lib/libshm/socket.h`

## File Metadata

- **Path**: `torch/lib/libshm/socket.h`
- **Size**: 4,322 bytes (4.22 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

#include <libshm/alloc_info.h>
#include <libshm/err.h>

class Socket {
 public:
  int socket_fd;
  Socket(const Socket& other) = delete;

 protected:
  Socket() {
    SYSCHECK_ERR_RETURN_NEG1(socket_fd = socket(AF_UNIX, SOCK_STREAM, 0));
  }
  Socket(Socket&& other) noexcept : socket_fd(other.socket_fd) {
    other.socket_fd = -1;
  };
  explicit Socket(int fd) : socket_fd(fd) {}

  virtual ~Socket() {
    if (socket_fd != -1)
      close(socket_fd);
  }

  struct sockaddr_un prepare_address(const char* path) {
    struct sockaddr_un address;
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, path);
    return address;
  }

  // Implemented based on https://man7.org/linux/man-pages/man7/unix.7.html
  size_t address_length(struct sockaddr_un address) {
    return offsetof(sockaddr_un, sun_path) + strlen(address.sun_path) + 1;
  }

  void recv(void* _buffer, size_t num_bytes) {
    char* buffer = (char*)_buffer;
    size_t bytes_received = 0;
    ssize_t step_received;
    struct pollfd pfd = {};
    pfd.fd = socket_fd;
    pfd.events = POLLIN;
    while (bytes_received < num_bytes) {
      SYSCHECK_ERR_RETURN_NEG1(poll(&pfd, 1, 1000));
      if (pfd.revents & POLLIN) {
        SYSCHECK_ERR_RETURN_NEG1(
            step_received =
                ::read(socket_fd, buffer, num_bytes - bytes_received));
        TORCH_CHECK(step_received != 0, "Other end has closed the connection");
        bytes_received += step_received;
        buffer += step_received;
      } else if (pfd.revents & (POLLERR | POLLHUP)) {
        TORCH_CHECK(false, "An error occurred while waiting for the data");
      } else {
        TORCH_CHECK(false, "Shared memory manager connection has timed out");
      }
    }
  }

  void send(const void* _buffer, size_t num_bytes) {
    const char* buffer = (const char*)_buffer;
    size_t bytes_sent = 0;
    ssize_t step_sent;
    while (bytes_sent < num_bytes) {
      SYSCHECK_ERR_RETURN_NEG1(
          step_sent = ::write(socket_fd, buffer, num_bytes));
      bytes_sent += step_sent;
      buffer += step_sent;
    }
  }
};

class ManagerSocket : public Socket {
 public:
  explicit ManagerSocket(int fd) : Socket(fd) {}

  AllocInfo receive() {
    AllocInfo info;
    recv(&info, sizeof(info));
    return info;
  }

  void confirm() {
    send("OK", 2);
  }
};

class ManagerServerSocket : public Socket {
 public:
  explicit ManagerServerSocket(const std::string& path) {
    socket_path = path;
    try {
      struct sockaddr_un address = prepare_address(path.c_str());
      size_t len = address_length(address);
      SYSCHECK_ERR_RETURN_NEG1(
          bind(socket_fd, (struct sockaddr*)&address, len));
      SYSCHECK_ERR_RETURN_NEG1(listen(socket_fd, 10));
    } catch (std::exception&) {
      SYSCHECK_ERR_RETURN_NEG1(close(socket_fd));
      throw;
    }
  }

  void remove() {
    struct stat file_stat;
    if (fstat(socket_fd, &file_stat) == 0)
      SYSCHECK_ERR_RETURN_NEG1(unlink(socket_path.c_str()));
  }

  ~ManagerServerSocket() override {
    unlink(socket_path.c_str());
  }

  ManagerSocket accept() {
    int client_fd;
    struct sockaddr_un addr;
    socklen_t addr_len = sizeof(addr);
    SYSCHECK_ERR_RETURN_NEG1(
        client_fd = ::accept(socket_fd, (struct sockaddr*)&addr, &addr_len));
    return ManagerSocket(client_fd);
  }

  std::string socket_path;
};

class ClientSocket : public Socket {
 public:
  explicit ClientSocket(const std::string& path) {
    try {
      struct sockaddr_un address = prepare_address(path.c_str());
      size_t len = address_length(address);
      SYSCHECK_ERR_RETURN_NEG1(
          connect(socket_fd, (struct sockaddr*)&address, len));
    } catch (std::exception&) {
      SYSCHECK_ERR_RETURN_NEG1(close(socket_fd));
      throw;
    }
  }

  void register_allocation(AllocInfo& info) {
    char buffer[3] = {0, 0, 0};
    send(&info, sizeof(info));
    recv(buffer, 2);
    TORCH_CHECK(
        strcmp(buffer, "OK") == 0,
        "Shared memory manager didn't respond with an OK");
  }

  void register_deallocation(AllocInfo& info) {
    send(&info, sizeof(info));
  }
};

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `Socket`, `sockaddr_un`, `sockaddr_un`, `sockaddr_un`, `pollfd`, `ManagerSocket`, `ManagerServerSocket`, `sockaddr_un`, `sockaddr`, `stat`, `sockaddr_un`, `sockaddr`, `ClientSocket`, `sockaddr_un`, `sockaddr`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/lib/libshm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `poll.h`
- `sys/socket.h`
- `sys/stat.h`
- `sys/types.h`
- `sys/un.h`
- `unistd.h`
- `cstddef`
- `cstdio`
- `cstring`
- `string`
- `libshm/alloc_info.h`
- `libshm/err.h`


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
- [`core.cpp_docs.md`](./core.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`libshm.h_docs.md`](./libshm.h_docs.md)
- [`err.h_docs.md`](./err.h_docs.md)
- [`alloc_info.h_docs.md`](./alloc_info.h_docs.md)


## Cross-References

- **File Documentation**: `socket.h_docs.md`
- **Keyword Index**: `socket.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/lib/libshm`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/lib/libshm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/lib/libshm`):

- [`libshm.h_kw.md_docs.md`](./libshm.h_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`alloc_info.h_docs.md_docs.md`](./alloc_info.h_docs.md_docs.md)
- [`core.cpp_docs.md_docs.md`](./core.cpp_docs.md_docs.md)
- [`core.cpp_kw.md_docs.md`](./core.cpp_kw.md_docs.md)
- [`manager.cpp_kw.md_docs.md`](./manager.cpp_kw.md_docs.md)
- [`err.h_kw.md_docs.md`](./err.h_kw.md_docs.md)
- [`CMakeLists.txt_kw.md_docs.md`](./CMakeLists.txt_kw.md_docs.md)
- [`manager.cpp_docs.md_docs.md`](./manager.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `socket.h_docs.md_docs.md`
- **Keyword Index**: `socket.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
