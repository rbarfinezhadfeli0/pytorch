# Documentation: `docs/torch/csrc/distributed/c10d/socket.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/socket.h_docs.md`
- **Size**: 4,936 bytes (4.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/socket.h`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/socket.h`
- **Size**: 2,460 bytes (2.40 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/Backoff.hpp>
#include <torch/csrc/distributed/c10d/exception.h>

namespace c10d::detail {

class SocketOptions {
 public:
  SocketOptions& prefer_ipv6(bool value) noexcept {
    prefer_ipv6_ = value;

    return *this;
  }

  bool prefer_ipv6() const noexcept {
    return prefer_ipv6_;
  }

  SocketOptions& connect_timeout(std::chrono::milliseconds value) noexcept {
    connect_timeout_ = value;

    return *this;
  }

  std::chrono::milliseconds connect_timeout() const noexcept {
    return connect_timeout_;
  }

  // Sets the backoff policy to use for socket connect ops.
  SocketOptions& connect_backoff(std::shared_ptr<Backoff> value) noexcept {
    connect_backoff_ = std::move(value);

    return *this;
  }

  const std::shared_ptr<Backoff>& connect_backoff() const noexcept {
    return connect_backoff_;
  }

 private:
  bool prefer_ipv6_ = true;
  std::chrono::milliseconds connect_timeout_{std::chrono::seconds{30}};
  std::shared_ptr<Backoff> connect_backoff_{
      std::make_shared<FixedBackoff>(std::chrono::milliseconds(1000))};
};

class SocketImpl;

class Socket {
 public:
  // This function initializes the underlying socket library and must be called
  // before any other socket function.
  static void initialize();

  static Socket listen(std::uint16_t port, const SocketOptions& opts = {});

  static Socket listenFromFd(int fd, std::uint16_t expected_port);

  static Socket connect(
      const std::string& host,
      std::uint16_t port,
      const SocketOptions& opts = {});

  Socket() noexcept = default;

  Socket(const Socket& other) = delete;

  Socket& operator=(const Socket& other) = delete;

  Socket(Socket&& other) noexcept;

  Socket& operator=(Socket&& other) noexcept;

  ~Socket();

  Socket accept() const;

  int handle() const noexcept;

  std::uint16_t port() const;

  bool waitForInput(std::chrono::milliseconds timeout);

  std::string repr() const;

 private:
  explicit Socket(std::unique_ptr<SocketImpl>&& impl) noexcept;

  std::unique_ptr<SocketImpl> impl_;
};
} // namespace c10d::detail

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `SocketOptions`, `SocketImpl`, `Socket`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `chrono`
- `cstdint`
- `memory`
- `string`
- `c10/macros/Macros.h`
- `c10/util/Exception.h`
- `torch/csrc/distributed/c10d/Backoff.hpp`
- `torch/csrc/distributed/c10d/exception.h`


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

- **File Documentation**: `socket.h_docs.md`
- **Keyword Index**: `socket.h_kw.md`
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

- **File Documentation**: `socket.h_docs.md_docs.md`
- **Keyword Index**: `socket.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
