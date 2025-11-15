# Documentation: `docs/torch/csrc/distributed/c10d/GlooDeviceFactory.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/GlooDeviceFactory.cpp_docs.md`
- **Size**: 9,464 bytes (9.24 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/GlooDeviceFactory.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/GlooDeviceFactory.cpp`
- **Size**: 6,887 bytes (6.73 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/GlooDeviceFactory.hpp>

#include <torch/csrc/distributed/c10d/Utils.hpp>

#ifdef USE_C10D_GLOO

#include <cstdlib>

#include <c10/util/Exception.h>
#include <c10/util/env.h>

#if GLOO_HAVE_TRANSPORT_TCP
#include <gloo/transport/tcp/device.h>
#endif

#if GLOO_HAVE_TRANSPORT_TCP_TLS
#include <gloo/transport/tcp/tls/device.h>
#endif

#if GLOO_HAVE_TRANSPORT_UV
#include <gloo/transport/uv/device.h>
#endif

#if GLOO_HAVE_TRANSPORT_IBVERBS
#include <gloo/transport/ibverbs/device.h>
#endif

// On Linux, check that the tcp transport is available.
#ifdef __linux__
#if !GLOO_HAVE_TRANSPORT_TCP
#error "Expected the tcp transport to be available on Linux."
#endif
#endif

// On macOS, check that the uv transport is available.
#ifdef __APPLE__
#if !GLOO_HAVE_TRANSPORT_UV
#error "Expected the uv transport to be available on macOS."
#endif
#endif

namespace c10d {

C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING(
    GlooDeviceRegistry,
    ::gloo::transport::Device,
    const std::string& /* interface */,
    const std::string& /* hostname */,
    bool /* lazyInit */)

#if GLOO_HAVE_TRANSPORT_TCP
static std::shared_ptr<::gloo::transport::Device> makeTCPDevice(
    const std::string& interfaceName,
    const std::string& hostname,
    bool lazyInit) {
  TORCH_CHECK(
      !interfaceName.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeTCPDevice(): interface or hostname "
      "can't be empty");

  ::gloo::transport::tcp::attr attr;
  if (!interfaceName.empty()) {
    attr.iface = interfaceName;
  } else {
    attr.hostname = hostname;
  }
  if (lazyInit) {
    return ::gloo::transport::tcp::CreateLazyDevice(attr);
  } else {
    return ::gloo::transport::tcp::CreateDevice(attr);
  }
}

// Registry priority is per key identifier. We register TCP to `LINUX` for
// the flexibility of other application to override by priority. Register
// TCP to `TCP` for env "GLOO_DEVICE_TRANSPORT" override.
C10_REGISTER_CREATOR(GlooDeviceRegistry, LINUX, makeTCPDevice)
C10_REGISTER_CREATOR(GlooDeviceRegistry, TCP, makeTCPDevice)
#endif

#if GLOO_HAVE_TRANSPORT_TCP_TLS
static std::shared_ptr<::gloo::transport::Device> makeTCPTLSDevice(
    const std::string& interface,
    const std::string& hostname,
    bool lazyInit) {
  TORCH_CHECK(
      !interface.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeTCPTLSDevice(): interface or hostname "
      "can't be empty");

  TORCH_CHECK(!lazyInit, "TCP_TLS transport does not support lazy init");

  ::gloo::transport::tcp::attr attr;
  if (!interface.empty()) {
    attr.iface = interface;
  } else {
    attr.hostname = hostname;
  }
  const auto pkey_env =
      c10::utils::get_env("GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY");
  const auto pkey = pkey_env.has_value() ? pkey_env.value() : std::string();
  const auto cert_env =
      c10::utils::get_env("GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT");
  const auto cert = cert_env.has_value() ? cert_env.value() : std::string();
  const auto caFile_env =
      c10::utils::get_env("GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE");
  const auto caFile =
      caFile_env.has_value() ? caFile_env.value() : std::string();
  const auto caPath_env =
      c10::utils::get_env("GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_PATH");
  const auto caPath =
      caPath_env.has_value() ? caPath_env.value() : std::string();
  return ::gloo::transport::tcp::tls::CreateDevice(
      attr, pkey, cert, caFile, caPath);
}

C10_REGISTER_CREATOR(GlooDeviceRegistry, TCP_TLS, makeTCPTLSDevice)
#endif

#if GLOO_HAVE_TRANSPORT_UV
static std::shared_ptr<::gloo::transport::Device> makeUVDevice(
    const std::string& interfaceName,
    const std::string& hostname,
    bool lazyInit) {
  TORCH_CHECK(
      !interfaceName.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeUVDevice(): interface or hostname "
      "can't be empty");

  TORCH_CHECK(!lazyInit, "UV transport does not support lazy init");

  ::gloo::transport::uv::attr attr;
  if (!interfaceName.empty()) {
    attr.iface = interfaceName;
  } else {
    attr.hostname = hostname;
  }
  return ::gloo::transport::uv::CreateDevice(attr);
}

// Registry priority is per key identifier. We register UV to `APPLE` for
// the flexibility of other application to override by priority. Register
// UV to `UV` for env "GLOO_DEVICE_TRANSPORT" override.
C10_REGISTER_CREATOR(GlooDeviceRegistry, APPLE, makeUVDevice)
C10_REGISTER_CREATOR(GlooDeviceRegistry, WIN32, makeUVDevice)
C10_REGISTER_CREATOR(GlooDeviceRegistry, UV, makeUVDevice)
#endif

#if GLOO_HAVE_TRANSPORT_IBVERBS
static std::shared_ptr<::gloo::transport::Device> makeIBVerbsDevice(
    const std::string& interface,
    const std::string& hostname,
    bool lazyInit) {
  if (!hostname.empty()) {
    TORCH_WARN(
        "ibverbs transport does not support hostname, defaulting to any");
  }

  TORCH_CHECK(!lazyInit, "transport does not support lazy init");

  ::gloo::transport::ibverbs::attr attr;
  attr.name = getCvarString(
      {
          "TORCH_GLOO_IBV_NAME",
      },
      "");
  attr.port = getCvarInt(
      {
          "TORCH_GLOO_IBV_PORT",
      },
      1);
  attr.index = getCvarInt(
      {
          "TORCH_GLOO_IBV_INDEX",
      },
      0);

  if (!interface.empty()) {
    attr.name = interface;
  }

  // use global port
  attr.port = 1;

  return ::gloo::transport::ibverbs::CreateDevice(attr);
}

C10_REGISTER_CREATOR(GlooDeviceRegistry, IBVERBS, makeIBVerbsDevice)
#endif

namespace {
std::shared_ptr<::gloo::transport::Device> makeGlooDevice(
    const std::string& interfaceName,
    const std::string& hostName,
    bool lazyInit) {
  static auto transportName = c10::utils::get_env("GLOO_DEVICE_TRANSPORT");
  if (transportName.has_value()) {
    return GlooDeviceRegistry()->Create(
        transportName.value(), interfaceName, hostName, lazyInit);
  }

#ifdef __linux__
  return GlooDeviceRegistry()->Create(
      "LINUX", interfaceName, hostName, lazyInit);
#endif

#ifdef __APPLE__
  return GlooDeviceRegistry()->Create(
      "APPLE", interfaceName, hostName, lazyInit);
#endif

#ifdef _WIN32
  return GlooDeviceRegistry()->Create(
      "WIN32", interfaceName, hostName, lazyInit);
#endif

  return nullptr;
}
} // anonymous namespace

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForInterface(const std::string& interfaceName, bool lazyInit) {
  auto device = makeGlooDevice(interfaceName, "", lazyInit);
  if (!device) {
    TORCH_CHECK(false, "makeDeviceForInterface(): unsupported gloo device");
  }
  return device;
}

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForHostname(const std::string& hostname, bool lazyInit) {
  auto device = makeGlooDevice("", hostname, lazyInit);
  if (!device) {
    TORCH_CHECK(false, "makeDeviceForHostname(): unsupported gloo device");
  }
  return device;
}

} // namespace c10d

#endif // USE_C10D_GLOO

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `std`, `c10d`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/GlooDeviceFactory.hpp`
- `torch/csrc/distributed/c10d/Utils.hpp`
- `cstdlib`
- `c10/util/Exception.h`
- `c10/util/env.h`
- `gloo/transport/tcp/device.h`
- `gloo/transport/tcp/tls/device.h`
- `gloo/transport/uv/device.h`
- `gloo/transport/ibverbs/device.h`


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

- **File Documentation**: `GlooDeviceFactory.cpp_docs.md`
- **Keyword Index**: `GlooDeviceFactory.cpp_kw.md`
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

- **File Documentation**: `GlooDeviceFactory.cpp_docs.md_docs.md`
- **Keyword Index**: `GlooDeviceFactory.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
