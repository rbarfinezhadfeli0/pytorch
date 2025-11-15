# Documentation: `c10/core/Device.cpp`

## File Metadata

- **Path**: `c10/core/Device.cpp`
- **Size**: 4,921 bytes (4.81 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <string>
#include <vector>

namespace c10 {
namespace {
DeviceType parse_type(const std::string& device_string) {
  static const std::array<
      std::pair<const char*, DeviceType>,
      static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
      types = {{
          {"cpu", DeviceType::CPU},
          {"cuda", DeviceType::CUDA},
          {"ipu", DeviceType::IPU},
          {"xpu", DeviceType::XPU},
          {"mkldnn", DeviceType::MKLDNN},
          {"opengl", DeviceType::OPENGL},
          {"opencl", DeviceType::OPENCL},
          {"ideep", DeviceType::IDEEP},
          {"hip", DeviceType::HIP},
          {"ve", DeviceType::VE},
          {"fpga", DeviceType::FPGA},
          {"maia", DeviceType::MAIA},
          {"xla", DeviceType::XLA},
          {"lazy", DeviceType::Lazy},
          {"vulkan", DeviceType::Vulkan},
          {"mps", DeviceType::MPS},
          {"meta", DeviceType::Meta},
          {"hpu", DeviceType::HPU},
          {"mtia", DeviceType::MTIA},
          {"privateuseone", DeviceType::PrivateUse1},
      }};
  if (device_string == "mkldnn") {
    TORCH_WARN_ONCE(
        "'mkldnn' is no longer used as device type. So torch.device('mkldnn') will be "
        "deprecated and removed in the future. Please use other valid device types instead.");
  }
  if (device_string == get_privateuse1_backend()) {
    return DeviceType::PrivateUse1;
  }
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [&device_string](const std::pair<const char*, DeviceType>& p) {
        return p.first && p.first == device_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  std::vector<const char*> device_names;
  for (const auto& it : types) {
    if (it.first) {
      device_names.push_back(it.first);
    }
  }
  TORCH_CHECK(
      false,
      "Expected one of ",
      c10::Join(", ", device_names),
      " device type at start of device string: ",
      device_string);
}
enum DeviceStringParsingState { START, INDEX_START, INDEX_REST, ERROR };

} // namespace

Device::Device(const std::string& device_string) : Device(Type::CPU) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");

  std::string device_name, device_index_str;
  DeviceStringParsingState pstate = DeviceStringParsingState::START;

  // The code below tries to match the string in the variable
  // device_string against the regular expression:
  // ([a-zA-Z_]+)(?::([1-9]\\d*|0))?
  for (size_t i = 0;
       pstate != DeviceStringParsingState::ERROR && i < device_string.size();
       ++i) {
    const char ch = device_string.at(i);
    const unsigned char uch = static_cast<unsigned char>(ch);
    switch (pstate) {
      case DeviceStringParsingState::START:
        if (ch != ':') {
          if (std::isalpha(uch) || ch == '_') {
            device_name.push_back(ch);
          } else {
            pstate = DeviceStringParsingState::ERROR;
          }
        } else {
          pstate = DeviceStringParsingState::INDEX_START;
        }
        break;

      case DeviceStringParsingState::INDEX_START:
        if (std::isdigit(uch)) {
          device_index_str.push_back(ch);
          pstate = DeviceStringParsingState::INDEX_REST;
        } else {
          pstate = DeviceStringParsingState::ERROR;
        }
        break;

      case DeviceStringParsingState::INDEX_REST:
        if (device_index_str.at(0) == '0') {
          pstate = DeviceStringParsingState::ERROR;
          break;
        }
        if (std::isdigit(uch)) {
          device_index_str.push_back(ch);
        } else {
          pstate = DeviceStringParsingState::ERROR;
        }
        break;

      case DeviceStringParsingState::ERROR:
        // Execution won't reach here.
        break;
    }
  }

  const bool has_error = device_name.empty() ||
      pstate == DeviceStringParsingState::ERROR ||
      (pstate == DeviceStringParsingState::INDEX_START &&
       device_index_str.empty());

  TORCH_CHECK(!has_error, "Invalid device string: '", device_string, "'");

  try {
    if (!device_index_str.empty()) {
      index_ = static_cast<c10::DeviceIndex>(std::stoi(device_index_str));
    }
  } catch (const std::exception&) {
    TORCH_CHECK(
        false,
        "Could not parse device index '",
        device_index_str,
        "' in device string '",
        device_string,
        "'");
  }
  type_ = parse_type(device_name);
  validate();
}

std::string Device::str() const {
  std::string str = DeviceTypeName(type(), /* lower case */ true);
  if (has_index()) {
    str.push_back(':');
    str.append(std::to_string(index()));
  }
  return str;
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.str();
  return stream;
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`, `Device`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Device.h`
- `c10/util/Exception.h`
- `algorithm`
- `array`
- `cctype`
- `exception`
- `string`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`c10/core`):

- [`DispatchKey.cpp_docs.md`](./DispatchKey.cpp_docs.md)
- [`CopyBytes.h_docs.md`](./CopyBytes.h_docs.md)
- [`OptionalRef.h_docs.md`](./OptionalRef.h_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`SafePyObject.cpp_docs.md`](./SafePyObject.cpp_docs.md)
- [`DeviceType.cpp_docs.md`](./DeviceType.cpp_docs.md)
- [`SymBool.cpp_docs.md`](./SymBool.cpp_docs.md)
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`SymbolicShapeMeta.cpp_docs.md`](./SymbolicShapeMeta.cpp_docs.md)


## Cross-References

- **File Documentation**: `Device.cpp_docs.md`
- **Keyword Index**: `Device.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
