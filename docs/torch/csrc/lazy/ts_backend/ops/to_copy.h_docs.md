# Documentation: `torch/csrc/lazy/ts_backend/ops/to_copy.h`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/ops/to_copy.h`
- **Size**: 4,079 bytes (3.98 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch::lazy {

// This IR was copied from code-generated output, but the entire _to_copy
// operator cannot be trivially code generated since it is only desirable to
// capture IR for certain permutations of _to_copy (e.g. dtype), and for the
// others it is difficult to even invoke the aten/eager fallback necessitating
// directly implementing the right to(device) behavior
class ToCopy : public torch::lazy::TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::_to_copy);
  }

  ToCopy(
      const torch::lazy::Value& self,
      const std::optional<at::ScalarType>& dtype,
      const std::optional<at::Layout>& layout,
      const std::optional<at::Device>& device,
      const std::optional<bool>& pin_memory,
      const bool& non_blocking,
      const std::optional<at::MemoryFormat>& memory_format,
      std::vector<torch::lazy::Shape>&& shapes)
      : torch::lazy::TsNode(
            ClassOpKind(),
            {self},
            std::move(shapes),
            /* num_outputs */ 1,
            torch::lazy::MHash(
                dtype,
                layout,
                device,
                pin_memory,
                non_blocking,
                memory_format)),

        dtype(dtype),
        layout(layout),
        device(device),
        pin_memory(pin_memory),
        non_blocking(non_blocking),
        memory_format(memory_format) {}

  bool CanBeReused(
      const torch::lazy::Value& self,
      const std::optional<at::ScalarType>& dtype,
      const std::optional<at::Layout>& layout,
      const std::optional<at::Device>& device,
      const std::optional<bool>& pin_memory,
      const bool& non_blocking,
      const std::optional<at::MemoryFormat>& memory_format) const {
    size_t i = 0;
    return (
        operand(i++) == self && this->dtype == dtype &&
        this->layout == layout && this->device == device &&
        this->pin_memory == pin_memory && this->non_blocking == non_blocking &&
        this->memory_format == memory_format);
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << torch::lazy::TsNode::ToString();
    if (dtype.has_value()) {
      ss << ", dtype=" << dtype.value();
    } else {
      ss << ", dtype=null";
    }
    if (layout.has_value()) {
      ss << ", layout=" << layout.value();
    } else {
      ss << ", layout=null";
    }
    if (device.has_value()) {
      ss << ", device=" << device.value();
    } else {
      ss << ", device=null";
    }
    if (pin_memory.has_value()) {
      ss << ", pin_memory=" << pin_memory.value();
    } else {
      ss << ", pin_memory=null";
    }
    ss << ", non_blocking=" << non_blocking;
    if (memory_format.has_value()) {
      ss << ", memory_format=" << memory_format.value();
    } else {
      ss << ", memory_format=null";
    }
    return ss.str();
  }

  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      torch::lazy::TSLoweringContext* loctx) const override {
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve(1);
    kwarguments.reserve(6);
    size_t i = 0;
    arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
    kwarguments.emplace_back("dtype", dtype);
    kwarguments.emplace_back("layout", layout);
    kwarguments.emplace_back("device", device);
    kwarguments.emplace_back("pin_memory", pin_memory);
    kwarguments.emplace_back("non_blocking", non_blocking);
    kwarguments.emplace_back("memory_format", memory_format);
    torch::lazy::TSOpVector _to_copy_out =
        torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    TORCH_CHECK_EQ(_to_copy_out.size(), 1);

    return _to_copy_out;
  }

  std::optional<at::ScalarType> dtype;
  std::optional<at::Layout> layout;
  std::optional<at::Device> device;
  std::optional<bool> pin_memory;
  bool non_blocking;
  std::optional<at::MemoryFormat> memory_format;
};

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ToCopy`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/ts_backend/ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/ts_backend/ts_node.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/lazy/ts_backend/ops`):

- [`generic.h_docs.md`](./generic.h_docs.md)
- [`device_data.h_docs.md`](./device_data.h_docs.md)
- [`device_data.cpp_docs.md`](./device_data.cpp_docs.md)
- [`generic.cpp_docs.md`](./generic.cpp_docs.md)


## Cross-References

- **File Documentation**: `to_copy.h_docs.md`
- **Keyword Index**: `to_copy.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
