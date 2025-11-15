# Documentation: to_copy.h

## File Metadata
- **Path**: `torch/csrc/lazy/ts_backend/ops/to_copy.h`
- **Size**: 4079 bytes
- **Lines**: 125
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
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

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): ToCopy


## Key Components

The file contains 349 words across 125 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4079 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
