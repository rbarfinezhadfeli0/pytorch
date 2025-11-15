# Documentation: `torch/csrc/lazy/backend/lowering_context.h`

## File Metadata

- **Path**: `torch/csrc/lazy/backend/lowering_context.h`
- **Size**: 3,238 bytes (3.16 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_util.h>

namespace torch::lazy {

class TORCH_API Computation {
 public:
  virtual int parameters_size() const = 0;

  virtual const std::vector<Shape>& parameter_shapes() const = 0;

  virtual const std::vector<std::string>& parameter_names() const = 0;

  virtual const Shape& result_shape() const = 0;

  virtual const std::string to_string() const = 0;

  virtual ~Computation() = default;

  // Indicates whether this computation is being executed inside a mark step
  // Assume false unless set otherwise
  bool in_mark_step = false;
};

using ComputationPtr = std::shared_ptr<Computation>;

// Keeps track of the code generation state.
class TORCH_API LoweringContext {
 public:
  LoweringContext(const std::string& name, BackendDevice device);
  LoweringContext(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status);

  virtual ~LoweringContext() = default;

  static std::unique_ptr<LoweringContext> Create(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      Util::EmissionMap emit_status);

  static std::unique_ptr<LoweringContext> Create(
      const std::string& name,
      BackendDevice device);

  const BackendDevice& device() const {
    return device_;
  }

  // Retrieves the vector holding all the tensors associated with the parameter
  // instructions which have been created.
  const std::vector<BackendDataPtr>& GetParametersData() const;

  // Adds a new input/output alias.
  virtual void SetUpAlias(
      const std::vector<int64_t>& output_index,
      int64_t param_number,
      const std::vector<int64_t>& param_index,
      bool must_alias = false) {
    // Dummy default implementation to do nothing.
  }

  // Check if parameter shape matches result at index.
  virtual bool CheckResultShape(
      const BackendDataPtr& parameter_data,
      size_t result_idx) {
    // Dummy default implementation to do nothing.
    return false;
  }

  // Adds the given output as a component of the result tuple and returns its
  // assigned position within the tuple.
  virtual size_t AddResult(const torch::lazy::Output& output) = 0;

  // Associates the given output with the input parameter of the given index and
  // shape. Only used for the operator-by-operator execution, mostly for
  // debugging purposes.
  virtual void AddParameter(
      const torch::lazy::Output& output,
      size_t index,
      const Shape& shape,
      const std::string& name) = 0;

  // Build the computation capturing all the operations created with the
  // embedded builder (returned by the builder() API).
  virtual ComputationPtr Build() = 0;

  size_t GetEmittedNodeCount() const {
    return emit_status_.size();
  }

 protected:
  BackendDevice device_;
  std::vector<BackendDataPtr> parameters_;
  std::vector<size_t> parameter_sequence_;
  Util::EmissionMap emit_status_;
};

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `memory`
- `string`
- `vector`
- `torch/csrc/lazy/backend/backend_data.h`
- `torch/csrc/lazy/backend/backend_device.h`
- `torch/csrc/lazy/core/ir.h`
- `torch/csrc/lazy/core/ir_util.h`


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

Files in the same folder (`torch/csrc/lazy/backend`):

- [`backend_device.h_docs.md`](./backend_device.h_docs.md)
- [`backend_device.cpp_docs.md`](./backend_device.cpp_docs.md)
- [`backend_data.h_docs.md`](./backend_data.h_docs.md)
- [`lowering_context.cpp_docs.md`](./lowering_context.cpp_docs.md)
- [`backend_interface.cpp_docs.md`](./backend_interface.cpp_docs.md)
- [`backend_interface.h_docs.md`](./backend_interface.h_docs.md)


## Cross-References

- **File Documentation**: `lowering_context.h_docs.md`
- **Keyword Index**: `lowering_context.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
