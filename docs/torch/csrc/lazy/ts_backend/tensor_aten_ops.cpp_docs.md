# Documentation: `torch/csrc/lazy/ts_backend/tensor_aten_ops.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/tensor_aten_ops.cpp`
- **Size**: 2,346 bytes (2.29 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/lazy/ts_backend/tensor_aten_ops.h>

#include <ATen/InferSize.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/ops/arithmetic_ir_ops.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/generated/LazyIr.h>
#include <algorithm>
#include <functional>
#include <optional>

namespace torch::lazy {
namespace {

// to enable operator+-*/ for Value
using namespace torch::lazy;

torch::lazy::Value MaybeExpand(
    const torch::lazy::Value& input,
    const torch::lazy::Shape& target_shape) {
  if (input.shape().sizes() == target_shape.sizes()) {
    return input;
  }
  return torch::lazy::MakeExpand(
      input,
      target_shape.sizes().vec(),
      /*is_scalar_expand=*/false);
}

} // namespace

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////

void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value) {
  torch::lazy::Value constant =
      torch::lazy::LazyGraphExecutor::Get()->GetIrValueForExpandedScalar(
          value, input->shape(), input->GetDevice());
  input->SetInPlaceIrValue(std::move(constant));
}

void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src) {
  if (input->GetDevice() == src->GetDevice()) {
    torch::lazy::Value copy_value;
    if (input->dtype() == src->dtype()) {
      copy_value = src->GetIrValue();
    } else {
      copy_value = torch::lazy::MakeCast(
          src->GetIrValue(), input->dtype(), src->dtype());
    }
    input->SetIrValue(MaybeExpand(copy_value, input->shape()));
  } else {
    auto input_shape = input->shape();
    at::Tensor src_tensor = src->ToTensor(/*detached=*/true);
    if (src_tensor.sizes() != input_shape.Get().sizes()) {
      src_tensor = src_tensor.expand(input_shape.Get().sizes().vec());
    }
    input->UpdateFromTensor(src_tensor, /*sync=*/false);
  }
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/ts_backend/tensor_aten_ops.h`
- `ATen/InferSize.h`
- `torch/csrc/autograd/variable.h`
- `torch/csrc/lazy/core/helpers.h`
- `torch/csrc/lazy/core/ir_builder.h`
- `torch/csrc/lazy/core/ir_util.h`
- `torch/csrc/lazy/core/lazy_graph_executor.h`
- `torch/csrc/lazy/core/metrics.h`
- `torch/csrc/lazy/core/ops/arithmetic_ir_ops.h`
- `torch/csrc/lazy/core/ops/utils.h`
- `torch/csrc/lazy/core/tensor.h`
- `torch/csrc/lazy/core/util.h`
- `torch/csrc/lazy/generated/LazyIr.h`
- `algorithm`
- `functional`
- `optional`


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

Files in the same folder (`torch/csrc/lazy/ts_backend`):

- [`ts_node.h_docs.md`](./ts_node.h_docs.md)
- [`dynamic_ir.cpp_docs.md`](./dynamic_ir.cpp_docs.md)
- [`ts_backend_impl.h_docs.md`](./ts_backend_impl.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)
- [`ts_autograd_functions.cpp_docs.md`](./ts_autograd_functions.cpp_docs.md)
- [`ts_eager_fallback.h_docs.md`](./ts_eager_fallback.h_docs.md)
- [`dynamic_ir.h_docs.md`](./dynamic_ir.h_docs.md)
- [`tensor_aten_ops.h_docs.md`](./tensor_aten_ops.h_docs.md)
- [`ts_lowering_context.cpp_docs.md`](./ts_lowering_context.cpp_docs.md)


## Cross-References

- **File Documentation**: `tensor_aten_ops.cpp_docs.md`
- **Keyword Index**: `tensor_aten_ops.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
