# Documentation: `docs/torch/csrc/jit/runtime/register_special_ops.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/register_special_ops.cpp_docs.md`
- **Size**: 20,416 bytes (19.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/register_special_ops.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/register_special_ops.cpp`
- **Size**: 17,445 bytes (17.04 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Context.h>
#include <torch/library.h>

#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/jit_type.h>
#include <c10/core/DefaultDtype.h>
#include <c10/util/irange.h>
#include <torch/csrc/api/include/torch/utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#include <ATen/InitialTensorOptions.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/frontend/error_report.h>

#include <sstream>

namespace torch::jit {

namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

c10::AliasAnalysisKind aliasAnalysisConservative() {
  return c10::AliasAnalysisKind::CONSERVATIVE;
}

void checkListInputType(const c10::TypePtr& elem_type, bool empty_list) {
  if (!elem_type->isSubtypeOf(*NumberType::get()) &&
      !elem_type->isSubtypeOf(*BoolType::get())) {
    std::stringstream error;
    error << "Input must be of ints, floats, or bools, "
          << "got " << elem_type->repr_str();
    // special case empty list torch.tensor([])
    if (elem_type->isSubtypeOf(*TensorType::get())) {
      if (empty_list) {
        error << "\nEmpty lists default to List[Tensor]. Add a variable "
                 "annotation to the assignment to create an empty list "
                 "of another type (torch.jit.annotate(List[T, []]) where T "
                 "is the type of elements in the list for Python 2)";
      }
    }
    throw std::runtime_error(error.str());
  }
}

at::Tensor castTensorTo(
    at::Tensor self,
    const IValue& dtype,
    const IValue& device) {
  at::ScalarType scalar_type =
      dtype.isNone() ? self.scalar_type() : dtype.toScalarType();
  c10::Device dev = device.isNone() ? self.device() : device.toDevice();
  if (scalar_type != self.scalar_type() || dev != self.device()) {
    self = self.to(dev, scalar_type);
  }
  return self;
}

std::vector<int64_t> compute_sizes(const IValue& seq) {
  std::vector<int64_t> sizes;
  auto seq_recur = seq.toList();
  while (true) {
    sizes.push_back(seq_recur.size());
    if (seq_recur.empty() || !seq_recur.get(0).isList()) {
      break;
    }
    seq_recur = seq_recur.get(0).toList();
  }
  return sizes;
}

void checkSequenceSize(int64_t n, int64_t dim, int64_t seq_size) {
  if (seq_size != n) {
    TORCH_CHECK(
        false,
        "Expected sequence of length ",
        n,
        " at dim ",
        dim,
        " (got ",
        seq_size,
        ")");
  }
}

template <typename DTYPE>
void storeLastDimension(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<IValue> obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (const auto i : c10::irange(n)) {
    *(DTYPE*)data = obj[i].to<DTYPE>();
    data += strides[dim] * elementSize;
  }
}

void storeLastDimensionFloat(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<IValue> obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (const auto i : c10::irange(n)) {
    *(float*)data = static_cast<float>(obj[i].to<double>());
    data += strides[dim] * elementSize;
  }
}

void storeLastDimensionHalf(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<IValue> obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (const auto i : c10::irange(n)) {
    *(at::Half*)data = at::convert<at::Half, double>(obj[i].to<double>());
    data += strides[dim] * elementSize;
  }
}

// reference python implementation recursive_store in tensor_new.cpp
void recursiveStore(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int tenElementSize,
    const IValue& obj) {
  auto ndim = sizes.size();
  auto n = sizes[dim];
  auto seq = obj.toListRef();
  checkSequenceSize(n, dim, seq.size());
  if (dim + 1 < static_cast<long>(ndim)) {
    for (const auto i : c10::irange(n)) {
      recursiveStore(data, sizes, strides, dim + 1, tenElementSize, seq[i]);
      data += strides[dim] * tenElementSize;
    }
  } else {
    if (obj.isIntList()) {
      storeLastDimension<int64_t>(
          data, sizes, strides, dim, tenElementSize, seq);
    } else if (obj.isBoolList()) {
      storeLastDimension<bool>(data, sizes, strides, dim, tenElementSize, seq);
    } else if (obj.isDoubleList()) {
      if (tenElementSize ==
          static_cast<int>(elementSize(at::ScalarType::Double))) {
        storeLastDimension<double>(
            data, sizes, strides, dim, tenElementSize, seq);
      } else if (
          tenElementSize ==
          static_cast<int>(elementSize(at::ScalarType::Float))) {
        storeLastDimensionFloat(data, sizes, strides, dim, tenElementSize, seq);
      } else if (
          tenElementSize ==
          static_cast<int>(elementSize(at::ScalarType::Half))) {
        storeLastDimensionHalf(data, sizes, strides, dim, tenElementSize, seq);
      } else {
        TORCH_INTERNAL_ASSERT(false);
      }
    } else {
      TORCH_INTERNAL_ASSERT(false);
    }
  }
}

template <bool if_set_requires_grad>
void createTensorFromList(Stack& stack) {
  // torch.tensor has a fourth requires_grad arg but torch.as_tensor not, so
  // we use the template arg to distinguish between these two cases
  bool requires_grad = false;
  IValue data;
  IValue dtype;
  IValue device;
  if (if_set_requires_grad) {
    pop(stack, data, dtype, device, requires_grad);
  } else {
    pop(stack, data, dtype, device);
  }
  auto elem_type = data.type();
  while (elem_type->isSubtypeOf(AnyListType::get())) {
    elem_type = elem_type->containedType(0);
  }
  auto sizes = compute_sizes(data);
  checkListInputType(elem_type, sizes.size() == 1 && sizes[0] == 0);
  at::ScalarType initial_scalar_type = scalarTypeFromJitType(*elem_type);
  if (initial_scalar_type == at::ScalarType::Double) {
    initial_scalar_type = typeMetaToScalarType(c10::get_default_dtype());
  }

  auto tensor =
      at::empty(sizes, at::initialTensorOptions().dtype(initial_scalar_type));

  if (tensor.numel() != 0) {
    recursiveStore(
        (char*)tensor.data_ptr(),
        sizes,
        tensor.strides(),
        0,
        tensor.element_size(),
        data);
  }

  tensor = castTensorTo(tensor, dtype, device);
  auto default_type = at::typeMetaToScalarType(at::get_default_dtype());

  if (dtype.isNone() && tensor.scalar_type() != default_type &&
      tensor.numel() == 0) {
    TORCH_WARN(
        "Creating a tensor from an empty ",
        elem_type->repr_str(),
        "list will create a tensor of default floating point type  (currently ",
        default_type,
        ") in python but a tensor of type ",
        elem_type->repr_str(),
        " in torchscript.\n",
        "Pass in a dtype argument to ensure consistent behavior");
  }
  if (if_set_requires_grad) {
    tensor.set_requires_grad(requires_grad);
  }
  push(stack, std::move(tensor));
}

RegisterOperators reg({
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]"),
        [](Stack& stack) {
          RECORD_FUNCTION("split_with_sizes", last(stack, 3));

          auto result = at::split_with_sizes(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toDimVector(),
              (std::move(peek(stack, 2, 3))).toInt());
          drop(stack, 3);
          pack(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),

#define DEFINE_TORCH_TENSOR_OP(operator_type, c_type, tensor_creation_op)       \
  OperatorGenerator(                                                            \
      TORCH_SELECTIVE_SCHEMA(                                                   \
          "aten::tensor." #operator_type "(" #operator_type                     \
          " t, *, ScalarType? dtype=None, Device? device=None"                  \
          ", bool requires_grad=False) -> Tensor"),                             \
      [](Stack& stack) {                                                        \
        c_type scalar_val;                                                      \
        IValue dtype;                                                           \
        IValue device;                                                          \
        bool requires_grad;                                                     \
        pop(stack, scalar_val, dtype, device, requires_grad);                   \
        auto tensor = tensor_creation_op;                                       \
        tensor = castTensorTo(tensor, dtype, device);                           \
        tensor.set_requires_grad(requires_grad);                                \
        push(stack, std::move(tensor));                                         \
      },                                                                        \
      aliasAnalysisFromSchema()),                                               \
      OperatorGenerator(                                                        \
          TORCH_SELECTIVE_SCHEMA(                                               \
              "aten::as_tensor." #operator_type "(" #operator_type              \
              " t, *, ScalarType? dtype=None, Device? device=None) -> Tensor"), \
          [](Stack& stack) {                                                    \
            c_type scalar_val;                                                  \
            IValue dtype;                                                       \
            IValue device;                                                      \
            pop(stack, scalar_val, dtype, device);                              \
            auto tensor = tensor_creation_op;                                   \
            tensor = castTensorTo(tensor, dtype, device);                       \
            push(stack, std::move(tensor));                                     \
          },                                                                    \
          aliasAnalysisFromSchema()),

    DEFINE_TORCH_TENSOR_OP(
        bool,
        bool,
        at::empty({}, at::device(at::kCPU).dtype(at::kBool)).fill_(scalar_val))
        DEFINE_TORCH_TENSOR_OP(
            float,
            double,
            at::native::scalar_tensor(
                scalar_val,
                typeMetaToScalarType(c10::get_default_dtype()),
                std::nullopt /* layout */,
                at::kCPU,
                std::nullopt /* pin_memory*/))
            DEFINE_TORCH_TENSOR_OP(
                int,
                int64_t,
                at::scalar_to_tensor(scalar_val))
                DEFINE_TORCH_TENSOR_OP(
                    complex,
                    c10::complex<double>,
                    at::native::scalar_tensor(
                        scalar_val,
                        typeMetaToScalarType(c10::get_default_complex_dtype()),
                        std::nullopt /* layout */,
                        at::kCPU,
                        std::nullopt /* pin_memory */))

    // reference python implementation: internal_new_from_data in
    // tensor_new.cpp
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::_infer_size(int[] a, int[] b) -> int[]"),
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, at::infer_size(a.toDimVector(), b.toDimVector()));
        },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor"),
        [](Stack& stack) {
          at::Tensor weight;
          at::Tensor input;
          double max_norm = 0;
          double norm_type = 0;
          pop(stack, weight, input, max_norm, norm_type);

          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor result =
              at::embedding_renorm_(weight, input, max_norm, norm_type);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor"),
        createTensorFromList<true>,
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::as_tensor(Tensor(a) data, *, ScalarType? dtype=None, Device? device=None) -> Tensor(a|b)"),
        [](Stack& stack) {
          auto device = pop(stack).toOptional<c10::Device>();
          auto dtype = pop(stack).toOptional<at::ScalarType>();
          at::Tensor data = pop(stack).toTensor();
          at::ScalarType scalar_type =
              dtype ? dtype.value() : data.scalar_type();
          c10::Device dev = device ? device.value() : data.device();

          if (scalar_type != data.scalar_type() || dev != data.device()) {
            data = data.to(
                dev, scalar_type, /*non_blocking=*/false, /*copy=*/false);
          }
          push(stack, std::move(data));
        },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::as_tensor.list(t[] data, *, ScalarType? dtype=None, Device? device=None) -> Tensor"),
        createTensorFromList<false>,
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_pack_sequence(Tensor output, Tensor batch_sizes, Tensor? sorted_indices, "
            "Tensor? unsorted_indices) -> (Tensor, Tensor, Tensor?, Tensor?)"),
        [](Stack& stack) {},
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::_get_tracing_state() -> bool"),
        [](Stack& stack) { push(stack, false); },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::is_scripting() -> bool"),
        [](Stack& stack) { push(stack, true); },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::has_torch_function(...) -> bool"),
        [](Stack& stack) { push(stack, false); },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_uniform_(Tensor(a!) tensor, float a, float b, Generator? generator=None) -> Tensor(a!)"),
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          std::optional<at::Generator> generator =
              pop(stack).toOptional<at::Generator>();

          double a = 0;
          double b = 0;
          pop(stack, tensor, a, b);
          push(stack, tensor.uniform_(a, b, generator));
        },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_normal_(Tensor(a!) tensor, float mean, float std, Generator? generator=None) -> Tensor(a!)"),
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          double mean = 0;
          double std = 0;
          std::optional<at::Generator> generator =
              pop(stack).toOptional<at::Generator>();

          pop(stack, tensor, mean, std);
          push(stack, tensor.normal_(mean, std, generator));
        },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_fill_(Tensor(a!) tensor, float val) -> Tensor(a!)"),
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          double val = 0;
          pop(stack, tensor, val);
          push(stack, at::fill_(tensor, val));
        },
        aliasAnalysisFromSchema()),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_no_grad_zero_(Tensor(a!) tensor) -> Tensor(a!)"),
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          pop(stack, tensor);
          push(stack, at::zero_(tensor));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::is_grad_enabled() -> bool",
        [](Stack& stack) { push(stack, torch::GradMode::is_enabled()); },
        aliasAnalysisConservative()),
    Operator(
        "aten::set_grad_enabled(bool val) -> ()",
        [](Stack& stack) { torch::GradMode::set_enabled(pop(stack).toBool()); },
        aliasAnalysisConservative()),
    Operator(
        "aten::_get_cpu_capability() -> str",
        [](Stack& stack) { push(stack, at::get_cpu_capability()); },
        aliasAnalysisConservative()),
});
} // namespace
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Context.h`
- `torch/library.h`
- `ATen/ExpandUtils.h`
- `ATen/NativeFunctions.h`
- `ATen/core/jit_type.h`
- `c10/core/DefaultDtype.h`
- `c10/util/irange.h`
- `torch/csrc/api/include/torch/utils.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/runtime/custom_operator.h`
- `torch/csrc/jit/runtime/operator.h`
- `torch/csrc/jit/runtime/vararg_functions.h`
- `ATen/InitialTensorOptions.h`
- `c10/core/ScalarType.h`
- `torch/csrc/jit/frontend/error_report.h`
- `sstream`


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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `register_special_ops.cpp_docs.md`
- **Keyword Index**: `register_special_ops.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/csrc/jit/runtime`):

- [`register_ops_utils.h_docs.md_docs.md`](./register_ops_utils.h_docs.md_docs.md)
- [`register_c10_ops.cpp_docs.md_docs.md`](./register_c10_ops.cpp_docs.md_docs.md)
- [`exception_message.h_kw.md_docs.md`](./exception_message.h_kw.md_docs.md)
- [`register_prim_ops.cpp_kw.md_docs.md`](./register_prim_ops.cpp_kw.md_docs.md)
- [`autodiff.cpp_kw.md_docs.md`](./autodiff.cpp_kw.md_docs.md)
- [`decomposition_registry_util.h_docs.md_docs.md`](./decomposition_registry_util.h_docs.md_docs.md)
- [`slice_indices_adjust.cpp_docs.md_docs.md`](./slice_indices_adjust.cpp_docs.md_docs.md)
- [`graph_iterator.h_kw.md_docs.md`](./graph_iterator.h_kw.md_docs.md)
- [`shape_function_registry.h_docs.md_docs.md`](./shape_function_registry.h_docs.md_docs.md)
- [`symbolic_script.cpp_docs.md_docs.md`](./symbolic_script.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `register_special_ops.cpp_docs.md_docs.md`
- **Keyword Index**: `register_special_ops.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
