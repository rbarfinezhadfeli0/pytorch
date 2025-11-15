# Documentation: `docs/aten/src/ATen/functorch/ADInterpreters.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/functorch/ADInterpreters.cpp_docs.md`
- **Size**: 11,391 bytes (11.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/functorch/ADInterpreters.cpp`

## File Metadata

- **Path**: `aten/src/ATen/functorch/ADInterpreters.cpp`
- **Size**: 8,799 bytes (8.59 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/functorch/ADInterpreters.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/TensorWrapper.h>
#include <bitset>

namespace at::functorch {

constexpr size_t default_bitset_size = 64;

static void checkForInvalidMutationOnCaptures(
    const c10::OperatorHandle& op,
    const torch::jit::Stack* stack,
    int64_t cur_level) {
  if (!isInplaceOp(op.schema())) {
    return;
  }
  auto args = torch::jit::last(stack, op.schema().arguments().size());
  auto mutated_arg = unwrapIfDead(args[0].toTensor());
  auto* wrapper = maybeGetTensorWrapper(mutated_arg);
  if (wrapper && wrapper->level() == cur_level && !(wrapper->is_immutable())) {
    return;
  }
  TORCH_CHECK(false,
      "During a grad (vjp, jvp, grad, etc) transform, the function provided ",
      "attempted to call in-place operation (", op.schema().operator_name(), ") ",
      "that would mutate a captured Tensor. This is not supported; please rewrite ",
      "the function being transformed to explicitly accept the mutated Tensor(s) ",
      "as inputs.");
}

static Tensor materializeGradWrappers(const Tensor& tensor, int64_t current_level) {
  if (!tensor.defined()) {
    return tensor;
  }
  // TensorWrapper creation may call dispatcher ops (e.g. aten.sym_storage_offset).
  // We need to ensure that they pass through the functorch stack properly.
  // In order to do that, we want to call those dispatcher ops at the next layer,
  // hence we disable DynamicLayerFrontMode so the call to the op automatically
  // goes to DynamicLayerBackMode which will then send it to the next layer.
  c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::FuncTorchDynamicLayerFrontMode);
  auto* wrapper = maybeGetTensorWrapper(tensor);
  if (!wrapper) {
    return makeTensorWrapper(tensor, current_level, /*is_immutable=*/true);
  }
  auto level = wrapper->level();
  TORCH_INTERNAL_ASSERT(level.has_value() && level <= current_level, "escaped?");
  if (level == current_level) {
    TORCH_INTERNAL_ASSERT(tensor.defined());
    return tensor;
  }
  return makeTensorWrapper(tensor, current_level, /*is_immutable=*/true);
}

Tensor GradInterpreterPtr::lift(const Tensor& tensor) const {
  return materializeGradWrappers(tensor, level());
}

Tensor JvpInterpreterPtr::lift(const Tensor& tensor) const {
  return materializeGradWrappers(tensor, level());
}

static void autogradBasedTransformProcess(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    int64_t current_level,
    TransformType transform_type) {
  // if is a grad transform, and the operation is in-place, and the mutated
  // argument is not currently wrapped in a TensorWrapper, then we need to
  // error out otherwise the result is silently incorrect
  checkForInvalidMutationOnCaptures(op, stack, current_level);

  // materialize live GradWrappers
  auto maybeTransformGradWrappers = [&](const Tensor& tensor) {
    return materializeGradWrappers(tensor, current_level);
  };
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, static_cast<int64_t>(stack->size() - num_args), static_cast<int64_t>(stack->size()), maybeTransformGradWrappers);

  setup_dispatch_key_tls(transform_type, {});
  op.callBoxed(stack);
}

static void autogradBasedTransformSendToNext(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const Interpreter& interpreter,
    TransformType transform_type,
    std::optional<bool> prev_grad_mode,
    std::optional<bool> prev_fwd_grad_mode,
    bool grad_special_case) {
  auto current_level = interpreter.level();
  if (transform_type == TransformType::Grad) {
    TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
  }
  if (transform_type == TransformType::Jvp) {
    TORCH_INTERNAL_ASSERT(prev_fwd_grad_mode.has_value());
  }
  auto unwrap = [&](const Tensor& tensor) {
    if (!tensor.defined()) {
      return tensor;
    }
    auto* maybe_tensor_wrapper = maybeGetTensorWrapper(tensor);
    if (!maybe_tensor_wrapper) {
      return tensor;
    }
    auto tensor_wrapper_level = maybe_tensor_wrapper->level().value();
    TORCH_INTERNAL_ASSERT(tensor_wrapper_level <= current_level);
    if (tensor_wrapper_level == current_level) {
      return maybe_tensor_wrapper->value();
    }
    return tensor;
  };
  auto wrap = [&](const Tensor& tensor, bool is_immutable) {
    if (!tensor.defined()) {
      return tensor;
    }
    return makeTensorWrapper(tensor, interpreter, is_immutable);
  };

  // TODO: we only need to do the following (marked with !) on in-place functions
  // that modify sizes or strides. There aren't many of them.
  // If autograd dispatch key:
  // 1. (!) Put a copy of all of the args onto the stack
  // 2. Unwrap all the args in the copy set
  // 3. Call the operator
  // 4. Wrap the output
  // 5. (!) refreshMetadata for all the args in the original set
  // 6. (!) Pop those args off.

  // Step 1 & 2
  auto args_size = op.schema().arguments().size();
  const auto ret_size = op.schema().returns().size();
  // Step 1
  auto front = static_cast<int64_t>(stack->size()) - args_size;
  for (const auto arg_idx : c10::irange(0, args_size)) {
    stack->push_back((*stack)[front + arg_idx]);
  }

  std::bitset<default_bitset_size> outputs_aliasing_immutable; // set = 1 for all bits
  if(!grad_special_case) {
    for (auto idx = stack->size() - args_size; idx < stack->size(); idx++) {
      const auto ivalue = (*stack)[idx];
      if (!ivalue.isTensor()) {
        continue; // only input that can be aliased is a tensor, not a tensor list (expect in ops without returns)
      }
      const auto& tensor = ivalue.toTensor();
      auto* maybe_tensor_wrapper = maybeGetTensorWrapper(tensor);
      if (!maybe_tensor_wrapper || maybe_tensor_wrapper->is_immutable()) {
        // if the input is immutable, we find if it aliases anything, noting that
        // args are in reverse order on stack, so the last arg is at the top of the stack
        const auto relative_pos = idx - (stack->size() - args_size);
        const auto aliased_out = findAliasedOutput(op.schema(), static_cast<int64_t>(relative_pos));
        if (aliased_out.has_value()) {
          outputs_aliasing_immutable.flip(*aliased_out); // each output aliases at most one input, so we can only hit this once
        }
      }
    }
  }

  // Step 2
  foreachTensorInplace(*stack, static_cast<int64_t>(stack->size() - args_size), static_cast<int64_t>(stack->size()), unwrap);

  // See NOTE [grad and vjp interaction with no_grad]
  std::optional<c10::AutoGradMode> grad_guard;
  if (transform_type == TransformType::Grad && prev_grad_mode.has_value() && *prev_grad_mode == false) {
    grad_guard.emplace(*prev_grad_mode);
  }
  std::optional<c10::AutoFwGradMode> fw_grad_guard;
  if (transform_type == TransformType::Jvp &&
      prev_fwd_grad_mode.has_value() && prev_fwd_grad_mode.value() == false) {
    fw_grad_guard.emplace(*prev_fwd_grad_mode);
  }

  // Re-dispatch
  if (getDynamicLayerStack().empty()) {
    sanityCheckStack(op, stack);
  }

  // Step 4, 5, 6

  op.callBoxed(stack);

  // Step 4
  foreachTensorInplaceWithFlag(*stack, static_cast<int64_t>(stack->size() - ret_size), static_cast<int64_t>(stack->size()), outputs_aliasing_immutable, wrap);

  // Step 5
  auto args_front = stack->size() - args_size - ret_size;
  for (const auto arg_idx : c10::irange(0, args_size)) {
    auto& ivalue = (*stack)[args_front + arg_idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    auto maybe_tensor_wrapper = maybeGetTensorWrapper(ivalue.toTensor());
    if (!maybe_tensor_wrapper) {
      continue;
    }
    maybe_tensor_wrapper->refreshMetadata();
  }

  // Step 6
  stack->erase(stack->end() - static_cast<std::ptrdiff_t>(args_size + ret_size), stack->end() - static_cast<std::ptrdiff_t>(ret_size));
}

void GradInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  autogradBasedTransformProcess(op, stack, level(), TransformType::Grad);
}

void GradInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool grad_special_case) {
  autogradBasedTransformSendToNext(
      op, stack, *base_,
      TransformType::Grad,
      prevGradMode(),
      std::nullopt,
      grad_special_case);
}

void JvpInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  autogradBasedTransformProcess(op, stack, level(), TransformType::Jvp);
}

void JvpInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool grad_special_case) {
  autogradBasedTransformSendToNext(
      op, stack, *base_,
      TransformType::Jvp,
      std::nullopt,
      prevFwdGradMode(),
      grad_special_case);
}

} // namespace at::functorch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/functorch/ADInterpreters.h`
- `ATen/functorch/DynamicLayer.h`
- `ATen/functorch/TensorWrapper.h`
- `bitset`


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

Files in the same folder (`aten/src/ATen/functorch`):

- [`Interpreter.cpp_docs.md`](./Interpreter.cpp_docs.md)
- [`Interpreter.h_docs.md`](./Interpreter.h_docs.md)
- [`BatchRulesScatterOps.cpp_docs.md`](./BatchRulesScatterOps.cpp_docs.md)
- [`BatchRulesHelper.h_docs.md`](./BatchRulesHelper.h_docs.md)
- [`BatchedFallback.cpp_docs.md`](./BatchedFallback.cpp_docs.md)
- [`BatchRulesLinearAlgebra.cpp_docs.md`](./BatchRulesLinearAlgebra.cpp_docs.md)
- [`VmapModeRegistrations.cpp_docs.md`](./VmapModeRegistrations.cpp_docs.md)
- [`PlumbingHelper.h_docs.md`](./PlumbingHelper.h_docs.md)
- [`BatchRulesFactory.cpp_docs.md`](./BatchRulesFactory.cpp_docs.md)
- [`BatchedTensorImpl.cpp_docs.md`](./BatchedTensorImpl.cpp_docs.md)


## Cross-References

- **File Documentation**: `ADInterpreters.cpp_docs.md`
- **Keyword Index**: `ADInterpreters.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/functorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/functorch`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/aten/src/ATen/functorch`):

- [`BatchRulesNorm.cpp_docs.md_docs.md`](./BatchRulesNorm.cpp_docs.md_docs.md)
- [`FunctionalizeInterpreter.h_kw.md_docs.md`](./FunctionalizeInterpreter.h_kw.md_docs.md)
- [`TensorWrapper.cpp_kw.md_docs.md`](./TensorWrapper.cpp_kw.md_docs.md)
- [`PlumbingHelper.h_docs.md_docs.md`](./PlumbingHelper.h_docs.md_docs.md)
- [`BatchRulesNorm.cpp_kw.md_docs.md`](./BatchRulesNorm.cpp_kw.md_docs.md)
- [`LegacyBatchingRegistrations.cpp_kw.md_docs.md`](./LegacyBatchingRegistrations.cpp_kw.md_docs.md)
- [`BatchRulesHelper.h_docs.md_docs.md`](./BatchRulesHelper.h_docs.md_docs.md)
- [`Interpreter.h_docs.md_docs.md`](./Interpreter.h_docs.md_docs.md)
- [`BatchedTensorImpl.cpp_docs.md_docs.md`](./BatchedTensorImpl.cpp_docs.md_docs.md)
- [`BatchRulesDecompositions.cpp_kw.md_docs.md`](./BatchRulesDecompositions.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ADInterpreters.cpp_docs.md_docs.md`
- **Keyword Index**: `ADInterpreters.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
