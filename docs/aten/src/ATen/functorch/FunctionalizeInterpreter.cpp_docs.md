# Documentation: `aten/src/ATen/functorch/FunctionalizeInterpreter.cpp`

## File Metadata

- **Path**: `aten/src/ATen/functorch/FunctionalizeInterpreter.cpp`
- **Size**: 3,020 bytes (2.95 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/functorch/FunctionalizeInterpreter.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/FunctionalTensorWrapper.h>

namespace at::functorch {

static void sanityCheckNotFunctional(const c10::OperatorHandle& op, torch::jit::Stack* stack, size_t num_args) {
  foreachTensorInplace(*stack, static_cast<std::ptrdiff_t>(stack->size() - num_args), static_cast<std::ptrdiff_t>(stack->size()),
      [](const Tensor& tensor) {
        TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(tensor));
        return tensor;
      });
}

void FunctionalizeInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  // We always want to call the functionalization kernels if functionalize() is on the layer stack.
  // It's the responsibility of the functionalization kernel to no-op and redispatch
  // if none of the input tensors are functional.
  setup_dispatch_key_tls(TransformType::Functionalize, DispatchKeySet(DispatchKey::Functionalize));
  auto functionalization_add_back_views = functionalizeAddBackViews();
  // We have some side-car TLS that we can set to toggle the functionaliation behavior.
  // If set, then we functionalization will only remove mutations, instead of
  // removing both mutations AND view operators.
  at::functionalization::impl::FunctionalizationReapplyViewsGuard functional_guard(functionalization_add_back_views);

  op.callBoxed(stack);

  auto ret_size = op.schema().returns().size();
  foreachTensorInplace(*stack, static_cast<std::ptrdiff_t>(stack->size() - ret_size), static_cast<std::ptrdiff_t>(stack->size()),
    [&](const Tensor& tensor) {
      if (at::functionalization::impl::isFunctionalTensor(tensor)) {
        auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
        // Functorch is responsible for setting the level on the wrapper, since we don't
        // have that info available in core (for now).
        // We could just "propagate" the level from the input tensors inside of the functionalize kernels,
        // but unfortunately we can't do that for factory operators.
        wrapper->set_level(level());
      }
      return tensor;
    }
  );
}

void FunctionalizeInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool grad_special_case) {
  // For now, we don't support nested functionalization calls.
  // This check just enforces that - after the functionalize kernel runs
  // and we hit the BackModeFallback, we'll have unwrapped our FunctionalTensors
  // so we can check that the unwrapped thing is not another (nested) FunctionalTensor.
  auto args_size = op.schema().arguments().size();
  sanityCheckNotFunctional(op, stack, args_size);

  // Re-dispatch
  if (getDynamicLayerStack().empty()) {
    sanityCheckStack(op, stack);
  }
  op.callBoxed(stack);

  auto ret_size = op.schema().returns().size();
  sanityCheckNotFunctional(op, stack, ret_size);
}

} // namespace at::functorch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

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

- `ATen/functorch/FunctionalizeInterpreter.h`
- `ATen/functorch/DynamicLayer.h`
- `ATen/FunctionalTensorWrapper.h`


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

- **File Documentation**: `FunctionalizeInterpreter.cpp_docs.md`
- **Keyword Index**: `FunctionalizeInterpreter.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
