# Documentation: `aten/src/ATen/functorch/BatchRulesDynamic.cpp`

## File Metadata

- **Path**: `aten/src/ATen/functorch/BatchRulesDynamic.cpp`
- **Size**: 3,565 bytes (3.48 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Metaprogramming.h>

// This file contains batching rules for operations that return Tensors of
// dynamic shape. We generally don't support those with vmap so we raise
// errors for them.


namespace at::functorch {

namespace {
void unsupportedDynamicOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false, "vmap: We do not support batching operators that can output dynamic shape. ",
        "Attempted to vmap over ", op.schema().operator_name(), ". ",
        "Please voice your support in https://github.com/pytorch/functorch/issues/256");
}
#define UNSUPPORTED_DYNAMIC(op) \
    m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&unsupportedDynamicOp>());

// NB: item and is_nonzero can decompose to this...
void unsupportedLocalScalarDense(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "vmap: It looks like you're either (1) calling .item() on a Tensor or ",
        "(2) attempting to use a Tensor in some data-dependent control flow or ",
        "(3) encountering this error in PyTorch internals. ",
        "For (1): we don't support vmap over calling .item() on a Tensor, please try to ",
        "rewrite what you're doing with other operations. ",
        "For (2): If you're doing some ",
        "control flow instead, we don't support that yet, please shout over at ",
        "https://github.com/pytorch/functorch/issues/257 . ",
        "For (3): please file an issue.");
}

void unsupportedItem(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "vmap: It looks like you're calling .item() on a Tensor. ",
        "We don't support vmap over calling .item() on a Tensor, please try to ",
        "rewrite what you're doing with other operations. If error is occurring ",
        "somewhere inside PyTorch internals, please file a bug report.");
}

void unsupportedIsNonzero(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "vmap: It looks like you're attempting to use a Tensor in some ",
        "data-dependent control flow. ",
        "We don't support that yet, please shout over at ",
        "https://github.com/pytorch/functorch/issues/257 .");
}

void unsupportedAllclose(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "vmap over torch.allclose isn't supported yet. Please voice your ",
        "support over at github.com/pytorch/functorch/issues/275");
}
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
    UNSUPPORTED_DYNAMIC(nonzero);
    UNSUPPORTED_DYNAMIC(where);
    UNSUPPORTED_DYNAMIC(unique_dim);
    UNSUPPORTED_DYNAMIC(unique_consecutive);
    UNSUPPORTED_DYNAMIC(unique_dim_consecutive);
    UNSUPPORTED_DYNAMIC(_unique2);
    m.impl("_local_scalar_dense", torch::CppFunction::makeFromBoxedFunction<&unsupportedLocalScalarDense>());
    m.impl("item", torch::CppFunction::makeFromBoxedFunction<&unsupportedItem>());
    m.impl("is_nonzero", torch::CppFunction::makeFromBoxedFunction<&unsupportedIsNonzero>());
    m.impl("allclose", torch::CppFunction::makeFromBoxedFunction<&unsupportedAllclose>());
}

} // namespace at::functorch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

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

- `ATen/ATen.h`
- `ATen/functorch/BatchRulesHelper.h`
- `ATen/functorch/BatchedFallback.h`
- `ATen/core/dispatch/Dispatcher.h`
- `c10/util/Metaprogramming.h`


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

- **File Documentation**: `BatchRulesDynamic.cpp_docs.md`
- **Keyword Index**: `BatchRulesDynamic.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
