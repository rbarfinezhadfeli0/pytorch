# Documentation: `docs/aten/src/ATen/functorch/VmapModeRegistrations.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/functorch/VmapModeRegistrations.cpp_docs.md`
- **Size**: 5,266 bytes (5.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/functorch/VmapModeRegistrations.cpp`

## File Metadata

- **Path**: `aten/src/ATen/functorch/VmapModeRegistrations.cpp`
- **Size**: 2,617 bytes (2.56 KB)
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

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/core/dispatch/Dispatcher.h>

// functorch's vmap has two Dispatch Keys that implement it:
// FuncTorchBatched and FuncTorchVmapMode. This file contains registrations for
// FuncTorchVmapMode -- these registrations are to error out on operations
// that we don't support on regular Tensors.

namespace at::functorch {

static void unsupportedRandomOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "vmap: We do not support calling out variants of random operations inside of vmap. ",
              "Please use non-out variants as a workaround");
}

TORCH_LIBRARY_IMPL(_, FuncTorchVmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

static void nyiRandomOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "vmap: we do not yet support ", op.schema().operator_name(),
              ". Please file an issue");
}

#define UNSUPPORTED_RANDOM(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define UNSUPPORTED_RANDOM2(op, overload) \
  m.impl(#op"."#overload, torch::CppFunction::makeFromBoxedFunction<&unsupportedRandomOp>());

#define NYI_RANDOM(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&nyiRandomOp>());

#define NYI_RANDOM2(op, overload) \
  m.impl(#op"."#overload, torch::CppFunction::makeFromBoxedFunction<&nyiRandomOp>());

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  UNSUPPORTED_RANDOM2(bernoulli, out);
  UNSUPPORTED_RANDOM2(rand, generator_out);
  UNSUPPORTED_RANDOM2(rand, out);
  UNSUPPORTED_RANDOM2(randint, generator_out);
  UNSUPPORTED_RANDOM2(randint, out);
  UNSUPPORTED_RANDOM2(randn, generator_out);
  UNSUPPORTED_RANDOM2(randn, out);
  UNSUPPORTED_RANDOM2(randperm, generator_out);
  UNSUPPORTED_RANDOM2(randperm, out);
  UNSUPPORTED_RANDOM2(multinomial, out);
  UNSUPPORTED_RANDOM2(normal, float_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, Tensor_Tensor_out);
  UNSUPPORTED_RANDOM2(normal, float_float_out);
  UNSUPPORTED_RANDOM2(rrelu_with_noise, out);

  NYI_RANDOM(rrelu_with_noise);
  NYI_RANDOM(rrelu_with_noise_);
  NYI_RANDOM(rrelu_);
  NYI_RANDOM(rrelu);
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

- `torch/library.h`
- `ATen/ATen.h`
- `ATen/functorch/LegacyVmapTransforms.h`
- `ATen/functorch/BatchedTensorImpl.h`
- `ATen/functorch/PlumbingHelper.h`
- `ATen/functorch/DynamicLayer.h`
- `ATen/core/dispatch/Dispatcher.h`


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
- [`PlumbingHelper.h_docs.md`](./PlumbingHelper.h_docs.md)
- [`BatchRulesFactory.cpp_docs.md`](./BatchRulesFactory.cpp_docs.md)
- [`BatchedTensorImpl.cpp_docs.md`](./BatchedTensorImpl.cpp_docs.md)


## Cross-References

- **File Documentation**: `VmapModeRegistrations.cpp_docs.md`
- **Keyword Index**: `VmapModeRegistrations.cpp_kw.md`
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

- **File Documentation**: `VmapModeRegistrations.cpp_docs.md_docs.md`
- **Keyword Index**: `VmapModeRegistrations.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
