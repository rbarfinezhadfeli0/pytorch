# Documentation: `docs/aten/src/ATen/VmapModeRegistrations.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/VmapModeRegistrations.cpp_docs.md`
- **Size**: 10,717 bytes (10.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/VmapModeRegistrations.cpp`

## File Metadata

- **Path**: `aten/src/ATen/VmapModeRegistrations.cpp`
- **Size**: 8,269 bytes (8.08 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/library.h>
#include <ATen/core/boxing/KernelFunction.h>

using torch::CppFunction;

namespace at {

// Note: [DispatchKey::VmapMode usage]
// Whenever we're inside a vmap, all Tensors dispatch on this key. At the moment,
// this key is used to disable random operations inside of vmap. If you are looking
// for Batching Rules, those are registered with DispatchKey::Batched instead.
//
// Note: [Ambiguity of random operations inside vmap]
// Random operations have an ambiguity where it isn't clear if they should
// apply the same randomness or apply different randomness. For example:
//
// >>> vmap(lambda t: torch.rand(1))(torch.zeros(5))
// Should the above return the same random number 5 times, or a different one?
//
// We haven't made a decision on that yet so we are temporarily banning random
// operations inside of vmap while we gather user feedback.

template <typename... Args> static Tensor unsupportedRandomOp(Args... args) {
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

template <typename... Args> static Tensor& unsupportedRandomOp_(Args... args) {
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

TORCH_LIBRARY_IMPL(_, VmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, VmapMode, m) {
  // NB: I'd really like to register a special kernel like
  // CppFunction::makeNamedNotSupported() to avoid listing out the types of everything.
  // However, registering e.g. CppFunction::makeNamedNotSupported() as an implementation
  // only works for operators that support boxing.
#define TENSOROPTIONS std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>

  // random operations (out-of-place)
  m.impl("bernoulli", unsupportedRandomOp<const Tensor&, std::optional<Generator>>);
  m.impl("bernoulli.out", unsupportedRandomOp_<const Tensor&, std::optional<Generator>, Tensor&>);
  m.impl("bernoulli.p", unsupportedRandomOp<const Tensor&, double, std::optional<Generator>>);
  m.impl("bernoulli_.Tensor", unsupportedRandomOp_<Tensor&, const Tensor&, std::optional<Generator>>);
  m.impl("bernoulli_.float", unsupportedRandomOp_<Tensor&, double, std::optional<Generator>>);

  m.impl("cauchy_", unsupportedRandomOp_<Tensor&, double, double, std::optional<Generator>>);
  m.impl("exponential_", unsupportedRandomOp_<Tensor&, double, std::optional<Generator>>);
  m.impl("geometric_", unsupportedRandomOp_<Tensor&, double, std::optional<Generator>>);
  m.impl("log_normal_", unsupportedRandomOp_<Tensor&, double, double, std::optional<Generator>>);
  m.impl("multinomial", unsupportedRandomOp<const Tensor&, int64_t, bool, std::optional<Generator>>);
  m.impl("multinomial.out", unsupportedRandomOp_<const Tensor&, int64_t, bool, std::optional<Generator>, Tensor&>);

  m.impl("normal.Tensor_float", unsupportedRandomOp<const Tensor&, double, std::optional<Generator>>);
  m.impl("normal.Tensor_float_out", unsupportedRandomOp_<const Tensor&, double, std::optional<Generator>, Tensor&>);
  m.impl("normal.float_Tensor_out", unsupportedRandomOp_<double, const Tensor&, std::optional<Generator>, Tensor&>);
  m.impl("normal.float_Tensor", unsupportedRandomOp<double, const Tensor&, std::optional<Generator>>);
  m.impl("normal.Tensor_Tensor", unsupportedRandomOp<const Tensor&, const Tensor&, std::optional<Generator>>);
  m.impl("normal.Tensor_Tensor_out", unsupportedRandomOp_<const Tensor&, const Tensor&, std::optional<Generator>, Tensor&>);
  m.impl("normal.float_float", unsupportedRandomOp<double, double, IntArrayRef, std::optional<Generator>, TENSOROPTIONS>);
  m.impl("normal.float_float_out", unsupportedRandomOp_<double, double, IntArrayRef, std::optional<Generator>, Tensor&>);
  m.impl("normal_", unsupportedRandomOp_<Tensor&, double, double, std::optional<Generator>>);

  m.impl("poisson", unsupportedRandomOp<const Tensor&, std::optional<Generator>>);

  m.impl("random_.from", unsupportedRandomOp_<Tensor&, int64_t, std::optional<int64_t>, std::optional<Generator>>);
  m.impl("random_.to", unsupportedRandomOp_<Tensor&, int64_t, std::optional<Generator>>);
  m.impl("random_", unsupportedRandomOp_<Tensor&, std::optional<Generator>>);

  m.impl("rand_like", unsupportedRandomOp<const Tensor&, TENSOROPTIONS, std::optional<MemoryFormat>>);
  m.impl("rand_like.generator", unsupportedRandomOp<const Tensor&, std::optional<Generator>, TENSOROPTIONS, std::optional<MemoryFormat>>);
  m.impl("randn_like", unsupportedRandomOp<const Tensor&, TENSOROPTIONS, std::optional<MemoryFormat>>);
  m.impl("randn_like.generator", unsupportedRandomOp<const Tensor&, std::optional<Generator>, TENSOROPTIONS, std::optional<MemoryFormat>>);

  m.impl("randint_like", unsupportedRandomOp<const Tensor&, int64_t, TENSOROPTIONS, std::optional<MemoryFormat>>);
  m.impl("randint_like.Tensor", unsupportedRandomOp<const Tensor&, const Tensor&, TENSOROPTIONS, std::optional<MemoryFormat>>);
  m.impl("randint_like.low_dtype", unsupportedRandomOp<const Tensor&, int64_t, int64_t, TENSOROPTIONS, std::optional<MemoryFormat>>);
  m.impl("randint_like.generator", unsupportedRandomOp<const Tensor&, int64_t, std::optional<Generator>, TENSOROPTIONS, std::optional<MemoryFormat>>);
  m.impl("randint_like.Tensor_generator", unsupportedRandomOp<const Tensor&, const Tensor&, std::optional<Generator>, TENSOROPTIONS, std::optional<MemoryFormat>>);
  m.impl("randint_like.low_generator_dtype", unsupportedRandomOp<const Tensor&, int64_t, int64_t, std::optional<Generator>, TENSOROPTIONS, std::optional<MemoryFormat>>);

  m.impl("rand", unsupportedRandomOp<IntArrayRef, TENSOROPTIONS>);
  m.impl("rand.generator", unsupportedRandomOp<IntArrayRef, std::optional<Generator>, TENSOROPTIONS>);
  m.impl("rand.names", unsupportedRandomOp<IntArrayRef, std::optional<DimnameList>, TENSOROPTIONS>);
  m.impl("rand.generator_with_names", unsupportedRandomOp<IntArrayRef, std::optional<Generator>, std::optional<DimnameList>, TENSOROPTIONS>);
  m.impl("rand.out", unsupportedRandomOp_<IntArrayRef, Tensor&>);
  m.impl("rand.generator_out", unsupportedRandomOp_<IntArrayRef, std::optional<Generator>, Tensor&>);

  m.impl("randn", unsupportedRandomOp<IntArrayRef, TENSOROPTIONS>);
  m.impl("randn.generator", unsupportedRandomOp<IntArrayRef, std::optional<Generator>, TENSOROPTIONS>);
  m.impl("randn.names", unsupportedRandomOp<IntArrayRef, std::optional<DimnameList>, TENSOROPTIONS>);
  m.impl("randn.generator_with_names", unsupportedRandomOp<IntArrayRef, std::optional<Generator>, std::optional<DimnameList>, TENSOROPTIONS>);
  m.impl("randn.out", unsupportedRandomOp_<IntArrayRef, Tensor&>);
  m.impl("randn.generator_out", unsupportedRandomOp_<IntArrayRef, std::optional<Generator>, Tensor&>);

  m.impl("randperm", unsupportedRandomOp<int64_t, TENSOROPTIONS>);
  m.impl("randperm.generator", unsupportedRandomOp<int64_t, std::optional<Generator>, TENSOROPTIONS>);
  m.impl("randperm.out", unsupportedRandomOp_<int64_t, Tensor&>);
  m.impl("randperm.generator_out", unsupportedRandomOp_<int64_t, std::optional<Generator>, Tensor&>);

  m.impl("randint", unsupportedRandomOp<int64_t, IntArrayRef, TENSOROPTIONS>);
  m.impl("randint.generator", unsupportedRandomOp<int64_t, IntArrayRef, std::optional<Generator>, TENSOROPTIONS>);
  m.impl("randint.low", unsupportedRandomOp<int64_t, int64_t, IntArrayRef, TENSOROPTIONS>);
  m.impl("randint.low_generator", unsupportedRandomOp<int64_t, int64_t, IntArrayRef, std::optional<Generator>, TENSOROPTIONS>);
  m.impl("randint.out", unsupportedRandomOp_<int64_t, IntArrayRef, Tensor&>);
  m.impl("randint.generator_out", unsupportedRandomOp_<int64_t, IntArrayRef, std::optional<Generator>, Tensor&>);
  m.impl("randint.low_out", unsupportedRandomOp_<int64_t, int64_t, IntArrayRef, Tensor&>);
  m.impl("randint.low_generator_out", unsupportedRandomOp_<int64_t, int64_t, IntArrayRef, std::optional<Generator>, Tensor&>);

  m.impl("uniform_", unsupportedRandomOp_<Tensor&, double, double, std::optional<Generator>>);

#undef TENSOROPTIONS
}


} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `torch/library.h`
- `ATen/core/boxing/KernelFunction.h`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `VmapModeRegistrations.cpp_docs.md`
- **Keyword Index**: `VmapModeRegistrations.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen`):

- [`Dispatch.cpp_docs.md_docs.md`](./Dispatch.cpp_docs.md_docs.md)
- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`ThreadLocalState.cpp_docs.md_docs.md`](./ThreadLocalState.cpp_docs.md_docs.md)
- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`FunctionalInverses.cpp_kw.md_docs.md`](./FunctionalInverses.cpp_kw.md_docs.md)
- [`SequenceNumber.h_kw.md_docs.md`](./SequenceNumber.h_kw.md_docs.md)
- [`ThreadLocalPythonObjects.h_docs.md_docs.md`](./ThreadLocalPythonObjects.h_docs.md_docs.md)
- [`TensorNames.h_docs.md_docs.md`](./TensorNames.h_docs.md_docs.md)
- [`LegacyBatchedTensorImpl.h_docs.md_docs.md`](./LegacyBatchedTensorImpl.h_docs.md_docs.md)
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `VmapModeRegistrations.cpp_docs.md_docs.md`
- **Keyword Index**: `VmapModeRegistrations.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
