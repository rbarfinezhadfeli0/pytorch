# Documentation: `aten/src/ATen/templates/RegisterFunctionalization.cpp`

## File Metadata

- **Path**: `aten/src/ATen/templates/RegisterFunctionalization.cpp`
- **Size**: 3,461 bytes (3.38 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// ${generated_comment}

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/ViewMetaClasses.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>

#include <c10/util/env.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#include <ATen/NativeFunctions.h>
#else
// needed for the meta tensor calls to get stride info in functionalization
#include <ATen/ops/empty_strided_native.h>
// needed for special handling of copy_().
// See Note [functionalizating copy_() and not preserving strides]
#include <ATen/ops/to_ops.h>
#include <ATen/ops/expand_copy_ops.h>

$ops_headers
#endif

namespace at {
namespace functionalization {

// This keyset is used by functionalization when it calls into meta kernels
// to accurately propagate stride metadata.
// Exclude any modes: the purpose of calling into meta kernels is only as an implementation
// detail to perform shape inference, and we don't want any modal keys to run.
// Specifically, we want to prevent functionalization and Python modes from running.
constexpr auto exclude_keys_for_meta_dispatch =
    c10::functorch_transforms_ks |
    c10::DispatchKeySet({
        c10::DispatchKey::FuncTorchDynamicLayerBackMode,
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
        c10::DispatchKey::Python,
        c10::DispatchKey::PreDispatch,

    });

// Helper around at::has_internal_overlap.
// The ATen util is used in hot-path eager mode: it's always fast,
// but might return TOO_HARD sometimes.
// During functionalization, we're ok taking a bit longer
// to detect memory overlap.
inline bool has_internal_overlap_helper(const at::Tensor t) {
  auto has_overlap = at::has_internal_overlap(t);
  if (has_overlap == at::MemOverlap::Yes) return true;
  if (has_overlap == at::MemOverlap::No) return false;
  return false;
}


inline Tensor to_meta(const Tensor& t) {
    if (!t.defined()) return t;
    return at::native::empty_strided_meta_symint(t.sym_sizes(), t.sym_strides(),
/*dtype=*/t.scalar_type(), /*layout=*/t.layout(),
/*device=*/c10::Device(kMeta), /*pin_memory=*/std::nullopt);
}

inline std::optional<Tensor> to_meta(const std::optional<Tensor>& t) {
  if (t.has_value()) {
    return to_meta(*t);
  }
  return std::nullopt;
}

inline std::vector<Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    outputs.push_back(to_meta(tensor));
  }
  return outputs;
}

inline c10::List<Tensor> to_meta(const c10::List<Tensor>& t_list) {
  c10::List<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(to_meta(t_list[i]));
  }
  return outputs;
}

inline c10::List<::std::optional<Tensor>> to_meta(const c10::List<::std::optional<Tensor>>& t_list) {
  c10::List<::std::optional<Tensor>> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(to_meta(t_list[i]));
  }
  return outputs;
}

static bool disable_meta_reference() {
  static auto env = c10::utils::get_env("TORCH_DISABLE_FUNCTIONALIZATION_META_REFERENCE");
  return env == "1";
}


${func_definitions}

}  // namespace functionalization

namespace {

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  ${func_registrations};
}

}  // namespace

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `functionalization`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/templates`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/LegacyTypeDispatch.h`
- `ATen/EmptyTensor.h`
- `ATen/FunctionalTensorWrapper.h`
- `ATen/ViewMetaClasses.h`
- `ATen/MemoryOverlap.h`
- `torch/library.h`
- `c10/util/env.h`
- `ATen/Operators.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty_strided_native.h`
- `ATen/ops/to_ops.h`
- `ATen/ops/expand_copy_ops.h`


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

Files in the same folder (`aten/src/ATen/templates`):

- [`NativeFunction.h_docs.md`](./NativeFunction.h_docs.md)
- [`DispatchKeyFunctions.h_docs.md`](./DispatchKeyFunctions.h_docs.md)
- [`aten_interned_strings.h_docs.md`](./aten_interned_strings.h_docs.md)
- [`UfuncCPUKernel.cpp_docs.md`](./UfuncCPUKernel.cpp_docs.md)
- [`DispatchKeyFunction.h_docs.md`](./DispatchKeyFunction.h_docs.md)
- [`LazyIr.h_docs.md`](./LazyIr.h_docs.md)
- [`RegisterDispatchDefinitions.ini_docs.md`](./RegisterDispatchDefinitions.ini_docs.md)
- [`Functions.cpp_docs.md`](./Functions.cpp_docs.md)
- [`RegisterDispatchKey.cpp_docs.md`](./RegisterDispatchKey.cpp_docs.md)
- [`MethodOperators.h_docs.md`](./MethodOperators.h_docs.md)


## Cross-References

- **File Documentation**: `RegisterFunctionalization.cpp_docs.md`
- **Keyword Index**: `RegisterFunctionalization.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
