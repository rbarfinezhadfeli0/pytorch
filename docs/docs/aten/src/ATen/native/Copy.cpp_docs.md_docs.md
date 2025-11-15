# Documentation: `docs/aten/src/ATen/native/Copy.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Copy.cpp_docs.md`
- **Size**: 17,482 bytes (17.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Copy.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Copy.cpp`
- **Size**: 14,172 bytes (13.84 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Copy.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/quantized/Copy.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/TensorShape.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/vulkan/Context.h>
#include <ATen/metal/Context.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_copy_from.h>
#include <ATen/ops/_propagate_xla_data.h>
#include <ATen/ops/_propagate_xla_data_native.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/_foreach_copy.h>
#include <ATen/ops/_foreach_copy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/expand_copy.h>
#endif

#ifdef USE_FBGEMM
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wextra-semi")
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
C10_DIAGNOSTIC_POP()
#endif

namespace {

using namespace at;

bool copy_transpose_valid(const Tensor& self, const Tensor& src) {
  const int MIN_SZ = 60 * 60;
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.scalar_type() == src.scalar_type() &&
      !isBitsType(self.scalar_type()) &&
      self.sizes().equals(src.sizes()) &&
      self.is_neg() == src.is_neg() &&
      self.is_conj() == src.is_conj() &&
      self.numel() >= MIN_SZ;
}

#if !defined(C10_MOBILE)
#define _AT_DISPATCH_CP_TYPES(TYPE, NAME, ...)                              \
        AT_DISPATCH_V2(                             \
            TYPE, NAME, AT_WRAP(__VA_ARGS__), kComplexHalf, kHalf, kBool, kBFloat16,            \
            AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES))
#else
#define _AT_DISPATCH_CP_TYPES(TYPE, NAME, ...)     \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(    \
            kComplexHalf, kHalf, kBool, kBFloat16, \
            TYPE, NAME, __VA_ARGS__)
#endif

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's trickier
void copy_same_type_transpose_(Tensor& self, const Tensor& src) {
  int64_t BLOCK_SZ = 0;
  if (self.scalar_type() == kByte) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 120;
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 60;
  }
  Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());

  // The code below is implemented with the assumption that sizes are equal
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.sizes().equals(src.sizes()));

  _AT_DISPATCH_CP_TYPES(self.scalar_type(), "copy_", [&] {
    const scalar_t* sp = src.const_data_ptr<scalar_t>();
    scalar_t* rp = self.data_ptr<scalar_t>();
    scalar_t* bp = buf.data_ptr<scalar_t>();

    int64_t NR = src.size(0);
    int64_t NC = src.size(1);
    for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
      for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
        const scalar_t* spo = sp + R + C * NR;
        scalar_t* rpo = rp + C + R * NC;

        int nr = std::min(NR - R, BLOCK_SZ);
        int nc = std::min(NC - C, BLOCK_SZ);

        // 1. copy columns from src to buf
        for (const auto c : c10::irange(nc)) {
          memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
        }

        // 2. transpose buf in place
        int rc_max = std::max(nr, nc);
        int rc_min = std::min(nr, nc);
        for (const auto r : c10::irange(rc_max)) {
          int end = std::min(r, rc_min);
          for (const auto c : c10::irange(end)) {
            scalar_t tmp = bp[r + BLOCK_SZ * c];
            bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
            bp[r * BLOCK_SZ + c] = tmp;
          }
        }

        // 3. copy rows from buf to dst
        for (const auto r : c10::irange(nr)) {
          memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
        }
      }
    }
  });
}

// Devices directly supported by this copy implementation. Other device types
// (e.g. XLA) may be supported by overriding copy_ and _copy_from.
bool is_supported_device(Device device) {
  DeviceType device_type = device.type();
  return device_type == kCPU || device_type == kCUDA || device_type == kHIP || device_type == kVulkan || device_type == kMetal || device_type == kMPS || device_type == kXPU;
}

} // namespace

namespace at::native {

static Tensor & copy_impl(Tensor & self, const Tensor & src, bool non_blocking) {
  // TODO: this should be handled during dispatch, but that's missing...
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  // FBGeMM kernel support exists only for the following case,
  // 1. Memory Format for source and destination tensors is contiguous.
  // 2. Device for both the source and destination tensor is CPU.
  // 3. dtype conversion between FP32->FP16 and FP16->FP32.
  // This checks that self.sizes() == src.sizes() because this code path doesn't
  // support broadcasting. This also guards against out of bounds memory access
  // when copying, see fbgemm::Float16ToFloat_ref.
  // https://github.com/pytorch/pytorch/issues/88543
  #ifdef USE_FBGEMM
    if (((self.dtype() == at::kFloat && src.dtype() == at::kHalf) ||
         (self.dtype() == at::kHalf && src.dtype() == at::kFloat)) &&
        (self.device().is_cpu() && src.device().is_cpu()) &&
        ((self.is_contiguous() && src.is_contiguous()) ||
         (self.is_non_overlapping_and_dense() && self.strides() == src.strides())) &&
        (self.sizes() == src.sizes())) {
      if (src.dtype() == at::kFloat && self.dtype() == at::kHalf) {
        auto* output_ptr =
            reinterpret_cast<fbgemm::float16*>(self.data_ptr<at::Half>());
        if (self.numel() < at::internal::GRAIN_SIZE) {
          fbgemm::FloatToFloat16_simd(src.const_data_ptr<float>(), output_ptr, self.numel());
        } else {
          at::parallel_for(
              0,
              self.numel(),
              at::internal::GRAIN_SIZE,
              [&](int64_t begin, int64_t end) {
                fbgemm::FloatToFloat16_simd(
                    src.const_data_ptr<float>() + begin,
                    output_ptr + begin,
                  end - begin);
              });
        }
      } else {
        auto in_data = reinterpret_cast<const fbgemm::float16*>(
            src.const_data_ptr<at::Half>());
        auto* output_ptr = self.data_ptr<float>();
        if (self.numel() < at::internal::GRAIN_SIZE) {
          fbgemm::Float16ToFloat_simd(in_data, output_ptr, self.numel());
        } else {
          at::parallel_for(
              0,
              self.numel(),
              at::internal::GRAIN_SIZE,
              [&](int64_t begin, int64_t end) {
                fbgemm::Float16ToFloat_simd(
                    in_data + begin, output_ptr + begin, end - begin);
              });
        }
      }
      return self;
    }
  #endif

  if (self.is_same(src)) {
    return self;
  }

  // Copies into meta self are OK and just ignored (similar to inplace)
  if (self.is_meta()) {
    auto shape = infer_size_symdimvector(self.sym_sizes(), src.sym_sizes());
    TORCH_CHECK(
        self.sym_sizes().equals(shape),
        "output with shape ",
        self.sym_sizes(),
        " doesn't match the broadcast shape ",
        shape);
    return self;
  }

  if (src.is_meta()) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Cannot copy out of meta tensor; no data!")
  }

  // Re-dispatch copies when either src or self device not implemented here (e.g. XLA).
  // _copy_from has a proper device dispatch setup.
  // This includes:
  //   cpu_tensor.copy_(xla_tensor) => xla_tensor._copy_from(cpu_tensor)
  //   xla_tensor.copy_(cpu_tensor) => cpu_tensor._copy_from(xla_tensor)
  // Both the _copy_from calls above will be dispatched to XLA's _copy_from kernels.

  if (!is_supported_device(src.device()) || !is_supported_device(self.device())) {
    at::_copy_from(src, self, non_blocking);
    return self;
  }

  if (self.is_quantized() && !src.is_quantized()) {
    return quantized_copy_from_float_(self, src);
  }

  if (self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(self.qscheme() == src.qscheme(),
                "Quantized Copy only works with same qscheme");
    TORCH_CHECK(self.scalar_type() == src.scalar_type());
    set_quantizer_(self, src.quantizer());
  }

  if (!self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(false, "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor");
  }

  if (self.device().type() == at::kVulkan || src.device().type() == at::kVulkan) {
  #ifdef USE_VULKAN_API
    return vulkan::ops::copy_(self, src);
  #else
    return at::vulkan::vulkan_copy_(self, src);
  #endif
  }

  if (self.device().type() == at::kMetal || src.device().type() == at::kMetal) {
    return at::metal::metal_copy_(self, src);
  }

  // Exit early if self and src are views of the same data
  const bool is_same_data = (
      self.is_alias_of(src) &&
      self.storage_offset() == src.storage_offset() &&
      self.strides().equals(src.strides()) &&
      self.sizes().equals(src.sizes()) &&
      self.scalar_type() == src.scalar_type() &&
      self.is_conj() == src.is_conj() &&
      self.is_neg() == src.is_neg()
    );
  if (is_same_data) {
    return self;
  }


  auto iter = TensorIteratorConfig()
    .add_output(self)
    .add_const_input(src)
    .resize_outputs(false)
    .check_all_same_dtype(false)
    .check_all_same_device(false)
    .build();

  if (iter.numel() == 0) {
    return self;
  }

  DeviceType device_type = iter.device_type(0);
  if (iter.device_type(1) == kCUDA) {
    device_type = kCUDA;
  } else if (iter.device_type(1) == kHIP) {
    device_type = kHIP;
  } else if (iter.device_type(1) == kMPS) {
    device_type = kMPS;
  } else if (iter.device_type(1) == kXPU){
    device_type = kXPU;
  }

  // TODO: if we need to, we can also enable this path for quantized tensor
  if (device_type == kCPU && copy_transpose_valid(self, src) && !self.is_quantized()) {
    copy_same_type_transpose_(self, src);
    return self;
  }

#ifdef USE_MPS
  if (self.device().type() == at::kMPS || src.device().type() == at::kMPS) {
    return at::native::mps::mps_copy_(self, src, non_blocking);
  }
#endif

  if(!(self.is_complex() || self.dtype() == at::kBool) && src.is_complex()) {
    TORCH_WARN_ONCE("Casting complex values to real discards the imaginary part");
  }
  copy_stub(device_type, iter, non_blocking);
  return self;
}

Tensor copy_meta(const Tensor& self, const Tensor& src, bool non_blocking) {
  // Must directly use self(), so we can dispatch properly is self is a subclass
  auto r = clone_preserve_strides(self);
  r.copy_(src, non_blocking);
  return r;
}

Tensor copy(const Tensor& self, const Tensor& src, bool non_blocking) {
  at::Tensor r;
  // copy() is the "functional" form of copy_(). It exists so we can properly functionalize copy_(), but:
  // (1) It isn't exposed to the frontend (no python bindings)
  // (2) It isn't exposed to the backend (it's a composite, that decomposes into to() and expand_as() calls.
  auto self_storage = self.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl();
  // If self has no real storage, we can't actually clone it.
  // Instead, generate an empty tensor with the right sizes/strides, since we should be able to assume
  // that copy_() will fully overwrite all data with that of src
  if (self_storage->nbytes() == 0) {
    r = at::empty_strided(self.sizes(), self.strides(), self.options());
  } else {
    r = clone_preserve_strides(self);
  }
  r.copy_(src, non_blocking);
  return r;
}

::std::vector<at::Tensor> _foreach_copy(at::TensorList self, at::TensorList src, bool non_blocking) {
  std::vector<at::Tensor> outs;
  outs.reserve(self.size());
  // This is a very slow implementation, but needs to directly call the copy() kernel above to handle
  // when self has zero storage.
  // This kernel should never really be run, except with debugging using compile(backend="aot_eager")
  for (const auto i : c10::irange(src.size())) {
    const auto& curr_src = src[i];
    const auto& curr_self = self[i];
    outs.push_back(at::copy(curr_self, curr_src, non_blocking));
  }
  return outs;
}

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  auto maybe_outnames = namedinference::compute_broadcast_outnames(self, src);
  {
    NoNamesGuard guard;
    if (self._is_zerotensor()) {
     TORCH_CHECK(false, "ZeroTensors are immutable. Please materialize the tensor using `.clone()`, if you want a mutable zero tensor.");
    }
    if (src._is_zerotensor()) {
      return self.zero_();
    }
    copy_impl(self, src, non_blocking);
  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

void copy_ignoring_overlaps(const TensorBase &dst, const TensorBase &src) {
  // Called when we are copying into an overlapping index `dst`, but we don't
  // care which writer wins. Hacky but it works. This is only used by
  // CUDA_tensor_apply2 in case that there are write overlaps.
  // FIXME: really, overlapping writes should be illegal/an error in Torch
  auto iter = TensorIteratorConfig()
      .add_output(dst)
      .add_const_input(src)
      .resize_outputs(false)
      .set_check_mem_overlap(false)
      .check_all_same_dtype(true)
      .check_all_same_device(true)
      .build();
  copy_stub(iter.device_type(), iter, /*non_blocking=*/false);
}

void _propagate_xla_data(const Tensor& input, const Tensor& output) {
  TORCH_INTERNAL_ASSERT(input.device().type() == kXLA, "This op should only be called by XLA")
}

DEFINE_DISPATCH(copy_stub);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 35 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `at`

**Classes/Structs**: `auto`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/Copy.h`
- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/Dispatch_v2.h`
- `ATen/ExpandUtils.h`
- `ATen/FunctionalTensorWrapper.h`
- `ATen/TensorIterator.h`
- `ATen/native/quantized/Copy.h`
- `ATen/native/mps/Copy.h`
- `ATen/native/vulkan/ops/Copy.h`
- `ATen/native/TensorShape.h`
- `ATen/quantized/Quantizer.h`
- `ATen/vulkan/Context.h`
- `ATen/metal/Context.h`
- `ATen/NamedTensorUtils.h`
- `ATen/Parallel.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_copy_from.h`
- `ATen/ops/_propagate_xla_data.h`
- `ATen/ops/_propagate_xla_data_native.h`
- `ATen/ops/copy.h`
- `ATen/ops/copy_native.h`
- `ATen/ops/_foreach_copy.h`
- `ATen/ops/_foreach_copy_native.h`
- `ATen/ops/empty.h`
- `ATen/ops/empty_strided.h`
- `ATen/ops/expand_copy.h`
- `fbgemm/Fbgemm.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `Copy.cpp_docs.md`
- **Keyword Index**: `Copy.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Copy.cpp_docs.md_docs.md`
- **Keyword Index**: `Copy.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
