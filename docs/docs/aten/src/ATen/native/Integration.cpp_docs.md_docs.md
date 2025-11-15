# Documentation: `docs/aten/src/ATen/native/Integration.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Integration.cpp_docs.md`
- **Size**: 9,947 bytes (9.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Integration.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Integration.cpp`
- **Size**: 7,231 bytes (7.06 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/DimVector.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Scalar.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cumulative_trapezoid_native.h>
#include <ATen/ops/trapezoid_native.h>
#include <ATen/ops/trapz_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace {

// The estimated integral of a function y of x,
// sampled at points (y_1, ..., y_n) that are separated by distance (dx_1, ..., dx_{n-1}),
// is given by the trapezoid rule:
//
// \sum_{i=1}^{n-1}  dx_i * (y_i + y_{i+1}) / 2
//
// TODO: if we extend TensorIterator to accept 3 inputs,
// we can probably make this a bit more performant.
Tensor do_trapezoid(const Tensor& y, const Tensor& dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);
    // If the dimensions of 'dx' and '(left + right)' do not match
    // broadcasting is attempted here.
    return ((left + right) * dx).sum(dim) / 2.;
}

// When dx is constant, the above formula simplifies
// to dx * [(\sum_{i=1}^n y_i) - (y_1 + y_n)/2]
Tensor do_trapezoid(const Tensor& y, double dx, int64_t dim) {
    return (y.sum(dim) - (y.select(dim, 0) + y.select(dim, -1)) * 0.5) * dx;
}

Tensor zeros_like_except(const Tensor& y, int64_t dim) {
    auto sizes = y.sym_sizes().vec();
    dim = maybe_wrap_dim(dim, y.dim());
    sizes.erase(sizes.begin() + dim);
    return at::zeros_symint(sizes, y.options());
}

Tensor do_cumulative_trapezoid(const Tensor& y, const Tensor& dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);

    return ((left + right) * dx).cumsum(dim) / 2.;
}

Tensor do_cumulative_trapezoid(const Tensor& y, double dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);

    return (dx /2. * (left + right)).cumsum(dim);
}
// Given the current shape of a Tensor and a target number of dimensions,
// returns a new shape with the same values as the original shape,
// but with '1's padded in the beginning to match the target number of dimensions.
// For example, curr_shape = (5,5,5) and target_n_dim = 6 ==> (1,1,1,5,5,5)
// Note that no padding will be added if the current shape has the greater than or equal
// number of dimensions than the target numbers of dimensions.
SymDimVector add_padding_to_shape(SymIntArrayRef curr_shape, int64_t target_n_dim) {
    const auto curr_size = static_cast<int64_t>(curr_shape.size());
    if (curr_size >= target_n_dim){
        target_n_dim = curr_size;
    }
    SymDimVector new_shape(target_n_dim, 1);
    for (const auto i : c10::irange(curr_size)) {
        new_shape[target_n_dim-i-1] = curr_shape[curr_size-i-1];
    }
    return new_shape;
}
}

Tensor trapezoid(const Tensor& y, const Tensor& x, int64_t dim) {
    dim = maybe_wrap_dim(dim, y);
    // asking for the integral with zero samples is a bit nonsensical,
    // but we'll return "0" to match numpy behavior.
    if (y.sym_size(dim) == 0) {
        return zeros_like_except(y, dim);
    }
    TORCH_CHECK(y.scalar_type() != kBool && x.scalar_type() != kBool, "trapezoid: received a bool input for `x` or `y`, but bool is not supported")
    Tensor x_viewed;
    // Note that we explicitly choose not to broadcast 'x' to match the shape of 'y' here because
    // we want to follow NumPy's behavior of broadcasting 'dx' and 'dy' together after the differences are taken.
    if (x.dim() == 1) {
        // This step takes 'x' with dimension (n,), and returns 'x_view' with
        // dimension (1,1,...,n,...,1,1) based on dim and y.dim() so that, later on, 'dx'
        // can be broadcast to match 'dy' at the correct dimensions.
        TORCH_CHECK(x.sym_size(0) == y.sym_size(dim), "trapezoid: There must be one `x` value for each sample point");
        SymDimVector new_sizes(y.dim(), 1); // shape = [1] * y.
        new_sizes[dim] = x.sym_size(0); // shape[axis] = d.shape[0]
        x_viewed = x.view_symint(new_sizes);
    } else if (x.dim() < y.dim()) {
        // When 'y' has more dimension than 'x', this step takes 'x' with dimension (n_1, n_2, ...),
        // and add '1's as dimensions in front to become (1, 1, ..., n_1, n_2), matching the dimension of 'y'.
        // This allows the subsequent slicing operations to proceed with any 'dim' without going out of bound.
        SymDimVector new_sizes = add_padding_to_shape(x.sym_sizes(), y.dim());
        x_viewed = x.view_symint(new_sizes);
    } else {
        x_viewed = x;
    }
    // Note the .slice operation reduces the dimension along 'dim' by 1,
    // while the sizes of other dimensions are untouched.
    Tensor x_left = x_viewed.slice(dim, 0, -1);
    Tensor x_right = x_viewed.slice(dim, 1);

    Tensor dx = x_right - x_left;
    return do_trapezoid(y, dx, dim);
}

Tensor trapezoid(const Tensor& y, const Scalar& dx, int64_t dim) {
    // see above
    if (y.sym_size(dim) == 0) {
        return zeros_like_except(y, dim);
    }
    TORCH_CHECK(y.scalar_type() != kBool, "trapezoid: received a bool input for `y`, but bool is not supported")
    TORCH_CHECK(!(dx.isComplex() ||  dx.isBoolean()), "trapezoid: Currently, we only support dx as a real number.");
    return do_trapezoid(y, dx.toDouble(), dim);
}

Tensor trapz(const Tensor& y, const Tensor& x, int64_t dim) {
    return at::native::trapezoid(y, x, dim);
}

Tensor trapz(const Tensor& y, double dx, int64_t dim) {
    return at::native::trapezoid(y, dx, dim);
}

Tensor cumulative_trapezoid(const Tensor& y, const Tensor& x, int64_t dim) {
    dim = maybe_wrap_dim(dim, y);
    TORCH_CHECK(y.scalar_type() != kBool && x.scalar_type() != kBool, "cumulative_trapezoid: received a bool input for `x` or `y`, but bool is not supported")
    Tensor x_viewed;
    if (x.dim() == 1) {
        // See trapezoid for implementation notes
        TORCH_CHECK(x.sym_size(0) == y.sym_size(dim), "cumulative_trapezoid: There must be one `x` value for each sample point");
        SymDimVector new_sizes(y.dim(), 1); // shape = [1] * y.
        new_sizes[dim] = x.sym_size(0); // shape[axis] = d.shape[0]
        x_viewed = x.view_symint(new_sizes);
    } else if (x.dim() < y.dim()) {
        // See trapezoid for implementation notes
        SymDimVector new_sizes = add_padding_to_shape(x.sym_sizes(), y.dim());
        x_viewed = x.view_symint(new_sizes);
    } else {
        x_viewed = x;
    }
    Tensor x_left = x_viewed.slice(dim, 0, -1);
    Tensor x_right = x_viewed.slice(dim, 1);
    Tensor dx = x_right - x_left;

    return do_cumulative_trapezoid(y, dx, dim);
}

Tensor cumulative_trapezoid(const Tensor& y, const Scalar& dx, int64_t dim) {
    TORCH_CHECK(y.scalar_type() != kBool, "cumulative_trapezoid: received a bool input for `y`, but bool is not supported")
    TORCH_CHECK(!(dx.isComplex() || dx.isBoolean()), "cumulative_trapezoid: Currently, we only support dx as a real number.");

    return do_cumulative_trapezoid(y, dx.toDouble(), dim);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 30 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/core/DimVector.h`
- `ATen/TensorOperators.h`
- `ATen/WrapDimUtils.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`
- `c10/core/ScalarType.h`
- `c10/core/Scalar.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/cumulative_trapezoid_native.h`
- `ATen/ops/trapezoid_native.h`
- `ATen/ops/trapz_native.h`
- `ATen/ops/zeros.h`


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

- **File Documentation**: `Integration.cpp_docs.md`
- **Keyword Index**: `Integration.cpp_kw.md`
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

- **File Documentation**: `Integration.cpp_docs.md_docs.md`
- **Keyword Index**: `Integration.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
