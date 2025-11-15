# Documentation: `docs/aten/src/ATen/native/Histogram.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Histogram.cpp_docs.md`
- **Size**: 23,961 bytes (23.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Histogram.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Histogram.cpp`
- **Size**: 20,875 bytes (20.39 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

#include <ATen/native/Histogram.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_histogramdd_bin_edges.h>
#include <ATen/ops/_histogramdd_bin_edges_native.h>
#include <ATen/ops/_histogramdd_from_bin_cts.h>
#include <ATen/ops/_histogramdd_from_bin_cts_native.h>
#include <ATen/ops/_histogramdd_from_bin_tensors.h>
#include <ATen/ops/_histogramdd_from_bin_tensors_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/histc_native.h>
#include <ATen/ops/histogram_native.h>
#include <ATen/ops/histogramdd_native.h>
#include <ATen/ops/linspace.h>
#endif

#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>
#include <functional>
#include <c10/util/ArrayRef.h>
#include <c10/core/ScalarType.h>
#include <c10/core/DefaultDtype.h>
#include <c10/util/irange.h>

/* Implements a numpy-like histogramdd function running on cpu
 * https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
 *
 * See the docstr for torch.histogramdd in torch/functional.py for further explanation.
 *
 * - torch.histogramdd(input, bins, range=None, weight=None, density=False)
 *   input     - tensor with shape (M, N). input is interpreted as M coordinates in N-dimensional space.
 *               If a tensor with more than 2 dimensions is passed, all but the last dimension will be flattened.
 *   bins      - int[] of length N or tensor list of length N. If int[], defines the number of equal-width bins
 *               in each dimension. If tensor list, defines the sequences of bin edges, including rightmost edges,
 *               for each dimension.
 *   range     - float[] of length 2 * N, optional. If specified, defines the leftmost and rightmost bin edges
 *               for each dimension.
 *   weight    - tensor, optional. If provided, weight should have the same shape as input excluding its last dimension.
 *               Each N-dimensional value in input contributes its associated weight towards its bin's result.
 *               If weight is not specified, each value has weight 1 by default.
 *   density   - bool, optional. If false (default), the result will contain the total count (weight) in each bin.
 *               If True, each count (weight) is divided by the total count (total weight), then divided by the
 *               volume of its associated bin.
 *
 * Returns:
 *   hist      - N-dimensional tensor containing the values of the histogram.
 *   bin_edges - tensor list of length N containing the edges of the histogram bins in each dimension.
 *               Bins include their left edge and exclude their right edge, with the exception of the
 *               rightmost bin in each dimension which includes both of its edges.
 *
 * Restrictions are defined in histogram_check_inputs() and in select_outer_bin_edges().
 */

namespace at::native {

DEFINE_DISPATCH(histogramdd_stub);
DEFINE_DISPATCH(histogramdd_linear_stub);
DEFINE_DISPATCH(histogram_select_outer_bin_edges_stub);

namespace {

/* Checks properties of input tensors input, bins, and weight.
 */
void histogramdd_check_inputs(const Tensor& input, const TensorList& bins, const std::optional<Tensor>& weight) {
    TORCH_CHECK(input.dim() >= 2, "torch.histogramdd: input tensor should have at least 2 dimensions, but got ",
                input.dim());

    const int64_t N = input.size(-1);

    TORCH_CHECK(static_cast<int64_t>(bins.size()) == N, "torch.histogramdd: expected ", N, " sequences of bin edges for a ", N,
                "-dimensional histogram but got ", bins.size());

    auto input_dtype = input.dtype();
    for (const auto dim : c10::irange(N)) {
        const Tensor& dim_bins = bins[dim];

        auto bins_dtype = dim_bins.dtype();
        TORCH_CHECK(input_dtype == bins_dtype, "torch.histogramdd: input tensor and bins tensors should",
                " have the same dtype, but got input with dtype ", input_dtype,
                " and bins for dimension ", dim, " with dtype ", bins_dtype);

        const int64_t dim_bins_dim = dim_bins.dim();
        TORCH_CHECK(dim_bins_dim == 1, "torch.histogramdd: bins tensor should have one dimension,",
                " but got ", dim_bins_dim, " dimensions in the bins tensor for dimension ", dim);

        const int64_t numel = dim_bins.numel();
        TORCH_CHECK(numel > 0, "torch.histogramdd: bins tensor should have at least 1 element,",
                " but got ", numel, " elements in the bins tensor for dimension ", dim);
    }

    if (weight.has_value()) {
        TORCH_CHECK(input.dtype() == weight.value().dtype(), "torch.histogramdd: if weight tensor is provided,"
                " input tensor and weight tensor should have the same dtype, but got input(", input.dtype(), ")",
                ", and weight(", weight.value().dtype(), ")");

        /* If a weight tensor is provided, we expect its shape to match that of
         * the input tensor excluding its innermost dimension N.
         */
        auto input_sizes = input.sizes().vec();
        input_sizes.pop_back();

        auto weight_sizes = weight.value().sizes().vec();
        if (weight_sizes.empty()) {
            // correctly handle scalars
            weight_sizes = {1};
        }

        TORCH_CHECK(input_sizes == weight_sizes, "torch.histogramdd: if weight tensor is provided it should have"
                " the same shape as the input tensor excluding its innermost dimension, but got input with shape ",
                input.sizes(), " and weight with shape ", weight.value().sizes());
    }
}

/* Checks properties of output tensors hist and bin_edges, then resizes them.
 */
void histogramdd_prepare_out(const Tensor& input, const std::vector<int64_t>& bin_ct,
        const Tensor& hist, const TensorList& bin_edges) {
    const int64_t N = input.size(-1);

    TORCH_INTERNAL_ASSERT((int64_t)bin_ct.size() == N);
    TORCH_INTERNAL_ASSERT((int64_t)bin_edges.size() == N);

    TORCH_CHECK(input.dtype() == hist.dtype(), "torch.histogram: input tensor and hist tensor should",
            " have the same dtype, but got input ", input.dtype(), " and hist ", hist.dtype());

    for (const auto dim : c10::irange(N)) {
        TORCH_CHECK(input.dtype() == bin_edges[dim].dtype(), "torch.histogram: input tensor and bin_edges tensor should",
                " have the same dtype, but got input ", input.dtype(), " and bin_edges ", bin_edges[dim].dtype(),
                " for dimension ", dim);

        TORCH_CHECK(bin_ct[dim] > 0,
                "torch.histogram(): bins must be > 0, but got ", bin_ct[dim], " for dimension ", dim);

        at::native::resize_output(bin_edges[dim], bin_ct[dim] + 1);
    }

    at::native::resize_output(hist, bin_ct);
}

void histogramdd_prepare_out(const Tensor& input, TensorList bins,
        const Tensor& hist, const TensorList& bin_edges) {
    std::vector<int64_t> bin_ct(bins.size());
    std::transform(bins.begin(), bins.end(), bin_ct.begin(), [](Tensor t) { return t.numel() - 1; });
    histogramdd_prepare_out(input, bin_ct, hist, bin_edges);
}

/* Determines the outermost bin edges. For simplicity when calling into aminmax,
 * assumes that input has already been reshaped to (M, N).
 */
std::pair<std::vector<double>, std::vector<double>>
select_outer_bin_edges(const Tensor& input, std::optional<c10::ArrayRef<double>> range) {
    TORCH_INTERNAL_ASSERT(input.dim() == 2, "expected input to have shape (M, N)");
    const int64_t N = input.size(-1);

    // Default ranges for empty input matching numpy.histogram's default
    std::vector<double> leftmost_edges(N, 0.);
    std::vector<double> rightmost_edges(N, 1.);

    if (range.has_value()) {
        // range is specified
        TORCH_CHECK((int64_t)range.value().size() == 2 * N, "torch.histogramdd: for a ", N, "-dimensional histogram",
                " range should have ", 2 * N, " elements, but got ", range.value().size());

        for (const auto dim : c10::irange(N)) {
            leftmost_edges[dim] = range.value()[2 * dim];
            rightmost_edges[dim] = range.value()[2 * dim + 1];
        }
    } else if (input.numel() > 0) {
        // non-empty input

        histogram_select_outer_bin_edges_stub(input.device().type(), input, N, leftmost_edges, rightmost_edges);
    }

    for (const auto dim : c10::irange(N)) {
        double leftmost_edge = leftmost_edges[dim];
        double rightmost_edge = rightmost_edges[dim];

        TORCH_CHECK(std::isfinite(leftmost_edge) && std::isfinite(rightmost_edge),
                "torch.histogramdd: dimension ", dim, "'s range [",
                leftmost_edge, ", ", rightmost_edge, "] is not finite");

        TORCH_CHECK(leftmost_edge <= rightmost_edge, "torch.histogramdd: min should not exceed max, but got",
                " min ", leftmost_edge, " max ", rightmost_edge, " for dimension ", dim);

        // Expand empty range to match numpy behavior and avoid division by 0 in normalization
        if (leftmost_edge == rightmost_edge) {
            leftmost_edges[dim] -= 0.5;
            rightmost_edges[dim] += 0.5;
        }
    }

    return std::make_pair(leftmost_edges, rightmost_edges);
}


/* Bin edges correction based on the precision representation.
 * To maintain the backward compatibility we take max(std::nextafter<>, +1)
 * and min(std::nextafter<>, -1) for scalar types. For other types +/- 1 as usual.
 */
void bins_edges_correction(const ScalarType& t, double &leftmost_edge, double &rightmost_edge)
{
#define UPDATE_WITH_LIMIT(real_type, scalartype) \
  case ScalarType::scalartype:                   \
    leftmost_edge = std::min(                    \
        static_cast<double>(                     \
            std::nexttoward(                     \
                static_cast<real_type>(leftmost_edge),   \
                std::numeric_limits<real_type>::lowest() \
            )                                    \
        ),                                       \
        leftmost_edge - 1.                       \
    );                                           \
    rightmost_edge = std::max(                   \
        static_cast<double>(                     \
            std::nexttoward(                     \
                static_cast<real_type>(rightmost_edge), \
                std::numeric_limits<real_type>::max()   \
            )                                    \
        ),                                       \
        rightmost_edge + 1.                      \
    );                                           \
    break;

    switch (t) {
        UPDATE_WITH_LIMIT(double, Double)
        UPDATE_WITH_LIMIT(float, Float)
        default:
            // Fallback to the default behavior for other types
            leftmost_edge -= 1;
            rightmost_edge += 1;
    }
#undef UPDATE_WITH_LIMIT
}

/* histc's version of the logic for outermost bin edges.
 */
std::pair<double, double> histc_select_outer_bin_edges(const Tensor& input,
        const Scalar& min, const Scalar& max) {
    double leftmost_edge = min.to<double>();
    double rightmost_edge = max.to<double>();

    if (leftmost_edge == rightmost_edge && input.numel() > 0) {
        auto extrema = aminmax(input);
        leftmost_edge = std::get<0>(extrema).item<double>();
        rightmost_edge = std::get<1>(extrema).item<double>();
    }

    if (leftmost_edge == rightmost_edge) {
        bins_edges_correction(input.dtype().toScalarType(), leftmost_edge, rightmost_edge);
    }

    TORCH_CHECK(!(std::isinf(leftmost_edge) || std::isinf(rightmost_edge) ||
            std::isnan(leftmost_edge) || std::isnan(rightmost_edge)),
            "torch.histc: range of [", leftmost_edge, ", ", rightmost_edge, "] is not finite");

    TORCH_CHECK(leftmost_edge < rightmost_edge, "torch.histc: max must be larger than min");

    return std::make_pair(leftmost_edge, rightmost_edge);
}

} // namespace

static std::vector<Tensor> allocate_bin_edges_tensors(const Tensor& self) {
    TORCH_CHECK(self.dim() >= 2, "torch.histogramdd: input tensor should have at least 2 dimensions");
    const int64_t N = self.size(-1);
    std::vector<Tensor> bin_edges_out(N);
    for (const auto dim : c10::irange(N)) {
        bin_edges_out[dim] = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    }
    return bin_edges_out;
}

/* Versions of histogramdd in which bins is a Tensor[] defining the sequences of bin edges.
 */
static Tensor& histogramdd_out(const Tensor& self, TensorList bins,
        const std::optional<Tensor>& weight, bool density,
        Tensor& hist, TensorList& bin_edges) {
    histogramdd_check_inputs(self, bins, weight);
    histogramdd_prepare_out(self, bins, hist, bin_edges);

    for (const auto dim : c10::irange(bins.size())) {
        bin_edges[dim].copy_(bins[dim]);
    }

    histogramdd_stub(self.device().type(), self, weight, density, hist, bin_edges);
    return hist;
}

Tensor _histogramdd(const Tensor& self, TensorList bins,
        const std::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    TensorList bin_edges_out_tl(bin_edges_out);

    histogramdd_out(self, bins, weight, density, hist, bin_edges_out_tl);
    return hist;
}

/* Versions of histogramdd in which bins is an int[]
 * defining the number of bins in each dimension.
 */
static std::vector<Tensor>& histogramdd_bin_edges_out(const Tensor& self, IntArrayRef bin_ct,
        std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density,
        std::vector<Tensor>& bin_edges_out) {
    TensorList bin_edges_out_tl(bin_edges_out);

    const int64_t N = self.size(-1);
    const int64_t M = std::accumulate(self.sizes().begin(), self.sizes().end() - 1,
            static_cast<int64_t>(1), std::multiplies<int64_t>());
    Tensor reshaped_self = self.reshape({ M, N });

    auto outer_bin_edges = select_outer_bin_edges(reshaped_self, range);

    const int64_t bin_size = bin_ct.size();
    TORCH_CHECK(
        N == bin_size,
        "histogramdd: The size of bins must be equal to the innermost dimension of the input.");
    for (const auto dim : c10::irange(N)) {
        at::linspace_out(bin_edges_out[dim], outer_bin_edges.first[dim], outer_bin_edges.second[dim],
                bin_ct[dim] + 1);
    }

    return bin_edges_out;
}

std::vector<Tensor> histogramdd_bin_edges(const Tensor& self, IntArrayRef bin_ct,
        std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    return histogramdd_bin_edges_out(self, bin_ct, range, weight, density, bin_edges_out);
}

static Tensor& histogramdd_out(const Tensor& self, IntArrayRef bin_ct,
        std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density,
        Tensor& hist, TensorList& bin_edges) {
    std::vector<Tensor> bins = histogramdd_bin_edges(self, bin_ct, range, weight, density);

    histogramdd_check_inputs(self, bins, weight);
    histogramdd_prepare_out(self, bins, hist, bin_edges);

    for (const auto dim : c10::irange(bins.size())) {
        bin_edges[dim].copy_(bins[dim]);
    }

    histogramdd_linear_stub(self.device().type(), self, weight, density, hist, bin_edges, true);
    return hist;
}

Tensor _histogramdd(const Tensor& self, IntArrayRef bin_ct,
        std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    TensorList bin_edges_out_tl(bin_edges_out);

    histogramdd_out(self, bin_ct, range, weight, density, hist, bin_edges_out_tl);
    return hist;
}

/* Versions of histogram in which bins is a Tensor defining the sequence of bin edges.
 */
std::tuple<Tensor&, Tensor&>
histogram_out(const Tensor& self, const Tensor& bins,
        const std::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    Tensor reshaped_self = self.reshape({ self.numel(), 1 });
    std::optional<Tensor> reshaped_weight = weight.has_value()
        ? weight.value().reshape({ weight.value().numel() }) : weight;
    TensorList bins_in = bins;
    TensorList bins_out = bin_edges;

    histogramdd_out(reshaped_self, bins_in, reshaped_weight, density, hist, bins_out);

    return std::forward_as_tuple(hist, bin_edges);
}

std::tuple<Tensor, Tensor>
histogram(const Tensor& self, const Tensor& bins,
        const std::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges = at::empty({0}, bins.options(), MemoryFormat::Contiguous);
    return histogram_out(self, bins, weight, density, hist, bin_edges);
}

/* Versions of histogram in which bins is an integer specifying the number of equal-width bins.
 */
std::tuple<Tensor&, Tensor&>
histogram_out(const Tensor& self, int64_t bin_ct, std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    Tensor reshaped_self = self.reshape({ self.numel(), 1 });
    std::optional<Tensor> reshaped_weight = weight.has_value()
        ? weight.value().reshape({ weight.value().numel() }) : weight;
    TensorList bins_in = bin_edges;
    TensorList bins_out = bin_edges;

    histogramdd_prepare_out(reshaped_self, std::vector<int64_t>{bin_ct}, hist, bins_out);
    auto outer_bin_edges = select_outer_bin_edges(reshaped_self, range);
    at::linspace_out(bin_edges, outer_bin_edges.first[0], outer_bin_edges.second[0], bin_ct + 1);

    histogramdd_check_inputs(reshaped_self, bins_in, reshaped_weight);

    histogramdd_linear_stub(reshaped_self.device().type(), reshaped_self, reshaped_weight, density, hist, bin_edges, true);
    return std::forward_as_tuple(hist, bin_edges);
}

std::tuple<Tensor, Tensor>
histogram(const Tensor& self, int64_t bin_ct, std::optional<c10::ArrayRef<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges_out = at::empty({0}, self.options());
    return histogram_out(self, bin_ct, range, weight, density, hist, bin_edges_out);
}

/* Narrowed interface for the legacy torch.histc function.
 */
Tensor& histogram_histc_out(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max, Tensor& hist) {
    Tensor bin_edges = at::empty({0}, self.options());

    Tensor reshaped = self.reshape({ self.numel(), 1 });
    TensorList bins_in = bin_edges;
    TensorList bins_out = bin_edges;

    histogramdd_prepare_out(reshaped, std::vector<int64_t>{bin_ct}, hist, bins_out);

    auto outer_bin_edges = histc_select_outer_bin_edges(self, min, max);
    at::linspace_out(bin_edges, outer_bin_edges.first, outer_bin_edges.second, bin_ct + 1);

    histogramdd_check_inputs(reshaped, bins_in, {});

    histogramdd_linear_stub(reshaped.device().type(), reshaped,
            std::optional<Tensor>(), false, hist, bin_edges, false);
    return hist;
}

Tensor histogram_histc(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    return histogram_histc_out(self, bin_ct, min, max, hist);
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd(
    const Tensor &self, TensorList bins, std::optional<ArrayRef<double>> /*range*/,
    const std::optional<Tensor> &weight, bool density) {
  auto hist = at::_histogramdd_from_bin_tensors(self, bins, weight, density);
  return std::tuple<Tensor, std::vector<Tensor>>{
      std::move(hist), bins.vec()};
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd(
    const Tensor &self, IntArrayRef bins, std::optional<ArrayRef<double>> range,
    const std::optional<Tensor> &weight, bool density) {
  auto bin_edges = at::_histogramdd_bin_edges(self, bins, range, weight, density);
  auto hist = at::_histogramdd_from_bin_cts(self, bins, range, weight, density);
  return std::tuple<Tensor, std::vector<Tensor>>{
      std::move(hist), std::move(bin_edges)};
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd(
    const Tensor &self, int64_t bins, std::optional<ArrayRef<double>> range,
    const std::optional<Tensor> &weight, bool density) {
  DimVector bins_v(self.size(-1), bins);
  return at::native::histogramdd(self, bins_v, range, weight, density);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 33 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `static`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/native/Histogram.h`
- `ATen/native/Resize.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_histogramdd_bin_edges.h`
- `ATen/ops/_histogramdd_bin_edges_native.h`
- `ATen/ops/_histogramdd_from_bin_cts.h`
- `ATen/ops/_histogramdd_from_bin_cts_native.h`
- `ATen/ops/_histogramdd_from_bin_tensors.h`
- `ATen/ops/_histogramdd_from_bin_tensors_native.h`
- `ATen/ops/aminmax.h`
- `ATen/ops/empty.h`
- `ATen/ops/histc_native.h`
- `ATen/ops/histogram_native.h`
- `ATen/ops/histogramdd_native.h`
- `ATen/ops/linspace.h`
- `cmath`
- `numeric`
- `tuple`
- `vector`
- `functional`
- `c10/util/ArrayRef.h`
- `c10/core/ScalarType.h`
- `c10/core/DefaultDtype.h`
- `c10/util/irange.h`


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

- **File Documentation**: `Histogram.cpp_docs.md`
- **Keyword Index**: `Histogram.cpp_kw.md`
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

- **File Documentation**: `Histogram.cpp_docs.md_docs.md`
- **Keyword Index**: `Histogram.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
