# Documentation: `docs/aten/src/ATen/core/jit_type.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/jit_type.h_docs.md`
- **Size**: 53,030 bytes (51.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/jit_type.h`

## File Metadata

- **Path**: `aten/src/ATen/core/jit_type.h`
- **Size**: 72,153 bytes (70.46 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/custom_class.h>
#include <ATen/core/jit_type_base.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/functional.h>
#include <ATen/core/symbol.h>
#include <ATen/core/type_factory.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/TypeList.h>
#include <c10/util/Exception.h>
#include <optional>
#include <c10/core/SymFloat.h>
#include <c10/core/SymBool.h>
#include <c10/core/Device.h>

#include <array>
#include <memory>
#include <ostream>
#include <sstream>
#include <utility>


namespace torch::jit {
struct Function;
} // namespace torch::jit


namespace c10 {

template<class Key, class Value>
class Dict;
struct IValue;
struct FunctionSchema;
struct NamedType;
using OptNameList = std::optional<std::vector<std::string>>;

void standardizeVectorForUnion(std::vector<TypePtr>& reference, std::vector<TypePtr>* to_fill);
void standardizeVectorForUnion(std::vector<TypePtr>* to_flatten);

inline bool is_contiguous_strides(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  size_t n_dim = sizes.size();
  if (n_dim == 0) {
    return true;
  }

  if (strides[n_dim - 1] != 1) {
    return false;
  }

  for (int i = static_cast<int>(n_dim) - 2; i >= 0; i--) {
    if (strides[i] != strides[i + 1] * sizes[i + 1]) {
      return false;
    }
  }
  return true;
}

struct AnyType;
using AnyTypePtr = SingletonTypePtr<AnyType>;
// Any is the top of the type hierarchy, all other types are subtypes
// T <: Any, forall T
struct TORCH_API AnyType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Any";
  }
  static const TypeKind Kind = TypeKind::AnyType;
  // global singleton
  static AnyTypePtr get();

 private:
  AnyType() : Type(TypeKind::AnyType) {}
};

inline std::string toString(const Type& type) {
  return type.str();
}

// Shim for compatibility with code that uses TypePtr.
inline std::string toString(const TypePtr& typePtr) {
  return toString(*typePtr);
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
  return !(lhs == rhs);
}

// common base for all types that have a single sub element
// e.g. Future[T], Optional[T], List[T]
template <TypeKind K, typename T>
struct SingleElementType : public SharedType {
  static const TypeKind Kind = K;

  const TypePtr& getElementType() const {
    return elem;
  }

  bool hasFreeVariables() const override {
    return getElementType()->hasFreeVariables();
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return elem;
  }

  bool equals(const Type& rhs) const override {
    if (auto rhs_ = rhs.cast<T>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }

 protected:
  SingleElementType(TypePtr elem) : SharedType(Kind), elem(std::move(elem)) {
    TORCH_CHECK(this->elem, c10::str(
            "Can not create ", typeKindToString(Kind), " with None type"));
  }

 private:
  TypePtr elem;
};

struct UnionType;
using UnionTypePtr = std::shared_ptr<UnionType>;
struct TORCH_API UnionType : public SharedType {
  friend struct Type;

  static const TypeKind Kind = TypeKind::UnionType;

  bool isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const override;

  std::string str() const override;

  static UnionTypePtr create(std::vector<TypePtr> reference);

  bool equals(const Type& rhs) const override;

  bool isUnionType() const override {
    return true;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return types_;
  }

  // For testing purposes only
  at::ArrayRef<TypePtr> getTypes() const {
    return types_;
  }

  TypePtr createWithContained(std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types));
  }

  bool canHoldType(const Type& type) const;

  bool hasFreeVariables() const override {
    return has_free_variables_;
  }

  std::optional<TypePtr> toOptional() const;

  std::optional<TypePtr> subtractTypeSet(std::vector<TypePtr>& to_subtract) const;

 protected:
    explicit UnionType(std::vector<TypePtr> types, TypeKind kind=TypeKind::UnionType);
    std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override;
    std::string unionStr(
        const TypePrinter& printer = nullptr,
        bool is_annotation_str = false) const;
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    bool has_free_variables_;
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::vector<TypePtr> types_;
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    bool can_hold_none_;

};

struct OptionalType;
using OptionalTypePtr = std::shared_ptr<OptionalType>;
// This type represents an optional type. There is one `Optional` for
// each element type. `Optional[T]` can accept both `T` and
// `None`(`std::nullopt` in C++)
// Subtype hierarchy for Optional:
//     - Optional[T] <: Optional[R] iff T <: R
//     - T <: Optional[R] if T <: R
//     - None <: Optional[T] for all T
//     - Optional[T] == Union[T, None] for all T
struct TORCH_API OptionalType : public UnionType {
  static OptionalTypePtr create(const TypePtr& contained);

  static const TypeKind Kind = TypeKind::OptionalType;

  friend struct Type;

  bool equals(const Type& rhs) const override;

  const TypePtr& getElementType() const {
    return contained_;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return contained_;
  }

  std::string str() const override {
    std::stringstream ss;
    ss << getElementType()->str() << "?";
    return ss.str();
  }

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    AT_ASSERT(contained_types.size() == 1);
    return create(contained_types[0]);
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  bool isUnionType() const override {
    return true;
  }

  // common cast Optional[Tensor] for undefined tensor type
  static TypePtr ofTensor();
  //
  // global singleton
  static TypePtr get(TypePtr inner);

 private:
  explicit OptionalType(const TypePtr& contained);

  TypePtr contained_;

  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    std::stringstream ss;
    ss << "Optional[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

template <typename T>
inline std::optional<T> merge_primitive(
    const std::optional<T>& a,
    const std::optional<T>& b) {
  if (a.has_value() && b.has_value() && a.value() == b.value()) {
    return a;
  }
  return std::optional<T>{};
}

// If we see `a + b + c`  and know that a, b, and c are the same size and have
// two dimensions (WxH), then we can generate a fused kernel for them. That
// fused kernel would likely have indexing math to handling both the W and H
// dimensions. However, if we knew the WxH dimensions were contiguous, we can
// pretend like we only have a single dimension, simplifying the indexing logic.
// This can be performed even if the dimensions are transposed,
// as long as a, b, and c are transposed in the same way.
// We'd like to have the compiler be able to do this dimensionality reduction,
// but simply knowing sizes is not enough.
// We can extend profiling to also record stride information.
// Rather than recording specific strides,
// we can simply order the strides from smallest to largest with
// `stride_indices` A contiguity marker on the smallest stride (c0) indicates
// the stride is precisely 1, otherwise a contiguity marker means that $stride_n
// = size_{n-1}*stride_{n-1}$
struct TORCH_API Stride {
  Stride() = default;
  Stride(
      const std::optional<size_t>& stride_index,
      std::optional<bool> contiguous,
      const std::optional<size_t>& stride)
      : stride_index_(stride_index), contiguous_(contiguous), stride_(stride) {}

  bool operator==(const Stride& b) const {
    return stride_index_ == b.stride_index_ && contiguous_ == b.contiguous_ &&
        stride_ == b.stride_;
  }

  bool isComplete() const {
    return stride_index_ && contiguous_ && stride_;
  }

  std::optional<size_t> stride_index_;
  std::optional<bool> contiguous_;
  std::optional<size_t> stride_;
};

template <>
inline std::optional<Stride> merge_primitive(
    const std::optional<Stride>& a,
    const std::optional<Stride>& b) {
  std::optional<Stride> left = a;
  std::optional<Stride> right = b;
  if (!left.has_value()) {
    left = {Stride()};
  }
  if (!right.has_value()) {
    right = {Stride()};
  }

  auto merged_index =
      merge_primitive(left->stride_index_, right->stride_index_);
  auto merged_cont = merge_primitive(left->contiguous_, right->contiguous_);
  auto merged_stride = merge_primitive(left->stride_, right->stride_);
  auto r = Stride(merged_index, merged_cont, merged_stride);
  // normalize
  if (!r.stride_index_.has_value() && !r.contiguous_.has_value() &&
      !r.stride_.has_value()) {
    return std::optional<Stride>{};
  }

  return r;
}

struct TORCH_API ShapeSymbol {
  // needed for use in `std::map`
  ShapeSymbol() : value_(-1) {}
  // is this symbol a fixed/static dimension
  bool is_static() const {
    return value_ >= 0;
  }
  bool operator==(const ShapeSymbol& b) const {
    return value_ == b.value_;
  }
  bool operator<(const ShapeSymbol& b) const {
    return value_ < b.value_;
  }

  static ShapeSymbol fromStaticSize(int64_t val) {
    return ShapeSymbol(val);
  }
  int64_t static_size() const {
    TORCH_CHECK(is_static());
    return value_;
  }

  int64_t value() const {
    return value_;
  }

  static ShapeSymbol newSymbol() {
    return fromStaticSize(-static_cast<int64_t>(++num_symbols));
  }
  friend TORCH_API std::ostream& operator<<(
      std::ostream& os,
      const ShapeSymbol& s);

 private:
  ShapeSymbol(int64_t val) : value_(val) {}
  int64_t value_;
  static std::atomic<size_t> num_symbols;
};

inline ShapeSymbol merge_primitive(
    const ShapeSymbol& a,
    const ShapeSymbol& b) {
  if (a.is_static() && b.is_static() && a == b) {
    return a;
  }
  return ShapeSymbol::newSymbol();
}

// Shape of a Tensor represented with ShapeSymbol's. Unranked, ranked unknown
// dims, partially known and fully known shapes are all supported.
struct TORCH_API SymbolicShape {
  // Unranked shape constructor.
  SymbolicShape() : dims_(std::nullopt) {}

  // Known rank but unknown dimensions.
  SymbolicShape(std::optional<size_t> rank) : dims_(std::nullopt) {
    if(!rank) {
      return;
    }

    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(*rank);
    for(size_t i = 0; i < *rank; ++i) {
      shape_symbols.push_back(ShapeSymbol::newSymbol());
    }
    dims_ = shape_symbols;
  }

  // Mix of known and unknown ranks
  SymbolicShape(const std::vector<std::optional<int64_t>>& dims) {
    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(dims.size());
    for(std::optional<int64_t> dim: dims) {
      if(!dim) {
        shape_symbols.push_back(ShapeSymbol::newSymbol());
      } else {
        shape_symbols.push_back(ShapeSymbol::fromStaticSize(*dim));
      }
    }
    dims_ = shape_symbols;
  }

  void dump() const;

  SymbolicShape(std::vector<ShapeSymbol> dims) : dims_(std::move(dims)) {}

  SymbolicShape(c10::IntArrayRef dims) {
    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(dims.size());
    for(int64_t dim : dims) {
      shape_symbols.push_back(ShapeSymbol::fromStaticSize(dim));
    }
    dims_ = shape_symbols;
  }

  ShapeSymbol operator[](size_t i) const {
    TORCH_CHECK(dims_, "Rank isn't fixed");
    return (*dims_).at(i);
  }

  ShapeSymbol at(size_t i) const {
    TORCH_CHECK(dims_, "Rank isn't fixed");
    return (*dims_).at(i);
  }

  // Returns rank or nullopt in case of unranked shape.
  std::optional<size_t> rank() const {
    if(!dims_) {
      return std::nullopt;
    }
    return dims_->size();
  }

  std::optional<std::vector<ShapeSymbol>> sizes() const {
    return dims_;
  }

  std::optional<std::vector<bool>> symbolicDims() const {
    if (!dims_) {
      return std::nullopt;
    }
    auto symbolic_dims = std::vector<bool>();
    for (const ShapeSymbol& s : *dims_) {
      symbolic_dims.push_back(!s.is_static());
    }
    return symbolic_dims;
  }

  // Checks whether the shape is fully defined/complete, ie. rank and sizes
  // of every dimension are known.
  bool isComplete() const {
    if(!dims_) {
      return false;
    }
    for(auto d : *dims_) {
      if(!d.is_static()) {
        return false;
      }
    }
    return true;
  }

  // Create new SymbolicShape that is result of merging self and another
  // SymbolicShape. Only dimensions that are static and equal will be
  // preserved.
  // If either of two shapes are of unknown rank or they have unmatching rank,
  // result will be unranked.
  SymbolicShape merge(const SymbolicShape& other) const;

  friend bool operator==(const SymbolicShape& lhs, const SymbolicShape& rhs) {
    return lhs.dims_ == rhs.dims_;
  }

  friend bool operator!=(const SymbolicShape& lhs, const SymbolicShape& rhs) {
    return !(lhs == rhs);
  }

  private:
    std::optional<std::vector<ShapeSymbol>> dims_;
};

namespace detail {
inline bool isComplete(const Stride& s) {
  return s.isComplete();
}

template<typename T>
inline bool isComplete(const T& /*t*/) {
  return true;
}
}

template <typename T>
struct VaryingShape {
  using ListOfOptionalElements = std::vector<std::optional<T>>;
  VaryingShape(const std::vector<T>& vec)
      : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

  VaryingShape(c10::ArrayRef<T> vec)
      : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

  VaryingShape(std::optional<size_t> size = std::nullopt) : dims_(std::nullopt) {
    if (size) {
      dims_ = ListOfOptionalElements(*size);
    }
  }

  VaryingShape(ListOfOptionalElements dims) : dims_(std::move(dims)) {}

  VaryingShape(size_t size) : VaryingShape(std::optional<size_t>(size)) {}

  bool operator==(const VaryingShape& other) const {
    return dims_ == other.dims_;
  }

  const std::optional<T> &operator[](size_t i) const {
    TORCH_CHECK(dims_, "Rank isn't fixed");
    return (*dims_).at(i);
  }

  std::optional<size_t> size() const {
    if (!dims_) {
      return std::nullopt;
    }
    const auto& dims = dims_.value();
    return dims.size();
  }

  const std::optional<ListOfOptionalElements>& sizes() const {
    return dims_;
  }

  TORCH_API VaryingShape merge(const VaryingShape& other) const;

  std::optional<std::vector<T>> concrete_sizes() const {
    if (!dims_) {
      return std::nullopt;
    }
    std::vector<T> sizes;
    sizes.reserve(dims_.value().size());
    for (auto d : *dims_) {
      if (!d) {
        return std::nullopt;
      }
      sizes.push_back(d.value());
    }
    return sizes;
  }

  bool isComplete() const {
    if (!dims_) {
      return false;
    }
    for (auto d : *dims_) {
      if (!d || !detail::isComplete(*d)) {
        return false;
      }
    }
    return true;
  }

 private:
  std::optional<ListOfOptionalElements> dims_;
};

struct TensorType;
// TODO: investigate making this SingletonOrSharedTypePtr<TensorType>
using TensorTypePtr = std::shared_ptr<TensorType>;
// This type represents a single Tensor with a specific size
struct TORCH_API TensorType : public SharedType {
  static TensorTypePtr create(const at::Tensor& t);

  // used by TensorType::create(size_t dim) which in turn used by
  // shape_analysis.cpp
  static TensorTypePtr create(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      const VaryingShape<int64_t>& sizes,
      const VaryingShape<int64_t>& strides,
      std::optional<bool> requires_grad,
      std::optional<bool> undefined = false,
      bool tensor_contiguity = false);

  static TensorTypePtr create(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      SymbolicShape sizes,
      VaryingShape<Stride> stride_,
      std::optional<bool> requires_grad,
      std::optional<bool> undefined = false);

  static TensorTypePtr create(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      std::optional<size_t> dim,
      std::optional<bool> requires_grad);

  // overloaded create variadic template argument as it could not distinguish
  // initializer list
  static TensorTypePtr createContiguous(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes);

  static TypePtr fromNumberType(const Type& typ);
  static TypePtr fromBoolType();

  std::optional<size_t> dim() const {
    return sizes().size();
  }

  VaryingShape<int64_t> sizes() const;

  VaryingShape<int64_t> strides() const;

  const VaryingShape<Stride>& stride_properties() const {
    return strides_;
  }

  const std::optional<at::Device>& device() const {
    return device_;
  }
  const std::optional<at::ScalarType>& scalarType() const {
    return scalar_type_;
  }
  const std::optional<bool>& requiresGrad() const {
    return requires_grad_;
  }
  bool requires_grad() const override {
    return requires_grad_ ? *requires_grad_ : true;
  }

  bool equals(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  std::string str() const override;

  std::string repr_str() const override {
    if (isInferredType()) {
      return str() + " (inferred)";
    } else {
      return str();
    }
  }

  std::optional<size_t> numel() const {
    size_t prod = 1;
    const auto& shape = sizes();

    for (size_t i = 0; i < shape.size(); i++) {
      auto const &s = shape[i];
      if (!s.has_value()) {
        return std::optional<size_t>{};
      }
      prod *= s.value();
    }
    return prod;
  }

  TensorTypePtr withRequiresGrad(std::optional<bool> s) {
    auto copy = clone();
    copy->requires_grad_ = s;
    return copy;
  }

  TensorTypePtr withScalarType(std::optional<ScalarType> st) {
    auto copy = clone();
    copy->scalar_type_ = st;
    return copy;
  }

  TensorTypePtr withDim(std::optional<size_t> d) {
    auto copy = clone();
    // withDim is only used by the legacy executor
    // that only cares about the rank, so create dummy symbols)) :
    copy->sizes_ = SymbolicShape(d);
    copy->strides_ = VaryingShape<Stride>(d);
    return copy;
  }

  TensorTypePtr withStrides(VaryingShape<Stride> sstrides) const {
    auto cloned = clone();
    cloned->strides_ = std::move(sstrides);
    return cloned;
  }

  TensorTypePtr withSizesStrides(
      at::IntArrayRef sizes,
      at::IntArrayRef strides) const {
    auto cloned = clone();
    auto ssizes = SymbolicShape(sizes);
    cloned->sizes_ = ssizes;
    cloned->strides_ = computeStrideProps(sizes, strides);
    return cloned;
  }

  TensorTypePtr withSymbolicShapes(SymbolicShape ssizes) const {
    auto cloned = clone();
    cloned->sizes_ = std::move(ssizes);
    return cloned;
  }

  TensorTypePtr withSizes(at::IntArrayRef sizes) const {
    return withSizesStrides(
        sizes, contiguousStridesOf(sizes));
  }

  TensorTypePtr withDevice(const std::optional<at::Device> device) const {
    auto copy = clone();
    copy->device_ = device;
    return copy;
  }

  TensorTypePtr dimensionedOnly() const {
    auto copy = clone();
    copy->sizes_ = SymbolicShape(sizes().size());
    copy->strides_ = VaryingShape<Stride>(sizes().size());
    return copy;
  }

  TensorTypePtr contiguous() const {
    auto cloned = clone();
    auto concrete_sizes =  sizes().concrete_sizes();
    TORCH_INTERNAL_ASSERT(concrete_sizes.has_value());
    auto strides = computeStrideProps(
        *concrete_sizes,
        contiguousStridesOf(*concrete_sizes));
    cloned->strides_ = strides;
    return cloned;
  }

  const SymbolicShape& symbolic_sizes() const;

  TensorTypePtr merge(const TensorType& other, bool merge_sizes = true) const;

  bool matchTensor(const at::Tensor& t);

  // is all information about the type specified except for autograd?
  // This replaces the notion of a 'CompleteTensorType' that used to exist
  // in the type-hierarchy. Excluding require_grad and undefined allows
  // this to match the old behavior.
  bool isComplete() const {
    return scalar_type_ && device_ && sizes_.isComplete() && strides_.isComplete();
  }

  bool isInferredType() const {
    return is_inferred_;
  }

  static TensorTypePtr getInferred() {
    static auto valueInferred = TensorType::create(
        /*scalar_type=*/{},
        /*device=*/{},
        /*sizes=*/SymbolicShape(),
        /*stride=*/VaryingShape<Stride>{},
        /*requires_grad=*/{},
        /*undefined=*/false);
    valueInferred->is_inferred_ = true;
    return valueInferred;
  }

  // this property is used by GuardElimination
  // please see `checkInputs` for more details
  bool isSummarized() const {
    return !(isComplete() && requiresGrad().has_value() &&
             undefined().has_value());
  }

  TensorTypePtr withUndefined() {
    auto r = clone();
    r->undefined_ = true;
    return r;
  }

  TensorTypePtr withPossiblyUndefined() {
    auto r = clone();
    r->undefined_ = std::nullopt;
    return r;
  }

  std::optional<bool> undefined() const { return undefined_; }

  static const TensorTypePtr& get();

  static const TypeKind Kind = TypeKind::TensorType;

  static std::vector<int64_t> contiguousStridesOf(
      at::IntArrayRef in_sizes,
      at::MemoryFormat memory_format = MemoryFormat::Contiguous) {
    auto contiguous_fn = [](const at::IntArrayRef& sizes,
                            const std::vector<int64_t>& dim_order) {
      std::vector<int64_t> strides(sizes.size());
      if (sizes.empty()) // zero-dim case
        return strides;

      strides[dim_order[0]] = 1;
      for (size_t i = 1; i < dim_order.size(); i++) {
        auto cur_dim = dim_order[i];
        auto pre_dim = dim_order[i - 1];
        strides[cur_dim] = strides[pre_dim] * sizes[pre_dim];
      }
      return strides;
    };

    std::vector<int64_t> dim_order(in_sizes.size());
    if (memory_format == MemoryFormat::ChannelsLast) {
      dim_order = {1, 3, 2, 0};
    } else if (memory_format == MemoryFormat::ChannelsLast3d) {
      dim_order = {1, 4, 3, 2, 0};
    } else {
      auto ndims = in_sizes.size();
      for (size_t i = 0; i < ndims; i++) {
        dim_order[i] = static_cast<int64_t>(ndims - i - 1); // Reverse
      }
    }
    return contiguous_fn(in_sizes, dim_order);
  }

 private:
  TensorType(
      std::optional<at::ScalarType> scalar_type,
      std::optional<Device> device,
      SymbolicShape sizes,
      VaryingShape<Stride> strides,
      std::optional<bool> requires_grad,
      std::optional<bool> undefined = false);

  TensorTypePtr clone() const {
    return TensorTypePtr(new TensorType(
        scalar_type_, device_, sizes_, strides_, requires_grad_, undefined_));
  }

  static VaryingShape<Stride> computeStrideProps(
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      bool tensor_contiguity = false);

  std::optional<at::ScalarType> scalar_type_;
  std::optional<at::Device> device_;
  SymbolicShape sizes_;
  VaryingShape<Stride> strides_;
  std::optional<bool> requires_grad_;
  // we exploit the fact certain tensors must be zero in the autograd to
  // optimize gradient computation. Such zero tensors are currently implemented
  // with `UndefinedTensorImpl.` They can be handled only by special operators
  // (e.g. `AutogradAdd`) and their `Tensor::defined()` property returns false.
  // Normally, `undefined_` is set to false, unless a type was created
  // with `withUndefined`
  // This will also mean that `undefined` tensors will fail
  // `subtypeOf(TensorType::get())` check
  // undefined_ may become `std::nullopt` if the tensor was observed to be both
  // defined and undefined. However, no tensor type starts out with
  // `undefined_` set to `std::nullopt`
  std::optional<bool> undefined_;
  // Represents whether or not this type was inferred.
  bool is_inferred_ = false;
};

struct ListType;
using ListTypePtr = std::shared_ptr<ListType>;
struct TORCH_API ListType
    : public SingleElementType<TypeKind::ListType, ListType> {
  // It's not exactly a singleton, but there should be exactly one instance of
  // List[T] for every T
  friend struct Type;
  template <typename... T>
  static ListTypePtr create(T&&... all) {
    return ListTypePtr(
        new ListType(std::forward<T>(all)...)); // NOLINT(modernize-make-shared)
  }

  std::string str() const override {
    std::stringstream ss;
    ss << getElementType()->str() << "[]";
    return ss.str();
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types.at(0)));
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // global singleton
  // Given an inner type T and an identifier,
  // this function will return the global singleton type pointer
  // the type List<T>.
  // The extra "identifier" argument is needed because we have multiple container types
  // that all re-use this function (List<T>, array<T, N>, etc.)
  static TypePtr get(const std::string& identifier, TypePtr inner);

  // common cast List[Tensor]
  static ListTypePtr ofTensors();
  static ListTypePtr ofOptionalTensors();
  static ListTypePtr ofInts();
  static ListTypePtr ofSymInts();
  static ListTypePtr ofFloats();
  static ListTypePtr ofComplexDoubles();
  static ListTypePtr ofBools();
  static ListTypePtr ofStrings();
  static ListTypePtr ofNumbers();

 private:
  ListType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    std::stringstream ss;
    ss << "List[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

struct DictType;
using DictTypePtr = std::shared_ptr<DictType>;
struct TORCH_API DictType : public SharedType {
  friend struct Type;
  static const TypeKind Kind = TypeKind::DictType;

  static DictTypePtr create(TypePtr key, TypePtr value) {
    auto kind = key->kind();
    if (auto dyn = key->castRaw<DynamicType>()) {
      kind = dyn->dynamicKind();
    }
    C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")
    switch (kind) {
      case TypeKind::AnyType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::ComplexType:
      case TypeKind::StringType:
      case TypeKind::TensorType:
      case TypeKind::DeviceObjType:
        return DictTypePtr(new DictType(std::move(key), std::move(value)));
      default:
        TORCH_CHECK(false,
            "Cannot create dict for key type '",
            key->str(),
            "', only int, float, complex, Tensor, device and string keys are supported");
    }
    C10_DIAGNOSTIC_POP()
  }

  // aligned with the format in FunctionSchema
  std::string str() const override {
    std::stringstream ss;
    ss << "Dict(" << getKeyType()->str() << ", " << getValueType()->str()
       << ")";
    return ss.str();
  }

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    TORCH_CHECK(contained_types.size() == 2, "Expected 2 contained types");
    return create(std::move(contained_types.at(0)), std::move(contained_types.at(1)));
  }

  const TypePtr& getKeyType() const {
    return types.at(0);
  }

  const TypePtr& getValueType() const {
    return types.at(1);
  }

  bool hasFreeVariables() const override {
    return has_free_variables;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return types;
  }

  bool equals(const Type& rhs) const override {
    if (auto* dict_rhs = rhs.castRaw<DictType>()) {
      return *getKeyType() == *(dict_rhs->getKeyType()) &&
          *getValueType() == *(dict_rhs->getValueType());
    }
    return false;
  }

  // global singleton
  // Given an inner type T and an identifier,
  // this function will return the global singleton type pointer
  // the type List<T>.
  // The extra "identifier" argument is needed because we have multiple container types
  // that all re-use this function (Dict<K, V> and unordered_map<K, V>)
  static TypePtr get(const std::string& identifier, TypePtr key, TypePtr val);

 private:
  DictType(TypePtr key, TypePtr value)
      : SharedType(TypeKind::DictType),
        has_free_variables(
            key->hasFreeVariables() || value->hasFreeVariables()) {
    types.reserve(2);
    types.push_back(std::move(key));
    types.push_back(std::move(value));
  }

  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override;

  std::vector<TypePtr> types;
  bool has_free_variables;
};

struct FutureType;
using FutureTypePtr = std::shared_ptr<FutureType>;

struct TORCH_API FutureType
    : public SingleElementType<TypeKind::FutureType, FutureType> {
  friend struct Type;
  template <typename... T>
  static FutureTypePtr create(TypePtr elem) {
    return FutureTypePtr(
        new FutureType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  std::string str() const override {
    std::stringstream ss;
    ss << "Future(" << getElementType()->str() << ")";
    return ss.str();
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types.at(0)));
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    if (Type::isSubtypeOfExt(rhs, why_not)) {
      return true;
    }
    if (auto rhs_ = rhs.castRaw<FutureType>()) {
      return getElementType()->isSubtypeOfExt(*rhs_->getElementType(), why_not);
    }
    return false;
  }

 private:
  FutureType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    std::stringstream ss;
    ss << "Future[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

struct AwaitType;
using AwaitTypePtr = std::shared_ptr<AwaitType>;

struct TORCH_API AwaitType
    : public SingleElementType<TypeKind::AwaitType, AwaitType> {
  friend struct Type;
  template <typename... T>
  static AwaitTypePtr create(TypePtr elem) {
    return AwaitTypePtr(
        new AwaitType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  std::string str() const override {
    std::stringstream ss;
    ss << "Await(" << getElementType()->str() << ")";
    return ss.str();
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types.at(0)));
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    if (Type::isSubtypeOfExt(rhs, why_not)) {
      return true;
    }
    if (auto rhs_ = rhs.castRaw<AwaitType>()) {
      return getElementType()->isSubtypeOfExt(*rhs_->getElementType(), why_not);
    }
    return false;
  }

 private:
  AwaitType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    std::stringstream ss;
    ss << "Await[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

struct RRefType;
using RRefTypePtr = std::shared_ptr<RRefType>;

struct TORCH_API RRefType
    : public SingleElementType<TypeKind::RRefType, RRefType> {
  friend struct Type;
  template <typename... T>
  static RRefTypePtr create(TypePtr elem) {
    return RRefTypePtr(
        new RRefType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  std::string str() const override {
    std::stringstream ss;
    ss << "RRef(" << getElementType()->str() << ")";
    return ss.str();
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types.at(0)));
  }

 private:
  RRefType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override {
    std::stringstream ss;
    ss << "RRef[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

// Any should never appear in a named type like a class, namedtuple or
// interface. If it does, then dynamic type information will be lost in the
// Pickler, leading to hard-to-track-down bugs that will only occur
// after saving or loading a model. This is because we rely on the
// static types in named types to reconstruct type tags of loaded
// values. Lifting this restriction requires solving the serialization
// problem first.
TORCH_API void checkNoAny(
    const Type& base,
    const char* what,
    const std::string& attrname,
    const TypePtr& attrtype);

struct TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;
using NameList = std::vector<std::string>;
// This type represents a Tuple
struct TORCH_API TupleType : public NamedType {

  static TupleTypePtr createNamed(const std::optional<c10::QualifiedName>& name,
      const std::vector<std::string>& field_names,
      const std::vector<TypePtr>& field_types,
      std::vector<IValue>& field_defaults);

  static TupleTypePtr createNamed(const std::optional<c10::QualifiedName>& name,
      const std::vector<std::string>& field_names,
      const std::vector<TypePtr>& field_types);

  static TupleTypePtr createNamed(const std::optional<c10::QualifiedName>& name,
      const std::vector<std::string_view>& field_names,
      const std::vector<TypePtr>& field_types);

  static TupleTypePtr create(
      std::vector<TypePtr> types) {
    return TupleTypePtr(new TupleType(
        std::move(types),
        std::nullopt,
        nullptr)); // NOLINT(modernize-make-shared)
  }
  static TupleTypePtr create() {
    return create({});
  }

  at::ArrayRef<TypePtr> elements() const {
    return elements_;
  }

  bool equals(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const override;

  std::string str() const override;
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }
  at::ArrayRef<TypePtr> containedTypes() const override {
    return elements_;
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return std::shared_ptr<TupleType>(
        new TupleType(std::move(contained_types), name(), schema()));
  }
  const std::shared_ptr<FunctionSchema>& schema() const {
    return schema_;
  }
  std::optional<std::vector<std::string_view>> names() const;

  static const TypeKind Kind = TypeKind::TupleType;

 private:
  template <typename S>
  static TupleTypePtr createWithSpec(
      const std::optional<c10::QualifiedName>& name,
      const std::vector<S>& field_names,
      const std::vector<TypePtr>& field_types,
      std::vector<IValue>& field_defaults);

  TupleType(
      std::vector<TypePtr> elements_,
      std::optional<c10::QualifiedName> name,
      std::shared_ptr<FunctionSchema> schema);

  bool compare(
      const Type& rhs,
      const std::function<bool(const Type&, const Type&)>& fn) const {
    if (rhs.kind() != kind()) {
      return false;
    }

    const auto& l_elements = elements();
    const auto& r_elements = rhs.castRaw<TupleType>()->elements();
    if (l_elements.size() != r_elements.size())
      return false;
    for (size_t i = 0; i < l_elements.size(); ++i) {
      if (!fn(*l_elements[i], *r_elements[i]))
        return false;
    }
    return true;
  }

  std::string annotation_str_impl(const TypePrinter& printer = nullptr) const override;

  std::vector<TypePtr> elements_;
  bool has_free_variables_;
  std::shared_ptr<FunctionSchema> schema_;
};

// the common supertype of all Enums, only used in operator registration.
// EnumType <: AnyEnumType for all Enums
struct AnyEnumType;
using AnyEnumTypePtr = SingletonTypePtr<AnyEnumType>;
struct TORCH_API AnyEnumType final : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "AnyEnumType";
  }
  static const TypeKind Kind = TypeKind::AnyEnumType;
  // global singleton
  static AnyEnumTypePtr get();
private:
  AnyEnumType()
  : Type(TypeKind::AnyEnumType) {}
};

struct NumberType;
using NumberTypePtr = SingletonTypePtr<NumberType>;
// This type represents a Python number
// Subtype hierarchy for Number Types (NumberType as the base type):
// IntType <: NumberType
// FloatType <: NumberType
// ComplexType <:NumberType
//
// WARNING: if you add a new subtype of NumberType that is not
// represented by a global singleton, you need to change NumberTypePtr
// to a SingletonOrSharedTypePtr and deal with NumberType needing to
// both inherit and not inherit from SharedType!
struct TORCH_API NumberType : public Type {
  bool equals(const Type& rhs) const override;

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  std::string str() const override {
    return "Scalar"; // match what PythonArgParser says for clarity
  }
  static const TypeKind Kind = TypeKind::NumberType;
  // global singleton
  static NumberTypePtr get();

 protected:
  NumberType(TypeKind kind = TypeKind::NumberType) : Type(kind) {}

  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "number"; // technically not a valid python type, but
                     // we need to use it when parsing back in annotations
                     // for implicit conversions
  }
};

struct FloatType;
using FloatTypePtr = SingletonTypePtr<FloatType>;
// This type represents a Python float number
struct TORCH_API FloatType : public NumberType {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "float";
  }
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::FloatType;
  // global singleton
  static FloatTypePtr get();

 private:
  FloatType() : NumberType(TypeKind::FloatType) {}
  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "float";
  }
};

struct ComplexType;
using ComplexTypePtr = SingletonTypePtr<ComplexType>;
// This type represents a Python float number
struct TORCH_API ComplexType : public NumberType {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "complex";
  }
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::ComplexType;
  // global singleton
  static ComplexTypePtr get();

 private:
  ComplexType() : NumberType(TypeKind::ComplexType) {}
  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "complex";
  }
};

// We need to introduce `SymIntType` to represent the `SymInt` type
// used in function schemas e.g. `aten::narrow_copy(... SymInt length)
// `SymInt` will be used to enable tracing arithmetic operations on
// dimension values. Please see [SymInt.h] for more information
struct SymIntType;
using SymIntTypePtr = SingletonTypePtr<SymIntType>;
struct TORCH_API SymIntType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "SymInt";
  }
  std::string annotation_str_impl(const TypePrinter& printer [[maybe_unused]] = nullptr) const override {
    return "int";
  }
  static const TypeKind Kind = TypeKind::SymIntType;
  // global singleton
  static SymIntTypePtr get();

 private:
  SymIntType() : Type(TypeKind::SymIntType) {}
};

struct SymFloatType;
using SymFloatTypePtr = SingletonTypePtr<SymFloatType>;
struct TORCH_API SymFloatType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "SymFloat";
  }
  std::string annotation_str_impl(const TypePrinter& printer [[maybe_unused]] = nullptr) const override {
    return "float";
  }
  static const TypeKind Kind = TypeKind::SymFloatType;
  // global singleton
  static SymFloatTypePtr get();

 private:
  SymFloatType() : Type(TypeKind::SymFloatType) {}
};

struct SymBoolType;
using SymBoolTypePtr = SingletonTypePtr<SymBoolType>;
struct TORCH_API SymBoolType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "SymBool";
  }
  std::string annotation_str_impl(const TypePrinter& printer [[maybe_unused]] = nullptr) const override {
    return "bool";
  }
  static const TypeKind Kind = TypeKind::SymBoolType;
  // global singleton
  static SymBoolTypePtr get();

 private:
  SymBoolType() : Type(TypeKind::SymBoolType) {}
};

struct IntType;
using IntTypePtr = SingletonTypePtr<IntType>;
// This type represents a Python int number
struct TORCH_API IntType : public NumberType {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "int";
  }
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::IntType;
  // global singleton
  static IntTypePtr get();

 private:
  IntType() : NumberType(TypeKind::IntType) {}
  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "int";
  }
};

struct BoolType;
using BoolTypePtr = SingletonTypePtr<BoolType>;
// This node represents a Python bool value
struct TORCH_API BoolType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "bool";
  }
  static const TypeKind Kind = TypeKind::BoolType;
  // global singleton
  static BoolTypePtr get();

 private:
  BoolType() : Type(TypeKind::BoolType) {}
};

struct StringType;
using StringTypePtr = SingletonTypePtr<StringType>;
// This type represents a Python string
struct TORCH_API StringType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    // we only use "str" (not "string") in both FunctionSchema and script
    return annotation_str();
  }
  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "str";
  }
  static const TypeKind Kind = TypeKind::StringType;
  // global singleton
  static StringTypePtr get();

 private:
  StringType() : Type(TypeKind::StringType) {}
};

struct StorageType;
using StorageTypePtr = SingletonTypePtr<StorageType>;
struct TORCH_API StorageType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return annotation_str();
  }
  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return "Storage";
  }
  static const TypeKind Kind = TypeKind::StorageType;
  // global singleton
  static StorageTypePtr get();

 private:
  StorageType() : Type(TypeKind::StorageType) {}
};

struct FunctionType;
using FunctionTypePtr = std::shared_ptr<FunctionType>;
struct TORCH_API FunctionType : public NamedType {
  static FunctionTypePtr create(torch::jit::Function* function) {
    return FunctionTypePtr(
        new FunctionType(function)); // NOLINT(modernize-make-shared)
  }
  bool equals(const Type& rhs) const override {
    if (auto func_type = rhs.cast<FunctionType>()) {
      return func_type->function_ == function_;
    }

    return false;
  }
  std::string str() const override {
    return "Function";
  }
  torch::jit::Function* function() const {
    return function_;
  }
  static const TypeKind Kind = TypeKind::FunctionType;

 private:
  FunctionType(torch::jit::Function* function);
  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return name()->qualifiedName();
  }
  torch::jit::Function* function_;
};

struct NoneType;
using NoneTypePtr = SingletonTypePtr<NoneType>;
// This type represents a Python None
struct TORCH_API NoneType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "NoneType";
  }
  bool isSubtypeOfExt(const Type& rhs, std::ostream *why_not) const override;

  static const TypeKind Kind = TypeKind::NoneType;
  // global singleton
  static NoneTypePtr get();

 private:
  NoneType() : Type(TypeKind::NoneType) {}
};

struct GeneratorType;
using GeneratorTypePtr = SingletonTypePtr<GeneratorType>;
// This type represents a Generator
struct TORCH_API GeneratorType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Generator";
  }
  static const TypeKind Kind = TypeKind::GeneratorType;
  // global singleton
  static GeneratorTypePtr get();

 private:
  GeneratorType() : Type(TypeKind::GeneratorType) {}
};

struct QuantizerType;
using QuantizerTypePtr = SingletonTypePtr<QuantizerType>;
// This type represents a Quantizer
struct TORCH_API QuantizerType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Quantizer";
  }
  static const TypeKind Kind = TypeKind::QuantizerType;
  // global singleton
  static QuantizerTypePtr get();

 private:
  QuantizerType() : Type(TypeKind::QuantizerType) {}
};

struct QSchemeType;
using QSchemeTypePtr = SingletonTypePtr<QSchemeType>;
// This type represents a QScheme
struct TORCH_API QSchemeType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "QScheme";
  }
  static const TypeKind Kind = TypeKind::QSchemeType;
  // global singleton
  static QSchemeTypePtr get();

 private:
  QSchemeType() : Type(TypeKind::QSchemeType) {}
};

struct DeviceObjType;
using DeviceObjTypePtr = SingletonTypePtr<DeviceObjType>;
// This type represents a Device
struct TORCH_API DeviceObjType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Device";
  }
  static const TypeKind Kind = TypeKind::DeviceObjType;
  // global singleton
  static DeviceObjTypePtr get();

 private:
  DeviceObjType() : Type(TypeKind::DeviceObjType) {}
};

struct StreamObjType;
using StreamObjTypePtr = SingletonTypePtr<StreamObjType>;
// This type represents a Generator
struct TORCH_API StreamObjType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Stream";
  }
  static const TypeKind Kind = TypeKind::StreamObjType;
  // global singleton
  static StreamObjTypePtr get();

private:
  StreamObjType() : Type(TypeKind::StreamObjType) {}
};

struct VarType;
using VarTypePtr = std::shared_ptr<VarType>;
// This type represents a type variable, used in FunctionSchema
struct VarType : public SharedType {
  static VarTypePtr create(std::string name_) {
    return VarTypePtr(new VarType(std::move(name_)));
  }
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return name();
  }
  const std::string& name() const {
    return name_;
  }
  bool hasFreeVariables() const override {
    return true;
  }
  static const TypeKind Kind = TypeKind::VarType;

 private:
  VarType(std::string name_)
      : SharedType(TypeKind::VarType), name_(std::move(name_)) {}
  std::string name_;
};

struct CapsuleType;
using CapsuleTypePtr = SingletonTypePtr<CapsuleType>;
// This type represents a Python Capsule.
// It does not appear in the IR and is only used during runtime
struct TORCH_API CapsuleType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Capsule";
  }
  static const TypeKind Kind = TypeKind::CapsuleType;
  // global singleton
  static CapsuleTypePtr get();
private:
  CapsuleType()
  : Type(TypeKind::CapsuleType) {}
};

struct PyObjectType;
using PyObjectTypePtr = SingletonTypePtr<PyObjectType>;
// This type represents a PyObject Type
struct TORCH_API PyObjectType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "PyObject";
  }
  static const TypeKind Kind = TypeKind::PyObjectType;
  // global singleton
  static PyObjectTypePtr get();
private:
  PyObjectType()
  : Type(TypeKind::PyObjectType) {}
};

enum class TypeVerbosity {
  None,
  Type,
  TypeAndStride,
  Full,
  Symbolic,
  Default = Full,
};

TORCH_API TypeVerbosity type_verbosity();

TORCH_API std::ostre
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/core`):

- [`operator_name.cpp_docs.md_docs.md`](./operator_name.cpp_docs.md_docs.md)
- [`builtin_function.h_kw.md_docs.md`](./builtin_function.h_kw.md_docs.md)
- [`QuantizerBase.h_docs.md_docs.md`](./QuantizerBase.h_docs.md_docs.md)
- [`MT19937RNGEngine.h_docs.md_docs.md`](./MT19937RNGEngine.h_docs.md_docs.md)
- [`UndefinedTensorImpl.h_docs.md_docs.md`](./UndefinedTensorImpl.h_docs.md_docs.md)
- [`IListRef_test.cpp_docs.md_docs.md`](./IListRef_test.cpp_docs.md_docs.md)
- [`CheckMemoryFormat.h_docs.md_docs.md`](./CheckMemoryFormat.h_docs.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`PythonFallbackKernel.cpp_docs.md_docs.md`](./PythonFallbackKernel.cpp_docs.md_docs.md)
- [`Dict.h_kw.md_docs.md`](./Dict.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `jit_type.h_docs.md_docs.md`
- **Keyword Index**: `jit_type.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
