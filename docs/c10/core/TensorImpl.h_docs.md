# Documentation: `c10/core/TensorImpl.h`

## File Metadata

- **Path**: `c10/core/TensorImpl.h`
- **Size**: 115,952 bytes (113.23 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/SymbolicShapeMeta.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/PyObjectSlot.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/Flags.h>
#include <c10/util/accumulate.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10/util/safe_numerics.h>
#include <c10/util/typeid.h>
#include <optional>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// A global boolean variable to control whether we free memory when a Tensor
// is shrunk to a smaller size. As a result, a Tensor is always going to
// keep the memory allocated for its maximum capacity reshaped to so far.
//
// This parameter is respected "upper-case" methods which call Resize()
// (e.g., CopyFrom, ResizeLike); it is NOT respected by Tensor::resize_
// or ShrinkTo, both of which guarantee to never to free memory.
C10_DECLARE_bool(caffe2_keep_on_shrink);

// Since we can have high variance in blob memory allocated across different
// inputs in the same run, we will shrink the blob only if the memory gain
// is larger than this flag in bytes.  This only applies to functions which
// respect caffe2_keep_on_shrink.
C10_DECLARE_int64(caffe2_max_keep_on_shrink_memory);

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-default")

namespace at {
class Tensor;
class TensorBase;
} // namespace at

namespace c10 {

/**
 * A utility function to convert vector<int> to vector<int64_t>.
 */
inline std::vector<int64_t> ToVectorint64_t(const ArrayRef<int>& src) {
  return std::vector<int64_t>(src.begin(), src.end());
}

/**
 * Return product of all dimensions starting from k
 */
inline int64_t size_from_dim_(int k, IntArrayRef dims) {
  int64_t r = 1;
  for (const auto i : c10::irange(k, dims.size())) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to k (not including dims[k])
inline int64_t size_to_dim_(int k, IntArrayRef dims) {
  TORCH_CHECK(k >= 0 && static_cast<size_t>(k) <= dims.size());
  int64_t r = 1;
  for (const auto i : c10::irange(k)) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims between k and l (not including dims[k] and dims[l])
inline int64_t size_between_dim_(int k, int l, IntArrayRef dims) {
  TORCH_CHECK((unsigned)l < dims.size() && (unsigned)k < dims.size());
  int64_t r = 1;
  if (k < l) {
    for (int i = k + 1; i < l; ++i) {
      r *= dims[i];
    }
  } else {
    for (int i = l + 1; i < k; ++i) {
      r *= dims[i];
    }
  }
  return r;
}

// Wrap around axis_index if it is negative, s.t., -1 is the last dim
inline int canonical_axis_index_(int axis_index, int ndims) {
  TORCH_CHECK(axis_index >= -ndims);
  TORCH_CHECK(axis_index < ndims);
  if (axis_index < 0) {
    return axis_index + ndims;
  }
  return axis_index;
}

using PlacementDtor = void (*)(void*, size_t);

/*
 * A Context that will call extra placement deleter during
 * deconstruction.
 *
 * Accept a already constructed DataPtr and store it as member
 * during destruction, we'll call extra deleter on the underlying
 * data pointer before the DataPtr is destructed.
 * `data_ptr_` owns the memory.
 */
struct C10_API PlacementDeleteContext {
  DataPtr data_ptr_;
  PlacementDtor placement_dtor_;
  size_t size_;

  PlacementDeleteContext(
      DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size)
      : data_ptr_(std::move(data_ptr)),
        placement_dtor_(placement_dtor),
        size_(size) {}

  PlacementDeleteContext(PlacementDeleteContext&&) noexcept = delete;
  PlacementDeleteContext(const PlacementDeleteContext&) = delete;
  PlacementDeleteContext& operator=(const PlacementDeleteContext&) = delete;
  PlacementDeleteContext& operator=(PlacementDeleteContext&&) = delete;
  static DataPtr makeDataPtr(
      DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size,
      Device device);
  ~PlacementDeleteContext() {
    placement_dtor_(data_ptr_.get(), size_);
    // original memory will be freed when data_ptr_ is destructed
  }
};

struct C10_API AutogradMetaInterface {
  virtual void set_requires_grad(
      bool requires_grad,
      at::TensorImpl* self_impl) = 0;
  virtual bool requires_grad() const = 0;
  virtual at::Tensor& mutable_grad() = 0;
  virtual const at::Tensor& grad() const = 0;
  virtual const at::Tensor& fw_grad(uint64_t level, const at::TensorBase& self)
      const = 0;
  virtual void set_fw_grad(
      const at::TensorBase& new_grad,
      const at::TensorBase& self,
      uint64_t level,
      bool is_inplace_op) = 0;
  virtual ~AutogradMetaInterface();
};

namespace impl {

// Unfortunately, the definition of AutogradMeta lives in a separate
// compilation unit than TensorImpl (libtorch.so versus libc10.so)
// which means that we cannot construct an AutogradMeta from TensorImpl,
// not even from the cpp file.  So we have to indirect it through a factory
// function which will be initialized when we load libtorch.so.

struct C10_API AutogradMetaFactory {
  virtual ~AutogradMetaFactory() = default;
  virtual std::unique_ptr<AutogradMetaInterface> make() const = 0;
  // This method is the dumbest method.  But I don't have access
  // to Tensor (not TensorImpl) which is undefined in this header.
  virtual const at::Tensor& undefined_tensor() const = 0;
};

C10_API void SetAutogradMetaFactory(AutogradMetaFactory* factory);
C10_API AutogradMetaFactory* GetAutogradMetaFactory();

struct C10_API AutogradMetaFactoryRegisterer{
    explicit AutogradMetaFactoryRegisterer(AutogradMetaFactory * factory){
        SetAutogradMetaFactory(factory);
} // namespace impl
}; // namespace c10

} // namespace impl

struct C10_API NamedTensorMetaInterface {
  virtual ~NamedTensorMetaInterface() = default;
  virtual std::unique_ptr<NamedTensorMetaInterface> clone() const {
    TORCH_INTERNAL_ASSERT(
        false, "Not implemented: NamedTensorMetaInterface::clone");
  }
  virtual int64_t slow_dim() const {
    TORCH_INTERNAL_ASSERT(
        false, "Not implemented: NamedTensorMetaInterface::slow_dim");
  }
};

// For ease of copy pasting
#if 0
is_contiguous
is_channels_last_contiguous
is_channels_last_3d_contiguous
is_channels_last
is_channels_last_3d
is_non_overlapping_and_dense
#endif

/**
 * This structure is intended to hold additional metadata of the specific device
 * backend.
 **/
struct C10_API BackendMeta : intrusive_ptr_target {
  ~BackendMeta() override = default;
  virtual intrusive_ptr<BackendMeta> clone(
      const intrusive_ptr<BackendMeta>& ptr) const {
    return ptr;
  }
};

struct C10_API ExtraMeta {
  std::unique_ptr<c10::SymbolicShapeMeta> symbolic_shape_meta_ = nullptr;
  std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta_ = nullptr;
  intrusive_ptr<c10::BackendMeta> backend_meta_ = nullptr;
  std::optional<std::string> custom_data_ptr_error_msg_ = std::nullopt;
  std::optional<std::string> custom_storage_error_msg_ = std::nullopt;

  ExtraMeta() = default;
  ~ExtraMeta() = default;
  ExtraMeta(const ExtraMeta& other) {
    if (other.symbolic_shape_meta_) {
      symbolic_shape_meta_ =
          std::make_unique<c10::SymbolicShapeMeta>(*other.symbolic_shape_meta_);
    }
    if (other.named_tensor_meta_) {
      named_tensor_meta_ = other.named_tensor_meta_->clone();
    }
    if (other.backend_meta_) {
      backend_meta_ = other.backend_meta_->clone(other.backend_meta_);
    }
    if (other.custom_data_ptr_error_msg_) {
      custom_data_ptr_error_msg_ = other.custom_data_ptr_error_msg_;
    }
    if (other.custom_storage_error_msg_) {
      custom_storage_error_msg_ = other.custom_storage_error_msg_;
    }
  }
  ExtraMeta& operator=(const ExtraMeta& other) = delete;
  ExtraMeta(ExtraMeta&& other) = delete;
  ExtraMeta& operator=(ExtraMeta&& other) = delete;

  ExtraMeta(
      std::unique_ptr<c10::SymbolicShapeMeta> symbolic_shape_meta,
      std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta,
      intrusive_ptr<c10::BackendMeta> backend_meta,
      std::optional<std::string> custom_data_ptr_error_msg = std::nullopt,
      std::optional<std::string> custom_storage_access_error_msg = std::nullopt)
      : symbolic_shape_meta_(std::move(symbolic_shape_meta)),
        named_tensor_meta_(std::move(named_tensor_meta)),
        backend_meta_(std::move(backend_meta)),
        custom_data_ptr_error_msg_(std::move(custom_data_ptr_error_msg)),
        custom_storage_error_msg_(std::move(custom_storage_access_error_msg)) {}

  std::unique_ptr<ExtraMeta> clone() const {
    return std::make_unique<ExtraMeta>(*this);
  }
};

// NOTE [ Version Counter Sharing ]
//
// Every Tensor has a version counter. Version counters are incremented whenever
// the data or size of a tensor changes through in-place Variable operations.
// Version counters are used to detect modifications to saved variables which
// would result in incorrect gradient calculations. Version counters may be
// shared between Variables:
//
// 1. A view shares the version counter of the base Variable,
// 2. `x.detach()` shares the version counter of `x`,
// 3. Unpacked saved variables share the version counter of the source.
//
// Version counters are not shared in these scenarios:
//
// 1. When we replace a `Variable`'s underlying `Tensor` by calling
// `set_data(...)`,
// 2. `x.data` does not share the version counter of `x`. (See discussion at
// https://github.com/pytorch/pytorch/issues/5396)
//
// Question: Why do we put the version counter in TensorImpl instead of
// AutogradMeta?
//
// Answer: After the Variable/Tensor merge, a tensor will not have AutogradMeta
// when its `requires_grad_` is false, but when we use this tensor in the
// forward pass of a function that requires saving this tensor for backward, we
// need to keep track of this tensor's version to make sure it's always valid in
// the autograd graph.
//
// To achieve this goal, we put the version counter in TensorImpl instead of
// AutogradMeta, and have it always be available. This allows us to have the
// optimization of not carrying AutogradMeta when a tensor doesn't require
// gradient.
//
// A hypothetical alternative way to achieve this goal is to initialize
// AutogradMeta and create the version counter for the non-requires-grad tensor
// only when it's saved for backward. However, since saving a tensor for
// backward happens in the forward pass, and our invariant is that forward pass
// needs to be thread-safe, lazy-initializing AutogradMeta when saving a tensor
// can introduce race conditions when we are running the forward pass in
// multi-thread scenarios, thus making the forward pass not thread-safe anymore,
// which breaks the invariant.
struct C10_API VariableVersion {
 private:
  struct VersionCounter : intrusive_ptr_target {
    VersionCounter(uint32_t version) : version_(version) {}
    std::atomic<uint32_t> version_;
  };
  c10::intrusive_ptr<VersionCounter> version_counter_;

 public:
  // Note [Disabled VariableVersion]
  // VariableVersion struct has an intrusive_ptr pointing VersionCounter struct
  // with an atomic variable. Thus `VariableVersion(/*version=*/0)` is not as
  // cheap as we expected. In some cases constructing a VariableVersion with
  // version 0 is not necessary so we add a cheap constructor which
  // doesn't allocate the intrusive_ptr.
  // Example use cases are:
  //  - Inference tensors don't track version counter, so they'll just always
  //    have disabled VariableVersion.
  //  - In SavedVariable class we override version_counter_ inside its
  //  constructor
  //    so that we can use the cheap constructor there.
  enum Disabled { DISABLED };
  // It's okay to return true even for inference tensor which
  // doesn't have version counter enabled.
  // We want to be permissive here since in many cases (e.g. make_variable)
  // we can std::move a TensorImpl if there's no other uses which saves us
  // an additional TensorImpl allocation.
  bool unique() const {
    return version_counter_ ? 1 == version_counter_.use_count() : true;
  }
  // NOTE: As of C++11 and 14, default-constructing a std::atomic variable
  // leaves it in a persistently undefined state. See
  // https://cplusplus.github.io/LWG/issue2334.
  VariableVersion(uint32_t version)
      : version_counter_(c10::make_intrusive<VersionCounter>(version)) {}
  VariableVersion(Disabled /*unused*/ = DISABLED) {}

  bool enabled() const {
    return version_counter_;
  }

  // Note [Inplace update inference tensor]
  // 1. Inplace update to inference tensor is forbidden in normal mode.
  //   For example:
  //     inference_tensor.copy_(normal_tensor_requires_grad)
  //   This inplace makes inference_tensor have requires_grad=True and
  //   have a grad_fn.  This is bad because views of `inference_tensor`
  //   created in InferenceMode won't be able to know the grad_fn since
  //   their ViewMeta were not recorded. To match NoGradMode behavior
  //   that "inplace update to a view created in NoGradMode raise an error",
  //   we just ban inplace update to inference tensor since we can't tell
  //   if an inference tensor is a view created in InferenceMode.
  //
  //   Note that views of normal tensor created in InferenceMode has proper
  //   ViewMeta so that they're aware of the grad_fn correctly.
  //
  // 2. Inplace update to inference tensor in inference tensor doesn't bump
  //    version counter.
  //    * It either doesn't call bump() by skipping ADInplaceOrView kernel,
  //      - e.g. inference_tensor.add_(1)
  //    * or bump() is a no-op for inference tensor.
  //      - e.g. inference_tensor.add_(normal_tensor)
  void bump() {
    // TODO: Replace the link to the documentation once it's available.
    TORCH_CHECK(
        version_counter_ || InferenceMode::is_enabled(),
        "Inplace update to inference tensor outside InferenceMode is not allowed."
        "You can make a clone to get a normal tensor before doing inplace update."
        "See https://github.com/pytorch/rfcs/pull/17 for more details.");
    if (version_counter_) {
      ++version_counter_->version_;
    }
  }

  void set_version(int64_t i) {
    TORCH_CHECK(
        version_counter_,
        "Tried to call torch.autograd._unsafe_set_version() on a tensor "
        "that does not have a version counter. Was it created in inference mode?");
    TORCH_CHECK(i >= 0, "Cannot set a version_counter to a value below 0: ", i);
    version_counter_->version_ = i;
  }

  // Inference tensor doesn't have version counter so it shouldn't be
  // accessed.
  uint32_t current_version() const {
    TORCH_CHECK(
        version_counter_, "Inference tensors do not track version counter.");
    return version_counter_->version_;
  }
};

// Forward declaration of TensorImpl needed for forward declaration of
// C10_TensorImpl_Size_Check_Dummy_Class
struct C10_API TensorImpl;

/**
 * NOTE: Some TensorImpl methods are small and not overridden in the
 * PyTorch codebase itself, but may theoretically need to be
 * overridden by third-party TensorImpl subclasses. This macro allows
 * users that need maximum performance and don't need these extension
 * points to disable them with a build-time flag. (In particular,
 * XLA's XLATensorImpl currently overrides these methods, so we can't
 * enable this flag by default.)
 */
#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
#define TENSORIMPL_MAYBE_VIRTUAL
#else
#define TENSORIMPL_MAYBE_VIRTUAL virtual
#endif

/**
 * The low-level representation of a tensor, which contains a pointer
 * to a storage (which contains the actual data) and metadata (e.g., sizes and
 * strides) describing this particular view of the data as a tensor.
 *
 * Some basic characteristics about our in-memory representation of
 * tensors:
 *
 *  - It contains a pointer to a storage struct (Storage/StorageImpl)
 *    which contains the pointer to the actual data and records the
 *    data type and device of the view.  This allows multiple tensors
 *    to alias the same underlying data, which allows to efficiently
 *    implement differing *views* on a tensor.
 *
 *  - The tensor struct itself records view-specific metadata about
 *    the tensor, e.g., sizes, strides and offset into storage.
 *    Each view of a storage can have a different size or offset.
 *
 *  - This class is intrusively refcounted.  It is refcounted so that
 *    we can support prompt deallocation of large tensors; it is
 *    intrusively refcounted so that we can still perform reference
 *    counted operations on raw pointers, which is often more convenient
 *    when passing tensors across language boundaries.
 *
 *  - For backwards-compatibility reasons, a tensor may be in an
 *    uninitialized state.  A tensor may be uninitialized in the following
 *    two ways:
 *
 *      - A tensor may be DTYPE UNINITIALIZED.  A tensor of this
 *        form has an uninitialized dtype.  This situation most
 *        frequently arises when a user writes Tensor x(CPU).  The dtype
 *        is subsequently initialized when mutable_data<T>() is
 *        invoked for the first time.
 *
 *      - A tensor may be STORAGE UNINITIALIZED.  A tensor of this form
 *        has non-zero size, but has a storage with a null data pointer.
 *        This situation most frequently arises when a user calls
 *        Resize() or FreeMemory().  This is because Caffe2 historically
 *        does lazy allocation: allocation of data doesn't occur until
 *        mutable_data<T>() is invoked.  A tensor with zero size is
 *        always storage initialized, because no allocation is necessary
 *        in this case.
 *
 *    All combinations of these two uninitialized states are possible.
 *    Consider the following transcript in idiomatic Caffe2 API:
 *
 *      Tensor x(CPU); // x is storage-initialized, dtype-UNINITIALIZED
 *      x.Resize(4); // x is storage-UNINITIALIZED, dtype-UNINITIALIZED
 *      x.mutable_data<float>(); // x is storage-initialized, dtype-initialized
 *      x.FreeMemory(); // x is storage-UNINITIALIZED, dtype-initialized.
 *
 *    All other fields on tensor are always initialized.  In particular,
 *    size is always valid. (Historically, a tensor declared as Tensor x(CPU)
 *    also had uninitialized size, encoded as numel == -1, but we have now
 *    decided to default to zero size, resulting in numel == 0).
 *
 *    Uninitialized storages MUST be uniquely owned, to keep our model
 *    simple.  Thus, we will reject operations which could cause an
 *    uninitialized storage to become shared (or a shared storage to
 *    become uninitialized, e.g., from FreeMemory).
 *
 *    In practice, tensors which are storage-UNINITIALIZED and
 *    dtype-UNINITIALIZED are *extremely* ephemeral: essentially,
 *    after you do a Resize(), you basically always call mutable_data()
 *    immediately afterwards.  Most functions are not designed to
 *    work if given a storage-UNINITIALIZED, dtype-UNINITIALIZED tensor.
 *
 *    We intend to eliminate all uninitialized states, so that every
 *    tensor is fully initialized in all fields.  Please do not write new code
 *    that depends on these uninitialized states.
 */
struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  TensorImpl() = delete;
  ~TensorImpl() override;
  // Note [Enum ImplType]
  // This enum is temporary. In the followup refactor we should
  // think about how to specialize TensorImpl creation for view
  // tensors. Currently we only special case its key_set_ but
  // there's also potential to share version_counter_ directly
  // without creating first and then override in as_view.
  enum ImplType { VIEW };

  /**
   * Construct a 1-dim 0-size tensor backed by the given storage.
   */
  TensorImpl(
      Storage&& storage,
      DispatchKeySet /*key_set*/,
      const caffe2::TypeMeta data_type);

  // See Note [Enum ImplType]
  TensorImpl(
      ImplType /*unused*/,
      Storage&& storage,
      DispatchKeySet /*key_set*/,
      const caffe2::TypeMeta data_type);

  /**
   * Construct a 1-dim 0 size tensor that doesn't have a storage.
   */
  TensorImpl(
      DispatchKeySet /*key_set*/,
      const caffe2::TypeMeta data_type,
      std::optional<c10::Device> device_opt);

  // Legacy constructors so I don't have to go update call sites.
  // TODO: When Variable is added, delete these constructors
  TensorImpl(
      Storage&& storage,
      DispatchKey dispatch_key,
      const caffe2::TypeMeta data_type)
      : TensorImpl(
            std::move(storage),
            DispatchKeySet(dispatch_key),
            data_type) {}
  TensorImpl(
      DispatchKey dispatch_key,
      const caffe2::TypeMeta data_type,
      std::optional<c10::Device> device_opt)
      : TensorImpl(DispatchKeySet(dispatch_key), data_type, device_opt) {}

 private:
  // This constructor is private, because the data_type is redundant with
  // storage.  Still, we pass it in separately because it's easier to write
  // the initializer list if we're not worried about storage being moved out
  // from under us.
  TensorImpl(
      Storage&& storage,
      DispatchKeySet /*key_set*/,
      const caffe2::TypeMeta data_type,
      std::optional<c10::Device> /*device_opt*/);

 public:
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

  /**
   * Release (decref) storage, and any other external allocations.  This
   * override is for `intrusive_ptr_target` and is used to implement weak
   * tensors.
   */
  void release_resources() override;

 public:
  /**
   * Return the DispatchKeySet corresponding to this Tensor, specifying
   * all of the DispatchKeys that this Tensor identifies as.  This is the
   * information used to dispatch operations on this tensor.
   */
  DispatchKeySet key_set() const {
    return key_set_;
  }

 private:
  [[noreturn]] void throw_cannot_call_with_symbolic(const char* meth) const;

  // NOTE: The general recipe for customizable methods is that the fastpath
  // function (e.g., sizes()) does an unlikely policy test, and if doesn't
  // trigger, it does the fast path implementation with no checks and going
  // directly to on-TensorImpl fields.  In particular, you never need to
  // check ExtraMeta if the policy doesn't trigger, as non-trivial ExtraMeta
  // implies the policy will always match.
  //
  // The default implementations of methods are "safe": they do extra tests
  // to make sure the internal state is consistent no matter if you are
  // doing symbolic shapes or not.  If you don't want the tests, directly
  // override the custom method (e.g., custom_sizes()) to do your preferred
  // behavior.

 public:
  /**
   * Return a reference to the sizes of this tensor.  This reference remains
   * valid as long as the tensor is live and not resized.
   */
  IntArrayRef sizes() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sizes_custom();
    }
    return sizes_and_strides_.sizes_arrayref();
  }

  SymIntArrayRef sym_sizes() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sym_sizes_custom();
    }
    // Sizes guaranteed to be non-negative, so unchecked cast is OK
    return c10::fromIntArrayRefKnownNonNegative(
        sizes_and_strides_.sizes_arrayref());
  }

  IntArrayRef sizes_default() const {
    if (C10_UNLIKELY(has_symbolic_sizes_strides_)) {
      throw_cannot_call_with_symbolic("sizes");
    }
    return sizes_and_strides_.sizes_arrayref();
  }

  SymIntArrayRef sym_sizes_default() const {
    if (has_symbolic_sizes_strides_) {
      return symbolic_shape_meta().sizes_;
    } else {
      // Sizes guaranteed to be non-negative, so unchecked cast is OK
      return c10::fromIntArrayRefKnownNonNegative(sizes_default());
    }
  }

  template <typename T>
  ArrayRef<T> generic_sizes() {
    static_assert(
        std::is_same_v<T, int64_t> || std::is_same_v<T, c10::SymInt>,
        "Only supports int64_t and c10::SymInt.");

    if constexpr (std::is_same_v<T, int64_t>) {
      return sizes();
    } else {
      return sym_sizes();
    }
  }

  template <typename T>
  ArrayRef<T> generic_strides() {
    static_assert(
        std::is_same_v<T, int64_t> || std::is_same_v<T, c10::SymInt>,
        "Only supports int64_t and c10::SymInt.");

    if constexpr (std::is_same_v<T, int64_t>) {
      return strides();
    } else {
      return sym_strides();
    }
  }

  template <typename T>
  T generic_storage_offset() {
    static_assert(
        std::is_same_v<T, int64_t> || std::is_same_v<T, c10::SymInt>,
        "Only supports int64_t and c10::SymInt.");

    if constexpr (std::is_same_v<T, int64_t>) {
      return storage_offset();
    } else {
      return sym_storage_offset();
    }
  }

  /**
   * The number of elements in a tensor.
   *
   * WARNING: Previously, if you were using the Caffe2 API, you could
   * test numel() == -1 to see if a tensor was uninitialized.  This
   * is no longer true; numel always accurately reports the product
   * of sizes of a tensor.
   */
  int64_t numel() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return numel_custom();
    }
    return numel_;
  }

  c10::SymInt sym_numel() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sym_numel_custom();
    }
    return c10::SymInt(SymInt::UNCHECKED, numel_);
  }

  int64_t numel_default() const {
    if (C10_UNLIKELY(has_symbolic_sizes_strides_)) {
      throw_cannot_call_with_symbolic("numel");
    }
    return numel_;
  }

  c10::SymInt sym_numel_default() const {
    if (has_symbolic_sizes_strides_) {
      return symbolic_shape_meta().numel();
    } else {
      return c10::SymInt(SymInt::UNCHECKED, numel_);
    }
  }

  /**
   * Return the number of dimensions of this tensor.  Note that 0-dimension
   * represents a Tensor that is a Scalar, e.g., one that has a single element.
   */
  int64_t dim() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return dim_custom();
    }
    return static_cast<int64_t>(sizes_and_strides_.size());
  }

  int64_t dim_default() const {
    if (has_symbolic_sizes_strides_) {
      return static_cast<int64_t>(symbolic_shape_meta().sizes_.size());
    } else {
      return static_cast<int64_t>(sizes_and_strides_.size());
    }
  }

  /**
   * Return the offset in number of elements into the storage that this
   * tensor points to.  Most tensors have storage_offset() == 0, but,
   * for example, an index into a tensor will have a non-zero storage_offset().
   *
   * WARNING: This is NOT computed in bytes.
   */
  int64_t storage_offset() const {
    // TODO: maybe this should be toggled by strides
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return storage_offset_custom();
    }
    return storage_offset_;
  }

  c10::SymInt sym_storage_offset() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sym_storage_offset_custom();
    }
    return c10::SymInt(SymInt::UNCHECKED, storage_offset_);
  }

  int64_t storage_offset_default() const {
    if (C10_UNLIKELY(has_symbolic_sizes_strides_)) {
      throw_cannot_call_with_symbolic("storage_offset");
    }
    return storage_offset_;
  }

  c10::SymInt sym_storage_offset_default() const {
    if (has_symbolic_sizes_strides_) {
      return symbolic_shape_meta().storage_offset_;
    } else {
      return c10::SymInt(SymInt::UNCHECKED, storage_offset_);
    }
  }

  /**
   * Return a reference to the strides of this tensor.  This reference remains
   * valid as long as the tensor is live and not restrided.
   */
  IntArrayRef strides() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
      return strides_custom();
    }
    return sizes_and_strides_.strides_arrayref();
  }

  c10::SymIntArrayRef sym_strides() const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
      return sym_strides_custom();
    }
    return c10::fromIntArrayRefKnownNonNegative(strides_default());
  }

  IntArrayRef strides_default() const {
    if (C10_UNLIKELY(has_symbolic_sizes_strides_)) {
      throw_cannot_call_with_symbolic("strides");
    }
    return sizes_and_strides_.strides_arrayref();
  }

  c10::SymIntArrayRef sym_strides_default() const {
    if (has_symbolic_sizes_strides_) {
      return symbolic_shape_meta().strides_;
    } else {
      return c10::fromIntArrayRefKnownNonNegative(strides_default());
    }
  }

  c10::SymBool sym_is_contiguous(
      at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
      return sym_is_contiguous_custom(memory_format);
    }
    return sym_is_contiguous_default(memory_format);
  }

  template <typename T>
  T is_contiguous_default_impl(at::MemoryFormat memory_format) const {
    if (!has_symbolic_sizes_strides_) {
      if (memory_format == at::MemoryFormat::ChannelsLast) {
        return is_channels_last_contiguous_;
      } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
        return is_channels_last_3d_contiguous_;
      }
      return is_contiguous_;
    }

    // Handle dynamic shapes.
    const auto& symbolic = symbolic_shape_meta().is_contiguous(memory_format);

    if constexpr (std::is_same_v<T, bool>) {
      return symbolic.guard_bool(__FILE__, __LINE__);
    } else {
      return symbolic;
    }
  }

  bool is_contiguous_default(at::MemoryFormat memory_format) const {
    return is_contiguous_default_impl<bool>(memory_format);
  }

  c10::SymBool sym_is_contiguous_default(at::MemoryFormat memory_format) const {
    return is_contiguous_default_impl<c10::SymBool>(memory_format);
  }

  /**
   * Whether or not a tensor is laid out in contiguous memory.
   *
   * Tensors with non-trivial strides are not contiguous.  See
   * compute_contiguous() for the exact definition of whether or not
   * a tensor is contiguous or not.
   */
  bool is_contiguous(
      at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
      return is_contiguous_custom(memory_format);
    }
    return is_contiguous_default(memory_format);
  }

  bool is_strides_like_default(at::MemoryFormat memory_format) const {
    if (has_symbolic_sizes_strides_) {
      if (memory_format == at::MemoryFormat::ChannelsLast) {
        return symbolic_shape_meta().is_channels_last().guard_bool(
            __FILE__, __LINE__);
      } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
        return symbolic_shape_meta().is_channels_last_3d().guard_bool(
            __FILE__, __LINE__);
      } else {
        return false;
      }
    }

    if (memory_format == at::MemoryFormat::ChannelsLast) {
      return is_channels_last_;
    } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
      return is_channels_last_3d_;
    } else {
      return false;
    }
  }

  SymBool sym_is_non_overlapping_and_dense_default() const {
    if (has_symbolic_sizes_strides_) {
      return symbolic_shape_meta().is_non_overlapping_and_dense();
    } else {
      return is_non_overlapping_and_dense_;
    }
  }

  bool is_non_overlapping_and_dense_default() const {
    if (has_symbolic_sizes_strides_) {
      return sym_is_non_overlapping_and_dense_default().guard_bool(
          __FILE__, __LINE__);
    } else {
      return is_non_overlapping_and_dense_;
    }
  }

  // NB: these dim accessor functions don't have _default(), as you can use
  // sizes_default/strides_default
  /**
   * Return the size of a tensor at some dimension, wrapping the dimension if
   * necessary.
   *
   * NOTE: if you know wrapping is unnecessary, do sizes()[d] instead; it will
   * be faster
   */
  int64_t size(int64_t d) const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return size_custom(d);
    }
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    return sizes_and_strides_.size_at_unchecked(d);
  }

  c10::SymInt sym_size(int64_t d) const {
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomSizes))) {
      return sym_size_custom(d);
    }
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    const auto sizes = this->sym_sizes();
    return sizes[d];
  }

  /**
   * Return the stride of a tensor at some dimension, wrapping the dimension
   * if necessary.
   *
   * NOTE: if you know wrapping is unnecessary, do sizes()[d] instead; it will
   * be faster
   */
  int64_t stride(int64_t d) const {
    d = maybe_wrap_dim(d, dim(), false);
    if (C10_UNLIKELY(matches_policy(SizesStridesPolicy::CustomStrides))) {
      // TODO: provide stride_custom, symmetrically with size_custom.
      // There is presently no user for it; only NestedTensor is using
      // size_custom overrideability
      return strides_custom()[d]; // unchecked (maybe_wrap_dim enforces bounds)
    }
    // Intentionally don't call default, which also handles symbolic
    return sizes_and_strides_.stride_at_unchecked(d);
  }

  enum class SizesStridesPolicy : uint8_t {
    // Default behavior, e.g., dense tensor.
    //
    // Can override: nothing
    Default = 0,
    // Customizable strides behavior, e.g., sparse tensor,
    // mkldnn tensor.
    //
    // Can override: strides(), is_contiguous()
    CustomStrides = 1,
    // Customizable sizes behavior, e.g., nested tensor
    //
    // Can override: strides(), is_contiguous(), sizes(), dim(), numel()
    CustomSizes = 2
  };

 protected:
  inline bool matches_policy(SizesStridesPolicy policy) const {
    return sizes_strides_policy_ >= static_cast<uint8_t>(policy);
  }

  inline bool matches_custom(SizesStridesPolicy policy) const {
    return custom_sizes_strides_ >= static_cast<uint8_t>(policy);
  }

  inline bool matches_python_custom(SizesStridesPolicy policy) const {
    auto r = python_custom_sizes_strides_ >= static_cast<uint8_t>(policy);
    if (r) {
      TORCH_INTERNAL_ASSERT(is_python_dispatch())
    }
    return r;
  }

  /**
   * Customization points for the functions above.  sizes_strides_policy_
   * must be set to enable these.
   *
   * NB: dim is overridable separately from sizes because it is possible
   * for a tensor to have rank, but not well defined sizes.
   */
  // sizes_strides_policy_ >= CustomStrides

  virtual bool is_strides_like_custom(at::MemoryFormat memory_format) const;

  virtual c10::SymBool sym_is_non_overlapping_and_dense_custom() const;

  bool is_non_overlapping_and_dense_custom() const {
    return sym_is_non_overlapping_and_dense_custom().guard_bool(
        __FILE__, __LINE__);
  }

  virtual c10::SymBool sym_is_contiguous_custom(
      at::MemoryFormat memory_format) const;

  bool is_contiguous_custom(at::MemoryFormat memory_format) const {
    return sym_is_contiguous_custom(memory_format)
        .guard_bool(__FILE__, __LINE__);
  }

  // sizes_strides_policy_ >= CustomSizes
  // Currently this method only exists to be overwritten by subclasses such as
  // NestedTensorImpl.
  virtual int64_t size_custom(int64_t d) const {
    // TODO: We could add support to Python dispatch here.
    // TODO: We could call into aten::size.int instead of
    // sizes_custom()[d] and enable use of the dispatcher.
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    return sizes_custom()[d]; // unchecked (maybe_wrap_dim enforces bounds)
  }

  virtual c10::SymInt sym_size_custom(int64_t d) const {
    // TODO: We could add support to Python dispatch here.
    // TODO: We could call into aten::size.int instead of
    // sym_sizes_custom()[d] and enable use of the dispatcher.
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    return sym_sizes_custom()[d]; // unchecked (maybe_wrap_dim enforces bounds)
  }

  virtual IntArrayRef sizes_custom() const;
  virtual IntArrayRef strides_custom() const;
  virtual int64_t numel_custom() const;
  virtual int64_t storage_offset_custom() const;
  virtual int64_t dim_custom() const;
  virtual Device device_custom() const;
  virtual Layout layout_custom() const;

  virtual c10::SymIntArrayRef sym_sizes_custom() const;
  virtual c10::SymIntArrayRef sym_strides_custom() const;
  virtual c10::SymInt sym_numel_custom() const;
  virtual c10::SymInt sym_storage_offset_custom() const;

 public:
/**
 * True if this tensor has storage. See storage() for details.
 */
#ifdef DEBUG
  // Allow subclasses to check that their storage_ is never getting set in debug
  // builds.
  virtual
#else
  TENSORIMPL_MAYBE_VIRTUAL
#endif
      bool
      has_storage() const
// NOTE: we devirtualize this because it arguably shouldn't be an
// error just to ask subclasses if they have storage.
// This used to throw for most subclasses, but OpaqueTensorImpl
// wanted it to successfully return false, so we went ahead and made
// it a non-error.
#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  {
    return storage_;
  }
#else
      ;
#endif

  /**
   * Return the underlying storage of a Tensor.  Multiple tensors may share
   * a single storage.  A Storage is an impoverished, Tensor-like class
   * which supports far less operations than Tensor.
   *
   * Avoid using this method if possible; try to use only Tensor APIs to perform
   * operations.
   */
  TENSORIMPL_MAYBE_VIRTUAL const Storage& storage() const {
    if (C10_UNLIKELY(storage_access_should_throw_)) {
      throw_storage_access_error();
    }
    return storage_;
  }

  /**
   * Return the underlying storage, unsafely assuming this is a basic strided
   * tensor. In cases where `storage` access would throw, this returns a
   * default-constructed Storage.
   */
  inline const Storage& unsafe_storage() const {
    return storage_;
  }

  bool unique_version() const {
    return version_counter_.unique();
  }

 protected:
  virtual Layout layout_impl() const {
    TORCH_CHECK(
        false, "layout_impl is only implemented for TensorImpl subclasses.");
  }

 public:
  // Whether a tensor is sparse COO or not.
  bool is_sparse() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has_all(c10::sparse_ks);
  }

  // Whether a tensor is sparse CSR or not.
  bool is_sparse_csr() const {
    return layout() == kSparseCsr;
  }

  // Whether a tensor is sparse CSR/CSC/BSR/BSC or not.
  bool is_sparse_compressed() const {
    return key_set_.has_all(c10::sparse_csr_ks);
  }

  bool is_quantized() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    constexpr auto quantized_ks = DispatchKeySet(DispatchKey::Quantized);
    return key_set_.has_all(quantized_ks);
  }

  bool is_meta() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_meta();
    }
    return device_opt_.has_value() && device_opt_->type() == kMeta;
  }

  bool is_cpu() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_cpu();
    }
    // Note: we cannot rely on dispatch keys to determine the device type
    // of a tensor, because "wrapper" tensors (like FunctionalTensorWrapper)
    // don't include backend dispatch keys.
    return device_opt_.has_value() && device_opt_->type() == kCPU;
  }

  bool is_cuda() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_cuda();
    }
    return device_opt_.has_value() && device_opt_->type() == kCUDA;
  }

  bool is_xpu() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_xpu();
    }
    return device_opt_.has_value() && device_opt_->type() == kXPU;
  }

  bool is_ipu() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_ipu();
    }
    return device_opt_.has_value() && device_opt_->type() == kIPU;
  }

  bool is_xla() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_xla();
    }
    return device_opt_.has_value() && device_opt_->type() == kXLA;
  }

  bool is_mtia() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_mtia();
    }
    return device_opt_.has_value() && device_opt_->type() == kMTIA;
  }

  bool is_hpu() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_hpu();
    }
    return device_opt_.has_value() && device_opt_->type() == kHPU;
  }

  bool is_lazy() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_lazy();
    }
    return device_opt_.has_value() && device_opt_->type() == kLazy;
  }

  bool is_hip() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_hip();
    }
    return device_opt_.has_value() && device_opt_->type() == kHIP;
  }

  bool is_ve() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_ve();
    }
    return device_opt_.has_value() && device_opt_->type() == kVE;
  }

  bool is_privateuseone() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_privateuseone();
    }
    return device_opt_.has_value() && device_opt_->type() == kPrivateUse1;
  }

  bool is_mkldnn() const {
    return key_set_.has_all(c10::mkldnn_ks);
  }

  bool is_vulkan() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_vulkan();
    }
    return device_opt_.has_value() && device_opt_->type() == kVulkan;
  }

  bool is_metal() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_metal();
    }
    return device_opt_.has_value() && device_opt_->type() == kMetal;
  }

  bool is_mps() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_mps();
    }
    return device_opt_.has_value() && device_opt_->type() == kMPS;
  }

  bool is_maia() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().is_maia();
    }
    return device_opt_.has_value() && device_opt_->type() == kMAIA;
  }

  bool is_nested() const {
    return key_set_.has(DispatchKey::NestedTensor);
  }

  // TODO: remove this once we don't automatically enabled Autograd dispatch
  // keys
  //       in TensorImpl constructor.
  // DON'T USE THIS API!! It's only created for testing purpose in
  // file aten/src/ATen/core/boxing/impl/test_helpers.h
  void remove_autograd_key() {
    key_set_ = key_set_ - autograd_dispatch_keyset;
  }

  // Inference tensor doesn't have autograd or ADInplaceOrView key.
  // Invariant:
  //   Inference tensor has version_counter_.enabled() == false
  bool is_inference() {
    bool no_ADInplaceOrView = !key_set_.has_any(c10::inplace_or_view_ks);
    bool no_Autograd = !key_set_.has_any(c10::autograd_dispatch_keyset);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        no_ADInplaceOrView == no_Autograd,
        "ADInplaceOrView and Autograd keys must be on/off at the same time.");
    return no_ADInplaceOrView && no_Autograd;
  }

  DeviceIndex get_device() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom().index();
    }
    return device_default().index();
  }

  Device device() const {
    if (C10_UNLIKELY(device_policy_)) {
      return device_custom();
    }
    return device_default();
  }

 protected:
  c10::Device device_default() const {
    TORCH_CHECK(device_opt_.has_value(), "tensor does not have a device");
    // See NOTE [std::optional operator usage in CUDA]
    return *device_opt_;
  }

 public:
  Layout layout() const {
    if (C10_UNLIKELY(layout_policy_)) {
      return layout_custom();
    }

    // NB: This method is not virtual and avoid dispatches for perf.
    // strided is also the most common layout type, so we check for
    // strided case first.
    // This keyset must also be kept in sync with the logic in
    // is_sparse() / is_sparse_csr() / is_mkldnn()
    constexpr auto sparse_and_sparsecsr_and_mkldnn_ks =
        c10::sparse_ks | c10::sparse_csr_ks | c10::mkldnn_ks;
    if (!key_set_.has_any(sparse_and_sparsecsr_and_mkldnn_ks)) {
      return kStrided;
    } else if (is_sparse()) {
      return kSparse;
    } else if (is_sparse_compressed()) {
      // Typically, the tensor dispatch keys define the tensor layout
      // uniquely. This allows using non-virtual layout method for
      // better performance. However, when tensor's layout depends,
      // say, on tensor attributes, one must use this execution path
      // where the corresponding tensor impl class overwrites virtual
      // layout_impl() method.
      //
      // TODO: implement layout() as native function/method so that
      // __torch_dispatch__ users will be able to redefine the
      // layout() method.
      return layout_impl();
    } else {
      TORCH_INTERNAL_ASSERT(
          is_mkldnn(), "There is an error in the layout calculation logic.");
      return kMkldnn;
    }
  }

  /**
   * True if a tensor was auto-wrapped from a C++ or Python number.
   * For example, when you write 't + 2', 2 is auto-wrapped into a Tensor
   * with `is_wrapped_number_` set to true.
   *
   * Wrapped numbers do not participate in the result type computation for
   * mixed-type operations if there are any Tensors that are not wrapped
   * numbers.  This is useful, because we want 't + 2' to work with
   * any type of tensor, not just LongTensor (which is what integers
   * in Python represent).
   *
   * Otherwise, they behave like their non-wrapped equivalents.
   * See [Result type computation] in TensorIterator.h.
   *
   * Why did we opt for wrapped numbers, as opposed to just having
   * an extra function add(Tensor, Scalar)?  This helps greatly reduce
   * the amount of code we have to write for add, when actually
   * a Tensor-Scalar addition is really just a Tensor-Tensor
   * addition when the RHS is 0-dim (except for promotion behavior.)
   */
  bool is_wrapped_number() const {
    return is_wrapped_number_;
  }

  /**
   * Set whether or not a tensor was auto-wrapped from a C++ or Python
   * number.  You probably don't want to call this, unless you are
   * writing binding code.
   */
  void set_wrapped_number(bool value) {
    TORCH_INTERNAL_ASSERT(dim() == 0);
    is_wrapped_number_ = value;
  }

  /**
   * Returns true if Tensor supports as_strided and as_strided_backward.
   * This is used in autograd to perform inplace update on view Tensors.
   * See Note [View + Inplace update for base tensor] and
   * [View + Inplace update for view tensor] for details.
   * Note this method only returns true for XLA backend, where it
   * simulates strided Tensor to support most view ops, but it cannot
   * fully support general `as_strided` case.
   * It can be expanded as needed in the future, e.g sparse Tensor.
   */
  inline bool support_as_strided() const {
    if (is_nested()) {
      return false;
    }
    if (key_set_.has(DispatchKey::Functionalize)) {
      return false;
    }
    return device().supports_as_strided();
  }

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.

  /**
   * Set whether or not a tensor requires gradient.
   */
  void set_requires_grad(bool requires_grad);

  /**
   * True if a tensor requires gradient.  Tensors which require gradient
   * have history tracked for any operations performed on them, so that
   * we can automatically differentiate back to them.  A tensor that
   * requires gradient and has no history is a "leaf" tensor, which we
   * accumulate gradients into.
   */
  bool requires_grad() const;

  /**
   * Return a mutable reference to the gradient.  This is conventionally
   * used as `t.grad() = x` to set a gradient to a completely new tensor.
   */
  at::Tensor& mutable_grad();

  /**
   * Return the accumulated gradient of a tensor.  This gradient is written
   * into when performing backwards, when this tensor is a leaf tensor.
   */
  const at::Tensor& grad() const;

  /**
   * Whether or not the imaginary part of the tensor should be negated
   */
  inline bool is_conj() const {
    constexpr auto conjugate_ks = DispatchKeySet(DispatchKey::Conjugate);
    return key_set_.has_all(conjugate_ks);
  }

  /**
   * Set whether or not to take the conjugate of the tensor (flip the imaginary
   * bit).
   */
  void _set_conj(bool value) {
    if (value) {
      key_set_ = key_set_.add(DispatchKey::Conjugate);
      TORCH_INTERNAL_ASSERT(isComplexType(typeMetaToScalarType(dtype())));
    } else {
      key_set_ = key_set_.remove(DispatchKey::Conjugate);
    }
  }

  /**
   * XXX: do not use, private api!
   * Update the bac
```



## High-Level Overview


This C++ file contains approximately 14 class(es)/struct(s) and 433 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`, `at`, `impl`

**Classes/Structs**: `Tensor`, `TensorBase`, `C10_API`, `C10_API`, `an`, `C10_API`, `C10_API`, `C10_API`, `C10_API`, `C10_API`, `C10_API`, `VersionCounter`, `has`, `we`, `C10_API`, `itself`, `is`, `C10_API`, `a`, `a`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Allocator.h`
- `c10/core/Device.h`
- `c10/core/DeviceType.h`
- `c10/core/DispatchKey.h`
- `c10/core/DispatchKeySet.h`
- `c10/core/InferenceMode.h`
- `c10/core/Layout.h`
- `c10/core/MemoryFormat.h`
- `c10/core/ScalarType.h`
- `c10/core/ScalarTypeToTypeMeta.h`
- `c10/core/Storage.h`
- `c10/core/SymBool.h`
- `c10/core/SymInt.h`
- `c10/core/SymIntArrayRef.h`
- `c10/core/SymbolicShapeMeta.h`
- `c10/core/WrapDimMinimal.h`
- `c10/core/impl/PyObjectSlot.h`
- `c10/core/impl/SizesAndStrides.h`
- `c10/macros/Export.h`
- `c10/macros/Macros.h`
- `c10/util/ArrayRef.h`
- `c10/util/DimVector.h`
- `c10/util/Exception.h`
- `c10/util/Flags.h`
- `c10/util/accumulate.h`
- `c10/util/intrusive_ptr.h`
- `c10/util/irange.h`
- `c10/util/safe_numerics.h`
- `c10/util/typeid.h`
- `optional`


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`c10/core`):

- [`DispatchKey.cpp_docs.md`](./DispatchKey.cpp_docs.md)
- [`CopyBytes.h_docs.md`](./CopyBytes.h_docs.md)
- [`OptionalRef.h_docs.md`](./OptionalRef.h_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`SafePyObject.cpp_docs.md`](./SafePyObject.cpp_docs.md)
- [`DeviceType.cpp_docs.md`](./DeviceType.cpp_docs.md)
- [`SymBool.cpp_docs.md`](./SymBool.cpp_docs.md)
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`SymbolicShapeMeta.cpp_docs.md`](./SymbolicShapeMeta.cpp_docs.md)


## Cross-References

- **File Documentation**: `TensorImpl.h_docs.md`
- **Keyword Index**: `TensorImpl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
