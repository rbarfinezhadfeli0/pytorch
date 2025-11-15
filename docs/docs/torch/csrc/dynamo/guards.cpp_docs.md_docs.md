# Documentation: `docs/torch/csrc/dynamo/guards.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/dynamo/guards.cpp_docs.md`
- **Size**: 53,733 bytes (52.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/dynamo/guards.cpp`

## File Metadata

- **Path**: `torch/csrc/dynamo/guards.cpp`
- **Size**: 265,913 bytes (259.68 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/autocast_mode.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/util/Exception.h>
#define PY_SSIZE_T_CLEAN
#include <ATen/EmptyTensor.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <c10/util/flat_hash_map.h>
#include <fmt/format.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_symnode.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <torch/extension.h>
#include <cstdint>

#include <torch/csrc/dynamo/debug_macros.h>

#include <nlohmann/json.hpp>

#ifdef USE_CUDA
#include <ATen/cuda/EmptyTensor.h>
#endif

#ifdef USE_XPU
#include <ATen/xpu/EmptyTensor.h>
#endif

#ifdef USE_MTIA
#include <ATen/native/mtia/EmptyTensor.h>
#endif

#include <chrono>
#include <sstream>
#include <tuple>
#include <utility>

// Uncomment next line to count instructions for guard eval.
// #define GUARD_INSTRUCTION_COUNT
#ifdef GUARD_INSTRUCTION_COUNT
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <functional>

int open_counter() {
  perf_event_attr attr{};
  attr.type = PERF_TYPE_HARDWARE;
  attr.size = sizeof(attr);
  attr.config = PERF_COUNT_HW_INSTRUCTIONS; // retired instructions
  attr.disabled = 1; // start stopped
  attr.exclude_kernel = 1; // user-space only
  attr.exclude_hv = 1;

  return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
}

uint64_t count_instructions(const std::function<void()>& fn) {
  int fd = open_counter();
  TORCH_CHECK(fd != -1, "perf_event_open failed");

  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  fn(); // run the code you care about
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

  uint64_t count;
  read(fd, &count, sizeof(count));
  close(fd);
  return count;
}
#endif

// Certain CPython data structures are defined in `.c` files in earlier Python
// versions, e.g., for TupleIteratorGetItemAccessor, we need a fast way to
// retrieve the underlying tuple and access the item. Before Python 3.12
// version, the data structure is in tupleobject.c file -
// https://github.com/python/cpython/blob/9afc6d102d16080535325f645849cd84eb04d57d/Objects/tupleobject.c#L1058-L1062
//
// To handle the older python versions, we manually copy the struct here and
// manually cast it to this new struct. For newer versions, the struct is
// included in the header file.
#if IS_PYTHON_3_12_PLUS

#define Py_BUILD_CORE
#include <internal/pycore_range.h> // _PyRangeIterObject
#include <internal/pycore_tuple.h> // _PyTupleIterObject
#undef Py_BUILD_CORE

#else

// Manually create _PyTupleIterObject struct
typedef struct {
  PyObject_HEAD
  Py_ssize_t it_index;
  PyTupleObject* it_seq; /* Set to NULL when iterator is exhausted */
} _PyTupleIterObject;

// Copied from CPython, and given a unified name for different Python versions.
// https://github.com/python/cpython/blob/7f71003b222ad398713514c2b55d34dc05dba6bc/Objects/rangeobject.c#L765-L771
typedef struct {
  PyObject_HEAD
  // NOTE for Python 3.12+, `index` is removed, and `start` is updated in place
  // instead, upon each `next(...)` call. See
  // https://github.com/python/cpython/pull/27986
  long index;
  long start;
  long step;
  long len;
} _PyRangeIterObject;

#endif // IS_PYTHON_3_12_PLUS

namespace torch::dynamo {

thread_local bool tls_is_in_mode_without_ignore_compile_internals = false;

void set_is_in_mode_without_ignore_compile_internals(bool value) {
  tls_is_in_mode_without_ignore_compile_internals = value;
}

bool get_is_in_mode_without_ignore_compile_internals() {
  return tls_is_in_mode_without_ignore_compile_internals;
}

// Macro to skip addition of duplicate guards like EQUALS_MATCH
#define SKIP_IF_GUARD_ALREADY_PRESENT(name) \
  if (self.is_leaf_guard_present(name)) {   \
    return;                                 \
  }                                         \
  self.insert_leaf_guard(name);

TensorCheck::TensorCheck(
    const LocalState& state,
    PyTypeObject* pt,
    const at::Tensor& v,
    c10::DispatchKeySet dispatch_key_set,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_strides)
    : pytype(pt),
      dispatch_key_(state.apply(dispatch_key_set).raw_repr()),
      dtype_(v.dtype().toScalarType()),
      device_index_(v.device().index()),
      requires_grad_(v.requires_grad()),
      sizes_(std::move(dynamic_dims_sizes)),
      strides_(std::move(dynamic_dims_strides)),
      dim_(static_cast<int64_t>(sizes_.size())) {
  // TODO(voz): In cases where sizes_ and strides_ are fully dynamic, should
  // we just treat this as optional?
}

TensorCheck::TensorCheck(
    const LocalState& state,
    PyTypeObject* pt,
    c10::DispatchKeySet dispatch_key_set,
    at::ScalarType dtype,
    at::DeviceIndex device_index,
    bool requires_grad,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_strides)
    : pytype(pt),
      dispatch_key_(state.apply(dispatch_key_set).raw_repr()),
      dtype_(dtype),
      device_index_(device_index),
      requires_grad_(requires_grad),
      sizes_(std::move(dynamic_dims_sizes)),
      strides_(std::move(dynamic_dims_strides)),
      dim_(static_cast<int64_t>(sizes_.size())) {}

// See note in guards.py [Note - On Export Tensor Guards]
// Logic parallel to here must be maintained in python
bool TensorCheck::check(const LocalState& state, const at::Tensor& v) {
  // In terms of a sparse_csr tensor, it does not support strides information
  c10::SymIntArrayRef sym_strides(std::vector<SymInt>(v.ndimension(), -1));
  bool does_not_support_stride = v.layout() == c10::kSparseCsr ||
      v.layout() == c10::kSparseCsc || v.layout() == c10::kSparseBsc ||
      v.layout() == c10::kSparseBsr;
  if (!does_not_support_stride) {
    sym_strides = v.sym_strides();
  }

  return check(
      state,
      v.key_set(),
      v.dtype().toScalarType(),
      v.device(),
      v.sym_sizes(),
      sym_strides,
      v.requires_grad());
}

bool TensorCheck::check(
    const LocalState& state,
    const c10::DispatchKeySet& dispatch_key_set,
    const at::ScalarType& dtype,
    const c10::Device& device,
    const c10::SymIntArrayRef& sym_sizes,
    const c10::SymIntArrayRef& sym_strides,
    const bool& requires_grad) {
  if (dispatch_key_ != state.apply(dispatch_key_set).raw_repr() ||
      dtype_ != dtype || device_index_ != device.index() ||
      requires_grad_ != requires_grad) {
    return false;
  }

  auto ndim = sym_sizes.size();
  if (ndim != static_cast<size_t>(dim_)) {
    return false;
  }

  const auto& sizes = sym_sizes;
  const auto& strides = sym_strides;
  for (auto i : c10::irange(ndim)) {
    auto known_size = sizes_[i];
    auto known_stride = strides_[i];
    if (known_size.has_value()) {
      if (known_size.value() != sizes[i]) {
        return false;
      }
    }
    if (known_stride.has_value()) {
      if (known_stride.value() != strides[i]) {
        return false;
      }
    }
  }
  return true;
}

std::string TensorCheck::check_verbose(
    const LocalState& state,
    const at::Tensor& v,
    const std::string& tensor_name) {
  std::stringstream fail_reason;
  fail_reason << "tensor '" << tensor_name << "' ";
  if (dispatch_key_ != state.apply(v.key_set()).raw_repr()) {
    // return fmt::format("tensor dispatch key mismatch. expected {}, actual
    // {}", dispatch_key_, state.apply(v.key_set()).raw_repr());
    fail_reason << "dispatch key set mismatch. expected "
                << c10::DispatchKeySet(c10::DispatchKeySet::RAW, dispatch_key_)
                << ", actual " << state.apply(v.key_set());
    return fail_reason.str();
  } else if (dtype_ != v.dtype().toScalarType()) {
    // return fmt::format("tensor dtype mismatch. expected {}, actual {}",
    // dtype_, v.dtype().toScalarType());
    fail_reason << "dtype mismatch. expected " << dtype_ << ", actual "
                << v.dtype().toScalarType();
    return fail_reason.str();
  } else if (device_index_ != v.device().index()) {
    fail_reason << "Tensor device index mismatch. Expected device index to be "
                << device_index_ << ", actual " << v.device().index();
    return fail_reason.str();
  } else if (requires_grad_ != v.requires_grad()) {
    // return fmt::format("tensor requires_grad mismatch. expected {}",
    // requires_grad_);
    fail_reason << "requires_grad mismatch. expected requires_grad="
                << requires_grad_;
    return fail_reason.str();
  }
  auto ndim = v.ndimension();
  if (ndim != dim_) {
    // return fmt::format("tensor rank mismatch. expected {}, actual {}",
    // sizes_.size(), ndim);
    fail_reason << "rank mismatch. expected " << sizes_.size() << ", actual "
                << ndim;
    return fail_reason.str();
  }
  const auto& sizes = v.sym_sizes();
  for (auto i : c10::irange(ndim)) {
    auto known_size = sizes_[i];
    if (known_size.has_value() && (known_size.value() != sizes[i])) {
      fail_reason << "size mismatch at index " << i << ". expected "
                  << known_size.value() << ", actual " << sizes[i];
      return fail_reason.str();
    }
  }
  const bool supports_stride =
      !v.is_sparse() && !at::sparse_csr::is_sparse_compressed(v);
  if (supports_stride) {
    const auto& strides = v.sym_strides();
    for (auto i : c10::irange(ndim)) {
      auto known_stride = strides_[i];
      if (known_stride.has_value() && known_stride.value() != strides[i]) {
        fail_reason << "stride mismatch at index " << i << ". expected "
                    << known_stride.value() << ", actual " << strides[i];
        return fail_reason.str();
      }
    }
  }
  return "";
}

namespace {

typedef std::vector<TensorCheck> ChecksList;

typedef struct {
  PyObject_HEAD
  ChecksList* checks;
} TensorGuards;

static void TensorGuards_dealloc(TensorGuards* self) {
  if (self->checks != nullptr) {
    delete self->checks;
    self->checks = nullptr;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* TensorGuards_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  TensorGuards* self = (TensorGuards*)type->tp_alloc(type, 0);
  if (self != nullptr) {
    self->checks = new ChecksList();
  }
  return (PyObject*)self;
}

static std::vector<std::optional<c10::SymInt>> wrapIntegersInOptional(
    const c10::SymIntArrayRef& intArray) {
  std::vector<std::optional<c10::SymInt>> optVec(intArray.size());
  std::transform(
      intArray.begin(),
      intArray.end(),
      optVec.begin(),
      [](const c10::SymInt& value) { return value; });
  return optVec;
}

static std::vector<std::optional<c10::SymInt>> pyListToVecOptInt(
    PyObject* pyList) {
  std::vector<std::optional<c10::SymInt>> vec;
  Py_ssize_t size = PyList_Size(pyList);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* item = PyList_GetItem(pyList, i);
    auto handle = py::handle(item);
    if (item == Py_None) {
      vec.emplace_back(std::nullopt);
    } else if (torch::is_symint(handle)) {
      vec.emplace_back(py::cast<c10::SymInt>(handle));
    } else {
      int64_t value = PyLong_AsLongLong(item);
      if (value == -1 && PyErr_Occurred()) {
        PyErr_SetString(
            PyExc_TypeError,
            "Size or stride list item is not a valid integer.");
        TORCH_CHECK(false, "Size or stride list item is not a valid integer.");
      }
      vec.emplace_back(c10::SymInt(value));
    }
  }
  return vec;
}

static std::vector<std::vector<std::optional<c10::SymInt>>> get_dynamic_dims(
    PyObject* dynamic_dims_py) {
  std::vector<std::vector<std::optional<c10::SymInt>>> per_tensor_dynamic_dims;
  if (dynamic_dims_py != Py_None) {
    Py_ssize_t size = PyList_Size(dynamic_dims_py);
    for (Py_ssize_t i = 0; i < size; i++) {
      PyObject* py_list = PyList_GetItem(dynamic_dims_py, i);
      std::vector<std::optional<c10::SymInt>> vec = pyListToVecOptInt(py_list);
      per_tensor_dynamic_dims.push_back(std::move(vec));
    }
  }
  return per_tensor_dynamic_dims;
}

static int TensorGuards_init(
    TensorGuards* self,
    PyObject* args,
    PyObject* kwds) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return -1;
  }
  // Top level structure is List[List[Union[int, None]]]
  PyObject* dynamic_dims_sizes_py =
      PyDict_GetItemString(kwds, "dynamic_dims_sizes");
  if (dynamic_dims_sizes_py == nullptr) {
    PyErr_SetString(PyExc_TypeError, "missing dynamic_dims_sizes=...");
    return -1;
  }
  PyObject* dynamic_dims_strides_py =
      PyDict_GetItemString(kwds, "dynamic_dims_strides");
  if (dynamic_dims_strides_py == nullptr) {
    PyErr_SetString(PyExc_TypeError, "missing dynamic_dims_strides=...");
    return -1;
  }

  // dynamic_dims_strides/sizes_py is None when dynamic_shapes=False - this is
  // an optimization to avoid invoking .size()/.stride() in python needlessly
  std::vector<std::vector<std::optional<c10::SymInt>>>
      per_tensor_dynamic_dims_sizes = get_dynamic_dims(dynamic_dims_sizes_py);
  std::vector<std::vector<std::optional<c10::SymInt>>>
      per_tensor_dynamic_dims_strides =
          get_dynamic_dims(dynamic_dims_strides_py);

  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);
  checks.reserve(len);
  LocalState state;

  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "expected Tensor()");
      return -1;
    }
    auto tensor = THPVariable_Unpack(item);
    std::vector<std::optional<c10::SymInt>> tensor_dims_size =
        per_tensor_dynamic_dims_sizes.empty()
        ? wrapIntegersInOptional(tensor.sym_sizes())
        : per_tensor_dynamic_dims_sizes[i];
    std::vector<std::optional<c10::SymInt>> tensor_dims_stride =
        per_tensor_dynamic_dims_strides.empty()
        ? wrapIntegersInOptional(tensor.sym_strides())
        : per_tensor_dynamic_dims_strides[i];

    checks.emplace_back(
        state,
        Py_TYPE(item),
        std::move(tensor),
        tensor.key_set(),
        std::move(tensor_dims_size),
        std::move(tensor_dims_stride));
  }
  return 0;
}

PyObject* TensorGuards_check(
    TensorGuards* self,
    PyObject* args,
    PyObject* kwargs) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return nullptr;
  }
  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);

  // kwargs is just ignored here

  if (static_cast<decltype(len)>(checks.size()) != len) {
    PyErr_SetString(PyExc_TypeError, "wrong length");
    return nullptr;
  }

  LocalState state;
  // Note - all the tensors that make it to guards must be unique. Dynamo
  // builder handles guarding for positive aliases (X is Y). However, we do not
  // create guards for negative alias (X is not Y) as that is an N^2
  // relationship. Instead, we rely on the uniqueness upstream to verify, at
  // check_fn time (this function).
  ska::flat_hash_map<PyObject*, std::nullptr_t> unique_tensors;
  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);

    if (Py_TYPE(item) != checks[i].pytype) {
      Py_RETURN_FALSE;
    }
    auto insertion = unique_tensors.insert({item, nullptr});
    if (!insertion.second) {
      // Violates uniqueness
      Py_RETURN_FALSE;
    }
    if (!checks[i].check(state, THPVariable_Unpack(item))) {
      Py_RETURN_FALSE;
    }
  }

  Py_RETURN_TRUE;
}

PyObject* TensorGuards_check_verbose(
    TensorGuards* self,
    PyObject* args,
    PyObject* kwargs) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return nullptr;
  }
  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);

  if (static_cast<decltype(len)>(checks.size()) != len) {
    PyErr_SetString(PyExc_TypeError, "wrong length");
    return nullptr;
  }

  PyObject* tensor_check_names_py =
      PyDict_GetItemString(kwargs, "tensor_check_names");
  if (tensor_check_names_py == nullptr) {
    PyErr_SetString(PyExc_TypeError, "missing tensor_check_names kwarg");
    return nullptr;
  }

  if (!PyList_Check(tensor_check_names_py)) {
    PyErr_SetString(PyExc_TypeError, "tensor_check_names kwarg must be a list");
    return nullptr;
  }

  auto names_size = PyList_Size(tensor_check_names_py);
  if (names_size != static_cast<decltype(names_size)>(checks.size())) {
    PyErr_SetString(
        PyExc_TypeError,
        "tensor_check_names should be the same size as # tensors");
    return nullptr;
  }

  std::vector<std::string> tensor_check_names;
  tensor_check_names.reserve(names_size);
  for (auto i : c10::irange(names_size)) {
    PyObject* value = PyList_GetItem(tensor_check_names_py, i);
    if (!PyUnicode_Check(value)) {
      PyErr_SetString(
          PyExc_TypeError, "tensor_check_names must only contain strings");
      return nullptr;
    }
    tensor_check_names.emplace_back(PyUnicode_AsUTF8(value));
  }

  LocalState state;
  ska::flat_hash_map<PyObject*, std::nullptr_t> unique_tensors;
  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (Py_TYPE(item) != checks[i].pytype) {
      std::stringstream fail_reason;
      PyObject* type_str = PyObject_Str(PyObject_Type(item));
      fail_reason << "expected type of '" << tensor_check_names[i]
                  << "' to be a tensor type, ";
      if (!type_str) {
        fail_reason << "but found a different type";
      } else {
        fail_reason << "' but found " << PyUnicode_AsUTF8(type_str);
      }
      return Py_BuildValue("s", fail_reason.str().c_str());
    }

    auto insertion = unique_tensors.insert({item, nullptr});
    if (!insertion.second) {
      std::stringstream fail_reason;
      fail_reason << "Duplicate tensor found where not expected! ";
      fail_reason << tensor_check_names[i]
                  << "should not alias to anything, but is aliased";
      return Py_BuildValue("s", fail_reason.str().c_str());
    }
    std::string fail_reason = checks[i].check_verbose(
        state, THPVariable_Unpack(item), tensor_check_names[i]);
    if (!fail_reason.empty()) {
      return Py_BuildValue("s", fail_reason.c_str());
    }
  }

  Py_RETURN_TRUE;
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static PyMethodDef TensorGuards_methods[] = {
    {"check",
     (PyCFunction)(void*)TensorGuards_check,
     METH_VARARGS | METH_KEYWORDS,
     ""},
    {"check_verbose",
     (PyCFunction)(void*)TensorGuards_check_verbose,
     METH_VARARGS | METH_KEYWORDS,
     "verbose fail reasons for failed checks"},
    {nullptr} /* Sentinel */
};

static PyTypeObject TensorGuardsType = {PyVarObject_HEAD_INIT(nullptr, 0)
};

struct AutocastState {
  static constexpr auto& DEVICES = at::autocast::_AUTOCAST_SUPPORTED_DEVICES;
  std::array<bool, DEVICES.size()> enabled{};
  std::array<at::ScalarType, DEVICES.size()> dtype{};
  bool cache_enabled;

  AutocastState() {
    for (size_t i = 0; i < DEVICES.size(); i++) {
      enabled[i] = at::autocast::is_autocast_enabled(DEVICES[i]);
      dtype[i] = at::autocast::get_autocast_dtype(DEVICES[i]);
    }
    cache_enabled = at::autocast::is_autocast_cache_enabled();
  }

  bool operator==(const AutocastState& o) const {
    for (size_t i = 0; i < DEVICES.size(); i++) {
      // If disabled audocast, autocast_dtype comparison not occur
      if (enabled[i] == false && o.enabled[i] == false) {
        continue;
      }
      if (enabled[i] != o.enabled[i] || dtype[i] != o.dtype[i]) {
        return false;
      }
    }
    if (cache_enabled != o.cache_enabled) {
      return false;
    }
    return true;
  }

  template <typename T>
  friend void to_json(T& json_j, const AutocastState& json_t) {
    json_j["enabled"] = json_t.enabled;
    json_j["dtype"] = json_t.dtype;
    json_j["cached_enabled"] = json_t.cache_enabled;
  }

  template <typename T>
  friend void from_json(const T& json_j, AutocastState& json_t) {
    json_t.enabled = json_j.at("enabled");
    json_t.dtype = json_j.at("dtype");
    json_t.cache_enabled = json_j.at("cached_enabled");
  }
};

// TODO (janimesh) - Remove the PyObject_HEAD part when C++ guard manager is
// merged.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct GlobalStateGuard {
  PyObject_HEAD

  void init() {
    auto& ctx = at::globalContext();
    _grad_mode = at::GradMode::is_enabled();
    _autocast_state = AutocastState();
    // The below two flags disambiguate
    // if torch function disabled state is
    // 1) enabled, 2) all disabled, 3) subclasses disabled
    // we guard on the stack separately
    _torch_function = torch::torch_function_enabled();
    _torch_function_all_disabled = at::impl::torch_function_all_disabled();
    _deterministic_algorithms = ctx.deterministicAlgorithms();
    _deterministic_algorithms_warn_only = ctx.deterministicAlgorithmsWarnOnly();
    _allow_tf32 =
        ctx.float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) ==
        at::Float32Precision::TF32;
    _allow_fp16_reduce = ctx.allowFP16ReductionCuBLAS();
    _allow_bf16_reduce = ctx.allowBF16ReductionCuBLAS();
    _num_threads = at::get_num_threads();
    _default_dtype = at::get_default_dtype();
  }

  bool check() const {
    auto& ctx = at::globalContext();
    return (_grad_mode == at::GradMode::is_enabled() &&
            _autocast_state == AutocastState() &&
            _torch_function == torch::torch_function_enabled() &&
            _torch_function_all_disabled ==
                at::impl::torch_function_all_disabled() &&
            _deterministic_algorithms == ctx.deterministicAlgorithms() &&
            _deterministic_algorithms_warn_only ==
                ctx.deterministicAlgorithmsWarnOnly() &&
            _allow_tf32 ==
                (ctx.float32Precision(
                     at::Float32Backend::CUDA, at::Float32Op::MATMUL) ==
                 at::Float32Precision::TF32) &&
            _allow_fp16_reduce == ctx.allowFP16ReductionCuBLAS() &&
            _allow_bf16_reduce == ctx.allowBF16ReductionCuBLAS() &&
            _num_threads == at::get_num_threads()) &&
        _default_dtype == at::get_default_dtype();
  }

  std::string reason() const {
    std::ostringstream os;
    auto& ctx = at::globalContext();
    if (_grad_mode != at::GradMode::is_enabled())
      os << "grad_mode ";
    if (!(_autocast_state == AutocastState()))
      os << "autocast ";
    if (_torch_function != torch::torch_function_enabled())
      os << "torch_function ";
    if (_deterministic_algorithms != ctx.deterministicAlgorithms())
      os << "deterministic_algorithms ";
    if (_deterministic_algorithms_warn_only !=
        ctx.deterministicAlgorithmsWarnOnly())
      os << "deterministic_algorithms_warn_only ";
    if (_allow_tf32 !=
        (ctx.float32Precision(
             at::Float32Backend::CUDA, at::Float32Op::MATMUL) ==
         at::Float32Precision::TF32))
      os << "allow_tf32 ";
    if (_allow_fp16_reduce != ctx.allowFP16ReductionCuBLAS())
      os << "allow_fp16_reduce ";
    if (_allow_bf16_reduce != ctx.allowBF16ReductionCuBLAS())
      os << "allow_bf16_reduce ";
    if (_num_threads != at::get_num_threads())
      os << "num_threads ";
    if (_default_dtype != at::get_default_dtype())
      os << "default_dtype ";
    return os.str();
  }

  template <typename T>
  friend void to_json(T& json_j, const GlobalStateGuard& json_t) {
    json_j["grad_mode"] = json_t._grad_mode;
    json_j["autocast_state"] = json_t._autocast_state;
    json_j["torch_function"] = json_t._torch_function;
    json_j["torch_function_all_disabled"] = json_t._torch_function_all_disabled;
    json_j["deterministic_algorithms"] = json_t._deterministic_algorithms;
    json_j["deterministic_algorithms_warn_only"] =
        json_t._deterministic_algorithms_warn_only;
    json_j["allow_tf32"] = json_t._allow_tf32;
    json_j["allow_fp16_reduce"] =
        static_cast<int64_t>(json_t._allow_fp16_reduce);
    json_j["allow_bf16_reduce"] =
        static_cast<int64_t>(json_t._allow_bf16_reduce);
    json_j["num_threads"] = json_t._num_threads;
    json_j["default_dtype"] = json_t._default_dtype.toScalarType();
  }

  template <typename T>
  friend void from_json(const T& json_j, GlobalStateGuard& json_t) {
    json_t._grad_mode = json_j.at("grad_mode");
    json_t._autocast_state = json_j.at("autocast_state");
    json_t._torch_function = json_j.at("torch_function");
    json_t._torch_function_all_disabled =
        json_j.at("torch_function_all_disabled");
    json_t._deterministic_algorithms = json_j.at("deterministic_algorithms");
    json_t._deterministic_algorithms_warn_only =
        json_j.at("deterministic_algorithms_warn_only");
    json_t._allow_tf32 = json_j.at("allow_tf32");
    json_t._allow_fp16_reduce = static_cast<at::CuBLASReductionOption>(
        static_cast<int64_t>(json_j.at("allow_fp16_reduce")));
    json_t._allow_bf16_reduce = static_cast<at::CuBLASReductionOption>(
        static_cast<int64_t>(json_j.at("allow_bf16_reduce")));
    json_t._num_threads = json_j.at("num_threads");
    json_t._default_dtype =
        caffe2::TypeMeta::fromScalarType(json_j.at("default_dtype"));
  }

  bool _grad_mode;
  AutocastState _autocast_state;
  bool _torch_function;
  bool _torch_function_all_disabled;
  bool _deterministic_algorithms;
  bool _deterministic_algorithms_warn_only;
  bool _allow_tf32;
  at::CuBLASReductionOption _allow_fp16_reduce;
  at::CuBLASReductionOption _allow_bf16_reduce;
  int _num_threads;
  caffe2::TypeMeta _default_dtype;
  // TODO(jansel): we should guard on more state as inductor starts using it
};

int GlobalStateGuard_init(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  self->init();
  return 0;
}

PyObject* GlobalStateGuard_check(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  if (self->check()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject* GlobalStateGuard_reason(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  return PyUnicode_FromString(self->reason().c_str());
}

PyObject* GlobalStateGuard_dump(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  return PyUnicode_FromString(nlohmann::json(*self).dump().c_str());
}

PyObject* GlobalStateGuard_load(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  char* json;
  TORCH_CHECK(
      PyArg_ParseTuple(args, "s", &json), "Cannot parse as json string.");
  nlohmann::json::parse(json).get_to(*self);
  Py_RETURN_NONE;
}

// NOLINTNEXTLINE(*array*)
static PyMethodDef GlobalStateGuard_methods[] = {
    {"check",
     (PyCFunction)(void*)GlobalStateGuard_check,
     METH_NOARGS,
     "Return true if global state was the same as at creation time"},
    {"reason",
     (PyCFunction)(void*)GlobalStateGuard_reason,
     METH_NOARGS,
     "Return string reason for guard check failing"},
    {"__getstate__",
     (PyCFunction)(void*)GlobalStateGuard_dump,
     METH_NOARGS,
     "Return serialized json format"},
    {"__setstate__",
     (PyCFunction)(void*)GlobalStateGuard_load,
     METH_VARARGS,
     "Parse serialized json format"},
    {nullptr}};
static PyTypeObject GlobalStateGuardType = {PyVarObject_HEAD_INIT(nullptr, 0)
};

static PyObject* check_type_id(PyObject* dummy, PyObject* args) {
  // faster `lambda obj, expected: id(type(obj)) == expected`
  PyObject* obj = nullptr;
  unsigned long long expected = 0;
  if (!PyArg_ParseTuple(args, "OK", &obj, &expected)) {
    return nullptr;
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  if (Py_TYPE(obj) == (void*)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject* check_obj_id(PyObject* dummy, PyObject* args) {
  // faster `lambda obj, expected: id(obj) == expected`
  PyObject* obj = nullptr;
  unsigned long long expected = 0;
  if (!PyArg_ParseTuple(args, "OK", &obj, &expected)) {
    return nullptr;
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  if (obj == (void*)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

#if IS_PYTHON_3_12_PLUS

static std::unordered_map<PyObject*, uint64_t> dict_version_map;
static int dict_version_watcher_id;
static int dict_recursive_tag_watcher_id;
static uint64_t global_dict_version_id = 1;
static int dict_version_watch_callback(
    PyDict_WatchEvent event,
    PyObject* dict,
    PyObject* key,
    PyObject* new_value) noexcept {
  if (event == PyDict_EVENT_DEALLOCATED) {
    dict_version_map.erase(dict);
  } else if (event != PyDict_EVENT_CLONED) {
    dict_version_map[dict] = global_dict_version_id++;
  }
  return 0;
}

#endif

static uint64_t get_dict_version_unchecked(PyObject* dict) {
#if IS_PYTHON_3_12_PLUS

  TORCH_CHECK(
      !PyDict_Watch(dict_version_watcher_id, dict),
      "failed to add version watcher to dict!");
  if (!dict_version_map.count(dict)) {
    dict_version_map[dict] = global_dict_version_id++;
  }
  return dict_version_map[dict];

#else

  return ((PyDictObject*)dict)->ma_version_tag;

#endif
}

static PyObject* dict_version(PyObject* dummy, PyObject* args) {
  // Retrieves the version of a dictionary.
  PyObject* obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return nullptr;
  }
  if (!PyDict_Check(obj)) {
    return nullptr;
  }
  return THPUtils_packUInt64(get_dict_version_unchecked(obj));
}

static PyObject* assert_size_stride(PyObject* dummy, PyObject* args) {
  /*
   Assert that a given tensor has a given size/stride, but ignore strides
   of size==1 dimensions.  Implemented in C++ as this is on the hot path.
  */
  PyObject* item = nullptr;
  PyObject* size = nullptr;
  PyObject* stride = nullptr;
  const char* op_name = nullptr;

  if (!PyArg_ParseTuple(args, "OOO|s", &item, &size, &stride, &op_name)) {
    return nullptr;
  }
  if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
    std::stringstream msg;
    msg << "expected Tensor()";
    if (op_name) {
      msg << " for op: " << op_name;
    }
    PyErr_SetString(PyExc_TypeError, msg.str().c_str());
    return nullptr;
  }
  if (!PyTuple_CheckExact(size) || !PyTuple_CheckExact(stride)) {
    std::stringstream msg;
    msg << "expected tuple()";
    if (op_name) {
      msg << " for op: " << op_name;
    }
    PyErr_SetString(PyExc_TypeError, msg.str().c_str());
    return nullptr;
  }
  at::Tensor tensor = THPVariable_Unpack(item);
  int64_t ndim = tensor.ndimension();
  if (PyTuple_GET_SIZE(size) != ndim || PyTuple_GET_SIZE(stride) != ndim) {
    std::stringstream msg;
    msg << "wrong number of dimensions" << ndim;
    if (op_name) {
      msg << " for op: " << op_name;
    }
    PyErr_SetString(PyExc_AssertionError, msg.str().c_str());
    return nullptr;
  }

  // We may add the size/stride assert at compile time due to unbacked symint,
  // but at runtime, the tensor can be empty.
  if (tensor.numel() == 0) {
    Py_RETURN_TRUE;
  }

  std::stringstream msg;
  int num_errors = 0;
  for (auto i : c10::irange(ndim)) {
    int64_t want_size = THPUtils_unpackLong(PyTuple_GET_ITEM(size, i));
    int64_t want_stride = THPUtils_unpackLong(PyTuple_GET_ITEM(stride, i));
    int64_t actual_size = tensor.size(i);
    int64_t actual_stride = tensor.stride(i);
    if (want_size != actual_size ||
        // ignore stride differences when size is 1
        (want_stride != actual_stride && actual_size > 1)) {
      if (num_errors > 0)
        msg << "; ";
      msg << "expected size " << actual_size << "==" << want_size << ", stride "
          << actual_stride << "==" << want_stride << " at dim=" << i;
      num_errors++;
    }
  }

  if (num_errors) {
    if (op_name) {
      msg << "\nError in op: " << op_name;
    }
    msg << "\nThis error most often comes from a incorrect fake (aka meta) kernel for a custom op.";
    msg << "\nUse torch.library.opcheck to test your custom op.";
    msg << "\nSee https://pytorch.org/docs/stable/library.html#torch.library.opcheck";
    PyErr_SetString(PyExc_AssertionError, msg.str().c_str());
    return nullptr;
  }

  Py_RETURN_TRUE;
}

static PyObject* assert_alignment(PyObject* dummy, PyObject* args) {
  /*
   * Asserts that a given tensor meets certain alignment.
   * This C++ version of torch._inductor.utils.tensor_is_aligned
   */
  PyObject* item = nullptr;
  unsigned long alignment = 0;
  const char* op_name = nullptr;

  if (!PyArg_ParseTuple(args, "Ok|s", &item, &alignment, &op_name)) {
    return nullptr;
  }
  if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
    std::stringstream msg;
    msg << "expected Tensor()";
    if (op_name) {
      msg << " for op: " << op_name;
    }
    PyErr_SetString(PyExc_TypeError, msg.str().c_str());
    return nullptr;
  }
  if (alignment == 0) {
    std::stringstream msg;
    msg << "alignment cannot be 0";
    if (op_name) {
      msg << " in op: " << op_name;
    }
    PyErr_SetString(PyExc_AssertionError, msg.str().c_str());
    return nullptr;
  }

  at::Tensor tensor = THPVariable_Unpack(item);

  int64_t storage_offset = tensor.storage_offset();
  size_t itemsize = tensor.itemsize();
  if (storage_offset * itemsize % alignment != 0) {
    std::stringstream msg;
    if (op_name) {
      msg << "\nError in op: " << op_name;
    }
    msg << "\nExpect the tensor to be " << alignment
        << " bytes aligned. Fail due to storage_offset=" << storage_offset
        << " itemsize=" << itemsize;
    PyErr_SetString(PyExc_AssertionError, msg.str().c_str());
    return nullptr;
  }

  Py_RETURN_TRUE;
}

template <typename T>
static void unwrap_size_tuple(PyObject* obj, T& output) {
  TORCH_CHECK(PyTuple_CheckExact(obj));
  size_t len = PyTuple_GET_SIZE(obj);
  output.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(obj, i));
    TORCH_CHECK(result >= 0);
    output.emplace_back(result);
  }
}

template <typename T>
static void _parse_empty_strided_args(
    PyObject* args,
    T& sizes,
    T& strides,
    at::ScalarType& dtype) {
  TORCH_CHECK(PyTuple_CheckExact(args));
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 3);
  // note PyTuple_GET_ITEM returns a borrowed ref, so no need for refcounts
  unwrap_size_tuple(PyTuple_GET_ITEM(args, 0), sizes);
  unwrap_size_tuple(PyTuple_GET_ITEM(args, 1), strides);
  PyObject* py_dtype = PyTuple_GET_ITEM(args, 2);
  TORCH_CHECK(THPDtype_Check(py_dtype));
  dtype = reinterpret_cast<THPDtype*>(py_dtype)->scalar_type;
}

static PyObject* _empty_strided_device(
    PyObject* dummy,
    PyObject* args,
    c10::DeviceType device_type,
    bool is_pinned = false) {
  HANDLE_TH_ERRORS;
  at::SmallVector<int64_t, 8> sizes;
  at::SmallVector<int64_t, 8> strides;
  at::ScalarType dtype{at::ScalarType::Undefined};
  _parse_empty_strided_args(args, sizes, strides, dtype);
  if (device_type == c10::DeviceType::CPU) {
    return THPVariable_Wrap(
        at::detail::empty_strided_cpu(sizes, strides, dtype, is_pinned));
  }
#ifdef USE_CUDA
  else if (device_type == c10::DeviceType::CUDA) {
    return THPVariable_Wrap(at::detail::empty_strided_cuda(
        sizes, strides, dtype, c10::DeviceType::CUDA));
  }
#endif
#ifdef USE_XPU
  else if (device_type == c10::DeviceType::XPU) {
    return THPVariable_Wrap(at::detail::empty_strided_xpu(
        sizes, strides, dtype, c10::DeviceType::XPU));
  }
#endif
#ifdef USE_MTIA
  else if (device_type == c10::DeviceType::MTIA) {
    return THPVariable_Wrap(at::detail::empty_strided_mtia(
        sizes, strides, dtype, c10::DeviceType::MTIA));
  }
#endif
  else {
    TORCH_CHECK(
        false, "PyTorch compiled without support for the specified device.");
  }

  END_HANDLE_TH_ERRORS;
}

static PyObject* _empty_strided_cpu(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is a lower-overhead
  // version that saves ~2us on every allocation.
  return _empty_strided_device(dummy, args, c10::DeviceType::CPU);
}

static PyObject* _empty_strided_cpu_pinned(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is a lower-overhead
  // version that saves ~2us on every allocation.
  return _empty_strided_device(
      dummy, args, c10::DeviceType::CPU, /*is_pinned=*/true);
}

static PyObject* _empty_strided_cuda(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is lower-overhead.
  return _empty_strided_device(dummy, args, c10::DeviceType::CUDA);
}

static PyObject* _empty_strided_xpu(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is lower-overhead.
  return _empty_strided_device(dummy, args, c10::DeviceType::XPU);
}

static PyObject* _empty_strided_mtia(PyObject* dummy, PyObject* args) {
  return _empty_strided_device(dummy, args, c10::DeviceType::MTIA);
}

static PyObject* _reinterpret_tensor(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS;
  static PythonArgParser parser(
      {"_reinterpret_tensor(Tensor base, IntArrayRef sizes, IntArrayRef strides, int64_t offset_increment=0)"},
      /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, /*kwargs=*/nullptr, parsed_args);

  Tensor self = r.tensor(0);
  auto sizes = r.intlist(1);
  auto strides = r.intlist(2);
  auto offset_increment = r.toInt64(3);

  auto res = torch::inductor::_reinterpret_tensor(
      self, sizes, strides, offset_increment);
  return torch::autograd::utils::wrap(res);

  END_HANDLE_TH_ERRORS;
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static PyMethodDef _methods[] = {
    {"check_type_id", check_type_id, METH_VARARGS, nullptr},
    {"check_obj_id", check_obj_id, METH_VARARGS, nullptr},
    {"assert_size_stride", assert_size_stride, METH_VARARGS, nullptr},
    {"assert_alignment", assert_alignment, METH_VARARGS, nullptr},
    {"dict_version", dict_version, METH_VARARGS, nullptr},
    {"_empty_strided_cpu", _empty_strided_cpu, METH_VARARGS, nullptr},
    {"_empty_strided_cpu_pinned",
     _empty_strided_cpu_pinned,
     METH_VARARGS,
     nullptr},
    {"_empty_strided_cuda", _empty_strided_cuda, METH_VARARGS, nullptr},
    {"_empty_strided_xpu", _empty_strided_xpu, METH_VARARGS, nullptr},
    {"_empty_strided_mtia", _empty_strided_mtia, METH_VARARGS, nullptr},
    {"_reinterpret_tensor", _reinterpret_tensor, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.guards",
    "Module containing checks on tensors",
    -1,
    _methods};

std::string get_exception_message() {
  PyObject *ptype = nullptr, *pvalue = nullptr, *ptraceback = nullptr;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);

  PyObject* exc_message_pyobj = PyObject_Str(pvalue);
  std::string exc_message = PyUnicode_AsUTF8(exc_message_pyobj);

  Py_DECREF(exc_message_pyobj);
  Py_XDECREF(ptype);
  Py_XDECREF(pvalue);
  Py_XDECREF(ptraceback);
  return exc_message;
}

bool is_immutable_object(py::handle example_value) {
  py::object config_module = py::module_::import("torch._dynamo.config");

  bool is_tensor_immutable =
      config_module.attr("skip_tensor_guards_with_matching_dict_tags")
          .cast<bool>();

  if (PyTuple_Check(example_value.ptr())) {
    // Check that each element is immutable
    for (Py_ssize_t i = 0; i < PyTuple_Size(example_value.ptr()); ++i) {
      if (!is_immutable_object(
              py::handle(PyTuple_GetItem(example_value.ptr(), i)))) {
        return false;
      }
    }
    return true;
  }

  return (example_value.ptr() == Py_None) ||
      PyLong_Check(example_value.ptr()) || PyFloat_Check(example_value.ptr()) ||
      PyBool_Check(example_value.ptr()) ||
      PyUnicode_Check(example_value.ptr()) ||
      PyCode_Check(example_value.ptr()) ||
      (Py_TYPE(example_value.ptr()) == &PyCFunction_Type) ||
      (is_tensor_immutable && THPVariable_Check(example_value.ptr()));
}

bool is_parameter(py::handle tensor) {
  py::object parameter = py::module::import("torch.nn").attr("Parameter");
  return py::isinstance(tensor, parameter);
}

/**
 * Dispatches metadata functions to the methods that return integer values,
 * i.e. used whenever static shapes are being used.
 *
 * These are used by the tensor storage overlapping check. Even though their
 * symbolic counterpart does work whenever static shapes are being used, the
 * introduced overhead might significantly worsen the performance.
 */
struct StaticMeta {
  static int64_t numel(const Tensor& t) {
    return t.numel();
  }

  static int64_t storage_offset(const Tensor& t) {
    return t.storage_offset();
  }

  static int64_t size(const Tensor& t, int64_t i) {
    return t.size(i);
  }

  static int64_t stride(const Tensor& t, int64_t i) {
    return t.stride(i);
  }
};

/**
 * Dispatches metadata functions to the methods that return c10::SymInt
 * values, i.e. used whenever dynamic shapes are being used.
 */
struct DynamicMeta {
  static SymInt numel(const Tensor& t) {
    return t.sym_numel();
  }

  static SymInt storage_offset(const Tensor& t) {
    return t.sym_storage_offset();
  }

  static SymInt size(const Tensor& t, int64_t i) {
    return t.sym_size(i);
  }

  static SymInt stride(const Tensor& t, int64_t i) {
    return t.sym_stride(i);
  }
};

/**
 * Assumption: x and y are known to share a storage, and we are trying to
 * determine if their memory is actually completely disjoint, based on
 * sizes/strides/storage_offset
 *
 * "Meta" should be one of the "*Meta" classes above. They dictate which
 * version of the metadata functions we should be using (symbolic vs.
 * concrete). Even though they have the same apparent behavior, the symbolic
 * version introduces a bit of overhead. Such an overhead might end up
 * becoming relevant if it's run enough times.
 */
template <class Meta>
bool tensors_definitely_do_not_overlap(const Tensor& x, const Tensor& y) {
  if (x.is_same(y)) {
    return false;
  }
  if (Meta::numel(x) == 0 || Meta::numel(y) == 0) {
    return true;
  }

  // Make x always on the left
  if (Meta::storage_offset(x) > Meta::storage_offset(y)) {
    return tensors_definitely_do_not_overlap<Meta>(y, x);
  }

  // Short-circuit in the "obvious" overlapping case: both tensors are
  // contiguous
  if (x.is_contiguous() && y.is_contiguous()) {
    if (Meta::storage_offset(x) + Meta::numel(x) > Meta::storage_offset(y)) {
      // definitely overlap
      return false;
    } else {
      // definitely no overlap
      return true;
    }
  }

  // Short-circuit: if last memory address of x is < start of y, then not
  // overlapping.
  auto x_last = Meta::storage_offset(x);
  for (int64_t i = 0; i < x.dim(); i++) {
    x_last += (Meta::size(x, i) - 1) * Meta::stride(x, i);
  }
  if (x_last < Meta::storage_offset(y)) {
    return true;
  }

  if (x.dim() == 2 && y.dim() == 2 && Meta::stride(x, 1) == 1 &&
      Meta::stride(y, 1) == 1) {
    // This cases is needed for the shampoo optimizer.
    // All tensors are 2d (non-contiguous), have the same outer stride, and have
    // an inner stride of 1 (so rows are contiguous)
    if (Meta::stride(x, 0) == Meta::stride(y, 0)) {
      auto offset_delta = Meta::storage_offset(y) - Meta::storage_offset(x);
      if (offset_delta < Meta::size(x, 1)) {
        // definitely overlaps (row 0 of y overlaps with row 0 of x)
        // Example:
        //   base = torch.arange(32).reshape(4, 8)
        //   x = base.narrow(1, 0, 4)
        //     x: size=(4, 4), stride=(8, 1), offset=0
        //   y = base.narrow(1, 3, 4)
        //     y: size=(4, 4), stride=(8, 1), offset=3
        return false;
      }
      auto x_total_elems_covered =
          Meta::stride(x, 0) * (Meta::size(x, 0) - 1) + Meta::size(x, 1);
      if (x_total_elems_covered <= offset_delta) {
        // definitely does not overlap (last byte of x is before start of y)
        // Example:
        //   x: size=(4, 4), stride=(8, 1), offset=0 (last byte is 27)
        //   y: size=(4, 4), stride=(8, 1), offset=28 (start byte is 28)
        return true;
      }
      // At this point, we want to check if the 0th row of y
      // overlaps with **some** row of x.
      // We can check this by shifting y backward by the shared stride,
      // repeatedly, until the first row of y is before the first row of x. Then
      // we can check if these rows overlap. We can accomplish this by modding
      // our offset by the stride.
      auto offset_delta_mod = offset_delta % Meta::stride(x, 0);
      // Example:
      // 0 1 2 3
      // 9 10 11 12
      // 18 19 20 21
      // 27 28 29 30
      //   x: size=(4, 4), stride=(9, 1), offset=0
      //   y: size=(4, 4), stride=(9, 1), offset=22 (this would not overlap)
      //   y: size=(4, 4), stride=(9, 1), offset=23 (this would not overlap)
      //   y: size=(4, 4), stride=(9, 1), offset=24 (this would overlap)
      //   y: size=(4, 4), stride=(9, 1), offset=25 (this would overlap)
      // If the interval [modded_offset, modded_offset + x_size] falls entirely
      // without
      if (offset_delta_mod + Meta::size(y, 1) <= Meta::stride(x, 0)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Computes the indices of the tensors that might overlap.
 *
 * Checks which of the given tensors have overlapping storages with ANY of
 * the other tensors.
 *
 * So, for example, if tensor 1 overlaps with tensor 2, and tensor 3 with
 * tensor 4, all of them will be in the output of this function. Even if
 * tensor 1 and 4 don't overlap.
 */
template <class Meta>
std::unordered_set<int64_t> compute_overlapping_tensors(
    const std::vector<Tensor>& tensors) {
  std::unordered_set<int64_t> aliased_tensor_indices;
  for (int64_t i = 0; i < static_cast<int64_t>(tensors.size()); i++) {
    const auto& tensor_i = tensors[i];
    for (int64_t j = 0; j < i; j++) {
      if (!tensors_definitely_do_not_overlap<Meta>(tensor_i, tensors[j])) {
        aliased_tensor_indices.insert(i);
        aliased_tensor_indices.insert(j);
      }
    }
  }
  return aliased_tensor_indices;
}

/**
 * Checks whether the storage overlapping relation is preserved.
 *
 * At this point, `non_overlapping` represents the tensors that should not
 * have overlapping storages. Similarly, `overlapping` represents the tensors
 * that should have overlapping storage in some way (or that we can't be sure).
 *
 * This function checks whether the assumption above is true or not.
 */
bool check_overlapping(
    const std::vector<Tensor>& overlapping,
    const std::vector<Tensor>& non_overlapping) {
  // Merge the tensor lists.
  std::vector<Tensor> tensors;
  tensors.reserve(overlapping.size() + non_overlapping.size());
  tensors.insert(tensors.end(), overlapping.begin(), overlapping.end());
  tensors.insert(tensors.end(), non_overlapping.begin(), non_overlapping.end());
  // Check what is the current storage overlapping relation.
  auto indices = compute_overlapping_tensors<StaticMeta>(tensors);
  // Check that the set of indices of tensors that might overlap is equal to
  // the indices of the first `overlapping.size()` tensors. That's because
  // `overlapping` tensors were in the beginning of `tensors` list.
  auto range = c10::irange(overlapping.size());
  return indices.size() == overlapping.size() &&
      std::all_of(range.begin(), range.end(), [&](int64_t i) {
           return indices.count(i) == 1;
         });
}

/**
 * Class responsible for collecting and checking the storage overlap relations.
 *
 * The way GuardManager is implemented, when STORAGE_OVERLAPPING guard check is
 * run on a given tensor, we don't know if it is an overlapping or
 * non-overlapping tensor. There's no order to which GuardManager runs the guard
 * check so that we can split it in 2.
 *
 * Since we are only interested in the classification of each tensor (not
 * necessarily the order), we can just issue 2 STORAGE_OVERLAPPING guards
 * representing the overlapping tensors and the non-overlapping ones.
 *
 * In order to collect the information from both guards (so that we can call
 * `check_overlapping` function correctly), we need this class which stores
 * both kinds of tensors, and knows when it has collected each one of them.
 */
class StorageOverlapChecker {
 public:
  StorageOverlapChecker(
      size_t expected_overlapping,
      size_t expected_non_overlapping)
      : _expected_overlapping(expected_overlapping),
        _expected_non_overlapping(expected_non_overlapping) {}

  /**
   * Adds a tensor to the corresponding storage, based on whether it should be
   * an `overlapping` tensor or not.
   */
  void add(PyObject* obj, bool overlapping) {
    // Just check that `obj` is actually a tensor, so that we can keep it alive
    // by incrementing its ref-count.
    TORCH_CHECK(THPVariable_CheckExact(obj) || THPVariable_Check(obj));
    Py_INCREF(obj);
    _get(overlapping).push_back(obj);
  }

  void reset(bool overlapping) {
    auto& vec = _get(overlapping);
    for (auto item : vec) {
      Py_DECREF(item);
    }
    vec.clear();
  }

  /**
   * Maybe checks the storage overlapping relation.
   *
   * Before actually calling `check_overlapping` function, this function makes
   * sure 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/csrc/dynamo`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`cpython_includes.h_kw.md_docs.md`](./cpython_includes.h_kw.md_docs.md)
- [`stackref_bridge.c_docs.md_docs.md`](./stackref_bridge.c_docs.md_docs.md)
- [`eval_frame.c_docs.md_docs.md`](./eval_frame.c_docs.md_docs.md)
- [`extra_state.h_docs.md_docs.md`](./extra_state.h_docs.md_docs.md)
- [`cache_entry.h_kw.md_docs.md`](./cache_entry.h_kw.md_docs.md)
- [`compiled_autograd.h_docs.md_docs.md`](./compiled_autograd.h_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`extra_state.h_kw.md_docs.md`](./extra_state.h_kw.md_docs.md)
- [`extra_state.cpp_kw.md_docs.md`](./extra_state.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `guards.cpp_docs.md_docs.md`
- **Keyword Index**: `guards.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
