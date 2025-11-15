# Documentation: guards.cpp

## File Metadata
- **Path**: `torch/csrc/dynamo/guards.cpp`
- **Size**: 265861 bytes
- **Lines**: 7852
- **Extension**: .cpp
- **Type**: Regular file

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
   * sure it has collected all expected tensors.
   */
  bool maybe_check() {
    TORCH_CHECK(_expected_overlapping >= _overlapping.size());
    TORCH_CHECK(_expected_non_overlapping >= _non_overlapping.size());
    if (_expected_overlapping == _overlapping.size() &&
        _expected_non_overlapping == _non_overlapping.size()) {
      // Transform each list of PyObject* into an actual list of Tensors.
      auto overlapping_tensors =
          _tensors_from(_overlapping, _expected_overlapping);
      auto non_overlapping_tensors =
          _tensors_from(_non_overlapping, _expected_non_overlapping);
      return check_overlapping(overlapping_tensors, non_overlapping_tensors);
    } else {
      // If we haven't collected them all yet, keep on running.
      return true;
    }
  }

 private:
  /**
   * Returns a reference to the container that corresponds to the given
   * overlapping relation.
   */
  std::vector<PyObject*>& _get(bool overlapping) {
    return overlapping ? _overlapping : _non_overlapping;
  }

  /**
   * Transforms a given list of PyObject* into a list of Tensor.
   */
  std::vector<Tensor> _tensors_from(
      const std::vector<PyObject*>& objects,
      size_t size) {
    std::vector<Tensor> tensors;
    tensors.reserve(size);
    std::transform(
        objects.begin(),
        objects.end(),
        std::back_inserter(tensors),
        [=](PyObject* obj) { return THPVariable_Unpack(obj); });
    return tensors;
  }

  // Expected number of possibly overlapping tensors.
  size_t _expected_overlapping;
  // Expected number of non-overlapping tensors.
  size_t _expected_non_overlapping;
  // Collected possibly overlapping tensors.
  std::vector<PyObject*> _overlapping;
  // Collected non-overlapping tensors.
  std::vector<PyObject*> _non_overlapping;
};

/**
 * Stores relevant guard debug information, e.g., failure str for a LeafGuard
 * failure. The data structure is also accessible in Python.
 */

class GuardDebugInfo {
 public:
  GuardDebugInfo(
      bool result,
      py::list verbose_code_parts,
      int num_guards_executed)
      : result(result),
        verbose_code_parts(std::move(verbose_code_parts)),
        num_guards_executed(num_guards_executed) {}

  // This constructor is used when guard succeeds.
  GuardDebugInfo(bool result, int num_guards_executed)
      : result(result), num_guards_executed(num_guards_executed) {}

  GuardDebugInfo(
      bool result,
      const std::string& failed_reason,
      int num_guards_executed)
      : GuardDebugInfo(result, num_guards_executed) {
    verbose_code_parts.append(failed_reason);
  }

  std::string to_string() {
    std::stringstream ss;
    ss << "GuardDebugInfo(\n"
       << "result=" << result << ",\n"
       << "verbose_code_parts=" << verbose_code_parts << ",\n"
       << "num_guards_executed=" << num_guards_executed << ")\n";
    return ss.str();
  }

  // Whether the guard passed or failed.
  bool result;

  // This is a list of verbose_code_parts for the failed guard. When there are
  // more than one verbose_code_parts, then recompilation reasoning infra on the
  // Python side can iterate over this list and eval each string to pinpoint the
  // exact code part that failed.
  py::list verbose_code_parts;

  // Total number of executed guards so far. This is helpful in debugging if
  // shuffling is working.
  int num_guards_executed;
};

class GuardManager;
class RootGuardManager;
class DictGuardManager;

// Global registry used by the *recursive-dict-tag* optimisation.
//
// Key   : `PyObject*` pointing to a watched `dict`
// Value : list of `GuardManager*` instances that have recorded that dict
//
// Why is this global?
// -------------------
// * CPython allows only a small, fixed number of dict-watcher IDs (≈64).
//   All `GuardManager`s therefore share a single watcher callback.
// * Different guard managers (possibly across different frames) can end up
//   watching the same dictionary pointer. Therefore, we have a list of guard
//   managers for each dict pointer.
//
// When is watch registered?
//  * During the recording phase of recursive dict tag matching in GuardManager.
//
// When are they watched?
//  * In the dict_recursive_tag_watch_callback function.
//
// When are the dict pointers unwatched?
//  * If a dict is mutated or the guard manager deallocates.
//  * Read `unwatch_all_saved_dict_pointers` docstring for more details.
//
// Expected size
// -------------
// Every compilation frame contributes its tag-safe dicts to this registry, so
// the container can grow large over the lifetime of the process.  That’s
// acceptable: lookup is by pointer (hash/equals = identity) and each entry
// stores only lightweight pointers.
std::unordered_map<PyObject*, std::list<GuardManager*>> dict_to_guard_managers;

/**
 * Base class for the leaf guard in the GuardManager hierarchy.
 */
class LeafGuard {
 public:
  LeafGuard(RootGuardManager* root_guard_manager, py::object verbose_code_parts)
      : _root_guard_manager(root_guard_manager),
        _verbose_code_parts(std::move(verbose_code_parts)) {}

  // check function could be called from python. This is useful for debugging
  // purpose.
  bool check(py::handle value) {
    return check_nopybind(value.ptr());
  }

  GuardDebugInfo check_verbose(py::handle value) {
    return check_verbose_nopybind(value.ptr());
  }

  virtual GuardDebugInfo check_verbose_nopybind(
      PyObject* value) { // borrowed ref
    bool result = check_nopybind(value);
    if (!result) {
      return GuardDebugInfo(result, _verbose_code_parts, 0);
    }
    return GuardDebugInfo(true, 0);
  }

  py::list verbose_code_parts() {
    return _verbose_code_parts;
  }

  // This is on the hot path and avoids any refcounting code from pybind. This
  // is not exposed to Python and can only be called from C++.
  virtual bool check_nopybind(PyObject* value) = 0;
  virtual bool check_nopybind(FrameLocalsMapping* map) {
    // throw std::runtime_error("fallback to python");
    // Could fallback to running check on the Python dict (lazily constructed)
    return check_nopybind((PyObject*)map->to_dict());
  }

  virtual ~LeafGuard() = default;

 protected:
  // RootGuardManager has state that is common across all guards like
  // LocalState.
  RootGuardManager* _root_guard_manager{nullptr};

 private:
  // This is set while constructing the leaf guard. This is used for identifying
  // the cause of recompilation.
  py::list _verbose_code_parts;
};

/**
 * Represents a leaf guard that accepts the python guard check function. We
 * would like to have most of the guards in C++ (to avoid a Python function
 * call).  But, it will take some time to reach that goal. Also, there might be
 * cases where its too tedious to write an equivalent C++ guard.
 *
 * LAMBDA_GUARD allows us to gradually move to C++. We can start from all
 * guards of type PythonLambaGuard and incrementally move expensive guards to
 * C++.
 */
class LAMBDA_GUARD : public LeafGuard {
 public:
  LAMBDA_GUARD(
      RootGuardManager* root_guard_manager,
      py::object guard_check_fn,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {
    if (py::isinstance<py::function>(guard_check_fn)) {
      _guard_check_fn = py::cast<py::function>(std::move(guard_check_fn));
    } else {
      throw py::type_error("LAMBDA_GUARD expects (callable, str)");
    }
  }

  // Runs the lambda function with the current f_locals value.
  bool check_nopybind(PyObject* value) override { // borrowed ref
    PyObject* x = PyObject_CallOneArg(_guard_check_fn.ptr(), value); // new ref
    if (x == nullptr) {
      // An exception is caught in the lambda function.
      PyErr_Clear();
      return false;
    }
    bool result = PyObject_IsTrue(x);
    Py_DECREF(x);
    return result;
  }

  GuardDebugInfo check_verbose_nopybind(PyObject* value) override {
    PyObject* x = PyObject_CallOneArg(_guard_check_fn.ptr(), value); // new ref
    if (x == nullptr) {
      // An exception is caught in the lambda function.
      std::string exc_message = get_exception_message();
      PyErr_Clear();
      return GuardDebugInfo(false, exc_message, 0);
    }
    bool result = PyObject_IsTrue(x);
    Py_DECREF(x);
    if (result) {
      return GuardDebugInfo(true, 0);
    }
    return GuardDebugInfo(false, verbose_code_parts(), 0);
  }

 private:
  // The user provided lambda function for check_fn.
  py::function _guard_check_fn;
};

class TYPE_MATCH : public LeafGuard {
 public:
  // type_id = id(type(obj))
  TYPE_MATCH(
      RootGuardManager* root_guard_manager,
      py::object type_id,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _expected(py::cast<intptr_t>(std::move(type_id))) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return Py_TYPE(value) == (void*)_expected;
  }

 private:
  // id of the type of the original object.
  intptr_t _expected;
};

class ID_MATCH : public LeafGuard {
 public:
  // obj_id = id(obj)
  ID_MATCH(
      RootGuardManager* root_guard_manager,
      py::object obj_id,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _expected(py::cast<intptr_t>(std::move(obj_id))) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return value == (void*)_expected;
  }

 private:
  // id of the original object.
  intptr_t _expected;
};

class NONE_MATCH : public LeafGuard {
 public:
  NONE_MATCH(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return value == Py_None;
  }
};

class TRUE_MATCH : public LeafGuard {
 public:
  TRUE_MATCH(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return value == Py_True;
  }
};

class FALSE_MATCH : public LeafGuard {
 public:
  FALSE_MATCH(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return value == Py_False;
  }
};

class EQUALS_MATCH : public LeafGuard {
 public:
  EQUALS_MATCH(
      RootGuardManager* root_guard_manager,
      py::object value,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _value(value),
        _value_type(Py_TYPE(value.ptr())) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // Fast path - pointer equality check. Pointer equality checks are ok
    // because objects guarded with EQUALS_MATCH are immutable.
    if (value != _value.ptr()) {
      // Check type
      if (Py_TYPE(value) != _value_type) {
        return false;
      }
      int result = PyObject_RichCompareBool(value, _value.ptr(), Py_EQ);
      // Check for exception
      if (result == -1) {
        PyErr_Clear();
        return false;
      }
      return result;
    }
    return true;
  }

 private:
  // value to compare against. This is py::object so that we hold on to the
  // original value and prevent garbage collection. We run EQUALS_MATCH only on
  // selected objects which do not have high memory footprint, so holding on to
  // these objects is ok.
  py::object _value;

  // Type of the value
  PyTypeObject* _value_type;
};

class RANGE_ITERATOR_MATCH : public LeafGuard {
 public:
  RANGE_ITERATOR_MATCH(
      RootGuardManager* root_guard_manager,
      py::object start,
      py::object stop,
      py::object step,
      py::object type_id,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _type_id(py::cast<intptr_t>(std::move(type_id))) {
    PyObject* start_obj = start.ptr();
    PyObject* stop_obj = stop.ptr();
    PyObject* step_obj = step.ptr();
    _start = THPUtils_unpackLong(start_obj);
    _stop = THPUtils_unpackLong(stop_obj);
    _step = THPUtils_unpackLong(step_obj);
    TORCH_CHECK(
        !PyErr_Occurred(), "values of start/stop/step must fit in a long type");
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // Do a type match first.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    if (Py_TYPE(value) != (void*)_type_id) {
      return false;
    }
    _PyRangeIterObject* iter = (_PyRangeIterObject*)value;

#if IS_PYTHON_3_12_PLUS
    long start = iter->start;
#else
    long start = iter->start + iter->index * iter->step;
#endif // IS_PYTHON_3_12_PLUS

    long stop = iter->start + iter->len * iter->step;
    return start == _start && stop == _stop && iter->step == _step;
  }

 private:
  intptr_t _type_id;
  // Normalized representation of a range iterator.
  long _start;
  long _stop;
  long _step;
};

class TUPLE_ITERATOR_LEN : public LeafGuard {
 public:
  TUPLE_ITERATOR_LEN(
      RootGuardManager* root_guard_manager,
      py::object length,
      py::object type_id,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _length(py::cast<Py_ssize_t>(std::move(length))),
        _type_id(py::cast<intptr_t>(std::move(type_id))) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // Do a type match first.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    if (Py_TYPE(value) != (void*)_type_id) {
      return false;
    }
    _PyTupleIterObject* it = (_PyTupleIterObject*)value;
    Py_ssize_t length = 0;
    if (it->it_seq)
      length = PyTuple_GET_SIZE(it->it_seq) - it->it_index;
    return length == _length;
  }

 private:
  // Length of the guarded list
  Py_ssize_t _length;
  intptr_t _type_id;
};

class LENGTH_CHECK : public LeafGuard {
 public:
  LENGTH_CHECK(
      RootGuardManager* root_guard_manager,
      py::object value,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _length(py::cast<Py_ssize_t>(std::move(value))) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // PySequence_Length returns -1 if the object is not a sequence. So, we
    // don't have to test for PySequence_Check.
    return PySequence_Length(value) == _length;
  }

 private:
  // Length of the guarded list
  Py_ssize_t _length;
};

class DICT_LENGTH : public LeafGuard {
 public:
  DICT_LENGTH(
      RootGuardManager* root_guard_manager,
      py::object value,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _length(py::cast<Py_ssize_t>(std::move(value))) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return PyDict_Check(value) && PyDict_Size(value) == _length;
  }

 private:
  // Length of the guarded dict
  Py_ssize_t _length;
};

class NOT_NONE : public LeafGuard {
 public:
  NOT_NONE(RootGuardManager* root_guard_manager, py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return value != Py_None;
  }
};

class MAPPING_KEYS_MATCH : public LeafGuard {
 public:
  MAPPING_KEYS_MATCH(
      RootGuardManager* root_guard_manager,
      py::object value,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {
    // This is ok to stash in the state because we only support
    // MappingProxyType objects with constant keys. So, the mem overhead is
    // negligible.
    _keys = py::list(value.attr("keys")());
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    PyObject* keys = PyMapping_Keys(value); // new ref
    int result = PyObject_RichCompareBool(keys, _keys.ptr(), Py_EQ);
    Py_DECREF(keys);
    return result;
  }

 private:
  py::object _keys;
};

class DEFAULT_DEVICE : public LeafGuard {
 public:
  DEFAULT_DEVICE(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {
    py::handle device_module = py::module::import("torch.utils._device");
    // Save the dict using py::object
    _utils_device_dict = device_module.attr("__dict__");
    _device = _utils_device_dict["CURRENT_DEVICE"];
  }

  template <typename T>
  bool check_nopybind_template(T* value) { // borrowed ref
    // Create a static interned string. Interned string is faster than creating
    // a new string every time. Even though its a new reference, we don't dec
    // ref it. Interned strings are used for things like variable names and are
    // leaked by design.
    static PyObject* current_device_str =
        PyUnicode_InternFromString("CURRENT_DEVICE");
    PyObject* device = PyDict_GetItem(
        _utils_device_dict.ptr(), current_device_str); // borrowed ref
    if (device != _device.ptr()) {
      int result = PyObject_RichCompareBool(device, _device.ptr(), Py_EQ);
      if (result == -1) {
        PyErr_Clear();
        return false;
      }
      return result;
    }
    return true;
  }

  bool check_nopybind(PyObject* value) override {
    return check_nopybind_template(value);
  }

  bool check_nopybind(FrameLocalsMapping* value) override {
    return check_nopybind_template(value);
  }

 private:
  // Save the current device and the module dict during the guard construction.
  py::object _utils_device_dict;
  py::object _device;
};

class GLOBAL_STATE : public LeafGuard {
 public:
  GLOBAL_STATE(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _guard(PyObject_New(GlobalStateGuard, &GlobalStateGuardType)) {
    _guard->init();
    owner_ = py::reinterpret_steal<py::object>((PyObject*)_guard);
  }

  GLOBAL_STATE(
      RootGuardManager* root,
      py::object initial_state,
      py::object verbose_code_parts)
      : LeafGuard(root, std::move(verbose_code_parts)),
        owner_(std::move(initial_state)),
        _guard((GlobalStateGuard*)owner_.ptr()) {
    if (!PyObject_TypeCheck(owner_.ptr(), &GlobalStateGuardType)) {
      throw py::type_error("GLOBAL_STATE expects a GlobalStateGuard");
    }
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // Ignore value arg, this is just to satisfy the interface.
    return _guard->check();
  }

  bool check_nopybind(FrameLocalsMapping* value) override {
    // Ignore value arg, this is just to satisfy the interface.
    return _guard->check();
  }

  GuardDebugInfo check_verbose_nopybind(PyObject* value) override {
    if (!_guard->check()) {
      return GuardDebugInfo(
          false, "GLOBAL_STATE changed: " + _guard->reason(), 0);
    }
    return GuardDebugInfo(true, 1);
  }

 private:
  py::object owner_;
  GlobalStateGuard* _guard;
};

// Checks that an attr is absent in the object. We don't need the opposite
// HASATTR guard because we can just rely on GetAttrGuardAccessor to act as
// HASATTR guard.
class NO_HASATTR : public LeafGuard {
 public:
  NO_HASATTR(
      RootGuardManager* root_guard_manager,
      py::object attr_name,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _attr_name(std::move(attr_name)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return PyObject_HasAttr(value, _attr_name.ptr()) == 0;
  }

 private:
  py::object _attr_name;
};

// Checks that dict contains or does not contain a key. This happens for
// PythonSysModulesVariable tracker.
// TODO(janimesh) - Check if we can use DictGuardManager. The downside could be
// large number of keys for sys module, so DICT_CONTAINS might still end up
// being faster.
class DICT_CONTAINS : public LeafGuard {
 public:
  DICT_CONTAINS(
      RootGuardManager* root_guard_manager,
      bool contains,
      py::object key,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _contains(contains ? 1 : 0),
        _key(std::move(key)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    int result = PyDict_Check(value) && PyDict_Contains(value, _key.ptr());
    if (result == -1) {
      PyErr_Clear();
      return false;
    }
    return result == _contains;
  }

 private:
  int _contains;
  py::object _key;
};

// Check that set contains an item.
class SET_CONTAINS : public LeafGuard {
 public:
  SET_CONTAINS(
      RootGuardManager* root_guard_manager,
      bool contains,
      py::object item,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _contains(contains ? 1 : 0),
        _item(std::move(item)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    int result = (PySet_Check(value) || PyFrozenSet_Check(value)) &&
        PySet_Contains(value, _item.ptr());
    if (result == -1) {
      PyErr_Clear();
      return false;
    }
    return result == _contains;
  }

 private:
  int _contains;
  py::object _item;
};

// Check if the float is nan
class FLOAT_IS_NAN : public LeafGuard {
 public:
  FLOAT_IS_NAN(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    if (!PyFloat_CheckExact(value)) {
      return false;
    }
    return std::isnan(PyFloat_AsDouble(value));
  }
};

// Check if the float is nan
class COMPLEX_IS_NAN : public LeafGuard {
 public:
  COMPLEX_IS_NAN(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    if (!PyComplex_CheckExact(value)) {
      return false;
    }
    Py_complex c_value = PyComplex_AsCComplex(value);
    return std::isnan(c_value.real) || std::isnan(c_value.imag);
  }
};

// Check if the dual level is the same as the one in fx graph
class DUAL_LEVEL_MATCH : public LeafGuard {
 public:
  DUAL_LEVEL_MATCH(
      RootGuardManager* root_guard_manager,
      int64_t level,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _level(level) {
    forward_ad_module = py::module_::import("torch.autograd.forward_ad");
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // Ignore value arg, this is just to satisfy the interface.
    return _check();
  }

  bool check_nopybind(FrameLocalsMapping* value) override {
    // Ignore value arg, this is just to satisfy the interface.
    return _check();
  }

  bool _check() {
    PyObject* current_level = PyObject_GetAttrString(
        forward_ad_module.ptr(), "_current_level"); // new ref
    if (current_level == nullptr) {
      // Attribute absent, clear the exception and return false.
      PyErr_Clear();
      return false;
    }
    if (!PyLong_CheckExact(current_level)) {
      Py_DECREF(current_level);
      return false;
    } else {
      int64_t current_level_int = PyLong_AsLongLong(current_level);
      Py_DECREF(current_level);
      return current_level_int == _level;
    }
  }

 private:
  int64_t _level;
  py::object forward_ad_module;
};

/**
 * Relational guards compare more than one value. We implement Relational
 * guards by capturing some state in the guard object. For example for tensor
 * aliasing guards - tensor X is not tensor Y - we construct one leaf guard
 * and install it at as a leaf of two guard managers (one for X and
 * another for Y). Therefore, this guard is run twice. In the first
 * invocation, it saves the first value (state) and returns True. In the
 * second invocation, it compares the saved value with the new value and
 * returns True if they do not alias.
 *
 * We have to be careful about resetting in case the other guards fail and we
 * have some state in the relational guard. This is done by virtual method
 * reset_state(). This is called by the RootGuardManager before it exits.
 *
 */
class RelationalGuard : public LeafGuard {
 public:
  RelationalGuard(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {}

  // reset the relational guard state on guard failure. This is called by the
  // guard manager.
  virtual void reset_state() = 0;
};

/**
 * Checks that object x is object y.
 */
class OBJECT_ALIASING : public RelationalGuard {
 public:
  OBJECT_ALIASING(
      RootGuardManager* root_guard_manager,
      py::object verbose_code_parts)
      : RelationalGuard(root_guard_manager, std::move(verbose_code_parts)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    if (_is_first_call) {
      _first_tensor = value;
      _is_first_call = false;
      return true;
    }
    return _first_tensor == value;
  }

  void reset_state() final {
    _is_first_call = true;
  }

 private:
  bool _is_first_call{true};
  PyObject* _first_tensor{nullptr};
};

/**
 * Checks that none of the tensors alias.
 */
class NO_TENSOR_ALIASING : public RelationalGuard {
 public:
  NO_TENSOR_ALIASING(
      RootGuardManager* root_guard_manager,
      const py::list& tensor_names,
      py::object verbose_code_parts)
      : RelationalGuard(root_guard_manager, std::move(verbose_code_parts)),
        _tensor_names(tensor_names) {
    _unique_tensors.reserve(tensor_names.size());
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    auto insertion = _unique_tensors.insert({value, nullptr});
    if (!insertion.second) {
      // No need to clear _unique_tensors, reset_state will do
      // it.
      return false;
    }
    return true;
  }

  GuardDebugInfo check_verbose_nopybind(PyObject* value) override {
    bool result = check_nopybind(value);

    if (!result) {
      return GuardDebugInfo(
          false, "Duplicate tensor found where not expected!", 0);
    }
    return GuardDebugInfo(true, 1);
  }

  void reset_state() final {
    _unique_tensors.clear();
  }

 private:
  py::list _tensor_names;
  ska::flat_hash_map<PyObject*, std::nullptr_t> _unique_tensors;
};

/**
 * Checks the storage overlapping relation of input tensors.
 *
 * This guard is always installed in pairs: one for the possibly overlapping
 * tensors, and another one for the non-overlapping tensors. This is so we can
 * correctly identify the given tensor in the check method as one of the 2
 * classes mentioned above.
 *
 * In the end, the one responsible for storing and checking is the
 * `StorageOverlapChecker` class.
 */
class STORAGE_OVERLAPPING : public RelationalGuard {
 public:
  STORAGE_OVERLAPPING(
      RootGuardManager* root_guard_manager,
      bool overlapping,
      std::shared_ptr<StorageOverlapChecker> checker,
      py::object verbose_code_parts)
      : RelationalGuard(root_guard_manager, std::move(verbose_code_parts)),
        _overlapping(overlapping),
        _checker(std::move(checker)) {}

  bool check_nopybind(PyObject* value) override {
    _checker->add(value, _overlapping);
    return _checker->maybe_check();
  }

  void reset_state() final {
    _checker->reset(_overlapping);
  }

 private:
  // Flag that indicates which kind of tensor this guard is collecting:
  //   1. Possibly overlapping tensors; or
  //   2. Non-overlapping tensors.
  bool _overlapping;
  // Actual checker for this guard.
  std::shared_ptr<StorageOverlapChecker> _checker;
};

/**
 * Symbolic Shape Guard.
 */
class SYMBOLIC_SHAPE_GUARD : public RelationalGuard {
 public:
  SYMBOLIC_SHAPE_GUARD(
      RootGuardManager* root_guard_manager,
      py::int_ nargs_int,
      py::int_ nargs_float,
      py::int_ py_addr,
      py::object py_addr_keep_alive,
      py::object verbose_code_parts)
      : RelationalGuard(root_guard_manager, std::move(verbose_code_parts)),
        _py_addr_keep_alive(std::move(py_addr_keep_alive)) {
    _nargs_int = PyLong_AsSize_t(nargs_int.ptr());
    _nargs_float = PyLong_AsSize_t(nargs_float.ptr());
    _nargs = _nargs_int + _nargs_float;
    if (PyErr_Occurred()) {
      throw py::value_error(
          "SYMBOLIC_SHAPE_GUARD expected a non-negative number of arguments.");
    }
    uintptr_t addr = PyLong_AsUnsignedLongLong(py_addr.ptr());
    if (PyErr_Occurred()) {
      throw py::value_error(
          "SYMBOLIC_SHAPE_GUARD expected an address to a C function.");
    }
    _guard_check_fn = reinterpret_cast<int8_t (*)(int64_t*, double*)>(addr);
    _args_int = std::vector<int64_t>(_nargs_int);
    _args_float = std::vector<double>(_nargs_float);
  }

  bool check_nopybind(PyObject* value) override {
    // We know that these arguments came from
    // IndexedSource(TensorPropertyGuard) and therefore no need to check that
    // the value is a Tuple[int, int].
    PyObject* py_idx = PyTuple_GET_ITEM(value, 0);
    PyObject* py_val = PyTuple_GET_ITEM(value, 1);
    size_t iarg = PyLong_AsSize_t(py_idx);
    if (iarg < _nargs_int) {
      if (!PyLong_Check(py_val)) {
        return false;
      }
      _args_int[iarg] = PyLong_AsLongLong(py_val);
    } else {
      if (!PyFloat_Check(py_val)) {
        return false;
      }
      _args_float[iarg - _nargs_int] = PyFloat_AS_DOUBLE(py_val);
    }
    _args_seen++;

    if (_args_seen == _nargs) {
      _args_seen = 0;
      return _guard_check_fn(_args_int.data(), _args_float.data());
    } else {
      // We don't have all the values yet. Return true until we get all.
      return true;
    }
  }

  GuardDebugInfo check_verbose_nopybind(PyObject* value) override {
    if (!PyTuple_Check(value)) {
      return GuardDebugInfo(false, "Non tuple found!", 0);
    } else if (PyTuple_Size(value) != 2) {
      return GuardDebugInfo(false, "Tuple of size not 2 found!", 0);
    } else {
      PyObject* py_idx = PyTuple_GET_ITEM(value, 0);
      PyObject* py_val = PyTuple_GET_ITEM(value, 1);
      if (!PyLong_Check(py_idx)) {
        return GuardDebugInfo(false, "Non integer index found!", 0);
      }
      size_t iarg = PyLong_AsSize_t(py_idx);
      if (iarg >= _nargs) {
        return GuardDebugInfo(false, "Index out of bounds!", 0);
      } else if (iarg < _nargs_int && !PyLong_Check(py_val)) {
        return GuardDebugInfo(false, "Non integer found!", 0);
      } else if (iarg >= _nargs_int && !PyFloat_Check(py_val)) {
        return GuardDebugInfo(false, "Non float found!", 0);
      }
    }
    bool result = check_nopybind(value);

    if (!result) {
      std::string msg = "\"Shape guard failed with values: ";
      for (auto v : _args_int) {
        msg += std::to_string(v) + ",";
      }
      for (auto v : _args_float) {
        msg += std::to_string(v) + ",";
      }
      msg.pop_back();
      msg += "\"";
      auto msgs = py::list();
      for (auto code_part : verbose_code_parts()) {
        msgs.append(code_part);
      }
      msgs.append(msg);
      return GuardDebugInfo(false, msgs, 0);
    }
    return GuardDebugInfo(true, 1);
  }

  void reset_state() final {
    _args_seen = 0;
  }

 private:
  py::object _py_addr_keep_alive;
  size_t _args_seen{0}, _nargs_float, _nargs_int, _nargs;
  std::vector<int64_t> _args_int;
  std::vector<double> _args_float;
  std::function<int8_t(int64_t*, double*)> _guard_check_fn;
};

class DYNAMIC_INDICES : public LeafGuard {
  // C++ equivalent of
  //  code.append(
  //      f"(({tensor_name}._dynamo_dynamic_indices.issubset({value._dynamo_dynamic_indices}))
  //      if hasattr({tensor_name}, '_dynamo_dynamic_indices') else True)"  #
  //      noqa: B950
 public:
  DYNAMIC_INDICES(
      RootGuardManager* root_guard_manager,
      py::set dynamic_indices,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _dynamic_indices(std::move(dynamic_indices)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // Make an interned string
    static PyObject* dynamic_indices_str =
        PyUnicode_InternFromString("_dynamo_dynamic_indices");
    PyObject* indices = PyObject_GetAttr(value, dynamic_indices_str); // new ref
    if (indices == nullptr) {
      // Attr absent. Clear exception.
      PyErr_Clear();
      // This is true deliberately. If hasattr fails, we return true.
      return true;
    }

    static PyObject* issubset_str = PyUnicode_InternFromString("issubset");
    PyObject* call_result = PyObject_CallMethodObjArgs(
        indices, issubset_str, _dynamic_indices.ptr(), nullptr); // new ref
    bool result = PyObject_IsTrue(call_result);
    Py_DECREF(call_result);
    Py_DECREF(indices);
    return result;
  }

 private:
  py::set _dynamic_indices;
};

class DICT_VERSION : public LeafGuard {
 public:
  DICT_VERSION(
      RootGuardManager* root_guard_manager,
      py::object value,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)) {
    if (!PyDict_Check(value.ptr())) {
      throw py::type_error("DICT_VERSION expects a dict");
    }
    _tag = get_dict_version_unchecked(value.ptr());
  }
  bool check_nopybind(PyObject* value) override { // borrowed ref
    return PyDict_Check(value) && get_dict_version_unchecked(value) == _tag;
  }

  // Saved dict version.
  uint64_t _tag;
};

// GuardManager can be a pointer to DictGuardManager, but at this point the
// compiler does not know that DictGuardManager is a derived class of
// GuardManager (no way to define inheritance relationships in forward
// declarations), so we forward declare a factory function and define it when
// both DictGuardManager and GuardManager are fully defined.
std::unique_ptr<GuardManager> make_guard_manager(
    RootGuardManager* root,
    std::string source,
    py::handle example_value,
    py::handle guard_manager_enum);

// Forward declarations for tag safe related helpers. All of these require some
// interaction between RootGuardManager and GuardManager. Since both of the
// classes are forward declared, we have to forward declare these helpers as
// well.
void start_recording_dict_pointers(
    RootGuardManager* root,
    GuardManager* tag_safe_root);
void stop_recording_dict_pointers(
    RootGuardManager* root,
    PyObject* value,
    bool result);
bool is_recording_dict_pointers(RootGuardManager* root);
void record_dict_pointer(RootGuardManager* root, PyObject* dict_pointer);
void record_tensor_pointer(RootGuardManager* root, PyObject* tensor_pointer);

GuardManager* clone_guard_manager(
    GuardManager* from,
    RootGuardManager* root,
    const py::function& clone_filter_fn);
void add_relational_guard_resetter_to_cloned_root(
    RootGuardManager* root,
    std::shared_ptr<RelationalGuard> guard);
std::shared_ptr<RelationalGuard> get_no_tensor_aliasing_guard(
    RootGuardManager* _root);
// std::string get_compile_id(RootGuardManager* root);

struct WeakEntry {
  PyObject* wr; // weakref
  PyObject* cap; // capsule whose m_self is used by the callback
};
/**
 * Base class representing a pair of accessor and the associated guard
 * manager. The accessor defines how to access the child value from the
 * py::object given to the parent check function.
 *
 * GuardAccessors can be considered equivalent to name() method of Source
 * objects in guards.py. In python, name() method returns a str which we can
 * then eval in f_locals and f_globals to retrieve the actual py object.
 * GuardAccessor serves the same purpose. The minor difference is that
 * GuardManager is a tree structure, so a GuardAccessor just has to retrieve
 * the value in the next level in this tree and pass it to the child
 * GuardAccessor.
 *
 * GuardAccessor also owns the GuardManager associated with the retrieved
 * value from the GuardAccessor.
 */
class GuardAccessor {
 public:
  GuardAccessor(
      RootGuardManager* root,
      py::object accessor_key,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum);

  // Return by reference as GuardAccessor owns the GuardManager.
  std::unique_ptr<GuardManager>& get_guard_manager() {
    return _guard_manager;
  }

  bool matches_key(const py::handle& key) const {
    return _accessor_key.equal(key);
  }

  std::string get_source() {
    return _source;
  }

  // matches_dict_tag is used by the DictGetItemGuardAccessor to skip the guard
  // subtree on immutable dict getitems.
  virtual bool check_nopybind(PyObject* obj, bool matches_dict_tag = false) = 0;
  virtual bool check_nopybind(FrameLocalsMapping* map, bool matches_dict_tag) {
    // throw std::runtime_error("fallback to python");
    // Could fallback to running check on the Python dict (lazily constructed)
    return check_nopybind((PyObject*)map->to_dict(), matches_dict_tag);
  }
  virtual GuardDebugInfo check_verbose_nopybind(PyObject* obj) = 0;
  virtual std::string repr() const = 0;

  virtual ~GuardAccessor() = default;

  // Cloning related functions
  GuardAccessor(GuardManager* guard_manager, GuardAccessor* from);

  virtual GuardAccessor* clone(
      RootGuardManager* cloned_root,
      const py::function& clone_filter_fn) = 0;

  void clone_visitor(GuardAccessor* to) {
    to->_source = this->_source;
    to->_accessor_key = this->_accessor_key;
  }

  template <typename DerivedGuardAccessor>
  GuardAccessor* clone_common(
      RootGuardManager* cloned_root,
      const py::function& clone_filter_fn) {
    GuardManager* cloned_mgr = clone_guard_manager(
        get_guard_manager().get(), cloned_root, clone_filter_fn);
    if (cloned_mgr == nullptr) {
      return nullptr;
    }
    DerivedGuardAccessor* cloned_accessor =
        new DerivedGuardAccessor(cloned_mgr, (DerivedGuardAccessor*)this);
    return cloned_accessor;
  }

 protected:
  // Guard manager corresponding to the retrieved value from the
  // GuardAccessor.
  std::unique_ptr<GuardManager> _guard_manager;
  // accessor key could be py::str for getattr, getitem or py::function for
  // lambda accessor. It is a py::object because we need to keep these accessor
  // keys alive.
  py::object _accessor_key;

  // A string that can be eval'd on f_locals or f_globals to access the variable
  // value. Only used for debugging.
  std::string _source;
};

/**
 * GuardManager encapsulates all the guards related to a particular
 * py::object. It is a tree structure and consists of 1) Leaf guards - Guards
 * that are run on the user given object 2) Accessors - Guard accessors (like
 * getattr, getitem) to access the next value in the tree hierarchy. Accessor
 * object also holds the child GuardManager.
 *
 * Lets look at an example to understand how it works.
 * class Pair:
 *     int x = 1;
 *     int y = 2;
 *
 * At compile time
 * >> guard_mananger = GuardManager()
 * >> guard_mananger.x.add_lambda_guard(
 *        lambda x: isinstance(x, Pair),
 *        lambda x: f"expected Pair, found {type(x)}"
 *    )
 * >> guard_mananger.x.add_lambda_guard(lambda x: x == 1, lambda x: f"found
 * {x}, expected 1")
 * >> guard_mananger.y.add_lambda_guard(lambda x: x == 2, lambda x: f"found
 * {x}, expected 2")
 *
 * At runtime
 * >> guard_mananger.check(Pair())
 *
 * At compile time we build the tree structure. When we do `guard_manager.x`,
 * it creates an AttrGuardAccessorNode, initializes a child guard manager with
 * this accessor node, and adds it as a child. When we do
 * `guard_manager.x.add_lambda_guard`, we call add_lambda_guard on the newly
 * created guard manager and register a new leaf guard on it.
 *
 * At runtime, the accessor node has an important function of providing a way
 * to access the value for the child guard. In the above example,
 * guard_manager.x adds an AttrGuardAccessorNode with attr_name x. When check
 * function is called, parent GuardManager calls getattr(value, "x") on its
 * value passed to the check function to call the check function of the child
 * guard manager.
 *
 * Performance optimization for fail fast - An optimization for runtime here is
 * to sort the execution of child guards depending on the failure count.  This
 * ensures that we run the guards that are more prone to fail statistically
 * first. This can improve the cache lookup time when we have multiple cache
 * entries.
 */

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class GuardManager {
 public:
  GuardManager() = delete;
  GuardManager(RootGuardManager* root, std::string source)
      : _root(root), _source(std::move(source)), _is_dict(false) {}

  GuardManager(
      RootGuardManager* root,
      std::string source,
      py::handle example_value)
      : _root(root),
        _source(std::move(source)),
        _is_dict(py::isinstance<py::dict>(example_value)),
        _is_immutable(is_immutable_object(example_value)) {
    if (_is_dict) {
      _dict_tag = get_dict_version_unchecked(example_value.ptr());
    }
    py::object typ = py::type::of(example_value);
    py::object weakref_mod = py::module_::import("weakref");
    _weak_type = weakref_mod.attr("ref")(typ);
    py::object config_module = py::module_::import("torch._dynamo.config");
    _max_saved_pointers_for_recursive_dict_tags_check =
        config_module.attr("max_saved_pointers_for_recursive_dict_tags_check")
            .cast<uint64_t>();
  }

  GuardManager(const GuardManager& m) = delete;
  GuardManager& operator=(const GuardManager&) = delete;

  virtual ~GuardManager() {
    cleanup_tag_safe_entries();
    disable_recursive_dict_tag_optimization();
  }

  void cleanup_tag_safe_entries() {
    for (auto& e : _tag_safe_entries) {
      // unset the pycapsule to make it invalid. This ensures that the weakref
      // callback does not look into a dangling pointer.
      if (PyCapsule_IsValid(e.cap, "GuardManager*")) {
        PyCapsule_SetName(e.cap, "DeadGuardManager");
      }
      Py_CLEAR(e.wr); // kills weakref (may remove callback)
    }
    _tag_safe_entries.clear();
  }

  RootGuardManager* get_root() {
    return _root;
  }

  std::string get_source() {
    return _source;
  }

  virtual void add_leaf_guard(std::shared_ptr<LeafGuard> leaf_guard) {
    _leaf_guards.emplace_back(std::move(leaf_guard));
  }

 public:
  // relational guard helpers
  void set_has_object_aliasing_guard() {
    _has_object_aliasing_guard = true;
  }

  void set_has_no_tensor_aliasing_guard() {
    _has_no_tensor_aliasing_guard = true;
  }

  bool has_object_aliasing_guard() {
    return _has_object_aliasing_guard;
  }

  bool has_no_tensor_aliasing_guard() {
    return _has_no_tensor_aliasing_guard;
  }

 public:
  // type related helpers
  bool is_guarded_value_immutable() {
    return _is_immutable;
  }

  bool is_recursive_dict_tag_matching_disabled() {
    return _disable_dict_tag_matching;
  }

  py::object get_type_of_guarded_value() {
    if (!_weak_type || _weak_type.is_none()) {
      return py::type::of(py::none());
    }

    TORCH_CHECK_TYPE(
        PyCallable_Check(_weak_type.ptr()), "_weak_type is not callable");
    return _weak_type();
  }

 public:
  // tag safety related helpers
  // Seen docstring in guards.py ``find_tag_safe_roots`` for full context
  void mark_tag_safe() {
    _is_tag_safe = true;
  }

  void mark_tag_safe_root() {
    TORCH_CHECK(
        _is_tag_safe, "Marking a node tag_safe_root when its not tag safe");
    _is_tag_safe_root = true;
  }

  bool is_tag_safe() {
    return _is_tag_safe;
  }

  bool is_tag_safe_root() {
    return _is_tag_safe_root;
  }

 public:
  // tag safe optimizations
  void stash_dict_pointers(
      PyObject* value,
      std::vector<std::pair<PyObject*, uint64_t>> dict_pointers) {
    _dict_pointers[value] = dict_pointers;
  }

  void stash_tensor_pointers(
      PyObject* value,
      std::vector<PyObject*> tensor_pointers) {
    _tensor_pointers[value] = tensor_pointers;
  }

  void disable_recursive_dict_tag_optimization() {
    unwatch_all_saved_dict_pointers();
    _disable_dict_tag_matching = true;
  }

 public:
  // For cloning
  GuardManager(
      RootGuardManager* root,
      std::string source,
      bool is_dict,
      bool is_immutable,
      py::object weak_type)
      : _root(root),
        _source(std::move(source)),
        _is_dict(is_dict),
        _is_immutable(is_immutable),
        _weak_type(weak_type) {}

  void clone_common(
      RootGuardManager* cloned_root,
      GuardManager* cloned_mgr,
      const py::function& clone_filter_fn) {
    for (const auto& guard : _leaf_guards) {
      cloned_mgr->_leaf_guards.emplace_back(guard);
      if (std::shared_ptr<RelationalGuard> relational_guard =
              std::dynamic_pointer_cast<RelationalGuard>(guard)) {
        add_relational_guard_resetter_to_cloned_root(
            cloned_root, relational_guard);
      }
    }

    for (const auto& accessor : _accessors) {
      GuardAccessor* cloned_accessor =
          accessor->clone(cloned_root, clone_filter_fn);
      if (cloned_accessor != nullptr) {
        cloned_mgr->_accessors.emplace_back(
            std::unique_ptr<GuardAccessor>(cloned_accessor));
      }
    }
  }

  virtual GuardManager* clone(
      RootGuardManager* cloned_root,
      const py::function& clone_filter_fn) {
    if (!py::cast<bool>(clone_filter_fn(this))) {
      return nullptr;
    }
    GuardManager* cloned_mgr = new GuardManager(
        cloned_root, _source, _is_dict, _is_immutable, _weak_type);
    if (is_tag_safe()) {
      cloned_mgr->mark_tag_safe();
      if (is_tag_safe_root()) {
        cloned_mgr->mark_tag_safe_root();
      }
    }
    clone_common(cloned_root, cloned_mgr, clone_filter_fn);
    return cloned_mgr;
  }

  /**
   * Adds a new guard manager with appropriate Accessor. If the accessor is
   * already present, we just return the guard manager.
   */
  template <typename GuardAccessorT>
  GuardManager* get_child_manager(
      const py::object& accessor_key,
      const std::string& source,
      py::handle example_value,
      py::handle guard_manager_enum) {
    // accessor_key type depends on the GuardAccessorT
    // for example for GetAttrGuardAccessor - py::str name

    // Return the manager if the guard accessor exists
    for (const auto& accessor : _accessors) {
      if (accessor->matches_key(accessor_key) &&
          source == accessor->get_source()) {
        return accessor->get_guard_manager().get();
      }
    }

    // Construct a new guard accessor
    _accessors.emplace_back(std::make_unique<GuardAccessorT>(
        _root,
        std::move(accessor_key),
        source,
        example_value,
        guard_manager_enum));
    return _accessors.back()->get_guard_manager().get();
  }

  // Runs the leaf guards check and then child managers check function.
  //
  // NB: There is some code DUPLICATION between this and check_verbose
  // function. This is intentional. check function is in the hot path and is
  // kept very simple. The purpose of check_verbose function is to get guard
  // failure reasoning to understand recompilations. check_verbose function
  // does not change the state of the guard, e.g., it does not shuffle the
  // guards and does not change the fail count. For simplicity, we duplicate
  // the code here.
  template <typename T>
  bool check_nopybind_template(T* value) { // borrowed ref

    if (!this->check_leaf_guards_nopybind(value)) {
      return false;
    }

    return this->check_accessors_nopybind(value);
  }

  bool check_dict_pointer_tags(PyObject* value) {
    if (_dict_callback_installed) {
      // This means that for 3.12+, there are callbacks watching dict pointers.
      return true;
    }
    for (auto& kv : _dict_pointers[value]) {
      PyObject* dict_pointer = kv.first;
      uint64_t old_tag = kv.second;
      uint64_t new_tag = get_dict_version_unchecked(dict_pointer);
      if (old_tag != new_tag) {
        return false;
      }
    }
    return true;
  }

  bool check_no_tensor_aliasing_guards_fast(PyObject* value) {
    std::shared_ptr<RelationalGuard> no_tensor_aliasing_guard =
        get_no_tensor_aliasing_guard(_root);
    for (auto* tensor_pointer : _tensor_pointers[value]) {
      if (!no_tensor_aliasing_guard->check_nopybind(tensor_pointer)) {
        return false;
      }
    }
    return true;
  }

  virtual bool check_nopybind(PyObject* value) {
    // -----------------------------------------------------------------------------
    // Recursive Dict‑Tag Matching
    // -----------------------------------------------------------------------------
    // The GuardManager implements recursive dictionary‑tag matching.
    // During compilation we pre‑compute every `tag_safe_node` and its
    // corresponding `tag_safe_root` (see `find_tag_safe_nodes` in guards.py).
    // These annotations allow the runtime to validate large subtrees with a
    // single, cheap check.
    //
    // Key idea
    // --------
    // For a `tag_safe_root`, the input pointer called `value`, the object the
    // guard is inspecting, serves as a proxy for the entire nested dictionary
    // structure beneath that node.  If this `value` pointer is one we have
    // already recorded, then verifying each dictionary’s tag is sufficient to
    // prove that nothing inside the subtree has changed.
    //
    // Runtime flow
    // -------------
    // 1) Previously‑seen `value` pointer
    //    • Look up the current `value` pointer in our cache.
    //    • If found, perform a recursive tag comparison on the cached subtree.
    //      All tags match means guard passes with no further traversal.
    //
    // 2) First‑time `value` pointer
    //    • Enter recording mode; walk the subtree, each tag safe root collects
    //      dict tag, and cache the new `value` pointer.
    //    • Future executions with this pointer now hit the fast path above.
    //
    // 3) Supporting multiple pointers
    //    • We deliberately cache a bounded number of distinct 

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 79 class(es): Meta, Meta, which, StorageOverlapChecker, GuardDebugInfo, GuardManager, RootGuardManager, DictGuardManager, for, LeafGuard, LAMBDA_GUARD, TYPE_MATCH, ID_MATCH, NONE_MATCH, TRUE_MATCH, FALSE_MATCH, EQUALS_MATCH, RANGE_ITERATOR_MATCH, TUPLE_ITERATOR_LEN, LENGTH_CHECK

### Structures
This file defines 12 struct(s): here, is, typedef, AutocastState, GlobalStateGuard, PyModuleDef, StaticMeta, DynamicMeta, one, WeakEntry, a, the


## Key Components

The file contains 24071 words across 7852 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 265861 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
