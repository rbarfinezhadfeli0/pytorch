# Documentation: `docs/torch/csrc/distributed/c10d/init.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/init.cpp_docs.md`
- **Size**: 53,995 bytes (52.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/init.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/init.cpp`
- **Size**: 175,207 bytes (171.10 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/python_headers.h>

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/Functional.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/ControlCollectives.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/StoreCollectives.hpp>
#include <torch/csrc/distributed/c10d/control_plane/WorkerServer.hpp>
#include <string_view>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#endif
#include <torch/csrc/distributed/c10d/FakeProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/PyProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/python_callback_work.hpp>

#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>
#endif

#ifdef USE_C10D_XCCL
#include <torch/csrc/distributed/c10d/ProcessGroupXCCL.hpp>
#endif

#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>
#endif

#ifdef USE_C10D_MPI
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#endif

#ifdef USE_C10D_UCC
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#endif

#include <fmt/format.h>
#include <pybind11/chrono.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#ifdef USE_NVSHMEM
#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh>
#endif

#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/reducer.hpp>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/custom_class.h>

namespace {

#ifdef USE_C10D_NCCL

bool acquire_gil() {
  // basically if this function can acquire the gil, it will return quickly.
  // if not, it will hang forever.  The idea is to call this from a thread
  // wrapped in a future, and then check the future after a timeout, to
  // determine whether we're facing gil contention.
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    return true;
  }

  // If we end up here, its probably still a "pass" from the perspective of
  // checking whether python is stuck. but currently we don't check the return
  // value of this function anyway, just check whether it returned quickly vs
  // timing out.  Taking a long time is the main sign of trouble.  Fast return
  // with true or with false is both OK from the perspective of debugging python
  // hangs.
  return false;
}

bool registerGilChecker() {
  c10d::get_gil_checker() = &acquire_gil;
  return true;
}

static bool registered = registerGilChecker();
#endif // USE_C10D_NCCL

// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
// TODO: move this somewhere more generally useful
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_{};

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) noexcept = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(
      IntrusivePtrNoGilDestructor&&) noexcept = default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      // NOLINTNEXTLINE(bugprone-exception-escape)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  [[nodiscard]] T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true)

namespace torch::distributed::c10d {

namespace {

py::bytes toPyBytes(const std::vector<uint8_t>& data) {
  return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

std::vector<py::bytes> toPyBytes(
    const std::vector<std::vector<uint8_t>>& data) {
  std::vector<py::bytes> out;
  out.reserve(data.size());
  for (const std::vector<uint8_t>& data_ : data) {
    out.emplace_back(reinterpret_cast<const char*>(data_.data()), data_.size());
  }
  return out;
}

std::vector<uint8_t> toVec8(const std::string& data) {
  std::vector<uint8_t> out{data.begin(), data.end()};
  return out;
}

std::vector<std::vector<uint8_t>> toVec8(const std::vector<std::string>& data) {
  std::vector<std::vector<uint8_t>> out;
  out.reserve(data.size());
  for (auto& data_ : data) {
    out.emplace_back(toVec8(data_));
  }
  return out;
}

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

constexpr auto kDeprecationWarning =
    "{} API is being deprecated, please ping "
    "https://github.com/pytorch/pytorch/issues/46291 "
    "if you see this warning";
template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;

template <typename T, typename Trampoline>
using intrusive_ptr_no_gil_destructor_trampoline_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>, Trampoline>;

// PythonStore is a pybind11 trampoline class to allow a Python
// class to inherit from c10d.Store and implement its interface.
class PythonStore : public ::c10d::Store {
 public:
  using ::c10d::Store::Store;

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that we can call the Python-side
  // function with a std::string instead of a std::vector<uint8_t>.
  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn =
        pybind11::get_overload(static_cast<const ::c10d::Store*>(this), "set");
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // Call function with a py::bytes object for the value.
    fn(key, toPyBytes(value));
  }

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that the Python-side function can
  // return a py::bytes instead of a std::vector<uint8_t>.
  std::vector<uint8_t> get(const std::string& key) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn =
        pybind11::get_overload(static_cast<const ::c10d::Store*>(this), "get");
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // Cast return value from Python to py::bytes, then implicitly
    // convert that to a std::string, so that we can construct a
    // std::vector<uint8_t>. There is no API for directly accessing
    // the contents of the py::bytes object.
    std::string str = pybind11::cast<py::bytes>(fn(key));
    return toVec8(str);
  }

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that the Python-side function can
  // return a py::bytes instead of a std::vector<uint8_t>.
  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "compare_set");
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // Cast return value from Python to py::bytes, then implicitly
    // convert that to a std::string, so that we can construct a
    // std::vector<uint8_t>. There is no API for directly accessing
    // the contents of the py::bytes object.
    std::string str = pybind11::cast<py::bytes>(
        fn(key, toPyBytes(expectedValue), toPyBytes(desiredValue)));
    return toVec8(str);
  }

  int64_t add(const std::string& key, int64_t value) override {
    PYBIND11_OVERLOAD_PURE(int64_t, ::c10d::Store, add, key, value);
  }

  int64_t getNumKeys() override {
    PYBIND11_OVERLOAD_PURE(int64_t, ::c10d::Store, getNumKeys);
  }

  bool deleteKey(const std::string& key) override {
    PYBIND11_OVERLOAD_PURE(bool, ::c10d::Store, deleteKey, key);
  }

  bool check(const std::vector<std::string>& keys) override {
    PYBIND11_OVERLOAD_PURE(bool, ::c10d::Store, check, keys);
  }

  void wait(const std::vector<std::string>& keys) override {
    PYBIND11_OVERLOAD_PURE(void, ::c10d::Store, wait, keys);
  }

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override {
    PYBIND11_OVERLOAD_PURE(void, ::c10d::Store, wait, keys, timeout);
  }

  c10::intrusive_ptr<Store> clone() override {
    PYBIND11_OVERLOAD_PURE(c10::intrusive_ptr<Store>, ::c10d::Store, clone);
  }

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that we can call the Python-side
  // function with a std::string instead of a std::vector<uint8_t>.
  void append(const std::string& key, const std::vector<uint8_t>& value)
      override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "append");
    if (!fn) {
      return Store::append(key, value);
    }
    // Call function with a py::bytes object for the value.
    fn(key, toPyBytes(value));
  }

  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "multi_get");
    if (!fn) {
      return Store::multiGet(keys);
    }
    std::vector<std::string> py_list =
        pybind11::cast<std::vector<std::string>>(fn(keys));
    std::vector<std::vector<uint8_t>> res;
    res.reserve(py_list.size());

    for (auto& str : py_list) {
      res.emplace_back(str.begin(), str.end());
    }

    return res;
  }

  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "multi_set");
    if (!fn) {
      return Store::multiSet(keys, values);
    }

    fn(keys, toPyBytes(values));
  }

  bool hasExtendedApi() const override {
    PYBIND11_OVERLOAD_NAME(
        bool, ::c10d::Store, "has_extended_api", hasExtendedApi);
  }
};

class PythonRequest : public ::c10d::control_plane::Request {
 public:
  const std::string& body() const override {
    PYBIND11_OVERRIDE_PURE(
        const std::string&, ::c10d::control_plane::Request, body);
  }

  const std::multimap<std::string, std::string>& params() const override {
    using MultiMap = const std::multimap<std::string, std::string>&;
    PYBIND11_OVERRIDE_PURE(MultiMap, ::c10d::control_plane::Request, params);
  }
};
class PythonResponse : public ::c10d::control_plane::Response {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  void setContent(std::string&& content, const std::string& content_type)
      override {
    PYBIND11_OVERRIDE_PURE_NAME(
        void,
        ::c10d::control_plane::Response,
        "set_content",
        setContent,
        content,
        content_type);
  }
  void setStatus(int status) override {
    PYBIND11_OVERRIDE_PURE_NAME(
        void, ::c10d::control_plane::Response, "set_status", setStatus, status);
  }
};

// Called from DDP's Python API to create a c10d Python comm hook object.
// The input state and callable comm_hook are Python objects. It later calls
// register_comm_hook function of the reducer input to register the hook.
void _register_comm_hook(
    ::c10d::Reducer& reducer,
    py::object state,
    py::object comm_hook) {
  reducer.register_comm_hook(std::make_unique<::c10d::PythonCommHook>(
      std::move(state), std::move(comm_hook)));
}

// Called from DDP's Python API to create a c10d C++ comm hook.
// The input is an enum hook type. It later calls register_builtin_comm_hook
// function of the reducer input to set the hook type.
void _register_builtin_comm_hook(
    ::c10d::Reducer& reducer,
    ::c10d::BuiltinCommHookType comm_hook_type) {
  reducer.register_builtin_comm_hook(comm_hook_type);
}

// Customize the metaclass of ::c10d::ReduceOp for the backward compatibility.
// https://github.com/pytorch/pytorch/pull/84243 changed ::c10d::ReduceOp to
// struct from enum, sacrificing some of the Python built-in function supports
// such as `isinstance` (see https://github.com/pytorch/pytorch/issues/87191)
// and `copy` (see
// https://github.com/pytorch/pytorch/pull/87303#discussion_r1002879700). Below,
// we define a custom `isinstance` in CPython/pybind11
// (`reduceopmeta___instancecheck__`) and modify the default metaclass of
// pybind11 (`GetReduceOpMetaclass`) so that
// `isinstance(torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp)`
// returns :obj:`True` as if `ReduceOp` is enum.
// Ref:
//   - https://docs.python.org/3/extending/newtypes_tutorial.html
//   - https://docs.python.org/3/c-api/typeobj.html?highlight=tp_methods
//   - https://github.com/pybind/pybind11/issues/2696
static PyObject* reduceopmeta___instancecheck__(
    PyObject* self,
    PyObject* args) {
  if (Py_TYPE(self) == Py_TYPE(args)) {
    Py_RETURN_TRUE;
  }
  if (std::string_view(args->ob_type->tp_name).find("RedOpType") !=
      std::string_view::npos) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}
// NOLINTNEXTLINE(*c-arrays)
static PyMethodDef reduceopmeta_methods[] = {
    {"__instancecheck__",
     reduceopmeta___instancecheck__,
     METH_O,
     "Custom `__instancecheck__` for ReduceOp"},
    {nullptr, nullptr}};
PyTypeObject* GetReduceOpMetaclass() {
  static auto* metaclass = [] {
    PyTypeObject* base_metaclass =
        pybind11::detail::get_internals().default_metaclass;
    // NOLINTNEXTLINE(*c-arrays)
    PyType_Slot slots[] = {
        {Py_tp_base, base_metaclass},
        {Py_tp_methods, reduceopmeta_methods},
        {0},
    };
    PyType_Spec spec = {};
    spec.name = "torch._C._distributed_c10d._ReduceOpMeta";
    // NOLINTNEXTLINE(*-narrowing-conversions)
    spec.basicsize = base_metaclass->tp_basicsize;
    spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    spec.slots = slots;
    PyTypeObject* metaclass =
        reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));
    if (!metaclass)
      throw py::error_already_set();
    return metaclass;
  }();
  return metaclass;
}

PyObject* c10d_init(PyObject* _unused, PyObject* noargs) {
  C10_LOG_API_USAGE_ONCE("c10d.python.import");

  auto c10d_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!c10d_module) {
    throw python_error();
  }

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m =
      torch_C_m.def_submodule("_distributed_c10d", "distributed c10d bindings");

  auto module = py::handle(m).cast<py::module>();

  module
      .def(
          "_register_comm_hook",
          &_register_comm_hook,
          py::arg("reducer"),
          py::arg("state"),
          py::arg("comm_hook"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_register_builtin_comm_hook",
          &_register_builtin_comm_hook,
          py::arg("reducer"),
          py::arg("comm_hook_type"));

  shared_ptr_class_<::c10d::GradBucket>(
      module,
      "GradBucket",
      R"(
This class mainly passes a flattened gradient tensor
(returned by :meth:`~torch.distributed.GradBucket.buffer`)
to DDP communication hook.
This tensor can be further decomposed into a list of per-parameter tensors within this bucket
(returned by :meth:`~torch.distributed.GradBucket.get_per_parameter_tensors`)
to apply layer-wise operations.
)")
      .def(
          "index",
          &::c10d::GradBucket::getIndex,
          py::call_guard<py::gil_scoped_release>(),
          R"(
.. warning::
    Since the buckets are rebuilt after the first iteration, should not rely on the indices at the beginning of training.

Returns:
    The index of a bucket that stores gradients of a few contiguous layers.
    All the gradients are bucketized.
)")
      .def(
          "buffer",
          &::c10d::GradBucket::getBuffer,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A flattened 1D ``torch.Tensor`` buffer,
    which can be further decomposed into a list of per-parameter tensors within this bucket.
)")
      .def(
          "gradients",
          &::c10d::GradBucket::getGradients,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A list of ``torch.Tensor``. Each tensor in the list corresponds to a gradient.
)")
      .def(
          "parameters",
          &::c10d::GradBucket::getParameters,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A list of ``torch.Tensor``. Each tensor in the list corresponds to a model
    parameter.
)")
      .def(
          "is_last",
          &::c10d::GradBucket::isLast,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    Whether this bucket is the last bucket to allreduce in an iteration.
    This also means that this bucket corresponds to the first few layers in the forward pass.
)")
      .def(
          "set_buffer",
          &::c10d::GradBucket::setBuffer,
          py::arg("buffer"),
          py::call_guard<py::gil_scoped_release>(),
          R"(
Replaces the tensor in the bucket with the input tensor buffer.
)");

  py::enum_<::c10d::BuiltinCommHookType>(module, "BuiltinCommHookType", R"(
An enum-like class for built-in communication hooks: ``ALLREDUCE`` and ``FP16_COMPRESS``.)")
      .value("ALLREDUCE", ::c10d::BuiltinCommHookType::ALLREDUCE)
      .value("FP16_COMPRESS", ::c10d::BuiltinCommHookType::FP16_COMPRESS);

  shared_ptr_class_<::c10d::Reducer>(module, "Reducer")
      .def(
          py::init(
              [](std::vector<at::Tensor> params,
                 std::vector<std::vector<size_t>> bucket_indices,
                 const std::vector<size_t>& per_bucket_size_limits,
                 c10::intrusive_ptr<::c10d::ProcessGroup> process_group,
                 std::vector<bool> expect_sparse_gradients,
                 int64_t bucket_bytes_cap,
                 bool find_unused_parameters,
                 bool gradient_as_bucket_view,
                 std::unordered_map<size_t, std::string> param_to_name_mapping,
                 int64_t first_bucket_bytes_cap,
                 bool skip_all_reduce_unused_params,
                 bool use_python_reducer) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                return std::make_unique<::c10d::Reducer>(
                    std::move(params),
                    std::move(bucket_indices),
                    std::move(process_group),
                    std::move(expect_sparse_gradients),
                    bucket_bytes_cap,
                    find_unused_parameters,
                    gradient_as_bucket_view,
                    std::move(param_to_name_mapping),
                    first_bucket_bytes_cap,
                    skip_all_reduce_unused_params,
                    use_python_reducer);
              }),
          py::arg("params"),
          py::arg("bucket_indices"),
          py::arg("per_bucket_size_limits"),
          py::arg("process_group"),
          py::arg("expect_sparse_gradients") = std::vector<bool>(),
          py::arg("bucket_bytes_cap") = ::c10d::kDefaultBucketBytesCap,
          py::arg("find_unused_parameters") = false,
          py::arg("gradient_as_bucket_view") = false,
          py::arg("param_to_name_mapping") =
              std::unordered_map<size_t, std::string>(),
          py::arg("first_bucket_bytes_cap") = ::c10d::kDefaultFirstBucketBytes,
          py::arg("skip_all_reduce_unused_params") = false,
          py::arg("use_python_reducer") = false)
      .def(
          "prepare_for_forward",
          &::c10d::Reducer::prepare_for_forward,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_backward",
          &::c10d::Reducer::prepare_for_backward,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_backward",
          [](::c10d::Reducer& reducer, const at::Tensor& output) -> void {
            reducer.prepare_for_backward({output});
          },
          py::call_guard<py::gil_scoped_release>())
      .def("get_backward_stats", &::c10d::Reducer::get_backward_stats)
      .def(
          "_install_post_backward_futures",
          [](::c10d::Reducer& reducer,
             const std::vector<std::shared_ptr<jit::PythonFutureWrapper>>&
                 futs) {
            c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(
                c10::FutureType::create(c10::TensorType::get()));
            for (const auto& fut : futs) {
              futures.push_back(fut->fut);
            }
            reducer.install_futures(futures);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_rebuild_buckets",
          &::c10d::Reducer::rebuild_buckets,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_zeros_like_grad_buckets",
          [](::c10d::Reducer& reducer) {
            return reducer.get_grad_buckets(/* return_zero_tensors */ true);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_optimizer_in_backward",
          [](::c10d::Reducer& reducer) { reducer.set_optimizer_in_backward(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_sparse_metadata",
          &::c10d::Reducer::setSparseMetadata,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_mixed_precision_param_dtype",
          [](::c10d::Reducer& reducer, py::object data_type_obj) {
            auto scalar_type =
                reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;
            reducer.set_mixed_precision_param_dtype(scalar_type);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_push_all_rebuilt_params",
          &::c10d::Reducer::push_rebuilt_params_for_all_indices,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_forward_pass_work_handle",
          &::c10d::Reducer::set_forward_pass_work_handle,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_local_used_map", &::c10d::Reducer::get_local_used_map_on_device)
      .def(
          "_set_ddp_runtime_logging_sample_rate",
          &::c10d::Reducer::set_ddp_runtime_logging_sample_rate,
          py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_static_graph",
          &::c10d::Reducer::set_static_graph,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_ddp_graph_static",
          &::c10d::Reducer::ddp_graph_static,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_delay_all_reduce",
          &::c10d::Reducer::delay_all_reduce,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_run_comm_hook",
          [](::c10d::Reducer& reducer, ::c10d::GradBucket& bucket)
              -> std::shared_ptr<jit::PythonFutureWrapper> {
            c10::intrusive_ptr<c10::ivalue::Future> fut =
                reducer.run_comm_hook(bucket);
            return std::make_shared<jit::PythonFutureWrapper>(fut);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_run_allreduce_hook",
          [](::c10d::Reducer& reducer, ::c10d::GradBucket& bucket)
              -> std::shared_ptr<jit::PythonFutureWrapper> {
            c10::intrusive_ptr<c10::ivalue::Future> fut =
                reducer.run_allreduce_hook(bucket);
            return std::make_shared<jit::PythonFutureWrapper>(fut);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_autograd_hook",
          [](::c10d::Reducer& reducer, int index) -> void {
            reducer.autograd_hook(index);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_logger",
          [](::c10d::Reducer& reducer,
             const std::shared_ptr<::c10d::Logger>& logger) {
            std::weak_ptr<::c10d::Logger> logger_weakref = logger;
            reducer.set_logger(logger_weakref);
          })
      .def(
          "_remove_autograd_hooks",
          [](::c10d::Reducer& reducer) { reducer.remove_autograd_hooks(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_check_reducer_finalized",
          [](::c10d::Reducer& reducer) { return reducer.check_finalized(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_reset_state",
          [](::c10d::Reducer& reducer) { return reducer.reset_state(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_update_process_group",
          [](::c10d::Reducer& reducer,
             c10::intrusive_ptr<::c10d::ProcessGroup> new_process_group) {
            return reducer.update_process_group(std::move(new_process_group));
          },
          py::call_guard<py::gil_scoped_release>());

  shared_ptr_class_<::c10d::Logger>(module, "Logger")
      .def(
          py::init([](const std::shared_ptr<::c10d::Reducer>& reducer) {
            // gil_scoped_release is not safe as a call_guard in init.
            // https://github.com/pybind/pybind11/issues/5473
            py::gil_scoped_release nogil{};

            return std::make_unique<::c10d::Logger>(reducer);
          }),
          py::arg("reducer"))
      .def(
          "set_construction_data_and_log",
          &::c10d::Logger::set_construction_data_and_log,
          py::arg("module_name"),
          py::arg("device_ids"),
          py::arg("output_device"),
          py::arg("broadcast_buffers"),
          py::arg("has_sync_bn"),
          py::arg("static_graph"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_runtime_stats_and_log",
          &::c10d::Logger::set_runtime_stats_and_log,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_error_and_log",
          [](::c10d::Logger& logger, const std::string& error) {
            logger.set_error_and_log(error);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_ddp_logging_data",
          &::c10d::Logger::get_ddp_logging_data,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_comm_hook_name",
          &::c10d::Logger::set_comm_hook,
          py::arg("comm_hook"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_uneven_input_join",
          &::c10d::Logger::set_uneven_input_join,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_static_graph",
          &::c10d::Logger::set_static_graph,
          py::call_guard<py::gil_scoped_release>());

  py::enum_<::c10d::DebugLevel>(module, "DebugLevel", R"(
      An enum whose values correspond to different debug levels of the
      torch.distributed package. Currently supporting OFF, INFO, and DETAIL,
      which can be set via the TORCH_DISTRIBUTED_DEBUG environment variable
      or via ``set_debug_level()`` function.
  )")
      .value("OFF", ::c10d::DebugLevel::Off)
      .value("INFO", ::c10d::DebugLevel::Info)
      .value("DETAIL", ::c10d::DebugLevel::Detail);

  module
      .def(
          "get_debug_level",
          ::c10d::debug_level,
          R"(Gets the debug level of the torch.distributed package.)")
      .def(
          "set_debug_level",
          ::c10d::setDebugLevel,
          R"(Sets the debug level of the torch.distributed package.)")
      .def(
          "set_debug_level_from_env",
          ::c10d::setDebugLevelFromEnvironment,
          R"(Sets the debug level of the torch.distributed package from the
          ``TORCH_DISTRIBUTED_DEBUG`` environment variable.)");

  // TODO(crcrpar): Hardening `ReduceOp`.
  //    While keeping most op types as enum value,
  //    making `PREMUL_SUM` callable, i.e., allowing for
  //    `ReduceOp.PREMUL_SUM(scale)` might be better as per @wanchaol.
  // https://pybind11.readthedocs.io/en/stable/classes.html#enumerations-and-internal-types
  py::class_<::c10d::ReduceOp> reduce_op(
      module,
      "ReduceOp",
      py::metaclass(reinterpret_cast<PyObject*>(GetReduceOpMetaclass())),
      R"(
An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``,
``MIN``, ``MAX``, ``BAND``, ``BOR``, ``BXOR``, and ``PREMUL_SUM``.

``BAND``, ``BOR``, and ``BXOR`` reductions are not available when
using the ``NCCL`` backend.

``AVG`` divides values by the world size before summing across ranks.
``AVG`` is only available with the ``NCCL`` backend,
and only for NCCL versions 2.10 or later.

``PREMUL_SUM`` multiplies inputs by a given scalar locally before reduction.
``PREMUL_SUM`` is only available with the ``NCCL`` backend,
and only available for NCCL versions 2.11 or later. Users are supposed to
use ``torch.distributed._make_nccl_premul_sum``.

Additionally, ``MAX``, ``MIN`` and ``PRODUCT`` are not supported for complex tensors.

The values of this class can be accessed as attributes, e.g., ``ReduceOp.SUM``.
They are used in specifying strategies for reduction collectives, e.g.,
:func:`reduce`.

This class does not support ``__members__`` property.)");

  reduce_op.def(py::init<::c10d::ReduceOp::RedOpType>())
      .def_readwrite("op", &::c10d::ReduceOp::op_);
  // The following are for some kind of backward compatibility.
  // Since c10d::ReduceOp had been an `enum class`, users can do comparison and
  // take hash of `::c10d::ReduceOp`. To avoid losing these functionality, here
  // I define some member methods.
  reduce_op
      // todo(crcrpar): Support `RedOpType == ReduceOp`.
      .def(
          // This calls `operator==(const ReduceOp::RedOpType)`
          "__eq__",
          [](const ::c10d::ReduceOp& self,
             const ::c10d::ReduceOp::RedOpType& other) {
            return self == other;
          })
      .def(
          // This calls `operator==(const ReduceOp)` for the future support of
          // `PREMUL_SUM` comparison
          "__eq__",
          [](const ::c10d::ReduceOp& self, const ::c10d::ReduceOp& other) {
            return self == other;
          })
      .def(
          // With the above custom `__eq__`'s, I have to manually support the
          // other types.
          "__eq__",
          // NOLINTNEXTLINE(performance-unnecessary-value-param)
          [](const ::c10d::ReduceOp& self, py::object) { return false; })
      .def(
          "__hash__",
          [](const ::c10d::ReduceOp& self) {
            return static_cast<uint8_t>(self.op_);
          })
      .def(
          "__copy__",
          [](const ::c10d::ReduceOp& self) { return ::c10d::ReduceOp(self); })
      .def(
          "__deepcopy__",
          [](const ::c10d::ReduceOp& self, const py::dict& memo) {
            return ::c10d::ReduceOp(self);
          })
      .def(py::pickle(
          [](const ::c10d::ReduceOp& r) {
            // __getstate__
            if (r.op_ != ::c10d::ReduceOp::RedOpType::PREMUL_SUM) {
              return py::make_tuple(r.op_, py::none());
            }
            TORCH_CHECK(r.supplement_.defined(), "Invalid PREMUL_SUM ReduceOp");
            const auto* preMulSupplement =
                reinterpret_cast<::c10d::NCCLPreMulSumSupplement*>(
                    r.supplement_.get());
            if (!preMulSupplement->tensor_factor.defined()) {
              return py::make_tuple(r.op_, preMulSupplement->double_factor);
            } else {
              return py::make_tuple(r.op_, preMulSupplement->tensor_factor);
            }
          },
          [](const py::tuple& t) {
            // __setstate__
            TORCH_CHECK(t.size() == 2, "Invalid state");
            const auto op =
                static_cast<::c10d::ReduceOp::RedOpType>(t[0].cast<uint8_t>());
            if (op != ::c10d::ReduceOp::RedOpType::PREMUL_SUM) {
              return ::c10d::ReduceOp(op);
            }
            const auto preMulSupplement_factor = t[1];
            if (py::isinstance<py::float_>(preMulSupplement_factor)) {
              return ::c10d::makeNCCLPreMulSum(t[1].cast<double>());
            } else {
              return ::c10d::makeNCCLPreMulSum(t[1].cast<at::Tensor>());
            }
          }));

  py::enum_<::c10d::ReduceOp::RedOpType>(reduce_op, "RedOpType")
      .value("SUM", ::c10d::ReduceOp::RedOpType::SUM)
      .value("AVG", ::c10d::ReduceOp::RedOpType::AVG)
      .value("PRODUCT", ::c10d::ReduceOp::RedOpType::PRODUCT)
      .value("MIN", ::c10d::ReduceOp::RedOpType::MIN)
      .value("MAX", ::c10d::ReduceOp::RedOpType::MAX)
      .value("BAND", ::c10d::ReduceOp::RedOpType::BAND)
      .value("BOR", ::c10d::ReduceOp::RedOpType::BOR)
      .value("BXOR", ::c10d::ReduceOp::RedOpType::BXOR)
      .value("PREMUL_SUM", ::c10d::ReduceOp::RedOpType::PREMUL_SUM)
      .export_values();

  // note(crcrpar): This could be removed because users will not pass
  // `RedOpType` to reduce collective ops Ref: [Implicit
  // conversions](https://pybind11.readthedocs.io/en/stable/advanced/classes.html#implicit-conversions)
  // Let us skip the explicit construction of `c10d::ReduceOp` from
  // `c10d::ReduceOp::RedOpType` in Python.
  py::implicitly_convertible<::c10d::ReduceOp::RedOpType, ::c10d::ReduceOp>();

  module
      .def(
          "_make_nccl_premul_sum",
          &::c10d::makeNCCLPreMulSum<double>,
          py::arg("factor").noconvert(),
          py::return_value_policy::copy, // seems safest
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_make_nccl_premul_sum",
          &::c10d::makeNCCLPreMulSum<at::Tensor>,
          py::arg("factor").noconvert(),
          py::return_value_policy::copy, // seems safest
          py::call_guard<py::gil_scoped_release>());

  module.def(
      "_set_thread_isolation_mode",
      &::c10d::set_thread_isolation_mode,
      py::arg("enable"));

  // Bindings for GroupRegistry.hpp
  //
  // Register a process group in the native registry. Process groups registered
  // via `_register_process_group` can be resolved from both Python and C++.
  module.def(
      "_register_process_group",
      [](const std::string& group_name,
         const c10::intrusive_ptr<::c10d::ProcessGroup>& group) {
        ::c10d::register_process_group(group_name, group);
      },
      py::arg("group_name"),
      py::arg("group"));

  // Resolve a process group from the native registry
  module.def(
      "_resolve_process_group",
      [](const std::string& group_name) {
        return ::c10d::resolve_process_group(group_name);
      },
      py::arg("group_name"));

  module.def(
      "_register_work",
      [](const at::Tensor& tensor,
         const c10::intrusive_ptr<::c10d::Work>& work) {
        py::object obj = py::cast(work);
        auto holder = c10::make_intrusive<::c10d::PyProcessGroup::PyWorkHolder>(
            work, obj);
        ::c10d::register_work(tensor, holder);
      },
      py::arg("tensor"),
      py::arg("work"));

  module.def("_get_work_registry_size", []() {
    return ::c10d::get_work_registry_size();
  });

  module.def(
      "_set_allow_inflight_collective_as_graph_input",
      [](bool value) {
        return ::c10d::set_allow_inflight_collective_as_graph_input(value);
      },
      py::arg("value"));

  module.def("_allow_inflight_collective_as_graph_input", []() {
    return ::c10d::allow_inflight_collective_as_graph_input();
  });

  // Remove a group from the native registry
  module.def(
      "_unregister_process_group",
      [](const std::string& group_name) {
        return ::c10d::unregister_process_group(group_name);
      },
      py::arg("group_name"));

  // Remove all process groups from the native registry
  module.def("_unregister_all_process_groups", []() {
    return ::c10d::unregister_all_process_groups();
  });

#ifdef USE_NVSHMEM
  // Initializes the device state in CUmodule so that it’s able to perform
  // NVSHMEM operations.
  module.def(
      "_nvshmemx_cumodule_init",
      ::c10d::nvshmem_extension::nvshmemx_cumodule_init,
      py::arg("module"));

  // Check if NVSHMEM is available on current system.
  module.def(
      "_is_nvshmem_available", ::c10d::nvshmem_extension::is_nvshmem_available);
#endif

  py::class_<::c10d::BroadcastOptions>(module, "BroadcastOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::BroadcastOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::BroadcastOptions::rootTensor)
      .def_readwrite("timeout", &::c10d::BroadcastOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::BroadcastOptions::asyncOp);

  py::class_<::c10d::AllreduceOptions>(module, "AllreduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::AllreduceOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::AllreduceOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::AllreduceOptions::asyncOp);

  py::class_<::c10d::AllreduceCoalescedOptions>(
      module, "AllreduceCoalescedOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::AllreduceCoalescedOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::AllreduceCoalescedOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::AllreduceCoalescedOptions::asyncOp);

  py::class_<::c10d::ReduceOptions>(module, "ReduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::ReduceOptions::reduceOp)
      .def_readwrite("rootRank", &::c10d::ReduceOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::ReduceOptions::rootTensor)
      .def_readwrite("timeout", &::c10d::ReduceOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::ReduceOptions::asyncOp);

  py::class_<::c10d::AllgatherOptions>(module, "AllgatherOptions")
      .def(py::init<>())
      .def_readwrite("timeout", &::c10d::AllgatherOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::AllgatherOptions::asyncOp);

  py::class_<::c10d::GatherOptions>(module, "GatherOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::GatherOptions::rootRank)
      .def_readwrite("timeout", &::c10d::GatherOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::GatherOptions::asyncOp);

  py::class_<::c10d::ScatterOptions>(module, "ScatterOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::ScatterOptions::rootRank)
      .def_readwrite("timeout", &::c10d::ScatterOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::ScatterOptions::asyncOp);

  py::class_<::c10d::ReduceScatterOptions>(module, "ReduceScatterOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::ReduceScatterOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::ReduceScatterOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::ReduceScatterOptions::asyncOp);

  py::class_<::c10d::BarrierOptions>(module, "BarrierOptions")
      .def(py::init<>())
      .def_readwrite("device_ids", &::c10d::BarrierOptions::device_ids)
      .def_readwrite("timeout", &::c10d::BarrierOptions::timeout)
      .def_readwrite("device", &::c10d::BarrierOptions::device)
      .def_readwrite("asyncOp", &::c10d::BarrierOptions::asyncOp);

  py::class_<::c10d::AllToAllOptions>(module, "AllToAllOptions")
      .def(py::init<>())
      .def_readwrite("timeout", &::c10d::AllToAllOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::AllToAllOptions::asyncOp);

  py::class_<::c10d::DistributedBackendOptions>(
      module, "_DistributedBackendOptions")
      .def(py::init<>())
      .def_readwrite("store", &::c10d::DistributedBackendOptions::store)
      .def_readwrite(
          "group_rank", &::c10d::DistributedBackendOptions::group_rank)
      .def_readwrite(
          "group_size", &::c10d::DistributedBackendOptions::group_size)
      .def_readwrite("timeout", &::c10d::DistributedBackendOptions::timeout)
      .def_readwrite("group_id", &::c10d::DistributedBackendOptions::group_id)
      .def_readwrite(
          "global_ranks_in_group",
          &::c10d::DistributedBackendOptions::global_ranks_in_group);

  py::class_<
      ::c10d::DMAConnectivity,
      c10::intrusive_ptr<::c10d::DMAConnectivity>>(module, "_DMAConnectivity")
      .def_readonly("device_type", &::c10d::DMAConnectivity::device_type)
      .def_readonly(
          "connection_type", &::c10d::DMAConnectivity::connection_type)
      .def_readonly("matrix", &::c10d::DMAConnectivity::matrix);

  module.def("_detect_dma_connectivity", ::c10d::detect_dma_connectivity);

  using SymmetricMemory = ::c10d::symmetric_memory::SymmetricMemory;
  py::class_<SymmetricMemory, c10::intrusive_ptr<SymmetricMemory>>(
      module, "_SymmetricMemory")
      .def_static("set_group_info", &::c10d::symmetric_memory::set_group_info)
      .def_static(
          "empty_strided_p2p",
          ::c10d::symmetric_memory::empty_strided_p2p,
          py::arg("size"),
          py::arg("stride"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("group_name") = py::none(),
          py::arg("alloc_id") = py::none())
      .def_static(
          "rendezvous",
          &::c10d::symmetric_memory::rendezvous,
          py::arg("tensor"),
          py::arg("group_name") = py::none())
      .def_static(
          "has_multicast_support",
          &::c10d::symmetric_memory::has_multicast_support)
      .def_static("set_backend", &::c10d::symmetric_memory::set_backend)
      .def_static("get_backend", &::c10d::symmetric_memory::get_backend)
      .def_static(
          "get_mempool_allocator",
          &::c10d::symmetric_memory::get_mempool_allocator)
      .def_property_readonly("rank", &SymmetricMemory::get_rank)
      .def_property_readonly("world_size", &SymmetricMemory::get_world_size)
      .def_property_readonly(
          "buffer_ptrs",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            std::vector<uintptr_t> ret;
            for (auto ptr : symm_mem->get_buffer_ptrs()) {
              ret.push_back(reinterpret_cast<uintptr_t>(ptr));
            }
            return ret;
          })
      .def_property_readonly(
          "buffer_ptrs_dev",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(symm_mem->get_buffer_ptrs_dev());
          })
      .def_property_readonly(
          "signal_pad_ptrs",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            std::vector<uintptr_t> ret;
            for (auto ptr : symm_mem->get_signal_pad_ptrs()) {
              ret.push_back(reinterpret_cast<uintptr_t>(ptr));
            }
            return ret;
          })
      .def_property_readonly(
          "signal_pad_ptrs_dev",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(
                symm_mem->get_signal_pad_ptrs_dev());
          })
      .def_property_readonly(
          "multicast_ptr",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(symm_mem->get_multicast_ptr());
          })
      .def_property_readonly("buffer_size", &SymmetricMemory::get_buffer_size)
      .def_property_readonly(
          "signal_pad_size", &SymmetricMemory::get_signal_pad_size)
      .def_property_readonly("offset", &SymmetricMemory::get_offset)
      .def(
          "get_buffer",
          &SymmetricMemory::get_buffer,
          py::arg("rank"),
          py::arg("sizes"),
          py::arg("dtype"),
          py::arg("storage_offset") = 0)
      .def(
          "get_signal_pad",
          &SymmetricMemory::get_signal_pad,
          py::arg("rank"),
          py::arg("sizes") = py::list(),
          py::arg("dtype") = py::none(),
          py::arg("storage_offset") = 0)
      .def(
          "barrier",
          &SymmetricMemory::barrier,
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "put_signal",
          &SymmetricMemory::put_signal,
          py::arg("dst_rank"),
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "wait_signal",
          &SymmetricMemory::wait_signal,
          py::arg("src_rank"),
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "get_remote_tensor",
          &SymmetricMemory::get_remote_tensor,
          py::arg("peer"),
          py::arg("sizes"),
          py::arg("dtype"))
      // Util functions that are often used together with symmetric memory but
      // not necessarily directly on symmetric memory.
      .def_static(
          "stream_write_value32",
          [](at::Tensor& input, int64_t offset, int64_t val) {
            // The range of `val` is checked inside the op
            auto op =
                c10::Dispatcher::singleton()
                    .findSchemaOrThrow("symm_mem::stream_write_value32_", "")
                    .typed<at::Tensor(at::Tensor&, int64_t, int64_t)>();
            return op.call(input, offset, val);
          },
          py::arg("input"),
          py::arg("offset"),
          py::arg("val"))
      .def_static(
          "memset32",
          [](at::Tensor& input, int64_t offset, int64_t val, int64_t count) {
            // The range of `val` is checked inside the op
            auto op = c10::Dispatcher::singleton()
                          .findSchemaOrThrow("symm_mem::memset32_", "")
                          .typed<at::Tensor(
                              at::Tensor&, int64_t, int64_t, int64_t)>();
            return op.call(input, offset, val, count);
          },
          py::arg("input"),
          py::arg("offset"),
          py::arg("val"),
          py::arg("count") = 1);

  auto store =
      py::class_<::c10d::Store, c10::intrusive_ptr<::c10d::Store>, PythonStore>(
          module,
          "Store",
          R"(
Base class for all store implementations, such as the 3 provided by PyTorch
distributed: (:class:`~torch.distributed.TCPStore`, :class:`~torch.distributed.FileStore`,
and :class:`~torch.distributed.HashStore`).
)")
          // Default constructor.
          .def(py::init<>())
          .def(
              "clone",
              &::c10d::Store::clone,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Clones the store and returns a new object that points to the same underlying
store. The returned store can be used concurrently with the original object.
This is intended to provide a safe way to use a store from multiple threads by
cloning one store per thread.
)")
          // Convert from std::string to std::vector<uint8>.
          .def(
              "set",
              [](::c10d::Store& store,
                 const std::string& key,
                 const std::string& value) { store.set(key, toVec8(value)); },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Inserts the key-value pair into the store based on the supplied ``key`` and
``value``. If ``key`` already exists in the store, it will overwrite the old
value with the new supplied ``value``.

Arguments:
    key (str): The key to be added to the store.
    value (str): The value associated with ``key`` to be added to the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set("first_key", "first_value")
    >>> # Should return "first_value"
    >>
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/distributed/c10d`):

- [`ProcessGroupWrapper.cpp_docs.md_docs.md`](./ProcessGroupWrapper.cpp_docs.md_docs.md)
- [`c10d.h_kw.md_docs.md`](./c10d.h_kw.md_docs.md)
- [`TCPStoreLibUvBackend.cpp_kw.md_docs.md`](./TCPStoreLibUvBackend.cpp_kw.md_docs.md)
- [`ProcessGroupGlooCuda.cpp_docs.md_docs.md`](./ProcessGroupGlooCuda.cpp_docs.md_docs.md)
- [`NanCheck.cu_docs.md_docs.md`](./NanCheck.cu_docs.md_docs.md)
- [`python_callback_work.hpp_kw.md_docs.md`](./python_callback_work.hpp_kw.md_docs.md)
- [`sequence_num.hpp_kw.md_docs.md`](./sequence_num.hpp_kw.md_docs.md)
- [`Functional.hpp_kw.md_docs.md`](./Functional.hpp_kw.md_docs.md)
- [`TCPStoreBackend.cpp_kw.md_docs.md`](./TCPStoreBackend.cpp_kw.md_docs.md)
- [`ProcessGroupUCC.cpp_kw.md_docs.md`](./ProcessGroupUCC.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `init.cpp_docs.md_docs.md`
- **Keyword Index**: `init.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
