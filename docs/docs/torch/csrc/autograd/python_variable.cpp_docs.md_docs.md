# Documentation: `docs/torch/csrc/autograd/python_variable.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/python_variable.cpp_docs.md`
- **Size**: 53,849 bytes (52.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/python_variable.cpp`

## File Metadata

- **Path**: `torch/csrc/autograd/python_variable.cpp`
- **Size**: 149,760 bytes (146.25 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/DTensorState.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Resize.h>
#include <c10/core/DeviceType.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/util/FbcodeMaps.h>
#include <c10/util/SmallVector.h>
#include <c10/util/irange.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_torch_functions.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/csrc/autograd/utils/error_messages.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/Placement.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/pyobject_preservation.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>

#include <torch/csrc/utils/torch_dispatch_mode.h>

#include <ATen/ATen.h>

#include <structmember.h>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using namespace at;
using namespace torch;
using namespace torch::autograd;

namespace {
class OperatorArgsKwargsView {
 public:
  OperatorArgsKwargsView(
      const c10::OperatorHandle& op,
      const std::vector<c10::IValue>& arguments);
  using args_iterator = const c10::IValue*;

  args_iterator args_begin() const {
    return arguments_.data();
  }

  args_iterator args_end() const {
    return arguments_.data() + positional_default_start_;
  }

  auto num_positional_args() const {
    return positional_default_start_;
  }

  auto kwarg_start_index() const {
    return first_non_default_kwarg_;
  }

  struct kwargs_iterator {
    kwargs_iterator() = default;
    kwargs_iterator(const OperatorArgsKwargsView* parent, size_t current)
        : parent_(parent), current_(current) {}

    kwargs_iterator(const kwargs_iterator&) = default;
    kwargs_iterator& operator=(const kwargs_iterator&) = default;

    kwargs_iterator& operator++() {
      do {
        current_++;
      } while (current_ < parent_->arguments_.size() &&
               parent_->is_default(current_));
      return *this;
    }

    kwargs_iterator operator++(int) {
      auto copy = *this;
      ++(*this);
      return copy;
    }

    const c10::IValue& operator*() const {
      return parent_->arguments_[current_];
    }

    const c10::IValue* operator->() const {
      return &operator*();
    }

    int64_t underlying_index() const {
      return current_;
    }

    bool operator==(const kwargs_iterator& rhs) const {
      return parent_ == rhs.parent_ && current_ == rhs.current_;
    }

    bool operator!=(const kwargs_iterator& rhs) {
      return !(*this == rhs);
    }

   private:
    const OperatorArgsKwargsView* parent_ = nullptr;
    size_t current_ = 0;
  };

  kwargs_iterator kwargs_begin() const {
    return kwargs_iterator(this, first_non_default_kwarg_);
  }

  kwargs_iterator kwargs_end() const {
    return kwargs_iterator(this, arguments_.size());
  }

 private:
  bool is_default(size_t idx) const {
    const auto& arg = op_.schema().arguments()[idx];
    if (!arg.default_value().has_value()) {
      return false;
    }
    const auto& default_ivalue = *arg.default_value();
    const auto& ivalue = arguments_[idx];
    if (default_ivalue != ivalue) {
      return false;
    }
    return true;
  }

  const c10::OperatorHandle& op_;
  c10::ArrayRef<c10::IValue> arguments_;
  // About all the pointers:
  //
  // f(int x, int y = 0, *, int z = 0)
  //                                  ^- arguments.size()
  //                        ^- kwarg_only_start
  //          ^- positional_default_start
  //   ^- 0
  int64_t positional_default_start_;
  int64_t first_non_default_kwarg_;
};

OperatorArgsKwargsView::OperatorArgsKwargsView(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments)
    : op_(op), arguments_(arguments) {
  // Find the split point between kwarg-only and regular.  Since most functions
  // don't have kwarg-only arguments, it is more efficient to scan from the
  // right (but ideally, this would just be precomputed in FunctionSchema
  // itself).  (NB: minus one in the loop is because we're testing if the
  // *next* argument is kwarg-only before we advance the starting index)
  const int64_t signed_arguments_size = static_cast<int64_t>(arguments.size());
  int64_t kwarg_only_start = signed_arguments_size;
  for (; kwarg_only_start > 0; kwarg_only_start--) {
    const auto& arg = op.schema().arguments()[kwarg_only_start - 1];
    if (!arg.kwarg_only()) {
      break;
    }
  }

  // Find the first positional argument that isn't defaulted
  positional_default_start_ = kwarg_only_start;
  for (; positional_default_start_ > 0; positional_default_start_--) {
    if (!is_default(positional_default_start_ - 1)) {
      break;
    }
  }

  // kwargs_iterator will skip default kwargs when incremented, but we
  // need to skip any initial run of default kwargs ourselves.
  first_non_default_kwarg_ = kwarg_only_start;
  for (; first_non_default_kwarg_ < signed_arguments_size;
       ++first_non_default_kwarg_) {
    if (!is_default(first_non_default_kwarg_)) {
      break;
    }
  }
}
} // namespace

std::pair<py::object, py::dict> parseIValuesToPyArgsKwargs(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments) {
  TORCH_CHECK(
      PyGILState_Check(),
      "GIL must be held before you call parseIValuesToPyArgsKwargs");
  const auto& schema = op.schema();
  py::dict kwargs;

  OperatorArgsKwargsView args_kwargs(op, arguments);
  auto args = py::reinterpret_steal<py::object>(
      PyTuple_New(args_kwargs.num_positional_args()));

  auto schemaAwareToPyObject =
      [&schema](size_t idx, const c10::IValue& argument) -> py::object {
    const auto& arg = schema.arguments()[idx];
    auto match = [&](c10::TypeKind kind) {
      const auto& t = arg.real_type();
      if (t->kind() == kind)
        return true;
      if (auto opt_t = t->cast<c10::OptionalType>()) {
        if (opt_t->getElementType()->kind() == kind)
          return true;
      }
      return false;
    };
    if (argument.isNone()) {
      return py::none();
    } else if (match(c10::ScalarTypeType::Kind)) {
      auto* obj = getTHPDtype(static_cast<c10::ScalarType>(argument.toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::LayoutType::Kind)) {
      auto* obj = getTHPLayout(static_cast<c10::Layout>(argument.toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::MemoryFormatType::Kind)) {
      return py::cast(static_cast<c10::MemoryFormat>(argument.toInt()));
    } else {
      return torch::jit::toPyObject(argument);
    }
  };

  // Populate positional arguments
  size_t idx = 0;
  for (auto argument_it = args_kwargs.args_begin();
       argument_it != args_kwargs.args_end();
       ++argument_it) {
    PyTuple_SET_ITEM(
        args.ptr(),
        idx,
        schemaAwareToPyObject(idx, *argument_it).release().ptr());
    idx++;
  }

  // Populate keyword arguments
  for (auto argument_it = args_kwargs.kwargs_begin();
       argument_it != args_kwargs.kwargs_end();
       ++argument_it) {
    const auto& arg = schema.arguments()[argument_it.underlying_index()];
    kwargs[py::cast(arg.name())] =
        schemaAwareToPyObject(argument_it.underlying_index(), *argument_it);
  }
  return std::make_pair(std::move(args), std::move(kwargs));
}

void pushPyOutToStack(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    py::object out,
    const char* msg) {
  TORCH_CHECK(
      PyGILState_Check(), "GIL must be held before you call pushPyOutToStack");
  const auto& schema_returns = op.schema().returns();
  const auto num_returns = schema_returns.size();
  if (num_returns == 0) {
    // Check that we got a None return from Python. Anything else is an error.
    TORCH_CHECK(
        out.is_none(),
        "Expected ",
        msg,
        " for ",
        op.operator_name(),
        " to return None but it returned something else instead.");
  } else if (num_returns == 1) {
    torch::jit::push(
        stack, torch::jit::toIValue(out.ptr(), schema_returns[0].real_type()));
  } else {
    auto outs = py::cast<py::sequence>(out);
    for (const auto idx : c10::irange(outs.size())) {
      torch::jit::push(
          stack,
          torch::jit::toIValue(
              outs[idx].ptr(), schema_returns[idx].real_type()));
    }
  }
}

namespace {

c10::TensorImpl::SizesStridesPolicy parseSizesStridesPolicyArgument(
    std::string_view arg) {
  if (arg == "strides") {
    return c10::TensorImpl::SizesStridesPolicy::CustomStrides;
  }

  if (arg == "sizes") {
    return c10::TensorImpl::SizesStridesPolicy::CustomSizes;
  }

  TORCH_CHECK_VALUE(
      false,
      "Unknown sizes_strides_policy: ",
      arg,
      "; expected 'strides' or 'sizes'");
}
} // anonymous namespace

PyObject* THPVariableClass = nullptr;

PyObject* ParameterClass = nullptr;

static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    const at::TensorBase& _var,
    bool allow_preexisting_pyobj = false,
    std::optional<bool> has_torch_dispatch_if_known = std::nullopt);

// clang-tidy gets confused by static const
static constexpr const char* VOLATILE_WARNING =
    "volatile was removed and now has no effect. Use "
    "`with torch.no_grad():` instead.";

static bool check_has_torch_dispatch(PyObject* obj) {
  PyTypeObject* tp = Py_TYPE(obj);
  if (THPVariable_CheckTypeExact(tp)) {
    return false;
  }
  py::object attr = PyObject_FastGetAttrString(obj, "__torch_dispatch__");
  return (
      attr.ptr() != nullptr &&
      attr.ptr() != torch::disabled_torch_dispatch_impl());
}

// NOLINTNEXTLINE(*-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyObject* device_to_py_class_[static_cast<size_t>(
    c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

void registerPythonTensorClass(
    const std::string& device,
    PyObject* python_tensor_class) {
  c10::Device dev(device);

  TORCH_CHECK(
      dev.type() == kXLA, "Only the python class for XLA can be overridden");
  if (device_to_py_class_[static_cast<size_t>(dev.type())] != nullptr) {
    TORCH_WARN(
        "Overriding a previously registered python class for ", dev.str());
  }

  device_to_py_class_[static_cast<size_t>(dev.type())] = python_tensor_class;
}

static PyObject* getPythonTensorClass(c10::Device d) {
  return device_to_py_class_[static_cast<size_t>(d.type())];
}

void activateGPUTrace() {
  c10::impl::GPUTrace::set_trace(getPyInterpreter());
}

PyObject* THPVariable_Wrap(const at::TensorBase& var) {
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  if (c10::impl::HermeticPyObjectTLS::get_state()) {
    return THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, var);
  }

  std::optional<PyObject*> mb_obj =
      var.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          /*ignore_hermetic_tls=*/false);
  if (mb_obj.has_value()) {
    auto obj = *mb_obj;
    if (obj) {
      if (var.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj()) {
        // C++ owns the Python object; this implies there weren't any other
        // owning references to the Python object.  Since we're making the
        // object "live" again on Python side, let's flip back the ownership
        // (Python owns C++) as it would now be unsound to deallocate the C++
        // object if all C++ references go to zero
        var.unsafeGetTensorImpl()->pyobj_slot()->set_owns_pyobj(false);
        reinterpret_cast<THPVariable*>(obj)->cdata =
            MaybeOwned<Variable>::owned(Variable(var));
        // NB: incref is not necessary, because we are "stealing" the previous
        // ownership from the Variable to return it here for the wrap
        return obj;
      }
      Py_INCREF(obj);
      return obj;
    }
    // TODO: a better invariant is that if we tagged, we MUST have a valid
    // PyObject.  That's PyObject preservation
    // (https://github.com/pytorch/pytorch/pull/56017).  Prior to this PR
    // being a thing, the PyObject field will get cleared when all references
    // to the Python object are removed.
  }

  if (C10_LIKELY(var.device().type() != c10::kXLA)) {
    return THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, var);
  }

  if (auto clazz = getPythonTensorClass(var.device())) {
    return THPVariable_NewWithVar((PyTypeObject*)clazz, var);
  }

  return THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, var);
}

static bool isResurrectable(THPVariable* self) {
  // We want to divide this check into 2 cases.

  // 1. C++ owns PyObject (in this case, self->cdata.unsafeIsBorrowed() is
  // true). You might think that in this case, it is impossible for tp_clear to
  // be called: surely the C++ reference to the PyObject is keeping it live? And
  // you'd be right! In fact, when C++ owns the PyObject, we have an invariant
  // that the refcount on the PyObject should be precisely one (because if you
  // take out another reference to the PyObject, we're supposed to flip the
  // ownership pointer back). In reality, you can violate this invariant
  // temporarily with weak references, so we don't test for it in asserts.

  // 2. PyObject owns C++ (in this case, self->cdata.unsafeIsBorrowed() is
  // false). In this case, tp_clear can get called if the PyObject is referenced
  // from a dead cycle, and nowhere else. But if resurrection did not occur,
  // then the reference to C++ from the PyObject must be the ONLY reference to
  // the C++ object.
  if (self->cdata.unsafeIsBorrowed()) {
    return false;
  }
  auto const& tensor = THPVariable_Unpack(self);
  if (!tensor.defined() || tensor.use_count() <= 1) {
    return false;
  }
  // Check if this is hermetic. If it is, no resurrection.
  if (tensor.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          /*ignore_hermetic_tls=*/false) != (PyObject*)self) {
    return false;
  }
  return true;
}

// returns true if successfully rezzed; if so, cancel the
// rest of deallocation
static bool THPVariable_tryResurrect(THPVariable* self) {
  const auto& tensor = THPVariable_Unpack(self);

  if (!isResurrectable(self)) {
    return false;
  }

  // At this point, we are definitely going to resurrect the tensor. So, the
  // tensor better be defined :)
  TORCH_INTERNAL_ASSERT(tensor.defined());

  // There are other C++ owners of the tensor.  Flip ownership
  // so that C++ owns this Python object, and cancel deallocation.
  TORCH_INTERNAL_ASSERT(
      !tensor.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj());

  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  auto maybe_pyobj = tensor_impl->pyobj_slot()->check_pyobj(
      /*ignore_hermetic_tls=*/false);

  TORCH_INTERNAL_ASSERT(
      maybe_pyobj.has_value(),
      "Trying to preserve a Python tensor whose PyObjectSlot does not have a PyObject");

  tensor_impl->pyobj_slot()->set_owns_pyobj(true);

  // Resurrect the Python object.  This is something CPython does
  // internally occasionally, see
  // https://github.com/python/cpython/blob/b98eba5bc2ffbe7a0ed49d540ebc4f756ae61985/Objects/object.c#L248-L259
  // so we just copy the pattern here.  Note that we don't have to worry
  // about saving and restoring the refcount (as the quoted code does)
  // because we actually DO need to reset the refcount to one here, we
  // can't assume that some other code has taken care of it.
  // NB: this will overreport _Py_RefTotal but based on inspection of object.c
  // there is no way to avoid this

  // When resurrecting, we MUST use _Py_NewReference and not Py_INCREF to
  // ensure the PyObject is in a valid state
  _Py_NewReference((PyObject*)self);

  // Flip THPVariable to be non-owning
  // (near use-after-free miss here: fresh MaybeOwned is created breaking
  // reference on Tensor in struct BEFORE we overwrite the old one)
  TORCH_INTERNAL_ASSERT(!c10::impl::HermeticPyObjectTLS::get_state());
  self->cdata = MaybeOwned<Variable>::borrowed(tensor);

  // NB: At this point, tensor *could* be dead (e.g., some other C++ thread
  // decrefed it.)  At this point, it is probably waiting on the GIL to
  // deallocate the Python object and will kill self, BUT NOT YET.

  return true;
}

static int THPFake_traverse(THPVariable* self, visitproc visit, void* arg) {
  TORCH_INTERNAL_ASSERT(
      false, "TensorBase tp_traverse function was not overridden properly");
  return 0;
}

static int THPFake_clear(THPVariable* self) {
  TORCH_INTERNAL_ASSERT(
      false, "TensorBase tp_clear function was not overridden properly");
  return 0;
}

static PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs);

static PyObject* THPVariable_fix_weakref(PyObject* self, PyObject* noargs) {
  const auto& var = THPVariable_Unpack(self);
  Py_DECREF(THPVariable_Wrap(var));
  Py_RETURN_NONE;
}

// Maps the given python callable over a vector of items, returning a vector
// of the same type of items.
template <typename T>
static std::vector<T> map_py_func(
    const py::function& func,
    const std::vector<T>& items) {
  std::vector<T> new_items;
  new_items.reserve(items.size());
  for (auto& item : items) {
    new_items.emplace_back(py::cast<T>(func(item)));
  }
  return new_items;
}

template <>
std::vector<at::Tensor> map_py_func(
    const py::function& func,
    const std::vector<at::Tensor>& items) {
  std::vector<at::Tensor> new_items;
  new_items.reserve(items.size());
  for (auto& item : items) {
    auto output = func(item);
    if (output.is(py::none())) {
      // treat None value as an undefined tensor
      new_items.emplace_back();
    } else {
      new_items.emplace_back(py::cast<at::Tensor>(output));
    }
  }
  return new_items;
}

static PyObject* view_func_impl(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs,
    bool check_has_same_meta) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(_self);

  static PythonArgParser parser({
      "_view_func(Tensor new_base, PyObject* symint_visitor_fn=None, PyObject* tensor_visitor_fn=None)",
  });
  ParsedArgs<3> parsed_args{};
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  auto new_base = r.tensor(0);
  PyObject* symint_visitor_fn = r.pyobject(1);
  PyObject* tensor_visitor_fn = r.pyobject(2);

  // Ensure that self is indeed a backward differentiable view
  // If not, we return an undefined Tensor (None) and let the user handle it.
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  at::Tensor out;
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    const auto& view_info = diff_view_meta->get_backward_view();
    // Ensure that the newly provided base is similar to the original base
    if (!check_has_same_meta ||
        torch::autograd::utils::has_same_meta(new_base, view_info.base_)) {
      // Do the actual view replay
      if (view_info.has_view_fn()) {
        auto& view_func = view_info.view_fn();

        // Determine new SymInt / tensor state as needed.
        std::optional<std::vector<c10::SymInt>> new_symints = std::nullopt;
        if (symint_visitor_fn != Py_None) {
          new_symints = map_py_func(
              py::cast<py::function>(symint_visitor_fn),
              view_func.get_symints());
        }

        std::optional<std::vector<at::Tensor>> new_tensors = std::nullopt;
        if (tensor_visitor_fn != Py_None) {
          new_tensors = map_py_func(
              py::cast<py::function>(tensor_visitor_fn),
              view_func.get_tensors());
        }

        // call view func
        if (new_symints.has_value() || new_tensors.has_value()) {
          out = (*view_func.clone_and_set(new_symints, new_tensors))(new_base);
        } else {
          out = view_func(new_base);
        }
      } else {
        out = new_base.as_strided(
            self.sizes(), self.strides(), self.storage_offset());
      }
    }
  }
  return THPVariable_Wrap(out);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_view_func(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  return view_func_impl(self_, args, kwargs, /*check_has_same_meta=*/true);
}

static PyObject* THPVariable_view_func_unsafe(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  return view_func_impl(self_, args, kwargs, /*check_has_same_meta=*/false);
}

static PyObject* rev_view_func_impl(PyObject* self_, PyObject* arg) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(self_);
  TORCH_CHECK(
      THPVariable_Check(arg),
      "_rev_view_func expect a single argument that is a Tensor");
  const auto& new_view = THPVariable_Unpack(arg);

  // Ensure that self is indeed a backward differentiable view
  // If not, we return an undefined Tensor (None) and let the user handle it.
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  at::Tensor out;
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    const auto& view_info = diff_view_meta->get_backward_view();
    // Do the actual view replay
    TORCH_CHECK(view_info.has_view_fn(), "No _rev_view_func() found");
    out = view_info.rev_view_fn()(new_view);
  }
  return THPVariable_Wrap(out);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_rev_view_func_unsafe(
    PyObject* self_,
    PyObject* arg) {
  return rev_view_func_impl(self_, arg);
}

// Instantiates a subclass of self with the same data.
static PyObject* THPVariable_as_subclass(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(_self);
  static PythonArgParser parser({
      "as_subclass(PyObject* cls)",
  });
  ParsedArgs<1> parsed_args{};
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");
  // guard completely turns off torch dispatch modes, doesn't just pop off the
  // stack
  torch_dispatch_mode::StashTorchDispatchStackGuard td_g;
  c10::impl::DisablePythonDispatcher dpd_g;
  return THPVariable_NewWithVar((PyTypeObject*)cls, self.alias());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_make_subclass(
    PyObject* _ignored,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "_make_subclass(PyObject* cls, Tensor data, bool require_grad=False, *, std::string_view? dispatch_sizes_strides_policy=None, bool dispatch_device=False, bool dispatch_layout=False, Device? device_for_backend_keys=None)",
  });
  ParsedArgs<7> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");
  // guard completely turns off torch dispatch modes, doesn't just pop off the
  // stack
  torch_dispatch_mode::StashTorchDispatchStackGuard td_g;
  c10::impl::DisablePythonDispatcher dpd_g;
  auto data =
      r.tensor(1).detach(); // creates a fresh Tensor (DEFINITELY_UNINITIALIZED)
  // We set `data`'s `allow_tensor_metadata_change` to true here, because we
  // want to allow the following use case for backward compatibility:
  //
  // ```python
  // rnn = torch.nn.RNN(100, 100, 2)
  // # The following calls `torch._cudnn_rnn_flatten_weight(rnn._flat_weights,
  // ...)`, # which changes storage of `rnn`'s weights in-place
  // rnn.flatten_parameters()
  // ```
  data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
  data.set_requires_grad(r.toBool(2));
  const auto sizes_strides_policy = r.stringViewOptional(3);
  if (sizes_strides_policy.has_value()) {
    data.unsafeGetTensorImpl()->set_python_custom_sizes_strides(
        parseSizesStridesPolicyArgument(*sizes_strides_policy));
  }
  if (r.toBool(4)) {
    data.unsafeGetTensorImpl()->set_python_custom_device(true);
  }
  if (r.toBool(5)) {
    data.unsafeGetTensorImpl()->set_python_custom_layout(true);
  }
  if (!r.isNone(6)) {
    data.unsafeGetTensorImpl()->_change_backend_component_keys(r.device(6));
  }

  return THPVariable_NewWithVar((PyTypeObject*)cls, data);
  END_HANDLE_TH_ERRORS
}

// Shared code factored out of THPVariable_make_wrapper_subclass and
// THPVariable_dtensor__new__.
static Tensor make_tensor_for_subclass_helper(
    SymIntArrayRef sym_sizes,
    OptionalSymIntArrayRef sym_strides,
    const std::optional<c10::SymInt>& sym_storage_offset,
    const TensorOptions& options,
    const std::optional<c10::SymInt>& storage_size,
    std::optional<DispatchKeySet> extra_dispatch_keys) {
  AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
  tracer::impl::NoTracerDispatchMode tracer_guard{};

  c10::SymInt size_bytes;
  auto dtype_itemsize = static_cast<int64_t>(options.dtype().itemsize());

  if (storage_size.has_value()) {
    size_bytes = storage_size.value();
  } else if (sym_strides.has_value()) {
    size_bytes = at::detail::computeStorageNbytes(
        sym_sizes,
        sym_strides.value(),
        dtype_itemsize,
        sym_storage_offset.value_or(0));
  } else {
    size_bytes = at::detail::computeStorageNbytesContiguous(
        sym_sizes, dtype_itemsize, sym_storage_offset.value_or(0));
  }

  // We use storages **only** to track aliasing of subclasses during tracing.
  // The actual data pointers are not valid.
  Storage storage{
      Storage::use_byte_size_t{},
      size_bytes,
      at::DataPtr{nullptr, options.device()},
      /*allocator=*/c10::GetAllocator(c10::kMeta),
      /*resizable=*/true};

  auto keys = c10::DispatchKeySet({options.computeDispatchKey()});
  if (extra_dispatch_keys.has_value()) {
    keys = keys | *extra_dispatch_keys;
  }
  Tensor tensor = at::detail::make_tensor<TensorImpl>(
      std::move(storage), keys, options.dtype());

  TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();

  if (sym_strides.has_value()) {
    tensor_impl->set_sizes_and_strides(
        sym_sizes, sym_strides.value(), sym_storage_offset);
  } else {
    TORCH_CHECK(
        !sym_storage_offset.has_value(),
        "setting storage offset without stride not supported");
    tensor_impl->generic_set_sizes_contiguous(sym_sizes);
  }
  return tensor;
}

static PyObject* THPVariable_make_wrapper_subclass(
    PyObject* /*unused*/,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // NB: pin_memory doesn't actually do anything
  // TODO: strides variant?

  // cls: Python subclass type
  // size, strides, storage_offset, memory_format, dtype: self-explanatory
  // layout: memory layout, e.g. for types of Nested Tensors or other sparse
  //         tensors
  // pin_memory, requires_grad: self-explanatory
  // dispatch_sizes_strides_policy: string - which sizes/strides we should
  //                                dispatch to a custom python implementation.
  // dispatch_device: whether to dispatch to a custom python implementation
  //                  for device
  // dispatch_layout: whether to dispatch to a custom python implementation
  //                  for layout
  // _extra_dispatch_keys: additional dispatch keys to add to the tensor
  // storage_size: if provided, skip storage size calculation and just use the
  //               value provided. One use case is for Nested Tensor, where the
  //               storage size cannot be calculated from the sizes/strides
  //               (because they contain a NestedInt).
  static PythonArgParser parser({
      "_make_wrapper_subclass(PyObject* cls, SymIntArrayRef size, SymIntArrayRef? strides=None, "
      "SymInt? storage_offset=None, MemoryFormat? memory_format=None, ScalarType dtype=None, "
      "Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False, "
      "std::string_view? dispatch_sizes_strides_policy=None, bool dispatch_device=False, bool dispatch_layout=False, "
      "DispatchKeySet _extra_dispatch_keys=None, SymInt? storage_size=None)",
  });
  ParsedArgs<15> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);

  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");

  // This is an important safety check; without it, the default behavior will be
  // to continue on to the underlying CPU/CUDA kernel advertised by the dispatch
  // key, which will immediately segfault because the data pointer is null.  By
  // forcing users to define __torch_dispatch__ we ensure this does not happen
  // TODO: This check is not complete; because the user can disable torch
  // dispatch and then go again, triggering segfault.  TBH I'm thinking I want
  // to delete this function entirely
  py::object attr = PyObject_FastGetAttrString(cls, "__torch_dispatch__");
  TORCH_CHECK_TYPE(
      attr.ptr() != nullptr &&
          attr.ptr() != torch::disabled_torch_dispatch_impl(),
      ((PyTypeObject*)cls)->tp_name,
      " must define __torch_dispatch__");

  const auto options = TensorOptions()
                           .dtype(r.scalartype(5))
                           .device(r.device(7))
                           .layout(r.layoutOptional(6))
                           // NB: long standing issue, requires_grad is not
                           // respected here; you have to set it post facto, see
                           // https://github.com/pytorch/pytorch/issues/26428
                           // .requires_grad(r.toBool(7))
                           .pinned_memory(r.toBool(8));

  // don't bother releasing GIL here, as we are not allocating any nontrivial
  // data
  auto sym_sizes = r.symintlist(1);
  auto sym_strides_own = r.symintlistOptional(2);
  Tensor tensor = make_tensor_for_subclass_helper(
      /*sym_sizes=*/r.symintlist(1),
      /*sym_strides=*/r.symintlistOptional(2),
      /*sym_storage_offset=*/r.toSymIntOptional(3),
      options,
      /*storage_size=*/r.toSymIntOptional(14),
      r.toDispatchKeySetOptional(13));

  const auto sizes_strides_policy = r.stringViewOptional(10);
  if (sizes_strides_policy.has_value()) {
    tensor.unsafeGetTensorImpl()->set_python_custom_sizes_strides(
        parseSizesStridesPolicyArgument(*sizes_strides_policy));
  }

  tensor.set_requires_grad(r.toBool(9));

  if (r.toBool(11)) {
    tensor.unsafeGetTensorImpl()->set_python_custom_device(true);
  }
  if (r.toBool(12)) {
    tensor.unsafeGetTensorImpl()->set_python_custom_layout(true);
  }

  return THPVariable_NewWithVar(
      (PyTypeObject*)cls,
      tensor,
      // false is the default
      /*allow_preexisting_pyobj=*/false,
      // we checked __torch_dispatch__ above; avoid checking again.
      /*has_torch_dispatch_if_known=*/true);
  END_HANDLE_TH_ERRORS
}

#if IS_PYBIND_2_13_PLUS
#define DEFINE_CACHING_PYTHON_IMPORT_GETTER(name, import_expr)             \
  static py::handle name() {                                               \
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> \
        storage;                                                           \
    return storage                                                         \
        .call_once_and_store_result(                                       \
            []() -> py::object { return import_expr; })                    \
        .get_stored();                                                     \
  }
#else
#define DEFINE_CACHING_PYTHON_IMPORT_GETTER(name, import_expr)     \
  static py::handle name() {                                       \
    static py::handle storage = py::object(import_expr).release(); \
    return storage;                                                \
  }
#endif

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_dtensor_class_impl,
    py::module::import("torch.distributed.tensor").attr("DTensor"))

py::handle get_dtensor_class() {
  return get_dtensor_class_impl();
}

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_dtensor_spec_class,
    py::module::import("torch.distributed.tensor")
        .attr("_dtensor_spec")
        .attr("DTensorSpec"))

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_replicate_class,
    py::module::import("torch.distributed.tensor")
        .attr("placement_types")
        .attr("Replicate"))

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_tensor_meta_class,
    py::module::import("torch.distributed.tensor")
        .attr("_dtensor_spec")
        .attr("TensorMeta"))

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_dtensor_op_dispatcher,
    py::module::import("torch.distributed.tensor")
        .attr("DTensor")
        .attr("_op_dispatcher"))

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_dtensor_dispatch,
    py::module::import("torch.distributed.tensor")
        .attr("DTensor")
        .attr("_op_dispatcher")
        .attr("_dispatch_fast_path_python_tail"))

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_dtensor_dispatcher_wrap,
    py::module::import("torch.distributed.tensor")
        .attr("DTensor")
        .attr("_op_dispatcher")
        .attr("wrap"))

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_dtensor_get_local_results_slow_path,
    py::module::import("torch")
        .attr("distributed")
        .attr("tensor")
        .attr("DTensor")
        .attr("_op_dispatcher")
        .attr("_dispatch_get_local_results_slow_path"))

DEFINE_CACHING_PYTHON_IMPORT_GETTER(
    get_output_sharding_class,
    py::module::import("torch.distributed.tensor")
        .attr("_op_schema")
        .attr("OutputSharding"))

static bool arg_type_tensor_or_tensor_list_like(py::handle arg) {
  const auto dtensor_spec_class = get_dtensor_spec_class();
  if (py::isinstance(arg, dtensor_spec_class)) {
    return true;
  }
  if (!PyList_Check(arg.ptr())) {
    return false;
  }
  py::list arg_list = py::reinterpret_borrow<py::list>(arg);
  for (const auto e : arg_list) {
    if (!e.is_none() && !py::isinstance(e, dtensor_spec_class)) {
      return false;
    }
  }
  return true;
}

#if IS_PYTHON_3_11_PLUS
#define MAYBE_FOR_EACH_PYTHON_3_10_MINUS_DTENSOR_INTERNED_STRING(_)
#else
#define MAYBE_FOR_EACH_PYTHON_3_10_MINUS_DTENSOR_INTERNED_STRING(_) _(__name__)
#endif

#define FOR_EACH_DTENSOR_INTERNED_STRING(_)                   \
  MAYBE_FOR_EACH_PYTHON_3_10_MINUS_DTENSOR_INTERNED_STRING(_) \
  _(_comparison_key)                                          \
  _(_custom_op_handlers)                                      \
  _(_local_tensor)                                            \
  _(_spec)                                                    \
  _(_unwrap_to_op_info_impl)                                  \
  _(args_schema)                                              \
  _(compute_mesh)                                             \
  _(device_mesh)                                              \
  _(dtype)                                                    \
  _(get_coordinate)                                           \
  _(kwargs_schema)                                            \
  _(ndim)                                                     \
  _(needs_pytree)                                             \
  _(needs_redistribute)                                       \
  _(op)                                                       \
  _(op_to_schema_info)                                        \
  _(output_sharding)                                          \
  _(output_spec)                                              \
  _(schema_info)                                              \
  _(shape)                                                    \
  _(sharding_propagator)                                      \
  _(size)                                                     \
  _(static_argnum)                                            \
  _(static_kwargkey)                                          \
  _(stride)                                                   \
  _(tensor_meta)

struct DTensorInternedStrings {
#define DECLARE_INTERNED_STRING_VARIABLE(s) PyObject* s;
  FOR_EACH_DTENSOR_INTERNED_STRING(DECLARE_INTERNED_STRING_VARIABLE)
#undef DECLARE_INTERNED_STRING_VARIABLE
};

static DTensorInternedStrings dtensor_interned_strings;

#ifdef USE_DISTRIBUTED
static bool intern_dtensor_strings() {
#define INTERN_DTENSOR_STRING(s)                                           \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dtensor_interned_strings.s == nullptr); \
  dtensor_interned_strings.s = PyUnicode_InternFromString(#s);             \
  if (dtensor_interned_strings.s == nullptr) {                             \
    return false;                                                          \
  }

  FOR_EACH_DTENSOR_INTERNED_STRING(INTERN_DTENSOR_STRING);
#undef INTERN_DTENSOR_STRING
  return true;
}
#endif

static bool checked_not(PyObject* obj) {
  int result = PyObject_Not(obj);
  if (result == -1) {
    throw py::error_already_set();
  }
  return result;
}

static bool checked_istrue(PyObject* obj) {
  int result = PyObject_IsTrue(obj);
  if (result == -1) {
    throw py::error_already_set();
  }
  return result;
}

// pybind11 does not not use PyObject_Vectorcall currently; it seems
// to materialize a tuple of args instead.
template <std::size_t N>
static py::object checked_vectorcall(
    PyObject* obj,
    std::array<PyObject*, N> args) {
  PyObject* result = PyObject_Vectorcall(obj, args.data(), N, nullptr);
  if (!result) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(result);
}

template <typename... Args>
static py::object checked_vectorcall(PyObject* obj, Args... args) {
  static_assert(
      (std::is_same_v<Args, PyObject*> && ...),
      "must pass PyObject* to checked_vectorcall!");
  std::array<PyObject*, sizeof...(Args)> arr = {args...};
  return checked_vectorcall(obj, arr);
}

static c10::SymDimVector tuple_to_symintlist(PyObject* obj) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(PyTuple_Check(obj));
  c10::SymDimVector res;
  const auto size = PyTuple_GET_SIZE(obj);
  res.reserve(size);
  for (const auto idx : c10::irange(size)) {
    PyObject* item = PyTuple_GET_ITEM(obj, idx);
    if (THPUtils_checkLongExact(item)) {
      res.emplace_back(THPUtils_unpackLong(item));
    } else if (torch::is_symint(py::handle(item))) {
      res.push_back(py::handle(item).cast<c10::SymInt>());
    } else {
      // N.B. torch.Tensor.__index__ exists, so this should handle
      // scalar Tensors fine.
      res.emplace_back(THPUtils_unpackIndex(item));
    }
  }
  return res;
}

// As a Python object, DTensorSpec can be stored directly within
// IValue, but doing so is inefficient -- it requires a
// heap-allocated, reference counted intermediate
// ivalue::PyObjectHolder.
// Representation options:
// 1) Add an IValue tag to represent a placeholder object.
// 2) Play representational tricks -- stuff information into an IValue
// payload, such as by creating impossible
// intrusive_ptr_target*. Problem: this would cause IValue copying and
// possibly destruction to crash and so would be horribly unsafe.
// 3) Represent DTensorSpec directly inside IValue despite the inefficiency.
// 4) Leave the actual DTensor in the list of IValues, but detect it efficiently
// and transparently replace.
// 5) Just use a 24-byte struct of IValue + extra py::object.
//
// Given the high blast radius of (1), the unsafety of (2), the likely
// poor performance of (3), and detection of (4) looking less
// efficient than (5), (5) seems like the best path forward.

// We can't safely steal bits from IValue, so we just use 24 bytes of
// space. If dtensor_spec is non-null (truthy) then it's the active
// member, otherwise it's iv.
struct IValueOrDTensorSpec {
  IValueOrDTensorSpec() = default;
  explicit IValueOrDTensorSpec(c10::IValue v) : iv(std::move(v)) {}
  explicit IValueOrDTensorSpec(py::object dts) : dtensor_spec(std::move(dts)) {}
  c10::IValue iv;
  py::object dtensor_spec;

  bool operator==(const IValueOrDTensorSpec& rhs) const {
    return dtensor_spec
        ? (rhs.dtensor_spec && dtensor_spec.equal(rhs.dtensor_spec))
        : (iv == rhs.iv);
  }
};

// This corresponds to the Python OpSchema class in that it is the key
// for the (native version of the) sharding propagator cache. It is
// missing essentially everything else from the Python OpSchema
// though.
class NativeOpSchema {
 public:
  NativeOpSchema(
      const c10::OperatorHandle& op,
      c10::SmallVector<IValueOrDTensorSpec, 8> comparison_key,
      std::size_t comparison_key_hash,
      std::size_t args_schema_len)
      : op_(op),
        hash_(hash_combine(
            hash_combine(
                std::hash<c10::OperatorHandle>()(op),
                comparison_key_hash),
            args_schema_len)),
        args_schema_len_(args_schema_len),
        comparison_key_(std::move(comparison_key)) {}

  bool operator==(const NativeOpSchema& rhs) const {
    // If two NativeOpSchema are being compared, they are probably
    // equal, because comparison is occurring during a hash table
    // lookup and we know the hashes are already equal. Therefore, we
    // don't bother checking hash_ first.
    return op_ == rhs.op_ && args_schema_len_ == rhs.args_schema_len_ &&
        comparison_key_ == rhs.comparison_key_;
  }

  std::size_t hash() const {
    return hash_;
  }

 private:
  // It would *not* be correct to store this by reference, because we
  // have no guarantees about its lifetime. This class is cheap anyway.
  c10::OperatorHandle op_;
  std::size_t hash_;
  // Subtle point: consider clamp.Tensor(Tensor self, Tensor?
  // min=None, Tensor? max=None). The invocations clamp(t1, None, t2)
  // and clamp(t1, t2, None) have the same comparison key (t1, t2)
  // because we drop non-static non-tensor args from comparison. The
  // only way we happen to be able to tell them apart is that we omit
  // trailing defaulted arguments from the args tuple passed to
  // __torch_dispatch__ (and hence to DTensor dispatch as well), so
  // they have different args_schema_len_.
  //
  // I am preserving this existing behavior, but I suspect we should
  // make an algorithm change to be less brittle, such as including
  // None defaults for Tensor arguments in the comparison.
  std::size_t args_schema_len_;
  // There is no particular justification for the choice of 8
  // here. Feel free to change it.
  c10::SmallVector<IValueOrDTensorSpec, 8> comparison_key_;
};

namespace std {
template <>
struct hash<NativeOpSchema> {
  std::size_t operator()(const NativeOpSchema& schema) const {
    return schema.hash();
  }
};
} // namespace std

// Map from OpSchema to pyobject sharding propagation config.
class NativeShardingPropagatorCache {
 public:
  // Returns an invalid (falsey) py::object if the lookup fails.
  py::object find(const NativeOpSchema& op_schema) const {
    if (auto it = repr_.find(op_schema); it != repr_.end()) {
      hits_++;
      return py::object(it->second);
    }
    misses_++;
    return py::object();
  }

  void insert(NativeOpSchema&& op_schema, py::object output_sharding) {
    auto [it, inserted] =
        repr_.emplace(std::move(op_schema), std::move(output_sharding));
    TORCH_INTERNAL_ASSERT(
        inserted,
        "tried to insert already-present element in NativeShardingPropagatorCache!");
  }

  auto hits() const {
    return hits_;
  }

  auto misses() const {
    return misses_;
  }

 private:
  c10::FastMap<NativeOpSchema, py::object> repr_;
  // Cache is thread-local, so we don't take any further action for
  // thread-safety of these.
  mutable std::size_t hits_ = 0;
  mutable std::size_t misses_ = 0;
};

static std::optional<std::pair<NativeOpSchema, /*ComputeMesh*/ py::object>>
create_native_op_schema(
    const c10::OperatorHandle& op,
    py::handle py_op,
    torch::jit::Stack* stack);

static std::mutex native_sharding_propagator_cache_cleanup_mutex;
static c10::
    FastMap<std::thread::id, std::optional<NativeShardingPropagatorCache>*>
        all_thread_caches;
thread_local std::optional<NativeShardingPropagatorCache>
    native_sharding_propagator_cache_DO_NOT_USE;

NativeShardingPropagatorCache&
get_thread_local_native_sharding_propagator_cache() {
  if (!native_sharding_propagator_cache_DO_NOT_USE.has_value()) {
    native_sharding_propagator_cache_DO_NOT_USE.emplace();
    std::lock_guard<std::mutex> lock(
        native_sharding_propagator_cache_cleanup_mutex);
    const auto this_thread_id = std::this_thread::get_id();
    all_thread_caches[this_thread_id] =
        &native_sharding_propagator_cache_DO_NOT_USE;
    py::dict thread_dict =
        py::reinterpret_borrow<py::dict>(PyThreadState_GetDict());
    // We need to clean up before Python detaches from the thread if
    // the thread is being destroyed.
    thread_dict["__DTensor_fastpath_thread_cache_cleanup"] =
        py::capsule(new std::thread::id(this_thread_id), [](void* p) {
          auto* ptid = reinterpret_cast<std::thread::id*>(p);
          {
            std::lock_guard<std::mutex> inner_lock(
                native_sharding_propagator_cache_cleanup_mutex);
            auto it = all_thread_caches.find(*ptid);
            if (it != all_thread_caches.end()) {
              // We need to both:
              // 1) free python objects, and
              it->second->reset();
              // 2) make sure we don't try to come back and mess with
              // a destroyed thread-local at module unload (e.g.,
              // process exit) time.
              all_thread_caches.erase(it);
            }
          }
          delete ptid;
        });
  }
  return native_sharding_propagator_cache_DO_NOT_USE.value();
}

// We need to clean up all thread_locals if our module is getting
// unloaded.
void cleanup_thread_local_native_sharding_propagator_caches() {
  std::lock_guard<std::mutex> lock(
      native_sharding_propagator_cache_cleanup_mutex);
  for (auto& [_, popt_cache] : all_thread_caches) {
    popt_cache->reset();
  }
  all_thread_caches.clear();
}

static void replace_dtensors_with_local_tensor(torch::jit::Stack& stack);

static bool is_default_overload(const std::string& overload_name) {
  return overload_name.empty() || overload_name == "default";
}

static bool is_random_op(const c10::OperatorHandle& op) {
  // NOTE: must stay in sync with _random_ops in
  // torch/distributed/tensor/_dispatch.py
  constexpr auto aten_namespace_prefix_len = 6;
  const auto& op_name = op.operator_name();
  if (op_name.name.size() <= aten_namespace_prefix_len ||
      memcmp(op_name.name.data(), "aten::", aten_namespace_prefix_len) != 0) {
    return false;
  }
  static constexpr std::array<std::string_view, 6> random_names = {{
      "native_dropout",
      "normal_",
      "rand_like",
      "randn_like",
      "uniform_",
      "bernoulli",
  }};
  std::string_view name_without_namespace(
      op_name.name.c_str() + aten_namespace_prefix_len,
      op_name.name.size() - aten_namespace_prefix_len);
  if (name_without_namespace == "bernoulli_") {
    return op_name.overload_name == "float";
  }
  if (name_without_namespace == "randint_like") {
    return is_default_overload(op_name.overload_name) ||
        op_name.overload_name == "low_dtype" ||
        op_name.overload_name == "low_dtype_out";
  }
  const auto it = std::find(
      random_names.begin(), random_names.end(), name_without_namespace);
  if (it == random_names.end()) {
    return false;
  }
  return is_default_overload(op_name.overload_name);
}

// Puts local results on the stack. Return true for success, false for bailout
// to slow path.
static bool get_local_results(
    const c10::OperatorHandle& op,
    py::handle output_sharding,
    py::handle compute_mesh,
    bool participating,
    torch::jit::Stack* stack) {
  if (participating) {
    // computation that happens in the current rank of the mesh, normal case
    if (checked_istrue(
            output_sharding.attr(dtensor_interned_strings.needs_redistribute)
                .ptr()) ||
        is_random_op(op)) {
      // Bail out to slow path.
      return false;
    }
    // normal case, run local sharded op computation.

    // It is slightly inefficient that we take another pass over
    // arguments here when we just did one in create_native_op_schema to
    // create the comparison key. However, we have a crucial difference:
    // in the NativeOpSchema, we don't want to waste time dealing with
    // defaulted args. Here, we need to provide defaulted args because
    // we are going to make a local op call.
    replace_dtensors_with_local_tensor(*stack);
    op.callBoxed(*stack);
  } else {
    // For a non-participating device (happens on rank that does not
    // belong to the device mesh), we
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/autograd`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/autograd`):

- [`python_cpp_function.h_kw.md_docs.md`](./python_cpp_function.h_kw.md_docs.md)
- [`anomaly_mode.cpp_kw.md_docs.md`](./anomaly_mode.cpp_kw.md_docs.md)
- [`python_nested_functions_manual.cpp_kw.md_docs.md`](./python_nested_functions_manual.cpp_kw.md_docs.md)
- [`variable_info.h_docs.md_docs.md`](./variable_info.h_docs.md_docs.md)
- [`python_nn_functions.h_docs.md_docs.md`](./python_nn_functions.h_docs.md_docs.md)
- [`python_cpp_function.h_docs.md_docs.md`](./python_cpp_function.h_docs.md_docs.md)
- [`profiler_legacy.cpp_kw.md_docs.md`](./profiler_legacy.cpp_kw.md_docs.md)
- [`saved_variable.cpp_docs.md_docs.md`](./saved_variable.cpp_docs.md_docs.md)
- [`python_fft_functions.h_docs.md_docs.md`](./python_fft_functions.h_docs.md_docs.md)
- [`python_autograd.h_kw.md_docs.md`](./python_autograd.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `python_variable.cpp_docs.md_docs.md`
- **Keyword Index**: `python_variable.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
