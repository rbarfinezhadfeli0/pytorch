# Documentation: `docs/torch/csrc/utils/tensor_new.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/tensor_new.cpp_docs.md`
- **Size**: 53,380 bytes (52.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/tensor_new.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/tensor_new.cpp`
- **Size**: 65,460 bytes (63.93 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/tensor_new.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_scalars.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_numpy.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/dlpack.h>
#include <c10/core/Backend.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Layout.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <optional>

#include <stdexcept>
#include <vector>

using at::Device;
using at::IntArrayRef;
using at::kInt;
using at::kLong;
using at::ScalarType;
using at::Storage;
using at::Tensor;
using at::TensorOptions;
using std::optional;

namespace torch::utils {
namespace {
const int MAX_DIMS = 128;

thread_local bool kOnlyLiftCPUTensors = false;

TensorOptions build_options(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    const std::optional<Device>& device = std::nullopt) {
  options = options.dtype(scalar_type);
  if (device.has_value()) {
    return options.device(device);
  }
  return options;
}

// NB: It appears there is some consistency invariant between options and
// device, where if device is non-empty, its type must be consistent with the
// device type in options.
// TODO: Refactor this so we just pass everything in via options

Tensor new_with_sizes(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    const std::optional<Device>& device,
    c10::SymIntArrayRef sizes) {
  maybe_initialize_device(options.device());
  pybind11::gil_scoped_release no_gil;
  return at::empty_symint(sizes, build_options(options, scalar_type, device));
}

Tensor new_with_storage(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    Storage storage) {
  auto tensor = at::empty({}, build_options(options, scalar_type));
  tensor.set_(std::move(storage));
  return tensor;
}

std::vector<int64_t> compute_sizes(PyObject* seq, ScalarType scalar_type) {
  bool is_storage = isStorage(seq);
  std::vector<int64_t> sizes;
  // Note that after the first iteration, obj is the only thing that keeps
  // the seq raw pointer alive.
  THPObjectPtr obj;
  while (PySequence_Check(seq)) {
    auto length = PySequence_Length(seq);
    if (length < 0)
      throw python_error();
    if (is_storage) {
      length /= static_cast<int64_t>(elementSize(scalar_type));
    }
    sizes.push_back(length);
    TORCH_CHECK_VALUE(
        sizes.size() <= MAX_DIMS,
        "too many dimensions '",
        Py_TYPE(seq)->tp_name,
        "'");
    if (length == 0)
      break;
    PyObject* new_obj = PySequence_GetItem(seq, 0);
    // This line uses seq so we must NOT override obj before this line
    TORCH_CHECK_VALUE(
        new_obj,
        "could not determine the shape of object type '",
        Py_TYPE(seq)->tp_name,
        "'");
    obj = THPObjectPtr(new_obj);
    seq = obj.get();
  }

  return sizes;
}

ScalarType infer_scalar_type(PyObject* obj) {
  if (torch::is_symint(obj)) {
    return ScalarType::Long;
  }
  if (torch::is_symfloat(obj)) {
    return torch::tensors::get_default_scalar_type();
  }
#ifdef USE_NUMPY
  if (is_numpy_available()) {
    if (PyArray_Check(obj)) {
      return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)obj));
    }
    if (PyArray_CheckScalar(obj)) {
      THPObjectPtr arr(PyArray_FromScalar(obj, nullptr));
      return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)arr.get()));
    }
  }
#endif
  if (PyFloat_Check(obj)) {
    // this is always guaranteed to be a floating-point type, and makes it more
    // convenient to write e.g. torch.tensor(0.) than torch.tensor(0.,
    // dtype=torch.Tensor.dtype).
    return torch::tensors::get_default_scalar_type();
  }
  if (THPUtils_checkLong(obj)) {
    return ScalarType::Long;
  }
  if (PyBool_Check(obj)) {
    return ScalarType::Bool;
  }
  if (PyComplex_Check(obj)) {
    switch (torch::tensors::get_default_scalar_type()) {
      case ScalarType::Float:
        return ScalarType::ComplexFloat;
      case ScalarType::Double:
        return ScalarType::ComplexDouble;
      case ScalarType::Half:
        return ScalarType::ComplexHalf;
      default:
        TORCH_CHECK(false, "invalid default scalar type for complex");
    }
  }
  if (THPVariable_Check(obj)) {
    const auto& var = THPVariable_Unpack(obj);
    return var.scalar_type();
  }
  TORCH_CHECK_TYPE(
      !THPUtils_checkString(obj),
      "new(): invalid data type '",
      Py_TYPE(obj)->tp_name,
      "'");
  if (PySequence_Check(obj)) {
    auto length = PySequence_Length(obj);
    if (length < 0)
      throw python_error();
    // match NumPy semantics, except use default tensor type instead of double.
    if (length == 0)
      return torch::tensors::get_default_scalar_type();
    ScalarType scalarType{};
    for (const auto i : c10::irange(length)) {
      THPObjectPtr handle(PySequence_GetItem(obj, i));
      if (!handle)
        throw python_error();
      auto cur_item = handle.get();
      TORCH_CHECK_TYPE(
          cur_item != obj, "new(): self-referential lists are incompatible");
      ScalarType item_scalarType = infer_scalar_type(cur_item);
      scalarType = (i > 0) ? at::promoteTypes(scalarType, item_scalarType)
                           : item_scalarType;
      if (scalarType == ScalarType::ComplexDouble) {
        // this won't change (unless we hit undefined, but that will fail
        // later).
        return scalarType;
      }
    }
    return scalarType;
  }
  TORCH_CHECK(false, "Could not infer dtype of ", Py_TYPE(obj)->tp_name);
}

void recursive_store(
    char* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t dim,
    ScalarType scalarType,
    size_t elementSize,
    PyObject* obj) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(data != nullptr);

  int64_t ndim = static_cast<int64_t>(sizes.size());
  bool is_symfloat = torch::is_symfloat(obj);
  bool is_symint = torch::is_symint(obj);
  if (dim == ndim) {
    if (is_symfloat) {
      auto new_obj = py::reinterpret_borrow<py::object>(obj);
      auto val = new_obj.cast<c10::SymFloat>();
      const double double_val = val.guard_float(__FILE__, __LINE__);
      obj = Py_BuildValue("d", double_val);
    }
    if (is_symint) {
      auto new_obj = py::reinterpret_borrow<py::object>(obj);
      auto val = new_obj.cast<c10::SymInt>();
      const int64_t int_val = val.guard_int(__FILE__, __LINE__);
      obj = Py_BuildValue("i", int_val);
    }
    torch::utils::store_scalar(data, scalarType, obj);
    return;
  }

  auto n = sizes[dim];
  auto seq = THPObjectPtr(PySequence_Fast(obj, "not a sequence"));
  if (!seq)
    throw python_error();
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto seq_size = PySequence_Fast_GET_SIZE(seq.get());
  TORCH_CHECK_VALUE(
      seq_size == n,
      "expected sequence of length ",
      n,
      " at dim ",
      dim,
      " (got ",
      seq_size,
      ")");

  PyObject** items = PySequence_Fast_ITEMS(seq.get());
  for (const auto i : c10::irange(n)) {
#ifdef USE_NUMPY
    if (is_numpy_available() && PyArray_Check(items[i])) {
      TORCH_WARN_ONCE(
          "Creating a tensor from a list of numpy.ndarrays is extremely slow. "
          "Please consider converting the list to a single numpy.ndarray with "
          "numpy.array() before converting to a tensor.");
    }
#endif
    recursive_store(
        data, sizes, strides, dim + 1, scalarType, elementSize, items[i]);
    data += strides[dim] * elementSize;
  }
}

Tensor internal_new_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<Device> device_opt,
    PyObject* data,
    bool copy_variables,
    bool copy_numpy,
    bool type_inference,
    bool pin_memory = false) {
  TORCH_CHECK_TYPE(
      !THPUtils_checkString(data),
      "new(): invalid data type '",
      Py_TYPE(data)->tp_name,
      "'");

  if (THPVariable_Check(data)) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from a variable");
    // TODO: use MaybeOwned
    auto var = THPVariable_Unpack(data);
    if (copy_variables) {
      var = var.detach();
    }
    // infer the scalar type and device type; it's not expected to infer the
    // layout since these constructors are defined per-layout-type (e.g. tensor
    // vs sparse_coo_tensor).
    const auto& inferred_scalar_type =
        type_inference ? var.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : var.device();
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_device(device);
    return var.to(
        device,
        inferred_scalar_type,
        /*non_blocking=*/false,
        /*copy=*/copy_variables);
  }

#ifdef USE_NUMPY
  if (PyObject_HasAttrString(data, "__cuda_array_interface__")) {
    TORCH_CHECK(
        !pin_memory,
        "Can't pin tensor constructed from __cuda_array_interface__");
    auto tensor = tensor_from_cuda_array_interface(data, device_opt);
    const auto& inferred_scalar_type =
        type_inference ? tensor.scalar_type() : scalar_type;

    // Device preference is:
    //  - explicitly user specified device in `device_opt`
    //      - either by setting device='...'
    //      - or setting torch.set_default_device(...)
    //  - device of already constructed tensor
    // This prevents an unnecessary device -> host copy when the tensor is
    // already on the device, while respecting a default device and allows the
    // user to overwrite the behavior explicitly.
    at::Device device = device_opt.has_value() ? *device_opt : tensor.device();

    pybind11::gil_scoped_release no_gil;
    maybe_initialize_device(device);
    return tensor.to(
        device,
        inferred_scalar_type,
        /*non_blocking=*/false,
        /*copy=*/copy_numpy);
  }

  if (is_numpy_available() && PyArray_Check(data)) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from numpy");
    auto tensor =
        tensor_from_numpy(data, /*warn_if_not_writeable=*/!copy_numpy);
    const auto& inferred_scalar_type =
        type_inference ? tensor.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : options.device();
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_device(device);
    return tensor.to(
        device,
        inferred_scalar_type,
        /*non_blocking=*/false,
        /*copy=*/copy_numpy);
  }
#endif

  if (PyObject_HasAttrString(data, "__dlpack__")) {
    py::object tensor_o =
        py::module::import("torch").attr("utils").attr("dlpack").attr(
            "from_dlpack")(py::handle(data));
    Tensor tensor = py::cast<Tensor>(tensor_o);
    const auto& inferred_scalar_type =
        type_inference ? tensor.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : tensor.device();
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_device(device);
    return tensor.to(
        device,
        inferred_scalar_type,
        /*non_blocking=*/false,
        /*copy=*/copy_variables);
  }

  auto device = device_opt.has_value() ? *device_opt : options.device();

  auto sizes = compute_sizes(data, scalar_type);

  ScalarType inferred_scalar_type =
      type_inference ? infer_scalar_type(data) : scalar_type;
  // This exists to prevent us from tracing the call to empty().  The actual
  // autograd code doesn't really matter, because requires_grad is always false
  // here.
  // What are the semantics of tensor_new()?
  // We manually construct a tensor and place on it on the correct device with
  // empty() and to(). We then have to "lift" the newly constructed tensor in
  // some cases, like when we're performing a functorch transform or running
  // functionalization. The exclude guards are all to ensure that extra logic
  // doesn't run when we're constructing the raw tensor.
  Tensor tensor;
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    c10::impl::ExcludeDispatchKeyGuard torchdispatchmode_guard(
        c10::DispatchKey::Python);
    c10::impl::ExcludeDispatchKeyGuard torchdispatchmode_snapshot_guard(
        c10::DispatchKey::PythonTLSSnapshot);
    // functorch uses FuncTorchDynamicLayerBackMode as a mode key to wrap all
    // tensors returned from operators in special TensorWrapper tensor extension
    c10::impl::ExcludeDispatchKeyGuard functorch_front_guard(
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode);
    c10::impl::ExcludeDispatchKeyGuard functorch_back_guard(
        c10::DispatchKey::FuncTorchDynamicLayerBackMode);
    // We disable Fake and DeferredInit handlers for similar reasons as
    // functorch.
    c10::impl::ExcludeDispatchKeyGuard fake_and_deferred_init_guard(
        c10::DispatchKeySet{
            c10::DispatchKey::Fake, c10::DispatchKey::DeferredInit});
    // Note [Functionalization <> torch.Tensor constructor]
    // Functionalization "lifts" the newly constructed tensor into a wrapper
    // using aten::lift().
    c10::impl::ExcludeDispatchKeyGuard functionalize_guard(
        c10::DispatchKey::Functionalize);
    {
      // Tracing should probably also use the "lift" operator to add the tensor
      // to a trace, but it's technically BC-breaking to do that, since we
      // currently trace .to() calls.
      at::tracer::impl::NoTracerDispatchMode tracer_guard;

      if (isStorage(data)) {
        auto [storage, storage_scalar_type, is_typed_storage] =
            createStorageGetType(data);

        TORCH_CHECK(
            !is_typed_storage || storage_scalar_type == scalar_type,
            "Expected a Storage of type ",
            scalar_type,
            " or an UntypedStorage, but got ",
            storage_scalar_type);
        tensor = at::empty(
            {0}, // sizes. Storage will be set later.
            at::initialTensorOptions()
                .dtype(
                    is_typed_storage ? storage_scalar_type
                                     : inferred_scalar_type)
                .pinned_memory(pin_memory)
                .device(storage.device()));
        tensor.set_(storage);

      } else {
        TensorOptions opts =
            at::initialTensorOptions().dtype(inferred_scalar_type);

        // If the device is Meta, take the shortcut. We don't want to allocate
        // an empty CPU tensor which would break our contract for meta tensors.
        if (device == at::kMeta) {
          return at::empty(sizes, opts.device(device));
        }
        tensor = at::empty(sizes, opts.pinned_memory(pin_memory));
        if (c10::multiply_integers(tensor.sizes()) != 0) {
          recursive_store(
              (char*)tensor.data_ptr(),
              tensor.sizes(),
              tensor.strides(),
              0,
              inferred_scalar_type,
              tensor.dtype().itemsize(),
              data);
        }
      }
    }
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_device(device);
    // However, it is VERY important that we trace the to() call here (even
    // though the reason this is important is a hack).  Without *some* factory
    // function call that is traced at construction time, we will consider
    // a tensor constant as originating from "outside" the trace, and if you
    // try to return it directly we will fail with the error saying no
    // "no observable data dependence".  In an ideal world, we wouldn't trace
    // a to() call but I need to think harder about what exactly we should trace
    // in this case.
    if (only_lift_cpu_tensors()) {
      tensor = tensor.to(
          inferred_scalar_type, /*non_blocking=*/false, /*copy=*/false);

    } else {
      tensor = tensor.to(
          device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/false);
    }
  }

  // torch.jit.trace will continue to trace out `.to()` instead of `.lift()`,
  // since changing it is BC-breaking.
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  {
    // lift has no autograd implementation, so we need to make sure we don't try
    // to dispatch to it.
    // TODO: arguably it should have an autograd implementation that noops
    at::AutoDispatchBelowADInplaceOrView guard;
    tensor = at::lift_fresh(tensor);
  }
  if (only_lift_cpu_tensors() && device.type() != DeviceType::CPU) {
    if (!device.has_index() &&
        !torch::utils::is_device_initialized(device.type())) {
      // Infer device 0 to avoid device init
      device = c10::Device(device.type(), 0);
    }
    tensor = tensor.to(device, /*non_blocking=*/false, /*copy=*/false);
  }
  return tensor;
}

Tensor new_from_data_copy(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<Device> device,
    PyObject* data) {
  return internal_new_from_data(
      options,
      scalar_type,
      device,
      data,
      /*copy_variables=*/true,
      /*copy_numpy=*/true,
      /*type_inference=*/false);
}

Tensor legacy_new_from_sequence(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<Device> device,
    PyObject* data) {
  TORCH_CHECK_TYPE(
      PySequence_Check(data),
      "new(): data must be a sequence (got ",
      Py_TYPE(data)->tp_name,
      ")");
  return internal_new_from_data(
      options,
      scalar_type,
      device,
      data,
      /*copy_variables=*/false,
      /*copy_numpy=*/false,
      /*type_inference=*/false);
}

// "base" here refers to the Tensor type on which the function was invoked,
// e.g.: in x.new(y), 'x' is the base.
// TODO: Rewrite this using dispatchKeyToTensorOptions
void check_base_legacy_new(
    c10::DispatchKey dispatch_key,
    at::Layout expected_layout) {
  if (expected_layout == c10::kStrided) {
    constexpr c10::DispatchKeySet expected_key_set({
        c10::DispatchKey::CPU,
        c10::DispatchKey::CUDA,
        c10::DispatchKey::HIP,
        c10::DispatchKey::XLA,
        c10::DispatchKey::Lazy,
        c10::DispatchKey::IPU,
        c10::DispatchKey::XPU,
        c10::DispatchKey::HPU,
        c10::DispatchKey::MPS,
        c10::DispatchKey::Meta,
        c10::DispatchKey::PrivateUse1,
    });
    TORCH_CHECK(
        expected_key_set.has(dispatch_key),
        "new(): expected key in ",
        expected_key_set,
        " but got: ",
        dispatch_key);
  } else if (expected_layout == c10::kSparse) {
    // NOTE: no sparse XLA or Lazy
    constexpr c10::DispatchKeySet expected_key_set({
        c10::DispatchKey::SparseCPU,
        c10::DispatchKey::SparseCUDA,
        c10::DispatchKey::SparseHIP,
        c10::DispatchKey::SparseXPU,
        c10::DispatchKey::SparseMPS,
        c10::DispatchKey::SparsePrivateUse1,
    });
    TORCH_CHECK(
        expected_key_set.has(dispatch_key),
        "new(): expected key in ",
        expected_key_set,
        " but got: ",
        dispatch_key);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unexpected layout");
  }
}

// TODO: Make this accept options instead of dispatch key
void check_legacy_ctor_device(
    c10::DispatchKey dispatch_key,
    std::optional<Device> device) {
  if (device.has_value()) {
    TORCH_CHECK(
        dispatchKeyToDeviceType(dispatch_key) == device.value().type(),
        "legacy constructor expects device type: ",
        dispatchKeyToDeviceType(dispatch_key),
        " but device type: ",
        device.value().type(),
        " was passed");
  }
}

std::optional<Device> device_or_from_dispatch_key(
    std::optional<Device> device,
    c10::DispatchKey dispatch_key) {
  if (device.has_value()) {
    return device;
  } else {
    return Device(dispatchKeyToDeviceType(dispatch_key));
  }
}

enum class CtorOrNew {
  BASE_CTOR,
  CTOR,
  NEW,
};

Tensor legacy_sparse_tensor_generic_ctor_new(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs,
    CtorOrNew ctor_or_new) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  static PythonArgParser parser({
      "new(*, Device? device=None)",
      "new(*, int64_t cdata)|hidden",
      "new(Tensor indices, Tensor values, *, Device? device=None)",
      "new(Tensor indices, Tensor values, IntArrayRef size, *, Device? device=None)",
      "new(SymIntArrayRef size, *, Device? device=None)",
  });
  if (ctor_or_new == CtorOrNew::NEW)
    check_base_legacy_new(dispatch_key, c10::kSparse);
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (ctor_or_new == CtorOrNew::CTOR) {
      TORCH_WARN_ONCE(
          "torch.sparse.SparseTensor() is deprecated."
          "  Please use torch.sparse_coo_tensor((0,), dtype=).");
    }
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    return at::empty({0}, build_options(options, scalar_type, deviceOptional));
  } else if (r.idx == 1) {
    if (ctor_or_new == CtorOrNew::CTOR) {
      TORCH_WARN_ONCE(
          "torch.sparse.SparseTensor(cdata=x._cdata) is deprecated."
          "  Please use torch.sparse_coo_tensor(x._indices(), x._values(), x.shape).");
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return at::unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 2) {
    if (ctor_or_new == CtorOrNew::CTOR) {
      TORCH_WARN_ONCE(
          "torch.sparse.SparseTensor(indices, values, *, device=) is deprecated."
          "  Please use torch.sparse_coo_tensor(indices, values, dtype=, device=).");
    }
    // Note: this signature doesn't have a dtype, even though it has a device;
    // it probably shouldn't have a device (we should infer it).
    auto deviceOptional = r.deviceOptional(2);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    if (ctor_or_new == CtorOrNew::CTOR) {
      TORCH_WARN_ONCE(
          "torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated."
          "  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).");
    }
    // Note: this signature doesn't have a dtype, even though it has a device;
    // it probably shouldn't have a device (we should infer it).
    auto deviceOptional = r.deviceOptional(3);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 &&
        arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      if (ctor_or_new == CtorOrNew::CTOR) {
        TORCH_CHECK_TYPE(
            false,
            "torch.sparse.SparseTensor(sequence) only accepts sizes.  Please use torch.sparse_coo_tensor() "
            "or construct a strided tensor and convert it to sparse via to_sparse.");
      } else {
        TORCH_CHECK_TYPE(
            false,
            "SparseTensor.new(sequence) only accepts sizes.  Please use torch.sparse_coo_tensor() "
            "or construct a strided tensor and convert it to sparse via to_sparse.");
      }
    }
    if (ctor_or_new == CtorOrNew::CTOR) {
      TORCH_WARN_ONCE(
          "torch.sparse.SparseTensor(shape, *, device=) is deprecated."
          "  Please use torch.sparse_coo_tensor(shape, dtype=, device=).");
    }
    return new_with_sizes(
        options, scalar_type, deviceOptional, r.symintlist(0));
  }
  TORCH_CHECK(false, "new(): invalid arguments");
}

// NB: device_idx here is NOT a DeviceIndex, but index into PythonArgs
c10::TensorOptions typeIdWithDefault(
    PythonArgs& r,
    int64_t device_idx,
    c10::DispatchKey dispatch_key) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  if (!r.isNone(static_cast<int>(device_idx))) {
    // TODO: This line doesn't seem to be exercised at all in tests
    options = options.device(r.device(static_cast<int>(device_idx)).type());
  }
  return options;
}

} // namespace

static Tensor legacy_tensor_generic_ctor_new(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs,
    CtorOrNew ctor_or_new) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  static PythonArgParser parser({
      "new(*, Device? device=None)",
      "new(Storage storage)",
      "new(*, int64_t cdata)|hidden",
      // This constructor is no longer legacy, it will also be usable for
      // subclass initialization
      "new(Tensor other)",
      "new(Tensor other, *, Device? device=None)|hidden", // prevent Tensor
                                                          // matching with
                                                          // IntArrayRef,
                                                          // PyObject*
      "new(SymIntArrayRef size, *, Device? device=None)",
      "new(PyObject* data, *, Device? device=None)",
  });

  if (isSparse(dispatchKeyToBackend(dispatch_key))) {
    return legacy_sparse_tensor_generic_ctor_new(
        dispatch_key, scalar_type, args, kwargs, ctor_or_new);
  }

  if (ctor_or_new == CtorOrNew::NEW)
    check_base_legacy_new(dispatch_key, c10::kStrided);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, build_options(options, scalar_type));
  } else if (r.idx == 1) {
    at::ScalarType storage_scalar_type{at::ScalarType::Undefined};
    bool is_typed_storage = false;
    at::Storage storage = r.storage(0, storage_scalar_type, is_typed_storage);
    if (storage_scalar_type != at::ScalarType::Undefined && is_typed_storage) {
      TORCH_CHECK(
          storage_scalar_type == scalar_type,
          "Expected a Storage of type ",
          scalar_type,
          " or an UntypedStorage, but got type ",
          storage_scalar_type,
          " for argument 1 'storage'");
    }
    return new_with_storage(options, scalar_type, storage);
  } else if (r.idx == 2) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return at::unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    const auto& other = r.tensor(0);
    // BASE_CTOR (aka torch.Tensor) is now relaxed to accept any
    // dtype; previously it was "float" biased
    if (ctor_or_new != CtorOrNew::BASE_CTOR) {
      options = options.dtype(scalar_type);
      TORCH_CHECK_TYPE(
          other.options().type_equal(options),
          "expected ",
          options,
          " (got ",
          other.options(),
          ")");
    }
    return other.alias();
  } else if (r.idx == 4) {
    if (ctor_or_new == CtorOrNew::CTOR || ctor_or_new == CtorOrNew::BASE_CTOR) {
      TORCH_CHECK(
          false,
          "Legacy tensor constructor of the form torch.Tensor(tensor, device=device) "
          "is not supported.  Use torch.tensor(...) or torch.as_tensor(...) instead.");
    } else {
      TORCH_CHECK(
          false,
          "Legacy tensor new of the form tensor.new(tensor, device=device) "
          "is not supported.  Use torch.as_tensor(...) instead.");
    }
  } else if (r.idx == 5) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 &&
        arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(
          options, scalar_type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(
        options, scalar_type, deviceOptional, r.symintlist(0));
  } else if (r.idx == 6) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    return legacy_new_from_sequence(
        options, scalar_type, deviceOptional, r.pyobject(0));
  }
  TORCH_CHECK(false, "new(): invalid arguments");
}

// Handles ONLY torch.Tensor
// Unlike the legacy dtype/device specialized constructors, this one is
// relaxed to accept any device/dtype input tensor (even if it doesn't
// match the default)
Tensor base_tensor_ctor(PyObject* args, PyObject* kwargs) {
  return legacy_tensor_generic_ctor_new(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      args,
      kwargs,
      CtorOrNew::BASE_CTOR);
}

// Handles calls like torch.DoubleTensor, torch.cuda.FloatTensor,
// torch.sparse.FloatTensor, etc.
Tensor legacy_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  return legacy_tensor_generic_ctor_new(
      dispatch_key, scalar_type, args, kwargs, CtorOrNew::CTOR);
}

// Handles tensor.new(...)
Tensor legacy_tensor_new(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  return legacy_tensor_generic_ctor_new(
      dispatch_key, scalar_type, args, kwargs, CtorOrNew::NEW);
}

Tensor indexing_tensor_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<Device> device,
    PyObject* data) {
  // Specific to tensor indexing, converts an indexing list to an
  // indexing tensor (type Byte or Long)
  ScalarType inferred_scalar_type = infer_scalar_type(data);
  if (inferred_scalar_type == ScalarType::Byte ||
      inferred_scalar_type == ScalarType::Bool) {
    return internal_new_from_data(
        options,
        inferred_scalar_type,
        device,
        data,
        /*copy_variables=*/false,
        /*copy_numpy=*/false,
        /*type_inference=*/false);
  } else {
    return internal_new_from_data(
        options,
        scalar_type,
        device,
        data,
        /*copy_variables=*/false,
        /*copy_numpy=*/false,
        /*type_inference=*/false);
  }
}

class CheckSparseTensorInvariantsContext {
 public:
  CheckSparseTensorInvariantsContext()
      : state{at::globalContext().checkSparseTensorInvariants()} {}
  ~CheckSparseTensorInvariantsContext() {
    at::globalContext().setCheckSparseTensorInvariants(state);
  }
  CheckSparseTensorInvariantsContext(
      const CheckSparseTensorInvariantsContext&) = delete;
  CheckSparseTensorInvariantsContext(CheckSparseTensorInvariantsContext&&) =
      delete;
  CheckSparseTensorInvariantsContext& operator=(
      const CheckSparseTensorInvariantsContext&) = delete;
  CheckSparseTensorInvariantsContext& operator=(
      CheckSparseTensorInvariantsContext&&) = delete;

 private:
  bool state;
};

static Tensor sparse_compressed_tensor_ctor_worker(
    const std::string& name,
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r,
    std::optional<c10::Layout> required_layout) {
  TORCH_INTERNAL_ASSERT(!isSparseCsr(dispatchKeyToBackend(dispatch_key)));
  TORCH_INTERNAL_ASSERT(!isSparse(dispatchKeyToBackend(dispatch_key)));
  enum {
    ARG_COMPRESSED_INDICES = 0,
    ARG_PLAIN_INDICES,
    ARG_VALUES,
    ARG_SIZE,
    ARG_TYPE,
    ARG_LAYOUT,
    ARG_DEVICE,
    ARG_PIN_MEMORY,
    ARG_REQUIRES_GRAD,
    ARG_CHECK_INVARIANTS,
    ARGS_COUNT
  };
  enum {
    ARG_VALUES1 = ARG_VALUES,
    ARG_TYPE1,
    ARG_LAYOUT1,
    ARG_DEVICE1,
    ARG_PIN_MEMORY1,
    ARG_REQUIRES_GRAD1,
    ARG_CHECK_INVARIANTS1,
    ARGS_COUNT1
  };

  auto safe_get_attr_string = [](PyObject* o,
                                 const char* attr_name) -> PyObject* {
    // Clear error indicator if attribute does not exists.
    // Otherwise subsequent Python C API calls might return bogus values.
    // See https://github.com/pytorch/pytorch/issues/58520 for more details
    auto rc = PyObject_GetAttrString(o, attr_name);
    if (!rc) {
      if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
        throw python_error();
      }
      // Warning: a wrong attribute error may be suppressed here
      PyErr_Clear();
    }
    return rc;
  };
  THPObjectPtr compressed_indices_dtype_attr(
      safe_get_attr_string(r.pyobject(ARG_COMPRESSED_INDICES), "dtype"));
  THPObjectPtr plain_indices_dtype_attr(
      safe_get_attr_string(r.pyobject(ARG_PLAIN_INDICES), "dtype"));
  at::ScalarType compressed_indices_scalar_type = compressed_indices_dtype_attr
      ? reinterpret_cast<THPDtype*>(compressed_indices_dtype_attr.get())
            ->scalar_type
      : kInt;
  at::ScalarType plain_indices_scalar_type = plain_indices_dtype_attr
      ? reinterpret_cast<THPDtype*>(plain_indices_dtype_attr.get())->scalar_type
      : kInt;
  CheckSparseTensorInvariantsContext
      restores_check_sparse_tensor_invariants_global_state{};
  bool default_check_invariants =
      at::globalContext().checkSparseTensorInvariants();

  if (r.idx == 0) {
    const bool pin_memory = r.toBool(ARG_PIN_MEMORY);
    bool type_inference = r.isNone(ARG_TYPE);
    const auto inferred_options =
        typeIdWithDefault(r, ARG_DEVICE, dispatch_key);
    const auto inferred_scalar_type =
        r.scalartypeWithDefault(ARG_TYPE, scalar_type);
    auto deviceOptional = r.deviceOptional(ARG_DEVICE);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    // the global state of invariants check flag will be restored via
    // CheckSparseTensorInvariantsContext destructor
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS, default_check_invariants));
    Tensor values = internal_new_from_data(
        inferred_options,
        inferred_scalar_type,
        deviceOptional,
        r.pyobject(ARG_VALUES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference);
    Tensor compressed_indices = internal_new_from_data(
        values.options(),
        compressed_indices_scalar_type,
        deviceOptional,
        r.pyobject(ARG_COMPRESSED_INDICES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/true);
    Tensor plain_indices = internal_new_from_data(
        values.options(),
        plain_indices_scalar_type,
        deviceOptional,
        r.pyobject(ARG_PLAIN_INDICES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/true);
    std::optional<c10::Layout> layout =
        (required_layout
             ? r.layoutWithDefault(ARG_LAYOUT, required_layout.value())
             : r.layoutOptional(ARG_LAYOUT));
    if (required_layout.has_value()) {
      TORCH_CHECK(
          layout.has_value() && layout == required_layout,
          name,
          ": layout must be ",
          required_layout.value(),
          " but got ",
          layout);
    }
    return at::sparse_compressed_tensor(
               compressed_indices,
               plain_indices,
               values,
               r.intlist(ARG_SIZE),
               values.options().layout(layout).pinned_memory(pin_memory))
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD));
  } else if (r.idx == 1) {
    bool type_inference = r.isNone(ARG_TYPE1);
    const auto inferred_options =
        typeIdWithDefault(r, ARG_DEVICE1, dispatch_key);
    const auto inferred_scalar_type =
        r.scalartypeWithDefault(ARG_TYPE1, scalar_type);
    auto deviceOptional = r.deviceOptional(ARG_DEVICE1);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    const bool pin_memory = r.toBool(ARG_PIN_MEMORY1);
    // the global state of invariants check flag will be restored via
    // CheckSparseTensorInvariantsContext destructor
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS1, default_check_invariants));
    Tensor values = internal_new_from_data(
        inferred_options,
        inferred_scalar_type,
        deviceOptional,
        r.pyobject(ARG_VALUES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference);
    Tensor compressed_indices = internal_new_from_data(
        values.options(),
        compressed_indices_scalar_type,
        deviceOptional,
        r.pyobject(ARG_COMPRESSED_INDICES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/true);
    Tensor plain_indices = internal_new_from_data(
        values.options(),
        plain_indices_scalar_type,
        deviceOptional,
        r.pyobject(ARG_PLAIN_INDICES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/true);
    std::optional<c10::Layout> layout =
        (required_layout
             ? r.layoutWithDefault(ARG_LAYOUT1, required_layout.value())
             : r.layoutOptional(ARG_LAYOUT1));
    if (required_layout.has_value()) {
      TORCH_CHECK(
          layout == required_layout,
          name,
          ": layout must be ",
          required_layout.value(),
          " but got ",
          layout);
    }
    return at::sparse_compressed_tensor(
               compressed_indices,
               plain_indices,
               values,
               values.options().layout(layout).pinned_memory(pin_memory))
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD1));
  }
  TORCH_CHECK(false, name + ": invalid arguments");
}

Tensor sparse_compressed_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  std::optional<c10::Layout> required_layout{};
  return sparse_compressed_tensor_ctor_worker(
      "sparse_compressed_tensor",
      dispatch_key,
      scalar_type,
      r,
      required_layout);
}

Tensor sparse_csr_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  std::optional<c10::Layout> required_layout(c10::Layout::SparseCsr);
  return sparse_compressed_tensor_ctor_worker(
      "sparse_csr_tensor", dispatch_key, scalar_type, r, required_layout);
}

Tensor sparse_csc_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  std::optional<c10::Layout> required_layout(c10::Layout::SparseCsc);
  return sparse_compressed_tensor_ctor_worker(
      "sparse_csc_tensor", dispatch_key, scalar_type, r, required_layout);
}

Tensor sparse_bsr_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  std::optional<c10::Layout> required_layout(c10::Layout::SparseBsr);
  return sparse_compressed_tensor_ctor_worker(
      "sparse_bsr_tensor", dispatch_key, scalar_type, r, required_layout);
}

Tensor sparse_bsc_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  std::optional<c10::Layout> required_layout(c10::Layout::SparseBsc);
  return sparse_compressed_tensor_ctor_worker(
      "sparse_bsc_tensor", dispatch_key, scalar_type, r, required_layout);
}

// Note [Ensuring sparse values and indices match devices]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In all places where we construct indices, we read out options from values
// (rather than use inferred_options).  Why?  This handles the case when
// values is a CUDA tensor, but indices is a non-Tensor value (and the device
// argument is not set).  Example:
//
//  torch.sparse_coo_tensor(([0, 1],), self.empty(2, 0).cuda(), (4, 0))
//
// Sparse tensors require both indices and values to live on the same device.
// If values lives on CUDA, we can infer where the indices should live, and
// should accept even ordinary index sequences (and just make sure we write them
// into the correct device).  values is the ONLY way we know that the index
// tensor should go to CUDA, so we have to get the information in somehow.
//
// This code is kind of jank.  For one, the dtype in options is silently ignored
// by internal_new_from_data.  Also, in classic janky code style, it used to
// not work quite right: if values lives on "cuda:1", before all we said was
// "this needs to be CUDA" and indices would be allocated on the wrong tensor.
// Options is more right and gets this correct.

Tensor sparse_coo_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  TORCH_INTERNAL_ASSERT(!isSparse(dispatchKeyToBackend(dispatch_key)));
  TORCH_INTERNAL_ASSERT(!isSparseCsr(dispatchKeyToBackend(dispatch_key)));
  enum {
    ARG_INDICES = 0,
    ARG_VALUES,
    ARG_TYPE,
    ARG_DEVICE,
    ARG_PIN_MEMORY,
    ARG_REQUIRES_GRAD,
    ARG_CHECK_INVARIANTS,
    ARG_IS_COALESCED,
    ARGS_COUNT
  };
  enum {
    ARG_INDICES1 = 0,
    ARG_VALUES1,
    ARG_SIZE1,
    ARG_TYPE1,
    ARG_DEVICE1,
    ARG_PIN_MEMORY1,
    ARG_REQUIRES_GRAD1,
    ARG_CHECK_INVARIANTS1,
    ARG_IS_COALESCED1,
    ARGS_COUNT1
  };
  enum {
    ARG_SIZE2 = 0,
    ARG_TYPE2,
    ARG_DEVICE2,
    ARG_REQUIRES_GRAD2,
    ARG_CHECK_INVARIANTS2,
    ARGS_COUNT2
  };

  CheckSparseTensorInvariantsContext
      restores_check_sparse_tensor_invariants_global_state{};
  bool default_check_invariants =
      at::globalContext().checkSparseTensorInvariants();

  if (r.idx == 0) {
    bool pin_memory = r.toBool(ARG_PIN_MEMORY);
    bool type_inference = r.isNone(ARG_TYPE);
    const auto inferred_options =
        typeIdWithDefault(r, ARG_DEVICE, dispatch_key);
    const auto inferred_scalar_type =
        r.scalartypeWithDefault(ARG_TYPE, scalar_type);
    auto deviceOptional = r.deviceOptional(ARG_DEVICE);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS, default_check_invariants));

    // if no dtype provided, infer type based on value type.
    Tensor values = internal_new_from_data(
        inferred_options,
        inferred_scalar_type,
        deviceOptional,
        r.pyobject(ARG_VALUES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference);
    // See Note [Ensuring sparse values and indices match devices]
    Tensor indices = internal_new_from_data(
        values.options(),
        kLong,
        deviceOptional,
        r.pyobject(ARG_INDICES),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/false);
    return at::sparse_coo_tensor(
               indices,
               values,
               values.options().layout(at::kSparse).pinned_memory(pin_memory),
               r.toBoolOptional(ARG_IS_COALESCED))
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD));
  } else if (r.idx == 1) {
    bool pin_memory = r.toBool(ARG_PIN_MEMORY1);
    bool type_inference = r.isNone(ARG_TYPE1);
    const auto inferred_options =
        typeIdWithDefault(r, ARG_DEVICE1, dispatch_key);
    const auto inferred_scalar_type =
        r.scalartypeWithDefault(ARG_TYPE1, scalar_type);
    auto deviceOptional = r.deviceOptional(ARG_DEVICE1);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS1, default_check_invariants));

    Tensor values = internal_new_from_data(
        inferred_options,
        inferred_scalar_type,
        deviceOptional,
        r.pyobject(ARG_VALUES1),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/type_inference);
    // See Note [Ensuring sparse values and indices match devices]
    Tensor indices = internal_new_from_data(
        values.options(),
        kLong,
        deviceOptional,
        r.pyobject(ARG_INDICES1),
        /*copy_variables=*/false,
        /*copy_numpy=*/true,
        /*type_inference=*/false);
    return at::sparse_coo_tensor(
               indices,
               values,
               r.intlist(ARG_SIZE1),
               values.options().layout(at::kSparse).pinned_memory(pin_memory),
               r.toBoolOptional(ARG_IS_COALESCED1))
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD1));
  } else if (r.idx == 2) {
    const auto inferred_options =
        typeIdWithDefault(r, ARG_DEVICE2, dispatch_key);
    const auto inferred_scalar_type =
        r.scalartypeWithDefault(ARG_TYPE2, scalar_type);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(ARG_DEVICE2));
    at::globalContext().setCheckSparseTensorInvariants(
        r.toBoolWithDefault(ARG_CHECK_INVARIANTS2, default_check_invariants));

    return at::sparse_coo_tensor(
               r.intlist(ARG_SIZE2),
               inferred_options.dtype(inferred_scalar_type).layout(at::kSparse))
        .set_requires_grad(r.toBool(ARG_REQUIRES_GRAD2));
  }
  TORCH_CHECK(false, "sparse_coo_tensor(): invalid arguments");
}

void _validate_sparse_coo_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  static PythonArgParser parser({
      "_validate_sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  Tensor values = internal_new_from_data(
      options,
      scalar_type,
      std::nullopt,
      r.pyobject(1),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
  // See Note [Ensuring sparse values and indices match devices]
  Tensor indices = internal_new_from_data(
      values.options(),
      kLong,
      std::nullopt,
      r.pyobject(0),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/false);
  at::native::_validate_sparse_coo_tensor_args(indices, values, r.intlist(2));
}

void _validate_sparse_compressed_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  enum {
    ARG_COMPRESSED_INDICES = 0,
    ARG_PLAIN_INDICES,
    ARG_VALUES,
    ARG_SIZE,
    ARG_LAYOUT,
    ARGS_COUNT
  };

  const std::string signature =
      "_validate_sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, IntArrayRef size, Layout layout)";
  static PythonArgParser parser({signature});

  ParsedArgs<ARGS_COUNT> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  Tensor values = internal_new_from_data(
      options,
      scalar_type,
      std::nullopt,
      r.pyobject(ARG_VALUES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
  // See Note [Ensuring sparse values and indices match devices]
  Tensor compressed_indices = internal_new_from_data(
      values.options(),
      kInt,
      std::nullopt,
      r.pyobject(ARG_COMPRESSED_INDICES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
  Tensor plain_indices = internal_new_from_data(
      values.options(),
      kInt,
      std::nullopt,
      r.pyobject(ARG_PLAIN_INDICES),
      /*copy_variables=*/false,
      /*copy_numpy=*/true,
      /*type_inference=*/true);
  at::native::_validate_sparse_compressed_tensor_args(
      compressed_indices,
      plain_indices,
      values,
      r.intlist(ARG_SIZE),
      r.layout(ARG_LAYOUT));
}

template <c10::Layout required_layout>
static void _validate_sparse_compressed_tensor_args_template(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  enum {
    ARG_COMPRESSED_INDICES = 0,
    ARG_PLAIN_INDICES,
    ARG_VALUES,
    ARG_SIZE,
    ARGS_COUNT
  };
  static std::string sig;
  switch (required_layout) {
    case c10::Layout::SparseCsr:
      sig =
          "_validate_sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size)";
      break;
    case c10::Layout::SparseCsc:
      sig =
          "_validate_sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size)";
      break;
    case c10::Layout::SparseBsr:
      sig =
          "_validate_sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size)";
      break;
    case c10::Layout::SparseBsc:
      sig =
          "_validate_sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size)";
      break;
    default:;
  }
  static PythonArgParser parser({sig});

  ParsedArgs<ARGS_COUNT> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  Tensor values = internal_new_from_data(
      options,
      scalar_type,
      std::nullopt,
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/utils`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/utils`):

- [`python_tuples.h_kw.md_docs.md`](./python_tuples.h_kw.md_docs.md)
- [`six.h_kw.md_docs.md`](./six.h_kw.md_docs.md)
- [`tensor_types.cpp_docs.md_docs.md`](./tensor_types.cpp_docs.md_docs.md)
- [`tensor_list.h_kw.md_docs.md`](./tensor_list.h_kw.md_docs.md)
- [`verbose.h_kw.md_docs.md`](./verbose.h_kw.md_docs.md)
- [`invalid_arguments.cpp_kw.md_docs.md`](./invalid_arguments.cpp_kw.md_docs.md)
- [`tensor_apply.h_kw.md_docs.md`](./tensor_apply.h_kw.md_docs.md)
- [`cuda_enabled.h_docs.md_docs.md`](./cuda_enabled.h_docs.md_docs.md)
- [`tensor_layouts.h_docs.md_docs.md`](./tensor_layouts.h_docs.md_docs.md)
- [`variadic.h_kw.md_docs.md`](./variadic.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `tensor_new.cpp_docs.md_docs.md`
- **Keyword Index**: `tensor_new.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
