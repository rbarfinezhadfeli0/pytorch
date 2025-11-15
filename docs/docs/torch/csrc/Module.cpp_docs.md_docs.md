# Documentation: `docs/torch/csrc/Module.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/Module.cpp_docs.md`
- **Size**: 53,257 bytes (52.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/Module.cpp`

## File Metadata

- **Path**: `torch/csrc/Module.cpp`
- **Size**: 92,677 bytes (90.50 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/DeviceAccelerator.h>
#include <fmt/core.h>
#include <sys/types.h>
#include <torch/csrc/python_headers.h>
#include <csignal>
#include <optional>

#ifndef _MSC_VER
#include <sys/socket.h>
#endif

#include <ATen/ATen.h>
#include <ATen/BlasBackend.h>
#include <ATen/CachedTensorUtils.h>
#include <ATen/DLConvertor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/LegacyVmapMode.h>
#include <ATen/LinalgBackend.h>

#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/core/Vitals.h>
#include <ATen/dlpack.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/Normalization.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/AbortHandler.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <libshm.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/THConcat.h>
#include <torch/csrc/utils/pybind.h>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include <ATen/ThreadLocalPythonObjects.h>
#include <torch/csrc/DataLoader.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DeviceAccelerator.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/TypeInfo.h>
#include <torch/csrc/acc/Module.h>
#include <torch/csrc/api/include/torch/python/init.h>
#include <torch/csrc/autograd/generated/python_return_types.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_enum_tag.h>
#include <torch/csrc/autograd/python_fft_functions.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_legacy_variable.h>
#include <torch/csrc/autograd/python_linalg_functions.h>
#include <torch/csrc/autograd/python_nested_functions.h>
#include <torch/csrc/autograd/python_nn_functions.h>
#include <torch/csrc/autograd/python_sparse_functions.h>
#include <torch/csrc/autograd/python_special_functions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/distributed/python_placement.h>
#include <torch/csrc/dynamo/init.h>
#include <torch/csrc/export/pybind.h>
#include <torch/csrc/functionalization/Module.h>
#include <torch/csrc/functorch/init.h>
#include <torch/csrc/fx/node.h>
#include <torch/csrc/inductor/aoti_package/pybind.h>
#include <torch/csrc/inductor/aoti_runner/pybind.h>
#include <torch/csrc/instruction_counter/Module.h>
#include <torch/csrc/jit/python/init.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/lazy/python/init.h>
#include <torch/csrc/monitor/python_init.h>
#include <torch/csrc/mps/Module.h>
#include <torch/csrc/mtia/Module.h>
#include <torch/csrc/multiprocessing/init.h>
#include <torch/csrc/onnx/init.h>
#include <torch/csrc/profiler/python/init.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/init.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/utils/tensor_qschemes.h>
#include <torch/csrc/utils/verbose.h>

#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/profiler/kineto_client_interface.h>
#include <sstream>

#ifdef USE_CUDA
#include <ATen/ROCmFABackend.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <torch/csrc/inductor/static_cuda_launcher.h>
#ifdef __HIP_PLATFORM_AMD__
#include <ATen/native/cudnn/hip/BatchNorm.h>
#else
#include <ATen/native/cudnn/BatchNorm.h>
#endif
#endif

#ifdef USE_DISTRIBUTED
#ifdef USE_C10D
#include <torch/csrc/distributed/autograd/python_autograd.h>
#include <torch/csrc/distributed/c10d/c10d.h>
#include <torch/csrc/distributed/rpc/rpc.h>
#include <torch/csrc/distributed/rpc/testing/testing.h>
#endif
#endif

#if defined(USE_VALGRIND)
#include <callgrind.h>
#endif

#ifdef USE_ITT
#include <torch/csrc/itt.h>
#endif

#include <torch/nativert/python/Bindings.h>

namespace py = pybind11;

static PyObject* module;

static THPGenerator* THPDefaultCPUGenerator = nullptr;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static PyObject* THPModule_initNames(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  static std::vector<std::string> names;

  THPObjectPtr types(PySequence_Fast(arg, "expected a sequence"));
  if (!types)
    return nullptr;

  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto num_classes = PySequence_Fast_GET_SIZE(types.get());
  names.reserve(names.size() + num_classes);
  for (Py_ssize_t i = 0; i < num_classes; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(types.get(), i);
    TORCH_CHECK(PyType_Check(obj), "expected a PyTypeObject");
    PyTypeObject* type = reinterpret_cast<PyTypeObject*>(obj);

    THPObjectPtr module_name(PyObject_GetAttrString(obj, "__module__"));
    if (!module_name)
      return nullptr;
    TORCH_CHECK(
        THPUtils_checkString(module_name.get()),
        "expected __module__ to be a string");
    std::string name = THPUtils_unpackString(module_name.get());
    names.emplace_back(name + "." + type->tp_name);
    type->tp_name = names.back().c_str();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
//
// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THPModule_initExtension(
    PyObject* _unused,
    PyObject* shm_manager_path) {
  HANDLE_TH_ERRORS
#if !defined(FBCODE_CAFFE2) && !defined(__aarch64__)
  if (torch::get_cpp_stacktraces_enabled()) {
    c10::SetStackTraceFetcher([]() -> std::string {
      auto tb = torch::CapturedTraceback::gather(false, false, true);
      if (torch::get_symbolize_mode() == torch::unwind::Mode::addr2line) {
        LOG(WARNING)
            << "symbolizing C++ stack trace for exception; if this hangs, rerun with TORCH_DISABLE_ADDR2LINE=1..."
            << '\n';
      }
      auto s_tbs = torch::symbolize({tb.get()});
      std::stringstream oss;
      oss << "C++ CapturedTraceback:" << '\n';
      const auto& s_tb = s_tbs.tracebacks.at(0);
      for (auto idx : c10::irange(s_tb.size())) {
        // Skip the first few frames:
        //  #1 torch::CapturedTraceback::gather(bool, bool, bool)
        //  #2 THPModule_initExtension
        //  #3 THPModule_initExtension(_object*, _object*)::{lambda()#1}
        if (idx <= 3) {
          continue;
        }
        auto frame_id = s_tb[idx];
        const auto& frame = s_tbs.all_frames.at(frame_id);
        oss << "#" << idx << " " << frame.funcname << " from " << frame.filename
            << ":" << frame.lineno << '\n';
      }
      return oss.str();
    });
  }
#endif
  if (!THPUtils_checkString(shm_manager_path)) {
    THPUtils_setError(
        "initialization error - expected bytes/string object as shm_manager_path!");
    return nullptr;
  }
  torch::utils::initializeLayouts();
  torch::utils::initializeMemoryFormats();
  torch::utils::initializeQSchemes();
  torch::utils::initializeDtypes();
  torch::tensors::initialize_python_bindings();
  std::string path = THPUtils_unpackString(shm_manager_path);
  libshm_init(path.c_str());

  auto module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!module)
    throw python_error();

  THPStorage_postInit(module);
  THPAutograd_initFunctions();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// The idea behind these functions is to make it easy to test if we are
// built with ASAN: they're designed not to crash if ASAN is not enabled, but
// to trigger ASAN if it is enabled.  This lets us run a "canary" tests which
// checks if our build environment is misconfigured.

static PyObject* THPModule_crashIfCsrcASAN(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "crash_if_csrc_asan expects an int, but got ",
      THPUtils_typename(arg));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
  volatile char x[3];
  x[THPUtils_unpackInt(arg)] = 0;
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return THPUtils_packInt32(x[0]);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_crashIfCsrcUBSAN(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "crash_if_csrc_ubsan expects an int, but got ",
      THPUtils_typename(arg));
  int32_t x = THPUtils_unpackInt(arg);
  double y = 1.0 / x;
  return THPUtils_packInt32(static_cast<int>(y));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_crashIfvptrUBSAN(PyObject* module, PyObject* noarg) {
  // This code should work perfectly fine, as vtables are identical for Foo and
  // Baz unless rtti and ubsan are enabled
  struct Foo {
    virtual int bar() = 0;
    virtual ~Foo() = default;
  };
  struct Baz {
    virtual int bar() {
      return 17;
    }
    virtual ~Baz() = default;
  };
  Baz x{};
  // Purposely cast through `void*` so there's no fixups applied.
  // NOLINTNEXTLINE(bugprone-casting-through-void,-warnings-as-errors)
  auto y = static_cast<Foo*>(static_cast<void*>(&x));
  auto rc = y->bar();
  return THPUtils_packInt32(rc);
}

static PyObject* THPModule_crashIfATenASAN(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "crash_if_aten_asan expects an int, "
      "but got ",
      THPUtils_typename(arg));
  return THPUtils_packInt32(at::_crash_if_asan(THPUtils_unpackInt(arg)));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_abort(PyObject* module, PyObject* noargs) {
  std::terminate();
  Py_RETURN_NONE;
}

static PyObject* THPModule_crashIfDebugAssertsFail(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "crash_if_debug_asserts_fail expects an int, but got ",
      THPUtils_typename(arg));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      THPUtils_unpackInt(arg) != 424242,
      "Expect anything but 424242 as an input for debug builds");
  return THPUtils_packInt32(0);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getNumThreads(PyObject* module, PyObject* noargs) {
  return THPUtils_packInt32(at::get_num_threads());
}

static PyObject* THPModule_setNumThreads(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "set_num_threads expects an int, but got ",
      THPUtils_typename(arg));
  int nthreads = THPUtils_unpackInt(arg);
  TORCH_CHECK(nthreads > 0, "set_num_threads expects a positive integer");
  at::set_num_threads(nthreads);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getNumInteropThreads(
    PyObject* module,
    PyObject* noargs) {
  return THPUtils_packUInt64(at::get_num_interop_threads());
}

static PyObject* THPModule_setNumInteropThreads(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "set_num_interop_threads expects an int, "
      "but got ",
      THPUtils_typename(arg));
  int nthreads = THPUtils_unpackInt(arg);
  TORCH_CHECK(
      nthreads > 0, "set_num_interop_threads expects a positive integer");
  at::set_num_interop_threads(nthreads);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setDefaultTensorType(
    PyObject* _unused,
    PyObject* type) {
  HANDLE_TH_ERRORS
  torch::tensors::py_set_default_tensor_type(type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setDefaultDtype(PyObject* _unused, PyObject* dtype) {
  HANDLE_TH_ERRORS
  torch::tensors::py_set_default_dtype(dtype);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_swap_tensor_impl(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* a_ = nullptr;
  PyObject* b_ = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &a_, &b_)) {
    return nullptr;
  }

  // Ensure we have Tensors
  TORCH_CHECK(THPVariable_Check(a_));
  TORCH_CHECK(THPVariable_Check(b_));

  THPVariable* a = reinterpret_cast<THPVariable*>(a_);
  THPVariable* b = reinterpret_cast<THPVariable*>(b_);

  // weak_use_count() adds 1 if use_count is non-zero
  TORCH_CHECK(
      a->cdata->weak_use_count() == 1,
      "Expected no weakrefs to t1's Tensor object but got  ",
      a->cdata->weak_use_count() - 1);
  TORCH_CHECK(
      b->cdata->weak_use_count() == 1,
      "Expected no weakrefs to t2's Tensor object but got  ",
      b->cdata->weak_use_count() - 1);

  // Swap the Tensor Impl
  c10::MaybeOwned<at::Tensor> tmp = a->cdata;

  // The TensorImpls contain PyObjectSlots that have a reference to the PyObject
  // associated with the TensorImpl. Swap this field as well.
  std::optional<PyObject*> mb_obj_a =
      a->cdata->unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          /*ignore_hermetic_tls=*/false);
  std::optional<PyObject*> mb_obj_b =
      b->cdata->unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          /*ignore_hermetic_tls=*/false);
  TORCH_INTERNAL_ASSERT(
      mb_obj_a.has_value() && mb_obj_b.has_value(),
      "Both tensors should have PyObjects tagged by the current python interpreter");
  TORCH_CHECK(mb_obj_a.value() == a_);
  TORCH_CHECK(mb_obj_b.value() == b_);

  a->cdata = b->cdata;
  b->cdata = tmp;

  a->cdata->unsafeGetTensorImpl()->pyobj_slot()->init_pyobj(a_);
  b->cdata->unsafeGetTensorImpl()->pyobj_slot()->init_pyobj(b_);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_addDocStr(PyObject* _unused, PyObject* args) {
  // adds a __doc__ string to a function, similar to numpy's arr_add_docstring
  static std::vector<std::string> all_docs;
  PyObject* obj = nullptr;
  PyObject* doc_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &obj, &doc_obj)) {
    return nullptr;
  }

  const char* doc_str = "<invalid string>";
  if (THPUtils_checkString(doc_obj)) {
    all_docs.push_back(THPUtils_unpackString(doc_obj));
    doc_str = all_docs.back().c_str();
  }

  if (Py_TYPE(obj) == &PyCFunction_Type) {
    PyCFunctionObject* f = reinterpret_cast<PyCFunctionObject*>(obj);
    if (f->m_ml->ml_doc) {
      return PyErr_Format(
          PyExc_RuntimeError,
          "function '%s' already has a docstring",
          f->m_ml->ml_name);
    }
    f->m_ml->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* m = reinterpret_cast<PyMethodDescrObject*>(obj);
    if (m->d_method->ml_doc) {
      return PyErr_Format(
          PyExc_RuntimeError,
          "method '%s' already has a docstring",
          m->d_method->ml_name);
    }
    m->d_method->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    PyGetSetDescrObject* m = reinterpret_cast<PyGetSetDescrObject*>(obj);
    if (m->d_getset->doc) {
      return PyErr_Format(
          PyExc_RuntimeError,
          "attribute '%s' already has a docstring",
          m->d_getset->name);
    }
    m->d_getset->doc = doc_str;
  } else if (Py_TYPE(obj) == &PyType_Type) {
    PyTypeObject* t = reinterpret_cast<PyTypeObject*>(obj);
    if (t->tp_doc) {
      return PyErr_Format(
          PyExc_RuntimeError, "Type '%s' already has a docstring", t->tp_name);
    }
    t->tp_doc = doc_str;
  } else {
    return PyErr_Format(
        PyExc_TypeError,
        "don't know how to add docstring to type '%s'",
        Py_TYPE(obj)->tp_name);
  }

  Py_INCREF(obj);
  return obj;
}

static PyObject* THPModule_inferSize(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;
  TORCH_CHECK(num_args == 2, "expected exactly 2 arguments");
  PyObject* arg1 = PyTuple_GET_ITEM(args, 0);
  TORCH_CHECK(THPSize_Check(arg1), "expected a torch.Size as argument 1");
  PyObject* arg2 = PyTuple_GET_ITEM(args, 1);
  TORCH_CHECK(THPSize_Check(arg2), "expected a torch.Size as argument 2");

  auto size1 = THPUtils_unpackLongs(arg1);
  auto size2 = THPUtils_unpackLongs(arg2);
  auto sizes = at::infer_size(size1, size2);
  return THPSize_NewFromSizes(static_cast<int64_t>(sizes.size()), sizes.data());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setBackcompatBroadcastWarn(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_backcompat_broadcast_warn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  setBackCompatBroadcastWarn(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getBackcompatBroadcastWarn(
    PyObject* module,
    PyObject* noargs) {
  if (getBackCompatBroadcastWarn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setBackcompatKeepdimWarn(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_backcompat_keepdim_warn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  setBackCompatKeepdimWarn(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getBackcompatKeepdimWarn(
    PyObject* module,
    PyObject* noargs) {
  if (getBackCompatKeepdimWarn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_hasDistributed(PyObject* _unused, PyObject* noargs) {
#ifdef USE_DISTRIBUTED
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyObject* THPModule_showConfig(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::show_config());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_cxxFlags(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_cxx_flags());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_parallelInfo(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_parallel_info());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getCpuCapability(
    PyObject* module,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_cpu_capability());
  END_HANDLE_TH_ERRORS
}

namespace {

template <class T>
void DLPack_Capsule_Destructor(PyObject* data) {
  if (C10_LIKELY(!PyCapsule_IsValid(data, at::DLPackTraits<T>::capsule))) {
    // early out, see DLPack spec: if a consuming library sets the capsule
    // name to something else, they own it and we don't need to do anything
    return;
  }
  HANDLE_TH_ERRORS
  // Causes overheads for validity checks again, but this case is rare
  // since consuming libraries should rename the capsule according to spec.
  // Note that this cannot set a python error (we checked validity above),
  // so we don't need to handle python error state here.
  T* tensor = (T*)PyCapsule_GetPointer(data, at::DLPackTraits<T>::capsule);
  // the dlMTensor has not been consumed, call deleter ourselves.
  // DLPack spec mentions that deleter may be NULL, but deleter from
  // `at::toDLPack` is never NULL, so no need for an additional check here.
  tensor->deleter(tensor);
  END_HANDLE_TH_ERRORS_RET()
}

template <class T>
PyObject* THPModule_toDLPackImpl(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"_to_dlpack(Tensor data, *, IntArrayRef? dl_device=None, bool? copy=None)"});
  torch::ParsedArgs<3> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);

  TORCH_INTERNAL_ASSERT(r.idx == 0);

  auto data = r.tensor(0);
  auto dl_device = r.intlist(1);
  auto copy = r.toBoolOptional(2);

  // Parse the int list into a tuple.
  std::optional<DLDevice> optional_dl_device;

  if (!dl_device.empty()) {
    TORCH_CHECK(
        dl_device.size() == 2,
        "dl_device must be either None or a tuple of ints");
    optional_dl_device = DLDevice{
        static_cast<DLDeviceType>(dl_device[0]),
        static_cast<int32_t>(dl_device[1])};
  }

  auto tensor = at::DLPackTraits<T>::toDLPack(
      at::maybeCopyTensor(data, optional_dl_device, copy));
  return PyCapsule_New(
      tensor, at::DLPackTraits<T>::capsule, DLPack_Capsule_Destructor<T>);

  END_HANDLE_TH_ERRORS
}

} // namespace

static PyObject* THPModule_toDLPack(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  return THPModule_toDLPackImpl<DLManagedTensor>(self, args, kwargs);
}

static PyObject* THPModule_toDLPackVersioned(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  return THPModule_toDLPackImpl<DLManagedTensorVersioned>(self, args, kwargs);
}

static PyObject* THPModule_fromDLPack(PyObject* _unused, PyObject* data) {
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  auto tensor = torch::utils::tensor_fromDLPack(data);
  return THPVariable_Wrap(tensor);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_torchDeviceToDLDevice(
    PyObject* _unused,
    PyObject* data) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPDevice_Check(data),
      "torchDeviceToDLDevice: expected torch.device argument.");
  auto device = reinterpret_cast<THPDevice*>(data)->device;
  auto dl_device = at::torchDeviceToDLDevice(device);

  auto tuple = PyTuple_New(2);
  if (!tuple) {
    throw python_error();
  }

  PyTuple_SET_ITEM(tuple, 0, THPUtils_packInt64(dl_device.device_type));
  PyTuple_SET_ITEM(tuple, 1, THPUtils_packInt64(dl_device.device_id));

  return tuple;
  END_HANDLE_TH_ERRORS
}

static PyObject* THModule_getCppBacktrace(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  size_t frames_to_skip = 0;
  size_t maximum_number_of_frames = 0;
  if (!PyArg_ParseTuple(
          args, "LL", &frames_to_skip, &maximum_number_of_frames)) {
    return nullptr;
  }
  return THPUtils_packString(
      c10::get_backtrace(frames_to_skip, maximum_number_of_frames, true));
  END_HANDLE_TH_ERRORS
}

static PyObject* THModule_rename_privateuse1_backend(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkString(arg),
      "_rename_privateuse1_backend expects a str, but got ",
      THPUtils_typename(arg));
  const std::string backend_name = THPUtils_unpackString(arg);
  c10::register_privateuse1_backend(backend_name);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THModule_get_privateuse1_backend_name(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(c10::get_privateuse1_backend());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setAllowTF32CuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_tf32_cublas expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setAllowTF32CuDNN(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_allowTF32CuDNN(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::globalContext().allowTF32CuDNN())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setFloat32MatmulPrecision(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkString(arg),
      "set_float32_matmul_precision expects a str, "
      "but got ",
      THPUtils_typename(arg));
  std::string s = THPUtils_unpackString(arg);
  at::globalContext().setFloat32MatmulPrecision(s);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_float32MatmulPrecision(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  std::string s = "highest";
  auto p = at::globalContext().float32MatmulPrecision();
  if (p == at::Float32MatmulPrecision::HIGH) {
    s = "high";
  } else if (p == at::Float32MatmulPrecision::MEDIUM) {
    s = "medium";
  }
  return THPUtils_packString(s);
  END_HANDLE_TH_ERRORS
}
static PyObject* THPModule_setSDPPriorityOrder(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  auto priority_order = THPUtils_unpackLongs(arg);
  at::globalContext().setSDPPriorityOrder(priority_order);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject* THPModule_sDPPriorityOrder(
    PyObject* _unused,
    PyObject* noargs) {
  auto ordervec = at::globalContext().sDPPriorityOrder();
  auto order =
      THPObjectPtr(PyList_New(static_cast<Py_ssize_t>(ordervec.size())));
  for (const auto i : c10::irange(ordervec.size())) {
    PyObject* i64 = THPUtils_packInt64(static_cast<int64_t>(ordervec[i]));
    if (!i64)
      return nullptr;
    PyList_SET_ITEM(order.get(), i, i64);
  }
  return order.release();
}
static PyObject* THPModule_setSDPUseFlash(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseFlash(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject* THPModule_userEnabledFlashSDP(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().userEnabledFlashSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
static PyObject* THPModule_setSDPUseMemEfficient(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseMemEfficient(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject* userEnabledMemEfficientSDP(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().userEnabledMemEfficientSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
static PyObject* THPModule_setSDPUseMath(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseMath(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject* THPModule_userEnabledMathSDP(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().userEnabledMathSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
static PyObject* THPModule_setAllowFP16BF16ReductionMathSDP(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setAllowFP16BF16ReductionMathSDP(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject* THPModule_allowFP16BF16ReductionMathSDP(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().allowFP16BF16ReductionMathSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
static PyObject* THPModule_setSDPUseOverrideable(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_overrideable expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseOverrideable(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject* THPModule_userEnabledOverrideableSDP(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().userEnabledOverrideableSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
static PyObject* THPModule_setSDPUseCuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_sdp_use_cudnn expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseCuDNN(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject* THPModule_userEnabledCuDNNSDP(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().userEnabledCuDNNSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setUserEnabledCuDNN(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_enabled_cudnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setUserEnabledCuDNN(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_userEnabledCuDNN(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().userEnabledCuDNN())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setUserEnabledMkldnn(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_enabled_mkldnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setUserEnabledMkldnn(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_userEnabledMkldnn(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().userEnabledMkldnn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setDeterministicCuDNN(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_deterministic_cudnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setDeterministicCuDNN(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_deterministicCuDNN(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().deterministicCuDNN())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setDeterministicMkldnn(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_deterministic_mkldnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setDeterministicMkldnn(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_deterministicMkldnn(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().deterministicMkldnn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setDeterministicAlgorithms(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"_set_deterministic_algorithms(bool mode, *, bool warn_only=False)"});
  torch::ParsedArgs<2> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  bool mode = r.toBool(0);
  bool warn_only = r.toBool(1);
  at::globalContext().setDeterministicAlgorithms(mode, warn_only);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setAllowTF32OneDNN(
    PyObject* _unsued,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "_set_onednn_allow_tf32 expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setAllowTF32OneDNN(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_allowTF32OneDNN(
    PyObject* _unused,
    PyObject* noargs) {
#ifdef USE_XPU
  if (at::globalContext().allowTF32OneDNN())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
#else
  Py_RETURN_NONE;
#endif
}

static PyObject* THPModule_deterministicAlgorithms(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().deterministicAlgorithms()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* THPModule_deterministicAlgorithmsWarnOnly(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().deterministicAlgorithmsWarnOnly()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* THPModule_setDeterministicFillUninitializedMemory(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg), "expected a bool, but got ", THPUtils_typename(arg));
  at::globalContext().setDeterministicFillUninitializedMemory(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_deterministicFillUninitializedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().deterministicFillUninitializedMemory())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setUserEnabledNNPACK(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_enabled_NNPACK expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setUserEnabledNNPACK(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_userEnabledNNPACK(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().userEnabledNNPACK())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setWarnAlways(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "setWarnOnlyOnce expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  c10::WarningUtils::set_warnAlways(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_warnAlways(PyObject* _unused, PyObject* noargs) {
  if (c10::WarningUtils::get_warnAlways()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

// Used only for testing C++ to Python warning translations.
static PyObject* THPModule_warn(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_WARN("Test message for TORCH_WARN");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Used only for testing C++ to Python warning translations.
static PyObject* THPModule_warnDeprecation(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_WARN_DEPRECATION("Test message for TORCH_WARN_DEPRECATION");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setBenchmarkCuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_benchmark_cudnn expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setBenchmarkCuDNN(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_benchmarkCuDNN(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().benchmarkCuDNN()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* THPModule_setImmediateMiopen(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_immediate_miopen expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setImmediateMiopen(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_immediateMiopen(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().immediateMiopen()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* THPModule_setAllowTF32CuBLAS(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_tf32_cublas expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setAllowTF32CuBLAS(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_allowTF32CuBLAS(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::globalContext().allowTF32CuBLAS()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setAllowFP16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* allow_reduction_obj = nullptr;
  PyObject* allow_splitk_obj = Py_None;
  if (!PyArg_ParseTuple(args, "O|O", &allow_reduction_obj, &allow_splitk_obj)) {
    return nullptr;
  }
  TORCH_CHECK(
      PyBool_Check(allow_reduction_obj),
      "set_allow_fp16_reduction_cublas expects a bool for allow_reduced_precision, "
      "but got ",
      THPUtils_typename(allow_reduction_obj));
  bool allow_reduction = allow_reduction_obj == Py_True;
  bool allow_splitk = true;
  if (allow_splitk_obj != Py_None) {
    TORCH_CHECK(
        PyBool_Check(allow_splitk_obj),
        "set_allow_fp16_reduction_cublas expects a bool for allow_splitk, "
        "but got ",
        THPUtils_typename(allow_splitk_obj));
    allow_splitk = allow_splitk_obj == Py_True;
  }
  at::globalContext().setAllowFP16ReductionCuBLAS(
      allow_reduction, allow_splitk);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_allowFP16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* noargs) {
  auto option = at::globalContext().allowFP16ReductionCuBLAS();
  bool allow_reduced_precision =
      option == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK;
  bool allow_splitk = option !=
      at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK;
  return PyTuple_Pack(
      2,
      allow_reduced_precision ? Py_True : Py_False,
      allow_splitk ? Py_True : Py_False);
}

static PyObject* THPModule_setAllowBF16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* allow_reduction_obj = nullptr;
  PyObject* allow_splitk_obj = Py_None;
  if (!PyArg_ParseTuple(args, "O|O", &allow_reduction_obj, &allow_splitk_obj)) {
    return nullptr;
  }
  TORCH_CHECK(
      PyBool_Check(allow_reduction_obj),
      "set_allow_bf16_reduction_cublas expects a bool for allow_reduced_precision, "
      "but got ",
      THPUtils_typename(allow_reduction_obj));
  bool allow_reduction = allow_reduction_obj == Py_True;
  bool allow_splitk = true;
  if (allow_splitk_obj != Py_None) {
    TORCH_CHECK(
        PyBool_Check(allow_splitk_obj),
        "set_allow_bf16_reduction_cublas expects a bool for allow_splitk, "
        "but got ",
        THPUtils_typename(allow_splitk_obj));
    allow_splitk = allow_splitk_obj == Py_True;
  }
  at::globalContext().setAllowBF16ReductionCuBLAS(
      allow_reduction, allow_splitk);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_allowBF16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* noargs) {
  auto option = at::globalContext().allowBF16ReductionCuBLAS();
  bool allow_reduced_precision =
      option == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK;
  bool allow_splitk = option !=
      at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK;
  return PyTuple_Pack(
      2,
      allow_reduced_precision ? Py_True : Py_False,
      allow_splitk ? Py_True : Py_False);
}

static PyObject* THPModule_setAllowFP16AccumulationCuBLAS(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_fp16_accumulation_cublas expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setAllowFP16AccumulationCuBLAS(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_allowFP16AccumulationCuBLAS(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().allowFP16AccumulationCuBLAS()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* THPModule_setAllowFP16ReductionCPU(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_fp16_reduction_cpu expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setAllowFP16ReductionCPU(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_allowFP16ReductionCPU(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().allowFP16ReductionCPU()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject* THPModule_setFlushDenormal(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "flush_denormal expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  if (!at::globalContext().setFlushDenormal(arg == Py_True)) {
    Py_RETURN_FALSE;
  };
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getDefaultDtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto scalar_type = torch::tensors::get_default_scalar_type();
  return Py_NewRef(torch::getTHPDtype(scalar_type));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getDefaultDevice(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(c10::DeviceTypeName(
      dispatchKeyToDeviceType(torch::tensors::get_default_dispatch_key()),
      /*lower_case=*/true));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setQEngine(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "set_qengine expects an int, "
      "but got ",
      THPUtils_typename(arg));
  auto qengine = THPUtils_unpackLong(arg);
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  at::globalContext().setQEngine(static_cast<at::QEngine>(qengine));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_qEngine(PyObject* _unused, PyObject* noargs) {
  return THPUtils_packInt64(
      static_cast<int64_t>(at::globalContext().qEngine()));
}

static PyObject* THPModule_supportedQEngines(
    PyObject* _unused,
    PyObject* noargs) {
  const auto& qengines = at::globalContext().supportedQEngines();
  auto list =
      THPObjectPtr(PyList_New(static_cast<Py_ssize_t>(qengines.size())));
  if (!list)
    return nullptr;
  for (const auto i : c10::irange(qengines.size())) {
    PyObject* i64 = THPUtils_packInt64(static_cast<int64_t>(qengines[i]));
    if (!i64)
      return nullptr;
    PyList_SET_ITEM(list.get(), i, i64);
  }
  return list.release();
}

static PyObject* THPModule_isEnabledXNNPACK(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().isXNNPACKAvailable())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setCheckSparseTensorInvariants(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_check_sparse_tensor_invariants expects a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setCheckSparseTensorInvariants(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_checkSparseTensorInvariants(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().checkSparseTensorInvariants())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_willEngineExecuteNode(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  bool isTHPFunction = THPFunction_Check(arg);
  bool isTHPCppFunction = torch::autograd::THPCppFunction_Check(arg);
  TORCH_CHECK(
      isTHPFunction || isTHPCppFunction,
      "_will_engine_execute_node expects an grad_fn, "
      "but got ",
      THPUtils_typename(arg));
  const auto exec_info = torch::autograd::get_current_graph_task_exec_info();
  TORCH_CHECK(
      exec_info,
      "_get_should_execute_nodes should only be called during the backward pass");
  torch::autograd::Node* node = nullptr;
  std::shared_ptr<torch::autograd::Node> node_sp;
  if (isTHPFunction) {
    node_sp = (reinterpret_cast<THPFunction*>(arg))->cdata.lock();
    node = node_sp.get();
  } else {
    node =
        (reinterpret_cast<torch::autograd::THPCppFunction*>(arg))->cdata.get();
  }
  const auto nodes_in_graph =
      torch::autograd::get_current_graph_task_nodes_in_graph();
  bool ret = nodes_in_graph->find(node) != nodes_in_graph->end();
  if (ret && !exec_info->empty()) {
    auto it = exec_info->find(node);
    if (it == exec_info->end() || !it->second.should_execute()) {
      ret = false;
    } else {
      TORCH_CHECK(
          !(node->topological_nr() == 0 && it->second.captures_),
          "A leaf node was passed to _will_engine_execute_node but we are "
          "currently running autograd.grad(). This is currently not supported.");
    }
  }
  if (ret) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getCurrentGraphTaskExecutionOrder(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  std::vector<torch::autograd::Node*> nodes =
      torch::autograd::get_current_graph_task_execution_order();
  TORCH_CHECK(
      !nodes.empty(),
      "_current_graph_task_execution_order should only be called during the backward pass");
  auto list = THPObjectPtr(PyList_New(static_cast<Py_ssize_t>(nodes.size())));
  if (!list)
    return nullptr;
  for (const auto i : c10::irange(nodes.size())) {
    // This node is guaranteed to be alive since the backward is still running
    PyObject* pyobj_node =
        torch::autograd::functionToPyObject(nodes[i]->getptr());
    PyList_SET_ITEM(list.get(), i, pyobj_node);
  }
  return list.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getCurrentGraphTaskId(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(torch::autograd::get_current_graph_task_id());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getCurrentNode(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return torch::autograd::functionToPyObject(
      torch::autograd::get_current_node());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_isDefaultMobileCPUAllocatorSet(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(at::globalContext().isDefaultMobileCPUAllocatorSet());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setDefaultMobileCPUAllocator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::globalContext().setDefaultMobileCPUAllocator();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_unsetDefaultMobileCPUAllocator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::globalContext().unsetDefaultMobileCPUAllocator();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_vmapmode_increment_nesting(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::impl::VmapMode::increment_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_vmapmode_decrement_nesting(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::impl::VmapMode::decrement_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_set_display_vmap_fallback_warnings_mode(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "enabled must be a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setDisplayVmapFallbackWarnings(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_are_vmap_fallback_warnings_enabled(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::globalContext().areVmapFallbackWarningsEnabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_set_warn_on_accumulate_grad_stream_mismatch(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "enabled must be a bool, "
      "but got ",
      THPUtils_typename(arg));
  at::globalContext().setWarnOnAccumulateGradStreamMismatch(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_warn_on_accumulate_grad_stream_mismatch(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::globalContext().warnOnAccumulateGradStreamMismatch()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPModule_ensureCUDADeviceGuardSet(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::impl::ensureCUDADeviceGuardSet();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static std::initializer_list<PyMethodDef> TorchMethods = {
    {"_initExtension", THPModule_initExtension, METH_O, nullptr},
    {"_autograd_init", THPAutograd_initExtension, METH_NOARGS, nullptr},
    {"_add_docstr", THPModule_addDocStr, METH_VARARGS, nullptr},
    {"_swap_tensor_impl", THPModule_swap_tensor_impl, METH_VARARGS, nullptr},
    {"_init_names", THPModule_initNames, METH_O, nullptr},
    {"_has_distributed", THPModule_hasDistributed, METH_NOARGS, nullptr},
    {"_set_default_tensor_type",
     THPModule_setDefaultTensorType,
     METH_O,
     nullptr},
    {"_set_default_dtype", THPModule_setDefaultDtype, METH_O, nullptr},
    {"_infer_size", THPModule_inferSize, METH_VARARGS, nullptr},
    {"_abort", THPModule_abort, METH_NOARGS, nullptr},
    {"_crash_if_csrc_asan", THPModule_crashIfCsrcASAN, METH_O, nullptr},
    {"_crash_if_csrc_ubsan", THPModule_crashIfCsrcUBSAN, METH_O, nullptr},
    {"_crash_if_vptr_ubsan", THPModule_crashIfvptrUBSAN, METH_NOARGS, nullptr},
    {"_crash_if_aten_asan", THPModule_crashIfATenASAN, METH_O, nullptr},
    {"_crash_if_debug_asserts_fail",
     THPModule_crashIfDebugAssertsFail,
     METH_O,
     nullptr},
   
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/csrc`):

- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`Exceptions.cpp_docs.md_docs.md`](./Exceptions.cpp_docs.md_docs.md)
- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`Exceptions.h_docs.md_docs.md`](./Exceptions.h_docs.md_docs.md)
- [`serialization.cpp_kw.md_docs.md`](./serialization.cpp_kw.md_docs.md)
- [`QScheme.cpp_kw.md_docs.md`](./QScheme.cpp_kw.md_docs.md)
- [`DataLoader.cpp_kw.md_docs.md`](./DataLoader.cpp_kw.md_docs.md)
- [`Size.h_docs.md_docs.md`](./Size.h_docs.md_docs.md)
- [`DeviceAccelerator.h_kw.md_docs.md`](./DeviceAccelerator.h_kw.md_docs.md)
- [`Device.cpp_kw.md_docs.md`](./Device.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Module.cpp_docs.md_docs.md`
- **Keyword Index**: `Module.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
