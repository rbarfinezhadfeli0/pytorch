# Documentation: `torch/csrc/jit/python/script_init.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/python/script_init.cpp`
- **Size**: 96,072 bytes (93.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/utils/pybind.h>

#include <caffe2/serialize/versions.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/compatibility/backport.h>
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/quantization.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_sugared_value.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/graph_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <torch/csrc/jit/python/python_list.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/logging.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/testing/hooks_for_testing.h>

#include <torch/csrc/api/include/torch/ordered_dict.h>

#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/csrc/jit/mobile/train/export_data.h>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace torch::jit {

using ::c10::Argument;
using ::c10::FunctionSchema;

using FunctionDefaults = std::unordered_map<std::string, py::object>;
using ClassMethodDefaults = std::unordered_map<std::string, FunctionDefaults>;

namespace {

// A resolver that will inspect the outer Python scope to find `name`.
struct PythonResolver : public Resolver {
  explicit PythonResolver(ResolutionCallback rcb) : rcb_(std::move(rcb)) {}

  /**
   * While compiling classes, the class type we're compiling will not be
   * available in Python, since we haven't fowner_ defining the class yet. So
   * in order to make the class type available to its own methods, we need to
   * explicitly resolve it.
   *
   * @param rcb Python function to resolve a name to its Python object in the
   *            enclosing scope
   * @param classname The unqualified classname of the class currently being
   *                  compiled.
   * @param classType The class's type.
   */
  explicit PythonResolver(
      ResolutionCallback rcb,
      std::string classname,
      ClassTypePtr classType)
      : rcb_(std::move(rcb)),
        classname_(std::move(classname)),
        classType_(std::move(classType)) {}

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override {
    pybind11::gil_scoped_acquire ag;
    py::object obj = rcb_(name);
    if (obj.is_none()) {
      return nullptr;
    }
    return toSugaredValue(obj, m, loc);
  }

  static bool isNamedTupleClass(py::object obj) {
    auto tuple_type = reinterpret_cast<PyObject*>(&PyTuple_Type);
    return PyObject_IsSubclass(obj.ptr(), tuple_type) &&
        py::hasattr(obj, "_fields");
  }

  TypePtr resolveTypeFromObject(const py::object& obj, const SourceRange& loc) {
    if (py::isinstance<ScriptClass>(obj)) {
      auto script_class = py::cast<ScriptClass>(obj);
      return script_class.class_type_.type_;
    }

    py::bool_ isClass = py::module::import("inspect").attr("isclass")(obj);
    if (!py::cast<bool>(isClass)) {
      return nullptr;
    }

    if (isNamedTupleClass(obj)) {
      return registerNamedTuple(obj, loc, rcb_);
    }

    auto qualifiedName = c10::QualifiedName(
        py::cast<std::string>(py::module::import("torch._jit_internal")
                                  .attr("_qualified_name")(obj)));

    return get_python_cu()->get_type(qualifiedName);
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    if (classType_ && name == classname_) {
      return classType_;
    }
    pybind11::gil_scoped_acquire ag;
    py::object obj = rcb_(name);
    if (obj.is_none()) {
      return nullptr;
    }

    auto annotation_type =
        py::module::import("torch.jit.annotations")
            .attr("try_ann_to_type")(obj, loc, py::cpp_function(rcb_));
    if (!annotation_type.is_none()) {
      return py::cast<TypePtr>(annotation_type);
    }
    return resolveTypeFromObject(obj, loc);
  }

 private:
  ResolutionCallback rcb_;
  std::string classname_;
  ClassTypePtr classType_;
};

std::shared_ptr<PythonResolver> pythonResolver(const ResolutionCallback& rcb) {
  return std::make_shared<PythonResolver>(rcb);
}
std::shared_ptr<PythonResolver> pythonResolver(
    const ResolutionCallback& rcb,
    std::string classname,
    ClassTypePtr classType) {
  return std::make_shared<PythonResolver>(
      rcb, std::move(classname), std::move(classType));
}

void checkOverloadDecl(const Decl& new_decl, const Decl& old_decl) {
  const auto& new_params = new_decl.params();
  const auto& old_params = old_decl.params();

  // TODO. same number of parameters not strictly necessary.
  TORCH_INTERNAL_ASSERT(
      new_params.size() == old_params.size(),
      "Overload must have same number of parameters\n",
      new_decl.range(),
      old_decl.range());
  for (const auto i : c10::irange(new_decl.params().size())) {
    TORCH_INTERNAL_ASSERT(
        new_params[i].ident().name() == old_params[i].ident().name(),
        "Overload parameters must have the same names\n",
        new_params[i].ident(),
        old_params[i].ident());
  }
}

std::optional<IValue> tryCalculateDefaultParam(
    const Argument& arg,
    const py::object& def_value) {
  auto n = arg.N();
  auto list_type = arg.type()->cast<ListType>();
  try {
    if (n && *n > 0 && list_type) {
      // BroadcastingList, allow default values T for arg types List[T]
      return toIValue(def_value, list_type->getElementType());
    } else {
      return toIValue(def_value, arg.type());
    }
  } catch (...) {
    return std::nullopt;
  }
}

// An overloaded function may have a default that does not subtype all overloads
// @overload
// def foo(x: str)
// def foo(x=1)
FunctionDefaults calcOverloadedFunctionDefaults(
    const FunctionSchema& schema,
    const FunctionDefaults& defaults) {
  FunctionDefaults updated_defaults;
  for (const auto& arg : schema.arguments()) {
    const std::string& arg_name = arg.name();
    auto value = defaults.find(arg_name);
    if (value == defaults.end()) {
      continue;
    }
    auto maybe_ivalue = tryCalculateDefaultParam(arg, value->second);
    if (maybe_ivalue) {
      updated_defaults[arg_name] = value->second;
    }
  }
  return updated_defaults;
}

} // namespace

static bool checkMutableFunctionDefault(const py::object& def_arg) {
  if (py::isinstance<py::list>(def_arg) || py::isinstance<py::dict>(def_arg)) {
    return true;
  }
  if (py::isinstance<py::tuple>(def_arg)) {
    auto pytuple = def_arg.cast<py::tuple>();
    for (py::handle t : pytuple) {
      py::object obj = py::reinterpret_borrow<py::object>(t);
      if (checkMutableFunctionDefault(obj)) {
        return true;
      }
    }
  }
  return false;
}

static void checkMutableFunctionDefault(
    const SourceRange& range,
    const Argument& arg,
    const py::object& def_arg) {
  if (checkMutableFunctionDefault(def_arg) || arg.type()->cast<ClassType>()) {
    throw(
        ErrorReport(range)
        << "Mutable default parameters are not supported because Python binds them to the function"
        << " and they persist across function calls.\n As a workaround, make the default None and instantiate"
        << " the default parameter within the body of the function. Found "
        << py::type::handle_of(def_arg) << " on parameter " << arg.name());
  }
}

static FunctionSchema getSchemaWithNameAndDefaults(
    const SourceRange& range,
    const FunctionSchema& schema,
    const std::optional<std::string>& new_name,
    const FunctionDefaults& default_args) {
  std::vector<Argument> new_args;
  for (auto& arg : schema.arguments()) {
    auto it = default_args.find(arg.name());
    if (it != default_args.end()) {
      checkMutableFunctionDefault(range, arg, it->second);
      std::optional<IValue> value = tryCalculateDefaultParam(arg, it->second);
      if (!value) {
        ErrorReport error(range);
        error << "Expected a default value of type " << arg.type()->repr_str()
              << " on parameter \"" << arg.name() << "\".";
        if (arg.is_inferred_type()) {
          error << "Because \"" << arg.name()
                << "\" was not annotated with an explicit type "
                << "it is assumed to be type 'Tensor'.";
        }
        throw ErrorReport(error);
      }
      new_args.emplace_back(
          arg.name(), arg.type(), arg.N(), *value, arg.kwarg_only());
    } else {
      new_args.push_back(arg);
    }
  }
  return FunctionSchema(
      new_name.value_or(schema.name()),
      schema.overload_name(),
      new_args,
      schema.returns(),
      schema.is_vararg(),
      schema.is_varret());
}

static Decl mergeDefaultsAndExtraParametersToOverloadDecl(
    const Decl& overload_decl,
    const Decl& impl_decl,
    const FunctionDefaults& defaults) {
  std::vector<Param> adjusted_params;
  const auto& overload_params = overload_decl.params();
  const auto& impl_params = impl_decl.params();

  // following PEP specification that the following should work:
  // @overload
  // def mouse_event(x1: int, y1: int) -> ClickEvent: ...
  // ...
  // def mouse_event(x1: int, y1: int, x2: Optional[int] = None, y2:
  // Optional[int] = None)
  TORCH_CHECK(
      overload_params.size() <= impl_params.size(),
      "Overload should not have more parameters than implementation function",
      overload_decl.range(),
      impl_decl.range());

  for (const auto i : c10::irange(overload_params.size())) {
    auto overload_name = overload_params[i].ident().name();
    auto impl_name = impl_params[i].ident().name();
    if (overload_name != impl_name) {
      throw(
          ErrorReport(overload_decl.range())
          << "Overload parameters must have the same names. "
          << "Found " << overload_name << " and " << impl_name
          << " on argument " << i);
    }
    adjusted_params.push_back(overload_params[i]);
  }
  for (size_t i = overload_params.size(); i < impl_params.size(); ++i) {
    if (!defaults.count(impl_params[i].ident().name())) {
      throw(
          ErrorReport(impl_decl.range())
          << "Expected to find default parameter on argument"
          << impl_params[i].ident().name()
          << " because it is not defined on the overloaded declaration");
    }
    if (!impl_params[i].type().present()) {
      throw(
          ErrorReport(impl_decl.range())
          << "Parameters not specified on the overloaded declaration must have a type annotation in the implementation function."
          << " Did not find type for param " << impl_params[i].ident().name());
    }
    adjusted_params.push_back(impl_params[i]);
  }
  return Decl::create(
      overload_decl.range(),
      List<Param>::create(overload_decl.range(), adjusted_params),
      overload_decl.return_type());
}

static StrongFunctionPtr script_compile_overloaded_function(
    const c10::QualifiedName& name,
    const Decl& overload_decl,
    const Def& implementation_def,
    const ResolutionCallback& rcb,
    const FunctionDefaults& implementation_defaults,
    const py::object& signature) {
  if (signature.is_none()) {
    throw(
        ErrorReport(overload_decl.range())
        << "Must explicitly add type annotations to overloaded functions");
  }

  auto adjusted_decl = mergeDefaultsAndExtraParametersToOverloadDecl(
      overload_decl, implementation_def.decl(), implementation_defaults);
  auto new_def = implementation_def.withDecl(adjusted_decl);
  auto cu = get_python_cu();
  auto defined_functions = cu->define(
      QualifiedName(name.prefix()),
      /*properties=*/{},
      /*propResolvers=*/{},
      {new_def},
      {pythonResolver(rcb)},
      nullptr,
      true);
  TORCH_INTERNAL_ASSERT(defined_functions.size() == 1);
  auto& defined = defined_functions[0];
  FunctionDefaults updated_defaults = calcOverloadedFunctionDefaults(
      defined->getSchema(), implementation_defaults);
  defined->setSchema(getSchemaWithNameAndDefaults(
      new_def.range(),
      defined->getSchema(),
      new_def.name().name(),
      updated_defaults));
  StrongFunctionPtr ret(std::move(cu), defined);
  didFinishEmitFunction(ret);
  return ret;
}

static StrongFunctionPtr script_compile_function(
    const c10::QualifiedName& name,
    const Def& def,
    const FunctionDefaults& defaults,
    const ResolutionCallback& rcb) {
  auto cu = get_python_cu();
  auto defined_functions = cu->define(
      QualifiedName(name.prefix()),
      /*properties=*/{},
      /*propResolvers=*/{},
      {def},
      {pythonResolver(rcb)},
      nullptr,
      true);
  TORCH_INTERNAL_ASSERT(defined_functions.size() == 1);
  auto& defined = defined_functions[0];
  defined->setSchema(getSchemaWithNameAndDefaults(
      def.range(), defined->getSchema(), def.name().name(), defaults));
  StrongFunctionPtr ret(std::move(cu), defined);
  didFinishEmitFunction(ret);
  return ret;
}

struct VISIBILITY_HIDDEN ModuleSelf : public Self {
  ModuleSelf(std::shared_ptr<ConcreteModuleType> concreteType)
      : Self(), concreteType_(std::move(concreteType)) {}

  std::shared_ptr<SugaredValue> makeSugared(Value* v) const override {
    v->setType(getClassType());
    return std::make_shared<ModuleValue>(v, concreteType_);
  }

  ClassTypePtr getClassType() const override {
    return concreteType_->getJitType()->expect<ClassType>();
  }

 private:
  std::shared_ptr<ConcreteModuleType> concreteType_;
};

static std::shared_ptr<Graph> _propagate_shapes(
    Graph& graph,
    std::vector<at::Tensor> inputs,
    bool with_grad = false) {
  Stack stack(inputs.begin(), inputs.end());
  auto retval = graph.copy();
  setInputTensorTypes(*retval, stack, /*complete=*/false);
  PropagateInputShapes(retval);
  return retval;
}

static std::shared_ptr<Graph> _propagate_and_assign_input_shapes(
    Graph& graph,
    const std::vector<at::Tensor>& inputs,
    const std::vector<int>& param_count_list,
    bool with_grad = false,
    bool propagate = true) {
  auto retval = graph.copy();
  setInputTensorTypes(
      *retval, fmap<IValue>(inputs), /*complete=*/true, param_count_list);
  if (propagate) {
    PropagateInputShapes(retval);
  }
  return retval;
}

static void addFunctionToModule(Module& module, const StrongFunctionPtr& func) {
  // Make a graph with a fake self argument
  auto graph = toGraphFunction(*func.function_).graph()->copy();
  auto v = graph->insertInput(0, "self");
  v->setType(module._ivalue()->type());
  const auto name = QualifiedName(*module.type()->name(), "forward");
  auto method =
      module._ivalue()->compilation_unit()->create_function(name, graph);
  module.type()->addMethod(method);
}

// this is used in our test suite to check that we correctly preserved type tags
static bool ivalue_tags_match(const Module& lhs, const Module& rhs) {
  struct Work {
    IValue a;
    IValue b;
  };
  std::unordered_set<const void*> visited;
  std::vector<Work> work = {{lhs._ivalue(), rhs._ivalue()}};
  while (!work.empty()) {
    Work item = work.back();
    work.pop_back();
    if (item.a.isPtrType()) {
      // uncomment to debug type matching errors
      // std::cout << "MATCHING " << /*item.a <<*/ "(" << *item.a.type() << ") "
      //          << item.a.internalToPointer() << " " << /*item.b <<*/ " ("
      //          << *item.b.type() << ") " << item.b.internalToPointer() <<
      //          "\n";

      if (visited.count(item.a.internalToPointer())) {
        continue;
      }
      visited.emplace(item.a.internalToPointer());
    }
    if (!unshapedType(item.b.type())
             ->isSubtypeOf(unshapedType(item.b.type()))) {
      // Since named types are saved and loaded in the test suite, we cannot
      // expect them to be equal. We should still check their slots however.
      if (!item.a.type()->cast<c10::NamedType>()) {
        return false;
      }
    }
    // check tags for objects that contain subobjects
    if (item.a.isObject()) {
      auto ao = item.a.toObject();
      auto bo = item.b.toObject();
      for (size_t i = 0; i < ao->slots().size(); ++i) {
        work.emplace_back(Work{ao->slots().at(i), bo->slots().at(i)});
      }
    } else if (item.a.isTuple()) {
      auto at = item.a.toTuple();
      auto bt = item.b.toTuple();
      for (size_t i = 0; i < at->elements().size(); ++i) {
        work.emplace_back(Work{at->elements().at(i), bt->elements().at(i)});
      }
    } else if (item.a.isList()) {
      auto al = item.a.toList();
      auto bl = item.b.toList();
      for (const auto i : c10::irange(al.size())) {
        work.emplace_back(Work{al.get(i), bl.get(i)});
      }
    } else if (item.a.isGenericDict()) {
      auto ad = item.a.toGenericDict();
      auto bd = item.b.toGenericDict();
      for (auto& item : ad) {
        // Dictionaory keys cannot contain List/Dicts that require tags
        // so we do not have to check them.
        // Furthermore without ordered dicts it is expensive to find the
        // equivalent key
        work.emplace_back(Work{item.value(), bd.at(item.key())});
      }
    } else if (item.a.isFuture()) {
      auto af = item.a.toFuture();
      auto bf = item.b.toFuture();
      af->wait();
      bf->wait();
      work.emplace_back(Work{af->value(), bf->value()});
    }
  }

  return true;
}

// helper used to implement ._parameters, ._buffers, ._modules dicts
// inside of script nn.Module
template <typename Policy>
struct slot_dict_impl {
  slot_dict_impl(ModulePtr module) : module_(std::move(module)) {}
  bool contains(const std::string& name) const {
    if (auto slot = module_->type()->findAttributeSlot(name)) {
      if (Policy::valid(module_->type(), *slot, module_->getSlot(*slot))) {
        return true;
      }
    }
    return false;
  }

  std::vector<std::pair<std::string, py::object>> items() const {
    std::vector<std::pair<std::string, py::object>> result;
    for (size_t i = 0, N = module_->type()->numAttributes(); i < N; ++i) {
      if (Policy::valid(module_->type(), i, module_->getSlot(i))) {
        result.emplace_back(
            module_->type()->getAttributeName(i),
            toPyObject(module_->getSlot(i)));
      }
    }
    return result;
  }

  void setattr(const std::string& name, py::object value) {
    const TypePtr& type = module_->type()->getAttribute(name);
    Module(module_).setattr(name, toIValue(std::move(value), type));
  }

  py::object getattr(const std::string& name) {
    return toPyObject(Module(module_).attr(name));
  }

  static void bind(const py::module& m, const char* name) {
    py::class_<slot_dict_impl<Policy>>(m, name)
        .def(py::init(
            [](Module& m) { return slot_dict_impl<Policy>(m._ivalue()); }))
        .def("contains", &slot_dict_impl<Policy>::contains)
        .def("items", &slot_dict_impl<Policy>::items)
        .def("setattr", &slot_dict_impl<Policy>::setattr)
        .def("getattr", &slot_dict_impl<Policy>::getattr);
  }

 private:
  ModulePtr module_;
};

template <typename T>
static py::list debugMakeList(const T& list) {
  py::list result;
  for (const auto& elem : list) {
    result.append(py::cast(elem));
  }
  return result;
}
template <typename T>
static py::list debugMakeNamedList(const T& list) {
  py::list result;
  for (auto elem : list) {
    result.append(py::cast(std::make_pair(elem.name, elem.value)));
  }
  return result;
}
template <typename T>
static py::set debugMakeSet(const T& list) {
  py::set result;
  for (const auto& elem : list) {
    result.add(py::cast(elem));
  }
  return result;
}

static py::dict _jit_debug_module_iterators(Module& module) {
  py::dict result;
  result["children"] = debugMakeList(module.children());
  result["named_children"] = debugMakeNamedList(module.named_children());
  result["modules"] = debugMakeList(module.modules());
  result["named_modules"] = debugMakeNamedList(module.named_modules());

  result["parameters"] = debugMakeList(module.parameters(false));
  result["named_parameters"] =
      debugMakeNamedList(module.named_parameters(false));
  result["parameters_r"] = debugMakeList(module.parameters(true));
  result["named_parameters_r"] =
      debugMakeNamedList(module.named_parameters(true));

  result["buffers"] = debugMakeList(module.buffers(false));
  result["named_buffers"] = debugMakeNamedList(module.named_buffers(false));
  result["buffers_r"] = debugMakeList(module.buffers(true));
  result["named_buffers_r"] = debugMakeNamedList(module.named_buffers(true));

  result["named_attributes"] =
      debugMakeNamedList(module.named_attributes(false));
  result["named_attributes_r"] =
      debugMakeNamedList(module.named_attributes(true));
  return result;
}

static constexpr std::array<const char*, 48> magic_method_names = {
    "__lt__",      "__le__",      "__eq__",        "__ne__",
    "__ge__",      "__gt__",      "__not__",       "__abs__",
    "__add__",     "__and__",     "__floordiv__",  "__index__",
    "__inv__",     "__invert__",  "__lshift__",    "__mod__",
    "__mul__",     "__matmul__",  "__neg__",       "__or__",
    "__pos__",     "__pow__",     "__rshift__",    "__sub__",
    "__truediv__", "__xor__",     "__concat__",    "__contains__",
    "__delitem__", "__getitem__", "__setitem__",   "__iadd__",
    "__iand__",    "__iconcat__", "__ifloordiv__", "__ilshift__",
    "__imod__",    "__imul__",    "__imatmul__",   "__ior__",
    "__ipow__",    "__irshift__", "__isub__",      "__itruediv__",
    "__ixor__",    "__str__",     "__len__",       "__repr__",
};

struct DeepCopyMemoTable {
  std::shared_ptr<IValue::HashIdentityIValueMap> map;
};

static IValue pyIValueDeepcopy(const IValue& ivalue, const py::dict& memo) {
  if (!memo.contains(py::str("__torch_script_memo_table"))) {
    memo["__torch_script_memo_table"] =
        DeepCopyMemoTable{std::make_shared<IValue::HashIdentityIValueMap>()};
  }
  auto& ivalue_memo =
      *py::cast<DeepCopyMemoTable>(memo["__torch_script_memo_table"]).map;
  return ivalue.deepcopy(ivalue_memo);
}

static ExtraFilesMap extra_files_from_python(const py::dict& pydict) {
  ExtraFilesMap r;
  for (const auto& it : pydict) {
    r[py::cast<std::string>(it.first)] = "";
  }
  return r;
}

static void extra_files_to_python(
    const ExtraFilesMap& m,
    const py::dict& pydict) {
  // py::dict is pointer-like type so it gets modified despite const&
  for (const auto& it : m) {
    pydict[py::str(it.first)] = py::bytes(it.second);
  }
}

static void pyCompilationUnitDefine(
    CompilationUnit& cu,
    const std::string& src,
    const ResolutionCallback* rcb,
    const uint32_t _frames_up) {
  if (rcb && *rcb) {
    cu.define(std::nullopt, src, pythonResolver(*rcb), nullptr);
  } else {
    py::object py_default_rcb =
        py::module::import("torch._jit_internal")
            .attr("createResolutionCallbackFromFrame")(_frames_up);
    auto default_rcb = py_default_rcb.cast<ResolutionCallback>();
    cu.define(std::nullopt, src, pythonResolver(default_rcb), nullptr);
  }
}

// This function will copy bytes into a shared_ptr of chars aligned
// at kFlatbufferDataAlignmentBytes boundary (currently 16).
// This is required because tensors need to be aligned at 16 bytes boundary.
static std::shared_ptr<char> copyStr(const std::string& bytes) {
  size_t size = (bytes.size() / kFlatbufferDataAlignmentBytes + 1) *
      kFlatbufferDataAlignmentBytes;
#ifdef _WIN32
  std::shared_ptr<char> bytes_copy(
      static_cast<char*>(_aligned_malloc(size, kFlatbufferDataAlignmentBytes)),
      _aligned_free);
#elif defined(__APPLE__)
  void* p;
  ::posix_memalign(&p, kFlatbufferDataAlignmentBytes, size);
  TORCH_INTERNAL_ASSERT(p, "Could not allocate memory for flatbuffer");
  std::shared_ptr<char> bytes_copy(static_cast<char*>(p), free);
#else
  std::shared_ptr<char> bytes_copy(
      static_cast<char*>(aligned_alloc(kFlatbufferDataAlignmentBytes, size)),
      free);
#endif
  memcpy(bytes_copy.get(), bytes.data(), bytes.size());
  return bytes_copy;
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<c10::Capsule>(m, "Capsule");

  auto object_class =
      py::class_<Object>(m, "ScriptObject")
          .def("_type", [](Object& o) { return o.type(); })
          .def(
              "_get_method",
              [](Object& self, const std::string& name) -> Method {
                return self.get_method(name);
              },
              py::keep_alive<0, 1>())
          .def(
              "setattr",
              [](Object& self, const std::string& name, py::object value) {
                if (self.type()->hasConstant(name)) {
                  TORCH_CHECK(
                      false,
                      "Can't set constant '",
                      name,
                      "' which has value:",
                      self.type()->getConstant(name));
                }
                TypePtr type = self.type()->getAttribute(name);
                try {
                  auto ivalue = toIValue(std::move(value), type);
                  self.setattr(name, ivalue);
                } catch (std::exception& e) {
                  throw py::cast_error(c10::str(
                      "Could not cast attribute '",
                      name,
                      "' to type ",
                      type->repr_str(),
                      ": ",
                      e.what()));
                }
              })
          .def(
              "getattr",
              [](Object& self, const std::string& name) {
                try {
                  return toPyObject(self.attr(name));
                } catch (const ObjectAttributeError& err) {
                  PyErr_SetString(PyExc_AttributeError, err.what());
                  throw py::error_already_set();
                }
              })
          .def(
              "__getattr__",
              [](Object& self, const std::string& name) -> py::object {
                try {
                  if (name == "__qualname__") {
                    return py::cast(self.type()->name()->name());
                  }
                  if (auto method = self.find_method(name)) {
                    return py::cast(*method);
                  }
                  if (self.has_property(name)) {
                    auto prop = self.get_property(name);
                    // wrap the Method into callable PyObject
                    auto getter_func = py::cast(prop.getter_func);
                    return getter_func();
                  }
                  return toPyObject(self.attr(name));
                } catch (const ObjectAttributeError& err) {
                  PyErr_SetString(PyExc_AttributeError, err.what());
                  throw py::error_already_set();
                }
              })
          .def(
              "__setattr__",
              [](Object& self, const std::string& name, py::object value) {
                try {
                  if (self.has_property(name)) {
                    auto prop = self.get_property(name);
                    if (!prop.setter_func.has_value()) {
                      TORCH_CHECK(false, "can't set attribute");
                    }
                    // wrap the Method into callable PyObject
                    auto setter_func = py::cast(prop.setter_func);
                    setter_func(value);
                    return;
                  }

                  if (self.type()->hasConstant(name)) {
                    TORCH_CHECK(
                        false,
                        "Can't set constant '",
                        name,
                        "' which has value:",
                        self.type()->getConstant(name));
                  }
                  TypePtr type = self.type()->getAttribute(name);
                  auto ivalue = toIValue(std::move(value), type);
                  self.setattr(name, ivalue);
                } catch (const ObjectAttributeError& err) {
                  PyErr_SetString(PyExc_AttributeError, err.what());
                  throw py::error_already_set();
                }
              })
          .def(
              "hasattr",
              [](Object& self, const std::string& name) {
                return self.hasattr(name);
              })
          .def(
              "_has_method",
              [](Object& self, const std::string& name) {
                return bool(self.find_method(name));
              })
          .def(
              "_method_names",
              [](Object& self) {
                return fmap(self.get_methods(), [](const Method& method) {
                  return method.name();
                });
              })
          .def(
              "_properties", [](Object& self) { return self.get_properties(); })
          .def("__copy__", &Object::copy)
          .def(
              "__hash__",
              [](const Object& self) {
                // Similar to Tensor's `__hash__`, which is `id()`.
                return std::hash<c10::ivalue::Object*>{}(self._ivalue().get());
              })
          .def(
              "__deepcopy__",
              [](const Object& self, const py::dict& memo) {
                if (auto getstate_method = self.find_method("__getstate__")) {
                  auto object_state = toPyObject((*getstate_method)(Stack{}));

                  if (auto qualname = self.type()->name()) {
                    auto class_type = getCustomClass(qualname->qualifiedName());
                    auto self = Object(c10::ivalue::Object::create(
                        c10::StrongTypePtr(
                            std::shared_ptr<torch::jit::CompilationUnit>(),
                            class_type),
                        1));

                    if (auto setstate_method =
                            self.find_method("__setstate__")) {
                      auto setstate_schema =
                          setstate_method->function().getSchema();
                      TORCH_INTERNAL_ASSERT(
                          setstate_schema.arguments().size() == 2,
                          "__setstate__ method for class ",
                          class_type->repr_str(),
                          " must have exactly 2 arguments!");
                      auto state_type =
                          setstate_schema.arguments().at(1).type();
                      (*setstate_method)(
                          Stack{toIValue(object_state, state_type)});
                      return self;
                    }
                    std::stringstream err;
                    err << "Tried to deepcopy object ";
                    if (auto qualname = class_type->name()) {
                      err << qualname->qualifiedName() << " ";
                    }
                    err << "which does not have a __setstate__ method defined!";
                    throw std::runtime_error(err.str());
                  }
                }

                std::stringstream err;
                err << "Tried to deepcopy object ";
                if (auto qualname = self.type()->name()) {
                  err << qualname->qualifiedName() << " ";
                }
                err << "which does not have a __getstate__ method defined!";
                throw std::runtime_error(err.str());
              })
          .def(py::pickle(
              [](const Object& self)
                  -> std::tuple<py::object, std::string> { // __getstate__
                if (auto getstate_method = self.find_method("__getstate__")) {
                  auto object_state = toPyObject((*getstate_method)(Stack{}));
                  TORCH_INTERNAL_ASSERT(self.type()->name());
                  return std::make_tuple(
                      object_state, self.type()->name()->qualifiedName());
                }
                std::stringstream err;
                err << "Tried to serialize object ";
                if (auto qualname = self.type()->name()) {
                  err << qualname->qualifiedName() << " ";
                }
                err << "which does not have a __getstate__ method defined!";
                throw std::runtime_error(err.str());
              },
              [](const std::tuple<py::object, std::string>& state_tup)
                  -> Object {
                auto [state, qualname] = state_tup;
                auto class_type = getCustomClass(qualname);
                TORCH_CHECK(
                    class_type,
                    "Tried to deserialize class ",
                    qualname,
                    " which is not known to the runtime. "
                    "If this is a custom C++ class, make "
                    "sure the appropriate code is linked.");

                auto self = Object(c10::ivalue::Object::create(
                    c10::StrongTypePtr(
                        std::shared_ptr<torch::jit::CompilationUnit>(),
                        class_type),
                    1));
                if (auto setstate_method = self.find_method("__setstate__")) {
                  auto setstate_schema =
                      setstate_method->function().getSchema();
                  TORCH_INTERNAL_ASSERT(
                      setstate_schema.arguments().size() == 2,
                      "__setstate__ method for class ",
                      class_type->repr_str(),
                      " must have exactly 2 arguments!");
                  auto state_type = setstate_schema.arguments().at(1).type();
                  (*setstate_method)(Stack{toIValue(state, state_type)});
                  return self;
                }
                std::stringstream err;
                err << "Tried to deserialize object ";
                if (auto qualname = class_type->name()) {
                  err << qualname->qualifiedName() << " ";
                }
                err << "which does not have a __setstate__ method defined!";
                throw std::runtime_error(err.str());
              }));

  py::class_<Object::Property>(m, "ScriptObjectProperty")
      .def_property_readonly(
          "name", [](const Object::Property& self) { return self.name; })
      .def_property_readonly(
          "getter",
          [](const Object::Property& self) { return self.getter_func; })
      .def_property_readonly("setter", [](const Object::Property& self) {
        return self.setter_func;
      });

  // Special case __str__ and __repr__ to make sure we can print Objects/Modules
  // regardless of if the user defined __str__/__repr__
  using MagicMethodImplType = std::function<py::object(
      const Object& self, py::args args, py::kwargs kwargs)>;

  std::unordered_map<std::string, MagicMethodImplType> special_magic_methods;
  special_magic_methods.emplace(
      "__str__",
      [](const Object& self,
         const py::args& args,
         const py::kwargs& kwargs) -> py::object {
        auto method = self.find_method("__str__");
        if (!method) {
          return py::str("ScriptObject <" + self.type()->str() + ">");
        }
        return invokeScriptMethodFromPython(*method, args, kwargs);
      });

  special_magic_methods.emplace(
      "__repr__",
      [](const Object& self,
         const py::args& args,
         const py::kwargs& kwargs) -> py::object {
        auto method = self.find_method("__repr__");
        if (!method) {
          std::stringstream ss;
          ss << std::hex << static_cast<const void*>(&self);
          return py::str("<torch.ScriptObject object at " + ss.str() + ">");
        }
        return invokeScriptMethodFromPython(*method, args, kwargs);
      });

  for (const char* mm_name : magic_method_names) {
    if (special_magic_methods.count(mm_name)) {
      object_class.def(mm_name, special_magic_methods[mm_name]);
    } else {
      object_class.def(
          mm_name,
          [mm_name](
              const Object& self,
              const py::args& args,
              const py::kwargs& kwargs) {
            auto method = self.find_method(mm_name);
            if (!method) {
              std::string msg = fmt::format(
                  "'{}' is not implemented for {}",
                  mm_name,
                  self.type()->str());
              throw c10::NotImplementedError(msg);
            }
            return invokeScriptMethodFromPython(*method, args, kwargs);
          });
    }
  }

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<DeepCopyMemoTable>(m, "DeepCopyMemoTable");

  py::class_<UpgraderEntry>(m, "_UpgraderEntry")
      .def(py::init<int, std::string, std::string>())
      .def_property_readonly(
          "bumped_at_version",
          [](const UpgraderEntry& self) { return self.bumped_at_version; })
      .def_property_readonly(
          "upgrader_name",
          [](const UpgraderEntry& self) { return self.upgrader_name; })
      .def_property_readonly("old_schema", [](const UpgraderEntry& self) {
        return self.old_schema;
      });

  py::class_<UpgraderRange>(m, "_UpgraderRange")
      .def(py::init<int, int>())
      .def_property_readonly(
          "min_version",
          [](const UpgraderRange& self) { return self.min_version; })
      .def_property_readonly("max_version", [](const UpgraderRange& self) {
        return self.max_version;
      });

  object_class.def(
      "__deepcopy__", [](const Object& self, const py::dict& memo) {
        return Object(
            pyIValueDeepcopy(IValue(self._ivalue()), memo).toObject());
      });

  // Used by torch.package to save ScriptModule objects in unified format.
  py::class_<ScriptModuleSerializer>(m, "ScriptModuleSerializer")
      .def(py::init<caffe2::serialize::PyTorchStreamWriter&>())
      .def("serialize", &ScriptModuleSerializer::serialize_unified_format)
      .def(
          "write_files",
          &ScriptModuleSerializer::writeFiles,
          py::arg("code_dir") = ".data/ts_code/code/")
      .def(
          "storage_context",
          &ScriptModuleSerializer::storage_context,
          pybind11::return_value_policy::reference_internal);

  // Used by torch.package to coordinate sharing of storages between eager
  // and ScriptModules.
  py::class_<
      SerializationStorageContext,
      std::shared_ptr<SerializationStorageContext>>(
      m, "SerializationStorageContext")
      .def("has_storage", &SerializationStorageContext::hasStorage)
      .def("get_or_add_storage", &SerializationStorageContext::getOrAddStorage);

  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module, Object>(m, "ScriptModule")
      .def(py::init<std::string, std::shared_ptr<CompilationUnit>, bool>())
      .def(
          "save",
          [](Module& m,
             const std::string& filename,
             const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            m.save(filename, _extra_files);
          },
          py::arg("filename"),
          py::arg("_extra_files") = ExtraFilesMap())
      .def(
          "save_to_buffer",
          [](Module& m, const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
            std::ostringstream buf;
            m.save(buf, _extra_files);
            return py::bytes(buf.str());
          },
          py::arg("_extra_files") = ExtraFilesMap())
      .def(
          "_save_for_mobile",
          [](Module& m,
             const std::string& filename,
             const ExtraFilesMap& _extra_files = ExtraFilesMap(),
             bool _save_mobile_debug_info = false,
             bool _use_flatbuffer = false) {
            m._save_for_mobile(
                filename,
                _extra_files,
                _save_mobile_debug_info,
                _use_flatbuffer);
          },
          py::arg("filename"),
          py::arg("_extra_files") = ExtraFilesMap(),
          py::arg("_save_mobile_debug_info") = false,
          py::arg("_use_flatbuffer") = false)
      .def(
          "_save_to_buffer_for_mobile",
          [](Module& m,
             const ExtraFilesMap& _extra_files = ExtraFilesMap(),
             bool _save_mobile_debug_info = false,
             bool _use_flatbuffer = false) {
            std::ostringstream buf;
            m._save_for_mobile(
                buf, _extra_files, _save_mobile_debug_info, _use_flatbuffer);
            return py::bytes(buf.str());
          },
          py::arg("_extra_files") = ExtraFilesMap(),
          py::arg("_save_mobile_debug_info") = false,
          py::arg("_use_flatbuffer") = false)
      .def("_set_optimized", &Module::set_optimized)
      .def(
          "dump",
          &Module::dump,
          py::arg("code") = true,
          py::arg("attrs") = true,
          py::arg("params") = true)
      .def(
          "dump_to_str",
          &Module::dump_to_str,
          py::arg("code") = true,
          py::arg("attrs") = true,
          py::arg("params") = true)
      .def(
          "_replicate_for_data_parallel",
          [](Module& module) {
            const ModulePtr& obj = module._ivalue();
            auto copy = c10::ivalue::Object::create(
                c10::StrongTypePtr(obj->compilation_unit(), obj->type()),
                obj->slots().size());
            for (size_t i = 0; i < obj->slots().size(); ++i) {
              copy->setSlot(i, obj->getSlot(i));
            }
            return Module(std::move(copy));
          })
      .def(
          "get_debug_state",
          [](Module& self) {
            if (auto m = self.find_method("forward")) {
              return m->get_executor().getDebugState();
            }
            throw std::runtime_error(
                "Attempted to call get_debug_state on a Module without a compiled forward()");
          })
      .def(
          "_define",
          [](Module& m,
             std::shared_ptr<ConcreteModuleType> concreteType,
             const std::string& script,
             const ResolutionCallback& rcb) {
            const auto self = ModuleSelf(std::move(concreteType));
            m._ivalue()->compilation_unit()->define(
                m.type()->name(), script, pythonResolver(rcb), &self);
            didFinishEmitModule(m);
          })
      .def(
          "_register_attribute",
          [](Module& m,
             const std::string& name,
             const TypePtr& type,
             py::handle value) {
            m.register_attribute(name, type, toIValue(value, type));
          })
      .def(
          "_create_method_from_trace",
          [](Module& self,
             const std::string& name,
             const py::function& func,
             const py::tuple& input_tuple,
             const py::function& var_name_lookup_fn,
             bool strict,
             bool force_outplace,
             const std::vector<std::string>& argument_names,
             bool store_inputs) {
            // prereq: Module's buffers and parameters are unique
            // this was ensured in python before calling this function
            auto typed_inputs = toTraceableStack(input_tuple);

            std::shared_ptr<Graph> graph =
                std::get<0>(tracer::createGraphByTracing(
                    func,
                    typed_inputs,
                    var_name_lookup_fn,
                    strict,
                    force_outplace,
                    &self,
                    argument_names));
            const auto method_name = QualifiedName(*self.type()->name(), name);
            auto fn = self._ivalue()->compilation_unit()->create_function(
                method_name, graph);
            self.type()->addMethod(fn);
            if (store_inputs) {
              self.store_traced_inputs(name, typed_inputs);
            }
            didFinishEmitModule(self);
          },
          py::arg("name"),
          py::arg("func"),
          py::arg("input_tuple"),
          py::arg("var_name_lookup_fn"),
          py::arg("strict"),
          py::arg("force_outplace"),
          py::arg("argument_names") = std::vector<std::string>(),
          py::arg("store_inputs"))
      .def(
          "_create_method_from_trace_with_dict",
          [](Module& self,
             const std::string& name,
             const py::function& func,
             const py::dict& input_dict,
             const py::function& var_name_lookup_fn,
             bool strict,
             bool force_outplace,
             const std::vector<std::string>& argument_names,
             bool store_inputs) {
            // prereq: Module's buffers and parameters are unique
            // this was ensured in python before calling this function
            auto typed_inputs = toTraceableStack(input_dict);

            std::shared_ptr<Graph> graph =
                std::get<0>(tracer::createGraphByTracingWithDict(
                    func,
                    input_dict,
                    typed_inputs,
                    var_name_lookup_fn,
                    strict,
                    force_outplace,
                    &self,
                    argument_names));
            const auto method_name = QualifiedName(*self.type()->name(), name);
            auto fn = self._ivalue()->compilation_unit()->create_function(
                method_name, graph);
            if (store_inputs) {
              self.store_traced_inputs(name, typed_inputs);
            }
            self.type()->addMethod(fn);
            didFinishEmitModule(self);
          },
          py::arg("name"),
          py::arg("func"),
          py::arg("input_dict"),
          py::arg("var_name_lookup_fn"),
          py::arg("strict"),
          py::arg("force_outplace"),
          py::arg("argument_names") = std::vector<std::string>(),
          py::arg("store_inputs"))
      .def(
          "_get_forward_hooks",
          [](const Module& m) {
            std::vector<StrongFunctionPtr> funcs;
            for (auto& hook : m.type()->getForwardHooks()) {
              funcs.emplace_back(m.type()->compilation_unit(), hook);
            }
            return funcs;
          })
      .def(
          "_get_forward_pre_hooks",
          [](const Module& m) {
            std::vector<StrongFunctionPtr> funcs;
            for (auto& pre_hook : m.type()->getForwardPreHooks()) {
              funcs.emplace_back(m.type()->compilation_unit(), pre_hook);
            }
            return funcs;
          })
      .def(
          "_retrieve_traced_inputs",
          [](const Module& m) {
            return ScriptDict(m.retrieve_traced_inputs());
          })
      .def_property_readonly(
          "code",
          [](Module& self) {
            std::vector<at::IValue> constants;
            PrintDepsTable deps;
            PythonPrint pp(constants, deps);
            pp.printNamedType(self.type());
            return pp.str();
          })
      .def_property_readonly(
          "code_with_constants",
          [](Module& self) {
            std::vector<at::IValue> constants;
            PrintDepsTable deps;
            PythonPrint pp(constants, deps);
            pp.printNamedType(self.type());
            std::map<std::string, at::IValue> consts;
            int i = 0;
            for (auto const& constant : constants) {
              consts["c" + std::to_string(i)] = constant;
              i += 1;
            }
            return std::make_tuple(pp.str(), std::move(consts));
          })
      .def("apply", &Module::apply)
      .def("__copy__", &Module::copy)
      .def(
          "__hash__",
          [](const Module& self) {
            // Similar to Tensor's `__hash__`, which is `id()`.
            return std::hash<c10::ivalue::Object*>{}(self._ivalue().get());
          })
      .def(
          "__eq__",
          [](const Module& self, const py::object& other) {
            // TODO: call UDF if it exists
            if (!py::isinstance<Module>(other)) {
              return false;
            }
            return self._ivalue().get() ==
                py::cast<Module>(other)._ivalue().get();
          })
      .def(
          "__deepcopy__",
          [](const Module
```



## High-Level Overview


This C++ file contains approximately 8 class(es)/struct(s) and 127 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `static`, `torch`

**Classes/Structs**: `PythonResolver`, `type`, `yet`, `type`, `currently`, `VISIBILITY_HIDDEN`, `Work`, `slot_dict_impl`, `DeepCopyMemoTable`, `of`, `M`, `inheritance`, `bodies`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `pybind11/detail/common.h`
- `pybind11/pytypes.h`
- `torch/csrc/jit/api/object.h`
- `torch/csrc/jit/python/script_init.h`
- `torch/csrc/utils/pybind.h`
- `caffe2/serialize/versions.h`
- `torch/csrc/Device.h`
- `torch/csrc/DynamicTypes.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/frontend/ir_emitter.h`
- `torch/csrc/jit/frontend/sugared_value.h`
- `torch/csrc/jit/mobile/code.h`
- `torch/csrc/jit/mobile/compatibility/backport.h`
- `torch/csrc/jit/mobile/compatibility/model_compatibility.h`
- `torch/csrc/jit/mobile/file_format.h`
- `torch/csrc/jit/mobile/flatbuffer_loader.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/mobile/quantization.h`
- `torch/csrc/jit/operator_upgraders/upgraders.h`
- `torch/csrc/jit/operator_upgraders/upgraders_entry.h`
- `torch/csrc/jit/operator_upgraders/utils.h`
- `torch/csrc/jit/operator_upgraders/version_map.h`
- `torch/csrc/jit/python/module_python.h`
- `torch/csrc/jit/python/python_ivalue.h`
- `torch/csrc/jit/python/python_sugared_value.h`
- `torch/csrc/jit/serialization/export_bytecode.h`
- `torch/csrc/jit/serialization/flatbuffer_serializer.h`
- `torch/csrc/jit/serialization/import.h`
- `torch/csrc/jit/testing/file_check.h`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/csrc/jit/python`):

- [`python_ir.cpp_docs.md`](./python_ir.cpp_docs.md)
- [`python_custom_class.cpp_docs.md`](./python_custom_class.cpp_docs.md)
- [`python_ivalue.h_docs.md`](./python_ivalue.h_docs.md)
- [`pybind_utils.cpp_docs.md`](./pybind_utils.cpp_docs.md)
- [`opaque_obj.h_docs.md`](./opaque_obj.h_docs.md)
- [`update_graph_executor_opt.h_docs.md`](./update_graph_executor_opt.h_docs.md)
- [`python_tree_views.h_docs.md`](./python_tree_views.h_docs.md)
- [`python_ir.h_docs.md`](./python_ir.h_docs.md)
- [`python_tracer.h_docs.md`](./python_tracer.h_docs.md)
- [`python_arg_flatten.h_docs.md`](./python_arg_flatten.h_docs.md)


## Cross-References

- **File Documentation**: `script_init.cpp_docs.md`
- **Keyword Index**: `script_init.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
