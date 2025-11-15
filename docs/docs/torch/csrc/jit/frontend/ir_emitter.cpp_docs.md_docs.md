# Documentation: `docs/torch/csrc/jit/frontend/ir_emitter.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/frontend/ir_emitter.cpp_docs.md`
- **Size**: 54,019 bytes (52.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/frontend/ir_emitter.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/frontend/ir_emitter.cpp`
- **Size**: 215,832 bytes (210.77 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/tree_views.h>

#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/canonicalize_modified_loop.h>
#include <torch/csrc/jit/frontend/convert_to_ssa.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/annotate_warns.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inline_forked_closures.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lift_closures.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
#include <torch/csrc/jit/testing/hooks_for_testing.h>

#include <torch/csrc/jit/ir/constants.h>

#include <c10/util/hash.h>
#include <optional>

#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <climits>
#include <set>
#include <stack>

namespace {
bool reportSourceLocation(size_t file_size) {
  if (file_size < 512ull * 1024) {
    return true;
  }
  const auto enable_env =
      c10::utils::get_env("PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION");
  bool flag = true;
  if (!enable_env.has_value() || enable_env == "0" || enable_env == "FALSE" ||
      enable_env == "false") {
    flag = false;
  }
  return flag;
}
} // namespace

namespace torch::jit {

using FunctionTable = std::unordered_map<std::string, Function&>;
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using TypeTable = std::unordered_map<std::string, TypePtr>;
using AttributeMap = std::unordered_map<std::string, Const>;
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

struct Refinement {
  Refinement(std::string identifier, TypePtr type)
      : identifier_(std::move(identifier)), type_(std::move(type)) {}
  const std::string& identifier() const {
    return identifier_;
  }
  TypePtr type() const {
    return type_;
  }

 private:
  std::string identifier_;
  TypePtr type_;
};

struct RefinementSet {
  // When a comparison like x is None is made, we associate type refinements
  // with its true value and its false value. If a boolean that has refinements
  // associated with it is used in a conditional of an if statement, the true
  // and false refinements are inserted into the corresponding blocks
  using Refinements = std::vector<Refinement>;

  RefinementSet(Refinements true_refinements, Refinements false_refinements)
      : true_refinements_(std::move(true_refinements)),
        false_refinements_(std::move(false_refinements)) {}
  RefinementSet(Refinement single) : RefinementSet({std::move(single)}, {}) {}
  RefinementSet(Refinement single_true, Refinement single_false)
      : RefinementSet(
            Refinements({std::move(single_true)}),
            Refinements({std::move(single_false)})) {}
  RefinementSet() = default; // empty
  RefinementSet And(const RefinementSet& rhs) const {
    // if the result of an AND is true, both a & b had to be true,
    // so we take the union of a.true_refinements and b.true_refinements.
    // if the result is false, either a or b could have been false,
    // so we take their intersection.
    return RefinementSet(
        unionSet(true_refinements_, rhs.true_refinements_),
        intersectSet(false_refinements_, rhs.false_refinements_));
  }
  RefinementSet Or(const RefinementSet& rhs) const {
    // if the result of an OR is true, either a & b could have been true,
    // so we take the intersection of a.true_refinements & b.true_refinements.
    // if the result is false, both a and b had to be false,
    // so we take their union.
    return RefinementSet(
        intersectSet(true_refinements_, rhs.true_refinements_),
        unionSet(false_refinements_, rhs.false_refinements_));
  }

  RefinementSet Not() const {
    return RefinementSet(false_refinements_, true_refinements_);
  }
  const std::vector<Refinement> activeRefinements() const {
    return true_refinements_;
  }

 private:
  static bool sameVar(const Refinement& a, const Refinement& b) {
    return a.identifier() == b.identifier();
  }
  static Refinements unionSet(const Refinements& a, const Refinements& b) {
    Refinements result = a;
    for (const Refinement& r : b) {
      auto it =
          std::find_if(result.begin(), result.end(), [&](const Refinement& e) {
            return e.identifier() == r.identifier();
          });
      if (it == result.end()) {
        result.push_back(r);
      } else if (*it->type() != *r.type()) {
        // we only keep refinements when they exactly match one
        // refinement type, for instance, we do not attempt to refine:
        // isinstance(x, float) and isinstance(x, int)
        result.erase(it);
      }
    }
    return result;
  }
  static Refinements intersectSet(const Refinements& a, const Refinements& b) {
    Refinements result;
    for (const Refinement& r : a) {
      auto it = std::find_if(b.begin(), b.end(), [&](const Refinement& e) {
        return e.identifier() == r.identifier();
      });
      if (it != b.end() && r.type() == it->type()) {
        result.push_back(r);
      }
    }
    return result;
  }

  Refinements true_refinements_;
  Refinements false_refinements_;
};

struct CondValue {
  CondValue(
      Value* value,
      RefinementSet refinements,
      std::optional<bool> static_if)
      : value_(value),
        refinements_(std::move(refinements)),
        static_if_(static_if) {}
  CondValue(
      Graph& g,
      const SourceRange& loc,
      bool static_value,
      RefinementSet refinements)
      : value_(g.insertConstant(static_value, loc)),
        refinements_(std::move(refinements)),
        static_if_(static_value) {}
  Value* value() const {
    return value_;
  }
  const RefinementSet& refinements() const {
    return refinements_;
  }
  std::optional<bool> staticIf() const {
    return static_if_;
  }

 private:
  Value* value_;
  RefinementSet refinements_;
  std::optional<bool>
      static_if_; // certain expression cause us to emit a static if statement
                  // this value is present if this is the case.
                  // this is not equivalent to value_ being a constant
                  // it is possible for value_ to be constant but for
                  // the expression that produced it to not trigger the
                  // static if behavior. e.g. use of a variable assigned
                  // to a constant
};

enum NoneStatus { ALWAYS, MAYBE, NEVER };
static NoneStatus canBeNone(Value* v) {
  if (v->node()->mustBeNone()) {
    return ALWAYS;
  }
  if (v->type()->kind() == OptionalType::Kind ||
      (v->type()->kind() == UnionType::Kind &&
       v->type()->expect<UnionType>()->canHoldType(*NoneType::get()))) {
    return MAYBE;
  }
  return NEVER;
}

static Value* asSimple(const SugaredValuePtr& value) {
  if (SimpleValue* sv = dynamic_cast<SimpleValue*>(value.get())) {
    return sv->getValue();
  }
  return nullptr;
}

static std::shared_ptr<MagicMethod> makeMagic(
    const std::string& name,
    const SugaredValuePtr& base) {
  return std::make_shared<MagicMethod>(name, base);
}

// Auxiliary data structure for desugaring variable binding into our always
// explicitly scoped language as we descend down nested control structures in
// the frontend (which themselves don't introduce scopes)
//
// The Environment keeps track of two tables, one for values which are not first
// class and a type table for values which are. When a first class value
// is set in the environment, we emit a prim::Store which sets the
// name of the variable to appropriate type, and when a first-class value is
// referenced we emit a prim::Load that generates a value of the appropriate
// type.
//
// a = 1
// print(a)
// becomes:
// = prim::Store[name="a"](%a.1)
// %a : int = prim::Load[name="a"]()
// prim::Print(%a)

struct Environment {
  Environment(
      GraphFunction& method,
      ResolverPtr resolver,
      Block* b,
      std::shared_ptr<Environment> next = nullptr)
      : method(method),
        resolver(std::move(resolver)),
        b(b),
        next(std::move(next)) {}

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  GraphFunction& method;
  ResolverPtr resolver;
  std::unordered_map<std::string, std::function<std::string()>> error_messages;
  Block* b;

  std::shared_ptr<Environment> next;

  // set type error in the lowest environment. if the variable is used after an
  // error has been set, then we will use the more informative error message
  void setVariableTypeError(
      const std::string& name,
      std::function<std::string()> msg) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();
    }
    runner->error_messages[name] = std::move(msg);
  }

  // see if type error has been set for a variable
  std::optional<std::string> findVariableTypeError(const std::string& name) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();
    }
    auto msg = runner->error_messages.find(name);
    if (msg != runner->error_messages.end()) {
      return msg->second();
    } else {
      return std::nullopt;
    }
  }

  SugaredValuePtr insertLoad(const std::string& name, const TypePtr& type) {
    auto g = b->owningGraph();
    auto load = g->insertNode(g->createLoad(name, type));
    if (meaningfulName(name)) {
      load->output()->setDebugName(name);
    }
    return std::make_shared<SimpleValue>(load->output());
  }

  // note: type is not always the same as v->type(), e.g.
  // type: Optional[Tensor]
  // v->type(): Tensor
  void insertStore(
      const std::string& name,
      const SourceRange& loc,
      Value* v,
      TypePtr type) {
    auto g = b->owningGraph();
    g->insertNode(g->createStore(name, v))->setSourceRange(loc);
    type_table[name] = std::move(type);
  }

  SugaredValuePtr findInThisFrame(const std::string& name) {
    auto it = value_table.find(name);
    if (it != value_table.end()) {
      return it->second;
    }
    auto it2 = type_table.find(name);
    if (it2 != type_table.end()) {
      return insertLoad(name, it2->second);
    }
    return nullptr;
  }

  SugaredValuePtr findInParentFrame(const std::string& name) {
    return next ? next->findInAnyFrame(name) : nullptr;
  }

  void setType(const std::string& name, TypePtr type) {
    type_table[name] = std::move(type);
  }

  SugaredValuePtr findInAnyFrame(const std::string& name) {
    for (auto runner = this; runner; runner = runner->next.get()) {
      if (auto r = runner->findInThisFrame(name)) {
        return r;
      }
    }
    return nullptr;
  }

  Block* block() {
    return b;
  }

  void setVar(const SourceRange& loc, const std::string& name, Value* value) {
    setSugaredVar(
        loc,
        name,
        std::make_shared<SimpleValue>(value),
        /*annotated_type=*/nullptr);
  }

  void setSugaredVar(
      const SourceRange& loc,
      const std::string& name,
      SugaredValuePtr value,
      const TypePtr& annotated_type) {
    Value* as_simple_value = asSimple(value);
    if (as_simple_value && !as_simple_value->hasDebugName() &&
        meaningfulName(name) &&
        // note: if the value wasn't defined in this block, we might be giving a
        // name only used inside this block to a value outside of this. this is
        // not normally helpful for debugging and causes import/export jitter.
        as_simple_value->node()->owningBlock() == block()) {
      as_simple_value->setDebugName(name);
    }
    // prevent re-assignment involving any sugared values
    // any reassignment like:
    // a = ...
    // while ...
    //   a = ..
    // requires 'a' to be first-class in the graph since its value depends on
    // control flow
    if (auto parent = findInParentFrame(name)) {
      if (annotated_type) {
        throw(
            ErrorReport(loc)
            << "Attempting to declare and annotate the type of variable '"
            << name << "' but it is already defined in an outer block");
      }
      if (!as_simple_value) {
        throw(
            ErrorReport(loc)
            << "Cannot re-assign '" << name << "' to a value of type "
            << value->kind() << " because " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed");
      }
      Value* simple_parent = asSimple(parent);
      if (!simple_parent) {
        throw(
            ErrorReport(loc)
            << "Cannot re-assign '" << name << "' because it has type "
            << value->kind() << " and " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed");
      }

      auto parent_type = unshapedType(simple_parent->type());
      as_simple_value = tryConvertToType(
          loc,
          *b->owningGraph(),
          parent_type,
          as_simple_value,
          /*allow_conversions=*/true);
      std::stringstream why_not;
      if (!as_simple_value->type()->isSubtypeOfExt(*parent_type, &why_not)) {
        auto error = ErrorReport(loc);
        error << "Variable '" << name << "' previously had type "
              << simple_parent->type()->repr_str()
              << " but is now being assigned to a value of type "
              << as_simple_value->type()->repr_str();

        // Special-cased error msg if we're trying to assign to a tensor list.
        if (simple_parent->type()->kind() == TypeKind::ListType &&
            as_simple_value->type()->kind() == TypeKind::ListType) {
          error << "\nEmpty lists default to List[Tensor]. Add a variable "
                   "annotation to the assignment to create an empty list "
                   "of another type (torch.jit.annotate(List[T, []]) where T "
                   "is the type of elements in the list for Python 2)";
        }
        error << "\n" << why_not.str();
        throw ErrorReport(error);
      }
    }
    if (as_simple_value) {
      if (annotated_type &&
          !as_simple_value->type()->isSubtypeOf(*annotated_type)) {
        throw(
            ErrorReport(loc)
            << "Variable '" << name << "' is annotated with type "
            << annotated_type->repr_str()
            << " but is being assigned to a value of type "
            << as_simple_value->type()->repr_str());
      }
      auto value_store_type =
          annotated_type ? annotated_type : as_simple_value->type();
      insertStore(name, loc, as_simple_value, value_store_type);
    } else {
      value_table[name] = std::move(value);
    }
  }

  SugaredValuePtr getSugaredVar(const Ident& ident, bool required = true) {
    return getSugaredVar(ident.name(), ident.range());
  }
  Value* getVar(const Ident& ident) {
    return getSugaredVar(ident)->asValue(ident.range(), method);
  }

  void throwVarNotFoundError(
      const std::string& ident,
      const SourceRange& range) {
    // check if this value was not emitted in an if statement because of a
    // type mismatch. if it was, then we print a more informative error msg
    if (auto msg = findVariableTypeError(ident)) {
      throw(ErrorReport(range) << *msg << "and was used here");
    }
    throw(ErrorReport(range) << "undefined value " << ident);
  }

  SugaredValuePtr getSugaredVar(
      const std::string& ident,
      const SourceRange& range,
      bool required = true) {
    auto retval = findInAnyFrame(ident);

    if (!retval) {
      static std::unordered_map<std::string, SugaredValuePtr> globals = {
          {"print", std::make_shared<PrintValue>()},
          {"tuple", SpecialFormValue::create(prim::TupleConstruct)},
          {"float",
           makeMagic(
               "__float__",
               std::make_shared<CastValue>(FloatType::get(), aten::Float))},
          {"complex",
           makeMagic(
               "__complex__",
               std::make_shared<CastValue>(ComplexType::get(), aten::Complex))},
          {"int",
           makeMagic(
               "__int__",
               std::make_shared<CastValue>(IntType::get(), aten::Int))},
          {"bool",
           makeMagic(
               "__bool__",
               std::make_shared<CastValue>(BoolType::get(), aten::Bool))},
          {"str",
           makeMagic(
               "__str__",
               std::make_shared<CastValue>(StringType::get(), aten::str))},
          {"getattr", SpecialFormValue::create(prim::GetAttr)},
          {"hasattr", SpecialFormValue::create(prim::HasAttr)},
          {"isinstance", SpecialFormValue::create(prim::isinstance)},
          // todo(zach): remove when we can correctly export torch.full via ONNX
          // or we have implicit conversion that can convert numbers to tensors
          {"_to_tensor",
           std::make_shared<CastValue>(TensorType::get(), prim::NumToTensor)},
          {"len",
           makeMagic(
               "__len__",
               std::make_shared<BuiltinFunction>(aten::len, std::nullopt))},
          {"hex",
           makeMagic(
               "__hex__",
               std::make_shared<BuiltinFunction>(aten::hex, std::nullopt))},
          {"oct",
           makeMagic(
               "__oct__",
               std::make_shared<BuiltinFunction>(aten::oct, std::nullopt))},
          {"round",
           makeMagic(
               "__round__",
               std::make_shared<BuiltinFunction>(aten::round, std::nullopt))},
          {"hash", std::make_shared<BuiltinFunction>(aten::hash, std::nullopt)},
          {"id", std::make_shared<BuiltinFunction>(prim::id, std::nullopt)},
          {"min", std::make_shared<BuiltinFunction>(prim::min, std::nullopt)},
          {"max", std::make_shared<BuiltinFunction>(prim::max, std::nullopt)},
          {"abs", std::make_shared<BuiltinFunction>(prim::abs, std::nullopt)},
          {"all", std::make_shared<BuiltinFunction>(aten::all, std::nullopt)},
          {"any", std::make_shared<BuiltinFunction>(aten::any, std::nullopt)},
          {"divmod",
           std::make_shared<BuiltinFunction>(aten::divmod, std::nullopt)},
          {"sum", std::make_shared<BuiltinFunction>(aten::sum, std::nullopt)},
          {"list", SpecialFormValue::create(prim::list)},
          {"dict", SpecialFormValue::create(prim::dict)},
          {"ord", std::make_shared<BuiltinFunction>(aten::ord, std::nullopt)},
          {"chr", std::make_shared<BuiltinFunction>(aten::chr, std::nullopt)},
          {"bin", std::make_shared<BuiltinFunction>(aten::bin, std::nullopt)},
          {"pow", std::make_shared<BuiltinFunction>(aten::pow, std::nullopt)},
          {"range", SpecialFormValue::create(prim::range)},
          {"zip", SpecialFormValue::create(prim::zip)},
          {"enumerate", SpecialFormValue::create(prim::enumerate)},
          {"rangelist",
           std::make_shared<BuiltinFunction>(prim::rangelist, std::nullopt)},
          {"sorted",
           std::make_shared<BuiltinFunction>(aten::sorted, std::nullopt)},
          // Only AssertionError is bound so that we can use it from emitAssert,
          // all other exceptions should be resolved at the Python level
          {"AssertionError",
           std::make_shared<ExceptionValue>("AssertionError")},
      };
      auto it = globals.find(ident);
      if (it != globals.end()) {
        retval = it->second;
      }
    }

    if (!retval) {
      if (auto type = resolver->resolveType(ident, range)) {
        if (auto tuple_type = type->cast<TupleType>()) {
          retval = std::make_shared<NamedTupleConstructor>(tuple_type);
        }
      }
    }

    if (!retval) {
      retval = resolver->resolveValue(ident, method, range);
    }

    if (!retval) {
      if (auto type = resolver->resolveType(ident, range)) {
        if (auto class_type = type->cast<ClassType>()) {
          retval = std::make_shared<ClassValue>(class_type);
        }
      }
    }

    if (!retval && required) {
      throwVarNotFoundError(ident, range);
    }

    return retval;
  }

  Value* getVar(const std::string& ident, const SourceRange& range) {
    return getSugaredVar(ident, range)->asValue(range, method);
  }

  void removeVar(const Ident& ident, bool check_if_removed = false) {
    bool removed = false;

    for (auto runner = this; runner; runner = runner->next.get()) {
      auto a = runner->value_table.erase(ident.name());
      auto b = runner->type_table.erase(ident.name());
      removed = a || b;
    }

    if (check_if_removed && !removed) {
      throwVarNotFoundError(ident.name(), ident.range());
    }
  }

  std::vector<std::string> definedVariables() {
    std::vector<std::string> result;
    for (auto& kv : type_table) {
      result.push_back(kv.first);
    }
    return result;
  }

 private:
  TypeTable type_table;
  ValueTable value_table;
};

template <class T, class Hash>
static Value* materializeConstant(
    T val,
    Graph& graph,
    const SourceRange& r,
    std::unordered_map<T, Value*, Hash>& map) {
  auto existing_constant = map.find(val);
  if (existing_constant != map.end()) {
    return existing_constant->second;
  }

  WithInsertPoint guard(graph.block()->nodes().front());
  auto new_constant = graph.insertConstant(val, r);
  map[val] = new_constant;

  return new_constant;
}

// Information for each def being emitted.
// Defs can be nested to support closures so we need a stack of this information
// Currently records information about the functions return type.
struct DefContext {
  TypePtr declared_return_type_; // nullptr if not annotated
  TypePtr merged_return_type_; // nullptr if a Return has not been seen yet
};

enum class LoopStatus { NOT_IN_LOOP, IN_LOOP, IN_UNROLLED_LOOP };

struct WithLoopStatus {
  WithLoopStatus(LoopStatus* prev, LoopStatus new_status)
      : prev_ptr_(prev), prev_value_(*prev) {
    *prev = new_status;
  }
  ~WithLoopStatus() {
    *prev_ptr_ = prev_value_;
  }

 private:
  LoopStatus* prev_ptr_;
  LoopStatus prev_value_;
};

struct to_ir {
  to_ir(
      const Def& def,
      ResolverPtr resolver_,
      const Self* self,
      GraphFunction& method) // method being constructed
      : method(method),
        graph(method.graph()),
        resolver(std::move(resolver_)),
        typeParser_(resolver),
        environment_stack(nullptr) {
    AT_ASSERT(resolver);
    pushFrame(graph->block(), /*starts_def=*/true);

    // Type annotations exclude explicitly typing the "self" parameter, so in
    // the case that this is a method with self we expect one fewer parameter
    // annotation than the number of parameters this Def takes.
    if (self && def.decl().params().empty()) {
      throw(
          ErrorReport(def.decl().params().range())
          << "methods must have a self argument");
    }
    method.setSchema(emitDef(def, self, graph->block()));

    // At this point, we might have received a graph that is compiled with
    // old operator schemas that might not exist in the system anymore.
    // Therefore, we replace such ops with its' valid upgrader.
    ReplaceOldOperatorsWithUpgraders(graph);

    // NB ORDERING: SSA conversion has to occur before
    // lifting of closures and forks, this way closures are converted
    // to SSA while part of their original graph, and closures are ready to
    // be inlined into forked closures
    ConvertToSSA(graph);

    // convert loops with an iter and body condition specified to
    // python-recognize while loops. we do this so they can be exported,
    // and run the pass early to avoid jitter. Like conversion to SSA,
    // it only needs to run once.
    CanonicalizeModifiedLoops(graph);

    // Convert Ops to a Normalized Form
    NormalizeOps(graph);

    runCleanupPasses(graph);
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  GraphFunction& method;
  std::shared_ptr<Graph> graph;
  ResolverPtr resolver;
  std::unordered_map<int64_t, Value*, std::hash<int64_t>> integral_constants;
  std::unordered_map<double, Value*, std::hash<double>> fp_constants;
  std::unordered_map<
      c10::complex<double>,
      Value*,
      c10::hash<c10::complex<double>>>
      complex_constants;
  std::unordered_set<Block*> exit_blocks;
  ScriptTypeParser typeParser_;
  LoopStatus loop_status_ = LoopStatus::NOT_IN_LOOP;

  // Singly-linked list of environments. This top element contains a member
  // `next` that points to the most immediate enclosing scope's value.
  std::shared_ptr<Environment> environment_stack;
  std::vector<DefContext> def_stack_;
  size_t temp_name_count_ = 0;
  std::string createTempName(const std::string& prefix) {
    return prefix + std::to_string(temp_name_count_++);
  }

  void pushFrame(Block* b, bool starts_def = false) {
    if (starts_def) {
      def_stack_.emplace_back();
    }
    environment_stack =
        std::make_shared<Environment>(method, resolver, b, environment_stack);
  }
  std::shared_ptr<Environment> popFrame(bool ends_def = false) {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    if (ends_def) {
      def_stack_.pop_back();
    }
    return old_frame;
  }

  // If the graph might not return, add an implicit None return at the end
  void handleMaybeNoReturn(const Def& def, Block* block) {
    auto decl_ret = def_stack_.back().declared_return_type_;
    if (exit_blocks.count(block) == 0) {
      auto decl_ret = def_stack_.back().declared_return_type_;
      if (decl_ret && decl_ret != NoneType::get()) {
        throw(
            ErrorReport(def.range())
            << "Function was not annotated as having type None, but does not "
            << "return along all paths");
      }
      WithInsertPoint b(*block->nodes().end());
      emitReturn(Return::create(
          def.range(), Expr(Compound::create(TK_NONE, def.range(), {}))));
    } else {
      // if we haven't seen any return statements, but the graph block exits
      // (the function always throws) then we accept the declared return type if
      // it exists or set it to none
      if (def_stack_.back().merged_return_type_ == nullptr) {
        def_stack_.back().merged_return_type_ =
            decl_ret != nullptr ? decl_ret : NoneType::get();
      }
    }
  }

  FunctionSchema emitDef(const Def& def, const Self* self, Block* block) {
    auto schema = typeParser_.parseSchemaFromDef(def, bool(self));
    // TODO need guards on init returning none
    if (schema.returns().size() == 1) {
      def_stack_.back().declared_return_type_ = schema.returns().at(0).type();
    }
    std::vector<Argument> arguments =
        emitFormalArguments(def, self, schema, block);

    // body
    auto stmts_list = def.statements();
    emitStatements(stmts_list.begin(), stmts_list.end());
    handleMaybeNoReturn(def, block);
    std::vector<Argument> returns = {emitOutput(def.range(), schema, block)};
    return {def.name().name(), "", std::move(arguments), std::move(returns)};
  }

  // see [setstate type]
  static TypePtr getTypeForSetStateArg(const Def& def, const Self* self) {
    TORCH_CHECK(self, "Expected __setstate__ to have a `self` argument");
    auto getstate = self->getClassType()->findMethod("__getstate__");
    if (!getstate) {
      throw(
          ErrorReport(def.range())
          << "`__setstate__` defined but not `__getstate__`. "
          << "You must have both defined on a ScriptModule "
          << "to customize serialization.\n"
          << "Did you forget to use `@torch.jit.export`?");
    }
    getstate->ensure_defined();
    return self->getClassType()
        ->getMethod("__getstate__")
        .getSchema()
        .returns()
        .at(0)
        .type();
  }

  // see [setstate type]
  static bool shouldDeriveSetStateType(
      const Def& def,
      const FunctionSchema& schema) {
    const bool noTypeAnnotations = std::all_of(
        schema.arguments().begin(),
        schema.arguments().end(),
        [](const Argument& arg) { return arg.is_inferred_type(); });

    bool shouldInfer = def.name().name() == "__setstate__" && noTypeAnnotations;
    if (!shouldInfer) {
      return false;
    }

    // Do some additional basic validation that the __setstate__ func is
    // well-formed
    TORCH_INTERNAL_ASSERT(def.name().name() == "__setstate__");
    const auto numDeclParams = def.decl().params().size();
    if (numDeclParams != 2) {
      throw(
          ErrorReport(def.range())
          << "Expected 2 arguments for `__setstate__`, got: " << numDeclParams);
    }
    return true;
  }

  std::vector<Argument> emitFormalArguments(
      const Def& def,
      const Self* self,
      const FunctionSchema& schema,
      Block* block) {
    std::vector<Argument> arguments; // for schema
    // inputs
    auto it = def.decl().params().begin();
    auto end = def.decl().params().end();
    auto expected_annotation_size = def.decl().params().size();
    if (self) {
      expected_annotation_size--;
    }
    if (schema.arguments().size() != expected_annotation_size) {
      throw(
          ErrorReport(def.decl().params().range())
          << "Number of type annotations for"
          << " function parameters (" << schema.arguments().size() << ")"
          << " does not match the number of parameters on the function ("
          << expected_annotation_size << ")!");
    }

    if (self) {
      AT_ASSERT(it != end);
      const auto& name = (*it).ident().name();
      Value* new_input = block->addInput()->setDebugName(name);
      environment_stack->setSugaredVar(
          (*it).ident().range(),
          name,
          self->makeSugared(new_input),
          /*annotated_type=*/nullptr);
      arguments.emplace_back(name, new_input->type());
      ++it;
    }

    // [setstate type]
    // __setstate__ is special, because if the user leaves it un-annotated we
    // will derive the type for `state` from the output type of __getstate__.
    // This is necessary so that we can allow submodules to appear in `state`.
    bool shouldDeriveType = shouldDeriveSetStateType(def, schema);
    size_t arg_annotation_idx = 0;
    for (; it != end; ++it) {
      auto& name = (*it).ident().name();
      // Add the input to the graph
      Value* new_input = block->addInput();
      if (meaningfulName(name)) {
        new_input->setDebugName(name);
      }
      // Record the type for the schema and set the Type on the Value*
      auto arg = schema.arguments().at(arg_annotation_idx++);
      if (shouldDeriveType) {
        TORCH_INTERNAL_ASSERT(schema.arguments().size() == 1);
        const auto& inferredStateType = getTypeForSetStateArg(def, self);
        arg = arg.cloneWithType(inferredStateType);
      }

      arguments.push_back(arg);
      new_input->setType(arguments.back().type());

      // NB: set type of new_input before setVar call so the Store is
      // typed appropriately
      environment_stack->setVar((*it).ident().range(), name, new_input);
    }
    return arguments;
  }

  Argument emitOutput(
      const SourceRange& range,
      const FunctionSchema& schema,
      Block* block) {
    // handleMaybeNoReturn ensures that merged_return_type_ is always set
    auto ret_type = def_stack_.back().merged_return_type_;
    TORCH_INTERNAL_ASSERT(ret_type);

    // in the ConvertToSSA pass, prim::ReturnStmts are lowered so that the
    // correct return value is set. Until then, we have a correctly-typed
    // placeholder return value. This is needed so that closures & graphs
    // are correctly typed.
    auto placeholder_return =
        graph->insertNode(graph->createUninitialized(ret_type))->output();
    block->registerOutput(placeholder_return);
    return Argument("", def_stack_.back().merged_return_type_);
  }

  void emitStatements(const List<Stmt>& statements) {
    return emitStatements(statements.begin(), statements.end());
  }

  // XXX: Right now closures are not generically implemented and are only used
  // as an intermediate form for special tasks, like defining gradients or
  // forked functions.
  //
  // There are several unfinished aspects that make them unusable generally
  // 1. We do not have a type, ivalue, operator to represent prim::Closure, so
  // closure_node has type None
  // 2. There is no export logic for it yet, so it cannot be
  // exported/python_printed
  // 3. There is nothing preventing the assignment of already existing variables
  // inside the closures
  //    the changes to those variables will just get forgotten.
  // 4. There is no parsing support in frontend.py, this is intentional since it
  //    prevents people from accidentally using this feature.
  //
  // This function leaves in the graph something like:
  //
  //   %2 : None = prim::Closure()
  //     block0():
  //       %1 : Tensor = prim::DoSomething(%0)
  //       -> (%1)
  //
  // A separate pass is required to erase this closure and replace it with
  // something actually executable (see liftClosure and inlineForkedClosure).
  std::shared_ptr<ClosureValue> emitClosure(
      const std::function<void(Block*)>& emit_body) {
    Node* closure_node = graph->insertNode(graph->create(prim::Closure, 1));
    // it is not a real thing yet, so just say the type is None
    closure_node->output()->setType(NoneType::get());
    Block* block = closure_node->addBlock();
    WithLoopStatus loop_guard(&loop_status_, LoopStatus::NOT_IN_LOOP);
    {
      WithInsertPoint guard(block);
      pushFrame(block, /*starts_def=*/true);
      emit_body(block);
      popFrame(/*ends_def=*/true);
    }
    return std::make_shared<ClosureValue>(closure_node->output());
  }

  void emitClosure(const Def& def) {
    // invoked once the closure block is set as the environment
    auto emit_body = [&](Block* closure_block) {
      emitDef(
          def,
          nullptr,
          closure_block); // ignore schema return, we just won't use it for now
                          // since we never create a Method for the closure
    };
    auto closure_value = emitClosure(emit_body);
    environment_stack->setSugaredVar(
        def.name().range(),
        def.name().name(),
        closure_value,
        /*annotated_type=*/nullptr);
  }

  void checkBreakContinue(
      const SourceRange& loc,
      const std::string& stmt_name) {
    if (loop_status_ == LoopStatus::NOT_IN_LOOP) {
      throw(
          ErrorReport(loc) << "SyntaxError: '" << stmt_name << "'"
                           << " outside loop");
    } else if (loop_status_ == LoopStatus::IN_UNROLLED_LOOP) {
      throw(
          ErrorReport(loc)
          << "Because we emit iteration over modulelists or tuples as "
             "unrolled loops, we do not support break or continue inside the body of these loops");
    }
  }

  void emitBreak(const Break& stmt) {
    checkBreakContinue(stmt.range(), "break");
    auto break_node =
        graph->create(prim::BreakStmt, {}, 0)->setSourceRange(stmt.range());
    graph->insertNode(break_node);
  }

  void emitContinue(const Continue& stmt) {
    checkBreakContinue(stmt.range(), "continue");
    auto continue_node =
        graph->create(prim::ContinueStmt, {}, 0)->setSourceRange(stmt.range());
    graph->insertNode(continue_node);
  }

  void emitDelete(const Delete& stmt) {
    for (const auto& target : stmt.targets()) {
      if (target.kind() == TK_SUBSCRIPT) {
        Subscript subscript(target);
        const List<Expr>& subscript_exprs = subscript.subscript_exprs();
        if (subscript_exprs[0].kind() == TK_SLICE_EXPR) {
          throw(
              ErrorReport(target.range())
              << "del statements only support deletion at a single index, "
                 "slicing is not supported"
                 " (see https://github.com/pytorch/pytorch/issues/31430)");
        }
        const SugaredValuePtr sv = emitSugaredExpr(subscript.value(), 1);
        const SourceRange& val_range = subscript.value().range();
        Value* idx = emitExpr(subscript_exprs[0]);
        Value* val = sv->asValue(val_range, method);

        // If val is a class instance, this is a method call to a type-specific
        // implementation of del defined in a __delitem__ method.
        if (auto cls = val->type()->cast<ClassType>()) {
          if (!cls->findMethod("__delitem__")) {
            throw(
                ErrorReport(target.range())
                << "Class does not define __delitem__");
          }

          // Use MethodValue to call the method to handle recursion.
          MethodValue(val, "__delitem__")
              .call(stmt.range(), method, {idx}, {}, 0);
        } else {
          auto node = graph->create(aten::Delete, {val, idx}, 0)
                          ->setSourceRange(target.range());
          graph->insertNode(node);
        }
      } else if (target.kind() == TK_VAR) {
        Var var(target);
        environment_stack->removeVar(var.name(), /*check_if_removed=*/true);
      } else {
        throw(
            ErrorReport(target.range())
            << "del statements are only supported for deleting"
               " list and dict items and variables");
      }
    }
  }

  void emitReturn(const Return& stmt) {
    TypePtr declared_return_type =
        def_stack_.back().declared_return_type_; // nullptr if not annotated
    auto actual_return = emitExpr(stmt.expr(), declared_return_type);

    // result type is annotated, every return must convert to that type
    if (declared_return_type) {
      // this guard skips implicit conversion from None -> Tensor for the return
      // type. otherwise forgetting a return a function returning a tensor will
      // cause a None to be converted to a tensor.
      if (!(actual_return->type()->isSubtypeOf(*TensorType::get()) &&
            actual_return->type()->isSubtypeOf(*NoneType::get()))) {
        actual_return = tryConvertToType(
            stmt.range(),
            *graph,
            declared_return_type,
            actual_return,
            /*allow_conversions=*/true);
      }
      if (!actual_return->type()->isSubtypeOf(*declared_return_type)) {
        throw(
            ErrorReport(stmt.range())
            << "Return value was annotated as having type "
            << declared_return_type->repr_str() << " but is actually of type "
            << actual_return->type()->repr_str());
      }
    } else {
      declared_return_type = def_stack_.back().merged_return_type_;
      if (!declared_return_type) {
        declared_return_type = actual_return->type();
      }
      auto merged_return_type =
          unifyTypes(declared_return_type, actual_return->type());
      if (!merged_return_type) {
        throw(
            ErrorReport(stmt.range())
            << "Previous return statement returned a value of type "
            << declared_return_type->repr_str()
            << " but this return statement returns a value of type "
            << actual_return->type()->repr_str());
      }
      declared_return_type = merged_return_type.value();
    }
    AT_ASSERT(declared_return_type);

    def_stack_.back().merged_return_type_ = declared_return_type;

    // If the annotated return type is Any and the result type is not Any,
    // cast the result to Any to facilitate type unification between return
    // statements on different code paths (e.g. different branches of an if,
    // body and containing scope of a loop).
    if (declared_return_type == AnyType::get() &&
        actual_return->type() != AnyType::get()) {
      actual_return =
          graph->insertUncheckedCast(actual_return, declared_return_type);
    }

    graph->insertNode(graph->create(prim::ReturnStmt, {actual_return}, 0));
    exit_blocks.insert(environment_stack->block());
  }

  void emitStatements(
      List<Stmt>::const_iterator begin,
      List<Stmt>::const_iterator end) {
    for (; begin != end; ++begin) {
      auto stmt = *begin;
      ErrorReport::CallStack::update_pending_range(stmt.range());
      switch (stmt.kind()) {
        case TK_IF:
          emitIf(If(stmt));
          break;
        case TK_WHILE:
          emitWhile(While(stmt));
          break;
        case TK_FOR:
          emitFor(For(stmt));
          break;
        case TK_ASSIGN:
          emitAssignment(Assign(stmt));
          break;
        case TK_AUG_ASSIGN:
          emitAugAssignment(AugAssign(stmt));
          break;
        case TK_EXPR_STMT: {
          auto expr = ExprStmt(stmt).expr();
          emitSugaredExpr(expr, 0);
        } break;
        case TK_RAISE:
          emitRaise(Raise(stmt));
          break;
        case TK_ASSERT:
          emitAssert(Assert(stmt));
          break;
        case TK_RETURN: {
          emitReturn(Return(stmt));
        } break;
        case TK_CONTINUE: {
          emitContinue(Continue(stmt));
        } break;
        case TK_BREAK: {
          emitBreak(Break(stmt));
        } break;
        case TK_PASS:
          // Emit nothing for pass
          break;
        case TK_DEF:
          emitClosure(Def(stmt));
          break;
        case TK_DELETE:
          emitDelete(Delete(stmt));
          break;
        case TK_WITH:
          emitWith(With(stmt));
          break;
        default:
          throw(
              ErrorReport(stmt)
              << "Unrecognized statement kind " << kindToString(stmt.kind()));
      }
      // Found an exit statement in this block. The remaining statements aren't
      // reachable so we don't emit them.
      if (exit_blocks.count(environment_stack->block()))
        return;
    }
  }

  RefinementSet findIsNoneRefinements(
      const Expr& lhs,
      Value* lhs_value,
      const Expr& rhs,
      Value* rhs_value,
      int tok) {
    if (rhs.kind() != TK_NONE && lhs.kind() == TK_NONE) {
      // make 'None is var' into 'var is None'
      return findIsNoneRefinements(rhs, rhs_value, lhs, lhs_value, tok);
    }
    if (rhs.kind() != TK_NONE || lhs.kind() != TK_VAR) {
      return {};
    }
    // statement must be var {is, is not} None
    const std::string& name = Var(lhs).name().name();
    // While it should in theory be possible to specialize
    // the `x is None` to know x has type NoneType, we have previously
    // not done this. Unfortunately, doing this will make the type None
    // propagate further in all loaded models. The handling of
    // unwrap_optional will fail in these cases since export did
    // not expect that the input would be none and an unannotated None.
    // To enable this, we need to (1) implement a real casting operator
    // annotated(T, X) that stays in the graph and does the cast
    // and (2) only enable this OPTIONAL_NONE when loading newer
    // graphs because it is incompatible with older graphs.
    // Refinement none(name, RefinementKind::OPTIONAL_NONE);
    if (const auto optional_type = lhs_value->type()->cast<OptionalType>()) {
      Refinement present(name, optional_type->getElementType());
      if (tok == TK_IS) {
        return RefinementSet({}, {present});
      } else { // TK_ISNOT
        return RefinementSet({present}, {});
      }
    }
    if (const auto union_type = lhs_value->type()->cast<UnionType>()) {
      std::vector<TypePtr> to_subtract{NoneType::get()};
      std::optional<TypePtr> remaining =
          union_type->subtractTypeSet(to_subtract);
      std::vector<Refinement> all_present;
      if (remaining) {
        Refinement present{name, *remaining};
        all_present.push_back(std::move(present));
      }
      if (tok == TK_IS) {
        return RefinementSet({}, all_present);
      } else { // TK_ISNOT
        return RefinementSet(all_present, {});
      }
    }
    return RefinementSet();
  }

  CondValue emitCondExpr(const Expr& expr) {
    switch (expr.kind()) {
      case TK_AND:
      case TK_OR: {
        auto binop = BinOp(expr);
        return emitShortCircuitLogical(
            binop.range(), binop.lhs(), binop.rhs(), expr.kind() == TK_OR);
      }
      case TK_NOT: {
        CondValue v = emitCondExpr(Expr(expr.tree()->trees()[0]));
        Value* result = emitBuiltinCall(
            expr.range(), *graph, aten::__not__, {v.value()}, {});
        std::optional<bool> static_if;
        if (v.staticIf()) {
          static_if = !*v.staticIf();
        }
        return CondValue(result, v.refinements().Not(), static_if);
      } break;
      case TK_IS:
      case TK_ISNOT: {
        // meta programming on AST for is/is not cases and emit branches base on
        auto cond_op = BinOp(expr);
        Value* lhs_val = emitExpr(cond_op.lhs());
        Value* rhs_val = emitExpr(cond_op.rhs());

        auto lhs_none = canBeNone(lhs_val);
        auto rhs_none = canBeNone(rhs_val);

        // Dispatch logic (A: ALWAYS, N: NEVER, M: MAYBE):
        //
        // AA, -> statically IS always holds, IS_NOT never holds
        // AN , NA-> statically IS_NOT always holds, IS never holds
        // MA, MM, MN, NM, NN, AM -> cannot prove anything statically
        bool its_is = expr.kind() == TK_IS;
        if (lhs_none == ALWAYS && rhs_none == ALWAYS) {
          return CondValue(*graph, expr.range(), its_is, {});
        } else if (
            (lhs_none == ALWAYS && rhs_none == NEVER) ||
            (lhs_none == NEVER && rhs_none == ALWAYS)) {
          // lhs_val/rhs_val with A/M: only emit never_none_branch
          return CondValue(*graph, expr.range(), !its_is, {});
        } else {
          auto kind = getNodeKind(expr.kind(), expr.get()->trees().size());
          Value* cond_value = emitBuiltinCall(
              expr.get()->range(),
              *method.graph(),
              kind,
              {lhs_val, rhs_val},
              {});
          auto refinements = RefinementSet(findIsNoneRefinements(
              cond_op.lhs(), lhs_val, cond_op.rhs(), rhs_val, expr.kind()));
          return CondValue(cond_value, refinements, std::nullopt);
        }
      } break;
      default: {
        if (expr.kind() == TK_APPLY) {
          auto apply = Apply(expr);
          auto callee = Apply(expr).callee();
          if (callee.kind() == TK_VAR) {
            if (Var(callee).name().name() == "isinstance") {
              checkApplyNumInputs(apply, 2);
              return emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
            }
            if (Var(callee).name().name() == "hasattr") {
              checkApplyNumInputs(apply, 2);
              return emitHasAttr(apply.inputs()[0], apply.inputs()[1]);
            }
          }
          auto sv = emitSugaredExpr(apply.callee(), 1);
          auto loc = apply.callee().range();
          if (auto special_form = dynamic_cast<SpecialFormValue*>(sv.get())) {
            if (special_form->form() == prim::isinstance) {
              checkApplyNumInputs(apply, 2);
              return emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
            }
          }
        }
        auto expr_out = emitToBool(expr.range(), emitExpr(expr));
        std::optional<bool> static_if = std::nullopt;
        auto kind = expr_out->node()->kind();
        if (kind == aten::is_scripting) {
          static_if = true;
        } else if (kind == aten::has_torch_function) {
          static_if = false;
        }
        // MetaCompile on boolean literals and constants
        if (auto maybe_ivalue = toIValue(expr_out)) {
          static_if = maybe_ivalue->toBool();
        }
        return CondValue(expr_out, RefinementSet({}), static_if);
      } break;
    }
  }

  std::shared_ptr<Environment> emitSingleIfBranch(
      Block* b,
      const List<Stmt>& branch,
      const RefinementSet& refinements) {
    pushFrame(b);
    WithInsertPoint guard(b);
    insertRefinements(branch.range(), refinements);
    emitStatements(branch);
    return popFrame();
  }

  Node* create(Symbol kind, const SourceRange& loc, size_t n_outputs) {
    return graph->create(kind, n_outputs)->setSourceRange(loc);
  }

  Value* emitTernaryIf(
      const TernaryIf& expr,
      const TypePtr& type_hint = nullptr) {
    CondValue cond_value = emitCondExpr(expr.cond());
    // If the cond expr is a static value, then we metacompile the `if`
    // statemement and only emit true or false branch
    if (cond_value.staticIf()) {
      if (*cond_value.staticIf()) {
        return emitExpr(expr.true_expr(), type_hint);
      } else {
        return emitExpr(expr.false_expr(), type_hint);
      }
    }
    auto true_expr = [&] { return emitExpr(expr.true_expr(), type_hint); };
    auto false_expr = [&] { return emitExpr(expr.false_expr(), type_hint); };
    return emitIfExpr(expr.range(), cond_value, true_expr, false_expr);
  }

  template <class F1, class F2, class F3>
  void refineAndSetUnionTypeHintOrPopulateCandidatesVector(
      const TypePtr& type_hint,
      TypePtr* refined_type_hint_ptr,
      std::vector<TypePtr>* all_candidates,
      const std::string& match_repr,
      const Expr& src,
      const F1& type_match,
      const F2& do_if_match,
      const F3& do_if_anytype,
      bool is_
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/frontend`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/frontend`):

- [`strtod.h_kw.md_docs.md`](./strtod.h_kw.md_docs.md)
- [`tree_views.cpp_docs.md_docs.md`](./tree_views.cpp_docs.md_docs.md)
- [`function_schema_parser.cpp_docs.md_docs.md`](./function_schema_parser.cpp_docs.md_docs.md)
- [`tree.h_kw.md_docs.md`](./tree.h_kw.md_docs.md)
- [`versioned_symbols.cpp_kw.md_docs.md`](./versioned_symbols.cpp_kw.md_docs.md)
- [`parser.cpp_kw.md_docs.md`](./parser.cpp_kw.md_docs.md)
- [`lexer.h_kw.md_docs.md`](./lexer.h_kw.md_docs.md)
- [`parser.cpp_docs.md_docs.md`](./parser.cpp_docs.md_docs.md)
- [`convert_to_ssa.h_docs.md_docs.md`](./convert_to_ssa.h_docs.md_docs.md)
- [`error_report.cpp_kw.md_docs.md`](./error_report.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ir_emitter.cpp_docs.md_docs.md`
- **Keyword Index**: `ir_emitter.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
