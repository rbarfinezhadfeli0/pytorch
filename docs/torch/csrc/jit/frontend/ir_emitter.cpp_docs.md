# Documentation: ir_emitter.cpp

## File Metadata
- **Path**: `torch/csrc/jit/frontend/ir_emitter.cpp`
- **Size**: 215832 bytes
- **Lines**: 5846
- **Extension**: .cpp
- **Type**: Regular file

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
      bool is_dict_constructor = false) {
    if (auto union_type_hint = (*refined_type_hint_ptr)->cast<UnionType>()) {
      // `candidate_types` holds all List types that were in the Union
      // annotation
      std::vector<TypePtr> candidate_types;

      std::copy_if(
          union_type_hint->containedTypes().begin(),
          union_type_hint->containedTypes().end(),
          std::back_inserter(candidate_types),
          [&](TypePtr type_ptr) { return type_match(type_ptr); });

      if (!is_dict_constructor && candidate_types.empty()) {
        throw(
            ErrorReport(src)
            << "Expected an Union type annotation "
            << "with an inner " << match_repr << " type, but got "
            << (*refined_type_hint_ptr)->repr_str());
      } else if (candidate_types.size() == 1) {
        // The Union only had a single type of the container we want to
        // match, so we can unconditionally refine it to that type
        (*refined_type_hint_ptr) = candidate_types[0];
      } else {
        // We can't refine the Union yet, since it contains multiple
        // types of the container we want to match, but we do at least
        // have a list of possiblee types (e.g. `Union[List[int],
        // List[str], float, str]` -> candidates={List[int], List[str]})
        (*all_candidates) = std::move(candidate_types);
      }
    } else if (
        auto optional_type_hint =
            (*refined_type_hint_ptr)->cast<OptionalType>()) {
      (*refined_type_hint_ptr) = optional_type_hint->getElementType();
    }

    // This case handles code like `dict([(x, y), (a, b)])` that would
    // otherwise fail the following error checks
    if (is_dict_constructor) {
      return;
    }

    // If we had any annotation that was NOT a Union that can hold more
    // than one type of the container we want to match
    if (all_candidates->empty()) {
      if (type_match(*refined_type_hint_ptr)) {
        do_if_match();
      } else if ((*refined_type_hint_ptr)->kind() == AnyType::Kind) {
        do_if_anytype();
      } else {
        throw(
            ErrorReport(src) << "Expected an annotation of type " << match_repr
                             << " but got " << type_hint->repr_str());
      }
    }
  }

  void refineAndSetListTypeHintFromCandidatesVector(
      const std::vector<TypePtr>& all_candidates,
      const TypePtr& type_hint,
      TypePtr* refined_type_hint_ptr,
      const TypePtr& unified_elem_type,
      const Expr& src) {
    TypePtr greatest_elem_type = nullptr;
    std::for_each(
        all_candidates.begin(),
        all_candidates.end(),
        [&](const TypePtr& candidate) {
          auto candidate_elem_type =
              candidate->expect<ListType>()->getElementType();
          if (unified_elem_type->isSubtypeOf(candidate_elem_type)) {
            if (!greatest_elem_type) {
              greatest_elem_type = candidate_elem_type;
            } else {
              greatest_elem_type =
                  *(unifyTypes(greatest_elem_type, candidate_elem_type));
            }
          }
        });
    if (!greatest_elem_type) {
      std::stringstream vector_repr;
      for (size_t i = 0; i < all_candidates.size(); ++i) {
        if (i > 0 && all_candidates.size() > 2) {
          vector_repr << ", ";
        }
        if (i != 0 && i == all_candidates.size() - 1) {
          vector_repr << " or ";
        }
        vector_repr << all_candidates[i]->repr_str();
      }
      throw(
          ErrorReport(src) << "Union type annotation `" << type_hint->repr_str()
                           << "` can hold " << vector_repr.str()
                           << ", but none of "
                           << "those types match the types of the given list "
                           << "elements, which were unified to "
                           << unified_elem_type->repr_str());
    } else {
      (*refined_type_hint_ptr) = ListType::create(greatest_elem_type);
      ;
    }
  }

  void refineAndSetDictTypeHintFromCandidatesVector(
      const std::vector<TypePtr>& all_candidates,
      const TypePtr& type_hint,
      TypePtr* refined_type_hint_ptr,
      const TypePtr& known_key_type,
      const TypePtr& known_value_type,
      const Expr& src) {
    TypePtr candidate_key_type = nullptr;
    TypePtr candidate_value_type = nullptr;
    TypePtr candidate = nullptr;

    for (const auto& current_candidate : all_candidates) {
      auto current_key_type =
          current_candidate->expect<DictType>()->getKeyType();
      auto current_value_type =
          current_candidate->expect<DictType>()->getValueType();

      if (known_key_type->isSubtypeOf(current_key_type) &&
          known_value_type->isSubtypeOf(current_value_type)) {
        if (!candidate ||
            (candidate_key_type->isSubtypeOf(current_key_type) &&
             candidate_value_type->isSubtypeOf(current_value_type))) {
          candidate_key_type = current_key_type;
          candidate_value_type = current_value_type;
          candidate = current_candidate;
        }
      }
    }

    if (!candidate) {
      std::stringstream vector_repr;
      for (size_t i = 0; i < all_candidates.size(); ++i) {
        if (i > 0 && all_candidates.size() > 2) {
          vector_repr << ", ";
        }
        if (i != 0 && i == all_candidates.size() - 1) {
          vector_repr << " or ";
        }
        vector_repr << all_candidates[i]->repr_str();
      }
      throw(
          ErrorReport(src) << "Union type annotation `" << type_hint->repr_str()
                           << "` can hold " << vector_repr.str()
                           << ", but none of "
                           << "those dict types can hold the types of the given"
                           << " keys and values, which were unified to Dict["
                           << known_key_type->repr_str() << ", "
                           << known_value_type->repr_str());
    } else {
      (*refined_type_hint_ptr) = candidate;
    }
  }

  Value* emitListComprehension(const ListComp& lc, const TypePtr& type_hint) {
    const auto loc = lc.range();
    const auto targets_list = List<Expr>::create(lc.range(), {lc.target()});
    const auto itrs = List<Expr>::create(lc.range(), {lc.iter()});

    // If there is no type hint, and this is emitted over an iterable that is
    // unrolled and of length 0, then we emit a List of tensors
    Value* list_value = graph->insertNode(graph->create(prim::ListConstruct, 1))
                            ->output()
                            ->setType(ListType::ofTensors());

    TypePtr refined_type_hint = type_hint;
    std::vector<TypePtr> all_candidates = {};

    if (refined_type_hint) {
      auto do_if_type_match = [&]() { list_value->setType(refined_type_hint); };

      auto type_match = [&](const TypePtr& t) {
        return t->isSubtypeOf(AnyListType::get());
      };

      refineAndSetUnionTypeHintOrPopulateCandidatesVector(
          type_hint,
          &refined_type_hint,
          &all_candidates,
          "List",
          lc,
          type_match,
          do_if_type_match,
          do_if_type_match);
    }

    bool seen_first_elem = false;

    // A list comprehension introduces its own scope
    Node* n =
        graph->insertNode(create(prim::ComprehensionScope, lc.range(), 0));
    auto* comprehension_block = n->addBlock();
    pushFrame(comprehension_block);
    WithInsertPoint guard(comprehension_block);
    auto emit_body = [&]() {
      Value* out = emitExpr(lc.elt());

      // If we didn't have a type annotation, the type of the list would
      // be set to `Tensor`. We don't want to unify this default type
      // with the actual elements in the list, so let the type begin as
      // the first element in the list
      if (!seen_first_elem) {
        list_value->setType(ListType::create(out->type()));
        seen_first_elem = true;
      }

      const auto elem_type_hint =
          refined_type_hint && refined_type_hint->kind() == ListType::Kind
          ? refined_type_hint->cast<ListType>()->getElementType()
          : nullptr;

      std::optional<TypePtr> unified_elem_type = unifyTypes(
          list_value->type()->expect<ListType>()->getElementType(),
          out->type(),
          /*default_to_union=*/true,
          elem_type_hint);

      // Case: The list comprehension generated heterogeneous values,
      // and we don't have a type hint to suggest that this is what the
      // user expected
      if (!type_hint && (*unified_elem_type)->isUnionType()) {
        TORCH_WARN(
            "List consists of heterogeneous types, which means",
            " that it has been typed as containing ",
            (*unified_elem_type)->repr_str(),
            ". To use any of the "
            "values in this List, it will be necessary to add an "
            "`assert isinstance` statement before first use to trigger "
            "type refinement. The first non-matching element was typed",
            " as ",
            out->type()->repr_str(),
            ", while the elements "
            " before it were ",
            list_value->type()
                ->expect<ListType>()
                ->getElementType()
                ->repr_str(),
            "\n",
            lc.range().str());
      }

      // Case: We had an annotation that we were able to narrow down to
      // a single ListType, but the most recently generated element in
      // the list comprehension doesn't match that annotation
      if (all_candidates.empty() && refined_type_hint &&
          !(*unified_elem_type)
               ->isSubtypeOf(*refined_type_hint->expectRef<ListType>()
                                  .getElementType())) {
        throw(
            ErrorReport(lc)
            << "List type annotation `" << refined_type_hint->repr_str()
            << "` did not match the types of the given list elements,"
            << " which were unified to " << (*unified_elem_type)->repr_str());
      }

      if (!all_candidates.empty()) {
        // If we had a Union type annotation that could hold more than
        // one different type of `List`
        refineAndSetListTypeHintFromCandidatesVector(
            all_candidates,
            type_hint,
            &refined_type_hint,
            *unified_elem_type,
            lc);
      } else if (!refined_type_hint) {
        refined_type_hint = ListType::create(*unified_elem_type);
      }

      list_value->setType(refined_type_hint);
      out->setType(refined_type_hint->expect<ListType>()->getElementType());

      NamedValue self = NamedValue(loc, "self", list_value);
      NamedValue input = NamedValue(loc, "", out);
      emitBuiltinCall(loc, *graph, aten::append, {input}, {}, self);
    };
    emitFor(targets_list, itrs, loc, emit_body);
    popFrame();
    return list_value;
  }

  Value* emitDictComprehension(const DictComp& dc, const TypePtr& type_hint) {
    const auto loc = dc.range();
    const auto targets_list = List<Expr>::create(dc.range(), {dc.target()});
    const auto itrs = List<Expr>::create(dc.range(), {dc.iter()});

    Value* dict_value =
        graph->insertNode(graph->create(prim::DictConstruct, 1))->output();

    // Set the default type to be Dict[str, Tensor]
    dict_value->setType(DictType::create(StringType::get(), TensorType::get()));

    TypePtr refined_type_hint = type_hint;
    TypePtr annotated_union_type =
        type_hint && type_hint->isUnionType() ? type_hint : nullptr;

    std::vector<TypePtr> all_candidates = {};

    if (refined_type_hint) {
      auto type_match = [&](const TypePtr& t) {
        return t->kind() == DictType::Kind;
      };

      auto do_if_match = [&]() { dict_value->setType(refined_type_hint); };

      refineAndSetUnionTypeHintOrPopulateCandidatesVector(
          type_hint,
          &refined_type_hint,
          &all_candidates,
          "Dict",
          dc,
          type_match,
          do_if_match,
          do_if_match);
    }

    TypePtr first_generated_key_type = nullptr;
    TypePtr first_generated_value_type = nullptr;

    // A dict comprehension introduces its own scope. No variable assigned
    // may leak into the rest of the graph
    Node* n =
        graph->insertNode(create(prim::ComprehensionScope, dc.range(), 0));
    auto* comprehension_block = n->addBlock();
    pushFrame(comprehension_block);
    WithInsertPoint guard(comprehension_block);
    auto emit_body = [&]() {
      auto k = emitExpr(dc.key());
      auto v = emitExpr(dc.value());

      // If we didn't have a type annotation, the type of the dict would
      // be set to `(str, Tensor)`. We don't want to unify this default
      // type with the actual elements in the dict, so let the type
      // begin as the first element in the dict
      if (k->type()->kind() == UnionType::Kind) {
        throw(
            ErrorReport(dc)
            << "Dicts may only contain homogeneous keys, but the type of "
            << "the first generated key was " << k->type()->repr_str());
      } else if (
          first_generated_key_type && first_generated_key_type != k->type()) {
        // Values can be heterogeneous, so we only need to check that the
        // key types are all the same
        throw(
            ErrorReport(dc)
            << "Dicts may only contain homogeneous keys. Expected "
            << "dict comprehension to generate type "
            << first_generated_key_type->repr_str() << ", but got "
            << k->type()->repr_str());
      } else {
        dict_value->setType(DictType::create(k->type(), v->type()));
        first_generated_key_type = k->type();
        first_generated_value_type = v->type();
      }

      // If we had any annotation OTHER THAN a Union that can hold more
      // than one type of Dict
      if (refined_type_hint && all_candidates.empty()) {
        DictTypePtr dict_type_hint = refined_type_hint->expect<DictType>();

        std::stringstream ss;
        std::stringstream err;

        bool is_key_subtype =
            k->type()->isSubtypeOfExt(*dict_type_hint->getKeyType(), &ss);

        if (!is_key_subtype) {
          err << "Dict type annotation `" << dict_type_hint->repr_str()
              << "` did not match the "
              << "type of an actual key type `" << k->type()->repr_str()
              << "`\n"
              << ss.str();
        }

        ss.str(std::string());
        bool is_value_subtype =
            v->type()->isSubtypeOfExt(*dict_type_hint->getValueType(), &ss);

        if (!is_value_subtype) {
          err << "Dict type annotation `" << dict_type_hint->repr_str()
              << "` did not match the "
              << "type of an actual value type `" << v->type()->repr_str()
              << "`\n"
              << ss.str();
        }

        if (!is_key_subtype || !is_value_subtype) {
          throw(ErrorReport(dc) << err.str());
        }
      }

      const TypePtr value_type_hint =
          refined_type_hint && refined_type_hint->kind() == DictType::Kind
          ? refined_type_hint->expect<DictType>()->getValueType()
          : nullptr;

      std::optional<TypePtr> unified_value_type = unifyTypes(
          first_generated_value_type,
          v->type(),
          /*default_to_union=*/true,
          value_type_hint);

      if (!type_hint && (*unified_value_type)->isUnionType()) {
        TORCH_WARN(
            "Dict values consist of heterogeneous types, which means",
            " that they have been typed as being ",
            (*unified_value_type)->repr_str(),
            ". To use any of the "
            "values in this dict, it will be necessary to add an "
            "`assert isinstance` statement before first use to trigger "
            "type refinement. The first non-matching element was typed",
            " as ",
            v->type()->repr_str(),
            ", while the elements "
            " before it were ",
            first_generated_value_type->repr_str(),
            "\n",
            dc.range().str());
      }

      if (type_hint) {
        if (type_hint->kind() == DictType::Kind) {
          dict_value->setType(type_hint);
          k->setType(type_hint->expect<DictType>()->getKeyType());
          v->setType(type_hint->expect<DictType>()->getValueType());
        } else {
          if (!all_candidates.empty()) {
            refineAndSetDictTypeHintFromCandidatesVector(
                all_candidates,
                type_hint,
                &refined_type_hint,
                k->type(),
                *unified_value_type,
                dc);
          }
          dict_value->setType(refined_type_hint);
          k->setType(refined_type_hint->expect<DictType>()->getKeyType());
          v->setType(refined_type_hint->expect<DictType>()->getValueType());
        }
      } else {
        dict_value->setType(DictType::create(k->type(), *unified_value_type));
      }

      NamedValue self = NamedValue(loc, "self", dict_value);
      NamedValue input_k = NamedValue(loc, "", k);
      NamedValue input_v = NamedValue(loc, "", v);
      emitBuiltinCall(
          loc, *graph, aten::_set_item, {self, input_k, input_v}, {});
    };
    emitFor(targets_list, itrs, loc, emit_body);
    popFrame();

    if (annotated_union_type) {
      Node* n =
          graph->insertNode(graph->create(prim::unchecked_cast, {dict_value}));
      n->output()->setType(std::move(annotated_union_type));
      dict_value = n->output();
    }

    return dict_value;
  }

  // Insert subtyping refinements
  void insertRefinements(const SourceRange& loc, const RefinementSet& ref) {
    for (const Refinement& r : ref.activeRefinements()) {
      Value* v = environment_stack->getVar(r.identifier(), loc);
      Value* new_v = graph->insertUncheckedCast(v, r.type());
      environment_stack->setVar(loc, r.identifier(), new_v);
    }
  }

  CondValue emitShortCircuitLogical(
      const SourceRange& loc,
      const Expr& first_expr,
      const Expr& second_expr,
      bool is_or) {
    CondValue lhs = emitCondExpr(first_expr);
    // if the continue expr in the short circuit is not evaluated,
    // than the const expression is False if the short circuit
    // is an `and` and True if the short circuit is an `or`.
    // `False and expr` -> False, `True or expr` -> True
    //
    // inserting it as a constant makes optimization easier

    // if it's an OR the first expr is emitted in the true branch
    // and the second expr in the false branch, if it's an AND the opposite
    auto get_const_expr = [&] { return graph->insertConstant(is_or, loc); };

    std::optional<CondValue> rhs;
    auto get_continue_expr = [&] {
      rhs = emitCondExpr(second_expr);
      return rhs->value();
    };

    // if this is an OR, eval second expression if first expr is False
    // If this is an AND, eval second expression if first expr is True
    Value* new_result = nullptr;
    std::optional<RefinementSet> refinements;
    std::optional<bool> static_if;
    if (is_or) {
      new_result = emitIfExpr(loc, lhs, get_const_expr, get_continue_expr);
      refinements = lhs.refinements().Or(rhs->refinements());
      if ((lhs.staticIf() && *lhs.staticIf()) ||
          (rhs->staticIf() && *rhs->staticIf())) {
        static_if = true;
      } else if (lhs.staticIf() && rhs->staticIf()) {
        static_if = *lhs.staticIf() || *rhs->staticIf();
      }
    } else {
      new_result = emitIfExpr(loc, lhs, get_continue_expr, get_const_expr);
      refinements = lhs.refinements().And(rhs->refinements());
      if (((lhs.staticIf() && !*lhs.staticIf()) ||
           (rhs->staticIf() && !*rhs->staticIf()))) {
        static_if = false;
      } else if (lhs.staticIf() && rhs->staticIf()) {
        static_if = *lhs.staticIf() && *rhs->staticIf();
      }
    }
    return CondValue(new_result, std::move(*refinements), static_if);
  }

  Value* emitIfExpr(
      const SourceRange& range,
      const CondValue& cond_value,
      const std::function<Value*()>& true_expr,
      const std::function<Value*()>& false_expr) {
    Node* n = graph->insertNode(create(prim::If, range, 0));
    n->addInput(cond_value.value());
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    auto emit_if_expr = [this, &range](
                            Block* b,
                            const RefinementSet& refinements,
                            const std::function<Value*()>& expr_value) {
      pushFrame(b);
      WithInsertPoint guard(b);
      insertRefinements(range, refinements);
      Value* out_val = expr_value();
      b->registerOutput(out_val);
      popFrame();
    };

    emit_if_expr(true_block, cond_value.refinements(), true_expr);
    emit_if_expr(false_block, cond_value.refinements().Not(), false_expr);

    auto true_type = true_block->outputs().at(0)->type();
    auto false_type = false_block->outputs().at(0)->type();
    auto unified = unifyTypes(true_type, false_type);
    if (!unified) {
      throw(
          ErrorReport(range)
          << "if-expression's true branch has type " << true_type->repr_str()
          << " but false branch has type " << false_type->repr_str());
    }

    // Add op outputs
    auto expr_value = n->addOutput()->setType(*unified); // Resulting value

    return expr_value;
  }
  Value* emitToBool(const SourceRange& loc, Value* v) {
    Value* out = nullptr;
    try {
      auto bool_cast = environment_stack->getSugaredVar("bool", loc);
      out = asSimple(bool_cast->call(loc, method, {v}, {}, 0));
    } catch (...) {
      throw(
          ErrorReport(loc) << "Could not cast value of type "
                           << v->type()->repr_str() << " to bool");
    }
    if (!out) {
      throw(
          ErrorReport(loc) << "Could not cast value of type "
                           << v->type()->repr_str() << " to bool");
    }
    // cast value not response for checking output type
    if (!out->type()->isSubtypeOf(*BoolType::get())) {
      throw(
          ErrorReport(loc)
          << "expected a bool expression for condition but found "
          << out->type()->repr_str());
    }
    return out;
  }

  void emitIfElseBlocks(
      const SourceRange& loc,
      const CondValue& cond_value,
      const List<Stmt>& trueBranch,
      const List<Stmt>& falseBranch) {
    // this is a static if statement: that is, it contains a subset
    // of operators where we are willing to specialize the if statement
    // to be only the true or false branch when the condition is statically
    // known. This is used to meta-program modules, for instance, when a
    // submodule is absent, an is None check can be used to ensure the
    // accesses to the None check, which would error, are not compiled.
    if (cond_value.staticIf()) {
      if (*cond_value.staticIf()) {
        insertRefinements(loc, cond_value.refinements());
        emitStatements(trueBranch);
      } else {
        insertRefinements(loc, cond_value.refinements().Not());
        emitStatements(falseBranch);
      }
      return;
    }

    Node* n = graph->insertNode(create(prim::If, loc, 0));
    n->addInput(cond_value.value());
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    // Emit both blocks once to get the union of all mutated values
    auto save_true =
        emitSingleIfBranch(true_block, trueBranch, cond_value.refinements());
    auto save_false = emitSingleIfBranch(
        false_block, falseBranch, cond_value.refinements().Not());

    bool true_exits = exit_blocks.count(true_block);
    bool false_exits = exit_blocks.count(false_block);
    if (true_exits && false_exits) {
      exit_blocks.insert(n->owningBlock());
    }

    // In python, every variable assigned in an if statement escapes
    // the scope of the if statement (all variables are scoped to the function).
    // Script is a subset of python: we consider variables to be in scope
    // as long as there is a definition of the variable along all paths
    // through the if statement
    // ----
    // if ...:
    //   a =
    // else:
    //   ...
    // ... = a  # error, a is not defined along all paths
    // ----
    // if ...:
    //   a =
    // else:
    //   a =
    // ... = a # OK, a is defined along all paths
    // ----
    // a = ...
    // if ...:
    //   a =
    // ... = a # OK, a is defined along all paths
    // if ...:
    //   a =
    // else:
    //   return
    // ... = a # OK, a is always defined

    // ordered set, because we want deterministic graph output
    std::set<std::string> mutated_variables;

    // When we access either the true or false environment,
    // we need to set the insertion point so the prim::Load is inserted
    // into the right block.
    // if var is only defined in one branch save error in case it's used later
    for (auto& v : save_true->definedVariables()) {
      {
        WithInsertPoint insert(false_block);
        if (save_false->findInAnyFrame(v) || false_exits) {
          mutated_variables.insert(v);
        } else {
          if (reportSourceLocation(loc.source()->size())) {
            ErrorReport error(loc);
            environment_stack->setVariableTypeError(v, [=]() -> std::string {
              error << v << " is not defined in the false branch";
              return error.what();
            });
          } else {
            environment_stack->setVariableTypeError(v, [=]() -> std::string {
              std::stringstream ss;
              ss << v << " is not defined in the false branch. "
                 << "The source info is eliminated due to the source file is too large. "
                 << "To get it back, please set PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION=1 "
                 << "as env var";
              return ss.str();
            });
          }
        }
      }
    }
    for (auto& v : save_false->definedVariables()) {
      {
        WithInsertPoint insert(true_block);
        if (save_true->findInAnyFrame(v) || true_exits) {
          mutated_variables.insert(v);
        } else {
          if (reportSourceLocation(loc.source()->size())) {
            ErrorReport error(loc);
            environment_stack->setVariableTypeError(v, [=]() -> std::string {
              error << v << " is not defined in the true branch";
              return error.what();
            });
          } else {
            environment_stack->setVariableTypeError(v, [=]() -> std::string {
              std::stringstream ss;
              ss << v << " is not defined in the false branch. "
                 << "The source info is eliminated due to the source file is too large. "
                 << "To get it back, please set PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION=1 "
                 << "as env var";
              return ss.str();
            });
          }
        }
      }
    }

    // Register outputs in each block
    for (const auto& x : mutated_variables) {
      Value* tv = nullptr;
      Value* fv = nullptr;

      {
        WithInsertPoint insert(true_block);
        if (!true_exits) {
          tv = save_true->getVar(x, loc);
        }
      }
      {
        WithInsertPoint insert(false_block);
        if (!false_exits) {
          fv = save_false->getVar(x, loc);
        }
      }

      // if both branches exit don't emit any variables
      // if one branch exits then we allow the all variables in the other branch
      // to escape scope since they are well-defined
      if (true_exits && false_exits) {
        continue;
      } else if (true_exits) {
        tv = graph->createUninitialized(fv->type())
                 ->insertBefore(true_block->return_node())
                 ->output();
        graph->createStore(x, tv)->insertBefore(true_block->return_node());
      } else if (false_exits) {
        fv = graph->createUninitialized(tv->type())
                 ->insertBefore(false_block->return_node())
                 ->output();
        graph->createStore(x, fv)->insertBefore(false_block->return_node());
      }

      SugaredValuePtr maybe_sugared_x = environment_stack->findInAnyFrame(x);
      TypePtr full_type = nullptr;
      if (maybe_sugared_x) {
        Value* maybe_simple = asSimple(maybe_sugared_x);
        if (maybe_simple) {
          full_type = maybe_simple->type();
        }
      }

      // Try to unify the types. If we found a type annotation earlier
      // in the environment, and if that type annotation is some form
      // of union, then we need to tell `unifyTypes` not to throw an
      // error if the branched return types we found are heterogeneous
      bool default_to_union = full_type &&
          (full_type->kind() == UnionType::Kind ||
           full_type->kind() == OptionalType::Kind ||
           full_type->kind() == NumberType::Kind);
      auto unified = unifyTypes(
          tv->type(), fv->type(), /*default_to_union=*/default_to_union);

      // We allow variables to be set to different types in each branch
      // as long as that variable is not already in scope or if that
      // variable does not get used later. Here, we save the error so
      // that the error message will be more informative in the case
      // that is used later. When `a` is accessed in `(a + 1)`, the
      // error will get printed:
      // if cond:
      //    a = 1
      // else:
      //    a = tensor
      // b = a + 1
      //
      if (!unified) {
        ErrorReport error(loc);
        error << "Type mismatch: " << x << " is set to type "
              << tv->type()->repr_str() << " in the true branch"
              << " and type " << fv->type()->repr_str()
              << " in the false branch";
        if (save_true->findInParentFrame(x) ||
            save_false->findInParentFrame(x)) {
          throw ErrorReport(error);
        } else {
          environment_stack->setVariableTypeError(
              x, [=]() -> std::string { return error.what(); });
          continue;
        }
      }
      environment_stack->setType(x, *unified);
    }
  }

  CondValue emitHasAttr(const Expr& objExpr, const Expr& attrExpr) {
    auto obj = emitSugaredExpr(objExpr, 1);
    if (attrExpr.kind() != TK_STRINGLITERAL) {
      throw(
          ErrorReport(attrExpr)
          << "hasattr's second argument must be a string literal");
    }
    const std::string& name = StringLiteral(attrExpr).text();
    const bool hasAttr = obj->hasAttr(objExpr.range(), method, name);
    return CondValue(*graph, objExpr.range(), hasAttr, {});
  }

  CondValue emitIsInstance(const Expr& obj, const Expr& classinfo) {
    Value* lhs_val = emitExpr(obj);
    std::vector<TypePtr> lhs_types;
    std::vector<TypePtr> rhs_types;

    std::function<void(const Expr&)> gather_rhs = [&](const Expr& expr) {
      if (expr.kind() == TK_TUPLE_LITERAL) {
        for (Expr e : TupleLiteral(expr).inputs()) {
          gather_rhs(e);
        }
        return;
      }
      TypePtr type = typeParser_.parseTypeFromExpr(expr);
      rhs_types.emplace_back(type);
    };

    lhs_types.push_back(lhs_val->type());
    gather_rhs(classinfo);

    standardizeVectorForUnion(&lhs_types);
    standardizeVectorForUnion(&rhs_types);

    RefinementSet refinement;

    TypePtr unified_true = nullptr;
    TypePtr unified_false = nullptr;

    std::vector<TypePtr> isinstance_types;
    std::vector<TypePtr> not_isinstance_types;

    std::vector<Refinement> true_refinements;
    std::vector<Refinement> false_refinements;

    bool all_lhs_subtype_some_rhs = true;

    // We can discard any rhs types that we know statically would be
    // impossible. For example, if we had:
    //
    //    def fn(x: Optional[str]):
    //        if isinstance(x, (List[str], str, int)):
    //            ...
    //
    // then `x` would be `str` in the true branch and `None` in the
    // false branch, not `(List[str], str, int)` in the true branch
    // and `None` in the false branch
    for (const TypePtr& lhs_type : lhs_types) {
      if (lhs_type == AnyType::get()) {
        isinstance_types.insert(
            isinstance_types.end(), rhs_types.begin(), rhs_types.end());
        not_isinstance_types.emplace_back(AnyType::get());
        // Edge case: we can still say that all lhs types subtype some
        // rhs type if `lhs` is `Any` and `rhs` is `Any`
        if (isinstance_types.size() != 1 ||
            isinstance_types[0] != AnyType::get()) {
          all_lhs_subtype_some_rhs = false;
        }
        break;
      }

      auto get_smaller_type = [&](const TypePtr& t1,
                                  const TypePtr& t2) -> TypePtr {
        if (t1->isSubtypeOf(*t2)) {
          return t1;
        } else if (t2->isSubtypeOf(*t1)) {
          return t2;
        } else {
          return nullptr;
        }
      };

      TypePtr found_refinement = nullptr;
      for (const TypePtr& rhs_type : rhs_types) {
        TypePtr maybe_smaller_type = get_smaller_type(lhs_type, rhs_type);
        if (!maybe_smaller_type) {
          continue;
        } else if (*maybe_smaller_type == *lhs_type) {
          // Cover the case that we have something like
          // lhs = `List[str]` and rhs = `list`
          found_refinement = lhs_type;
        } else if (*maybe_smaller_type == *rhs_type) {
          // We want the narrowest possible type
          found_refinement = found_refinement
              ? *(unifyTypes(found_refinement, rhs_type))
              : rhs_type;
        }
      }

      if (found_refinement) {
        if (*found_refinement == *lhs_type) {
          all_lhs_subtype_some_rhs &= true;
        }
        isinstance_types.push_back(found_refinement);
      } else {
        // If the lhs couldn't be a subtype of the rhs (or couldn't
        // be "refined" to itself, as in the `List[str]` and `list`
        // case above), then we add `lhs_type` to the false branch
        // refinements. This is because the type can still be itself
        // if the `isinstance` check is false
        not_isinstance_types.push_back(lhs_type);
        all_lhs_subtype_some_rhs = false;
      }
    }

    // For use with `unifyTypeList`
    std::stringstream nowhere;

    // Get a single type for the true and false branches
    if (!isinstance_types.empty()) {
      unified_true =
          *unifyTypeList(isinstance_types, nowhere, /*default_to_union=*/true);
    }
    if (obj.kind() == TK_VAR && unified_true) {
      std::string ident = Var(obj).name().name();
      true_refinements = {Refinement(ident, unified_true)};
    }

    // Get a single type for the true and false branches
    if (!not_isinstance_types.empty()) {
      unified_false = *unifyTypeList(
          not_isinstance_types, nowhere, /*default_to_union=*/true);
    }
    if (obj.kind() == TK_VAR && unified_false) {
      std::string ident = Var(obj).name().name();
      false_refinements = {Refinement(ident, unified_false)};
    }

    refinement = RefinementSet(true_refinements, false_refinements);

    bool is_statically_false = isinstance_types.empty();

    // If the statement is statically true
    if (all_lhs_subtype_some_rhs) {
      return CondValue(*graph, obj.range(), true, std::move(refinement));
    }

    if (is_statically_false) {
      return CondValue(*graph, obj.range(), false, std::move(refinement));
    }

    // check maybe true/false at runtime, need an actual op
    Value* result =
        graph->insertNode(graph->createIsInstance(lhs_val, rhs_types))
            ->output();
    return CondValue(result, std::move(refinement), std::nullopt);
  }

  void emitIf(const If& stmt) {
    Expr cond = stmt.cond();
    CondValue cond_value = emitCondExpr(cond);
    emitIfElseBlocks(
        stmt.range(), cond_value, stmt.trueBranch(), stmt.falseBranch());
  }

  // *********************** Loop Operators ************************************
  // Emits a loop operator with the form:
  // Loop(max_trip_count)
  // block0(loop_counter) {
  //   <body>
  // }
  // block1 {
  //   <loop condition>
  //   -> (condition)
  // }
  // For loops will have an empty loop condition block with condition set to
  // true. In the convert to ssa pass, the loop condition will correctly
  // inlined. and inputs and outputs added so that the loop conforms to the
  // semantics specified at
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop
  void emitLoopCommon(
      const SourceRange& range,
      const std::function<void()>& emit_body,
      const SugaredValuePtr& iter_val,
      std::optional<List<Expr>> targets,
      std::optional<Expr> cond) {
    Value* max_trip_count_val = nullptr;
    if (iter_val != nullptr) {
      max_trip_count_val = iter_val->len(range, method);
    } else {
      max_trip_count_val = materializeConstant(
          std::numeric_limits<int64_t>::max(),
          *graph,
          range,
          integral_constants);
    }

    Node* n = graph->insertNode(create(prim::Loop, range, 0));
    auto* body_block = n->addBlock();
    {
      Block* condition_block = n->addBlock();
      pushFrame(condition_block);
      Value* out = nullptr;
      if (cond) {
        WithInsertPoint insert(condition_block);
        out = emitToBool(cond.value().range(), emitExpr(cond.value()));
      } else {
        WithInsertPoint insert(n);
        out = graph->insertConstant(true, range);
      }
      condition_block->registerOutput(out);
      popFrame();
    }
    n->addInput(max_trip_count_val);

    WithLoopStatus loop_guard(&loop_status_, LoopStatus::IN_LOOP);
    Value* trip_count =
        body_block->addInput()->setType(IntType::get()); // Iteration num
    {
      pushFrame(body_block);
      WithInsertPoint guard(body_block);

      // if the FOR iters and targets are present, emit FOR target assignments
      if (iter_val != nullptr && targets) {
        Value* cur_elem = iter_val->getitem(range, method, trip_count)
                              ->asValue(range, method);
        SugaredValuePtr sv = std::make_shared<SimpleValue>(cur_elem);
        List<Expr> target_exprs = targets.value();
        validateAssignLhsExpr(target_exprs, range);

        // if target exprs are more than 1, it means iteration unpacking on LHS
        // we create Tuple literal to wrap those target exprs for assignments
        if (target_exprs.size() > 1) {
          Expr tl = TupleLiteral::create(range, target_exprs);
          target_exprs = List<Expr>::create(range, {tl});
        }
        emitExprsAssign(target_exprs, {sv}, range, /*n_binders=*/1);
      }
      emit_body();
      popFrame();
    }
  }

  void emitUnrolledLoop(
      const SourceRange& loc,
      const std::function<void()>& emit_body,
      const SugaredValuePtr& iterable,
      const List<Expr>& targets) {
    auto static_len = iterable->staticLen();
    TORCH_INTERNAL_ASSERT(
        static_len, "Unrolled loop iter should have static length");
    int64_t len = *static_len;
    WithLoopStatus loop_guard(&loop_status_, LoopStatus::IN_UNROLLED_LOOP);
    // In order to support ModuleLists which return different types,
    // as with an nn.Sequential which has a module that returns a Dict and then
    // a module which returns a Tensor,
    // we do not push a new environment frame because if we did all intermediary
    // values would have to subtype the input type.
    for (const auto i : c10::irange(len)) {
      auto index =
          materializeConstant(i, *method.graph(), loc, integral_constants);
      auto sugared_value = iterable->getitem(loc, method, index);
      emitExprsAssign(
          targets, {sugared_value}, targets.range(), /*n_binders=*/1);
      emit_body();
    }
  }

  void emitFor(
      const List<Expr>& targets,
      const List<Expr>& itrs,
      const SourceRange& loc,
      const std::function<void()>& emit_body) {
    if (itrs.size() != 1) {
      throw(ErrorReport(loc) << "List of iterables is not supported currently");
    }

    // Emit loop information for builtinFunction values like range(), zip(),
    // enumerate() or SimpleValue like List, Tensor, Dict, etc.
    SugaredValuePtr sv = emitSugaredExpr(itrs[0], 1);
    SugaredValuePtr iterable = sv->iter(loc, method);

    // We unroll the loop for iterables that contain ModuleLists so that we can
    // compile Heterogeneous module lists.
    if (!iterable->shouldEmitUnrolled()) {
      emitLoopCommon(loc, emit_body, iterable, targets, {});
    } else {
      emitUnrolledLoop(loc, emit_body, iterable, targets);
    }
  }

  void emitFor(const For& stmt) {
    auto emit_body = [&]() { emitStatements(stmt.body()); };
    emitFor(stmt.targets(), stmt.itrs(), stmt.range(), emit_body);
  }

  void emitWhile(const While& stmt) {
    auto cond = stmt.cond();
    auto emit_body = [&]() { emitStatements(stmt.body()); };
    emitLoopCommon(stmt.range(), emit_body, nullptr, {}, cond);
  }

  void emitWith(const With& stmt) {
    auto targets = stmt.targets();
    // Keep a stack of entered objects so they can be exited
    // in the right order.
    std::stack<Value*> entered;

    for (const auto& target : targets) {
      Expr e = target.target();

      auto* rhs = emitExpr(e);
      auto* n = graph->insertNode(graph->create(prim::Enter, {rhs}));
      entered.push(rhs);

      if (rhs->type()->kind() != TypeKind::ClassType) {
        throw(
            ErrorReport(e.range())
            << "With item expression must return an object");
      }

      auto rhsClass = rhs->type()->expect<ClassType>();
      auto* enterMethod = rhsClass->findMethod("__enter__");
      auto* exitMethod = rhsClass->findMethod("__exit__");

      if (!enterMethod || !exitMethod) {
        throw(
            ErrorReport(e.range())
            << "Object returned by with item expression does not define __enter__ and __exit__ methods");
      }

      // Check the schema of __enter__.
      auto& enterSchema = enterMethod->getSchema();
      if (enterSchema.arguments().size() != 1) {
        throw(
            ErrorReport(e.range())
            << "__enter__ must have only one argument and one return value");
      }

      // Check the schema of __exit__.
      auto& exitSchema = exitMethod->getSchema();
      if (exitSchema.arguments().size() != 4) {
        throw(ErrorReport(e.range()) << "__exit__ must have four arguments");
      } else {
        for (unsigned i = 1; i < 4; ++i) {
          if (exitSchema.arguments().at(i).type() != AnyType::get()) {
            throw(
                ErrorReport(e.range())
                << "argument " << i
                << " of __exit__ must have Any type; TorchScript does not currently support passing exception type, value, or traceback to the __exit__ function.");
          }
        }
      }

      // Set the output of the enter node to be the return type of __enter__.
      n->output(0)->setType(enterSchema.returns().at(0).type());

      // Set i = e.__enter__() so that references to i in the body of the with
      // will resolve correctly.
      if (target.var().present()) {
        Var i = target.var().get();
        environment_stack->setVar(i.range(), i.name().name(), n->output(0));
      }
    }

    emitStatements(stmt.body());

    // Insert all the corresponding prim::Exit nodes.
    while (!entered.empty()) {
      auto* input = entered.top();
      entered.pop();
      auto* n = graph->create(prim::Exit);
      graph->insertNode(n);
      n->addInput(input);
    }
  }

  // Currently we do not support assigning exceptions to variables,
  // a = Exception("hi")
  // raise a
  //
  // We ignore the expression following raise
  void emitRaise(const Raise& raise) {
    auto sv = emitSugaredExpr(raise.expr(), 1);
    Value* error_message = nullptr;
    Value* qualified_class_name = nullptr;

    if (auto exception_instance =
            std::dynamic_pointer_cast<ExceptionMessageValue>(sv)) {
      // The typical case, an instance of the exception class was thrown:
      //    raise RuntimeError("error")
      error_message = exception_instance->getValue();
      qualified_class_name = exception_instance->getQualifiedClassName();
    } else if (
        auto exception_class = std::dynamic_pointer_cast<ExceptionValue>(sv)) {
      // A bare exception was thrown so add an empty message. e.g.
      //    raise RuntimeError
      error_message = insertConstant(*graph, "", raise.range());
    } else {
      // The raise was not followed by an exception (i.e. it was something like
      // `raise "error"` instead of `raise RuntimeError("error")`)
      throw(
          ErrorReport(raise.range())
          << "exceptions must derive from BaseException");
    }

    if (!error_message->type()->isSubtypeOf(*StringType::get())) {
      error_message = graph->insert(aten::str, {error_message});
    }

    graph->insert(
        prim::RaiseException,
        {error_message, qualified_class_name},
        {},
        raise.range());
    exit_blocks.insert(environment_stack->block());
  }

  // emit assserions as an if branch so that assertions will reuse the
  // message
  void emitAssert(const Assert& stmt) {
    CondValue cond_value = emitCondExpr(stmt.test());
    List<Stmt> true_branch = List<Stmt>::create(stmt.range(), {});
    // Create an `AssertionError("the_message")` call
    auto message = (stmt.msg().present())
        ? stmt.msg().get()
        : StringLiteral::create(stmt.range(), "");
    auto callee = Var::create(
        stmt.range(), Ident::create(stmt.range(), "AssertionError"));
    auto apply = Apply::create(
        stmt.range(),
        callee,
        List<Expr>::create(stmt.range(), {message}),
        List<Attribute>::create(stmt.range(), {}));

    List<Stmt> false_branch =
        List<Stmt>::create(stmt.range(), {Raise::create(stmt.range(), apply)});
    emitIfElseBlocks(stmt.range(), cond_value, true_branch, false_branch);
  }

  // Validate that the `lhs` Expr's in an assignment statement are valid. That
  // is:
  //
  // 1) All lhs Expr's are either Var, Tuple or Starred nodes
  // 2) There is at most one Starred node in the lhs Expr
  // 3) A Starred node can only appear when there is another non-Starred lhs
  //    Expr. Concretely this means that `*abc = func()` is illegal. Unpacking
  //    all outputs into a tuple is covered by `abc = func()`.
  bool validateAssignLhsExpr(const List<Expr>& lhs, const SourceRange& r) {
    size_t num_normal_assign = 0;
    size_t num_starred = 0;
    for (const auto& assignee : lhs) {
      if (assignee.kind() == TK_VAR || assignee.kind() == TK_SUBSCRIPT ||
          assignee.kind() == TK_TUPLE_LITERAL || assignee.kind() == '.') {
        num_normal_assign++;
      } else if (assignee.kind() == TK_STARRED) {
        num_starred++;
      } else {
        throw(
            ErrorReport(assignee) << "lhs of assignment must be a variable, "
                                  << "subscript, or starred expression");
      }
    }

    if (num_starred > 1) {
      throw(
          ErrorReport(r)
          << "Only one starred expression is allowed on the lhs");
    }

    if (num_starred > 0 && num_normal_assign == 0) {
      throw(
          ErrorReport(r) << "A Starred expression may only appear on the "
                         << "lhs within the presence of another non-starred"
                         << " expression");
    }

    return num_starred;
  }

  // Get the appropriate builtin op for this augmented assignment
  // If the RHS is a tensor, return the corresponding ATen in-place op
  // If it's a list of scalars, then return the corresponding list augment op
  Symbol getAugOp(const AugAssign& stmt, const TypePtr& type) {
    bool use_inplace_op = type->isSubtypeOf(*TensorType::get()) ||
        type->kind() == TypeKind::ListType;
    switch (stmt.aug_op()) {
      case '+':
        return use_inplace_op ? aten::add_ : aten::add;
      case '-':
        return use_inplace_op ? aten::sub_ : aten::sub;
      case '/':
        return use_inplace_op ? aten::div_ : aten::div;
      case '*':
        return use_inplace_op ? aten::mul_ : aten::mul;
      case '%':
        return use_inplace_op ? aten::fmod_ : aten::fmod;
      case '|':
        return use_inplace_op ? aten::bitwise_or : aten::__or__;
      case '&':
        return use_inplace_op ? aten::bitwise_and : aten::__and__;
      case '^':
        return use_inplace_op ? aten::bitwise_xor : aten::__xor__;
      case TK_LSHIFT:
        return use_inplace_op ? aten::__ilshift__ : aten::__lshift__;
      case TK_RSHIFT:
        return use_inplace_op ? aten::__irshift__ : aten::__rshift__;
      case TK_POW:
        return aten::pow;
      default:
        throw(
            ErrorReport(stmt)
            << "Unknown augmented assignment: " << kindToString(stmt.aug_op()));
    }
  }

  // Get a pair of <in place magic method name, out of place magic method name>
  // since the out of place method is called if the in place method is not
  // present
  std::pair<std::string, std::string> getAugMagicMethod(const AugAssign& stmt) {
    switch (stmt.aug_op()) {
      case '+':
        return std::make_pair(std::string("__iadd__"), std::string("__add__"));
      case '-':
        return std::make_pair(std::string("__isub__"), std::string("__sub__"));
      case '/':
        return std::make_pair(
            std::string("__itruediv__"), std::string("__truediv__"));
      case '*':
        return std::make_pair(std::string("__imul__"), std::string("__mul__"));
      case '%':
        return std::make_pair(std::string("__imod__"), std::string("__mod__"));
      default:
        throw(
            ErrorReport(stmt)
            << "Unknown augmented assignment: " << kindToString(stmt.aug_op()));
    }
  }

  // Emit nodes for augmented assignments like `+=`
  void emitAugAssignment(const AugAssign& stmt) {
    switch (stmt.lhs().kind()) {
      case TK_VAR: {
        emitAugAssignmentToVar(stmt);
      } break;
      case '.': {
        emitAugAssignmentToSelectVar(stmt);
      } break;
      case TK_SUBSCRIPT: {
        emitAugAssignmentToSubscript(stmt);
      } break;
      default:
        throw(
            ErrorReport(stmt.lhs())
            << "unexpected expression on "
            << "left-hand side of augmented assignment");
    }
  }

  // This will be called when there is a class param or module buffer
  // mutation which make the LHS of the expr be a select expression
  //
  

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 25 class(es): and, value, value, in, value, values, value, values, T, Hash, LoopStatus, instance, F1, F2, F3, was, param, A, types, and

### Structures
This file defines 11 struct(s): Refinement, RefinementSet, CondValue, Environment, DefContext, WithLoopStatus, to_ir, the, kwargs, FunctionResolver, CompilationUnit


## Key Components

The file contains 20701 words across 5846 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 215832 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
