# Documentation: `torch/csrc/jit/passes/shape_analysis.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/shape_analysis.cpp`
- **Size**: 88,170 bytes (86.10 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/shape_analysis.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/utils/op_registry.h>
#include <torch/csrc/jit/runtime/exception_message.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/symbol.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_strided.h>
#endif

#include <exception>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace torch::jit {

bool mergeTypes(
    ArrayRef<Value*> lhs,
    ArrayRef<Value*> rhs,
    ArrayRef<Value*> outputs) {
  AT_ASSERT(lhs.size() == rhs.size() && rhs.size() == outputs.size());
  bool changed = false;
  for (const auto i : c10::irange(lhs.size())) {
    auto old_output_type = outputs[i]->type();
    auto new_type =
        unifyTypes(lhs[i]->type(), rhs[i]->type(), /*default_to_union=*/true);
    AT_ASSERT(new_type);
    outputs[i]->setType(*new_type);
    if (*old_output_type != *outputs[i]->type())
      changed = true;
  }
  return changed;
}

static void applyTypes(ArrayRef<Value*> src, ArrayRef<Value*> dst) {
  AT_ASSERT(src.size() == dst.size());
  for (const auto i : c10::irange(src.size())) {
    dst[i]->setType(src[i]->type());
  }
}

void PropertyPropBase::propagateBlock(Block* block, bool insert_expands) {
  for (Node* node : block->nodes()) {
    try {
      propagateNode(node, insert_expands);
    } catch (propagation_error& e) {
      setUnshapedType(node);
    } catch (std::exception& e) {
      throw(
          ErrorReport(node->sourceRange())
          << ExceptionMessage(e)
          << "\nThe above operation failed shape propagation in this context");
    }
  }
}

void PropertyPropBase::processIf(Node* node) {
  auto then_block = node->blocks().at(0);
  auto else_block = node->blocks().at(1);
  propagateBlock(then_block);
  propagateBlock(else_block);
  mergeTypes(then_block->outputs(), else_block->outputs(), node->outputs());
}

void PropertyPropBase::processLoop(Node* node) {
  LoopView loop(node);
  // propagate counter type
  loop.currentTripCount()->setType(loop.maxTripCount()->type());
  applyTypes(loop.carriedInputs(), loop.bodyCarriedInputs());

  do {
    propagateBlock(loop.bodyBlock(), /*insert_expands=*/false);
    // note: inserting expands is unsafe at this point, we don't know
    // if the types are stable yet, so the arguments to expand may change
  } while (mergeTypes(
      loop.bodyCarriedInputs(),
      loop.bodyCarriedOutputs(),
      loop.bodyCarriedInputs()));

  // now that the types are stable, we can insert the expands
  propagateBlock(loop.bodyBlock(), /*insert_expands=*/true);
  applyTypes(loop.bodyCarriedInputs(), loop.carriedOutputs());
}

void PropertyPropBase::setUnshapedType(Value* o) {
  o->setType(unshapedType(o->type()));
}

void PropertyPropBase::setUnshapedType(Node* node) {
  for (auto o : node->outputs()) {
    setUnshapedType(o);
  }
}

namespace prim {
using namespace ::c10::prim;
}

#define SHAPE_ASSERT(cond) \
  if (!(cond))             \
  throw propagation_error()

namespace {

bool isValidArgumentForRunning(Value* v) {
  // allow constants
  if (toIValue(v))
    return true;
  if (TensorTypePtr tt = v->type()->cast<TensorType>()) {
    if (!tt->scalarType()) {
      return false;
    }
    return !at::isIntegralType(*tt->scalarType(), /*includeBool=*/false);
  }
  return v->type()->isSubtypeOf(*FloatType::get());
}

bool isValidReturnForRunning(Value* v) {
  return v->type()->isSubtypeOf(*TensorType::get()) ||
      v->type()->isSubtypeOf(*NumberType::get());
}

bool containsTensorType(const TypePtr& t) {
  auto n_contained = t->containedTypes().size();
  if (n_contained == 1) {
    return t->containedTypes().at(0)->isSubtypeOf(*TensorType::get());
  } else if (n_contained > 1) {
    return std::any_of(
        t->containedTypes().begin(),
        t->containedTypes().end(),
        containsTensorType);
  }
  return false;
}

// for each node in the schema with type Tensor, extract the T type
// returns std::nullopt if any Tensor in the schema does not have a known
// shape ignores non-tensor in the list of inputs
std::optional<std::vector<TensorTypePtr>> gatherTensorTypes(
    Node* node,
    bool complete = false) {
  std::vector<TensorTypePtr> tensor_types;

  auto schema_opt = node->maybeSchema();
  if (!schema_opt) {
    return std::nullopt;
  }
  auto& schema = *schema_opt;
  auto& args = schema.arguments();
  // can't handle varargs primitives because we don't know what should be a
  // Tensor
  if (schema.is_vararg()) {
    return std::nullopt;
  }
  for (const auto i : c10::irange(args.size())) {
    if (args[i].type()->isSubtypeOf(*ListType::ofTensors())) {
      return std::nullopt;
    } else if (args[i].type()->isSubtypeOf(*TensorType::get())) {
      if (auto type = node->input(i)->type()->cast<TensorType>()) {
        if (complete && !type->isComplete()) {
          return std::nullopt;
        }
        tensor_types.push_back(type);
      } else {
        return std::nullopt;
      }
    } else /* non-tensor type */ {
      continue;
    }
  }
  return tensor_types;
}

int64_t wrapDim(int64_t dim, at::IntArrayRef sizes) {
  if (dim < 0) {
    dim += (int64_t)sizes.size();
  }
  return dim;
}

c10::ScalarType unionScalarTypes(
    c10::ScalarType original,
    c10::ScalarType next) {
  if (original == c10::ScalarType::Undefined) {
    return next;
  } else {
    return c10::promoteTypes(original, next);
  }
}

// Promotes result types for arithmetic operations on Tensor operands using
// new type promotion logic. See tensor_attributes.rst for details.
// This doesn't handle the case of arithmetic ops with Scalar arguments (when
// `Tensor.getUnsafeTensorImpl()->is_wrapped_number()` would return true)
std::optional<c10::ScalarType> getPromotedTypeForArithmeticOp(Node* node) {
  c10::ScalarType dimmed = c10::ScalarType::Undefined;
  c10::ScalarType zerodim = c10::ScalarType::Undefined;
  // binary arithmetic ops, more than 2 args is alpha.
  for (const auto i : c10::irange(2)) {
    auto dtt = node->inputs()[i]->type()->expect<TensorType>();
    auto inputDtype = dtt->scalarType();
    if (!dtt || !inputDtype) {
      return std::nullopt;
    }
    if (dtt->dim() && *dtt->dim() > 0) {
      dimmed = unionScalarTypes(dimmed, *inputDtype);
    } else if (!isFloatingType(dimmed)) {
      // if no dimensions
      zerodim = unionScalarTypes(zerodim, *inputDtype);
    }
  }
  // if a tensor with dimensions is already of the highest category, don't
  // need to check zero-dim tensors.
  if (isFloatingType(dimmed)) {
    return dimmed;
  }
  // int_tensor * zero_dim_floating -> floating_tensor
  if (isIntegralType(dimmed, false) && isFloatingType(zerodim)) {
    return zerodim;
  }
  // bool_tensor * non_bool_scalar -> non_bool_tensor
  if (c10::ScalarType::Bool == dimmed &&
      c10::ScalarType::Undefined != zerodim) {
    return zerodim;
  }
  // types of dimensioned tensors generally take precedence over zero-dim
  // tensors if not promoting due to category. e.g.:
  // int_tensor * long -> int_tensor
  if (c10::ScalarType::Undefined != dimmed) {
    return dimmed;
  }

  // no dimmed tensors. e.g. zero_dim_tensor + zero_dim_tensor.
  return zerodim;
}

class ShapePropagator : public PropertyPropBase {
 public:
  explicit ShapePropagator(const std::shared_ptr<Graph>& graph)
      : PropertyPropBase(graph), aliasDb_(graph) {
    collectResizeSet(graph->block());
  }

 private:
  ValueSet resized_alias_set;
  const AliasDb aliasDb_;

  bool resizesInput(Node* n) {
    static std::unordered_set<Symbol> resize_ops{
        aten::resize_,
        aten::resize_as_,
        aten::copy_,
        aten::set_,
        aten::unsqueeze_,
        aten::t_,
        aten::transpose_,
    };

    if (resize_ops.count(n->kind()))
      return true;

    if (!n->maybeSchema())
      return false;

    // ops which take the result and write to input "out"
    if (auto out_arg_index = n->schema().argumentIndexWithName("out")) {
      auto arg = n->schema().arguments().at(*out_arg_index);
      return arg.kwarg_only() && arg.type()->isSubtypeOf(*TensorType::get());
    }
    return false;
  }

  void collectResizeSet(Block* block) {
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        collectResizeSet(b);
      }
      if (resizesInput(n)) {
        for (const auto input : n->inputs()) {
          if (aliasDb_.writesToAlias(n, {input})) {
            resized_alias_set.insert(input);
          }
        }
      }
    }
  }

  IValue representativeValue(Value* v) {
    TypePtr type_ = v->type();
    // if the value is actually constant, just use it!
    if (auto iv = toIValue(v)) {
      return *iv;
    }
    if (TensorTypePtr type = type_->cast<TensorType>()) {
      if (type->isComplete()) {
        at::DeviceGuard device_guard(*type->device());
        return at::empty_strided(
                   *type->sizes().concrete_sizes(),
                   *type->strides().concrete_sizes(),
                   at::TensorOptions(*type->device()).dtype(type->scalarType()))
            .zero_();
      }
      // fallthrough
    } else if (type_->isSubtypeOf(*FloatType::get())) {
      return 0.f;
    }
    // we should not get here because isValidArgumentForRunning should have
    // prevented it
    std::stringstream ss;
    ss << "unable to create representative value for: " << type_->str()
       << ". File a bug report";
    throw std::runtime_error(ss.str());
  }

  void broadcastBinary(
      Node* node,
      std::vector<TensorTypePtr>& types,
      size_t idx1,
      size_t idx2) {
    auto expected_size = at::infer_size(
        *types[idx1]->sizes().concrete_sizes(),
        *types[idx2]->sizes().concrete_sizes());
    auto broadcast = [&](size_t input_idx) {
      TensorTypePtr input_type = types.at(input_idx);
      if (input_type->sizes() == expected_size)
        return;
      auto graph = node->owningGraph();
      WithInsertPoint point_guard{node};
      Node* expand = graph
                         ->create(
                             aten::expand,
                             {node->inputs().at(input_idx),
                              graph->insertConstant(expected_size),
                              graph->insertConstant(false)})
                         ->insertBefore(node);
      propagateNode(expand);
      node->replaceInput(input_idx, expand->output());
    };
    broadcast(idx1);
    broadcast(idx2);
    types[0] = node->inputs().at(idx1)->type()->expect<TensorType>();
    types[1] = node->inputs().at(idx2)->type()->expect<TensorType>();
  }

  OperatorSet cannot_propagate_shape_by_running_it = {
      "aten::inverse(Tensor self) -> Tensor",
  };

  // Check if this node depends on a value that has been mutated previously. If
  // it has, then it's not safe to run this node in isolation, since we don't
  // know whether the dependency has been executed.
  std::unordered_map<Node*, bool> dependsOnMutationMemo_;
  bool dependsOnMutation(Node* node) {
    if (dependsOnMutationMemo_.count(node) != 0) {
      return dependsOnMutationMemo_[node];
    }

    if (aliasDb_.hasWriters(node)) {
      // If something could have written to a value used by this node, we can't
      // guarantee the result is the same when running it in isolation.
      dependsOnMutationMemo_[node] = true;
      return true;
    }

    // recursively check the producers of its inputs. We need to do this if the
    // mutable value has been laundered through a pure function:
    //   a += 1
    //   c = a + b
    //   d = c + 1
    // In this case, `d` cares whether `a` has been mutated even though it's not
    // a direct input.
    auto depends = false;
    for (auto input : node->inputs()) {
      depends |= dependsOnMutation(input->node());
    }

    dependsOnMutationMemo_[node] = depends;
    return depends;
  }

  bool canPropagateShapeByRunningIt(Node* node) {
    if (node->isMemberOf(cannot_propagate_shape_by_running_it)) {
      return false;
    }

    if (dependsOnMutation(node)) {
      return false;
    }

    bool valid_args = std::all_of(
        node->inputs().begin(),
        node->inputs().end(),
        isValidArgumentForRunning);
    if (!valid_args)
      return false;

    bool valid_returns = std::all_of(
        node->outputs().begin(),
        node->outputs().end(),
        isValidReturnForRunning);
    if (!valid_returns)
      return false;

    return true;
  }

  // If there's no Tensor in outputs, e.g float / float,
  // we don't need to propagate shape.
  bool DoesntRefineOutputs(Node* node) {
    auto outputs = node->outputs();
    for (auto& out : outputs) {
      if (containsTensorType(out->type())) {
        return false;
      }
    }
    return true;
  }

  bool PropagateShapeOnNodeByRunningIt(Node* node, Operation op = nullptr) {
    if (!canPropagateShapeByRunningIt(node))
      return false;

    if (!op)
      op = node->getOperation();

    Stack stack;

    for (auto input : node->inputs()) {
      stack.push_back(representativeValue(input));
    }

    // XXX: we're not catching any exceptions from the op for now. This
    // is to uncover any mistakes we could make when editing this code,
    // and eventually it shouldn't matter, because this phase should be
    // preceded by schema checking.
    op(stack);

    AT_ASSERT(stack.size() == node->outputs().size());
    for (const auto i : c10::irange(stack.size())) {
      // some ops may have mixed tensor/primitive outputs
      // for primitives, we don't need to change the type because it is already
      // its most constrained form.
      auto tensor_type = node->outputs()[i]->type()->cast<TensorType>();
      if (stack[i].isTensor() && tensor_type) {
        // gradient information isn't always available or part of representative
        // inputs, maintain original grad property
        auto tensor_grad = tensor_type->requiresGrad();
        node->outputs()[i]->setType(TensorType::create(stack[i].toTensor())
                                        ->withRequiresGrad(tensor_grad));
      }
    }
    return true;
  }

  void PropagateCatShape(Node* cat_node) {
    static const auto propagate_complete =
        [](Node* node, at::ArrayRef<Value*> tensors) -> bool {
      auto input_types =
          fmap(tensors, [](Value* v) { return v->type()->cast<TensorType>(); });
      if (!std::all_of(
              input_types.begin(),
              input_types.end(),
              [](const TensorTypePtr& tp) {
                return tp != nullptr && tp->isComplete();
              })) {
        return false;
      }
      if (!node->is_constant(attr::dim))
        return false;
      std::vector<int64_t> sizes = *input_types[0]->sizes().concrete_sizes();
      const int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      const int64_t ndim = (int64_t)sizes.size();

      if (dim < 0 || dim >= ndim)
        return false;

      sizes[dim] = 0;
      for (auto& tp : input_types) {
        auto tp_sizes = tp->sizes().concrete_sizes().value();
        if (sizes.size() != tp_sizes.size())
          return false;
        for (const auto i : c10::irange(ndim)) {
          if (sizes[i] != tp_sizes[i] && i != dim) {
            return false;
          }
        }
        sizes[dim] += tp_sizes[dim];
      }
      node->output()->setType(input_types[0]->withSizes(sizes));
      return true;
    };
    static const auto propagate = [](Node* node,
                                     at::ArrayRef<Value*> tensors) -> bool {
      for (Value* v : tensors) {
        if (auto type = v->type()->cast<TensorType>()) {
          node->output()->setType(type->dimensionedOnly());
          return true;
        }
      }
      return false;
    };
    auto list_node =
        ((cat_node->kind() == prim::FusedConcat)
             ? cat_node
             : cat_node->namedInput(attr::tensors)->node());
    if (list_node->kind() == prim::ListConstruct ||
        cat_node->kind() == prim::FusedConcat) {
      auto tensors = list_node->inputs();
      if (!tensors.empty()) {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (propagate_complete(cat_node, tensors)) {
          return;
        } else if (propagate(cat_node, tensors)) {
          return;
        }
      }
    }
    setUnshapedType(cat_node);
  }

  void propagateTorchTensorShape(Node* node) {
    auto input_type = node->inputs().at(0)->type();

    size_t dims = 0;
    auto input_base_type = input_type;
    auto list_type = input_type->cast<ListType>();
    while (list_type) {
      dims++;
      input_base_type = list_type->getElementType();
      list_type = input_base_type->cast<ListType>();
    }

    std::optional<at::ScalarType> default_type =
        tryScalarTypeFromJitType(*input_base_type);
    if (auto grad_index = node->schema().argumentIndexWithName("dtype")) {
      auto inp = toIValue(node->inputs().at(*grad_index));
      if (inp == std::nullopt) {
        return;
      } else if (!inp->isNone()) {
        default_type = inp->toScalarType();
      }
    }

    at::Device default_device = at::kCPU;
    if (auto device_index = node->schema().argumentIndexWithName("device")) {
      auto inp = toIValue(node->inputs().at(*device_index));
      if (inp == std::nullopt) {
        return;
      } else if (!inp->isNone()) {
        default_device = inp->toDevice();
      }
    }
    node->output()->setType(TensorType::create(
        default_type, default_device, dims, /*requires_grad=*/std::nullopt));
  }

  // returns whether any such values were found
  bool setUnshapedTypeIfAliasResizedSet(at::ArrayRef<Value*> vs) {
    bool in_resize = false;
    for (auto v : vs) {
      if (aliasDb_.mayAlias(ValueSet{v}, resized_alias_set)) {
        setUnshapedType(v);
        in_resize = true;
      }
    }
    return in_resize;
  }

  void propagateNode(Node* node, bool insert_expands = true) override {
    // Certain ops like resize_ change the input tensors size. Because our
    // analysis is flow invariant, we set any Tensor that can alias a resized
    // Tensor to the base Tensor Type without size information.
    if (setUnshapedTypeIfAliasResizedSet(node->inputs())) {
      return setUnshapedType(node);
    }

    // These don't require the types, and have complicated schema. Return early
    // after we process them.
    switch (node->kind()) {
      case prim::If:
        return processIf(node);
      case prim::Loop: {
        return processLoop(node);
      }
      case aten::Bool:
      case aten::Int:
      case aten::Float:
      case aten::ScalarImplicit:
      case aten::FloatImplicit:
      case aten::IntImplicit:
        return; // correct num type is already set
      case prim::NumToTensor: {
        TypePtr typ = node->input()->type();
        if (typ->isSubtypeOf(*IntType::get()) ||
            typ->isSubtypeOf(*BoolType::get())) {
          node->output()->setType(TensorType::create(
              at::kLong, at::kCPU, 0, /*requires_grad=*/std::nullopt));
        } else if (node->input()->type()->isSubtypeOf(*FloatType::get())) {
          node->output()->setType(TensorType::create(
              at::kDouble, at::kCPU, 0, /*requires_grad=*/std::nullopt));
        }
        return;
      }
      case aten::tensor:
      case aten::as_tensor: {
        // as_tensor has an overloaded schema and can either have a tensor or
        // a list as the first input, if the input is a tensor, we delegate
        // the shape propagation in PropagateTensorShapeOnNode
        if (node->inputs().at(0)->type()->isSubtypeOf(*TensorType::get())) {
          break;
        }
        return propagateTorchTensorShape(node);
      }
      case prim::TupleConstruct: {
        // We refresh the tuple type, because the input types could have been
        // refined.
        auto orig_type = node->output()->type()->expect<TupleType>();
        auto new_types =
            fmap(node->inputs(), [](Value* v) { return v->type(); });
        node->output()->setType(
            orig_type->createWithContained(std::move(new_types)));
        return;
      }
      case prim::TupleUnpack: {
        auto tuple_type = node->input()->type()->cast<TupleType>();
        AT_ASSERT(
            tuple_type &&
            tuple_type->elements().size() == node->outputs().size());
        auto elems = tuple_type->elements();
        for (size_t i = 0; i < node->outputs().size(); ++i) {
          node->output(i)->setType(elems[i]);
        }
        return;
      }
      case prim::Constant: {
        if (node->output()->type()->isSubtypeOf(*TensorType::get())) {
          node->output()->inferTypeFrom(node->t(attr::value));
        }
        return;
      }
      case prim::unchecked_unwrap_optional: {
        // If we have specialized the optional type to the element type,
        // we want to pass it down. We write this as input.isSubtypeOf(output)
        // to be sure that we don't screw up nested optionals.
        if (node->input()->type()->isSubtypeOf(*node->output()->type())) {
          node->output()->setType(node->input()->type());
        }
        return;
      }
      case prim::ConstantChunk: {
        Value* tensor = node->input();
        if (auto type = tensor->type()->cast<TensorType>()) {
          type = type->dimensionedOnly();
          for (Value* output : node->outputs()) {
            output->setType(type);
          }
        } else {
          setUnshapedType(node);
        }
        return;
      }
      case prim::grad: {
        auto tt = node->input()->type()->expect<TensorType>();
        // grad may be undefined
        // requires_grad may be required
        auto grad_type = TensorType::get()->withPossiblyUndefined();
        node->output()->setType(std::move(grad_type));
        return;
      }
      case prim::CallFunction:
      case prim::CallMethod:
      case prim::AutogradZero: {
        setUnshapedType(node);
        return;
      }
      case prim::GetAttr: {
        auto cls = node->input()->type()->expect<ClassType>();
        // propagate any type specializations encoded in the type of the class
        node->output()->setType(cls->getAttribute(node->s(attr::name)));
        return;
      }
      case aten::_unwrap_optional: {
        // If we have specialized the optional type to the element type,
        // we want to pass it down. We write this as input.isSubtypeOf(output)
        // to be sure that we don't screw up nested optionals.
        if (node->input()->type()->isSubtypeOf(*node->output()->type())) {
          node->output()->setType(node->input()->type());
        }
        return;
      }
      default:
        break; // fall-through
    }

    if (node->hasSideEffects()) {
      return;
    }

    if (node->matches("aten::cat(Tensor[] tensors, int dim) -> Tensor") ||
        node->kind() == prim::FusedConcat) {
      return PropagateCatShape(node);
    }

    if (auto maybe_complete_types =
            gatherTensorTypes(node, /*complete=*/true)) {
      if (PropagateCompleteShapeOnNode(
              node, insert_expands, std::move(*maybe_complete_types))) {
        return;
      }
    }

    if (PropagateTensorShapeOnNode(node, insert_expands)) {
      return;
    }

    if (DoesntRefineOutputs(node)) {
      return;
    }

    if (PropagateShapeOnNodeByRunningIt(node)) {
      return;
    }
    return setUnshapedType(node);
  }

  static std::optional<size_t> determineListSize(Value* list) {
    AT_ASSERT(list->type()->cast<ListType>());
    if (auto shape = constant_as<c10::List<int64_t>>(list)) {
      return shape->size();
    }
    auto input_node = list->node();
    if (input_node->kind() == prim::ListConstruct) {
      return input_node->inputs().size();
    }
    return std::nullopt;
  }

  // is it ok to try to run the op
  // If an input is a constant, then we assume that the input is valid
  // and we can try to run it.
  // Otherwise:
  // Integral typed _inputs_ are often an indicator that we're indexing into
  // a tensor, so we should special-case these ops in the shape propagation.
  // Additionally, passing in a zero representative tensor into an integer
  // division op causes divide-by-zero errors
  // _Outputs_ must be tensors or primitives
  // We will call inferTypeFrom on the tensors, and ignore the primitives.
  // However, we allow primitive returns because we want to support mixed
  // primitive/tensor outputs.

  bool PropagateTensorShapeOnNode(Node* node, bool insert_expands) {
    static const auto broadcast =
        [](std::vector<TensorTypePtr>& tensor_types,
           std::optional<at::ScalarType> t) -> TensorTypePtr {
      if (tensor_types.size() == 1) {
        return tensor_types[0]->dimensionedOnly()->withScalarType(t);
      }
      AT_ASSERT(!tensor_types.empty());
      auto any_type = tensor_types[0];
      auto max_dims = any_type->dim();
      for (auto& type : tensor_types) {
        if (!max_dims || !type->dim()) {
          max_dims = std::nullopt;
        } else {
          max_dims = std::max(*max_dims, *type->dim());
        }
      }
      return TensorType::create(
          t,
          any_type->device(),
          max_dims,
          /*requires_grad=*/std::nullopt);
    };

    using type_vec_t = std::vector<TensorTypePtr>;
    // Formula is expected to return a vector of length equal to the number of
    // tensor outputs of the node, or an empty vector which implies that it
    // failed to propagate.
    using formula_t = std::function<type_vec_t(Node*)>;
    static std::mutex shape_formulas_mutex;
    static std::vector<std::pair<OperatorSet, formula_t>> shape_formulas;
    struct register_formula_for {
      register_formula_for(OperatorSet operators, formula_t formula) {
        std::unique_lock<std::mutex> lock{shape_formulas_mutex};
        shape_formulas.emplace_back(std::move(operators), std::move(formula));
      }
    };

    // Requirements:
    //   dims           : preserved
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for simple_unary_ops{
        {
            "aten::acos(Tensor self) -> Tensor",
            "aten::neg(Tensor self) -> Tensor",
            "aten::t(Tensor self) -> Tensor",
            "aten::sigmoid(Tensor self) -> Tensor",
            "aten::logit(Tensor self, float? eps=None) -> Tensor",
            "aten::tanh(Tensor self) -> Tensor",
            "aten::relu(Tensor self) -> Tensor",
            "aten::asin(Tensor self) -> Tensor",
            "aten::atan(Tensor self) -> Tensor",
            "aten::ceil(Tensor self) -> Tensor",
            "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
            "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)",
            "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
            "aten::celu(Tensor self, Scalar alpha) -> Tensor",
            "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor",
            "aten::clamp_max(Tensor self, Scalar max) -> Tensor",
            "aten::clamp_min(Tensor self, Scalar min) -> Tensor",
            "aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
            "aten::cos(Tensor self) -> Tensor",
            "aten::cosh(Tensor self) -> Tensor",
            "aten::digamma(Tensor self) -> Tensor",
            "aten::dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::elu(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor",
            "aten::erf(Tensor self) -> Tensor",
            "aten::erfc(Tensor self) -> Tensor",
            "aten::erfinv(Tensor self) -> Tensor",
            "aten::exp(Tensor self) -> Tensor",
            "aten::expm1(Tensor self) -> Tensor",
            "aten::log(Tensor self) -> Tensor",
            "aten::log10(Tensor self) -> Tensor",
            "aten::log1p(Tensor self) -> Tensor",
            "aten::log2(Tensor self) -> Tensor",
            "aten::log_sigmoid(Tensor self) -> Tensor",
            "aten::floor(Tensor self) -> Tensor",
            "aten::frac(Tensor self) -> Tensor",
            "aten::flip(Tensor self, int[] dims) -> Tensor",
            "aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::feature_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::hardshrink(Tensor self, Scalar lambd) -> Tensor",
            "aten::hardtanh(Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
            "aten::glu(Tensor self, int dim) -> Tensor",
            "aten::inverse(Tensor self) -> Tensor",
            "aten::leaky_relu(Tensor self, Scalar negative_slope) -> Tensor",
            "aten::lgamma(Tensor self) -> Tensor",
            "aten::mvlgamma(Tensor self, int p) -> Tensor",
            "aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor",
            "aten::permute(Tensor self, int[] dims) -> Tensor",
            "aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)",
            "aten::pinverse(Tensor self, float rcond) -> Tensor",
            "aten::reciprocal(Tensor self) -> Tensor",
            "aten::relu(Tensor self) -> Tensor",
            "aten::round(Tensor self) -> Tensor",
            "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
            "aten::rsqrt(Tensor self) -> Tensor",
            "aten::selu(Tensor self) -> Tensor",
            "aten::gelu(Tensor self, *, str approximate='none') -> Tensor",
            "aten::sigmoid(Tensor self) -> Tensor",
            "aten::sign(Tensor self) -> Tensor",
            "aten::sin(Tensor self) -> Tensor",
            "aten::sinh(Tensor self) -> Tensor",
            "aten::softplus(Tensor self, Scalar beta, Scalar threshold) -> Tensor",
            "aten::softshrink(Tensor self, Scalar lambd) -> Tensor",
            "aten::sqrt(Tensor self) -> Tensor",
            "aten::tan(Tensor self) -> Tensor",
            "aten::tanh(Tensor self) -> Tensor",
            "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
            "aten::transpose(Tensor self, int dim0, int dim1) -> Tensor",
            "aten::tril(Tensor self, int diagonal) -> Tensor",
            "aten::triu(Tensor self, int diagonal) -> Tensor",
            "aten::trunc(Tensor self) -> Tensor",
            "aten::rot90(Tensor self, int k, int[] dims) -> Tensor",
            "aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
            "aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor",
            "aten::alias(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto input_type = node->input(0)->type()->cast<TensorType>();
          return input_type ? type_vec_t{input_type->dimensionedOnly()}
                            : type_vec_t{};
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : preserved, except complex maps to float
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for simple_unary_ops_complex_to_float{
        {
            "aten::abs(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto input_type = node->input(0)->type()->cast<TensorType>();

          // Maps complex -> float
          if (input_type->scalarType()) {
            const auto scalar_type = *(input_type->scalarType());
            if (isComplexType(scalar_type)) {
              const auto out_type = c10::toRealValueType(scalar_type);
              return type_vec_t{
                  input_type->dimensionedOnly()->withScalarType(out_type)};
            }
          }

          return input_type ? type_vec_t{input_type->dimensionedOnly()}
                            : type_vec_t{};
        }};

    // Requirements:
    //   dims           : broadcast all tensor args
    //   scalar type    : promoted from input dtypes
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for broadcasting_ops_arithmetic{
        {
            // Tensor-Tensor operators
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::mul(Tensor self, Tensor other) -> Tensor",
            "aten::div(Tensor self, Tensor other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            AT_ASSERT(maybe_tensor_types->size() >= 2);
            auto dtype = getPromotedTypeForArithmeticOp(node);
            return {broadcast(*maybe_tensor_types, dtype)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : broadcast all tensor args
    //   scalar type    : always matching and preserved
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for broadcasting_ops{
        {
            "aten::pow(Tensor self, Tensor exponent) -> Tensor",
            "aten::fmod(Tensor self, Tensor other) -> Tensor",
            "aten::remainder(Tensor self, Tensor other) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor",
            "aten::max(Tensor self, Tensor other) -> Tensor",
            "aten::min(Tensor self, Tensor other) -> Tensor",
            "aten::__and__(Tensor self, Tensor other) -> Tensor",
            "aten::__or__(Tensor self, Tensor other) -> Tensor",
            "aten::__xor__(Tensor self, Tensor other) -> Tensor",
            "aten::__lshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__rshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__iand__(Tensor self, Tensor other) -> Tensor",
            "aten::__ior__(Tensor self, Tensor other) -> Tensor",
            "aten::__ixor__(Tensor self, Tensor other) -> Tensor",
            "aten::__ilshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__irshift__(Tensor self, Tensor other) -> Tensor",

            // Ops with Tensor-Tensor overloads only
            "aten::atan2(Tensor self, Tensor other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            AT_ASSERT(maybe_tensor_types->size() >= 2);
            auto first_scalar_type = (*maybe_tensor_types)[0]->scalarType();
            auto second_scalar_type = (*maybe_tensor_types)[1]->scalarType();
            if (!first_scalar_type || !second_scalar_type) {
              return {};
            }
            size_t arg_for_type = 0;
            if (c10::promoteTypes(*first_scalar_type, *second_scalar_type) !=
                first_scalar_type) {
              arg_for_type = 1;
            }
            auto t = (*maybe_tensor_types)[arg_for_type]->scalarType();
            return {broadcast(*maybe_tensor_types, t)};
          }
          return {};
        }};

    static const register_formula_for fused_accum_binary_ops{
        {
            // Non-binary ops
            "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
            "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            auto dtype = (*maybe_tensor_types)[0]->scalarType();
            if (!dtype) {
              return {};
            }
            return {broadcast(*maybe_tensor_types, dtype)};
          }
          return {};
        }};

    static const register_formula_for broadcasting_tensor_scalar_ops_arithmetic{
        {
            // Tensor-Scalar operators
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::mul(Tensor self, Scalar other) -> Tensor",
            "aten::div(Tensor self, Scalar other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            auto first_scalar_type = (*maybe_tensor_types)[0]->scalarType();
            auto second_scalar_type =
                tryScalarTypeFromJitType(*node->inputs()[1]->type());
            if (!first_scalar_type || !second_scalar_type) {
              return {};
            }
            if (isIntegralType(*first_scalar_type, false) &&
                isFloatingType(*second_scalar_type)) {
              auto default_dtype =
                  at::typeMetaToScalarType(caffe2::get_default_dtype());
              return {broadcast(*maybe_tensor_types, default_dtype)};
            }
            if (c10::ScalarType::Bool == *first_scalar_type &&
                c10::ScalarType::Bool != *second_scalar_type) {
              auto result_type =
                  c10::promoteTypes(*first_scalar_type, *second_scalar_type);
              return {broadcast(*maybe_tensor_types, result_type)};
            }
            return {broadcast(*maybe_tensor_types, first_scalar_type)};
          }
          return {};
        }};

    // NB: we always take the scalar type of the Tensor
    static const register_formula_for broadcasting_tensor_scalar_ops{
        {

            "aten::pow(Tensor self, Scalar exponent) -> Tensor",
            "aten::fmod(Tensor self, Scalar other) -> Tensor",
            "aten::remainder(Tensor self, Scalar other) -> Tensor",
            "aten::pow(Scalar self, Tensor exponent) -> Tensor",
            "aten::__and__(Tensor self, Scalar other) -> Tensor",
            "aten::__or__(Tensor self, Scalar other) -> Tensor",
            "aten::__xor__(Tensor self, Scalar other) -> Tensor",
            "aten::__lshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__rshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__iand__(Tensor self, Scalar other) -> Tensor",
            "aten::__ior__(Tensor self, Scalar other) -> Tensor",
            "aten::__ixor__(Tensor self, Scalar other) -> Tensor",
            "aten::__ilshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__irshift__(Tensor self, Scalar other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(
                *maybe_tensor_types, (*maybe_tensor_types)[0]->scalarType())};
          }
          return {};
        }};

    // aten::where is special in that its return type is the second argument's
    // (self) type rather than the that of condition
    static const register_formula_for where_op{
        {
            "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(
                *maybe_tensor_types, (*maybe_tensor_types)[1]->scalarType())};
          }
          return {};
        }};

    static const auto any_tensor_type = [](Node* node) -> TensorTypePtr {
      for (Value* input : node->inputs()) {
        if (auto type = input->type()->cast<TensorType>()) {
          if (type->dim().has_value()) {
            return type;
          }
        }
      }
      return nullptr;
    };

    // Requirements:
    //   dims           : always matching and preserved
    //   scalar type    : always matching and preserved
    //   device         : always matching and preserved
    //   tensor inputs  : 2
    //   tensor outputs : 1
    static const register_formula_for binary_ops_strict_match{
        {
            "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::mm(Tensor self, Tensor mat2) -> Tensor",
            "aten::bmm(Tensor self, Tensor mat2) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = any_tensor_type(node)) {
            return {std::move(type)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : all tensor args are broadcast
    //   scalar type    : byte/uint8
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for comparison_ops{
        {
            "aten::lt(Tensor self, Tensor other) -> Tensor",
            "aten::le(Tensor self, Tensor other) -> Tensor",
            "aten::gt(Tensor self, Tensor other) -> Tensor",
            "aten::ge(Tensor self, Tensor other) -> Tensor",
            "aten::eq(Tensor self, Tensor other) -> Tensor",
            "aten::ne(Tensor self, Tensor other) -> Tensor",
            "aten::lt(Tensor self, Scalar other) -> Tensor",
            "aten::le(Tensor self, Scalar other) -> Tensor",
            "aten::gt(Tensor self, Scalar other) -> Tensor",
            "aten::ge(Tensor self, Scalar other) -> Tensor",
            "aten::eq(Tensor self, Scalar other) -> Tensor",
            "aten::ne(Tensor self, Scalar other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(*maybe_tensor_types, at::kBool)};
          }
          return {};
        }};

    static const register_formula_for nn_ops_first_input_formula{
        *nn_ops_first_input_preserving(), [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type->dimensionedOnly()};
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for all_reduce_ops{
        {
            "aten::det(Tensor self) -> Tensor",
            "aten::logdet(Tensor self) -> Tensor",
            "aten::max(Tensor self) -> Tensor",
            "aten::min(Tensor self) -> Tensor",
            "aten::median(Tensor self) -> Tensor",
            "aten::nanmedian(Tensor self) -> Tensor",
            "aten::norm(Tensor self, Scalar p) -> Tensor",
            "aten::std(Tensor self, bool unbiased) -> Tensor",
            "aten::trace(Tensor self) -> Tensor",
            "aten::var(Tensor self, bool unbiased) -> Tensor",
            "aten::all(Tensor self) -> Tensor",
            "aten::any(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type->withDim(0)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : dtype if specified, else preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for reduce_ops_with_opt_dtype{
        {"aten::mean(Tensor self, *, int? dtype) -> Tensor"},
        [](Node* node) -> type_vec_t {
          std::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            auto ret = type->withDim(0);
            if (maybe_dtype_option && !maybe_dtype_option->isNone()) {
              return {ret->withScalarType(maybe_dtype_option->toScalarType())};
            } else {
              return {std::move(ret)};
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : dtype if specified, else preserved if floating point,
    //   otherwise long/int64 device         : preserved tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for
        all_reduce_ops_with_integer_upcast_and_dtype{
            {
                "aten::sum(Tensor self, *, int? dtype) -> Tensor",
                "aten::prod(Tensor self, *, int? dtype) -> Tensor",
            },
            [](Node* node) -> type_vec_t {
              if (auto type = node->input(0)->type()->cast<TensorType>()) {
                type = type->withDim(0);
                std::optional<IValue> maybe_dtype_option =
                    node->get(attr::dtype);
                if (maybe_dtype_option && !maybe_dtype_option->isNone()) {
                  return {
                      type->withScalarType(maybe_dtype_option->toScalarType())};
                }
                if (type->scalarType()) {
                  return {
                      at::isFloatingType(*type->scalarType())
                          ? std::move(type)
                          : type->withScalarType(at::kLong)};
                } else {
                  return {std::move(type)};
                }
              }
              return {};
            }};

    static const auto reduce_op_handler = [](Node* node,
                                             int64_t num_reduced_dim = 0,
                                             bool upcast_integer = false,
                                             std::optional<IValue> opt_dtype =
                                                 std::nullopt) -> type_vec_t {
      if (auto type = node->input(0)->type()->cast<TensorType>()) {
        if (!type->scalarType() || !type->dim()) {
          return {};
        }
        if (opt_dtype && !opt_dtype->isNone()) {
          type = type->withScalarType(opt_dtype->toScalarType());
        } else if (upcast_integer && !at::isFloatingType(*type->scalarType())) {
          type = type->withScalarType(at::kLong);
        }
        if (static_cast<int64_t>(*type->dim()) >= num_reduced_dim &&
            num_reduced_dim > 0) {
          return {type->withDim(*type->dim() - num_reduced_dim)};
        } else {
          return {std::move(type)};
        }
      }
      return {};
    };

    static const auto multidim_reduce_with_keepdim =
        [](Node* node,
           int64_t num_reduced_dim,
           bool upcast_integer) -> type_vec_t {
      auto maybe_keepdim = node->get<bool>(attr::keepdim);
      if (!maybe_keepdim)
        return {};
      return reduce_op_handler(
          node, *maybe_keepdim ? 0 : num_reduced_dim, upcast_integer);
    };

    // Requirements:
    //   dims           : 0 if dim is None, otherwise preserved if keepdim ==
    //   false or 1 smaller otherwise scalar type    : preserved device :
    //   preserved tensor inputs  : 1 tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    //   - Has a bool keepdim argument
    static const register_formula_for argminmax{
        {
            "aten::argmax(Tensor self, int? dim, bool keepdim) -> Tensor",
            "aten::argmin(Tensor self, int? dim, bool keepdim) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            if (node->input(1)->type()->kind() == c10::TypeKind::NoneType) {
              return {type->withDim(0)};
            } else {
              return multidim_reduce_with_keepdim(
                  node, /*num_reduced_dim=*/1, /*upcast_integer=*/false);
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved if keepdim == false, 1 smaller otherwise
    //   scalar type    : preserved for first output, byte/uint8 for second
    //   output if exists device         : preserved tensor inputs  : 1 tensor
    //   outputs : 1 or 2
    // Additionally:
    //   - First input should be the only tensor input
    //   - Has a bool keepdim argument
    static const register_formula_for dim_reduce_ops{
        {
            "aten::all(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::any(Tensor self, int dim, bool keepdim) -> Tensor",

            // Ops returning indices as second output
            "aten::kthvalue(Tensor self, int k, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::max(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::min(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::median(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::nanmedian(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::mode(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
        },
        [](Node* node) -> type_vec_t {
          // NB: Note that while this function is generally meant to be used
          // with ops that have a single output, we will fix up its return right
          /
```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 125 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `prim`, `void`

**Classes/Structs**: `ShapePropagator`, `node`, `register_formula_for`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/shape_analysis.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`
- `torch/csrc/jit/frontend/error_report.h`
- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/ir/constants.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/ir_views.h`
- `torch/csrc/jit/passes/utils/op_registry.h`
- `torch/csrc/jit/runtime/exception_message.h`
- `torch/csrc/jit/runtime/operator.h`
- `torch/csrc/autograd/variable.h`
- `ATen/DeviceGuard.h`
- `ATen/ExpandUtils.h`
- `ATen/core/symbol.h`
- `ATen/Functions.h`
- `ATen/ops/empty_strided.h`
- `exception`
- `memory`
- `sstream`
- `utility`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/jit/passes`):

- [`inline_fork_wait.h_docs.md`](./inline_fork_wait.h_docs.md)
- [`subgraph_rewrite.cpp_docs.md`](./subgraph_rewrite.cpp_docs.md)
- [`value_refinement_utils.cpp_docs.md`](./value_refinement_utils.cpp_docs.md)
- [`create_autodiff_subgraphs.cpp_docs.md`](./create_autodiff_subgraphs.cpp_docs.md)
- [`update_differentiable_graph_requires_grad.h_docs.md`](./update_differentiable_graph_requires_grad.h_docs.md)
- [`inplace_check.h_docs.md`](./inplace_check.h_docs.md)
- [`common_subexpression_elimination.h_docs.md`](./common_subexpression_elimination.h_docs.md)
- [`dtype_analysis.cpp_docs.md`](./dtype_analysis.cpp_docs.md)
- [`canonicalize.h_docs.md`](./canonicalize.h_docs.md)
- [`add_if_then_else.h_docs.md`](./add_if_then_else.h_docs.md)


## Cross-References

- **File Documentation**: `shape_analysis.cpp_docs.md`
- **Keyword Index**: `shape_analysis.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
