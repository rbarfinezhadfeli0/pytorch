# Documentation: `docs/torch/csrc/jit/passes/onnx/shape_type_inference.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/onnx/shape_type_inference.cpp_docs.md`
- **Size**: 53,354 bytes (52.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/onnx/shape_type_inference.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/shape_type_inference.cpp`
- **Size**: 92,576 bytes (90.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <torch/csrc/jit/passes/onnx/constant_map.h>
#include <torch/csrc/jit/passes/onnx/fixup_onnx_controlflow.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/onnx.h>
#include <torch/csrc/utils/python_strings.h>

#include <onnx/shape_inference/implementation.h>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <unordered_set>
#include <utility>

namespace torch::jit {

static inline bool PyNone_Check(PyObject* o) {
  return o == Py_None;
}

std::pair<TypePtr, bool> MergeInferredType(
    const TypePtr& existing_type,
    const TypePtr& inferred_type) {
  auto new_list_type = inferred_type->cast<ListType>();
  auto use_inferred_type = false;
  if (new_list_type) {
    return std::make_pair(inferred_type, true);
  }
  auto new_tensor_type = inferred_type->cast<TensorType>();
  auto old_tensor_type = existing_type->cast<TensorType>();

  if (new_tensor_type && old_tensor_type) {
    if (!old_tensor_type->device()) {
      // device not available means this is an invalid tensor type (most likely
      // an empty one) return inferred type directly.
      return std::make_pair(new_tensor_type, true);
    }
    auto type = old_tensor_type;
    if (new_tensor_type->dim()) {
      type = type->withSymbolicShapes(new_tensor_type->symbolic_sizes());
      use_inferred_type = true;
    }
    if (new_tensor_type->scalarType().has_value()) {
      type = type->withScalarType(new_tensor_type->scalarType());
      use_inferred_type = true;
    }
    return std::make_pair(type, use_inferred_type);
  }

  if (old_tensor_type) {
    return std::make_pair(existing_type, false);
  }

  auto old_list_type = existing_type->cast<ListType>();
  if (new_tensor_type && old_list_type) {
    if (new_tensor_type->sizes().isComplete()) {
      return std::make_pair(inferred_type, true);
    }
    return std::make_pair(existing_type, false);
  }

  return std::make_pair(inferred_type, true);
}

void MergeInferredTypeAndSetMap(
    Value* dest_v,
    const TypePtr& existing_type,
    const TypePtr& inferred_type) {
  auto [mergedType, inferred] = MergeInferredType(existing_type, inferred_type);
  dest_v->setType(mergedType);
  ConstantValueMap::SetUseInferredType(dest_v->debugName(), inferred);
}

namespace {
namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

// SymbolDimMap is a Torch-to-ONNX shape look-up. This is built so it can be
// returned by the export function. During the export however, when we come
// across new ONNX shapes, the reverse look-up is needed. To avoid incurring
// a linear-time look-up, we maintain DimSymbolMap in parallel.
c10::ShapeSymbol ONNXDimToShapeSymbol(
    const onnx::TensorShapeProto_Dimension& dim,
    SymbolDimMap& symbol_dim_map,
    DimSymbolMap& dim_symbol_map) {
  if (dim.has_dim_value()) {
    return c10::ShapeSymbol::fromStaticSize(dim.dim_value());
  }
  std::optional<c10::ShapeSymbol> sym = std::nullopt;
  if (dim.has_dim_param()) {
    // If this param is already known, assign the same Symbol.
    GRAPH_UPDATE("Got dim_param:", dim.dim_param());
    auto maybe_symbol = dim_symbol_map.find(dim.dim_param());
    if (maybe_symbol != dim_symbol_map.end()) {
      sym = maybe_symbol->second;
    }
  }
  if (!sym) {
    sym = c10::ShapeSymbol::newSymbol();
    // If dim.dim_param() is empty, no need to keep track
    // because there won't be duplicates.
    symbol_dim_map[sym.value()] = dim.dim_param();
    dim_symbol_map[dim.dim_param()] = sym.value();
  }
  return sym.value();
}

TensorTypePtr TorchTensorTypeFromONNX(
    const onnx::TypeProto_Tensor& onnx_tensor_type,
    SymbolDimMap& symbol_dim_map,
    DimSymbolMap& dim_symbol_map) {
  std::optional<at::ScalarType> scalar_type;
  if (onnx_tensor_type.has_elem_type()) {
    scalar_type = ONNXTypeToATenType(onnx_tensor_type.elem_type());
  }

  auto v_type = TensorType::create(
      scalar_type,
      at::kCPU,
      c10::SymbolicShape(),
      c10::VaryingShape<c10::Stride>{},
      {});
  if (onnx_tensor_type.has_shape()) {
    std::vector<c10::ShapeSymbol> sizes;
    const auto& onnx_shape = onnx_tensor_type.shape();

    for (const auto i : c10::irange(onnx_shape.dim_size())) {
      sizes.emplace_back(ONNXDimToShapeSymbol(
          onnx_shape.dim(i), symbol_dim_map, dim_symbol_map));
    }
    v_type = TensorType::create(scalar_type, at::kCPU, sizes.size(), {});
    v_type = v_type->withSymbolicShapes(c10::SymbolicShape(sizes));

    if (v_type->sizes().concrete_sizes().has_value()) {
      // Populate strides based on sizes info, if sizes are all static.
      // Creating strides ensures yielding True for isCompleteTensor.
      v_type = v_type->contiguous();
    }
  }

  return v_type;
}

ListTypePtr TorchListTypeFromONNX(
    const onnx::TypeProto_Sequence& onnx_sequence_type,
    SymbolDimMap& symbol_dim_map,
    DimSymbolMap& dim_symbol_map) {
  if (onnx_sequence_type.has_elem_type()) {
    const auto& onnx_seq_elem_type = onnx_sequence_type.elem_type();
    if (onnx_seq_elem_type.has_tensor_type()) {
      const auto& onnx_tensor_type = onnx_seq_elem_type.tensor_type();
      const auto v_tensor_type = TorchTensorTypeFromONNX(
          onnx_tensor_type, symbol_dim_map, dim_symbol_map);
      auto v_type = ListType::create(v_tensor_type);
      return v_type;
    }
  }
  return nullptr;
}

void UpdateTorchValueByOnnxValueInfo(
    Value* v,
    const onnx::ValueInfoProto& p_info,
    SymbolDimMap& symbol_dim_map,
    DimSymbolMap& dim_symbol_map) {
  if (!p_info.has_type()) {
    return;
  }

  const auto& p_type = p_info.type();
  if (p_type.has_tensor_type()) {
    const auto torch_tensor_type = TorchTensorTypeFromONNX(
        p_type.tensor_type(), symbol_dim_map, dim_symbol_map);
    if (torch_tensor_type) {
      MergeInferredTypeAndSetMap(v, v->type(), torch_tensor_type);
    }
  } else if (p_type.has_sequence_type()) {
    const auto torch_list_type = TorchListTypeFromONNX(
        p_type.sequence_type(), symbol_dim_map, dim_symbol_map);
    if (torch_list_type) {
      MergeInferredTypeAndSetMap(v, v->type(), torch_list_type);
    }
  }
}

bool IsValidONNXControlflowNode(const Node* n) {
  // Skip when block size is zero. This is when the node is being created,
  // and doesn't have subblocks attached yet. Run shape inference for these
  // nodes later, when the subgraph has already completed shape inferencing.
  auto node_kind = n->kind();
  if (node_kind == ::c10::onnx::Loop || node_kind == ::c10::onnx::If) {
    if (n->blocks().empty()) {
      return false;
    }
  }

  return true;
}

bool IsValidONNXNode(const Node* n) {
  auto node_kind = n->kind();

  if (!node_kind.is_onnx()) {
    // node kind is not ONNX, skipped.
    return false;
  }

  if (!IsValidONNXControlflowNode(n)) {
    return false;
  }

  for (auto b : n->blocks()) {
    for (auto b_n : b->nodes()) {
      if (!IsValidONNXNode(b_n)) {
        return false;
      }
    }
  }

  return true;
}

bool CustomSettype(Node* node) {
  // This is a helper function to decide if the non-ONNX node actually has
  // custom setType from user
  // Go through every symbolic_sizes and if any one of them is static, we say
  // this is set by user. On the other hand, if all of them are * (dynamic), we
  // take this node does not have given type, since unreliable nodes have *
  // shape anyway.
  auto all_output_has_type = [](Value* output) {
    if (auto output_type = output->type()->cast<TensorType>()) {
      if (auto sizes = output_type->symbolic_sizes().sizes()) {
        return std::any_of(std::begin(*sizes), std::end(*sizes), [](auto size) {
          return size.is_static();
        });
      }
    }
    return false;
  };

  return std::all_of(
      node->outputs().begin(), node->outputs().end(), all_output_has_type);
}

Value* CloneValueFromListConstruct(
    Value* v,
    const std::shared_ptr<Graph>& n_graph,
    int opset_version) {
  auto lc_node = v->node();
  TORCH_INTERNAL_ASSERT(lc_node->kind() == ::c10::prim::ListConstruct);
  // In jit/passes/onnx/peephole.cpp::eraseListConstruct,
  // prim::ListConstruct is converted to onnx::Concat. The conversion should
  // eventually be moved to symbolic. For now, treat this operator as
  // special case, and change from list type to tensor type. The scalar type
  // is preserved. If the elemtype is Int, insert a onnx::Concat node into
  // the graph.
  TypePtr elem = v->type()->castRaw<ListType>()->getElementType();
  std::optional<at::ScalarType> scalar_type = std::nullopt;
  if (elem->cast<IntType>()) {
    scalar_type = at::kLong;
    if (isValidToTransformToONNXConcatNode(v->node())) {
      auto concat_node = transformToONNXConcatNode(
          n_graph.get(), v->node(), true, opset_version);
      return concat_node->output();
    }
  } else if (elem->cast<FloatType>()) {
    scalar_type = at::kFloat;
  } else if (elem->cast<BoolType>()) {
    scalar_type = at::kBool;
  } else if (auto t_type = elem->cast<TensorType>()) {
    scalar_type = t_type->scalarType();
  }

  auto input = n_graph->addInput();
  if (scalar_type) {
    auto v_type = TensorType::create(
        scalar_type,
        at::kCPU,
        c10::SymbolicShape(),
        c10::VaryingShape<c10::Stride>{},
        {});
    input->setType(v_type);
  }
  return input;
}

// Clone the node n for the new graph.
Node* CloneNodeToGraph(
    Node* n,
    std::shared_ptr<Graph> n_graph,
    const ParamMap& params_dict,
    int opset_version) {
  auto clone_node = n_graph->createClone(
      n, [&n_graph, &params_dict, opset_version](Value* v) {
        auto v_n = v->node();
        switch (v_n->kind()) {
          case ::c10::prim::Constant:
          case ::c10::onnx::Constant: {
            // Clone the input if it is constant.
            auto constant_n = n_graph->insertNode(
                n_graph->createClone(v_n, [](Value* v) { return v; }));
            return constant_n->output();
          }
          case ::c10::prim::ListConstruct: {
            return CloneValueFromListConstruct(v, n_graph, opset_version);
          }
          case ::c10::prim::PackPadded: {
            auto input = n_graph->addInput();
            if (v == v_n->output(0)) {
              // Only the first output requires this workaround.
              // In `peephole` pass, user nodes are modified to consume the
              // input instead.
              input->copyMetadata(v_n->input(0));
            } else {
              input->copyMetadata(v);
            }
            return input;
          }
          default: {
            // Try to lookup input value and insert it into the graph.
            // If the input value is unknown, set it to graph input in the new
            // graph, and copy over metadata, such as datatype and shape.
            ::std::optional<at::Tensor> val = ::std::nullopt;
            auto v0 = params_dict.find(v->debugName());
            if (v0 != params_dict.end()) {
              val = v0->second.toTensor();
            } else {
              val = ConstantValueMap::GetValue(v->debugName());
            }

            if (val.has_value()) {
              return n_graph
                  ->insertNode(n_graph->create(::c10::onnx::Constant)
                                   ->t_(attr::value, val.value()))
                  ->output();
            }
            auto input = n_graph->addInput();
            input->copyMetadata(v);
            return input;
          }
        }
      });
  return clone_node;
}

bool HasValidType(const TypePtr& type, const std::string& name) {
  if (auto t_type = type->cast<TensorType>()) {
    if (!t_type->scalarType().has_value()) {
      GRAPH_UPDATE("Input ", name, " is missing tensor datatype.");
      return false;
    }
  } else if (auto s_type = type->cast<ListType>()) {
    auto e_type = s_type->getElementType();
    return HasValidType(e_type, name);
  } else if (auto o_type = type->cast<OptionalType>()) {
    auto e_type = o_type->getElementType();
    return HasValidType(e_type, name);
  }
  return true;
}

bool IsGraphValidForInference(const std::shared_ptr<Graph>& graph) {
  // Verify if every input has type (either Tensor, Sequence or Optional) and
  // scalar type. This is a requirement for ONNX graph inputs.
  for (auto in : graph->inputs()) {
    return HasValidType(in->type(), in->debugName());
  }
  return true;
}

void ConvertGraphToONNXProto(
    const std::shared_ptr<Graph>& graph,
    std::shared_ptr<onnx::ModelProto>& model_proto,
    SymbolDimMap& symbol_dim_map,
    DimSymbolMap& dim_symbol_map,
    int opset_version) {
  auto
      [model_proto_tmp,
       export_map,
       new_symbol_dim_map,
       val_use_external_data_format,
       node_names] =
          export_onnx(
              graph,
              {},
              opset_version,
              {},
              false,
              onnx_torch::OperatorExportTypes::ONNX,
              true,
              true,
              {},
              true,
              false,
              std::string());
  model_proto = std::move(model_proto_tmp);
  symbol_dim_map.insert(new_symbol_dim_map.begin(), new_symbol_dim_map.end());
  for (const auto& pair : new_symbol_dim_map) {
    dim_symbol_map[pair.second] = pair.first;
  }
  for (int i = 0; i < model_proto->graph().output_size(); ++i) {
    model_proto->mutable_graph()->mutable_output(i)->clear_type();
  }
}

std::optional<at::Tensor> ComputeConstantFolding(
    const Node* n,
    int opset_version) {
  if (n->inputs().empty()) {
    return std::nullopt;
  }
  std::vector<at::Tensor> inputTensorValues;
  for (auto i : c10::irange(n->inputs().size())) {
    if (TensorTypePtr input_type = n->input(i)->type()->cast<TensorType>()) {
      if (!ConstantValueMap::HasValue(n->input(i)->debugName())) {
        return std::nullopt;
      }
      auto tensor_value =
          ConstantValueMap::GetValue(n->input(i)->debugName()).value();
      inputTensorValues.emplace_back(tensor_value);
    }
  }
  if (inputTensorValues.size() < n->inputs().size()) {
    return std::nullopt;
  }
  try {
    return onnx_constant_fold::runTorchBackendForOnnx(
        n, inputTensorValues, opset_version);
  } catch (const std::exception& ex) {
    auto ex_str = std::string(ex.what());
    ex_str = ex_str.substr(0, ex_str.find('\n'));
    TORCH_WARN("Constant folding in symbolic shape inference fails: ", ex_str);
    return std::nullopt;
  }
}

// Similar to the function above, but for symbolic shapes.
std::optional<::c10::SymbolicShape> ComputeShapeFromReshape(
    Node* n,
    const c10::SymbolicShape& input_shape,
    const c10::SymbolicShape& shape,
    int opset_version) {
  std::vector<c10::ShapeSymbol> input_shape_vector =
      input_shape.sizes().value();
  std::vector<c10::ShapeSymbol> shape_vector = shape.sizes().value();
  TORCH_INTERNAL_ASSERT(
      !input_shape_vector.empty() || !shape_vector.empty(),
      "Reshape node should have at least one input size > 0 when constant folding.");
  if (shape_vector.empty()) {
    return input_shape;
  }
  if (input_shape_vector.empty()) {
    return shape;
  }

  auto is_zero = [](c10::ShapeSymbol& ss) { return ss.value() == 0; };
  auto it_0 = std::find_if(shape_vector.begin(), shape_vector.end(), is_zero);
  bool shape_has_zero = it_0 != shape_vector.end();

  int64_t minus_one_pos = -1;
  for (auto i : c10::irange(shape_vector.size())) {
    if (shape_vector[i].value() == -1) {
      minus_one_pos = i;
      break;
    }
  }

  int allowzero = 0;
  if (opset_version >= 14 && n->hasAttributeS("allowzero")) {
    allowzero = n->i(attr::allowzero);
  }

  TORCH_CHECK(
      !(shape_has_zero && allowzero == 1 && minus_one_pos != -1),
      "0 and -1 cannot both be present in `Shape` input of `Reshape` node, when `allowzero=1`.");

  if (minus_one_pos == -1 && (!shape_has_zero || allowzero)) {
    return shape;
  }
  std::vector<c10::ShapeSymbol> final_shape;
  uint64_t shape_ratio = 1;
  std::unordered_map<int64_t, int64_t> sym_map;
  for (const c10::ShapeSymbol& input_shape : input_shape_vector) {
    // input_shape.static_size() could be zero when torch.tensor([]) is used.
    if (input_shape.is_static() && input_shape.static_size() != 0) {
      if (shape_ratio >=
          std::numeric_limits<uint64_t>::max() / input_shape.static_size()) {
        TORCH_WARN(
            "ComputeShapeFromReshape(), shape_ratio overflows, skip shape inference.");
        return std::nullopt;
      } else {
        shape_ratio *= static_cast<uint64_t>(input_shape.static_size());
      }
    } else {
      auto value = input_shape.value();
      sym_map.emplace(value, 0).first->second += 1;
    }
  }
  int shape_size = static_cast<int>(shape_vector.size());
  for (const int i : c10::irange(shape_size)) {
    if (i == minus_one_pos) {
      continue;
    }
    c10::ShapeSymbol& target_shape = shape_vector[i];
    if (target_shape.value() == 0) {
      target_shape = input_shape_vector[i];
    }
    if (target_shape.is_static()) {
      shape_ratio /= static_cast<uint64_t>(target_shape.static_size());
    } else {
      auto value = target_shape.value();
      if (sym_map.find(value) == sym_map.end()) {
        return std::nullopt;
      }
      sym_map[value]--;
      if (sym_map[value] == 0) {
        sym_map.erase(value);
      }
    }
  }

  // sym_map is used to match shape symbols between the input and shape.
  // If there is a mismatch, the output shape cannot be estimated.
  if (!sym_map.empty()) {
    return std::nullopt;
  }

  TORCH_INTERNAL_ASSERT(
      minus_one_pos != -1,
      "There are no examples for shape_has_zero = true && minus_one_pos == -1.");

  for (const auto i : c10::irange(minus_one_pos)) {
    c10::ShapeSymbol cur_shape(
        shape_vector[i].value() == 0 ? input_shape_vector[i] : shape_vector[i]);
    final_shape.push_back(cur_shape);
  }
  if (minus_one_pos != -1) {
    final_shape.push_back(
        c10::ShapeSymbol::fromStaticSize(static_cast<int64_t>(shape_ratio)));
  }
  for (auto i = minus_one_pos + 1; i < shape_size; i++) {
    c10::ShapeSymbol cur_shape(
        shape_vector[i].value() == 0 ? input_shape_vector[i] : shape_vector[i]);
    final_shape.push_back(cur_shape);
  }
  c10::SymbolicShape final_shape_0(final_shape);
  return final_shape_0;
}

std::optional<::c10::SymbolicShape> ComputeShapeFromExpand(
    const std::vector<::c10::ShapeSymbol>& input_shape,
    const std::vector<int64_t>& reshape) {
  for (const auto& it : reshape) {
    if (it < 0) {
      return std::nullopt;
    }
  }
  std::vector<::c10::ShapeSymbol> final_shape;
  if (input_shape.size() >= reshape.size()) {
    final_shape = input_shape;
  } else {
    for (auto v : reshape) {
      final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(v));
    }
  }
  auto min_size = std::min(input_shape.size(), reshape.size());
  for (const auto i : c10::irange(min_size)) {
    auto idx = final_shape.size() - i - 1;
    auto input_shape_idx = input_shape.size() - i - 1;
    auto reshape_idx = reshape.size() - i - 1;
    if (input_shape[input_shape_idx].is_static()) {
      auto input_shape_value = input_shape[input_shape_idx].static_size();
      auto reshape_value = reshape[reshape_idx];
      TORCH_INTERNAL_ASSERT(
          input_shape_value == reshape_value || input_shape_value == 1 ||
              reshape_value == 1,
          "ONNX Expand input shape constraint not satisfied.");
      final_shape[idx] = ::c10::ShapeSymbol::fromStaticSize(
          std::max(input_shape_value, reshape_value));

    } else {
      final_shape[idx] = ::c10::ShapeSymbol::newSymbol();
    }
  }
  ::c10::SymbolicShape shape(final_shape);
  return shape;
}

std::optional<::c10::SymbolicShape> ComputeShapeFromTile(
    const std::vector<::c10::ShapeSymbol>& input_shape,
    const std::vector<int64_t>& reshape) {
  TORCH_INTERNAL_ASSERT(
      input_shape.size() == reshape.size(),
      "ONNX Tile input shapes do not match.");
  for (const auto& it : reshape) {
    if (it < 0) {
      return std::nullopt;
    }
  }
  std::vector<::c10::ShapeSymbol> final_shape;
  final_shape.reserve(input_shape.size());
  for (const auto i : c10::irange(input_shape.size())) {
    if (input_shape[i].is_static()) {
      final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(
          input_shape[i].static_size() * reshape[i]));
    } else {
      final_shape.emplace_back(::c10::ShapeSymbol::newSymbol());
    }
  }
  ::c10::SymbolicShape shape(final_shape);
  return shape;
}

void UpdateRank(Value* value, size_t rank) {
  ConstantValueMap::SetRank(value->debugName(), rank);
  if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
    std::optional<size_t> rank_opt = rank;
    auto shape = ::c10::SymbolicShape(rank_opt);
    value->setType(value_type->withSymbolicShapes(shape));
  }
}

void UpdateShapeFromVector(
    Value* value,
    const std::vector<int64_t>& shape_size) {
  ::c10::SymbolicShape shape(shape_size);
  ConstantValueMap::SetShape(value->debugName(), shape);
  if (shape_size.empty()) {
    UpdateRank(value, 0);
    return;
  }
  ConstantValueMap::SetRank(value->debugName(), shape_size.size());
  if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
    value->setType(value_type->withSymbolicShapes(shape));
  }
}

void UpdateShape(Value* value, const ::c10::SymbolicShape& shape) {
  ConstantValueMap::SetShape(value->debugName(), shape);
  if (shape.rank().has_value()) {
    auto rank = shape.rank().value();
    if (rank == 0) {
      UpdateRank(value, 0);
      return;
    }
    ConstantValueMap::SetRank(value->debugName(), rank);
    if (TensorTypePtr value_type = value->type()->cast<TensorType>()) {
      value->setType(value_type->withSymbolicShapes(shape));
    }
  }
}

void UpdateShapeConstantValueMap(
    const Value* value,
    const ::c10::SymbolicShape& shape) {
  ConstantValueMap::SetShape(value->debugName(), shape);
  if (shape.rank().has_value()) {
    auto rank = shape.rank().value();
    ConstantValueMap::SetRank(value->debugName(), rank);
  }
}

std::optional<std::vector<int64_t>> GetValueFromListConstructNode(
    Node* lc_node) {
  std::vector<int64_t> shape_size;
  for (const auto& input : lc_node->inputs()) {
    if (input->type()->cast<TensorType>() &&
        ConstantValueMap::HasValue(input->debugName())) {
      auto lc_value = ConstantValueMap::GetValue(input->debugName()).value();
      if (lc_value.dim() == 0) {
        int64_t lc_value_0 = lc_value.item<int64_t>();
        shape_size.emplace_back(lc_value_0);
      }
    }
  }
  return lc_node->inputs().size() == shape_size.size()
      ? std::optional<std::vector<int64_t>>(shape_size)
      : std::nullopt;
}

void SetShapeValueFromListConstructNode(Node* lc_node) {
  std::vector<c10::ShapeSymbol> shape_size;
  for (const auto& input : lc_node->inputs()) {
    if (TensorTypePtr shape_type = input->type()->cast<TensorType>()) {
      if (ConstantValueMap::HasValue(input->debugName())) {
        auto lc_value = ConstantValueMap::GetValue(input->debugName()).value();
        if (lc_value.dim() == 0) {
          int64_t lc_value_0 = lc_value.item<int64_t>();
          shape_size.emplace_back(c10::ShapeSymbol::fromStaticSize(lc_value_0));
        }
      } else if (ConstantValueMap::HasShapeValue(input->debugName())) {
        auto lc_value =
            ConstantValueMap::GetShapeValue(input->debugName()).value();
        if (lc_value.rank() == 1U) {
          shape_size.emplace_back(lc_value.at(0));
        }
      }
    }
  }
  if (lc_node->inputs().size() == shape_size.size()) {
    c10::SymbolicShape final_shape(shape_size);
    ConstantValueMap::SetShapeValue(
        lc_node->output()->debugName(), final_shape);
  }
}

std::vector<::c10::ShapeSymbol> Broadcast(
    const std::vector<::c10::ShapeSymbol>& input_shape_value_0,
    const std::vector<::c10::ShapeSymbol>& input_shape_value_1) {
  size_t rank_0 = input_shape_value_0.size();
  size_t rank_1 = input_shape_value_1.size();
  size_t rank_max = std::max(rank_0, rank_1);
  size_t rank_min = std::min(rank_0, rank_1);
  std::vector<::c10::ShapeSymbol> final_shape;
  final_shape.reserve(rank_max);
  std::generate_n(
      std::back_inserter(final_shape), rank_max, ::c10::ShapeSymbol::newSymbol);
  for (auto idx : c10::irange(rank_min)) {
    const c10::ShapeSymbol& ss_shape_0 = input_shape_value_0[rank_0 - 1 - idx];
    const c10::ShapeSymbol& ss_shape_1 = input_shape_value_1[rank_1 - 1 - idx];
    bool is_static_0 = ss_shape_0.is_static();
    bool is_static_1 = ss_shape_1.is_static();
    size_t shape_idx = rank_max - 1 - idx;
    if (is_static_0 && is_static_1) {
      int64_t static_0_sz = ss_shape_0.static_size();
      int64_t static_1_sz = ss_shape_1.static_size();
      // condition for corner case of 0d tensor
      // 0d tensor with 1d tensor would give us 0d tensor
      if (std::min(static_0_sz, static_1_sz) == 0) {
        final_shape[shape_idx] = ::c10::ShapeSymbol::fromStaticSize(
            std::min(static_0_sz, static_1_sz));
      } else {
        final_shape[shape_idx] = ::c10::ShapeSymbol::fromStaticSize(
            std::max(static_0_sz, static_1_sz));
      }
    } else if (!is_static_0 && !is_static_1) {
      if (ss_shape_0.value() == ss_shape_1.value()) {
        final_shape[shape_idx] = ss_shape_0;
      }
    }
  }
  if (rank_0 < rank_1) {
    for (size_t idx = rank_min; idx < rank_max; idx++) {
      size_t shape_idx = rank_max - 1 - idx;
      final_shape[shape_idx] = input_shape_value_1[shape_idx];
    }
  } else {
    for (size_t idx = rank_min; idx < rank_max; idx++) {
      size_t shape_idx = rank_max - 1 - idx;
      final_shape[shape_idx] = input_shape_value_0[shape_idx];
    }
  }
  return final_shape;
}

void ProcessBroadcastNode(Node* n) {
  TORCH_INTERNAL_ASSERT(n->inputs().size() == 2);
  if (ConstantValueMap::HasShape(n->input(0)->debugName()) &&
      ConstantValueMap::HasShape(n->input(1)->debugName())) {
    auto input_shape_0 = ConstantValueMap::GetShape(n->input(0)->debugName());
    auto input_shape_value_0 = input_shape_0.value().sizes().value();
    auto input_shape_1 = ConstantValueMap::GetShape(n->input(1)->debugName());
    auto input_shape_value_1 = input_shape_1.value().sizes().value();
    auto final_shape = Broadcast(input_shape_value_0, input_shape_value_1);
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
}

void ProcessShapeForConcatNode(Node* n) {
  auto axis = n->i(attr::axis);
  if (ConstantValueMap::HasRank(n->input(0)->debugName())) {
    auto rank = ConstantValueMap::GetRank(n->input(0)->debugName()).value();
    size_t axis_adjust = 0;
    if (axis >= 0) {
      axis_adjust = static_cast<size_t>(axis);
    } else {
      axis_adjust = static_cast<size_t>(axis + static_cast<int>(rank));
    }
    std::vector<::c10::ShapeSymbol> final_shape;
    final_shape.reserve(rank);
    for (auto idx : c10::irange(rank)) {
      if (idx == axis_adjust) {
        auto flag = true;
        int64_t size_total = 0;
        for (auto input_idx : c10::irange(n->inputs().size())) {
          if (ConstantValueMap::HasShape(n->input(input_idx)->debugName())) {
            auto input_shape =
                ConstantValueMap::GetShape(n->input(input_idx)->debugName());
            auto input_shape_value = input_shape.value().sizes();
            auto shape_symbol = input_shape_value.value()[idx];
            if (shape_symbol.is_static()) {
              size_total += shape_symbol.static_size();
            } else {
              flag = false;
              break;
            }
          }
        }
        if (flag) {
          final_shape.emplace_back(
              ::c10::ShapeSymbol::fromStaticSize(size_total));
        } else {
          final_shape.emplace_back(::c10::ShapeSymbol::newSymbol());
        }
      } else {
        auto flag = false;
        for (auto input_idx : c10::irange(n->inputs().size())) {
          if (ConstantValueMap::HasShape(n->input(input_idx)->debugName())) {
            auto input_shape =
                ConstantValueMap::GetShape(n->input(input_idx)->debugName());
            auto input_shape_value = input_shape.value().sizes();
            auto shape_symbol = input_shape_value.value()[idx];
            if (shape_symbol.is_static()) {
              final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(
                  shape_symbol.static_size()));
              flag = true;
              break;
            }
          }
        }
        if (!flag) {
          final_shape.emplace_back(::c10::ShapeSymbol::newSymbol());
        }
      }
    }
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
}

void ProcessShapeValueForConcatNode(Node* n) {
  auto rank = n->inputs().size();
  std::vector<c10::ShapeSymbol> shape_size;
  for (const auto& input : n->inputs()) {
    if (ConstantValueMap::HasValue(input->debugName())) {
      auto concat_value =
          ConstantValueMap::GetValue(input->debugName()).value();
      if (concat_value.dim() == 1 && concat_value.size(0) == 1) {
        auto concat_value_0 = concat_value[0].item<int64_t>();
        shape_size.emplace_back(
            c10::ShapeSymbol::fromStaticSize(concat_value_0));
      }
    } else if (ConstantValueMap::HasShapeValue(input->debugName())) {
      auto concat_value =
          ConstantValueMap::GetShapeValue(input->debugName()).value();
      if (concat_value.rank() == 1U) {
        shape_size.emplace_back(concat_value.at(0));
      }
    }
  }
  if (rank == shape_size.size()) {
    c10::SymbolicShape final_shape(shape_size);
    ConstantValueMap::SetShapeValue(n->output(0)->debugName(), final_shape);
  }
}

void ProcessConcatNode(Node* n) {
  ProcessShapeForConcatNode(n);
  ProcessShapeValueForConcatNode(n);
}

void ProcessMatMulNode(Node* n) {
  if (ConstantValueMap::HasShape(n->input(0)->debugName()) &&
      ConstantValueMap::HasShape(n->input(1)->debugName())) {
    auto input_shape_0 =
        ConstantValueMap::GetShape(n->input(0)->debugName()).value();
    auto input_shape_value_0 = input_shape_0.sizes().value();
    auto input_shape_1 =
        ConstantValueMap::GetShape(n->input(1)->debugName()).value();
    auto input_shape_value_1 = input_shape_1.sizes().value();
    size_t rank_0 = input_shape_value_0.size();
    size_t rank_1 = input_shape_value_1.size();
    // Handle inputs of rank 1 just like numpy.matmul:
    // https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    auto is_rank_0_1 = false;
    if (rank_0 == 1) {
      input_shape_value_0.insert(
          input_shape_value_0.begin(), ::c10::ShapeSymbol::fromStaticSize(1));
      rank_0 = 2;
      is_rank_0_1 = true;
    }
    auto is_rank_1_1 = false;
    if (rank_1 == 1) {
      input_shape_value_1.emplace_back(::c10::ShapeSymbol::fromStaticSize(1));
      rank_1 = 2;
      is_rank_1_1 = true;
    }
    // Per https://pytorch.org/docs/stable/generated/torch.matmul.html
    // the broadcasting logic only applies to the batch dimensions, and not the
    // matrix dimensions so we remove the matrix dimensions which are the last 2
    // dimensions before broadcasting
    auto final_shape = Broadcast(
        std::vector<::c10::ShapeSymbol>(
            input_shape_value_0.begin(), input_shape_value_0.end() - 2),
        std::vector<::c10::ShapeSymbol>(
            input_shape_value_1.begin(), input_shape_value_1.end() - 2));
    // add the last 2 dimensions back, unless they do not exist in the first
    // place and inserted by this function Then apply [n,k]X[k,m]=[n,m], where
    // n=input_shape_value_0[rank_0 - 2], m=input_shape_value_1[rank_1 - 1]
    if (!is_rank_0_1) {
      final_shape.emplace_back(input_shape_value_0[rank_0 - 2]);
    }
    if (!is_rank_1_1) {
      final_shape.emplace_back(input_shape_value_1[rank_1 - 1]);
    }
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
}

void ProcessReduceNode(Node* n) {
  if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
    auto input_shape_0 = ConstantValueMap::GetShape(n->input(0)->debugName());
    auto input_shape_value_0 = input_shape_0.value().sizes();
    size_t rank_0 = input_shape_value_0.value().size();
    std::vector<::c10::ShapeSymbol> final_shape;
    std::vector<int64_t> axes_vector(rank_0);
    if (n->hasAttributeS("axes")) {
      axes_vector = n->is(attr::axes);
    } else if (n->inputs().size() > 1) {
      axes_vector =
          ConstantValueMap::GetValueInto1DInt64Vector(n->input(1)->debugName());
    } else {
      std::iota(axes_vector.begin(), axes_vector.end(), 0);
    }

    for (auto idx : c10::irange(axes_vector.size())) {
      if (axes_vector[idx] < 0) {
        axes_vector[idx] += rank_0;
      }
    }
    final_shape.reserve(rank_0);
    // ONNX keepdims defaults to 1 when not set.
    int64_t keepdims = 1;
    if (n->hasAttributeS("keepdims")) {
      keepdims = n->i(attr::keepdims);
    }
    for (auto idx : c10::irange(rank_0)) {
      auto it = std::find(axes_vector.begin(), axes_vector.end(), idx);
      if (it != axes_vector.end()) {
        if (keepdims != 0) {
          final_shape.emplace_back(::c10::ShapeSymbol::fromStaticSize(1));
        }
      } else {
        final_shape.emplace_back(input_shape_value_0.value()[idx]);
      }
    }
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
}

void ProcessReshapeNode(Node* n, int opset_version) {
  const auto& input_name = n->input(0)->debugName();
  const auto& shape_name = n->input(1)->debugName();

  // When `shape` input value is statically known, compute output shape.
  if (ConstantValueMap::HasValue(shape_name)) {
    auto static_shape_value =
        ConstantValueMap::GetValueInto1DInt64Vector(shape_name);
    auto symbolic_input_shape = ConstantValueMap::GetShape(input_name);
    if (symbolic_input_shape && !static_shape_value.empty()) {
      auto final_shape = ComputeShapeFromReshape(
          n,
          symbolic_input_shape.value(),
          c10::SymbolicShape(static_shape_value),
          opset_version);
      if (final_shape) {
        UpdateShape(n->output(), final_shape.value());
        return;
      }
    }
  }

  // When `shape` input value is symbolically known, compute output shape.
  if (ConstantValueMap::HasShapeValue(shape_name) &&
      ConstantValueMap::HasShape(input_name)) {
    auto symbolic_input_shape = ConstantValueMap::GetShape(input_name).value();
    auto symbolic_shape_value =
        ConstantValueMap::GetShapeValue(shape_name).value();
    auto final_shape = ComputeShapeFromReshape(
        n, symbolic_input_shape, symbolic_shape_value, opset_version);
    if (final_shape.has_value()) {
      UpdateShape(n->output(), final_shape.value());
      return;
    }
  }

  // Only shape of new shape is known, assign output rank.
  if (ConstantValueMap::HasShape(shape_name)) {
    auto output_rank = ConstantValueMap::GetShapeInto1DInt64Vector(shape_name);
    if (output_rank.has_value()) {
      TORCH_INTERNAL_ASSERT(output_rank.value().size() == 1);
      UpdateRank(n->output(), output_rank.value()[0]);
      return;
    }
  }

  // ListConstruct is handled at the beginning of ProcessConstantValueMap, no
  // further process here.
  if (TensorTypePtr shape_type = n->input(1)->type()->cast<TensorType>()) {
    // Set rank to Reshape output if possible.
    // From shape inference, we have:
    // %4236 : Float(*, device=cpu) = onnx::Transpose[perm=[0]](%4235)
    // %4237 : Long(2, strides=[1], device=cpu) = onnx::Concat[axis=0](%4232)
    // %4238 : FloatTensor(device=cpu) = onnx::Reshape(%4236, %4237)
    // We can have it as SymbolicShape with known rank:
    // %4238 : Float(*, *, strides=[2480, 1], requires_grad=0, device=cpu) =
    // onnx::Reshape(%4236, %4237)
    auto shape_type_dim = shape_type->dim();
    if (shape_type_dim.has_value()) {
      auto shape_type_size = shape_type->sizes()[0];
      if (shape_type_size.has_value()) {
        size_t rank = shape_type_size.value();
        UpdateRank(n->output(), rank);
      }
    }
  }
}

c10::SymbolicShape ComputeShapeForSlice(
    const std::vector<c10::ShapeSymbol>& input_shape,
    const std::vector<int64_t>& start_vector,
    const std::vector<int64_t>& end_vector,
    const std::vector<int64_t>& axes_vector,
    const std::vector<int64_t>& step_vector) {
  TORCH_INTERNAL_ASSERT(axes_vector.size() <= input_shape.size());
  TORCH_INTERNAL_ASSERT(axes_vector.size() == start_vector.size());
  TORCH_INTERNAL_ASSERT(axes_vector.size() == end_vector.size());
  TORCH_INTERNAL_ASSERT(axes_vector.size() == step_vector.size());
  std::vector<c10::ShapeSymbol> final_shape;
  final_shape = input_shape;
  for (const auto idx : c10::irange(axes_vector.size())) {
    auto axis = axes_vector[idx];
    TORCH_INTERNAL_ASSERT(axis >= 0);
    if (!input_shape[axis].is_static()) {
      final_shape[axis] = c10::ShapeSymbol::newSymbol();
      continue;
    }
    auto input_shape_axis_value = input_shape[axis].static_size();
    auto cur_start = start_vector[idx];
    auto cur_end = end_vector[idx];
    auto cur_step = step_vector[idx];
    if (cur_start < -input_shape_axis_value) {
      cur_start = 0;
    } else if (cur_start < 0) {
      cur_start = input_shape_axis_value + cur_start;
    } else if (cur_start > input_shape_axis_value - 1) {
      cur_start = input_shape_axis_value;
    }
    if (cur_end < -input_shape_axis_value) {
      cur_end = -1;
    } else if (cur_end < 0) {
      cur_end = input_shape_axis_value + cur_end;
    } else if (cur_end > input_shape_axis_value - 1) {
      cur_end = input_shape_axis_value;
    }
    TORCH_INTERNAL_ASSERT(cur_step != 0);
    if (cur_step > 0) {
      final_shape[axis] = c10::ShapeSymbol::fromStaticSize(
          (cur_end - cur_start - 1) / cur_step + 1);
    } else {
      final_shape[axis] = c10::ShapeSymbol::fromStaticSize(
          (cur_start - cur_end - 1) / (-cur_step) + 1);
    }
  }
  return c10::SymbolicShape(final_shape);
}

void ProcessSliceNode(Node* n, int opset_version) {
  bool valid = ConstantValueMap::HasShape(n->input(0)->debugName());

  // For opset version <= 9, starts, ends, axes, steps are attributes,
  // so their values are always valid.
  if (opset_version >= 10) {
    // We can only infer shapes if 'axes' is known.
    if (n->inputs().size() > 3) {
      valid = valid && ConstantValueMap::HasValue(n->input(3)->debugName());
    }
  }

  if (!valid) {
    if (ConstantValueMap::HasRank(n->input(0)->debugName())) {
      auto rank = ConstantValueMap::GetRank(n->input(0)->debugName()).value();
      UpdateRank(n->output(), rank);
    }
    return;
  } else {
    auto shape_size_0 =
        ConstantValueMap::GetShape(n->input(0)->debugName()).value();
    if (shape_size_0.rank().has_value()) {
      auto input0_shape_value = shape_size_0.sizes().value();

      std::vector<int64_t> start_vector;
      std::vector<int64_t> end_vector;
      std::vector<int64_t> step_vector;

      std::vector<int64_t> axes_vector(input0_shape_value.size(), 0);
      for (const auto i : c10::irange(input0_shape_value.size())) {
        axes_vector[i] = i;
      }
      if (opset_version >= 10 && n->inputs().size() > 3) {
        axes_vector = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input(3)->debugName());
      } else if (opset_version < 10 && n->hasAttributeS("axes")) {
        axes_vector = n->is(attr::axes);
      }
      for (auto& axis : axes_vector) {
        if (axis < 0) {
          axis += input0_shape_value.size();
        }
      }

      if (opset_version < 10) {
        start_vector = n->is(attr::starts);
        end_vector = n->is(attr::ends);
      } else {
        // If starts, ends, or step are unknown,
        // then mark all dimensions in 'axes' as unknown.
        std::vector<uint64_t> indices = {1U, 2U, 4U};
        bool start_end_step_known =
            std::all_of(indices.begin(), indices.end(), [&n](auto i) {
              return (i >= n->inputs().size()) ||
                  ConstantValueMap::HasValue(n->input(i)->debugName());
            });
        if (!start_end_step_known) {
          auto final_shape = input0_shape_value;
          for (const auto axis : axes_vector) {
            final_shape[axis] = c10::ShapeSymbol::newSymbol();
          }
          UpdateShape(n->output(), final_shape);
          return;
        }

        start_vector = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input(1)->debugName());
        end_vector = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input(2)->debugName());
        if (n->inputs().size() > 4) {
          step_vector = ConstantValueMap::GetValueInto1DInt64Vector(
              n->input(4)->debugName());
        }
      }

      if (step_vector.empty()) {
        step_vector = std::vector<int64_t>(axes_vector.size(), 1);
      }

      auto final_shape = ComputeShapeForSlice(
          input0_shape_value,
          start_vector,
          end_vector,
          axes_vector,
          step_vector);
      UpdateShape(n->output(), final_shape);
    }
  }
}

void ProcessUnchangeNode(Node* n) {
  if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
    auto shape_size_0 =
        ConstantValueMap::GetShape(n->input(0)->debugName()).value();
    UpdateShape(n->output(), shape_size_0);
  }
}

void ProcessTimeSeriesNode(Node* n) {
  auto input0_shape = ConstantValueMap::GetShape(n->input(0)->debugName());
  auto input1_shape = ConstantValueMap::GetShape(n->input(1)->debugName());
  if (!(input0_shape.has_value() && input1_shape.has_value())) {
    return;
  }
  auto input0_shape_value = input0_shape.value().sizes();
  auto input1_shape_value = input1_shape.value().sizes();
  c10::ShapeSymbol seq_length;
  c10::ShapeSymbol num_directions;
  c10::ShapeSymbol batch_size;
  c10::ShapeSymbol hidden_size;
  if (input0_shape_value.has_value()) {
    seq_length = input0_shape_value.value()[0];
    batch_size = input0_shape_value.value()[1];
  }

  if (input1_shape_value.has_value()) {
    num_directions = input1_shape_value.value()[0];
    if (input1_shape_value.value()[1].is_static()) {
      auto input1_value = input1_shape_value.value()[1].static_size();
      switch (n->kind()) {
        case ::c10::onnx::RNN:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value);
          break;
        case ::c10::onnx::LSTM:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value / 4);
          break;
        case ::c10::onnx::GRU:
          hidden_size = c10::ShapeSymbol::fromStaticSize(input1_value / 3);
          break;
        default:
          throw std::runtime_error(
              std::string() + "This is not a valid TimeSeries Node with type " +
              n->kind().toDisplayString());
      }
    } else {
      hidden_size = c10::ShapeSymbol::newSymbol();
    }
  }

  if (n->outputs().size() > 1) {
    std::vector<c10::ShapeSymbol> final_shape = {
        seq_length, num_directions, batch_size, hidden_size};
    UpdateShape(n->output(0), c10::SymbolicShape(final_shape));
  }
  for (const auto idx : c10::irange(2U, 4U)) {
    if (n->outputs().size() > idx) {
      std::vector<c10::ShapeSymbol> final_shape = {
          num_directions, batch_size, hidden_size};
      UpdateShape(n->output(idx - 1), c10::SymbolicShape(final_shape));
    }
  }
}

void ProcessUnsqueezeNode(Node* n) {
  TensorTypePtr output_type = n->output(0)->type()->cast<TensorType>();
  if (output_type == nullptr) {
    return;
  }
  if (output_type->dim().has_value() && output_type->dim().value() == 1 &&
      ConstantValueMap::HasShapeValue(n->input(0)->debugName())) {
    auto shape_value =
        ConstantValueMap::GetShapeValue(n->input(0)->debugName()).value();
    // When the scalar represents a shape, it is the same as the shape value
    // when it gets unsqueezed.
    ConstantValueMap::SetShapeValue(n->output()->debugName(), shape_value);
  }
}

// As an addition to onnx shape inference, this function leverages constant
// folding and a per-Op based process to update rank/shape for the graph, also
// it update ConstantValueMap accordingly.
void ComputeConstant(Node* n, int opset_version) {
  if (n->kind() == ::c10::onnx::Constant) {
    if (n->kindOf(attr::value) == AttributeKind::t) {
      const at::Tensor& const_val = n->t(attr::value);
      at::Tensor const_val_copy =
          at::empty(const_val.sizes(), const_val.options());
      const_val_copy.copy_(const_val);
      ConstantValueMap::SetValue(n->output()->debugName(), const_val_copy);
    }
    return;
  }
  auto only_rank_available = false;
  size_t rank = 0;

  // Constant folding.
  auto const_fold_val = ComputeConstantFolding(n, opset_version);
  if (const_fold_val.has_value()) {
    at::Tensor const_fold_val_copy = at::empty(
        const_fold_val.value().sizes(), const_fold_val.value().options());
    const_fold_val_copy.copy_(const_fold_val.value());
    ConstantValueMap::SetValue(n->output()->debugName(), const_fold_val_copy);
    UpdateShapeFromVector(n->output(), const_fold_val_copy.sizes().vec());
    return;
  }

  switch (n->kind()) {
    case ::c10::onnx::Add:
    case ::c10::onnx::Div:
    case ::c10::onnx::Equal:
    case ::c10::onnx::Greater:
    case ::c10::onnx::GreaterOrEqual:
    case ::c10::onnx::Less:
    case ::c10::onnx::LessOrEqual:
    case ::c10::onnx::Mod:
    case ::c10::onnx::Mul:
    case ::c10::onnx::Pow:
    case ::c10::onnx::Sub: {
      ProcessBroadcastNode(n);
      break;
    }
    case ::c10::onnx::Shape: {
      auto input_shape =
          ConstantValueMap::GetShapeInto1DInt64Vector(n->input()->debugName());
      if (input_shape.has_value()) {
        auto shape_value = input_shape.value();
        // TODO: getDevice() ?
        auto options = c10::TensorOptions().dtype(at::kLong).device(at::kCPU);
        auto shape_value_size = static_cast<int64_t>(shape_value.size());
        auto f =
            at::from_blob(shape_value.data(), {shape_value_size}, at::kLong)
                .to(at::kCPU);
        // Need copy here
        at::Tensor f_copy = at::empty({shape_value_size}, options);
        f_copy.copy_(f);
        ConstantValueMap::SetValue(n->output()->debugName(), f_copy);
        std::vector<::c10::ShapeSymbol> final_shape_vector(
            1, c10::ShapeSymbol::fromStaticSize(shape_value_size));
        ::c10::SymbolicShape final_shape(final_shape_vector);
        UpdateShape(n->output(), final_shape);
      }
      break;
    }
    case ::c10::onnx::Reshape: {
      ProcessReshapeNode(n, opset_version);
      break;
    }
    case ::c10::onnx::Transpose: {
      if (n->hasAttributeS("perm")) {
        auto perm_v = n->is(attr::perm);
        rank = perm_v.size();
        auto is_default_perm = false;
        if (rank == 2 && perm_v[0] == 1 && perm_v[1] == 0) {
          is_default_perm = true;
        }
        auto shape_updated = false;
        if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
          auto shape_size_0 =
              ConstantValueMap::GetShape(n->input(0)->debugName())
                  .value()
                  .sizes();
          if (shape_size_0.has_value()) {
            auto shape_vector_0 = shape_size_0.value();
            std::vector<::c10::ShapeSymbol> final_shape_vector(
                shape_vector_0.size(), ::c10::ShapeSymbol());
            if (is_default_perm) {
              std::reverse_copy(
                  std::begin(shape_vector_0),
                  std::end(shape_vector_0),
                  std::begin(final_shape_vector));
            } else {
              for (const auto i : c10::irange(shape_vector_0.size())) {
                final_shape_vector[i] = shape_vector_0[perm_v[i]];
              }
            }
            ::c10::SymbolicShape final_shape(final_shape_vector);
            UpdateShape(n->output(), final_shape);
            shape_updated = true;
          }
        }
        if (!shape_updated) {
          if (!is_default_perm) {
            only_rank_available = true;
          } else if (ConstantValueMap::HasRank(n->input(0)->debugName())) {
            rank = ConstantValueMap::GetRank(n->input(0)->debugName()).value();
            only_rank_available = true;
          }
        }
      }
      break;
    }
    case ::c10::onnx::Concat: {
      ProcessConcatNode(n);
      break;
    }
    case ::c10::onnx::ConstantOfShape: {
      if (ConstantValueMap::HasValue(n->input()->debugName())) {
        auto shape_temp = ConstantValueMap::GetValueInto1DInt64Vector(
            n->input()->debugName());
        UpdateShapeFromVector(n->output(), shape_temp);
        if (!shape_temp.empty()) {
          if (n->hasAttributeS("value")) {
            auto value = n->t(attr::value).repeat(shape_temp);
            ConstantValueMap::SetValue(n->output()->debugName(), value);
          } else {
            auto options =
                c10::TensorOptions().dtype(at::kFloat).device(at::kCPU);
            auto value = at::full({1}, 0.0, options).repeat(shape_temp);
            ConstantValueMap::SetValue(n->output()->debugName(), value);
          }
        }
      }
      break;
    }
    case ::c10::onnx::Expand: {
      if (ConstantValueMap::HasShape(n->input(0)->debugName())) {
        auto input0_shape_size =
            ConstantValueMap::GetShape(n->input(0)->debugName())
                .valu
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes/onnx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/passes/onnx`):

- [`constant_map.cpp_kw.md_docs.md`](./constant_map.cpp_kw.md_docs.md)
- [`deduplicate_initializers.h_docs.md_docs.md`](./deduplicate_initializers.h_docs.md_docs.md)
- [`shape_type_inference.h_docs.md_docs.md`](./shape_type_inference.h_docs.md_docs.md)
- [`function_substitution.h_kw.md_docs.md`](./function_substitution.h_kw.md_docs.md)
- [`eliminate_unused_items.h_kw.md_docs.md`](./eliminate_unused_items.h_kw.md_docs.md)
- [`prepare_division_for_onnx.cpp_docs.md_docs.md`](./prepare_division_for_onnx.cpp_docs.md_docs.md)
- [`fixup_onnx_controlflow.cpp_kw.md_docs.md`](./fixup_onnx_controlflow.cpp_kw.md_docs.md)
- [`constant_fold.cpp_docs.md_docs.md`](./constant_fold.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`onnx_log.cpp_kw.md_docs.md`](./onnx_log.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `shape_type_inference.cpp_docs.md_docs.md`
- **Keyword Index**: `shape_type_inference.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
