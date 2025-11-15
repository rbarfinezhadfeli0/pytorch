# Documentation: `torch/csrc/jit/ir/alias_analysis.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/ir/alias_analysis.cpp`
- **Size**: 66,698 bytes (65.13 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/ir/alias_analysis.h>

#include <ATen/core/interned_strings.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <fstream>
#include <iostream>

namespace torch::jit {

namespace {

c10::MaybeOwned<TypePtr> toSingleType(const AliasTypeSet& mut_types) {
  return mut_types.size() == 1
      ? c10::MaybeOwned<TypePtr>::borrowed(mut_types[0])
      : c10::MaybeOwned<TypePtr>::owned(c10::UnionType::create(mut_types));
}

// This class determines whether a type is mutable, and, if so, it maps
// the type to its "mutable equivalent" (see definition in
// `mapTypeToAliasTypeSet`). It uses a cache of TypePtrs to speed up these
// type lookups
class MutableTypePtrHelper {
 public:
  explicit MutableTypePtrHelper(
      ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache)
      : mutable_type_cache_(mutable_type_cache) {}

  // Map any mutable type to a type such that all other types which the
  // mutable type can alias will be mapped to the same type. For
  // example, calling this method on `Optional[List[int]]` should be
  // the same as calling this method on `List[int]`.
  //
  // Rules:
  //   - If the type is not mutable, return `nullopt`
  //   - If the type is a `Tuple`, that means that it's an immutable
  //     object that can itself contain mutable objects. We want to make
  //     sure that the mutable objects are correctly aliased, so we
  //     remove the immutable objects. (For example,
  //     `Tuple[int, Tensor]` would become `Tuple[Tensor]`, while
  //     `Tuple[int, str]` would be returned as `nullopt`.) This is a
  //     convenience that makes it easy to check if the `Tuple`
  //     contains only immutable objects, though it's not technically
  //     necessary
  //   - For any Tensor type (including Tensor types that are part of
  //     a larger container, e.g. `List[Tensor]`), return the
  //     "unshaped" version of that Tensor. An "unshaped" Tensor is a
  //     Tensor with shape information removed. For example, a Tensor
  //     of dimension 4 would map to the same type as a Tensor of
  //     dimension 1. This allows us to treat all subclasses of Tensor
  //     as a single, homogeneous "Tensor" type.
  std::optional<AliasTypeSet> mapTypeToAliasTypeSet(const TypePtr& type) {
    if (mutable_type_cache_) {
      const AliasTypeSet* result = mapTypeToBorrowedAliasTypeSet(type);
      if (result) {
        return *result;
      }
    }
    return mapTypeToAliasTypeSetImpl(type);
  }

  const AliasTypeSet* mapTypeToBorrowedAliasTypeSet(const TypePtr& type) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mutable_type_cache_ != nullptr);
    auto maybe_type_mapping = mutable_type_cache_->find(type);
    if (maybe_type_mapping != mutable_type_cache_->end()) {
      return &maybe_type_mapping->second;
    }

    auto mutable_types = mapTypeToAliasTypeSetImpl(type);
    if (mutable_types) {
      auto it =
          mutable_type_cache_->emplace(type, std::move(*mutable_types)).first;
      return &it->second;
    } else {
      return nullptr;
    }
  }

 private:
  std::optional<AliasTypeSet> mapTypeToAliasTypeSetImpl(const TypePtr& type) {
    switch (type->kind()) {
      case TypeKind::ListType:
      case TypeKind::DictType:
      case TypeKind::ClassType:
      case TypeKind::TensorType:
        // TODO: Look up cached contained types. this is kind of tricky
        // because a `List[Optional[T]]` should still be
        // `List[Optional[Unshaped(T)]]`, but
        // `mapTypeToAliasTypeSet(Optional[T])` should be `T`
        return AliasTypeSet{unshapedType(type)};
      case TypeKind::UnionType: {
        AliasTypeSet mutable_types;
        for (const TypePtr& inner :
             type->expectRef<UnionType>().containedTypes()) {
          if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
            mutable_types.insert(
                mutable_types.end(),
                (*maybe_inner_types).begin(),
                (*maybe_inner_types).end());
          }
        }
        if (mutable_types.empty()) {
          return std::nullopt;
        }
        return mutable_types;
      }
      case TypeKind::OptionalType: {
        auto inner = type->castRaw<OptionalType>()->getElementType();
        return mapTypeToAliasTypeSet(inner);
      }
      case TypeKind::AnyType:
        return {AliasTypeSet{type}};
      case TypeKind::FutureType: {
        if (auto maybe_mut_types = mapTypeToAliasTypeSet(
                type->castRaw<FutureType>()->getElementType())) {
          return {AliasTypeSet{
              FutureType::create(*toSingleType(*maybe_mut_types))}};
        }
        return std::nullopt;
      }
      case TypeKind::AwaitType: {
        if (auto maybe_mut_types = mapTypeToAliasTypeSet(
                type->castRaw<AwaitType>()->getElementType())) {
          return {
              AliasTypeSet{AwaitType::create(*toSingleType(*maybe_mut_types))}};
        }
        return std::nullopt;
      }
      case TypeKind::TupleType: {
        std::vector<TypePtr> mutable_types;
        for (const TypePtr& inner : type->expectRef<TupleType>().elements()) {
          if (auto maybe_inner_types = mapTypeToAliasTypeSet(inner)) {
            mutable_types.insert(
                mutable_types.end(),
                (*maybe_inner_types).begin(),
                (*maybe_inner_types).end());
          }
        }
        if (mutable_types.empty()) {
          return std::nullopt;
        }
        return {AliasTypeSet{TupleType::create(mutable_types)}};
      }
      default:
        return std::nullopt;
    }
  }
  ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache_;
};

bool isMutableTypeImpl(
    const TypePtr& type,
    ska::flat_hash_map<TypePtr, AliasTypeSet>* mutable_type_cache) {
  // Check common cases to avoid recursively constructing type in
  // `mapTypeToAliasTypeSetPtrImpl`
  auto kind = type->kind();
  if (kind == TypeKind::TensorType || kind == TypeKind::ListType ||
      kind == TypeKind::ClassType || kind == TypeKind::DictType) {
    return true;
  }
  MutableTypePtrHelper helper(mutable_type_cache);
  if (mutable_type_cache) {
    return helper.mapTypeToBorrowedAliasTypeSet(type) != nullptr;
  } else {
    return helper.mapTypeToAliasTypeSet(type).has_value();
  }
}

} // namespace

// Static `isMutableType` does not use cache of type -> mutable type equivalent
bool AliasDb::isMutableType(const TypePtr& type) {
  return isMutableTypeImpl(type, nullptr);
}

bool AliasDb::isMutableType(const Value* v) {
  return isMutableType(v->type());
}

// Make use of type -> mutable cache
bool AliasDb::isMutableTypeInternal(const TypePtr& type) const {
  return isMutableTypeImpl(type, &mapped_mutable_types_);
}

bool AliasDb::isMutableTypeInternal(const Value* v) const {
  return isMutableTypeInternal(v->type());
}

const AliasTypeSet* AliasDb::mapTypeToAliasTypeSetPtr(
    const TypePtr& type) const {
  MutableTypePtrHelper helper(&mapped_mutable_types_);
  return helper.mapTypeToBorrowedAliasTypeSet(type);
}

// Structure used during analysis to keep track of all writes at a high
// level. When the analysis is completed, this will be used to construct
// a more efficient WriteIndex
struct AliasDb::WriteRegistry {
  void registerWrite(const Value* v, Node* n) {
    writes_[n].emplace_back(v);
  }
  void registerWriteToAllContained(const Value* v, Node* n) {
    containedWrites_[n].emplace_back(v);
  }
  void registerWriteToAllWildcards(Node* n) {
    writesToAllWildcards_.insert(n);
  }
  std::unordered_map<Node*, std::vector<const Value*>> writes_;
  std::unordered_map<Node*, std::vector<const Value*>> containedWrites_;
  std::unordered_set<Node*> writesToAllWildcards_;
};

AliasDb::AliasDb(
    std::shared_ptr<Graph> graph,
    bool isFrozen,
    bool descendFunctionCalls)
    : graph_(std::move(graph)),
      isFrozen_(isFrozen),
      descend_function_calls_(descendFunctionCalls),
      memoryDAGBuilder_(std::make_unique<MemoryDAGBuilder>()),
      writeRegistry_(std::make_unique<AliasDb::WriteRegistry>()) {
  analyze(graph_);

  memoryDAG_ = std::move(*memoryDAGBuilder_).createMemoryDAG();
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  memoryDAGBuilder_ = nullptr; // to make further access a hard error

  memoryDAG_->setWildcards(
      wildcards_, elementMap_, [&](const Value* v) -> Element* {
        return getWildcard(v->type());
      });

  // Now we build up the various write indices based on information in the write
  // registry that we populated during analysis

  // Initialize the write index
  writeIndex_ = TWriteIndex();
  auto& writeIndex = *writeIndex_; // to make operator[] less ugly

  // Build the write index
  for (const auto& write : writeRegistry_->writes_) {
    Node* node = write.first;
    const std::vector<const Value*> writtenValues = write.second;
    for (const Value* writtenValue : writtenValues) {
      auto it = elementMap_.find(writtenValue);
      TORCH_INTERNAL_ASSERT(
          it != elementMap_.end(), "Tried to write to value not in MemoryDAG");
      const auto& writtenMemoryLocations =
          memoryDAG_->getMemoryLocations(it->second);
      writeIndex[node] |= writtenMemoryLocations;
    }
  }

  for (const auto& write : writeRegistry_->containedWrites_) {
    Node* node = write.first;
    const std::vector<const Value*>& writtenValues = write.second;
    for (const Value* writtenValue : writtenValues) {
      auto elem = elementMap_.at(writtenValue);
      MemoryLocations writtenMemoryLocations;
      memoryDAG_->collectAllContainedMemoryLocations(
          elem, writtenMemoryLocations);
      writeIndex[node] |= writtenMemoryLocations;
    }
  }

  for (const auto& write : writeRegistry_->writesToAllWildcards_) {
    for (const auto& pr : wildcardIndex_) {
      writeIndex[write].set(pr.second->index);
    }
  }

  // Now that we've built the write index, we can null out the WriteRegistry to
  // make future access an error. In this way we prevent the index from getting
  // out of sync (since we have no way of registering new writes)
  writeRegistry_ = nullptr;

  // Initialize the write cache
  buildWrittenToLocationsIndex();
  GRAPH_DEBUG(toString());
}

AliasDb::~AliasDb() = default;

bool AliasDb::isMutable(Node* n) const {
  ValueSet vs;
  for (const auto input : n->inputs()) {
    vs.insert(input);
  }
  return writesToAlias(n, vs);
}

bool AliasDb::hasInputWriters(const Node* n) const {
  for (const auto input : n->inputs()) {
    if (hasWriters(input)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::hasOutputWriters(const Node* n) const {
  for (const auto output : n->outputs()) {
    if (hasWriters(output)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::hasWriters(const Node* n) const {
  return hasInputWriters(n) || hasOutputWriters(n);
}

bool AliasDb::hasWriters(const Value* v) const {
  if (v->mustBeNone()) {
    return false;
  }

  auto it = elementMap_.find(v);
  if (it == elementMap_.end()) {
    return false;
  }

  const auto& el = it->second;
  return writtenToLocationsIndex_->intersects(
      memoryDAG_->getMemoryLocations(el));
}

void AliasDb::getWritesImpl(Node* n, MemoryLocations& ret) const {
  if (writeIndex_->count(n)) {
    const auto& writes = writeIndex_->at(n);
    ret |= writes;
  }

  for (auto block : n->blocks()) {
    for (auto node : block->nodes()) {
      getWritesImpl(node, ret);
    }
  }
}

// Does `n` write to an alias of one of the values in `vs`?
bool AliasDb::writesToAlias(Node* n, const ValueSet& vs) const {
  const auto writtenTo = getWrites(n);
  if (writtenTo.empty()) {
    return false;
  }

  MemoryLocations locs;
  for (const auto v : vs) {
    auto it = elementMap_.find(v);
    if (it != elementMap_.end()) {
      const auto& vlocs = memoryDAG_->getMemoryLocations(it->second);
      if (writtenTo.intersects(vlocs)) {
        return true;
      }
    }
  }

  return false;
}

MemoryLocations AliasDb::getWrites(Node* n) const {
  MemoryLocations writes;
  getWritesImpl(n, writes);
  return writes;
}

void AliasDb::getReadsImpl(Node* n, MemoryLocations& ret) const {
  for (const auto input : n->inputs()) {
    auto it = elementMap_.find(input);
    if (it != elementMap_.end()) {
      auto el = it->second;

      // Add all memory locations this element may alias and their contained
      // elements
      memoryDAG_->collectAllContainedMemoryLocations(el, ret);
    }
  }

  for (auto block : n->blocks()) {
    for (auto node : block->nodes()) {
      getReadsImpl(node, ret);
    }
  }
}

MemoryLocations AliasDb::getReads(Node* n) const {
  MemoryLocations reads;
  getReadsImpl(n, reads);
  return reads;
}

MemoryLocations AliasDb::getMemoryLocations(Value* v) const {
  auto it = elementMap_.find(v);
  if (it != elementMap_.end()) {
    return memoryDAG_->getMemoryLocations(it->second);
  }
  return MemoryLocations();
}

std::string AliasDb::getElementName(const Element* e) const {
  if (e->values.empty()) {
    // Not the most efficient way, but given the fact there are
    // not too many types and even fewer of them will end up in
    // `wildcardIndex_`, we should be fine with a linear search
    // each time we hit a Wildcard leaf
    for (const auto& ent : wildcardIndex_) {
      if (ent.second == e) {
        return std::string("WILDCARD for type ") + ent.first->str();
      }
    }
    return "WILDCARD";
  } else {
    std::ostringstream ss;
    if (e->values.size() == 1) {
      ss << "%" << (*e->values.begin())->debugName();
      return ss.str();
    }
    ss << "(";
    for (const Value* v : e->values) {
      ss << "%" << v->debugName() << ", ";
    }
    ss << ")";
    return ss.str();
  }
}

void AliasDb::dump() const {
  std::cout << toString();
}

std::string AliasDb::toString() const {
  std::stringstream ss{};

  ss << "\n===1. GRAPH===\n";
  ss << graph_->toString();

  ss << "\n===2. ALIAS DB===\n";
  for (const auto& ptrPair : elementMap_) {
    const auto element = ptrPair.second;
    int ct = 0;
    if (!element->pointsTo.empty()) {
      ss << getElementName(element) << " points to: ";
      for (const auto pointedTo : element->pointsTo) {
        if (ct > 0) {
          ss << ", ";
        }
        ++ct;
        ss << getElementName(memoryDAG_->fromIndex(pointedTo));
      }
      ss << "\n";
    }
    ct = 0;
    if (!element->containedElements.empty()) {
      ss << getElementName(element) << " contains: ";
      for (const auto contained : element->containedElements) {
        ss << getElementName(memoryDAG_->fromIndex(contained));
        if (ct > 0) {
          ss << ", ";
        }
        ++ct;
      }
      ss << "\n";
    }
  }

  ss << "\n===3. Writes===\n";
  for (const auto& pr : *writeIndex_) {
    const auto node = pr.first;
    const auto& values = pr.second;
    ss << *node;
    ss << "  ";
    for (const auto value : values) {
      ss << getElementName(memoryDAG_->fromIndex(value)) << ", ";
    }
    ss << "\n";
  }
  ss << "\n";
  return ss.str();
}

bool AliasDb::dumpToGraphvizFile(const char* filename) const {
  std::ofstream dot_file(filename);
  if (!dot_file.good()) {
    std::cout << "Failed to create Graphviz file: '" << filename << "'\n";
    return false;
  }
  dot_file << toGraphviz();
  return true;
}

std::string AliasDb::toGraphviz() const {
  std::stringstream dot;

  // Local helper to generate a graphviz-friendly name encoding
  // See also AliasDb::getElementName()
  const auto name = [this](const Element* e) -> std::string {
    if (e->values.empty()) {
      for (const auto& ent : wildcardIndex_) {
        if (ent.second == e) {
          return std::string("\"WILDCARD for ") + ent.first->str() + "\"";
        }
      }
      return "\"WILDCARD\"";
    } else {
      std::ostringstream ss;
      if (e->values.size() == 1) {
        ss << "\"\\%" << (*e->values.begin())->debugName() << "\"";
        return ss.str();
      }
      ss << "\"(";
      for (const Value* v : e->values) {
        ss << "\\%" << v->debugName() << ", ";
      }
      ss << ")\"";
      return ss.str();
    }
  };

  // Include the textual representation for reference
  dot << "/*\n";
  dot << toString();
  dot << "*/\n";

  dot << "digraph alias_db {\n"
      << "  rankdir=LR\n"
      << "  node [shape=rect, color=gray];\n"
      << "  edge [color=black];\n";

  for (const auto& ptrPair : elementMap_) {
    const auto element = ptrPair.second;
    if (!element->pointsTo.empty()) {
      for (const auto pointedTo : element->pointsTo) {
        dot << "  " << name(element) << " -> "
            << name(memoryDAG_->fromIndex(pointedTo)) << "\n";
      }
    }
    if (!element->containedElements.empty()) {
      for (const auto contained : element->containedElements) {
        dot << "  " << name(element) << " -> "
            << name(memoryDAG_->fromIndex(contained))
            << " [style=dashed, color=blue]\n";
      }
    }
  }

  dot << "}\n";
  return dot.str();
}

void AliasDb::analyze(const std::shared_ptr<Graph>& graph) {
  for (auto input : graph->inputs()) {
    setWildcard(input);
  }
  analyze(graph->block());
}

void AliasDb::analyze(Block* block) {
  for (auto node : block->nodes()) {
    analyze(node);
  }
}

void AliasDb::analyze(Node* node) {
  analyzeImpl(node);
}

// Returns true if analysis was run using
// the registered analyzer.
bool AliasDb::tryRegisteredAnalysis(Node* node) {
  const Operator& op = node->getOperator();
  auto analysis = op.aliasAnalysisKind();
  if (AliasAnalysisKind::PURE_FUNCTION == analysis) {
    analyzeCreator(node);
    return true;
  }
  return false;
}

// The basic strategy is:
//   1. Retrieve alias information for every input.
//   2. Use the node's schema's alias annotations to propgagate alias/write
//      information to the outputs. For unschematized nodes, a special analyzer
//      will have to be handwritten.
void AliasDb::analyzeImpl(Node* node) {
  auto op = node->maybeOperator();
  const bool hasSpecialCase = aliasAnalysisHasSpecialCaseFor(node->kind());
  if (op) {
    const auto analysis = op->aliasAnalysisKind();

    const bool registeredAsSpecialCase =
        analysis == AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
    if (C10_UNLIKELY(registeredAsSpecialCase && !hasSpecialCase)) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Op ",
          node->kind().toDisplayString(),
          " is registered with AliasAnalysisKind::INTERNAL_SPECIAL_CASE but doesn't have a special case.");
    } else if (C10_UNLIKELY(!registeredAsSpecialCase && hasSpecialCase)) {
      TORCH_INTERNAL_ASSERT(
          false,
          "Op ",
          node->kind().toDisplayString(),
          " has a special case and should be registered with AliasAnalysisKind::INTERNAL_SPECIAL_CASE but is registered with ",
          c10::toString(analysis));
    }
  } else {
    if (!hasSpecialCase) {
      std::ostringstream oss;
      for (const auto input : node->inputs()) {
        oss << input->type()->str() << ", ";
      }
      oss << "\n\nCandidates:";
      const auto& candidates = getAllOperatorsFor(node->kind());
      for (const auto& candidate : candidates) {
        oss << "\n\t" << candidate->schema();
      }
      TORCH_INTERNAL_ASSERT(
          0,
          "We don't have an op for ",
          node->kind().toDisplayString(),
          " but it isn't a special case.  ",
          "Argument types: ",
          oss.str());
    }
  }

  // These nodes are not schematized, so we need to handle them specially
  switch (node->kind()) {
    case prim::If:
      return analyzeIf(node);
    case prim::Loop:
      return analyzeLoop(node);
    case prim::FusionGroup:
    case prim::CudaFusionGroup:
    case prim::oneDNNFusionGroup:
    case prim::FunctionalGraph:
    case prim::DifferentiableGraph:
    case prim::FallbackGraph:
      return analyzeSubgraph(node);
    case prim::fork:
      return analyzeFork(node);
    case aten::wait:
      return analyzeWait(node);
    case prim::awaitable:
    case prim::awaitable_nowait:
      return analyzeAwaitable(node);
    case prim::awaitable_wait:
      return analyzeAwaitableWait(node);
    case prim::rpc_async:
    case prim::rpc_sync:
    case prim::rpc_remote:
      return analyzeRpcAsync(node);
    case aten::batch_norm:
      return analyzeBatchNorm(node);
    case aten::instance_norm:
      return analyzeInstanceNorm(node);
    case prim::GradOf:
      return analyzeGradOf(node);
    case prim::BroadcastMKLDNNTensors: {
      makePointerTo(node->outputs().at(0), node->inputs().at(0));
      makePointerTo(node->outputs().at(1), node->inputs().at(1));
      return;
    }
    // TODO: think more about TensorExpr alias correctness
    case prim::TensorExprGroup:
    case prim::TensorExprDynamicGroup:
    case prim::MKLDNNGroup:
    case prim::ConstantMKLDNNTensor:
    case prim::StaticSubgraph:
    case prim::Constant:
    case prim::AutogradZero:
    case prim::AutogradAdd:
    case prim::FusedConcat:
    case prim::MMTreeReduce:
    case prim::MMBatchSide:
    case prim::BroadcastSizes:
    case prim::ChunkSizes:
    // this should never be seen outside of initial compilation
    // but because of some dependencies with closure invoking alias
    // db needs to be handled here
    case prim::EmptyListLiteral:
    case prim::Closure:
    case prim::CreateObject:
    case prim::tolist:
    case prim::Uninitialized:
      return analyzeCreator(node);
    case prim::TupleConstruct:
    case prim::DictConstruct:
    case prim::ListConstruct:
      return analyzeContainerConstruct(node);
    case prim::TupleUnpack:
    case prim::TupleIndex:
    case prim::TupleSlice:
    case prim::ListUnpack:
    case prim::PythonOp:
    case prim::GetAttr:
      if (isFrozen_ && node->kind() == prim::GetAttr) {
        auto& ty = node->input()->type();
        if (ty->expectRef<ClassType>().is_module()) {
          return analyzeCreator(node);
        }
      }
      return analyzeExtractor(node);
    case prim::unchecked_cast:
      return makePointerTo(node->output(), node->input());
    case prim::ConstantChunk:
      return analyzeChunk(node);
    case prim::BroadcastingChunk:
      return analyzeBroadcastingChunk(node);
    case prim::SetAttr:
      return analyzeSetAttr(node);
    case prim::profile_ivalue:
    case prim::profile:
      makePointerTo(node->output(), node->inputs().at(0));
      return;
    case prim::TypeCheck:
    case prim::RequiresGradCheck: {
      auto num_inputs = node->inputs().size();
      for (const auto i : c10::irange(num_inputs)) {
        makePointerTo(node->outputs().at(i), node->inputs().at(i));
      }
      return;
    }
    case prim::BailOut:
      TORCH_INTERNAL_ASSERT(
          node->inputs().at(0)->node()->kind() == prim::BailoutTemplate);
      makePointerTo(node->output(), node->inputs().at(1));
      return;
    case prim::Guard:
      makePointerTo(node->output(), node->inputs().at(0));
      return;
    case prim::CallFunction:
    case prim::CallMethod: {
      // TODO: this can be improved with summarizes of what the function does
      // for now we assume the worst
      if (!descend_function_calls_) {
        return analyzeConservative(node);
      }
      auto g = tryToGraphFunction(node);
      if (!g) {
        return analyzeConservative(node);
      }
      // this is an unoptimized path - we copy the subgraph for each function
      // call past the first - so we do not generally enable the recursive
      // analysis. use cases for fine-grained alias analysis without inlining
      // are very uncommon
      auto graph = g->optimized_graph();
      // alias analysis will use Value* as mappings for information,
      // so for each analysis of a particular function call we need a new graph
      // for all copies made, store them for duration of analysis so we do not
      // run into lifetime issues with the graph
      std::vector<std::shared_ptr<Graph>>& graphs =
          function_call_copies_[graph.get()];
      if (graphs.empty()) {
        graphs.push_back(graph);
        analyzeSubgraph(node, graph);
      } else {
        auto copied_graph = graph->copy();
        graphs.push_back(copied_graph);
        analyzeSubgraph(node, copied_graph);
      }
      return;
    }
    case prim::Enter:
    case prim::Exit:
      // TODO: this can be improved with summarizes of what the function does
      // for now we assume the worst
      // NB: update safeToChangeAliasingRelationship if changed
      return analyzeConservative(node);
    case prim::Print:
    case prim::isinstance:
      // These ops do nothing
      return;
    default:
      if (tryRegisteredAnalysis(node)) {
        return;
      }
  }

  TORCH_INTERNAL_ASSERT(op, "We should have an op schema if we get to here");
  const AliasAnalysisKind analysis = op->aliasAnalysisKind();
  TORCH_INTERNAL_ASSERT(
      analysis != AliasAnalysisKind::INTERNAL_SPECIAL_CASE &&
          !aliasAnalysisHasSpecialCaseFor(node->kind()),
      "Special cases should be handled already if we're here.");

  if (node->kind().is_aten() || node->kind().is_prim() ||
      node->kind().is_cuda()) {
    // TODO There is nothing in the system that relies on aten:: and prim::
    // ops using AliasAnalysisKind::FROM_SCHEMA or
    // AliasAnalysisKind::INTERNAL_SPECIAL_CASE, but this is the intended
    // behavior for all current ops and a good error check. We can consider
    // lifting this constraint later if we have a use case for it.
    TORCH_INTERNAL_ASSERT(
        analysis == AliasAnalysisKind::FROM_SCHEMA ||
            analysis == AliasAnalysisKind::CONSERVATIVE,
        "aten:: and prim:: operators should use AliasAnalysisKind::FROM_SCHEMA or "
        "AliasAnalysisKind::CONSERVATIVE(if really necessary), but ",
        node->kind().toDisplayString(),
        " doesn't. Note: Ideally, prim:: operators actually shouldn't have a schema ",
        "and then use AliasAnalysisKind::INTERNAL_SPECIAL_CASE instead.");
  }

  if (analysis == AliasAnalysisKind::CONSERVATIVE) {
    // TODO A previous implementation of alias analysis always accessed
    // node->schema , which cause the schema caches in the Node class to be
    // filled for the full graph. Unfortunately, our JIT passes started relying
    // on that, so we need to keep doing this. Details: in
    // caffe2/torch/onnx/utils.py, _jit_pass_onnx is called on an invalid JIT
    // graph because we called _jit_pass_erase_number_types right before and
    // ints are now Tensors instead. So if _jit_pass_onnx tries to look up
    // operator schemas, it will crash. However, _jit_pass_constant_propagation,
    // which is called before it, runs alias analysis and prefills the schema
    // cache in the all Node instances so that _jit_pass_onnx doesn't look up
    // operators to get the schemas anymore. We should fix this.
    node->schema(); // fill the schema cache in the Node class

    return analyzeConservative(node);
  }

  TORCH_INTERNAL_ASSERT(
      analysis == AliasAnalysisKind::FROM_SCHEMA,
      "AliasAnalysisKind::CONSERVATIVE/PURE_FUNCTION/INTERNAL_SPECIAL_CASE should already have been handled above");
  const auto& schema = node->schema();

  // Bind the schema's "formal" alias annotation to the actual values those
  // schema arguments represent
  std::unordered_map<Symbol, Value*> formalToActual;
  for (const auto i : c10::irange(schema.arguments().size())) {
    const at::AliasInfo* formal = schema.arguments()[i].alias_info();
    const auto& actualValue = node->inputs().at(i);

    // Skip if there's no alias annotation
    if (!formal) {
      continue;
    }

    // If this type cannot alias, continue. Can occur with a VarType schema
    if (!isMutableTypeInternal(actualValue)) {
      continue;
    }

    // Do sanity checks on the alias annotation
    TORCH_INTERNAL_ASSERT(
        formal->containedTypes().size() <= 1,
        "Composite types for alias analysis not yet supported");
    TORCH_INTERNAL_ASSERT(
        !formal->isWildcardBefore(),
        "Doesn't make sense for a input value to begin as a wildcard");
    // This is a special case where we have alias info before [] but not after,
    // such as `Tensor(a!)[]`
    if (formal->containedTypes().size() == 1 && formal->beforeSets().empty()) {
      // Use the first containedType in alias info.
      formal = &(formal->containedTypes()[0]);
    }

    const auto& formalAlias = formal->beforeSet();

    // skip if we've already bound this alias
    if (formalToActual.count(formalAlias) != 0) {
      continue;
    }

    // Bind the formal to the actual
    formalToActual[formalAlias] = actualValue;

    // Record writes
    if (formal->isWrite()) {
      registerWrite(actualValue, node);
    }

    // Now deal with sets after the '->'
    if (formal->isWildcardAfter()) {
      TORCH_INTERNAL_ASSERT(
          formal->afterSets().size() == 1,
          "If the after set contains a wildcard, "
          "there should be no other alias sets specified.");
      setWildcard(actualValue);
    } else {
      // We don't understand anything else in the after yet, so assert there's
      // been no change.
      TORCH_INTERNAL_ASSERT(formal->beforeSets() == formal->afterSets());
    }
  }

  // Use the formal-actual mapping to give aliases to the outputs
  for (const auto i : c10::irange(schema.returns().size())) {
    const auto actual = node->outputs().at(i);
    const at::AliasInfo* formal = schema.returns()[i].alias_info();
    if (!formal) {
      // This is a fresh tensor
      giveFreshAlias(actual);
      continue;
    }

    // If this type cannot alias, continue. Can occur with a VarType schema
    if (!isMutableType(actual)) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        formal->containedTypes().size() <= 1,
        "Composite types for alias analysis not yet supported");
    TORCH_INTERNAL_ASSERT(formal->beforeSets() == formal->afterSets());
    if (formal->containedTypes().size() == 1 && formal->beforeSets().empty()) {
      // Use the first containedType in alias info.
      formal = &(formal->containedTypes()[0]);
    }
    if (formal->isWildcardBefore()) {
      TORCH_INTERNAL_ASSERT(
          formal->beforeSets().size() == 1,
          "If an output is a wildcard, "
          "there should be no other alias sets specified.");
      setWildcard(actual);
      continue;
    }

    bool inputs_has_alias = false;
    for (const auto& formalAlias : formal->beforeSets()) {
      if (formalToActual.count(formalAlias)) {
        inputs_has_alias = true;
        auto toAlias = formalToActual.at(formalAlias);
        makePointerTo(actual, toAlias);
      }
    }
    // If all the alias annotation that we encounter weren't in the inputs:
    //   e.g. foo(Tensor(a) self) -> Tensor(b)
    //   or foo(Tensor(a) self) -> Tensor(b|c)
    // Otherwise it is the form of a|fresh, which we can ignore, taking the
    // conservative assumption that the output must alias `a`, e.g
    //   aten::cuda(Tensor(a) self) -> Tensor(a|fresh)
    if (!inputs_has_alias && !formal->beforeSets().empty()) {
      giveFreshAlias(actual);
    }

    // Record writes
    if (formal->isWrite()) {
      registerWrite(actual, node);
    }
  }
}

// Register the fact that `n` writes to `v`.
void AliasDb::registerWrite(const Value* v, Node* n, bool writeToContained) {
  if (!isMutableTypeInternal(v)) {
    // don't need to register a write if the value isn't mutable
    return;
  }
  if (writeToContained) {
    writeRegistry_->registerWriteToAllContained(v, n);
  } else {
    writeRegistry_->registerWrite(v, n);
  }
}

void AliasDb::analyzeIf(Node* node) {
  // For if statements, the alias set of an output is the union of the
  // alias sets generated by the if and else block
  const auto trueBlock = node->blocks().at(0);
  const auto falseBlock = node->blocks().at(1);
  analyze(trueBlock);
  analyze(falseBlock);

  for (const auto i : c10::irange(node->outputs().size())) {
    const auto nodeOutput = node->outputs()[i];

    const auto trueOutput = trueBlock->outputs().at(i);
    const auto falseOutput = falseBlock->outputs().at(i);

    makePointerTo(nodeOutput, trueOutput);
    makePointerTo(nodeOutput, falseOutput);
  }
}

void AliasDb::analyzeLoop(Node* node) {
  const auto bodyBlock = node->blocks().at(0);
  const auto loopCarriedInputs = node->inputs().slice(2); // skip max, cond
  const auto blockInputs = bodyBlock->inputs().slice(1); // skip trip
  const auto blockOutputs = bodyBlock->outputs().slice(1); // skip trip
  TORCH_INTERNAL_ASSERT(loopCarriedInputs.size() == blockInputs.size());
  TORCH_INTERNAL_ASSERT(blockOutputs.size() == node->outputs().size());

  // Run alias analysis on the loop body, iterating until the block output
  // alias info converges. Copy node input aliases to block input
  mapAliases(blockInputs, loopCarriedInputs);

  // Populate block output alias info by analyzing the body
  analyze(bodyBlock);

  // Copy the alias info from the block output to the node output
  mapAliases(node->outputs(), blockOutputs);
}

void AliasDb::analyzeGradOf(Node* node) {
  const auto grad_of_block = node->blocks().at(0);
  analyze(grad_of_block);
  mapAliases(node->outputs(), grad_of_block->outputs());
}

void AliasDb::analyzeSubgraph(
    Node* node,
    const std::shared_ptr<Graph>& subgraph) {
  const auto subgraphBlock = subgraph->block();
  // CallFunction nodes have an extra first parameter
  if (node->kind() == prim::CallFunction) {
    mapAliases(subgraphBlock->inputs(), node->inputs().slice(1));
  } else {
    mapAliases(subgraphBlock->inputs(), node->inputs());
  }

  analyze(subgraphBlock);

  // Note: the subgraph outputs and node outputs are NOT NECESSARILY the
  // same length. Autodifferentiation maybe capture additional outputs in the
  // subgraph block.
  TORCH_INTERNAL_ASSERT(
      subgraphBlock->outputs().size() >= node->outputs().size());
  for (size_t i = 0; i < node->outputs().size(); i++) {
    makePointerTo(node->outputs()[i], subgraphBlock->outputs()[i]);
  }
}

void AliasDb::analyzeSubgraph(Node* node) {
  const auto subgraph = node->g(attr::Subgraph);
  return analyzeSubgraph(node, subgraph);
}
// For nodes that generate a fresh value from nothing
void AliasDb::analyzeCreator(Node* node) {
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }
}

// For nodes that extract values from a composite type. Right now, this just
// gives up and creates wildcards for everything.
void AliasDb::analyzeExtractor(Node* node) {
  for (const auto output : node->outputs()) {
    setWildcard(output);
  }
}

// For torch.chunk(), all returned tensors may alias the input tensor
void AliasDb::analyzeChunk(Node* node) {
  for (auto output : node->outputs()) {
    makePointerTo(output, node->input());
  }
}

void AliasDb::analyzeFork(Node* node) {
  for (const auto input : node->inputs()) {
    setWildcard(input);
  }

  // Give the future that the fork emits a fresh value
  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

void AliasDb::analyzeWait(Node* node) {
  TORCH_INTERNAL_ASSERT(node->kind() == aten::wait);
  for (const auto output : node->outputs()) {
    setWildcard(output);
  }
  // the forked subgraph that `wait` is waiting on may write to any of its
  // inputs. We don't have a reliable way of recovering the fork inputs, so
  // for safety we just register a write to every wildcard.
  writeRegistry_->registerWriteToAllWildcards(node);
}

void AliasDb::analyzeAwaitable(Node* node) {
  for (const auto input : node->inputs()) {
    setWildcard(input);
  }

  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

void AliasDb::analyzeAwaitableWait(Node* node) {
  TORCH_INTERNAL_ASSERT(node->kind() == prim::awaitable_wait);
  for (const auto output : node->outputs()) {
    setWildcard(output);
  }
  // the awaitable subgraph that `wait` is waiting on may write to any of its
  // inputs. We don't have a reliable way of recovering the awaitable inputs, so
  // for safety we just register a write to every wildcard.
  writeRegistry_->registerWriteToAllWildcards(node);
}

void AliasDb::analyzeRpcAsync(Node* node) {
  for (const auto input : node->inputs()) {
    setWildcard(input);
  }

  // Give the future that the rpc_async emits a fresh value
  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

namespace {
std::optional<bool> getConstantBooleanInput(
    Node* node,
    const std::string& inputName) {
  TORCH_INTERNAL_ASSERT(
      node->hasNamedInput(inputName), inputName + " input is expected");
  auto value = node->namedInput(inputName);
  TORCH_INTERNAL_ASSERT(
      value->type() == BoolType::get(),
      inputName + "training input is expected to be a bool");
  return constant_as<bool>(value);
}
} // namespace

// custom behavior for batch_norm because (a!)? annotations currently
// aren't supported, and because behavior differs depending on the value of
// training
void AliasDb::analyzeBatchNorm(Node* node) {
  // we invoking freezing for inference, so we assume training will be folded to
  // a constant false to avoid needing to invoke freezing multiple times in
  // order to make batch norm weights constant
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }

  if (isFrozen_) {
    return;
  }

  auto isTraining = getConstantBooleanInput(node, "training");

  if (!isTraining.has_value() || *isTraining) {
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_mean"), "running_mean input is expected");
    auto runningMean = node->namedInput("running_mean");
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_var"), "running_var input is expected");
    auto runningVar = node->namedInput("running_var");

    registerWrite(runningMean, node);
    registerWrite(runningVar, node);
  }
}

// custom behavior for instance_norm, because (a!)? annotations currently
// aren't supported, and because behavior differs depending on the value of
// use_input_stats
void AliasDb::analyzeInstanceNorm(Node* node) {
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }

  auto useInputStats = getConstantBooleanInput(node, "use_input_stats");

  if (!useInputStats.has_value() || *useInputStats) {
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_mean"), "running_mean input is expected");
    auto runningMean = node->namedInput("running_mean");
    TORCH_INTERNAL_ASSERT(
        node->hasNamedInput("running_var"), "running_var input is expected");
    auto runningVar = node->namedInput("running_var");

    registerWrite(runningMean, node);
    registerWrite(runningVar, node);
  }
}

// SetAttr: writes to the `self` field
void AliasDb::analyzeSetAttr(Node* node) {
  const auto self = node->inputs().at(0);
  TORCH_INTERNAL_ASSERT(self->type()->kind() == TypeKind::ClassType);
  registerWrite(self, node);
  // Also the value being set must become a wildcard.
  const auto newValue = node->inputs().at(1);
  setWildcard(newValue);
}

// Used for anything where we do not have accurate alias summaries
// may write to any input and produce wildcards
void AliasDb::analyzeConservative(Node* node) {
  for (const auto input : node->inputs()) {
    if (!isMutableTypeInternal(input)) {
      continue;
    }
    registerWrite(input, node, /*writeToContained=*/true);
    setWildcard(input);
  }

  for (const auto output : node->outputs()) {
    setWildcard(output);
  }
}

bool AliasDb::functionalNonEscapingListUse(const Use& use) const {
  Node* n = use.user;
  size_t offset = use.offset;
  Value* container = n->inputs().at(offset);

  // only consider aten op uses of lists
  if (!container->type()->cast<ListType>()) {
    return false;
  }

  /*
  in the general case, we consider any Value that enters another container as
  entering the heap, and thus aliasing all other heap values of the same type.
  the advantage of this approach are:
  - there are many composite list/container ops that would be tricky to
  schematize if we did something more complicated
  - limits the size of the AliasDb, because a container of size 10 only contains
  1 memory dag element instead of 10
  - we do not need to worry about adding contained elements to the wildcard set
  when a container escapes the graph.
  The downside of this approach is we are unable to handle the common case of a
  list constructed and passed into an aten op. Here, optimize for a set of
  common ops where the output does not alias the list or the list elements
  */

  // only used in output of graph - no further uses,
  // so there will be no use of it where the contained element leaks
  if (use.user->kind() == prim::Return) {
    return use.user->owningBlock() == graph_->block();
  }

  switch (use.user->kind()) {
    case aten::cat:
    case aten::broadcast_tensors:
    case aten::stack:
    case aten::vstack:
    case aten::hstack:
    case aten::dstack:
      return true;
  }
  auto op = use.user->maybeOperator();
  if (op && op->aliasAnalysisKind() == AliasAnalysisKind::PURE_FUNCTION) {
    return true;
  }
  return false;
}

bool AliasDb::functionalNonEscapingTupleUse(const Use& use) const {
  Node* n = use.user;
  size_t offset = use.offset;
  Value* container = n->inputs().at(offset);
  if (!container->type()->cast<TupleType>()) {
    return false;
  }
  // TODO(T97387453): Cover more ops that do not let escape tuples' elements.
  bool in_return_outputs = use.user->kind() == prim::Return;
  bool not_in_nested_subgraph = use.user->owningBlock() == graph_->block();
  return in_return_outputs && not_in_nested_subgraph;
}

// List or dict or tuple construct: create an aliasing element for the actual
// container, then mark all inputs as wildcards, since they've gone inside the
// container. Then, add the wildcard sets of appropriate type to the contained
// elements of the container.
void AliasDb::analyzeContainerConstruct(Node* node) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == prim::ListConstruct ||
      node->kind() == prim::DictConstruct ||
      node->kind() == prim::TupleConstruct);

  // tuples which contain immutable types are immutable
  if (!isMutableTypeInternal(node->output())) {
    return;
  }

  TORCH_INTERNAL_ASSERT(node->outputs().size() == 1);
  auto container = node->output();

  // optimization:
  // if a list is only used once in an aten op, and the op output
  // doesn't alias the input, then we can add all inputs to the list's
  // contained elements instead of the wildcard set.
  if (container->uses().size() == 1 &&
      (functionalNonEscapingListUse(container->uses().at(0)) ||
       functionalNonEscapingTupleUse(container->uses().at(0)))) {
    giveFreshAlias(container, false);
    for (Value* v : node->inputs()) {
      addToContainedElements(v, container);
    }
    return;
  }

  giveFreshAlias(container);
  auto container_elem = elementMap_.at(container);
  for (auto input : node->inputs()) {
    auto maybe_wildcard_elem = setWildcard(input);
    if (maybe_wildcard_elem) {
      memoryDAGBuilder_->addToContainedElements(
          *maybe_wildcard_elem, container_elem);
    }
  }
}

// BroadcastingChunk: all inputs are broadcasted, and then individually chunked.
// This is an intermediate node used only in the graph fuser.
void AliasDb::analyzeBroadcastingChunk(Node* node) {
  auto inputs = node->inputs();
  auto outputs = node->outputs();
  auto nchunks = node->i(attr::chunks);
  for (const auto index : c10::irange(inputs.size())) {
    // Each inputs[i] is aliased by exactly `nchunks` distinct output tensors:
    // inputs[i] produces chunks outputs[i * nchunks + k] for k in [0..nchunks)
    auto output_begin = outputs.begin() + index * nchunks;
    for (auto it = output_begin; it != output_begin + nchunks; ++it) {
      makePointerTo(*it, inputs.at(index));
    }
  }
}

bool AliasDb::nonAliasingValue(const Value* elem) const {
  // these are values which can point to aliasing types in the graph,
  // as with a None value pointing to an optional if node output,
  // but will never alias themselves
  return elem->mustBeNone() || elem->node()->kind() == prim::Uninitialized;
}

// Register the fact that `from` is a pointer to `to`
void AliasDb::makePointerTo(const Value* from, const Value* to) {
  if (nonAliasingValue(from) || nonAliasingValue(to)) {
    // if either value is guaranteed to be non-aliasing, we do not need to
    // connect the two elements. however, it is invariant that aliasing types
    // that are not wildcards have a memory dag element, so we create one if
    // needed
    giveFreshAlias(from);
    giveFreshAlias(to);
    return;
  }

  // The contained types of immutable type containers (`Optional`,
  // `Tuple`, `Future`, and `Union`) are unified, so these types can be
  // mutable or immutable and point to a type which is mutable or
  // immutable. `Any` is mutable but can point to an immutable type
  // through refinement
  if (isMutableTypeInternal(from) != isMutableTypeInternal(to)) {
    return;
  }
  // both immutable
  if (!isMutableTypeInternal(from)) {
    return;
  }
  if (from == to) {
    return;
  }

  // At this point, we are dealing with two mutable types
  auto from_el = getOrCreateElement(from);
  auto to_el = getOrCreateElement(to);

  memoryDAGBuilder_->makePointerTo(from_el, to_el);
}

void AliasDb::addToContainedElements(
    const Value* inner,
    const Value* container) {
  if (!isMutableTypeInternal(inner)) {
    return;
  }

  auto inner_el = getOrCreateElement(inner);
  auto cont_el = getOrCreateElement(container);

  memoryDAGBuilder_->addToContainedElements(inner_el, cont_el);
}

bool AliasDb::mayAlias(const Value* a, const Value* b) const {
  if (!isMutableTypeInternal(a) || !isMutableTypeInternal(b)) {
    return false;
  }

  return memoryDAG_->mayAlias(elementMap_.at(a), elementMap_.at(b));
}

bool AliasDb::mayAlias(const ValueSet& a, const ValueSet& b) const {
  if (a.empty() || b.empty()) {
    return false;
  }

  // Record all memory locations from group `a`
  MemoryLocations aMemLocs;
  for (const auto value : a) {
    auto it = elementMap_.find(value);
    if (it != elementMap_.end()) {
      aMemLocs |= memoryDAG_->getMemoryLocations(it->second);
    }
  }

  // If any of group `b`s memory locations overlap, return true.
  for (const auto value : b) {
    auto it = elementMap_.find(value);
    if (it != elementMap_.end()) {
      if (aMemLocs.intersects(memoryDAG_->getMemoryLocations(it->second))) {
        return true;
      }
    }
  }
  // No overlap, so group `a` and `b` do not share a memory location
  return false;
}

bool AliasDb::mayContainAlias(Value* a, Value* b) const {
  if (!isMutableTypeInternal(a) || !isMutableTypeInternal(b)) {
    return false;
  }
  return memoryDAG_->mayContainAlias(elementMap_.at(a), elementMap_.at(b));
}

std::vector<Element*> AliasDb::getElements(at::ArrayRef<Value*> vs) const {
  std::vector<Element*> elements;
  for (const auto& val : vs) {
    if (isMutableTypeInternal(val)) {
      elements.push_back(elementMap_.at(val));
    }
  }
  return elements;
}

bool AliasDb::mayContainAlias(
    const at::ArrayRef<Value*> a,
    const at::ArrayRef<Value*> b) const {
  auto a_elems = getElements(a);
  return a_elems.empty() ? false
                         : memoryDAG_->mayContainAlias(a_elems, getElements(b));
}

bool AliasDb::mayContainAlias(Value* a, const at::ArrayRef<Value*> b) const {
  if (!isMutableTypeInternal(a)) {
    return false;
  }
  auto b_elems = getElements(b);
  return b_elems.empty()
      ? false
      : memoryDAG_->mayContainAlias(elementMap_.at(a), b_elems);
}

// Make each value in the `from` list point to its partner in the `to` list
void AliasDb::mapAliases(at::ArrayRef<Value*> from, at::ArrayRef<Value*> to) {
  TORCH_INTERNAL_ASSERT(to.size() == from.size());
  for (const auto i : c10::irange(to.size())) {
    makePointerTo(from[i], to[i]);
  }
}

// Should only be called from create_functional_graphs.
// The asserts are to guard against unintentional use.
// FIXME refactor aliasdb construction to be more robust to mutation so this
// hack isn't necessary.
void AliasDb::createValue(const Value* value) {
  TORCH_INTERNAL_ASSERT(isMutableTypeInternal(value->type()));
  auto new_elem = memoryDAG_->unsafeMakeFreshValue(value);
  elementMap_[value] = new_elem;
}

void AliasDb::giveFreshAlias(
    const Value* value,
    bool add_wildcard_to_contained_elems) {
  auto maybe_mut_types = mapTypeToAliasTypeSetPtr(value->type());
  if (!maybe_mut_types) {
    return;
  }

  if (elementMap_.count(value)) {
    // Inside a loop, we may have given a fresh alias to this value already, so
    // skip
    return;
  }

  auto new_elem = memoryDAGBuilder_->makeFreshValue(value);
  elementMap_[value] = new_elem;
  if (add_wildcard_to_contained_elems) {
    if (maybe_mut_types->size() > 1) {
      pointUnionTypeElementToAllContainedTypes(new_elem, *maybe_mut_types);
    } else {
      addContainedTypesToFreshElement(new_elem, *maybe_mut_types);
    }
  }
}

Element* AliasDb::getOrCreateElement(const Value* value) {
  if (!elementMap_.count(value)) {
    giveFreshAlias(value);
  }
  return elementMap_.at(value);
}

void AliasDb::replaceWithNewValue(Value* existing, Value* new_value) {
  TORCH_INTERNAL_ASSERT(
      *unshapedType(existing->type()) == *unshapedType(new_value->type()),
      "Types must be strictly equal if you are replacing aliasing information. ",
      "Got existing: '",
      existing->type()->repr_str(),
      "', new_value: '",
      new_value->type()->repr_str(),
      "'");
  if (!isMutableTypeInternal(existing)) {
    return;
  }
  auto existing_elem = elementMap_.at(existing);
  elementMap_[new_value] = existing_elem;
  elementMap_.erase(exis
```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 108 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `determines`, `MutableTypePtrHelper`, `AliasDb`, `to`, `return`, `AliasDb`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/ir`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/alias_analysis.h`
- `ATen/core/interned_strings.h`
- `c10/util/flat_hash_map.h`
- `c10/util/irange.h`
- `torch/csrc/jit/api/function_impl.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/inliner.h`
- `torch/csrc/jit/passes/utils/subgraph_utils.h`
- `torch/csrc/jit/runtime/operator.h`
- `fstream`
- `iostream`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/jit/ir`):

- [`node_hashing.h_docs.md`](./node_hashing.h_docs.md)
- [`constants.cpp_docs.md`](./constants.cpp_docs.md)
- [`subgraph_matcher.h_docs.md`](./subgraph_matcher.h_docs.md)
- [`scope.cpp_docs.md`](./scope.cpp_docs.md)
- [`graph_node_list.h_docs.md`](./graph_node_list.h_docs.md)
- [`type_hashing.cpp_docs.md`](./type_hashing.cpp_docs.md)
- [`ir.h_docs.md`](./ir.h_docs.md)
- [`ir.cpp_docs.md`](./ir.cpp_docs.md)
- [`irparser.cpp_docs.md`](./irparser.cpp_docs.md)
- [`node_hashing.cpp_docs.md`](./node_hashing.cpp_docs.md)


## Cross-References

- **File Documentation**: `alias_analysis.cpp_docs.md`
- **Keyword Index**: `alias_analysis.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
