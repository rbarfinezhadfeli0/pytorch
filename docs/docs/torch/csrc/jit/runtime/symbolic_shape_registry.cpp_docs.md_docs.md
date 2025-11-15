# Documentation: `docs/torch/csrc/jit/runtime/symbolic_shape_registry.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/symbolic_shape_registry.cpp_docs.md`
- **Size**: 18,830 bytes (18.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/symbolic_shape_registry.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/symbolic_shape_registry.cpp`
- **Size**: 15,781 bytes (15.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/serialized_shape_function_registry.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <unordered_map>

namespace torch::jit {
namespace {
std::mutex lock;

// split here to satisfy MSVC++
// https://docs.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026?view=msvc-170
const std::string _xnnpack_shape_compute_functions =
#ifdef USE_XNNPACK
    R"(def prepacked_conv2d_clamp_run(input: List[int], conv2dOpContext: Any):
    assert isinstance(conv2dOpContext, __torch__.torch.classes.xnnpack.Conv2dOpContext)
    (weight, bias, stride, padding, dilation, groups) = unchecked_cast(
        Tuple[List[int], Optional[List[int]], List[int], List[int], List[int], int],
        ops.prepacked.unpack_prepacked_sizes_conv2d(conv2dOpContext),
    )
    return conv2d(input, weight, bias, stride, padding, dilation, groups)

def prepacked_linear_clamp_run(input: List[int], linearOpContext: Any):
    assert isinstance(linearOpContext, __torch__.torch.classes.xnnpack.LinearOpContext)
    (weight, bias) = unchecked_cast(
        Tuple[List[int], Optional[List[int]]],
        ops.prepacked.unpack_prepacked_sizes_linear(linearOpContext),
    )
    return linear(input, weight, bias)
    )"
#else
    ""
#endif
    ;

// mapping function schema to shape compute graphs allows multiple functions to
// share the same shape compute graph, which is memory efficient and also will
// help speed up shape analysis by caching the result of running consecutive ops
// for a particular set of inputs with the same graph, e.g. running a series
// of pointwise ops
// we need a map from schema to shape compute graph, because the aten schema
// is not recoverable from the shape compute graph, since the shape compute
// graph replaces Tensor inputs with List[int] and there are operators like Conv
// which natively have List[int] inputs
// TODO: consider storing shape compute graph directly on operator,
// and merge into native_functions.yaml

// wrapped in function so that operators get registered before map is
// initialized
// Conditionally defined ops not yet supported in python serialized
// operators
static const OperatorMap<std::string>& conditionally_defined_ops() {
  // clang-format off
  static const OperatorMap<std::string> schema_to_function_graph{
#ifdef USE_XNNPACK
      {"prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y", "prepacked_conv2d_clamp_run"},
      {"prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y", "prepacked_linear_clamp_run"},
#endif
  };
  // clang-format on
  return schema_to_function_graph;
}

std::unordered_map<const FunctionSchema*, std::shared_ptr<Graph>>
    cached_schema_to_graph;

std::unordered_map<const FunctionSchema*, BoundedShapeGraphs>
    cached_bounded_schema_to_graph;

// CompilationUnit that holds all these Functions and keeps them alive.
auto compilation_unit = std::make_shared<CompilationUnit>();

const std::optional<const FunctionSchema*> getInplaceVariant(
    const FunctionSchema& base_schema) {
  auto& inplace_variants =
      getAllOperatorsFor(c10::Symbol::fromQualString(base_schema.name() + "_"));

  for (const auto& variant : inplace_variants) {
    // Need to check that all args are the same except for the first, which
    // is almost the same except for the Alias info
    const FunctionSchema* schema = &variant->schema();
    if (!schema->isSubtypeOf(base_schema, false)) {
      continue;
    }

    Argument self_arg = schema->arguments()[0];
    if (!self_arg.alias_info()->isWrite()) {
      continue;
    }

    Argument ret_arg = schema->returns()[0];
    if (!ret_arg.alias_info()->isWrite()) {
      continue;
    }

    return schema;
  }
  return std::nullopt;
}

TypePtr mapTensorToListOfInts(TypePtr type) {
  if (type->cast<TensorType>()) {
    return ListType::ofInts();
  }
  at::ArrayRef<TypePtr> contained = type->containedTypes();
  if (contained.empty()) {
    return type;
  }
  return type->withContained(
      fmap(type->containedTypes(), mapTensorToListOfInts));
}

void checkForWhileLoop(
    const FunctionSchema* schema,
    std::shared_ptr<Graph> graph) {
  DepthFirstGraphNodeIterator graph_it(graph);
  for (auto* node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    if (node->kind() != prim::Loop) {
      continue;
    }
    LoopView loop(node);
    if (loop.loopType() != LoopView::For) {
      TORCH_WARN(
          "While loops are not yet implemented in unrolling which may make this shape function difficult to partially evaluate: ",
          *node,
          " for schema ",
          *schema);
    }
  }
}

void checkInputReturnedAsOutput(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  // Could use alias db here as well but would have to warn because it's
  // imprecise
  for (size_t i : c10::irange(graph->inputs().size())) {
    Value* input = graph->inputs().at(i);
    for (size_t j : c10::irange(graph->outputs().size())) {
      Value* output = graph->outputs().at(j);
      TORCH_CHECK(
          input != output,
          "For schema: ",
          *schema,
          " input index ",
          i,
          " is returned as output index ",
          j,
          ". Shape functions must return new unaliased lists");
    }
  }
}

void checkInputAndOutputTypes(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  // allow extra unused arguments to map multiple functions to e.g. unary
  TORCH_CHECK(
      graph->inputs().size() <= schema->arguments().size(),
      "Shape function must have fewer arguments than schema. Got ",
      graph->inputs().size(),
      " graph arguments and ",
      schema->arguments().size(),
      " schema arguments of schema: ",
      *schema);

  for (auto i : c10::irange(graph->inputs().size())) {
    auto inp_type = schema->arguments().at(i).type();
    auto mapped_type = mapTensorToListOfInts(inp_type);
    auto graph_type = graph->inputs().at(i)->type();
    TORCH_INTERNAL_ASSERT(
        mapped_type->isSubtypeOf(graph->inputs().at(i)->type()),
        "For schema type: ",
        inp_type->str(),
        " Expected supertype of ",
        mapped_type->str(),
        " but got graph_type ",
        graph_type->str(),
        " at index ",
        i,
        " of schema: ",
        *schema);
  }

  TORCH_CHECK(
      graph->outputs().size() == schema->returns().size(),
      "Shape function equal number of outputs as schema. Got ",
      graph->outputs().size(),
      " graph outputs and ",
      schema->returns().size(),
      " schema returns of schema: ",
      *schema);

  for (auto i : c10::irange(schema->returns().size())) {
    auto out_type = schema->returns().at(i).type();
    auto mapped_type = mapTensorToListOfInts(out_type);
    auto graph_type = graph->outputs().at(i)->type();
    TORCH_INTERNAL_ASSERT(
        mapped_type->isSubtypeOf(graph->outputs().at(i)->type()),
        "For schema type: ",
        out_type->str(),
        " Expected supertype of ",
        mapped_type->str(),
        " but got graph_type ",
        graph_type->str(),
        " at output index ",
        i,
        " of schema: ",
        *schema);
  }
}

void transformShapeFunction(
    const FunctionSchema* schema_string,
    const std::shared_ptr<Graph>& graph) {
  Inline(*graph);

  // ATEN operators can return multiple unboxed values, this in contrast to
  // functions defined in TorchScript or User-Registered Operators
  // Which must use a Tuple
  // Here, modify the shape graph of aten operators with multiple outputs
  // so that they correspond to each other
  if (schema_string->returns().size() > 1) {
    TORCH_INTERNAL_ASSERT(
        graph->outputs().size() == 1 &&
        graph->outputs().at(0)->type()->cast<TupleType>());
    auto tuple_node = graph->outputs().at(0)->node();
    WithInsertPoint guard(graph->return_node());
    auto tuple_unpack_values = createTupleUnpack(tuple_node->output());
    graph->eraseOutput(0);
    for (Value* v : tuple_unpack_values) {
      graph->registerOutput(v);
    }
    GRAPH_DUMP("After Output Tuple Unpacking", graph);
  }
}

std::shared_ptr<Graph> genShapeComputeFn(
    const FunctionSchema* schema_string,
    const std::string& shape_compute_function_name,
    std::unordered_map<std::string, std::shared_ptr<Graph>>& reused_functions,
    const CompilationUnit& module) {
  std::shared_ptr<Graph> graph;
  GRAPH_DEBUG(
      "Registering schema: ",
      *schema_string,
      " with shape compute func: ",
      shape_compute_function_name);
  if (reused_functions.count(shape_compute_function_name)) {
    GRAPH_DEBUG("Registering reused schema");
    graph = reused_functions[shape_compute_function_name];
  } else {
    Function& shape_compute_function =
        module.get_function(shape_compute_function_name);
    graph = toGraphFunction(shape_compute_function).graph();

    transformShapeFunction(schema_string, graph);
    // NB: we lint the shape functions registered in source
    // in a test file
    // LintShapeComputeGraph(schema_string, graph);

    reused_functions[shape_compute_function_name] = graph;
  }
  // allow extra unused arguments to map multiple functions to e.g. unary
  TORCH_INTERNAL_ASSERT(
      graph->inputs().size() <= schema_string->arguments().size());
  return graph;
}

void registerSchema(
    const FunctionSchema* schema_string,
    const std::string& shape_compute_function_name,
    std::unordered_map<std::string, std::shared_ptr<Graph>>& reused_functions,
    const CompilationUnit& module) {
  auto graph = genShapeComputeFn(
      schema_string, shape_compute_function_name, reused_functions, module);

  cached_schema_to_graph[schema_string] = graph;
}

void registerBoundedSchema(
    const FunctionSchema* schema_string,
    const std::string& lower_bound_function_name,
    const std::string& upper_bound_function_name,
    std::unordered_map<std::string, std::shared_ptr<Graph>>& reused_functions,
    const CompilationUnit& module) {
  auto lower_graph = genShapeComputeFn(
      schema_string, lower_bound_function_name, reused_functions, module);
  auto upper_graph = genShapeComputeFn(
      schema_string, upper_bound_function_name, reused_functions, module);
  cached_bounded_schema_to_graph[schema_string] = {lower_graph, upper_graph};
}

void loadModule(const CompilationUnit& module) {
  std::unordered_map<std::string, std::shared_ptr<Graph>> reused_functions;

  std::vector<std::pair<std::shared_ptr<Operator>, std::string>>
      operator_pairs = conditionally_defined_ops().getAllKeysAndValues();
  auto te_ops = get_tensorexpr_elementwise_set().getAllKeysAndValues();
  operator_pairs.insert(operator_pairs.end(), te_ops.begin(), te_ops.end());
  auto more_mappings = GetShapeFunctionMappings().getAllKeysAndValues();
  operator_pairs.insert(
      operator_pairs.end(), more_mappings.begin(), more_mappings.end());

  for (const auto& pair : operator_pairs) {
    const FunctionSchema* schema_string = &pair.first->schema();
    const std::string& shape_compute_function_name = pair.second;

    registerSchema(
        schema_string, shape_compute_function_name, reused_functions, module);

    // Register the inplace variant if any for functions with common shape forms
    if (shape_compute_function_name == "unary") {
      auto inplace_schema = getInplaceVariant(*schema_string);
      if (inplace_schema.has_value()) {
        registerSchema(
            inplace_schema.value(), "unary", reused_functions, module);
      }
    }
    if (shape_compute_function_name == "broadcast") {
      auto inplace_schema = getInplaceVariant(*schema_string);
      if (inplace_schema.has_value()) {
        registerSchema(
            inplace_schema.value(),
            "broadcast_inplace",
            reused_functions,
            module);
      }
    }
  }

  // Now register the bounded schemas
  for (const auto& pair : GetBoundedShapeMappings().getAllKeysAndValues()) {
    const FunctionSchema* schema_string = &pair.first->schema();
    const std::string& lower_bound_function_name = pair.second.first;
    const std::string& upper_bound_function_name = pair.second.second;

    registerBoundedSchema(
        schema_string,
        lower_bound_function_name,
        upper_bound_function_name,
        reused_functions,
        module);
  }
}

void loadFunctions() {
  try {
    auto shape_compute_functions =
        GetSerializedShapeFunctions() + _xnnpack_shape_compute_functions;

    auto src = std::make_shared<Source>(shape_compute_functions);
    std::stringstream ss;
    std::vector<at::IValue> constantTable;
    auto resolver = std::make_shared<SourceImporterImpl>(
        compilation_unit,
        &constantTable,
        [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
        1);
    compilation_unit->define(
        std::nullopt, shape_compute_functions, resolver, nullptr);
    loadModule(*compilation_unit);
  } catch (...) {
    // Reset the cache and compilation unit so that we don't get weird errors
    // in later tests when one of the shape functions is invalid.
    compilation_unit = std::make_shared<CompilationUnit>();
    cached_schema_to_graph.clear();
    throw;
  }
}
} // anonymous namespace

std::optional<std::shared_ptr<Graph>> shapeComputeGraphForSchema(
    const FunctionSchema& schema) {
  std::lock_guard<std::mutex> guard(lock);
  if (cached_schema_to_graph.empty()) {
    loadFunctions();
  }

  GRAPH_DEBUG("Trying to find schema: ", schema);
  auto cache_it = cached_schema_to_graph.find(&schema);
  if (cache_it != cached_schema_to_graph.end()) {
    return cache_it->second;
  }
  GRAPH_DEBUG("Could not find schema: ", schema);

  return std::nullopt;
}

TORCH_API std::optional<BoundedShapeGraphs> boundedGraphsForSchema(
    const FunctionSchema& schema) {
  std::lock_guard<std::mutex> guard(lock);
  if (cached_bounded_schema_to_graph.empty()) {
    loadFunctions();
  }
  GRAPH_DEBUG("Trying to find schema in bounded graphs: ", schema);
  auto cache_it = cached_bounded_schema_to_graph.find(&schema);
  if (cache_it != cached_bounded_schema_to_graph.end()) {
    return cache_it->second;
  }

  return std::nullopt;
}

void RegisterShapeComputeGraphForSchema(
    const FunctionSchema& schema,
    const std::shared_ptr<Graph>& g) {
  std::lock_guard<std::mutex> guard(lock);
  if (cached_schema_to_graph.empty()) {
    loadFunctions();
  }
  transformShapeFunction(&schema, g);
  LintShapeComputeGraph(&schema, g);

  cached_schema_to_graph[&schema] = g;
}

std::vector<const FunctionSchema*> RegisteredShapeComputeSchemas() {
  std::lock_guard<std::mutex> guard(lock);
  if (cached_schema_to_graph.empty()) {
    loadFunctions();
  }

  std::vector<const FunctionSchema*> schemas;
  schemas.reserve(cached_schema_to_graph.size());
  for (const auto& pair : cached_schema_to_graph) {
    schemas.push_back(pair.first);
  }
  return schemas;
}

void LintShapeComputeGraph(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  checkInputAndOutputTypes(schema, graph);
  checkForWhileLoop(schema, graph);
  checkInputReturnedAsOutput(schema, graph);
  // TODO: other checks ? list ops which we don't symbolically optimize, etc ?
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `std`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `torch/csrc/jit/frontend/ir_emitter.h`
- `torch/csrc/jit/ir/ir_views.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/inliner.h`
- `torch/csrc/jit/runtime/graph_iterator.h`
- `torch/csrc/jit/runtime/operator.h`
- `torch/csrc/jit/runtime/serialized_shape_function_registry.h`
- `torch/csrc/jit/runtime/symbolic_shape_registry.h`
- `torch/csrc/jit/runtime/symbolic_shape_registry_util.h`
- `torch/csrc/jit/serialization/import_source.h`
- `unordered_map`


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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `symbolic_shape_registry.cpp_docs.md`
- **Keyword Index**: `symbolic_shape_registry.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/jit/runtime`):

- [`register_ops_utils.h_docs.md_docs.md`](./register_ops_utils.h_docs.md_docs.md)
- [`register_c10_ops.cpp_docs.md_docs.md`](./register_c10_ops.cpp_docs.md_docs.md)
- [`exception_message.h_kw.md_docs.md`](./exception_message.h_kw.md_docs.md)
- [`register_prim_ops.cpp_kw.md_docs.md`](./register_prim_ops.cpp_kw.md_docs.md)
- [`autodiff.cpp_kw.md_docs.md`](./autodiff.cpp_kw.md_docs.md)
- [`decomposition_registry_util.h_docs.md_docs.md`](./decomposition_registry_util.h_docs.md_docs.md)
- [`slice_indices_adjust.cpp_docs.md_docs.md`](./slice_indices_adjust.cpp_docs.md_docs.md)
- [`graph_iterator.h_kw.md_docs.md`](./graph_iterator.h_kw.md_docs.md)
- [`shape_function_registry.h_docs.md_docs.md`](./shape_function_registry.h_docs.md_docs.md)
- [`symbolic_script.cpp_docs.md_docs.md`](./symbolic_script.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `symbolic_shape_registry.cpp_docs.md_docs.md`
- **Keyword Index**: `symbolic_shape_registry.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
