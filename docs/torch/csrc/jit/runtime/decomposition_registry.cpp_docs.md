# Documentation: `torch/csrc/jit/runtime/decomposition_registry.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/decomposition_registry.cpp`
- **Size**: 7,043 bytes (6.88 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/runtime/decomposition_registry.h>
#include <torch/csrc/jit/runtime/decomposition_registry_util.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/serialization/import_source.h>

#include <c10/util/Exception.h>
#include <torch/csrc/autograd/jit_decomp_interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <memory>
#include <unordered_map>

namespace torch::jit {
namespace {
std::mutex lock;

// CompilationUnit that holds all these Functions and keeps them alive.
auto compilation_unit = std::make_shared<CompilationUnit>();
std::unordered_map<const FunctionSchema*, std::shared_ptr<Graph>>
    schema_to_decomposition;

// Holds User-Registered Functions and keeps them alive
std::unordered_map<const FunctionSchema*, std::unique_ptr<Function>>
    user_registered_funcs;

std::unordered_map<const FunctionSchema*, Function*> schema_to_function;

void loadModule(const CompilationUnit& module) {
  const auto& mappings = GetDecompositionMapping().getAllKeysAndValues();
  for (const auto& pair : mappings) {
    const FunctionSchema* schema = &pair.first->schema();
    const std::string& decomposition_function_name = pair.second;

    Function& decomposition_function =
        module.get_function(decomposition_function_name);
    std::shared_ptr<Graph> graph =
        toGraphFunction(decomposition_function).graph();

    schema_to_function[schema] = &decomposition_function;
    schema_to_decomposition[schema] = graph;
  }
}

void loadDecompositionFunctions() {
  std::lock_guard<std::mutex> guard(lock);
  if (!schema_to_decomposition.empty()) {
    return;
  }

  auto src = std::make_shared<Source>(GetSerializedDecompositions());
  std::stringstream ss;
  std::vector<at::IValue> constantTable;
  auto resolver = std::make_shared<SourceImporterImpl>(
      compilation_unit,
      &constantTable,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
      1);
  compilation_unit->define(
      std::nullopt, GetSerializedDecompositions(), resolver, nullptr);
  loadModule(*compilation_unit);
}

} // anonymous namespace

static void DecomposeOp(Node* n) {
  auto schema = n->maybeSchema();
  if (!schema) {
    return;
  }
  auto decomposition = GetDecomposition(n->schema());
  if (!decomposition) {
    return;
  }
  WithInsertPoint guard(n);
  auto outputs = insertGraph(*n->owningGraph(), **decomposition, n->inputs());
  TORCH_INTERNAL_ASSERT(outputs.size() == n->outputs().size());
  for (size_t i : c10::irange(outputs.size())) {
    n->outputs().at(i)->replaceAllUsesWith(outputs[i]);
  }
  n->destroy();
}

static void RunDecompositions(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // advance iterator bc the current node may be destroyed
    for (Block* b : n->blocks()) {
      RunDecompositions(b);
    }
    DecomposeOp(n);
  }
}

void RunDecompositions(std::shared_ptr<Graph> g) {
  RunDecompositions(g->block());
  for ([[maybe_unused]] const auto _ : c10::irange(2)) {
    PeepholeOptimize(g, /*disable_shape_peephole*/ true);
    ConstantPropagation(g);
  }
}

std::optional<std::shared_ptr<Graph>> GetDecomposition(
    const FunctionSchema& schema) {
  loadDecompositionFunctions();
  GRAPH_DEBUG("Trying to find schema: ", schema);
  auto cache_it = schema_to_decomposition.find(&schema);
  if (cache_it != schema_to_decomposition.end()) {
    return cache_it->second;
  }
  GRAPH_DEBUG("Could not find schema: ", schema);

  return std::nullopt;
}

std::optional<GraphFunction*> GetDecompositionFunction(
    const FunctionSchema& schema) {
  loadDecompositionFunctions();
  auto cache_it = schema_to_function.find(&schema);
  GRAPH_DEBUG("Trying to find schema: ", schema);
  if (cache_it == schema_to_function.end()) {
    GRAPH_DEBUG("Could not find schema: ", schema);
    return std::nullopt;
  }
  auto& func = toGraphFunction(*cache_it->second);
  // Simple Executor:
  // To allow decomposition to run on tensor subclasses such as batched tensors,
  // we set decomposition execution to use the simple executor so that
  // optimizations that do not compose with arbitrary subclasses (such as
  // fusion) do not run
  func._set_initial_executor_execution_mode(ExecutorExecutionMode::SIMPLE);
  return &func;
}

// Decomposition registers a Graph so that we can initialize a GraphFunction
// that will run with Simple Executor
void RegisterDecomposition(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g) {
  loadDecompositionFunctions();
  std::lock_guard<std::mutex> guard(lock);
  Inline(*g);
  for (const auto i : c10::irange(2)) {
    (void)i; // Suppress unused variable warning
    PeepholeOptimize(g);
    ConstantPropagationImmutableTypes(g);
  }

  auto new_func = std::make_unique<GraphFunction>(
      schema.name(), g, nullptr, ExecutorExecutionMode::SIMPLE);
  user_registered_funcs.emplace(&schema, std::move(new_func));
  schema_to_function[&schema] = user_registered_funcs[&schema].get();
  schema_to_decomposition[&schema] = g;
}

// see NOTE: [Jit Decomposition Interface]
struct JitDecomp final : torch::autograd::impl::JitDecompInterface {
  bool has_jit_decomposition(const c10::FunctionSchema& schema) const override;
  void run_jit_decomposition(
      const c10::OperatorHandle& op,
      torch::jit::Stack* stack) const override;
};

static JitDecomp jitDecomp;
static torch::autograd::impl::JitDecompRegisterer registerJitDecomp(&jitDecomp);

void JitDecomp::run_jit_decomposition(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) const {
  ::torch::jit::run_jit_decomposition(op, stack);
}

bool JitDecomp::has_jit_decomposition(const FunctionSchema& schema) const {
  return ::torch::jit::has_jit_decomposition(schema);
}

void run_jit_decomposition(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  // TODO: templatize based on op and keep static trace_exec
  auto* trace_exec = torch::jit::GetDecompositionExecutor(schema);
  trace_exec->run((*stack));
  if (stack->back().isTuple()) {
    at::IValue tup = std::move(stack->back());
    stack->pop_back();
    for (const auto& elem : tup.toTuple()->elements()) {
      stack->push_back(elem);
    }
  }
}

bool has_jit_decomposition(const FunctionSchema& schema) {
  return GetDecompositionFunction(schema).has_value();
}

Function* GetDecompositionExecutor(const FunctionSchema& schema) {
  auto maybe_func = GetDecompositionFunction(schema);
  TORCH_INTERNAL_ASSERT(maybe_func);
  return *maybe_func;
}

Function* GetDecompositionExecutor(const char* schema_literal) {
  auto& schema = getOperatorForLiteral(schema_literal)->schema();
  return GetDecompositionExecutor(schema);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `static`, `torch`

**Classes/Structs**: `JitDecomp`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/frontend/ir_emitter.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/constant_propagation.h`
- `torch/csrc/jit/passes/peephole.h`
- `torch/csrc/jit/runtime/decomposition_registry.h`
- `torch/csrc/jit/runtime/decomposition_registry_util.h`
- `torch/csrc/jit/runtime/operator.h`
- `torch/csrc/jit/serialization/import_source.h`
- `c10/util/Exception.h`
- `torch/csrc/autograd/jit_decomp_interface.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/passes/inliner.h`
- `torch/csrc/jit/runtime/graph_executor.h`
- `memory`
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

- **File Documentation**: `decomposition_registry.cpp_docs.md`
- **Keyword Index**: `decomposition_registry.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
