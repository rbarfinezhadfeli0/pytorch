# Documentation: `docs/torch/csrc/jit/passes/symbolic_shape_cache.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/symbolic_shape_cache.cpp_docs.md`
- **Size**: 9,623 bytes (9.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/symbolic_shape_cache.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/symbolic_shape_cache.cpp`
- **Size**: 6,812 bytes (6.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_cache.h>
#include <torch/csrc/lazy/core/cache.h>

#include <utility>

// SHAPE CACHING CODE

namespace torch::jit {
namespace {
using CanonicalArg = std::variant<CanonicalizedSymbolicShape, IValue>;
using CanonicalArgVec = std::vector<CanonicalArg>;
using CanonicalRet = std::vector<CanonicalizedSymbolicShape>;
using ShapeCacheKey = std::tuple<c10::OperatorName, CanonicalArgVec>;

CanonicalArgVec cannonicalizeVec(
    const std::vector<SSAInput>& arg_vec,
    std::unordered_map<int64_t, int64_t>& ss_map,
    bool deep_copy = true) {
  CanonicalArgVec canonical_args;
  canonical_args.reserve(arg_vec.size());
  for (auto& arg : arg_vec) {
    if (const IValue* iv = std::get_if<IValue>(&arg)) {
      if (deep_copy) {
        canonical_args.emplace_back(iv->deepcopy());
      } else {
        canonical_args.emplace_back(*iv);
      }
    } else {
      auto& ss = std::get<at::SymbolicShape>(arg);
      canonical_args.emplace_back(CanonicalizedSymbolicShape(ss, ss_map));
    }
  }
  return canonical_args;
}

std::vector<CanonicalizedSymbolicShape> cannonicalizeVec(
    const std::vector<at::SymbolicShape>& ret_vec,
    std::unordered_map<int64_t, int64_t>& ss_map) {
  std::vector<CanonicalizedSymbolicShape> canonical_rets;
  canonical_rets.reserve(ret_vec.size());
  for (auto& ss : ret_vec) {
    canonical_rets.emplace_back(ss, ss_map);
  }
  return canonical_rets;
}

struct ArgumentsHasher {
  size_t operator()(const ShapeCacheKey& cacheKey) const {
    // TODO: ignore arguments that are not used in shape function (not needed
    // initially)
    auto& op_name = std::get<0>(cacheKey);
    auto& arg_vec = std::get<1>(cacheKey);

    size_t hash_val = c10::hash<c10::OperatorName>()(op_name);

    hash_val = at::hash_combine(std::hash<size_t>{}(arg_vec.size()), hash_val);
    for (const CanonicalArg& arg : arg_vec) {
      size_t cur_arg = 0;
      if (const IValue* ival = std::get_if<IValue>(&arg)) {
        // IValue doesn't hash List (as Python doesn't), so we will do a custom
        // list hash
        if (ival->isList()) {
          TORCH_INTERNAL_ASSERT(ival->isIntList(), "Unexpected Args in List");
          cur_arg = ival->toListRef().size();
          for (const IValue& elem_ival : ival->toListRef()) {
            cur_arg = at::hash_combine(cur_arg, IValue::hash(elem_ival));
          }
        } else {
          cur_arg = IValue::hash(ival);
        }
      } else {
        cur_arg = std::get<CanonicalizedSymbolicShape>(arg).hash();
      }
      hash_val = at::hash_combine(hash_val, cur_arg);
    }
    return hash_val;
  }
};

using ShapeCache = lazy::Cache<
    ShapeCacheKey,
    std::vector<CanonicalizedSymbolicShape>,
    ArgumentsHasher>;

constexpr size_t kShapeCacheSize = 1024;
ShapeCache shapeCache(kShapeCacheSize);

ShapeCacheKey get_cache_key(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    std::unordered_map<int64_t, int64_t>& ss_map,
    bool deep_copy = true) {
  CanonicalArgVec canonical_args = cannonicalizeVec(arg_vec, ss_map, deep_copy);
  return std::make_tuple(schema->operator_name(), canonical_args);
}

} // namespace

TORCH_API void cache_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    const std::vector<at::SymbolicShape>& ret_vec) {
  // TODO: compare perf using std::vector<std::tuple<int64_t, int64_t>>
  auto ss_map = std::unordered_map<int64_t, int64_t>();
  auto cache_key = get_cache_key(schema, arg_vec, ss_map, /* deep_copy */ true);
  auto can_ret_vec = std::make_shared<std::vector<CanonicalizedSymbolicShape>>(
      cannonicalizeVec(ret_vec, ss_map));
  shapeCache.Add(std::move(cache_key), std::move(can_ret_vec));
}

TORCH_API std::optional<std::vector<at::SymbolicShape>>
get_cached_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec) {
  // TODO: compare perf using std::vector<std::tuple<int64_t, int64_t>> for both
  // ss_map and inverse_ss_map
  auto ss_map = std::unordered_map<int64_t, int64_t>();
  auto cache_key =
      get_cache_key(schema, arg_vec, ss_map, /* deep_copy */ false);
  auto cached_ret_vec = shapeCache.Get(cache_key);
  if (cached_ret_vec == nullptr) {
    return std::nullopt;
  }
  // Decanonicalize the return values
  auto inverse_ss_map = std::unordered_map<int64_t, int64_t>();
  for (auto& ss_val : ss_map) {
    inverse_ss_map[ss_val.second] = ss_val.first;
  }
  std::vector<at::SymbolicShape> ret_vec;
  for (auto& css : *cached_ret_vec) {
    ret_vec.emplace_back(css.toSymbolicShape(inverse_ss_map));
  }
  return ret_vec;
}

// Function only to access the cache, used for testing
TORCH_API void clear_shape_cache() {
  shapeCache.Clear();
}

TORCH_API size_t get_shape_cache_size() {
  return shapeCache.Numel();
}

void CanonicalizedSymbolicShape::init(
    const c10::SymbolicShape& orig_shape,
    std::unordered_map<int64_t, int64_t>& ss_map) {
  auto sizes = orig_shape.sizes();
  if (!sizes) {
    values_ = std::nullopt;
    return;
  }
  values_ = std::vector<int64_t>();
  int64_t cur_symbolic_index = -static_cast<int64_t>(ss_map.size()) - 1;
  for (auto& cur_shape : *sizes) {
    if (cur_shape.is_static()) {
      values_->push_back(cur_shape.static_size());
    } else {
      // Check for aliasing
      auto it = ss_map.find(cur_shape.value());

      if (it == ss_map.end()) {
        values_->push_back(cur_symbolic_index);
        ss_map.insert({cur_shape.value(), cur_symbolic_index});
        cur_symbolic_index--;
      } else {
        values_->push_back(it->second);
      }
    }
  }
}

c10::SymbolicShape CanonicalizedSymbolicShape::toSymbolicShape(
    std::unordered_map<int64_t, int64_t>& inverse_ss_map) const {
  if (!values_.has_value()) {
    return c10::SymbolicShape();
  }
  std::vector<at::ShapeSymbol> sizes;
  for (long long cur_val : *values_) {
    if (cur_val >= 0) {
      sizes.push_back(at::ShapeSymbol::fromStaticSize(cur_val));
      continue;
    }
    auto res = inverse_ss_map.find(cur_val);
    if (res != inverse_ss_map.end()) {
      sizes.push_back(at::ShapeSymbol::fromStaticSize(res->second));
    } else {
      auto new_symbol = at::ShapeSymbol::newSymbol();
      inverse_ss_map.insert({cur_val, new_symbol.value()});
      sizes.push_back(new_symbol);
    }
  }
  return c10::SymbolicShape(std::move(sizes));
}

size_t CanonicalizedSymbolicShape::hash() const {
  if (!values_.has_value()) {
    return 0x8cc80c80; // random value to prevent hash collisions
  }
  return c10::hash<std::vector<int64_t>>()(values_.value());
}

bool operator==(
    const CanonicalizedSymbolicShape& a,
    const CanonicalizedSymbolicShape& b) {
  return a.values_ == b.values_;
}
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `TORCH_API`

**Classes/Structs**: `ArgumentsHasher`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/symbolic_shape_analysis.h`
- `torch/csrc/jit/passes/symbolic_shape_cache.h`
- `torch/csrc/lazy/core/cache.h`
- `utility`


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

- **File Documentation**: `symbolic_shape_cache.cpp_docs.md`
- **Keyword Index**: `symbolic_shape_cache.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/passes`):

- [`peephole_dict_idioms.h_docs.md_docs.md`](./peephole_dict_idioms.h_docs.md_docs.md)
- [`remove_redundant_profiles.h_kw.md_docs.md`](./remove_redundant_profiles.h_kw.md_docs.md)
- [`loop_unrolling.cpp_kw.md_docs.md`](./loop_unrolling.cpp_kw.md_docs.md)
- [`onnx.h_kw.md_docs.md`](./onnx.h_kw.md_docs.md)
- [`guard_elimination.h_docs.md_docs.md`](./guard_elimination.h_docs.md_docs.md)
- [`frozen_conv_add_relu_fusion.cpp_docs.md_docs.md`](./frozen_conv_add_relu_fusion.cpp_docs.md_docs.md)
- [`hoist_conv_packed_params.h_kw.md_docs.md`](./hoist_conv_packed_params.h_kw.md_docs.md)
- [`lift_closures.h_kw.md_docs.md`](./lift_closures.h_kw.md_docs.md)
- [`frozen_conv_folding.h_kw.md_docs.md`](./frozen_conv_folding.h_kw.md_docs.md)
- [`frozen_graph_optimizations.h_docs.md_docs.md`](./frozen_graph_optimizations.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `symbolic_shape_cache.cpp_docs.md_docs.md`
- **Keyword Index**: `symbolic_shape_cache.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
