# Documentation: `docs/torch/csrc/jit/runtime/static/te_wrapper.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/static/te_wrapper.cpp_docs.md`
- **Size**: 11,789 bytes (11.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/static/te_wrapper.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/static/te_wrapper.cpp`
- **Size**: 9,124 bytes (8.91 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/runtime/static/te_wrapper.h>

#include <ATen/CPUFunctions.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

#include <utility>

namespace torch::jit {

using namespace torch::jit::tensorexpr;

// Use the width of an AVX-512 vector by default; this happens to work OK for
// AVX2 as well. Some ops benefit from using multiple AVX ports, in which case
// they are vectorized by twice this constant.  An exception is logit, since it
// contains FP divide, which is single-ported.
static constexpr int kVectorWidth = 16;

#ifdef TORCH_ENABLE_LLVM

void TEWrapper::update(std::unique_ptr<LLVMCodeGen>&& cg_) {
  cg = std::move(cg_);
}

void TEWrapper::call(const std::vector<void*>& args) {
  cg->call_raw(args);
}

static void optimizePointwise(LoopNest* ln, Tensor target, int width) {
  std::vector<ForPtr> loops = ln->getLoopStmtsFor(target);
  ForPtr inner, tail;
  TORCH_CHECK(loops.size() > 0, "No loops created for pointwise op");
  ln->splitWithTail(loops[0], width, &inner, &tail);
  ln->vectorize(inner);
}

static std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    Tensor out,
    std::vector<CodeGen::BufferArg> args,
    int width = kVectorWidth) {
  LoopNest ln({out});
  optimizePointwise(&ln, out, width);
  ln.prepareForCodegen();
  StmtPtr s = ln.root_stmt();
  s = IRSimplifier::simplify(s);
  args.insert(args.begin(), out);
  auto cg = std::make_unique<LLVMCodeGen>(s, args);
  cg->cleanup_memory();
  wrap->update(std::move(cg));
  return wrap;
}

static std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    LoopNest* ln,
    std::vector<CodeGen::BufferArg> args) {
  auto cg = std::make_unique<LLVMCodeGen>(ln->root_stmt(), args);
  wrap->update(std::move(cg));
  return wrap;
}

#else

void TEWrapper::call(const std::vector<void*>& args) {
  DCHECK(0 && "Invalid call");
}

static std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    const Tensor& out,
    const std::vector<CodeGen::BufferArg>& args,
    int width = kVectorWidth) {
  return wrap;
}

static std::shared_ptr<TEWrapper> wrapTECompute(
    std::shared_ptr<TEWrapper> wrap,
    LoopNest* ln,
    const std::vector<CodeGen::BufferArg>& args) {
  return wrap;
}

#endif

namespace {

std::mutex& getNNCCacheMutex() {
  static std::mutex nncCacheMutex;
  return nncCacheMutex;
}

c10::FastMap<NodeKind, std::shared_ptr<TEWrapper>>& getNNCCache() {
  static c10::FastMap<NodeKind, std::shared_ptr<TEWrapper>> nncCache;
  return nncCache;
}

std::shared_ptr<TEWrapper> lookupNNCCache(NodeKind kind) {
  std::lock_guard<std::mutex> lock(getNNCCacheMutex());
  auto it = getNNCCache().find(kind);
  if (it != getNNCCache().end()) {
    return it->second;
  }
  return nullptr;
}

void updateNNCCache(NodeKind kind, std::shared_ptr<TEWrapper> code) {
  std::lock_guard<std::mutex> lock(getNNCCacheMutex());
  getNNCCache()[kind] = std::move(code);
}

} // namespace

std::shared_ptr<TEWrapper> createDiv() {
  auto wrap = lookupNNCCache(aten::div);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();

  auto dim = VarHandle("dim", kInt);
  auto mode = VarHandle("mode", kInt);
  BufHandle A("A", {dim}, kFloat);
  BufHandle B("B", {dim}, kFloat);

  using axis = const VarHandle&;
  Tensor C = Compute("C", {dim}, [&](axis x) {
    auto true_div_result = A.load(x) / B.load(x);

    auto mode_default = IntImm::make(0);
    auto mode_trunc = IntImm::make(1);
    auto mode_floor = IntImm::make(2);

    // this is a glorified ternary choice operator train
    return CompareSelect::make(
        mode,
        mode_default,
        true_div_result,
        CompareSelect::make(
            mode,
            mode_trunc,
            trunc(true_div_result),
            floor(true_div_result),
            kEQ),
        kEQ);
  });

  wrap = wrapTECompute(wrap, C, {A, B, mode, dim});

  updateNNCCache(aten::div, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createLogit() {
  auto wrap = lookupNNCCache(aten::logit);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  auto C = VarHandle("C", kFloat);
  BufHandle A("A", {N}, kFloat);
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto A_elem = [&]() {
      auto elem = A.load(i);
      auto one = FloatImm::make(1.0f);
      const auto& min = C;
      auto max = one - C;
      elem = CompareSelect::make(elem, min, min, elem, kLT);
      return CompareSelect::make(elem, max, max, elem, kGT);
    }();
    return log_vml(A_elem / (FloatImm::make(1.0f) - A_elem));
  });
  wrap = wrapTECompute(wrap, B, {A, N, C});
  updateNNCCache(aten::logit, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createRelu() {
  auto wrap = lookupNNCCache(aten::relu);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto zero = FloatImm::make(0.f);
    auto a = A.load(i);
    return CompareSelect::make(a, zero, zero, a, kLT);
  });
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::relu, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createTanh() {
  auto wrap = lookupNNCCache(aten::tanh);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  Tensor B = Compute("B", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    return fast_tanh(a);
  });
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::tanh, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createSigmoid() {
  auto wrap = lookupNNCCache(aten::sigmoid);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  Tensor B = Compute(
      "B", {N}, [&](const VarHandle& i) { return fast_sigmoid(A.load(i)); });
  wrap = wrapTECompute(wrap, B, {A, N});
  updateNNCCache(aten::sigmoid, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createClamp() {
  static auto clamp_symbol = c10::Symbol::fromQualString("aten::clamp");
  auto wrap = lookupNNCCache(clamp_symbol);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  auto min_handle = VarHandle("min", kFloat);
  auto max_handle = VarHandle("max", kFloat);

  BufHandle A("A", {N}, kFloat);
  Tensor result = Compute("aten_clamp", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    return tensorexpr::clamp(min_handle, max_handle, a);
  });
  wrap = wrapTECompute(wrap, result, {A, min_handle, max_handle, N});
  updateNNCCache(clamp_symbol, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createClampNanToNum() {
  static auto symbol =
      c10::Symbol::fromQualString("static_runtime::clamp_nan_to_num");
  auto wrap = lookupNNCCache(symbol);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  auto min_handle = VarHandle("min", kFloat);
  auto max_handle = VarHandle("max", kFloat);
  auto nan_replace_val = VarHandle("nan_replace_val", kFloat);

  BufHandle A("A", {N}, kFloat);
  Tensor result = Compute("aten_clamp", {N}, [&](const VarHandle& i) {
    auto a = A.load(i);
    auto clamp = tensorexpr::clamp(min_handle, max_handle, a);
    auto is_nan = tensorexpr::isnan(clamp);
    auto nans_replaced =
        tensorexpr::CompareSelect::make(is_nan, 1, nan_replace_val, clamp, kEQ);
    return nans_replaced;
  });
  wrap = wrapTECompute(
      wrap, result, {A, min_handle, max_handle, nan_replace_val, N});
  updateNNCCache(symbol, wrap);
  return wrap;
}

std::shared_ptr<TEWrapper> createSignedLog1p() {
  static auto signed_log1p_symbol =
      c10::Symbol::fromQualString("static_runtime::signed_log1p");
  auto wrap = lookupNNCCache(signed_log1p_symbol);
  if (wrap) {
    return wrap;
  }
  wrap = std::make_shared<TEWrapper>();
  auto N = VarHandle("N", kInt);
  BufHandle A("A", {N}, kFloat);
  Tensor abs_result = Compute("aten_abs", {N}, [&](const VarHandle& i) {
    return tensorexpr::abs(A.load(i));
  });
  Tensor log1p_result = Compute("aten_log1p", {N}, [&](const VarHandle& i) {
    return log1p(abs_result.load(i));
  });
  Tensor sign = computeSign({A}, {N});
  Tensor output = Compute("aten_mul", {N}, [&](const VarHandle& i) {
    return sign.load(i) * log1p_result.load(i);
  });
  LoopNest ln({output}, {abs_result, log1p_result, sign, output});
  GRAPH_DEBUG("Original stmt: ", *ln.root_stmt());
  ln.inlineIntermediateBufs(true);
  ln.prepareForCodegen();
  ln.simplify();
  ln.vectorizeInnerLoops();
  ln.simplify();
  GRAPH_DEBUG("Final stmt: ", *ln.root_stmt());
  wrap = wrapTECompute(wrap, &ln, {output, A, N});
  updateNNCCache(signed_log1p_symbol, wrap);
  return wrap;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `std`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime/static`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/runtime/static/te_wrapper.h`
- `ATen/CPUFunctions.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/runtime/static/impl.h`
- `torch/csrc/jit/tensorexpr/expr.h`
- `torch/csrc/jit/tensorexpr/operators/misc.h`
- `torch/csrc/jit/tensorexpr/operators/operators.h`
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

Files in the same folder (`torch/csrc/jit/runtime/static`):

- [`memory_planner.h_docs.md`](./memory_planner.h_docs.md)
- [`ops.h_docs.md`](./ops.h_docs.md)
- [`fusion.h_docs.md`](./fusion.h_docs.md)
- [`fusion.cpp_docs.md`](./fusion.cpp_docs.md)
- [`memory_planner.cpp_docs.md`](./memory_planner.cpp_docs.md)
- [`generated_ops.cpp_docs.md`](./generated_ops.cpp_docs.md)
- [`init.h_docs.md`](./init.h_docs.md)
- [`passes.cpp_docs.md`](./passes.cpp_docs.md)
- [`passes.h_docs.md`](./passes.h_docs.md)
- [`impl.h_docs.md`](./impl.h_docs.md)


## Cross-References

- **File Documentation**: `te_wrapper.cpp_docs.md`
- **Keyword Index**: `te_wrapper.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime/static`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime/static`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/runtime/static`):

- [`fusion.h_kw.md_docs.md`](./fusion.h_kw.md_docs.md)
- [`ProcessedNodeInputs.cpp_docs.md_docs.md`](./ProcessedNodeInputs.cpp_docs.md_docs.md)
- [`impl.h_docs.md_docs.md`](./impl.h_docs.md_docs.md)
- [`memory_planner.cpp_kw.md_docs.md`](./memory_planner.cpp_kw.md_docs.md)
- [`te_wrapper.cpp_kw.md_docs.md`](./te_wrapper.cpp_kw.md_docs.md)
- [`generated_ops.cpp_kw.md_docs.md`](./generated_ops.cpp_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`te_wrapper.h_docs.md_docs.md`](./te_wrapper.h_docs.md_docs.md)
- [`ProcessedNodeInputs.h_kw.md_docs.md`](./ProcessedNodeInputs.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `te_wrapper.cpp_docs.md_docs.md`
- **Keyword Index**: `te_wrapper.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
