# Documentation: `benchmarks/static_runtime/deep_wide_pt_bench.cc`

## File Metadata

- **Path**: `benchmarks/static_runtime/deep_wide_pt_bench.cc`
- **Size**: 6,219 bytes (6.07 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```cpp
#include <benchmark/benchmark.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include "deep_wide_pt.h"

const int embedding_size = 32;
const int num_features = 50;

using namespace torch;

static void BM_deep_wide_base(benchmark::State& state) {
  std::shared_ptr<DeepAndWide> net =
      std::make_shared<DeepAndWide>(num_features);

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});
  // warmup
  net->forward(ad_emb_packed, user_emb, wide);
  for (auto _ : state) {
    net->forward(ad_emb_packed, user_emb, wide);
  }
}

static void BM_deep_wide_fast(benchmark::State& state) {
  std::shared_ptr<DeepAndWideFast> net =
      std::make_shared<DeepAndWideFast>(num_features);

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});
  // warmup
  net->forward(ad_emb_packed, user_emb, wide);
  for (auto _ : state) {
    net->forward(ad_emb_packed, user_emb, wide);
  }
}

static void BM_deep_wide_jit_graph_executor(benchmark::State& state) {
  auto mod = getDeepAndWideSciptModel();

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<IValue> inputs({ad_emb_packed, user_emb, wide});

  TORCH_CHECK_EQ(setenv("TORCH_JIT_DISABLE_NEW_EXECUTOR", "1", 1), 0);

  mod.forward(inputs);
  for (auto _ : state) {
    mod.forward(inputs);
  }
}

static void BM_deep_wide_jit_profiling_executor(benchmark::State& state) {
  auto mod = getDeepAndWideSciptModel();

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<IValue> inputs({ad_emb_packed, user_emb, wide});

  TORCH_CHECK_EQ(unsetenv("TORCH_JIT_DISABLE_NEW_EXECUTOR"), 0);

  mod.forward(inputs);
  for (auto _ : state) {
    mod.forward(inputs);
  }
}

static void BM_deep_wide_static(benchmark::State& state) {
  auto mod = getDeepAndWideSciptModel();
  torch::jit::StaticModule smod(mod);

  const int batch_size = state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<c10::IValue> inputs({ad_emb_packed, user_emb, wide});

  smod(inputs, {});
  for (auto _ : state) {
    smod(inputs, {});
  }
}

std::shared_ptr<torch::jit::StaticModule> getStaticModule() {
  static auto smod =
      std::make_shared<torch::jit::StaticModule>(getDeepAndWideSciptModel());
  return smod;
}

static void BM_deep_wide_static_threaded(benchmark::State& state) {
  auto sm = getStaticModule();
  torch::jit::StaticRuntime sr(*sm);

  const int batch_size = 1; // state.range(0);
  auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
  auto user_emb = torch::randn({batch_size, 1, embedding_size});
  auto wide = torch::randn({batch_size, num_features});

  std::vector<c10::IValue> inputs({ad_emb_packed, user_emb, wide});

  sr(inputs, {});
  for (auto _ : state) {
    sr(inputs, {});
  }
}

static void BM_leaky_relu_const(benchmark::State& state) {
  auto mod = getLeakyReLUConstScriptModel();
  torch::jit::StaticModule smod(mod);

  const int batch_size = state.range(0);
  auto data = torch::randn({batch_size, num_features});
  std::vector<c10::IValue> inputs({data});

  smod(inputs, {});
  for (auto _ : state) {
    smod(inputs, {});
  }
}

static void BM_leaky_relu(benchmark::State& state) {
  auto mod = getLeakyReLUScriptModel();
  torch::jit::StaticModule smod(mod);

  const int batch_size = state.range(0);
  auto neg_slope = torch::randn(1);
  auto data = torch::randn({batch_size, num_features});
  std::vector<c10::IValue> inputs({data, neg_slope[0]});

  smod(inputs, {});
  for (auto _ : state) {
    smod(inputs, {});
  }
}

BENCHMARK(BM_leaky_relu)->RangeMultiplier(8)->Ranges({{1, 20}});
BENCHMARK(BM_leaky_relu_const)->RangeMultiplier(8)->Ranges({{1, 20}});

static void BM_signed_log1p(benchmark::State& state) {
  auto mod = getSignedLog1pModel();
  torch::jit::StaticModule smod(mod);

  const int num_elements = state.range(0);
  auto data = torch::randn({num_elements});
  std::vector<c10::IValue> inputs({data});

  smod(inputs, {});
  for (auto _ : state) {
    smod(inputs, {});
  }
}

BENCHMARK(BM_signed_log1p)->RangeMultiplier(8)->Ranges({{16, 65536}});

static void BM_long_static_memory_optimization(benchmark::State& state) {
  auto mod = getLongScriptModel();
  torch::jit::StaticModuleOptions opts;
  opts.optimize_memory = state.range(1);
  torch::jit::StaticModule smod(mod, false, opts);

  const auto N = state.range(0);
  auto a = torch::randn({N, N});
  auto b = torch::randn({N, N});
  auto c = torch::randn({N, N});
  std::vector<c10::IValue> inputs({a, b, c});

  smod(inputs, {});
  for (auto _ : state) {
    smod(inputs, {});
  }
}

BENCHMARK(BM_deep_wide_base)->RangeMultiplier(8)->Ranges({{1, 20}});
BENCHMARK(BM_deep_wide_fast)->RangeMultiplier(8)->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_jit_graph_executor)
    ->RangeMultiplier(8)
    ->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_jit_profiling_executor)
    ->RangeMultiplier(8)
    ->Ranges({{1, 20}});

BENCHMARK(BM_deep_wide_static)->RangeMultiplier(8)->Ranges({{1, 20}});
BENCHMARK(BM_deep_wide_static_threaded)->Threads(8);

BENCHMARK(BM_long_static_memory_optimization)
    ->Args({2 << 0, 0})
    ->Args({2 << 2, 0})
    ->Args({2 << 4, 0})
    ->Args({2 << 8, 0})
    ->Args({2 << 0, 1})
    ->Args({2 << 2, 1})
    ->Args({2 << 4, 1})
    ->Args({2 << 8, 1});

int main(int argc, char** argv) {
  c10::ParseCommandLineFlags(&argc, &argv);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/static_runtime`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `benchmark/benchmark.h`
- `torch/csrc/jit/runtime/static/impl.h`
- `deep_wide_pt.h`


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

Files in the same folder (`benchmarks/static_runtime`):

- [`test_cpu_fusion.cc_docs.md`](./test_cpu_fusion.cc_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_utils.cc_docs.md`](./test_utils.cc_docs.md)
- [`test_static_runtime.cc_docs.md`](./test_static_runtime.cc_docs.md)
- [`test_generated_ops.cc_docs.md`](./test_generated_ops.cc_docs.md)
- [`test_utils.h_docs.md`](./test_utils.h_docs.md)
- [`deep_wide_pt.cc_docs.md`](./deep_wide_pt.cc_docs.md)
- [`deep_wide_pt.h_docs.md`](./deep_wide_pt.h_docs.md)
- [`test_static_module.cc_docs.md`](./test_static_module.cc_docs.md)


## Cross-References

- **File Documentation**: `deep_wide_pt_bench.cc_docs.md`
- **Keyword Index**: `deep_wide_pt_bench.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
