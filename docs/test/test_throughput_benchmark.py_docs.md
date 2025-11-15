# Documentation: `test/test_throughput_benchmark.py`

## File Metadata

- **Path**: `test/test_throughput_benchmark.py`
- **Size**: 4,018 bytes (3.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import run_tests, TemporaryFileName, TestCase
from torch.utils import ThroughputBenchmark


class TwoLayerNet(torch.jit.ScriptModule):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    @torch.jit.script_method
    def forward(self, x1, x2):
        h1_relu = self.linear1(x1).clamp(min=0)
        h2_relu = self.linear1(x2).clamp(min=0)
        cat = torch.cat((h1_relu, h2_relu), 1)
        y_pred = self.linear2(cat)
        return y_pred


class TwoLayerNetModule(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    def forward(self, x1, x2):
        h1_relu = self.linear1(x1).clamp(min=0)
        h2_relu = self.linear1(x2).clamp(min=0)
        cat = torch.cat((h1_relu, h2_relu), 1)
        y_pred = self.linear2(cat)
        return y_pred


class TestThroughputBenchmark(TestCase):
    def linear_test(self, Module, profiler_output_path=""):
        D_in = 10
        H = 5
        D_out = 15
        B = 8
        NUM_INPUTS = 2

        module = Module(D_in, H, D_out)

        inputs = []

        for _ in range(NUM_INPUTS):
            inputs.append([torch.randn(B, D_in), torch.randn(B, D_in)])
        bench = ThroughputBenchmark(module)

        for input in inputs:
            # can do both args and kwargs here
            bench.add_input(input[0], x2=input[1])

        for i in range(NUM_INPUTS):
            # or just unpack the list of inputs
            module_result = module(*inputs[i])
            bench_result = bench.run_once(*inputs[i])
            torch.testing.assert_close(bench_result, module_result)

        stats = bench.benchmark(
            num_calling_threads=4,
            num_warmup_iters=100,
            num_iters=1000,
            profiler_output_path=profiler_output_path,
        )

        print(stats)

    def test_script_module(self):
        self.linear_test(TwoLayerNet)

    def test_module(self):
        self.linear_test(TwoLayerNetModule)

    def test_profiling(self):
        with TemporaryFileName() as fname:
            self.linear_test(TwoLayerNetModule, profiler_output_path=fname)

    def linear_with_compile_test(self, Module, dtype):
        from contextlib import nullcontext

        from torch._dynamo import config
        from torch._inductor import config as inductor_config

        config.error_on_recompile = True
        inductor_config.cpp_wrapper = True
        inductor_config.freezing = True
        D_in = 10
        H = 5
        D_out = 15
        B = 8

        autocast = dtype != torch.float32
        module = Module(D_in, H, D_out)

        input = (torch.randn(B, D_in), torch.randn(B, D_in))

        with torch.no_grad(), torch.amp.autocast("cpu", enabled=autocast, dtype=dtype):
            torch._dynamo.reset()
            module(*input)
            module = torch.compile(module)
            module(*input)
            module(*input)

        ctx = nullcontext()
        if dtype == torch.float16 or dtype == torch.bfloat16:
            ctx = torch.amp.autocast("cpu", enabled=autocast, dtype=dtype)
        with torch.no_grad(), ctx:
            bench = ThroughputBenchmark(module)
            bench.add_input(*input)

            module_result = module(*input)
            bench_result = bench.run_once(*input)
            torch.testing.assert_close(bench_result, module_result)

            stats = bench.benchmark(
                num_calling_threads=4, num_warmup_iters=100, num_iters=1000
            )

            print(stats)

    def test_compile(self):
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            self.linear_with_compile_test(TwoLayerNetModule, dtype)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TwoLayerNet`, `TwoLayerNetModule`, `TestThroughputBenchmark`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `linear_test`, `test_script_module`, `test_module`, `test_profiling`, `linear_with_compile_test`, `test_compile`

**Key imports**: torch, run_tests, TemporaryFileName, TestCase, ThroughputBenchmark, nullcontext, config, config as inductor_config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_utils`: run_tests, TemporaryFileName, TestCase
- `torch.utils`: ThroughputBenchmark
- `contextlib`: nullcontext
- `torch._dynamo`: config
- `torch._inductor`: config as inductor_config


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python test/test_throughput_benchmark.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_throughput_benchmark.py_docs.md`
- **Keyword Index**: `test_throughput_benchmark.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
