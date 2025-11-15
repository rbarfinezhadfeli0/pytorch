# Documentation: `benchmarks/dynamo/microbenchmarks/bench_mm_fusion.py`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/bench_mm_fusion.py`
- **Size**: 3,256 bytes (3.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
# flake8: noqa: B902

from prettytable import PrettyTable

import torch
import torch._dynamo
import torch._inductor.config
from torch._inductor.runtime.benchmarking import benchmarker


# torch._inductor.config.debug = True
torch._inductor.config.triton.dense_indexing = True
torch.manual_seed(0)


# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True


class Func:
    # mm
    @torch._dynamo.optimize("inductor")
    def mm(a, b, bias):
        y = torch.mm(a, b)
        return y

    # mm+bias
    @torch._dynamo.optimize("inductor")
    def mm_add(a, b, bias):
        y = torch.mm(a, b)
        return y + bias

    # relu(mm)
    @torch._dynamo.optimize("inductor")
    def mm_relu(a, b, bias):
        y = torch.mm(a, b)
        return torch.relu(y)

    # relu(mm+bias)
    @torch._dynamo.optimize("inductor")
    def mm_add_relu(a, b, bias):
        y = torch.mm(a, b)
        y += bias
        return torch.relu(y)


def bench(shape, layer_id, p, fusion_types=None):
    torch._logging.set_logs(inductor_metrics=True)
    if fusion_types is None:
        fusion_types = [""]
    dtype = torch.float16
    M, K = shape[0]
    _, N = shape[1]
    torch.manual_seed(0)
    # allocate inputs
    a = torch.randn(shape[0], device="cuda", dtype=dtype)
    b = torch.randn(shape[1], device="cuda", dtype=dtype)

    def tflops(ms):
        return M * K * N / ms * 1e-9

    row = [layer_id]
    for fusion_type in fusion_types:
        if fusion_type == "":
            fn_mm = Func.mm
        else:
            fn_mm = getattr(Func, f"mm_{fusion_type}")

        if "add" in fusion_type:
            bias = torch.randn((M, N), dtype=dtype, device="cuda")
        else:
            bias = None

        args = (a, b, bias)

        def fn():
            return fn_mm(*args)

        torch._inductor.config.triton.mm = "aten"
        torch_mm_ms, _, _ = benchmarker.benchmark_gpu(fn)
        torch._inductor.config.triton.mm = "triton"
        # reset to force code gen new python code
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        triton_mm_ms, _, _ = benchmarker.benchmark_gpu(fn)
        assert torch._inductor.metrics.generated_kernel_count == 1, (
            "codegen #kernel != 1"
        )
        row.extend([tflops(torch_mm_ms), tflops(triton_mm_ms)])

    p.add_row(row)
    torch._logging.set_logs()


fusion_types = ["", "add", "relu", "add_relu"]
shapes = [
    # alexnet
    ([128, 9216], [9216, 4096]),
    ([128, 4096], [4096, 4096]),
    ([128, 4096], [4096, 1000]),
    # BERT
    ([2048, 768], [768, 768]),
    ([2048, 768], [768, 3072]),
    ([2048, 3072], [3072, 768]),
    # hf_GPT2
    ([1024, 768], [768, 768]),
    ([1024, 768], [768, 3072]),
    ([1024, 3072], [3072, 768]),
    ([1024, 768], [768, 2304]),
]
p = PrettyTable()
field_names = ["layer"]
for fusion_type in fusion_types:
    if fusion_type == "":
        field_names.append("torch mm")
        field_names.append("triton mm")
    else:
        field_names.append(f"torch mm+{fusion_type}")
        field_names.append(f"triton mm+{fusion_type}")

p.field_names = field_names
p.float_format = ".3"
for id, shape in enumerate(shapes):
    bench(shape, id, p, fusion_types)

print(p)

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Func`

**Functions defined**: `mm`, `mm_add`, `mm_relu`, `mm_add_relu`, `bench`, `tflops`, `fn`

**Key imports**: PrettyTable, torch, torch._dynamo, torch._inductor.config, benchmarker


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/microbenchmarks`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `prettytable`: PrettyTable
- `torch`
- `torch._dynamo`
- `torch._inductor.config`
- `torch._inductor.runtime.benchmarking`: benchmarker


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`benchmarks/dynamo/microbenchmarks`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`dynamo_microbenchmarks.py_docs.md`](./dynamo_microbenchmarks.py_docs.md)
- [`benchmark_helper.py_docs.md`](./benchmark_helper.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`model.py_docs.md`](./model.py_docs.md)
- [`dynamo_guard_eval.py_docs.md`](./dynamo_guard_eval.py_docs.md)
- [`overheads.py_docs.md`](./overheads.py_docs.md)
- [`operatorbench.py_docs.md`](./operatorbench.py_docs.md)
- [`tensor_layout_mini_benchmark.py_docs.md`](./tensor_layout_mini_benchmark.py_docs.md)


## Cross-References

- **File Documentation**: `bench_mm_fusion.py_docs.md`
- **Keyword Index**: `bench_mm_fusion.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
