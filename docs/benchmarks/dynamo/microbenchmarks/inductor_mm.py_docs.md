# Documentation: `benchmarks/dynamo/microbenchmarks/inductor_mm.py`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/inductor_mm.py`
- **Size**: 5,736 bytes (5.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import triton
from benchmark_helper import time_with_torch_timer

import torch
import torch._dynamo
import torch._dynamo.config
import torch._inductor.config as config
from torch._inductor.runtime.benchmarking import benchmarker


# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_aten_mm(a, b):
    return torch.mm(a, b)


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_mm(a, b):
    return torch.mm(a, b)


def torch_mm(a, b):
    return torch.mm(a, b)


def triton_mm(a, b):
    return triton.ops.matmul(a, b)


def test_total_time(shapes):
    print("shape; torch mm; triton mm; inductor aten mm; inductor triton mm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

        config.triton.mm = "aten"
        inductor_aten_mm(a, b)

        config.triton.mm = "triton"
        inductor_triton_mm(a, b)

        torch_ms = time_with_torch_timer(torch_mm, (a, b)).mean * 1000

        triton_ms = time_with_torch_timer(triton_mm, (a, b)).mean * 1000

        config.triton.mm = "aten"
        ind_aten_ms = time_with_torch_timer(inductor_aten_mm, (a, b)).mean * 1000

        config.triton.mm = "triton"
        ind_triton_ms = time_with_torch_timer(inductor_triton_mm, (a, b)).mean * 1000

        print(torch_ms, triton_ms, ind_aten_ms, ind_triton_ms, sep="; ")

        torch._dynamo.reset()


def test_GPU_time(shapes):
    print("shape; torch mm; triton mm; inductor aten mm; inductor triton mm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

        config.triton.mm = "aten"
        inductor_aten_mm(a, b)

        config.triton.mm = "triton"
        inductor_triton_mm(a, b)

        torch_ms, _, _ = benchmarker.benchmark_gpu(lambda: torch_mm(a, b))
        triton_ms, _, _ = benchmarker.benchmark_gpu(lambda: triton_mm(a, b))
        ind_aten_ms, _, _ = benchmarker.benchmark_gpu(lambda: inductor_aten_mm(a, b))
        ind_triton_ms, _, _ = benchmarker.benchmark_gpu(
            lambda: inductor_triton_mm(a, b)
        )
        print(torch_ms, triton_ms, ind_aten_ms, ind_triton_ms, sep="; ")

        torch._dynamo.reset()


if __name__ == "__main__":
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
    print("test total time")
    test_total_time(shapes)

    print("test GPU time")
    test_GPU_time(shapes)


# Results Preview on AWS AI cluster
"""
test total time
shape; torch mm; triton mm; inductor aten mm; inductor triton mm
[128, 9216] x [9216, 4096]; 0.07240759208798409; 0.10885953903198242; 0.20063146017491817; 0.20054904278367758
[128, 4096] x [4096, 4096]; 0.03640300128608942; 0.10960095096379519; 0.09948539081960917; 0.0996188772842288
[128, 4096] x [4096, 1000]; 0.02215010579675436; 0.12592008337378502; 0.031120930798351765; 0.0370654184371233
[2048, 768] x [768, 768]; 0.023501068353652954; 0.10804693214595318; 0.03004650119692087; 0.0276932492852211
[2048, 768] x [768, 3072]; 0.045639658346772194; 0.10883208829909563; 0.062736920081079; 0.06480381824076176
[2048, 3072] x [3072, 768]; 0.054093082435429096; 0.10804777964949608; 0.08744294755160809; 0.07766005117446184
[1024, 768] x [768, 768]; 0.021525858901441097; 0.10909941978752613; 0.02656651195138693; 0.02683836966753006
[1024, 768] x [768, 3072]; 0.027319076471030712; 0.10825308971107006; 0.040118801407516; 0.039282338693737984
[1024, 3072] x [3072, 768]; 0.034132059663534164; 0.10594133753329515; 0.05069758277386427; 0.04572632722556591
[1024, 768] x [768, 2304]; 0.02529360819607973; 0.10486091021448374; 0.03724239766597748; 0.036449190229177475
test GPU time
shape; torch mm; triton mm; inductor aten mm; inductor triton mm
[128, 9216] x [9216, 4096]; 0.09113600105047226; 0.09011200070381165; 0.21606400609016418; 0.21606400609016418
[128, 4096] x [4096, 4096]; 0.053247999399900436; 0.05222399905323982; 0.1157120019197464; 0.1157120019197464
[128, 4096] x [4096, 1000]; 0.026623999699950218; 0.02969600073993206; 0.04710400104522705; 0.05222399905323982
[2048, 768] x [768, 768]; 0.02457600086927414; 0.020479999482631683; 0.04095999896526337; 0.03993599861860275
[2048, 768] x [768, 3072]; 0.05119999870657921; 0.05222399905323982; 0.07475200295448303; 0.07577600330114365
[2048, 3072] x [3072, 768]; 0.05939200147986412; 0.05222399905323982; 0.09830400347709656; 0.0870399996638298
[1024, 768] x [768, 768]; 0.01945599913597107; 0.016383999958634377; 0.03276799991726875; 0.03276799991726875
[1024, 768] x [768, 3072]; 0.03174399957060814; 0.03276799991726875; 0.053247999399900436; 0.053247999399900436
[1024, 3072] x [3072, 768]; 0.04403200000524521; 0.03379200026392937; 0.06860800087451935; 0.062463998794555664
[1024, 768] x [768, 2304]; 0.02969600073993206; 0.02969600073993206; 0.04915200173854828; 0.048128001391887665
"""

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `inductor_aten_mm`, `inductor_triton_mm`, `torch_mm`, `triton_mm`, `test_total_time`, `test_GPU_time`

**Key imports**: triton, time_with_torch_timer, torch, torch._dynamo, torch._dynamo.config, torch._inductor.config as config, benchmarker


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/microbenchmarks`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `triton`
- `benchmark_helper`: time_with_torch_timer
- `torch`
- `torch._dynamo`
- `torch._dynamo.config`
- `torch._inductor.config as config`
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
- [`bench_mm_fusion.py_docs.md`](./bench_mm_fusion.py_docs.md)
- [`dynamo_microbenchmarks.py_docs.md`](./dynamo_microbenchmarks.py_docs.md)
- [`benchmark_helper.py_docs.md`](./benchmark_helper.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`model.py_docs.md`](./model.py_docs.md)
- [`dynamo_guard_eval.py_docs.md`](./dynamo_guard_eval.py_docs.md)
- [`overheads.py_docs.md`](./overheads.py_docs.md)
- [`operatorbench.py_docs.md`](./operatorbench.py_docs.md)
- [`tensor_layout_mini_benchmark.py_docs.md`](./tensor_layout_mini_benchmark.py_docs.md)


## Cross-References

- **File Documentation**: `inductor_mm.py_docs.md`
- **Keyword Index**: `inductor_mm.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
