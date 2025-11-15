# Documentation: `docs/functorch/benchmarks/cse.py_docs.md`

## File Metadata

- **Path**: `docs/functorch/benchmarks/cse.py_docs.md`
- **Size**: 4,967 bytes (4.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file contains **examples or benchmarks**.

## Original Source

```markdown
# Documentation: `functorch/benchmarks/cse.py`

## File Metadata

- **Path**: `functorch/benchmarks/cse.py`
- **Size**: 2,432 bytes (2.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
import torch
import torch.fx as fx
from functorch import make_fx
from torch._functorch.compile_utils import fx_graph_cse
from torch.profiler import profile, ProfilerActivity


def profile_it(f, inp):
    for _ in range(5):
        f(inp)

    itr = 5
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(itr):
            f(inp)

    timing = prof.key_averages()
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total
    return cuda_time_total / itr


def profile_function(name, f, inp):
    fx_g = make_fx(f)(inp)

    new_g = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_g)
    # do not benchmark against the scripted version because script already does some CSE
    # script_f = torch.jit.script(fx_g)
    # script_g = torch.jit.script(new_g)
    # avg_cuda_time_f = profile_it(script_f, inp)
    # avg_cuda_time_g = profile_it(script_g, inp)
    avg_cuda_time_f = profile_it(fx_g, inp)
    avg_cuda_time_g = profile_it(new_g, inp)
    num_node_decrease = len(fx_g.graph.nodes) - len(new_g.graph.nodes)

    print(
        f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {num_node_decrease}, {len(fx_g.graph.nodes)}"
    )


g_gpu = torch.Generator(device="cuda")
g_gpu.manual_seed(2147483647)
inp = torch.randn(2**20, device="cuda", generator=g_gpu)


def f1(x):
    return x.cos().cos()


profile_function("f1", f1, inp)


def fsum(x):
    a = x.sum()
    b = x.sum()
    c = x.sum()
    d = x.sum()
    return a + b + c + d


profile_function("fsum", fsum, inp)


def fconcat(x):
    a = torch.cat((x, x))
    b = torch.cat((x, x))
    return a + b


profile_function("fconcat", fconcat, inp)


def fsum2(x):
    a = x.sum()
    for _ in range(30):
        a = a + x.sum()
    return a


profile_function("fsum2", fsum2, inp)


def fsummulti(x):
    a = 0
    for _ in range(3):
        a = a + x.sum()
        a = a * x.sum()
    return a


profile_function("fsummulti", fsummulti, inp)


def fsummulti2(x):
    a = 0
    for _ in range(30):
        a = a + x.sum()
        a = a * x.sum()
    return a


profile_function("fsummulti2", fsummulti2, inp)


def fcos(x):
    a = 0
    for _ in range(3):
        a = a + x.cos()
    return a


profile_function("fcos", fcos, inp)


def fcos2(x):
    a = 0
    for _ in range(30):
        a = a + x.cos()
    return a


profile_function("fcos2", fcos2, inp)

```



## High-Level Overview


This Python file contains 0 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `profile_it`, `profile_function`, `f1`, `fsum`, `fconcat`, `fsum2`, `fsummulti`, `fsummulti2`, `fcos`, `fcos2`

**Key imports**: torch, torch.fx as fx, make_fx, fx_graph_cse, profile, ProfilerActivity


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `functorch/benchmarks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.fx as fx`
- `functorch`: make_fx
- `torch._functorch.compile_utils`: fx_graph_cse
- `torch.profiler`: profile, ProfilerActivity


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

Files in the same folder (`functorch/benchmarks`):

- [`operator_authoring.py_docs.md`](./operator_authoring.py_docs.md)
- [`pointwise_scorecard.py_docs.md`](./pointwise_scorecard.py_docs.md)
- [`per_sample_grads.py_docs.md`](./per_sample_grads.py_docs.md)
- [`chrome_trace_parser.py_docs.md`](./chrome_trace_parser.py_docs.md)
- [`process_scorecard.py_docs.md`](./process_scorecard.py_docs.md)


## Cross-References

- **File Documentation**: `cse.py_docs.md`
- **Keyword Index**: `cse.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/functorch/benchmarks`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/functorch/benchmarks`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/functorch/benchmarks`):

- [`cse.py_kw.md_docs.md`](./cse.py_kw.md_docs.md)
- [`pointwise_scorecard.py_kw.md_docs.md`](./pointwise_scorecard.py_kw.md_docs.md)
- [`process_scorecard.py_kw.md_docs.md`](./process_scorecard.py_kw.md_docs.md)
- [`per_sample_grads.py_kw.md_docs.md`](./per_sample_grads.py_kw.md_docs.md)
- [`chrome_trace_parser.py_docs.md_docs.md`](./chrome_trace_parser.py_docs.md_docs.md)
- [`pointwise_scorecard.py_docs.md_docs.md`](./pointwise_scorecard.py_docs.md_docs.md)
- [`operator_authoring.py_docs.md_docs.md`](./operator_authoring.py_docs.md_docs.md)
- [`process_scorecard.py_docs.md_docs.md`](./process_scorecard.py_docs.md_docs.md)
- [`chrome_trace_parser.py_kw.md_docs.md`](./chrome_trace_parser.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cse.py_docs.md_docs.md`
- **Keyword Index**: `cse.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
