# Documentation: `benchmarks/dynamo/microbenchmarks/operatorbench.py`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operatorbench.py`
- **Size**: 12,350 bytes (12.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
import csv
import itertools
import sys
import time
import warnings
from contextlib import nullcontext

import click
import numpy as np
from operator_inp_utils import OperatorInputsLoader
from tqdm import tqdm

import torch
from torch._dynamo.backends.cudagraphs import cudagraphs_inner
from torch._dynamo.testing import same
from torch._inductor.compile_fx import compile_fx
from torch._inductor.decomposition import decompositions
from torch._inductor.lowering import lowerings
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.utils import gen_gm_and_inputs
from torch.utils._pytree import tree_map_only


aten = torch.ops.aten
profile_enabled = False
inductor_config_options = {
    "halide": {"cpu_backend": "halide", "cuda_backend": "halide"},
    "autotune": {
        "max_autotune_pointwise": True,
        "max_autotune": True,
        "max_autotune_gemm": True,
        "coordinate_descent_tuning": True,
    },
}


def maybe_record_function(name):
    return torch.profiler.record_function(name) if profile_enabled else nullcontext()


def compute_speedups(
    operator, models, example_inputs, repeats, accuracy_checking=False, device="cuda"
):
    expected = models[0](*example_inputs)
    if accuracy_checking:
        for model in models[1:]:
            actual = model(*example_inputs)
            # change to assert later
            try:
                same(actual, expected, cos_similarity=True, equal_nan=True)
            except AssertionError as e:
                print(e)
                print(f"Accuracy check failed: {operator}")
                print((expected[0] - actual[0]).abs().max())

    timings = np.zeros((repeats, len(models)), np.float64)
    for rep in range(repeats):
        with maybe_record_function(f"rep_{rep}"):
            # interleave the runs to handle frequency scaling and load changes
            for m, model in enumerate(models):
                with maybe_record_function(f"model_{m}"):
                    if device == "cuda":
                        model(*example_inputs)

                        # benchmarker.benchmark_gpu() clears L2 cache to hide the latency of CPU launch time
                        # along with cuda synchronization
                        timings[rep, m] = benchmarker.benchmark_gpu(
                            lambda: model(*example_inputs)
                        )
                    else:
                        from torch._inductor.utils import timed

                        timings[rep, m] = timed(model, example_inputs)
    return np.median(timings, axis=0)


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


def convert_to_jit(gm, gm_args):
    strip_overloads(gm)
    try:
        return torch.jit.script(gm)
    except Exception:
        pass
    return torch.jit.trace(gm, gm_args)


def to_channels_last(ten):
    return ten if ten.ndim != 4 else ten.to(memory_format=torch.channels_last)


def microbenchmark(
    operator,
    args,
    kwargs,
    accuracy_checking,
    repeats,
    inductor_configs,
    measure_nvfuser,
    device,
):
    gm, gm_args = gen_gm_and_inputs(operator, args, kwargs)
    torch.jit._builtins._register_builtin(
        torch.ops.aten.convolution_backward.default, "aten::convolution_backward"
    )
    compiled = [gm]
    for config in inductor_configs:
        t = -time.perf_counter()
        compiled.append(compile_fx(gm, gm_args, config_patches=config))
        t += time.perf_counter()
        if t > 10:
            print(f"slow compile inductor {t:.1f}s {config}")

    if measure_nvfuser:
        g = convert_to_jit(gm, gm_args)
        cudagraphs_jit = cudagraphs_inner(
            g, gm_args, copy_outputs=False, copy_inputs=False
        )
        compiled += [cudagraphs_jit]
    if accuracy_checking:
        repeats = 1

    medians = compute_speedups(
        operator, compiled, gm_args, repeats, accuracy_checking, device
    )
    return medians


quantiles_thresholds = (0.2, 0.5, 0.8)


def quantiles(timings):
    return np.quantile(timings, quantiles_thresholds).tolist()


def skip_operator(operator):
    nyi_strings = (
        "aten.gather.default",
        "nll_loss",
        "aten.index",
        "aten.scatter_",
        "masked_fill_.Scalar",
    )

    if any(nyi_string in str(operator) for nyi_string in nyi_strings):
        # maybe disable aten.native_layer_norm.default
        # TODO - inputs cannot be randomly initialized, causes cyda failures
        print(f"Skipping {operator}, input generator nyi")
        return True

    # not covered by other non-compute operator heuristics
    if operator == torch.ops.aten._unsafe_view.default:
        print(f"Skipping {operator}, non compute operator")
        return True

    # some of inductor registered to the OpOverload, some registered to OpOverloadPacket
    op_impls = [operator]
    if isinstance(operator, torch._ops.OpOverload):
        op_impls.append(operator.overloadpacket)

    # TODO - skip benchmarking fallbacks. for some ops we have both lowerings and fallbacks
    # so its not clear just from operator what will be lowered.

    if all(op not in decompositions and op not in lowerings for op in op_impls):
        print(f"Skipping {operator}, no inductor impl")
        return True

    if "convolution" in str(operator):
        return True

    return False


@click.command()
@click.option(
    "--suite",
    help="suite to load inps from: options: timm, huggingface, torchbench",
    default="torchbench",
)
@click.option("--op", help="operator overload to benchmark", default="all")
@click.option("--dtype", help="dtype to benchmark", default="float32")
@click.option("--max-samples", help="max samples per op", default=15)
@click.option("--accuracy-checking", help="check accuracy", default=False)
@click.option(
    "--repeats", help="how many times to repeat for perf measurement", default=3
)
@click.option(
    "--inductor-config",
    multiple=True,
    help="Custom inductor config, options: " + ", ".join(inductor_config_options),
)
@click.option(
    "--measure-nvfuser/--no-measure-nvfuser",
    help="default we only measure inductor",
    default=False,
)
@click.option("--device", help="cpu or cuda", default="cuda")
@click.option("--inp-file", help="use custom input file instead of suite", default=None)
@click.option("--start-idx", help="specify start index of samples", default=0)
@click.option(
    "--channels-last", help="force inputs to channels last", is_flag=True, default=False
)
@click.option("--profile", help="profile the benchmark", is_flag=True, default=False)
def benchmark(
    suite,
    op,
    dtype,
    max_samples,
    accuracy_checking,
    repeats,
    inductor_config,
    measure_nvfuser,
    device,
    inp_file,
    start_idx,
    channels_last,
    profile,
):
    warnings.filterwarnings("ignore", module="torch.jit._check")
    torch.set_float32_matmul_precision("high")
    global profile_enabled

    if inp_file is not None:
        loader = OperatorInputsLoader(inp_file)
    else:
        assert suite in ("timm", "huggingface", "torchbench"), f"got {suite}"
        if suite == "timm":
            loader = OperatorInputsLoader.get_timm_loader()
        elif suite == "huggingface":
            loader = OperatorInputsLoader.get_huggingface_loader()
        else:
            loader = OperatorInputsLoader.get_torchbench_loader()

    assert dtype in ("float16", "float32"), f"got {dtype}"

    inductor_configs = [{}]
    backend_names = ["inductor"]
    for name in inductor_config or ():
        backend_names.append(name)
        inductor_configs.append(inductor_config_options[name])
    if measure_nvfuser:
        backend_names.append("nvfuser")

    compare2 = len(backend_names) == 2
    if compare2:
        a, b = backend_names
        backend_names.append(f"{a}/{b}")

    output_fd = None
    output_csv = None
    if op == "all":
        filename = f"operatorbench_{suite}_{dtype}.csv"
        output_fd = open(filename, "w")
        output_csv = csv.writer(output_fd)
        output_csv.writerow(
            [
                "operator",
                *[
                    f"{a} {b}"
                    for a, b in itertools.product(
                        backend_names,
                        [f"{x * 100:.0f}th" for x in quantiles_thresholds],
                    )
                ],
                "elapsed",
                *map("{} abs".format, ["eager", *backend_names]),
            ]
        )

    dtype = torch.float16 if dtype == "float16" else torch.float32

    if op == "all":
        ops = loader.get_all_ops()
    else:
        ops = [eval(op)]

    max_samples = max_samples + start_idx
    profile_enabled = profile

    for operator in ops:
        if skip_operator(operator):
            continue
        start = time.perf_counter()
        inp_gen = loader.get_inputs_for_operator(operator, dtype=dtype, device=device)
        timings = []
        inputs_list = []
        for _ in range(min(max_samples, 1000000)):
            try:
                inps = next(inp_gen)
                inputs_list.append(inps)
            except StopIteration:
                break

        profiler_context = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"./log/operator_{operator}", use_gzip=True
                ),
            )
            if profile_enabled
            else nullcontext()
        )
        with profiler_context:
            for i, inps in enumerate(tqdm(inputs_list[start_idx:], desc=str(operator))):
                if inps is None:
                    break
                args, kwargs = inps
                if channels_last:
                    args, kwargs = tree_map_only(
                        torch.Tensor, to_channels_last, (args, kwargs)
                    )
                try:
                    with maybe_record_function(f"iter_{i}"):
                        # aten, nvfuser, inductor
                        timings.append(
                            microbenchmark(
                                operator,
                                args,
                                kwargs,
                                accuracy_checking,
                                repeats,
                                inductor_configs,
                                measure_nvfuser,
                                device,
                            )
                        )
                except Exception as e:
                    print(f"error {operator} input {i}: {type(e).__name__}: {e}")
                    # comment out this line to avoid blocking other tests
                    # raise e

        if not timings:
            continue

        timings = np.stack(timings)
        speedups = [
            quantiles(timings[:, 0] / timings[:, x]) for x in range(1, timings.shape[1])
        ]
        if compare2:
            speedups.append(quantiles(timings[:, 1] / timings[:, 2]))
        assert len(backend_names) == len(speedups)

        row = [f"{operator}"]
        sys.stdout.write(f"{operator}: ")
        for backend, (low, mid, high) in zip(backend_names, speedups):
            sys.stdout.write(f"{backend}={mid:.4f}x ({low:.4f}-{high:.4f}) ")
            row.extend(map("{:.6f}".format, [low, mid, high]))
        elapsed = time.perf_counter() - start
        row.append(f"{elapsed:1f}")
        row.extend(map("{:.8f}".format, np.mean(timings, axis=0).tolist()))
        sys.stdout.write(f"took {elapsed:.0f}s\n")
        sys.stdout.flush()
        if output_csv:
            output_csv.writerow(row)
            output_fd.flush()

    if output_fd:
        print(f"Wrote {filename}")
        output_fd.close()


if __name__ == "__main__":
    benchmark()

```



## High-Level Overview


This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `maybe_record_function`, `compute_speedups`, `strip_overloads`, `convert_to_jit`, `to_channels_last`, `microbenchmark`, `quantiles`, `skip_operator`, `benchmark`

**Key imports**: csv, itertools, sys, time, warnings, nullcontext, click, numpy as np, OperatorInputsLoader, tqdm


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/microbenchmarks`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `csv`
- `itertools`
- `sys`
- `time`
- `warnings`
- `contextlib`: nullcontext
- `click`
- `numpy as np`
- `operator_inp_utils`: OperatorInputsLoader
- `tqdm`: tqdm
- `torch`
- `torch._dynamo.backends.cudagraphs`: cudagraphs_inner
- `torch._dynamo.testing`: same
- `torch._inductor.compile_fx`: compile_fx
- `torch._inductor.decomposition`: decompositions
- `torch._inductor.lowering`: lowerings
- `torch._inductor.runtime.benchmarking`: benchmarker
- `torch._inductor.utils`: gen_gm_and_inputs
- `torch.utils._pytree`: tree_map_only


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

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
- [`tensor_layout_mini_benchmark.py_docs.md`](./tensor_layout_mini_benchmark.py_docs.md)


## Cross-References

- **File Documentation**: `operatorbench.py_docs.md`
- **Keyword Index**: `operatorbench.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
