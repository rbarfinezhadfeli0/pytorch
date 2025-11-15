# Documentation: `benchmarks/gpt_fast/benchmark.py`

## File Metadata

- **Path**: `benchmarks/gpt_fast/benchmark.py`
- **Size**: 10,670 bytes (10.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import csv
import dataclasses
import json
import os

from common import all_experiments, Experiment, register_experiment
from generate import get_arch_name

import torch
import torch.nn as nn
from torch._inductor.runtime.benchmarking import benchmarker
from torch.utils.flop_counter import FlopCounterMode


WARMUP_ITER = 5

A100_40G_BF16_TFLOPS = 312


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim, dtype=dtype),
                nn.LayerNorm(hidden_dim, dtype=dtype),
                nn.Linear(hidden_dim, output_dim, dtype=dtype),
                nn.LayerNorm(output_dim, dtype=dtype),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@register_experiment(name="mlp_layer_norm_gelu")
def run_mlp_layer_norm_gelu(device: str = "cuda"):
    dtype_flops_utilization_map = {
        torch.bfloat16: "0.8",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    intermediate_size = 14336
    results = []
    for dtype, expected_flops_utilization in dtype_flops_utilization_map.items():
        flops_utilization = 0
        for D in input_shapes:
            mod = SimpleMLP(
                input_dim=D, hidden_dim=intermediate_size, output_dim=D, dtype=dtype
            ).to(device)

            x = torch.randn(D, device=device, dtype=torch.bfloat16)

            with FlopCounterMode(display=False) as mode:
                mod(x)

            flops = mode.get_total_flops()

            compiled_mod = torch.compile(mod, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_mod(x)

            us_per_iter = benchmarker.benchmark(compiled_mod, (x,), {}) * 1000
            flops_utilization += us_per_iter * flops / 1e9 / A100_40G_BF16_TFLOPS

        flops_utilization = flops_utilization / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "mlp_layer_norm_gelu",
                "flops_utilization",
                expected_flops_utilization,
                f"{flops_utilization:.02f}",
                dtype_str,
                device,
                get_arch_name(),
            )
        )
    return results


@register_experiment(name="layer_norm")
def run_layer_norm(device: str = "cuda"):
    dtype_memory_bandwidth_map = {
        torch.bfloat16: "950",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    BS = 4096
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:
            mod = nn.LayerNorm(D).to(device)

            x = torch.randn(BS, D, device=device, dtype=dtype)

            compiled_mod = torch.compile(mod, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_mod(x)

            us_per_iter = benchmarker.benchmark(compiled_mod, (x,), {}) * 1000
            memory_bandwidth += (1e6 / us_per_iter) * 2 * BS * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "layer_norm",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                dtype_str,
                device,
                get_arch_name(),
            )
        )
    return results


@register_experiment(name="gather_gemv")
@torch._inductor.config.patch(coordinate_descent_tuning=True)
def run_gather_gemv(device: str = "cuda"):
    E = 8
    dtype_memory_bandwidth_map = {
        torch.int8: "990",
        torch.bfloat16: "1060",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:

            def gather_gemv(W, score_idxs, x):
                return W[score_idxs].to(x.dtype) @ x

            W = torch.randn(E, D, D, device=device).to(dtype=dtype)
            x = torch.randn(D, device=device, dtype=torch.bfloat16)
            score_idxs = torch.tensor([3, 5], device=device)

            compiled_fn = torch.compile(gather_gemv, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_fn(W, score_idxs, x)

            us_per_iter = (
                benchmarker.benchmark(
                    compiled_fn,
                    (
                        W,
                        score_idxs,
                        x,
                    ),
                    {},
                )
                * 1000
            )
            memory_bandwidth += (1e6 / us_per_iter) * 2 * D * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "gather_gemv",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                dtype_str,
                device,
                get_arch_name(),
            )
        )
    return results


@register_experiment(name="gemv")
@torch._inductor.config.patch(coordinate_descent_tuning=True)
def run_gemv(device: str = "cuda"):
    dtype_memory_bandwidth_map = {
        torch.int8: "870",
        torch.bfloat16: "990",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:

            def gemv(W, x):
                return W.to(x.dtype) @ x

            W = torch.randn(D, D, device=device).to(dtype=dtype)
            x = torch.randn(D, device=device, dtype=torch.bfloat16)

            compiled_fn = torch.compile(gemv, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_fn(W, x)

            us_per_iter = (
                benchmarker.benchmark(
                    compiled_fn,
                    (
                        W,
                        x,
                    ),
                    {},
                )
                * 1000
            )
            memory_bandwidth += (1e6 / us_per_iter) * D * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                "gemv",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                dtype_str,
                device,
                get_arch_name(),
            )
        )
    return results


def output_csv(output_file, headers, row):
    if os.path.exists(output_file):
        with open(output_file) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                # if prior results failed the header might not be filled in yet
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]

    if output_file != DEFAULT_OUTPUT_FILE:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    with open(output_file, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


def output_json(output_file, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    record = {
        "benchmark": {
            "name": "PyTorch gpt-fast benchmark",
            "mode": "inference",
            "dtype": mapping_headers["dtype"],
            "extra_info": {
                "device": mapping_headers["device"],
                "arch": mapping_headers["arch"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "type": "OSS model" if mapping_headers["is_model"] else "micro-benchmark",
            "origins": ["pytorch"],
        },
        "metric": {
            "name": mapping_headers["metric"],
            "benchmark_values": [mapping_headers["actual"]],
            "target_value": mapping_headers["target"],
        },
    }

    with open(f"{os.path.splitext(output_file)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)


DEFAULT_OUTPUT_FILE = "gpt_fast_benchmark.csv"


def main(output_file=DEFAULT_OUTPUT_FILE, only_model=None):
    results = []

    if not only_model:
        experiments = all_experiments.values()
    else:
        if only_model not in all_experiments:
            print(
                f"Unknown model: {only_model}, all available models: {all_experiments.keys()}"
            )
        # only run the specified model
        experiments = [all_experiments[only_model]]
    for func in experiments:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except AssertionError:
            # This happens when torch is compiled with CUDA turning off completely
            device = "cpu"

        torch.compiler.cudagraph_mark_step_begin()
        lst = func(device)
        for x in lst:
            results.append(dataclasses.astuple(x))

    headers = [field.name for field in dataclasses.fields(Experiment)]

    for row in results:
        output_csv(output_file, headers, row)
        # Also write the output in JSON format so that it can be ingested into the OSS benchmark database
        output_json(output_file, headers, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Set the output CSV file to save the benchmark results",
    )
    parser.add_argument(
        "--only",
        help="Specify a model or micro-benchmark name to run exclusively",
    )
    args = parser.parse_args()

    main(output_file=args.output, only_model=args.only)

```



## High-Level Overview


This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SimpleMLP`

**Functions defined**: `__init__`, `forward`, `run_mlp_layer_norm_gelu`, `run_layer_norm`, `run_gather_gemv`, `gather_gemv`, `run_gemv`, `gemv`, `output_csv`, `output_json`, `main`

**Key imports**: argparse, csv, dataclasses, json, os, all_experiments, Experiment, register_experiment, get_arch_name, torch, torch.nn as nn, benchmarker


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/gpt_fast`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `csv`
- `dataclasses`
- `json`
- `os`
- `common`: all_experiments, Experiment, register_experiment
- `generate`: get_arch_name
- `torch`
- `torch.nn as nn`
- `torch._inductor.runtime.benchmarking`: benchmarker
- `torch.utils.flop_counter`: FlopCounterMode


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/gpt_fast`):

- [`model.py_docs.md`](./model.py_docs.md)
- [`mixtral_moe_quantize.py_docs.md`](./mixtral_moe_quantize.py_docs.md)
- [`quantize.py_docs.md`](./quantize.py_docs.md)
- [`mixtral_moe_model.py_docs.md`](./mixtral_moe_model.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)
- [`generate.py_docs.md`](./generate.py_docs.md)


## Cross-References

- **File Documentation**: `benchmark.py_docs.md`
- **Keyword Index**: `benchmark.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
