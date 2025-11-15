# Documentation: `benchmarks/dynamo/genai_layers/utils.py`

## File Metadata

- **Path**: `benchmarks/dynamo/genai_layers/utils.py`
- **Size**: 10,690 bytes (10.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
from scipy.stats import gmean

import torch
from torch._inductor.runtime.benchmarking import benchmarker


def benchmark_kernel_in_milliseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    with torch.compiler.set_stance("fail_on_recompile"):
        return benchmarker.benchmark_gpu(lambda: func(*args, **kwargs))


@dataclass
class Performance:
    # Benchmark setting usually the shape of the input tensor
    setting: str

    # Latency in milliseconds
    latency: float

    # Number of  memory access in bytes
    memory_bytes: float

    # Memory bandwidth in GB/s
    memory_bandwidth: float = 0.0

    # Compute intensity in FLOPs/byte
    compute_intensity: float = 0.0

    def __post_init__(self):
        self.memory_bandwidth = self.memory_bytes / (self.latency / 1000) / 1e9

    def __str__(self):
        return f"setting: {self.setting}, latency: {self.latency} ms, memory bandwidth: {self.memory_bandwidth} GB/s"


class BenchmarkKernel:
    def __init__(self, script_args):
        self.script_args = script_args
        self.name = self.__class__.__name__
        self.available_backends: list[str] = []
        self.compile_mode: str = script_args.compile_mode

        # mapping from backend to list of performance results
        self.profiling_results: defaultdict[str, list[Performance]] = defaultdict(list)

    def get_memory_bytes(self, args, kwargs) -> int:
        # Get the necessary memory access in bytes for the kernelßß
        raise NotImplementedError

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # Get a list of input shapes to benchmark the kernel
        raise NotImplementedError

    def eager(self, args, kwargs) -> Any:
        raise NotImplementedError

    def compiled(self, args, kwargs) -> Any:
        raise NotImplementedError

    def helion(self, args, kwargs) -> Any:
        raise NotImplementedError

    def quack(self, args, kwargs) -> Any:
        raise NotImplementedError

    def liger(self, args, kwargs) -> Any:
        raise NotImplementedError

    def triton(self, args, kwargs) -> Any:
        raise NotImplementedError

    def benchmark(self):
        raise NotImplementedError

    def clone_inputs(self, args, kwargs) -> Any:
        args_ref = [
            arg.clone().detach().requires_grad_(arg.requires_grad) for arg in args
        ]

        kwargs_ref = (
            {
                k: (
                    v.clone().detach().requires_grad_(v.requires_grad)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in kwargs.items()
            }
            if kwargs
            else kwargs
        )

        return args_ref, kwargs_ref

    def check_accuracy(self, args, kwargs) -> None:
        res = {}
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            res[backend] = getattr(self, backend)(args_ref, kwargs_ref)()

        if (
            "compiled" in self.available_backends
            and self.script_args.custom_compile_options
        ):
            torch._dynamo.reset()  # cause recompile
            with torch._inductor.config.patch(self.script_args.custom_compile_options):
                args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
                res[self.script_args.custom_compile_name] = self.compiled(
                    args_ref, kwargs_ref
                )()

        gold = res["eager"]

        tol = {}
        if self.script_args.tolerance:
            tol = {
                "atol": self.script_args.tolerance,
                "rtol": self.script_args.tolerance,
            }
        for backend in res:
            if backend == "eager":
                continue
            try:
                torch.testing.assert_close(res[backend], gold, **tol)
                for t, gold_t in zip(res[backend], gold):
                    if t.requires_grad:
                        torch.testing.assert_close(t.grad, gold_t.grad, **tol)
                print(
                    f"Accuracy check \033[92m✓ succeed\033[0m for {backend} backend on {self.name} kernel"
                )
            except Exception as e:
                print(
                    f"Accuracy check \033[91m✗ failed\033[0m for {backend} backend on {self.name} kernel. Error {e}"
                )
                if self.script_args.exit_on_accuracy_failure:
                    print("Exit right away since --exit-on-accuracy-failure is set")
                    sys.exit(1)

    def benchmark_single_shape_for_backend(
        self, backend, args, kwargs, setting, fn=None
    ) -> bool:
        if fn is None:
            fn = getattr(self, backend)
        args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
        try:
            avg_time = benchmark_kernel_in_milliseconds(fn(args_ref, kwargs_ref))
        except Exception as e:
            print(
                f"Failed to run {backend} backend on {self.name} kernel for {setting} due to {e}"
            )
            self.available_backends.remove(backend)  # noqa: B909
            return False
        mem_bytes = self.get_memory_bytes(args_ref, kwargs_ref)
        perf = Performance(setting, avg_time, mem_bytes)
        print(f"{self.name} kernel on {backend} backend. {perf}")
        self.profiling_results[backend].append(perf)
        return True

    def benchmark_single_shape(
        self, args, kwargs=None, should_check_accuracy=True, setting: str = ""
    ):
        for backend in self.available_backends:
            self.benchmark_single_shape_for_backend(backend, args, kwargs, setting)
        if (
            "compiled" in self.available_backends
            and self.script_args.custom_compile_options
        ):
            torch._dynamo.reset()  # cause recompile
            with torch._inductor.config.patch(self.script_args.custom_compile_options):
                status = self.benchmark_single_shape_for_backend(
                    self.script_args.custom_compile_name,
                    args,
                    kwargs,
                    setting,
                    fn=self.compiled,
                )
            if not status:
                self.script_args.custom_compile_options = (
                    None  # once fail, don't run again
                )

        if should_check_accuracy:
            self.check_accuracy(args, kwargs)

    def visualize(self) -> None:
        device_name = torch.cuda.get_device_name(0)
        visualize_comparison(
            self.profiling_results,
            title=f"{self.name} ({device_name})",
            output_path=f"{self.name}_bench",
        )
        return

    def report_geomean_speedup(self) -> None:
        print(f"Geomean speedup for benchmark {self.name}")
        eager_result = {
            result.setting: result for result in self.profiling_results["eager"]
        }
        print(f"  eager {len(eager_result)} data points")
        for backend, backend_result in self.profiling_results.items():
            if backend == "eager":
                continue
            speeduplist = []
            for result in backend_result:
                eager_latency = eager_result[result.setting].latency
                backend_latency = result.latency
                speeduplist.append(
                    eager_latency / backend_latency if backend_latency != 0 else 0.0
                )

            if len(speeduplist) > 0:
                print(
                    f"  {backend} {len(speeduplist)} data points, {gmean(speeduplist):.2f}x speedup"
                )


def get_backend_colors() -> dict[str, str]:
    """Get consistent color scheme for different backends."""
    return {
        "eager": "#1f77b4",  # blue
        "compiled": "#ff7f0e",  # orange
        "quack": "#2ca02c",  # green
        "liger": "#d62728",  # red
        "helion": "#9467bd",  # purple
        "triton": "#8c564b",  # brown
        "cutlass": "#e377c2",  # pink
        "flash_attn": "#7f7f7f",  # gray
        "default": "#000000",  # black
    }


def visualize_comparison(
    profiling_results: dict[str, list[Performance]],
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Create a single memory_bandwidth comparison plot from profiling results.

    Args:
        profiling_results: Dict mapping backend names to lists of Performance objects
        output_path: Path to save the plot (optional)
    """
    # Get backend colors
    backend_colors = get_backend_colors()

    # Extract settings from eager backend which runs all settings
    all_settings = []
    for perf in profiling_results["eager"]:
        all_settings.append(perf.setting)

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for backend in profiling_results:
        backend_perfs = profiling_results[backend]
        perf_dict = {perf.setting: perf for perf in backend_perfs}

        x_vals = []
        y_vals = []
        for i, setting in enumerate(all_settings):
            if setting in perf_dict:
                x_vals.append(i)
                y_vals.append(perf_dict[setting].memory_bandwidth)

        if x_vals:  # Only plot if we have data
            color = backend_colors.get(backend, backend_colors["default"])
            ax.plot(
                x_vals,
                y_vals,
                "o-",
                label=backend,
                color=color,
                linewidth=2,
                markersize=8,
                alpha=0.8,
            )

    # Configure the plot
    ax.set_title(title or "Memory Bandwidth Comparison", fontsize=16)
    ax.set_xlabel("Shape", fontsize=12)
    ax.set_ylabel("memory bandwidth (GB/s)", fontsize=12)
    ax.set_xticks(range(len(all_settings)))
    ax.set_xticklabels(
        [
            s.replace("shape: ", "").replace("[", "").replace("]", "")
            for s in all_settings
        ],
        rotation=45,
        ha="right",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot if output path is provided
    if output_path:
        # Save as PNG
        os.makedirs("pics", exist_ok=True)
        full_path = os.path.join("pics", output_path + ".png")
        plt.savefig(full_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to {full_path}")

    plt.close()

```



## High-Level Overview


This Python file contains 3 class(es) and 21 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Performance`, `BenchmarkKernel`

**Functions defined**: `benchmark_kernel_in_milliseconds`, `__post_init__`, `__str__`, `__init__`, `get_memory_bytes`, `get_shapes`, `eager`, `compiled`, `helion`, `quack`, `liger`, `triton`, `benchmark`, `clone_inputs`, `check_accuracy`, `benchmark_single_shape_for_backend`, `benchmark_single_shape`, `visualize`, `report_geomean_speedup`, `get_backend_colors`

**Key imports**: os, sys, defaultdict, Callable, dataclass, Any, Optional, matplotlib.pyplot as plt, gmean, torch, benchmarker


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/genai_layers`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `collections`: defaultdict
- `collections.abc`: Callable
- `dataclasses`: dataclass
- `typing`: Any, Optional
- `matplotlib.pyplot as plt`
- `scipy.stats`: gmean
- `torch`
- `torch._inductor.runtime.benchmarking`: benchmarker


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`benchmarks/dynamo/genai_layers`):

- [`kernels.py_docs.md`](./kernels.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`requirements.txt_docs.md`](./requirements.txt_docs.md)
- [`benchmark.py_docs.md`](./benchmark.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
