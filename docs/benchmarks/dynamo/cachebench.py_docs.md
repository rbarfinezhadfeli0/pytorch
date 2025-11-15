# Documentation: `benchmarks/dynamo/cachebench.py`

## File Metadata

- **Path**: `benchmarks/dynamo/cachebench.py`
- **Size**: 7,121 bytes (6.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable

from torch._inductor.utils import fresh_cache


logger: logging.Logger = logging.getLogger(__name__)

TIMEOUT: int = 2000


# Keep in sync with .ci/pytorch/test.sh
TORCHBENCH_MODELS: list[str] = [
    "nanogpt",
    "BERT_pytorch",
    "resnet50",
    "moco",
    "llama",
]
HUGGINGFACE_MODELS: list[str] = [
    "AllenaiLongformerBase",
    "BertForMaskedLM",
    "GPT2ForSequenceClassification",
]


@dataclasses.dataclass
class RunResult:
    model: str
    mode: str  # inference or training
    benchmark: str
    dynamic: bool
    device: str  # cuda or cpu
    cold_compile_s: list[float]
    warm_compile_s: list[float]
    speedup_pct: float


def get_compile_time(file: tempfile._TemporaryFileWrapper) -> float:
    lines = file.readlines()
    # Decode from byte string, remove new lines, parse csv
    lines = [line.decode("utf-8").strip().split(",") for line in lines]
    compilation_time_idx = lines[0].index("compilation_latency")
    compilation_time = lines[1][compilation_time_idx]
    return float(compilation_time)


def _run_torchbench_from_args(
    cmd_args: argparse.Namespace,
    model: str,
    args: list[str],
) -> tuple[list[float], list[float]]:
    cold_compile_time: list[float] = []
    warm_compile_time: list[float] = []

    for _ in range(cmd_args.repeat):
        with fresh_cache():
            env = os.environ.copy()
            with tempfile.NamedTemporaryFile(suffix=".csv") as file:
                args.append("--output=" + file.name)
                logger.info(f"Performing cold-start run for {model}")  # noqa: G004
                subprocess.check_call(args, timeout=TIMEOUT, env=env)
                cold_compile_time.append(get_compile_time(file))

            args.pop()
            with tempfile.NamedTemporaryFile(suffix=".csv") as file:
                args.append("--output=" + file.name)
                logger.info(f"Performing warm-start run for {model}")  # noqa: G004
                subprocess.check_call(args, timeout=TIMEOUT, env=env)
                warm_compile_time.append(get_compile_time(file))

    return cold_compile_time, warm_compile_time


MODE_ARGS_DICT = {
    "inference": ["--inference", "--bfloat16"],
    "training": ["--training", "--amp"],
}


BENCHMARK_FILE = {
    "torchbench": "torchbench.py",
    "huggingface": "huggingface.py",
}


def _run_torchbench_model(
    cmd_args: argparse.Namespace,
    results: list[RunResult],
    model: str,
) -> None:
    cur_file = os.path.abspath(__file__)
    torchbench_file = os.path.join(
        os.path.dirname(cur_file), BENCHMARK_FILE[cmd_args.benchmark]
    )
    assert os.path.exists(torchbench_file), (
        f"Torchbench does not exist at {torchbench_file}"
    )

    dynamic = cmd_args.dynamic
    dynamic_args = ["--dynamic-shapes", "--dynamic-batch-only"] if dynamic else []

    args = (
        [
            sys.executable,
            torchbench_file,
            f"--only={model}",
            "--repeat=1",
            "--performance",
            "--backend=inductor",
            f"--device={cmd_args.device}",
        ]
        + MODE_ARGS_DICT[cmd_args.mode]
        + dynamic_args
    )

    logger.info(f"Command: {args}")  # noqa: G004
    try:
        cold_compile_t, warm_compile_t = _run_torchbench_from_args(
            cmd_args, model, args
        )
        speedup_pct = (1 - (sum(warm_compile_t) / sum(cold_compile_t))) * 100
        results.append(
            RunResult(
                model=model,
                mode=cmd_args.mode,
                benchmark=cmd_args.benchmark,
                dynamic=dynamic,
                device=cmd_args.device,
                cold_compile_s=cold_compile_t,
                warm_compile_s=warm_compile_t,
                speedup_pct=speedup_pct,
            )
        )
    except Exception:
        logger.info("fail", exc_info=True)
        return None


def _write_results_to_json(
    cmd_args: argparse.Namespace,
    results: list[RunResult],
) -> None:
    if len(results) == 0:
        # do not write empty results
        return

    records = []
    for result in results:
        for metric_name, value in [
            ("Cold compile time (s)", result.cold_compile_s),
            ("Warm compile time (s)", result.warm_compile_s),
            ("Speedup (%)", [result.speedup_pct]),
        ]:
            records.append(
                {
                    "benchmark": {
                        "name": "TorchCache Benchmark",
                        "mode": result.mode,
                        "extra_info": {
                            "is_dynamic": result.dynamic,
                            "device": result.device,
                        },
                    },
                    "model": {
                        "name": result.model,
                        "backend": "inductor",
                        "origins": [result.benchmark],
                    },
                    "metric": {
                        "name": metric_name,
                        "type": "OSS model",
                        "benchmark_values": value,
                    },
                }
            )
    with open(cmd_args.output, "w") as f:
        json.dump(records, f)


def parse_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TorchCache benchmark.")
    parser.add_argument(
        "-m",
        "--model",
        help="Name of the model to run",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Whether to run with dynamic enabled",
    )
    parser.add_argument(
        "--benchmark",
        choices=("torchbench", "huggingface"),
        required=True,
        help="Name of benchmark suite to run",
    )
    parser.add_argument(
        "--mode",
        choices=("inference", "training"),
        default="training",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The output filename (json)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        choices=range(1, 10),
        help="Number of times to repeat the compilation (reduce noise)",
    )
    args, _ = parser.parse_known_args()
    return args


Dispatch_fn_t = Callable[[argparse.Namespace, list[RunResult], str], None]


def main() -> None:
    cmd_args = parse_cmd_args()

    dispatcher: dict[str, tuple[Dispatch_fn_t, list[str]]] = {
        "torchbench": (_run_torchbench_model, TORCHBENCH_MODELS),
        "huggingface": (_run_torchbench_model, HUGGINGFACE_MODELS),
    }
    fn, models = dispatcher[cmd_args.benchmark]
    if cmd_args.model is not None:
        models = [cmd_args.model]

    results: list[RunResult] = []
    for model in models:
        fn(cmd_args, results, model)

    _write_results_to_json(cmd_args, results)


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RunResult`

**Functions defined**: `get_compile_time`, `_run_torchbench_from_args`, `_run_torchbench_model`, `_write_results_to_json`, `parse_cmd_args`, `main`

**Key imports**: argparse, dataclasses, json, logging, os, subprocess, sys, tempfile, Callable, fresh_cache


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `dataclasses`
- `json`
- `logging`
- `os`
- `subprocess`
- `sys`
- `tempfile`
- `collections.abc`: Callable
- `torch._inductor.utils`: fresh_cache


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

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/dynamo`):

- [`timm_models_list_cpu.txt_docs.md`](./timm_models_list_cpu.txt_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`benchmarks.py_docs.md`](./benchmarks.py_docs.md)
- [`check_graph_breaks.py_docs.md`](./check_graph_breaks.py_docs.md)
- [`check_csv.py_docs.md`](./check_csv.py_docs.md)
- [`all_torchbench_models_list.txt_docs.md`](./all_torchbench_models_list.txt_docs.md)
- [`check_accuracy.py_docs.md`](./check_accuracy.py_docs.md)
- [`torchbench_models_list_cpu.txt_docs.md`](./torchbench_models_list_cpu.txt_docs.md)
- [`timm_models.py_docs.md`](./timm_models.py_docs.md)


## Cross-References

- **File Documentation**: `cachebench.py_docs.md`
- **Keyword Index**: `cachebench.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
