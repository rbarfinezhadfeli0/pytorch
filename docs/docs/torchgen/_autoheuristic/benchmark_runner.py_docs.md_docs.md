# Documentation: `docs/torchgen/_autoheuristic/benchmark_runner.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/_autoheuristic/benchmark_runner.py_docs.md`
- **Size**: 5,377 bytes (5.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/_autoheuristic/benchmark_runner.py`

## File Metadata

- **Path**: `torchgen/_autoheuristic/benchmark_runner.py`
- **Size**: 2,493 bytes (2.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import argparse
import random
import time
from abc import abstractmethod
from typing import Any

from tqdm import tqdm  # type: ignore[import-untyped]

import torch


class BenchmarkRunner:
    """
    BenchmarkRunner is a base class for all benchmark runners. It provides an interface to run benchmarks in order to
    collect data with AutoHeuristic.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

    def add_base_arguments(self) -> None:
        self.parser.add_argument(
            "--device",
            type=int,
            default=None,
            help="torch.cuda.set_device(device) will be used",
        )
        self.parser.add_argument(
            "--use-heuristic",
            action="store_true",
            help="Use learned heuristic instead of collecting data.",
        )
        self.parser.add_argument(
            "-o",
            type=str,
            default="ah_data.txt",
            help="Path to file where AutoHeuristic will log results.",
        )
        self.parser.add_argument(
            "--num-samples",
            type=int,
            default=1000,
            help="Number of samples to collect.",
        )
        self.parser.add_argument(
            "--num-reps",
            type=int,
            default=3,
            help="Number of measurements to collect for each input.",
        )

    def run(self) -> None:
        torch.set_default_device("cuda")
        args = self.parser.parse_args()
        if args.use_heuristic:
            torch._inductor.config.autoheuristic_use = self.name
            torch._inductor.config.autoheuristic_collect = ""
        else:
            torch._inductor.config.autoheuristic_use = ""
            torch._inductor.config.autoheuristic_collect = self.name
        torch._inductor.config.autoheuristic_log_path = args.o
        if args.device is not None:
            torch.cuda.set_device(args.device)
        random.seed(time.time())
        self.main(args.num_samples, args.num_reps)

    @abstractmethod
    def run_benchmark(self, *args: Any) -> None: ...

    @abstractmethod
    def create_input(self) -> tuple[Any, ...]: ...

    def main(self, num_samples: int, num_reps: int) -> None:
        for _ in tqdm(range(num_samples)):
            input = self.create_input()
            for _ in range(num_reps):
                self.run_benchmark(*input)

```



## High-Level Overview

"""    BenchmarkRunner is a base class for all benchmark runners. It provides an interface to run benchmarks in order to    collect data with AutoHeuristic.

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BenchmarkRunner`

**Functions defined**: `__init__`, `add_base_arguments`, `run`, `run_benchmark`, `create_input`, `main`

**Key imports**: argparse, random, time, abstractmethod, Any, tqdm  , torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/_autoheuristic`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `random`
- `time`
- `abc`: abstractmethod
- `typing`: Any
- `tqdm`: tqdm  
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`torchgen/_autoheuristic`):

- [`train_regression.py_docs.md`](./train_regression.py_docs.md)
- [`merge_data.py_docs.md`](./merge_data.py_docs.md)
- [`generate_heuristic.sh_docs.md`](./generate_heuristic.sh_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`ah_tree.py_docs.md`](./ah_tree.py_docs.md)
- [`benchmark_utils.py_docs.md`](./benchmark_utils.py_docs.md)
- [`train_decision.py_docs.md`](./train_decision.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`test.sh_docs.md`](./test.sh_docs.md)


## Cross-References

- **File Documentation**: `benchmark_runner.py_docs.md`
- **Keyword Index**: `benchmark_runner.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen/_autoheuristic`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen/_autoheuristic`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`docs/torchgen/_autoheuristic`):

- [`ah_tree.py_kw.md_docs.md`](./ah_tree.py_kw.md_docs.md)
- [`test.sh_kw.md_docs.md`](./test.sh_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`train_regression.py_docs.md_docs.md`](./train_regression.py_docs.md_docs.md)
- [`collect_data.sh_docs.md_docs.md`](./collect_data.sh_docs.md_docs.md)
- [`benchmark_utils.py_docs.md_docs.md`](./benchmark_utils.py_docs.md_docs.md)
- [`benchmark_runner.py_kw.md_docs.md`](./benchmark_runner.py_kw.md_docs.md)
- [`requirements.txt_docs.md_docs.md`](./requirements.txt_docs.md_docs.md)
- [`test_utils.py_kw.md_docs.md`](./test_utils.py_kw.md_docs.md)
- [`requirements.txt_kw.md_docs.md`](./requirements.txt_kw.md_docs.md)


## Cross-References

- **File Documentation**: `benchmark_runner.py_docs.md_docs.md`
- **Keyword Index**: `benchmark_runner.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
