# Documentation: `docs/tools/experimental/torchfuzz/runner.py_docs.md`

## File Metadata

- **Path**: `docs/tools/experimental/torchfuzz/runner.py_docs.md`
- **Size**: 6,952 bytes (6.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/experimental/torchfuzz/runner.py`

## File Metadata

- **Path**: `tools/experimental/torchfuzz/runner.py`
- **Size**: 4,250 bytes (4.15 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
"""
Program runner utilities for PyTorch fuzzer.
This module handles running and testing generated PyTorch programs.
"""

import os
import random
import subprocess
import sys


class ProgramRunner:
    """Runs generated PyTorch programs and handles output/error reporting."""

    def __init__(self):
        pass

    def run_program(self, program_path):
        """
        Run a generated Python program and handle output/errors.

        Args:
            program_path: Path to the Python program to run

        Returns:
            bool: True if program ran successfully, False otherwise
        """
        abs_path = os.path.abspath(program_path)
        print(f"Running: {abs_path}")

        # Select a random CUDA device if available
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            devices = [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]
        else:
            # Default to all GPUs if not set
            try:
                import torch

                num_gpus = torch.cuda.device_count()
                devices = [str(i) for i in range(num_gpus)]
            except ImportError:
                devices = []
        if devices:
            selected_device = random.choice(devices)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = selected_device
            print(f"Selected CUDA_VISIBLE_DEVICES={selected_device}")
        else:
            env = None  # No GPU available or torch not installed

        try:
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            print("=== Program Output ===")
            print(result.stdout)
            print(result.stderr)
            return True

        except subprocess.CalledProcessError as e:
            print("=== Program Output (Failure) ===")
            print(e.stdout)
            print(e.stderr)
            print("===============================")
            print("=== Program Source ===")
            with open(abs_path) as f:
                print(f.read())
            print("======================")
            print(f"Program exited with code: {e.returncode}")
            sys.exit(1)

    def run_and_validate(self, program_path):
        """
        Run a program and return detailed results for validation.

        Args:
            program_path: Path to the Python program to run

        Returns:
            dict: Dictionary with 'success', 'stdout', 'stderr', 'returncode'
        """
        abs_path = os.path.abspath(program_path)

        # Select a random CUDA device if available
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            devices = [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]
        else:
            try:
                import torch

                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    devices = [str(i) for i in range(1, num_gpus)]
                else:
                    devices = [str(i) for i in range(num_gpus)]
            except ImportError:
                devices = []
        if devices:
            selected_device = random.choice(devices)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = selected_device
            print(f"Selected CUDA_VISIBLE_DEVICES={selected_device}")
        else:
            env = None

        try:
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode,
            }

```



## High-Level Overview

"""Program runner utilities for PyTorch fuzzer.This module handles running and testing generated PyTorch programs.

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ProgramRunner`

**Functions defined**: `__init__`, `run_program`, `run_and_validate`

**Key imports**: os, random, subprocess, sys, torch, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/experimental/torchfuzz`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `random`
- `subprocess`
- `sys`
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`tools/experimental/torchfuzz`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`codegen.py_docs.md`](./codegen.py_docs.md)
- [`tensor_fuzzer.py_docs.md`](./tensor_fuzzer.py_docs.md)
- [`fuzzer.py_docs.md`](./fuzzer.py_docs.md)
- [`visualize_graph.py_docs.md`](./visualize_graph.py_docs.md)
- [`checks.py_docs.md`](./checks.py_docs.md)
- [`test_determinism.py_docs.md`](./test_determinism.py_docs.md)
- [`type_promotion.py_docs.md`](./type_promotion.py_docs.md)
- [`ops_fuzzer.py_docs.md`](./ops_fuzzer.py_docs.md)
- [`multi_process_fuzzer.py_docs.md`](./multi_process_fuzzer.py_docs.md)


## Cross-References

- **File Documentation**: `runner.py_docs.md`
- **Keyword Index**: `runner.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/experimental/torchfuzz`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/experimental/torchfuzz`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/tools/experimental/torchfuzz`):

- [`ops_fuzzer.py_docs.md_docs.md`](./ops_fuzzer.py_docs.md_docs.md)
- [`multi_process_fuzzer.py_docs.md_docs.md`](./multi_process_fuzzer.py_docs.md_docs.md)
- [`multi_process_fuzzer.py_kw.md_docs.md`](./multi_process_fuzzer.py_kw.md_docs.md)
- [`checks.py_kw.md_docs.md`](./checks.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`checks.py_docs.md_docs.md`](./checks.py_docs.md_docs.md)
- [`type_promotion.py_docs.md_docs.md`](./type_promotion.py_docs.md_docs.md)
- [`fuzzer.py_kw.md_docs.md`](./fuzzer.py_kw.md_docs.md)
- [`test_determinism.py_kw.md_docs.md`](./test_determinism.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `runner.py_docs.md_docs.md`
- **Keyword Index**: `runner.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
