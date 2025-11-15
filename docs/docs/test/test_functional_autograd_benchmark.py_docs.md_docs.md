# Documentation: `docs/test/test_functional_autograd_benchmark.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_functional_autograd_benchmark.py_docs.md`
- **Size**: 5,518 bytes (5.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_functional_autograd_benchmark.py`

## File Metadata

- **Path**: `test/test_functional_autograd_benchmark.py`
- **Size**: 2,659 bytes (2.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: autograd"]

import os

import subprocess
import unittest

from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    slowTest,
    TemporaryFileName,
    TestCase,
)

PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))


# This is a very simple smoke test for the functional autograd benchmarking script.
class TestFunctionalAutogradBenchmark(TestCase):
    def _test_runner(self, model, disable_gpu=False):
        # Note about windows:
        # The temporary file is exclusively open by this process and the child process
        # is not allowed to open it again. As this is a simple smoke test, we choose for now
        # not to run this on windows and keep the code here simple.
        with TemporaryFileName() as out_file:
            cmd = [
                "python3",
                "../benchmarks/functional_autograd_benchmark/functional_autograd_benchmark.py",
            ]
            if IS_WINDOWS:
                cmd[0] = "python"
            # Only run the warmup
            cmd += ["--num-iters", "0"]
            # Only run the vjp task (fastest one)
            cmd += ["--task-filter", "vjp"]
            # Only run the specified model
            cmd += ["--model-filter", model]
            # Output file
            cmd += ["--output", out_file]
            if disable_gpu:
                cmd += ["--gpu", "-1"]

            res = subprocess.run(cmd, check=False)

            self.assertTrue(res.returncode == 0)
            # Check that something was written to the file
            self.assertTrue(os.stat(out_file).st_size > 0)

    @unittest.skipIf(
        PYTORCH_COLLECT_COVERAGE,
        "Can deadlocks with gcov, see https://github.com/pytorch/pytorch/issues/49656",
    )
    def test_fast_tasks(self):
        fast_tasks = [
            "resnet18",
            "ppl_simple_reg",
            "ppl_robust_reg",
            "wav2letter",
            "transformer",
            "multiheadattn",
        ]

        for task in fast_tasks:
            self._test_runner(task)

    @slowTest
    @unittest.skipIf(
        IS_WINDOWS,
        "NamedTemporaryFile on windows does not have all the features we need.",
    )
    def test_slow_tasks(self):
        slow_tasks = ["fcn_resnet", "detr"]
        # deepspeech is voluntarily excluded as it takes too long to run without
        # proper tuning of the number of threads it should use.

        for task in slow_tasks:
            # Disable GPU for slow test as the CI GPU don't have enough memory
            self._test_runner(task, disable_gpu=True)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFunctionalAutogradBenchmark`

**Functions defined**: `_test_runner`, `test_fast_tasks`, `test_slow_tasks`

**Key imports**: os, subprocess, unittest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `subprocess`
- `unittest`


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

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_functional_autograd_benchmark.py
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

- **File Documentation**: `test_functional_autograd_benchmark.py_docs.md`
- **Keyword Index**: `test_functional_autograd_benchmark.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_functional_autograd_benchmark.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_functional_autograd_benchmark.py_docs.md_docs.md`
- **Keyword Index**: `test_functional_autograd_benchmark.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
