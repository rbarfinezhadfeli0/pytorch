# Documentation: `test/distributed/_tools/test_sac_estimator.py`

## File Metadata

- **Path**: `test/distributed/_tools/test_sac_estimator.py`
- **Size**: 3,142 bytes (3.07 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.sac_estimator import SACEstimator
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


class TestSACEstimator(TestCase):
    def _sac_estimation(
        self,
        estimate_mode: str,
        model: torch.nn.Module,
        inp: torch.Tensor,
    ):
        sace = SACEstimator()
        with sace(estimate_mode_type=estimate_mode):
            loss = model(inp).sum()
        loss.backward()
        sace.pwlf_sac_tradeoff_curve(n_segments=2, save_tradeoff_graphs=False)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_transformer_sac_estimation(self):
        """Runs a basic GPT-2 model"""
        dev = torch.cuda.current_device()
        vocab_size = 8192
        bsz, seq_len = 8, 1024
        model_args = ModelArgs(
            n_layers=4,
            n_heads=12,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dim=768,
            dropout_p=0.1,
        )
        with FakeTensorMode():
            with torch.device(dev):
                model = Transformer(model_args)
            inp = torch.randint(
                0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev
            )

            self._sac_estimation("operator-level-benchmark", model, inp)
            self._sac_estimation("operator-level-cost-model", model, inp)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_simple_model_sac_estimation(self):
        """This test checks the correctness of view_ops, random_ops and inplace_ops"""

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(5, 10)
                self.relu1 = torch.nn.ReLU(inplace=True)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = torch.cos_(x)
                x = torch.sin_(x)
                return x

        dev = torch.cuda.current_device()
        with FakeTensorMode():
            with torch.device(dev):
                model = Foo()
            x = torch.rand((10, 5), device=dev)

            sac_estimator = SACEstimator()
            with sac_estimator(estimate_mode_type="operator-level-benchmark"):
                loss = model(x).sum()
            loss.backward()

            self.assertEqual(sac_estimator.sac_mod_stats["Foo"].view_like_ops, [0])
            self.assertEqual(sac_estimator.sac_mod_stats["Foo"].rand_ops, [])
            self.assertEqual(
                sac_estimator.sac_mod_stats["Foo"].inplace_ops, [(2, 1), (3, 1), (4, 1)]
            )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Runs a basic GPT-2 model"""        dev = torch.cuda.current_device()        vocab_size = 8192        bsz, seq_len = 8, 1024        model_args = ModelArgs(            n_layers=4,            n_heads=12,            vocab_size=vocab_size,            max_seq_len=seq_len,            dim=768,            dropout_p=0.1,        )        with FakeTensorMode():            with torch.device(dev):                model = Transformer(model_args)            inp = torch.randint(                0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev            )            self._sac_estimation("operator-level-benchmark", model, inp)

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSACEstimator`, `Foo`

**Functions defined**: `_sac_estimation`, `test_transformer_sac_estimation`, `test_simple_model_sac_estimation`, `__init__`, `forward`

**Key imports**: unittest, torch, FakeTensorMode, SACEstimator, TEST_CUDA, run_tests, skipIfTorchDynamo, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_tools`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch.distributed._tools.sac_estimator`: SACEstimator
- `torch.testing._internal.common_cuda`: TEST_CUDA
- `torch.testing._internal.common_utils`: run_tests, skipIfTorchDynamo, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python test/distributed/_tools/test_sac_estimator.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_tools`):

- [`test_fsdp2_mem_tracker.py_docs.md`](./test_fsdp2_mem_tracker.py_docs.md)
- [`test_runtime_estimator.py_docs.md`](./test_runtime_estimator.py_docs.md)
- [`test_mod_tracker.py_docs.md`](./test_mod_tracker.py_docs.md)
- [`test_memory_tracker.py_docs.md`](./test_memory_tracker.py_docs.md)
- [`test_sac_ilp.py_docs.md`](./test_sac_ilp.py_docs.md)
- [`test_mem_tracker.py_docs.md`](./test_mem_tracker.py_docs.md)
- [`test_fake_collectives.py_docs.md`](./test_fake_collectives.py_docs.md)


## Cross-References

- **File Documentation**: `test_sac_estimator.py_docs.md`
- **Keyword Index**: `test_sac_estimator.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
