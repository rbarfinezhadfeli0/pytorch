# Documentation: `docs/test/distributed/fsdp/test_fsdp_memory.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_fsdp_memory.py_docs.md`
- **Size**: 12,805 bytes (12.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_fsdp_memory.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_memory.py`
- **Size**: 8,982 bytes (8.77 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys
import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TEST_HPU,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.utils.checkpoint import checkpoint


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def get_cur_mem(rank, result, prefix):
    """Collect memory allocated values in a result dict in MB"""
    if TEST_CUDA:
        torch._C._cuda_clearCublasWorkspaces()
    result[prefix] = round(torch.accelerator.memory_allocated() / 1024 / 1024)


class Model(nn.Module):
    def __init__(self, hidden_dim, with_fsdp=False, with_checkpoint=False):
        super().__init__()
        if with_fsdp:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3),
                FSDP(nn.BatchNorm2d(64)),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        if with_fsdp:
            self.blocks = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),
                FSDP(nn.BatchNorm2d(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                FSDP(nn.BatchNorm2d(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                FSDP(nn.BatchNorm2d(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
            )
        else:
            self.blocks = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
            )

        self.head = nn.Linear(hidden_dim, 10)
        self.with_checkpoint = with_checkpoint

    def forward(self, x):
        if self.with_checkpoint:
            return self.head(checkpoint(self.blocks, self.stem(x), use_reentrant=True))
        else:
            return self.head(self.blocks(self.stem(x)))


def create_model(with_fsdp, with_checkpoint, model_hidden_dim):
    torch.manual_seed(0)
    model = Model(model_hidden_dim, with_fsdp, with_checkpoint)
    if with_fsdp:
        model.stem = FSDP(model.stem)
        model.blocks = FSDP(model.blocks)
        model.head = FSDP(model.head)

    return model


class TestFSDPMemory(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _dist_train(self, with_checkpoint, expected, model_hidden_dim, iterations):
        gpu_id = self.rank
        batch = torch.randn(size=(2, 3, 224, 224)).to(device_type)

        model = create_model(
            with_fsdp=True,
            with_checkpoint=with_checkpoint,
            model_hidden_dim=model_hidden_dim,
        )
        model = model.to(device_type)
        model = FSDP(model)

        # We enable momentum so that after the first iteration, the optimizer state is added
        # to the total memory used.
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

        results = {}  # results of memory stats
        for iteration in range(iterations):
            get_cur_mem(gpu_id, results, f"iter {iteration}: start")

            out = model(batch)
            get_cur_mem(gpu_id, results, f"iter {iteration}: after fwd")

            out = sum(o.sum() for o in out[0])
            fake_loss = criterion(out, torch.tensor(0.0).to(device_type))
            get_cur_mem(gpu_id, results, f"iter {iteration}: after loss")

            fake_loss.backward()
            get_cur_mem(gpu_id, results, f"iter {iteration}: after bwd")

            optimizer.step()
            get_cur_mem(gpu_id, results, f"iter {iteration}: after step")

            # It is important to use `set_to_none` below, not optimizer.zero_grad() to reclaim memory.
            model.zero_grad(set_to_none=True)
            get_cur_mem(gpu_id, results, f"iter {iteration}: done")

        def cmp(results, expected):
            ret = ""
            self.assertEqual(results.keys(), expected.keys())
            for k, v in results.items():
                exp = expected[k]
                if abs(exp - v) > 1:  # allow 1MB rounding differences
                    ret += f"{k}: got {v}, expected {exp}\n"
            return ret

        output = cmp(results, expected)
        self.assertEqual(output, "")

    @unittest.skipIf(TEST_HPU, "Memory will be different for CUDA and HPU, skipping")
    @skip_if_lt_x_gpu(2)
    @parametrize("ckpt", ["no_ckpt", "ckpt"])
    def test_fsdp_memory(self, ckpt):
        # hidden_dim 128: model size ~4MB
        model_hidden_dim = 128

        model = create_model(
            with_fsdp=False, with_checkpoint=False, model_hidden_dim=model_hidden_dim
        ).to(device_type)
        model_size_mb = round(torch.accelerator.memory_allocated() / 1024 / 1024)
        del model

        sharded_model_size_mb = int(model_size_mb / self.world_size)

        # We have observed that sometimes after 3rd iteration, 4th one can fail (not on this
        # test but on much bigger scale tests). We run 4 iterations here just in case it happens.
        iterations = 4

        expected = {}

        for iteration in range(iterations):
            if iteration == 0:
                # sharded model size + 1MB temp memory
                expected[f"iter {iteration}: start"] = sharded_model_size_mb + 1
                # it is hard to calculate this memory size, get it from printed memory usage
                if ckpt == "ckpt":
                    expected[f"iter {iteration}: after fwd"] = 51
                    expected[f"iter {iteration}: after loss"] = 51
                else:
                    expected[f"iter {iteration}: after fwd"] = 340
                    expected[f"iter {iteration}: after loss"] = 340
                # sharded model size + sharded grad size + 1M temp memory
                expected[f"iter {iteration}: after bwd"] = 2 * sharded_model_size_mb + 1
            else:
                # after optimizer step in the first iteration, memory usage increased by
                # sharded_model_size_mb because of increased optimizer states memory usage
                expected[f"iter {iteration}: start"] = 2 * sharded_model_size_mb + 1
                if ckpt == "ckpt":
                    expected[f"iter {iteration}: after fwd"] = (
                        51 + sharded_model_size_mb
                    )
                    expected[f"iter {iteration}: after loss"] = (
                        51 + sharded_model_size_mb
                    )
                else:
                    expected[f"iter {iteration}: after fwd"] = (
                        340 + sharded_model_size_mb
                    )
                    expected[f"iter {iteration}: after loss"] = (
                        340 + sharded_model_size_mb
                    )
                expected[f"iter {iteration}: after bwd"] = 3 * sharded_model_size_mb + 1

            # sharded model size + sharded grad size + optimizer states + 1M temp memory
            expected[f"iter {iteration}: after step"] = 3 * sharded_model_size_mb + 1
            # grad memory is claimed after setting grad = None
            # sharded model size + optimizer states + 1M temp memory
            expected[f"iter {iteration}: done"] = 2 * sharded_model_size_mb + 1

        # Get the fsdp and checkpoint flags.
        with_ckpt = ckpt == "ckpt"

        self._dist_train(
            with_ckpt,
            expected,
            model_hidden_dim,
            iterations,
        )


instantiate_parametrized_tests(TestFSDPMemory)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Collect memory allocated values in a result dict in MB"""    if TEST_CUDA:        torch._C._cuda_clearCublasWorkspaces()    result[prefix] = round(torch.accelerator.memory_allocated() / 1024 / 1024)class Model(nn.Module):    def __init__(self, hidden_dim, with_fsdp=False, with_checkpoint=False):        super().__init__()        if with_fsdp:            self.stem = nn.Sequential(                nn.Conv2d(3, 64, kernel_size=3),

This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Model`, `TestFSDPMemory`

**Functions defined**: `get_cur_mem`, `__init__`, `forward`, `create_model`, `world_size`, `_dist_train`, `cmp`, `test_fsdp_memory`

**Key imports**: sys, unittest, torch, torch.nn as nn, torch.optim as optim, distributed as dist, FullyShardedDataParallel as FSDP, skip_if_lt_x_gpu, FSDPTest, checkpoint


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `unittest`
- `torch`
- `torch.nn as nn`
- `torch.optim as optim`
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_fsdp`: FSDPTest
- `torch.utils.checkpoint`: checkpoint


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/fsdp/test_fsdp_memory.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/fsdp`):

- [`test_fsdp_mixed_precision.py_docs.md`](./test_fsdp_mixed_precision.py_docs.md)
- [`test_fsdp_uneven.py_docs.md`](./test_fsdp_uneven.py_docs.md)
- [`test_fsdp_dtensor_state_dict.py_docs.md`](./test_fsdp_dtensor_state_dict.py_docs.md)
- [`test_fsdp_tp_integration.py_docs.md`](./test_fsdp_tp_integration.py_docs.md)
- [`test_distributed_checkpoint.py_docs.md`](./test_distributed_checkpoint.py_docs.md)
- [`test_fsdp_multiple_forward.py_docs.md`](./test_fsdp_multiple_forward.py_docs.md)
- [`test_checkpoint_wrapper.py_docs.md`](./test_checkpoint_wrapper.py_docs.md)
- [`test_fsdp_clip_grad_norm.py_docs.md`](./test_fsdp_clip_grad_norm.py_docs.md)
- [`test_fsdp_use_orig_params.py_docs.md`](./test_fsdp_use_orig_params.py_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_memory.py_docs.md`
- **Keyword Index**: `test_fsdp_memory.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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
python docs/test/distributed/fsdp/test_fsdp_memory.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/fsdp`):

- [`test_fsdp_grad_acc.py_docs.md_docs.md`](./test_fsdp_grad_acc.py_docs.md_docs.md)
- [`test_fsdp_ignored_modules.py_kw.md_docs.md`](./test_fsdp_ignored_modules.py_kw.md_docs.md)
- [`test_fsdp_meta.py_kw.md_docs.md`](./test_fsdp_meta.py_kw.md_docs.md)
- [`test_fsdp_apply.py_docs.md_docs.md`](./test_fsdp_apply.py_docs.md_docs.md)
- [`test_fsdp_tp_integration.py_kw.md_docs.md`](./test_fsdp_tp_integration.py_kw.md_docs.md)
- [`test_fsdp_fx.py_docs.md_docs.md`](./test_fsdp_fx.py_docs.md_docs.md)
- [`test_fsdp_memory.py_kw.md_docs.md`](./test_fsdp_memory.py_kw.md_docs.md)
- [`test_fsdp_apply.py_kw.md_docs.md`](./test_fsdp_apply.py_kw.md_docs.md)
- [`test_fsdp_tp_integration.py_docs.md_docs.md`](./test_fsdp_tp_integration.py_docs.md_docs.md)
- [`test_fsdp_multiple_forward.py_kw.md_docs.md`](./test_fsdp_multiple_forward.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_memory.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_memory.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
