# Documentation: `docs/test/distributed/checkpoint/e2e/test_fine_tuning.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/e2e/test_fine_tuning.py_docs.md`
- **Size**: 10,211 bytes (9.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/e2e/test_fine_tuning.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/e2e/test_fine_tuning.py`
- **Size**: 7,139 bytes (6.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


DIM = 500


class PreTrainedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.sequential = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
        self.module_list = nn.ModuleList([nn.Linear(DIM, DIM), nn.ReLU()])
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = self.relu(self.layer1(batch))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sequential(x)
        x = self.module_list[1](self.module_list[0](x))
        return x


class FineTuningModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain = PreTrainedModel()
        for p in self.pretrain.parameters():
            p.requires_grad = False

        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = self.relu(self.pretrain(batch))
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x


class TestFineTuning(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return min(4, torch.accelerator.device_count())

    @property
    def backend(self):
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr_backend}"

    def pretrain(self, pretrain_dir: str) -> None:
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        model = PreTrainedModel().to(self.device_type)
        model = FSDP(model, device_mesh=device_mesh)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training
        for _ in range(3):
            batch = torch.rand(32, DIM, device=self.device_type)
            loss = model(batch).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

        # Save state_dict
        model_state_dict, optim_state_dict = get_state_dict(model, optimizers=optim)
        saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
        dist_cp.save(
            state_dict=saved_state_dict,
            storage_writer=dist_cp.FileSystemWriter(pretrain_dir),
        )

    def finetune(self, pretrain_dir: str, finetune_dir: str) -> None:
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        model = FineTuningModel().to(self.device_type)
        # TODO: make the parallelism more complicated, e.g., using 2D + DDP.
        model = FSDP(model, use_orig_params=True, device_mesh=device_mesh)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simulate that the fine tuning restart after 3 iterations
        for i in range(2):
            # Load pretrain submodules checkpoint
            pretrain_state_dict = get_model_state_dict(
                model,
                submodules={model.pretrain},
                options=StateDictOptions(keep_submodule_prefixes=False),
            )
            dist_cp.load(
                {"model": pretrain_state_dict},
                storage_reader=dist_cp.FileSystemReader(pretrain_dir),
            )
            set_model_state_dict(
                model,
                model_state_dict={model.pretrain: pretrain_state_dict},
                options=StateDictOptions(strict=False),
            )

            try:
                # Load training submodules checkpoint
                model_state_dict, optim_state_dict = get_state_dict(
                    model,
                    optimizers=optim,
                    options=StateDictOptions(ignore_frozen_params=True),
                )
                dist_cp.load_state_dict(
                    {"model": model_state_dict, "optim": optim_state_dict},
                    storage_reader=dist_cp.FileSystemReader(pretrain_dir),
                )
                set_state_dict(
                    model,
                    optimizers=optim,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                    options=StateDictOptions(strict=False),
                )
            except KeyError:
                # If this is the first round of the fine tuning, then nothing is saved.
                # If this is the restart of the fine tuning, then checkpoint should exit.
                self.assertEqual(i, 0)

            # Training
            for _ in range(3):
                batch = torch.rand(32, DIM, device=self.device_type)
                loss = model(batch).sum()
                loss.backward()
                optim.step()
                optim.zero_grad()

            # Save state_dict
            model_state_dict, optim_state_dict = get_state_dict(
                model,
                optimizers=optim,
                options=StateDictOptions(ignore_frozen_params=True),
            )
            saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
            dist_cp.save(
                state_dict=saved_state_dict,
                storage_writer=dist_cp.FileSystemWriter(finetune_dir),
            )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_fine_tuning(self) -> None:
        self.assertTrue(os.path.exists(self.temp_dir))
        pretrain_dir = os.path.join(self.temp_dir, "pretrain")
        finetune_dir = os.path.join(self.temp_dir, "finetune")
        print(pretrain_dir, finetune_dir)
        if self.rank == 0:
            os.mkdir(pretrain_dir)
            os.mkdir(finetune_dir)
        dist.barrier()
        os.sync()
        self.assertTrue(os.path.exists(pretrain_dir))
        self.assertTrue(os.path.exists(finetune_dir))

        self.pretrain(pretrain_dir)
        self.finetune(pretrain_dir, finetune_dir)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PreTrainedModel`, `FineTuningModel`, `TestFineTuning`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `world_size`, `backend`, `pretrain`, `finetune`, `test_fine_tuning`

**Key imports**: os, sys, torch, torch.distributed as dist, torch.distributed.checkpoint as dist_cp, torch.nn as nn, init_device_mesh, FullyShardedDataParallel as FSDP, skip_if_lt_x_gpu, run_tests, TEST_WITH_DEV_DBG_ASAN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint/e2e`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `torch`
- `torch.distributed as dist`
- `torch.distributed.checkpoint as dist_cp`
- `torch.nn as nn`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN
- `torch.testing._internal.distributed.checkpoint_utils`: with_temp_dir


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python test/distributed/checkpoint/e2e/test_fine_tuning.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/checkpoint/e2e`):

- [`test_e2e_save_and_load.py_docs.md`](./test_e2e_save_and_load.py_docs.md)
- [`test_fsdp_ep.py_docs.md`](./test_fsdp_ep.py_docs.md)


## Cross-References

- **File Documentation**: `test_fine_tuning.py_docs.md`
- **Keyword Index**: `test_fine_tuning.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/checkpoint/e2e`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/checkpoint/e2e`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python docs/test/distributed/checkpoint/e2e/test_fine_tuning.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/checkpoint/e2e`):

- [`test_fine_tuning.py_kw.md_docs.md`](./test_fine_tuning.py_kw.md_docs.md)
- [`test_e2e_save_and_load.py_kw.md_docs.md`](./test_e2e_save_and_load.py_kw.md_docs.md)
- [`test_e2e_save_and_load.py_docs.md_docs.md`](./test_e2e_save_and_load.py_docs.md_docs.md)
- [`test_fsdp_ep.py_kw.md_docs.md`](./test_fsdp_ep.py_kw.md_docs.md)
- [`test_fsdp_ep.py_docs.md_docs.md`](./test_fsdp_ep.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_fine_tuning.py_docs.md_docs.md`
- **Keyword Index**: `test_fine_tuning.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
