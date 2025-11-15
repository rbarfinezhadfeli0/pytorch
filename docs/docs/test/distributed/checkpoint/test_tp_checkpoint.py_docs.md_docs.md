# Documentation: `docs/test/distributed/checkpoint/test_tp_checkpoint.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/test_tp_checkpoint.py_docs.md`
- **Size**: 8,897 bytes (8.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/test_tp_checkpoint.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_tp_checkpoint.py`
- **Size**: 5,633 bytes (5.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class UnevenShardedModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(5, 10, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 15, device=device)
        self.net3 = torch.nn.Linear(15, 1, device=device)

    def forward(self, x):
        return self.net3(self.net2(self.relu(self.net1(x))))


class TestTpCheckpoint(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_tp_checkpoint(self):
        CHECKPOINT_DIR = self.temp_dir
        mesh_shpe = (self.world_size,)
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)

        # create model and move it to GPU with id rank
        model = MLPModule(self.device_type).to(self.rank)
        # Parallelize the module based on the given Parallel Style.
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, parallelize_plan)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        original_state_dict = deepcopy(model.state_dict())

        dcp.save(
            state_dict=original_state_dict,
            storage_writer=dcp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # Update the parameters so model.state_dict() will be different from original_state_dict.
        torch.manual_seed(0)
        inp = torch.rand(20, 10).to(self.rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
        state_dict = model.state_dict()

        # ensure the current model parameters are different from original_state_dict before loading from checkpoint
        for param1, param2 in zip(original_state_dict.values(), state_dict.values()):
            self.assertNotEqual(param1.to_local(), param2.to_local())

        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )

        # now load from checkpoint to check current model parameters are the same as original_state_dict
        for param1, param2 in zip(original_state_dict.values(), state_dict.values()):
            self.assertEqual(param1.to_local(), param2.to_local())

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_tp_checkpoint_load_on_meta_device(self):
        CHECKPOINT_DIR = self.temp_dir
        mesh_shpe = (self.world_size,)
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)

        # create model and move it to GPU with id rank
        model = UnevenShardedModel(self.device_type).to(self.rank)
        # Parallelize the module based on the given Parallel Style.
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
            "net3": ColwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, parallelize_plan=parallelize_plan)
        original_state_dict = {
            "model": model.state_dict(),
        }

        dcp.save(
            state_dict=original_state_dict,
            storage_writer=dcp.FileSystemWriter(CHECKPOINT_DIR),
        )

        model2 = parallelize_module(
            UnevenShardedModel("meta"), tp_mesh, parallelize_plan=parallelize_plan
        )
        model2_sd_before_load = model2.state_dict()
        state_dict_to_load = {"model": model2_sd_before_load}

        dcp.load(
            state_dict=state_dict_to_load,
            storage_reader=dcp.FileSystemReader(CHECKPOINT_DIR),
        )
        # We need to make sure state_dict_to_load["model"] is the same as state_dict_after_load["model"],
        # since we are doing in-place loading.
        self.assertTrue(state_dict_to_load["model"] is model2_sd_before_load)

        model2.load_state_dict(state_dict_to_load["model"], assign=True)
        state_dict_after_load = {"model": model2.state_dict()}

        self.assertEqual(
            len(original_state_dict["model"]), len(state_dict_to_load["model"])
        )
        self.assertEqual(
            len(original_state_dict["model"]), len(state_dict_after_load["model"])
        )

        for name, param in original_state_dict["model"].items():
            param_to_load = state_dict_to_load["model"][name]
            param_after_load = state_dict_after_load["model"][name]

            # we need to explicitly check the device is not meta as the assertEqual check
            # currently doesn't handle DTensor with meta device.
            self.assertTrue(not param_to_load.is_meta)
            self.assertTrue(not param_after_load.is_meta)
            self.assertEqual(param.to_local(), param_to_load.to_local())
            self.assertEqual(param.to_local(), param_after_load.to_local())


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `UnevenShardedModel`, `TestTpCheckpoint`

**Functions defined**: `__init__`, `forward`, `test_tp_checkpoint`, `test_tp_checkpoint_load_on_meta_device`

**Key imports**: deepcopy, torch, torch.distributed.checkpoint as dcp, init_device_mesh, run_tests, with_temp_dir


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`: deepcopy
- `torch`
- `torch.distributed.checkpoint as dcp`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.testing._internal.common_utils`: run_tests
- `torch.testing._internal.distributed.checkpoint_utils`: with_temp_dir


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
python test/distributed/checkpoint/test_tp_checkpoint.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/checkpoint`):

- [`test_format_utils.py_docs.md`](./test_format_utils.py_docs.md)
- [`test_save_load_api.py_docs.md`](./test_save_load_api.py_docs.md)
- [`test_pg_transport.py_docs.md`](./test_pg_transport.py_docs.md)
- [`test_async_process_executor.py_docs.md`](./test_async_process_executor.py_docs.md)
- [`test_file_system_checkpoint.py_docs.md`](./test_file_system_checkpoint.py_docs.md)
- [`test_nested_dict.py_docs.md`](./test_nested_dict.py_docs.md)
- [`test_hf_storage.py_docs.md`](./test_hf_storage.py_docs.md)
- [`test_hf_safetensor_e2e.py_docs.md`](./test_hf_safetensor_e2e.py_docs.md)
- [`test_fsdp_optim_state.py_docs.md`](./test_fsdp_optim_state.py_docs.md)
- [`test_state_dict_stager.py_docs.md`](./test_state_dict_stager.py_docs.md)


## Cross-References

- **File Documentation**: `test_tp_checkpoint.py_docs.md`
- **Keyword Index**: `test_tp_checkpoint.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/checkpoint`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/checkpoint`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/checkpoint/test_tp_checkpoint.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/checkpoint`):

- [`test_state_dict.py_docs.md_docs.md`](./test_state_dict.py_docs.md_docs.md)
- [`test_checkpoint.py_kw.md_docs.md`](./test_checkpoint.py_kw.md_docs.md)
- [`test_dtensor_checkpoint.py_docs.md_docs.md`](./test_dtensor_checkpoint.py_docs.md_docs.md)
- [`test_file_system_checkpoint_cpu.py_docs.md_docs.md`](./test_file_system_checkpoint_cpu.py_docs.md_docs.md)
- [`test_dedup_tensors.py_docs.md_docs.md`](./test_dedup_tensors.py_docs.md_docs.md)
- [`test_fsspec.py_kw.md_docs.md`](./test_fsspec.py_kw.md_docs.md)
- [`test_quantized_hf_storage.py_kw.md_docs.md`](./test_quantized_hf_storage.py_kw.md_docs.md)
- [`test_pg_transport.py_kw.md_docs.md`](./test_pg_transport.py_kw.md_docs.md)
- [`test_dedup_tensors.py_kw.md_docs.md`](./test_dedup_tensors.py_kw.md_docs.md)
- [`test_hsdp_checkpoint.py_docs.md_docs.md`](./test_hsdp_checkpoint.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_tp_checkpoint.py_docs.md_docs.md`
- **Keyword Index**: `test_tp_checkpoint.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
