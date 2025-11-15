# Documentation: `test/distributed/checkpoint/test_format_utils.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_format_utils.py`
- **Size**: 3,386 bytes (3.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    dcp_to_torch_save,
    DynamicMetaLoadPlanner,
    torch_save_to_dcp,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class SimpleModelUneven(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    def get_input(self):
        return torch.rand(4, 5, device=device_type)


class TestFormatUtils(DTensorTestBase):
    @with_temp_dir
    def test_dcp_to_torch_save(self) -> None:
        model = SimpleModelUneven()
        dcp.save({"model": model}, checkpoint_id=self.temp_dir)

        torch_path = self.temp_dir + "/model.pt"
        dcp_to_torch_save(self.temp_dir, torch_path)

        loaded_sd = torch.load(torch_path)
        self.assertEqual(loaded_sd, {"model": model.state_dict()})

    @with_temp_dir
    def test_torch_save_to_dcp(self) -> None:
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}
        torch_path = self.temp_dir + "/model.pt"
        torch.save(sd, torch_path)

        torch_save_to_dcp(torch_path, self.temp_dir)

        model = SimpleModelUneven()
        dcp.load({"model": model}, checkpoint_id=self.temp_dir)

        self.assertEqual({"model": model.state_dict()}, sd)

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_online_torch_save_to_dcp(self) -> None:
        """Tests loading a model saved by torch.save directly into a sharded model
        using dcp.load
        """
        # Save a model with torch.save
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}

        torch_fn = self.temp_dir + "/model.pt"
        if dist.get_rank() == 0:
            torch.save(sd, torch_fn)
        dist.barrier()

        # Load into a sharded model
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = SimpleModelUneven().to(self.device_type)
        model = FSDP(
            model,
            device_mesh=device_mesh,
            use_orig_params=True,
        )
        dcp.load(
            {"model": model},
            planner=DynamicMetaLoadPlanner(),
            storage_reader=BroadcastingTorchSaveReader(),
            checkpoint_id=torch_fn,
        )

        self.assertEqual(sd["model"], model.state_dict())


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SimpleModelUneven`, `TestFormatUtils`

**Functions defined**: `__init__`, `forward`, `get_input`, `test_dcp_to_torch_save`, `test_torch_save_to_dcp`, `test_online_torch_save_to_dcp`

**Key imports**: torch, torch.distributed as dist, torch.distributed.checkpoint as dcp, torch.nn as nn, torch.nn.functional as F, init_device_mesh, FullyShardedDataParallel as FSDP, skip_if_lt_x_gpu, run_tests, with_temp_dir


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed as dist`
- `torch.distributed.checkpoint as dcp`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
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
python test/distributed/checkpoint/test_format_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/checkpoint`):

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

- **File Documentation**: `test_format_utils.py_docs.md`
- **Keyword Index**: `test_format_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
