# Documentation: `test/distributed/checkpoint/test_hsdp_checkpoint.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_hsdp_checkpoint.py`
- **Size**: 7,814 bytes (7.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.tensor import Replicate
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(8, 4)
        self.net3 = nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x

    def get_input(self):
        return torch.rand(4, 5, device=device_type)


class SimpleModelUneven(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x

    def get_input(self):
        return torch.rand(4, 5, device=device_type)


class TestHSDPCheckpoint(DTensorTestBase):
    @property
    def backend(self):
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr_backend}"

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    @parametrize("is_even_sharded_model", [True, False])
    def test_hsdp_checkpoint(self, is_even_sharded_model) -> None:
        CHECKPOINT_DIR = self.temp_dir
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        model = FSDP(
            simple_model().to(self.device_type),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh_2d,
        )
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        state_dict = {"model": model.state_dict()}
        state_dict_to_save = deepcopy(state_dict)

        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # Update the parameters so current model state_dict now be different from state_dict_to_save.
        model(model.get_input()).sum().backward()
        optim.step()

        # At this point, the current state dict is different from state_dict_to_save.
        for (k1, v1), (k2, v2) in zip(
            state_dict_to_save["model"].items(), model.state_dict().items()
        ):
            self.assertEqual(k1, k2)
            self.assertEqual(v1.device_mesh, v2.device_mesh)
            self.assertEqual(v1.placements, v2.placements)
            self.assertNotEqual(v1.to_local(), v2.to_local())

        dist_cp.load(
            state_dict=state_dict_to_save,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )
        model.load_state_dict(state_dict_to_save["model"])

        # After loading, the current model state dict should be the same as state_dict_to_save.
        for (k1, v1), (k2, v2) in zip(
            state_dict_to_save["model"].items(), model.state_dict().items()
        ):
            self.assertEqual(k1, k2)
            self.assertEqual(v1.device_mesh, v2.device_mesh)
            self.assertEqual(v1.placements, v2.placements)
            self.assertEqual(v1.to_local(), v2.to_local())

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    @parametrize("is_even_sharded_model", [True, False])
    def test_hsdp_fsdp_checkpoint_conversion(self, is_even_sharded_model) -> None:
        CHECKPOINT_DIR = self.temp_dir
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # save the hsdp model state_dict
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        hsdp_model = FSDP(
            simple_model().to(self.device_type),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh_2d,
        )
        FSDP.set_state_dict_type(
            hsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        hsdp_state_dict = {"model": hsdp_model.state_dict()}
        dist_cp.save_state_dict(
            state_dict=hsdp_state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # initialize a fsdp model to load checkpoint into
        mesh_1d = init_device_mesh(self.device_type, (self.world_size,))
        fsdp_model = FSDP(
            simple_model().to(self.device_type),
            device_mesh=mesh_1d,
        )
        FSDP.set_state_dict_type(
            fsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        fsdp_state_dict = {"model": fsdp_model.state_dict()}

        # at this point, the hsdp model parameters are different from fsdp model parameters.
        for (k1, v1), (k2, v2) in zip(
            hsdp_state_dict["model"].items(), fsdp_state_dict["model"].items()
        ):
            self.assertEqual(k1, k2)
            self.assertNotEqual(v1.device_mesh, v2.device_mesh)
            self.assertNotEqual(v1.placements, v2.placements)
            v1_all_gather = v1.redistribute(
                mesh_2d, placements=(Replicate(), Replicate())
            )
            v2_all_gather = v2.redistribute(mesh_1d, placements=(Replicate(),))
            self.assertNotEqual(v1_all_gather.to_local(), v2_all_gather.to_local())

        # load the fsdp state_dict from storage
        dist_cp.load_state_dict(
            state_dict=fsdp_state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )
        fsdp_model.load_state_dict(fsdp_state_dict["model"])

        state_dict_after_load = fsdp_model.state_dict()
        # After loading, the current model state dict should be the same as hsdp_state_dict.
        for (k1, v1), (k2, v2) in zip(
            hsdp_state_dict["model"].items(), state_dict_after_load.items()
        ):
            self.assertEqual(k1, k2)
            self.assertNotEqual(v1.device_mesh, v2.device_mesh)
            self.assertNotEqual(v1.placements, v2.placements)
            v1_all_gather = v1.redistribute(
                mesh_2d, placements=(Replicate(), Replicate())
            )
            v2_all_gather = v2.redistribute(mesh_1d, placements=(Replicate(),))
            self.assertEqual(v1_all_gather.to_local(), v2_all_gather.to_local())


instantiate_parametrized_tests(TestHSDPCheckpoint)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SimpleModel`, `SimpleModelUneven`, `TestHSDPCheckpoint`

**Functions defined**: `__init__`, `forward`, `get_input`, `__init__`, `forward`, `get_input`, `backend`, `test_hsdp_checkpoint`, `test_hsdp_fsdp_checkpoint_conversion`

**Key imports**: deepcopy, torch, torch.distributed as dist, torch.distributed.checkpoint as dist_cp, torch.nn as nn, torch.nn.functional as F, init_device_mesh, FullyShardedDataParallel as FSDP, Replicate, skip_if_lt_x_gpu


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`: deepcopy
- `torch`
- `torch.distributed as dist`
- `torch.distributed.checkpoint as dist_cp`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.tensor`: Replicate
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
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
python test/distributed/checkpoint/test_hsdp_checkpoint.py
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

- **File Documentation**: `test_hsdp_checkpoint.py_docs.md`
- **Keyword Index**: `test_hsdp_checkpoint.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
