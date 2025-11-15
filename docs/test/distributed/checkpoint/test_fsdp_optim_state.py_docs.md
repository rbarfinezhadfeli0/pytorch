# Documentation: `test/distributed/checkpoint/test_fsdp_optim_state.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_fsdp_optim_state.py`
- **Size**: 5,062 bytes (4.94 KB)
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
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
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


class FsdpOptimStateCheckpoint(DTensorTestBase):
    def _create_model(self):
        # make weight tensor dim_0 as large as the world size for scaling test
        layer1_weight_dim = self.world_size
        layer2_weight_dim = self.world_size * 2
        layer3_weight_dim = self.world_size * 3

        class TestDummyModel(torch.nn.Module):
            def __init__(self, device_type) -> None:
                super().__init__()
                self.device_type = device_type
                self.net1 = nn.Sequential(nn.Linear(8, layer1_weight_dim), nn.ReLU())
                self.net2 = nn.Sequential(
                    nn.Linear(layer1_weight_dim, layer2_weight_dim), nn.ReLU()
                )
                self.net3 = nn.Sequential(
                    nn.Linear(layer2_weight_dim, layer3_weight_dim), nn.ReLU()
                )

            def forward(self, x):
                return self.net3(self.net2(self.net1(x)))

            def get_input(self):
                return torch.rand(8, 8, device=self.device_type)

        model = TestDummyModel(self.device_type).to(self.device_type)
        return model

    @property
    def backend(self):
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr_backend}"

    @skip_if_lt_x_gpu(2)
    @with_comms
    @with_temp_dir
    @parametrize("pass_planner", [True, False])
    def test_load_sharded_optimizer_state_dict(self, pass_planner) -> None:
        CHECKPOINT_DIR = self.temp_dir
        planner = dcp.DefaultLoadPlanner() if pass_planner else None

        model = self._create_model()
        model = FSDP(model)
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        # step ahead to initialize the optimizer
        model(model.get_input()).sum().backward()
        optim.step()

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        optim_osd = FSDP.optim_state_dict(model, optim)

        state_dict = {
            "model": model.state_dict(),
            "optim": optim_osd,
        }
        dcp.save(
            state_dict=state_dict,
            storage_writer=dcp.FileSystemWriter(CHECKPOINT_DIR),
        )

        # now load the model and ensure the values are the same
        model_2 = self._create_model()
        model_2 = FSDP(model_2)
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)

        FSDP.set_state_dict_type(
            model_2,
            StateDictType.SHARDED_STATE_DICT,
        )
        # Adam lazily creates its state
        self.assertEqual(0, len(optim_2.state))

        state_dict = {
            "model": model_2.state_dict(),
            # cannot load the optimizer together with the model
        }
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(CHECKPOINT_DIR),
        )
        model_2.load_state_dict(state_dict["model"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=dcp.FileSystemReader(CHECKPOINT_DIR),
            planner=planner,
        )
        flattened_osd = FSDP.optim_state_dict_to_load(
            model_2, optim_2, optim_state["optim"]
        )
        optim_2.load_state_dict(flattened_osd)
        osd_after_load = FSDP.optim_state_dict(model_2, optim_2)

        # Compare optim_state_dict prior to save and after load
        before_optim_state = optim_osd["state"]
        after_optim_state = osd_after_load["state"]
        self.assertEqual(len(before_optim_state), len(after_optim_state))
        for fqn, states in before_optim_state.items():
            for state_name, state in states.items():
                state2 = after_optim_state.get(fqn).get(state_name)
                if isinstance(state, ShardedTensor):
                    self.assertTrue(isinstance(state2, ShardedTensor))
                    self.assertTrue(torch.allclose(state, state2))
                else:
                    self.assertEqual(state, state2)


instantiate_parametrized_tests(FsdpOptimStateCheckpoint)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FsdpOptimStateCheckpoint`, `TestDummyModel`

**Functions defined**: `_create_model`, `__init__`, `forward`, `get_input`, `backend`, `test_load_sharded_optimizer_state_dict`

**Key imports**: torch, torch.distributed as dist, torch.distributed.checkpoint as dcp, torch.nn as nn, ShardedTensor, load_sharded_optimizer_state_dict, FullyShardedDataParallel as FSDP, StateDictType, skip_if_lt_x_gpu, with_temp_dir


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
- `torch.distributed._shard.sharded_tensor.api`: ShardedTensor
- `torch.distributed.checkpoint.optimizer`: load_sharded_optimizer_state_dict
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.fsdp.fully_sharded_data_parallel`: StateDictType
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
python test/distributed/checkpoint/test_fsdp_optim_state.py
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
- [`test_state_dict_stager.py_docs.md`](./test_state_dict_stager.py_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_optim_state.py_docs.md`
- **Keyword Index**: `test_fsdp_optim_state.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
