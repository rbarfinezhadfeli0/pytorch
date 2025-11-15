# Documentation: `docs/test/distributed/checkpoint/test_fsdp_model_state.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/test_fsdp_model_state.py_docs.md`
- **Size**: 6,949 bytes (6.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/test_fsdp_model_state.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_fsdp_model_state.py`
- **Size**: 3,459 bytes (3.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class FsdpModelStateCheckpoint(DTensorTestBase):
    @property
    def backend(self):
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr_backend}"

    def _test_fsdp_model_state(self, process_group) -> None:
        CHECKPOINT_DIR = self.temp_dir

        model = FSDP(torch.nn.Linear(8, 8, device="meta"))
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
            }

            dist_cp.save(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
                planner=DefaultSavePlanner(),
            )

        model_2 = FSDP(
            torch.nn.Linear(8, 8, device="meta"), process_group=process_group
        )

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertNotEqual(model.weight, model_2.weight)
                self.assertNotEqual(model.bias, model_2.bias)

        # now load the model and ensure the values are the same
        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model_2.state_dict(),
            }

            dist_cp.load(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                planner=DefaultLoadPlanner(),
            )
            model_2.load_state_dict(state_dict["model"])

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertEqual(model.weight, model_2.weight)
                self.assertEqual(model.bias, model_2.bias)

    @skip_if_lt_x_gpu(2)
    @with_comms
    @with_temp_dir
    def test_fsdp_model_state_no_resharding(self):
        self._test_fsdp_model_state(process_group=None)

    def _create_new_dist_group(self):
        world_size = dist.get_world_size()
        group1 = [i for i in range(world_size) if i % 2 == 0]
        group2 = [i for i in range(world_size) if i % 2 != 0]

        # create new fsdp group for resharding
        fsdp_0 = dist.new_group(ranks=group1)
        fsdp_1 = dist.new_group(ranks=group2)
        if dist.get_rank() % 2 == 0:
            my_fsdp = fsdp_0
        else:
            my_fsdp = fsdp_1

        return my_fsdp

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_fsdp_model_state_with_resharding(self):
        self._test_fsdp_model_state(process_group=self._create_new_dist_group())


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FsdpModelStateCheckpoint`

**Functions defined**: `backend`, `_test_fsdp_model_state`, `test_fsdp_model_state_no_resharding`, `_create_new_dist_group`, `test_fsdp_model_state_with_resharding`

**Key imports**: torch, torch.distributed as dist, torch.distributed.checkpoint as dist_cp, FullyShardedDataParallel as FSDP, StateDictType, skip_if_lt_x_gpu, run_tests, with_temp_dir


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed as dist`
- `torch.distributed.checkpoint as dist_cp`
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.fsdp.fully_sharded_data_parallel`: StateDictType
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_utils`: run_tests
- `torch.testing._internal.distributed.checkpoint_utils`: with_temp_dir


## Code Patterns & Idioms

### Common Patterns

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
python test/distributed/checkpoint/test_fsdp_model_state.py
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

- **File Documentation**: `test_fsdp_model_state.py_docs.md`
- **Keyword Index**: `test_fsdp_model_state.py_kw.md`
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
python docs/test/distributed/checkpoint/test_fsdp_model_state.py_docs.md
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

- **File Documentation**: `test_fsdp_model_state.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_model_state.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
