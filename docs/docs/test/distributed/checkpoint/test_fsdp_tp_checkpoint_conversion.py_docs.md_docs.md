# Documentation: `docs/test/distributed/checkpoint/test_fsdp_tp_checkpoint_conversion.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/test_fsdp_tp_checkpoint_conversion.py_docs.md`
- **Size**: 7,597 bytes (7.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/test_fsdp_tp_checkpoint_conversion.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_fsdp_tp_checkpoint_conversion.py`
- **Size**: 4,016 bytes (3.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._state_dict_utils import _all_gather_sharded_tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor import DTensor, Replicate
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


# TODO: modularize this test and add test for checkpoint conversion in both direction.
class TestFsdpTpCheckpointConversion(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_fsdp_to_tp(self):
        CHECKPOINT_DIR = self.temp_dir

        model = MLPModule(self.device_type).to(self.rank)
        # create a FSDP wrapped model
        fsdp_model = FSDP(model, use_orig_params=True)

        FSDP.set_state_dict_type(
            fsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        fsdp_state_dict = fsdp_model.state_dict()

        # save fsdp_state_dict to storage
        dist_cp.save(
            state_dict=fsdp_state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        )

        # create a TP wrapped model
        mesh_shape = (self.world_size,)
        device_mesh = init_device_mesh(self.device_type, mesh_shape)
        model = MLPModule(self.device_type).to(self.rank)
        # Parallelize the module based on the given Parallel Style.
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, device_mesh, parallelize_plan)
        optimizer = torch.optim.SGD(tp_model.parameters(), lr=0.25)

        # Update the parameters so tp_model.state_dict() will be different from fsdp_model.state_dict().
        torch.manual_seed(0)
        inp = torch.rand(20, 10).to(self.rank)
        output = tp_model(inp)
        output.sum().backward()
        optimizer.step()
        tp_state_dict = tp_model.state_dict()

        # Check parameters are indeed different prior to loading.
        for fsdp_item, tp_item in zip(fsdp_state_dict.items(), tp_state_dict.items()):
            fsdp_k, fsdp_v = fsdp_item
            tp_k, tp_v = tp_item

            self.assertEqual(fsdp_k, tp_k)

            if isinstance(fsdp_v, ShardedTensor) and isinstance(tp_v, DTensor):
                fsdp_redistributed = _all_gather_sharded_tensor(fsdp_v)
                tp_redistributed = tp_v.redistribute(
                    device_mesh, placements=[Replicate()]
                ).to_local()
                self.assertNotEqual(fsdp_redistributed, tp_redistributed)

        dist_cp.load(
            state_dict=tp_state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
        tp_model.load_state_dict(tp_state_dict)

        # Check parameters are equal after loading.
        for fsdp_item, tp_item in zip(fsdp_state_dict.items(), tp_state_dict.items()):
            fsdp_k, fsdp_v = fsdp_item
            tp_k, tp_v = tp_item

            self.assertEqual(fsdp_k, tp_k)

            if isinstance(fsdp_v, ShardedTensor) and isinstance(tp_v, DTensor):
                fsdp_redistributed = _all_gather_sharded_tensor(fsdp_v)
                tp_redistributed = tp_v.redistribute(
                    device_mesh, placements=[Replicate()]
                ).to_local()
                self.assertEqual(fsdp_redistributed, tp_redistributed)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFsdpTpCheckpointConversion`

**Functions defined**: `test_fsdp_to_tp`

**Key imports**: torch, torch.distributed.checkpoint as dist_cp, ShardedTensor, _all_gather_sharded_tensor, init_device_mesh, FullyShardedDataParallel as FSDP, StateDictType, DTensor, Replicate, run_tests, with_temp_dir


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.checkpoint as dist_cp`
- `torch.distributed._shard.sharded_tensor`: ShardedTensor
- `torch.distributed._state_dict_utils`: _all_gather_sharded_tensor
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.fsdp.fully_sharded_data_parallel`: StateDictType
- `torch.distributed.tensor`: DTensor, Replicate
- `torch.testing._internal.common_utils`: run_tests
- `torch.testing._internal.distributed.checkpoint_utils`: with_temp_dir


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/distributed/checkpoint/test_fsdp_tp_checkpoint_conversion.py
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

- **File Documentation**: `test_fsdp_tp_checkpoint_conversion.py_docs.md`
- **Keyword Index**: `test_fsdp_tp_checkpoint_conversion.py_kw.md`
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

*No specific patterns automatically detected.*


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
python docs/test/distributed/checkpoint/test_fsdp_tp_checkpoint_conversion.py_docs.md
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

- **File Documentation**: `test_fsdp_tp_checkpoint_conversion.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_tp_checkpoint_conversion.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
