# Documentation: `docs/test/distributed/fsdp/test_distributed_checkpoint.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_distributed_checkpoint.py_docs.md`
- **Size**: 6,947 bytes (6.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_distributed_checkpoint.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_distributed_checkpoint.py`
- **Size**: 3,346 bytes (3.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch import distributed as dist
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, SkipModel
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
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


_DISTRIBUTED_STATE_DICT_IMPLS = {
    StateDictType.LOCAL_STATE_DICT,
    StateDictType.SHARDED_STATE_DICT,
}


class TestDistributedCheckpoint(FSDPTest):
    @property
    def world_size(self):
        if torch.accelerator.is_available():
            gpu_cnt = torch.accelerator.device_count()
            if gpu_cnt < 2:
                return gpu_cnt
        return 2

    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    @parametrize("state_dict_type", _DISTRIBUTED_STATE_DICT_IMPLS)
    def test_distributed_checkpoint(self, state_dict_type) -> None:
        with enable_wrap(wrapper_cls=FSDP):
            torch.manual_seed(100)
            model = wrap(SkipModel(double_nest=True))
            torch.manual_seed(200)
            new_model = wrap(SkipModel(double_nest=True))

        with (
            FullyShardedDataParallel.summon_full_params(model),
            FullyShardedDataParallel.summon_full_params(new_model),
        ):
            params = list(model.parameters())
            new_params = list(new_model.parameters())
            self.assertNotEqual(params, new_params)

        writer = FileSystemWriter(self.temp_dir)
        reader = FileSystemReader(self.temp_dir)
        with (
            FSDP.state_dict_type(model, state_dict_type),
            FSDP.state_dict_type(new_model, state_dict_type),
        ):
            state_dict = model.state_dict()

        save(state_dict, writer)

        with (
            FSDP.state_dict_type(model, state_dict_type),
            FSDP.state_dict_type(new_model, state_dict_type),
        ):
            state_dict = new_model.state_dict()
            load(state_dict, reader)
            new_model.load_state_dict(state_dict)

        with (
            FullyShardedDataParallel.summon_full_params(model),
            FullyShardedDataParallel.summon_full_params(new_model),
        ):
            params = list(model.parameters())
            new_params = list(new_model.parameters())
            self.assertEqual(params, new_params)

        # TODO: add resharding test case.


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestDistributedCheckpoint, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDistributedCheckpoint`

**Functions defined**: `world_size`, `test_distributed_checkpoint`

**Key imports**: sys, torch, distributed as dist, FileSystemReader, FileSystemWriter, load, save, FullyShardedDataParallel as FSDP, StateDictType, FullyShardedDataParallel, enable_wrap, wrap, instantiate_device_type_tests, skip_if_lt_x_gpu, FSDPTest, SkipModel


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.distributed.checkpoint`: FileSystemReader, FileSystemWriter, load, save
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP, StateDictType
- `torch.distributed.fsdp.fully_sharded_data_parallel`: FullyShardedDataParallel
- `torch.distributed.fsdp.wrap`: enable_wrap, wrap
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_fsdp`: FSDPTest, SkipModel
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
python test/distributed/fsdp/test_distributed_checkpoint.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/fsdp`):

- [`test_fsdp_memory.py_docs.md`](./test_fsdp_memory.py_docs.md)
- [`test_fsdp_mixed_precision.py_docs.md`](./test_fsdp_mixed_precision.py_docs.md)
- [`test_fsdp_uneven.py_docs.md`](./test_fsdp_uneven.py_docs.md)
- [`test_fsdp_dtensor_state_dict.py_docs.md`](./test_fsdp_dtensor_state_dict.py_docs.md)
- [`test_fsdp_tp_integration.py_docs.md`](./test_fsdp_tp_integration.py_docs.md)
- [`test_fsdp_multiple_forward.py_docs.md`](./test_fsdp_multiple_forward.py_docs.md)
- [`test_checkpoint_wrapper.py_docs.md`](./test_checkpoint_wrapper.py_docs.md)
- [`test_fsdp_clip_grad_norm.py_docs.md`](./test_fsdp_clip_grad_norm.py_docs.md)
- [`test_fsdp_use_orig_params.py_docs.md`](./test_fsdp_use_orig_params.py_docs.md)


## Cross-References

- **File Documentation**: `test_distributed_checkpoint.py_docs.md`
- **Keyword Index**: `test_distributed_checkpoint.py_kw.md`
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
python docs/test/distributed/fsdp/test_distributed_checkpoint.py_docs.md
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

- **File Documentation**: `test_distributed_checkpoint.py_docs.md_docs.md`
- **Keyword Index**: `test_distributed_checkpoint.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
