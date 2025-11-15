# Documentation: `docs/test/distributed/checkpoint/test_fsspec.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/test_fsspec.py_docs.md`
- **Size**: 10,782 bytes (10.53 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/test_fsspec.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_fsspec.py`
- **Size**: 7,071 bytes (6.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import shutil
import tempfile
from collections.abc import Callable
from functools import wraps
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint._fsspec_filesystem import (
    FileSystem,
    FsspecReader,
    FsspecWriter,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.checkpoint.utils import CheckpointException
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
BACKEND = torch.distributed.get_default_backend_for_device(device_type)


def with_temp_dir(
    func: Optional[Callable] = None,
) -> Optional[Callable]:
    """
    Wrapper to initialize temp directory for distributed checkpoint.
    """
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: tuple[object], **kwargs: dict[str, Any]) -> None:
        # Only create temp_dir when rank is 0 (or no pg)
        if not dist.is_initialized() or dist.get_rank() == 0:
            temp_dir = tempfile.mkdtemp()
            print(f"Using temp directory: {temp_dir}")
        else:
            temp_dir = ""
        object_list = [temp_dir]

        # Broadcast temp_dir to all the other ranks
        if dist.is_initialized():
            dist.broadcast_object_list(object_list)
        self.temp_dir = object_list[0]

        try:
            func(self, *args, **kwargs)
        finally:
            if not dist.is_initialized() or dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    return wrapper


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))


class TestFSSpec(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(backend=BACKEND, init_rpc=False)
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_fsspec(self):
        CHECKPOINT_DIR = self.temp_dir

        model = FSDP(MyTestModule().to(device_type))
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()
        optim.step()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
                "optim": FSDP.optim_state_dict(model, optim),
            }

            dcp.save(
                state_dict=state_dict,
                storage_writer=FsspecWriter(CHECKPOINT_DIR),
                planner=dcp.DefaultSavePlanner(),
            )

        model_2 = FSDP(MyTestModule().to(device_type))
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                for n_p1, n_p2 in zip(
                    model.named_parameters(), model_2.named_parameters()
                ):
                    self.assertNotEqual(n_p1[1], n_p2[1])

        # now load the model and ensure the values are the same
        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model_2.state_dict(),
            }

            dcp.load(
                state_dict=state_dict,
                storage_reader=FsspecReader(CHECKPOINT_DIR),
                planner=dcp.DefaultLoadPlanner(),
            )
            model_2.load_state_dict(state_dict["model"])

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=FsspecReader(CHECKPOINT_DIR),
            )

            flattened_osd = FSDP.optim_state_dict_to_load(
                model_2, optim_2, optim_state["optim"]
            )
            optim_2.load_state_dict(flattened_osd)

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                for n_p1, n_p2 in zip(
                    model.named_parameters(), model_2.named_parameters()
                ):
                    self.assertEqual(n_p1[1], n_p2[1])

        def opt_at(opt, idx):
            return list(iter(opt.state.values()))[idx]

        # Adam lazily creates its state
        self.assertEqual(opt_at(optim, 0)["exp_avg"], opt_at(optim_2, 0)["exp_avg"])
        self.assertEqual(
            opt_at(optim, 0)["exp_avg_sq"], opt_at(optim_2, 0)["exp_avg_sq"]
        )

    @with_comms(backend=BACKEND, init_rpc=False)
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_overwrite(self):
        t1, t2 = torch.randn(10), torch.randn(10)

        dcp.save(
            {"random": t1}, storage_writer=FsspecWriter(self.temp_dir, overwrite=False)
        )
        dcp.save(
            {"random": t2}, storage_writer=FsspecWriter(self.temp_dir, overwrite=True)
        )

        sd = {"random": torch.zeros(10)}
        dcp.load(sd, checkpoint_id=self.temp_dir)
        self.assertTrue(torch.allclose(sd["random"], t2))

        with self.assertRaisesRegex(
            CheckpointException, ".*Checkpoint already exists.*"
        ):
            dcp.save(
                {"random": t2},
                storage_writer=FsspecWriter(self.temp_dir, overwrite=False),
            )


class TestFileSystem(TestCase):
    @with_temp_dir
    def test_remove_on_fail(self):
        fs = FileSystem()
        path = fs.init_path(self.temp_dir)

        write_file = fs.concat_path(path, "writeable")
        with self.assertRaises(OSError):
            with fs.create_stream(write_file, "w") as s:
                s.write("aaa")
                raise OSError("fail")
        self.assertFalse(fs.exists(write_file))

        read_file = fs.concat_path(path, "readable")
        with fs.create_stream(read_file, "w") as s:
            s.write("bbb")
        self.assertTrue(fs.exists(read_file))

        with self.assertRaises(OSError):
            with fs.create_stream(read_file, "r") as s:
                raise OSError("fail")
        self.assertTrue(fs.exists(read_file))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""    Wrapper to initialize temp directory for distributed checkpoint.

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyTestModule`, `TestFSSpec`, `TestFileSystem`

**Functions defined**: `with_temp_dir`, `wrapper`, `__init__`, `forward`, `world_size`, `test_fsspec`, `opt_at`, `test_overwrite`, `test_remove_on_fail`

**Key imports**: shutil, tempfile, Callable, wraps, Any, Optional, torch, torch.distributed as dist, torch.distributed.checkpoint as dcp, torch.nn as nn, load_sharded_optimizer_state_dict


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `shutil`
- `tempfile`
- `collections.abc`: Callable
- `functools`: wraps
- `typing`: Any, Optional
- `torch`
- `torch.distributed as dist`
- `torch.distributed.checkpoint as dcp`
- `torch.nn as nn`
- `torch.distributed.checkpoint.optimizer`: load_sharded_optimizer_state_dict
- `torch.distributed.checkpoint.utils`: CheckpointException
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.fsdp.fully_sharded_data_parallel`: StateDictType
- `torch.testing._internal.common_utils`: run_tests, TestCase


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
python test/distributed/checkpoint/test_fsspec.py
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

- **File Documentation**: `test_fsspec.py_docs.md`
- **Keyword Index**: `test_fsspec.py_kw.md`
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
python docs/test/distributed/checkpoint/test_fsspec.py_docs.md
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

- **File Documentation**: `test_fsspec.py_docs.md_docs.md`
- **Keyword Index**: `test_fsspec.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
