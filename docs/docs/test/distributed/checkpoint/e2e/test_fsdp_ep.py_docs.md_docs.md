# Documentation: `docs/test/distributed/checkpoint/e2e/test_fsdp_ep.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/e2e/test_fsdp_ep.py_docs.md`
- **Size**: 7,098 bytes (6.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/e2e/test_fsdp_ep.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/e2e/test_fsdp_ep.py`
- **Size**: 4,128 bytes (4.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin


class Dummymodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class EPModel(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class SecondTier(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.ep_layers = nn.ModuleList(
            [EPModel(rank) if rank % 4 == i else Dummymodel() for i in range(4)]
        )
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class TopModel(nn.Module):
    def __init__(self, rank):
        super().__init__()
        torch.manual_seed(0)

        self.second = SecondTier(rank)
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class TestFSDPWithEP(DTensorTestBase, VerifyStateDictMixin):
    @property
    def world_size(self) -> int:
        return min(8, torch.accelerator.device_count())

    @with_comms
    @skip_if_lt_x_gpu(8)
    @with_temp_dir
    def test_e2e(self):
        model = TopModel(self.rank).to(self.device_type)

        mesh_fsdp_tp = init_device_mesh(
            self.device_type, (2, 4), mesh_dim_names=("dp", "tp")
        )
        # TODO: we are using an internal API atm. Change to a public API once it is ready.
        mesh_fsdp_ep = mesh_fsdp_tp["dp"]
        mesh_fsdp_ep._root_mesh = None

        mesh_fsdp = init_device_mesh(self.device_type, (8,))
        for i, l in enumerate(model.second.ep_layers):
            model.second.ep_layers[i] = FSDP(
                l, use_orig_params=True, device_mesh=mesh_fsdp_ep
            )
        model.second = FSDP(model.second, use_orig_params=True, device_mesh=mesh_fsdp)
        model = FSDP(model, use_orig_params=True, device_mesh=mesh_fsdp)
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        msd, osd = get_state_dict(model, optim)

        # FSDP only params
        for key in (
            "net.0.weight",
            "net.0.bias",
            "second.net.0.weight",
            "second.net.0.bias",
        ):
            msd_v = msd[key]
            osd_v = osd["state"][key]["exp_avg"]
            for v in (msd_v, osd_v):
                self.assertTrue(isinstance(v, DTensor))
                self.assertEqual(tuple(v.device_mesh.mesh), tuple(range(8)))

        # FSDP/EP params
        layer = self.rank % 4
        ranks = (layer, layer + 4)
        for i in range(4):
            for key in (
                f"second.ep_layers.{i}.net1.0.weight",
                f"second.ep_layers.{i}.net1.0.bias",
                f"second.ep_layers.{i}.net2.0.weight",
                f"second.ep_layers.{i}.net2.0.bias",
            ):
                if layer != i:
                    self.assertTrue(key not in msd)
                else:
                    msd_v = msd[key]
                    osd_v = osd["state"][key]["exp_avg"]
                    for v in (msd_v, osd_v):
                        self.assertTrue(isinstance(v, DTensor))
                        self.assertEqual(tuple(v.device_mesh.mesh), ranks)

        self.assertEqual(set(osd["state"].keys()), set(msd.keys()))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Dummymodel`, `EPModel`, `SecondTier`, `TopModel`, `TestFSDPWithEP`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `world_size`, `test_e2e`

**Key imports**: torch, torch.nn as nn, get_state_dict, init_device_mesh, FullyShardedDataParallel as FSDP, DTensor, run_tests, with_temp_dir, VerifyStateDictMixin


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint/e2e`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn as nn`
- `torch.distributed.checkpoint.state_dict`: get_state_dict
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.tensor`: DTensor
- `torch.testing._internal.common_utils`: run_tests
- `torch.testing._internal.distributed.checkpoint_utils`: with_temp_dir
- `torch.testing._internal.distributed.common_state_dict`: VerifyStateDictMixin


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
python test/distributed/checkpoint/e2e/test_fsdp_ep.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/checkpoint/e2e`):

- [`test_e2e_save_and_load.py_docs.md`](./test_e2e_save_and_load.py_docs.md)
- [`test_fine_tuning.py_docs.md`](./test_fine_tuning.py_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_ep.py_docs.md`
- **Keyword Index**: `test_fsdp_ep.py_kw.md`
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
python docs/test/distributed/checkpoint/e2e/test_fsdp_ep.py_docs.md
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
- [`test_fine_tuning.py_docs.md_docs.md`](./test_fine_tuning.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_ep.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_ep.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
