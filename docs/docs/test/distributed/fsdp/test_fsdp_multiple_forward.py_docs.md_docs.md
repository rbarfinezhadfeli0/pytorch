# Documentation: `docs/test/distributed/fsdp/test_fsdp_multiple_forward.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_fsdp_multiple_forward.py_docs.md`
- **Size**: 6,134 bytes (5.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_fsdp_multiple_forward.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_multiple_forward.py`
- **Size**: 2,587 bytes (2.53 KB)
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, get_full_params
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


device_type = torch.device(get_devtype())

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class Model(Module):
    def __init__(self, wrap_fsdp):
        super().__init__()
        # keep everything deterministic for model initialization
        torch.manual_seed(0)
        self.inner = Linear(4, 4)
        if wrap_fsdp:
            self.inner = FSDP(self.inner)
        self.outer = Linear(4, 5)

    def forward(self, x):
        # Forward twice.
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)


class TestMultiForward(FSDPTest):
    def _dist_train(self, wrap_fsdp):
        # keep everything deterministic for input data
        torch.manual_seed(0)
        model = Model(wrap_fsdp).to(device_type.type)
        if wrap_fsdp:
            model = FSDP(model, device_id=device_type.type)
        else:
            model = DistributedDataParallel(model, device_ids=[device_type.type])
        optim = SGD(model.parameters(), lr=0.1)
        in_data = torch.rand(64, 4).to(device_type.type)
        in_data.requires_grad = True
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()
        if wrap_fsdp:
            return get_full_params(model)
        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    def test_multi_forward(self):
        # DDP
        ddp_state = self._dist_train(wrap_fsdp=False)
        # FSDP
        fsdp_state = self._dist_train(wrap_fsdp=True)
        self.assertEqual(ddp_state, fsdp_state)


devices = ("cpu", "hpu", "xpu")
instantiate_device_type_tests(
    TestMultiForward, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Model`, `TestMultiForward`

**Functions defined**: `__init__`, `forward`, `_dist_train`, `test_multi_forward`

**Key imports**: sys, torch, distributed as dist, FullyShardedDataParallel as FSDP, Linear, Module, DistributedDataParallel, SGD, instantiate_device_type_tests, skip_if_lt_x_gpu, FSDPTest, get_devtype, get_full_params


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.nn`: Linear, Module
- `torch.nn.parallel`: DistributedDataParallel
- `torch.optim`: SGD
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_fsdp`: FSDPTest, get_devtype, get_full_params
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


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
python test/distributed/fsdp/test_fsdp_multiple_forward.py
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
- [`test_distributed_checkpoint.py_docs.md`](./test_distributed_checkpoint.py_docs.md)
- [`test_checkpoint_wrapper.py_docs.md`](./test_checkpoint_wrapper.py_docs.md)
- [`test_fsdp_clip_grad_norm.py_docs.md`](./test_fsdp_clip_grad_norm.py_docs.md)
- [`test_fsdp_use_orig_params.py_docs.md`](./test_fsdp_use_orig_params.py_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_multiple_forward.py_docs.md`
- **Keyword Index**: `test_fsdp_multiple_forward.py_kw.md`
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
python docs/test/distributed/fsdp/test_fsdp_multiple_forward.py_docs.md
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

- **File Documentation**: `test_fsdp_multiple_forward.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_multiple_forward.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
