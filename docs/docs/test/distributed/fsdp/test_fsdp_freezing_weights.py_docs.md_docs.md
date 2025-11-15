# Documentation: `docs/test/distributed/fsdp/test_fsdp_freezing_weights.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_fsdp_freezing_weights.py_docs.md`
- **Size**: 11,035 bytes (10.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_fsdp_freezing_weights.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_freezing_weights.py`
- **Size**: 7,454 bytes (7.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import contextlib
import sys
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_full_params
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class Model(nn.Module):
    def __init__(
        self,
        with_fsdp,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 10)
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap(fsdp_kwargs)
        self.autograd_ctx = (
            torch.no_grad if disable_autograd else contextlib.nullcontext
        )

    def fsdp_wrap(self, fsdp_kwargs):
        self.trunk = FSDP(self.trunk, **fsdp_kwargs)
        self.head = FSDP(self.head, **fsdp_kwargs)

    def forward(self, x):
        with self.autograd_ctx():
            x = self.trunk(x)
        return self.head(x)


class NestedTrunkModel(nn.Module):
    def __init__(
        self,
        with_fsdp,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            self._create_block(3, 64, with_fsdp, freeze_after_wrap_fsdp),
            self._create_block(64, 64, with_fsdp, freeze_after_wrap_fsdp),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap(fsdp_kwargs)
        self.autograd_ctx = (
            torch.no_grad if disable_autograd else contextlib.nullcontext
        )

    def fsdp_wrap(self, fsdp_kwargs):
        for name, child in self.trunk.named_children():
            wrapped_child = FSDP(child, **fsdp_kwargs)
            setattr(self.trunk, name, wrapped_child)
        self.trunk = FSDP(self.trunk, **fsdp_kwargs)
        self.head = FSDP(self.head, **fsdp_kwargs)

    def forward(self, x):
        with self.autograd_ctx():
            x = self.trunk(x)
        return self.head(x)

    def _create_block(
        self, in_channels, out_channels, with_fsdp, freeze_after_wrap_fsdp
    ):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        return block


class FreezingMethod(str, Enum):
    GradToNone = "grad_to_none"
    RequiresGrad = "requires_grad"


class TestFreezingWeights(FSDPTest):
    def _create_model(
        self,
        with_fsdp,
        with_nested_trunk,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        if with_nested_trunk:
            model = NestedTrunkModel(
                with_fsdp, freeze_after_wrap_fsdp, disable_autograd, fsdp_kwargs
            )
        else:
            model = Model(
                with_fsdp, freeze_after_wrap_fsdp, disable_autograd, fsdp_kwargs
            )
        return model

    def _dist_train(
        self,
        with_nested_trunk,
        freezing_method,
        freeze_after_wrap_fsdp,
        with_fsdp,
        disable_autograd,
        forward_prefetch,
    ):
        torch.manual_seed(0)
        batch = torch.randn(size=(2, 3, 224, 224)).to(device_type)

        fsdp_kwargs = {
            "device_id": self.rank,
            "forward_prefetch": forward_prefetch,
        }

        ddp_kwargs = {
            "device_ids": [self.rank],
            "find_unused_parameters": bool(disable_autograd),
        }

        model = self._create_model(
            with_fsdp,
            with_nested_trunk,
            freeze_after_wrap_fsdp,
            disable_autograd,
            fsdp_kwargs,
        )
        model = model.to(device_type)

        # freezing the trunk using requires_grad.
        if freezing_method == FreezingMethod.RequiresGrad:
            for param in model.trunk.parameters():
                param.requires_grad = False

        if with_fsdp:
            if not freeze_after_wrap_fsdp:
                model.fsdp_wrap(fsdp_kwargs)
            model = FSDP(model, **fsdp_kwargs)
        else:
            model = DistributedDataParallel(model, **ddp_kwargs)

        target = torch.tensor([0, 1], dtype=torch.long).to(device_type)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        for _ in range(3):
            out = model(batch)
            fake_loss = criterion(out, target)
            optimizer.zero_grad()
            fake_loss.backward()
            if freezing_method == FreezingMethod.GradToNone:
                for param in model.module.trunk.parameters():
                    param.grad = None
            optimizer.step()

        if with_fsdp:
            return get_full_params(model)

        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("with_nested_trunk", [True, False])
    @parametrize(
        "freezing_method", [FreezingMethod.RequiresGrad, FreezingMethod.GradToNone]
    )
    @parametrize("freeze_after_wrap_fsdp", [True, False])
    @parametrize("disable_autograd", [True, False])
    @parametrize("forward_prefetch", [True, False])
    def test_freezing_weights(
        self,
        with_nested_trunk,
        freezing_method,
        freeze_after_wrap_fsdp,
        disable_autograd,
        forward_prefetch,
    ):
        # DDP
        ddp_state = self._dist_train(
            with_nested_trunk,
            freezing_method,
            freeze_after_wrap_fsdp,
            with_fsdp=False,
            disable_autograd=disable_autograd,
            forward_prefetch=False,  # does not apply to DDP
        )

        # FSDP
        fsdp_state = self._dist_train(
            with_nested_trunk,
            freezing_method,
            freeze_after_wrap_fsdp,
            with_fsdp=True,
            disable_autograd=disable_autograd,
            forward_prefetch=forward_prefetch,
        )

        self.assertEqual(
            ddp_state,
            fsdp_state,
            exact_device=True,
            msg="FullyShardedDataParallel states didn't match PyTorch DDP states",
        )

        if freezing_method == FreezingMethod.RequiresGrad:
            for ddp_param, fsdp_param in zip(ddp_state, fsdp_state):
                self.assertEqual(ddp_param.requires_grad, fsdp_param.requires_grad)


instantiate_parametrized_tests(TestFreezingWeights)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Model`, `NestedTrunkModel`, `FreezingMethod`, `TestFreezingWeights`

**Functions defined**: `__init__`, `fsdp_wrap`, `forward`, `__init__`, `fsdp_wrap`, `forward`, `_create_block`, `_create_model`, `_dist_train`, `test_freezing_weights`

**Key imports**: contextlib, sys, Enum, torch, torch.nn as nn, torch.optim as optim, distributed as dist, FullyShardedDataParallel as FSDP, DistributedDataParallel, skip_if_lt_x_gpu


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `sys`
- `enum`: Enum
- `torch`
- `torch.nn as nn`
- `torch.optim as optim`
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.nn.parallel`: DistributedDataParallel
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_fsdp`: FSDPTest, get_full_params


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
python test/distributed/fsdp/test_fsdp_freezing_weights.py
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
- [`test_fsdp_multiple_forward.py_docs.md`](./test_fsdp_multiple_forward.py_docs.md)
- [`test_checkpoint_wrapper.py_docs.md`](./test_checkpoint_wrapper.py_docs.md)
- [`test_fsdp_clip_grad_norm.py_docs.md`](./test_fsdp_clip_grad_norm.py_docs.md)
- [`test_fsdp_use_orig_params.py_docs.md`](./test_fsdp_use_orig_params.py_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_freezing_weights.py_docs.md`
- **Keyword Index**: `test_fsdp_freezing_weights.py_kw.md`
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
python docs/test/distributed/fsdp/test_fsdp_freezing_weights.py_docs.md
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

- **File Documentation**: `test_fsdp_freezing_weights.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_freezing_weights.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
