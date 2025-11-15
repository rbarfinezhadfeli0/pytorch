# Documentation: `docs/test/distributed/fsdp/test_fsdp_apply.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_fsdp_apply.py_docs.md`
- **Size**: 7,542 bytes (7.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_fsdp_apply.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_apply.py`
- **Size**: 4,089 bytes (3.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    get_devtype,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = torch.device(get_devtype())


class TestApply(FSDPTest):
    @property
    def world_size(self):
        if torch.accelerator.is_available():
            gpu_cnt = torch.accelerator.device_count()
            if gpu_cnt < 2:
                return gpu_cnt
        return 2

    @torch.no_grad()
    def _init_linear_weights(self, m):
        if type(m) is nn.Linear:
            m.weight.fill_(1.0)
            m.bias.fill_(1.0)

    def check_weights(self, fsdp, expected_tensor_fn, check):
        with FSDP.summon_full_params(fsdp, recurse=True):
            linear_modules = [
                module for module in fsdp.modules() if type(module) is nn.Linear
            ]
            for module in linear_modules:
                for param in module.parameters():
                    expected = expected_tensor_fn(param)
                    check(param, expected, f"Got {param} but expected {expected}")

    def _check_apply(self, fsdp):
        # Assert linear weights are not all 1.0
        self.check_weights(
            fsdp, lambda param: torch.empty_like(param).fill_(1.0), self.assertNotEqual
        )

        fsdp.apply(self._init_linear_weights)

        # Ensure all weights are 1.0
        self.check_weights(
            fsdp, lambda param: torch.empty_like(param).fill_(1.0), self.assertEqual
        )

    @skip_if_lt_x_gpu(2)
    def test_nested_module_apply(self):
        """Tests that ``apply()`` modifies parameter values in-place on a
        non-FSDP-root nested FSDP-wrapped model."""
        fsdp_kwargs = {"device_id": device_type.type}
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs=fsdp_kwargs,
        )
        self._check_apply(nested_wrapped_module)

    @skip_if_lt_x_gpu(2)
    def test_transformer_module_apply(self):
        """Tests that ``apply()`` modifies parameter values in-place on an
        FSDP-wrapped transformer model with shared parameters."""
        fsdp_kwargs = {"device_id": device_type.type}
        transformer = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs=fsdp_kwargs,
        )
        self._check_apply(transformer)

    @skip_if_lt_x_gpu(2)
    def test_apply_in_summon_raises_error(self):
        """Tests that calling ``apply()`` on an FSDP instance inside the
        ``summon_full_params()`` context raises an error."""
        fsdp_kwargs = {"device_id": device_type.type}
        transformer = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
            fsdp_kwargs=fsdp_kwargs,
        )
        with transformer.summon_full_params(transformer):
            with self.assertRaisesRegex(ValueError, "expected to be in states"):
                transformer.apply(self._init_linear_weights)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(TestApply, globals(), only_for=devices, allow_xpu=True)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestApply`

**Functions defined**: `world_size`, `_init_linear_weights`, `check_weights`, `_check_apply`, `test_nested_module_apply`, `test_transformer_module_apply`, `test_apply_in_summon_raises_error`

**Key imports**: sys, torch, torch.distributed as dist, torch.nn as nn, FullyShardedDataParallel as FSDP, instantiate_device_type_tests, skip_if_lt_x_gpu, run_tests, TEST_WITH_DEV_DBG_ASAN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.distributed as dist`
- `torch.nn as nn`
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


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
python test/distributed/fsdp/test_fsdp_apply.py
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

- **File Documentation**: `test_fsdp_apply.py_docs.md`
- **Keyword Index**: `test_fsdp_apply.py_kw.md`
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
python docs/test/distributed/fsdp/test_fsdp_apply.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/fsdp`):

- [`test_fsdp_grad_acc.py_docs.md_docs.md`](./test_fsdp_grad_acc.py_docs.md_docs.md)
- [`test_fsdp_ignored_modules.py_kw.md_docs.md`](./test_fsdp_ignored_modules.py_kw.md_docs.md)
- [`test_fsdp_meta.py_kw.md_docs.md`](./test_fsdp_meta.py_kw.md_docs.md)
- [`test_fsdp_tp_integration.py_kw.md_docs.md`](./test_fsdp_tp_integration.py_kw.md_docs.md)
- [`test_fsdp_fx.py_docs.md_docs.md`](./test_fsdp_fx.py_docs.md_docs.md)
- [`test_fsdp_memory.py_kw.md_docs.md`](./test_fsdp_memory.py_kw.md_docs.md)
- [`test_fsdp_apply.py_kw.md_docs.md`](./test_fsdp_apply.py_kw.md_docs.md)
- [`test_fsdp_tp_integration.py_docs.md_docs.md`](./test_fsdp_tp_integration.py_docs.md_docs.md)
- [`test_fsdp_multiple_forward.py_kw.md_docs.md`](./test_fsdp_multiple_forward.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_apply.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_apply.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
