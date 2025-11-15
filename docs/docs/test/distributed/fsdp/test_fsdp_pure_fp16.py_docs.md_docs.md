# Documentation: `docs/test/distributed/fsdp/test_fsdp_pure_fp16.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_fsdp_pure_fp16.py_docs.md`
- **Size**: 8,924 bytes (8.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_fsdp_pure_fp16.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_pure_fp16.py`
- **Size**: 5,520 bytes (5.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch import distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    get_devtype,
    NestedWrappedModule,
)
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


class TestPureFP16(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_pure_fp16_training(self):
        """Tests pure FP16 training, including when the parameter's dtype is
        changed after FSDP initialization and before training."""
        self.run_subtests(
            {
                "cpu_offload": [
                    CPUOffload(offload_params=True),
                    CPUOffload(offload_params=False),
                ]
            },
            self._test_pure_fp16_training,
        )

    def _test_pure_fp16_training(self, cpu_offload: CPUOffload):
        self._test_fsdp_parity(
            NestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            device_init_mode=DEVICEInitMode.DEVICE_BEFORE,
            # Run one iteration to avoid NaN without a gradient scaler
            num_iters=1,
            cpu_offload=cpu_offload,
            use_pure_fp16=True,
        )

    @skip_if_lt_x_gpu(2)
    def test_fp16_dtypes(self):
        """
        Tests that both user-facing parameter/gradient dtypes and internal
        saved dtype attributes are as expected when using an FP16 model
        possibly with explicit mixed precision enabled.
        """
        self.run_subtests(
            {
                "to_half_before_fsdp_init": [False, True],
                "use_orig_params": [False, True],
                "mixed_precision": [
                    MixedPrecision(),
                    MixedPrecision(
                        param_dtype=torch.float16,
                        reduce_dtype=torch.float32,
                    ),
                    MixedPrecision(
                        param_dtype=torch.float32,
                    ),
                ],
            },
            self._test_fp16_dtypes,
        )

    def _test_fp16_dtypes(
        self,
        to_half_before_fsdp_init: bool,
        use_orig_params: bool,
        mixed_precision: MixedPrecision,
    ):
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_NEVER,
            {
                "device_id": device_type,
            },
        )
        fsdp_kwargs = {
            "use_orig_params": use_orig_params,
            "device_id": device_type,
            "mixed_precision": mixed_precision,
        }
        if to_half_before_fsdp_init:
            model = model.half()
        fsdp_model = FSDP(model, **fsdp_kwargs)
        if not to_half_before_fsdp_init:
            fsdp_model = fsdp_model.half()
        for param in fsdp_model.parameters():
            self.assertEqual(param.dtype, torch.float16)
        inp = tuple(
            t.half() if torch.is_tensor(t) else t
            for t in fsdp_model.module.get_input(self.device_type)
        )
        out = fsdp_model(*inp)
        out.sum().backward()

        # Check handle dtype attributes
        for handle in traversal_utils._get_fsdp_handles(fsdp_model):
            self.assertEqual(handle.flat_param.dtype, torch.float16)
            self.assertEqual(handle.flat_param.grad.dtype, torch.float16)
            self.assertEqual(handle._orig_param_dtype, torch.float16)
            # Specifying `mixed_precision` takes precedence over the model
            # dtype for both `param_dtype` and `reduce_dtype`
            if mixed_precision.param_dtype is not None:
                self.assertEqual(
                    handle._fwd_bwd_param_dtype, mixed_precision.param_dtype
                )
            else:
                self.assertEqual(handle._fwd_bwd_param_dtype, torch.float16)
            if mixed_precision.reduce_dtype is not None:
                self.assertEqual(handle._reduce_dtype, mixed_precision.reduce_dtype)
            elif (
                mixed_precision.reduce_dtype is None
                and mixed_precision.param_dtype is not None
            ):
                # Special case: infer reduce dtype from parameter dtype
                self.assertEqual(handle._reduce_dtype, mixed_precision.param_dtype)
            else:
                self.assertEqual(handle._reduce_dtype, torch.float16)

        # Check parameter/gradient dtypes
        for param in fsdp_model.parameters():
            self.assertEqual(param.dtype, torch.float16)
            if param.grad is not None:
                self.assertEqual(param.grad.dtype, torch.float16)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(TestPureFP16, globals(), only_for=devices, allow_xpu=True)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Tests pure FP16 training, including when the parameter's dtype is

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPureFP16`

**Functions defined**: `test_pure_fp16_training`, `_test_pure_fp16_training`, `test_fp16_dtypes`, `_test_fp16_dtypes`

**Key imports**: sys, torch, torch.distributed.fsdp._traversal_utils as traversal_utils, distributed as dist, instantiate_device_type_tests, skip_if_lt_x_gpu, run_tests, TEST_WITH_DEV_DBG_ASAN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.distributed.fsdp._traversal_utils as traversal_utils`
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


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
python test/distributed/fsdp/test_fsdp_pure_fp16.py
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

- **File Documentation**: `test_fsdp_pure_fp16.py_docs.md`
- **Keyword Index**: `test_fsdp_pure_fp16.py_kw.md`
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
python docs/test/distributed/fsdp/test_fsdp_pure_fp16.py_docs.md
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

- **File Documentation**: `test_fsdp_pure_fp16.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_pure_fp16.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
