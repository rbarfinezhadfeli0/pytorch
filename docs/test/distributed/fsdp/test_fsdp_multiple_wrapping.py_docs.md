# Documentation: `test/distributed/fsdp/test_fsdp_multiple_wrapping.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_multiple_wrapping.py`
- **Size**: 2,450 bytes (2.39 KB)
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
from torch.nn import Linear, Module, Sequential
from torch.optim import SGD
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
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


class InnerModel(Module):
    def __init__(self, device):
        super().__init__()
        self.layers = Sequential(FSDP(Linear(5, 5), device_id=device_type.type))

    def forward(self, x):
        return self.layers(x)


class TestMultipleWrapping(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_multiple_wrapping(self, device):
        """
        This test simulates wrapping the module after training to run inference.
        This is required in cases where later in a session, the model is wrapped again in FSDP but
        contains nested FSDP wrappers within the module.
        """
        inner_model = InnerModel(device)
        model = FSDP(inner_model).to(device_type.type)
        optim = SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            input = torch.rand((1, 5), dtype=torch.float).to(device_type.type)
            input.requires_grad = True
            output = model(input)
            output.sum().backward()
            optim.step()
            optim.zero_grad()
        input = torch.rand((1, 5), dtype=torch.float).to(device_type.type)
        output = model(input)
        # second time to rewrap the inner model
        # rewrapped_model = FSDP(inner_model, device_id=device)
        rewrapped_model = FSDP(inner_model).to(device_type.type)
        rewrapped_output = rewrapped_model(input)
        self.assertEqual(output, rewrapped_output)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestMultipleWrapping, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""        This test simulates wrapping the module after training to run inference.        This is required in cases where later in a session, the model is wrapped again in FSDP but        contains nested FSDP wrappers within the module.

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `InnerModel`, `TestMultipleWrapping`

**Functions defined**: `__init__`, `forward`, `test_multiple_wrapping`

**Key imports**: sys, torch, distributed as dist, FullyShardedDataParallel as FSDP, Linear, Module, Sequential, SGD, instantiate_device_type_tests, skip_if_lt_x_gpu, FSDPTest, get_devtype, run_tests, TEST_WITH_DEV_DBG_ASAN


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
- `torch.nn`: Linear, Module, Sequential
- `torch.optim`: SGD
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_fsdp`: FSDPTest, get_devtype
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
python test/distributed/fsdp/test_fsdp_multiple_wrapping.py
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

- **File Documentation**: `test_fsdp_multiple_wrapping.py_docs.md`
- **Keyword Index**: `test_fsdp_multiple_wrapping.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
