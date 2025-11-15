# Documentation: `docs/test/distributed/tensor/test_xla_integration.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/tensor/test_xla_integration.py_docs.md`
- **Size**: 9,889 bytes (9.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/tensor/test_xla_integration.py`

## File Metadata

- **Path**: `test/distributed/tensor/test_xla_integration.py`
- **Size**: 6,636 bytes (6.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import os
import unittest
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

import torch
from torch import nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# wrapper to check xla test requirements
def with_xla(func: Callable) -> Callable:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self,
        *args: tuple[object],
        **kwargs: dict[str, Any],  # type: ignore[misc]
    ) -> None:
        # TODO(yeounoh) replace this with xr.use_spmd() when we deprecate the flag.
        os.environ["XLA_USE_SPMD"] = "1"
        try:
            import torch_xla  # type:ignore[import]  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("torch_xla is not installed.") from exc
        self.device_type = "xla"
        func(self, *args, **kwargs)  # type: ignore[misc]
        os.environ["XLA_USE_SPMD"] = "0"

    return wrapper


class DTensorXLAIntegrationTest(TestCase):
    class SimpleLinear(nn.Module):
        def __init__(self) -> None:
            super(DTensorXLAIntegrationTest.SimpleLinear, self).__init__()
            self.fc1 = nn.Linear(128, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            y = self.relu(self.fc1(x))
            z = self.fc2(y)
            return z

    @with_xla
    def test_xla_distribute_tensor_1d_shard(self):
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        if device_count > 1:
            device_mesh = DeviceMesh("xla", list(range(device_count)))
            shard_spec = [Shard(0)]

            for requires_grad in [True, False]:
                tensor_to_shard = torch.randn(
                    3 * device_count, 3, requires_grad=requires_grad
                )
                dist_tensor = distribute_tensor(
                    tensor_to_shard, device_mesh, shard_spec
                )
                # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
                assert type(dist_tensor).__name__ == "XLAShardedTensor"
                global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
                self.assertEqual(
                    global_tensor.size(), torch.Size([3 * device_count, 3])
                )
                local_tensor = dist_tensor.local_shards[0].data
                self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
                if requires_grad:
                    self.assertTrue(dist_tensor.global_tensor.requires_grad)
                    self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def test_xla_distribute_tensor_1d_replicate(self):
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        device_mesh = DeviceMesh("xla", list(range(device_count)))
        shard_spec = [Replicate()]

        for requires_grad in [True, False]:
            tensor_to_shard = torch.randn(
                3 * device_count, 3, requires_grad=requires_grad
            )
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
            assert type(dist_tensor).__name__ == "XLAShardedTensor"
            global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
            self.assertEqual(global_tensor.size(), torch.Size([3 * device_count, 3]))
            local_tensor = dist_tensor.local_shards[0].data
            self.assertEqual(local_tensor.size(), torch.Size([3 * device_count, 3]))
            if requires_grad:
                self.assertTrue(dist_tensor.global_tensor.requires_grad)
                self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def test_xla_distribute_tensor_2d(self):
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        if device_count > 1:
            device_mesh = DeviceMesh(
                "xla", np.array(range(device_count)).reshape(2, device_count // 2)
            )
            shard_spec = [Replicate(), Shard(0)]

            for requires_grad in [True, False]:
                tensor_to_shard = torch.randn(
                    3 * device_count // 2, 3, requires_grad=requires_grad
                )
                dist_tensor = distribute_tensor(
                    tensor_to_shard, device_mesh, shard_spec
                )
                # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
                assert type(dist_tensor).__name__ == "XLAShardedTensor"
                global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
                self.assertEqual(
                    global_tensor.size(), torch.Size([3 * device_count // 2, 3])
                )
                local_tensor = dist_tensor.local_shards[0].data
                self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
                if requires_grad:
                    self.assertTrue(dist_tensor.global_tensor.requires_grad)
                    self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def text_xla_distribute_module(self):
        import torch_xla  # type:ignore[import]
        import torch_xla.core.xla_model as xm  # type:ignore[import]
        import torch_xla.runtime as xr  # type:ignore[import]

        model = self.SimpleLinear().to(xm.xla_device())

        device_count = xr.global_runtime_device_count()
        device_mesh = DeviceMesh("xla", list(range(device_count)))

        def shard_params(mod_name, mod, mesh):
            shard_spec = [Shard(0)]
            # annotate fc1 and fc2
            if isinstance(mod, nn.Linear):
                for _, param in mod.named_parameters():
                    # annotate the parameter tensors directly
                    distribute_tensor(param, mesh, shard_spec)

        sharded_model = distribute_module(model, device_mesh, shard_params)
        self.assertTrue(
            torch_xla._XLAC._get_xla_sharding_spec(sharded_model.fc1.weight) != ""
        )
        self.assertTrue(
            torch_xla._XLAC._get_xla_sharding_spec(sharded_model.fc2.weight) != ""
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DTensorXLAIntegrationTest`, `SimpleLinear`

**Functions defined**: `with_xla`, `wrapper`, `__init__`, `forward`, `test_xla_distribute_tensor_1d_shard`, `test_xla_distribute_tensor_1d_replicate`, `test_xla_distribute_tensor_2d`, `text_xla_distribute_module`, `shard_params`

**Key imports**: os, unittest, Callable, wraps, Any, numpy as np, torch, nn, run_tests, TestCase, torch_xla  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest`
- `collections.abc`: Callable
- `functools`: wraps
- `typing`: Any
- `numpy as np`
- `torch`
- `torch.testing._internal.common_utils`: run_tests, TestCase
- `torch_xla  `
- `torch_xla.runtime as xr  `
- `torch_xla.core.xla_model as xm  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/tensor/test_xla_integration.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_dtensor.py_docs.md`](./test_dtensor.py_docs.md)
- [`test_dtensor_testbase.py_docs.md`](./test_dtensor_testbase.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_dtensor_dispatch_overhead.py_docs.md`](./test_dtensor_dispatch_overhead.py_docs.md)
- [`test_tensor_ops.py_docs.md`](./test_tensor_ops.py_docs.md)
- [`test_matrix_ops.py_docs.md`](./test_matrix_ops.py_docs.md)
- [`test_op_schema.py_docs.md`](./test_op_schema.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_attention.py_docs.md`](./test_attention.py_docs.md)


## Cross-References

- **File Documentation**: `test_xla_integration.py_docs.md`
- **Keyword Index**: `test_xla_integration.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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
python docs/test/distributed/tensor/test_xla_integration.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/tensor`):

- [`test_math_ops.py_docs.md_docs.md`](./test_math_ops.py_docs.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_dtensor_export.py_docs.md_docs.md`](./test_dtensor_export.py_docs.md_docs.md)
- [`test_placement_types.py_docs.md_docs.md`](./test_placement_types.py_docs.md_docs.md)
- [`test_convolution_ops.py_kw.md_docs.md`](./test_convolution_ops.py_kw.md_docs.md)
- [`test_placement_types.py_kw.md_docs.md`](./test_placement_types.py_kw.md_docs.md)
- [`test_common_rules.py_kw.md_docs.md`](./test_common_rules.py_kw.md_docs.md)
- [`test_dtensor_compile.py_kw.md_docs.md`](./test_dtensor_compile.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_api.py_docs.md_docs.md`](./test_api.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_xla_integration.py_docs.md_docs.md`
- **Keyword Index**: `test_xla_integration.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
