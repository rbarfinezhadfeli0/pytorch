# Documentation: `docs/test/distributed/tensor/test_dtensor_dispatch_overhead.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/tensor/test_dtensor_dispatch_overhead.py_docs.md`
- **Size**: 8,464 bytes (8.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/tensor/test_dtensor_dispatch_overhead.py`

## File Metadata

- **Path**: `test/distributed/tensor/test_dtensor_dispatch_overhead.py`
- **Size**: 5,249 bytes (5.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import functools
import logging
import statistics
import time
from collections import namedtuple

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils._python_dispatch import TorchDispatchMode


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TimeCaptureMode(TorchDispatchMode):
    def __init__(self, repeat_count=10):
        # repeat each op call `repeat_count` times
        self.repeat_count = repeat_count
        # recorded time is scaled to micro seconds
        self.time_list = []
        self.op_to_time = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.time_list.clear()

        @functools.wraps(func)
        def repeated_func(*args, **kwargs):
            result = None
            for _ in range(self.repeat_count):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                self.time_list.append(elapsed_time)
            return result

        res = repeated_func(*args, **(kwargs or {}))

        Timing = namedtuple(
            "Timing", ["dispatch_with_cache_miss", "dispatch_with_cache_hit"]
        )
        if func.__name__ not in self.op_to_time:
            self.op_to_time[func.__name__] = []
        self.op_to_time[func.__name__].append(
            Timing(
                round(self.time_list[0] * 1e6, 2),
                round(statistics.median(self.time_list) * 1e6, 2),
            )
        )
        return res


class DistOpDispatchOverHead(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_dtensor_add_op_dispatch_overhead(self):
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_name(0)
            gpu_name = device_props
            logger.info("running on %s", gpu_name)
            # TODO: adjust `expected_propagate_time` and `expected_dispatch_time` to target different hardware
        else:
            self.skipTest("CUDA not available")
        expected_propagate_time = 880  # noqa: F841
        expected_dispatch_time = 90  # noqa: F841
        diff_percent_threshold = 0.20  # noqa: F841
        propagator = DTensor._op_dispatcher.sharding_propagator
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        input_data = torch.rand(512, 512, device="cuda")
        a = distribute_tensor(input_data, device_mesh, [Shard(0)])
        # warm up
        with TimeCaptureMode() as tcm:
            for _ in range(100):
                propagator.propagate_op_sharding.cache.cache_clear()
                _ = a + a
            # record number
            propagator.propagate_op_sharding.cache.cache_clear()
            _ = a + a
            add_dispatch_cache_miss, add_dispatch_cache_hit = tcm.op_to_time[
                "add.Tensor"
            ][-1]

        all_miss_performance = [0] * self.world_size
        all_hit_performance = [0] * self.world_size
        torch.distributed.all_gather_object(
            all_miss_performance, add_dispatch_cache_miss
        )
        torch.distributed.all_gather_object(all_hit_performance, add_dispatch_cache_hit)
        if self.rank == 0:
            logger.info(
                "add op dispatch cache miss from %s ranks: %s us, \n"
                "add op dispatch cache hit from %s ranks: %s us",
                self.world_size,
                all_miss_performance,
                self.world_size,
                all_hit_performance,
            )
        # compare median with expected range
        miss_performance = statistics.median(all_miss_performance)
        hit_performance = statistics.median(all_hit_performance)
        extra_time_spend_on_strategy_propagate = miss_performance - hit_performance  # noqa: F841
        # Do not enabling the assertion check due to flaky performance concern
        # self.assertTrue(
        #     (extra_time_spend_on_strategy_propagate - expected_propagate_time)
        #     / expected_propagate_time
        #     < diff_percent_threshold,
        #     msg=(
        #         f"extra time spend on strategy propagate is {extra_time_spend_on_strategy_propagate} us, "
        #         f"performance diff is {diff_percent_threshold * 100}% greater than expected {expected_propagate_time} us"
        #     ),
        # )
        # self.assertTrue(
        #     (hit_performance - expected_dispatch_time) / expected_dispatch_time
        #     < diff_percent_threshold,
        #     msg=(
        #         f"DTensor dispatch time is {hit_performance} us, "
        #         f"performance diff is {diff_percent_threshold * 100}% greater than "
        #         f"expected {expected_dispatch_time} us"
        #     ),
        # )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TimeCaptureMode`, `DistOpDispatchOverHead`

**Functions defined**: `__init__`, `__torch_dispatch__`, `repeated_func`, `world_size`, `test_dtensor_add_op_dispatch_overhead`

**Key imports**: functools, logging, statistics, time, namedtuple, torch, init_device_mesh, distribute_tensor, DTensor, Shard, run_tests, TorchDispatchMode


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `logging`
- `statistics`
- `time`
- `collections`: namedtuple
- `torch`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.tensor`: distribute_tensor, DTensor, Shard
- `torch.testing._internal.common_utils`: run_tests
- `torch.utils._python_dispatch`: TorchDispatchMode


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/tensor/test_dtensor_dispatch_overhead.py
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
- [`test_tensor_ops.py_docs.md`](./test_tensor_ops.py_docs.md)
- [`test_matrix_ops.py_docs.md`](./test_matrix_ops.py_docs.md)
- [`test_op_schema.py_docs.md`](./test_op_schema.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_attention.py_docs.md`](./test_attention.py_docs.md)


## Cross-References

- **File Documentation**: `test_dtensor_dispatch_overhead.py_docs.md`
- **Keyword Index**: `test_dtensor_dispatch_overhead.py_kw.md`
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


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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
python docs/test/distributed/tensor/test_dtensor_dispatch_overhead.py_docs.md
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

- **File Documentation**: `test_dtensor_dispatch_overhead.py_docs.md_docs.md`
- **Keyword Index**: `test_dtensor_dispatch_overhead.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
