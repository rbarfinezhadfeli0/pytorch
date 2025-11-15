# Documentation: `docs/test/distributed/test_c10d_logger.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/test_c10d_logger.py_docs.md`
- **Size**: 7,834 bytes (7.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/test_c10d_logger.py`

## File Metadata

- **Path**: `test/distributed/test_c10d_logger.py`
- **Size**: 4,555 bytes (4.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import json
import logging
import re
import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.distributed.c10d_logger import _c10d_logger, _exception_logger


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import DistributedTestBase, TEST_SKIPS
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


device_type = str(get_devtype())

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

WORLD_SIZE = min(4, max(2, torch.get_device_module(device_type).device_count()))


def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if torch.get_device_module(device_type).device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        self.create_pg(device_type)
        func(self)
        self.destroy_comms()

    return wrapper


class C10dErrorLoggerTest(DistributedTestBase):
    @property
    def world_size(self):
        return WORLD_SIZE

    @property
    def process_group(self):
        return dist.group.WORLD

    def destroy_comms(self):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def test_get_or_create_logger(self):
        self.assertIsNotNone(_c10d_logger)
        self.assertEqual(1, len(_c10d_logger.handlers))
        self.assertIsInstance(_c10d_logger.handlers[0], logging.NullHandler)

    @_exception_logger
    def _failed_broadcast_raise_exception(self):
        tensor = torch.arange(2, dtype=torch.int64)
        dist.broadcast(tensor, self.world_size + 1)

    @_exception_logger
    def _failed_broadcast_not_raise_exception(self):
        try:
            tensor = torch.arange(2, dtype=torch.int64)
            dist.broadcast(tensor, self.world_size + 1)
        except Exception:
            pass

    @with_comms
    def test_exception_logger(self) -> None:
        with self.assertRaises(Exception):
            self._failed_broadcast_raise_exception()

        with self.assertLogs(_c10d_logger, level="DEBUG") as captured:
            self._failed_broadcast_not_raise_exception()
            error_msg_dict = json.loads(
                re.search("({.+})", captured.output[0]).group(0).replace("'", '"')
            )

            # NCCL adds additional nccl_version data to the error_msg_dict
            if self.backend(device_type) == dist.Backend.NCCL:
                self.assertEqual(len(error_msg_dict), 9)
            else:
                self.assertEqual(len(error_msg_dict), 8)

            self.assertIn("pg_name", error_msg_dict.keys())
            self.assertEqual("None", error_msg_dict["pg_name"])

            self.assertIn("func_name", error_msg_dict.keys())
            self.assertEqual("broadcast", error_msg_dict["func_name"])

            self.assertIn("backend", error_msg_dict.keys())
            self.assertEqual(self.backend(device_type), error_msg_dict["backend"])

            if self.backend(device_type) == dist.Backend.NCCL:
                self.assertIn("nccl_version", error_msg_dict.keys())
                nccl_ver = torch.cuda.nccl.version()
                self.assertEqual(
                    ".".join(str(v) for v in nccl_ver), error_msg_dict["nccl_version"]
                )

            # In this test case, group_size = world_size, since we don't have multiple processes on one node.
            self.assertIn("group_size", error_msg_dict.keys())
            self.assertEqual(str(self.world_size), error_msg_dict["group_size"])

            self.assertIn("world_size", error_msg_dict.keys())
            self.assertEqual(str(self.world_size), error_msg_dict["world_size"])

            self.assertIn("global_rank", error_msg_dict.keys())
            self.assertIn(str(dist.get_rank()), error_msg_dict["global_rank"])

            # In this test case, local_rank = global_rank, since we don't have multiple processes on one node.
            self.assertIn("local_rank", error_msg_dict.keys())
            self.assertIn(str(dist.get_rank()), error_msg_dict["local_rank"])


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `C10dErrorLoggerTest`

**Functions defined**: `with_comms`, `wrapper`, `world_size`, `process_group`, `destroy_comms`, `test_get_or_create_logger`, `_failed_broadcast_raise_exception`, `_failed_broadcast_not_raise_exception`, `test_exception_logger`

**Key imports**: json, logging, re, sys, partial, wraps, torch, torch.distributed as dist, _c10d_logger, _exception_logger, DistributedTestBase, TEST_SKIPS, get_devtype


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `logging`
- `re`
- `sys`
- `functools`: partial, wraps
- `torch`
- `torch.distributed as dist`
- `torch.distributed.c10d_logger`: _c10d_logger, _exception_logger
- `torch.testing._internal.common_distributed`: DistributedTestBase, TEST_SKIPS
- `torch.testing._internal.common_fsdp`: get_devtype
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/distributed/test_c10d_logger.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed`):

- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`test_dist2.py_docs.md`](./test_dist2.py_docs.md)
- [`test_c10d_functional_native.py_docs.md`](./test_c10d_functional_native.py_docs.md)
- [`test_c10d_object_collectives.py_docs.md`](./test_c10d_object_collectives.py_docs.md)
- [`test_c10d_spawn_ucc.py_docs.md`](./test_c10d_spawn_ucc.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_c10d_logger.py_docs.md`
- **Keyword Index**: `test_c10d_logger.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python docs/test/distributed/test_c10d_logger.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed`):

- [`test_run.py_kw.md_docs.md`](./test_run.py_kw.md_docs.md)
- [`test_inductor_collectives.py_docs.md_docs.md`](./test_inductor_collectives.py_docs.md_docs.md)
- [`test_control_collectives.py_kw.md_docs.md`](./test_control_collectives.py_kw.md_docs.md)
- [`test_c10d_gloo.py_docs.md_docs.md`](./test_c10d_gloo.py_docs.md_docs.md)
- [`test_collective_utils.py_kw.md_docs.md`](./test_collective_utils.py_kw.md_docs.md)
- [`test_data_parallel.py_kw.md_docs.md`](./test_data_parallel.py_kw.md_docs.md)
- [`test_overlap_bucketing_unit.py_kw.md_docs.md`](./test_overlap_bucketing_unit.py_kw.md_docs.md)
- [`test_c10d_nccl.py_kw.md_docs.md`](./test_c10d_nccl.py_kw.md_docs.md)
- [`test_multi_threaded_pg.py_docs.md_docs.md`](./test_multi_threaded_pg.py_docs.md_docs.md)
- [`argparse_util_test.py_kw.md_docs.md`](./argparse_util_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_c10d_logger.py_docs.md_docs.md`
- **Keyword Index**: `test_c10d_logger.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
