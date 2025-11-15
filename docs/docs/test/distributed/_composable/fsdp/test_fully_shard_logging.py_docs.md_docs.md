# Documentation: `docs/test/distributed/_composable/fsdp/test_fully_shard_logging.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/_composable/fsdp/test_fully_shard_logging.py_docs.md`
- **Size**: 6,104 bytes (5.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/_composable/fsdp/test_fully_shard_logging.py`

## File Metadata

- **Path**: `test/distributed/_composable/fsdp/test_fully_shard_logging.py`
- **Size**: 2,209 bytes (2.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fsdp"]
import functools
import os
import unittest

import torch.distributed as dist
from torch._dynamo.test_case import run_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.logging_utils import LoggingTestCase


requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)

import torch
from torch.testing._internal.common_fsdp import get_devtype


device_type = torch.device(get_devtype())


@skip_if_lt_x_gpu(2)
class LoggingTests(LoggingTestCase):
    @requires_distributed()
    def test_fsdp_logging(self):
        env = dict(os.environ)
        env["TORCH_LOGS"] = "fsdp"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_PORT"] = "34715"
        env["MASTER_ADDR"] = "localhost"
        _, stderr = self.run_process_no_exception(
            f"""\
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
logger = logging.getLogger("torch.distributed.fsdp.fully_shard")
logger.setLevel(logging.DEBUG)
device = '{device_type.type}'
torch.manual_seed(0)
model = nn.Sequential(*[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)])
for layer in model:
    fully_shard(layer)
fully_shard(model)
x = torch.randn((4, 4), device=device)
model(x).sum().backward()
""",
            env=env,
        )
        self.assertIn("FSDP::root_pre_forward", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_forward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_forward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_forward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_forward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_backward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::pre_backward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_backward (0)", stderr.decode("utf-8"))
        self.assertIn("FSDP::post_backward (1)", stderr.decode("utf-8"))
        self.assertIn("FSDP::root_post_backward", stderr.decode("utf-8"))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

f"""\import loggingimport torchimport torch.distributed as distimport torch.nn as nnfrom torch.distributed.fsdp import fully_shardlogger = logging.getLogger("torch.distributed.fsdp.fully_shard")logger.setLevel(logging.DEBUG)device = '{device_type.type}'torch.manual_seed(0)model = nn.Sequential(*[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)])for layer in model:    fully_shard(layer)fully_shard(model)x = torch.randn((4, 4), device=device)model(x).sum().backward()

This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LoggingTests`

**Functions defined**: `test_fsdp_logging`

**Key imports**: functools, os, unittest, torch.distributed as dist, run_tests, skip_if_lt_x_gpu, LoggingTestCase, torch, get_devtype, logging


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_composable/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `os`
- `unittest`
- `torch.distributed as dist`
- `torch._dynamo.test_case`: run_tests
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.logging_utils`: LoggingTestCase
- `torch`
- `torch.testing._internal.common_fsdp`: get_devtype
- `logging`
- `torch.nn as nn`
- `torch.distributed.fsdp`: fully_shard


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
python test/distributed/_composable/fsdp/test_fully_shard_logging.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_composable/fsdp`):

- [`test_fully_shard_extensions.py_docs.md`](./test_fully_shard_extensions.py_docs.md)
- [`test_fully_shard_mixed_precision.py_docs.md`](./test_fully_shard_mixed_precision.py_docs.md)
- [`test_fully_shard_ignore_params.py_docs.md`](./test_fully_shard_ignore_params.py_docs.md)
- [`test_fully_shard_frozen.py_docs.md`](./test_fully_shard_frozen.py_docs.md)
- [`test_fully_shard_clip_grad_norm_.py_docs.md`](./test_fully_shard_clip_grad_norm_.py_docs.md)
- [`test_fully_shard_state.py_docs.md`](./test_fully_shard_state.py_docs.md)
- [`test_fully_shard_overlap.py_docs.md`](./test_fully_shard_overlap.py_docs.md)
- [`test_fully_shard_state_dict.py_docs.md`](./test_fully_shard_state_dict.py_docs.md)
- [`test_fully_shard_init.py_docs.md`](./test_fully_shard_init.py_docs.md)


## Cross-References

- **File Documentation**: `test_fully_shard_logging.py_docs.md`
- **Keyword Index**: `test_fully_shard_logging.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/_composable/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/_composable/fsdp`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/_composable/fsdp/test_fully_shard_logging.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_composable/fsdp`):

- [`test_fully_shard_clip_grad_norm_.py_docs.md_docs.md`](./test_fully_shard_clip_grad_norm_.py_docs.md_docs.md)
- [`test_fully_shard_autograd.py_kw.md_docs.md`](./test_fully_shard_autograd.py_kw.md_docs.md)
- [`test_fully_shard_ignore_params.py_kw.md_docs.md`](./test_fully_shard_ignore_params.py_kw.md_docs.md)
- [`test_fully_shard_comm.py_docs.md_docs.md`](./test_fully_shard_comm.py_docs.md_docs.md)
- [`test_fully_shard_state.py_docs.md_docs.md`](./test_fully_shard_state.py_docs.md_docs.md)
- [`test_fully_shard_ignore_params.py_docs.md_docs.md`](./test_fully_shard_ignore_params.py_docs.md_docs.md)
- [`test_fully_shard_clip_grad_norm_.py_kw.md_docs.md`](./test_fully_shard_clip_grad_norm_.py_kw.md_docs.md)
- [`test_fully_shard_state.py_kw.md_docs.md`](./test_fully_shard_state.py_kw.md_docs.md)
- [`test_fully_shard_mixed_precision.py_docs.md_docs.md`](./test_fully_shard_mixed_precision.py_docs.md_docs.md)
- [`test_fully_shard_state_dict.py_kw.md_docs.md`](./test_fully_shard_state_dict.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fully_shard_logging.py_docs.md_docs.md`
- **Keyword Index**: `test_fully_shard_logging.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
