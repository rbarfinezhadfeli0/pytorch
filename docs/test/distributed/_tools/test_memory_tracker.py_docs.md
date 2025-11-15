# Documentation: `test/distributed/_tools/test_memory_tracker.py`

## File Metadata

- **Path**: `test/distributed/_tools/test_memory_tracker.py`
- **Size**: 2,452 bytes (2.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import os
import unittest

import torch
import torch.nn as nn
from torch.distributed._tools import MemoryTracker
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMemoryTracker(TestCase):
    @unittest.skipIf(not torch.accelerator.is_available(), "no accelerator")
    def test_local_model(self):
        """
        Minimal test case to check the memory tracker can collect the expected
        memory stats at operator level, as well as can print the summary result
        without crash.
        """
        device = torch.accelerator.current_accelerator()
        # Create a model with a hierarchy of modules
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(nn.Linear(64, 2), nn.ReLU(inplace=True)),
        ).to(device)

        # Run one iteration of forward and backward pass
        tracker = MemoryTracker()
        tracker.start_monitor(model)

        x = torch.randn(size=(2, 3, 224, 224), device=device)
        # torch.LongTensor expects cpu device type, not gpu device type in
        # constructor, so calling .to() outside constructor here.
        target = torch.LongTensor([0, 1]).to(device)
        criterion = nn.CrossEntropyLoss()
        criterion(model(x), target).backward()

        self.assertTrue(len(tracker._hooks) > 0)

        tracker.stop()

        self.assertTrue(len(tracker._hooks) == 0)

        path = "memory.trace"
        tracker.save_stats(path)
        tracker.load(path)
        tracker.summary()
        if os.path.exists(path):
            os.remove(path)

        self.assertTrue(tracker._op_index > 0)
        self.assertTrue(len(tracker._operator_names) > 0)
        self.assertEqual(len(tracker.memories_allocated), tracker._op_index)
        self.assertEqual(len(tracker.memories_active), tracker._op_index)
        self.assertEqual(len(tracker.memories_reserved), tracker._op_index)
        self.assertTrue(len(tracker._markers) == 2)
        self.assertTrue(tracker._cur_module_name != "")
        self.assertTrue(hasattr(tracker, "_num_alloc_retries"))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""        Minimal test case to check the memory tracker can collect the expected        memory stats at operator level, as well as can print the summary result        without crash.

This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestMemoryTracker`

**Functions defined**: `test_local_model`

**Key imports**: os, unittest, torch, torch.nn as nn, MemoryTracker, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_tools`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest`
- `torch`
- `torch.nn as nn`
- `torch.distributed._tools`: MemoryTracker
- `torch.testing._internal.common_utils`: run_tests, TestCase


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
python test/distributed/_tools/test_memory_tracker.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_tools`):

- [`test_fsdp2_mem_tracker.py_docs.md`](./test_fsdp2_mem_tracker.py_docs.md)
- [`test_runtime_estimator.py_docs.md`](./test_runtime_estimator.py_docs.md)
- [`test_mod_tracker.py_docs.md`](./test_mod_tracker.py_docs.md)
- [`test_sac_estimator.py_docs.md`](./test_sac_estimator.py_docs.md)
- [`test_sac_ilp.py_docs.md`](./test_sac_ilp.py_docs.md)
- [`test_mem_tracker.py_docs.md`](./test_mem_tracker.py_docs.md)
- [`test_fake_collectives.py_docs.md`](./test_fake_collectives.py_docs.md)


## Cross-References

- **File Documentation**: `test_memory_tracker.py_docs.md`
- **Keyword Index**: `test_memory_tracker.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
