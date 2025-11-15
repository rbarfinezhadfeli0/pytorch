# Documentation: `test/distributed/elastic/timer/api_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/timer/api_test.py`
- **Size**: 2,442 bytes (2.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import unittest.mock as mock

from torch.distributed.elastic.timer import TimerServer
from torch.distributed.elastic.timer.api import RequestQueue, TimerRequest


class MockRequestQueue(RequestQueue):
    def size(self):
        return 2

    def get(self, size, timeout):
        return [TimerRequest(1, "test_1", 0), TimerRequest(2, "test_2", 0)]


class MockTimerServer(TimerServer):
    """
     Mock implementation of TimerServer for testing purposes.
     This mock has the following behavior:

     1. reaping worker 1 throws
     2. reaping worker 2 succeeds
     3. reaping worker 3 fails (caught exception)

    For each workers 1 - 3 returns 2 expired timers
    """

    def __init__(self, request_queue, max_interval):
        super().__init__(request_queue, max_interval)

    def register_timers(self, timer_requests):
        pass

    def clear_timers(self, worker_ids):
        pass

    def get_expired_timers(self, deadline):
        return {
            i: [TimerRequest(i, f"test_{i}_0", 0), TimerRequest(i, f"test_{i}_1", 0)]
            for i in range(1, 4)
        }

    def _reap_worker(self, worker_id):
        if worker_id == 1:
            raise RuntimeError("test error")
        elif worker_id == 2:
            return True
        elif worker_id == 3:
            return False


class TimerApiTest(unittest.TestCase):
    @mock.patch.object(MockTimerServer, "register_timers")
    @mock.patch.object(MockTimerServer, "clear_timers")
    def test_run_watchdog(self, mock_clear_timers, mock_register_timers):
        """
        tests that when a ``_reap_worker()`` method throws an exception
        for a particular worker_id, the timers for successfully reaped workers
        are cleared properly
        """
        max_interval = 1
        request_queue = mock.Mock(wraps=MockRequestQueue())
        timer_server = MockTimerServer(request_queue, max_interval)
        timer_server._run_watchdog()

        request_queue.size.assert_called_once()
        request_queue.get.assert_called_with(request_queue.size(), max_interval)
        mock_register_timers.assert_called_with(request_queue.get(2, 1))
        mock_clear_timers.assert_called_with({1, 2})

```



## High-Level Overview

"""     Mock implementation of TimerServer for testing purposes.     This mock has the following behavior:     1. reaping worker 1 throws     2. reaping worker 2 succeeds     3. reaping worker 3 fails (caught exception)    For each workers 1 - 3 returns 2 expired timers

This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MockRequestQueue`, `MockTimerServer`, `TimerApiTest`

**Functions defined**: `size`, `get`, `__init__`, `register_timers`, `clear_timers`, `get_expired_timers`, `_reap_worker`, `test_run_watchdog`

**Key imports**: unittest, unittest.mock as mock, TimerServer, RequestQueue, TimerRequest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/timer`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `unittest.mock as mock`
- `torch.distributed.elastic.timer`: TimerServer
- `torch.distributed.elastic.timer.api`: RequestQueue, TimerRequest


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/distributed/elastic/timer/api_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/timer`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`local_timer_example.py_docs.md`](./local_timer_example.py_docs.md)
- [`local_timer_test.py_docs.md`](./local_timer_test.py_docs.md)
- [`file_based_local_timer_test.py_docs.md`](./file_based_local_timer_test.py_docs.md)


## Cross-References

- **File Documentation**: `api_test.py_docs.md`
- **Keyword Index**: `api_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
