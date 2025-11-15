# Documentation: `test/test_monitor.py`

## File Metadata

- **Path**: `test/test_monitor.py`
- **Size**: 5,013 bytes (4.90 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: r2p"]

import sys
import tempfile
import time
import unittest

from datetime import datetime, timedelta

from torch.monitor import (
    _WaitCounter,
    Aggregation,
    Event,
    log_event,
    register_event_handler,
    Stat,
    TensorboardEventHandler,
    unregister_event_handler,
)
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestMonitor(TestCase):
    def test_interval_stat(self) -> None:
        events = []

        def handler(event):
            events.append(event)

        handle = register_event_handler(handler)
        s = Stat(
            "asdf",
            (Aggregation.SUM, Aggregation.COUNT),
            timedelta(milliseconds=1),
        )
        self.assertEqual(s.name, "asdf")

        s.add(2)
        for _ in range(100):
            # NOTE: different platforms sleep may be inaccurate so we loop
            # instead (i.e. win)
            time.sleep(1 / 1000)  # ms
            s.add(3)
            if len(events) >= 1:
                break
        self.assertGreaterEqual(len(events), 1)
        unregister_event_handler(handle)

    def test_fixed_count_stat(self) -> None:
        s = Stat(
            "asdf",
            (Aggregation.SUM, Aggregation.COUNT),
            timedelta(hours=100),
            3,
        )
        s.add(1)
        s.add(2)
        name = s.name
        self.assertEqual(name, "asdf")
        self.assertEqual(s.count, 2)
        s.add(3)
        self.assertEqual(s.count, 0)
        self.assertEqual(s.get(), {Aggregation.SUM: 6.0, Aggregation.COUNT: 3})

    def test_log_event(self) -> None:
        e = Event(
            name="torch.monitor.TestEvent",
            timestamp=datetime.now(),
            data={
                "str": "a string",
                "float": 1234.0,
                "int": 1234,
            },
        )
        self.assertEqual(e.name, "torch.monitor.TestEvent")
        self.assertIsNotNone(e.timestamp)
        self.assertIsNotNone(e.data)
        log_event(e)

    @skipIfTorchDynamo("Really weird error")
    def test_event_handler(self) -> None:
        events = []

        def handler(event: Event) -> None:
            events.append(event)

        handle = register_event_handler(handler)
        e = Event(
            name="torch.monitor.TestEvent",
            timestamp=datetime.now(),
            data={},
        )
        log_event(e)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], e)
        log_event(e)
        self.assertEqual(len(events), 2)

        unregister_event_handler(handle)
        log_event(e)
        self.assertEqual(len(events), 2)

    def test_wait_counter(self) -> None:
        wait_counter = _WaitCounter(
            "test_wait_counter",
        )
        with wait_counter.guard():
            pass


@skipIfTorchDynamo("Really weird error")
class TestMonitorTensorboard(TestCase):
    def setUp(self):
        super().setUp()
        global SummaryWriter, event_multiplexer
        try:
            from tensorboard.backend.event_processing import (
                plugin_event_multiplexer as event_multiplexer,
            )
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return self.skipTest("Skip the test since TensorBoard is not installed")
        self.temp_dirs = []

    def create_summary_writer(self):
        temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
        self.temp_dirs.append(temp_dir)
        return SummaryWriter(temp_dir.name)

    def tearDown(self):
        # Remove directories created by SummaryWriter
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()

    @unittest.skipIf(
        sys.version_info >= (3, 13),
        "numpy failure, likely caused by old tensorboard version",
    )
    def test_event_handler(self):
        with self.create_summary_writer() as w:
            handle = register_event_handler(TensorboardEventHandler(w))

            s = Stat(
                "asdf",
                (Aggregation.SUM, Aggregation.COUNT),
                timedelta(hours=1),
                5,
            )
            for i in range(10):
                s.add(i)
            self.assertEqual(s.count, 0)

            unregister_event_handler(handle)

        mul = event_multiplexer.EventMultiplexer()
        mul.AddRunsFromDirectory(self.temp_dirs[-1].name)
        mul.Reload()
        scalar_dict = mul.PluginRunToTagToContent("scalars")
        raw_result = {
            tag: mul.Tensors(run, tag)
            for run, run_dict in scalar_dict.items()
            for tag in run_dict
        }
        scalars = {
            tag: [e.tensor_proto.float_val[0] for e in events]
            for tag, events in raw_result.items()
        }
        self.assertEqual(
            scalars,
            {
                "asdf.sum": [10],
                "asdf.count": [5],
            },
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestMonitor`, `TestMonitorTensorboard`

**Functions defined**: `test_interval_stat`, `handler`, `test_fixed_count_stat`, `test_log_event`, `test_event_handler`, `handler`, `test_wait_counter`, `setUp`, `create_summary_writer`, `tearDown`, `test_event_handler`

**Key imports**: sys, tempfile, time, unittest, datetime, timedelta, run_tests, skipIfTorchDynamo, TestCase, SummaryWriter


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `tempfile`
- `time`
- `unittest`
- `datetime`: datetime, timedelta
- `torch.testing._internal.common_utils`: run_tests, skipIfTorchDynamo, TestCase
- `torch.utils.tensorboard`: SummaryWriter


## Code Patterns & Idioms

### Common Patterns

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
python test/test_monitor.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_monitor.py_docs.md`
- **Keyword Index**: `test_monitor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
