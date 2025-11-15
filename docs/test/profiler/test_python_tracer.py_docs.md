# Documentation: `test/profiler/test_python_tracer.py`

## File Metadata

- **Path**: `test/profiler/test_python_tracer.py`
- **Size**: 3,018 bytes (2.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: profiler"]

import json
import subprocess
import sys
import time

from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfPythonVersionMismatch,
    TemporaryFileName,
    TestCase,
)


class TestPythonTracer(TestCase):
    @skipIfPythonVersionMismatch(lambda major, minor, micro: major == 3 and minor == 12)
    def test_method_with_c_function(self):
        class A:
            method_with_c_function = classmethod(repr)

        def get_key(x):
            A().method_with_c_function()
            time.sleep(1.2)
            return len(x)

        names = ["Alice", "Bob"]

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
        ) as prof:
            sorted(names, key=get_key)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]
                found = False
                for event in events:
                    if (
                        event.get("cat", "") == "python_function"
                        and event.get("name", "") == "<built-in function sorted>"
                    ):
                        duration = event.get("dur", 0)
                        if duration >= 2000000:
                            found = True
                            break
                self.assertTrue(found)

    @skipIfPythonVersionMismatch(lambda major, minor, micro: major == 3 and minor == 12)
    def test_monitoring_callback(self):
        vi = sys.version_info
        from sys import monitoring

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
        ):
            name = monitoring.get_tool(2)
            if vi.micro < 5:
                self.assertEqual(name, "PyTorch Profiler")
            else:
                self.assertEqual(name, None)
        name = monitoring.get_tool(2)
        self.assertEqual(name, None)

    def test_unexpected_c_return_events(self):
        code = """
import threading
import time
import torch

from threading import Event, Lock

lock = Lock()
lock.acquire()
event1 = Event()
event2 = Event()
event3 = Event()

def run():
    event1.set()
    event2.wait()
    lock.acquire()
    event3.set()

threading.Thread(target=run).start()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True):
    event1.wait()
    event2.set()
    time.sleep(1)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True):
    lock.release()
    event3.wait()
    """

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        self.assertFalse(
            "Python replay stack is empty during pop operation" in result.stderr
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPythonTracer`, `A`

**Functions defined**: `test_method_with_c_function`, `get_key`, `test_monitoring_callback`, `test_unexpected_c_return_events`, `run`

**Key imports**: json, subprocess, sys, time, profile, ProfilerActivity, monitoring, threading, time, torch, Event, Lock


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/profiler`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `subprocess`
- `sys`
- `time`
- `torch.profiler`: profile, ProfilerActivity
- `threading`
- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/profiler/test_python_tracer.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/profiler`):

- [`profiler_utils_mock_events.json_docs.md`](./profiler_utils_mock_events.json_docs.md)
- [`test_memory_profiler.py_docs.md`](./test_memory_profiler.py_docs.md)
- [`test_cpp_thread.cpp_docs.md`](./test_cpp_thread.cpp_docs.md)
- [`test_execution_trace.py_docs.md`](./test_execution_trace.py_docs.md)
- [`test_record_function.py_docs.md`](./test_record_function.py_docs.md)
- [`test_torch_tidy.py_docs.md`](./test_torch_tidy.py_docs.md)
- [`test_cpp_thread_lib.pyi_docs.md`](./test_cpp_thread_lib.pyi_docs.md)
- [`test_profiler_tree.py_docs.md`](./test_profiler_tree.py_docs.md)
- [`test_cpp_thread.py_docs.md`](./test_cpp_thread.py_docs.md)


## Cross-References

- **File Documentation**: `test_python_tracer.py_docs.md`
- **Keyword Index**: `test_python_tracer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
