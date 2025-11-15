# Documentation: `test/profiler/test_record_function.py`

## File Metadata

- **Path**: `test/profiler/test_record_function.py`
- **Size**: 8,324 bytes (8.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: profiler"]
# ruff: noqa: F841

from typing import Any

import torch
import torch.optim
import torch.utils.data
import torch.utils.data.datapipes as dp
from torch._dispatch.python import enable_python_dispatcher
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.autograd.profiler import profile as _profile
from torch.profiler import kineto_available, record_function
from torch.testing._internal.common_utils import run_tests, TestCase


# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

Json = dict[str, Any]


class TestRecordFunction(TestCase):
    def _record_function_with_param(self):
        u = torch.randn(3, 4, 5, requires_grad=True)
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            with record_function("## TEST 1 ##", "1, 2, 3"):
                rf_handle = _record_function_with_args_enter(
                    "## TEST 2 ##", 1, False, 2.5, [u, u], "hello", u
                )
                _record_function_with_args_exit(rf_handle)
            with record_function("## TEST 3 ##"):
                rf_handle = _record_function_with_args_enter("## TEST 4 ##")
                _record_function_with_args_exit(rf_handle)
        return prof

    def test_record_function(self):
        prof_result = self._record_function_with_param()
        found_test_1 = False
        found_test_2 = False
        found_test_3 = False
        found_test_4 = False
        for e in prof_result.function_events:
            if "## TEST 1 ##" == e.name:
                found_test_1 = True
                self.assertTrue(e.input_shapes == [[]])
            elif "## TEST 2 ##" == e.name:
                found_test_2 = True
                self.assertTrue(e.input_shapes == [[], [], [], [], [], [3, 4, 5]])
            elif "## TEST 3 ##" == e.name:
                found_test_3 = True
                self.assertTrue(e.input_shapes == [])
            elif "## TEST 4 ##" == e.name:
                found_test_4 = True
                self.assertTrue(e.input_shapes == [])
        self.assertTrue(found_test_1)
        self.assertTrue(found_test_2)
        self.assertTrue(found_test_3)
        self.assertTrue(found_test_4)

    def test_datapipe_with_record_function(self):
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            input_dp1 = dp.iter.IterableWrapper(range(4))
            input_dp2 = dp.iter.IterableWrapper(range(4, 8))
            input_dp3 = dp.iter.IterableWrapper(range(8, 12))
            output_dp = input_dp1.mux(input_dp2, input_dp3)
            output = list(output_dp)

        has_iter = False
        has_mux = False
        for e in prof.function_events:
            if has_iter and has_mux:
                break

            if not has_iter and "IterableWrapper" in e.name:
                has_iter = True
            if not has_mux and "Multiplexer" in e.name:
                has_mux = True
        self.assertTrue(has_iter)
        self.assertTrue(has_mux)

    def test_datapipe_delegation_with_profiler(self):
        class IDPIterator(torch.utils.data.IterDataPipe):
            def __init__(self) -> None:
                self.data = list(range(10))
                self._idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._idx >= 10:
                    self._idx = 0
                    raise StopIteration
                self._idx += 1
                return self.data[self._idx - 1]

            def get_value(self, idx):
                return self.data[idx]

        dp1 = IDPIterator()  # The object itself is an iterator
        self.assertEqual(5, dp1.get_value(5))
        it_dp1 = iter(dp1)  # This creates the 1st iterator
        self.assertEqual(5, it_dp1.get_value(5))  # type: ignore[attr-defined]
        self.assertEqual(list(range(10)), list(it_dp1))

        class IDPDelegator(torch.utils.data.IterDataPipe):
            def __init__(self, datapipe):
                self.datapipe = datapipe

            def __iter__(self):
                return iter(self.datapipe)

        dp2 = IDPDelegator(dp1)
        it_dp2 = iter(dp2)
        self.assertEqual(5, it_dp2.get_value(5))
        self.assertEqual(list(range(10)), list(it_dp2))

    def test_datapipe_with_record_function_fork(self):
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            input_dp = dp.iter.IterableWrapper(range(10))
            dp1, dp2, dp3 = input_dp.fork(num_instances=3)
            output1 = list(dp1)
        has_iter = False
        has_child = False
        for e in prof.function_events:
            if has_iter and has_child:
                break

            if not has_iter and "IterableWrapper" in e.name:
                has_iter = True
            if not has_child and "_ChildDataPipe" in e.name:
                has_child = True
        self.assertTrue(has_iter)
        self.assertTrue(has_child)

    def test_python_dispatch_mode_record_function(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class TestDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with _profile() as prof:
            with enable_python_dispatcher():
                with TestDispatchMode():
                    x = torch.randn(3, 4)
                    y = torch.sin(x)

        found_python_dispatch_mode = False
        for e in prof.function_events:
            if e.name == "PythonDispatchMode":
                found_python_dispatch_mode = True
                break
        self.assertTrue(
            found_python_dispatch_mode,
            "PythonDispatchMode record function not found in profiler events",
        )

    def test_python_subclass_record_function(self):
        class TestTensorSubclass(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                def unwrap(x):
                    return x.elem if isinstance(x, TestTensorSubclass) else x

                def wrap(x):
                    return TestTensorSubclass(x) if isinstance(x, torch.Tensor) else x

                unwrapped_args = tuple(unwrap(arg) for arg in args)
                unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
                result = func(*unwrapped_args, **unwrapped_kwargs)

                if isinstance(result, torch.Tensor):
                    return TestTensorSubclass(result)
                return result

        with _profile() as prof:
            with enable_python_dispatcher():
                x = TestTensorSubclass(torch.randn(3, 4))
                y = torch.sin(x)

        found_python_subclass = False
        for e in prof.function_events:
            if e.name == "PythonSubclass":
                found_python_subclass = True
                break
        self.assertTrue(
            found_python_subclass,
            "PythonSubclass record function not found in profiler events",
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 6 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestRecordFunction`, `IDPIterator`, `IDPDelegator`, `TestDispatchMode`, `TestTensorSubclass`

**Functions defined**: `_record_function_with_param`, `test_record_function`, `test_datapipe_with_record_function`, `test_datapipe_delegation_with_profiler`, `__init__`, `__iter__`, `__next__`, `get_value`, `__init__`, `__iter__`, `test_datapipe_with_record_function_fork`, `test_python_dispatch_mode_record_function`, `__torch_dispatch__`, `test_python_subclass_record_function`, `__new__`, `__torch_dispatch__`, `unwrap`, `wrap`

**Key imports**: Any, torch, torch.optim, torch.utils.data, torch.utils.data.datapipes as dp, enable_python_dispatcher, profile as _profile, kineto_available, record_function, run_tests, TestCase, tqdm


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/profiler`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch`
- `torch.optim`
- `torch.utils.data`
- `torch.utils.data.datapipes as dp`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch.autograd.profiler`: profile as _profile
- `torch.profiler`: kineto_available, record_function
- `torch.testing._internal.common_utils`: run_tests, TestCase
- `tqdm`
- `torch.utils._python_dispatch`: TorchDispatchMode


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/profiler/test_record_function.py
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
- [`test_python_tracer.py_docs.md`](./test_python_tracer.py_docs.md)
- [`test_torch_tidy.py_docs.md`](./test_torch_tidy.py_docs.md)
- [`test_cpp_thread_lib.pyi_docs.md`](./test_cpp_thread_lib.pyi_docs.md)
- [`test_profiler_tree.py_docs.md`](./test_profiler_tree.py_docs.md)
- [`test_cpp_thread.py_docs.md`](./test_cpp_thread.py_docs.md)


## Cross-References

- **File Documentation**: `test_record_function.py_docs.md`
- **Keyword Index**: `test_record_function.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
