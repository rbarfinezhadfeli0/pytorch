# Documentation: `docs/test/dynamo/test_callback.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_callback.py_docs.md`
- **Size**: 9,153 bytes (8.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_callback.py`

## File Metadata

- **Path**: `test/dynamo/test_callback.py`
- **Size**: 5,770 bytes (5.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import unittest
from unittest.mock import Mock

import torch
from torch._dynamo.callback import callback_handler, CallbackArgs, CallbackTrigger
from torch._dynamo.test_case import run_tests, TestCase
from torch._guards import CompileId
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.triton_utils import HAS_CUDA_AND_TRITON, requires_gpu


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class CallbackTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._on_compile_start = Mock()
        self._on_compile_end = Mock()
        callback_handler.register_start_callback(self._on_compile_start)
        callback_handler.register_end_callback(self._on_compile_end)

    def tearDown(self) -> None:
        callback_handler.clear()
        return super().tearDown()

    def test_callbacks_with_duplicate_prevention(self) -> None:
        trigger = CallbackTrigger.DYNAMO
        compile_id = CompileId(frame_id=0, frame_compile_id=0)
        with (
            callback_handler.install_callbacks(trigger, compile_id),
            callback_handler.install_callbacks(trigger, compile_id),
        ):
            self._on_compile_start.assert_called_once()
        self._on_compile_end.assert_called_once()

    def test_counter(self) -> None:
        trigger = CallbackTrigger.DYNAMO
        compile_id = CompileId(frame_id=0, frame_compile_id=0)
        with callback_handler.install_callbacks(trigger, compile_id):
            self.assertEqual(
                callback_handler._CompilationCallbackHandler__pending_callbacks_counter,
                1,
            )
        self.assertEqual(
            callback_handler._CompilationCallbackHandler__pending_callbacks_counter, 0
        )

    def test_counter_assertion(self) -> None:
        callback_handler._CompilationCallbackHandler__pending_callbacks_counter -= 1
        with self.assertRaisesRegex(
            AssertionError, "Pending callbacks counter cannot become negative."
        ):
            trigger = CallbackTrigger.DYNAMO
            compile_id = CompileId(frame_id=0, frame_compile_id=0)
            with callback_handler.install_callbacks(trigger, str(compile_id)):
                pass
        self.assertEqual(
            callback_handler._CompilationCallbackHandler__pending_callbacks_counter, 0
        )

    @unittest.skipIf(
        TEST_WITH_ROCM, "ROCm outputs a different number of autotuning logs"
    )
    @requires_gpu
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_triggers(self) -> None:
        torch._dynamo.reset()
        order = []

        def on_start(args: CallbackArgs):
            nonlocal order
            order.append(f"start={args}")

        def on_end(args: CallbackArgs):
            nonlocal order
            order.append(f"end={args}")

        torch._dynamo.callback.on_compile_start(on_start)
        torch._dynamo.callback.on_compile_start(on_end)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                temp = self.fc1(x)
                temp = self.relu(temp)
                torch._dynamo.graph_break()
                return self.fc2(temp)

        model = TinyModel().to(device_type)
        compiled_model = torch.compile(model, mode="max-autotune")
        x = torch.randn(10, 10, device=device_type)

        loss = compiled_model(x).sum()
        loss.backward()
        self.assertExpectedInline(
            "\n".join(order),
            """\
start=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='0/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='0/0')""",  # noqa: B950
        )
        order.clear()

        if not HAS_CUDA_AND_TRITON:
            return

        compiled_model.zero_grad()
        loss = compiled_model(x).sum()
        loss.backward()

        self.assertExpectedInline(
            "\n".join(order),
            """\
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')""",  # noqa: B950
        )
        order.clear()

        compiled_model.zero_grad()
        loss = compiled_model(x).sum()
        loss.backward()
        self.assertEqual(len(order), 0)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CallbackTests`, `TinyModel`

**Functions defined**: `setUp`, `tearDown`, `test_callbacks_with_duplicate_prevention`, `test_counter`, `test_counter_assertion`, `test_triggers`, `on_start`, `on_end`, `__init__`, `forward`

**Key imports**: unittest, Mock, torch, callback_handler, CallbackArgs, CallbackTrigger, run_tests, TestCase, CompileId, TEST_WITH_ROCM, HAS_CUDA_AND_TRITON, requires_gpu


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `unittest.mock`: Mock
- `torch`
- `torch._dynamo.callback`: callback_handler, CallbackArgs, CallbackTrigger
- `torch._dynamo.test_case`: run_tests, TestCase
- `torch._guards`: CompileId
- `torch.testing._internal.common_utils`: TEST_WITH_ROCM
- `torch.testing._internal.triton_utils`: HAS_CUDA_AND_TRITON, requires_gpu


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/test_callback.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)


## Cross-References

- **File Documentation**: `test_callback.py_docs.md`
- **Keyword Index**: `test_callback.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/dynamo/test_callback.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo`):

- [`test_error_messages.py_docs.md_docs.md`](./test_error_messages.py_docs.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_unittest.py_docs.md_docs.md`](./test_unittest.py_docs.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_aot_autograd.py_kw.md_docs.md`](./test_aot_autograd.py_kw.md_docs.md)
- [`test_einops.py_docs.md_docs.md`](./test_einops.py_docs.md_docs.md)
- [`test_compile.py_kw.md_docs.md`](./test_compile.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_buffers_override.py_kw.md_docs.md`](./test_buffers_override.py_kw.md_docs.md)
- [`test_frame_init.py_docs.md_docs.md`](./test_frame_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_callback.py_docs.md_docs.md`
- **Keyword Index**: `test_callback.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
