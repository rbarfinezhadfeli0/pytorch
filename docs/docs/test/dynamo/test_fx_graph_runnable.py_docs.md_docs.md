# Documentation: `docs/test/dynamo/test_fx_graph_runnable.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_fx_graph_runnable.py_docs.md`
- **Size**: 17,251 bytes (16.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_fx_graph_runnable.py`

## File Metadata

- **Path**: `test/dynamo/test_fx_graph_runnable.py`
- **Size**: 13,132 bytes (12.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import io
import logging
import subprocess
import sys
import unittest

import torch
import torch._logging.structured
import torch.distributed as dist
from torch._inductor.codecache import WritableTempFile
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE
from torch.utils._triton import has_triton


if torch.distributed.is_available():
    from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
    from torch.testing._internal.distributed.fake_pg import FakeStore

if has_triton():
    import triton
    import triton.language as tl

    def init_to_zero(name):
        return lambda nargs: nargs[name].zero_()

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.atomic_add(output_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE": 1024},
                num_warps=4,
                num_stages=2,
                pre_hook=init_to_zero("output_ptr"),
            )
        ],
        pre_hook=init_to_zero("output_ptr"),
        post_hook=init_to_zero("output_ptr"),
        key=["n_elements"],
    )
    @triton.jit
    def add_kernel_autotune(
        x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.atomic_add(output_ptr + offsets, output, mask=mask)


from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu


class FxGraphRunnableArtifactFilter(logging.Filter):
    def filter(self, record):
        return (
            "artifact" in record.metadata
            and record.metadata["artifact"]["name"] == "fx_graph_runnable"
        )


class StructuredTracePayloadFormatter(logging.Formatter):
    def format(self, record):
        return record.payload.strip()


trace_log = logging.getLogger("torch.__trace")


class ToyModel(torch.nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
class FxGraphRunnableTest(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        # Create a custom filter specifically for fx_graph_runnable entries
        self.filter = FxGraphRunnableArtifactFilter()

        # Create a separate buffer and handler for capturing fx_graph_runnable entries
        self.buffer = io.StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTracePayloadFormatter())
        self.handler.addFilter(self.filter)
        trace_log.addHandler(self.handler)

    def tearDown(self):
        trace_log.removeHandler(self.handler)
        trace_log.setLevel(self.old_level)

    def _exec_and_verify_payload(self):
        # Write captured payload & run it in a fresh Python process
        payload = self.buffer.getvalue().strip()
        self.assertTrue(payload, "Expected fx_graph_runnable payload but got nothing")
        self.assertIn("def forward", payload)  # sanity-check for actual FX code

        with WritableTempFile("w", suffix=".py") as tmp:
            tmp.write(payload)
            tmp.flush()
            res = subprocess.run(
                [sys.executable, tmp.name], capture_output=True, text=True, timeout=30
            )

            self.assertEqual(
                res.returncode,
                0,
                f"Standalone fx_graph_runnable failed:\nSTDERR:\n{res.stderr}",
            )

    # basic tests
    def test_basic_tensor_add(self):
        def f(x):
            return x + 1

        torch.compile(f)(torch.randn(4))
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_user_defined_triton_kernel_autotune(self):
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = output.numel()

            def grid(
                meta,
            ):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_kernel_autotune[grid](x, y, output, n_elements)
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)

        torch.compile(add)(x, y)
        self._exec_and_verify_payload()

    @unittest.skipUnless(has_triton(), "Triton not available")
    @requires_gpu
    def test_user_defined_triton_kernel(self):
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = x.numel()
            add_kernel[n_elements,](x, y, output, n_elements, BLOCK_SIZE=4)
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)

        torch.compile(add)(x, y)
        self._exec_and_verify_payload()

    def test_two_inputs_matmul(self):
        def f(a, b):
            return (a @ b).relu()

        a, b = torch.randn(2, 3), torch.randn(3, 4)
        torch.compile(f)(a, b)
        self._exec_and_verify_payload()

    def test_scalar_multiply(self):
        def f(x):
            return x * 2

        torch.compile(f)(torch.randn(5))
        self._exec_and_verify_payload()

    # testing dynamic shapes
    def test_dynamic_shapes_run(self):
        def f(x):
            return (x @ x.transpose(0, 1)).relu()

        a = torch.randn(10, 12)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)

        torch.compile(f)(a)
        self._exec_and_verify_payload()

    def test_broadcast_add_dynamic(self):
        def f(x, y):
            return x + y * 2

        x = torch.randn(5, 1)
        y = torch.randn(1, 8)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 1)

        torch.compile(f)(x, y)
        self._exec_and_verify_payload()

    def test_toy_model_basic(self):
        model = ToyModel(input_size=8, hidden_size=16, output_size=4)
        model.eval()  # Set to eval mode to avoid dropout randomness

        x = torch.randn(3, 8)
        torch.compile(model)(x)
        self._exec_and_verify_payload()

    def test_toy_model_batch_processing(self):
        model = ToyModel(input_size=12, hidden_size=24, output_size=6)
        model.eval()

        x = torch.randn(16, 12)
        torch.compile(model)(x)
        self._exec_and_verify_payload()

    def test_toy_model_dynamic_batch(self):
        model = ToyModel(input_size=10, hidden_size=20, output_size=5)
        model.eval()

        x = torch.randn(7, 10)
        torch._dynamo.mark_dynamic(x, 0)

        torch.compile(model)(x)
        self._exec_and_verify_payload()

    # Distributed collectives tests with FakeProcessGroup
    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_all_reduce_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            dist.all_reduce(x)
            return x * 2

        try:
            x = torch.randn(4, 4)
            torch.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_all_gather_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            output_tensors = [torch.empty_like(x) for _ in range(2)]
            dist.all_gather(output_tensors, x)
            return output_tensors[0] + output_tensors[1]

        try:
            x = torch.randn(3, 3)
            torch.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_broadcast_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            dist.broadcast(x, src=0)
            return x.sum()

        try:
            x = torch.randn(5, 5)
            torch.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_reduce_scatter_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            input_list = [x, x.clone()]
            output = torch.empty_like(x)
            dist.reduce_scatter(output, input_list)
            return output

        try:
            x = torch.randn(4, 4)
            torch.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available"
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_dtensor_compile_redistribute(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        mesh = DeviceMesh("cpu", list(range(2)))

        def f(x, y):
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = torch.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out_redistribute.to_local()

        try:
            x = torch.arange(8, dtype=torch.float32)
            y = torch.arange(8, dtype=torch.float32)
            torch.compile(f)(x, y)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    def test_metrics_context(self):
        """
        When TORCH_COMPILE_DEBUG is set, provenance_tracking_level is set to 1, and
        the generated fx_graph_runnable crashed with,
        RuntimeError: Cannot add inductor_provenance outside of a MetricsContext
        """
        import torch._inductor.config as inductor_config

        def f(x):
            return x * 2 + 1

        # Enable provenance tracking to trigger the code path that adds metrics
        with inductor_config.patch(
            {"trace.enabled": True, "trace.provenance_tracking_level": 1}
        ):
            x = torch.randn(4, 4)
            torch.compile(f)(x)
            self._exec_and_verify_payload()

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic_expression(self):
        """
        Test not emitting something like "s27*s53**2 = 36"
        """

        def f(x):
            return torch.ops.aten._adaptive_avg_pool2d(
                x, (6, 6)
            ), torch.ops.aten._adaptive_avg_pool2d(x + 1, (2, 5))

        x = torch.randn(2, 4, 16, 16)
        torch.compile(f)(x)
        self._exec_and_verify_payload()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not (IS_FBCODE or IS_SANDCASTLE):
        # fbcode complains about not being able to find torch in subprocess
        run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 43 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FxGraphRunnableArtifactFilter`, `StructuredTracePayloadFormatter`, `ToyModel`, `FxGraphRunnableTest`

**Functions defined**: `init_to_zero`, `add_kernel`, `add_kernel_autotune`, `filter`, `format`, `__init__`, `forward`, `setUp`, `tearDown`, `_exec_and_verify_payload`, `test_basic_tensor_add`, `f`, `test_user_defined_triton_kernel_autotune`, `add`, `grid`, `test_user_defined_triton_kernel`, `add`, `test_two_inputs_matmul`, `f`, `test_scalar_multiply`

**Key imports**: io, logging, subprocess, sys, unittest, torch, torch._logging.structured, torch.distributed as dist, WritableTempFile, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `logging`
- `subprocess`
- `sys`
- `unittest`
- `torch`
- `torch._logging.structured`
- `torch.distributed as dist`
- `torch._inductor.codecache`: WritableTempFile
- `torch._inductor.test_case`: TestCase
- `torch.testing._internal.common_utils`: IS_FBCODE, IS_SANDCASTLE
- `torch.utils._triton`: has_triton
- `torch.distributed._tensor`: DeviceMesh, DTensor, Replicate, Shard
- `torch.testing._internal.distributed.fake_pg`: FakeStore
- `triton`
- `triton.language as tl`
- `torch.testing._internal.inductor_utils`: GPU_TYPE
- `torch.testing._internal.triton_utils`: requires_gpu
- `torch._inductor.config as inductor_config`
- `torch._dynamo.test_case`: run_tests


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/test_fx_graph_runnable.py
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
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_fx_graph_runnable.py_docs.md`
- **Keyword Index**: `test_fx_graph_runnable.py_kw.md`
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/dynamo/test_fx_graph_runnable.py_docs.md
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

- **File Documentation**: `test_fx_graph_runnable.py_docs.md_docs.md`
- **Keyword Index**: `test_fx_graph_runnable.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
