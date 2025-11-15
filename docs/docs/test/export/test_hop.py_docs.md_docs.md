# Documentation: `docs/test/export/test_hop.py_docs.md`

## File Metadata

- **Path**: `docs/test/export/test_hop.py_docs.md`
- **Size**: 8,456 bytes (8.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/export/test_hop.py`

## File Metadata

- **Path**: `test/export/test_hop.py`
- **Size**: 5,200 bytes (5.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import io
import unittest

import torch
import torch._dynamo as torchdynamo
import torch.utils._pytree as pytree
from torch._dynamo.test_case import TestCase
from torch.export import export, load, save
from torch.export._trace import _export
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests
from torch.testing._internal.hop_db import (
    FIXME_hop_that_doesnt_have_opinfo_test_allowlist,
    hop_db,
)


hop_tests = []

for op_info in hop_db:
    op_info_hop_name = op_info.name
    if op_info_hop_name in FIXME_hop_that_doesnt_have_opinfo_test_allowlist:
        continue
    hop_tests.append(op_info)


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestHOP(TestCase):
    def _compare(self, eager_model, export, args, kwargs):
        eager_args = copy.deepcopy(args)
        eager_kwargs = copy.deepcopy(kwargs)
        export_args = copy.deepcopy(args)
        export_kwargs = copy.deepcopy(kwargs)

        flat_orig_outputs = pytree.tree_leaves(eager_model(*eager_args, **eager_kwargs))
        flat_loaded_outputs = pytree.tree_leaves(
            export.module()(*export_args, **export_kwargs)
        )

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
            self.assertEqual(type(orig), type(loaded))
            self.assertEqual(orig, loaded)

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_aot_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = export(model, args, kwargs, strict=True)
            self._compare(model, ep, args, kwargs)
        # With PYTORCH_TEST_CUDA_MEM_LEAK_CHECK=1, a memory leak occurs during
        # strict-mode export. We need to manually reset the cache of backends.
        # Specifically, `cached_backends.clear()` is required.
        # Upon examining the items in `cached_backends`,
        # we notice that under strict-mode export, there exists
        # the `dynamo_normalization_capturing_compiler`, which must be
        # cleared to avoid memory leaks. An educated guess is that
        # the `dynamo_normalization_capturing_compiler` references input tensors
        # on CUDA devices and fails to free them.
        torchdynamo._reset_guarded_backend_cache()

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_pre_dispatch_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            self._compare(model, ep, args, kwargs)
        torchdynamo._reset_guarded_backend_cache()

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_retrace_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            ep = ep.run_decompositions()
            self._compare(model, ep, args, kwargs)
        torchdynamo._reset_guarded_backend_cache()

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_serialize_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            ep = ep.run_decompositions()
            buffer = io.BytesIO()
            save(ep, buffer)
            buffer.seek(0)
            ep = load(buffer)
            self._compare(model, ep, args, kwargs)
        torchdynamo._reset_guarded_backend_cache()


instantiate_device_type_tests(TestHOP, globals())

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestHOP`, `Foo`, `Foo`, `Foo`, `Foo`

**Functions defined**: `_compare`, `test_aot_export`, `forward`, `test_pre_dispatch_export`, `forward`, `test_retrace_export`, `forward`, `test_serialize_export`, `forward`

**Key imports**: copy, io, unittest, torch, torch._dynamo as torchdynamo, torch.utils._pytree as pytree, TestCase, export, load, save, _export, IS_WINDOWS, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `io`
- `unittest`
- `torch`
- `torch._dynamo as torchdynamo`
- `torch.utils._pytree as pytree`
- `torch._dynamo.test_case`: TestCase
- `torch.export`: export, load, save
- `torch.export._trace`: _export
- `torch.testing._internal.common_utils`: IS_WINDOWS, run_tests


## Code Patterns & Idioms

### Common Patterns

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
python test/export/test_hop.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `test_hop.py_docs.md`
- **Keyword Index**: `test_hop.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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
python docs/test/export/test_hop.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/export`):

- [`test_serialize.py_docs.md_docs.md`](./test_serialize.py_docs.md_docs.md)
- [`test_verifier.py_kw.md_docs.md`](./test_verifier.py_kw.md_docs.md)
- [`test_upgrader.py_kw.md_docs.md`](./test_upgrader.py_kw.md_docs.md)
- [`test_db.py_docs.md_docs.md`](./test_db.py_docs.md_docs.md)
- [`test_export.py_docs.md_docs.md`](./test_export.py_docs.md_docs.md)
- [`test_dynamic_shapes.py_kw.md_docs.md`](./test_dynamic_shapes.py_kw.md_docs.md)
- [`test_passes.py_kw.md_docs.md`](./test_passes.py_kw.md_docs.md)
- [`test_unflatten.py_docs.md_docs.md`](./test_unflatten.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_functionalized_assertions.py_kw.md_docs.md`](./test_functionalized_assertions.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_hop.py_docs.md_docs.md`
- **Keyword Index**: `test_hop.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
