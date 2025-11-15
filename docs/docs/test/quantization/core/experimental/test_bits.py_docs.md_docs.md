# Documentation: `docs/test/quantization/core/experimental/test_bits.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/core/experimental/test_bits.py_docs.md`
- **Size**: 6,548 bytes (6.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/core/experimental/test_bits.py`

## File Metadata

- **Path**: `test/quantization/core/experimental/test_bits.py`
- **Size**: 3,538 bytes (3.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests

from torch.testing._internal.common_utils import run_tests, TestCase, skipIfRocm
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_map

import itertools

class Int16Tensor(torch.Tensor):
    def __new__(cls, elem):
        assert elem.dtype == torch.bits16
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __init__(self, elem):
        super().__init__()

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, torch.Tensor):
                with no_dispatch():
                    return t.view(torch.int16)
            return t
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        with no_dispatch():
            out = func(*args, **kwargs)

        def wrap(t):
            if isinstance(t, torch.Tensor):
                with no_dispatch():
                    return t.view(torch.bits16)
            return t
        out = tree_map(wrap, out)
        return out

    # This most likely should be removed (and thus use the disabled impl)
    # but the test below fail under Dynamo in that case.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)

    def __repr__(self) -> str:
        with no_dispatch():
            self.view(torch.int16)
            return f"TensorSubclassDemo{self.view(torch.int16)}"


class TestBits(TestCase):
    @skipIfRocm
    def test_types(self, device):
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        for bits_type in bits_types:
            _ = torch.zeros(20, dtype=torch.int32, device=device).view(bits_type)
            _ = torch.empty(20, dtype=bits_type, device=device)
            x = torch.randint(100, (20, 20), dtype=torch.int8, device=device).view(bits_type)
            y = x.t().contiguous()
            view_type = torch.int8 if x.element_size() == 1 else torch.int16
            self.assertEqual(x.t().view(view_type), y.view(view_type))
            y = x.t().clone()
            self.assertEqual(x.t().view(view_type), y.view(view_type))

    def test_cat(self, device):
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        for bits_type in bits_types:
            view_type = torch.int8 if bits_type.itemsize == 1 else torch.int16
            x_int = torch.randint(100, (512, 512), dtype=view_type, device=device)
            x = x_int.view(bits_type)
            y_int = torch.randint(100, (512, 512), dtype=view_type, device=device)
            y = y_int.view(bits_type)
            for dim, transpose in itertools.product(range(x_int.ndim), (True, False)):
                y_ref = y_int.t() if transpose else y_int
                y_b = y.t() if transpose else y
                z_ref = torch.cat([x_int, y_ref], dim=dim)
                z = torch.cat([x, y_b], dim=dim)
                self.assertEqual(z_ref, z.view(view_type))


    def test_subclass(self):
        t = torch.zeros(20, dtype=torch.int16).view(torch.bits16)
        s = Int16Tensor(t)
        s = s + 1 - 1
        self.assertTrue(torch.allclose(s, torch.zeros(20, dtype=torch.bits16)))

instantiate_device_type_tests(TestBits, globals())


if __name__ == '__main__':
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Int16Tensor`, `TestBits`

**Functions defined**: `__new__`, `__init__`, `__torch_dispatch__`, `unwrap`, `wrap`, `__torch_function__`, `__repr__`, `test_types`, `test_cat`, `test_subclass`

**Key imports**: torch, instantiate_device_type_tests, run_tests, TestCase, skipIfRocm, no_dispatch, tree_map, itertools


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/core/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_utils`: run_tests, TestCase, skipIfRocm
- `torch.utils._mode_utils`: no_dispatch
- `torch.utils._pytree`: tree_map
- `itertools`


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
python test/quantization/core/experimental/test_bits.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/core/experimental`):

- [`test_adaround_eager.py_docs.md`](./test_adaround_eager.py_docs.md)
- [`test_fake_quantize.py_docs.md`](./test_fake_quantize.py_docs.md)
- [`test_floatx.py_docs.md`](./test_floatx.py_docs.md)
- [`test_quantizer.py_docs.md`](./test_quantizer.py_docs.md)
- [`apot_fx_graph_mode_qat.py_docs.md`](./apot_fx_graph_mode_qat.py_docs.md)
- [`apot_fx_graph_mode_ptq.py_docs.md`](./apot_fx_graph_mode_ptq.py_docs.md)
- [`test_quantized_tensor.py_docs.md`](./test_quantized_tensor.py_docs.md)
- [`test_nonuniform_observer.py_docs.md`](./test_nonuniform_observer.py_docs.md)
- [`test_linear.py_docs.md`](./test_linear.py_docs.md)


## Cross-References

- **File Documentation**: `test_bits.py_docs.md`
- **Keyword Index**: `test_bits.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/quantization/core/experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/core/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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
python docs/test/quantization/core/experimental/test_bits.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/core/experimental`):

- [`test_quantizer.py_docs.md_docs.md`](./test_quantizer.py_docs.md_docs.md)
- [`test_adaround_eager.py_docs.md_docs.md`](./test_adaround_eager.py_docs.md_docs.md)
- [`apot_fx_graph_mode_qat.py_kw.md_docs.md`](./apot_fx_graph_mode_qat.py_kw.md_docs.md)
- [`test_quantized_tensor.py_kw.md_docs.md`](./test_quantized_tensor.py_kw.md_docs.md)
- [`apot_fx_graph_mode_ptq.py_kw.md_docs.md`](./apot_fx_graph_mode_ptq.py_kw.md_docs.md)
- [`test_fake_quantize.py_kw.md_docs.md`](./test_fake_quantize.py_kw.md_docs.md)
- [`test_nonuniform_observer.py_kw.md_docs.md`](./test_nonuniform_observer.py_kw.md_docs.md)
- [`apot_fx_graph_mode_qat.py_docs.md_docs.md`](./apot_fx_graph_mode_qat.py_docs.md_docs.md)
- [`test_floatx.py_docs.md_docs.md`](./test_floatx.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_bits.py_docs.md_docs.md`
- **Keyword Index**: `test_bits.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
