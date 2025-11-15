# Documentation: `docs/test/quantization/core/test_utils.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/core/test_utils.py_docs.md`
- **Size**: 12,061 bytes (11.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/core/test_utils.py`

## File Metadata

- **Path**: `test/quantization/core/test_utils.py`
- **Size**: 8,542 bytes (8.34 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase
from torch.ao.quantization.utils import get_fqn_to_example_inputs
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


class TestUtils(TestCase):
    def _test_get_fqn_to_example_inputs(self, M, example_inputs, expected_fqn_to_dim):
        m = M().eval()
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)
        for fqn, expected_dims in expected_fqn_to_dim.items():
            assert fqn in expected_fqn_to_dim
            example_inputs = fqn_to_example_inputs[fqn]
            for example_input, expected_dim in zip(example_inputs, expected_dims):
                assert example_input.dim() == expected_dim

    def test_get_fqn_to_example_inputs_simple(self):
        class Sub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x)
                return x

        expected_fqn_to_dim = {
            "": (2,),
            "linear1": (2,),
            "linear2": (2,),
            "sub": (2,),
            "sub.linear1": (2,),
            "sub.linear2": (2,)
        }
        example_inputs = (torch.rand(1, 5),)
        self._test_get_fqn_to_example_inputs(M, example_inputs, expected_fqn_to_dim)

    def test_get_fqn_to_example_inputs_default_kwargs(self):
        """ Test that we can get example inputs for functions with default keyword arguments
        """
        class Sub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x, key1=torch.rand(1), key2=torch.rand(1)):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                # only override `key2`, `key1` will use default
                x = self.sub(x, key2=torch.rand(1, 2))
                return x

        expected_fqn_to_dim = {
            "": (2,),
            "linear1": (2,),
            "linear2": (2,),
            # second arg is `key1`, which is using default argument
            # third arg is `key2`, override by callsite
            "sub": (2, 1, 2),
            "sub.linear1": (2,),
            "sub.linear2": (2,)
        }
        example_inputs = (torch.rand(1, 5),)
        self._test_get_fqn_to_example_inputs(M, example_inputs, expected_fqn_to_dim)

    def test_get_fqn_to_example_inputs_complex_args(self):
        """ Test that we can record complex example inputs such as lists and dicts
        """
        class Sub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x, list_arg, dict_arg):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x, [x], {"3": x})
                return x

        example_inputs = (torch.rand(1, 5),)
        m = M().eval()
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)
        assert "sub" in fqn_to_example_inputs
        assert isinstance(fqn_to_example_inputs["sub"][1], list)
        assert isinstance(fqn_to_example_inputs["sub"][2], dict) and \
            "3" in fqn_to_example_inputs["sub"][2]

    def test_quantize_weight_clamping_per_tensor(self):
        """ Test quant_{min, max} from per tensor observer is honored by `_quantize_weight` method
        """
        fp_min, fp_max = -1000.0, 1000.0
        q8_min, q8_max = -10, 10

        float_tensor = torch.tensor([fp_min, fp_max])

        observer = MovingAverageMinMaxObserver(
            averaging_constant=1.0,
            dtype=torch.qint8,
            quant_min=q8_min,
            quant_max=q8_max,
            qscheme=torch.per_tensor_symmetric,
        )

        observer(float_tensor)
        assert observer.min_val == fp_min
        assert observer.max_val == fp_max

        quantized_tensor = _quantize_weight(float_tensor, observer)
        assert quantized_tensor.int_repr().max().item() == q8_max
        assert quantized_tensor.int_repr().min().item() == q8_min

        # Actual weight values can be outside than observer [min_val, max_val] for the moving average observer
        float_tensor *= 1.2

        quantized_tensor = _quantize_weight(float_tensor, observer)
        assert quantized_tensor.int_repr().max().item() == q8_max
        assert quantized_tensor.int_repr().min().item() == q8_min

    def test_quantize_weight_clamping_per_channel(self):
        """ Test quant_{min, max} from per channel observer is honored by `_quantize_weight` method
        """
        fp_min, fp_max = -1000.0, 1000.0
        q8_min, q8_max = -10, 10

        float_tensor = torch.tensor([[fp_min, fp_max]])

        observer = MovingAveragePerChannelMinMaxObserver(
            averaging_constant=1.0,
            dtype=torch.qint8,
            quant_min=q8_min,
            quant_max=q8_max,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        observer(float_tensor)
        assert observer.min_val == fp_min
        assert observer.max_val == fp_max

        quantized_tensor = _quantize_weight(float_tensor, observer)
        assert quantized_tensor.int_repr().max().item() == q8_max
        assert quantized_tensor.int_repr().min().item() == q8_min

        # Actual weight values can be outside than observer [min_val, max_val] for the moving average observer
        float_tensor *= 1.2

        quantized_tensor = _quantize_weight(float_tensor, observer)
        assert quantized_tensor.int_repr().max().item() == q8_max
        assert quantized_tensor.int_repr().min().item() == q8_min

    def test_uint4_int4_dtype(self):

        def up_size(size):
            return (*size[:-1], size[-1] * 2)

        for dtype in [torch.uint4, torch.int4]:
            class UInt4OrInt4Tensor(torch.Tensor):
                @staticmethod
                def __new__(cls, elem, **kwargs):
                    assert elem.dtype is torch.uint8
                    assert not kwargs.get("requires_grad", False)
                    kwargs["requires_grad"] = False
                    return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=dtype, **kwargs)

                def __init__(self, elem):
                    self.elem = elem

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs=None):
                    pass

            # make sure it runs
            x = UInt4OrInt4Tensor(torch.tensor([
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            ], dtype=torch.uint8))
            assert x.dtype == dtype

if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")

```



## High-Level Overview


This Python file contains 8 class(es) and 23 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestUtils`, `Sub`, `M`, `Sub`, `M`, `Sub`, `M`, `UInt4OrInt4Tensor`

**Functions defined**: `_test_get_fqn_to_example_inputs`, `test_get_fqn_to_example_inputs_simple`, `__init__`, `forward`, `__init__`, `forward`, `test_get_fqn_to_example_inputs_default_kwargs`, `__init__`, `forward`, `__init__`, `forward`, `test_get_fqn_to_example_inputs_complex_args`, `__init__`, `forward`, `__init__`, `forward`, `test_quantize_weight_clamping_per_tensor`, `test_quantize_weight_clamping_per_channel`, `test_uint4_int4_dtype`, `up_size`

**Key imports**: torch, raise_on_run_directly, TestCase, get_fqn_to_example_inputs, _quantize_weight, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/core`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly, TestCase
- `torch.ao.quantization.utils`: get_fqn_to_example_inputs
- `torch.ao.nn.quantized.modules.utils`: _quantize_weight
- `torch.ao.quantization`: MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/quantization/core/test_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/core`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_quantized_module.py_docs.md`](./test_quantized_module.py_docs.md)
- [`test_backend_config.py_docs.md`](./test_backend_config.py_docs.md)
- [`test_workflow_module.py_docs.md`](./test_workflow_module.py_docs.md)
- [`test_workflow_ops.py_docs.md`](./test_workflow_ops.py_docs.md)
- [`test_quantized_functional.py_docs.md`](./test_quantized_functional.py_docs.md)
- [`test_quantized_tensor.py_docs.md`](./test_quantized_tensor.py_docs.md)
- [`test_quantized_op.py_docs.md`](./test_quantized_op.py_docs.md)
- [`test_top_level_apis.py_docs.md`](./test_top_level_apis.py_docs.md)


## Cross-References

- **File Documentation**: `test_utils.py_docs.md`
- **Keyword Index**: `test_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/quantization/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/core`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/quantization/core/test_utils.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/core`):

- [`test_quantized_op.py_kw.md_docs.md`](./test_quantized_op.py_kw.md_docs.md)
- [`test_workflow_module.py_kw.md_docs.md`](./test_workflow_module.py_kw.md_docs.md)
- [`test_quantized_tensor.py_kw.md_docs.md`](./test_quantized_tensor.py_kw.md_docs.md)
- [`test_backend_config.py_docs.md_docs.md`](./test_backend_config.py_docs.md_docs.md)
- [`test_workflow_module.py_docs.md_docs.md`](./test_workflow_module.py_docs.md_docs.md)
- [`test_top_level_apis.py_docs.md_docs.md`](./test_top_level_apis.py_docs.md_docs.md)
- [`test_quantized_module.py_docs.md_docs.md`](./test_quantized_module.py_docs.md_docs.md)
- [`test_quantized_functional.py_kw.md_docs.md`](./test_quantized_functional.py_kw.md_docs.md)
- [`test_utils.py_kw.md_docs.md`](./test_utils.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_utils.py_docs.md_docs.md`
- **Keyword Index**: `test_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
