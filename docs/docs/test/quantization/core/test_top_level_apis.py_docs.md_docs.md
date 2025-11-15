# Documentation: `docs/test/quantization/core/test_top_level_apis.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/core/test_top_level_apis.py_docs.md`
- **Size**: 6,431 bytes (6.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/core/test_top_level_apis.py`

## File Metadata

- **Path**: `test/quantization/core/test_top_level_apis.py`
- **Size**: 3,666 bytes (3.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

import torch
import torch.ao.quantization
from torch.testing._internal.common_utils import TestCase


class TestDefaultObservers(TestCase):
    observers = [
        "default_affine_fixed_qparams_observer",
        "default_debug_observer",
        "default_dynamic_quant_observer",
        "default_placeholder_observer",
        "default_fixed_qparams_range_0to1_observer",
        "default_fixed_qparams_range_neg1to1_observer",
        "default_float_qparams_observer",
        "default_float_qparams_observer_4bit",
        "default_histogram_observer",
        "default_observer",
        "default_per_channel_weight_observer",
        "default_reuse_input_observer",
        "default_symmetric_fixed_qparams_observer",
        "default_weight_observer",
        "per_channel_weight_observer_range_neg_127_to_127",
        "weight_observer_range_neg_127_to_127",
    ]

    fake_quants = [
        "default_affine_fixed_qparams_fake_quant",
        "default_dynamic_fake_quant",
        "default_embedding_fake_quant",
        "default_embedding_fake_quant_4bit",
        "default_fake_quant",
        "default_fixed_qparams_range_0to1_fake_quant",
        "default_fixed_qparams_range_neg1to1_fake_quant",
        "default_fused_act_fake_quant",
        "default_fused_per_channel_wt_fake_quant",
        "default_fused_wt_fake_quant",
        "default_histogram_fake_quant",
        "default_per_channel_weight_fake_quant",
        "default_symmetric_fixed_qparams_fake_quant",
        "default_weight_fake_quant",
        "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
        "fused_wt_fake_quant_range_neg_127_to_127",
    ]

    def _get_observer_ins(self, observer):
        obs_func = getattr(torch.ao.quantization, observer)
        return obs_func()

    def test_observers(self) -> None:
        t = torch.rand(1, 2, 3, 4)
        for observer in self.observers:
            obs = self._get_observer_ins(observer)
            obs.forward(t)

    def test_fake_quants(self) -> None:
        t = torch.rand(1, 2, 3, 4)
        for observer in self.fake_quants:
            obs = self._get_observer_ins(observer)
            obs.forward(t)


class TestQConfig(TestCase):

    REDUCE_RANGE_DICT = {
        'fbgemm': (True, False),
        'qnnpack': (False, False),
        'onednn': (False, False),
        'x86': (True, False),
    }

    def test_reduce_range_qat(self) -> None:
        for backend, reduce_ranges in self.REDUCE_RANGE_DICT.items():
            for version in range(2):
                qconfig = torch.ao.quantization.get_default_qat_qconfig(backend, version)

                fake_quantize_activ = qconfig.activation()
                self.assertEqual(fake_quantize_activ.activation_post_process.reduce_range, reduce_ranges[0])

                fake_quantize_weight = qconfig.weight()
                self.assertEqual(fake_quantize_weight.activation_post_process.reduce_range, reduce_ranges[1])

    def test_reduce_range(self) -> None:
        for backend, reduce_ranges in self.REDUCE_RANGE_DICT.items():
            for version in range(1):
                qconfig = torch.ao.quantization.get_default_qconfig(backend, version)

                fake_quantize_activ = qconfig.activation()
                self.assertEqual(fake_quantize_activ.reduce_range, reduce_ranges[0])

                fake_quantize_weight = qconfig.weight()
                self.assertEqual(fake_quantize_weight.reduce_range, reduce_ranges[1])

if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDefaultObservers`, `TestQConfig`

**Functions defined**: `_get_observer_ins`, `test_observers`, `test_fake_quants`, `test_reduce_range_qat`, `test_reduce_range`

**Key imports**: torch, torch.ao.quantization, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/core`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.quantization`
- `torch.testing._internal.common_utils`: TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/quantization/core/test_top_level_apis.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/core`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_quantized_module.py_docs.md`](./test_quantized_module.py_docs.md)
- [`test_backend_config.py_docs.md`](./test_backend_config.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_workflow_module.py_docs.md`](./test_workflow_module.py_docs.md)
- [`test_workflow_ops.py_docs.md`](./test_workflow_ops.py_docs.md)
- [`test_quantized_functional.py_docs.md`](./test_quantized_functional.py_docs.md)
- [`test_quantized_tensor.py_docs.md`](./test_quantized_tensor.py_docs.md)
- [`test_quantized_op.py_docs.md`](./test_quantized_op.py_docs.md)


## Cross-References

- **File Documentation**: `test_top_level_apis.py_docs.md`
- **Keyword Index**: `test_top_level_apis.py_kw.md`
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

*No specific patterns automatically detected.*


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
python docs/test/quantization/core/test_top_level_apis.py_docs.md
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
- [`test_quantized_module.py_docs.md_docs.md`](./test_quantized_module.py_docs.md_docs.md)
- [`test_quantized_functional.py_kw.md_docs.md`](./test_quantized_functional.py_kw.md_docs.md)
- [`test_utils.py_kw.md_docs.md`](./test_utils.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_top_level_apis.py_docs.md_docs.md`
- **Keyword Index**: `test_top_level_apis.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
