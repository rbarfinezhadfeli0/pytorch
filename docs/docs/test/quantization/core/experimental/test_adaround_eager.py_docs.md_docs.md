# Documentation: `docs/test/quantization/core/experimental/test_adaround_eager.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/core/experimental/test_adaround_eager.py_docs.md`
- **Size**: 8,199 bytes (8.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/core/experimental/test_adaround_eager.py`

## File Metadata

- **Path**: `test/quantization/core/experimental/test_adaround_eager.py`
- **Size**: 4,968 bytes (4.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: speech_infra"]

import copy

import torch
import torch.nn as nn
from torch.ao.quantization.experimental.adaround_optimization import (
    AdaptiveRoundingOptimizer,
)
from torch.nn import functional as F
from torch.quantization.observer import MinMaxObserver
from torch.testing._internal.common_quantization import QuantizationTestCase


def forward_wrapper(fetcher):
    def forward(module, input, output):
        fetcher.append(input[0].detach())
        fetcher.append(output.detach())

    return forward


class TestAdaround(QuantizationTestCase):
    def feedforawrd_callback(
        self,
        model,
        data,
    ) -> None:
        model(data)

    def feedforawrd_callback_with_wrapper(self, model, data, wrapper) -> None:
        wrapper(model, data)

    def run_adaround(self, model, img_data, wrapper=None):
        adaround_optimizer = AdaptiveRoundingOptimizer(
            model,
            self.feedforawrd_callback
            if wrapper is None
            else self.feedforawrd_callback_with_wrapper,
            forward_wrapper,
            img_data,
            max_iter=100,
            batch_size=10,
            feed_forward_wrapper=wrapper,
        )
        adarounded_model = adaround_optimizer.run_adaround()
        return adarounded_model

    def get_fake_quant(self, model):
        hard_fake_quant_model = copy.deepcopy(model)
        for _, module in hard_fake_quant_model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                weight_observer = MinMaxObserver(
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                )
                weight_observer(module.weight)
                scale, zero_point = weight_observer.calculate_qparams()
                fake_quant_module = torch.fake_quantize_per_tensor_affine(
                    module.weight,
                    scale=scale,
                    zero_point=zero_point,
                    quant_min=-128,
                    quant_max=127,
                )
                module.weight.data.copy_(fake_quant_module)
        return hard_fake_quant_model

    def get_feed_forward_wrapper(self):
        class FeedForwardWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, model, sample):
                return model(sample)

        wrapper_module = FeedForwardWrapper()
        return wrapper_module

    def test_linear_chain(self):
        class LinearChain(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        float_model = LinearChain()
        img_data = [torch.rand(10, 3, dtype=torch.float) for _ in range(50)]
        adarounded_model = self.run_adaround(
            float_model, img_data, self.get_feed_forward_wrapper()
        )
        fq_model = self.get_fake_quant(float_model)
        rand_input = torch.rand(10, 3)
        with torch.no_grad():
            ada_out = adarounded_model(rand_input)
            fq_out = fq_model(rand_input)
            float_out = float_model(rand_input)
            ada_loss = F.mse_loss(ada_out, float_out)
            fq_loss = F.mse_loss(fq_out, float_out)
            self.assertTrue(ada_loss.item() < fq_loss.item())

    def test_conv_chain(self):
        class ConvChain(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
                self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
                self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

            def forward(self, x):
                x = self.conv2d1(x)
                x = self.conv2d2(x)
                x = self.conv2d3(x)
                return x

        float_model = ConvChain()
        img_data = [torch.rand(10, 3, 125, 125, dtype=torch.float) for _ in range(50)]
        adarounded_model = self.run_adaround(float_model, img_data)
        fq_model = self.get_fake_quant(float_model)
        rand_input = torch.rand(10, 3, 256, 256)
        with torch.no_grad():
            ada_out = adarounded_model(rand_input)
            fq_out = fq_model(rand_input)
            float_out = float_model(rand_input)
            ada_loss = F.mse_loss(ada_out, float_out)
            fq_loss = F.mse_loss(fq_out, float_out)
            self.assertTrue(ada_loss.item() < fq_loss.item())


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 4 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAdaround`, `FeedForwardWrapper`, `LinearChain`, `ConvChain`

**Functions defined**: `forward_wrapper`, `forward`, `feedforawrd_callback`, `feedforawrd_callback_with_wrapper`, `run_adaround`, `get_fake_quant`, `get_feed_forward_wrapper`, `__init__`, `forward`, `test_linear_chain`, `__init__`, `forward`, `test_conv_chain`, `__init__`, `forward`

**Key imports**: copy, torch, torch.nn as nn, functional as F, MinMaxObserver, QuantizationTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/core/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch`
- `torch.nn as nn`
- `torch.nn`: functional as F
- `torch.quantization.observer`: MinMaxObserver
- `torch.testing._internal.common_quantization`: QuantizationTestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/quantization/core/experimental/test_adaround_eager.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/core/experimental`):

- [`test_fake_quantize.py_docs.md`](./test_fake_quantize.py_docs.md)
- [`test_floatx.py_docs.md`](./test_floatx.py_docs.md)
- [`test_quantizer.py_docs.md`](./test_quantizer.py_docs.md)
- [`test_bits.py_docs.md`](./test_bits.py_docs.md)
- [`apot_fx_graph_mode_qat.py_docs.md`](./apot_fx_graph_mode_qat.py_docs.md)
- [`apot_fx_graph_mode_ptq.py_docs.md`](./apot_fx_graph_mode_ptq.py_docs.md)
- [`test_quantized_tensor.py_docs.md`](./test_quantized_tensor.py_docs.md)
- [`test_nonuniform_observer.py_docs.md`](./test_nonuniform_observer.py_docs.md)
- [`test_linear.py_docs.md`](./test_linear.py_docs.md)


## Cross-References

- **File Documentation**: `test_adaround_eager.py_docs.md`
- **Keyword Index**: `test_adaround_eager.py_kw.md`
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
- **Neural Network**: Defines or uses PyTorch neural network components


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
python docs/test/quantization/core/experimental/test_adaround_eager.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/core/experimental`):

- [`test_bits.py_docs.md_docs.md`](./test_bits.py_docs.md_docs.md)
- [`test_quantizer.py_docs.md_docs.md`](./test_quantizer.py_docs.md_docs.md)
- [`apot_fx_graph_mode_qat.py_kw.md_docs.md`](./apot_fx_graph_mode_qat.py_kw.md_docs.md)
- [`test_quantized_tensor.py_kw.md_docs.md`](./test_quantized_tensor.py_kw.md_docs.md)
- [`apot_fx_graph_mode_ptq.py_kw.md_docs.md`](./apot_fx_graph_mode_ptq.py_kw.md_docs.md)
- [`test_fake_quantize.py_kw.md_docs.md`](./test_fake_quantize.py_kw.md_docs.md)
- [`test_nonuniform_observer.py_kw.md_docs.md`](./test_nonuniform_observer.py_kw.md_docs.md)
- [`apot_fx_graph_mode_qat.py_docs.md_docs.md`](./apot_fx_graph_mode_qat.py_docs.md_docs.md)
- [`test_floatx.py_docs.md_docs.md`](./test_floatx.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_adaround_eager.py_docs.md_docs.md`
- **Keyword Index**: `test_adaround_eager.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
