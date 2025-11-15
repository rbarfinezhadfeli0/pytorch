# Documentation: `benchmarks/gpt_fast/quantize.py`

## File Metadata

- **Path**: `benchmarks/gpt_fast/quantize.py`
- **Size**: 3,565 bytes (3.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
# flake8: noqa: E266, C417, B950
import torch
import torch.nn as nn
import torch.nn.functional as F


##### Quantization Primitives ######


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


##### Weight-only int8 per-channel quantized code ######


def replace_linear_weight_only_int8_per_channel(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                WeightOnlyInt8Linear(child.in_features, child.out_features),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8
                )
                cur_state_dict[f"{fqn}.weight"] = int8_weight.to("cpu")
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype).to("cpu")

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales

```



## High-Level Overview


This Python file contains 2 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WeightOnlyInt8QuantHandler`, `WeightOnlyInt8Linear`

**Functions defined**: `dynamically_quantize_per_channel`, `replace_linear_weight_only_int8_per_channel`, `__init__`, `create_quantized_state_dict`, `convert_for_runtime`, `__init__`, `forward`

**Key imports**: torch, torch.nn as nn, torch.nn.functional as F


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/gpt_fast`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/gpt_fast`):

- [`model.py_docs.md`](./model.py_docs.md)
- [`mixtral_moe_quantize.py_docs.md`](./mixtral_moe_quantize.py_docs.md)
- [`mixtral_moe_model.py_docs.md`](./mixtral_moe_model.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)
- [`generate.py_docs.md`](./generate.py_docs.md)
- [`benchmark.py_docs.md`](./benchmark.py_docs.md)


## Cross-References

- **File Documentation**: `quantize.py_docs.md`
- **Keyword Index**: `quantize.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
