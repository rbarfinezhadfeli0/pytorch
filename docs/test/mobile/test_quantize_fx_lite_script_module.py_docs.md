# Documentation: `test/mobile/test_quantize_fx_lite_script_module.py`

## File Metadata

- **Path**: `test/mobile/test_quantize_fx_lite_script_module.py`
- **Size**: 3,121 bytes (3.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: mobile"]

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.utils.bundled_inputs
from torch.ao.quantization import default_qconfig, float_qparams_weight_only_qconfig

# graph mode quantization based on fx
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.testing._internal.common_quantization import (
    LinearModelWithSubmodule,
    NodeSpec as ns,
    QuantizationLiteTestCase,
)


class TestLiteFuseFx(QuantizationLiteTestCase):
    # Tests from:
    # ./caffe2/test/quantization/fx/test_quantize_fx.py

    def test_embedding(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                return self.emb(indices)

        model = M().eval()
        indices = torch.randint(low=0, high=10, size=(20,))

        ns.call_module(nnq.Embedding)
        configs = [
            (float_qparams_weight_only_qconfig, ns.call_module(nnq.Embedding)),
            (None, ns.call_module(nn.Embedding)),
            (default_qconfig, ns.call_module(nn.Embedding)),
        ]

        for qconfig, _ in configs:
            qconfig_dict = {"": qconfig}
            m = prepare_fx(
                model,
                qconfig_dict,
                example_inputs=torch.randint(low=0, high=10, size=(20,)),
            )
            m = convert_fx(m)
            self._compare_script_and_mobile(m, input=indices)

    def test_conv2d(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        m = M().eval()
        qconfig_dict = {"": default_qconfig, "module_name": [("conv1", None)]}
        m = prepare_fx(m, qconfig_dict, example_inputs=torch.randn(1, 1, 1, 1))
        data = torch.randn(1, 1, 1, 1)
        m = convert_fx(m)
        # first conv is quantized, second conv is not quantized
        self._compare_script_and_mobile(m, input=data)

    def test_submodule(self):
        # test quantizing complete module, submodule and linear layer
        configs = [
            {},
            {"module_name": [("subm", None)]},
            {"module_name": [("fc", None)]},
        ]
        for config in configs:
            model = LinearModelWithSubmodule().eval()
            qconfig_dict = {
                "": torch.ao.quantization.get_default_qconfig("qnnpack"),
                **config,
            }
            model = prepare_fx(
                model,
                qconfig_dict,
                example_inputs=torch.randn(5, 5),
            )
            quant = convert_fx(model)

            x = torch.randn(5, 5)
            self._compare_script_and_mobile(quant, input=x)


if __name__ == "__main__":
    run_tests()  # noqa: F821

```



## High-Level Overview


This Python file contains 3 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestLiteFuseFx`, `M`, `M`

**Functions defined**: `test_embedding`, `__init__`, `forward`, `test_conv2d`, `__init__`, `forward`, `test_submodule`

**Key imports**: torch, torch.ao.nn.quantized as nnq, torch.nn as nn, torch.utils.bundled_inputs, default_qconfig, float_qparams_weight_only_qconfig, convert_fx, prepare_fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/mobile`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.nn.quantized as nnq`
- `torch.nn as nn`
- `torch.utils.bundled_inputs`
- `torch.ao.quantization`: default_qconfig, float_qparams_weight_only_qconfig
- `torch.ao.quantization.quantize_fx`: convert_fx, prepare_fx


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
python test/mobile/test_quantize_fx_lite_script_module.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/mobile`):

- [`test_upgrader_codegen.py_docs.md`](./test_upgrader_codegen.py_docs.md)
- [`test_upgrader_bytecode_table_example.cpp_docs.md`](./test_upgrader_bytecode_table_example.cpp_docs.md)
- [`test_lite_script_module.py_docs.md`](./test_lite_script_module.py_docs.md)
- [`test_lite_script_type.py_docs.md`](./test_lite_script_type.py_docs.md)
- [`test_upgraders.py_docs.md`](./test_upgraders.py_docs.md)
- [`test_bytecode.py_docs.md`](./test_bytecode.py_docs.md)


## Cross-References

- **File Documentation**: `test_quantize_fx_lite_script_module.py_docs.md`
- **Keyword Index**: `test_quantize_fx_lite_script_module.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
