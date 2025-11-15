# Documentation: `docs/test/quantization/eager/test_model_numerics.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/eager/test_model_numerics.py_docs.md`
- **Size**: 10,230 bytes (9.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/eager/test_model_numerics.py`

## File Metadata

- **Path**: `test/quantization/eager/test_model_numerics.py`
- **Size**: 7,618 bytes (7.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_quantization import (
    ModelMultipleOps,
    ModelMultipleOpsNoAvgPool,
    QuantizationTestCase,
)
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
)


class TestModelNumericsEager(QuantizationTestCase):
    def test_float_quant_compare_per_tensor(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(42)
                my_model = ModelMultipleOps().to(torch.float32)
                my_model.eval()
                calib_data = torch.rand(1024, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(1, 3, 15, 15, dtype=torch.float32)
                out_ref = my_model(eval_data)
                qModel = torch.ao.quantization.QuantWrapper(my_model)
                qModel.eval()
                qModel.qconfig = torch.ao.quantization.default_qconfig
                torch.ao.quantization.fuse_modules(
                    qModel.module, [["conv1", "bn1", "relu1"]], inplace=True
                )
                torch.ao.quantization.prepare(qModel, inplace=True)
                qModel(calib_data)
                torch.ao.quantization.convert(qModel, inplace=True)
                out_q = qModel(eval_data)
                SQNRdB = 20 * torch.log10(
                    torch.norm(out_ref) / torch.norm(out_ref - out_q)
                )
                # Quantized model output should be close to floating point model output numerically
                # Setting target SQNR to be 30 dB so that relative error is 1e-3 below the desired
                # output
                self.assertGreater(
                    SQNRdB,
                    30,
                    msg="Quantized model numerics diverge from float, expect SQNR > 30 dB",
                )

    def test_float_quant_compare_per_channel(self):
        # Test for per-channel Quant
        torch.manual_seed(67)
        my_model = ModelMultipleOps().to(torch.float32)
        my_model.eval()
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
        out_ref = my_model(eval_data)
        q_model = torch.ao.quantization.QuantWrapper(my_model)
        q_model.eval()
        q_model.qconfig = torch.ao.quantization.default_per_channel_qconfig
        torch.ao.quantization.fuse_modules(
            q_model.module, [["conv1", "bn1", "relu1"]], inplace=True
        )
        torch.ao.quantization.prepare(q_model)
        q_model(calib_data)
        torch.ao.quantization.convert(q_model)
        out_q = q_model(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 35 dB
        self.assertGreater(
            SQNRdB,
            35,
            msg="Quantized model numerics diverge from float, expect SQNR > 35 dB",
        )

    def test_fake_quant_true_quant_compare(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(67)
                my_model = ModelMultipleOpsNoAvgPool().to(torch.float32)
                calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
                my_model.eval()
                out_ref = my_model(eval_data)
                fq_model = torch.ao.quantization.QuantWrapper(my_model)
                fq_model.train()
                fq_model.qconfig = torch.ao.quantization.default_qat_qconfig
                torch.ao.quantization.fuse_modules_qat(
                    fq_model.module, [["conv1", "bn1", "relu1"]], inplace=True
                )
                torch.ao.quantization.prepare_qat(fq_model)
                fq_model.eval()
                fq_model.apply(torch.ao.quantization.disable_fake_quant)
                fq_model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                fq_model(calib_data)
                fq_model.apply(torch.ao.quantization.enable_fake_quant)
                fq_model.apply(torch.ao.quantization.disable_observer)
                out_fq = fq_model(eval_data)
                SQNRdB = 20 * torch.log10(
                    torch.norm(out_ref) / torch.norm(out_ref - out_fq)
                )
                # Quantized model output should be close to floating point model output numerically
                # Setting target SQNR to be 35 dB
                self.assertGreater(
                    SQNRdB,
                    35,
                    msg="Quantized model numerics diverge from float, expect SQNR > 35 dB",
                )
                torch.ao.quantization.convert(fq_model)
                out_q = fq_model(eval_data)
                SQNRdB = 20 * torch.log10(
                    torch.norm(out_fq) / (torch.norm(out_fq - out_q) + 1e-10)
                )
                self.assertGreater(
                    SQNRdB,
                    60,
                    msg="Fake quant and true quant numerics diverge, expect SQNR > 60 dB",
                )

    # Test to compare weight only quantized model numerics and
    # activation only quantized model numerics with float
    def test_weight_only_activation_only_fakequant(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(67)
                calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
                qconfigset = {
                    torch.ao.quantization.default_weight_only_qconfig,
                    torch.ao.quantization.default_activation_only_qconfig,
                }
                SQNRTarget = [35, 45]
                for idx, qconfig in enumerate(qconfigset):
                    my_model = ModelMultipleOpsNoAvgPool().to(torch.float32)
                    my_model.eval()
                    out_ref = my_model(eval_data)
                    fq_model = torch.ao.quantization.QuantWrapper(my_model)
                    fq_model.train()
                    fq_model.qconfig = qconfig
                    torch.ao.quantization.fuse_modules_qat(
                        fq_model.module, [["conv1", "bn1", "relu1"]], inplace=True
                    )
                    torch.ao.quantization.prepare_qat(fq_model)
                    fq_model.eval()
                    fq_model.apply(torch.ao.quantization.disable_fake_quant)
                    fq_model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                    fq_model(calib_data)
                    fq_model.apply(torch.ao.quantization.enable_fake_quant)
                    fq_model.apply(torch.ao.quantization.disable_observer)
                    out_fq = fq_model(eval_data)
                    SQNRdB = 20 * torch.log10(
                        torch.norm(out_ref) / torch.norm(out_ref - out_fq)
                    )
                    self.assertGreater(
                        SQNRdB,
                        SQNRTarget[idx],
                        msg="Quantized model numerics diverge from float",
                    )


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_quantization.py TESTNAME\n\n"
        "instead."
    )

```



## High-Level Overview


This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestModelNumericsEager`

**Functions defined**: `test_float_quant_compare_per_tensor`, `test_float_quant_compare_per_channel`, `test_fake_quant_true_quant_compare`, `test_weight_only_activation_only_fakequant`

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/eager`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/quantization/eager/test_model_numerics.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/eager`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_quantize_eager_ptq.py_docs.md`](./test_quantize_eager_ptq.py_docs.md)
- [`test_quantize_eager_qat.py_docs.md`](./test_quantize_eager_qat.py_docs.md)
- [`test_equalize_eager.py_docs.md`](./test_equalize_eager.py_docs.md)
- [`test_bias_correction_eager.py_docs.md`](./test_bias_correction_eager.py_docs.md)
- [`test_fuse_eager.py_docs.md`](./test_fuse_eager.py_docs.md)
- [`test_numeric_suite_eager.py_docs.md`](./test_numeric_suite_eager.py_docs.md)


## Cross-References

- **File Documentation**: `test_model_numerics.py_docs.md`
- **Keyword Index**: `test_model_numerics.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/quantization/eager`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/eager`, which is part of the **testing infrastructure**.



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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/quantization/eager/test_model_numerics.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/eager`):

- [`test_numeric_suite_eager.py_kw.md_docs.md`](./test_numeric_suite_eager.py_kw.md_docs.md)
- [`test_equalize_eager.py_docs.md_docs.md`](./test_equalize_eager.py_docs.md_docs.md)
- [`test_fuse_eager.py_docs.md_docs.md`](./test_fuse_eager.py_docs.md_docs.md)
- [`test_numeric_suite_eager.py_docs.md_docs.md`](./test_numeric_suite_eager.py_docs.md_docs.md)
- [`test_quantize_eager_qat.py_docs.md_docs.md`](./test_quantize_eager_qat.py_docs.md_docs.md)
- [`test_quantize_eager_ptq.py_docs.md_docs.md`](./test_quantize_eager_ptq.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`test_bias_correction_eager.py_docs.md_docs.md`](./test_bias_correction_eager.py_docs.md_docs.md)
- [`test_equalize_eager.py_kw.md_docs.md`](./test_equalize_eager.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_model_numerics.py_docs.md_docs.md`
- **Keyword Index**: `test_model_numerics.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
