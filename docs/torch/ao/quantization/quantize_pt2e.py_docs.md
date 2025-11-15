# Documentation: `torch/ao/quantization/quantize_pt2e.py`

## File Metadata

- **Path**: `torch/ao/quantization/quantize_pt2e.py`
- **Size**: 9,476 bytes (9.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import typing_extensions

import torch
from torch._export.passes.constant_folding import constant_fold
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch.ao.quantization.quantizer import (  # noqa: F401
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    QuantizationSpecBase,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassManager

from .pt2e.prepare import prepare
from .pt2e.qat_utils import _fold_conv_bn_qat, _fuse_conv_bn_qat
from .pt2e.representation import reference_representation_rewrite
from .pt2e.utils import _disallow_eval_train, _fuse_conv_bn_, _get_node_name_to_scope
from .quantize_fx import _convert_to_reference_decomposed_fx
from .utils import DEPRECATION_WARNING


__all__ = [
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "convert_pt2e",
]


@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """Prepare a model for post training quantization

    Args:
      * `model` (torch.fx.GraphModule): a model captured by `torch.export.export_for_training` API.
      * `quantizer`: A backend specific quantizer that conveys how user want the
        model to be quantized. Tutorial for how to write a quantizer can be found here:
        https://pytorch.org/tutorials/prototype/pt2e_quantizer.html

    Return:
      A GraphModule with observer (based on quantizer annotation), ready for calibration

    Example::

        import torch
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e
        from torch.ao.quantization.quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

           def forward(self, x):
               return self.linear(x)

        # initialize a floating point model
        float_model = M().eval()

        # define calibration function
        def calibrate(model, data_loader):
            model.eval()
            with torch.no_grad():
                for image, target in data_loader:
                    model(image)

        # Step 1. program capture
        # NOTE: this API will be updated to torch.export API in the future, but the captured
        # result should mostly stay the same
        m = torch.export.export_for_training(m, *example_inputs).module()
        # we get a model with aten ops

        # Step 2. quantization
        # backend developer will write their own Quantizer and expose methods to allow
        # users to express how they
        # want the model to be quantized
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        m = prepare_pt2e(m, quantizer)

        # run calibration
        # calibrate(m, sample_inference_data)
    """
    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.prepare_pt2e")
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    # TODO: check qconfig_mapping to make sure conv and bn are both configured
    # to be quantized before fusion
    # TODO: (maybe) rewrite this with subgraph_rewriter
    _fuse_conv_bn_(model)
    model = quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    model = prepare(
        model,
        node_name_to_scope,
        is_qat=False,
        obs_or_fq_callback=quantizer.prepare_obs_or_fq_callback,
    )
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model


@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare_qat_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """Prepare a model for quantization aware training

    Args:
      * `model` (torch.fx.GraphModule): see :func:`~torch.ao.quantization.quantize_pt2e.prepare_pt2e`
      * `quantizer`: see :func:`~torch.ao.quantization.quantize_pt2e.prepare_pt2e`

    Return:
      A GraphModule with fake quant modules (based on quantizer annotation), ready for
      quantization aware training

    Example::
        import torch
        from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
        from torch.ao.quantization.quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

           def forward(self, x):
               return self.linear(x)

        # initialize a floating point model
        float_model = M().eval()

        # define the training loop for quantization aware training
        def train_loop(model, train_data):
            model.train()
            for image, target in data_loader:
                ...

        # Step 1. program capture
        # NOTE: this API will be updated to torch.export API in the future, but the captured
        # result should mostly stay the same
        m = torch.export.export_for_training(m, *example_inputs).module()
        # we get a model with aten ops

        # Step 2. quantization
        # backend developer will write their own Quantizer and expose methods to allow
        # users to express how they
        # want the model to be quantized
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        m = prepare_qat_pt2e(m, quantizer)

        # run quantization aware training
        train_loop(prepared_model, train_loop)

    """
    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.prepare_qat_pt2e")
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    model = quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    # Perform fusion after annotate to avoid quantizing ops in the new
    # subgraph that don't need to be quantized
    # TODO: only fuse if conv and bn are both configured to be quantized
    _fuse_conv_bn_qat(model)
    model = prepare(
        model,
        node_name_to_scope,
        is_qat=True,
        obs_or_fq_callback=quantizer.prepare_obs_or_fq_callback,
    )
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model


_QUANT_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
    torch.ops.pt2e_quant.quantize_affine,
]


def _quant_node_constraint(n: Node) -> bool:
    """If there is any pure ops between get_attr and quantize op they will be const propagated
    e.g. get_attr(weight) -> transpose -> quantize -> dequantize*
    (Note: dequantize op is not going to be constant propagated)

    This filter is added because we don't want to constant fold the things that are not
    related to quantization
    """
    return n.op == "call_function" and n.target in _QUANT_OPS


@typing_extensions.deprecated(DEPRECATION_WARNING)
def convert_pt2e(
    model: GraphModule,
    use_reference_representation: bool = False,
    fold_quantize: bool = True,
) -> GraphModule:
    """Convert a calibrated/trained model to a quantized model

    Args:
      * `model` (torch.fx.GraphModule): calibrated/trained model
      * `use_reference_representation` (bool): boolean flag to indicate whether to produce reference representation or not
      * `fold_quantize` (bool): boolean flag for whether fold the quantize op or not

    Returns:
        quantized model, either in q/dq representation or reference representation

    Example::

        # prepared_model: the model produced by `prepare_pt2e`/`prepare_qat_pt2e` and calibration/training
        # `convert_pt2e` produces a quantized model that represents quantized computation with
        # quantize dequantize ops and fp32 ops by default.
        # Please refer to
        # https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_static.html#convert-the-calibrated-model-to-a-quantized-model
        # for detailed explanation of output quantized model
        quantized_model = convert_pt2e(prepared_model)

    """
    torch._C._log_api_usage_once("quantization_api.quantize_pt2e.convert_pt2e")
    if not isinstance(use_reference_representation, bool):
        raise ValueError(
            "Unexpected argument type for `use_reference_representation`, "
            f"please make sure you intend to pass argument {use_reference_representation} to convert_pt2e"
        )
    original_graph_meta = model.meta
    model = _convert_to_reference_decomposed_fx(model)
    model = _fold_conv_bn_qat(model)

    pm = PassManager([DuplicateDQPass()])
    model = pm(model).graph_module

    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module

    if fold_quantize:
        constant_fold(model, _quant_node_constraint)

    if use_reference_representation:
        model = reference_representation_rewrite(model)

    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model

```



## High-Level Overview

"""Prepare a model for post training quantization    Args:      * `model` (torch.fx.GraphModule): a model captured by `torch.export.export_for_training` API.      * `quantizer`: A backend specific quantizer that conveys how user want the        model to be quantized. Tutorial for how to write a quantizer can be found here:        https://pytorch.org/tutorials/prototype/pt2e_quantizer.html    Return:      A GraphModule with observer (based on quantizer annotation), ready for calibration    Example::

This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `M`, `M`

**Functions defined**: `prepare_pt2e`, `__init__`, `forward`, `calibrate`, `prepare_qat_pt2e`, `__init__`, `forward`, `train_loop`, `_quant_node_constraint`, `convert_pt2e`

**Key imports**: typing_extensions, torch, constant_fold, DuplicateDQPass, PortNodeMetaForQDQ, GraphModule, Node, PassManager, prepare, _fold_conv_bn_qat, _fuse_conv_bn_qat, reference_representation_rewrite


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing_extensions`
- `torch`
- `torch._export.passes.constant_folding`: constant_fold
- `torch.ao.quantization.pt2e.duplicate_dq_pass`: DuplicateDQPass
- `torch.ao.quantization.pt2e.port_metadata_pass`: PortNodeMetaForQDQ
- `torch.fx`: GraphModule, Node
- `torch.fx.passes.infra.pass_manager`: PassManager
- `.pt2e.prepare`: prepare
- `.pt2e.qat_utils`: _fold_conv_bn_qat, _fuse_conv_bn_qat
- `.pt2e.representation`: reference_representation_rewrite
- `.pt2e.utils`: _disallow_eval_train, _fuse_conv_bn_, _get_node_name_to_scope
- `.quantize_fx`: _convert_to_reference_decomposed_fx
- `.utils`: DEPRECATION_WARNING
- `torch.ao.quantization.quantize_pt2e`: prepare_pt2e


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/ao/quantization`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`quant_type.py_docs.md`](./quant_type.py_docs.md)
- [`fake_quantize.py_docs.md`](./fake_quantize.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`fuse_modules.py_docs.md`](./fuse_modules.py_docs.md)
- [`_equalize.py_docs.md`](./_equalize.py_docs.md)
- [`quantize.py_docs.md`](./quantize.py_docs.md)
- [`_learnable_fake_quantize.py_docs.md`](./_learnable_fake_quantize.py_docs.md)
- [`observer.py_docs.md`](./observer.py_docs.md)
- [`pattern.md_docs.md`](./pattern.md_docs.md)


## Cross-References

- **File Documentation**: `quantize_pt2e.py_docs.md`
- **Keyword Index**: `quantize_pt2e.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
