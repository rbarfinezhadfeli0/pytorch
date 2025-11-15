# Documentation: `docs/torch/ao/quantization/pt2e/export_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/pt2e/export_utils.py_docs.md`
- **Size**: 11,323 bytes (11.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/pt2e/export_utils.py`

## File Metadata

- **Path**: `torch/ao/quantization/pt2e/export_utils.py`
- **Size**: 7,990 bytes (7.80 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import types

import torch
import torch.nn.functional as F
from torch.ao.quantization.utils import _assert_and_get_unique_device


__all__ = [
    "model_is_exported",
]

_EXPORTED_TRAINING_ATTR = "_exported_training"


class _WrapperModule(torch.nn.Module):
    """Class to wrap a callable in an :class:`torch.nn.Module`. Use this if you
    are trying to export a callable.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        """Simple forward that just calls the ``fn`` provided to :meth:`WrapperModule.__init__`."""
        return self.fn(*args, **kwargs)


def model_is_exported(m: torch.nn.Module) -> bool:
    """
    Return True if the `torch.nn.Module` was exported, False otherwise
    (e.g. if the model was FX symbolically traced or not traced at all).
    """
    return isinstance(m, torch.fx.GraphModule) and any(
        "val" in n.meta for n in m.graph.nodes
    )


def _replace_dropout(m: torch.fx.GraphModule, train_to_eval: bool):
    """
    Switch dropout patterns in the model between train and eval modes.

    Dropout has different behavior in train vs eval mode. For exported models,
    however, calling `model.train()` or `model.eval()` does not automatically switch
    the dropout behavior between the two modes, so here we need to rewrite the aten
    dropout patterns manually to achieve the same effect.

    See https://github.com/pytorch/pytorch/issues/103681.
    """
    # Avoid circular dependencies
    from .utils import _get_aten_graph_module_for_pattern

    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()

    for inplace in [False, True]:

        def dropout_train(x):
            return F.dropout(x, p=0.5, training=True, inplace=inplace)

        def dropout_eval(x):
            return F.dropout(x, p=0.5, training=False, inplace=inplace)

        example_inputs = (torch.randn(1),)
        if train_to_eval:
            match_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_train),
                example_inputs,
            )
            replacement_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_eval),
                example_inputs,
            )
        else:
            match_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_eval),
                example_inputs,
            )
            replacement_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_train),
                example_inputs,
            )

        from torch.fx.subgraph_rewriter import replace_pattern_with_filters

        replace_pattern_with_filters(
            m,
            match_pattern,
            replacement_pattern,
            match_filters=[],
            ignore_literals=True,
        )
        m.recompile()


def _replace_batchnorm(m: torch.fx.GraphModule, train_to_eval: bool):
    """
    Switch batchnorm patterns in the model between train and eval modes.

    Batchnorm has different behavior in train vs eval mode. For exported models,
    however, calling `model.train()` or `model.eval()` does not automatically switch
    the batchnorm behavior between the two modes, so here we need to rewrite the aten
    batchnorm patterns manually to achieve the same effect.
    """
    # TODO(Leslie): This function still fails to support custom momentum and eps value.
    # Enable this support in future updates.

    # Avoid circular dependencies
    from .utils import _get_aten_graph_module_for_pattern

    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()

    def bn_train(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True
        )

    def bn_eval(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False
        )

    example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    device = _assert_and_get_unique_device(m)
    is_cuda = device is not None and device.type == "cuda"
    bn_train_aten = _get_aten_graph_module_for_pattern(
        _WrapperModule(bn_train),
        example_inputs,
        is_cuda,
    )
    bn_eval_aten = _get_aten_graph_module_for_pattern(
        _WrapperModule(bn_eval),
        example_inputs,
        is_cuda,
    )

    if train_to_eval:
        match_pattern = bn_train_aten
        replacement_pattern = bn_eval_aten
    else:
        match_pattern = bn_eval_aten
        replacement_pattern = bn_train_aten

    from torch.fx.subgraph_rewriter import replace_pattern_with_filters

    replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern,
        match_filters=[],
        ignore_literals=True,
    )
    m.recompile()


# TODO: expose these under this namespace?
def _move_exported_model_to_eval(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to eval mode.

    This is equivalent to model.eval() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing inference on the model.

    This call is idempotent; if the model is already in eval mode, nothing will happen.
    """
    is_training = getattr(model, _EXPORTED_TRAINING_ATTR, True)
    if not is_training:
        return model
    setattr(model, _EXPORTED_TRAINING_ATTR, False)
    _replace_dropout(model, train_to_eval=True)
    _replace_batchnorm(model, train_to_eval=True)
    return model


def _move_exported_model_to_train(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to train mode.

    This is equivalent to model.train() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing training on the model.

    This call is idempotent; if the model is already in train mode, nothing will happen.
    """
    is_training = getattr(model, _EXPORTED_TRAINING_ATTR, False)
    if is_training:
        return model
    setattr(model, _EXPORTED_TRAINING_ATTR, True)
    _replace_dropout(model, train_to_eval=False)
    _replace_batchnorm(model, train_to_eval=False)
    return model


def _allow_exported_model_train_eval(model: torch.fx.GraphModule):
    """
    Allow users to call `model.train()` and `model.eval()` on an exported model,
    but with the effect of changing behavior between the two modes limited to special
    ops only, which are currently dropout and batchnorm.

    Note: This does not achieve the same effect as what `model.train()` and `model.eval()`
    does in eager models, but only provides an approximation. In particular, user code
    branching on `training` flag will not function correctly in general because the branch
    is already specialized at export time. Additionally, other ops beyond dropout and batchnorm
    that have different train/eval behavior will also not be converted properly.
    """

    def _train(self, mode: bool = True):
        if mode:
            _move_exported_model_to_train(self)
        else:
            _move_exported_model_to_eval(self)

    def _eval(self):
        _move_exported_model_to_eval(self)

    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    return model

```



## High-Level Overview

"""Class to wrap a callable in an :class:`torch.nn.Module`. Use this if you    are trying to export a callable.

This Python file contains 1 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_WrapperModule`

**Functions defined**: `__init__`, `forward`, `model_is_exported`, `_replace_dropout`, `dropout_train`, `dropout_eval`, `_replace_batchnorm`, `bn_train`, `bn_eval`, `_move_exported_model_to_eval`, `_move_exported_model_to_train`, `_allow_exported_model_train_eval`, `_train`, `_eval`

**Key imports**: types, torch, torch.nn.functional as F, _assert_and_get_unique_device, _get_aten_graph_module_for_pattern, replace_pattern_with_filters, _get_aten_graph_module_for_pattern, replace_pattern_with_filters


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/pt2e`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `types`
- `torch`
- `torch.nn.functional as F`
- `torch.ao.quantization.utils`: _assert_and_get_unique_device
- `.utils`: _get_aten_graph_module_for_pattern
- `torch.fx.subgraph_rewriter`: replace_pattern_with_filters


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/ao/quantization/pt2e`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`port_metadata_pass.py_docs.md`](./port_metadata_pass.py_docs.md)
- [`_numeric_debugger.py_docs.md`](./_numeric_debugger.py_docs.md)
- [`duplicate_dq_pass.py_docs.md`](./duplicate_dq_pass.py_docs.md)
- [`lowering.py_docs.md`](./lowering.py_docs.md)
- [`_affine_quantization.py_docs.md`](./_affine_quantization.py_docs.md)
- [`qat_utils.py_docs.md`](./qat_utils.py_docs.md)
- [`prepare.py_docs.md`](./prepare.py_docs.md)


## Cross-References

- **File Documentation**: `export_utils.py_docs.md`
- **Keyword Index**: `export_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/pt2e`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/pt2e`, which is part of the **core PyTorch library**.



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
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/ao/quantization/pt2e`):

- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`_numeric_debugger.py_kw.md_docs.md`](./_numeric_debugger.py_kw.md_docs.md)
- [`duplicate_dq_pass.py_docs.md_docs.md`](./duplicate_dq_pass.py_docs.md_docs.md)
- [`prepare.py_kw.md_docs.md`](./prepare.py_kw.md_docs.md)
- [`qat_utils.py_docs.md_docs.md`](./qat_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`graph_utils.py_docs.md_docs.md`](./graph_utils.py_docs.md_docs.md)
- [`lowering.py_docs.md_docs.md`](./lowering.py_docs.md_docs.md)
- [`export_utils.py_kw.md_docs.md`](./export_utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `export_utils.py_docs.md_docs.md`
- **Keyword Index**: `export_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
