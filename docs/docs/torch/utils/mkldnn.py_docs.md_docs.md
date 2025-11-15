# Documentation: `docs/torch/utils/mkldnn.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/mkldnn.py_docs.md`
- **Size**: 11,502 bytes (11.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/mkldnn.py`

## File Metadata

- **Path**: `torch/utils/mkldnn.py`
- **Size**: 8,244 bytes (8.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch


class MkldnnLinear(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype) -> None:
        super().__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))
        if dense_module.bias is not None:
            # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
            # we use fp32 dtype.
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

    @torch.jit.script_method
    def forward(self, x):
        x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
        y_mkldnn = torch._C._nn.mkldnn_linear(x_mkldnn, self.weight, self.bias)
        y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
        return y


class _MkldnnConvNd(torch.jit.ScriptModule):
    """Common base of MkldnnConv1d and MkldnnConv2d."""

    __constants__ = ['stride', 'padding', 'dilation', 'groups']

    def __init__(self, dense_module) -> None:
        super().__init__()

        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups

        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
            # we use fp32 dtype.
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def forward(self, x):
        return torch.mkldnn_convolution(
            x,
            self.weight,
            self.bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups)


class MkldnnConv1d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None:
        super().__init__(dense_module)

        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]


class MkldnnConv2d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None:
        super().__init__(dense_module)

        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv2d_weight(
            dense_module.weight.to_mkldnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(
            state[0].to_mkldnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

class MkldnnConv3d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None:
        super().__init__(dense_module)

        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv3d_weight(
            dense_module.weight.to_mkldnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.mkldnn_reorder_conv3d_weight(
            state[0].to_mkldnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]


class MkldnnBatchNorm(torch.jit.ScriptModule):
    __constants__ = ['exponential_average_factor', 'eps']

    def __init__(self, dense_module) -> None:
        super().__init__()

        if dense_module.training:
            raise AssertionError("Only support eval mode batchnorm for mkldnn path now")
        if not dense_module.track_running_stats:
            raise AssertionError("Only support track_running_stats=True for mkldnn path now")
        if not dense_module.affine:
            raise AssertionError("Only support affine=True for mkldnn path now")

        if dense_module.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = dense_module.momentum
        self.eps = dense_module.eps

        self.register_buffer('weight', dense_module.weight.to_mkldnn())
        self.register_buffer('bias', dense_module.bias.to_mkldnn())
        self.register_buffer('running_mean', dense_module.running_mean.to_mkldnn())
        self.register_buffer('running_var', dense_module.running_var.to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        weight = self.weight.to_dense()
        bias = self.bias.to_dense()
        running_mean = self.running_mean.to_dense()
        running_var = self.running_var.to_dense()
        return (weight, bias, running_mean, running_var, self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.running_mean = state[2].to_mkldnn()
        self.running_var = state[3].to_mkldnn()
        self.training = state[4]

    @torch.jit.script_method
    def forward(self, x):
        return torch.batch_norm(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            False,  # training
            self.exponential_average_factor,
            self.eps,
            False,  # cuda_enabled
        )

class MkldnnPrelu(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype) -> None:
        super().__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.training = state[1]

    @torch.jit.script_method
    def forward(self, x):
        x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
        y_mkldnn = torch.prelu(x_mkldnn, self.weight)
        y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
        return y

def to_mkldnn(module, dtype=torch.float):
    if dtype not in (torch.float, torch.bfloat16, torch.half):
        raise AssertionError("MKLDNN only support float, bfloat16, and half path now")


    def m_fn(m, d):
        if isinstance(m, torch.nn.Linear):
            return MkldnnLinear(m, d)
        elif isinstance(m, torch.nn.Conv1d):
            return MkldnnConv1d(m, d)
        elif isinstance(m, torch.nn.Conv2d):
            return MkldnnConv2d(m, d)
        elif isinstance(m, torch.nn.Conv3d):
            return MkldnnConv3d(m, d)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            # For batchnorm bf16 path, OneDNN requires weight and bias need fp32 dtype.
            # so it doesn't need dtype argument.
            return MkldnnBatchNorm(m)
        elif isinstance(m, torch.nn.PReLU):
            return MkldnnPrelu(m, d)
        else:
            return m

    def m_fn_rec(m, d):
        new_m = m_fn(m, d)
        for name, sub_m in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m, d))
        return new_m

    return m_fn_rec(module, dtype)

```



## High-Level Overview

"""Common base of MkldnnConv1d and MkldnnConv2d."""    __constants__ = ['stride', 'padding', 'dilation', 'groups']    def __init__(self, dense_module) -> None:        super().__init__()        self.stride = dense_module.stride        self.padding = dense_module.padding        self.dilation = dense_module.dilation        self.groups = dense_module.groups        if dense_module.bias is not None:

This Python file contains 7 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MkldnnLinear`, `_MkldnnConvNd`, `MkldnnConv1d`, `MkldnnConv2d`, `MkldnnConv3d`, `MkldnnBatchNorm`, `MkldnnPrelu`

**Functions defined**: `__init__`, `__getstate__`, `__setstate__`, `forward`, `__init__`, `__getstate__`, `forward`, `__init__`, `__setstate__`, `__init__`, `__setstate__`, `__init__`, `__setstate__`, `__init__`, `__getstate__`, `__setstate__`, `forward`, `__init__`, `__getstate__`, `__setstate__`

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `mkldnn.py_docs.md`
- **Keyword Index**: `mkldnn.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/utils`):

- [`show_pickle.py_docs.md_docs.md`](./show_pickle.py_docs.md_docs.md)
- [`file_baton.py_docs.md_docs.md`](./file_baton.py_docs.md_docs.md)
- [`_filelock.py_kw.md_docs.md`](./_filelock.py_kw.md_docs.md)
- [`_config_module.py_docs.md_docs.md`](./_config_module.py_docs.md_docs.md)
- [`cpp_extension.py_docs.md_docs.md`](./cpp_extension.py_docs.md_docs.md)
- [`checkpoint.py_docs.md_docs.md`](./checkpoint.py_docs.md_docs.md)
- [`module_tracker.py_kw.md_docs.md`](./module_tracker.py_kw.md_docs.md)
- [`dlpack.py_docs.md_docs.md`](./dlpack.py_docs.md_docs.md)
- [`_import_utils.py_kw.md_docs.md`](./_import_utils.py_kw.md_docs.md)
- [`_traceback.py_kw.md_docs.md`](./_traceback.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mkldnn.py_docs.md_docs.md`
- **Keyword Index**: `mkldnn.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
