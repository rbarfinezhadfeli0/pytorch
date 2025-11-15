# Documentation: `torch/distributed/nn/jit/templates/remote_module_template.py`

## File Metadata

- **Path**: `torch/distributed/nn/jit/templates/remote_module_template.py`
- **Size**: 3,463 bytes (3.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
#!/usr/bin/python3
# mypy: allow-untyped-defs


def get_remote_module_template(enable_moving_cpu_tensors_to_cuda: bool):
    return _TEMPLATE_PREFIX + (
        _REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA
        if enable_moving_cpu_tensors_to_cuda
        else _REMOTE_FORWARD_TEMPLATE
    )


_TEMPLATE_PREFIX = """from typing import *

import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch._jit_internal import Future
from torch.distributed.rpc import RRef
from typing import Tuple  # pyre-ignore: unused import


{assign_module_interface_cls}


def forward_async(self, {arg_types}){arrow_and_future_return_type}:
    args = (self.module_rref, self.device, self.is_device_map_set, {args})
    kwargs = {{{kwargs}}}
    return rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )


def forward(self, {arg_types}){arrow_and_return_type}:
    args = (self.module_rref, self.device, self.is_device_map_set, {args})
    kwargs = {{{kwargs}}}
    ret_fut = rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )
    return ret_fut.wait()


_generated_methods = [
    forward_async,
    forward,
]


{jit_script_decorator}
"""

# This template may cause typing error (the mismatch between ``Tuple[()]`` and ``Tuple[Any]``)
# even if the code is only used for instantiation but not execution.
# Therefore, only include handling moving CPU tensors to a cuda device if necessary.
# TODO: Merge these two templates together in the future once TorchScript syntax is improved.
_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA = """
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()
    device = torch.device(device)

    if device.type != "cuda":
        return module.forward({args}, {kwargs})

    # If the module is on a cuda device,
    # move any CPU tensor in args or kwargs to the same cuda device.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.to(device) if isinstance(i, Tensor) else i for i in *args)``.
    args = ({args},)
    out_args: Tuple[()] = ()
    for arg in args:
        arg = (arg.to(device),) if isinstance(arg, Tensor) else (arg,)
        out_args = out_args + arg

    kwargs = {{{kwargs}}}
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            kwargs[k] = kwargs[k].to(device)

    if is_device_map_set:
        return module.forward(*out_args, {kwargs})

    # If the device map is empty, then only CPU tensors are allowed to send over wire,
    # so have to move any GPU tensor to CPU in the output.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.cpu() if isinstance(i, Tensor) else i for i in module.forward(*out_args, {kwargs}))``.
    ret: Tuple[()] = ()
    for i in module.forward(*out_args, {kwargs}):
        i = (i.cpu(),) if isinstance(i, Tensor) else (i,)
        ret = ret + i
    return ret
"""

_REMOTE_FORWARD_TEMPLATE = """
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()

    return module.forward({args}, {kwargs})
"""

```



## High-Level Overview

_TEMPLATE_PREFIX = """from typing import *import torchimport torch.distributed.rpc as rpcfrom torch import Tensorfrom torch._jit_internal import Futurefrom torch.distributed.rpc import RReffrom typing import Tuple  # pyre-ignore: unused import{assign_module_interface_cls}def forward_async(self, {arg_types}){arrow_and_future_return_type}:    args = (self.module_rref, self.device, self.is_device_map_set, {args})    kwargs = {{{kwargs}}}    return rpc.rpc_async(        self.module_rref.owner(),        _remote_forward,        args,        kwargs,    )def forward(self, {arg_types}){arrow_and_return_type}:    args = (self.module_rref, self.device, self.is_device_map_set, {args})    kwargs = {{{kwargs}}}    ret_fut = rpc.rpc_async(        self.module_rref.owner(),        _remote_forward,        args,        kwargs,    )    return ret_fut.wait()_generated_methods = [    forward_async,

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_remote_module_template`, `forward_async`, `forward`, `_remote_forward`, `_remote_forward`

**Key imports**: torch, torch.distributed.rpc as rpc, Tensor, Future, RRef, Tuple  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/nn/jit/templates`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.rpc as rpc`
- `torch._jit_internal`: Future
- `torch.distributed.rpc`: RRef
- `typing`: Tuple  


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/distributed/nn/jit/templates`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `remote_module_template.py_docs.md`
- **Keyword Index**: `remote_module_template.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
