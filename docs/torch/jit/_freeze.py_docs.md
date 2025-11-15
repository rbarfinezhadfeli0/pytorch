# Documentation: `torch/jit/_freeze.py`

## File Metadata

- **Path**: `torch/jit/_freeze.py`
- **Size**: 9,510 bytes (9.29 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""Freezing.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

from typing import Optional

import torch
from torch.jit._script import RecursiveScriptModule, ScriptModule


def freeze(
    mod, preserved_attrs: Optional[list[str]] = None, optimize_numerics: bool = True
):
    r"""Freeze ScriptModule, inline submodules, and attributes as constants.

    Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned
    module's submodules, parameters, and attributes as constants in the TorchScript IR Graph.
    By default, `forward` will be preserved, as well as attributes & methods specified in
    `preserved_attrs`. Additionally, any attribute that is modified within a preserved
    method will be preserved.

    Freezing currently only accepts ScriptModules that are in eval mode.

    Freezing applies generic optimization that will speed up your model regardless of machine.
    To further optimize using server-specific settings, run `optimize_for_inference` after
    freezing.

    Args:
        mod (:class:`ScriptModule`): a module to be frozen
        preserved_attrs (Optional[List[str]]): a list of attributes to preserve in addition to the forward method.
            Attributes modified in preserved methods will also be preserved.
        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
            preserve numerics. Full details of optimization can be found at `torch.jit.run_frozen_optimizations`.

    Returns:
        Frozen :class:`ScriptModule`.

    Example (Freezing a simple module with a Parameter):

    .. testcode::
        import torch
        class MyModule(torch.nn.Module):
            def __init__(self, N, M):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(N, M))
                self.linear = torch.nn.Linear(N, M)

            def forward(self, input):
                output = self.weight.mm(input)
                output = self.linear(output)
                return output

        scripted_module = torch.jit.script(MyModule(2, 3).eval())
        frozen_module = torch.jit.freeze(scripted_module)
        # parameters have been removed and inlined into the Graph as constants
        assert len(list(frozen_module.named_parameters())) == 0
        # See the compiled graph as Python code
        print(frozen_module.code)

    Example (Freezing a module with preserved attributes)

    .. testcode::
        import torch
        class MyModule2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.modified_tensor = torch.tensor(10.)
                self.version = 1

            def forward(self, input):
                self.modified_tensor += 1
                return input + self.modified_tensor

        scripted_module = torch.jit.script(MyModule2().eval())
        frozen_module = torch.jit.freeze(scripted_module, preserved_attrs=["version"])
        # we've manually preserved `version`, so it still exists on the frozen module and can be modified
        assert frozen_module.version == 1
        frozen_module.version = 2
        # `modified_tensor` is detected as being mutated in the forward, so freezing preserves
        # it to retain model semantics
        assert frozen_module(torch.tensor(1)) == torch.tensor(12)
        # now that we've run it once, the next result will be incremented by one
        assert frozen_module(torch.tensor(1)) == torch.tensor(13)

    Note:
        Freezing submodule attributes is also supported:
        frozen_module = torch.jit.freeze(scripted_module, preserved_attrs=["submodule.version"])

    Note:
        If you're not sure why an attribute is not being inlined as a constant, you can run
        `dump_alias_db` on frozen_module.forward.graph to see if freezing has detected the
        attribute is being modified.

    Note:
        Because freezing makes weights constants and removes module hierarchy, `to` and other
        nn.Module methods to manipulate device or dtype no longer work. As a workaround,
        You can remap devices by specifying `map_location` in `torch.jit.load`, however
        device-specific logic may have been baked into the model.
    """
    if not isinstance(mod, ScriptModule):
        raise RuntimeError(
            "Freezing expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
        )

    if mod.training:
        raise RuntimeError(
            "Freezing is currently only implemented for modules in eval mode. "
            "Please call .eval() on your module before freezing."
        )

    preserved_attrs = preserved_attrs if preserved_attrs is not None else []

    out = RecursiveScriptModule(torch._C._freeze_module(mod._c, preserved_attrs))
    RecursiveScriptModule._finalize_scriptmodule(out)

    preserved_methods = [x for x in preserved_attrs if mod._c._has_method(x)]
    run_frozen_optimizations(out, optimize_numerics, preserved_methods)

    return out


def run_frozen_optimizations(
    mod, optimize_numerics: bool = True, preserved_methods: Optional[list[str]] = None
) -> None:
    r"""
    Run a series of optimizations looking for patterns that occur in frozen graphs.

    The current set of optimizations includes:
        - Dropout Removal
        - Pretranspose Linear Layers
        - Concat Linear Layers with same input Tensor
        - Conv -> Batchnorm folding
        - Conv -> Add/Sub folding
        - Conv -> Mul/Div folding

    Args:
        mod (:class:`ScriptModule`): a frozen module to be optimized

        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
        preserve numerics. These optimizations preserve default rtol and atol of `torch.testing.assert_close`
        when applied on a single transformation, however in a module where many transformations are applied
        the rtol or atol may no longer fall within the default `assert_close` tolerance. Conv -> Batchnorm folding,
        Conv-Add/Sub, and Conv -> Mul/Div folding all may alter numerics.

    Returns:
        None

    Note:
        In rare occasions, this can result in slower execution.

    Example (Freezing a module with Conv->Batchnorm)
    .. code-block:: python
        import torch

        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True
        )
        bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        mod = torch.nn.Sequential(conv, bn)
        # set optimize to False here, by default freezing runs run_frozen_optimizations
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()), optimize=False)
        # inspect frozen mod
        assert "batch_norm" in str(frozen_mod.graph)
        torch.jit.run_frozen_optimizations(frozen_mod)
        assert "batch_norm" not in str(frozen_mod.graph)

    """
    if mod._c._has_method("forward"):
        torch._C._jit_pass_optimize_frozen_graph(mod.graph, optimize_numerics)

    if preserved_methods is None:
        preserved_methods = []

    for method in preserved_methods:
        torch._C._jit_pass_optimize_frozen_graph(
            mod.__getattr__(method).graph, optimize_numerics
        )


def optimize_for_inference(
    mod: ScriptModule, other_methods: Optional[list[str]] = None
) -> ScriptModule:
    """
    Perform a set of optimization passes to optimize a model for the purposes of inference.

    If the model is not already frozen, optimize_for_inference
    will invoke `torch.jit.freeze` automatically.

    In addition to generic optimizations that should speed up your model regardless
    of environment, prepare for inference will also bake in build specific settings
    such as the presence of CUDNN or MKLDNN, and may in the future make transformations
    which speed things up on one machine but slow things down on another. Accordingly,
    serialization is not implemented following invoking `optimize_for_inference` and
    is not guaranteed.

    This is still in prototype, and may have the potential to slow down your model.
    Primary use cases that have been targeted so far have been vision models on cpu
    and gpu to a lesser extent.

    Example (optimizing a module with Conv->Batchnorm)::

        import torch

        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True
        )
        bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        mod = torch.nn.Sequential(conv, bn)
        frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(mod.eval()))
        assert "batch_norm" not in str(frozen_mod.graph)
        # if built with MKLDNN, convolution will be run with MKLDNN weights
        assert "MKLDNN" in frozen_mod.graph
    """
    if not isinstance(mod, ScriptModule):
        raise RuntimeError(
            "optimize_for_inference expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
        )

    if other_methods is None:
        other_methods = []

    if hasattr(mod, "training"):
        mod = freeze(mod.eval(), preserved_attrs=other_methods)

    torch._C._jit_pass_optimize_for_inference(mod._c, other_methods)

    return mod

```



## High-Level Overview

"""Freezing.This is not intended to be imported directly; please use the exposedfunctionalities in `torch.jit`.

This Python file contains 2 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyModule`, `MyModule2`

**Functions defined**: `freeze`, `__init__`, `forward`, `__init__`, `forward`, `run_frozen_optimizations`, `optimize_for_inference`

**Key imports**: Optional, torch, RecursiveScriptModule, ScriptModule, torch, torch, torch, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch.jit._script`: RecursiveScriptModule, ScriptModule


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

Files in the same folder (`torch/jit`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_decompositions.py_docs.md`](./_decompositions.py_docs.md)
- [`_dataclass_impls.py_docs.md`](./_dataclass_impls.py_docs.md)
- [`quantized.py_docs.md`](./quantized.py_docs.md)
- [`frontend.py_docs.md`](./frontend.py_docs.md)
- [`_builtins.py_docs.md`](./_builtins.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)
- [`_state.py_docs.md`](./_state.py_docs.md)
- [`_await.py_docs.md`](./_await.py_docs.md)


## Cross-References

- **File Documentation**: `_freeze.py_docs.md`
- **Keyword Index**: `_freeze.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
