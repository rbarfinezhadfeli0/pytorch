# Documentation: `docs/torch/backends/_nnapi/prepare.py_docs.md`

## File Metadata

- **Path**: `docs/torch/backends/_nnapi/prepare.py_docs.md`
- **Size**: 9,035 bytes (8.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/backends/_nnapi/prepare.py`

## File Metadata

- **Path**: `torch/backends/_nnapi/prepare.py`
- **Size**: 6,559 bytes (6.41 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch.backends._nnapi.serializer import _NnapiSerializer


ANEURALNETWORKS_PREFER_LOW_POWER = 0
ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1
ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2


class NnapiModule(torch.nn.Module):
    """Torch Module that wraps an NNAPI Compilation.

    This module handles preparing the weights, initializing the
    NNAPI TorchBind object, and adjusting the memory formats
    of all inputs and outputs.
    """

    # _nnapi.Compilation is defined
    comp: Optional[torch.classes._nnapi.Compilation]  # type: ignore[name-defined]
    weights: list[torch.Tensor]
    out_templates: list[torch.Tensor]

    def __init__(
        self,
        shape_compute_module: torch.nn.Module,
        ser_model: torch.Tensor,
        weights: list[torch.Tensor],
        inp_mem_fmts: list[int],
        out_mem_fmts: list[int],
        compilation_preference: int,
        relax_f32_to_f16: bool,
    ):
        super().__init__()
        self.shape_compute_module = shape_compute_module
        self.ser_model = ser_model
        self.weights = weights
        self.inp_mem_fmts = inp_mem_fmts
        self.out_mem_fmts = out_mem_fmts
        self.out_templates = []
        self.comp = None
        self.compilation_preference = compilation_preference
        self.relax_f32_to_f16 = relax_f32_to_f16

    @torch.jit.export
    def init(self, args: list[torch.Tensor]):
        assert self.comp is None
        self.out_templates = self.shape_compute_module.prepare(self.ser_model, args)  # type: ignore[operator]
        self.weights = [w.contiguous() for w in self.weights]
        comp = torch.classes._nnapi.Compilation()
        comp.init2(
            self.ser_model,
            self.weights,
            self.compilation_preference,
            self.relax_f32_to_f16,
        )

        self.comp = comp

    def forward(self, args: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.comp is None:
            self.init(args)
        comp = self.comp
        assert comp is not None
        outs = [torch.empty_like(out) for out in self.out_templates]

        assert len(args) == len(self.inp_mem_fmts)
        fixed_args = []
        for idx in range(len(args)):
            fmt = self.inp_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt == 0:
                fixed_args.append(args[idx].contiguous())
            elif fmt == 1:
                fixed_args.append(args[idx].permute(0, 2, 3, 1).contiguous())
            else:
                raise ValueError("Invalid mem_fmt")
        comp.run(fixed_args, outs)
        assert len(outs) == len(self.out_mem_fmts)
        for idx in range(len(self.out_templates)):
            fmt = self.out_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt in (0, 2):
                pass
            elif fmt == 1:
                outs[idx] = outs[idx].permute(0, 3, 1, 2)
            else:
                raise ValueError("Invalid mem_fmt")
        return outs


def convert_model_to_nnapi(
    model,
    inputs,
    serializer=None,
    return_shapes=None,
    use_int16_for_qint16=False,
    compilation_preference=ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
    relax_f32_to_f16=False,
):
    (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    ) = process_for_nnapi(
        model, inputs, serializer, return_shapes, use_int16_for_qint16
    )

    nnapi_model = NnapiModule(
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        compilation_preference,
        relax_f32_to_f16,
    )

    class NnapiInterfaceWrapper(torch.nn.Module):
        """NNAPI list-ifying and de-list-ifying wrapper.

        NNAPI always expects a list of inputs and provides a list of outputs.
        This module allows us to accept inputs as separate arguments.
        It returns results as either a single tensor or tuple,
        matching the original module.
        """

        def __init__(self, mod):
            super().__init__()
            self.mod = mod

    wrapper_model_py = NnapiInterfaceWrapper(nnapi_model)
    wrapper_model = torch.jit.script(wrapper_model_py)
    # TODO: Maybe make these names match the original.
    arg_list = ", ".join(f"arg_{idx}" for idx in range(len(inputs)))
    if retval_count < 0:
        ret_expr = "retvals[0]"
    else:
        ret_expr = "".join(f"retvals[{idx}], " for idx in range(retval_count))
    wrapper_model.define(
        f"def forward(self, {arg_list}):\n"
        f"    retvals = self.mod([{arg_list}])\n"
        f"    return {ret_expr}\n"
    )
    return wrapper_model


def process_for_nnapi(
    model, inputs, serializer=None, return_shapes=None, use_int16_for_qint16=False
):
    model = torch.jit.freeze(model)

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    serializer = serializer or _NnapiSerializer(
        config=None, use_int16_for_qint16=use_int16_for_qint16
    )
    (
        ser_model,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        shape_compute_lines,
        retval_count,
    ) = serializer.serialize_model(model, inputs, return_shapes)
    ser_model_tensor = torch.tensor(ser_model, dtype=torch.int32)

    # We have to create a new class here every time this function is called
    # because module.define adds a method to the *class*, not the instance.
    class ShapeComputeModule(torch.nn.Module):
        """Code-gen-ed module for tensor shape computation.

        module.prepare will mutate ser_model according to the computed operand
        shapes, based on the shapes of args.  Returns a list of output templates.
        """

    shape_compute_module = torch.jit.script(ShapeComputeModule())
    real_shape_compute_lines = [
        "def prepare(self, ser_model: torch.Tensor, args: List[torch.Tensor]) -> List[torch.Tensor]:\n",
    ] + [f"    {line}\n" for line in shape_compute_lines]
    shape_compute_module.define("".join(real_shape_compute_lines))

    return (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    )

```



## High-Level Overview

"""Torch Module that wraps an NNAPI Compilation.    This module handles preparing the weights, initializing the    NNAPI TorchBind object, and adjusting the memory formats    of all inputs and outputs.

This Python file contains 4 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `NnapiModule`, `NnapiInterfaceWrapper`, `ShapeComputeModule`

**Functions defined**: `__init__`, `init`, `forward`, `convert_model_to_nnapi`, `__init__`, `forward`, `process_for_nnapi`, `prepare`

**Key imports**: Optional, torch, _NnapiSerializer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/backends/_nnapi`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch.backends._nnapi.serializer`: _NnapiSerializer


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/backends/_nnapi`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`serializer.py_docs.md`](./serializer.py_docs.md)


## Cross-References

- **File Documentation**: `prepare.py_docs.md`
- **Keyword Index**: `prepare.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/backends/_nnapi`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/backends/_nnapi`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/backends/_nnapi`):

- [`serializer.py_docs.md_docs.md`](./serializer.py_docs.md_docs.md)
- [`prepare.py_kw.md_docs.md`](./prepare.py_kw.md_docs.md)
- [`serializer.py_kw.md_docs.md`](./serializer.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `prepare.py_docs.md_docs.md`
- **Keyword Index**: `prepare.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
