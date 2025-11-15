# Documentation: `docs/torch/nativert/backends/_lower_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/backends/_lower_utils.py_docs.md`
- **Size**: 6,164 bytes (6.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/backends/_lower_utils.py`

## File Metadata

- **Path**: `torch/nativert/backends/_lower_utils.py`
- **Size**: 3,499 bytes (3.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import types

import torch
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.pt2_archive._package import AOTI_FILES, package_pt2
from torch.types import FileLike

from ._lowered_aoti_module import LoweredBackendModule


def get_new_ep_with_flat_inputs_outputs(ep: ExportedProgram) -> ExportedProgram:
    class FlattenedModule(torch.nn.Module):
        def __init__(
            self,
            original_module: torch.fx.GraphModule,
            in_spec: pytree.TreeSpec,
            out_spec: pytree.TreeSpec,
        ) -> None:
            super().__init__()
            self.original_module = original_module
            self.in_spec = in_spec
            self.out_spec = out_spec

        def forward(self, *flat_inputs):  # type: ignore[no-untyped-def]
            # Unflatten inputs to original structure
            inputs = pytree.tree_unflatten(flat_inputs, self.in_spec)
            args, kwargs = inputs
            outputs = self.original_module(*args, **kwargs)
            # Flatten outputs
            flat_outputs, _ = pytree.tree_flatten(outputs)
            return tuple(flat_outputs)

    flattened_module = FlattenedModule(
        ep.module(), ep.call_spec.in_spec, ep.call_spec.out_spec
    )
    args, kwargs = ep.example_inputs
    flat_inputs, _ = pytree.tree_flatten((args, kwargs))
    flat_ep = torch.export.export(flattened_module, tuple(flat_inputs))

    return flat_ep


def lower_exported_program(
    exported_program: ExportedProgram, model_name: str, backend_id: str
) -> tuple[ExportedProgram, AOTI_FILES]:
    """
    Lower an exported program to AOTInductor and return a delegate ExportedProgram
    with the `executorch_call_delegate` HOP
    """
    args, kwargs = exported_program.example_inputs
    out_spec = exported_program.call_spec.out_spec
    flat_ep = get_new_ep_with_flat_inputs_outputs(exported_program)
    flat_inputs, _ = pytree.tree_flatten((args, kwargs))

    aoti_files = torch._inductor.aot_compile(
        flat_ep.module(), tuple(flat_inputs), options={"aot_inductor.package": True}
    )
    assert isinstance(aoti_files, list)

    lowered_aoti_module = LoweredBackendModule(
        flat_ep, backend_id, module_name=model_name
    )

    def patched_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        flat_inputs, _ = pytree.tree_flatten((args, kwargs))
        flat_outputs = torch._higher_order_ops.executorch_call_delegate(
            self, *flat_inputs
        )
        if out_spec is not None and flat_outputs is not None:
            return pytree.tree_unflatten(flat_outputs, out_spec)
        else:
            return flat_outputs

    lowered_aoti_module.forward = types.MethodType(patched_forward, lowered_aoti_module)  # type: ignore[method-assign]

    aoti_delegate_ep = torch.export.export(lowered_aoti_module, args, kwargs)

    return aoti_delegate_ep, aoti_files


def package_nativert_with_aoti_delegate(
    f: FileLike,
    model_name: str,
    backend_id: str,
    original_ep: ExportedProgram,
    delegate_ep: ExportedProgram,
    delegate_files: AOTI_FILES,
) -> None:
    """
    Package a pt2 archive file that can be consumed by NativeRT with AOTI Delegate
    """
    package_pt2(
        f,
        exported_programs={
            model_name: original_ep,
            f"{model_name}-{backend_id}": delegate_ep,
        },
        aoti_files={f"{model_name}-{backend_id}": delegate_files},  # type: ignore[dict-item]
    )
    return

```



## High-Level Overview

"""    Lower an exported program to AOTInductor and return a delegate ExportedProgram    with the `executorch_call_delegate` HOP

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FlattenedModule`

**Functions defined**: `get_new_ep_with_flat_inputs_outputs`, `__init__`, `forward`, `lower_exported_program`, `patched_forward`, `package_nativert_with_aoti_delegate`

**Key imports**: types, torch, torch.utils._pytree as pytree, ExportedProgram, AOTI_FILES, package_pt2, FileLike, LoweredBackendModule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/backends`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `types`
- `torch`
- `torch.utils._pytree as pytree`
- `torch.export`: ExportedProgram
- `torch.export.pt2_archive._package`: AOTI_FILES, package_pt2
- `torch.types`: FileLike
- `._lowered_aoti_module`: LoweredBackendModule


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

Files in the same folder (`torch/nativert/backends`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_lowered_aoti_module.py_docs.md`](./_lowered_aoti_module.py_docs.md)


## Cross-References

- **File Documentation**: `_lower_utils.py_docs.md`
- **Keyword Index**: `_lower_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/backends`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/backends`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/nativert/backends`):

- [`_lowered_aoti_module.py_kw.md_docs.md`](./_lowered_aoti_module.py_kw.md_docs.md)
- [`_lower_utils.py_kw.md_docs.md`](./_lower_utils.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`_lowered_aoti_module.py_docs.md_docs.md`](./_lowered_aoti_module.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_lower_utils.py_docs.md_docs.md`
- **Keyword Index**: `_lower_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
