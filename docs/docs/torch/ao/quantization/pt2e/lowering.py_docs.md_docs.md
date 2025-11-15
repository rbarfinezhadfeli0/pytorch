# Documentation: `docs/torch/ao/quantization/pt2e/lowering.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/pt2e/lowering.py_docs.md`
- **Size**: 4,726 bytes (4.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/pt2e/lowering.py`

## File Metadata

- **Path**: `torch/ao/quantization/pt2e/lowering.py`
- **Size**: 1,892 bytes (1.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import torch
from torch._inductor.constant_folding import constant_fold
from torch._inductor.fx_passes.freezing_patterns import freezing_passes


__all__ = [
    "lower_pt2e_quantized_to_x86",
]


def lower_pt2e_quantized_to_x86(
    model: torch.fx.GraphModule,
    example_inputs: tuple[torch.Tensor, ...],
) -> torch.fx.GraphModule:
    """Lower a PT2E-quantized model to x86 backend.

    Args:
    * `model` (torch.fx.GraphModule): a model quantized by PT2E quantization flow.
    * `example_inputs` (tuple[torch.Tensor, ...]): example inputs for the model.

    Return:
    A GraphModule lowered to x86 backend.
    """

    def _post_autograd_decomp_table():  # type: ignore[no-untyped-def]
        decomp_table = torch.export.default_decompositions()

        # if we are post-autograd, we shouldn't
        # decomp prim ops.
        for k in list(decomp_table.keys()):
            if not torch._export.utils._is_cia_op(k):
                del decomp_table[k]

        return decomp_table

    def _node_replace(m):  # type: ignore[no-untyped-def]
        # Replace aten.t(x) with aten.permute(x, [1, 0])
        aten = torch.ops.aten
        g = m.graph
        for node in g.nodes:
            if node.target is aten.t.default:
                with g.inserting_before(node):
                    x = node.args[0]
                    dims = [1, 0]
                    perm_node = g.call_function(aten.permute.default, args=(x, dims))
                    node.replace_all_uses_with(perm_node)
                    g.erase_node(node)

        g.lint()
        m.recompile()

    lowered_model = (
        torch.export.export(model, example_inputs, strict=True)
        .run_decompositions(_post_autograd_decomp_table())
        .module()
    )
    _node_replace(lowered_model)
    freezing_passes(lowered_model, example_inputs)
    constant_fold(lowered_model)
    return lowered_model

```



## High-Level Overview

"""Lower a PT2E-quantized model to x86 backend.    Args:    * `model` (torch.fx.GraphModule): a model quantized by PT2E quantization flow.    * `example_inputs` (tuple[torch.Tensor, ...]): example inputs for the model.    Return:    A GraphModule lowered to x86 backend.

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `lower_pt2e_quantized_to_x86`, `_post_autograd_decomp_table`, `_node_replace`

**Key imports**: torch, constant_fold, freezing_passes


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/pt2e`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.constant_folding`: constant_fold
- `torch._inductor.fx_passes.freezing_patterns`: freezing_passes


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/ao/quantization/pt2e`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`port_metadata_pass.py_docs.md`](./port_metadata_pass.py_docs.md)
- [`_numeric_debugger.py_docs.md`](./_numeric_debugger.py_docs.md)
- [`duplicate_dq_pass.py_docs.md`](./duplicate_dq_pass.py_docs.md)
- [`_affine_quantization.py_docs.md`](./_affine_quantization.py_docs.md)
- [`qat_utils.py_docs.md`](./qat_utils.py_docs.md)
- [`prepare.py_docs.md`](./prepare.py_docs.md)
- [`export_utils.py_docs.md`](./export_utils.py_docs.md)


## Cross-References

- **File Documentation**: `lowering.py_docs.md`
- **Keyword Index**: `lowering.py_kw.md`
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

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/ao/quantization/pt2e`):

- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`_numeric_debugger.py_kw.md_docs.md`](./_numeric_debugger.py_kw.md_docs.md)
- [`duplicate_dq_pass.py_docs.md_docs.md`](./duplicate_dq_pass.py_docs.md_docs.md)
- [`prepare.py_kw.md_docs.md`](./prepare.py_kw.md_docs.md)
- [`qat_utils.py_docs.md_docs.md`](./qat_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`graph_utils.py_docs.md_docs.md`](./graph_utils.py_docs.md_docs.md)
- [`export_utils.py_docs.md_docs.md`](./export_utils.py_docs.md_docs.md)
- [`export_utils.py_kw.md_docs.md`](./export_utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `lowering.py_docs.md_docs.md`
- **Keyword Index**: `lowering.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
