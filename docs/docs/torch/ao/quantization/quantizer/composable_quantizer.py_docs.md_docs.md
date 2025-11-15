# Documentation: `docs/torch/ao/quantization/quantizer/composable_quantizer.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/quantizer/composable_quantizer.py_docs.md`
- **Size**: 6,585 bytes (6.43 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/quantizer/composable_quantizer.py`

## File Metadata

- **Path**: `torch/ao/quantization/quantizer/composable_quantizer.py`
- **Size**: 3,012 bytes (2.94 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from .quantizer import QuantizationAnnotation, Quantizer


if TYPE_CHECKING:
    import torch
    from torch.fx import Node

__all__ = [
    "ComposableQuantizer",
]


class ComposableQuantizer(Quantizer):
    """
    ComposableQuantizer allows users to combine more than one quantizer into a single quantizer.
    This allows users to quantize a model with multiple quantizers. E.g., embedding quantization
    maybe supported by one quantizer while linear layers and other ops might be supported by another
    quantizer.

    ComposableQuantizer is initialized with a list of `Quantizer` instances.
    The order of the composition matters since that is the order in which the quantizers will be
    applies.
    Example:
    ```
    embedding_quantizer = EmbeddingQuantizer()
    linear_quantizer = MyLinearQuantizer()
    xnnpack_quantizer = (
        XNNPackQuantizer()
    )  # to handle ops not quantized by previous two quantizers
    composed_quantizer = ComposableQuantizer(
        [embedding_quantizer, linear_quantizer, xnnpack_quantizer]
    )
    prepared_m = prepare_pt2e(model, composed_quantizer)
    ```
    """

    def __init__(self, quantizers: list[Quantizer]):
        super().__init__()
        self.quantizers = quantizers
        self._graph_annotations: dict[Node, QuantizationAnnotation] = {}

    def _record_and_validate_annotations(
        self, gm: torch.fx.GraphModule, quantizer: Quantizer
    ) -> None:
        for n in gm.graph.nodes:
            if "quantization_annotation" in n.meta:
                # check if the annotation has been changed by
                # comparing QuantizationAnnotation object id
                if n in self._graph_annotations and (
                    id(self._graph_annotations[n])
                    != id(n.meta["quantization_annotation"])
                ):
                    raise RuntimeError(
                        f"Quantizer {quantizer.__class__.__name__} has changed annotations on node {n}"
                    )
                else:
                    self._graph_annotations[n] = n.meta["quantization_annotation"]
            else:
                if n in self._graph_annotations:
                    raise RuntimeError(
                        f"Quantizer {quantizer.__class__.__name__} has removed annotations on node {n}"
                    )

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        for quantizer in self.quantizers:
            quantizer.annotate(model)
            self._record_and_validate_annotations(model, quantizer)
        return model

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        for quantizer in self.quantizers:
            model = quantizer.transform_for_annotation(model)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

```



## High-Level Overview

"""    ComposableQuantizer allows users to combine more than one quantizer into a single quantizer.    This allows users to quantize a model with multiple quantizers. E.g., embedding quantization    maybe supported by one quantizer while linear layers and other ops might be supported by another    quantizer.    ComposableQuantizer is initialized with a list of `Quantizer` instances.    The order of the composition matters since that is the order in which the quantizers will be    applies.    Example:    ```    embedding_quantizer = EmbeddingQuantizer()    linear_quantizer = MyLinearQuantizer()    xnnpack_quantizer = (        XNNPackQuantizer()    )  # to handle ops not quantized by previous two quantizers    composed_quantizer = ComposableQuantizer(        [embedding_quantizer, linear_quantizer, xnnpack_quantizer]    )    prepared_m = prepare_pt2e(model, composed_quantizer)    ```

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ComposableQuantizer`

**Functions defined**: `__init__`, `_record_and_validate_annotations`, `annotate`, `transform_for_annotation`, `validate`

**Key imports**: annotations, TYPE_CHECKING, QuantizationAnnotation, Quantizer, torch, Node


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/quantizer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TYPE_CHECKING
- `.quantizer`: QuantizationAnnotation, Quantizer
- `torch`
- `torch.fx`: Node


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/ao/quantization/quantizer`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`x86_inductor_quantizer.py_docs.md`](./x86_inductor_quantizer.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`xnnpack_quantizer_utils.py_docs.md`](./xnnpack_quantizer_utils.py_docs.md)
- [`xpu_inductor_quantizer.py_docs.md`](./xpu_inductor_quantizer.py_docs.md)
- [`quantizer.py_docs.md`](./quantizer.py_docs.md)
- [`xnnpack_quantizer.py_docs.md`](./xnnpack_quantizer.py_docs.md)
- [`embedding_quantizer.py_docs.md`](./embedding_quantizer.py_docs.md)


## Cross-References

- **File Documentation**: `composable_quantizer.py_docs.md`
- **Keyword Index**: `composable_quantizer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/quantizer`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/quantizer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/ao/quantization/quantizer`):

- [`xpu_inductor_quantizer.py_docs.md_docs.md`](./xpu_inductor_quantizer.py_docs.md_docs.md)
- [`xnnpack_quantizer_utils.py_kw.md_docs.md`](./xnnpack_quantizer_utils.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`x86_inductor_quantizer.py_kw.md_docs.md`](./x86_inductor_quantizer.py_kw.md_docs.md)
- [`embedding_quantizer.py_kw.md_docs.md`](./embedding_quantizer.py_kw.md_docs.md)
- [`embedding_quantizer.py_docs.md_docs.md`](./embedding_quantizer.py_docs.md_docs.md)
- [`xnnpack_quantizer_utils.py_docs.md_docs.md`](./xnnpack_quantizer_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`composable_quantizer.py_kw.md_docs.md`](./composable_quantizer.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `composable_quantizer.py_docs.md_docs.md`
- **Keyword Index**: `composable_quantizer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
