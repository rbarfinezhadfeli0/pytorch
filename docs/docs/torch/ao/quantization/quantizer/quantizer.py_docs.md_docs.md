# Documentation: `docs/torch/ao/quantization/quantizer/quantizer.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/quantizer/quantizer.py_docs.md`
- **Size**: 9,762 bytes (9.53 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/quantizer/quantizer.py`

## File Metadata

- **Path**: `torch/ao/quantization/quantizer/quantizer.py`
- **Size**: 6,617 bytes (6.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated

import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node


__all__ = [
    "Quantizer",
    "QuantizationSpecBase",
    "QuantizationSpec",
    "FixedQParamsQuantizationSpec",
    "EdgeOrNode",
    "SharedQuantizationSpec",
    "DerivedQuantizationSpec",
    "QuantizationAnnotation",
]


class QuantizationSpecBase(ABC):  # noqa: B024
    """Base class for different types of quantization specs that allows users to
    specify how to quantize a Tensor (input/output of a Node) in the model
    """


@dataclass(eq=True, frozen=True)
class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, quant_min, quant_max etc.
    """

    dtype: torch.dtype
    # observer or fake_quantize constructor such as
    # MinMaxObserver, PerChannelHistogramObserver etc.
    # or we can attach some custom args to them
    # e.g. MinMaxObserver.with_args(eps=eps)
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor
    quant_min: int | None = None
    quant_max: int | None = None
    qscheme: torch.qscheme | None = None
    ch_axis: int | None = None
    is_dynamic: bool = False

    def __post_init__(self):
        # TODO: add init for quant_min/quant_max
        # quant_min must be less than quant_max
        if (
            self.quant_min is not None
            and self.quant_max is not None
            and self.quant_min > self.quant_max
        ):
            raise ValueError(
                f"quant_min {self.quant_min} must be <= quant_max {self.quant_max}."
            )

        # ch_axis must be less than the number of channels
        # but no way to check here. Just check that it is not < 0.
        if self.ch_axis is not None and self.ch_axis < 0:
            raise ValueError("Ch_axis is < 0.")


@dataclass(eq=True, frozen=True)
class FixedQParamsQuantizationSpec(QuantizationSpecBase):
    dtype: torch.dtype
    scale: float
    zero_point: int
    quant_min: int | None = None
    quant_max: int | None = None
    qscheme: torch.qscheme | None = None
    is_dynamic: bool = False


"""
The way we refer to other points of quantization in the graph will be either
an input edge or an output value
input edge is the connection between input node and the node consuming the input, so it's a Tuple[Node, Node]
output value is an fx Node
"""
EdgeOrNode = Annotated[tuple[Node, Node] | Node, None]
EdgeOrNode.__module__ = "torch.ao.quantization.quantizer.quantizer"


@dataclass(eq=True, frozen=True)
class SharedQuantizationSpec(QuantizationSpecBase):
    """
    Quantization spec for the Tensors whose quantization parameters are shared with other Tensors
    """

    # the edge or node to share observer or fake quant instances with
    edge_or_node: EdgeOrNode


@dataclass(eq=True, frozen=True)
class DerivedQuantizationSpec(QuantizationSpecBase):
    """Quantization spec for the Tensors whose quantization parameters are derived from other Tensors"""

    derived_from: list[EdgeOrNode]
    derive_qparams_fn: Callable[[list[ObserverOrFakeQuantize]], tuple[Tensor, Tensor]]
    dtype: torch.dtype
    quant_min: int | None = None
    quant_max: int | None = None
    qscheme: torch.qscheme | None = None
    ch_axis: int | None = None
    is_dynamic: bool = False


@dataclass
class QuantizationAnnotation:
    """How are input argument or output should be quantized,
    expressed as QuantizationSpec, this corresponds to how a Tensor in the
    operator Graph is observed (PTQ) or fake quantized (QAT)
    """

    # a map from torch.fx.Node to a type of QuantizationSpecBase
    input_qspec_map: dict[Node, QuantizationSpecBase | None] = field(
        default_factory=dict
    )

    # How the output of this node is quantized, expressed as QuantizationSpec
    # TODO: change the value to QuantizationSpec in a separate PR
    output_qspec: QuantizationSpecBase | None = None

    # For a Node: node1 and edge: (node1, node2), since they are observing the same
    # Tensor, we may want to implicitly share observers, this flag allows people to
    # turn off this behavior for the output of the node
    allow_implicit_sharing: bool = True

    # whether the node is annotated or not
    _annotated: bool = False


class Quantizer(ABC):
    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Allows for user defined transforms to run before annotating the graph.
        This allows quantizer to allow quantizing part of the model that are otherwise not quantizable.
        For example quantizer can
        a) decompose a compound operator like scaled dot product attention,
        into bmm and softmax if quantizer knows how to quantize bmm/softmax but not sdpa
        or b) transform scalars to tensor to allow quantizing scalares.

        Note: this is an optional method
        """
        return model

    # annotate nodes in the graph with observer or fake quant constructors
    # to convey the desired way of quantization
    @abstractmethod
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        pass

    # validate the annotated graph is supported by the backend
    @abstractmethod
    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    def prepare_obs_or_fq_callback(
        self,
        model: torch.fx.GraphModule,
        edge_or_node_to_obs_or_fq: dict[EdgeOrNode, ObserverOrFakeQuantize],
    ) -> None:
        """A callback that will be called after the observers or fake quants are created
        for each sharing group, but before they are inserted into the graph. The
        callback can be used to make final quantization adjustments, such as enforcing
        specific scale and zero point on model input or output.

        Args:
          * `model`: the graph module being prepared.
          * `edge_or_node_to_obs_or_fq`: a dictionary mapping each annotated edge and
            node to the corresponding observer or fake quant object. Note that multiple
            edges and/or nodes can map to the same observer / fake quant instance if
            they were annotated with SharedQuantizationSpec. This dictionary can be
            modified by the callback.
        """
        return

```



## High-Level Overview

"""Base class for different types of quantization specs that allows users to    specify how to quantize a Tensor (input/output of a Node) in the model

This Python file contains 8 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QuantizationSpecBase`, `QuantizationSpec`, `FixedQParamsQuantizationSpec`, `SharedQuantizationSpec`, `DerivedQuantizationSpec`, `QuantizationAnnotation`, `Quantizer`

**Functions defined**: `__post_init__`, `transform_for_annotation`, `annotate`, `validate`, `prepare_obs_or_fq_callback`

**Key imports**: ABC, abstractmethod, Callable, dataclass, field, Annotated, torch, Tensor, ObserverOrFakeQuantize, _ObserverOrFakeQuantizeConstructor, Node


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/quantizer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`: ABC, abstractmethod
- `collections.abc`: Callable
- `dataclasses`: dataclass, field
- `typing`: Annotated
- `torch`
- `torch.ao.quantization`: ObserverOrFakeQuantize
- `torch.ao.quantization.qconfig`: _ObserverOrFakeQuantizeConstructor
- `torch.fx`: Node


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


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
- [`composable_quantizer.py_docs.md`](./composable_quantizer.py_docs.md)
- [`xnnpack_quantizer.py_docs.md`](./xnnpack_quantizer.py_docs.md)
- [`embedding_quantizer.py_docs.md`](./embedding_quantizer.py_docs.md)


## Cross-References

- **File Documentation**: `quantizer.py_docs.md`
- **Keyword Index**: `quantizer.py_kw.md`
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

- **Abstract Base Classes**: Defines abstract interfaces


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
- [`composable_quantizer.py_docs.md_docs.md`](./composable_quantizer.py_docs.md_docs.md)
- [`xnnpack_quantizer_utils.py_docs.md_docs.md`](./xnnpack_quantizer_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`composable_quantizer.py_kw.md_docs.md`](./composable_quantizer.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `quantizer.py_docs.md_docs.md`
- **Keyword Index**: `quantizer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
