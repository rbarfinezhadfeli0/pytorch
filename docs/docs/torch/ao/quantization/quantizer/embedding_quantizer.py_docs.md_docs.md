# Documentation: `docs/torch/ao/quantization/quantizer/embedding_quantizer.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/quantizer/embedding_quantizer.py_docs.md`
- **Size**: 6,323 bytes (6.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/quantizer/embedding_quantizer.py`

## File Metadata

- **Path**: `torch/ao/quantization/quantizer/embedding_quantizer.py`
- **Size**: 3,457 bytes (3.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OperatorConfig,
    OperatorPatternType,
    QuantizationConfig,
)


__all__ = [
    "get_embedding_operators_config",
    "EmbeddingQuantizer",
]


def get_embedding_operators_config() -> OperatorConfig:
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=torch.per_channel_affine_float_qparams,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2**-12),
    )
    quantization_config = QuantizationConfig(None, None, weight_quantization_spec, None)
    ops: list[OperatorPatternType] = [[torch.nn.Embedding]]
    ops.append([F.embedding])
    supported_config_and_operators = OperatorConfig(
        config=quantization_config, operators=ops
    )
    return copy.deepcopy(supported_config_and_operators)


class EmbeddingQuantizer(Quantizer):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_supported_quantization_configs(cls) -> list[QuantizationConfig]:
        op_configs: set[QuantizationConfig] = {
            spec for spec, _ in cls.get_supported_operators()
        }
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: QuantizationConfig
    ) -> list[OperatorPatternType]:
        for config, ops in cls.get_supported_operators():
            # note: this assumes each entry in cls.supported_spec_and_operators
            # corresponds to one spec, e.g. we don't have
            # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
            # where the first and second entry have the same spec but did not
            # merge the op list
            if config == quantization_config:
                return ops
        return []

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        self._annotate_embedding_ops(model.graph)
        return model

    def _annotate_embedding_ops(self, graph: torch.fx.Graph) -> None:
        embedding_config: OperatorConfig = get_embedding_operators_config()
        for node in graph.nodes:
            # Keep node parsing based annotations instead of module partitioners
            # just as an example of alternate ways of annotating
            if (
                node.op == "call_function"
                and node.target is torch.ops.aten.embedding.default
            ):
                if embedding_config.config.weight is None:
                    raise ValueError(
                        "Embedding config must have a valid weight quantization spec."
                    )
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                        node.args[0]: embedding_config.config.weight,
                    }
                )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> list[OperatorConfig]:
        return [get_embedding_operators_config()]

```



## High-Level Overview


This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EmbeddingQuantizer`

**Functions defined**: `get_embedding_operators_config`, `__init__`, `get_supported_quantization_configs`, `get_supported_operator_for_quantization_config`, `annotate`, `_annotate_embedding_ops`, `validate`, `get_supported_operators`

**Key imports**: annotations, copy, torch, torch.nn.functional as F, PerChannelMinMaxObserver


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/quantizer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `copy`
- `torch`
- `torch.nn.functional as F`
- `torch.ao.quantization.observer`: PerChannelMinMaxObserver


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
- [`composable_quantizer.py_docs.md`](./composable_quantizer.py_docs.md)
- [`xnnpack_quantizer.py_docs.md`](./xnnpack_quantizer.py_docs.md)


## Cross-References

- **File Documentation**: `embedding_quantizer.py_docs.md`
- **Keyword Index**: `embedding_quantizer.py_kw.md`
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
- **Neural Network**: Defines or uses PyTorch neural network components


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
- [`composable_quantizer.py_docs.md_docs.md`](./composable_quantizer.py_docs.md_docs.md)
- [`xnnpack_quantizer_utils.py_docs.md_docs.md`](./xnnpack_quantizer_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`composable_quantizer.py_kw.md_docs.md`](./composable_quantizer.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `embedding_quantizer.py_docs.md_docs.md`
- **Keyword Index**: `embedding_quantizer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
