# Documentation: `benchmarks/dynamo/torchao_backend.py`

## File Metadata

- **Path**: `benchmarks/dynamo/torchao_backend.py`
- **Size**: 2,250 bytes (2.20 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
from collections.abc import Callable
from typing import Any

import torch


def setup_baseline():
    from torchao.quantization.utils import recommended_inductor_config_setter

    recommended_inductor_config_setter()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.recompile_limit = 10000


def torchao_optimize_ctx(quantization: str):
    from torchao.quantization.quant_api import (
        autoquant,
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    from torchao.utils import unwrap_tensor_subclass

    def inner(model_iter_fn: Callable):
        def _torchao_apply(module: torch.nn.Module, example_inputs: Any):
            if getattr(module, "_quantized", None) is None:
                if quantization == "int8dynamic":
                    quantize_(
                        module,
                        int8_dynamic_activation_int8_weight(),
                        set_inductor_config=False,
                    )
                elif quantization == "int8weightonly":
                    quantize_(module, int8_weight_only(), set_inductor_config=False)
                elif quantization == "int4weightonly":
                    quantize_(module, int4_weight_only(), set_inductor_config=False)
                if quantization == "autoquant":
                    autoquant(module, error_on_unseen=False, set_inductor_config=False)
                    if isinstance(example_inputs, dict):
                        module(**example_inputs)
                    else:
                        module(*example_inputs)
                    from torchao.quantization.autoquant import AUTOQUANT_CACHE

                    if len(AUTOQUANT_CACHE) == 0:
                        raise Exception(  # noqa: TRY002
                            "NotAutoquantizable"
                            f"Found no autoquantizable layers in model {type(module)}, stopping autoquantized run"
                        )
                else:
                    unwrap_tensor_subclass(module)
                setattr(module, "_quantized", True)  # noqa: B010
            model_iter_fn(module, example_inputs)

        return _torchao_apply

    return inner

```



## High-Level Overview


This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `setup_baseline`, `torchao_optimize_ctx`, `inner`, `_torchao_apply`

**Key imports**: Callable, Any, torch, recommended_inductor_config_setter, unwrap_tensor_subclass, AUTOQUANT_CACHE


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Any
- `torch`
- `torchao.quantization.utils`: recommended_inductor_config_setter
- `torchao.utils`: unwrap_tensor_subclass
- `torchao.quantization.autoquant`: AUTOQUANT_CACHE


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`benchmarks/dynamo`):

- [`timm_models_list_cpu.txt_docs.md`](./timm_models_list_cpu.txt_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`benchmarks.py_docs.md`](./benchmarks.py_docs.md)
- [`check_graph_breaks.py_docs.md`](./check_graph_breaks.py_docs.md)
- [`check_csv.py_docs.md`](./check_csv.py_docs.md)
- [`all_torchbench_models_list.txt_docs.md`](./all_torchbench_models_list.txt_docs.md)
- [`check_accuracy.py_docs.md`](./check_accuracy.py_docs.md)
- [`torchbench_models_list_cpu.txt_docs.md`](./torchbench_models_list_cpu.txt_docs.md)
- [`timm_models.py_docs.md`](./timm_models.py_docs.md)


## Cross-References

- **File Documentation**: `torchao_backend.py_docs.md`
- **Keyword Index**: `torchao_backend.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
