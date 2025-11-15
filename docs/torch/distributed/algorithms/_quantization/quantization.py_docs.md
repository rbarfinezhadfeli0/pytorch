# Documentation: `torch/distributed/algorithms/_quantization/quantization.py`

## File Metadata

- **Path**: `torch/distributed/algorithms/_quantization/quantization.py`
- **Size**: 5,646 bytes (5.51 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools
from enum import Enum

import torch
import torch.distributed as dist


TORCH_HALF_MIN = torch.finfo(torch.float16).min
TORCH_HALF_MAX = torch.finfo(torch.float16).max


class DQuantType(Enum):
    """
    Different quantization methods for auto_quantize API are identified here.

    auto_quantize API currently supports fp16 and bfp16 methods.
    """

    FP16 = ("fp16",)
    BFP16 = "bfp16"

    def __str__(self) -> str:
        return self.value


def _fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, TORCH_HALF_MIN, TORCH_HALF_MAX).half()


def _quantize_tensor(tensor, qtype):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_quantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        return _fp32_to_fp16_with_clamp(tensor)
    elif qtype == DQuantType.BFP16:
        return torch.ops.quantization._FloatToBfloat16Quantized(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


def _quantize_tensor_list(tensor_list, qtype):
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_quantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    quantized_tensor_list = [_quantize_tensor(t, qtype) for t in tensor_list]
    return quantized_tensor_list


def _dequantize_tensor(tensor, qtype, quant_loss=None):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_dequantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        elif tensor.dtype == torch.float16 and quant_loss is None:
            return tensor.float()
        else:
            # pyrefly: ignore [unsupported-operation]
            return tensor.float() / quant_loss
    elif qtype == DQuantType.BFP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        else:
            return torch.ops.quantization._Bfloat16QuantizedToFloat(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


def _dequantize_tensor_list(tensor_list, qtype, quant_loss=None):
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_dequantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    dequantized_tensor_list = [_dequantize_tensor(t, qtype) for t in tensor_list]
    return dequantized_tensor_list


def auto_quantize(func, qtype, quant_loss=None):
    """
    Quantize the input tensors, choose the precision types, and pass other necessary arguments and then dequantizes the output.

    Currently it only supports:
        . FP16 and BFP16 quantization method supported for gloo and nccl backends
        . all_gather, all_to_all collective ops
    Note: BFP16 only supports 2D tensors.
    Args:
        func (Callable): A function representing collective operations.
        qtype (QuantType): Quantization method
        quant_loss (float, optional): This can be used to improve accuracy in the dequantization.
    Returns:
        (Callable): the same collective as func but enables automatic quantization/dequantization.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        group = kwargs.get("group")
        async_op = kwargs.get("async_op", False)
        if async_op is True:
            raise RuntimeError("The async_op=True mode is not supported yet.")
        if func is dist.all_gather:
            tensors = args[0]
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_gather(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        elif func is dist.all_to_all:
            tensors = args[0]
            input_tensors = _quantize_tensor_list(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_to_all(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        elif func is dist.all_to_all_single:
            tensors = args[0]
            out_splits = kwargs.get("out_splits")
            in_splits = kwargs.get("in_splits")
            # Quantizing the input/output tensor
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor(tensors, qtype)
            dist.all_to_all_single(
                out_tensors, input_tensors, out_splits, in_splits, group=group
            )
            for i, t in enumerate(
                _dequantize_tensor(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t
        else:
            raise RuntimeError(f"The collective op {func} is not supported yet")

    return wrapper

```



## High-Level Overview

"""    Different quantization methods for auto_quantize API are identified here.    auto_quantize API currently supports fp16 and bfp16 methods.

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DQuantType`

**Functions defined**: `__str__`, `_fp32_to_fp16_with_clamp`, `_quantize_tensor`, `_quantize_tensor_list`, `_dequantize_tensor`, `_dequantize_tensor_list`, `auto_quantize`, `wrapper`

**Key imports**: functools, Enum, torch, torch.distributed as dist


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/algorithms/_quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `enum`: Enum
- `torch`
- `torch.distributed as dist`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/distributed/algorithms/_quantization`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `quantization.py_docs.md`
- **Keyword Index**: `quantization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
