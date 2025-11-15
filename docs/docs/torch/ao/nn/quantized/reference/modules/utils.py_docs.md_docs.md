# Documentation: `docs/torch/ao/nn/quantized/reference/modules/utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/nn/quantized/reference/modules/utils.py_docs.md`
- **Size**: 18,015 bytes (17.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/nn/quantized/reference/modules/utils.py`

## File Metadata

- **Path**: `torch/ao/nn/quantized/reference/modules/utils.py`
- **Size**: 15,586 bytes (15.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import typing

import torch


__all__ = [
    "ReferenceQuantizedModule",
]


class ReferenceQuantizedModule(torch.nn.Module):
    def _init_weight_qparams(self, weight_qparams, device):
        if weight_qparams is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0,
            }
        # pyrefly: ignore [bad-assignment]
        self.weight_qscheme: torch.qscheme = weight_qparams["qscheme"]
        self.weight_dtype = weight_qparams["dtype"]
        assert self.weight_qscheme in [
            None,
            torch.per_tensor_affine,
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams,
        ], (
            f"qscheme: {self.weight_qscheme} is not support in reference quantized {self._get_name()}"
        )
        if self.weight_dtype in [
            torch.quint8,
            torch.qint8,
            torch.quint4x2,
            torch.qint32,
        ]:
            zero_point_dtype = (
                weight_qparams["zero_point"].dtype
                if isinstance(weight_qparams["zero_point"], torch.Tensor)
                else torch.int
            )
            w_scale = weight_qparams["scale"]
            w_scale_tensor = (
                w_scale.detach().clone()
                if isinstance(w_scale, torch.Tensor)
                else torch.tensor(w_scale, dtype=torch.float, device=device)
            )
            self.register_buffer("weight_scale", w_scale_tensor)
            w_zp = weight_qparams["zero_point"]
            w_zp_tensor = (
                w_zp.detach().clone()
                if isinstance(w_zp, torch.Tensor)
                else torch.tensor(w_zp, dtype=zero_point_dtype, device=device)
            )
            self.register_buffer("weight_zero_point", w_zp_tensor)
            if self.weight_qscheme in [
                torch.per_channel_affine,
                torch.per_channel_affine_float_qparams,
            ]:
                w_axis = weight_qparams["axis"]
                w_axis_tensor = (
                    w_axis.detach().clone()
                    if isinstance(w_axis, torch.Tensor)
                    else torch.tensor(w_axis, dtype=torch.int, device=device)
                )
                self.register_buffer("weight_axis", w_axis_tensor)
            else:
                # added for TorchScriptability, not used
                self.register_buffer(
                    "weight_axis", torch.tensor(0, dtype=torch.int, device=device)
                )
        else:
            # added for TorchScriptability, and for torch.float
            self.register_buffer(
                "weight_scale", torch.tensor(1.0, dtype=torch.float, device=device)
            )
            self.register_buffer(
                "weight_zero_point", torch.tensor(0, dtype=torch.int, device=device)
            )
            self.register_buffer(
                "weight_axis", torch.tensor(0, dtype=torch.int, device=device)
            )
        # pyrefly: ignore [bad-assignment]
        self.is_decomposed: bool = weight_qparams.get("is_decomposed", False)
        # store weight_axis as weight_axis_int due to some constraints of torchdynamo.export
        # for capturing `.item` operations
        self.weight_axis_int: int = self.weight_axis.item()  # type: ignore[operator, assignment]
        # pyrefly: ignore [bad-assignment]
        self.weight_quant_min: int | None = weight_qparams.get("quant_min")
        # pyrefly: ignore [bad-assignment]
        self.weight_quant_max: int | None = weight_qparams.get("quant_max")

    def get_weight(self):
        """
        Fake quantize (quantize and dequantize) the weight with
        the quantization parameters for weight, this is used to
        simulate the numerics for the quantized weight in a quantized
        model
        """
        # suppress mypy warning
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        if self.is_decomposed:
            return _quantize_and_dequantize_weight_decomposed(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                # pyrefly: ignore [bad-argument-type]
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max,
            )
        else:
            return _quantize_and_dequantize_weight(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                # pyrefly: ignore [bad-argument-type]
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
            )

    def get_quantized_weight(self):
        # suppress mypy warning
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        # assert isinstance(self.weight_axis, torch.Tensor)
        if self.is_decomposed:
            return _quantize_weight_decomposed(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                # pyrefly: ignore [bad-argument-type]
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max,
            )
        else:
            return _quantize_weight(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                # pyrefly: ignore [bad-argument-type]
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        _save_weight_qparams(
            destination,
            prefix,
            self.weight_qscheme,
            self.weight_dtype,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_axis,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for key in _get_weight_qparam_keys(state_dict, prefix):
            setattr(self, key, state_dict[prefix + key])
            state_dict.pop(prefix + key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _quantize_weight_decomposed(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> torch.Tensor:
    _DTYPE_TO_QVALUE_BOUNDS: dict[torch.dtype, tuple[int, int]] = {
        torch.uint8: (0, 255),
        torch.int8: (-128, 127),
        torch.int32: ((-(2**31)), (2**31 - 1)),
    }

    # TODO: add an util function for converting qdtype to dtype
    _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }
    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[
                    weight_dtype_
                ]
            weight = torch.ops.quantized_decomposed.quantize_per_tensor(
                weight,
                weight_scale,
                weight_zero_point,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )
            return weight
    elif weight_qscheme in [
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        # TODO: torch.quint4x2 is not supported
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[
                    weight_dtype_
                ]
            weight = torch.ops.quantized_decomposed.quantize_per_channel(
                weight,
                weight_scale,
                weight_zero_point,
                weight_axis,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _dequantize_weight_decomposed(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> torch.Tensor:
    # TODO: get the quant_min and quant_max from activation_post_process
    _DTYPE_TO_QVALUE_BOUNDS: dict[torch.dtype, tuple[int, int]] = {
        torch.uint8: (0, 255),
        torch.int8: (-128, 127),
        torch.int32: ((-(2**31)), (2**31 - 1)),
    }
    # TODO: add an util function for converting qdtype to dtype
    _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }
    weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
    if weight_quant_min is None or weight_quant_max is None:
        weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype_]
    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.ops.quantized_decomposed.dequantize_per_tensor(
                weight,
                weight_scale,
                weight_zero_point,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )
            return weight
    elif weight_qscheme in [
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        # TODO: torch.quint4x2 is not supported
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.ops.quantized_decomposed.dequantize_per_channel(
                weight,
                weight_scale,
                weight_zero_point,
                weight_axis,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_,
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _quantize_weight(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis_int: int,
) -> torch.Tensor:
    if weight_dtype == torch.float16:
        weight = weight.to(weight_dtype)
        return weight

    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.quantize_per_tensor(
                weight, weight_scale, weight_zero_point, weight_dtype
            )
            return weight
    elif weight_qscheme in [
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        if weight_dtype in [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32]:
            weight = torch.quantize_per_channel(
                weight, weight_scale, weight_zero_point, weight_axis_int, weight_dtype
            )  # type: ignore[arg-type]
            return weight
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _quantize_and_dequantize_weight_decomposed(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis_int: int,
    weight_quant_min: int | None,
    weight_quant_max: int | None,
) -> torch.Tensor:
    """Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [
        torch.per_tensor_affine,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        weight_quant = _quantize_weight_decomposed(
            weight,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
            weight_quant_min,
            weight_quant_max,
        )
        weight_dequant = _dequantize_weight_decomposed(
            weight_quant,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
            weight_quant_min,
            weight_quant_max,
        )
    else:
        weight_dequant = weight
    return weight_dequant


def _quantize_and_dequantize_weight(
    weight: torch.Tensor,
    weight_qscheme: torch.qscheme,
    weight_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_axis_int: int,
) -> torch.Tensor:
    """Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [
        torch.per_tensor_affine,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]:
        weight_quant = _quantize_weight(
            weight,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
        )
        weight_dequant = weight_quant.dequantize()
    else:
        weight_dequant = weight
    return weight_dequant


def _save_weight_qparams(
    destination,
    prefix,
    weight_qscheme,
    weight_dtype,
    weight_scale,
    weight_zero_point,
    weight_axis,
):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis


def _get_weight_qparam_keys(state_dict: dict[str, typing.Any], prefix: str):
    keys = ["weight_qscheme", "weight_dtype"]
    weight_qscheme = state_dict[prefix + "weight_qscheme"]
    if weight_qscheme is not None:
        keys.append("weight_scale")
        keys.append("weight_zero_point")
        if weight_qscheme == torch.quantize_per_channel:
            keys.append("weight_axis")
    return keys

```



## High-Level Overview


This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ReferenceQuantizedModule`

**Functions defined**: `_init_weight_qparams`, `get_weight`, `get_quantized_weight`, `_save_to_state_dict`, `_load_from_state_dict`, `_quantize_weight_decomposed`, `_dequantize_weight_decomposed`, `_quantize_weight`, `_quantize_and_dequantize_weight_decomposed`, `_quantize_and_dequantize_weight`, `_save_weight_qparams`, `_get_weight_qparam_keys`

**Key imports**: typing, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/quantized/reference/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`
- `torch`


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`torch/ao/nn/quantized/reference/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`sparse.py_docs.md`](./sparse.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/nn/quantized/reference/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/nn/quantized/reference/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`docs/torch/ao/nn/quantized/reference/modules`):

- [`sparse.py_docs.md_docs.md`](./sparse.py_docs.md_docs.md)
- [`sparse.py_kw.md_docs.md`](./sparse.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`rnn.py_docs.md_docs.md`](./rnn.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`linear.py_docs.md_docs.md`](./linear.py_docs.md_docs.md)
- [`conv.py_kw.md_docs.md`](./conv.py_kw.md_docs.md)
- [`linear.py_kw.md_docs.md`](./linear.py_kw.md_docs.md)
- [`conv.py_docs.md_docs.md`](./conv.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md_docs.md`
- **Keyword Index**: `utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
