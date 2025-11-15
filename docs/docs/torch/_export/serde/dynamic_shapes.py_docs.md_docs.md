# Documentation: `docs/torch/_export/serde/dynamic_shapes.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/serde/dynamic_shapes.py_docs.md`
- **Size**: 14,462 bytes (14.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_export/serde/dynamic_shapes.py`

## File Metadata

- **Path**: `torch/_export/serde/dynamic_shapes.py`
- **Size**: 11,653 bytes (11.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import dataclasses
from typing import Any, Optional, Union

import torch
from torch._dynamo.exc import UserError, UserErrorType
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _DerivedDim,
    _DimHint,
    _tree_map_with_path,
    Dim,
)
from torch.utils._pytree import tree_map

from .serialize import _dataclass_to_dict


@dataclasses.dataclass
class RootDim:
    """
    This represents a Dim object.
    """

    min: int
    max: Union[int, None]
    derived: list[str]


@dataclasses.dataclass
class DynamicShapesSpec:
    """
    This stores a dynamic_shapes spec for de/serialization.
    """

    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None]
    dims: dict[str, RootDim]


def _postprocess_serialized_shapes(
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
    dims: dict[str, dict[str, Union[int, list[str], None]]],
    to_dict: Optional[bool] = False,
) -> Union[DynamicShapesSpec, dict[str, Any]]:
    """
    Sorts dims and dumps to dictionary format.
    """
    from torch.utils._sympy.numbers import int_oo

    dims = {
        k: RootDim(
            min=v["min"],  # type: ignore[arg-type]
            max=None if v["max"] is int_oo else v["max"],  # type: ignore[arg-type]
            derived=sorted(v["derived"]),  # type: ignore[arg-type]
        )
        for k, v in sorted(dims.items())
    }
    # pyrefly: ignore [bad-argument-type]
    spec = DynamicShapesSpec(dynamic_shapes=dynamic_shapes, dims=dims)
    if to_dict:
        return _dataclass_to_dict(spec)
    else:
        return spec


def _dump_dynamic_shapes(
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
    args: tuple[Any],
    kwargs: Optional[dict[str, Any]] = None,
    to_dict: Optional[bool] = False,
) -> Union[DynamicShapesSpec, dict[str, Any]]:
    """
    Utility function for dynamic shapes serialization, serializing a dynamic_shapes spec.
    Returns a DynamicShapesSpec dataclass containing 2 fields, "dynamic_shapes" and "dims".
    Uses args & kwargs to distinguish between tensor-level and dim-level specs (only for Nones).

    dynamic_shapes: A pytree structure mirroring the dynamic_shapes input to export():
        - Each tensor input is represented with a list of values, non-tensor inputs with None.
        - dynamic dimensions (i.e. symbols) in tensors and Dim enums are represented with strings.
        - static dimensions are represented with ints.

    dims: A dictionary mapping each symbol name to the min/max range and derived dim names.

    For example:
    ```
    dx = Dim("dx", min=4, max=16)
    dy = dx + 1

    inputs = (
        [
            torch.randn(4, 4),
            torch.randn(5, 4),
        ],
        torch.randn(4),
        torch.randn(4, 4),
        "hello",
    )
    dynamic_shapes = {
        "a": [
            (dx, 4),
            (dy, 4),
        ],
        "b": (Dim.STATIC,),
        "c": None,
        "d": None,
    }
    out = _dump_dynamic_shapes(dynamic_shapes, inputs, to_dict=True)
    ```
    would generate the following output:
    ```
    {
        "dynamic_shapes": (
            [
                ["dx", 4],
                ["dx + 1", 4],
            ],
            ["_DimHint.STATIC"],
            ["_DimHint.STATIC", "_DimHint.STATIC"],
            None,
        ),
        "dims": {
            "dx": {
                "min": 4,
                "max": 16,
                "derived": ["dx + 1"],
            },
        },
    }
    ```
    """
    dims: dict[str, dict[str, Any]] = {}

    def _standardize_shapes(path, tensor, shape):  # type: ignore[no-untyped-def]
        """
        Helps standardize the dynamic_shapes tree structure we serialize,
        returning lists for each tensor shape, handling tensor-level Nones.
        """
        if not isinstance(tensor, torch.Tensor):
            return None
        if shape is None:
            return [Dim.STATIC] * len(tensor.shape)

        out = []
        if isinstance(shape, dict):
            for i, s in enumerate(tensor.shape):
                out.append(s if shape.get(i) is None else shape.get(i))
        else:
            assert isinstance(shape, (tuple, list))
            for i, s in enumerate(tensor.shape):
                out.append(s if shape[i] is None else shape[i])
        return out

    def _track_dim_from_dims(
        val: Union[None, int, _DimHint, Dim],
    ) -> Union[None, int, str]:
        """
        Tracks dims, ranges, derived dims from the standardized dynamic_shapes spec.
        """
        if val is None or isinstance(val, int):  # non-tensor input or static
            return val
        if isinstance(val, _DimHint):  # store enum as string
            return val.__class__.__name__ + "." + val.type.name

        assert isinstance(val, Dim)

        # track root dim
        root = val.root if isinstance(val, _DerivedDim) else val  # type: ignore[attr-defined]
        if root.__name__ not in dims:
            dims[root.__name__] = {
                "min": root.min,  # type: ignore[attr-defined,union-attr]
                "max": root.max,  # type: ignore[attr-defined,union-attr]
                "derived": set(),
            }

        # track derived dims
        if isinstance(val, _DerivedDim):
            dims[root.__name__]["derived"].add(val.__name__)

        return val.__name__

    if dynamic_shapes is None:
        return {"dynamic_shapes": None, "dims": {}}

    # convert to tuple of specs, for each arg/kwarg
    kwargs = kwargs or {}
    if isinstance(dynamic_shapes, dict):
        dynamic_shapes = dynamic_shapes.values()  # type: ignore[assignment]
    # pyrefly: ignore [bad-assignment, bad-argument-type]
    dynamic_shapes = tuple(dynamic_shapes)
    combined_args = tuple(args) + tuple(kwargs.values())

    # run same check when we're processing shapes for export - is this too lazy?
    _check_dynamic_shapes(dict(enumerate(combined_args)), dynamic_shapes)  # type: ignore[arg-type]

    tree_shapes = _tree_map_with_path(
        _standardize_shapes, combined_args, dynamic_shapes, tree_name="inputs"
    )
    serialized_shapes = tree_map(_track_dim_from_dims, tree_shapes)
    return _postprocess_serialized_shapes(serialized_shapes, dims, to_dict=to_dict)


def _load_dynamic_shapes(
    spec: Union[DynamicShapesSpec, dict[str, Any]],
    from_dict: Optional[bool] = False,
) -> Union[dict[str, Any], tuple[Any], list[Any], None]:
    """
    Utility function for dynamic shapes serialization.
    Deserializes a DynamicShapesSpec or corresponding dictionary into a dynamic_shapes input to export().
    """
    import sympy

    from torch.fx.experimental.symbolic_shapes import _is_supported_equivalence

    if from_dict:
        if not isinstance(spec, dict):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"With from_dict=True, expected `spec` to be a dict, got {type(spec)}",
            )
        if sorted(spec.keys()) != ["dims", "dynamic_shapes"]:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "With from_dict=True, expected `spec` to have keys `dims` and `dynamic_shapes`, "
                f"instead found {spec.keys()}",
            )
        dims = {}
        for k, v in spec["dims"].items():
            if not isinstance(k, str):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected `spec['dims']` keys to be strings for symbols, got key {type(k)}",
                )
            if sorted(v.keys()) != ["derived", "max", "min"]:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected `spec['dims']` values to have keys `derived`, `max`, and `min`, "
                    f"instead found {v.keys()}",
                )
            if not isinstance(v["min"], int):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dims in `spec['dims']` to map `min` to an int, got {k}: {v['min']}",
                )
            if not isinstance(v["max"], int) or v["max"] is None:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dims in `spec['dims']` to map `max` to an int or None, got {k}: {v['max']}",
                )
            if not isinstance(v["derived"], list) or any(
                not isinstance(d, str) for d in v["derived"]
            ):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    "Expected dims in `spec['dims']` to map `derived` to a list of derived expressions, "
                    f"got {k}: {v['derived']}",
                )
            dims[k] = RootDim(**v)
        dynamic_shapes = spec["dynamic_shapes"]
    else:
        if not isinstance(spec, DynamicShapesSpec):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Expected `spec` to be a DynamicShapesSpec, got {type(spec)}",
            )
        dims = spec.dims
        dynamic_shapes = spec.dynamic_shapes

    if dynamic_shapes is None:
        return None

    dim_cache = {}
    for name, info in dims.items():
        symbol = sympy.sympify(name)
        if not isinstance(symbol, sympy.Symbol):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Expected `spec['dims']` keys to be symbols, got {name}",
            )
        dim_cache[name] = Dim(name, min=info.min, max=info.max)  # cache root dim
        for _expr in info.derived:
            expr = sympy.sympify(_expr)
            if len(expr.free_symbols) != 1 or symbol not in expr.free_symbols:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected derived expressions in to have {name} as the only free symbol, got {expr}",
                )
            if not _is_supported_equivalence(expr):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected derived expressions to be linear expressions, got {expr}",
                )
            modulus, remainder = sympy.polys.polytools.div(expr, symbol)
            ddim = dim_cache[name]
            if modulus != 1:
                ddim = int(modulus) * ddim  # type: ignore[assignment, operator]
            if remainder != 0:
                ddim = ddim + int(remainder)  # type: ignore[assignment, operator]
            dim_cache[_expr] = ddim  # cache derived dims

    def deserialize_shape(
        val: Union[None, int, str],
    ) -> Union[None, int, Dim, _DimHint]:
        if val is None or isinstance(val, int):
            return val
        elif val == "_DimHint.AUTO":
            return _DimHint.AUTO()
        elif val == "_DimHint.DYNAMIC":
            return _DimHint.DYNAMIC()
        elif val == "_DimHint.STATIC":
            return _DimHint.STATIC()
        if not isinstance(val, str):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Expected leaves in `spec['dynamic_shapes']` to be ints, None, Dim.AUTO/STATIC, symbols, "
                f" or derived expressions, got {val}",
            )
        if val not in dim_cache:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Expected dims in `spec['dynamic_shapes']` to be tracked in `spec['dims']`, "
                f"got {val} which is not in {dims.keys()}",
            )
        return dim_cache[val]  # type: ignore[return-value]

    return tree_map(deserialize_shape, dynamic_shapes)

```



## High-Level Overview

"""    This represents a Dim object.

This Python file contains 3 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RootDim`, `DynamicShapesSpec`

**Functions defined**: `_postprocess_serialized_shapes`, `_dump_dynamic_shapes`, `_standardize_shapes`, `_track_dim_from_dims`, `_load_dynamic_shapes`, `deserialize_shape`

**Key imports**: dataclasses, Any, Optional, Union, torch, UserError, UserErrorType, tree_map, _dataclass_to_dict, int_oo, sympy, _is_supported_equivalence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/serde`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `typing`: Any, Optional, Union
- `torch`
- `torch._dynamo.exc`: UserError, UserErrorType
- `torch.utils._pytree`: tree_map
- `.serialize`: _dataclass_to_dict
- `torch.utils._sympy.numbers`: int_oo
- `sympy`
- `torch.fx.experimental.symbolic_shapes`: _is_supported_equivalence


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/_export/serde`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`schema.yaml_docs.md`](./schema.yaml_docs.md)
- [`export_schema.thrift_docs.md`](./export_schema.thrift_docs.md)
- [`schema.py_docs.md`](./schema.py_docs.md)
- [`schema_check.py_docs.md`](./schema_check.py_docs.md)
- [`union.py_docs.md`](./union.py_docs.md)
- [`serialize.py_docs.md`](./serialize.py_docs.md)


## Cross-References

- **File Documentation**: `dynamic_shapes.py_docs.md`
- **Keyword Index**: `dynamic_shapes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_export/serde`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export/serde`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_export/serde`):

- [`schema_check.py_kw.md_docs.md`](./schema_check.py_kw.md_docs.md)
- [`schema.py_docs.md_docs.md`](./schema.py_docs.md_docs.md)
- [`serialize.py_kw.md_docs.md`](./serialize.py_kw.md_docs.md)
- [`serialize.py_docs.md_docs.md`](./serialize.py_docs.md_docs.md)
- [`schema.yaml_kw.md_docs.md`](./schema.yaml_kw.md_docs.md)
- [`schema.yaml_docs.md_docs.md`](./schema.yaml_docs.md_docs.md)
- [`schema.py_kw.md_docs.md`](./schema.py_kw.md_docs.md)
- [`export_schema.thrift_kw.md_docs.md`](./export_schema.thrift_kw.md_docs.md)
- [`schema_check.py_docs.md_docs.md`](./schema_check.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `dynamic_shapes.py_docs.md_docs.md`
- **Keyword Index**: `dynamic_shapes.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
