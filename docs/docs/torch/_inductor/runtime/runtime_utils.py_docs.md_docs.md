# Documentation: `docs/torch/_inductor/runtime/runtime_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/runtime/runtime_utils.py_docs.md`
- **Size**: 9,434 bytes (9.21 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/runtime/runtime_utils.py`

## File Metadata

- **Path**: `torch/_inductor/runtime/runtime_utils.py`
- **Size**: 5,970 bytes (5.83 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import functools
import operator
from typing import Any, TYPE_CHECKING

import torch

# NOTE: other files rely on the imports below
from torch._dynamo import callback as compilation_callback  # noqa: F401
from torch._inductor.runtime.cache_dir_utils import (  # noqa: F401
    cache_dir,
    default_cache_dir,
    triton_cache_dir,
)


if TYPE_CHECKING:
    from collections.abc import Hashable

    from .triton_compat import Config


def conditional_product(*args: int) -> int:
    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(number: int, denom: int) -> int:
    return -(number // -denom)


def is_power_of_2(n: int) -> bool:
    """Returns whether n = 2 ** m for some integer m."""
    return n > 0 and n & n - 1 == 0


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor)
    )


def triton_config_to_hashable(cfg: Config) -> Hashable:
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
    # pyrefly: ignore [missing-attribute]
    items = sorted(cfg.kwargs.items())
    # pyrefly: ignore [missing-attribute]
    items.append(("num_warps", cfg.num_warps))
    # pyrefly: ignore [missing-attribute]
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


def validate_triton_config(cfg: Config) -> None:
    # [Note: Triton pre_hook in inductor]
    # pre-hook is a lambda function, which we don't attempt to serialize.
    # right now, if a pre-hook is attached to the config, it will not be saved;
    # and then it won't be used when the config is loaded from cache.
    # So we assert - if we do get a pre_hook, it might get ignored after caching.
    assert getattr(cfg, "pre_hook", None) is None, (
        "triton configs with pre_hooks not supported"
    )


def create_bandwidth_info_str(
    ms: float,
    num_gb: float,
    gb_per_s: float,
    prefix: str = "",
    suffix: str = "",
    color: bool = True,
) -> str:
    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    slow = ms > 0.012 and gb_per_s < 650
    return red_text(info_str) if color and slow else info_str


def get_max_y_grid() -> int:
    return 65535


try:
    # pyrefly: ignore [import-error]
    import colorama

    HAS_COLORAMA = True
except ModuleNotFoundError:
    HAS_COLORAMA = False
    colorama = None  # type: ignore[assignment]


if HAS_COLORAMA:

    def _color_text(msg: str, color: str) -> str:
        # pyrefly: ignore [missing-attribute]
        return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET

else:

    def _color_text(msg: str, color: str) -> str:
        return msg


def green_text(msg: str) -> str:
    return _color_text(msg, "green")


def yellow_text(msg: str) -> str:
    return _color_text(msg, "yellow")


def red_text(msg: str) -> str:
    return _color_text(msg, "red")


def blue_text(msg: str) -> str:
    return _color_text(msg, "blue")


def get_first_attr(obj: Any, *attrs: str) -> Any:
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


dynamo_timed = torch._dynamo.utils.dynamo_timed  # type: ignore[has-type]


def triton_hash_to_path_key(key: str) -> str:
    # In early versions of Triton, the hash is directly used in the path name.
    # Later, the hash is converted to base64 before being used in the path name.
    # Later, the base64 conversion was replaced to the base32
    #
    # This code tries to import _base64 and falls back to _base32 if _base64 is unavailable.
    #
    # To handle this, try to import the to-base64-conversion function.
    # If it exists, use it; otherwise, try using _base32; if both are unavailable, use the hash directly.
    try:
        from triton.runtime.cache import _base64

        return _base64(key)
    except Exception:
        try:
            from triton.runtime.cache import _base32

            return _base32(key)
        except Exception:
            return key


def compile_mps_shader(source: str) -> Any:
    """
    Compiles shader source but raise more actionable error message when needed
    """
    try:
        return torch.mps.compile_shader(source)
    except SyntaxError as err:
        raise SyntaxError(f"failed to compile {source} with {err.msg}") from err


def torch_dtype_to_jax(dtype: torch.dtype) -> str:
    """
    Map PyTorch dtype to JAX dtype expression.

    This helper is used at compile time in codegen to generate
    JAX dtype expressions for Pallas kernels.

    Args:
        dtype: PyTorch dtype to convert

    Returns:
        JAX dtype expression as string (e.g., "jnp.float32")
    """
    dtype_map = {
        torch.float32: "jnp.float32",
        torch.float64: "jnp.float64",
        torch.float16: "jnp.float16",
        torch.bfloat16: "jnp.bfloat16",
        torch.int32: "jnp.int32",
        torch.int64: "jnp.int64",
        torch.int16: "jnp.int16",
        torch.int8: "jnp.int8",
        torch.uint8: "jnp.uint8",
        torch.bool: "jnp.bool_",
    }
    return dtype_map.get(dtype, f"jnp.{dtype}")

```



## High-Level Overview

"""Returns whether n = 2 ** m for some integer m."""    return n > 0 and n & n - 1 == 0def next_power_of_2(n: int) -> int:

This Python file contains 0 class(es) and 19 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `conditional_product`, `ceildiv`, `is_power_of_2`, `next_power_of_2`, `get_num_bytes`, `triton_config_to_hashable`, `validate_triton_config`, `create_bandwidth_info_str`, `get_max_y_grid`, `_color_text`, `_color_text`, `green_text`, `yellow_text`, `red_text`, `blue_text`, `get_first_attr`, `triton_hash_to_path_key`, `compile_mps_shader`, `torch_dtype_to_jax`

**Key imports**: annotations, functools, operator, Any, TYPE_CHECKING, torch, callback as compilation_callback  , Hashable, Config, colorama, _base64 and falls back to _base32 if _base64 is unavailable.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `functools`
- `operator`
- `typing`: Any, TYPE_CHECKING
- `torch`
- `torch._dynamo`: callback as compilation_callback  
- `collections.abc`: Hashable
- `.triton_compat`: Config
- `colorama`
- `_base64 and falls back to _base32 if _base64 is unavailable.`
- `the to`
- `triton.runtime.cache`: _base64


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/_inductor/runtime`):

- [`static_cuda_launcher.py_docs.md`](./static_cuda_launcher.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`hints.py_docs.md`](./hints.py_docs.md)
- [`coordinate_descent_tuner.py_docs.md`](./coordinate_descent_tuner.py_docs.md)
- [`autotune_cache.py_docs.md`](./autotune_cache.py_docs.md)
- [`triton_heuristics.py_docs.md`](./triton_heuristics.py_docs.md)
- [`debug_utils.py_docs.md`](./debug_utils.py_docs.md)
- [`compile_tasks.py_docs.md`](./compile_tasks.py_docs.md)
- [`triton_compat.py_docs.md`](./triton_compat.py_docs.md)
- [`cache_dir_utils.py_docs.md`](./cache_dir_utils.py_docs.md)


## Cross-References

- **File Documentation**: `runtime_utils.py_docs.md`
- **Keyword Index**: `runtime_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_inductor/runtime`):

- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`hints.py_kw.md_docs.md`](./hints.py_kw.md_docs.md)
- [`cache_dir_utils.py_kw.md_docs.md`](./cache_dir_utils.py_kw.md_docs.md)
- [`cache_dir_utils.py_docs.md_docs.md`](./cache_dir_utils.py_docs.md_docs.md)
- [`halide_helpers.py_docs.md_docs.md`](./halide_helpers.py_docs.md_docs.md)
- [`debug_utils.py_docs.md_docs.md`](./debug_utils.py_docs.md_docs.md)
- [`runtime_utils.py_kw.md_docs.md`](./runtime_utils.py_kw.md_docs.md)
- [`static_cuda_launcher.py_docs.md_docs.md`](./static_cuda_launcher.py_docs.md_docs.md)
- [`static_cuda_launcher.py_kw.md_docs.md`](./static_cuda_launcher.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `runtime_utils.py_docs.md_docs.md`
- **Keyword Index**: `runtime_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
