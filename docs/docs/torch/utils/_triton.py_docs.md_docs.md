# Documentation: `docs/torch/utils/_triton.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/_triton.py_docs.md`
- **Size**: 8,422 bytes (8.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/_triton.py`

## File Metadata

- **Path**: `torch/utils/_triton.py`
- **Size**: 5,173 bytes (5.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import functools
import hashlib
from typing import Any


@functools.cache
def has_triton_package() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


@functools.cache
def get_triton_version(fallback: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    try:
        import triton

        major, minor = tuple(int(v) for v in triton.__version__.split(".")[:2])
        return (major, minor)
    except ImportError:
        return fallback


@functools.cache
def _device_supports_tma() -> bool:
    import torch

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (9, 0)
        and not torch.version.hip
    )


@functools.cache
def has_triton_experimental_host_tma() -> bool:
    if has_triton_package():
        if _device_supports_tma():
            try:
                from triton.tools.experimental_descriptor import (  # noqa: F401
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_triton_tensor_descriptor_host_tma() -> bool:
    if has_triton_package():
        if _device_supports_tma():
            try:
                from triton.tools.tensor_descriptor import (  # noqa: F401
                    TensorDescriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_triton_tma() -> bool:
    return has_triton_tensor_descriptor_host_tma() or has_triton_experimental_host_tma()


@functools.cache
def has_triton_tma_device() -> bool:
    if has_triton_package():
        import torch

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (9, 0)
            and not torch.version.hip
        ) or torch.xpu.is_available():
            # old API
            try:
                from triton.language.extra.cuda import (  # noqa: F401
                    experimental_device_tensormap_create1d,
                    experimental_device_tensormap_create2d,
                )

                return True
            except ImportError:
                pass

            # new API
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_datacenter_blackwell_tma_device() -> bool:
    import torch

    if (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (10, 0)
        and torch.cuda.get_device_capability() < (11, 0)
        and not torch.version.hip
    ):
        return has_triton_tma_device() and has_triton_tensor_descriptor_host_tma()

    return False


@functools.lru_cache(None)
def has_triton_stable_tma_api() -> bool:
    if has_triton_package():
        import torch

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (9, 0)
            and not torch.version.hip
        ) or torch.xpu.is_available():
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass
    return False


@functools.cache
def has_triton() -> bool:
    if not has_triton_package():
        return False

    from torch._inductor.config import triton_disable_device_detection

    if triton_disable_device_detection:
        return False

    from torch._dynamo.device_interface import get_interface_for_device

    def cuda_extra_check(device_interface: Any) -> bool:
        return device_interface.Worker.get_device_properties().major >= 7

    def cpu_extra_check(device_interface: Any) -> bool:
        import triton.backends

        return "cpu" in triton.backends.backends

    def _return_true(device_interface: Any) -> bool:
        return True

    triton_supported_devices = {
        "cuda": cuda_extra_check,
        "xpu": _return_true,
        "cpu": cpu_extra_check,
        "mtia": _return_true,
    }

    def is_device_compatible_with_triton() -> bool:
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    return is_device_compatible_with_triton()


@functools.cache
def triton_backend() -> Any:
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    return make_backend(target)


@functools.cache
def triton_hash_with_backend() -> str:
    from torch._inductor.runtime.triton_compat import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}"

    # Hash is upper case so that it can't contain any Python keywords.
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()

```



## High-Level Overview


This Python file contains 0 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `has_triton_package`, `get_triton_version`, `_device_supports_tma`, `has_triton_experimental_host_tma`, `has_triton_tensor_descriptor_host_tma`, `has_triton_tma`, `has_triton_tma_device`, `has_datacenter_blackwell_tma_device`, `has_triton_stable_tma_api`, `has_triton`, `cuda_extra_check`, `cpu_extra_check`, `_return_true`, `is_device_compatible_with_triton`, `triton_backend`, `triton_hash_with_backend`

**Key imports**: functools, hashlib, Any, triton  , triton, torch, torch, make_tensor_descriptor  , torch, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `hashlib`
- `typing`: Any
- `triton  `
- `triton`
- `torch`
- `triton.language`: make_tensor_descriptor  
- `torch._inductor.config`: triton_disable_device_detection
- `torch._dynamo.device_interface`: get_interface_for_device
- `triton.backends`
- `triton.compiler.compiler`: make_backend
- `triton.runtime.driver`: driver
- `torch._inductor.runtime.triton_compat`: triton_key


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `_triton.py_docs.md`
- **Keyword Index**: `_triton.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/utils`):

- [`show_pickle.py_docs.md_docs.md`](./show_pickle.py_docs.md_docs.md)
- [`file_baton.py_docs.md_docs.md`](./file_baton.py_docs.md_docs.md)
- [`_filelock.py_kw.md_docs.md`](./_filelock.py_kw.md_docs.md)
- [`_config_module.py_docs.md_docs.md`](./_config_module.py_docs.md_docs.md)
- [`cpp_extension.py_docs.md_docs.md`](./cpp_extension.py_docs.md_docs.md)
- [`checkpoint.py_docs.md_docs.md`](./checkpoint.py_docs.md_docs.md)
- [`module_tracker.py_kw.md_docs.md`](./module_tracker.py_kw.md_docs.md)
- [`dlpack.py_docs.md_docs.md`](./dlpack.py_docs.md_docs.md)
- [`_import_utils.py_kw.md_docs.md`](./_import_utils.py_kw.md_docs.md)
- [`_traceback.py_kw.md_docs.md`](./_traceback.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_triton.py_docs.md_docs.md`
- **Keyword Index**: `_triton.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
