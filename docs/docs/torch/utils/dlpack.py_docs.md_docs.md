# Documentation: `docs/torch/utils/dlpack.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/dlpack.py_docs.md`
- **Size**: 9,639 bytes (9.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/dlpack.py`

## File Metadata

- **Path**: `torch/utils/dlpack.py`
- **Size**: 6,390 bytes (6.24 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any

import torch
import enum

from torch._C import _to_dlpack as to_dlpack
from torch.types import Device as _Device

__all__ = [
    "DLDeviceType",
    "from_dlpack",
]

class DLDeviceType(enum.IntEnum):
    # Enums as in DLPack specification (aten/src/ATen/dlpack.h)
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
    kDLMAIA = 17,


torch._C._add_docstr(to_dlpack, r"""to_dlpack(tensor) -> PyCapsule

Returns an opaque object (a "DLPack capsule") representing the tensor.

.. note::
  ``to_dlpack`` is a legacy DLPack interface. The capsule it returns
  cannot be used for anything in Python other than use it as input to
  ``from_dlpack``. The more idiomatic use of DLPack is to call
  ``from_dlpack`` directly on the tensor object - this works when that
  object has a ``__dlpack__`` method, which PyTorch and most other
  libraries indeed have now.

.. warning::
  Only call ``from_dlpack`` once per capsule produced with ``to_dlpack``.
  Behavior when a capsule is consumed multiple times is undefined.

Args:
    tensor: a tensor to be exported

The DLPack capsule shares the tensor's memory.
""")


# TODO: add a typing.Protocol to be able to tell Mypy that only objects with
# __dlpack__ and __dlpack_device__ methods are accepted.
def from_dlpack(
    ext_tensor: Any,
    *,
    device: _Device | None = None,
    copy: bool | None = None
) -> 'torch.Tensor':
    """from_dlpack(ext_tensor) -> Tensor

    Converts a tensor from an external library into a ``torch.Tensor``.

    The returned PyTorch tensor will share the memory with the input tensor
    (which may have come from another library). Note that in-place operations
    will therefore also affect the data of the input tensor. This may lead to
    unexpected issues (e.g., other libraries may have read-only flags or
    immutable data structures), so the user should only do this if they know
    for sure that this is fine.

    Args:
        ext_tensor (object with ``__dlpack__`` attribute, or a DLPack capsule):
            The tensor or DLPack capsule to convert.

            If ``ext_tensor`` is a tensor (or ndarray) object, it must support
            the ``__dlpack__`` protocol (i.e., have a ``ext_tensor.__dlpack__``
            method). Otherwise ``ext_tensor`` may be a DLPack capsule, which is
            an opaque ``PyCapsule`` instance, typically produced by a
            ``to_dlpack`` function or method.

        device (torch.device or str or None): An optional PyTorch device
            specifying where to place the new tensor. If None (default), the
            new tensor will be on the same device as ``ext_tensor``.

        copy (bool or None): An optional boolean indicating whether or not to copy
            ``self``. If None, PyTorch will copy only if necessary.

    Examples::

        >>> import torch.utils.dlpack
        >>> t = torch.arange(4)

        # Convert a tensor directly (supported in PyTorch >= 1.10)
        >>> t2 = torch.from_dlpack(t)
        >>> t2[:2] = -1  # show that memory is shared
        >>> t2
        tensor([-1, -1,  2,  3])
        >>> t
        tensor([-1, -1,  2,  3])

        # The old-style DLPack usage, with an intermediate capsule object
        >>> capsule = torch.utils.dlpack.to_dlpack(t)
        >>> capsule
        <capsule object "dltensor" at ...>
        >>> t3 = torch.from_dlpack(capsule)
        >>> t3
        tensor([-1, -1,  2,  3])
        >>> t3[0] = -9  # now we're sharing memory between 3 tensors
        >>> t3
        tensor([-9, -1,  2,  3])
        >>> t2
        tensor([-9, -1,  2,  3])
        >>> t
        tensor([-9, -1,  2,  3])

    """
    if hasattr(ext_tensor, '__dlpack__'):
        # Only populate kwargs if any of the optional arguments are, in fact, not None. Otherwise,
        # leave them out, since we might end up falling back to no-extra-kwargs __dlpack__ call.
        kwargs: dict[str, Any] = {}
        kwargs["max_version"] = (1, 0)

        if copy is not None:
            kwargs["copy"] = copy

        # Parse the device parameter.
        # At this moment, it can either be a torch.device or a str representing
        # a torch.device, e.g. "cpu", "cuda", etc.
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if not isinstance(device, torch.device):
                raise AssertionError(f"from_dlpack: unsupported device type: {type(device)}")
            kwargs["dl_device"] = torch._C._torchDeviceToDLDevice(device)

        ext_device = ext_tensor.__dlpack_device__()
        # ext_device is either CUDA or ROCm, we need to pass the current
        # stream
        if ext_device[0] in (DLDeviceType.kDLCUDA, DLDeviceType.kDLROCM):
            stream = torch.cuda.current_stream(f'cuda:{ext_device[1]}')
            # cuda_stream is the pointer to the stream and it is a public
            # attribute, but it is not documented
            # The array API specify that the default legacy stream must be passed
            # with a value of 1 for CUDA
            # https://data-apis.org/array-api/latest/API_specification/array_object.html?dlpack-self-stream-none#dlpack-self-stream-none
            is_cuda = ext_device[0] == DLDeviceType.kDLCUDA
            # Since pytorch is not using PTDS by default, lets directly pass
            # the legacy stream
            stream_ptr = 1 if is_cuda and stream.cuda_stream == 0 else stream.cuda_stream
            kwargs["stream"] = stream_ptr

        try:
            # Try running __dlpack__ while specifying `max_version` argument.
            dlpack = ext_tensor.__dlpack__(**kwargs)
        except TypeError:
            # If that doesn't work, try removing the `max_version` argument.
            kwargs.pop("max_version")
            dlpack = ext_tensor.__dlpack__(**kwargs)

    else:
        if device is not None or copy is not None:
            raise AssertionError(
                "device and copy kwargs not supported when ext_tensor is already a DLPack capsule."
            )
        # Old versions just call the converter
        dlpack = ext_tensor
    return torch._C._from_dlpack(dlpack)

```



## High-Level Overview

torch._C._add_docstr(to_dlpack, r"""to_dlpack(tensor) -> PyCapsuleReturns an opaque object (a "DLPack capsule") representing the tensor... note::  ``to_dlpack`` is a legacy DLPack interface. The capsule it returns  cannot be used for anything in Python other than use it as input to  ``from_dlpack``. The more idiomatic use of DLPack is to call  ``from_dlpack`` directly on the tensor object - this works when that  object has a ``__dlpack__`` method, which PyTorch and most other  libraries indeed have now... warning::  Only call ``from_dlpack`` once per capsule produced with ``to_dlpack``.  Behavior when a capsule is consumed multiple times is undefined.Args:    tensor: a tensor to be exported

This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DLDeviceType`

**Functions defined**: `from_dlpack`

**Key imports**: Any, torch, enum, _to_dlpack as to_dlpack, Device as _Device, torch.utils.dlpack


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch`
- `enum`
- `torch._C`: _to_dlpack as to_dlpack
- `torch.types`: Device as _Device
- `torch.utils.dlpack`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `dlpack.py_docs.md`
- **Keyword Index**: `dlpack.py_kw.md`
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
- [`_import_utils.py_kw.md_docs.md`](./_import_utils.py_kw.md_docs.md)
- [`_traceback.py_kw.md_docs.md`](./_traceback.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `dlpack.py_docs.md_docs.md`
- **Keyword Index**: `dlpack.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
