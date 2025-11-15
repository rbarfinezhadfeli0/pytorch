# Documentation: `torch/distributed/remote_device.py`

## File Metadata

- **Path**: `torch/distributed/remote_device.py`
- **Size**: 4,761 bytes (4.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional, Union

import torch


class _remote_device:
    """
    Represents a device on a remote worker.

    Args:
        remote_device (str or torch.device): Represents a device on a remote worker.
            The string format should be one of the following:

                1. "<workername>/<device>", where the device field can be parsed as torch.device type.
                   E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
                   In addition, the device field can be optional and the default value is "cpu".
                2. "rank:<rank>/<device>", where <rank> is the rank of the
                   process and device can be parsed as torch.device type.
                   E.g., "rank:0/cpu", "rank:0", "rank:0/cuda:0"
                3. <workername> and <rank> are optional and formats like "cpu"
                    and "cuda:1", just represent local devices.
    """

    def __init__(self, remote_device: Union[str, torch.device]):
        PARSE_ERROR = (
            f"Could not parse remote_device: {remote_device}. The valid format is "
            "'<workername>/<device>' or 'rank:<rank>/<device>' or '<device>'"
        )
        self._worker_name = None
        self._rank = None
        self._device: Optional[Union[str, int, torch.device]] = None

        if isinstance(remote_device, torch.device):
            self._device = remote_device
        elif isinstance(remote_device, str):
            fields = remote_device.split("/")
            if len(fields) == 2:
                # pyrefly: ignore [bad-assignment]
                self._worker_name, self._device = fields
            elif len(fields) == 1:
                # Check if this is a valid device.
                if _remote_device._is_valid_local_device(fields[0]):
                    self._device = fields[0]
                else:
                    # pyrefly: ignore [bad-assignment]
                    self._worker_name = fields[0]
                    self._device = "cpu"
            else:
                raise ValueError(PARSE_ERROR)
        else:
            raise TypeError(f"Invalid type for remote_device: {type(remote_device)}")

        # Do some basic sanity check (no empty string)
        if self._worker_name is not None and not self._worker_name:
            raise ValueError(PARSE_ERROR)

        # Validate the device.
        self._device = torch.device(self._device)

        # Check for rank based format.
        if self._worker_name is not None:
            fields = self._worker_name.split(":")
            if len(fields) == 2:
                # rank:<rank>/device format, extract rank
                if fields[0] == "rank" and fields[1].isdigit():
                    self._rank = int(fields[1])  # type: ignore[assignment]
                    # pyrefly: ignore [bad-assignment]
                    self._worker_name = None
                else:
                    raise ValueError(PARSE_ERROR)
            elif len(fields) > 2:
                raise ValueError(PARSE_ERROR)

    @staticmethod
    def _is_valid_local_device(device):
        # Check for torch.device
        try:
            torch.device(device)
            return True
        except Exception:
            return False

    def worker_name(self) -> Optional[str]:
        """Return the name of remote worker representing the remote device and ``None`` if no worker name is available."""
        return self._worker_name

    def rank(self) -> Optional[int]:
        """
        Returns the rank of remote worker representing the remote device.
        Returns ``None`` if no rank is available.
        """
        return self._rank

    def device(self) -> torch.device:
        """Return the local device on the remote worker."""
        return self._device  # type: ignore[return-value]

    def __repr__(self):
        if self._device is not None:
            if self._worker_name is not None:
                return f"{self._worker_name}/{self._device}"
            elif self._rank is not None:
                return f"rank:{self._rank}/{self._device}"
            else:
                return str(self._device)
        else:
            if self._worker_name is not None:
                return f"{self._worker_name}"
            elif self._rank is not None:
                return f"{self._rank}"
            else:
                raise RuntimeError("Invalid state!")

    def __eq__(self, other):
        return isinstance(other, _remote_device) and (
            self._worker_name == other._worker_name
            and self._device == other._device
            and self._rank == other._rank
        )

    def __hash__(self):
        return hash(self._worker_name) ^ hash(self._device) ^ hash(self._rank)

```



## High-Level Overview

"""    Represents a device on a remote worker.    Args:        remote_device (str or torch.device): Represents a device on a remote worker.            The string format should be one of the following:                1. "<workername>/<device>", where the device field can be parsed as torch.device type.                   E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".                   In addition, the device field can be optional and the default value is "cpu".                2. "rank:<rank>/<device>", where <rank> is the rank of the                   process and device can be parsed as torch.device type.                   E.g., "rank:0/cpu", "rank:0", "rank:0/cuda:0"                3. <workername> and <rank> are optional and formats like "cpu"                    and "cuda:1", just represent local devices.

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_remote_device`

**Functions defined**: `__init__`, `_is_valid_local_device`, `worker_name`, `rank`, `device`, `__repr__`, `__eq__`, `__hash__`

**Key imports**: Optional, Union, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional, Union
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`torch/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_mesh_layout.py_docs.md`](./_mesh_layout.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`c10d_logger.py_docs.md`](./c10d_logger.py_docs.md)
- [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`CONTRIBUTING.md_docs.md`](./CONTRIBUTING.md_docs.md)
- [`_functional_collectives_impl.py_docs.md`](./_functional_collectives_impl.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)


## Cross-References

- **File Documentation**: `remote_device.py_docs.md`
- **Keyword Index**: `remote_device.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
