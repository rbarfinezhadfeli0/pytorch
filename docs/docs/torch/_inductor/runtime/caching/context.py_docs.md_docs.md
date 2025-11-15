# Documentation: `docs/torch/_inductor/runtime/caching/context.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/runtime/caching/context.py_docs.md`
- **Size**: 13,679 bytes (13.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/runtime/caching/context.py`

## File Metadata

- **Path**: `torch/_inductor/runtime/caching/context.py`
- **Size**: 10,315 bytes (10.07 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Context management for PyTorch Inductor runtime caching.

This module provides context classes for collecting configuration and environment
information used in caching decisions for PyTorch's Inductor runtime.
"""

import json
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Sequence
from functools import cache
from hashlib import sha256
from typing import Any
from typing_extensions import override, TypedDict

import torch


class _Context(ABC):
    """Abstract base class for context providers.

    Context providers collect specific configuration and environment information
    that affects compilation and runtime behavior.
    """

    @staticmethod
    @abstractmethod
    def forms_of_context() -> Sequence[str]:
        """Return a sequence of context form names provided by this context class.

        Returns:
            A sequence of strings representing the available context forms.
        """


class _RuntimeContext(_Context):
    """Context provider for runtime configuration and environment settings.

    Collects configuration settings that affect runtime behavior but not
    compilation, such as Inductor configs, determinism settings, and CUDA
    matmul precision configurations.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        """Return the runtime context forms provided by this class.

        Returns:
            A sequence containing the available runtime context forms:
            - "inductor_configs": PyTorch Inductor configuration settings
            - "torch_determinism_configs": Deterministic algorithm settings
            - "cuda_matmul_precision_configs": CUDA matrix multiplication precision settings
        """
        return (
            "inductor_configs",
            "torch_determinism_configs",
            "cuda_matmul_precision_configs",
        )

    @staticmethod
    def inductor_configs() -> dict[str, Any]:
        """Get portable Inductor configuration settings.

        Returns:
            A dictionary containing Inductor configuration settings,
            including private configs.
        """
        from torch._inductor import config

        return config.save_config_portable(ignore_private_configs=False)

    @staticmethod
    def torch_determinism_configs() -> dict[str, Any]:
        """Get PyTorch deterministic algorithm configuration settings.

        Returns:
            A dictionary containing deterministic algorithm settings:
            - Whether deterministic algorithms are enabled
            - Whether deterministic algorithm warnings are enabled
            - Fill uninitialized memory setting
        """
        return {
            "torch.are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "torch.is_deterministic_algorithms_warn_only_enabled": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "torch.utils.deterministic.fill_uninitialized_memory": (
                torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]
            ),
        }

    @staticmethod
    def cuda_matmul_precision_configs() -> dict[str, Any]:
        """Get CUDA matrix multiplication precision configuration settings.

        Returns:
            A dictionary containing CUDA matmul precision settings:
            - FP32 precision setting
            - FP16 reduced precision reduction allowance
            - BF16 reduced precision reduction allowance
        """
        return {
            "torch.backends.cuda.matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        }


class _CompileContext(_Context):
    """Context provider for compilation-related configuration and environment settings.

    Collects information that affects compilation behavior, such as PyTorch and Triton
    versions, runtime environment, and accelerator properties.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        """Return the compile context forms provided by this class.

        Returns:
            A sequence containing the available compile context forms:
            - "torch_version_hash": PyTorch version hash
            - "triton_version_hash": Triton version hash (if available)
            - "runtime": Runtime type (CUDA/HIP/None)
            - "runtime_version": Runtime version string
            - "accelerator_properties": GPU/accelerator properties
        """
        return (
            "torch_version_hash",
            "triton_version_hash",
            "runtime",
            "runtime_version",
            "accelerator_properties",
        )

    @cache
    @staticmethod
    def torch_version_hash() -> str:
        """Get base64-encoded PyTorch version hash.

        Returns:
            A base64-encoded string representing the PyTorch version hash.
        """
        from torch._inductor.codecache import torch_key

        return b64encode(torch_key()).decode()

    @cache
    @staticmethod
    def triton_version_hash() -> str | None:
        """Get Triton version key if Triton is available.

        Returns:
            Triton version key if Triton is available, None otherwise.
        """
        from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

        return triton_key() if HAS_TRITON else None

    @cache
    @staticmethod
    def runtime() -> str | None:
        """Determine the runtime type based on available backends.

        Returns:
            "CUDA" if CUDA is available, "HIP" if HIP is available, None otherwise.
        """
        return "CUDA" if torch.version.cuda else "HIP" if torch.version.hip else None

    @cache
    @staticmethod
    def runtime_version() -> str | None:
        """Get the version string for the detected runtime.

        Returns:
            Version string for the current runtime (CUDA or HIP), or None if
            no supported runtime is detected.
        """
        return {
            "CUDA": torch.version.cuda,
            "HIP": torch.version.hip,
        }.get(_CompileContext.runtime())  # type: ignore[arg-type]

    @cache
    @staticmethod
    def accelerator_properties() -> str | None:
        """Get string representation of CUDA device properties.

        Returns:
            String representation of CUDA device properties if a runtime is
            available, None otherwise.
        """
        return (
            repr(torch.cuda.get_device_properties())
            if _CompileContext.runtime() and torch.cuda.is_available()
            else None
        )


class SelectedRuntimeContext(TypedDict):
    inductor_configs: bool
    torch_determinism_configs: bool
    cuda_matmul_precision_configs: bool


class SelectedCompileContext(TypedDict):
    torch_version_hash: bool
    triton_version_hash: bool
    runtime: bool
    runtime_version: bool
    accelerator_properties: bool


class IsolationSchema(TypedDict):
    """Schema for specifying which context forms to include in cache isolation.

    Attributes:
        runtime_context: Either True (include all runtime context), False (exclude all),
                        or a SelectedRuntimeContext dict specifying which forms to include.
        compile_context: Either True (include all compile context), False (exclude all),
                        or a SelectedCompileContext dict specifying which forms to include.
    """

    runtime_context: SelectedRuntimeContext | bool
    compile_context: SelectedCompileContext | bool


_DEFAULT_ISOLATION_SCHEMA: IsolationSchema = IsolationSchema(
    runtime_context=True, compile_context=True
)


def _isolation_context(
    ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA,
) -> dict[str, Any]:
    """Generate context data based on the isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A dictionary containing the selected context data with keys
        "runtime_context" and "compile_context", where each value is
        either None (if excluded) or a dict of context form data.
    """
    isolation_context: dict[str, Any] = {}
    for context_name, context_cls in (
        ("runtime_context", _RuntimeContext),
        ("compile_context", _CompileContext),
    ):
        selected_context: dict[str, Any] | None = None
        if ischema[context_name] is True:  # type: ignore[literal-required]
            selected_context = {
                form_of_context: getattr(context_cls, form_of_context)()
                for form_of_context in context_cls.forms_of_context()
            }
        elif ischema[context_name] is False:  # type: ignore[literal-required]
            selected_context = None
        else:
            selected_context = {}
            for form_of_context in ischema[context_name]:  # type: ignore[literal-required]
                selected = ischema[context_name][form_of_context]  # type: ignore[literal-required]
                if selected:
                    selected_context[form_of_context] = getattr(
                        context_cls, form_of_context
                    )()
            selected_context = selected_context or None
        isolation_context[context_name] = selected_context
    return isolation_context


def _isolation_key(ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA) -> str:
    """Generate a unique key for the given isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A 32-character hexadecimal string that uniquely identifies
        the context specified by the isolation schema.
    """
    return sha256(
        json.dumps(_isolation_context(ischema), sort_keys=True).encode()
    ).hexdigest()[:32]

```



## High-Level Overview

"""Context management for PyTorch Inductor runtime caching.This module provides context classes for collecting configuration and environmentinformation used in caching decisions for PyTorch's Inductor runtime.

This Python file contains 7 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_Context`, `_RuntimeContext`, `_CompileContext`, `SelectedRuntimeContext`, `SelectedCompileContext`, `IsolationSchema`

**Functions defined**: `forms_of_context`, `forms_of_context`, `inductor_configs`, `torch_determinism_configs`, `cuda_matmul_precision_configs`, `forms_of_context`, `torch_version_hash`, `triton_version_hash`, `runtime`, `runtime_version`, `accelerator_properties`, `_isolation_context`, `_isolation_key`

**Key imports**: json, ABC, abstractmethod, b64encode, Sequence, cache, sha256, Any, override, TypedDict, torch, config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/runtime/caching`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `abc`: ABC, abstractmethod
- `base64`: b64encode
- `collections.abc`: Sequence
- `functools`: cache
- `hashlib`: sha256
- `typing`: Any
- `typing_extensions`: override, TypedDict
- `torch`
- `torch._inductor`: config
- `torch._inductor.codecache`: torch_key
- `torch._inductor.runtime.triton_compat`: HAS_TRITON, triton_key


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`torch/_inductor/runtime/caching`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`exceptions.py_docs.md`](./exceptions.py_docs.md)
- [`implementations.py_docs.md`](./implementations.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`interfaces.py_docs.md`](./interfaces.py_docs.md)
- [`locks.py_docs.md`](./locks.py_docs.md)


## Cross-References

- **File Documentation**: `context.py_docs.md`
- **Keyword Index**: `context.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/runtime/caching`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/runtime/caching`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`docs/torch/_inductor/runtime/caching`):

- [`exceptions.py_kw.md_docs.md`](./exceptions.py_kw.md_docs.md)
- [`context.py_kw.md_docs.md`](./context.py_kw.md_docs.md)
- [`locks.py_kw.md_docs.md`](./locks.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`exceptions.py_docs.md_docs.md`](./exceptions.py_docs.md_docs.md)
- [`interfaces.py_docs.md_docs.md`](./interfaces.py_docs.md_docs.md)
- [`implementations.py_kw.md_docs.md`](./implementations.py_kw.md_docs.md)
- [`locks.py_docs.md_docs.md`](./locks.py_docs.md_docs.md)
- [`config.py_kw.md_docs.md`](./config.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `context.py_docs.md_docs.md`
- **Keyword Index**: `context.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
