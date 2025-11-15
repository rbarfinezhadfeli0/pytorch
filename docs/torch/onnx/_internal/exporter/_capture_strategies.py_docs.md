# Documentation: `torch/onnx/_internal/exporter/_capture_strategies.py`

## File Metadata

- **Path**: `torch/onnx/_internal/exporter/_capture_strategies.py`
- **Size**: 10,578 bytes (10.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Strategies for capturing ExportedPrograms."""

# mypy: allow-untyped-defs
from __future__ import annotations

import abc
import contextlib
import dataclasses
import datetime
import logging
import pathlib
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx import _flags


if TYPE_CHECKING:
    import os
    from collections.abc import Callable


logger = logging.getLogger(__name__)


def _verbose_printer(verbose: bool | None) -> Callable[..., None]:
    """Prints messages based on `verbose`."""
    if verbose is False:
        return lambda *_, **__: None
    # pyrefly: ignore [not-iterable]
    return lambda *args, **kwargs: print("[torch.onnx]", *args, **kwargs)


def _take_first_line(text: str) -> str:
    """Take the first line of a text."""
    lines = text.split("\n", maxsplit=1)
    first_line = lines[0]
    if len(lines) > 1:
        first_line += "[...]"
    return first_line


@contextlib.contextmanager
def _patch_dynamo_unsupported_functions():
    """Patch PyTorch to bypass some functions torch.export.export does not support."""
    # TODO: Remove the patches once dynamo supports these functions.
    import torch.jit

    # Replace torch.jit.isinstance with isinstance
    jit_isinstance = torch.jit.isinstance
    # pyrefly: ignore [bad-assignment]
    torch.jit.isinstance = isinstance
    logger.info("Replaced torch.jit.isinstance with isinstance to allow dynamo tracing")
    try:
        yield
    finally:
        torch.jit.isinstance = jit_isinstance


@dataclasses.dataclass
class Result:
    exported_program: torch.export.ExportedProgram | None
    strategy: str
    exception: Exception | None = None

    @property
    def success(self) -> bool:
        """Whether the capture was successful.

        An exception can still be recorded even if the capture was successful. In
        this case the exception is informational only. For example, draft_export
        can record an exception if there are warnings during the export. The exceptions
        will go into the onnx export report when report=True.
        """
        return self.exported_program is not None


class CaptureStrategy(abc.ABC):
    """Strategy for capturing a module as ExportedProgram.

    To use a strategy, create an instance and call it with the model, args, kwargs, and dynamic_shapes.
    Example::

        strategy = TorchExportNonStrictStrategy(verbose=True)
        result = strategy(model, args, kwargs, dynamic_shapes)
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        dump: bool = False,
        artifacts_dir: str | os.PathLike = ".",
        timestamp: str | None = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            verbose: Whether to print verbose messages.
            dump: Whether to dump the intermediate artifacts to a file.
        """
        self._verbose_print = _verbose_printer(verbose)
        self._dump = dump
        self._artifacts_dir = pathlib.Path(artifacts_dir)
        self._timestamp = timestamp or datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-%f"
        )
        self._exception: Exception | None = None

    def __call__(
        self,
        model: torch.nn.Module | torch.jit.ScriptFunction,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        dynamic_shapes,
    ) -> Result:
        self._enter(model)
        if kwargs is None:
            kwargs = {}
        try:
            exported_program = self._capture(model, args, kwargs, dynamic_shapes)
        except Exception as e:
            self._failure(model, e)
            return Result(
                exported_program=None,
                strategy=self.__class__.__name__,
                exception=e,
            )
        self._success(model)
        return Result(
            exported_program,
            strategy=self.__class__.__name__,
            exception=self._exception,
        )

    @abc.abstractmethod
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        raise NotImplementedError

    def _enter(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None:
        return

    def _success(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None:
        return

    def _failure(
        self, model: torch.nn.Module | torch.jit.ScriptFunction, e: Exception
    ) -> None:
        return


class TorchExportStrictStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        with (
            _patch_dynamo_unsupported_functions(),
            # Support the dynamism with 0/1 input dim
            torch.fx.experimental._config.patch(backed_size_oblivious=True),  # type: ignore[attr-defined]
        ):
            try:
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=True,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )
            except torch._dynamo.exc.UserError as exc:
                # Refine the dynamic shapes based on the suggested fixes.
                try:
                    new_shapes = torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes(
                        exc.msg, dynamic_shapes
                    )
                except Exception:
                    # If the dynamic shapes cannot be refined, re-raise the exception.
                    raise exc from None
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=new_shapes,
                    strict=True,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`... ❌"
        )


class TorchExportNonStrictStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        with (
            # Support the dynamism with 0/1 input dim
            torch.fx.experimental._config.patch(backed_size_oblivious=True),  # type: ignore[attr-defined]
        ):
            try:
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )
            except torch._dynamo.exc.UserError as exc:
                # Refine the dynamic shapes based on the suggested fixes.
                try:
                    new_shapes = torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes(
                        exc.msg, dynamic_shapes
                    )
                except Exception:
                    # If the dynamic shapes cannot be refined, re-raise the exception.
                    raise exc from None
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=new_shapes,
                    strict=False,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ❌"
        )


class TorchExportDraftExportStrategy(CaptureStrategy):
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        ep = torch.export.draft_export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )
        report = ep._report  # type: ignore[attr-defined]
        if not report.successful():
            self._exception = RuntimeError(str(report))
            self._verbose_print(f"Draft Export report:\n{report}")
        return ep

    def _enter(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`..."
        )

    def _success(self, model) -> None:
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`... ✅"
        )

    def _failure(self, model, e) -> None:
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`... ❌"
        )


CAPTURE_STRATEGIES: tuple[type[CaptureStrategy], ...] = (
    TorchExportNonStrictStrategy,  # strict=False is preferred over strict=True because it does not have dynamo issues
    TorchExportStrictStrategy,
)

if _flags.ENABLE_DRAFT_EXPORT:
    CAPTURE_STRATEGIES = (*CAPTURE_STRATEGIES, TorchExportDraftExportStrategy)

```



## High-Level Overview

"""Strategies for capturing ExportedPrograms."""# mypy: allow-untyped-defsfrom __future__ import annotationsimport abcimport contextlibimport dataclassesimport datetimeimport loggingimport pathlibfrom typing import Any, TYPE_CHECKINGimport torchfrom torch.onnx import _flagsif TYPE_CHECKING:    import os    from collections.abc import Callablelogger = logging.getLogger(__name__)def _verbose_printer(verbose: bool | None) -> Callable[..., None]:

This Python file contains 5 class(es) and 22 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Result`, `CaptureStrategy`, `TorchExportStrictStrategy`, `TorchExportNonStrictStrategy`, `TorchExportDraftExportStrategy`

**Functions defined**: `_verbose_printer`, `_take_first_line`, `_patch_dynamo_unsupported_functions`, `success`, `__init__`, `__call__`, `_capture`, `_enter`, `_success`, `_failure`, `_capture`, `_enter`, `_success`, `_failure`, `_capture`, `_enter`, `_success`, `_failure`, `_capture`, `_enter`

**Key imports**: annotations, abc, contextlib, dataclasses, datetime, logging, pathlib, Any, TYPE_CHECKING, torch, _flags


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `abc`
- `contextlib`
- `dataclasses`
- `datetime`
- `logging`
- `pathlib`
- `typing`: Any, TYPE_CHECKING
- `torch`
- `torch.onnx`: _flags
- `os`
- `collections.abc`: Callable
- `torch.jit`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/onnx/_internal/exporter`):

- [`_registration.py_docs.md`](./_registration.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_flags.py_docs.md`](./_flags.py_docs.md)
- [`_building.py_docs.md`](./_building.py_docs.md)
- [`_ir_passes.py_docs.md`](./_ir_passes.py_docs.md)
- [`_analysis.py_docs.md`](./_analysis.py_docs.md)
- [`_verification.py_docs.md`](./_verification.py_docs.md)
- [`_tensors.py_docs.md`](./_tensors.py_docs.md)
- [`_dispatching.py_docs.md`](./_dispatching.py_docs.md)


## Cross-References

- **File Documentation**: `_capture_strategies.py_docs.md`
- **Keyword Index**: `_capture_strategies.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
