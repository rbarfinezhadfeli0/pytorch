# Documentation: `docs/torch/_inductor/exc.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/exc.py_docs.md`
- **Size**: 8,195 bytes (8.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/exc.py`

## File Metadata

- **Path**: `torch/_inductor/exc.py`
- **Size**: 4,869 bytes (4.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import os
import tempfile
import textwrap
from functools import lru_cache
from typing import Any, Optional, TYPE_CHECKING

from torch._dynamo.exc import BackendCompilerFailed, ShortenTraceback


if TYPE_CHECKING:
    import types

    from torch.cuda import _CudaDeviceProperties

if os.environ.get("TORCHINDUCTOR_WRITE_MISSING_OPS") == "1":

    @lru_cache(None)
    def _record_missing_op(target: Any) -> None:
        with open(f"{tempfile.gettempdir()}/missing_ops.txt", "a") as fd:
            fd.write(str(target) + "\n")

else:

    def _record_missing_op(target: Any) -> None:  # type: ignore[misc]
        pass


class OperatorIssue(RuntimeError):
    @staticmethod
    def operator_str(target: Any, args: list[Any], kwargs: dict[str, Any]) -> str:
        lines = [f"target: {target}"] + [
            f"args[{i}]: {arg}" for i, arg in enumerate(args)
        ]
        if kwargs:
            lines.append(f"kwargs: {kwargs}")
        return textwrap.indent("\n".join(lines), "  ")


class MissingOperatorWithoutDecomp(OperatorIssue):
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        _record_missing_op(target)
        super().__init__(f"missing lowering\n{self.operator_str(target, args, kwargs)}")


class MissingOperatorWithDecomp(OperatorIssue):
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        _record_missing_op(target)
        super().__init__(
            f"missing decomposition\n{self.operator_str(target, args, kwargs)}"
            + textwrap.dedent(
                f"""

                There is a decomposition available for {target} in
                torch._decomp.get_decompositions().  Please add this operator to the
                `decompositions` list in torch._inductor.decomposition
                """
            )
        )


class LoweringException(OperatorIssue):
    def __init__(
        self, exc: Exception, target: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> None:
        super().__init__(
            f"{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}"
        )


class SubgraphLoweringException(RuntimeError):
    pass


class InvalidCxxCompiler(RuntimeError):
    def __init__(self) -> None:
        from . import config

        super().__init__(
            f"No working C++ compiler found in {config.__name__}.cpp.cxx: {config.cpp.cxx}"
        )


class CppWrapperCodegenError(RuntimeError):
    def __init__(self, msg: str) -> None:
        super().__init__(f"C++ wrapper codegen error: {msg}")


class CppCompileError(RuntimeError):
    def __init__(self, cmd: list[str], output: str) -> None:
        if isinstance(output, bytes):
            output = output.decode("utf-8")

        self.cmd = cmd
        self.output = output

        super().__init__(
            textwrap.dedent(
                """
                    C++ compile error

                    Command:
                    {cmd}

                    Output:
                    {output}
                """
            )
            .strip()
            .format(cmd=" ".join(cmd), output=output)
        )

    def __reduce__(self) -> tuple[type, tuple[list[str], str]]:
        return (self.__class__, (self.cmd, self.output))


class CUDACompileError(CppCompileError):
    pass


class TritonMissing(ShortenTraceback):
    def __init__(self, first_useful_frame: Optional[types.FrameType]) -> None:
        super().__init__(
            "Cannot find a working triton installation. "
            "Either the package is not installed or it is too old. "
            "More information on installing Triton can be found at: https://github.com/triton-lang/triton",
            first_useful_frame=first_useful_frame,
        )


class GPUTooOldForTriton(ShortenTraceback):
    def __init__(
        self,
        # pyrefly: ignore [not-a-type]
        device_props: _CudaDeviceProperties,
        first_useful_frame: Optional[types.FrameType],
    ) -> None:
        super().__init__(
            f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, "
            "which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, "
            f"but your device is of CUDA capability {device_props.major}.{device_props.minor}",
            first_useful_frame=first_useful_frame,
        )


class InductorError(BackendCompilerFailed):
    backend_name = "inductor"

    def __init__(
        self,
        inner_exception: Exception,
        first_useful_frame: Optional[types.FrameType],
    ) -> None:
        self.inner_exception = inner_exception
        ShortenTraceback.__init__(
            self,
            f"{type(inner_exception).__name__}: {inner_exception}",
            first_useful_frame=first_useful_frame,
        )

```



## High-Level Overview


This Python file contains 12 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OperatorIssue`, `MissingOperatorWithoutDecomp`, `MissingOperatorWithDecomp`, `LoweringException`, `SubgraphLoweringException`, `InvalidCxxCompiler`, `CppWrapperCodegenError`, `CppCompileError`, `CUDACompileError`, `TritonMissing`, `GPUTooOldForTriton`, `InductorError`

**Functions defined**: `_record_missing_op`, `_record_missing_op`, `operator_str`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__reduce__`, `__init__`, `__init__`, `__init__`

**Key imports**: annotations, os, tempfile, textwrap, lru_cache, Any, Optional, TYPE_CHECKING, BackendCompilerFailed, ShortenTraceback, types, _CudaDeviceProperties, config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `os`
- `tempfile`
- `textwrap`
- `functools`: lru_cache
- `typing`: Any, Optional, TYPE_CHECKING
- `torch._dynamo.exc`: BackendCompilerFailed, ShortenTraceback
- `types`
- `torch.cuda`: _CudaDeviceProperties
- `.`: config


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `exc.py_docs.md`
- **Keyword Index**: `exc.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `exc.py_docs.md_docs.md`
- **Keyword Index**: `exc.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
