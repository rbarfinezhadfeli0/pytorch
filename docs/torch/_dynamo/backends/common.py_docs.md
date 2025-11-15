# Documentation: `torch/_dynamo/backends/common.py`

## File Metadata

- **Path**: `torch/_dynamo/backends/common.py`
- **Size**: 6,127 bytes (5.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
This module provides common utilities and base classes for TorchDynamo backends.

Key components:
- AotAutograd: Base class for implementing AOT (Ahead-of-Time) autograd backends
- Backend utilities for handling:
  - Fake tensor conversion
  - Device/dtype detection from inputs
  - Memory efficient fusion
  - Graph flattening
  - Common compiler configurations

The utilities here are used by various backend implementations to handle
common operations and provide consistent behavior across different backends.
AOT autograd functionality is particularly important as it enables ahead-of-time
optimization of both forward and backward passes.
"""

import contextlib
import functools
import logging
from collections.abc import Callable, Iterable
from typing import Any
from typing_extensions import ParamSpec, TypeVar
from unittest.mock import patch

import torch
from torch._dynamo import disable
from torch._dynamo.exc import TensorifyScalarRestartAnalysis
from torch._dynamo.utils import counters, defake, flatten_graph_inputs
from torch._functorch.aot_autograd import (
    aot_module_simplified,
    SerializableAOTDispatchCompiler,
)
from torch.utils._python_dispatch import _disable_current_modes


log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class AotAutograd:
    def __init__(self, **kwargs: Any) -> None:
        self.__name__ = "compiler_fn"
        self.kwargs = kwargs

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Iterable[Any], **kwargs: Any
    ) -> Callable[..., Any]:
        if kwargs:
            log.warning("aot_autograd-based backend ignoring extra kwargs %s", kwargs)

        if any(isinstance(x, (list, tuple, dict)) for x in example_inputs):
            return flatten_graph_inputs(
                gm,
                example_inputs,
                self,
            )

        # Hack to get around circular import problems with aot_eager_decomp_partition
        if callable(self.kwargs.get("decompositions")):
            self.kwargs["decompositions"] = self.kwargs["decompositions"]()

        # NB: dont delete counter increment
        counters["aot_autograd"]["total"] += 1
        use_fallback = False

        if use_fallback:
            log.debug("Unable to use AOT Autograd because graph has mutation")
            counters["aot_autograd"]["not_ok"] += 1
            return gm

        def wrap_bw_compiler(bw_compiler_fn: Callable[P, R]) -> Callable[..., R]:
            def _wrapped_bw_compiler(*args: P.args, **kwargs: P.kwargs) -> R:
                # Note [Wrapping bw_compiler in disable]
                # The two disables here:
                # - stop TorchDynamo from trying to compile the bw_compiler function itself
                # - stop TorchDynamo from trying to compile our the generated backwards pass bw_compiler produces
                return disable(
                    disable(
                        bw_compiler_fn, reason="do not trace backward compiler function"
                    )(*args, **kwargs),  # type: ignore[misc]
                    reason="do not trace generated backwards pass",
                )

            return _wrapped_bw_compiler

        bw_compiler = self.kwargs.get("bw_compiler") or self.kwargs["fw_compiler"]

        if isinstance(bw_compiler, SerializableAOTDispatchCompiler):
            bw_compiler.compiler_fn = wrap_bw_compiler(bw_compiler.compiler_fn)
        else:
            bw_compiler = wrap_bw_compiler(bw_compiler)

        self.kwargs["bw_compiler"] = bw_compiler
        self.kwargs["inference_compiler"] = (
            self.kwargs.get("inference_compiler") or self.kwargs["fw_compiler"]
        )

        from functorch.compile import nop
        from torch._inductor.debug import enable_aot_logging

        # debug asserts slow down compile time noticeably,
        # So only default them on when the aot_eager backend is used.
        if self.kwargs.get("fw_compiler", None) is nop:
            patch_config: contextlib.AbstractContextManager[Any] = patch(
                "functorch.compile.config.debug_assert", True
            )
        else:
            patch_config = contextlib.nullcontext()

        try:
            # NB: NOT cloned!
            with enable_aot_logging(), patch_config:
                cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
                counters["aot_autograd"]["ok"] += 1
                return disable(cg, reason="do not trace AOT-compiled graph")
        except TensorifyScalarRestartAnalysis:
            raise
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise


def aot_autograd(**kwargs: Any) -> AotAutograd:
    return AotAutograd(**kwargs)


def mem_efficient_fusion_kwargs(use_decomps: bool) -> dict[str, Any]:
    from functorch.compile import (
        default_decompositions,
        min_cut_rematerialization_partition,
        ts_compile,
    )

    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
    }

    if use_decomps:
        kwargs["decompositions"] = default_decompositions

    return kwargs


def fake_tensor_unsupported(fn: Callable[[Any, list[Any], Any], R]) -> Any:
    """
    Decorator for backends that need real inputs.  We swap out fake
    tensors for zero tensors.
    """

    @functools.wraps(fn)
    def wrapper(model: Any, inputs: Any, **kwargs: Any) -> Any:
        with _disable_current_modes():
            inputs = list(map(defake, inputs))
            return fn(model, inputs, **kwargs)  # type: ignore[call-arg]

    return wrapper


def device_from_inputs(example_inputs: Iterable[Any]) -> torch.device:
    for x in example_inputs:
        if hasattr(x, "device"):
            return x.device
    return torch.device("cpu")  # Default fallback


def dtype_from_inputs(example_inputs: Iterable[Any]) -> torch.dtype:
    for x in example_inputs:
        if hasattr(x, "dtype"):
            return x.dtype
    return torch.float32  # Default fallback

```



## High-Level Overview

"""This module provides common utilities and base classes for TorchDynamo backends.Key components:- AotAutograd: Base class for implementing AOT (Ahead-of-Time) autograd backends- Backend utilities for handling:  - Fake tensor conversion  - Device/dtype detection from inputs  - Memory efficient fusion  - Graph flattening  - Common compiler configurationsThe utilities here are used by various backend implementations to handlecommon operations and provide consistent behavior across different backends.AOT autograd functionality is particularly important as it enables ahead-of-timeoptimization of both forward and backward passes.

This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AotAutograd`

**Functions defined**: `__init__`, `__call__`, `wrap_bw_compiler`, `_wrapped_bw_compiler`, `aot_autograd`, `mem_efficient_fusion_kwargs`, `fake_tensor_unsupported`, `wrapper`, `device_from_inputs`, `dtype_from_inputs`

**Key imports**: contextlib, functools, logging, Callable, Iterable, Any, ParamSpec, TypeVar, patch, torch, disable, TensorifyScalarRestartAnalysis


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/backends`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `functools`
- `logging`
- `collections.abc`: Callable, Iterable
- `typing`: Any
- `typing_extensions`: ParamSpec, TypeVar
- `unittest.mock`: patch
- `torch`
- `torch._dynamo`: disable
- `torch._dynamo.exc`: TensorifyScalarRestartAnalysis
- `torch._dynamo.utils`: counters, defake, flatten_graph_inputs
- `torch.utils._python_dispatch`: _disable_current_modes
- `problems with aot_eager_decomp_partition`
- `functorch.compile`: nop
- `torch._inductor.debug`: enable_aot_logging


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/_dynamo/backends`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`onnxrt.py_docs.md`](./onnxrt.py_docs.md)
- [`cudagraphs.py_docs.md`](./cudagraphs.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`debugging.py_docs.md`](./debugging.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`torchxla.py_docs.md`](./torchxla.py_docs.md)
- [`tensorrt.py_docs.md`](./tensorrt.py_docs.md)
- [`tvm.py_docs.md`](./tvm.py_docs.md)


## Cross-References

- **File Documentation**: `common.py_docs.md`
- **Keyword Index**: `common.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
