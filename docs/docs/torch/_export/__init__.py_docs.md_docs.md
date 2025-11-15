# Documentation: `docs/torch/_export/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/__init__.py_docs.md`
- **Size**: 9,817 bytes (9.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/_export/__init__.py`

## File Metadata

- **Path**: `torch/_export/__init__.py`
- **Size**: 6,687 bytes (6.53 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import copy
import dataclasses
import functools
import io
import json
import logging
import os
import re
import sys
import types
import warnings
import weakref
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache

from typing import Any, Optional, TYPE_CHECKING, Union
from collections.abc import Callable
from unittest.mock import patch

import torch
import torch.fx
import torch.utils._pytree as pytree

from torch._dispatch.python import enable_python_dispatcher
from torch._guards import compile_context
from torch._utils_internal import log_export_usage
from torch.export._tree_utils import reorder_kwargs
from torch.export.graph_signature import (
    ArgumentSpec,
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    SymBoolArgument,
    SymFloatArgument,
    TensorArgument,
)
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from .wrappers import _wrap_submodules
from .utils import _materialize_cpp_cia_ops
from . import config

if TYPE_CHECKING:
    from torch._C._aoti import AOTIModelContainerRunner

log = logging.getLogger(__name__)

@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """
    allow_rnn: bool = True


# We only want to print this once to avoid flooding logs in workflows where aot_compile_warning
# is called multiple times.
@lru_cache
def aot_compile_warning():

    log.warning("+============================+")
    log.warning("|     !!!   WARNING   !!!    |")
    log.warning("+============================+")
    log.warning(
        "torch._export.aot_compile()/torch._export.aot_load() is being deprecated, please switch to "
        "directly calling torch._inductor.aoti_compile_and_package(torch.export.export())/"
        "torch._inductor.aoti_load_package() instead.")


def aot_compile(
    f: Callable,
    args: tuple[Any],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[dict[str, Any]] = None,
    options: Optional[dict[str, Any]] = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
    same_signature: bool = True,
) -> Union[list[Any], str]:
    """
    Note: this function is not stable yet

    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside, generates executable cpp code from the program, and returns
    the path to the generated shared library

    Args:
        f: the `nn.Module` or callable to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        dynamic_shapes: Should either be:
            1) a dict from argument names of ``f`` to their dynamic shape specifications,
            2) a tuple that specifies dynamic shape specifications for each input in original order.
            If you are specifying dynamism on keyword args, you will need to pass them in the order that
            is defined in the original function signature.

            The dynamic shape of a tensor argument can be specified as either
            (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
            not required to include static dimension indices in this dict, but when they are,
            they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
            where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
            are denoted by None. Arguments that are dicts or tuples / lists of tensors are
            recursively specified by using mappings or sequences of contained specifications.

        options: A dictionary of options to control inductor

        disable_constraint_solver: Whether the dim constraint solver must be disabled.

    Returns:
        Path to the generated shared library
    """
    from torch.export._trace import _export_to_torch_ir
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor import config as inductor_config

    aot_compile_warning()

    if inductor_config.is_predispatch:
        gm = torch.export._trace._export(f, args, kwargs, dynamic_shapes, pre_dispatch=True).module()
    else:
        # We want to export to Torch IR here to utilize the pre_grad passes in
        # inductor, which run on Torch IR.
        with torch._export.config.patch(use_new_tracer_experimental=True):
            gm = _export_to_torch_ir(
                f,
                args,
                kwargs,
                dynamic_shapes,
                disable_constraint_solver=disable_constraint_solver,
                same_signature=same_signature,
                # Disabling this flag, because instead we can rely on the mapping
                # dynamo_flat_name_to_original_fqn which is coming from Dynamo.
                restore_fqn=False,
            )

    with torch.no_grad():
        so_path = torch._inductor.aot_compile(gm, args, kwargs, options=options)  # type: ignore[arg-type]

    assert isinstance(so_path, (str, list))
    return so_path

def aot_load(so_path: str, device: str) -> Callable:
    """
    Loads a shared library generated by aot_compile and returns a callable

    Args:
        so_path: Path to the shared library

    Returns:
        A callable
    """
    aot_compile_warning()

    if device == "cpu":
        runner: AOTIModelContainerRunner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
    elif device == "cuda" or device.startswith("cuda:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)
    elif device == "xpu" or device.startswith("xpu:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerXpu(so_path, 1, device)
    elif device == "mps" or device.startswith("mps:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerMps(so_path, 1)
    else:
        raise RuntimeError("Unsupported device " + device)

    def optimized(*args, **kwargs):
        call_spec = runner.get_call_spec()
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
        flat_outputs = runner.run(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized

```



## High-Level Overview


This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExportDynamoConfig`

**Functions defined**: `aot_compile_warning`, `aot_compile`, `aot_load`, `optimized`

**Key imports**: copy, dataclasses, functools, io, json, logging, os, re, sys, types


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `dataclasses`
- `functools`
- `io`
- `json`
- `logging`
- `os`
- `re`
- `sys`
- `types`
- `warnings`
- `weakref`
- `zipfile`
- `collections`: OrderedDict
- `contextlib`: contextmanager
- `typing`: Any, Optional, TYPE_CHECKING, Union
- `collections.abc`: Callable
- `unittest.mock`: patch
- `torch`
- `torch.fx`
- `torch.utils._pytree as pytree`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._guards`: compile_context
- `torch._utils_internal`: log_export_usage
- `torch.export._tree_utils`: reorder_kwargs
- `torch.fx._compatibility`: compatibility
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.fx.graph`: _PyTreeCodeGen, _PyTreeInfo


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/_export`):

- [`utils.py_docs.md`](./utils.py_docs.md)
- [`error.py_docs.md`](./error.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`pass_base.py_docs.md`](./pass_base.py_docs.md)
- [`tools.py_docs.md`](./tools.py_docs.md)
- [`non_strict_utils.py_docs.md`](./non_strict_utils.py_docs.md)
- [`converter.py_docs.md`](./converter.py_docs.md)
- [`wrappers.py_docs.md`](./wrappers.py_docs.md)
- [`verifier.py_docs.md`](./verifier.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/_export`):

- [`error.py_kw.md_docs.md`](./error.py_kw.md_docs.md)
- [`converter.py_kw.md_docs.md`](./converter.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`pass_base.py_kw.md_docs.md`](./pass_base.py_kw.md_docs.md)
- [`wrappers.py_docs.md_docs.md`](./wrappers.py_docs.md_docs.md)
- [`converter.py_docs.md_docs.md`](./converter.py_docs.md_docs.md)
- [`config.py_kw.md_docs.md`](./config.py_kw.md_docs.md)
- [`verifier.py_kw.md_docs.md`](./verifier.py_kw.md_docs.md)
- [`verifier.py_docs.md_docs.md`](./verifier.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
