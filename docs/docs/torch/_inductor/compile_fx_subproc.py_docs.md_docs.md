# Documentation: `docs/torch/_inductor/compile_fx_subproc.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/compile_fx_subproc.py_docs.md`
- **Size**: 6,332 bytes (6.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/compile_fx_subproc.py`

## File Metadata

- **Path**: `torch/_inductor/compile_fx_subproc.py`
- **Size**: 3,171 bytes (3.10 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import atexit
import functools
import os
from typing import Optional, TYPE_CHECKING
from typing_extensions import final, override

import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch.fx
from torch._inductor.compile_worker.subproc_pool import (
    AnyPool,
    SubprocKind,
    SubprocPool,
)
from torch._inductor.utils import clear_caches

from .compile_fx_ext import (
    _OutOfProcessFxCompile,
    _WireProtocolPickledInput,
    _WireProtocolPickledOutput,
)
from .output_code import complex_memory_overlap  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import Mapping
    from concurrent.futures import Future


@final
class _SubprocessFxCompile(_OutOfProcessFxCompile):
    @override
    def _send_to_child_async(
        self, input: _WireProtocolPickledInput
    ) -> Future[_WireProtocolPickledOutput]:
        # TODO: Do we need to copy across some kind of logging IDs? (ChromiumEventLogger)

        pool = self.process_pool()

        # TODO: This is probably the wrong thing to do long-term - but for now
        # let's share the cache so we can identify tests broken by this later.
        env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
        extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}

        return pool.submit(
            _SubprocessFxCompile._run_in_child_subprocess, input, extra_env
        )

    @staticmethod
    @functools.cache
    def process_pool() -> AnyPool:
        pool = SubprocPool(
            # TODO: Consider raising this limit if we start using async w/
            # subprocess and want to compile multiple graphs in parallel.
            1,
            kind=SubprocKind.SPAWN,
        )

        atexit.register(pool.shutdown)

        return pool

    @classmethod
    def _run_in_child_subprocess(
        cls,
        pickled_input: _WireProtocolPickledInput,
        extra_env: Optional[Mapping[str, str]],
    ) -> _WireProtocolPickledOutput:
        # TODO: In subprocess mode we need to clear the inductor caches.
        # The problem:
        #   1. We compile in worker A which fills stuff in tmpdir
        #   2. parent clears inductor caches which deletes tmpdirs and tells
        #      cpp_prefix_path() to clear its LRU cache
        #   3. We compile a second time in subproc A - but since we never told
        #      cpp_prefix_path() in worker A to clear its LRU it thinks the
        #      tmpdir still exists and fails to compile.
        #
        # TODO: We probably should be using a separate tmpdir in the worker
        # anyway... but we should probably still respect clear_caches()
        # in the parent... maybe?
        #
        # TODO: We could be less aggressive by keeping a clock which gets
        # incremented when we clear the cache, send the clock to the worker and
        # only clear caches if the clock changed since last time.
        #
        clear_caches()
        torch._inductor.metrics.reset()

        # TODO: turn off config.fx_graph_async_compile

        result = cls._run_in_child(pickled_input, extra_env)
        return result

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_SubprocessFxCompile`

**Functions defined**: `_send_to_child_async`, `process_pool`, `_run_in_child_subprocess`

**Key imports**: annotations, atexit, functools, os, Optional, TYPE_CHECKING, final, override, torch._inductor.async_compile  , torch.fx, clear_caches, complex_memory_overlap  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `atexit`
- `functools`
- `os`
- `typing`: Optional, TYPE_CHECKING
- `typing_extensions`: final, override
- `torch._inductor.async_compile  `
- `torch.fx`
- `torch._inductor.utils`: clear_caches
- `.output_code`: complex_memory_overlap  
- `collections.abc`: Mapping
- `concurrent.futures`: Future


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

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

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

- **File Documentation**: `compile_fx_subproc.py_docs.md`
- **Keyword Index**: `compile_fx_subproc.py_kw.md`
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

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

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

- **File Documentation**: `compile_fx_subproc.py_docs.md_docs.md`
- **Keyword Index**: `compile_fx_subproc.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
