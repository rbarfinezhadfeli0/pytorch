# Documentation: `docs/torch/distributed/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/__init__.py_docs.md`
- **Size**: 8,710 bytes (8.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/distributed/__init__.py`

## File Metadata

- **Path**: `torch/distributed/__init__.py`
- **Size**: 5,505 bytes (5.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import logging
import pdb
import sys
import traceback
import typing
from datetime import timedelta

import torch


log = logging.getLogger(__name__)


def is_available() -> bool:
    """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch._C, "_c10d_init")


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize torch.distributed")

# Custom Runtime Errors thrown from the distributed package
DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError
QueueEmptyError = torch._C._DistQueueEmptyError

if is_available():
    from torch._C._distributed_c10d import (
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _ControlCollectives,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _make_nccl_premul_sum,
        _register_builtin_comm_hook,
        _register_comm_hook,
        _StoreCollectives,
        _test_python_store,
        _verify_params_across_processes,
        Backend as _Backend,
        BuiltinCommHookType,
        DebugLevel,
        FileStore,
        get_debug_level,
        GradBucket,
        Logger,
        PrefixStore,
        ProcessGroup as ProcessGroup,
        Reducer,
        set_debug_level,
        set_debug_level_from_env,
        Store,
        TCPStore,
        Work as _Work,
    )

    class _DistributedPdb(pdb.Pdb):
        """
        Supports using PDB from inside a multiprocessing child process.

        Usage:
        _DistributedPdb().set_trace()
        """

        def interaction(self, *args, **kwargs):
            _stdin = sys.stdin
            try:
                sys.stdin = open("/dev/stdin")
                pdb.Pdb.interaction(self, *args, **kwargs)
            finally:
                sys.stdin = _stdin

    _breakpoint_cache: dict[int, typing.Any] = {}

    def breakpoint(rank: int = 0, skip: int = 0, timeout_s=3600):
        """
        Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
        done with the breakpoint before continuing.

        Args:
            rank (int): Which rank to break on.  Default: ``0``
            skip (int): Skip the first ``skip`` calls to this breakpoint. Default: ``0``.
        """
        if skip > 0:
            key = hash(str(traceback.format_exc()))
            counter = _breakpoint_cache.get(key, 0) + 1
            _breakpoint_cache[key] = counter
            if counter <= skip:
                log.warning("Skip the breakpoint, counter=%d", counter)
                return

        # avoid having the default timeout (if short) interrupt your debug session
        if timeout_s is not None:
            for group in torch.distributed.distributed_c10d._pg_map:
                torch.distributed.distributed_c10d._set_pg_timeout(
                    timedelta(seconds=timeout_s), group
                )

        if get_rank() == rank:
            pdb = _DistributedPdb()
            pdb.message(
                "\n!!! ATTENTION !!!\n\n"
                f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
            )
            pdb.set_trace()
        # If Meta/Python keys are in the TLS, we want to make sure that we ignore them
        # and hit the (default) CPU/CUDA implementation of barrier.
        meta_in_tls = torch._C._meta_in_tls_dispatch_include()
        guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
        torch._C._set_meta_in_tls_dispatch_include(False)
        try:
            barrier()
        finally:
            torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)
            del guard

    if sys.platform != "win32":
        from torch._C._distributed_c10d import HashStore

    from .device_mesh import DeviceMesh, init_device_mesh

    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.
    # pyrefly: ignore [deprecated]
    from .distributed_c10d import *  # noqa: F403
    from .distributed_c10d import (  # pyrefly: ignore  # deprecated; pyrefly: ignore [deprecated]
        _all_gather_base,
        _coalescing_manager,
        _CoalescingManager,
        _create_process_group_wrapper,
        _get_process_group_name,
        _rank_not_in_group,
        _reduce_scatter_base,
        _time_estimator,
        get_node_local_rank,
    )
    from .remote_device import _remote_device
    from .rendezvous import (
        _create_store_from_options,
        register_rendezvous_handler,
        rendezvous,
    )

    set_debug_level_from_env()

else:
    # This stub is sufficient to get
    #   python test/test_public_bindings.py -k test_correct_module_names
    # working even when USE_DISTRIBUTED=0.  Feel free to add more
    # stubs as necessary.
    # We cannot define stubs directly because they confuse pyre

    class _ProcessGroupStub:
        pass

    sys.modules["torch.distributed"].ProcessGroup = _ProcessGroupStub  # type: ignore[attr-defined]

```



## High-Level Overview

"""    Return ``True`` if the distributed package is available.    Otherwise,    ``torch.distributed`` does not expose any other APIs. Currently,    ``torch.distributed`` is available on Linux, MacOS and Windows. Set    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,    ``USE_DISTRIBUTED=0`` for MacOS.

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_DistributedPdb`, `_ProcessGroupStub`

**Functions defined**: `is_available`, `interaction`, `breakpoint`

**Key imports**: logging, pdb, sys, traceback, typing, timedelta, torch, HashStore, DeviceMesh, init_device_mesh, _remote_device


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `pdb`
- `sys`
- `traceback`
- `typing`
- `datetime`: timedelta
- `torch`
- `torch._C._distributed_c10d`: HashStore
- `.device_mesh`: DeviceMesh, init_device_mesh
- `.remote_device`: _remote_device


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed`):

- [`_mesh_layout.py_docs.md_docs.md`](./_mesh_layout.py_docs.md_docs.md)
- [`run.py_docs.md_docs.md`](./run.py_docs.md_docs.md)
- [`device_mesh.py_docs.md_docs.md`](./device_mesh.py_docs.md_docs.md)
- [`_composable_state.py_docs.md_docs.md`](./_composable_state.py_docs.md_docs.md)
- [`run.py_kw.md_docs.md`](./run.py_kw.md_docs.md)
- [`_dist2.py_kw.md_docs.md`](./_dist2.py_kw.md_docs.md)
- [`_state_dict_utils.py_kw.md_docs.md`](./_state_dict_utils.py_kw.md_docs.md)
- [`rendezvous.py_kw.md_docs.md`](./rendezvous.py_kw.md_docs.md)
- [`rendezvous.py_docs.md_docs.md`](./rendezvous.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
