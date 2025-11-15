# Documentation: `torch/distributed/fsdp/_debug_utils.py`

## File Metadata

- **Path**: `torch/distributed/fsdp/_debug_utils.py`
- **Size**: 5,818 bytes (5.68 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum

import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
from torch.distributed.fsdp._common_utils import (
    _apply_to_modules,
    _get_module_fsdp_state,
    clean_tensor_name,
)


logger = logging.getLogger(__name__)


class SimpleProfiler:
    class Type(str, Enum):
        ALL = "all"
        ALLGATHER = "all_gather"
        ALLGATHER_OBJ = "all_gather_object"
        RESHARDING = "resharding"
        H2D = "H2D"
        D2H = "D2H"

    results: dict[str, float] = defaultdict(float)
    profiling: set[str] = set()

    @classmethod
    def reset(cls) -> None:
        cls.results.clear()
        cls.profiling.clear()

    @classmethod
    @contextmanager
    def profile(cls, profile_type: str) -> Iterator[None]:
        if profile_type in cls.profiling:
            raise AssertionError(
                f"{profile_type} is already being profiled. "
                "SimpleProfiler does not support profiling multiple instances at "
                "the same time. "
            )

        cls.profiling.add(profile_type)
        begin = time.monotonic()
        try:
            yield
        finally:
            end = time.monotonic()
            cls.results[profile_type] += end - begin
            cls.profiling.remove(profile_type)

    @classmethod
    def dump_and_reset(cls, msg: str) -> None:
        # This cannot be combined with DETAIL distributed log
        # as the profiling will be very incorrect.
        if dist.get_rank() == 0 and dist.get_debug_level() == dist.DebugLevel.INFO:
            logger.info("%s %s", msg, cls.results)
        cls.reset()


def _get_sharded_module_tree_with_module_name_to_fqns(
    model: torch.nn.Module,
) -> tuple[str, dict[str, list[str]]]:
    """
    It is used for composable fully_shard() code path, it returns
      1. sharded module tree info: each line represents a submodule name that contains the
    submodule's FQN and its submodule class name, if the submodule is sharded by `fully_shard`,
    the submodule name will add a postfix with ' FULLY SHARDED'. Each increased tree
    level adds 4 spaces before the printed name. A printed sharded module tree info for a toy model
    is like this:
        [CompositeModel] FULLY SHARDED
            l1[Linear]
            u1[UnitModule] FULLY SHARDED
                u1.l1[Linear]
                u1.seq[Sequential]
                    u1.seq.0[ReLU]
                    u1.seq.1[Linear]
                    u1.seq.2[ReLU]
                u1.l2[Linear]
            u2[UnitModule] FULLY SHARDED
                u2.l1[Linear]
                u2.seq[Sequential]
                    u2.seq.0[ReLU]
                    u2.seq.1[Linear]
                    u2.seq.2[ReLU]
                u2.l2[Linear]
            l2[Linear]
      2. a dict mapping from the concated module FQN and class name to a list of its managed
    original parameters' FQNs. An example of the dict for the above toy sharded model is like this:
            {'[CompositeModel]': ['l1.weight', 'l1.bias', 'l2.weight', 'l2.bias'],
             'u1[UnitModule]': ['u1.l1.weight', 'u1.l1.bias', 'u1.seq.1.weight', 'u1.seq.1.bias', 'u1.l2.weight', 'u1.l2.bias'],
             'u2[UnitModule]': ['u2.l1.weight', 'u2.l1.bias', 'u2.seq.1.weight', 'u2.seq.1.bias', 'u2.l2.weight', 'u2.l2.bias']
            }
    All FQNs are prefixed starting from ``model``.

    Args:
        model (torch.nn.Module): Root module (which may or may not be passed to
                                 composable `fully_shard()`).
    """

    def module_fn(
        module, prefix, tree_level, sharded_tree_info, sharded_module_name_to_fqns
    ):
        num_spaces = tree_level * 4
        trimed_prefix = (
            prefix[:-1] if (len(prefix) > 0 and prefix[-1] == ".") else prefix
        )
        prefixed_module_name = trimed_prefix + "[" + module.__class__.__name__ + "]"
        printed_prefixed_module_name = " " * num_spaces + prefixed_module_name

        state = _get_module_fsdp_state(module)
        if state is None:
            sharded_tree_info[0] += printed_prefixed_module_name + "\n"
            return

        handle = state._fully_sharded_module_to_handle.get(module, None)

        if handle:
            sharded_tree_info[0] += (
                printed_prefixed_module_name + " FULLY SHARDED" + "\n"
            )
        else:
            sharded_tree_info[0] += printed_prefixed_module_name + "\n"

        if handle:
            param = handle.flat_param
            if not isinstance(param, flat_param_file.FlatParameter):
                raise AssertionError(f"Expected FlatParameter, got {type(param)}")
            global_fqns = [
                clean_tensor_name(prefix + name) for name in param._fqns
            ]  # prefixed from the top level `model` (i.e. including `prefix`)

            if prefixed_module_name in sharded_module_name_to_fqns:
                sharded_module_name_to_fqns[prefixed_module_name].extend(global_fqns)
            else:
                sharded_module_name_to_fqns[prefixed_module_name] = global_fqns

    def return_fn(sharded_tree_info, sharded_module_name_to_fqns):
        return sharded_tree_info[0], sharded_module_name_to_fqns

    # Use List to mutate its value in place while running the recursive functions
    sharded_tree_info: list[str] = [
        "",
    ]
    sharded_module_name_to_fqns: dict[str, list[str]] = {}
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [key for key, _ in model.named_parameters()],
        sharded_tree_info,
        sharded_module_name_to_fqns,
    )

```



## High-Level Overview


This Python file contains 4 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SimpleProfiler`, `Type`

**Functions defined**: `reset`, `profile`, `dump_and_reset`, `_get_sharded_module_tree_with_module_name_to_fqns`, `module_fn`, `return_fn`

**Key imports**: logging, time, defaultdict, Iterator, contextmanager, Enum, torch, torch.distributed as dist, torch.distributed.fsdp._flat_param as flat_param_file


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/fsdp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `time`
- `collections`: defaultdict
- `collections.abc`: Iterator
- `contextlib`: contextmanager
- `enum`: Enum
- `torch`
- `torch.distributed as dist`
- `torch.distributed.fsdp._flat_param as flat_param_file`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/distributed/fsdp`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_limiter_utils.py_docs.md`](./_limiter_utils.py_docs.md)
- [`_traversal_utils.py_docs.md`](./_traversal_utils.py_docs.md)
- [`_runtime_utils.py_docs.md`](./_runtime_utils.py_docs.md)
- [`_common_utils.py_docs.md`](./_common_utils.py_docs.md)
- [`_wrap_utils.py_docs.md`](./_wrap_utils.py_docs.md)
- [`_exec_order_utils.py_docs.md`](./_exec_order_utils.py_docs.md)
- [`sharded_grad_scaler.py_docs.md`](./sharded_grad_scaler.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `_debug_utils.py_docs.md`
- **Keyword Index**: `_debug_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
