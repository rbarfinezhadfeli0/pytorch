# Documentation: `docs/torch/distributed/fsdp/_wrap_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/fsdp/_wrap_utils.py_docs.md`
- **Size**: 14,023 bytes (13.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/fsdp/_wrap_utils.py`

## File Metadata

- **Path**: `torch/distributed/fsdp/_wrap_utils.py`
- **Size**: 10,959 bytes (10.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import collections
import functools
import inspect
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, Union

import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state,
    _override_module_mixed_precision,
)
from torch.distributed.fsdp.wrap import (
    _construct_wrap_fn,
    _or_policy,
    _Policy,
    _post_order_apply,
    _recursive_wrap,
    _run_mixed_precision_override_policy,
    _wrap_module_cls_individually,
)


def _auto_wrap(
    root_module: nn.Module,
    policy: Union[Callable, _Policy],
    ignored_modules: set[nn.Module],
    ignored_params: set[nn.Parameter],
    root_kwargs: dict[str, Any],
    fsdp_fn: Callable,  # e.g. `FullyShardedDataParallel` or `fully_shard`
):
    """
    Auto wraps modules in ``root_module`` 's tree according to ``policy``
    following a post-order traversal.

    Precondition: ``root_kwargs`` should contain all arguments except
    ``module``. This function accepts the kwargs dict directly since it gets
    forwarded into the post-order traversal function.
    """
    mixed_precision = root_kwargs["mixed_precision"]
    is_wrapper = inspect.isclass(fsdp_fn)
    # TODO: We may relax this no-nested-wrapping constraint to support manual
    # wrapping followed by auto wrapping.
    _check_nested_wrapping(root_module)

    if isinstance(policy, _Policy):
        root_kwargs["auto_wrap_policy" if is_wrapper else "policy"] = None
        target_module_to_kwargs = policy._run_policy(
            root_module, ignored_modules, root_kwargs
        )
        if mixed_precision is not None:
            target_module_to_kwargs = _run_mixed_precision_override_policy(
                root_module,
                mixed_precision._module_classes_to_ignore,
                ignored_modules,
                root_kwargs,
                target_module_to_kwargs,
            )
            overridden_module_classes = _override_module_mixed_precision(
                root_module, mixed_precision._module_classes_to_ignore
            )
            _warn_on_overridden_mixed_precision(overridden_module_classes)
        use_orig_params = root_kwargs.get("use_orig_params", False)
        _validate_frozen_params(
            root_module,
            set(target_module_to_kwargs.keys()),
            ignored_params,
            use_orig_params,
        )
        wrap_fn = _construct_wrap_fn(root_module, target_module_to_kwargs, fsdp_fn)
        _post_order_apply(root_module, wrap_fn)
        return

    recursive_wrap_kwargs = {
        "module": root_module,
        "auto_wrap_policy": policy,
        "wrapper_cls": fsdp_fn,
        "ignored_modules": ignored_modules,
        "ignored_params": ignored_params,
        "only_wrap_children": True,
    }
    if mixed_precision is not None:
        # Wrap modules of the ignored types separately and register forward
        # hooks to cast to fp32 and back to the original dtype, respectively
        overridden_module_classes = _override_module_mixed_precision(
            root_module, mixed_precision._module_classes_to_ignore
        )
        policy = functools.partial(
            _or_policy,
            policies=[
                policy,
                partial(
                    _wrap_module_cls_individually,
                    module_classes=mixed_precision._module_classes_to_ignore,
                ),
            ],
        )
        recursive_wrap_kwargs["auto_wrap_policy"] = policy
        _warn_on_overridden_mixed_precision(overridden_module_classes)
    _recursive_wrap(**recursive_wrap_kwargs, **root_kwargs)  # type: ignore[arg-type]


def _check_nested_wrapping(root_module: nn.Module):
    for module_name, module in root_module.named_modules():
        if _get_module_fsdp_state(module) is not None:
            raise ValueError(
                "FSDP auto wrapping requires modules to not already have "
                f"FSDP applied but found {module_name} in\n{root_module}"
            )


def _warn_on_overridden_mixed_precision(
    overridden_module_classes: set[type[nn.Module]],
):
    if len(overridden_module_classes) == 0:
        return
    warnings.warn(
        "Both mixed precision and an auto_wrap_policy were specified to FSDP, "
        f"where the wrapped module has submodules of type:\n{overridden_module_classes}\n"
        "These modules will be wrapped as separate FSDP instacnes with mixed "
        "precision disabled.",
        stacklevel=2,
    )


def _validate_frozen_params(
    root_module: nn.Module,
    modules_to_wrap: set[nn.Module],
    ignored_params: set[nn.Parameter],
    use_orig_params: bool,
):
    """
    This checks that, given ``modules_to_wrap``, each module would manage
    parameters that are uniformly frozen or non-frozen. This uniformity
    requirement is strict for ``use_orig_params=False`` (hard error) and highly
    recommended for ``use_orig_params=True`` (user warning).
    """
    post_order_named_modules = _get_post_order_named_modules(root_module)
    visited_modules: set[nn.Module] = set()
    for module_name, module in post_order_named_modules:
        if module in modules_to_wrap:
            param_to_fqn = _get_managed_param_to_fqn(
                module, ignored_params, visited_modules, module_name
            )
            frozen_param_fqns: list[str] = []
            frozen_param_numel = 0
            nonfrozen_param_fqns: list[str] = []
            nonfrozen_param_numel = 0
            for param, fqn in param_to_fqn.items():
                if param.requires_grad:
                    nonfrozen_param_fqns.append(fqn)
                    nonfrozen_param_numel += param.numel()
                else:
                    frozen_param_fqns.append(fqn)
                    frozen_param_numel += param.numel()
            if len(frozen_param_fqns) > 0 and len(nonfrozen_param_fqns) > 0:
                msg = f"{module_name} has both parameters with requires_grad=True and False."
                if use_orig_params:
                    total_param_numel = frozen_param_numel + nonfrozen_param_numel
                    msg += (
                        " We do not recommend wrapping such modules since "
                        "the gradient memory usage will be higher than expected "
                        f"({total_param_numel} numel instead of {nonfrozen_param_numel} numel "
                        "before sharding via reduce-scatter). "
                    )
                else:
                    msg += " FSDP does not support wrapping such modules when use_orig_params=False. "
                msg += "If possible, wrap the frozen parameters with FSDP separately.\n"
                msg += (
                    f"The following parameters have requires_grad=True:\n{nonfrozen_param_fqns}\n"
                    f"The following parameters have requires_grad=False:\n{frozen_param_fqns}"
                )
                if use_orig_params:
                    warnings.warn(msg, stacklevel=2)
                else:
                    raise ValueError(msg)


def _get_post_order_named_modules(
    root_module: nn.Module,
) -> list[tuple[str, nn.Module]]:
    """
    This returns the named modules following a post-order traversal, which is a
    valid reverse topological sort. We achieve this using the reverse of a
    stack-based DFS order instead of reversing ``root_module.named_modules()``
    since the former gives the modules in registration order at each level in
    the module tree (as opposed to the reverse), which allows us to error/warn
    on the first registered module that violates the condition.

    For example, consider the following module structure:
        M(
          S1(),
          S2(
            SS1(),
            SS2(),
          ),
          S3(),
        )
    The reverse DFS order is [S1, SS1, SS2, S2, S3, M], while the reverse
    ``named_modules()`` order is [S3, SS2, SS1, S2, S1, M].
    """
    visited_modules = {root_module}
    stack = [("", root_module)]
    # Append and reverse at the end for linear-time algorithm
    reverse_post_order_named_modules: list[tuple[str, nn.Module]] = []
    while stack:
        module_name, module = stack.pop()
        reverse_post_order_named_modules.append((module_name, module))
        for child_module_name, child_module in module.named_children():
            if child_module is None:  # only for overrides of `named_children()`
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                if module_name != "":
                    child_module_name = module_name + "." + child_module_name
                stack.append((child_module_name, child_module))
    post_order_named_modules = list(reversed(reverse_post_order_named_modules))
    return post_order_named_modules


def _get_managed_param_to_fqn(
    module_to_wrap: nn.Module,
    ignored_params: set[nn.Parameter],
    visited_modules: set[nn.Module],
    root_prefix: str,
) -> dict[nn.Parameter, str]:
    """
    This returns a dict that maps managed parameter to its FQN for the given
    ``module_to_wrap``. The dict's keys are exactly the parameters that would
    be managed by the module, where this is achieved by calling this function
    on the modules to wrap in reverse topological order, destructively updating
    ``visited_modules``, and not traversing into those modules. The FQNs are
    prefixed from the root (via ``root_prefix``) to be more informative.

    NOTE: This function is meant to be called pre-wrapping and iteratively in
    reverse topological order to cover the full module tree. This differs from
    the ``_get_param_to_fqn()`` function meant to be called post-wrapping and
    on the full module tree in one shot. Given those differences, we do not try
    to unify the two.
    """
    param_to_fqn: dict[nn.Parameter, str] = {}
    # Run BFS (or any tree traversal works)
    queue = collections.deque([(module_to_wrap, root_prefix)])
    visited_modules.add(module_to_wrap)
    while queue:
        module, prefix = queue.popleft()
        for param_name, param in module.named_parameters(recurse=False):
            if param not in ignored_params:
                fqn = param_name if prefix == "" else prefix + "." + param_name
                param_to_fqn[param] = fqn
        for child_module_name, child_module in module.named_children():
            if child_module is None:  # only for overrides of `named_children()`
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                child_prefix = (
                    child_module_name
                    if prefix == ""
                    else prefix + "." + child_module_name
                )
                queue.append((child_module, child_prefix))
    return param_to_fqn

```



## High-Level Overview

"""    Auto wraps modules in ``root_module`` 's tree according to ``policy``    following a post-order traversal.    Precondition: ``root_kwargs`` should contain all arguments except    ``module``. This function accepts the kwargs dict directly since it gets    forwarded into the post-order traversal function.

This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_auto_wrap`, `_check_nested_wrapping`, `_warn_on_overridden_mixed_precision`, `_validate_frozen_params`, `_get_post_order_named_modules`, `_get_managed_param_to_fqn`

**Key imports**: collections, functools, inspect, warnings, Callable, partial, Any, Union, torch.nn as nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/fsdp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `functools`
- `inspect`
- `warnings`
- `collections.abc`: Callable
- `typing`: Any, Union
- `torch.nn as nn`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/distributed/fsdp`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_limiter_utils.py_docs.md`](./_limiter_utils.py_docs.md)
- [`_traversal_utils.py_docs.md`](./_traversal_utils.py_docs.md)
- [`_runtime_utils.py_docs.md`](./_runtime_utils.py_docs.md)
- [`_common_utils.py_docs.md`](./_common_utils.py_docs.md)
- [`_exec_order_utils.py_docs.md`](./_exec_order_utils.py_docs.md)
- [`sharded_grad_scaler.py_docs.md`](./sharded_grad_scaler.py_docs.md)
- [`_state_dict_utils.py_docs.md`](./_state_dict_utils.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `_wrap_utils.py_docs.md`
- **Keyword Index**: `_wrap_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/fsdp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/distributed/fsdp`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`_limiter_utils.py_kw.md_docs.md`](./_limiter_utils.py_kw.md_docs.md)
- [`_optim_utils.py_kw.md_docs.md`](./_optim_utils.py_kw.md_docs.md)
- [`fully_sharded_data_parallel.py_kw.md_docs.md`](./fully_sharded_data_parallel.py_kw.md_docs.md)
- [`_state_dict_utils.py_kw.md_docs.md`](./_state_dict_utils.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`_exec_order_utils.py_docs.md_docs.md`](./_exec_order_utils.py_docs.md_docs.md)
- [`_flat_param.py_docs.md_docs.md`](./_flat_param.py_docs.md_docs.md)
- [`_wrap_utils.py_kw.md_docs.md`](./_wrap_utils.py_kw.md_docs.md)
- [`sharded_grad_scaler.py_docs.md_docs.md`](./sharded_grad_scaler.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_wrap_utils.py_docs.md_docs.md`
- **Keyword Index**: `_wrap_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
