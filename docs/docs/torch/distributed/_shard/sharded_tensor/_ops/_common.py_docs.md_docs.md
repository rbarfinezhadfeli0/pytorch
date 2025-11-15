# Documentation: `docs/torch/distributed/_shard/sharded_tensor/_ops/_common.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/_shard/sharded_tensor/_ops/_common.py_docs.md`
- **Size**: 7,646 bytes (7.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/_shard/sharded_tensor/_ops/_common.py`

## File Metadata

- **Path**: `torch/distributed/_shard/sharded_tensor/_ops/_common.py`
- **Size**: 4,280 bytes (4.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools

from torch.distributed._shard.common_op_utils import _basic_validation
from torch.distributed._shard.sharded_tensor import (
    _sharded_op_impl,
    Shard,
    ShardedTensor,
)


def _sharded_op_common(op, early_stop_func, extra_check):
    """
    Inject sharded tensor op registration with common logics executed before
    different behaviors are done on either local shards or a local tensor.

    Example::
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> op = torch.transpose
        >>> @_sharded_op_impl(op)
        >>> @_sharded_op_common(op, early_stop_func, extra_check)
        >>> def sharded_tensor_op(types, args, kwargs, process_group):
        >>>   ...
        >>>
        >>> st = sharded_tensor.rand(32, 16)
        >>> st.transpose(1, 2)
        >>> # This will call '_sharded_op_common'

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.

    Return:
        func (Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.transpose)
    """

    def decorator_sharded_func(wrapped_func):
        @functools.wraps(wrapped_func)
        def wrapper(types, args=(), kwargs=None, pg=None):
            _basic_validation(op, args, kwargs)

            # pyrefly: ignore [index-error]
            st = args[0]
            if kwargs is None:
                kwargs = {}
            if extra_check:
                extra_check(*args, **kwargs)
            if early_stop_func:
                early_stop = early_stop_func(*args, **kwargs)
                if early_stop:
                    return st
            return wrapped_func(types, args, kwargs, pg)

        return wrapper

    return decorator_sharded_func


def _register_sharded_op_on_local_shards(
    op, early_stop_func=None, extra_check=None, customized_func=None
):
    """
    Handles ``__torch_function__`` dispatch for ops which are performed on
    each shard of the sharded tensor such as elementwise op like
    ``torch.nn.functional.gelu`` or ``torch.nn.functional.relu``.

    For more complicated ops, a customized func can be used to generate
    the new shards and sharded tensor size.

    This function expects that the original ShardingSpec for the ShardedTensor
    is preserved irrespective of whether or not a customized function is used.

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.
        customized_func (Callable, optional): the func for customized logic
            to generate new shards and sharded tensor size.
            Default: if ``None``, we simply lower to the real op call with
                all local shards of the st.

    Return:
        func (Callable): registered implementation for sharded op for
        ``__torch_function__`` dispatch.
    """

    @_sharded_op_impl(op)
    @_sharded_op_common(op, early_stop_func, extra_check)
    def sharded_tensor_op_on_local_shards(types, args=(), kwargs=None, pg=None):
        # pyrefly: ignore [index-error]
        st = args[0]
        st_metadata = st.metadata()
        local_shards = st.local_shards()
        local_shards_new = []
        if customized_func:
            local_shards_new, st_metadata = customized_func(args, kwargs, pg)
        else:
            for local_shard in local_shards:
                args = (local_shard.tensor, *args[1:])
                local_shards_new.append(
                    Shard(op(*args, **kwargs), local_shard.metadata)
                )
        return ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards_new,
            st_metadata,
            process_group=pg,
            init_rrefs=st._init_rrefs,
            sharding_spec=st.sharding_spec(),
        )

```



## High-Level Overview

"""    Inject sharded tensor op registration with common logics executed before    different behaviors are done on either local shards or a local tensor.    Example::        >>> # xdoctest: +SKIP("Undefined variables")        >>> op = torch.transpose        >>> @_sharded_op_impl(op)        >>> @_sharded_op_common(op, early_stop_func, extra_check)        >>> def sharded_tensor_op(types, args, kwargs, process_group):        >>>   ...        >>>        >>> st = sharded_tensor.rand(32, 16)        >>> st.transpose(1, 2)        >>> # This will call '_sharded_op_common'    Args:        op: The op to be registered and applied to all shards of the st.        early_stop_func (Callable, optional): the func for early stop.            Default: if ``None``, no early stop.        extra_check (Callable, optional): the func for extra condition check.            Default: if ``None``, no extra check.    Return:        func (Callable): Torch function for which we want to provide a sharded            implementation (ex: torch.transpose)

This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_sharded_op_common`, `sharded_tensor_op`, `decorator_sharded_func`, `wrapper`, `_register_sharded_op_on_local_shards`, `sharded_tensor_op_on_local_shards`

**Key imports**: functools, _basic_validation


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/_shard/sharded_tensor/_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `torch.distributed._shard.common_op_utils`: _basic_validation


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

Files in the same folder (`torch/distributed/_shard/sharded_tensor/_ops`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`misc_ops.py_docs.md`](./misc_ops.py_docs.md)
- [`binary_cmp.py_docs.md`](./binary_cmp.py_docs.md)
- [`init.py_docs.md`](./init.py_docs.md)
- [`tensor_ops.py_docs.md`](./tensor_ops.py_docs.md)


## Cross-References

- **File Documentation**: `_common.py_docs.md`
- **Keyword Index**: `_common.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/_shard/sharded_tensor/_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/_shard/sharded_tensor/_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/distributed/_shard/sharded_tensor/_ops`):

- [`init.py_docs.md_docs.md`](./init.py_docs.md_docs.md)
- [`tensor_ops.py_kw.md_docs.md`](./tensor_ops.py_kw.md_docs.md)
- [`binary_cmp.py_kw.md_docs.md`](./binary_cmp.py_kw.md_docs.md)
- [`tensor_ops.py_docs.md_docs.md`](./tensor_ops.py_docs.md_docs.md)
- [`misc_ops.py_docs.md_docs.md`](./misc_ops.py_docs.md_docs.md)
- [`init.py_kw.md_docs.md`](./init.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`_common.py_kw.md_docs.md`](./_common.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_common.py_docs.md_docs.md`
- **Keyword Index**: `_common.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
