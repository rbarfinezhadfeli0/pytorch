# Documentation: `docs/source/notes/broadcasting.rst`

## File Metadata

- **Path**: `docs/source/notes/broadcasting.rst`
- **Size**: 4,432 bytes (4.33 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
.. _broadcasting-semantics:

Broadcasting semantics
======================

Many PyTorch operations support NumPy's broadcasting semantics.
See https://numpy.org/doc/stable/user/basics.broadcasting.html for details.

In short, if a PyTorch operation supports broadcast, then its Tensor arguments can be
automatically expanded to be of equal sizes (without making copies of the data).

General semantics
-----------------
Two tensors are "broadcastable" if the following rules hold:

- When iterating over the dimension sizes, starting at the trailing dimension,
  the dimension sizes must either be equal, one of them is 1, or one of them
  does not exist.

For Example::

    >>> x=torch.empty(5,7,3)
    >>> y=torch.empty(5,7,3)
    # same shapes are always broadcastable (i.e. the above rules always hold)

    >>> x=torch.empty((0,))
    >>> y=torch.empty(2,2)
    # x and y are not broadcastable, because the 0-sized dimension of x
    # does not match the 2-sized dimension of y.

    # can line up trailing dimensions
    >>> x=torch.empty(5,3,4,1)
    >>> y=torch.empty(  3,1,1)
    # x and y are broadcastable.
    # 1st trailing dimension: both have size 1
    # 2nd trailing dimension: y has size 1
    # 3rd trailing dimension: x size == y size
    # 4th trailing dimension: y dimension doesn't exist

    # but:
    >>> x=torch.empty(5,2,4,1)
    >>> y=torch.empty(  3,1,1)
    # x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

If two tensors :attr:`x`, :attr:`y` are "broadcastable", the resulting tensor size
is calculated as follows:

- If the number of dimensions of :attr:`x` and :attr:`y` are not equal, prepend 1
  to the dimensions of the tensor with fewer dimensions to make them equal length.
- Then, for each dimension size, the resulting dimension size is the max of the sizes of
  :attr:`x` and :attr:`y` along that dimension.

For Example::

    # can line up trailing dimensions to make reading easier
    >>> x=torch.empty(5,1,4,1)
    >>> y=torch.empty(  3,1,1)
    >>> (x+y).size()
    torch.Size([5, 3, 4, 1])

    # but not necessary:
    >>> x=torch.empty(1)
    >>> y=torch.empty(3,1,7)
    >>> (x+y).size()
    torch.Size([3, 1, 7])

    >>> x=torch.empty(5,2,4,1)
    >>> y=torch.empty(3,1,1)
    >>> (x+y).size()
    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

In-place semantics
------------------
One complication is that in-place operations do not allow the in-place tensor to change shape
as a result of the broadcast.

For Example::

    >>> x=torch.empty(5,3,4,1)
    >>> y=torch.empty(3,1,1)
    >>> (x.add_(y)).size()
    torch.Size([5, 3, 4, 1])

    # but:
    >>> x=torch.empty(1,3,1)
    >>> y=torch.empty(3,1,7)
    >>> (x.add_(y)).size()
    RuntimeError: The expanded size of the tensor (1) must match the existing size (7) at non-singleton dimension 2.

Backwards compatibility
-----------------------
Prior versions of PyTorch allowed certain pointwise functions to execute on tensors with different shapes,
as long as the number of elements in each tensor was equal.  The pointwise operation would then be carried
out by viewing each tensor as 1-dimensional.  PyTorch now supports broadcasting and the "1-dimensional"
pointwise behavior is considered deprecated and will generate a Python warning in cases where tensors are
not broadcastable, but have the same number of elements.

Note that the introduction of broadcasting can cause backwards incompatible changes in the case where
two tensors do not have the same shape, but are broadcastable and have the same number of elements.
For Example::

    >>> torch.add(torch.ones(4,1), torch.randn(4))

would previously produce a Tensor with size: torch.Size([4,1]), but now produces a Tensor with size: torch.Size([4,4]).
In order to help identify cases in your code where backwards incompatibilities introduced by broadcasting may exist,
you may set `torch.utils.backcompat.broadcast_warning.enabled` to `True`, which will generate a python warning
in such cases.

For Example::

    >>> torch.utils.backcompat.broadcast_warning.enabled=True
    >>> torch.add(torch.ones(4,1), torch.ones(4))
    __main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
    Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 1-dimensional.

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source/notes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source/notes`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/source/notes`):

- [`windows.rst_docs.md`](./windows.rst_docs.md)
- [`get_start_xpu.rst_docs.md`](./get_start_xpu.rst_docs.md)
- [`amp_examples.rst_docs.md`](./amp_examples.rst_docs.md)
- [`autograd.rst_docs.md`](./autograd.rst_docs.md)
- [`cpu_threading_torchscript_inference.rst_docs.md`](./cpu_threading_torchscript_inference.rst_docs.md)
- [`hip.rst_docs.md`](./hip.rst_docs.md)
- [`libtorch_stable_abi.md_docs.md`](./libtorch_stable_abi.md_docs.md)
- [`cuda.rst_docs.md`](./cuda.rst_docs.md)
- [`out.rst_docs.md`](./out.rst_docs.md)


## Cross-References

- **File Documentation**: `broadcasting.rst_docs.md`
- **Keyword Index**: `broadcasting.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
