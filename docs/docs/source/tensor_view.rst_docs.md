# Documentation: `docs/source/tensor_view.rst`

## File Metadata

- **Path**: `docs/source/tensor_view.rst`
- **Size**: 4,087 bytes (3.99 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
.. currentmodule:: torch

.. _tensor-view-doc:

Tensor Views
=============

PyTorch allows a tensor to be a ``View`` of an existing tensor. View tensor shares the same underlying data
with its base tensor. Supporting ``View`` avoids explicit data copy, thus allows us to do fast and memory efficient
reshaping, slicing and element-wise operations.

For example, to get a view of an existing tensor ``t``, you can call ``t.view(...)``.

::

    >>> t = torch.rand(4, 4)
    >>> b = t.view(2, 8)
    >>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` share the same underlying data.
    True
    # Modifying view tensor changes base tensor as well.
    >>> b[0][0] = 3.14
    >>> t[0][0]
    tensor(3.14)

Since views share underlying data with its base tensor, if you edit the data
in the view, it will be reflected in the base tensor as well.

Typically a PyTorch op returns a new tensor as output, e.g. :meth:`~torch.Tensor.add`.
But in case of view ops, outputs are views of input tensors to avoid unnecessary data copy.
No data movement occurs when creating a view, view tensor just changes the way
it interprets the same data. Taking a view of contiguous tensor could potentially produce a non-contiguous tensor.
Users should pay additional attention as contiguity might have implicit performance impact.
:meth:`~torch.Tensor.transpose` is a common example.

::

    >>> base = torch.tensor([[0, 1],[2, 3]])
    >>> base.is_contiguous()
    True
    >>> t = base.transpose(0, 1)  # `t` is a view of `base`. No data movement happened here.
    # View tensors might be non-contiguous.
    >>> t.is_contiguous()
    False
    # To get a contiguous tensor, call `.contiguous()` to enforce
    # copying data when `t` is not contiguous.
    >>> c = t.contiguous()

For reference, here’s a full list of view ops in PyTorch:

- Basic slicing and indexing op, e.g. ``tensor[0, 2:, 1:7:2]`` returns a view of base ``tensor``, see note below.
- :meth:`~torch.Tensor.adjoint`
- :meth:`~torch.Tensor.as_strided`
- :meth:`~torch.Tensor.detach`
- :meth:`~torch.Tensor.diagonal`
- :meth:`~torch.Tensor.expand`
- :meth:`~torch.Tensor.expand_as`
- :meth:`~torch.Tensor.movedim`
- :meth:`~torch.Tensor.narrow`
- :meth:`~torch.Tensor.permute`
- :meth:`~torch.Tensor.select`
- :meth:`~torch.Tensor.squeeze`
- :meth:`~torch.Tensor.transpose`
- :meth:`~torch.Tensor.t`
- :attr:`~torch.Tensor.T`
- :attr:`~torch.Tensor.H`
- :attr:`~torch.Tensor.mT`
- :attr:`~torch.Tensor.mH`
- :attr:`~torch.Tensor.real`
- :attr:`~torch.Tensor.imag`
- :meth:`~torch.Tensor.view_as_real`
- :meth:`~torch.Tensor.unflatten`
- :meth:`~torch.Tensor.unfold`
- :meth:`~torch.Tensor.unsqueeze`
- :meth:`~torch.Tensor.view`
- :meth:`~torch.Tensor.view_as`
- :meth:`~torch.Tensor.unbind`
- :meth:`~torch.Tensor.split`
- :meth:`~torch.Tensor.hsplit`
- :meth:`~torch.Tensor.vsplit`
- :meth:`~torch.Tensor.tensor_split`
- :meth:`~torch.Tensor.split_with_sizes`
- :meth:`~torch.Tensor.swapaxes`
- :meth:`~torch.Tensor.swapdims`
- :meth:`~torch.Tensor.chunk`
- :meth:`~torch.Tensor.indices` (sparse tensor only)
- :meth:`~torch.Tensor.values`  (sparse tensor only)

.. note::
   When accessing the contents of a tensor via indexing, PyTorch follows Numpy behaviors
   that basic indexing returns views, while advanced indexing returns a copy.
   Assignment via either basic or advanced indexing is in-place. See more examples in
   `Numpy indexing documentation <https://numpy.org/doc/stable/user/basics.indexing.html>`_.

It's also worth mentioning a few ops with special behaviors:

- :meth:`~torch.Tensor.reshape`, :meth:`~torch.Tensor.reshape_as` and :meth:`~torch.Tensor.flatten` can return either a view or new tensor, user code shouldn't rely on whether it's view or not.
- :meth:`~torch.Tensor.contiguous` returns **itself** if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying data.

For a more detailed walk-through of PyTorch internal implementation,
please refer to `ezyang's blogpost about PyTorch Internals <http://blog.ezyang.com/2019/05/pytorch-internals/>`_.

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`docs/source`):

- [`torch.compiler_troubleshooting.md_docs.md`](./torch.compiler_troubleshooting.md_docs.md)
- [`torch.compiler_aot_inductor_debugging_guide.md_docs.md`](./torch.compiler_aot_inductor_debugging_guide.md_docs.md)
- [`mtia.memory.md_docs.md`](./mtia.memory.md_docs.md)
- [`torch.compiler_get_started.md_docs.md`](./torch.compiler_get_started.md_docs.md)
- [`torch.compiler_dynamo_deepdive.md_docs.md`](./torch.compiler_dynamo_deepdive.md_docs.md)
- [`mtia.mtia_graph.md_docs.md`](./mtia.mtia_graph.md_docs.md)
- [`hub.md_docs.md`](./hub.md_docs.md)
- [`torch_nccl_environment_variables.md_docs.md`](./torch_nccl_environment_variables.md_docs.md)
- [`optim.md_docs.md`](./optim.md_docs.md)
- [`torch.compiler_aot_inductor.md_docs.md`](./torch.compiler_aot_inductor.md_docs.md)


## Cross-References

- **File Documentation**: `tensor_view.rst_docs.md`
- **Keyword Index**: `tensor_view.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
