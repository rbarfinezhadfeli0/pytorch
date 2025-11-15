# Documentation: `docs/docs/cpp/source/notes/tensor_indexing.rst_docs.md`

## File Metadata

- **Path**: `docs/docs/cpp/source/notes/tensor_indexing.rst_docs.md`
- **Size**: 11,822 bytes (11.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/cpp/source/notes/tensor_indexing.rst`

## File Metadata

- **Path**: `docs/cpp/source/notes/tensor_indexing.rst`
- **Size**: 9,669 bytes (9.44 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
Tensor Indexing API
===================

Indexing a tensor in the PyTorch C++ API works very similar to the Python API.
All index types such as ``None`` / ``...`` / integer / boolean / slice / tensor
are available in the C++ API, making translation from Python indexing code to C++
very simple. The main difference is that, instead of using the ``[]``-operator
similar to the Python API syntax, in the C++ API the indexing methods are:

- ``torch::Tensor::index`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor5indexE8ArrayRefIN2at8indexing11TensorIndexEE>`_)
- ``torch::Tensor::index_put_`` (`link <https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4N2at6Tensor10index_put_E8ArrayRefIN2at8indexing11TensorIndexEERK6Tensor>`_)

It's also important to note that index types such as ``None`` / ``Ellipsis`` / ``Slice``
live in the ``torch::indexing`` namespace, and it's recommended to put ``using namespace torch::indexing``
before any indexing code for convenient use of those index types.

Here are some examples of translating Python indexing code to C++:

Getter
------

+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| Python                                                   | C++  (assuming ``using namespace torch::indexing``)                                  |
+==========================================================+======================================================================================+
| ``tensor[None]``                                         | ``tensor.index({None})``                                                             |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[Ellipsis, ...]``                                | ``tensor.index({Ellipsis, "..."})``                                                  |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[1, 2]``                                         | ``tensor.index({1, 2})``                                                             |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[True, False]``                                  | ``tensor.index({true, false})``                                                      |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[1::2]``                                         | ``tensor.index({Slice(1, None, 2)})``                                                |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[torch.tensor([1, 2])]``                         | ``tensor.index({torch::tensor({1, 2})})``                                            |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[..., 0, True, 1::2, torch.tensor([1, 2])]``     | ``tensor.index({"...", 0, true, Slice(1, None, 2), torch::tensor({1, 2})})``         |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+

Setter
------

+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| Python                                                   | C++  (assuming ``using namespace torch::indexing``)                                  |
+==========================================================+======================================================================================+
| ``tensor[None] = 1``                                     | ``tensor.index_put_({None}, 1)``                                                     |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[Ellipsis, ...] = 1``                            | ``tensor.index_put_({Ellipsis, "..."}, 1)``                                          |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[1, 2] = 1``                                     | ``tensor.index_put_({1, 2}, 1)``                                                     |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[True, False] = 1``                              | ``tensor.index_put_({true, false}, 1)``                                              |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[1::2] = 1``                                     | ``tensor.index_put_({Slice(1, None, 2)}, 1)``                                        |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[torch.tensor([1, 2])] = 1``                     | ``tensor.index_put_({torch::tensor({1, 2})}, 1)``                                    |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+
| ``tensor[..., 0, True, 1::2, torch.tensor([1, 2])] = 1`` | ``tensor.index_put_({"...", 0, true, Slice(1, None, 2), torch::tensor({1, 2})}, 1)`` |
+----------------------------------------------------------+--------------------------------------------------------------------------------------+


Translating between Python/C++ index types
------------------------------------------

The one-to-one translation between Python and C++ index types is as follows:

+-------------------------+------------------------------------------------------------------------+
| Python                  | C++ (assuming ``using namespace torch::indexing``)                     |
+=========================+========================================================================+
| ``None``                | ``None``                                                               |
+-------------------------+------------------------------------------------------------------------+
| ``Ellipsis``            | ``Ellipsis``                                                           |
+-------------------------+------------------------------------------------------------------------+
| ``...``                 | ``"..."``                                                              |
+-------------------------+------------------------------------------------------------------------+
| ``123``                 | ``123``                                                                |
+-------------------------+------------------------------------------------------------------------+
| ``True``                | ``true``                                                               |
+-------------------------+------------------------------------------------------------------------+
| ``False``               | ``false``                                                              |
+-------------------------+------------------------------------------------------------------------+
| ``:`` or ``::``         | ``Slice()`` or ``Slice(None, None)`` or ``Slice(None, None, None)``    |
+-------------------------+------------------------------------------------------------------------+
| ``1:`` or ``1::``       | ``Slice(1, None)`` or ``Slice(1, None, None)``                         |
+-------------------------+------------------------------------------------------------------------+
| ``:3`` or ``:3:``       | ``Slice(None, 3)`` or ``Slice(None, 3, None)``                         |
+-------------------------+------------------------------------------------------------------------+
| ``::2``                 | ``Slice(None, None, 2)``                                               |
+-------------------------+------------------------------------------------------------------------+
| ``1:3``                 | ``Slice(1, 3)``                                                        |
+-------------------------+------------------------------------------------------------------------+
| ``1::2``                | ``Slice(1, None, 2)``                                                  |
+-------------------------+------------------------------------------------------------------------+
| ``:3:2``                | ``Slice(None, 3, 2)``                                                  |
+-------------------------+------------------------------------------------------------------------+
| ``1:3:2``               | ``Slice(1, 3, 2)``                                                     |
+-------------------------+------------------------------------------------------------------------+
| ``torch.tensor([1, 2])``| ``torch::tensor({1, 2})``                                              |
+-------------------------+------------------------------------------------------------------------+

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/cpp/source/notes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/cpp/source/notes`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`docs/cpp/source/notes`):

- [`maybe_owned.rst_docs.md`](./maybe_owned.rst_docs.md)
- [`tensor_cuda_stream.rst_docs.md`](./tensor_cuda_stream.rst_docs.md)
- [`tensor_creation.rst_docs.md`](./tensor_creation.rst_docs.md)
- [`tensor_basics.rst_docs.md`](./tensor_basics.rst_docs.md)
- [`faq.rst_docs.md`](./faq.rst_docs.md)
- [`versioning.rst_docs.md`](./versioning.rst_docs.md)
- [`inference_mode.rst_docs.md`](./inference_mode.rst_docs.md)


## Cross-References

- **File Documentation**: `tensor_indexing.rst_docs.md`
- **Keyword Index**: `tensor_indexing.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/cpp/source/notes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/cpp/source/notes`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/docs/cpp/source/notes`):

- [`inference_mode.rst_kw.md_docs.md`](./inference_mode.rst_kw.md_docs.md)
- [`versioning.rst_docs.md_docs.md`](./versioning.rst_docs.md_docs.md)
- [`tensor_basics.rst_kw.md_docs.md`](./tensor_basics.rst_kw.md_docs.md)
- [`tensor_cuda_stream.rst_kw.md_docs.md`](./tensor_cuda_stream.rst_kw.md_docs.md)
- [`faq.rst_docs.md_docs.md`](./faq.rst_docs.md_docs.md)
- [`maybe_owned.rst_kw.md_docs.md`](./maybe_owned.rst_kw.md_docs.md)
- [`inference_mode.rst_docs.md_docs.md`](./inference_mode.rst_docs.md_docs.md)
- [`tensor_creation.rst_docs.md_docs.md`](./tensor_creation.rst_docs.md_docs.md)
- [`faq.rst_kw.md_docs.md`](./faq.rst_kw.md_docs.md)
- [`tensor_basics.rst_docs.md_docs.md`](./tensor_basics.rst_docs.md_docs.md)


## Cross-References

- **File Documentation**: `tensor_indexing.rst_docs.md_docs.md`
- **Keyword Index**: `tensor_indexing.rst_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
