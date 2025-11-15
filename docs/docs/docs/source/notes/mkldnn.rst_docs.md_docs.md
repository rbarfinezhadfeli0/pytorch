# Documentation: `docs/docs/source/notes/mkldnn.rst_docs.md`

## File Metadata

- **Path**: `docs/docs/source/notes/mkldnn.rst_docs.md`
- **Size**: 6,351 bytes (6.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/notes/mkldnn.rst`

## File Metadata

- **Path**: `docs/source/notes/mkldnn.rst`
- **Size**: 4,007 bytes (3.91 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
.. meta::
   :description: A guide to torch.backends.mkldnn, a PyTorch backend to run MKLDNN operations
   :keywords: optimize PyTorch, MKLDNN

.. _mkldnn_backend:

MKLDNN backend
---------------------------------------------------

MKLDNN is an open-source cross-platform performance library of basic building blocks
for deep learning applications.

.. code:: python

  # The flag below controls whether enable MKLDNN backend in Pytorch.
  torch.backends.mkldnn.enabled = True

Users can disable MKLDNN backend by:

.. code:: python

  torch.backends.mkldnn.enabled = False

.. _bf16_on_mkldnn:

Bfloat16 (BF16) on MKLDNN backend
---------------------------------------------------

Starting in PyTorch 2.9, there is a set of APIs to control the internal computation precision
for `float32` operators.

.. code:: python

  # The flag below controls the internal computation precision for mkldnn matmul. Default ieee is float32.
  torch.backends.mkldnn.matmul.fp32_precision = "ieee"

  # The flag below controls the internal computation precision for mkldnn conv. Default ieee is float32.
  torch.backends.mkldnn.conv.fp32_precision = "ieee"

  # The flag below controls the internal computation precision for mkldnn rnn. Default ieee is float32.
  torch.backends.mkldnn.rnn.fp32_precision = "ieee"

Note that besides matmuls and convolutions themselves, functions and nn modules that internally uses
matmuls or convolutions are also affected. These include :class:`torch.nn.Linear`, :class:`torch.nn._ConvNd`, :func:`torch.cdist`,
:func:`torch.tensordot`, :func:`torch.nn.functional.affine_grid` and :func:`torch.nn.functional.grid_sample`,
:class:`torch.nn.AdaptiveLogSoftmaxWithLoss`, :class:`torch.nn.GRU` and  :class:`torch.nn.LSTM`.

To get an idea of the precision and speed, see the example code and benchmark data (on SPR) below:

.. code:: python

  torch.manual_seed(0)
  a_full = torch.randn(10240, 10240, dtype=torch.double)
  b_full = torch.randn(10240, 10240, dtype=torch.double)
  ab_full = a_full @ b_full
  mean = ab_full.abs().mean()  # 80.7451

  a = a_full.float()
  b = b_full.float()

  # Do matmul at BF16 mode.
  torch.backends.mkldnn.matmul.fp32_precision = 'bf16'
  ab_bf16 = a @ b  # expected speedup with BF16 dot-product acceleration
  error = (ab_bf16 - ab_full).abs().max()  # 1.3704
  relative_error = error / mean  # 0.0170
  print(error, relative_error)

  # Do matmul at TF32 mode.
  torch.backends.mkldnn.matmul.fp32_precision = 'tf32'
  ab_tf32 = a @ b  # expected speedup with TF32 dot-product acceleration
  error = (ab_tf32 - ab_full).abs().max()  # 0.0004
  relative_error = error / mean  # 0.00000552
  print(error, relative_error)

  # Do matmul FP32 mode.
  torch.backends.mkldnn.matmul.fp32_precision = 'ieee'
  ab_fp32 = a @ b
  error = (ab_fp32 - ab_full).abs().max()  # 0.0003
  relative_error = error / mean  # 0.00000317
  print(error, relative_error)

From the above example, we can see that with BF16, the speed is ~7x faster on SPR, and that
relative error compared to double precision is approximately 2 orders of magnitude larger.
If full FP32 precision is needed, users can disable BF16 by:

.. code:: python

  torch.backends.mkldnn.matmul.fp32_precision = 'ieee'
  torch.backends.mkldnn.conv.fp32_precision = 'ieee'
  torch.backends.mkldnn.rnn.fp32_precision = 'ieee'

To toggle the BF16 flags off in C++, you can do

.. code:: C++

  at::globalContext().setFloat32Precision("ieee", "mkldnn", "matmul");
  at::globalContext().setFloat32Precision("ieee", "mkldnn", "conv");
  at::globalContext().setFloat32Precision("ieee", "mkldnn", "rnn");

We can override a generic setting for a specific operator or backend if the fp32_precision is set to `ieee`.

.. code:: python

  torch.backends.fp32_precision = "bf16"
  torch.backends.mkldnn.fp32_precision = "ieee"
  torch.backends.mkldnn.matmul.fp32_precision = "ieee"

For such case, both `torch.backends.mkldnn.fp32_precision` and `torch.backends.mkldnn.matmul.fp32_precision`
is overridden to bf16.

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

Files in the same folder (`docs/source/notes`):

- [`windows.rst_docs.md`](./windows.rst_docs.md)
- [`get_start_xpu.rst_docs.md`](./get_start_xpu.rst_docs.md)
- [`amp_examples.rst_docs.md`](./amp_examples.rst_docs.md)
- [`broadcasting.rst_docs.md`](./broadcasting.rst_docs.md)
- [`autograd.rst_docs.md`](./autograd.rst_docs.md)
- [`cpu_threading_torchscript_inference.rst_docs.md`](./cpu_threading_torchscript_inference.rst_docs.md)
- [`hip.rst_docs.md`](./hip.rst_docs.md)
- [`libtorch_stable_abi.md_docs.md`](./libtorch_stable_abi.md_docs.md)
- [`cuda.rst_docs.md`](./cuda.rst_docs.md)
- [`out.rst_docs.md`](./out.rst_docs.md)


## Cross-References

- **File Documentation**: `mkldnn.rst_docs.md`
- **Keyword Index**: `mkldnn.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source/notes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source/notes`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`docs/docs/source/notes`):

- [`amp_examples.rst_kw.md_docs.md`](./amp_examples.rst_kw.md_docs.md)
- [`libtorch_stable_abi.md_kw.md_docs.md`](./libtorch_stable_abi.md_kw.md_docs.md)
- [`autograd.rst_docs.md_docs.md`](./autograd.rst_docs.md_docs.md)
- [`multiprocessing.rst_kw.md_docs.md`](./multiprocessing.rst_kw.md_docs.md)
- [`numerical_accuracy.rst_kw.md_docs.md`](./numerical_accuracy.rst_kw.md_docs.md)
- [`cuda.rst_kw.md_docs.md`](./cuda.rst_kw.md_docs.md)
- [`windows.rst_kw.md_docs.md`](./windows.rst_kw.md_docs.md)
- [`extending.func.rst_docs.md_docs.md`](./extending.func.rst_docs.md_docs.md)
- [`extending.rst_docs.md_docs.md`](./extending.rst_docs.md_docs.md)
- [`modules.rst_kw.md_docs.md`](./modules.rst_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mkldnn.rst_docs.md_docs.md`
- **Keyword Index**: `mkldnn.rst_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
