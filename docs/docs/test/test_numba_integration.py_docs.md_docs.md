# Documentation: `docs/test/test_numba_integration.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_numba_integration.py_docs.md`
- **Size**: 19,477 bytes (19.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_numba_integration.py`

## File Metadata

- **Path**: `test/test_numba_integration.py`
- **Size**: 15,769 bytes (15.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: cuda"]

import unittest

import torch
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_cuda import (
    TEST_CUDA,
    TEST_MULTIGPU,
    TEST_NUMBA_CUDA,
)
from torch.testing._internal.common_utils import TEST_NUMPY


if TEST_NUMPY:
    import numpy

if TEST_NUMBA_CUDA:
    import numba.cuda


class TestNumbaIntegration(common.TestCase):
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_cuda_array_interface(self):
        """torch.Tensor exposes __cuda_array_interface__ for cuda tensors.

        An object t is considered a cuda-tensor if:
            hasattr(t, '__cuda_array_interface__')

        A cuda-tensor provides a tensor description dict:
            shape: (integer, ...) Tensor shape.
            strides: (integer, ...) Tensor strides, in bytes.
            typestr: (str) A numpy-style typestr.
            data: (int, boolean) A (data_ptr, read-only) tuple.
            version: (int) Version 0

        See:
        https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        """

        types = [
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.HalfTensor,
            torch.LongTensor,
            torch.IntTensor,
            torch.ShortTensor,
            torch.CharTensor,
            torch.ByteTensor,
        ]
        dtypes = [
            numpy.float64,
            numpy.float32,
            numpy.float16,
            numpy.int64,
            numpy.int32,
            numpy.int16,
            numpy.int8,
            numpy.uint8,
        ]
        for tp, npt in zip(types, dtypes):
            # CPU tensors do not implement the interface.
            cput = tp(10)

            self.assertFalse(hasattr(cput, "__cuda_array_interface__"))
            self.assertRaises(AttributeError, lambda: cput.__cuda_array_interface__)

            # Sparse CPU/CUDA tensors do not implement the interface
            if tp not in (torch.HalfTensor,):
                indices_t = torch.empty(1, cput.size(0), dtype=torch.long).clamp_(min=0)
                sparse_t = torch.sparse_coo_tensor(indices_t, cput)

                self.assertFalse(hasattr(sparse_t, "__cuda_array_interface__"))
                self.assertRaises(
                    AttributeError, lambda: sparse_t.__cuda_array_interface__
                )

                sparse_cuda_t = torch.sparse_coo_tensor(indices_t, cput).cuda()

                self.assertFalse(hasattr(sparse_cuda_t, "__cuda_array_interface__"))
                self.assertRaises(
                    AttributeError, lambda: sparse_cuda_t.__cuda_array_interface__
                )

            # CUDA tensors have the attribute and v2 interface
            cudat = tp(10).cuda()

            self.assertTrue(hasattr(cudat, "__cuda_array_interface__"))

            ar_dict = cudat.__cuda_array_interface__

            self.assertEqual(
                set(ar_dict.keys()), {"shape", "strides", "typestr", "data", "version"}
            )

            self.assertEqual(ar_dict["shape"], (10,))
            self.assertIs(ar_dict["strides"], None)
            # typestr from numpy, cuda-native little-endian
            self.assertEqual(ar_dict["typestr"], numpy.dtype(npt).newbyteorder("<").str)
            self.assertEqual(ar_dict["data"], (cudat.data_ptr(), False))
            self.assertEqual(ar_dict["version"], 2)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_array_adaptor(self):
        """Torch __cuda_array_adaptor__ exposes tensor data to numba.cuda."""

        torch_dtypes = [
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.uint8,
            torch.int8,
            torch.uint16,
            torch.int16,
            torch.uint32,
            torch.int32,
            torch.uint64,
            torch.int64,
            torch.bool,
        ]

        for dt in torch_dtypes:
            # CPU tensors of all types do not register as cuda arrays,
            # attempts to convert raise a type error.
            cput = torch.arange(10).to(dt)
            npt = cput.numpy()

            self.assertTrue(not numba.cuda.is_cuda_array(cput))
            with self.assertRaises(TypeError):
                numba.cuda.as_cuda_array(cput)

            # Any cuda tensor is a cuda array.
            cudat = cput.to(device="cuda")
            self.assertTrue(numba.cuda.is_cuda_array(cudat))

            numba_view = numba.cuda.as_cuda_array(cudat)
            self.assertIsInstance(numba_view, numba.cuda.devicearray.DeviceNDArray)

            # The reported type of the cuda array matches the numpy type of the cpu tensor.
            self.assertEqual(numba_view.dtype, npt.dtype)
            self.assertEqual(numba_view.strides, npt.strides)
            self.assertEqual(numba_view.shape, cudat.shape)

            # Pass back to cuda from host for all equality checks below, needed for
            # float16 comparisons, which aren't supported cpu-side.

            # The data is identical in the view.
            self.assertEqual(cudat, torch.tensor(numba_view.copy_to_host()).to("cuda"))

            # Writes to the torch.Tensor are reflected in the numba array.
            cudat[:5] = 11
            self.assertEqual(cudat, torch.tensor(numba_view.copy_to_host()).to("cuda"))

            # Strided tensors are supported.
            strided_cudat = cudat[::2]
            strided_npt = cput[::2].numpy()
            strided_numba_view = numba.cuda.as_cuda_array(strided_cudat)

            self.assertEqual(strided_numba_view.dtype, strided_npt.dtype)
            self.assertEqual(strided_numba_view.strides, strided_npt.strides)
            self.assertEqual(strided_numba_view.shape, strided_cudat.shape)

            # As of numba 0.40.0 support for strided views is ...limited...
            # Cannot verify correctness of strided view operations.

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_conversion_errors(self):
        """Numba properly detects array interface for tensor.Tensor variants."""

        # CPU tensors are not cuda arrays.
        cput = torch.arange(100)

        self.assertFalse(numba.cuda.is_cuda_array(cput))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cput)

        # Sparse tensors are not cuda arrays, regardless of device.
        sparset = torch.sparse_coo_tensor(cput[None, :], cput)

        self.assertFalse(numba.cuda.is_cuda_array(sparset))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(sparset)

        sparset.cuda()

        self.assertFalse(numba.cuda.is_cuda_array(sparset))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(sparset)

        # Device-status overrides gradient status.
        # CPU+gradient isn't a cuda array.
        cpu_gradt = torch.zeros(100).requires_grad_(True)

        self.assertFalse(numba.cuda.is_cuda_array(cpu_gradt))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cpu_gradt)

        # CUDA+gradient raises a RuntimeError on check or conversion.
        #
        # Use of hasattr for interface detection causes interface change in
        # python2; it swallows all exceptions not just AttributeError.
        cuda_gradt = torch.zeros(100).requires_grad_(True).cuda()

        # conversion raises RuntimeError
        with self.assertRaises(RuntimeError):
            numba.cuda.is_cuda_array(cuda_gradt)
        with self.assertRaises(RuntimeError):
            numba.cuda.as_cuda_array(cuda_gradt)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    @unittest.skipIf(not TEST_MULTIGPU, "No multigpu")
    def test_active_device(self):
        """'as_cuda_array' tensor device must match active numba context."""

        # Both torch/numba default to device 0 and can interop freely
        cudat = torch.arange(10, device="cuda")
        self.assertEqual(cudat.device.index, 0)
        self.assertIsInstance(
            numba.cuda.as_cuda_array(cudat), numba.cuda.devicearray.DeviceNDArray
        )

        # Tensors on non-default device raise api error if converted
        cudat = torch.arange(10, device=torch.device("cuda", 1))

        with self.assertRaises(numba.cuda.driver.CudaAPIError):
            numba.cuda.as_cuda_array(cudat)

        # but can be converted when switching to the device's context
        with numba.cuda.devices.gpus[cudat.device.index]:
            self.assertIsInstance(
                numba.cuda.as_cuda_array(cudat), numba.cuda.devicearray.DeviceNDArray
            )

    @unittest.skip(
        "Test is temporary disabled, see https://github.com/pytorch/pytorch/issues/54418"
    )
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_from_cuda_array_interface(self):
        """torch.as_tensor() and torch.tensor() supports the __cuda_array_interface__ protocol.

        If an object exposes the __cuda_array_interface__, .as_tensor() and .tensor()
        will use the exposed device memory.

        See:
        https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        """

        dtypes = [
            numpy.complex64,
            numpy.complex128,
            numpy.float64,
            numpy.float32,
            numpy.int64,
            numpy.int32,
            numpy.int16,
            numpy.int8,
            numpy.uint8,
        ]
        for dtype in dtypes:
            numpy_arys = [
                numpy.ones((), dtype=dtype),
                numpy.arange(6).reshape(2, 3).astype(dtype),
                numpy.arange(6)
                .reshape(2, 3)
                .astype(dtype)[1:],  # View offset should be ignored
                numpy.arange(6)
                .reshape(2, 3)
                .astype(dtype)[:, None],  # change the strides but still contiguous
            ]
            # Zero-copy when using `torch.as_tensor()`
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                torch_ary = torch.as_tensor(numba_ary, device="cuda")
                self.assertEqual(
                    numba_ary.__cuda_array_interface__,
                    torch_ary.__cuda_array_interface__,
                )
                self.assertEqual(
                    torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary, dtype=dtype)
                )

                # Check that `torch_ary` and `numba_ary` points to the same device memory
                torch_ary += 42
                self.assertEqual(
                    torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary, dtype=dtype)
                )

            # Implicit-copy because `torch_ary` is a CPU array
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                torch_ary = torch.as_tensor(numba_ary, device="cpu")
                self.assertEqual(
                    torch_ary.data.numpy(), numpy.asarray(numba_ary, dtype=dtype)
                )

                # Check that `torch_ary` and `numba_ary` points to different memory
                torch_ary += 42
                self.assertEqual(
                    torch_ary.data.numpy(), numpy.asarray(numba_ary, dtype=dtype) + 42
                )

            # Explicit-copy when using `torch.tensor()`
            for numpy_ary in numpy_arys:
                numba_ary = numba.cuda.to_device(numpy_ary)
                torch_ary = torch.tensor(numba_ary, device="cuda")
                self.assertEqual(
                    torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary, dtype=dtype)
                )

                # Check that `torch_ary` and `numba_ary` points to different memory
                torch_ary += 42
                self.assertEqual(
                    torch_ary.cpu().data.numpy(),
                    numpy.asarray(numba_ary, dtype=dtype) + 42,
                )

    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_from_cuda_array_interface_inferred_strides(self):
        """torch.as_tensor(numba_ary) should have correct inferred (contiguous) strides"""
        # This could, in theory, be combined with test_from_cuda_array_interface but that test
        # is overly strict: it checks that the exported protocols are exactly the same, which
        # cannot handle differing exported protocol versions.
        dtypes = [
            numpy.float64,
            numpy.float32,
            numpy.int64,
            numpy.int32,
            numpy.int16,
            numpy.int8,
            numpy.uint8,
        ]
        for dtype in dtypes:
            numpy_ary = numpy.arange(6).reshape(2, 3).astype(dtype)
            numba_ary = numba.cuda.to_device(numpy_ary)
            self.assertTrue(numba_ary.is_c_contiguous())
            torch_ary = torch.as_tensor(numba_ary, device="cuda")
            self.assertTrue(torch_ary.is_contiguous())

    @unittest.skip(
        "Test is temporary disabled, see https://github.com/pytorch/pytorch/issues/54418"
    )
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_from_cuda_array_interface_lifetime(self):
        """torch.as_tensor(obj) tensor grabs a reference to obj so that the lifetime of obj exceeds the tensor"""
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device="cuda")
        self.assertEqual(
            torch_ary.__cuda_array_interface__, numba_ary.__cuda_array_interface__
        )  # No copy
        del numba_ary
        self.assertEqual(
            torch_ary.cpu().data.numpy(), numpy.arange(6)
        )  # `torch_ary` is still alive

    @unittest.skip(
        "Test is temporary disabled, see https://github.com/pytorch/pytorch/issues/54418"
    )
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    @unittest.skipIf(not TEST_MULTIGPU, "No multigpu")
    def test_from_cuda_array_interface_active_device(self):
        """torch.as_tensor() tensor device must match active numba context."""

        # Zero-copy: both torch/numba default to device 0 and can interop freely
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device="cuda")
        self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary))
        self.assertEqual(
            torch_ary.__cuda_array_interface__, numba_ary.__cuda_array_interface__
        )

        # Implicit-copy: when the Numba and Torch device differ
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device=torch.device("cuda", 1))
        self.assertEqual(torch_ary.get_device(), 1)
        self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary))
        if1 = torch_ary.__cuda_array_interface__
        if2 = numba_ary.__cuda_array_interface__
        self.assertNotEqual(if1["data"], if2["data"])
        del if1["data"]
        del if2["data"]
        self.assertEqual(if1, if2)


if __name__ == "__main__":
    common.run_tests()

```



## High-Level Overview

"""torch.Tensor exposes __cuda_array_interface__ for cuda tensors.        An object t is considered a cuda-tensor if:            hasattr(t, '__cuda_array_interface__')        A cuda-tensor provides a tensor description dict:            shape: (integer, ...) Tensor shape.            strides: (integer, ...) Tensor strides, in bytes.            typestr: (str) A numpy-style typestr.            data: (int, boolean) A (data_ptr, read-only) tuple.            version: (int) Version 0        See:        https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNumbaIntegration`

**Functions defined**: `test_cuda_array_interface`, `test_array_adaptor`, `test_conversion_errors`, `test_active_device`, `test_from_cuda_array_interface`, `test_from_cuda_array_interface_inferred_strides`, `test_from_cuda_array_interface_lifetime`, `test_from_cuda_array_interface_active_device`

**Key imports**: unittest, torch, torch.testing._internal.common_utils as common, TEST_NUMPY, numpy, numba.cuda


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch.testing._internal.common_utils as common`
- `torch.testing._internal.common_utils`: TEST_NUMPY
- `numpy`
- `numba.cuda`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python test/test_numba_integration.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_numba_integration.py_docs.md`
- **Keyword Index**: `test_numba_integration.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_numba_integration.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_numba_integration.py_docs.md_docs.md`
- **Keyword Index**: `test_numba_integration.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
