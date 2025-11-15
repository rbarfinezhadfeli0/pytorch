# Documentation: `docs/test/torch_np/numpy_tests/fft/test_helper.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/numpy_tests/fft/test_helper.py_docs.md`
- **Size**: 9,197 bytes (8.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/numpy_tests/fft/test_helper.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/fft/test_helper.py`
- **Size**: 6,527 bytes (6.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

"""Test functions for fftpack.helper module

Copied from fftpack.helper by Pearu Peterson, October 2005

"""

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import fft, pi
    from numpy.testing import assert_array_almost_equal
else:
    import torch._numpy as np
    from torch._numpy import fft, pi
    from torch._numpy.testing import assert_array_almost_equal


class TestFFTShift(TestCase):
    def test_definition(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)
        assert_array_almost_equal(fft.ifftshift(y), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)
        assert_array_almost_equal(fft.ifftshift(y), x)

    def test_inverse(self):
        for n in [1, 4, 9, 100, 211]:
            x = np.random.random((n,))
            assert_array_almost_equal(fft.ifftshift(fft.fftshift(x)), x)

    def test_axes_keyword(self):
        freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
        shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shifted)
        assert_array_almost_equal(
            fft.fftshift(freqs, axes=0), fft.fftshift(freqs, axes=(0,))
        )
        assert_array_almost_equal(fft.ifftshift(shifted, axes=(0, 1)), freqs)
        assert_array_almost_equal(
            fft.ifftshift(shifted, axes=0), fft.ifftshift(shifted, axes=(0,))
        )

        assert_array_almost_equal(fft.fftshift(freqs), shifted)
        assert_array_almost_equal(fft.ifftshift(shifted), freqs)

    def test_uneven_dims(self):
        """Test 2D input, which has uneven dimension sizes"""
        freqs = [[0, 1], [2, 3], [4, 5]]

        # shift in dimension 0
        shift_dim0 = [[4, 5], [0, 1], [2, 3]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=0), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=0), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0,)), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=[0]), freqs)

        # shift in dimension 1
        shift_dim1 = [[1, 0], [3, 2], [5, 4]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=1), shift_dim1)
        assert_array_almost_equal(fft.ifftshift(shift_dim1, axes=1), freqs)

        # shift in both dimensions
        shift_dim_both = [[5, 4], [1, 0], [3, 2]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)

        # axes=None (default) shift in all dimensions
        assert_array_almost_equal(fft.fftshift(freqs, axes=None), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=None), freqs)
        assert_array_almost_equal(fft.fftshift(freqs), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both), freqs)

    def test_equal_to_original(self):
        """Test that the new (>=v1.15) implementation (see #10073) is equal to the original (<=v1.14)"""
        if TEST_WITH_TORCHDYNAMO:
            from numpy import arange, asarray, concatenate, take
        else:
            from torch._numpy import arange, asarray, concatenate, take

        def original_fftshift(x, axes=None):
            """How fftshift was implemented in v1.14"""
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        def original_ifftshift(x, axes=None):
            """How ifftshift was implemented in v1.14"""
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = n - (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        # create possible 2d array combinations and try all possible keywords
        # compare output to original functions
        for i in range(16):
            for j in range(16):
                for axes_keyword in [0, 1, None, (0,), (0, 1)]:
                    inp = np.random.rand(i, j)

                    assert_array_almost_equal(
                        fft.fftshift(inp, axes_keyword),
                        original_fftshift(inp, axes_keyword),
                    )

                    assert_array_almost_equal(
                        fft.ifftshift(inp, axes_keyword),
                        original_ifftshift(inp, axes_keyword),
                    )


class TestFFTFreq(TestCase):
    def test_definition(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        assert_array_almost_equal(9 * fft.fftfreq(9), x)
        assert_array_almost_equal(9 * pi * fft.fftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        assert_array_almost_equal(10 * fft.fftfreq(10), x)
        assert_array_almost_equal(10 * pi * fft.fftfreq(10, pi), x)


class TestRFFTFreq(TestCase):
    def test_definition(self):
        x = [0, 1, 2, 3, 4]
        assert_array_almost_equal(9 * fft.rfftfreq(9), x)
        assert_array_almost_equal(9 * pi * fft.rfftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, 5]
        assert_array_almost_equal(10 * fft.rfftfreq(10), x)
        assert_array_almost_equal(10 * pi * fft.rfftfreq(10, pi), x)


class TestIRFFTN(TestCase):
    def test_not_last_axis_success(self):
        ar, ai = np.random.random((2, 16, 8, 32))
        a = ar + 1j * ai

        axes = (-2,)

        # Should not raise error
        fft.irfftn(a, axes=axes)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test functions for fftpack.helper moduleCopied from fftpack.helper by Pearu Peterson, October 2005

This Python file contains 4 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFFTShift`, `TestFFTFreq`, `TestRFFTFreq`, `TestIRFFTN`

**Functions defined**: `test_definition`, `test_inverse`, `test_axes_keyword`, `test_uneven_dims`, `test_equal_to_original`, `original_fftshift`, `original_ifftshift`, `test_definition`, `test_definition`, `test_not_last_axis_success`

**Key imports**: numpy as np, fft, pi, assert_array_almost_equal, torch._numpy as np, fft, pi, assert_array_almost_equal, arange, asarray, concatenate, take, arange, asarray, concatenate, take


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np/numpy_tests/fft`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `numpy`: fft, pi
- `numpy.testing`: assert_array_almost_equal
- `torch._numpy as np`
- `torch._numpy`: fft, pi
- `torch._numpy.testing`: assert_array_almost_equal


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

This is a test file. Run it with:

```bash
python test/torch_np/numpy_tests/fft/test_helper.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np/numpy_tests/fft`):

- [`test_pocketfft.py_docs.md`](./test_pocketfft.py_docs.md)


## Cross-References

- **File Documentation**: `test_helper.py_docs.md`
- **Keyword Index**: `test_helper.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/torch_np/numpy_tests/fft`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/torch_np/numpy_tests/fft`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python docs/test/torch_np/numpy_tests/fft/test_helper.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np/numpy_tests/fft`):

- [`test_pocketfft.py_docs.md_docs.md`](./test_pocketfft.py_docs.md_docs.md)
- [`test_pocketfft.py_kw.md_docs.md`](./test_pocketfft.py_kw.md_docs.md)
- [`test_helper.py_kw.md_docs.md`](./test_helper.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_helper.py_docs.md_docs.md`
- **Keyword Index**: `test_helper.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
