# Documentation: `docs/torch/fft/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fft/__init__.py_docs.md`
- **Size**: 52,577 bytes (51.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/fft/__init__.py`

## File Metadata

- **Path**: `torch/fft/__init__.py`
- **Size**: 55,337 bytes (54.04 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
import torch
from torch._C import _add_docstr, _fft  # type: ignore[attr-defined]
from torch._torch_docs import common_args, factory_common_args


__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
    "Tensor",
]

Tensor = torch.Tensor

# Note: This not only adds the doc strings for the spectral ops, but
# connects the torch.fft Python namespace to the torch._C._fft builtins.

fft = _add_docstr(
    _fft.fft_fft,
    r"""
fft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional discrete Fourier transform of :attr:`input`.

Note:
    The Fourier domain representation of any real signal satisfies the
    Hermitian property: `X[i] = conj(X[-i])`. This function always returns both
    the positive and negative frequency terms even though, for real inputs, the
    negative frequencies are redundant. :func:`~torch.fft.rfft` returns the
    more compact one-sided representation where only the positive frequencies
    are returned.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimension.

Args:
    input (Tensor): the input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the FFT.
    dim (int, optional): The dimension along which to take the one dimensional FFT.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fft`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Calling the backward transform (:func:`~torch.fft.ifft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifft`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> torch.fft.fft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

    >>> t = torch.tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
    >>> torch.fft.fft(t)
    tensor([12.+16.j, -8.+0.j, -4.-4.j,  0.-8.j])
""".format(**common_args),
)

ifft = _add_docstr(
    _fft.fft_ifft,
    r"""
ifft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional inverse discrete Fourier transform of :attr:`input`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimension.

Args:
    input (Tensor): the input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the IFFT.
    dim (int, optional): The dimension along which to take the one dimensional IFFT.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ifft`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

        Calling the forward transform (:func:`~torch.fft.fft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifft`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> t = torch.tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    >>> torch.fft.ifft(t)
    tensor([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j])
""".format(**common_args),
)

fft2 = _add_docstr(
    _fft.fft_fft2,
    r"""
fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2 dimensional discrete Fourier transform of :attr:`input`.
Equivalent to :func:`~torch.fft.fftn` but FFTs only the last two dimensions by default.

Note:
    The Fourier domain representation of any real signal satisfies the
    Hermitian property: ``X[i, j] = conj(X[-i, -j])``. This
    function always returns all positive and negative frequency terms even
    though, for real inputs, half of these values are redundant.
    :func:`~torch.fft.rfft2` returns the more compact one-sided representation
    where only the positive frequencies of the last dimension are returned.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fft2`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ifft2`) with the same
        normalization mode will apply an overall normalization of ``1/n``
        between the two transforms. This is required to make
        :func:`~torch.fft.ifft2` the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> fft2 = torch.fft.fft2(x)

    The discrete Fourier transform is separable, so :func:`~torch.fft.fft2`
    here is equivalent to two one-dimensional :func:`~torch.fft.fft` calls:

    >>> two_ffts = torch.fft.fft(torch.fft.fft(x, dim=0), dim=1)
    >>> torch.testing.assert_close(fft2, two_ffts, check_stride=False)

""".format(**common_args),
)

ifft2 = _add_docstr(
    _fft.fft_ifft2,
    r"""
ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2 dimensional inverse discrete Fourier transform of :attr:`input`.
Equivalent to :func:`~torch.fft.ifftn` but IFFTs only the last two dimensions by default.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the IFFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ifft2`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.fft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifft2`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> ifft2 = torch.fft.ifft2(x)

    The discrete Fourier transform is separable, so :func:`~torch.fft.ifft2`
    here is equivalent to two one-dimensional :func:`~torch.fft.ifft` calls:

    >>> two_iffts = torch.fft.ifft(torch.fft.ifft(x, dim=0), dim=1)
    >>> torch.testing.assert_close(ifft2, two_iffts, check_stride=False)

""".format(**common_args),
)

fftn = _add_docstr(
    _fft.fft_fftn,
    r"""
fftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the N dimensional discrete Fourier transform of :attr:`input`.

Note:
    The Fourier domain representation of any real signal satisfies the
    Hermitian property: ``X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])``. This
    function always returns all positive and negative frequency terms even
    though, for real inputs, half of these values are redundant.
    :func:`~torch.fft.rfftn` returns the more compact one-sided representation
    where only the positive frequencies of the last dimension are returned.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fftn`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ifftn`) with the same
        normalization mode will apply an overall normalization of ``1/n``
        between the two transforms. This is required to make
        :func:`~torch.fft.ifftn` the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> fftn = torch.fft.fftn(x)

    The discrete Fourier transform is separable, so :func:`~torch.fft.fftn`
    here is equivalent to two one-dimensional :func:`~torch.fft.fft` calls:

    >>> two_ffts = torch.fft.fft(torch.fft.fft(x, dim=0), dim=1)
    >>> torch.testing.assert_close(fftn, two_ffts, check_stride=False)

""".format(**common_args),
)

ifftn = _add_docstr(
    _fft.fft_ifftn,
    r"""
ifftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the N dimensional inverse discrete Fourier transform of :attr:`input`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the IFFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ifftn`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.fftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifftn`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> ifftn = torch.fft.ifftn(x)

    The discrete Fourier transform is separable, so :func:`~torch.fft.ifftn`
    here is equivalent to two one-dimensional :func:`~torch.fft.ifft` calls:

    >>> two_iffts = torch.fft.ifft(torch.fft.ifft(x, dim=0), dim=1)
    >>> torch.testing.assert_close(ifftn, two_iffts, check_stride=False)

""".format(**common_args),
)

rfft = _add_docstr(
    _fft.fft_rfft,
    r"""
rfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional Fourier transform of real-valued :attr:`input`.

The FFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])`` so
the output contains only the positive frequencies below the Nyquist frequency.
To compute the full output, use :func:`~torch.fft.fft`

Note:
    Supports torch.half on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimension.

Args:
    input (Tensor): the real input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the real FFT.
    dim (int, optional): The dimension along which to take the one dimensional real FFT.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.rfft`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Calling the backward transform (:func:`~torch.fft.irfft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> torch.fft.rfft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j])

    Compare against the full output from :func:`~torch.fft.fft`:

    >>> torch.fft.fft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

    Notice that the symmetric element ``T[-1] == T[1].conj()`` is omitted.
    At the Nyquist frequency ``T[-2] == T[2]`` is it's own symmetric pair,
    and therefore must always be real-valued.
""".format(**common_args),
)

irfft = _add_docstr(
    _fft.fft_irfft,
    r"""
irfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfft`.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as produced by :func:`~torch.fft.rfft`. By the Hermitian property, the
output will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`n`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal length :attr:`n`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimension.
    With default arguments, size of the transformed dimension should be (2^n + 1) as argument
    `n` defaults to even output size = 2 * (transformed_dim_size - 1)

Args:
    input (Tensor): the input tensor representing a half-Hermitian signal
    n (int, optional): Output signal length. This determines the length of the
        output signal. If given, the input will either be zero-padded or trimmed to this
        length before computing the real IFFT.
        Defaults to even output: ``n=2*(input.size(dim) - 1)``.
    dim (int, optional): The dimension along which to take the one dimensional real IFFT.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.irfft`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

        Calling the forward transform (:func:`~torch.fft.rfft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> t = torch.linspace(0, 1, 5)
    >>> t
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    >>> T = torch.fft.rfft(t)
    >>> T
    tensor([ 2.5000+0.0000j, -0.6250+0.8602j, -0.6250+0.2031j])

    Without specifying the output length to :func:`~torch.fft.irfft`, the output
    will not round-trip properly because the input is odd-length:

    >>> torch.fft.irfft(T)
    tensor([0.1562, 0.3511, 0.7812, 1.2114])

    So, it is recommended to always pass the signal length :attr:`n`:

    >>> roundtrip = torch.fft.irfft(T, t.numel())
    >>> torch.testing.assert_close(roundtrip, t, check_stride=False)

""".format(**common_args),
)

rfft2 = _add_docstr(
    _fft.fft_rfft2,
    r"""
rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2-dimensional discrete Fourier transform of real :attr:`input`.
Equivalent to :func:`~torch.fft.rfftn` but FFTs only the last two dimensions by default.

The FFT of a real signal is Hermitian-symmetric, ``X[i, j] = conj(X[-i, -j])``,
so the full :func:`~torch.fft.fft2` output contains redundant information.
:func:`~torch.fft.rfft2` instead omits the negative frequencies in the last
dimension.

Note:
    Supports torch.half on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.rfft2`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.irfft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft2`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> t = torch.rand(10, 10)
    >>> rfft2 = torch.fft.rfft2(t)
    >>> rfft2.size()
    torch.Size([10, 6])

    Compared against the full output from :func:`~torch.fft.fft2`, we have all
    elements up to the Nyquist frequency.

    >>> fft2 = torch.fft.fft2(t)
    >>> torch.testing.assert_close(fft2[..., :6], rfft2, check_stride=False)

    The discrete Fourier transform is separable, so :func:`~torch.fft.rfft2`
    here is equivalent to a combination of :func:`~torch.fft.fft` and
    :func:`~torch.fft.rfft`:

    >>> two_ffts = torch.fft.fft(torch.fft.rfft(t, dim=1), dim=0)
    >>> torch.testing.assert_close(rfft2, two_ffts, check_stride=False)

""".format(**common_args),
)

irfft2 = _add_docstr(
    _fft.fft_irfft2,
    r"""
irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfft2`.
Equivalent to :func:`~torch.fft.irfftn` but IFFTs only the last two dimensions by default.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as produced by :func:`~torch.fft.rfft2`. By the Hermitian property, the
output will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`s`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal shape :attr:`s`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.
    With default arguments, the size of last dimension should be (2^n + 1) as argument
    `s` defaults to even output size = 2 * (last_dim_size - 1)

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Defaults to even output in the last dimension:
        ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
    dim (Tuple[int], optional): Dimensions to be transformed.
        The last dimension must be the half-Hermitian compressed dimension.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.irfft2`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.rfft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft2`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> t = torch.rand(10, 9)
    >>> T = torch.fft.rfft2(t)

    Without specifying the output length to :func:`~torch.fft.irfft2`, the output
    will not round-trip properly because the input is odd-length in the last
    dimension:

    >>> torch.fft.irfft2(T).size()
    torch.Size([10, 8])

    So, it is recommended to always pass the signal shape :attr:`s`.

    >>> roundtrip = torch.fft.irfft2(T, t.size())
    >>> roundtrip.size()
    torch.Size([10, 9])
    >>> torch.testing.assert_close(roundtrip, t, check_stride=False)

""".format(**common_args),
)

rfftn = _add_docstr(
    _fft.fft_rfftn,
    r"""
rfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the N-dimensional discrete Fourier transform of real :attr:`input`.

The FFT of a real signal is Hermitian-symmetric,
``X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])`` so the full
:func:`~torch.fft.fftn` output contains redundant information.
:func:`~torch.fft.rfftn` instead omits the negative frequencies in the
last dimension.

Note:
    Supports torch.half on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.rfftn`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.irfftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfftn`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> t = torch.rand(10, 10)
    >>> rfftn = torch.fft.rfftn(t)
    >>> rfftn.size()
    torch.Size([10, 6])

    Compared against the full output from :func:`~torch.fft.fftn`, we have all
    elements up to the Nyquist frequency.

    >>> fftn = torch.fft.fftn(t)
    >>> torch.testing.assert_close(fftn[..., :6], rfftn, check_stride=False)

    The discrete Fourier transform is separable, so :func:`~torch.fft.rfftn`
    here is equivalent to a combination of :func:`~torch.fft.fft` and
    :func:`~torch.fft.rfft`:

    >>> two_ffts = torch.fft.fft(torch.fft.rfft(t, dim=1), dim=0)
    >>> torch.testing.assert_close(rfftn, two_ffts, check_stride=False)

""".format(**common_args),
)

irfftn = _add_docstr(
    _fft.fft_irfftn,
    r"""
irfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfftn`.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as produced by :func:`~torch.fft.rfftn`. By the Hermitian property, the
output will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`s`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal shape :attr:`s`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.
    With default arguments, the size of last dimension should be (2^n + 1) as argument
    `s` defaults to even output size = 2 * (last_dim_size - 1)

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Defaults to even output in the last dimension:
        ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
    dim (Tuple[int], optional): Dimensions to be transformed.
        The last dimension must be the half-Hermitian compressed dimension.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.irfftn`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.rfftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfftn`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> t = torch.rand(10, 9)
    >>> T = torch.fft.rfftn(t)

    Without specifying the output length to :func:`~torch.fft.irfft`, the output
    will not round-trip properly because the input is odd-length in the last
    dimension:

    >>> torch.fft.irfftn(T).size()
    torch.Size([10, 8])

    So, it is recommended to always pass the signal shape :attr:`s`.

    >>> roundtrip = torch.fft.irfftn(T, t.size())
    >>> roundtrip.size()
    torch.Size([10, 9])
    >>> torch.testing.assert_close(roundtrip, t, check_stride=False)

""".format(**common_args),
)

hfft = _add_docstr(
    _fft.fft_hfft,
    r"""
hfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional discrete Fourier transform of a Hermitian
symmetric :attr:`input` signal.

Note:

    :func:`~torch.fft.hfft`/:func:`~torch.fft.ihfft` are analogous to
    :func:`~torch.fft.rfft`/:func:`~torch.fft.irfft`. The real FFT expects
    a real signal in the time-domain and gives a Hermitian symmetry in the
    frequency-domain. The Hermitian FFT is the opposite; Hermitian symmetric in
    the time-domain and real-valued in the frequency-domain. For this reason,
    special care needs to be taken with the length argument :attr:`n`, in the
    same way as with :func:`~torch.fft.irfft`.

Note:
    Because the signal is Hermitian in the time-domain, the result will be
    real in the frequency domain. Note that some input frequencies must be
    real-valued to satisfy the Hermitian property. In these cases the imaginary
    component will be ignored. For example, any imaginary component in
    ``input[0]`` would result in one or more complex frequency terms which
    cannot be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`n`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal length :attr:`n`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimension.
    With default arguments, size of the transformed dimension should be (2^n + 1) as argument
    `n` defaults to even output size = 2 * (transformed_dim_size - 1)

Args:
    input (Tensor): the input tensor representing a half-Hermitian signal
    n (int, optional): Output signal length. This determines the length of the
        real output. If given, the input will either be zero-padded or trimmed to this
        length before computing the Hermitian FFT.
        Defaults to even output: ``n=2*(input.size(dim) - 1)``.
    dim (int, optional): The dimension along which to take the one dimensional Hermitian FFT.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.hfft`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian FFT orthonormal)

        Calling the backward transform (:func:`~torch.fft.ihfft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfft`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    Taking a real-valued frequency signal and bringing it into the time domain
    gives Hermitian symmetric output:

    >>> t = torch.linspace(0, 1, 5)
    >>> t
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    >>> T = torch.fft.ifft(t)
    >>> T
    tensor([ 0.5000-0.0000j, -0.1250-0.1720j, -0.1250-0.0406j, -0.1250+0.0406j,
            -0.1250+0.1720j])

    Note that ``T[1] == T[-1].conj()`` and ``T[2] == T[-2].conj()`` is
    redundant. We can thus compute the forward transform without considering
    negative frequencies:

    >>> torch.fft.hfft(T[:3], n=5)
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

    Like with :func:`~torch.fft.irfft`, the output length must be given in order
    to recover an even length output:

    >>> torch.fft.hfft(T[:3])
    tensor([0.1250, 0.2809, 0.6250, 0.9691])
""".format(**common_args),
)

ihfft = _add_docstr(
    _fft.fft_ihfft,
    r"""
ihfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.hfft`.

:attr:`input` must be a real-valued signal, interpreted in the Fourier domain.
The IFFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])``.
:func:`~torch.fft.ihfft` represents this in the one-sided form where only the
positive frequencies below the Nyquist frequency are included. To compute the
full output, use :func:`~torch.fft.ifft`.

Note:
    Supports torch.half on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimension.

Args:
    input (Tensor): the real input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the Hermitian IFFT.
    dim (int, optional): The dimension along which to take the one dimensional Hermitian IFFT.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ihfft`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

        Calling the forward transform (:func:`~torch.fft.hfft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfft`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> t = torch.arange(5)
    >>> t
    tensor([0, 1, 2, 3, 4])
    >>> torch.fft.ihfft(t)
    tensor([ 2.0000-0.0000j, -0.5000-0.6882j, -0.5000-0.1625j])

    Compare against the full output from :func:`~torch.fft.ifft`:

    >>> torch.fft.ifft(t)
    tensor([ 2.0000-0.0000j, -0.5000-0.6882j, -0.5000-0.1625j, -0.5000+0.1625j,
            -0.5000+0.6882j])
""".format(**common_args),
)

hfft2 = _add_docstr(
    _fft.fft_hfft2,
    r"""
hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2-dimensional discrete Fourier transform of a Hermitian symmetric
:attr:`input` signal. Equivalent to :func:`~torch.fft.hfftn` but only
transforms the last two dimensions by default.

:attr:`input` is interpreted as a one-sided Hermitian signal in the time
domain. By the Hermitian property, the Fourier transform will be real-valued.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.
    With default arguments, the size of last dimension should be (2^n + 1) as argument
    `s` defaults to even output size = 2 * (last_dim_size - 1)

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the Hermitian FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Defaults to even output in the last dimension:
        ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
    dim (Tuple[int], optional): Dimensions to be transformed.
        The last dimension must be the half-Hermitian compressed dimension.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.hfft2`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ihfft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfft2`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    Starting from a real frequency-space signal, we can generate a
    Hermitian-symmetric time-domain signal:
    >>> T = torch.rand(10, 9)
    >>> t = torch.fft.ihfft2(T)

    Without specifying the output length to :func:`~torch.fft.hfftn`, the
    output will not round-trip properly because the input is odd-length in the
    last dimension:

    >>> torch.fft.hfft2(t).size()
    torch.Size([10, 10])

    So, it is recommended to always pass the signal shape :attr:`s`.

    >>> roundtrip = torch.fft.hfft2(t, T.size())
    >>> roundtrip.size()
    torch.Size([10, 9])
    >>> torch.allclose(roundtrip, T)
    True

""".format(**common_args),
)

ihfft2 = _add_docstr(
    _fft.fft_ihfft2,
    r"""
ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2-dimensional inverse discrete Fourier transform of real
:attr:`input`. Equivalent to :func:`~torch.fft.ihfftn` but transforms only the
two last dimensions by default.

Note:
    Supports torch.half on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the Hermitian IFFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ihfft2`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.hfft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfft2`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> T = torch.rand(10, 10)
    >>> t = torch.fft.ihfft2(t)
    >>> t.size()
    torch.Size([10, 6])

    Compared against the full output from :func:`~torch.fft.ifft2`, the
    Hermitian time-space signal takes up only half the space.

    >>> fftn = torch.fft.ifft2(t)
    >>> torch.allclose(fftn[..., :6], rfftn)
    True

    The discrete Fourier transform is separable, so :func:`~torch.fft.ihfft2`
    here is equivalent to a combination of :func:`~torch.fft.ifft` and
    :func:`~torch.fft.ihfft`:

    >>> two_ffts = torch.fft.ifft(torch.fft.ihfft(t, dim=1), dim=0)
    >>> torch.allclose(t, two_ffts)
    True

""".format(**common_args),
)

hfftn = _add_docstr(
    _fft.fft_hfftn,
    r"""
hfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the n-dimensional discrete Fourier transform of a Hermitian symmetric
:attr:`input` signal.

:attr:`input` is interpreted as a one-sided Hermitian signal in the time
domain. By the Hermitian property, the Fourier transform will be real-valued.

Note:
    :func:`~torch.fft.hfftn`/:func:`~torch.fft.ihfftn` are analogous to
    :func:`~torch.fft.rfftn`/:func:`~torch.fft.irfftn`. The real FFT expects
    a real signal in the time-domain and gives Hermitian symmetry in the
    frequency-domain. The Hermitian FFT is the opposite; Hermitian symmetric in
    the time-domain and real-valued in the frequency-domain. For this reason,
    special care needs to be taken with the shape argument :attr:`s`, in the
    same way as with :func:`~torch.fft.irfftn`.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`s`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. It is recommended to always pass the signal shape :attr:`s`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.
    With default arguments, the size of last dimension should be (2^n + 1) as argument
    `s` defaults to even output size = 2 * (last_dim_size - 1)

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Defaults to even output in the last dimension:
        ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
    dim (Tuple[int], optional): Dimensions to be transformed.
        The last dimension must be the half-Hermitian compressed dimension.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.hfftn`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ihfftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfftn`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    Starting from a real frequency-space signal, we can generate a
    Hermitian-symmetric time-domain signal:
    >>> T = torch.rand(10, 9)
    >>> t = torch.fft.ihfftn(T)

    Without specifying the output length to :func:`~torch.fft.hfftn`, the
    output will not round-trip properly because the input is odd-length in the
    last dimension:

    >>> torch.fft.hfftn(t).size()
    torch.Size([10, 10])

    So, it is recommended to always pass the signal shape :attr:`s`.

    >>> roundtrip = torch.fft.hfftn(t, T.size())
    >>> roundtrip.size()
    torch.Size([10, 9])
    >>> torch.allclose(roundtrip, T)
    True

""".format(**common_args),
)

ihfftn = _add_docstr(
    _fft.fft_ihfftn,
    r"""
ihfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

Computes the N-dimensional inverse discrete Fourier transform of real :attr:`input`.

:attr:`input` must be a real-valued signal, interpreted in the Fourier domain.
The n-dimensional IFFT of a real signal is Hermitian-symmetric,
``X[i, j, ...] = conj(X[-i, -j, ...])``. :func:`~torch.fft.ihfftn` represents
this in the one-sided form where only the positive frequencies below the
Nyquist frequency are included in the last signal dimension. To compute the
full output, use :func:`~torch.fft.ifftn`.

Note:
    Supports torch.half on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the Hermitian IFFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ihfftn`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.hfftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfftn`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}

Example:

    >>> T = torch.rand(10, 10)
    >>> ihfftn = torch.fft.ihfftn(T)
    >>> ihfftn.size()
    torch.Size([10, 6])

    Compared against the full output from :func:`~torch.fft.ifftn`, we have all
    elements up to the Nyquist frequency.

    >>> ifftn = torch.fft.ifftn(t)
    >>> torch.allclose(ifftn[..., :6], ihfftn)
    True

    The discrete Fourier transform is separable, so :func:`~torch.fft.ihfftn`
    here is equivalent to a combination of :func:`~torch.fft.ihfft` and
    :func:`~torch.fft.ifft`:

    >>> two_iffts = torch.fft.ifft(torch.fft.ihfft(t, dim=1), dim=0)
    >>> torch.allclose(ihfftn, two_iffts)
    True

""".format(**common_args),
)

fftfreq = _add_docstr(
    _fft.fft_fftfreq,
    r"""
fftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the discrete Fourier Transform sample frequencies for a signal of size :attr:`n`.

Note:
    By convention, :func:`~torch.fft.fft` returns positive frequency terms
    first, followed by the negative frequencies in reverse order, so that
    ``f[-i]`` for all :math:`0 < i \leq n/2`` in Python gives the negative
    frequency terms. For an FFT of length :attr:`n` and with inputs spaced in
    length unit :attr:`d`, the frequencies are::

        f = [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)

Note:
    For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
    either negative or positive. :func:`~torch.fft.fftfreq` follows NumPy's
    convention of taking it to be negative.

Args:
    n (int): the FFT length
    d (float, optional): The sampling length scale.
        The spacing between individual samples of the FFT input.
        The default assumes unit spacing, dividing that result by the actual
        spacing gives the result in physical frequency units.

Keyword Args:
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example:

    >>> torch.fft.fftfreq(5)
    tensor([ 0.0000,  0.2000,  0.4000, -0.4000, -0.2000])

    For even input, we can see the Nyquist frequency at ``f[2]`` is given as
    negative:

    >>> torch.fft.fftfreq(4)
    tensor([ 0.0000,  0.2500, -0.5000, -0.2500])

""".format(**factory_common_args),
)

rfftfreq = _add_docstr(
    _fft.fft_rfftfreq,
    r"""
rfftfreq(n, d=1.0, *, out=None, dtype=None, layout=
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fft`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fft`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/fft`):

- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
