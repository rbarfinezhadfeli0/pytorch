# Documentation: `docs/torch/quasirandom.py_docs.md`

## File Metadata

- **Path**: `docs/torch/quasirandom.py_docs.md`
- **Size**: 12,096 bytes (11.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/quasirandom.py`

## File Metadata

- **Path**: `torch/quasirandom.py`
- **Size**: 7,948 bytes (7.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional

import torch


class SobolEngine:
    r"""
    The :class:`torch.quasirandom.SobolEngine` is an engine for generating
    (scrambled) Sobol sequences. Sobol sequences are an example of low
    discrepancy quasi-random sequences.

    This implementation of an engine for Sobol sequences is capable of
    sampling sequences up to a maximum dimension of 21201. It uses direction
    numbers from https://web.maths.unsw.edu.au/~fkuo/sobol/ obtained using the
    search criterion D(6) up to the dimension 21201. This is the recommended
    choice by the authors.

    References:
      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
        Journal of Complexity, 14(4):466-489, December 1998.

      - I. M. Sobol. The distribution of points in a cube and the accurate
        evaluation of integrals.
        Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

    Args:
        dimension (Int): The dimensionality of the sequence to be drawn
        scramble (bool, optional): Setting this to ``True`` will produce
                                   scrambled Sobol sequences. Scrambling is
                                   capable of producing better Sobol
                                   sequences. Default: ``False``.
        seed (Int, optional): This is the seed for the scrambling. The seed
                              of the random number generator is set to this,
                              if specified. Otherwise, it uses a random seed.
                              Default: ``None``

    Examples::

        >>> # xdoctest: +SKIP("unseeded random state")
        >>> soboleng = torch.quasirandom.SobolEngine(dimension=5)
        >>> soboleng.draw(3)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.2500, 0.2500, 0.7500]])
    """

    MAXBIT = 30
    MAXDIM = 21201

    def __init__(self, dimension, scramble=False, seed=None):
        if dimension > self.MAXDIM or dimension < 1:
            raise ValueError(
                "Supported range of dimensionality "
                f"for SobolEngine is [1, {self.MAXDIM}]"
            )

        self.seed = seed
        self.scramble = scramble
        self.dimension = dimension

        cpu = torch.device("cpu")

        self.sobolstate = torch.zeros(
            dimension, self.MAXBIT, device=cpu, dtype=torch.long
        )
        torch._sobol_engine_initialize_state_(self.sobolstate, self.dimension)

        if not self.scramble:
            self.shift = torch.zeros(self.dimension, device=cpu, dtype=torch.long)
        else:
            self._scramble()

        self.quasi = self.shift.clone(memory_format=torch.contiguous_format)
        self._first_point = (self.quasi / 2**self.MAXBIT).reshape(1, -1)
        self.num_generated = 0

    def draw(
        self,
        n: int = 1,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""
        Function to draw a sequence of :attr:`n` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(n, dimension)`.

        Args:
            n (Int, optional): The length of sequence of points to draw.
                               Default: 1
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        if dtype is None:
            dtype = torch.get_default_dtype()

        if self.num_generated == 0:
            if n == 1:
                result = self._first_point.to(dtype)
            else:
                result, self.quasi = torch._sobol_engine_draw(
                    self.quasi,
                    n - 1,
                    self.sobolstate,
                    self.dimension,
                    self.num_generated,
                    dtype=dtype,
                )
                result = torch.cat((self._first_point.to(dtype), result), dim=-2)
        else:
            result, self.quasi = torch._sobol_engine_draw(
                self.quasi,
                n,
                self.sobolstate,
                self.dimension,
                self.num_generated - 1,
                dtype=dtype,
            )

        self.num_generated += n

        if out is not None:
            out.resize_as_(result).copy_(result)
            return out

        return result

    def draw_base2(
        self,
        m: int,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""
        Function to draw a sequence of :attr:`2**m` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(2**m, dimension)`.

        Args:
            m (Int): The (base2) exponent of the number of points to draw.
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        n = 2**m
        total_n = self.num_generated + n
        if not (total_n & (total_n - 1) == 0):
            raise ValueError(
                "The balance properties of Sobol' points require "
                f"n to be a power of 2. {self.num_generated} points have been "
                f"previously generated, then: n={self.num_generated}+2**{m}={total_n}. "
                "If you still want to do this, please use "
                "'SobolEngine.draw()' instead."
            )
        return self.draw(n=n, out=out, dtype=dtype)

    def reset(self):
        r"""
        Function to reset the ``SobolEngine`` to base state.
        """
        self.quasi.copy_(self.shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        r"""
        Function to fast-forward the state of the ``SobolEngine`` by
        :attr:`n` steps. This is equivalent to drawing :attr:`n` samples
        without using the samples.

        Args:
            n (Int): The number of steps to fast-forward by.
        """
        if self.num_generated == 0:
            torch._sobol_engine_ff_(
                self.quasi, n - 1, self.sobolstate, self.dimension, self.num_generated
            )
        else:
            torch._sobol_engine_ff_(
                self.quasi, n, self.sobolstate, self.dimension, self.num_generated - 1
            )
        self.num_generated += n
        return self

    def _scramble(self):
        g: Optional[torch.Generator] = None
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed)

        cpu = torch.device("cpu")

        # Generate shift vector
        shift_ints = torch.randint(
            2, (self.dimension, self.MAXBIT), device=cpu, generator=g
        )
        self.shift = torch.mv(
            shift_ints, torch.pow(2, torch.arange(0, self.MAXBIT, device=cpu))
        )

        # Generate lower triangular matrices (stacked across dimensions)
        ltm_dims = (self.dimension, self.MAXBIT, self.MAXBIT)
        ltm = torch.randint(2, ltm_dims, device=cpu, generator=g).tril()

        torch._sobol_engine_scramble_(self.sobolstate, ltm, self.dimension)

    def __repr__(self):
        fmt_string = [f"dimension={self.dimension}"]
        if self.scramble:
            fmt_string += ["scramble=True"]
        if self.seed is not None:
            fmt_string += [f"seed={self.seed}"]
        return self.__class__.__name__ + "(" + ", ".join(fmt_string) + ")"

```



## High-Level Overview

r"""    The :class:`torch.quasirandom.SobolEngine` is an engine for generating    (scrambled) Sobol sequences. Sobol sequences are an example of low    discrepancy quasi-random sequences.    This implementation of an engine for Sobol sequences is capable of    sampling sequences up to a maximum dimension of 21201. It uses direction    numbers from https://web.maths.unsw.edu.au/~fkuo/sobol/ obtained using the    search criterion D(6) up to the dimension 21201. This is the recommended    choice by the authors.    References:      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.        Journal of Complexity, 14(4):466-489, December 1998.      - I. M. Sobol. The distribution of points in a cube and the accurate        evaluation of integrals.        Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.    Args:        dimension (Int): The dimensionality of the sequence to be drawn        scramble (bool, optional): Setting this to ``True`` will produce                                   scrambled Sobol sequences. Scrambling is                                   capable of producing better Sobol                                   sequences. Default: ``False``.        seed (Int, optional): This is the seed for the scrambling. The seed                              of the random number generator is set to this,                              if specified. Otherwise, it uses a random seed.                              Default: ``None``    Examples::        >>> # xdoctest: +SKIP("unseeded random state")        >>> soboleng = torch.quasirandom.SobolEngine(dimension=5)        >>> soboleng.draw(3)        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],                [0.7500, 0.2500, 0.2500, 0.2500, 0.7500]])

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SobolEngine`

**Functions defined**: `__init__`, `draw`, `draw_base2`, `reset`, `fast_forward`, `_scramble`, `__repr__`

**Key imports**: Optional, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tensor_docs.py_docs.md`](./_tensor_docs.py_docs.md)
- [`_classes.py_docs.md`](./_classes.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`_meta_registrations.py_docs.md`](./_meta_registrations.py_docs.md)
- [`_appdirs.py_docs.md`](./_appdirs.py_docs.md)
- [`_tensor.py_docs.md`](./_tensor.py_docs.md)
- [`_streambase.py_docs.md`](./_streambase.py_docs.md)
- [`_lowrank.py_docs.md`](./_lowrank.py_docs.md)
- [`_size_docs.py_docs.md`](./_size_docs.py_docs.md)


## Cross-References

- **File Documentation**: `quasirandom.py_docs.md`
- **Keyword Index**: `quasirandom.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch`):

- [`types.py_kw.md_docs.md`](./types.py_kw.md_docs.md)
- [`storage.py_docs.md_docs.md`](./storage.py_docs.md_docs.md)
- [`serialization.py_kw.md_docs.md`](./serialization.py_kw.md_docs.md)
- [`serialization.py_docs.md_docs.md`](./serialization.py_docs.md_docs.md)
- [`library.py_kw.md_docs.md`](./library.py_kw.md_docs.md)
- [`overrides.py_docs.md_docs.md`](./overrides.py_docs.md_docs.md)
- [`script.h_kw.md_docs.md`](./script.h_kw.md_docs.md)
- [`_sources.py_kw.md_docs.md`](./_sources.py_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`_torch_docs.py_docs.md_docs.md`](./_torch_docs.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `quasirandom.py_docs.md_docs.md`
- **Keyword Index**: `quasirandom.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
