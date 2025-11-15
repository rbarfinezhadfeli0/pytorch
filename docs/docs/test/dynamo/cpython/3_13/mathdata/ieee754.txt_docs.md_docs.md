# Documentation: `docs/test/dynamo/cpython/3_13/mathdata/ieee754.txt_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/cpython/3_13/mathdata/ieee754.txt_docs.md`
- **Size**: 5,301 bytes (5.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/dynamo/cpython/3_13/mathdata/ieee754.txt`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/mathdata/ieee754.txt`
- **Size**: 3,226 bytes (3.15 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```
======================================
Python IEEE 754 floating point support
======================================

>>> from sys import float_info as FI
>>> from math import *
>>> PI = pi
>>> E = e

You must never compare two floats with == because you are not going to get
what you expect. We treat two floats as equal if the difference between them
is small than epsilon.
>>> EPS = 1E-15
>>> def equal(x, y):
...     """Almost equal helper for floats"""
...     return abs(x - y) < EPS


NaNs and INFs
=============

In Python 2.6 and newer NaNs (not a number) and infinity can be constructed
from the strings 'inf' and 'nan'.

>>> INF = float('inf')
>>> NINF = float('-inf')
>>> NAN = float('nan')

>>> INF
inf
>>> NINF
-inf
>>> NAN
nan

The math module's ``isnan`` and ``isinf`` functions can be used to detect INF
and NAN:
>>> isinf(INF), isinf(NINF), isnan(NAN)
(True, True, True)
>>> INF == -NINF
True

Infinity
--------

Ambiguous operations like ``0 * inf`` or ``inf - inf`` result in NaN.
>>> INF * 0
nan
>>> INF - INF
nan
>>> INF / INF
nan

However unambiguous operations with inf return inf:
>>> INF * INF
inf
>>> 1.5 * INF
inf
>>> 0.5 * INF
inf
>>> INF / 1000
inf

Not a Number
------------

NaNs are never equal to another number, even itself
>>> NAN == NAN
False
>>> NAN < 0
False
>>> NAN >= 0
False

All operations involving a NaN return a NaN except for nan**0 and 1**nan.
>>> 1 + NAN
nan
>>> 1 * NAN
nan
>>> 0 * NAN
nan
>>> 1 ** NAN
1.0
>>> NAN ** 0
1.0
>>> 0 ** NAN
nan
>>> (1.0 + FI.epsilon) * NAN
nan

Misc Functions
==============

The power of 1 raised to x is always 1.0, even for special values like 0,
infinity and NaN.

>>> pow(1, 0)
1.0
>>> pow(1, INF)
1.0
>>> pow(1, -INF)
1.0
>>> pow(1, NAN)
1.0

The power of 0 raised to x is defined as 0, if x is positive. Negative
finite values are a domain error or zero division error and NaN result in a
silent NaN.

>>> pow(0, 0)
1.0
>>> pow(0, INF)
0.0
>>> pow(0, -INF)
inf
>>> 0 ** -1
Traceback (most recent call last):
...
ZeroDivisionError: 0.0 cannot be raised to a negative power
>>> pow(0, NAN)
nan


Trigonometric Functions
=======================

>>> sin(INF)
Traceback (most recent call last):
...
ValueError: math domain error
>>> sin(NINF)
Traceback (most recent call last):
...
ValueError: math domain error
>>> sin(NAN)
nan
>>> cos(INF)
Traceback (most recent call last):
...
ValueError: math domain error
>>> cos(NINF)
Traceback (most recent call last):
...
ValueError: math domain error
>>> cos(NAN)
nan
>>> tan(INF)
Traceback (most recent call last):
...
ValueError: math domain error
>>> tan(NINF)
Traceback (most recent call last):
...
ValueError: math domain error
>>> tan(NAN)
nan

Neither pi nor tan are exact, but you can assume that tan(pi/2) is a large value
and tan(pi) is a very small value:
>>> tan(PI/2) > 1E10
True
>>> -tan(-PI/2) > 1E10
True
>>> tan(PI) < 1E-15
True

>>> asin(NAN), acos(NAN), atan(NAN)
(nan, nan, nan)
>>> asin(INF), asin(NINF)
Traceback (most recent call last):
...
ValueError: math domain error
>>> acos(INF), acos(NINF)
Traceback (most recent call last):
...
ValueError: math domain error
>>> equal(atan(INF), PI/2), equal(atan(NINF), -PI/2)
(True, True)


Hyberbolic Functions
====================


```



## High-Level Overview

This file is part of the PyTorch framework located at `test/dynamo/cpython/3_13/mathdata`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo/cpython/3_13/mathdata`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python test/dynamo/cpython/3_13/mathdata/ieee754.txt
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo/cpython/3_13/mathdata`):

- [`cmath_testcases.txt_docs.md`](./cmath_testcases.txt_docs.md)
- [`floating_points.txt_docs.md`](./floating_points.txt_docs.md)
- [`math_testcases.txt_docs.md`](./math_testcases.txt_docs.md)
- [`formatfloat_testcases.txt_docs.md`](./formatfloat_testcases.txt_docs.md)


## Cross-References

- **File Documentation**: `ieee754.txt_docs.md`
- **Keyword Index**: `ieee754.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo/cpython/3_13/mathdata`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo/cpython/3_13/mathdata`, which is part of the **testing infrastructure**.



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
python docs/test/dynamo/cpython/3_13/mathdata/ieee754.txt_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo/cpython/3_13/mathdata`):

- [`math_testcases.txt_docs.md_docs.md`](./math_testcases.txt_docs.md_docs.md)
- [`cmath_testcases.txt_kw.md_docs.md`](./cmath_testcases.txt_kw.md_docs.md)
- [`floating_points.txt_kw.md_docs.md`](./floating_points.txt_kw.md_docs.md)
- [`formatfloat_testcases.txt_docs.md_docs.md`](./formatfloat_testcases.txt_docs.md_docs.md)
- [`formatfloat_testcases.txt_kw.md_docs.md`](./formatfloat_testcases.txt_kw.md_docs.md)
- [`cmath_testcases.txt_docs.md_docs.md`](./cmath_testcases.txt_docs.md_docs.md)
- [`math_testcases.txt_kw.md_docs.md`](./math_testcases.txt_kw.md_docs.md)
- [`ieee754.txt_kw.md_docs.md`](./ieee754.txt_kw.md_docs.md)
- [`floating_points.txt_docs.md_docs.md`](./floating_points.txt_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ieee754.txt_docs.md_docs.md`
- **Keyword Index**: `ieee754.txt_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
