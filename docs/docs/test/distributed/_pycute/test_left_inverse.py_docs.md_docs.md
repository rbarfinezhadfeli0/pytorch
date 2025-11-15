# Documentation: `docs/test/distributed/_pycute/test_left_inverse.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/_pycute/test_left_inverse.py_docs.md`
- **Size**: 5,709 bytes (5.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/_pycute/test_left_inverse.py`

## File Metadata

- **Path**: `test/distributed/_pycute/test_left_inverse.py`
- **Size**: 3,245 bytes (3.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# ruff: noqa: PGH004, G004, F403
# flake8: noqa
# Owner(s): ["oncall: distributed"]
#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Unit tests for _pycute.left_inverse
"""

import logging

from torch.distributed._pycute import *
from torch.testing._internal.common_utils import run_tests, TestCase


_LOGGER = logging.getLogger(__name__)


class TestLeftInverse(TestCase):
    def helper_test_left_inverse(self, layout):
        inv_layout = left_inverse(layout)

        _LOGGER.debug(f"{layout}  =>  {inv_layout}")

        for i in range(size(layout)):
            self.assertEqual(inv_layout(layout(i)), i)

    def test_left_inverse(self):
        test = Layout(1, 0)
        self.helper_test_left_inverse(test)

        test = Layout((1, 1), (0, 0))
        self.helper_test_left_inverse(test)

        test = Layout(1, 1)
        self.helper_test_left_inverse(test)

        test = Layout(4, 1)
        self.helper_test_left_inverse(test)

        test = Layout(4, 2)
        self.helper_test_left_inverse(test)

        test = Layout((8, 4), (1, 8))
        self.helper_test_left_inverse(test)

        test = Layout((8, 4), (4, 1))
        self.helper_test_left_inverse(test)

        test = Layout((2, 4, 6), (1, 2, 8))
        self.helper_test_left_inverse(test)

        test = Layout((2, 4, 6), (4, 1, 8))
        self.helper_test_left_inverse(test)

        test = Layout((4, 2), (1, 16))
        self.helper_test_left_inverse(test)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Unit tests for _pycute.left_inverse

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestLeftInverse`

**Functions defined**: `helper_test_left_inverse`, `test_left_inverse`

**Key imports**: logging, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_pycute`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `torch.testing._internal.common_utils`: run_tests, TestCase


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
python test/distributed/_pycute/test_left_inverse.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_pycute`):

- [`test_coalesce.py_docs.md`](./test_coalesce.py_docs.md)
- [`test_typing.py_docs.md`](./test_typing.py_docs.md)
- [`test_int_tuple.py_docs.md`](./test_int_tuple.py_docs.md)
- [`test_complement.py_docs.md`](./test_complement.py_docs.md)
- [`test_composition.py_docs.md`](./test_composition.py_docs.md)
- [`test_right_inverse.py_docs.md`](./test_right_inverse.py_docs.md)


## Cross-References

- **File Documentation**: `test_left_inverse.py_docs.md`
- **Keyword Index**: `test_left_inverse.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/_pycute`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/_pycute`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/_pycute/test_left_inverse.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_pycute`):

- [`test_complement.py_kw.md_docs.md`](./test_complement.py_kw.md_docs.md)
- [`test_composition.py_docs.md_docs.md`](./test_composition.py_docs.md_docs.md)
- [`test_left_inverse.py_kw.md_docs.md`](./test_left_inverse.py_kw.md_docs.md)
- [`test_coalesce.py_docs.md_docs.md`](./test_coalesce.py_docs.md_docs.md)
- [`test_right_inverse.py_docs.md_docs.md`](./test_right_inverse.py_docs.md_docs.md)
- [`test_complement.py_docs.md_docs.md`](./test_complement.py_docs.md_docs.md)
- [`test_coalesce.py_kw.md_docs.md`](./test_coalesce.py_kw.md_docs.md)
- [`test_int_tuple.py_docs.md_docs.md`](./test_int_tuple.py_docs.md_docs.md)
- [`test_int_tuple.py_kw.md_docs.md`](./test_int_tuple.py_kw.md_docs.md)
- [`test_typing.py_docs.md_docs.md`](./test_typing.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_left_inverse.py_docs.md_docs.md`
- **Keyword Index**: `test_left_inverse.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
