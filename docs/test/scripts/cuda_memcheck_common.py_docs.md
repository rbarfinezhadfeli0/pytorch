# Documentation: `test/scripts/cuda_memcheck_common.py`

## File Metadata

- **Path**: `test/scripts/cuda_memcheck_common.py`
- **Size**: 4,509 bytes (4.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
# this file contains a simple parser that parses report
# from cuda-memcheck


class ParseError(Exception):
    """Whenever the simple parser is unable to parse the report, this exception will be raised"""


class Report:
    """A report is a container of errors, and a summary on how many errors are found"""

    def __init__(self, text, errors):
        # text is something like
        # ERROR SUMMARY: 1 error
        # or
        # ERROR SUMMARY: 2 errors
        self.text = text
        self.num_errors = int(text.strip().split()[2])
        self.errors = errors
        if len(errors) != self.num_errors:
            if len(errors) == 10000 and self.num_errors > 10000:
                # When there are more than 10k errors, cuda-memcheck only display 10k
                self.num_errors = 10000
            else:
                raise ParseError("Number of errors does not match")


class Error:
    """Each error is a section in the output of cuda-memcheck.
    Each error in the report has an error message and a backtrace. It looks like:

    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b77a05]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x39d6d1d]
    =========     .....
    """

    def __init__(self, lines):
        self.message = lines[0]
        lines = lines[2:]
        self.stack = [l.strip() for l in lines]


def parse(message):
    """A simple parser that parses the report of cuda-memcheck. This parser is meant to be simple
    and it only split the report into separate errors and a summary. Where each error is further
    splitted into error message and backtrace. No further details are parsed.

    A report contains multiple errors and a summary on how many errors are detected. It looks like:

    ========= CUDA-MEMCHECK
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaPointerGetAttributes.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaPointerGetAttributes + 0x1a9) [0x428b9]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b778a9]
    =========     .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]
    =========     .....
    =========
    ========= .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     .....
    =========     Host Frame:python (_PyEval_EvalFrameDefault + 0x6a0) [0x1d0ad0]
    =========     Host Frame:python (_PyEval_EvalCodeWithName + 0xbb9) [0x116db9]
    =========
    ========= ERROR SUMMARY: 4 errors
    """
    errors = []
    HEAD = "========="
    headlen = len(HEAD)
    started = False
    in_message = False
    message_lines = []
    lines = message.splitlines()
    for l in lines:
        if l == HEAD + " CUDA-MEMCHECK":
            started = True
            continue
        if not started or not l.startswith(HEAD):
            continue
        l = l[headlen + 1 :]
        if l.startswith("ERROR SUMMARY:"):
            return Report(l, errors)
        if not in_message:
            in_message = True
            message_lines = [l]
        elif l == "":
            errors.append(Error(message_lines))
            in_message = False
        else:
            message_lines.append(l)
    raise ParseError("No error summary found")

```



## High-Level Overview

"""Whenever the simple parser is unable to parse the report, this exception will be raised"""class Report:

This Python file contains 3 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ParseError`, `Report`, `Error`

**Functions defined**: `__init__`, `__init__`, `parse`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/scripts`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*No imports detected.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/scripts/cuda_memcheck_common.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/scripts`):

- [`run_cuda_memcheck.py_docs.md`](./run_cuda_memcheck.py_docs.md)


## Cross-References

- **File Documentation**: `cuda_memcheck_common.py_docs.md`
- **Keyword Index**: `cuda_memcheck_common.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
