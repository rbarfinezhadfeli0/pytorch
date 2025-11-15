# Documentation: `docs/test/distributed/elastic/multiprocessing/errors/error_handler_test.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/elastic/multiprocessing/errors/error_handler_test.py_docs.md`
- **Size**: 6,769 bytes (6.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/elastic/multiprocessing/errors/error_handler_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/multiprocessing/errors/error_handler_test.py`
- **Size**: 4,076 bytes (3.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

import filecmp
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler
from torch.distributed.elastic.multiprocessing.errors.handlers import get_error_handler


def raise_exception_fn():
    raise RuntimeError("foobar")


class GetErrorHandlerTest(unittest.TestCase):
    def test_get_error_handler(self):
        self.assertTrue(isinstance(get_error_handler(), ErrorHandler))


class ErrorHandlerTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        self.test_error_file = os.path.join(self.test_dir, "error.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("faulthandler.enable")
    def test_initialize(self, fh_enable_mock):
        ErrorHandler().initialize()
        fh_enable_mock.assert_called_once()

    @patch("faulthandler.enable", side_effect=RuntimeError)
    def test_initialize_error(self, fh_enable_mock):
        # makes sure that initialize handles errors gracefully
        ErrorHandler().initialize()
        fh_enable_mock.assert_called_once()

    def test_record_exception(self):
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}):
            eh = ErrorHandler()
            eh.initialize()

            try:
                raise_exception_fn()
            except Exception as e:
                eh.record_exception(e)

            with open(self.test_error_file) as fp:
                err = json.load(fp)
                # error file content example:
                # {
                #   "message": {
                #     "message": "RuntimeError: foobar",
                #     "extraInfo": {
                #       "py_callstack": "Traceback (most recent call last):\n  <... OMITTED ...>",
                #       "timestamp": "1605774851"
                #     }
                #   }
            self.assertIsNotNone(err["message"]["message"])
            self.assertIsNotNone(err["message"]["extraInfo"]["py_callstack"])
            self.assertIsNotNone(err["message"]["extraInfo"]["timestamp"])

    def test_record_exception_no_error_file(self):
        # make sure record does not fail when no error file is specified in env vars
        with patch.dict(os.environ, {}):
            eh = ErrorHandler()
            eh.initialize()
            try:
                raise_exception_fn()
            except Exception as e:
                eh.record_exception(e)

    def test_dump_error_file(self):
        src_error_file = os.path.join(self.test_dir, "src_error.json")
        eh = ErrorHandler()
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": src_error_file}):
            eh.record_exception(RuntimeError("foobar"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}):
            eh.dump_error_file(src_error_file)
            self.assertTrue(filecmp.cmp(src_error_file, self.test_error_file))

        with patch.dict(os.environ, {}):
            eh.dump_error_file(src_error_file)
            # just validate that dump_error_file works when
            # my error file is not set
            # should just log an error with src_error_file pretty printed

    def test_dump_error_file_overwrite_existing(self):
        dst_error_file = os.path.join(self.test_dir, "dst_error.json")
        src_error_file = os.path.join(self.test_dir, "src_error.json")
        eh = ErrorHandler()
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": dst_error_file}):
            eh.record_exception(RuntimeError("foo"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": src_error_file}):
            eh.record_exception(RuntimeError("bar"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": dst_error_file}):
            eh.dump_error_file(src_error_file)
            self.assertTrue(filecmp.cmp(src_error_file, dst_error_file))

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GetErrorHandlerTest`, `ErrorHandlerTest`

**Functions defined**: `raise_exception_fn`, `test_get_error_handler`, `setUp`, `tearDown`, `test_initialize`, `test_initialize_error`, `test_record_exception`, `test_record_exception_no_error_file`, `test_dump_error_file`, `test_dump_error_file_overwrite_existing`

**Key imports**: filecmp, json, os, shutil, tempfile, unittest, patch, ErrorHandler, get_error_handler


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/multiprocessing/errors`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `filecmp`
- `json`
- `os`
- `shutil`
- `tempfile`
- `unittest`
- `unittest.mock`: patch
- `torch.distributed.elastic.multiprocessing.errors.error_handler`: ErrorHandler
- `torch.distributed.elastic.multiprocessing.errors.handlers`: get_error_handler


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/distributed/elastic/multiprocessing/errors/error_handler_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/multiprocessing/errors`):

- [`api_test.py_docs.md`](./api_test.py_docs.md)


## Cross-References

- **File Documentation**: `error_handler_test.py_docs.md`
- **Keyword Index**: `error_handler_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/elastic/multiprocessing/errors`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/elastic/multiprocessing/errors`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python docs/test/distributed/elastic/multiprocessing/errors/error_handler_test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/elastic/multiprocessing/errors`):

- [`error_handler_test.py_kw.md_docs.md`](./error_handler_test.py_kw.md_docs.md)
- [`api_test.py_kw.md_docs.md`](./api_test.py_kw.md_docs.md)
- [`api_test.py_docs.md_docs.md`](./api_test.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `error_handler_test.py_docs.md_docs.md`
- **Keyword Index**: `error_handler_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
