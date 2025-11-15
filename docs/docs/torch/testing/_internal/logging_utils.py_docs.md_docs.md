# Documentation: `docs/torch/testing/_internal/logging_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/logging_utils.py_docs.md`
- **Size**: 11,474 bytes (11.21 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/logging_utils.py`

## File Metadata

- **Path**: `torch/testing/_internal/logging_utils.py`
- **Size**: 8,240 bytes (8.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: ignore-errors

import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from contextlib import AbstractContextManager
from collections.abc import Callable
from torch._dynamo.utils import LazyString
from torch._inductor import config as inductor_config
import logging
import io

@contextlib.contextmanager
def preserve_log_state():
    prev_state = torch._logging._internal._get_log_state()
    torch._logging._internal._set_log_state(torch._logging._internal.LogState())
    try:
        yield
    finally:
        torch._logging._internal._set_log_state(prev_state)
        torch._logging._internal._init_logs()

def log_settings(settings):
    exit_stack = contextlib.ExitStack()
    settings_patch = unittest.mock.patch.dict(os.environ, {"TORCH_LOGS": settings})
    exit_stack.enter_context(preserve_log_state())
    exit_stack.enter_context(settings_patch)
    torch._logging._internal._init_logs()
    return exit_stack

def log_api(**kwargs):
    exit_stack = contextlib.ExitStack()
    exit_stack.enter_context(preserve_log_state())
    torch._logging.set_logs(**kwargs)
    return exit_stack


def kwargs_to_settings(**kwargs):
    INT_TO_VERBOSITY = {10: "+", 20: "", 40: "-"}

    settings = []

    def append_setting(name, level):
        if isinstance(name, str) and isinstance(level, int) and level in INT_TO_VERBOSITY:
            settings.append(INT_TO_VERBOSITY[level] + name)
            return
        else:
            raise ValueError("Invalid value for setting")

    for name, val in kwargs.items():
        if isinstance(val, bool):
            settings.append(name)
        elif isinstance(val, int):
            append_setting(name, val)
        elif isinstance(val, dict) and name == "modules":
            for module_qname, level in val.items():
                append_setting(module_qname, level)
        else:
            raise ValueError("Invalid value for setting")

    return ",".join(settings)


# Note on testing strategy:
# This class does two things:
# 1. Runs two versions of a test:
#    1a. patches the env var log settings to some specific value
#    1b. calls torch._logging.set_logs(..)
# 2. patches the emit method of each setup handler to gather records
# that are emitted to each console stream
# 3. passes a ref to the gathered records to each test case for checking
#
# The goal of this testing in general is to ensure that given some settings env var
# that the logs are setup correctly and capturing the correct records.
def make_logging_test(**kwargs):
    def wrapper(fn):
        @inductor_config.patch({"fx_graph_cache": False})
        def test_fn(self):

            torch._dynamo.reset()
            records = []
            # run with env var
            if len(kwargs) == 0:
                with self._handler_watcher(records):
                    fn(self, records)
            else:
                with log_settings(kwargs_to_settings(**kwargs)), self._handler_watcher(records):
                    fn(self, records)

            # run with API
            torch._dynamo.reset()
            records.clear()
            with log_api(**kwargs), self._handler_watcher(records):
                fn(self, records)


        return test_fn

    return wrapper

def make_settings_test(settings):
    def wrapper(fn):
        def test_fn(self):
            torch._dynamo.reset()
            records = []
            # run with env var
            with log_settings(settings), self._handler_watcher(records):
                fn(self, records)

        return test_fn

    return wrapper

class LoggingTestCase(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.dict(os.environ, {"___LOG_TESTING": ""})
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch("torch._dynamo.config.suppress_errors", True)
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch("torch._dynamo.config.verbose", False)
        )

    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()
        torch._logging._internal.log_state.clear()
        torch._logging._init_logs()

    def hasRecord(self, records, m):
        return any(m in r.getMessage() for r in records)

    def getRecord(self, records, m):
        record = None
        for r in records:
            # NB: not r.msg because it looks like 3.11 changed how they
            # structure log records
            if m in r.getMessage():
                self.assertIsNone(
                    record,
                    msg=LazyString(
                        lambda: f"multiple matching records: {record} and {r} among {records}"
                    ),
                )
                record = r
        if record is None:
            self.fail(f"did not find record with {m} among {records}")
        return record

    # This patches the emit method of each handler to gather records
    # as they are emitted
    def _handler_watcher(self, record_list):
        exit_stack = contextlib.ExitStack()

        def emit_post_hook(record):
            nonlocal record_list
            record_list.append(record)

        # registered logs are the only ones with handlers, so patch those
        for log_qname in torch._logging._internal.log_registry.get_log_qnames():
            logger = logging.getLogger(log_qname)
            num_handlers = len(logger.handlers)
            self.assertLessEqual(
                num_handlers,
                2,
                "All pt2 loggers should only have at most two handlers (debug artifacts and messages above debug level).",
            )

            self.assertGreater(num_handlers, 0, "All pt2 loggers should have more than zero handlers")

            for handler in logger.handlers:
                old_emit = handler.emit

                def new_emit(record):
                    old_emit(record)
                    emit_post_hook(record)

                exit_stack.enter_context(
                    unittest.mock.patch.object(handler, "emit", new_emit)
                )

        return exit_stack


def logs_to_string(module, log_option):
    """Example:
    logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
    returns the output of TORCH_LOGS="post_grad_graphs" from the
    torch._inductor.compile_fx module.
    """
    log_stream = io.StringIO()
    handler = logging.StreamHandler(stream=log_stream)

    @contextlib.contextmanager
    def tmp_redirect_logs():
        try:
            logger = torch._logging.getArtifactLogger(module, log_option)
            logger.addHandler(handler)
            yield
        finally:
            logger.removeHandler(handler)

    def ctx_manager():
        exit_stack = log_settings(log_option)
        exit_stack.enter_context(tmp_redirect_logs())
        return exit_stack

    return log_stream, ctx_manager


def multiple_logs_to_string(module: str, *log_options: str) -> tuple[list[io.StringIO], Callable[[], AbstractContextManager[None]]]:
    """Example:
    multiple_logs_to_string("torch._inductor.compile_fx", "pre_grad_graphs", "post_grad_graphs")
    returns the output of TORCH_LOGS="pre_graph_graphs, post_grad_graphs" from the
    torch._inductor.compile_fx module.
    """
    log_streams = [io.StringIO() for _ in range(len(log_options))]
    handlers = [logging.StreamHandler(stream=log_stream) for log_stream in log_streams]

    @contextlib.contextmanager
    def tmp_redirect_logs():
        loggers = [torch._logging.getArtifactLogger(module, option) for option in log_options]
        try:
            for logger, handler in zip(loggers, handlers, strict=True):
                logger.addHandler(handler)
            yield
        finally:
            for logger, handler in zip(loggers, handlers, strict=True):
                logger.removeHandler(handler)

    def ctx_manager() -> AbstractContextManager[None]:
        exit_stack = log_settings(", ".join(log_options))
        exit_stack.enter_context(tmp_redirect_logs())
        return exit_stack  # type: ignore[return-value]

    return log_streams, ctx_manager

```



## High-Level Overview


This Python file contains 2 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LoggingTestCase`

**Functions defined**: `preserve_log_state`, `log_settings`, `log_api`, `kwargs_to_settings`, `append_setting`, `make_logging_test`, `wrapper`, `test_fn`, `make_settings_test`, `wrapper`, `test_fn`, `setUpClass`, `tearDownClass`, `hasRecord`, `getRecord`, `_handler_watcher`, `emit_post_hook`, `new_emit`, `logs_to_string`, `tmp_redirect_logs`

**Key imports**: torch._dynamo.test_case, unittest.mock, os, contextlib, torch._logging, torch._logging._internal, AbstractContextManager, Callable, LazyString, config as inductor_config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch._dynamo.test_case`
- `unittest.mock`
- `os`
- `contextlib`
- `torch._logging`
- `torch._logging._internal`
- `collections.abc`: Callable
- `torch._dynamo.utils`: LazyString
- `torch._inductor`: config as inductor_config
- `logging`
- `io`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python torch/testing/_internal/logging_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal`):

- [`common_jit.py_docs.md`](./common_jit.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_function_db.py_docs.md`](./autograd_function_db.py_docs.md)
- [`custom_op_db.py_docs.md`](./custom_op_db.py_docs.md)
- [`subclasses.py_docs.md`](./subclasses.py_docs.md)
- [`two_tensor.py_docs.md`](./two_tensor.py_docs.md)
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `logging_utils.py_docs.md`
- **Keyword Index**: `logging_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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
python docs/torch/testing/_internal/logging_utils.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal`):

- [`hypothesis_utils.py_kw.md_docs.md`](./hypothesis_utils.py_kw.md_docs.md)
- [`fake_config_module2.py_kw.md_docs.md`](./fake_config_module2.py_kw.md_docs.md)
- [`static_module.py_kw.md_docs.md`](./static_module.py_kw.md_docs.md)
- [`common_pruning.py_kw.md_docs.md`](./common_pruning.py_kw.md_docs.md)
- [`composite_compliance.py_kw.md_docs.md`](./composite_compliance.py_kw.md_docs.md)
- [`common_mkldnn.py_docs.md_docs.md`](./common_mkldnn.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`common_dtype.py_docs.md_docs.md`](./common_dtype.py_docs.md_docs.md)
- [`common_methods_invocations.py_docs.md_docs.md`](./common_methods_invocations.py_docs.md_docs.md)
- [`hypothesis_utils.py_docs.md_docs.md`](./hypothesis_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `logging_utils.py_docs.md_docs.md`
- **Keyword Index**: `logging_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
