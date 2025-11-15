# Documentation: `docs/test/dynamo/test_exc.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_exc.py_docs.md`
- **Size**: 14,462 bytes (14.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_exc.py`

## File Metadata

- **Path**: `test/dynamo/test_exc.py`
- **Size**: 10,619 bytes (10.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import logging
import unittest

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
from torch._dynamo.comptime import comptime
from torch._dynamo.exc import Unsupported
from torch.testing._internal.common_device_type import skipIf
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    munge_exc,
    skipIfWindows,
    TEST_Z3,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class ExcTests(LoggingTestCase):
    maxDiff = None

    def test_unsupported_real_stack(self):
        # exercise Unsupported constructor and augment_exc_message
        def fn002(x):
            torch._dynamo.graph_break()

        def fn001(x):
            x = x + 1
            fn002(x)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn001, backend="eager", fullgraph=True)(
                torch.randn(1)
            ),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_exc.py", line N, in fn001
    fn002(x)
  File "test_exc.py", line N, in fn002
    torch._dynamo.graph_break()""",
        )

    @torch._dynamo.config.patch(verbose=True, suppress_errors=True)
    @make_logging_test()
    @unittest.skipIf(IS_FBCODE, "stack trace slightly different in fbcode")
    def test_internal_error_suppress_errors(self, records):
        def fn001(x):
            def f(ctx):
                raise AssertionError

            comptime(f)

        torch.compile(fn001, backend="eager")(torch.randn(1))

        record = self.getRecord(records, "WON'T CONVERT")

        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT fn001 test_exc.py line N
========== TorchDynamo Stack Trace ==========
Traceback (most recent call last):
  File "test_exc.py", line N, in f
    raise AssertionError
AssertionError:

from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)


========== The above exception occurred while processing the following code ==========

  File "test_exc.py", line N, in test_internal_error_suppress_errors
    torch.compile(fn001, backend="eager")(torch.randn(1))
  File "test_exc.py", line N, in fn001
    comptime(f)

==========""",
        )

    @make_logging_test()
    def test_not_implemented_error(self, records):
        def fn001(x):
            def f(ctx):
                raise NotImplementedError

            # Ensure graph break is not possible
            for _ in range(3):
                comptime(f)

        torch.compile(fn001, backend="eager")(torch.randn(1))

        record = self.getRecord(records, "WON'T CONVERT")

        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT fn001 test_exc.py line N
due to:
Traceback (most recent call last):
  File "test_exc.py", line N, in f
    raise NotImplementedError
torch._dynamo.exc.InternalTorchDynamoError: NotImplementedError:

from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)""",
        )

    @torch._dynamo.config.patch(inject_BUILD_SET_unimplemented_TESTING_ONLY=True)
    @make_logging_test(dynamo=logging.DEBUG)
    def test_unsupported_error(self, records):
        def fn001(x):
            return {1, 2}

        torch.compile(fn001, backend="eager")(torch.randn(1))

        # TODO: There is no graph break log!  This is because the graph break
        # logging is not in a centralized location; unsupported
        # instruction bypasses it
        self.getRecord(records, "Graph break:")

    @torch._dynamo.config.patch(suppress_errors=False)
    def test_internal_error_no_suppress(self):
        def fn001(x):
            # NB: avoid decorator, as 3.11 changed the line number attributed
            # in this situation
            def f(ctx):
                raise AssertionError

            comptime(f)

        # NB: OK for user code to be truncated here, because the regular
        # exception backtrace has the rest of the crumbs
        self.assertExpectedInlineMunged(
            AssertionError,
            lambda: torch.compile(fn001, backend="eager")(torch.randn(1)),
            """\


from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)""",
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_log(self, records):
        def fn002(x):
            x = x + 1
            torch._dynamo.graph_break()
            x = x + 1
            return x

        def fn001(x):
            return fn002(x)

        torch.compile(fn001, backend="eager")(torch.randn(1))

        record = self.getRecord(records, "Graph break in user code")

        # TODO: This should also report the enclosing frames; need to plumb
        # frame object to it
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
Graph break in user code at test_exc.py:N
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html
User code traceback:
  File "test_exc.py", line N, in test_graph_break_log
    torch.compile(fn001, backend="eager")(torch.randn(1))
  File "test_exc.py", line N, in fn001
    return fn002(x)
  File "test_exc.py", line N, in fn002
    torch._dynamo.graph_break()
""",  # noqa: B950
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_log_generic_jump(self, records):
        def fn(x):
            if x.sum() > 0:
                return x + 1
            else:
                return x - 1

        torch.compile(fn, backend="eager")(torch.ones(3, 3))

        # check for record existence
        self.getRecord(records, "Graph break in user code")

    @torch._dynamo.config.patch(suppress_errors=False)
    def test_backend_suppress_line(self):
        def fn001(x):
            x = torch.relu(x)
            return x + 1

        # Do NOT let this get attributed to x + 1
        self.assertExpectedInlineMunged(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: torch.compile(fn001, backend="relu_compile_error_TESTING_ONLY")(
                torch.randn(1)
            ),
            """\
backend='relu_compile_error_TESTING_ONLY' raised:
ReluCompileError:""",
        )

    @skipIf(not TEST_Z3, "z3 not installed")
    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        suppress_errors=False,
    )
    @torch.fx.experimental._config.patch(
        inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True,
        translation_validation=True,
        translation_validation_no_bisect=True,
    )
    @skipIfWindows(
        msg='AssertionError: "tran[551 chars]s1 s2 s3) s0)\n  ==> (<= (+ s1 s2) (+ s0 (* -1[511 chars][0])'  # noqa: PLR0133
        != 'tran[551 chars]s1 s2) (+ s0 (* -1 s3)))\n  ==> (<= (+ s1 s2) [483 chars][0])"'
    )
    def test_trigger_on_error(self):
        from torch.fx.experimental.validator import ValidationException

        @torch.compile
        def fn(x, shape):
            return x.split(shape)

        self.assertExpectedInlineMunged(
            ValidationException,
            lambda: fn(torch.randn(20), (5, 10, 5)),
            """\
translation validation failed.

Model:
  ==> L['shape'][0]: 0
  ==> L['shape'][1]: 0
  ==> L['shape'][2]: 0
  ==> L['x'].size()[0]: 3
  ==> L['x'].storage_offset(): 0
  ==> L['x'].stride()[0]: 1
  ==> s3: 0
  ==> s52: 0
  ==> s77: 3
  ==> s86: 0

Assertions:
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 1)

Target Expressions:
  ==> (!= (+ s3 s52 s86) s77)
  ==> (<= 0 s3)
  ==> (<= 0 s52)
  ==> (<= 0 s86)
  ==> (<= 2 s77)
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 0)
  ==> (>= 0 s86)

Failed Source Expressions:
  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])""",
        )

    @skipIf(not TEST_Z3, "z3 not installed")
    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        suppress_errors=False,
    )
    @torch.fx.experimental._config.patch(
        inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True,
        translation_validation=True,
    )
    def test_trigger_bisect_on_error(self):
        from torch.fx.experimental.validator import BisectValidationException

        @torch.compile
        def fn(x, shape):
            return x.split(shape)

        self.assertExpectedInlineMunged(
            BisectValidationException,
            lambda: fn(torch.randn(20), (5, 10, 5)),
            """\
translation validation failed when evaluating: Eq(s3 + s52 + s86, s77)

Failure occurred while running node:
    %split : [num_users=3] = call_method[target=split](args = (%l_x_, (%l_shape_0_, %l_shape_1_, %l_shape_2_)), kwargs = {})

Model:
  ==> L['shape'][0]: 0
  ==> L['shape'][1]: 0
  ==> L['shape'][2]: 0
  ==> L['x'].size()[0]: 3
  ==> L['x'].storage_offset(): 0
  ==> L['x'].stride()[0]: 1
  ==> s3: 0
  ==> s52: 0
  ==> s77: 3
  ==> s86: 0

Assertions:
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 1)

Target Expressions:
  ==> (!= (+ s3 s52 s86) s77)
  ==> (<= 0 s3)
  ==> (<= 0 s52)
  ==> (<= 0 s86)
  ==> (<= 2 s77)
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 0)

Failed Source Expressions:
  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview

"""\Call to `torch._dynamo.graph_break()`  Explanation: User-inserted graph break. Message: None  Hint: Remove the `torch._dynamo.graph_break()` call.  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}` For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.htmlfrom user code:   File "test_exc.py", line N, in fn001    fn002(x)

This Python file contains 1 class(es) and 25 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExcTests`

**Functions defined**: `test_unsupported_real_stack`, `fn002`, `fn001`, `test_internal_error_suppress_errors`, `fn001`, `f`, `test_not_implemented_error`, `fn001`, `f`, `test_unsupported_error`, `fn001`, `test_internal_error_no_suppress`, `fn001`, `f`, `test_graph_break_log`, `fn002`, `fn001`, `test_graph_break_log_generic_jump`, `fn`, `test_backend_suppress_line`

**Key imports**: logging, unittest, torch, torch._dynamo, torch._dynamo.config, torch._dynamo.test_case, comptime, Unsupported, skipIf, LoggingTestCase, make_logging_test


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `unittest`
- `torch`
- `torch._dynamo`
- `torch._dynamo.config`
- `torch._dynamo.test_case`
- `torch._dynamo.comptime`: comptime
- `torch._dynamo.exc`: Unsupported
- `torch.testing._internal.common_device_type`: skipIf
- `torch.testing._internal.logging_utils`: LoggingTestCase, make_logging_test
- `torch.fx.experimental.validator`: ValidationException


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python test/dynamo/test_exc.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_exc.py_docs.md`
- **Keyword Index**: `test_exc.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python docs/test/dynamo/test_exc.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo`):

- [`test_error_messages.py_docs.md_docs.md`](./test_error_messages.py_docs.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_unittest.py_docs.md_docs.md`](./test_unittest.py_docs.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_aot_autograd.py_kw.md_docs.md`](./test_aot_autograd.py_kw.md_docs.md)
- [`test_einops.py_docs.md_docs.md`](./test_einops.py_docs.md_docs.md)
- [`test_compile.py_kw.md_docs.md`](./test_compile.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_buffers_override.py_kw.md_docs.md`](./test_buffers_override.py_kw.md_docs.md)
- [`test_frame_init.py_docs.md_docs.md`](./test_frame_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_exc.py_docs.md_docs.md`
- **Keyword Index**: `test_exc.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
