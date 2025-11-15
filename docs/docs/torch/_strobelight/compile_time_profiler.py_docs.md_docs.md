# Documentation: `docs/torch/_strobelight/compile_time_profiler.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_strobelight/compile_time_profiler.py_docs.md`
- **Size**: 10,123 bytes (9.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_strobelight/compile_time_profiler.py`

## File Metadata

- **Path**: `torch/_strobelight/compile_time_profiler.py`
- **Size**: 7,565 bytes (7.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: disallow-untyped-defs

import json
import logging
import os
import re
import subprocess
from datetime import datetime
from socket import gethostname
from typing import Any, Optional

from torch._strobelight.cli_function_profiler import StrobelightCLIFunctionProfiler


logger = logging.getLogger("strobelight_compile_time_profiler")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def get_fburl(url: str) -> str:
    short_url = url
    # Attempt to shorten the URL
    try:
        result = subprocess.run(
            ["fburl", url], capture_output=True, stdin=subprocess.DEVNULL
        )
        if result.returncode == 0:
            short_url = result.stdout.decode("utf-8")
    except Exception as e:
        logger.warning("URL shortening failed: %s, using long URL", repr(e))
    return short_url


def get_strobelight_url(identifier: str) -> str:
    scuba_json = {
        "aggregateList": [],
        "aggregation_field": "async_stack_complete",
        "b_constraints": [[]],
        "c_constraints": [[]],
        "cols": ["namespace_id", "namespace_process_id"],
        "compare": "none",
        "constraints": [
            [{"column": "sample_tags", "op": "all", "value": [f'["{identifier}"]']}]
        ],
        "derivedCols": [],
        "end": "now",
        "enumCols": [],
        "filterMode": "DEFAULT",
        "hideEmptyColumns": "false",
        "ignoreGroupByInComparison": "false",
        "is_timeseries": "false",
        "mappedCols": [],
        "metric": "count",
        "modifiers": [],
        "order": "weight",
        "order_desc": "true",
        "param_dimensions": [
            {"dim": "py_async_stack", "op": "edge", "param": "0", "anchor": "0"}
        ],
        "purposes": [],
        "return_remainder": "false",
        "samplingRatio": "1",
        "should_pivot": "false",
        "start": "-30 days",
        "timezone": "America/Los_Angeles",
        "top": 10000,
    }
    scuba_url_prefix = "https://www.internalfb.com/intern/scuba/query/?dataset=pyperf_experimental/on_demand&drillstate="
    scuba_url_suff = "&view=GraphProfilerView&&normalized=1726332703&pool=uber"
    long_url = scuba_url_prefix + json.dumps(scuba_json) + scuba_url_suff
    return get_fburl(long_url)


class StrobelightCompileTimeProfiler:
    success_profile_count: int = 0
    failed_profile_count: int = 0
    ignored_profile_runs: int = 0
    inside_profile_compile_time: bool = False
    enabled: bool = False

    # A regex that can be used to filter out what frames to profile. ex: "1/.*"
    frame_id_filter: Optional[str] = os.environ.get("COMPILE_STROBELIGHT_FRAME_FILTER")

    # A unique identifier that is used as the run_user_name in the strobelight profile to
    # associate all compile time profiles together.
    identifier: Optional[str] = None

    current_phase: Optional[str] = None

    profiler: Optional[Any] = None

    max_stack_length: int = int(
        os.environ.get("COMPILE_STROBELIGHT_MAX_STACK_LENGTH", 500)
    )
    max_profile_time: int = int(
        os.environ.get("COMPILE_STROBELIGHT_MAX_PROFILE_TIME", 60 * 30)
    )
    # Collect sample each x cycles.
    sample_each: int = int(
        float(os.environ.get("COMPILE_STROBELIGHT_SAMPLE_RATE", 1e7))
    )

    @classmethod
    def get_frame(cls) -> str:
        from torch._guards import CompileContext

        return (str)(CompileContext.current_trace_id())

    @classmethod
    def enable(cls, profiler_class: Any = StrobelightCLIFunctionProfiler) -> None:
        if cls.enabled:
            logger.info("compile time strobelight profiling already enabled")
            return

        logger.info("compile time strobelight profiling enabled")

        if profiler_class is StrobelightCLIFunctionProfiler:
            import shutil

            if not shutil.which("strobeclient"):
                logger.info(
                    "strobeclient not found, can't enable compile time strobelight profiling, seems"
                    "like you are not on a FB machine."
                )
                return

        cls.enabled = True
        cls._cls_init()
        # profiler_class should have public API similar to that of StrobelightCLIFunctionProfiler.
        # we have pass different functionProfilerClass for meta-internal fbcode targets.
        # NB: the actual implementation in Meta is at
        # fbcode/caffe2/fb/strobelight/function_profiler.py
        cls.profiler = profiler_class(
            sample_each=cls.sample_each,
            max_profile_duration_sec=cls.max_profile_time,
            stack_max_len=cls.max_stack_length,
            async_stack_max_len=cls.max_stack_length,
            run_user_name="pt2-profiler/"
            + os.environ.get("USER", os.environ.get("USERNAME", "")),
            sample_tags={cls.identifier},  # pyrefly: ignore  # bad-argument-type
        )

    @classmethod
    def _cls_init(cls) -> None:
        cls.identifier = "{date}{pid}{hostname}".format(
            date=datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            pid=os.getpid(),
            hostname=gethostname(),
        )

        logger.info("Unique sample tag for this run is: %s", cls.identifier)
        logger.info(
            "URL to access the strobelight profile at the end of the run: %s",
            get_strobelight_url(cls.identifier),
        )

    @classmethod
    def _log_stats(cls) -> None:
        logger.info(
            "%s strobelight success runs out of %s non-recursive compilation events.",
            cls.success_profile_count,
            cls.success_profile_count + cls.failed_profile_count,
        )

    # TODO use threadlevel meta data to tags to record phases.
    @classmethod
    def profile_compile_time(
        cls, func: Any, phase_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        def skip() -> Any:
            return func(*args, **kwargs)

        if not cls.enabled:
            return skip()

        if cls.profiler is None:
            logger.error("profiler is not set")
            return

        frame_id = cls.get_frame()

        if cls.inside_profile_compile_time:
            cls.ignored_profile_runs += 1
            logger.info(
                "profile_compile_time is requested for phase: %s, frame %s, while already in running phase: %s,"
                "frame %s, recursive call ignored",
                phase_name,
                frame_id,
                cls.current_phase,
                frame_id,
            )
            return skip()

        if cls.frame_id_filter is not None:
            should_run = re.match(cls.frame_id_filter, frame_id) is not None
            if not should_run:
                logger.info(
                    "profiling frame %s is skipped due to frame_id_filter %s",
                    frame_id,
                    cls.frame_id_filter,
                )
                return skip()

        cls.inside_profile_compile_time = True
        cls.current_phase = phase_name
        logger.info("profiling frame %s", frame_id)
        work_result = cls.profiler.profile(func, *args, **kwargs)

        if cls.profiler.profile_result is not None:
            cls.success_profile_count += 1
        else:
            cls.failed_profile_count += 1

        cls._log_stats()
        cls.inside_profile_compile_time = False
        return work_result

```



## High-Level Overview


This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StrobelightCompileTimeProfiler`

**Functions defined**: `get_fburl`, `get_strobelight_url`, `get_frame`, `enable`, `_cls_init`, `_log_stats`, `profile_compile_time`, `skip`

**Key imports**: json, logging, os, re, subprocess, datetime, gethostname, Any, Optional, StrobelightCLIFunctionProfiler, CompileContext


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_strobelight`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `logging`
- `os`
- `re`
- `subprocess`
- `datetime`: datetime
- `socket`: gethostname
- `typing`: Any, Optional
- `torch._strobelight.cli_function_profiler`: StrobelightCLIFunctionProfiler
- `torch._guards`: CompileContext
- `shutil`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_strobelight`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`cli_function_profiler.py_docs.md`](./cli_function_profiler.py_docs.md)


## Cross-References

- **File Documentation**: `compile_time_profiler.py_docs.md`
- **Keyword Index**: `compile_time_profiler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_strobelight`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_strobelight`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_strobelight`):

- [`cli_function_profiler.py_kw.md_docs.md`](./cli_function_profiler.py_kw.md_docs.md)
- [`cli_function_profiler.py_docs.md_docs.md`](./cli_function_profiler.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`compile_time_profiler.py_kw.md_docs.md`](./compile_time_profiler.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `compile_time_profiler.py_docs.md_docs.md`
- **Keyword Index**: `compile_time_profiler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
