# Documentation: `docs/torch/distributed/flight_recorder/components/loader.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/flight_recorder/components/loader.py_docs.md`
- **Size**: 5,381 bytes (5.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/flight_recorder/components/loader.py`

## File Metadata

- **Path**: `torch/distributed/flight_recorder/components/loader.py`
- **Size**: 2,908 bytes (2.84 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import os
import pickle
import re
import time
from collections import defaultdict
from typing import Any, Union

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger


__all__ = [
    "read_dump",
    "read_dir",
]


logger: FlightRecorderLogger = FlightRecorderLogger()


def read_dump(prefix: str, filename: str) -> dict[str, Union[str, int, list[Any]]]:
    basename = os.path.basename(filename)

    rank = int(basename[len(prefix) :])
    host_name = f"host_rank{rank}"

    with open(filename, "rb") as infile:
        dump = pickle.load(infile)

    entries = dump["entries"]
    version = dump["version"]
    pg_config = dump["pg_config"]

    return {
        "host_name": host_name,
        "rank": rank,
        "entries": entries,
        "version": version,
        "pg_config": pg_config,
    }


exp = re.compile(r"([\w\-\_]*?)(\d+)$")


def _determine_prefix(files: list[str]) -> str:
    """If the user doesn't specify a prefix, but does pass a dir full of similarly-prefixed files, we should be able to
    infer the common prefix most of the time.  But if we can't confidently infer, just fall back to requiring the user
    to specify it
    """
    possible_prefixes: defaultdict[str, set[int]] = defaultdict(set)
    for f in files:
        m = exp.search(f)
        if m:
            p, r = m.groups()
            possible_prefixes[p].add(int(r))
    if len(possible_prefixes) == 1:
        prefix = next(iter(possible_prefixes))
        logger.debug("Inferred common prefix %s", prefix)
        return prefix
    else:
        raise ValueError(
            "Unable to automatically determine the common prefix for the trace file names. "
            "Please specify --prefix argument manually"
        )


def read_dir(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str]:
    gc.disable()
    prefix = args.prefix
    details = {}
    t0 = time.time()
    version = ""
    filecount = 0
    assert os.path.isdir(args.trace_dir), f"folder {args.trace_dir} does not exist"
    for root, _, files in os.walk(args.trace_dir):
        if prefix is None:
            prefix = _determine_prefix(files)
        for f in files:
            if (offset := f.find(prefix)) == -1:
                continue
            details[f] = read_dump(f[:offset] + prefix, os.path.join(root, f))
            filecount += 1
            if not version:
                version = str(details[f]["version"])
    tb = time.time()
    assert len(details) > 0, (
        f"no files loaded from {args.trace_dir} with prefix {prefix}"
    )
    logger.debug("loaded %s files in %ss", filecount, tb - t0)
    return details, version

```



## High-Level Overview


This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `read_dump`, `_determine_prefix`, `read_dir`

**Key imports**: argparse, gc, os, pickle, re, time, defaultdict, Any, Union, FlightRecorderLogger


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/flight_recorder/components`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `gc`
- `os`
- `pickle`
- `re`
- `time`
- `collections`: defaultdict
- `typing`: Any, Union
- `torch.distributed.flight_recorder.components.fr_logger`: FlightRecorderLogger


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/flight_recorder/components`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`config_manager.py_docs.md`](./config_manager.py_docs.md)
- [`fr_logger.py_docs.md`](./fr_logger.py_docs.md)
- [`builder.py_docs.md`](./builder.py_docs.md)


## Cross-References

- **File Documentation**: `loader.py_docs.md`
- **Keyword Index**: `loader.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/flight_recorder/components`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/flight_recorder/components`, which is part of the **core PyTorch library**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/flight_recorder/components`):

- [`types.py_kw.md_docs.md`](./types.py_kw.md_docs.md)
- [`config_manager.py_kw.md_docs.md`](./config_manager.py_kw.md_docs.md)
- [`config_manager.py_docs.md_docs.md`](./config_manager.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`fr_logger.py_kw.md_docs.md`](./fr_logger.py_kw.md_docs.md)
- [`builder.py_docs.md_docs.md`](./builder.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`loader.py_kw.md_docs.md`](./loader.py_kw.md_docs.md)
- [`fr_logger.py_docs.md_docs.md`](./fr_logger.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `loader.py_docs.md_docs.md`
- **Keyword Index**: `loader.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
