# Documentation: `benchmarks/instruction_counts/applications/ci.py`

## File Metadata

- **Path**: `benchmarks/instruction_counts/applications/ci.py`
- **Size**: 2,475 bytes (2.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
"""Collect instruction counts for continuous integration."""

# mypy: ignore-errors

import argparse
import hashlib
import json
import time
from typing import Union

from core.expand import materialize
from definitions.standard import BENCHMARKS
from execution.runner import Runner
from execution.work import WorkOrder


REPEATS = 5
TIMEOUT = 600  # Seconds
RETRIES = 2

VERSION = 0
MD5 = "4d55e8abf881ad38bb617a96714c1296"


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--subset", action="store_true")
    args = parser.parse_args(argv)

    t0 = int(time.time())
    version = VERSION
    benchmarks = materialize(BENCHMARKS)

    # Useful for local development, since e2e time for the full suite is O(1 hour)
    in_debug_mode = args.subset or args.destination is None
    if args.subset:
        version = -1
        benchmarks = benchmarks[:10]

    work_orders = tuple(
        WorkOrder(label, autolabels, timer_args, timeout=TIMEOUT, retries=RETRIES)
        for label, autolabels, timer_args in benchmarks * REPEATS
    )

    keys = tuple({str(work_order): None for work_order in work_orders}.keys())
    md5 = hashlib.md5(usedforsecurity=False)
    for key in keys:
        md5.update(key.encode("utf-8"))

    # Warn early, since collection takes a long time.
    if md5.hexdigest() != MD5 and not args.subset:
        version = -1
        print(f"WARNING: Expected {MD5}, got {md5.hexdigest()} instead")

    results = Runner(work_orders, cadence=30.0).run()

    # TODO: Annotate with TypedDict when 3.8 is the minimum supported version.
    grouped_results: dict[str, dict[str, list[Union[float, int]]]] = {
        key: {"times": [], "counts": []} for key in keys
    }

    for work_order, r in results.items():
        key = str(work_order)
        grouped_results[key]["times"].extend(r.wall_times)
        grouped_results[key]["counts"].extend(r.instructions)

    final_results = {
        "version": version,
        "md5": md5.hexdigest(),
        "start_time": t0,
        "end_time": int(time.time()),
        "values": grouped_results,
    }

    if args.destination:
        with open(args.destination, "w") as f:
            json.dump(final_results, f)

    if in_debug_mode:
        result_str = json.dumps(final_results)
        print(f"{result_str[:30]} ... {result_str[-30:]}\n")
        import pdb

        pdb.set_trace()

```



## High-Level Overview

"""Collect instruction counts for continuous integration."""# mypy: ignore-errorsimport argparseimport hashlibimport jsonimport timefrom typing import Unionfrom core.expand import materializefrom definitions.standard import BENCHMARKSfrom execution.runner import Runnerfrom execution.work import WorkOrderREPEATS = 5TIMEOUT = 600  # SecondsRETRIES = 2VERSION = 0MD5 = "4d55e8abf881ad38bb617a96714c1296"def main(argv: list[str]) -> None:    parser = argparse.ArgumentParser()    parser.add_argument("--destination", type=str, default=None)    parser.add_argument("--subset", action="store_true")    args = parser.parse_args(argv)    t0 = int(time.time())    version = VERSION    benchmarks = materialize(BENCHMARKS)    # Useful for local development, since e2e time for the full suite is O(1 hour)    in_debug_mode = args.subset or args.destination is None    if args.subset:        version = -1        benchmarks = benchmarks[:10]    work_orders = tuple(        WorkOrder(label, autolabels, timer_args, timeout=TIMEOUT, retries=RETRIES)        for label, autolabels, timer_args in benchmarks * REPEATS    )    keys = tuple({str(work_order): None for work_order in work_orders}.keys())    md5 = hashlib.md5(usedforsecurity=False)    for key in keys:        md5.update(key.encode("utf-8"))

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `main`

**Key imports**: argparse, hashlib, json, time, Union, materialize, BENCHMARKS, Runner, WorkOrder, pdb


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/instruction_counts/applications`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `hashlib`
- `json`
- `time`
- `typing`: Union
- `core.expand`: materialize
- `definitions.standard`: BENCHMARKS
- `execution.runner`: Runner
- `execution.work`: WorkOrder
- `pdb`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/instruction_counts/applications`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `ci.py_docs.md`
- **Keyword Index**: `ci.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
