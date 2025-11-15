# Documentation: `torch/distributed/flight_recorder/components/config_manager.py`

## File Metadata

- **Path**: `torch/distributed/flight_recorder/components/config_manager.py`
- **Size**: 3,951 bytes (3.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file handles **configuration or setup**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from collections.abc import Sequence
from typing import Optional

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger


__all__ = ["JobConfig"]


logger: FlightRecorderLogger = FlightRecorderLogger()


class JobConfig:
    """
    A helper class to manage the script configuration.
    """

    def __init__(self: "JobConfig"):
        self.parser = argparse.ArgumentParser(
            description="PyTorch Flight recorder analyzing script."
        )
        self.parser.add_argument(
            "trace_dir",
            nargs="?",
            help="Directory containing one trace file per rank, named with <prefix>_<rank>.",
        )
        self.parser.add_argument(
            "--selected-ranks",
            default=None,
            nargs="+",
            type=int,
            help="List of ranks we want to show traces for.",
        )
        self.parser.add_argument(
            "--allow-incomplete-ranks",
            action="store_true",
            help=(
                "FR trace require all ranks to have dumps for analysis. "
                "This flag allows best-effort partial analysis of results "
                "and printing of collected data."
            ),
        )
        self.parser.add_argument(
            "--pg-filters",
            default=None,
            nargs="+",
            type=str,
            help=(
                "List of filter strings, it could be pg name or pg desc. "
                "If specified, only show traces for the given pg."
            ),
        )
        self.parser.add_argument("-o", "--output", default=None)
        self.parser.add_argument(
            "-p",
            "--prefix",
            help=(
                "Common filename prefix to strip such that rank can be extracted. "
                "If not specified, will attempt to infer a common prefix."
            ),
            default=None,
        )
        self.parser.add_argument("-j", "--just_print_entries", action="store_true")
        self.parser.add_argument("-v", "--verbose", action="store_true")
        self.parser.add_argument("--print_stack_trace", action="store_true")
        self.parser.add_argument(
            "--mismatch_cap",
            type=int,
            default=10,
            help="Maximum number of mismatches we print (from earliest).",
        )
        self.parser.add_argument(
            "--transform-ft",
            action="store_true",
            help="Transform PG config to use global ranks to analyze traces produced by torchft",
        )
        self.parser.add_argument(
            "--group-world-size",
            type=int,
            default=None,
            help="The number of ranks in 1 torchft replica group. Must be specified if --transform-ft is True",
        )

    def parse_args(
        self: "JobConfig", args: Optional[Sequence[str]]
    ) -> argparse.Namespace:
        # pyrefly: ignore [bad-assignment]
        args = self.parser.parse_args(args)
        # pyrefly: ignore [missing-attribute]
        if args.selected_ranks is not None:
            # pyrefly: ignore [missing-attribute]
            assert args.just_print_entries, (
                "Not support selecting ranks without printing entries"
            )
        # pyrefly: ignore [missing-attribute]
        if args.pg_filters is not None:
            # pyrefly: ignore [missing-attribute]
            assert args.just_print_entries, (
                "Not support selecting pg filters without printing entries"
            )
        # pyrefly: ignore [missing-attribute]
        if args.verbose:
            logger.set_log_level(logging.DEBUG)
        # pyrefly: ignore [bad-return]
        return args

```



## High-Level Overview

"""    A helper class to manage the script configuration.

This Python file contains 2 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `JobConfig`

**Functions defined**: `__init__`, `parse_args`

**Key imports**: argparse, logging, Sequence, Optional, FlightRecorderLogger


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/flight_recorder/components`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `logging`
- `collections.abc`: Sequence
- `typing`: Optional
- `torch.distributed.flight_recorder.components.fr_logger`: FlightRecorderLogger


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

Files in the same folder (`torch/distributed/flight_recorder/components`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`loader.py_docs.md`](./loader.py_docs.md)
- [`fr_logger.py_docs.md`](./fr_logger.py_docs.md)
- [`builder.py_docs.md`](./builder.py_docs.md)


## Cross-References

- **File Documentation**: `config_manager.py_docs.md`
- **Keyword Index**: `config_manager.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
