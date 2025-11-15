# Documentation: `benchmarks/upload_scribe.py`

## File Metadata

- **Path**: `benchmarks/upload_scribe.py`
- **Size**: 5,844 bytes (5.71 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
"""Scribe Uploader for Pytorch Benchmark Data

Currently supports data in pytest-benchmark format but can be extended.

New fields can be added just by modifying the schema in this file, schema
checking is only here to encourage reusing existing fields and avoiding typos.
"""

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

import requests


class ScribeUploader:
    def __init__(self, category):
        self.category = category

    def format_message(self, field_dict):
        assert "time" in field_dict, "Missing required Scribe field 'time'"
        message = defaultdict(dict)
        for field, value in field_dict.items():
            if field in self.schema["normal"]:
                message["normal"][field] = str(value)
            elif field in self.schema["int"]:
                message["int"][field] = int(value)
            elif field in self.schema["float"]:
                message["float"][field] = float(value)
            else:
                raise ValueError(
                    f"Field {field} is not currently used, be intentional about adding new fields"
                )
        return message

    def _upload_intern(self, messages):
        for m in messages:
            json_str = json.dumps(m)
            cmd = ["scribe_cat", self.category, json_str]
            subprocess.run(cmd)

    def upload(self, messages):
        if os.environ.get("SCRIBE_INTERN"):
            return self._upload_intern(messages)
        access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("Can't find access token from environment variable")
        url = "https://graph.facebook.com/scribe_logs"  # @lint-ignore
        r = requests.post(
            url,
            data={
                "access_token": access_token,
                "logs": json.dumps(
                    [
                        {
                            "category": self.category,
                            "message": json.dumps(message),
                            "line_escape": False,
                        }
                        for message in messages
                    ]
                ),
            },
        )
        print(r.text)
        r.raise_for_status()


class PytorchBenchmarkUploader(ScribeUploader):
    def __init__(self):
        super().__init__("perfpipe_pytorch_benchmarks")
        self.schema = {
            "int": [
                "time",
                "rounds",
            ],
            "normal": [
                "benchmark_group",
                "benchmark_name",
                "benchmark_executor",
                "benchmark_fuser",
                "benchmark_class",
                "benchmark_time",
                "pytorch_commit_id",
                "pytorch_branch",
                "pytorch_commit_time",
                "pytorch_version",
                "pytorch_git_dirty",
                "machine_kernel",
                "machine_processor",
                "machine_hostname",
                "circle_build_num",
                "circle_project_reponame",
            ],
            "float": [
                "stddev",
                "min",
                "median",
                "max",
                "mean",
            ],
        }

    def post_pytest_benchmarks(self, pytest_json):
        machine_info = pytest_json["machine_info"]
        commit_info = pytest_json["commit_info"]
        upload_time = int(time.time())
        messages = []
        for b in pytest_json["benchmarks"]:
            test = b["name"].split("[")[0]
            net_name = b["params"]["net_name"]
            benchmark_name = f"{test}[{net_name}]"
            executor = b["params"]["executor"]
            fuser = b["params"]["fuser"]
            m = self.format_message(
                {
                    "time": upload_time,
                    "benchmark_group": b["group"],
                    "benchmark_name": benchmark_name,
                    "benchmark_executor": executor,
                    "benchmark_fuser": fuser,
                    "benchmark_class": b["fullname"],
                    "benchmark_time": pytest_json["datetime"],
                    "pytorch_commit_id": commit_info["id"],
                    "pytorch_branch": commit_info["branch"],
                    "pytorch_commit_time": commit_info["time"],
                    "pytorch_version": None,
                    "pytorch_git_dirty": commit_info["dirty"],
                    "machine_kernel": machine_info["release"],
                    "machine_processor": machine_info["processor"],
                    "machine_hostname": machine_info["node"],
                    "circle_build_num": os.environ.get("CIRCLE_BUILD_NUM"),
                    "circle_project_reponame": os.environ.get(
                        "CIRCLE_PROJECT_REPONAME"
                    ),
                    "stddev": b["stats"]["stddev"],
                    "rounds": b["stats"]["rounds"],
                    "min": b["stats"]["min"],
                    "median": b["stats"]["median"],
                    "max": b["stats"]["max"],
                    "mean": b["stats"]["mean"],
                }
            )
            messages.append(m)
        self.upload(messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pytest-bench-json",
        "--pytest_bench_json",
        type=argparse.FileType("r"),
        help="Upload json data formatted by pytest-benchmark module",
    )
    args = parser.parse_args()
    if args.pytest_bench_json:
        benchmark_uploader = PytorchBenchmarkUploader()
        json_data = json.load(args.pytest_bench_json)
        benchmark_uploader.post_pytest_benchmarks(json_data)

```



## High-Level Overview

"""Scribe Uploader for Pytorch Benchmark DataCurrently supports data in pytest-benchmark format but can be extended.New fields can be added just by modifying the schema in this file, schemachecking is only here to encourage reusing existing fields and avoiding typos.

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ScribeUploader`, `PytorchBenchmarkUploader`

**Functions defined**: `__init__`, `format_message`, `_upload_intern`, `upload`, `__init__`, `post_pytest_benchmarks`

**Key imports**: argparse, json, os, subprocess, time, defaultdict, requests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `json`
- `os`
- `subprocess`
- `time`
- `collections`: defaultdict
- `requests`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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

Files in the same folder (`benchmarks`):

- [`compare-fastrnn-results.py_docs.md`](./compare-fastrnn-results.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`compare.sh_docs.md`](./compare.sh_docs.md)


## Cross-References

- **File Documentation**: `upload_scribe.py_docs.md`
- **Keyword Index**: `upload_scribe.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
