# Documentation: `scripts/compile_tests/download_reports.py`

## File Metadata

- **Path**: `scripts/compile_tests/download_reports.py`
- **Size**: 4,204 bytes (4.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
import json
import os
import pprint
import re
import subprocess

import requests


CONFIGS = {
    "dynamo39": {
        "linux-jammy-py3.9-clang12 / test (dynamo_wrapped, 1, 3, linux.2xlarge)",
        "linux-jammy-py3.9-clang12 / test (dynamo_wrapped, 2, 3, linux.2xlarge)",
        "linux-jammy-py3.9-clang12 / test (dynamo_wrapped, 3, 3, linux.2xlarge)",
    },
    "dynamo313": {
        "linux-jammy-py3.13-clang12 / test (dynamo_wrapped, 1, 3, linux.2xlarge)",
        "linux-jammy-py3.13-clang12 / test (dynamo_wrapped, 2, 3, linux.2xlarge)",
        "linux-jammy-py3.13-clang12 / test (dynamo_wrapped, 3, 3, linux.2xlarge)",
    },
    "eager313": {
        "linux-jammy-py3.13-clang12 / test (default, 1, 5, linux.4xlarge)",
        "linux-jammy-py3.13-clang12 / test (default, 2, 5, linux.4xlarge)",
        "linux-jammy-py3.13-clang12 / test (default, 3, 5, linux.4xlarge)",
        "linux-jammy-py3.13-clang12 / test (default, 4, 5, linux.4xlarge)",
        "linux-jammy-py3.13-clang12 / test (default, 5, 5, linux.4xlarge)",
    },
}


def download_reports(commit_sha, configs=("dynamo39", "dynamo313", "eager313")):
    log_dir = "tmp_test_reports_" + commit_sha

    def subdir_path(config):
        return f"{log_dir}/{config}"

    for config in configs:
        assert config in CONFIGS.keys(), config
    subdir_paths = [subdir_path(config) for config in configs]

    # See which configs we haven't downloaded logs for yet
    missing_configs = []
    for config, path in zip(configs, subdir_paths):
        if os.path.exists(path):
            continue
        missing_configs.append(config)
    if len(missing_configs) == 0:
        print(
            f"All required logs appear to exist, not downloading again. Run `rm -rf {log_dir}` if this is not the case"
        )
        return subdir_paths

    output = subprocess.check_output(
        ["gh", "run", "list", "-c", commit_sha, "-w", "pull", "--json", "databaseId"]
    ).decode()
    workflow_run_id = str(json.loads(output)[0]["databaseId"])
    output = subprocess.check_output(["gh", "run", "view", workflow_run_id])
    workflow_jobs = parse_workflow_jobs(output)
    print("found the following workflow jobs:")
    pprint.pprint(workflow_jobs)

    # Figure out which jobs we need to download logs for
    required_jobs = []
    for config in configs:
        required_jobs.extend(list(CONFIGS[config]))
    for job in required_jobs:
        assert job in workflow_jobs, (
            f"{job} not found, is the commit_sha correct? has the job finished running? The GitHub API may take a couple minutes to update."
        )

    # This page lists all artifacts.
    listings = requests.get(
        f"https://hud.pytorch.org/api/artifacts/s3/{workflow_run_id}"
    ).json()

    def download_report(job_name, subdir):
        job_id = workflow_jobs[job_name]
        for listing in listings:
            name = listing["name"]
            if not name.startswith("test-reports-"):
                continue
            if name.endswith(f"_{job_id}.zip"):
                url = listing["url"]
                subprocess.run(["wget", "-P", subdir, url], check=True)
                path_to_zip = f"{subdir}/{name}"
                dir_name = path_to_zip[:-4]
                subprocess.run(["unzip", path_to_zip, "-d", dir_name], check=True)
                return
        raise AssertionError("should not be hit")

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for config in set(configs) - set(missing_configs):
        print(
            f"Logs for {config} already exist, not downloading again. Run `rm -rf {subdir_path(config)}` if this is not the case."
        )
    for config in missing_configs:
        subdir = subdir_path(config)
        os.mkdir(subdir)
        job_names = CONFIGS[config]
        for job_name in job_names:
            download_report(job_name, subdir)

    return subdir_paths


def parse_workflow_jobs(output):
    result = {}
    lines = output.decode().split("\n")
    for line in lines:
        match = re.search(r"(\S+ / .*) in .* \(ID (\d+)\)", line)
        if match is None:
            continue
        result[match.group(1)] = match.group(2)
    return result

```



## High-Level Overview


This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `download_reports`, `subdir_path`, `download_report`, `parse_workflow_jobs`

**Key imports**: json, os, pprint, re, subprocess, requests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `scripts/compile_tests`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `os`
- `pprint`
- `re`
- `subprocess`
- `requests`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python scripts/compile_tests/download_reports.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`scripts/compile_tests`):

- [`update_failures.py_docs.md`](./update_failures.py_docs.md)
- [`failures_histogram.py_docs.md`](./failures_histogram.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)
- [`passrate.py_docs.md`](./passrate.py_docs.md)


## Cross-References

- **File Documentation**: `download_reports.py_docs.md`
- **Keyword Index**: `download_reports.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
