# Documentation: `docs/tools/testing/explicit_ci_jobs.py_docs.md`

## File Metadata

- **Path**: `docs/tools/testing/explicit_ci_jobs.py_docs.md`
- **Size**: 7,780 bytes (7.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/testing/explicit_ci_jobs.py`

## File Metadata

- **Path**: `tools/testing/explicit_ci_jobs.py`
- **Size**: 4,989 bytes (4.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).parents[2]
CONFIG_YML = REPO_ROOT / ".circleci" / "config.yml"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


WORKFLOWS_TO_CHECK = [
    "binary_builds",
    "build",
    "master_build",
    # These are formatted slightly differently, skip them
    # "scheduled-ci",
    # "debuggable-scheduled-ci",
    # "slow-gradcheck-scheduled-ci",
    # "promote",
]


def add_job(
    workflows: dict[str, Any],
    workflow_name: str,
    type: str,
    job: dict[str, Any],
    past_jobs: dict[str, Any],
) -> None:
    """
    Add job 'job' under 'type' and 'workflow_name' to 'workflow' in place. Also
    add any dependencies (they must already be in 'past_jobs')
    """
    if workflow_name not in workflows:
        workflows[workflow_name] = {"when": "always", "jobs": []}

    requires = job.get("requires")
    if requires is not None:
        for requirement in requires:
            dependency = past_jobs[requirement]
            add_job(
                workflows,
                dependency["workflow_name"],
                dependency["type"],
                dependency["job"],
                past_jobs,
            )

    workflows[workflow_name]["jobs"].append({type: job})


def get_filtered_circleci_config(
    workflows: dict[str, Any], relevant_jobs: list[str]
) -> dict[str, Any]:
    """
    Given an existing CircleCI config, remove every job that's not listed in
    'relevant_jobs'
    """
    new_workflows: dict[str, Any] = {}
    past_jobs: dict[str, Any] = {}
    for workflow_name, workflow in workflows.items():
        if workflow_name not in WORKFLOWS_TO_CHECK:
            # Don't care about this workflow, skip it entirely
            continue

        for job_dict in workflow["jobs"]:
            for type, job in job_dict.items():
                if "name" not in job:
                    # Job doesn't have a name so it can't be handled
                    print("Skipping", type)
                else:
                    if job["name"] in relevant_jobs:
                        # Found a job that was specified at the CLI, add it to
                        # the new result
                        add_job(new_workflows, workflow_name, type, job, past_jobs)

                    # Record the job in case it's needed as a dependency later
                    past_jobs[job["name"]] = {
                        "workflow_name": workflow_name,
                        "type": type,
                        "job": job,
                    }

    return new_workflows


def commit_ci(files: list[str], message: str) -> None:
    # Check that there are no other modified files than the ones edited by this
    # tool
    stdout = subprocess.run(
        ["git", "status", "--porcelain"], stdout=subprocess.PIPE
    ).stdout.decode()
    for line in stdout.split("\n"):
        if line == "":
            continue
        if line[0] != " ":
            raise RuntimeError(
                f"Refusing to commit while other changes are already staged: {line}"
            )

    # Make the commit
    subprocess.run(["git", "add"] + files)
    subprocess.run(["git", "commit", "-m", message])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="make .circleci/config.yml only have a specific set of jobs and delete GitHub actions"
    )
    parser.add_argument("--job", action="append", help="job name", default=[])
    parser.add_argument(
        "--filter-gha", help="keep only these github actions (glob match)", default=""
    )
    parser.add_argument(
        "--make-commit",
        action="store_true",
        help="add change to git with to a do-not-merge commit",
    )
    args = parser.parse_args()

    touched_files = [CONFIG_YML]
    with open(CONFIG_YML) as f:
        config_yml = yaml.safe_load(f.read())

    config_yml["workflows"] = get_filtered_circleci_config(
        config_yml["workflows"], args.job
    )

    with open(CONFIG_YML, "w") as f:
        yaml.dump(config_yml, f)

    if args.filter_gha:
        for relative_file in WORKFLOWS_DIR.iterdir():
            path = REPO_ROOT.joinpath(relative_file)
            if not fnmatch.fnmatch(path.name, args.filter_gha):
                touched_files.append(path)
                path.resolve().unlink()

    if args.make_commit:
        jobs_str = "\n".join([f" * {job}" for job in args.job])
        message = textwrap.dedent(
            f"""
        [skip ci][do not merge] Edit config.yml to filter specific jobs

        Filter CircleCI to only run:
        {jobs_str}

        See [Run Specific CI Jobs](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#run-specific-ci-jobs) for details.
        """
        ).strip()
        commit_ci([str(f.relative_to(REPO_ROOT)) for f in touched_files], message)

```



## High-Level Overview

"""    Add job 'job' under 'type' and 'workflow_name' to 'workflow' in place. Also    add any dependencies (they must already be in 'past_jobs')

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `add_job`, `get_filtered_circleci_config`, `commit_ci`

**Key imports**: annotations, argparse, fnmatch, subprocess, textwrap, Path, Any, yaml


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/testing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `fnmatch`
- `subprocess`
- `textwrap`
- `pathlib`: Path
- `typing`: Any
- `yaml`


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
python tools/testing/explicit_ci_jobs.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/testing`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`test_selections.py_docs.md`](./test_selections.py_docs.md)
- [`clickhouse.py_docs.md`](./clickhouse.py_docs.md)
- [`update_slow_tests.py_docs.md`](./update_slow_tests.py_docs.md)
- [`upload_artifacts.py_docs.md`](./upload_artifacts.py_docs.md)
- [`do_target_determination_for_s3.py_docs.md`](./do_target_determination_for_s3.py_docs.md)
- [`modulefinder_determinator.py_docs.md`](./modulefinder_determinator.py_docs.md)
- [`discover_tests.py_docs.md`](./discover_tests.py_docs.md)


## Cross-References

- **File Documentation**: `explicit_ci_jobs.py_docs.md`
- **Keyword Index**: `explicit_ci_jobs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/testing`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/testing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python docs/tools/testing/explicit_ci_jobs.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/testing`):

- [`test_run.py_kw.md_docs.md`](./test_run.py_kw.md_docs.md)
- [`discover_tests.py_docs.md_docs.md`](./discover_tests.py_docs.md_docs.md)
- [`upload_artifacts.py_docs.md_docs.md`](./upload_artifacts.py_docs.md_docs.md)
- [`test_selections.py_kw.md_docs.md`](./test_selections.py_kw.md_docs.md)
- [`modulefinder_determinator.py_docs.md_docs.md`](./modulefinder_determinator.py_docs.md_docs.md)
- [`explicit_ci_jobs.py_kw.md_docs.md`](./explicit_ci_jobs.py_kw.md_docs.md)
- [`test_selections.py_docs.md_docs.md`](./test_selections.py_docs.md_docs.md)
- [`clickhouse.py_kw.md_docs.md`](./clickhouse.py_kw.md_docs.md)
- [`discover_tests.py_kw.md_docs.md`](./discover_tests.py_kw.md_docs.md)
- [`modulefinder_determinator.py_kw.md_docs.md`](./modulefinder_determinator.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `explicit_ci_jobs.py_docs.md_docs.md`
- **Keyword Index**: `explicit_ci_jobs.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
