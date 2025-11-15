# Documentation: `tools/extract_scripts.py`

## File Metadata

- **Path**: `tools/extract_scripts.py`
- **Size**: 3,154 bytes (3.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any
from typing_extensions import TypedDict  # Python 3.11+

import yaml


Step = dict[str, Any]


class Script(TypedDict):
    extension: str
    script: str


def extract(step: Step) -> Script | None:
    run = step.get("run")

    # https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#using-a-specific-shell
    shell = step.get("shell", "bash")
    extension = {
        "bash": ".sh",
        "pwsh": ".ps1",
        "python": ".py",
        "sh": ".sh",
        "cmd": ".cmd",
        "powershell": ".ps1",
    }.get(shell)

    is_gh_script = step.get("uses", "").startswith("actions/github-script@")
    gh_script = step.get("with", {}).get("script")

    if run is not None and extension is not None:
        script = {
            "bash": f"#!/usr/bin/env bash\nset -eo pipefail\n{run}",
            "sh": f"#!/usr/bin/env sh\nset -e\n{run}",
        }.get(shell, run)
        return {"extension": extension, "script": script}  # type: ignore[typeddict-item]
    elif is_gh_script and gh_script is not None:
        return {"extension": ".js", "script": gh_script}
    else:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} already exists; aborting to avoid overwriting")

    gha_expressions_found = False

    for p in Path(".github/workflows").iterdir():
        with open(p, "rb") as f:
            workflow = yaml.safe_load(f)

        for job_name, job in workflow["jobs"].items():
            job_dir = out / p / job_name
            if "steps" not in job:
                continue
            steps = job["steps"]
            index_chars = len(str(len(steps) - 1))
            for i, step in enumerate(steps, start=1):
                extracted = extract(step)
                if extracted:
                    script = extracted["script"]
                    step_name = step.get("name", "")
                    if "${{" in script:
                        gha_expressions_found = True
                        print(
                            f"{p} job `{job_name}` step {i}: {step_name}",
                            file=sys.stderr,
                        )

                    job_dir.mkdir(parents=True, exist_ok=True)

                    sanitized = re.sub(
                        "[^a-zA-Z_]+",
                        "_",
                        f"_{step_name}",
                    ).rstrip("_")
                    extension = extracted["extension"]
                    filename = f"{i:0{index_chars}}{sanitized}{extension}"
                    (job_dir / filename).write_text(script)

    if gha_expressions_found:
        sys.exit(
            "Each of the above scripts contains a GitHub Actions "
            "${{ <expression> }} which must be replaced with an `env` variable"
            " for security reasons."
        )


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Script`

**Functions defined**: `extract`, `main`

**Key imports**: annotations, argparse, re, sys, Path, Any, TypedDict  , yaml


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `re`
- `sys`
- `pathlib`: Path
- `typing`: Any
- `typing_extensions`: TypedDict  
- `yaml`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`tools`):

- [`BUCK.bzl_docs.md`](./BUCK.bzl_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`render_junit.py_docs.md`](./render_junit.py_docs.md)
- [`nvcc_fix_deps.py_docs.md`](./nvcc_fix_deps.py_docs.md)
- [`update_masked_docs.py_docs.md`](./update_masked_docs.py_docs.md)
- [`optional_submodules.py_docs.md`](./optional_submodules.py_docs.md)
- [`gen_vulkan_spv.py_docs.md`](./gen_vulkan_spv.py_docs.md)
- [`generated_dirs.txt_docs.md`](./generated_dirs.txt_docs.md)
- [`build_libtorch.py_docs.md`](./build_libtorch.py_docs.md)


## Cross-References

- **File Documentation**: `extract_scripts.py_docs.md`
- **Keyword Index**: `extract_scripts.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
