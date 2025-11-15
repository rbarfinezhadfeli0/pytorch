# Documentation: `.github/scripts/build_triton_wheel.py`

## File Metadata

- **Path**: `.github/scripts/build_triton_wheel.py`
- **Size**: 5,152 bytes (5.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

import os
import shutil
import sys
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Optional


SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent.parent


def read_triton_pin(device: str = "cuda") -> str:
    triton_file = "triton.txt"
    if device == "xpu":
        triton_file = "triton-xpu.txt"
    with open(REPO_DIR / ".ci" / "docker" / "ci_commit_pins" / triton_file) as f:
        return f.read().strip()


def read_triton_version(device: str = "cuda") -> str:
    triton_version_file = "triton_version.txt"
    if device == "xpu":
        triton_version_file = "triton_xpu_version.txt"
    with open(REPO_DIR / ".ci" / "docker" / triton_version_file) as f:
        return f.read().strip()


def check_and_replace(inp: str, src: str, dst: str) -> str:
    """Checks that `src` can be found in `input` and replaces it with `dst`"""
    if src not in inp:
        raise RuntimeError(f"Can't find ${src} in the input")
    return inp.replace(src, dst)


def patch_init_py(
    path: Path, *, version: str, expected_version: Optional[str] = None
) -> None:
    if not expected_version:
        expected_version = read_triton_version()
    with open(path) as f:
        orig = f.read()
    # Replace version
    orig = check_and_replace(
        orig, f"__version__ = '{expected_version}'", f'__version__ = "{version}"'
    )
    with open(path, "w") as f:
        f.write(orig)


def build_triton(
    *,
    version: str,
    commit_hash: str,
    device: str = "cuda",
    py_version: Optional[str] = None,
    release: bool = False,
    with_clang_ldd: bool = False,
) -> Path:
    env = os.environ.copy()
    if "MAX_JOBS" not in env:
        max_jobs = os.cpu_count() or 1
        env["MAX_JOBS"] = str(max_jobs)

    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"

        triton_repo = "https://github.com/openai/triton"
        if device == "rocm":
            triton_pkg_name = "pytorch-triton-rocm"
        elif device == "xpu":
            triton_pkg_name = "pytorch-triton-xpu"
            triton_repo = "https://github.com/intel/intel-xpu-backend-for-triton"
        else:
            triton_pkg_name = "pytorch-triton"
        check_call(["git", "clone", triton_repo, "triton"], cwd=tmpdir)
        if release:
            ver, rev, patch = version.split(".")
            check_call(
                ["git", "checkout", f"release/{ver}.{rev}.x"], cwd=triton_basedir
            )
        else:
            check_call(["git", "fetch", "origin", commit_hash], cwd=triton_basedir)
            check_call(["git", "checkout", commit_hash], cwd=triton_basedir)

        # change built wheel name and version
        env["TRITON_WHEEL_NAME"] = triton_pkg_name
        if with_clang_ldd:
            env["TRITON_BUILD_WITH_CLANG_LLD"] = "1"

        patch_init_py(
            triton_pythondir / "triton" / "__init__.py",
            version=f"{version}",
            expected_version=read_triton_version(device),
        )

        if device == "rocm":
            check_call(
                [f"{SCRIPT_DIR}/amd/package_triton_wheel.sh"],
                cwd=triton_basedir,
                shell=True,
            )
            print("ROCm libraries setup for triton installation...")

        # old triton versions have setup.py in the python/ dir,
        # new versions have it in the root dir.
        triton_setupdir = (
            triton_basedir
            if (triton_basedir / "setup.py").exists()
            else triton_pythondir
        )

        check_call(
            [sys.executable, "setup.py", "bdist_wheel"], cwd=triton_setupdir, env=env
        )

        whl_path = next(iter((triton_setupdir / "dist").glob("*.whl")))
        shutil.copy(whl_path, Path.cwd())

        if device == "rocm":
            check_call(
                [f"{SCRIPT_DIR}/amd/patch_triton_wheel.sh", Path.cwd()],
                cwd=triton_basedir,
            )

        return Path.cwd() / whl_path.name


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser("Build Triton binaries")
    parser.add_argument("--release", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "rocm", "xpu", "aarch64"]
    )
    parser.add_argument("--py-version", type=str)
    parser.add_argument("--commit-hash", type=str)
    parser.add_argument("--with-clang-ldd", action="store_true")
    parser.add_argument("--triton-version", type=str, default=None)
    args = parser.parse_args()

    triton_version = read_triton_version(args.device)
    if args.triton_version:
        triton_version = args.triton_version

    build_triton(
        device=args.device,
        commit_hash=(
            args.commit_hash if args.commit_hash else read_triton_pin(args.device)
        ),
        version=triton_version,
        py_version=args.py_version,
        release=args.release,
        with_clang_ldd=args.with_clang_ldd,
    )


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Checks that `src` can be found in `input` and replaces it with `dst`"""    if src not in inp:        raise RuntimeError(f"Can't find ${src} in the input")    return inp.replace(src, dst)def patch_init_py(    path: Path, *, version: str, expected_version: Optional[str] = None) -> None:    if not expected_version:        expected_version = read_triton_version()    with open(path) as f:        orig = f.read()    # Replace version    orig = check_and_replace(        orig, f"__version__ = '{expected_version}'", f'__version__ = "{version}"'    )    with open(path, "w") as f:

This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `read_triton_pin`, `read_triton_version`, `check_and_replace`, `patch_init_py`, `build_triton`, `main`

**Key imports**: os, shutil, sys, Path, check_call, TemporaryDirectory, Optional, ArgumentParser


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `shutil`
- `sys`
- `pathlib`: Path
- `subprocess`: check_call
- `tempfile`: TemporaryDirectory
- `typing`: Optional
- `argparse`: ArgumentParser


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`.github/scripts`):

- [`convert_lintrunner_annotations_to_github.py_docs.md`](./convert_lintrunner_annotations_to_github.py_docs.md)
- [`gitutils.py_docs.md`](./gitutils.py_docs.md)
- [`collect_ciflow_labels.py_docs.md`](./collect_ciflow_labels.py_docs.md)
- [`generate_docker_release_matrix.py_docs.md`](./generate_docker_release_matrix.py_docs.md)
- [`github_utils.py_docs.md`](./github_utils.py_docs.md)
- [`filter_test_configs.py_docs.md`](./filter_test_configs.py_docs.md)
- [`test_runner_determinator.py_docs.md`](./test_runner_determinator.py_docs.md)
- [`trymerge.py_docs.md`](./trymerge.py_docs.md)
- [`comment_on_pr.py_docs.md`](./comment_on_pr.py_docs.md)
- [`generate_binary_build_matrix.py_docs.md`](./generate_binary_build_matrix.py_docs.md)


## Cross-References

- **File Documentation**: `build_triton_wheel.py_docs.md`
- **Keyword Index**: `build_triton_wheel.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
