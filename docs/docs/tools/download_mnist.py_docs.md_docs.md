# Documentation: `docs/tools/download_mnist.py_docs.md`

## File Metadata

- **Path**: `docs/tools/download_mnist.py_docs.md`
- **Size**: 5,302 bytes (5.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/download_mnist.py`

## File Metadata

- **Path**: `tools/download_mnist.py`
- **Size**: 2,807 bytes (2.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import gzip
import os
import sys
from urllib.error import URLError
from urllib.request import urlretrieve


MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",  # @lint-ignore
]

RESOURCES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def report_download_progress(
    chunk_number: int,
    chunk_size: int,
    file_size: int,
) -> None:
    if file_size != -1:
        # pyrefly: ignore [no-matching-overload]
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write(f"\r0% |{bar:<64}| {int(percent * 100)}%")


def download(destination_path: str, resource: str, quiet: bool) -> None:
    if os.path.exists(destination_path):
        if not quiet:
            print(f"{destination_path} already exists, skipping ...")
    else:
        for mirror in MIRRORS:
            url = mirror + resource
            print(f"Downloading {url} ...")
            try:
                hook = None if quiet else report_download_progress
                urlretrieve(url, destination_path, reporthook=hook)
            except (URLError, ConnectionError) as e:
                print(f"Failed to download (trying next):\n{e}")
                continue
            finally:
                if not quiet:
                    # Just a newline.
                    print()
            break
        else:
            raise RuntimeError("Error downloading resource!")


def unzip(zipped_path: str, quiet: bool) -> None:
    unzipped_path = os.path.splitext(zipped_path)[0]
    if os.path.exists(unzipped_path):
        if not quiet:
            print(f"{unzipped_path} already exists, skipping ... ")
        return
    with gzip.open(zipped_path, "rb") as zipped_file:
        with open(unzipped_path, "wb") as unzipped_file:
            unzipped_file.write(zipped_file.read())
            if not quiet:
                print(f"Unzipped {zipped_path} ...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the MNIST dataset from the internet"
    )
    parser.add_argument(
        "-d", "--destination", default=".", help="Destination directory"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Don't report about progress"
    )
    options = parser.parse_args()

    if not os.path.exists(options.destination):
        os.makedirs(options.destination)

    try:
        for resource in RESOURCES:
            path = os.path.join(options.destination, resource)
            download(path, resource, options.quiet)
            unzip(path, options.quiet)
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `report_download_progress`, `download`, `unzip`, `main`

**Key imports**: argparse, gzip, os, sys, URLError, urlretrieve


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `gzip`
- `os`
- `sys`
- `urllib.error`: URLError
- `urllib.request`: urlretrieve


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
- [`extract_scripts.py_docs.md`](./extract_scripts.py_docs.md)
- [`nvcc_fix_deps.py_docs.md`](./nvcc_fix_deps.py_docs.md)
- [`update_masked_docs.py_docs.md`](./update_masked_docs.py_docs.md)
- [`optional_submodules.py_docs.md`](./optional_submodules.py_docs.md)
- [`gen_vulkan_spv.py_docs.md`](./gen_vulkan_spv.py_docs.md)
- [`generated_dirs.txt_docs.md`](./generated_dirs.txt_docs.md)
- [`build_libtorch.py_docs.md`](./build_libtorch.py_docs.md)


## Cross-References

- **File Documentation**: `download_mnist.py_docs.md`
- **Keyword Index**: `download_mnist.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/tools`):

- [`git_add_generated_dirs.sh_docs.md_docs.md`](./git_add_generated_dirs.sh_docs.md_docs.md)
- [`update_masked_docs.py_docs.md_docs.md`](./update_masked_docs.py_docs.md_docs.md)
- [`bazel.bzl_docs.md_docs.md`](./bazel.bzl_docs.md_docs.md)
- [`nightly_hotpatch.py_docs.md_docs.md`](./nightly_hotpatch.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`build_with_debinfo.py_docs.md_docs.md`](./build_with_debinfo.py_docs.md_docs.md)
- [`extract_scripts.py_docs.md_docs.md`](./extract_scripts.py_docs.md_docs.md)
- [`bazel.bzl_kw.md_docs.md`](./bazel.bzl_kw.md_docs.md)
- [`build_with_debinfo.py_kw.md_docs.md`](./build_with_debinfo.py_kw.md_docs.md)
- [`gen_flatbuffers.sh_kw.md_docs.md`](./gen_flatbuffers.sh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `download_mnist.py_docs.md_docs.md`
- **Keyword Index**: `download_mnist.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
