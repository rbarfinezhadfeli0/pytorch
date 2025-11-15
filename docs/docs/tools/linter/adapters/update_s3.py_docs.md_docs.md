# Documentation: `docs/tools/linter/adapters/update_s3.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/update_s3.py_docs.md`
- **Size**: 5,490 bytes (5.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/update_s3.py`

## File Metadata

- **Path**: `tools/linter/adapters/update_s3.py`
- **Size**: 2,700 bytes (2.64 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""Uploads a new binary to s3 and updates its hash in the config file.

You'll need to have appropriate credentials on the PyTorch AWS buckets, see:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
for how to configure them.
"""

import argparse
import hashlib
import json
import logging
import os

import boto3  # type: ignore[import]


def compute_file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file and return it as a hex string."""
    # If the file doesn't exist, return an empty string.
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha256()

    # Open the file in binary mode and hash it.
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)

    # Return the hash as a hexadecimal string.
    return hash.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="s3 binary updater",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config-json",
        required=True,
        help="path to config json that you are trying to update",
    )
    parser.add_argument(
        "--linter",
        required=True,
        help="name of linter you're trying to update",
    )
    parser.add_argument(
        "--platform",
        required=True,
        help="which platform you are uploading the binary for",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="file to upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="if set, don't actually upload/write hash",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open(args.config_json) as f:
        config = json.load(f)
    linter_config = config[args.linter][args.platform]
    bucket = linter_config["s3_bucket"]
    object_name = linter_config["object_name"]

    # Upload the file
    logging.info(
        "Uploading file %s to s3 bucket: %s, object name: %s",
        args.file,
        bucket,
        object_name,
    )
    if not args.dry_run:
        s3_client = boto3.client("s3")
        s3_client.upload_file(args.file, bucket, object_name)

    # Update hash in repo
    hash_of_new_binary = compute_file_sha256(args.file)
    logging.info("Computed new hash for binary %s", hash_of_new_binary)

    linter_config["hash"] = hash_of_new_binary
    config_dump = json.dumps(config, indent=4, sort_keys=True)

    logging.info("Writing out new config:")
    logging.info(config_dump)
    if not args.dry_run:
        with open(args.config_json, "w") as f:
            f.write(config_dump)


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Uploads a new binary to s3 and updates its hash in the config file.You'll need to have appropriate credentials on the PyTorch AWS buckets, see:https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configurationfor how to configure them.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `compute_file_sha256`, `main`

**Key imports**: argparse, hashlib, json, logging, os, boto3  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `hashlib`
- `json`
- `logging`
- `os`
- `boto3  `


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

Files in the same folder (`tools/linter/adapters`):

- [`grep_linter.py_docs.md`](./grep_linter.py_docs.md)
- [`import_linter.py_docs.md`](./import_linter.py_docs.md)
- [`gha_linter.py_docs.md`](./gha_linter.py_docs.md)
- [`actionlint_linter.py_docs.md`](./actionlint_linter.py_docs.md)
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `update_s3.py_docs.md`
- **Keyword Index**: `update_s3.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/linter/adapters`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/linter/adapters`, which contains **development tools and scripts**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/linter/adapters`):

- [`pyrefly_linter.py_kw.md_docs.md`](./pyrefly_linter.py_kw.md_docs.md)
- [`codespell_linter.py_kw.md_docs.md`](./codespell_linter.py_kw.md_docs.md)
- [`no_workflows_on_fork.py_kw.md_docs.md`](./no_workflows_on_fork.py_kw.md_docs.md)
- [`bazel_linter.py_kw.md_docs.md`](./bazel_linter.py_kw.md_docs.md)
- [`mypy_linter.py_docs.md_docs.md`](./mypy_linter.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`exec_linter.py_kw.md_docs.md`](./exec_linter.py_kw.md_docs.md)
- [`clangformat_linter.py_docs.md_docs.md`](./clangformat_linter.py_docs.md_docs.md)
- [`pip_init.py_kw.md_docs.md`](./pip_init.py_kw.md_docs.md)
- [`testowners_linter.py_docs.md_docs.md`](./testowners_linter.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `update_s3.py_docs.md_docs.md`
- **Keyword Index**: `update_s3.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
