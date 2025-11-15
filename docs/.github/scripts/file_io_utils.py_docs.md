# Documentation: `.github/scripts/file_io_utils.py`

## File Metadata

- **Path**: `.github/scripts/file_io_utils.py`
- **Size**: 2,919 bytes (2.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
import json
import re
import shutil
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import]


def zip_folder(folder_to_zip: Path, dest_file_base_name: Path) -> Path:
    """
    Returns the path to the resulting zip file, with the appropriate extension added if needed
    """
    # shutil.make_archive will append .zip to the dest_file_name, so we need to remove it if it's already there
    if dest_file_base_name.suffix == ".zip":
        dest_file_base_name = dest_file_base_name.with_suffix("")

    ensure_dir_exists(dest_file_base_name.parent)

    print(f"Zipping {folder_to_zip}\n     to {dest_file_base_name}")
    # Convert to string because shutil.make_archive doesn't like Path objects
    return Path(shutil.make_archive(str(dest_file_base_name), "zip", folder_to_zip))


def unzip_folder(zip_file_path: Path, unzip_to_folder: Path) -> None:
    """
    Returns the path to the unzipped folder
    """
    print(f"Unzipping {zip_file_path}")
    print(f"       to {unzip_to_folder}")
    shutil.unpack_archive(zip_file_path, unzip_to_folder, "zip")


def ensure_dir_exists(dir: Path) -> None:
    dir.mkdir(parents=True, exist_ok=True)


def copy_file(source_file: Path, dest_file: Path) -> None:
    ensure_dir_exists(dest_file.parent)
    shutil.copyfile(source_file, dest_file)


def load_json_file(file_path: Path) -> Any:
    """
    Returns the deserialized json object
    """
    with open(file_path) as f:
        return json.load(f)


def write_json_file(file_path: Path, content: Any) -> None:
    dir = file_path.parent
    ensure_dir_exists(dir)

    with open(file_path, "w") as f:
        json.dump(content, f, indent=2)


def sanitize_for_s3(text: str) -> str:
    """
    S3 keys can only contain alphanumeric characters, underscores, and dashes.
    This function replaces all other characters with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def upload_file_to_s3(file_name: Path, bucket: str, key: str) -> None:
    print(f"Uploading {file_name}")
    print(f"       to s3://{bucket}/{key}")

    boto3.client("s3").upload_file(
        str(file_name),
        bucket,
        key,
    )


def download_s3_objects_with_prefix(
    bucket_name: str, prefix: str, download_folder: Path
) -> list[Path]:
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    downloads = []

    for obj in bucket.objects.filter(Prefix=prefix):
        download_path = download_folder / obj.key

        ensure_dir_exists(download_path.parent)
        print(f"Downloading s3://{bucket.name}/{obj.key}")
        print(f"         to {download_path}")

        s3.Object(bucket.name, obj.key).download_file(str(download_path))
        downloads.append(download_path)

    if len(downloads) == 0:
        print(
            f"There were no files matching the prefix `{prefix}` in bucket `{bucket.name}`"
        )

    return downloads

```



## High-Level Overview

"""    Returns the path to the resulting zip file, with the appropriate extension added if needed

This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `zip_folder`, `unzip_folder`, `ensure_dir_exists`, `copy_file`, `load_json_file`, `write_json_file`, `sanitize_for_s3`, `upload_file_to_s3`, `download_s3_objects_with_prefix`

**Key imports**: json, re, shutil, Path, Any, boto3  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `re`
- `shutil`
- `pathlib`: Path
- `typing`: Any
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

- **File Documentation**: `file_io_utils.py_docs.md`
- **Keyword Index**: `file_io_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
