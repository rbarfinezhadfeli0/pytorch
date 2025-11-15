# Documentation: `tools/linter/adapters/s3_init_config.json`

## File Metadata

- **Path**: `tools/linter/adapters/s3_init_config.json`
- **Size**: 3,413 bytes (3.33 KB)
- **Type**: JSON Configuration
- **Extension**: `.json`

## File Purpose

This file is a **utility or tool script**. This file handles **configuration or setup**.

## Original Source

```json
{
    "HOW TO UPDATE THE BINARIES": [
        "Upload the new file to S3 under a new folder with the version number embedded in (see actionlint for an example).",
        "(Don't override the old files, otherwise you'll break `lintrunner install` for anyone using an older commit of pytorch.)",
        "'Hash' is the sha256 of the uploaded file.",
        "Validate the new download url and hash by running 'lintrunner init' to pull the new binaries and then run 'lintrunner' to try linting the files.",
        "Some binaries have custom builds; see https://github.com/pytorch/test-infra/blob/main/.github/workflows/clang-tidy-linux.yml and https://github.com/pytorch/test-infra/blob/main/.github/workflows/clang-tidy-macos.yml"
    ],
    "clang-format": {
        "Darwin-arm": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/macos-arm/19.1.4/clang-format",
            "hash": "f0da3ecf0ab1e9b50e8c27bd2d7ca0baa619e2f4b824b35d79d46356581fa552"
        },
        "Darwin-i386": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/macos-i386/19.1.4/clang-format",
            "hash": "f5eb5037b9aa9d1d2de650fb2e0fe1a2517768a462fae8e98791a67b698302f4"
        },
        "Linux": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/19.1.4/clang-format",
            "hash": "bfa9ef6eccb372f79ffcb6196af966fd84519ea9567f5ae7b6ad30208cd82109"
        }
    },
    "clang-tidy": {
        "Darwin-i386": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/macos-i386/19.1.4/clang-tidy",
            "hash": "7b5da17d3f8b1c18c77d043999f05293f43402affb16de15dfcb276971984a3e"
        },
        "Darwin-arm": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/macos-arm/19.1.4/clang-tidy",
            "hash": "04243f4044fe6d95f6d51d15be803331c3cbb61f2d8fcfeba5a5dec1e7ae6dfb"
        },
        "Linux": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/19.1.4/clang-tidy",
            "hash": "5637bd0fca665d2797926fedf53ca5ad4655bb9dbed1e1c8654c8e032ce1e7a8"
        }
    },
    "actionlint": {
        "Darwin-i386": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/actionlint/1.7.7/Darwin_amd64/actionlint",
            "hash": "996affd492c57441c5ecfe00dedaef1fde056872d242c0cf7cc15de058d59d03"
        },
        "Darwin-arm": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/actionlint/1.7.7/Darwin_arm64/actionlint",
            "hash": "00aba386d026da33be6e85dd5a46d7af4dd9e4d6cbdb02335f4b267162fd2d9e"
        },
        "Linux": {
            "download_url": "https://oss-clang-format.s3.us-east-2.amazonaws.com/actionlint/1.7.7/Linux_x86_64/actionlint",
            "hash": "9f7dedb4e23f89f2922073d1a6720405b7b520d4f5832ebb96f0d55a2958886c"
        }
    },
    "bazel": {
        "Darwin": {
            "download_url": "https://raw.githubusercontent.com/bazelbuild/bazelisk/v1.16.0/bazelisk.py",
            "hash": "1f6d76d023ddd5f1625f34d934418e7334a267318d084f31be09df8a8835ed16"
        },
        "Linux": {
            "download_url": "https://raw.githubusercontent.com/bazelbuild/bazelisk/v1.16.0/bazelisk.py",
            "hash": "1f6d76d023ddd5f1625f34d934418e7334a267318d084f31be09df8a8835ed16"
        }
    }
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `tools/linter/adapters`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

- **File Documentation**: `s3_init_config.json_docs.md`
- **Keyword Index**: `s3_init_config.json_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
