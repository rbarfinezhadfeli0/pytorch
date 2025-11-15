# Documentation: `.github/scripts/lint_native_functions.py`

## File Metadata

- **Path**: `.github/scripts/lint_native_functions.py`
- **Size**: 2,273 bytes (2.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
#!/usr/bin/env python3
"""
Verify that it is possible to round-trip native_functions.yaml via ruamel under some
configuration.  Keeping native_functions.yaml consistent in this way allows us to
run codemods on the file using ruamel without introducing line noise.  Note that we don't
want to normalize the YAML file, as that would to lots of spurious lint failures.  Anything
that ruamel understands how to roundtrip, e.g., whitespace and comments, is OK!

ruamel is a bit picky about inconsistent indentation, so you will have to indent your
file properly.  Also, if you are working on changing the syntax of native_functions.yaml,
you may find that you want to use some format that is not what ruamel prefers.  If so,
it is OK to modify this script (instead of reformatting native_functions.yaml)--the point
is simply to make sure that there is *some* configuration of ruamel that can round trip
the YAML, not to be prescriptive about it.
"""

import difflib
import sys
from io import StringIO
from pathlib import Path

import ruamel.yaml  # type: ignore[import]


def fn(base: str) -> str:
    return str(base / Path("aten/src/ATen/native/native_functions.yaml"))


with open(Path(__file__).parents[2] / fn(".")) as f:
    contents = f.read()

yaml = ruamel.yaml.YAML()  # type: ignore[attr-defined]
yaml.preserve_quotes = True  # type: ignore[assignment]
yaml.width = 1000  # type: ignore[assignment]
yaml.boolean_representation = ["False", "True"]  # type: ignore[attr-defined]
r = yaml.load(contents)

# Cuz ruamel's author intentionally didn't include conversion to string
# https://stackoverflow.com/questions/47614862/best-way-to-use-ruamel-yaml-to-dump-to-string-not-to-stream
string_stream = StringIO()
yaml.dump(r, string_stream)
new_contents = string_stream.getvalue()
string_stream.close()

if contents != new_contents:
    print(
        """\

## LINT FAILURE: native_functions.yaml ##

native_functions.yaml failed lint; please apply the diff below to fix lint.
If you think this is in error, please see .github/scripts/lint_native_functions.py
""",
        file=sys.stderr,
    )
    sys.stdout.writelines(
        difflib.unified_diff(
            contents.splitlines(True), new_contents.splitlines(True), fn("a"), fn("b")
        )
    )
    sys.exit(1)

```



## High-Level Overview

"""Verify that it is possible to round-trip native_functions.yaml via ruamel under someconfiguration.  Keeping native_functions.yaml consistent in this way allows us torun codemods on the file using ruamel without introducing line noise.  Note that we don'twant to normalize the YAML file, as that would to lots of spurious lint failures.  Anythingthat ruamel understands how to roundtrip, e.g., whitespace and comments, is OK!ruamel is a bit picky about inconsistent indentation, so you will have to indent yourfile properly.  Also, if you are working on changing the syntax of native_functions.yaml,you may find that you want to use some format that is not what ruamel prefers.  If so,it is OK to modify this script (instead of reformatting native_functions.yaml)--the pointis simply to make sure that there is *some* configuration of ruamel that can round tripthe YAML, not to be prescriptive about it.

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `fn`

**Key imports**: difflib, sys, StringIO, Path, ruamel.yaml  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `difflib`
- `sys`
- `io`: StringIO
- `pathlib`: Path
- `ruamel.yaml  `


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

- **File Documentation**: `lint_native_functions.py_docs.md`
- **Keyword Index**: `lint_native_functions.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
