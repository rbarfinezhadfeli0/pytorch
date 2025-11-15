# Documentation: `tools/build_defs/glob_defs.bzl`

## File Metadata

- **Path**: `tools/build_defs/glob_defs.bzl`
- **Size**: 3,110 bytes (3.04 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```
# Only used for PyTorch open source BUCK build

"""Provides utility macros for working with globs."""

load("@bazel_skylib//lib:paths.bzl", "paths")

def subdir_glob(glob_specs, exclude = None, prefix = ""):
    """Returns a dict of sub-directory relative paths to full paths.

    The subdir_glob() function is useful for defining header maps for C/C++
    libraries which should be relative the given sub-directory.
    Given a list of tuples, the form of (relative-sub-directory, glob-pattern),
    it returns a dict of sub-directory relative paths to full paths.

    Please refer to native.glob() for explanations and examples of the pattern.

    Args:
      glob_specs: The array of tuples in form of
        (relative-sub-directory, glob-pattern inside relative-sub-directory).
        type: List[Tuple[str, str]]
      exclude: A list of patterns to identify files that should be removed
        from the set specified by the first argument. Defaults to [].
        type: Optional[List[str]]
      prefix: If is not None, prepends it to each key in the dictionary.
        Defaults to None.
        type: Optional[str]

    Returns:
      A dict of sub-directory relative paths to full paths.
    """
    if exclude == None:
        exclude = []

    results = []

    for dirpath, glob_pattern in glob_specs:
        results.append(
            _single_subdir_glob(dirpath, glob_pattern, exclude, prefix),
        )

    return _merge_maps(*results)

def _merge_maps(*file_maps):
    result = {}
    for file_map in file_maps:
        for key in file_map:
            if key in result and result[key] != file_map[key]:
                fail(
                    "Conflicting files in file search paths. " +
                    "\"%s\" maps to both \"%s\" and \"%s\"." %
                    (key, result[key], file_map[key]),
                )

            result[key] = file_map[key]

    return result

def _single_subdir_glob(dirpath, glob_pattern, exclude = None, prefix = None):
    if exclude == None:
        exclude = []
    results = {}
    files = native.glob([paths.join(dirpath, glob_pattern)], exclude = exclude)
    for f in files:
        if dirpath:
            key = f[len(dirpath) + 1:]
        else:
            key = f
        if prefix:
            key = paths.join(prefix, key)
        results[key] = f

    return results

# Using a flat list will trigger build errors on Android.
# cxx_library will generate an apple_library on iOS, a cxx_library on Android.
# Those rules have different behaviors. Using a map will make the behavior consistent.
#
def glob_private_headers(glob_patterns, exclude = []):
    result = {}
    headers = native.glob(glob_patterns, exclude = exclude)
    for header in headers:
        result[paths.basename(header)] = header
    return result

def glob(include, exclude = (), **kwargs):
    buildfile = native.read_config("buildfile", "name", "BUCK")
    subpkgs = [
        target[:-len(buildfile)] + "**/*"
        for target in native.glob(["*/**/" + buildfile])
    ]
    return native.glob(include, exclude = list(exclude) + subpkgs, **kwargs)

```



## High-Level Overview

This file is part of the PyTorch framework located at `tools/build_defs`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/build_defs`, which contains **development tools and scripts**.



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

Files in the same folder (`tools/build_defs`):

- [`select.bzl_docs.md`](./select.bzl_docs.md)
- [`default_platform_defs.bzl_docs.md`](./default_platform_defs.bzl_docs.md)
- [`expect.bzl_docs.md`](./expect.bzl_docs.md)
- [`fbsource_utils.bzl_docs.md`](./fbsource_utils.bzl_docs.md)
- [`buck_helpers.bzl_docs.md`](./buck_helpers.bzl_docs.md)
- [`type_defs.bzl_docs.md`](./type_defs.bzl_docs.md)
- [`platform_defs.bzl_docs.md`](./platform_defs.bzl_docs.md)
- [`fb_xplat_cxx_library.bzl_docs.md`](./fb_xplat_cxx_library.bzl_docs.md)
- [`fb_xplat_genrule.bzl_docs.md`](./fb_xplat_genrule.bzl_docs.md)


## Cross-References

- **File Documentation**: `glob_defs.bzl_docs.md`
- **Keyword Index**: `glob_defs.bzl_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
