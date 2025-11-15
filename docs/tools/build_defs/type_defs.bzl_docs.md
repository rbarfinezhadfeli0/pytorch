# Documentation: `tools/build_defs/type_defs.bzl`

## File Metadata

- **Path**: `tools/build_defs/type_defs.bzl`
- **Size**: 2,891 bytes (2.82 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```
# Only used for PyTorch open source BUCK build

"""Provides macros for queries type information."""

_SELECT_TYPE = type(select({"DEFAULT": []}))

def is_select(thing):
    return type(thing) == _SELECT_TYPE

def is_unicode(arg):
    """Checks if provided instance has a unicode type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for unicode instances, False otherwise. rtype: bool
    """
    return hasattr(arg, "encode")

_STRING_TYPE = type("")

def is_string(arg):
    """Checks if provided instance has a string type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for string instances, False otherwise. rtype: bool
    """
    return type(arg) == _STRING_TYPE

_LIST_TYPE = type([])

def is_list(arg):
    """Checks if provided instance has a list type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for list instances, False otherwise. rtype: bool
    """
    return type(arg) == _LIST_TYPE

_DICT_TYPE = type({})

def is_dict(arg):
    """Checks if provided instance has a dict type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for dict instances, False otherwise. rtype: bool
    """
    return type(arg) == _DICT_TYPE

_TUPLE_TYPE = type(())

def is_tuple(arg):
    """Checks if provided instance has a tuple type.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for tuple instances, False otherwise. rtype: bool
    """
    return type(arg) == _TUPLE_TYPE

def is_collection(arg):
    """Checks if provided instance is a collection subtype.

    This will either be a dict, list, or tuple.
    """
    return is_dict(arg) or is_list(arg) or is_tuple(arg)

_BOOL_TYPE = type(True)

def is_bool(arg):
    """Checks if provided instance is a boolean value.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for boolean values, False otherwise. rtype: bool
    """
    return type(arg) == _BOOL_TYPE

_NUMBER_TYPE = type(1)

def is_number(arg):
    """Checks if provided instance is a number value.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for number values, False otherwise. rtype: bool
    """
    return type(arg) == _NUMBER_TYPE

_STRUCT_TYPE = type(struct())  # Starlark returns the same type for all structs

def is_struct(arg):
    """Checks if provided instance is a struct value.

    Args:
      arg: An instance to check. type: Any

    Returns:
      True for struct values, False otherwise. rtype: bool
    """
    return type(arg) == _STRUCT_TYPE

type_utils = struct(
    is_bool = is_bool,
    is_number = is_number,
    is_string = is_string,
    is_unicode = is_unicode,
    is_list = is_list,
    is_dict = is_dict,
    is_tuple = is_tuple,
    is_collection = is_collection,
    is_select = is_select,
    is_struct = is_struct,
)

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
- [`platform_defs.bzl_docs.md`](./platform_defs.bzl_docs.md)
- [`fb_xplat_cxx_library.bzl_docs.md`](./fb_xplat_cxx_library.bzl_docs.md)
- [`glob_defs.bzl_docs.md`](./glob_defs.bzl_docs.md)
- [`fb_xplat_genrule.bzl_docs.md`](./fb_xplat_genrule.bzl_docs.md)


## Cross-References

- **File Documentation**: `type_defs.bzl_docs.md`
- **Keyword Index**: `type_defs.bzl_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
