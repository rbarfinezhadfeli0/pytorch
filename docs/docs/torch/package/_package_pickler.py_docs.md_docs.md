# Documentation: `docs/torch/package/_package_pickler.py_docs.md`

## File Metadata

- **Path**: `docs/torch/package/_package_pickler.py_docs.md`
- **Size**: 8,010 bytes (7.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/package/_package_pickler.py`

## File Metadata

- **Path**: `torch/package/_package_pickler.py`
- **Size**: 5,033 bytes (4.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# pyrefly: ignore [missing-module-attribute]
from pickle import (  # type: ignore[attr-defined]
    _compat_pickle,
    _extension_registry,
    _getattribute,
    _Pickler,
    EXT1,
    EXT2,
    EXT4,
    GLOBAL,
    PicklingError,
    STACK_GLOBAL,
)
from struct import pack
from types import FunctionType

from .importer import Importer, ObjMismatchError, ObjNotFoundError, sys_importer


class _PyTorchLegacyPickler(_Pickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._persistent_id = None

    def persistent_id(self, obj):
        if self._persistent_id is None:
            return super().persistent_id(obj)
        return self._persistent_id(obj)


class PackagePickler(_PyTorchLegacyPickler):
    """Package-aware pickler.

    This behaves the same as a normal pickler, except it uses an `Importer`
    to find objects and modules to save.
    """

    def __init__(self, importer: Importer, *args, **kwargs):
        self.importer = importer
        super().__init__(*args, **kwargs)

        # Make sure the dispatch table copied from _Pickler is up-to-date.
        # Previous issues have been encountered where a library (e.g. dill)
        # mutate _Pickler.dispatch, PackagePickler makes a copy when this lib
        # is imported, then the offending library removes its dispatch entries,
        # leaving PackagePickler with a stale dispatch table that may cause
        # unwanted behavior.
        self.dispatch = _Pickler.dispatch.copy()  # type: ignore[misc]
        self.dispatch[FunctionType] = PackagePickler.save_global  # type: ignore[assignment]

    def save_global(self, obj, name=None):
        # ruff: noqa: F841
        # unfortunately the pickler code is factored in a way that
        # forces us to copy/paste this function. The only change is marked
        # CHANGED below.
        write = self.write  # type: ignore[attr-defined]
        memo = self.memo  # type: ignore[attr-defined]

        # CHANGED: import module from module environment instead of __import__
        try:
            module_name, name = self.importer.get_name(obj, name)
        except (ObjNotFoundError, ObjMismatchError) as err:
            raise PicklingError(f"Can't pickle {obj}: {str(err)}") from err

        module = self.importer.import_module(module_name)
        _, parent = _getattribute(module, name)
        # END CHANGED

        if self.proto >= 2:  # type: ignore[attr-defined]
            code = _extension_registry.get((module_name, name))
            if code:
                assert code > 0
                if code <= 0xFF:
                    write(EXT1 + pack("<B", code))
                elif code <= 0xFFFF:
                    write(EXT2 + pack("<H", code))
                else:
                    write(EXT4 + pack("<i", code))
                return
        lastname = name.rpartition(".")[2]
        if parent is module:
            name = lastname
        # Non-ASCII identifiers are supported only with protocols >= 3.
        if self.proto >= 4:  # type: ignore[attr-defined]
            self.save(module_name)  # type: ignore[attr-defined]
            self.save(name)  # type: ignore[attr-defined]
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))  # type: ignore[attr-defined]
        elif self.proto >= 3:  # type: ignore[attr-defined]
            write(
                GLOBAL
                + bytes(module_name, "utf-8")
                + b"\n"
                + bytes(name, "utf-8")
                + b"\n"
            )
        else:
            if self.fix_imports:  # type: ignore[attr-defined]
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[(module_name, name)]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(
                    GLOBAL
                    + bytes(module_name, "ascii")
                    + b"\n"
                    + bytes(name, "ascii")
                    + b"\n"
                )
            except UnicodeEncodeError as exc:
                raise PicklingError(
                    f"can't pickle global identifier '{module}.{name}' using "
                    f"pickle protocol {self.proto:d}"  # type: ignore[attr-defined]
                ) from exc

        self.memoize(obj)  # type: ignore[attr-defined]


def create_pickler(data_buf, importer, protocol=4):
    if importer is sys_importer:
        # if we are using the normal import library system, then
        # we can use the C implementation of pickle which is faster
        return _PyTorchLegacyPickler(data_buf, protocol=protocol)
    else:
        return PackagePickler(importer, data_buf, protocol=protocol)

```



## High-Level Overview

"""Package-aware pickler.    This behaves the same as a normal pickler, except it uses an `Importer`    to find objects and modules to save.

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_PyTorchLegacyPickler`, `PackagePickler`

**Functions defined**: `__init__`, `persistent_id`, `__init__`, `save_global`, `create_pickler`

**Key imports**: pack, FunctionType, Importer, ObjMismatchError, ObjNotFoundError, sys_importer, module from module environment instead of __import__, library system, then


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `struct`: pack
- `types`: FunctionType
- `.importer`: Importer, ObjMismatchError, ObjNotFoundError, sys_importer
- `module from module environment instead of __import__`
- `library system, then`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/package`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`package_exporter.py_docs.md`](./package_exporter.py_docs.md)
- [`glob_group.py_docs.md`](./glob_group.py_docs.md)
- [`file_structure_representation.py_docs.md`](./file_structure_representation.py_docs.md)
- [`_directory_reader.py_docs.md`](./_directory_reader.py_docs.md)
- [`_mangling.py_docs.md`](./_mangling.py_docs.md)
- [`mangling.md_docs.md`](./mangling.md_docs.md)
- [`package_importer.py_docs.md`](./package_importer.py_docs.md)
- [`find_file_dependencies.py_docs.md`](./find_file_dependencies.py_docs.md)


## Cross-References

- **File Documentation**: `_package_pickler.py_docs.md`
- **Keyword Index**: `_package_pickler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/package`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/package`):

- [`importer.py_docs.md_docs.md`](./importer.py_docs.md_docs.md)
- [`file_structure_representation.py_kw.md_docs.md`](./file_structure_representation.py_kw.md_docs.md)
- [`_directory_reader.py_docs.md_docs.md`](./_directory_reader.py_docs.md_docs.md)
- [`_package_unpickler.py_kw.md_docs.md`](./_package_unpickler.py_kw.md_docs.md)
- [`_digraph.py_kw.md_docs.md`](./_digraph.py_kw.md_docs.md)
- [`_directory_reader.py_kw.md_docs.md`](./_directory_reader.py_kw.md_docs.md)
- [`mangling.md_docs.md_docs.md`](./mangling.md_docs.md_docs.md)
- [`mangling.md_kw.md_docs.md`](./mangling.md_kw.md_docs.md)
- [`package_importer.py_docs.md_docs.md`](./package_importer.py_docs.md_docs.md)
- [`package_importer.py_kw.md_docs.md`](./package_importer.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_package_pickler.py_docs.md_docs.md`
- **Keyword Index**: `_package_pickler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
