# Documentation: `torch/package/importer.py`

## File Metadata

- **Path**: `torch/package/importer.py`
- **Size**: 9,920 bytes (9.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import importlib
import logging
import sys
from abc import ABC, abstractmethod

# pyrefly: ignore [missing-module-attribute]
from pickle import (  # type: ignore[attr-defined]
    _getattribute,
    _Pickler,
    whichmodule as _pickle_whichmodule,  # pyrefly: ignore  # missing-module-attribute
)
from types import ModuleType
from typing import Any, Optional

from ._mangling import demangle, get_mangle_prefix, is_mangled


__all__ = ["ObjNotFoundError", "ObjMismatchError", "Importer", "OrderedImporter"]
log = logging.getLogger(__name__)


class ObjNotFoundError(Exception):
    """Raised when an importer cannot find an object by searching for its name."""


class ObjMismatchError(Exception):
    """Raised when an importer found a different object with the same name as the user-provided one."""


class Importer(ABC):
    """Represents an environment to import modules from.

    By default, you can figure out what module an object belongs by checking
    __module__ and importing the result using __import__ or importlib.import_module.

    torch.package introduces module importers other than the default one.
    Each PackageImporter introduces a new namespace. Potentially a single
    name (e.g. 'foo.bar') is present in multiple namespaces.

    It supports two main operations:
        import_module: module_name -> module object
        get_name: object -> (parent module name, name of obj within module)

    The guarantee is that following round-trip will succeed or throw an ObjNotFoundError/ObjMisMatchError.
        module_name, obj_name = env.get_name(obj)
        module = env.import_module(module_name)
        obj2 = getattr(module, obj_name)
        assert obj1 is obj2
    """

    modules: dict[str, ModuleType]

    @abstractmethod
    def import_module(self, module_name: str) -> ModuleType:
        """Import `module_name` from this environment.

        The contract is the same as for importlib.import_module.
        """

    def get_name(self, obj: Any, name: Optional[str] = None) -> tuple[str, str]:
        """Given an object, return a name that can be used to retrieve the
        object from this environment.

        Args:
            obj: An object to get the module-environment-relative name for.
            name: If set, use this name instead of looking up __name__ or __qualname__ on `obj`.
                This is only here to match how Pickler handles __reduce__ functions that return a string,
                don't use otherwise.
        Returns:
            A tuple (parent_module_name, attr_name) that can be used to retrieve `obj` from this environment.
            Use it like:
                mod = importer.import_module(parent_module_name)
                obj = getattr(mod, attr_name)

        Raises:
            ObjNotFoundError: we couldn't retrieve `obj by name.
            ObjMisMatchError: we found a different object with the same name as `obj`.
        """
        if name is None and obj and _Pickler.dispatch.get(type(obj)) is None:
            # Honor the string return variant of __reduce__, which will give us
            # a global name to search for in this environment.
            # TODO: I guess we should do copyreg too?
            reduce = getattr(obj, "__reduce__", None)
            if reduce is not None:
                try:
                    rv = reduce()
                    if isinstance(rv, str):
                        name = rv
                except Exception:
                    pass
        if name is None:
            name = getattr(obj, "__qualname__", None)
        if name is None:
            name = obj.__name__

        orig_module_name = self.whichmodule(obj, name)
        # Demangle the module name before importing. If this obj came out of a
        # PackageImporter, `__module__` will be mangled. See mangling.md for
        # details.
        module_name = demangle(orig_module_name)

        # Check that this name will indeed return the correct object
        try:
            module = self.import_module(module_name)
            if sys.version_info >= (3, 14):
                # pickle._getatribute signature changes in 3.14
                # to take iterable and return just one object
                obj2 = _getattribute(module, name.split("."))
            else:
                obj2, _ = _getattribute(module, name)
        except (ImportError, KeyError, AttributeError):
            raise ObjNotFoundError(
                f"{obj} was not found as {module_name}.{name}"
            ) from None

        if obj is obj2:
            return module_name, name

        def get_obj_info(obj):
            assert name is not None
            module_name = self.whichmodule(obj, name)
            is_mangled_ = is_mangled(module_name)
            location = (
                get_mangle_prefix(module_name)
                if is_mangled_
                else "the current Python environment"
            )
            importer_name = (
                f"the importer for {get_mangle_prefix(module_name)}"
                if is_mangled_
                else "'sys_importer'"
            )
            return module_name, location, importer_name

        obj_module_name, obj_location, obj_importer_name = get_obj_info(obj)
        obj2_module_name, obj2_location, obj2_importer_name = get_obj_info(obj2)
        msg = (
            f"\n\nThe object provided is from '{obj_module_name}', "
            f"which is coming from {obj_location}."
            f"\nHowever, when we import '{obj2_module_name}', it's coming from {obj2_location}."
            "\nTo fix this, make sure this 'PackageExporter's importer lists "
            f"{obj_importer_name} before {obj2_importer_name}."
        )
        raise ObjMismatchError(msg)

    def whichmodule(self, obj: Any, name: str) -> str:
        """Find the module name an object belongs to.

        This should be considered internal for end-users, but developers of
        an importer can override it to customize the behavior.

        Taken from pickle.py, but modified to exclude the search into sys.modules
        """
        module_name = getattr(obj, "__module__", None)
        if module_name is not None:
            return module_name

        # Protect the iteration by using a list copy of self.modules against dynamic
        # modules that trigger imports of other modules upon calls to getattr.
        for module_name, module in self.modules.copy().items():
            if (
                module_name == "__main__"
                or module_name == "__mp_main__"  # bpo-42406
                or module is None
            ):
                continue
            try:
                if _getattribute(module, name)[0] is obj:
                    return module_name
            except AttributeError:
                pass

        return "__main__"


class _SysImporter(Importer):
    """An importer that implements the default behavior of Python."""

    def import_module(self, module_name: str):
        return importlib.import_module(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        return _pickle_whichmodule(obj, name)


sys_importer = _SysImporter()


class OrderedImporter(Importer):
    """A compound importer that takes a list of importers and tries them one at a time.

    The first importer in the list that returns a result "wins".
    """

    def __init__(self, *args):
        self._importers: list[Importer] = list(args)

    def _is_torchpackage_dummy(self, module):
        """Returns true iff this module is an empty PackageNode in a torch.package.

        If you intern `a.b` but never use `a` in your code, then `a` will be an
        empty module with no source. This can break cases where we are trying to
        re-package an object after adding a real dependency on `a`, since
        OrderedImportere will resolve `a` to the dummy package and stop there.

        See: https://github.com/pytorch/pytorch/pull/71520#issuecomment-1029603769
        """
        if not getattr(module, "__torch_package__", False):
            return False
        if not hasattr(module, "__path__"):
            return False
        if not hasattr(module, "__file__"):
            return True
        return module.__file__ is None

    def get_name(self, obj: Any, name: Optional[str] = None) -> tuple[str, str]:
        for importer in self._importers:
            try:
                return importer.get_name(obj, name)
            except (ObjNotFoundError, ObjMismatchError) as e:
                warning_message = (
                    f"Tried to call get_name with obj {obj}, "
                    f"and name {name} on {importer} and got {e}"
                )
                log.warning(warning_message)
        raise ObjNotFoundError(
            f"Could not find obj {obj} and name {name} in any of the importers {self._importers}"
        )

    def import_module(self, module_name: str) -> ModuleType:
        last_err = None
        for importer in self._importers:
            if not isinstance(importer, Importer):
                raise TypeError(
                    f"{importer} is not a Importer. "
                    "All importers in OrderedImporter must inherit from Importer."
                )
            try:
                module = importer.import_module(module_name)
                if self._is_torchpackage_dummy(module):
                    continue
                return module
            except ModuleNotFoundError as err:
                last_err = err

        if last_err is not None:
            raise last_err
        else:
            raise ModuleNotFoundError(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        for importer in self._importers:
            module_name = importer.whichmodule(obj, name)
            if module_name != "__main__":
                return module_name

        return "__main__"

```



## High-Level Overview

"""Raised when an importer cannot find an object by searching for its name."""class ObjMismatchError(Exception):

This Python file contains 5 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ObjNotFoundError`, `ObjMismatchError`, `Importer`, `_SysImporter`, `OrderedImporter`

**Functions defined**: `import_module`, `get_name`, `get_obj_info`, `whichmodule`, `import_module`, `whichmodule`, `__init__`, `_is_torchpackage_dummy`, `get_name`, `import_module`, `whichmodule`

**Key imports**: importlib, logging, sys, ABC, abstractmethod, ModuleType, Any, Optional, demangle, get_mangle_prefix, is_mangled, modules from.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `importlib`
- `logging`
- `sys`
- `abc`: ABC, abstractmethod
- `types`: ModuleType
- `typing`: Any, Optional
- `._mangling`: demangle, get_mangle_prefix, is_mangled
- `modules from.`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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
- [`_package_pickler.py_docs.md`](./_package_pickler.py_docs.md)
- [`glob_group.py_docs.md`](./glob_group.py_docs.md)
- [`file_structure_representation.py_docs.md`](./file_structure_representation.py_docs.md)
- [`_directory_reader.py_docs.md`](./_directory_reader.py_docs.md)
- [`_mangling.py_docs.md`](./_mangling.py_docs.md)
- [`mangling.md_docs.md`](./mangling.md_docs.md)
- [`package_importer.py_docs.md`](./package_importer.py_docs.md)
- [`find_file_dependencies.py_docs.md`](./find_file_dependencies.py_docs.md)


## Cross-References

- **File Documentation**: `importer.py_docs.md`
- **Keyword Index**: `importer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
