# Documentation: `torch/package/find_file_dependencies.py`

## File Metadata

- **Path**: `torch/package/find_file_dependencies.py`
- **Size**: 3,979 bytes (3.89 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import ast
from typing import Optional

from ._importlib import _resolve_name


class _ExtractModuleReferences(ast.NodeVisitor):
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    def run(cls, src: str, package: str) -> list[tuple[str, Optional[str]]]:
        visitor = cls(package)
        tree = ast.parse(src)
        visitor.visit(tree)
        return list(visitor.references.keys())

    def __init__(self, package):
        super().__init__()
        self.package = package
        self.references = {}

    def _absmodule(self, module_name: str, level: int) -> str:
        if level > 0:
            return _resolve_name(module_name, self.package, level)
        return module_name

    def visit_Import(self, node):
        for alias in node.names:
            self.references[(alias.name, None)] = True

    def visit_ImportFrom(self, node):
        name = self._absmodule(node.module, 0 if node.level is None else node.level)
        for alias in node.names:
            # from my_package import foo
            # foo may be a module, so we have to add it to the list of
            # potential references, if import of it fails, we will ignore it
            if alias.name != "*":
                self.references[(name, alias.name)] = True
            else:
                self.references[(name, None)] = True

    def _grab_node_int(self, node):
        return node.value

    def _grab_node_str(self, node):
        return node.value

    def visit_Call(self, node):
        # __import__ calls aren't routed to the visit_Import/From nodes
        if hasattr(node.func, "id") and node.func.id == "__import__":
            try:
                name = self._grab_node_str(node.args[0])
                fromlist: list[str] = []
                level = 0
                if len(node.args) > 3:
                    fromlist.extend(self._grab_node_str(v) for v in node.args[3].elts)
                elif hasattr(node, "keywords"):
                    for keyword in node.keywords:
                        if keyword.arg == "fromlist":
                            fromlist.extend(
                                self._grab_node_str(v) for v in keyword.value.elts
                            )
                if len(node.args) > 4:
                    level = self._grab_node_int(node.args[4])
                elif hasattr(node, "keywords"):
                    for keyword in node.keywords:
                        if keyword.arg == "level":
                            level = self._grab_node_int(keyword.value)
                if fromlist == []:
                    # the top-level package (the name up till the first dot) is returned
                    # when the fromlist argument is empty in normal import system,
                    # we need to include top level package to match this behavior and last
                    # level package to capture the intended dependency of user
                    self.references[(name, None)] = True
                    top_name = name.rsplit(".", maxsplit=1)[0]
                    if top_name != name:
                        top_name = self._absmodule(top_name, level)
                        self.references[(top_name, None)] = True
                else:
                    name = self._absmodule(name, level)
                    for alias in fromlist:
                        # fromlist args may be submodules, so we have to add the fromlist args
                        # to the list of potential references. If import of an arg fails we
                        # will ignore it, similar to visit_ImportFrom
                        if alias != "*":
                            self.references[(name, alias)] = True
                        else:
                            self.references[(name, None)] = True
            except Exception:
                return


find_files_source_depends_on = _ExtractModuleReferences.run

```



## High-Level Overview

"""    Extract the list of global variables a block of code will read and write

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_ExtractModuleReferences`

**Functions defined**: `run`, `__init__`, `_absmodule`, `visit_Import`, `visit_ImportFrom`, `_grab_node_int`, `_grab_node_str`, `visit_Call`

**Key imports**: ast, Optional, _resolve_name, foo, of it fails, we will ignore it, system,, of an arg fails we


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `ast`
- `typing`: Optional
- `._importlib`: _resolve_name
- `my_package`: foo
- `of it fails, we will ignore it`
- `system,`
- `of an arg fails we`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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


## Cross-References

- **File Documentation**: `find_file_dependencies.py_docs.md`
- **Keyword Index**: `find_file_dependencies.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
