# Documentation: `torch/package/file_structure_representation.py`

## File Metadata

- **Path**: `torch/package/file_structure_representation.py`
- **Size**: 4,746 bytes (4.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

from .glob_group import GlobGroup, GlobPattern


__all__ = ["Directory"]


class Directory:
    """A file structure representation. Organized as Directory nodes that have lists of
    their Directory children. Directories for a package are created by calling
    :meth:`PackageImporter.file_structure`."""

    def __init__(self, name: str, is_dir: bool):
        self.name = name
        self.is_dir = is_dir
        self.children: dict[str, Directory] = {}

    def _get_dir(self, dirs: list[str]) -> "Directory":
        """Builds path of Directories if not yet built and returns last directory
        in list.

        Args:
            dirs (List[str]): List of directory names that are treated like a path.

        Returns:
            :class:`Directory`: The last Directory specified in the dirs list.
        """
        if len(dirs) == 0:
            return self
        dir_name = dirs[0]
        if dir_name not in self.children:
            self.children[dir_name] = Directory(dir_name, True)
        return self.children[dir_name]._get_dir(dirs[1:])

    def _add_file(self, file_path: str):
        """Adds a file to a Directory.

        Args:
            file_path (str): Path of file to add. Last element is added as a file while
                other paths items are added as directories.
        """
        *dirs, file = file_path.split("/")
        dir = self._get_dir(dirs)
        dir.children[file] = Directory(file, False)

    def has_file(self, filename: str) -> bool:
        """Checks if a file is present in a :class:`Directory`.

        Args:
            filename (str): Path of file to search for.
        Returns:
            bool: If a :class:`Directory` contains the specified file.
        """
        lineage = filename.split("/", maxsplit=1)
        child = lineage[0]
        grandchildren = lineage[1] if len(lineage) > 1 else None
        if child in self.children:
            if grandchildren is None:
                return True
            else:
                return self.children[child].has_file(grandchildren)
        return False

    def __str__(self):
        str_list: list[str] = []
        self._stringify_tree(str_list)
        return "".join(str_list)

    def _stringify_tree(
        self,
        str_list: list[str],
        preamble: str = "",
        dir_ptr: str = "\u2500\u2500\u2500 ",
    ):
        """Recursive method to generate print-friendly version of a Directory."""
        space = "    "
        branch = "\u2502   "
        tee = "\u251c\u2500\u2500 "
        last = "\u2514\u2500\u2500 "

        # add this directory's representation
        str_list.append(f"{preamble}{dir_ptr}{self.name}\n")

        # add directory's children representations
        if dir_ptr == tee:
            preamble = preamble + branch
        else:
            preamble = preamble + space

        file_keys: list[str] = []
        dir_keys: list[str] = []
        for key, val in self.children.items():
            if val.is_dir:
                dir_keys.append(key)
            else:
                file_keys.append(key)

        for index, key in enumerate(sorted(dir_keys)):
            if (index == len(dir_keys) - 1) and len(file_keys) == 0:
                self.children[key]._stringify_tree(str_list, preamble, last)
            else:
                self.children[key]._stringify_tree(str_list, preamble, tee)
        for index, file in enumerate(sorted(file_keys)):
            pointer = last if (index == len(file_keys) - 1) else tee
            str_list.append(f"{preamble}{pointer}{file}\n")


def _create_directory_from_file_list(
    filename: str,
    file_list: list[str],
    include: "GlobPattern" = "**",
    exclude: "GlobPattern" = (),
) -> Directory:
    """Return a :class:`Directory` file structure representation created from a list of files.

    Args:
        filename (str): The name given to the top-level directory that will be the
            relative root for all file paths found in the file_list.

        file_list (List[str]): List of files to add to the top-level directory.

        include (Union[List[str], str]): An optional pattern that limits what is included from the file_list to
            files whose name matches the pattern.

        exclude (Union[List[str], str]): An optional pattern that excludes files whose name match the pattern.

    Returns:
            :class:`Directory`: a :class:`Directory` file structure representation created from a list of files.
    """
    glob_pattern = GlobGroup(include, exclude=exclude, separator="/")

    top_dir = Directory(filename, True)
    for file in file_list:
        if glob_pattern.matches(file):
            top_dir._add_file(file)
    return top_dir

```



## High-Level Overview

"""A file structure representation. Organized as Directory nodes that have lists of    their Directory children. Directories for a package are created by calling

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Directory`

**Functions defined**: `__init__`, `_get_dir`, `_add_file`, `has_file`, `__str__`, `_stringify_tree`, `_create_directory_from_file_list`

**Key imports**: GlobGroup, GlobPattern


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `.glob_group`: GlobGroup, GlobPattern


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
- [`_directory_reader.py_docs.md`](./_directory_reader.py_docs.md)
- [`_mangling.py_docs.md`](./_mangling.py_docs.md)
- [`mangling.md_docs.md`](./mangling.md_docs.md)
- [`package_importer.py_docs.md`](./package_importer.py_docs.md)
- [`find_file_dependencies.py_docs.md`](./find_file_dependencies.py_docs.md)


## Cross-References

- **File Documentation**: `file_structure_representation.py_docs.md`
- **Keyword Index**: `file_structure_representation.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
