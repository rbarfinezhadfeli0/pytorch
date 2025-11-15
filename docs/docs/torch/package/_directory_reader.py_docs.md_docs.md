# Documentation: `docs/torch/package/_directory_reader.py_docs.md`

## File Metadata

- **Path**: `docs/torch/package/_directory_reader.py_docs.md`
- **Size**: 4,875 bytes (4.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/package/_directory_reader.py`

## File Metadata

- **Path**: `torch/package/_directory_reader.py`
- **Size**: 1,915 bytes (1.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import os.path
from glob import glob
from typing import cast

import torch
from torch.types import Storage


__serialization_id_record_name__ = ".data/serialization_id"


# because get_storage_from_record returns a tensor!?
class _HasStorage:
    def __init__(self, storage):
        self._storage = storage

    def storage(self):
        return self._storage


class DirectoryReader:
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the internal PyTorchFileReader class (which is used for
    accessing packages in all other cases).

    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader
    class due to ScriptObjects requiring an actual PyTorchFileReader instance.
    """

    def __init__(self, directory):
        self.directory = directory

    def get_record(self, name):
        filename = f"{self.directory}/{name}"
        with open(filename, "rb") as f:
            return f.read()

    def get_storage_from_record(self, name, numel, dtype):
        filename = f"{self.directory}/{name}"
        nbytes = torch._utils._element_size(dtype) * numel
        storage = cast(Storage, torch.UntypedStorage)
        return _HasStorage(storage.from_file(filename=filename, nbytes=nbytes))

    def has_record(self, path):
        full_path = os.path.join(self.directory, path)
        return os.path.isfile(full_path)

    def get_all_records(
        self,
    ):
        files = [
            filename[len(self.directory) + 1 :]
            for filename in glob(f"{self.directory}/**", recursive=True)
            if not os.path.isdir(filename)
        ]
        return files

    def serialization_id(
        self,
    ):
        if self.has_record(__serialization_id_record_name__):
            return self.get_record(__serialization_id_record_name__)
        else:
            return ""

```



## High-Level Overview

"""    Class to allow PackageImporter to operate on unzipped packages. Methods    copy the behavior of the internal PyTorchFileReader class (which is used for    accessing packages in all other cases).    N.B.: ScriptObjects are not depickleable or accessible via this DirectoryReader    class due to ScriptObjects requiring an actual PyTorchFileReader instance.

This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_HasStorage`, `DirectoryReader`

**Functions defined**: `__init__`, `storage`, `__init__`, `get_record`, `get_storage_from_record`, `has_record`, `get_all_records`, `serialization_id`

**Key imports**: os.path, glob, cast, torch, Storage


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os.path`
- `glob`: glob
- `typing`: cast
- `torch`
- `torch.types`: Storage


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
- [`_mangling.py_docs.md`](./_mangling.py_docs.md)
- [`mangling.md_docs.md`](./mangling.md_docs.md)
- [`package_importer.py_docs.md`](./package_importer.py_docs.md)
- [`find_file_dependencies.py_docs.md`](./find_file_dependencies.py_docs.md)


## Cross-References

- **File Documentation**: `_directory_reader.py_docs.md`
- **Keyword Index**: `_directory_reader.py_kw.md`
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
- [`_package_unpickler.py_kw.md_docs.md`](./_package_unpickler.py_kw.md_docs.md)
- [`_digraph.py_kw.md_docs.md`](./_digraph.py_kw.md_docs.md)
- [`_directory_reader.py_kw.md_docs.md`](./_directory_reader.py_kw.md_docs.md)
- [`mangling.md_docs.md_docs.md`](./mangling.md_docs.md_docs.md)
- [`mangling.md_kw.md_docs.md`](./mangling.md_kw.md_docs.md)
- [`package_importer.py_docs.md_docs.md`](./package_importer.py_docs.md_docs.md)
- [`package_importer.py_kw.md_docs.md`](./package_importer.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_directory_reader.py_docs.md_docs.md`
- **Keyword Index**: `_directory_reader.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
