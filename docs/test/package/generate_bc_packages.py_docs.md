# Documentation: `test/package/generate_bc_packages.py`

## File Metadata

- **Path**: `test/package/generate_bc_packages.py`
- **Size**: 1,407 bytes (1.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Can be **executed as a standalone script**.

## Original Source

```python
from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.package import PackageExporter
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE


packaging_directory = f"{Path(__file__).parent}/package_bc"
torch.package.package_exporter._gate_torchscript_serialization = False


def generate_bc_packages():
    """Function to create packages for testing backwards compatibility"""
    if not IS_FBCODE or IS_SANDCASTLE:
        from package_a.test_nn_module import TestNnModule

        test_nn_module = TestNnModule()
        test_torchscript_module = torch.jit.script(TestNnModule())
        test_fx_module: torch.fx.GraphModule = symbolic_trace(TestNnModule())
        with PackageExporter(f"{packaging_directory}/test_nn_module.pt") as pe1:
            pe1.intern("**")
            pe1.save_pickle("nn_module", "nn_module.pkl", test_nn_module)
        with PackageExporter(
            f"{packaging_directory}/test_torchscript_module.pt"
        ) as pe2:
            pe2.intern("**")
            pe2.save_pickle(
                "torchscript_module", "torchscript_module.pkl", test_torchscript_module
            )
        with PackageExporter(f"{packaging_directory}/test_fx_module.pt") as pe3:
            pe3.intern("**")
            pe3.save_pickle("fx_module", "fx_module.pkl", test_fx_module)


if __name__ == "__main__":
    generate_bc_packages()

```



## High-Level Overview

"""Function to create packages for testing backwards compatibility"""    if not IS_FBCODE or IS_SANDCASTLE:        from package_a.test_nn_module import TestNnModule        test_nn_module = TestNnModule()        test_torchscript_module = torch.jit.script(TestNnModule())        test_fx_module: torch.fx.GraphModule = symbolic_trace(TestNnModule())        with PackageExporter(f"{packaging_directory}/test_nn_module.pt") as pe1:            pe1.intern("**")            pe1.save_pickle("nn_module", "nn_module.pkl", test_nn_module)        with PackageExporter(            f"{packaging_directory}/test_torchscript_module.pt"        ) as pe2:            pe2.intern("**")            pe2.save_pickle(                "torchscript_module", "torchscript_module.pkl", test_torchscript_module            )        with PackageExporter(f"{packaging_directory}/test_fx_module.pt") as pe3:            pe3.intern("**")            pe3.save_pickle("fx_module", "fx_module.pkl", test_fx_module)if __name__ == "__main__":    generate_bc_packages()

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `generate_bc_packages`

**Key imports**: Path, torch, symbolic_trace, PackageExporter, IS_FBCODE, IS_SANDCASTLE, TestNnModule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `pathlib`: Path
- `torch`
- `torch.fx`: symbolic_trace
- `torch.package`: PackageExporter
- `torch.testing._internal.common_utils`: IS_FBCODE, IS_SANDCASTLE
- `package_a.test_nn_module`: TestNnModule


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/package/generate_bc_packages.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/package`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_directory_reader.py_docs.md`](./test_directory_reader.py_docs.md)
- [`test_digraph.py_docs.md`](./test_digraph.py_docs.md)
- [`test_dependency_api.py_docs.md`](./test_dependency_api.py_docs.md)
- [`module_a.py_docs.md`](./module_a.py_docs.md)
- [`test_model.py_docs.md`](./test_model.py_docs.md)
- [`module_a_remapped_path.py_docs.md`](./module_a_remapped_path.py_docs.md)
- [`test_glob_group.py_docs.md`](./test_glob_group.py_docs.md)
- [`test_load_bc_packages.py_docs.md`](./test_load_bc_packages.py_docs.md)
- [`test_mangling.py_docs.md`](./test_mangling.py_docs.md)


## Cross-References

- **File Documentation**: `generate_bc_packages.py_docs.md`
- **Keyword Index**: `generate_bc_packages.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
