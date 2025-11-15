# Documentation: `docs/torchgen/decompositions/gen_jit_decompositions.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/decompositions/gen_jit_decompositions.py_docs.md`
- **Size**: 4,967 bytes (4.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/decompositions/gen_jit_decompositions.py`

## File Metadata

- **Path**: `torchgen/decompositions/gen_jit_decompositions.py`
- **Size**: 2,424 bytes (2.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
import os
from pathlib import Path

from torch.jit._decompositions import decomposition_table


# from torchgen.code_template import CodeTemplate

DECOMP_HEADER = r"""
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python torchgen/decompositions/gen_jit_decompositions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/decomposition_registry_util.h>

namespace torch {
namespace jit {


const std::string decomp_funcs =
R"("""


DECOMP_CENTER = r"""
)";

const std::string& GetSerializedDecompositions() {
  return decomp_funcs;
}

const OperatorMap<std::string>& GetDecompositionMapping() {
  // clang-format off
 static const OperatorMap<std::string> decomposition_mapping {
"""

DECOMP_END = r"""
  };
  // clang-format on

  return decomposition_mapping;
}

} // namespace jit
} // namespace torch
"""


DECOMPOSITION_UTIL_FILE_NAME = "decomposition_registry_util.cpp"


def gen_serialized_decompisitions() -> str:
    return "\n".join(
        [scripted_func.code for scripted_func in decomposition_table.values()]  # type: ignore[misc]
    )


def gen_decomposition_mappings() -> str:
    decomposition_mappings = []
    for schema, scripted_func in decomposition_table.items():
        decomposition_mappings.append(
            '    {"' + schema + '", "' + scripted_func.name + '"},'  # type: ignore[operator]
        )
    return "\n".join(decomposition_mappings)


def write_decomposition_util_file(path: str) -> None:
    decomposition_str = gen_serialized_decompisitions()
    decomposition_mappings = gen_decomposition_mappings()
    file_components = [
        DECOMP_HEADER,
        decomposition_str,
        DECOMP_CENTER,
        decomposition_mappings,
        DECOMP_END,
    ]
    print("writing file to : ", path + "/" + DECOMPOSITION_UTIL_FILE_NAME)
    with open(os.path.join(path, DECOMPOSITION_UTIL_FILE_NAME), "wb") as out_file:
        final_output = "".join(file_components)
        out_file.write(final_output.encode("utf-8"))


def main() -> None:
    pytorch_dir = Path(__file__).resolve().parents[3]
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "runtime"
    write_decomposition_util_file(str(upgrader_path))


if __name__ == "__main__":
    main()

```



## High-Level Overview

DECOMP_HEADER = r"""/** * @generated * This is an auto-generated file. Please do not modify it by hand. * To re-generate, please run: * cd ~/pytorch && python torchgen/decompositions/gen_jit_decompositions.py */#include <torch/csrc/jit/jit_log.h>#include <torch/csrc/jit/passes/inliner.h>#include <torch/csrc/jit/runtime/operator.h>#include <torch/csrc/jit/runtime/decomposition_registry_util.h>namespace torch {namespace jit {const std::string decomp_funcs =

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `gen_serialized_decompisitions`, `gen_decomposition_mappings`, `write_decomposition_util_file`, `main`

**Key imports**: os, Path, decomposition_table, CodeTemplate


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/decompositions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `pathlib`: Path
- `torch.jit._decompositions`: decomposition_table
- `torchgen.code_template`: CodeTemplate


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torchgen/decompositions`):



## Cross-References

- **File Documentation**: `gen_jit_decompositions.py_docs.md`
- **Keyword Index**: `gen_jit_decompositions.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen/decompositions`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen/decompositions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torchgen/decompositions`):

- [`gen_jit_decompositions.py_kw.md_docs.md`](./gen_jit_decompositions.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `gen_jit_decompositions.py_docs.md_docs.md`
- **Keyword Index**: `gen_jit_decompositions.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
