# Documentation: `docs/tools/experimental/torchfuzz/type_promotion.py_docs.md`

## File Metadata

- **Path**: `docs/tools/experimental/torchfuzz/type_promotion.py_docs.md`
- **Size**: 9,284 bytes (9.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/experimental/torchfuzz/type_promotion.py`

## File Metadata

- **Path**: `tools/experimental/torchfuzz/type_promotion.py`
- **Size**: 5,641 bytes (5.51 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
"""Type promotion utilities for torchfuzz operators."""

import random

import torch


# Define promotion chains - types that can promote to the target
# PyTorch promotion hierarchy (simplified):
# - bool < int8 < int16 < int32 < int64 < float16 < float32 < float64 < complex64 < complex128
# - uint types have limited promotion support
PROMOTION_CHAINS = {
    torch.bool: [torch.bool],
    torch.int8: [torch.bool, torch.int8],
    torch.int16: [torch.bool, torch.int8, torch.int16],
    torch.int32: [torch.bool, torch.int8, torch.int16, torch.int32],
    torch.int64: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64],
    torch.float16: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
    ],
    torch.float32: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
    ],
    torch.float64: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    ],
    torch.complex64: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.complex64,
    ],
    torch.complex128: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
}


def get_promoted_dtypes(target_dtype: torch.dtype) -> list[torch.dtype]:
    """
    Generate two dtypes that will promote to target_dtype via PyTorch's type promotion rules.
    """
    # Get compatible input types for the target dtype
    compatible_types = PROMOTION_CHAINS.get(target_dtype, [target_dtype])

    # Strategy: Choose between same type or mixed promotion
    strategies = ["same_type", "mixed_promotion"]
    strategy = random.choice(strategies)

    if strategy == "same_type":
        # Both args same type as target
        return [target_dtype, target_dtype]

    else:  # mixed_promotion
        # Mixed types where the result will promote to target_dtype
        lower_types = compatible_types[:-1]  # All except the last (target_dtype)

        if lower_types:
            # One arg is target_dtype, one is lower (will promote to target)
            lower_dtype = random.choice(lower_types)
            if random.random() < 0.5:
                return [target_dtype, lower_dtype]
            else:
                return [lower_dtype, target_dtype]
        else:
            # Fallback to same type if no lower types available
            return [target_dtype, target_dtype]


def get_dtype_name(dtype: torch.dtype) -> str:
    """Get string name for a torch dtype."""
    return str(dtype).split(".")[-1]


def get_promotion_table_for_strings() -> dict:
    """
    Get promotion table using string dtype names for backward compatibility.
    Returns dictionary mapping output dtype string to possible input dtype string pairs.
    """
    return {
        "float32": [
            ("float32", "float32"),
            ("bfloat16", "float32"),
            ("float32", "bfloat16"),
            ("float16", "float32"),
            ("float32", "float16"),
        ],
        "bfloat16": [
            ("bfloat16", "bfloat16"),
            ("float32", "bfloat16"),
            ("bfloat16", "float32"),
        ],
        "float16": [
            ("float16", "float16"),
            ("float32", "float16"),
            ("float16", "float32"),
        ],
        "int32": [
            ("int32", "int32"),
            ("int64", "int32"),
            ("int32", "int64"),
        ],
        "int64": [
            ("int64", "int64"),
            ("int32", "int64"),
            ("int64", "int32"),
        ],
        "bool": [
            ("bool", "bool"),
        ],
    }


def get_dtype_map() -> dict:
    """Get mapping from string names to torch dtypes."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
        "int8": torch.int8,
        "int16": torch.int16,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }


def get_scalar_promotion_pairs(
    target_dtype: torch.dtype,
) -> list[tuple[torch.dtype, torch.dtype]]:
    """
    Get promotion pairs for scalar operations.
    Returns list of (dtype1, dtype2) tuples that promote to target_dtype.
    """
    return (
        [
            (torch.float32, torch.float32),
            (torch.float16, torch.float32),
            (torch.float32, torch.float16),
            (torch.int32, torch.float32),
            (torch.float32, torch.int32),
        ]
        if target_dtype == torch.float32
        else [
            (torch.float64, torch.float64),
            (torch.float32, torch.float64),
            (torch.float64, torch.float32),
        ]
        if target_dtype == torch.float64
        else [
            (torch.int32, torch.int32),
            (torch.int64, torch.int32),
            (torch.int32, torch.int64),
        ]
        if target_dtype == torch.int32
        else [
            (torch.int64, torch.int64),
            (torch.int32, torch.int64),
            (torch.int64, torch.int32),
        ]
        if target_dtype == torch.int64
        else [(target_dtype, target_dtype)]
    )

```



## High-Level Overview

"""Type promotion utilities for torchfuzz operators."""import randomimport torch# Define promotion chains - types that can promote to the target# PyTorch promotion hierarchy (simplified):# - bool < int8 < int16 < int32 < int64 < float16 < float32 < float64 < complex64 < complex128# - uint types have limited promotion supportPROMOTION_CHAINS = {    torch.bool: [torch.bool],    torch.int8: [torch.bool, torch.int8],    torch.int16: [torch.bool, torch.int8, torch.int16],    torch.int32: [torch.bool, torch.int8, torch.int16, torch.int32],    torch.int64: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64],    torch.float16: [        torch.bool,        torch.int8,        torch.int16,        torch.int32,        torch.int64,        torch.float16,    ],    torch.float32: [        torch.bool,        torch.int8,        torch.int16,        torch.int32,        torch.int64,        torch.float16,        torch.float32,    ],    torch.float64: [        torch.bool,        torch.int8,        torch.int16,        torch.int32,        torch.int64,        torch.float16,        torch.float32,        torch.float64,    ],    torch.complex64: [        torch.bool,        torch.int8,        torch.int16,        torch.int32,        torch.int64,

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_promoted_dtypes`, `get_dtype_name`, `get_promotion_table_for_strings`, `get_dtype_map`, `get_scalar_promotion_pairs`

**Key imports**: random, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/experimental/torchfuzz`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `random`
- `torch`


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

Files in the same folder (`tools/experimental/torchfuzz`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`codegen.py_docs.md`](./codegen.py_docs.md)
- [`tensor_fuzzer.py_docs.md`](./tensor_fuzzer.py_docs.md)
- [`fuzzer.py_docs.md`](./fuzzer.py_docs.md)
- [`visualize_graph.py_docs.md`](./visualize_graph.py_docs.md)
- [`checks.py_docs.md`](./checks.py_docs.md)
- [`test_determinism.py_docs.md`](./test_determinism.py_docs.md)
- [`ops_fuzzer.py_docs.md`](./ops_fuzzer.py_docs.md)
- [`multi_process_fuzzer.py_docs.md`](./multi_process_fuzzer.py_docs.md)


## Cross-References

- **File Documentation**: `type_promotion.py_docs.md`
- **Keyword Index**: `type_promotion.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/experimental/torchfuzz`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/experimental/torchfuzz`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/tools/experimental/torchfuzz`):

- [`ops_fuzzer.py_docs.md_docs.md`](./ops_fuzzer.py_docs.md_docs.md)
- [`multi_process_fuzzer.py_docs.md_docs.md`](./multi_process_fuzzer.py_docs.md_docs.md)
- [`multi_process_fuzzer.py_kw.md_docs.md`](./multi_process_fuzzer.py_kw.md_docs.md)
- [`checks.py_kw.md_docs.md`](./checks.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`checks.py_docs.md_docs.md`](./checks.py_docs.md_docs.md)
- [`runner.py_docs.md_docs.md`](./runner.py_docs.md_docs.md)
- [`fuzzer.py_kw.md_docs.md`](./fuzzer.py_kw.md_docs.md)
- [`test_determinism.py_kw.md_docs.md`](./test_determinism.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `type_promotion.py_docs.md_docs.md`
- **Keyword Index**: `type_promotion.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
