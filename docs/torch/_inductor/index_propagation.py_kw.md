# Keyword Index: `torch/_inductor/index_propagation.py`

## File Information

- **Original File**: [torch/_inductor/index_propagation.py](../../../torch/_inductor/index_propagation.py)
- **Documentation**: [`index_propagation.py_docs.md`](./index_propagation.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`IndexPropagation`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`SymPyOps`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`class`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`from`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)

### Functions

- **`__init__`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`__post_init__`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`_default`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`_is_constant`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`abs`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`add`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`constant`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`fallback`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`floordiv`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`identity`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`index_expr`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`indirect_indexing`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`is_constant`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`materialize_expr`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`maximum`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`minimum`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`mod`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`mul`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`neg`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`new_symbolic`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`propagate_sympy`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`remainder`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`square`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`statically_true`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`sub`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`to_dtype`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`unwrap`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`upper_bound`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`wrap`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`wrap_expr`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)

### Imports

- **`.ops_handler`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`.sizevars`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`.utils`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`.virtualized`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`Any`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`DefaultHandler`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`FloorDiv`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`Sequence`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`V`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`bound_sympy`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`collections.abc`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`dataclass`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`dataclasses`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`dtype_to_type`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`generate_assert`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`itertools`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`statically_known_true`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`sympy`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`torch`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`torch._prims_common`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`torch.utils._sympy.functions`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`torch.utils._sympy.value_ranges`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)
- **`typing`**: [index_propagation.py_docs.md](./index_propagation.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
