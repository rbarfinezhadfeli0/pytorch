# Keyword Index: `torch/testing/_internal/composite_compliance.py`

## File Information

- **Original File**: [torch/testing/_internal/composite_compliance.py](../../../../torch/testing/_internal/composite_compliance.py)
- **Documentation**: [`composite_compliance.py_docs.md`](./composite_compliance.py_docs.md)
- **Folder**: `torch/testing/_internal`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CompositeCompliantTensor`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`CompositeCompliantTensorMode`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`which`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)

### Functions

- **`__new__`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`__repr__`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`__torch_dispatch__`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`check_all_permutations`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`check_attr_consistency`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`check_backward_formula`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`check_forward_ad_formula`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`check_metadata_consistency`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`check_with_mode`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`compute_expected_grad`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`compute_expected_grads`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`gather_leaf_tensors`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`generate_cct_and_mode`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`generate_subclass_choices`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`generate_subclass_choices_args_kwargs`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`is_inplace`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`is_inplace_view_fn`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`is_tensorlist`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`is_view_fn`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`maybe_make_dual`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`maybe_map`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`maybe_tangent`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`raise_composite_compliance_error`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`unwrap`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`wrap`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)

### Imports

- **`Callable`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`Tensor`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`TorchDispatchMode`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`_pytree`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`collections.abc`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`functools`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`itertools`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`no_dispatch`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`partial`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`re`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`torch`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`torch.autograd.forward_ad`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`torch.utils`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`torch.utils._mode_utils`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`torch.utils._python_dispatch`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`torch.utils._pytree`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)
- **`tree_map`**: [composite_compliance.py_docs.md](./composite_compliance.py_docs.md)


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
