# Keyword Index: `torch/testing/_internal/jit_metaprogramming_utils.py`

## File Information

- **Original File**: [torch/testing/_internal/jit_metaprogramming_utils.py](../../../../torch/testing/_internal/jit_metaprogramming_utils.py)
- **Documentation**: [`jit_metaprogramming_utils.py_docs.md`](./jit_metaprogramming_utils.py_docs.md)
- **Folder**: `torch/testing/_internal`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`SplitInputs`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`TheModule`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`dont_convert`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)

### Functions

- **`__init__`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`_is_tensor_input`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`check_alias_annotation`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`conjugate`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`create_input`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`create_script_fn`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`create_script_module`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`create_traced_fn`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`forward`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`gen_script_fn_and_args`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_all_nn_module_tests`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_call`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_constant`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_nn_functional_compiled_fn_and_inputs`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_nn_functional_tests`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_nn_mod_test_name`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_nn_module_class_from_kwargs`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_nn_module_name_from_kwargs`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`get_script_args`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`make_module`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`map_arg`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`maybe_non_contig`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`new_fn`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`nontensors_match`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`partial_apply_nontensors`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`script_fn`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`script_module`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`the_method`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`traced_fn`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`try_get_nn_module_compiled_mod_and_inputs`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`unpack_variables`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`value_to_literal`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)

### Imports

- **`Any`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`BroadcastingList2`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`collections`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`copy`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`deepcopy`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`inf`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`is_iterable_of_tensors`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`math`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`module_tests`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch.cuda`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch.jit`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch.jit._logging`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch.jit.annotations`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch.jit.frontend`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch.nn.functional`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch.testing._internal.common_nn`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`torch.testing._internal.common_utils`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)
- **`typing`**: [jit_metaprogramming_utils.py_docs.md](./jit_metaprogramming_utils.py_docs.md)


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
