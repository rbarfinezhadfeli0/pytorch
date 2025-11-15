# Keyword Index: `torch/_functorch/autograd_function.py`

## File Information

- **Original File**: [torch/_functorch/autograd_function.py](../../../torch/_functorch/autograd_function.py)
- **Documentation**: [`autograd_function.py_docs.md`](./autograd_function.py_docs.md)
- **Folder**: `torch/_functorch`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ApplyTemplate`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`AutogradFunctionApply`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`CtxCustomSave`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`CtxWithSavedTensors`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`CustomFunctionHigherOrderOperator`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`MyExp`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`Sum`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`VmapInfo`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`VmappedSum`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`WrappedCtx`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`name`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`with`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)

### Functions

- **`__call__`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`__getattr__`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`__init__`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`__setattr__`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`autograd_function_forward_rewritten`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`backward`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`backward_no_context`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`custom_function_call_functionalize`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`custom_function_call_grad`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`custom_function_call_vmap`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`custom_function_call_vmap_generate_rule`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`custom_function_call_vmap_helper`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`forward`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`generate_single_level_function`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`get_tangents_in_dims`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`has_overridden_vmap_rule`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`inner`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`jvp`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`jvp_no_context`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`lower_to_next`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`new_forward`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`reductify`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`reductify_leaf`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`save_for_backward`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`save_for_forward`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`saved_tensors`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`setup_context`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`unpack_outputs`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`validate_vmap_returns_tuple_of_two_elements`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`vmapify_autograd_function`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`wrap_fn`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`wrap_outputs_maintaining_identity`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)

### Imports

- **`HigherOrderOperator`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`NamedTuple`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`_set_fwd_grad_enabled`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`enable_single_level_autograd_function`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`torch`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`torch._C._functorch`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`torch._functorch.apis`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`torch._functorch.utils`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`torch._functorch.vmap`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`torch._ops`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`torch.autograd.forward_ad`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`torch.utils._pytree`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`typing`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)
- **`vmap`**: [autograd_function.py_docs.md](./autograd_function.py_docs.md)


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
