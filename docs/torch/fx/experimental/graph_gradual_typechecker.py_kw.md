# Keyword Index: `torch/fx/experimental/graph_gradual_typechecker.py`

## File Information

- **Original File**: [torch/fx/experimental/graph_gradual_typechecker.py](../../../../torch/fx/experimental/graph_gradual_typechecker.py)
- **Documentation**: [`graph_gradual_typechecker.py_docs.md`](./graph_gradual_typechecker.py_docs.md)
- **Folder**: `torch/fx/experimental`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GraphTypeChecker`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`Refine`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)

### Functions

- **`__init__`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`adaptiveavgpool2d_check`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`adaptiveavgpool2d_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`add_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`all_eq`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`bn2d_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`broadcast_types`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`calculate_out_dimension`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`conv2d_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`conv_refinement_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`conv_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`convert_to_sympy_symbols`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`element_wise_eq`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`expand_to_tensor_dim`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`first_two_eq`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`flatten_check`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`flatten_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`flatten_refinement_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`get_attr_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`get_greatest_upper_bound`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`get_node_type`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`get_parameter`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`infer_symbolic_relations`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`linear_check`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`linear_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`linear_refinement_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`maxpool2d_check`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`maxpool2d_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`refine`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`refine_node`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`register`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`register_algebraic_expressions_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`register_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`register_refinement_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`relu_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`replace_dyn_with_fresh_var`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`reshape_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`symbolic_relations`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`transpose_inference_rule`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`type_check`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`type_check_node`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)

### Imports

- **`BatchNorm2d`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`Callable`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`Conv2d`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`Dyn`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`Equality`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`Node`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`ParamSpec`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`TypeVar`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`Var`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`collections.abc`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`functools`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`itertools`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`operator`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`reduce`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`sympy`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`torch`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`torch.fx.experimental.refinement_types`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`torch.fx.experimental.unification`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`torch.fx.node`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`torch.fx.tensor_type`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`torch.nn.modules.batchnorm`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`torch.nn.modules.conv`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`typing`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)
- **`typing_extensions`**: [graph_gradual_typechecker.py_docs.md](./graph_gradual_typechecker.py_docs.md)


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
