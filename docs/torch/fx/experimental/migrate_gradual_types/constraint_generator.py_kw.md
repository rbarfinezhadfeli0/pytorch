# Keyword Index: `torch/fx/experimental/migrate_gradual_types/constraint_generator.py`

## File Information

- **Original File**: [torch/fx/experimental/migrate_gradual_types/constraint_generator.py](../../../../../torch/fx/experimental/migrate_gradual_types/constraint_generator.py)
- **Documentation**: [`constraint_generator.py_docs.md`](./constraint_generator.py_docs.md)
- **Folder**: `torch/fx/experimental/migrate_gradual_types`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ConstraintGenerator`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)

### Functions

- **`__init__`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`adaptive_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`add_layer_norm_constraints`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`add_linear_constraints`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`arange_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`assert_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`batchnorm_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`bmm_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`broadcasting_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`conv2d_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`cumsum_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`embedding_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`embedding_inference_rule_functional`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`eq_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`equality_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`expand_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`flatten_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`full_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`gen_broadcasting_constraints`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`gen_embedding_rules`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`gen_layer_norm_constraints`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`generate_constraints`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`generate_constraints_node`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`generate_flatten_constraints`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`get_attr_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`getitem_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`gt_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`index_select_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`layer_norm_functional`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`layer_norm_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`linear_constraints`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`linear_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`lt_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`masked_fill_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`maxpool_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`neq_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`range_check`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`register`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`register_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`relu_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`reshape_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`size_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`tensor_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch_dim_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch_linear_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`transpose_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`type_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`view_inference_rule`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)

### Imports

- **`BatchNorm2d`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`Callable`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`Conv2d`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`Dyn`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`Node`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`ParamSpec`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`TypeVar`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`_assert_is_none`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`collections.abc`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`operator`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch.fx._symbolic_trace`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch.fx.experimental.migrate_gradual_types.constraint`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch.fx.experimental.migrate_gradual_types.operation`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch.fx.experimental.migrate_gradual_types.util`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch.fx.node`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch.fx.tensor_type`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch.nn.modules.batchnorm`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`torch.nn.modules.conv`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`typing`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`typing_extensions`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)
- **`warnings`**: [constraint_generator.py_docs.md](./constraint_generator.py_docs.md)


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
