# Keyword Index: `torch/onnx/_internal/fx/passes/type_promotion.py`

## File Information

- **Original File**: [torch/onnx/_internal/fx/passes/type_promotion.py](../../../../../../torch/onnx/_internal/fx/passes/type_promotion.py)
- **Documentation**: [`type_promotion.py_docs.md`](./type_promotion.py_docs.md)
- **Folder**: `torch/onnx/_internal/fx/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AllOrAnyReductionTypePromotionRule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`DivElementwiseTypePromotionRule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`ElementwiseTypePromotionRule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`ElementwiseTypePromotionRuleSetGenerator`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`InsertTypePromotion`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`ReductionTypePromotionRule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`SumLikeReductionTypePromotionRule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`TypePromotionRule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`TypePromotionTable`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_OpTraceDispatchMode`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_TypePromotionInterpreter`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`class`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`for`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`needs`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`of`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`that`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)

### Functions

- **`__eq__`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`__hash__`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`__init__`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`__repr__`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`__torch_dispatch__`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_consolidate_input_dtype`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_create_node`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_fetch_fake_args`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_maybe_promote_arg`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_maybe_promote_node`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_parse_torch_refs`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_parse_type_promotion_rule_from_refs_op`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_rerun_node_after_type_promotion`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_run`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_run_node_and_set_meta`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_try_getclosurevars`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`add_rule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`find_compatible_op_overload`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`generate_from_torch_refs`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`get_rule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`get_type_promotion_rule`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`is_valid`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`preview_type_promotion`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`run_node`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)

### Imports

- **`Any`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`Callable`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`ModuleType`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`__future__`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_pass`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_prims_common`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`_python_dispatch`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`abc`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`annotations`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`collections.abc`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`dataclasses`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`fake_tensor`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`functional`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`inspect`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`linalg`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`logging`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`proxy_tensor`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch._dispatch.python`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch._ops`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch._prims_common`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch._refs`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch._refs.nn`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch._subclasses`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch.fx`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch.fx.experimental`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch.fx.traceback`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch.onnx._internal.fx`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`torch.utils`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`types`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)
- **`typing`**: [type_promotion.py_docs.md](./type_promotion.py_docs.md)


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
