# Keyword Index: `test/inductor/test_custom_post_grad_passes.py`

## File Information

- **Original File**: [test/inductor/test_custom_post_grad_passes.py](../../../test/inductor/test_custom_post_grad_passes.py)
- **Documentation**: [`test_custom_post_grad_passes.py_docs.md`](./test_custom_post_grad_passes.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ChangeCosCustomPass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`CustomBackendPass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`TestCustomPassBase`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`TestPostGradCustomPrePostPass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`_ConvReLU`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`_CustomPass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)

### Functions

- **`__call__`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`__init__`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`_clone_inputs`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`_mkldnn_conv_relu_pattern`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`_register_fusion_lowering`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`_register_mkldnn_conv_relu_fusion`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`_test_common`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`change_cos_pass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`clone`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`dummy_check`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`f`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`fn`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`forward`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`g`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`inner_test`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`merge_mm_shared_rhs`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`register_custom_lowering_pattern`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`test_custom_backend_pass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`test_custom_joint_pass_post`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`test_custom_joint_pass_pre`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`test_custom_post_pass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`test_custom_pre_grad_pass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`test_custom_pre_pass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`uuid`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)

### Imports

- **`Arg`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`HAS_CPU`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`IS_LINUX`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`collections`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`config`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`contextlib`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`counters`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`defaultdict`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`get_custom_backend_pass_for_device`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`lowerings`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`operator`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`run_tests`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch._dynamo.utils`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch._inductor`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch._inductor.codegen.common`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch._inductor.custom_graph_pass`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch._inductor.lowering`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch._inductor.pattern_matcher`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch._inductor.test_case`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch.fx`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_custom_post_grad_passes.py_docs.md](./test_custom_post_grad_passes.py_docs.md)


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
