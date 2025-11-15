# Keyword Index: `torch/distributed/_tools/sac_estimator.py`

## File Information

- **Original File**: [torch/distributed/_tools/sac_estimator.py](../../../../torch/distributed/_tools/sac_estimator.py)
- **Documentation**: [`sac_estimator.py_docs.md`](./sac_estimator.py_docs.md)
- **Folder**: `torch/distributed/_tools`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MSPS`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`SACEstimator`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`class`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`for`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`from`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`is`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`provides`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)

### Functions

- **`__call__`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`__enter__`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`__exit__`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`__init__`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`__torch_dispatch__`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_display_stats_tabular`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_get_force_store_random`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_get_greedy_order_meta`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_get_inplace_metadata`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_get_sac_stats`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_get_sac_tradeoff_pwlf_stats`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_pack_hook`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_post_fw_hook`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`_pre_fw_hook`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`append_row`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`display_modulewise_sac_stats`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`display_sac_stats`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`display_sac_tradeoff_stats`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`pwlf_sac_tradeoff_curve`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`save_prediction_graph`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)

### Imports

- **`Any`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`FakeTensorMode`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`ModTracker`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`OrderedDict`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`RuntimeEstimator`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`SAC_IGNORED_OPS`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`Self`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`TorchDispatchMode`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`active_fake_mode`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`astuple`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`collections`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`dataclasses`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`get_untyped_storages`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`math`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`matplotlib.pyplot`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`nan`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`numpy`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`os`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`pwlf`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`sys`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`tabulate`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch._guards`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch.distributed._tools.common_utils`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch.distributed._tools.mod_tracker`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch.distributed._tools.runtime_estimator`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch.testing._internal.composite_compliance`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch.utils._python_dispatch`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch.utils._pytree`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`torch.utils.checkpoint`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`tree_flatten`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`typing`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)
- **`typing_extensions`**: [sac_estimator.py_docs.md](./sac_estimator.py_docs.md)


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
