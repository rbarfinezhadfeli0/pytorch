# Keyword Index: `test/quantization/fx/test_equalize_fx.py`

## File Information

- **Original File**: [test/quantization/fx/test_equalize_fx.py](../../../../test/quantization/fx/test_equalize_fx.py)
- **Documentation**: [`test_equalize_fx.py_docs.md`](./test_equalize_fx.py_docs.md)
- **Folder**: `test/quantization/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`M`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`TestBranchingWithEqualizationModel`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`TestBranchingWithoutEqualizationModel`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`TestEqualizeFx`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)

### Functions

- **`__init__`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`calculate_equalization_scale_ref`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`channel_minmax`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`check_orig_and_eq_graphs`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`forward`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`get_expected_eq_scales`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`get_expected_inp_act_vals`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`get_expected_weight_act_vals`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`get_expected_weights_bias`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_eq_observer`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_equalization_activation_values`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_equalization_branching`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_equalization_convert`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_equalization_equalization_scales`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_equalization_graphs`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_equalization_prepare`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_equalization_results`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_input_weight_equalization_weights_bias`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`test_selective_equalization`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)

### Imports

- **`MinMaxObserver`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`copy`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`default_qconfig`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`given`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`hypothesis`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`numpy`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`prepare_fx`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`raise_on_run_directly`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`strategies`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.ao.nn.intrinsic.quantized`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.ao.nn.quantized`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.ao.quantization`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.ao.quantization.fx._equalize`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.ao.quantization.observer`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.ao.quantization.quantize_fx`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.nn`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.nn.functional`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.testing._internal.common_quantization`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_equalize_fx.py_docs.md](./test_equalize_fx.py_docs.md)


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
