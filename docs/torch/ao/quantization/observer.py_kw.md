# Keyword Index: `torch/ao/quantization/observer.py`

## File Information

- **Original File**: [torch/ao/quantization/observer.py](../../../../torch/ao/quantization/observer.py)
- **Documentation**: [`observer.py_docs.md`](./observer.py_docs.md)
- **Folder**: `torch/ao/quantization`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AffineQuantizedObserverBase`**: [observer.py_docs.md](./observer.py_docs.md)
- **`FixedQParamsObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`Granularity`**: [observer.py_docs.md](./observer.py_docs.md)
- **`HistogramObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`MappingType`**: [observer.py_docs.md](./observer.py_docs.md)
- **`MinMaxObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`MovingAverageMinMaxObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`MovingAveragePerChannelMinMaxObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`NoopObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`ObserverBase`**: [observer.py_docs.md](./observer.py_docs.md)
- **`PerAxis`**: [observer.py_docs.md](./observer.py_docs.md)
- **`PerBlock`**: [observer.py_docs.md](./observer.py_docs.md)
- **`PerChannelMinMaxObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`PerGroup`**: [observer.py_docs.md](./observer.py_docs.md)
- **`PerRow`**: [observer.py_docs.md](./observer.py_docs.md)
- **`PerTensor`**: [observer.py_docs.md](./observer.py_docs.md)
- **`PerToken`**: [observer.py_docs.md](./observer.py_docs.md)
- **`PlaceholderObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`RecordingObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`ReuseInputObserver`**: [observer.py_docs.md](./observer.py_docs.md)
- **`TorchAODType`**: [observer.py_docs.md](./observer.py_docs.md)
- **`UniformQuantizationObserverBase`**: [observer.py_docs.md](./observer.py_docs.md)
- **`ZeroPointDomain`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_PartialWrapper`**: [observer.py_docs.md](./observer.py_docs.md)
- **`buffers`**: [observer.py_docs.md](./observer.py_docs.md)
- **`factories`**: [observer.py_docs.md](./observer.py_docs.md)
- **`for`**: [observer.py_docs.md](./observer.py_docs.md)
- **`from`**: [observer.py_docs.md](./observer.py_docs.md)
- **`of`**: [observer.py_docs.md](./observer.py_docs.md)
- **`only`**: [observer.py_docs.md](./observer.py_docs.md)
- **`serves`**: [observer.py_docs.md](./observer.py_docs.md)
- **`was`**: [observer.py_docs.md](./observer.py_docs.md)

### Functions

- **`__call__`**: [observer.py_docs.md](./observer.py_docs.md)
- **`__init__`**: [observer.py_docs.md](./observer.py_docs.md)
- **`__repr__`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_calculate_qparams`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_combine_histograms`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_compute_quantization_error`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_forward`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_get_norm`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_is_activation_post_process`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_is_observer_script_module`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_is_per_channel_script_obs_instance`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_load_from_state_dict`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_load_from_state_dict_script`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_non_linear_param_search`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_save_to_state_dict`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_upscale_histogram`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_validate_qmin_qmax`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_with_args`**: [observer.py_docs.md](./observer.py_docs.md)
- **`_with_callable_args`**: [observer.py_docs.md](./observer.py_docs.md)
- **`calculate_qparams`**: [observer.py_docs.md](./observer.py_docs.md)
- **`convert`**: [observer.py_docs.md](./observer.py_docs.md)
- **`extra_repr`**: [observer.py_docs.md](./observer.py_docs.md)
- **`forward`**: [observer.py_docs.md](./observer.py_docs.md)
- **`get_block_size`**: [observer.py_docs.md](./observer.py_docs.md)
- **`get_observer_state_dict`**: [observer.py_docs.md](./observer.py_docs.md)
- **`get_tensor_value`**: [observer.py_docs.md](./observer.py_docs.md)
- **`load_observer_state_dict`**: [observer.py_docs.md](./observer.py_docs.md)
- **`reset_histogram`**: [observer.py_docs.md](./observer.py_docs.md)
- **`reset_min_max_vals`**: [observer.py_docs.md](./observer.py_docs.md)
- **`with_args`**: [observer.py_docs.md](./observer.py_docs.md)
- **`with_callable_args`**: [observer.py_docs.md](./observer.py_docs.md)

### Imports

- **`ABCMeta`**: [observer.py_docs.md](./observer.py_docs.md)
- **`Any`**: [observer.py_docs.md](./observer.py_docs.md)
- **`Node`**: [observer.py_docs.md](./observer.py_docs.md)
- **`OrderedDict`**: [observer.py_docs.md](./observer.py_docs.md)
- **`abc`**: [observer.py_docs.md](./observer.py_docs.md)
- **`auto`**: [observer.py_docs.md](./observer.py_docs.md)
- **`collections`**: [observer.py_docs.md](./observer.py_docs.md)
- **`create_getattr_from_value`**: [observer.py_docs.md](./observer.py_docs.md)
- **`dataclass`**: [observer.py_docs.md](./observer.py_docs.md)
- **`dataclasses`**: [observer.py_docs.md](./observer.py_docs.md)
- **`enum`**: [observer.py_docs.md](./observer.py_docs.md)
- **`functools`**: [observer.py_docs.md](./observer.py_docs.md)
- **`operator`**: [observer.py_docs.md](./observer.py_docs.md)
- **`partial`**: [observer.py_docs.md](./observer.py_docs.md)
- **`re`**: [observer.py_docs.md](./observer.py_docs.md)
- **`torch`**: [observer.py_docs.md](./observer.py_docs.md)
- **`torch.ao.quantization.fx.utils`**: [observer.py_docs.md](./observer.py_docs.md)
- **`torch.ao.quantization.utils`**: [observer.py_docs.md](./observer.py_docs.md)
- **`torch.fx`**: [observer.py_docs.md](./observer.py_docs.md)
- **`torch.nn`**: [observer.py_docs.md](./observer.py_docs.md)
- **`typing`**: [observer.py_docs.md](./observer.py_docs.md)
- **`warnings`**: [observer.py_docs.md](./observer.py_docs.md)


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
