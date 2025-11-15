# Documentation: `docs/test/quantization/ao_migration/test_quantization.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/ao_migration/test_quantization.py_docs.md`
- **Size**: 10,846 bytes (10.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/ao_migration/test_quantization.py`

## File Metadata

- **Path**: `test/quantization/ao_migration/test_quantization.py`
- **Size**: 7,954 bytes (7.77 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

from torch.testing._internal.common_utils import raise_on_run_directly

from .common import AOMigrationTestCase


class TestAOMigrationQuantization(AOMigrationTestCase):
    r"""Modules and functions related to the
    `torch/quantization` migration to `torch/ao/quantization`.
    """

    def test_function_import_quantize(self):
        function_list = [
            "_convert",
            "_observer_forward_hook",
            "_propagate_qconfig_helper",
            "_remove_activation_post_process",
            "_remove_qconfig",
            "_add_observer_",
            "add_quant_dequant",
            "convert",
            "_get_observer_dict",
            "_get_unique_devices_",
            "_is_activation_post_process",
            "prepare",
            "prepare_qat",
            "propagate_qconfig_",
            "quantize",
            "quantize_dynamic",
            "quantize_qat",
            "_register_activation_post_process_hook",
            "swap_module",
        ]
        self._test_function_import("quantize", function_list)

    def test_function_import_stubs(self):
        function_list = [
            "QuantStub",
            "DeQuantStub",
            "QuantWrapper",
        ]
        self._test_function_import("stubs", function_list)

    def test_function_import_quantize_jit(self):
        function_list = [
            "_check_is_script_module",
            "_check_forward_method",
            "script_qconfig",
            "script_qconfig_dict",
            "fuse_conv_bn_jit",
            "_prepare_jit",
            "prepare_jit",
            "prepare_dynamic_jit",
            "_convert_jit",
            "convert_jit",
            "convert_dynamic_jit",
            "_quantize_jit",
            "quantize_jit",
            "quantize_dynamic_jit",
        ]
        self._test_function_import("quantize_jit", function_list)

    def test_function_import_fake_quantize(self):
        function_list = [
            "_is_per_channel",
            "_is_per_tensor",
            "_is_symmetric_quant",
            "FakeQuantizeBase",
            "FakeQuantize",
            "FixedQParamsFakeQuantize",
            "FusedMovingAvgObsFakeQuantize",
            "default_fake_quant",
            "default_weight_fake_quant",
            "default_fixed_qparams_range_neg1to1_fake_quant",
            "default_fixed_qparams_range_0to1_fake_quant",
            "default_per_channel_weight_fake_quant",
            "default_histogram_fake_quant",
            "default_fused_act_fake_quant",
            "default_fused_wt_fake_quant",
            "default_fused_per_channel_wt_fake_quant",
            "_is_fake_quant_script_module",
            "disable_fake_quant",
            "enable_fake_quant",
            "disable_observer",
            "enable_observer",
        ]
        self._test_function_import("fake_quantize", function_list)

    def test_function_import_fuse_modules(self):
        function_list = [
            "_fuse_modules",
            "_get_module",
            "_set_module",
            "fuse_conv_bn",
            "fuse_conv_bn_relu",
            "fuse_known_modules",
            "fuse_modules",
            "get_fuser_method",
        ]
        self._test_function_import("fuse_modules", function_list)

    def test_function_import_quant_type(self):
        function_list = [
            "QuantType",
            "_get_quant_type_to_str",
        ]
        self._test_function_import("quant_type", function_list)

    def test_function_import_observer(self):
        function_list = [
            "_PartialWrapper",
            "_with_args",
            "_with_callable_args",
            "ABC",
            "ObserverBase",
            "_ObserverBase",
            "MinMaxObserver",
            "MovingAverageMinMaxObserver",
            "PerChannelMinMaxObserver",
            "MovingAveragePerChannelMinMaxObserver",
            "HistogramObserver",
            "PlaceholderObserver",
            "RecordingObserver",
            "NoopObserver",
            "_is_activation_post_process",
            "_is_per_channel_script_obs_instance",
            "get_observer_state_dict",
            "load_observer_state_dict",
            "default_observer",
            "default_placeholder_observer",
            "default_debug_observer",
            "default_weight_observer",
            "default_histogram_observer",
            "default_per_channel_weight_observer",
            "default_dynamic_quant_observer",
            "default_float_qparams_observer",
        ]
        self._test_function_import("observer", function_list)

    def test_function_import_qconfig(self):
        function_list = [
            "QConfig",
            "default_qconfig",
            "default_debug_qconfig",
            "default_per_channel_qconfig",
            "QConfigDynamic",
            "default_dynamic_qconfig",
            "float16_dynamic_qconfig",
            "float16_static_qconfig",
            "per_channel_dynamic_qconfig",
            "float_qparams_weight_only_qconfig",
            "default_qat_qconfig",
            "default_weight_only_qconfig",
            "default_activation_only_qconfig",
            "default_qat_qconfig_v2",
            "get_default_qconfig",
            "get_default_qat_qconfig",
            "_assert_valid_qconfig",
            "QConfigAny",
            "_add_module_to_qconfig_obs_ctr",
            "qconfig_equals",
        ]
        self._test_function_import("qconfig", function_list)

    def test_function_import_quantization_mappings(self):
        function_list = [
            "no_observer_set",
            "get_default_static_quant_module_mappings",
            "get_static_quant_module_class",
            "get_dynamic_quant_module_class",
            "get_default_qat_module_mappings",
            "get_default_dynamic_quant_module_mappings",
            "get_default_qconfig_propagation_list",
            "get_default_compare_output_module_list",
            "get_default_float_to_quantized_operator_mappings",
            "get_quantized_operator",
            "_get_special_act_post_process",
            "_has_special_act_post_process",
        ]
        dict_list = [
            "DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS",
            "DEFAULT_STATIC_QUANT_MODULE_MAPPINGS",
            "DEFAULT_QAT_MODULE_MAPPINGS",
            "DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS",
            # "_INCLUDE_QCONFIG_PROPAGATE_LIST",
            "DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS",
            "DEFAULT_MODULE_TO_ACT_POST_PROCESS",
        ]
        self._test_function_import("quantization_mappings", function_list)
        self._test_dict_import("quantization_mappings", dict_list)

    def test_function_import_fuser_method_mappings(self):
        function_list = [
            "fuse_conv_bn",
            "fuse_conv_bn_relu",
            "fuse_linear_bn",
            "get_fuser_method",
        ]
        dict_list = ["_DEFAULT_OP_LIST_TO_FUSER_METHOD"]
        self._test_function_import("fuser_method_mappings", function_list)
        self._test_dict_import("fuser_method_mappings", dict_list)

    def test_function_import_utils(self):
        function_list = [
            "activation_dtype",
            "activation_is_int8_quantized",
            "activation_is_statically_quantized",
            "calculate_qmin_qmax",
            "check_min_max_valid",
            "get_combined_dict",
            "get_qconfig_dtypes",
            "get_qparam_dict",
            "get_quant_type",
            "get_swapped_custom_module_class",
            "getattr_from_fqn",
            "is_per_channel",
            "is_per_tensor",
            "weight_dtype",
            "weight_is_quantized",
            "weight_is_statically_quantized",
        ]
        self._test_function_import("utils", function_list)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")

```



## High-Level Overview

r"""Modules and functions related to the    `torch/quantization` migration to `torch/ao/quantization`.

This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAOMigrationQuantization`

**Functions defined**: `test_function_import_quantize`, `test_function_import_stubs`, `test_function_import_quantize_jit`, `test_function_import_fake_quantize`, `test_function_import_fuse_modules`, `test_function_import_quant_type`, `test_function_import_observer`, `test_function_import_qconfig`, `test_function_import_quantization_mappings`, `test_function_import_fuser_method_mappings`, `test_function_import_utils`

**Key imports**: raise_on_run_directly, AOMigrationTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/ao_migration`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.testing._internal.common_utils`: raise_on_run_directly
- `.common`: AOMigrationTestCase


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/quantization/ao_migration/test_quantization.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/ao_migration`):

- [`test_ao_migration.py_docs.md`](./test_ao_migration.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_quantization_fx.py_docs.md`](./test_quantization_fx.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)


## Cross-References

- **File Documentation**: `test_quantization.py_docs.md`
- **Keyword Index**: `test_quantization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/quantization/ao_migration`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/ao_migration`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/quantization/ao_migration/test_quantization.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/ao_migration`):

- [`test_quantization_fx.py_kw.md_docs.md`](./test_quantization_fx.py_kw.md_docs.md)
- [`common.py_docs.md_docs.md`](./common.py_docs.md_docs.md)
- [`common.py_kw.md_docs.md`](./common.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`test_ao_migration.py_kw.md_docs.md`](./test_ao_migration.py_kw.md_docs.md)
- [`test_quantization_fx.py_docs.md_docs.md`](./test_quantization_fx.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`test_quantization.py_kw.md_docs.md`](./test_quantization.py_kw.md_docs.md)
- [`test_ao_migration.py_docs.md_docs.md`](./test_ao_migration.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_quantization.py_docs.md_docs.md`
- **Keyword Index**: `test_quantization.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
