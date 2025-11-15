# Documentation: `docs/test/quantization/pt2e/test_metadata_porting.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/pt2e/test_metadata_porting.py_docs.md`
- **Size**: 25,349 bytes (24.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/pt2e/test_metadata_porting.py`

## File Metadata

- **Path**: `test/quantization/pt2e/test_metadata_porting.py`
- **Size**: 21,478 bytes (20.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]
import copy
import unittest

import torch
import torch._export
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import QuantizationAnnotation, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OP_TO_ANNOTATOR
from torch.fx import Node
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    raise_on_run_directly,
    skipIfCrossRef,
)


class TestHelperModules:
    class Conv2dWithObsSharingOps(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.hardtanh = torch.nn.Hardtanh()
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            x = self.hardtanh(x)
            x = x.view(-1, 3)
            x = self.linear(x)
            return x


def _tag_partitions(
    backend_name: str, op_name: str, annotated_partitions: list[list[Node]]
):
    for index, partition_nodes in enumerate(annotated_partitions):
        tag_name = backend_name + "_" + op_name + "_" + str(index)
        for node in partition_nodes:
            assert "quantization_tag" not in node.meta, f"{node} is already tagged"
            node.meta["quantization_tag"] = tag_name


_QUANT_OPS = {
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
    torch.ops.quantized_decomposed.choose_qparams.tensor,
}


# TODO: rename to TestPortMetadataPass to align with the util name?
@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestMetaDataPorting(QuantizationTestCase):
    def _test_quant_tag_preservation_through_decomp(
        self, model, example_inputs, from_node_to_tags
    ):
        ep = torch.export.export(model, example_inputs, strict=True)
        found_tags = True
        not_found_nodes = ""
        for from_node, tag in from_node_to_tags.items():
            for n in ep.graph_module.graph.nodes:
                from_node_meta = n.meta.get("from_node", None)
                if from_node_meta is None:
                    continue
                if not isinstance(from_node_meta, list):
                    raise ValueError(
                        f"from_node metadata is of type {type(from_node_meta)}, but expected list"
                    )
                for meta in from_node_meta:
                    node_target = meta.target
                    if node_target == str(from_node):
                        node_tag = n.meta.get("quantization_tag", None)
                        if node_tag is None or tag != node_tag:
                            not_found_nodes += str(n.target) + ", "
                            found_tags = False
                            break
                if not found_tags:
                    break
        self.assertTrue(
            found_tags,
            f"Decomposition did not preserve quantization tag for {not_found_nodes}",
        )

    def _test_metadata_porting(
        self,
        model,
        example_inputs,
        quantizer,
        node_tags=None,
    ) -> torch.fx.GraphModule:
        m_eager = model.eval()

        # program capture
        m = copy.deepcopy(m_eager)
        m = torch.export.export(m, example_inputs, strict=True).module()

        m = prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m)

        m(*example_inputs)
        recorded_node_tags = {}
        for n in m.graph.nodes:
            if "quantization_tag" not in n.meta:
                continue
            if n.op == "call_function" and n.target in _QUANT_OPS:
                key = n.target
            elif n.op == "get_attr":
                key = "get_attr"
            else:
                continue

            if key not in recorded_node_tags:
                recorded_node_tags[key] = set()

            if (
                n.op == "call_function"
                and n.meta["quantization_tag"] in recorded_node_tags[key]
            ):
                raise ValueError(
                    f"{key} {n.format_node()} has tag {n.meta['quantization_tag']} that "
                    "is associated with another node of the same type"
                )
            recorded_node_tags[key].add(n.meta["quantization_tag"])

        self.assertEqual(set(recorded_node_tags.keys()), set(node_tags.keys()))
        for k, v in recorded_node_tags.items():
            self.assertEqual(v, node_tags[k])
        return m

    @skipIfCrossRef  # mlazos: retracing FX graph with torch function mode doesn't propagate metadata, because the stack
    # trace of the mode torch function impl doesn't match the traced graph stored lineno.
    def test_simple_metadata_porting(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config
                )
                _tag_partitions(backend_string, "linear", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                _tag_partitions(backend_string, "conv2d", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["adaptive_avg_pool2d"](
                    gm, quantization_config
                )
                _tag_partitions(
                    backend_string, "adaptive_avg_pool2d", annotated_partitions
                )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        get_attr_tags = {
            "BackendA_conv2d_0",
            "BackendA_linear_0",
        }
        quantize_per_tensor_tags = {
            "BackendA_conv2d_0",
            "BackendA_adaptive_avg_pool2d_0",
            "BackendA_linear_0",
        }
        dequantize_per_tensor_tags = {
            "BackendA_adaptive_avg_pool2d_0",
            "BackendA_conv2d_0",
            "BackendA_linear_0",
        }
        dequantize_per_channel_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
        }
        m = self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

        from_node_to_tags = {
            torch.ops.aten.adaptive_avg_pool2d.default: "BackendA_adaptive_avg_pool2d_0",
            torch.ops.aten.linear.default: "BackendA_linear_0",
        }
        self._test_quant_tag_preservation_through_decomp(
            m, example_inputs, from_node_to_tags
        )

    def test_metadata_porting_with_no_quant_inbetween(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Dont quantize avgpool
        Check quantization tags on conv2d and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config
                )
                _tag_partitions(backend_string, "linear", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                _tag_partitions(backend_string, "conv2d", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        get_attr_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        quantize_per_tensor_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        dequantize_per_tensor_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        dequantize_per_channel_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
        }
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

    @unittest.skip("Temporarily disabled")
    def test_metadata_porting_for_dq(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Quantize all except linear.
        Quantize linear with dynamic quantization
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                # static quantiazation
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                _tag_partitions(backend_string, "conv2d", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["adaptive_avg_pool2d"](
                    gm, quantization_config
                )
                _tag_partitions(
                    backend_string, "adaptive_avg_pool2d", annotated_partitions
                )

                # dynamic quantization
                quantization_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config_dynamic
                )
                _tag_partitions(backend_string, "linear_dynamic", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        # TODO: add get_attr_tags when the test is re-enabled
        get_attr_tags = {}
        quantize_per_tensor_tags = {
            "BackendA_conv2d_0",
            "BackendA_adaptive_avg_pool2d_0",
        }
        quantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        choose_qparams_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_tensor_tags = {
            "BackendA_adaptive_avg_pool2d_0",
            "BackendA_conv2d_0",
        }
        dequantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_channel_tags = {
            "BackendA_conv2d_0",
            "BackendA_linear_dynamic_0",
        }
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: dequantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
            torch.ops.quantized_decomposed.choose_qparams.tensor: choose_qparams_tensor_tensor_tags,
        }
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

    def test_metadata_porting_for_two_dq(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Quantize linear and conv with dynamic quantization
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"

                # dynamic quantization
                quantization_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["conv"](
                    gm, quantization_config_dynamic
                )
                _tag_partitions(backend_string, "conv2d_dynamic", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config_dynamic
                )
                _tag_partitions(backend_string, "linear_dynamic", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        get_attr_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        choose_qparams_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        quantize_per_tensor_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        dequantize_per_tensor_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        dequantize_per_channel_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: dequantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
            torch.ops.quantized_decomposed.choose_qparams.tensor: choose_qparams_tensor_tags,
        }
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

    def test_metadata_porting_for_dq_no_static_q(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Dont quantize anything except linear.
        Quantize linear with dynamic quantization
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                # dynamic quantization
                quantization_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config_dynamic
                )
                _tag_partitions(backend_string, "linear_dynamic", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        get_attr_tags = {"BackendA_linear_dynamic_0"}
        choose_qparams_tensor_tags = {"BackendA_linear_dynamic_0"}
        quantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_channel_tags = {"BackendA_linear_dynamic_0"}
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: dequantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
            torch.ops.quantized_decomposed.choose_qparams.tensor: choose_qparams_tensor_tags,
        }
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

    def test_no_metadata_porting(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                OP_TO_ANNOTATOR["linear"](gm, quantization_config)
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                OP_TO_ANNOTATOR["adaptive_avg_pool2d"](gm, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_tags = {}
        m = self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

        from_node_to_tags = {}
        self._test_quant_tag_preservation_through_decomp(
            m, example_inputs, from_node_to_tags
        )

    def test_no_metadata_porting_through_unknown_ops(self):
        """
        Model under test
        matmul -> add -> relu
        matmul has get_attr as first input, but the quantization_tag should not be
        propagated to add even if it's part of a chain that ends at get_attr
        """

        class MatmulWithConstInput(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.rand(8, 16)))

            def forward(self, x, y):
                x = torch.matmul(self.w, x)
                z = x + y
                return torch.nn.functional.relu(z)

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                qconfig = get_symmetric_quantization_config()
                for n in gm.graph.nodes:
                    if n.op != "call_function":
                        continue

                    n.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map={n.args[0]: qconfig.input_activation},
                        output_qspec=qconfig.output_activation,
                    )

                    tag = str(n.target)
                    n.meta["quantization_tag"] = tag
                    for arg in n.args:
                        if arg.op == "get_attr":
                            arg.meta["quantization_tag"] = tag

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(16, 24), torch.randn(8, 24))
        get_attr_tags = {"aten.matmul.default"}
        quantize_per_tensor_tensor_tags = {
            "aten.matmul.default",
            "aten.add.Tensor",
            "aten.relu.default",
        }
        dequantize_per_tensor_tensor_tags = {
            "aten.matmul.default",
            "aten.add.Tensor",
            "aten.relu.default",
        }
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tensor_tags,
        }
        self._test_metadata_porting(
            MatmulWithConstInput(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")

```



## High-Level Overview


This Python file contains 11 class(es) and 28 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestHelperModules`, `Conv2dWithObsSharingOps`, `TestMetaDataPorting`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `MatmulWithConstInput`, `BackendAQuantizer`

**Functions defined**: `__init__`, `forward`, `_tag_partitions`, `_test_quant_tag_preservation_through_decomp`, `_test_metadata_porting`, `test_simple_metadata_porting`, `annotate`, `validate`, `test_metadata_porting_with_no_quant_inbetween`, `annotate`, `validate`, `test_metadata_porting_for_dq`, `annotate`, `validate`, `test_metadata_porting_for_two_dq`, `annotate`, `validate`, `test_metadata_porting_for_dq_no_static_q`, `annotate`, `validate`

**Key imports**: copy, unittest, torch, torch._export, convert_pt2e, prepare_pt2e, QuantizationAnnotation, Quantizer, OP_TO_ANNOTATOR, Node, QuantizationTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/pt2e`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `unittest`
- `torch`
- `torch._export`
- `torch.ao.quantization.quantize_pt2e`: convert_pt2e, prepare_pt2e
- `torch.ao.quantization.quantizer`: QuantizationAnnotation, Quantizer
- `torch.ao.quantization.quantizer.xnnpack_quantizer_utils`: OP_TO_ANNOTATOR
- `torch.fx`: Node
- `torch.testing._internal.common_quantization`: QuantizationTestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/quantization/pt2e/test_metadata_porting.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/pt2e`):

- [`test_graph_utils.py_docs.md`](./test_graph_utils.py_docs.md)
- [`test_numeric_debugger.py_docs.md`](./test_numeric_debugger.py_docs.md)
- [`test_quantize_pt2e.py_docs.md`](./test_quantize_pt2e.py_docs.md)
- [`test_quantize_pt2e_qat.py_docs.md`](./test_quantize_pt2e_qat.py_docs.md)
- [`test_representation.py_docs.md`](./test_representation.py_docs.md)
- [`test_xnnpack_quantizer.py_docs.md`](./test_xnnpack_quantizer.py_docs.md)
- [`test_x86inductor_quantizer.py_docs.md`](./test_x86inductor_quantizer.py_docs.md)
- [`test_duplicate_dq.py_docs.md`](./test_duplicate_dq.py_docs.md)


## Cross-References

- **File Documentation**: `test_metadata_porting.py_docs.md`
- **Keyword Index**: `test_metadata_porting.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/quantization/pt2e`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/pt2e`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/quantization/pt2e/test_metadata_porting.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/pt2e`):

- [`test_representation.py_kw.md_docs.md`](./test_representation.py_kw.md_docs.md)
- [`test_representation.py_docs.md_docs.md`](./test_representation.py_docs.md_docs.md)
- [`test_x86inductor_quantizer.py_docs.md_docs.md`](./test_x86inductor_quantizer.py_docs.md_docs.md)
- [`test_quantize_pt2e_qat.py_docs.md_docs.md`](./test_quantize_pt2e_qat.py_docs.md_docs.md)
- [`test_quantize_pt2e.py_kw.md_docs.md`](./test_quantize_pt2e.py_kw.md_docs.md)
- [`test_x86inductor_quantizer.py_kw.md_docs.md`](./test_x86inductor_quantizer.py_kw.md_docs.md)
- [`test_xnnpack_quantizer.py_kw.md_docs.md`](./test_xnnpack_quantizer.py_kw.md_docs.md)
- [`test_graph_utils.py_kw.md_docs.md`](./test_graph_utils.py_kw.md_docs.md)
- [`test_quantize_pt2e.py_docs.md_docs.md`](./test_quantize_pt2e.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_metadata_porting.py_docs.md_docs.md`
- **Keyword Index**: `test_metadata_porting.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
