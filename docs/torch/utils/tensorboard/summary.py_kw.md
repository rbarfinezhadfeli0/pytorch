# Keyword Index: `torch/utils/tensorboard/summary.py`

## File Information

- **Original File**: [torch/utils/tensorboard/summary.py](../../../../torch/utils/tensorboard/summary.py)
- **Documentation**: [`summary.py_docs.md`](./summary.py_docs.md)
- **Folder**: `torch/utils/tensorboard`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_calc_scale_factor`**: [summary.py_docs.md](./summary.py_docs.md)
- **`_draw_single_box`**: [summary.py_docs.md](./summary.py_docs.md)
- **`_get_json_config`**: [summary.py_docs.md](./summary.py_docs.md)
- **`_get_tensor_summary`**: [summary.py_docs.md](./summary.py_docs.md)
- **`_tensor_to_complex_val`**: [summary.py_docs.md](./summary.py_docs.md)
- **`_tensor_to_half_val`**: [summary.py_docs.md](./summary.py_docs.md)
- **`_tensor_to_list`**: [summary.py_docs.md](./summary.py_docs.md)
- **`audio`**: [summary.py_docs.md](./summary.py_docs.md)
- **`compute_curve`**: [summary.py_docs.md](./summary.py_docs.md)
- **`custom_scalars`**: [summary.py_docs.md](./summary.py_docs.md)
- **`draw_boxes`**: [summary.py_docs.md](./summary.py_docs.md)
- **`half_to_int`**: [summary.py_docs.md](./summary.py_docs.md)
- **`histogram`**: [summary.py_docs.md](./summary.py_docs.md)
- **`histogram_raw`**: [summary.py_docs.md](./summary.py_docs.md)
- **`hparams`**: [summary.py_docs.md](./summary.py_docs.md)
- **`image`**: [summary.py_docs.md](./summary.py_docs.md)
- **`image_boxes`**: [summary.py_docs.md](./summary.py_docs.md)
- **`int_to_half`**: [summary.py_docs.md](./summary.py_docs.md)
- **`make_histogram`**: [summary.py_docs.md](./summary.py_docs.md)
- **`make_image`**: [summary.py_docs.md](./summary.py_docs.md)
- **`make_video`**: [summary.py_docs.md](./summary.py_docs.md)
- **`mesh`**: [summary.py_docs.md](./summary.py_docs.md)
- **`pr_curve`**: [summary.py_docs.md](./summary.py_docs.md)
- **`pr_curve_raw`**: [summary.py_docs.md](./summary.py_docs.md)
- **`scalar`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensor_proto`**: [summary.py_docs.md](./summary.py_docs.md)
- **`text`**: [summary.py_docs.md](./summary.py_docs.md)
- **`video`**: [summary.py_docs.md](./summary.py_docs.md)

### Imports

- **`._convert_np`**: [summary.py_docs.md](./summary.py_docs.md)
- **`._utils`**: [summary.py_docs.md](./summary.py_docs.md)
- **`Any`**: [summary.py_docs.md](./summary.py_docs.md)
- **`Image`**: [summary.py_docs.md](./summary.py_docs.md)
- **`ImageDraw`**: [summary.py_docs.md](./summary.py_docs.md)
- **`MeshPluginData`**: [summary.py_docs.md](./summary.py_docs.md)
- **`PIL`**: [summary.py_docs.md](./summary.py_docs.md)
- **`PrCurvePluginData`**: [summary.py_docs.md](./summary.py_docs.md)
- **`TensorProto`**: [summary.py_docs.md](./summary.py_docs.md)
- **`TensorShapeProto`**: [summary.py_docs.md](./summary.py_docs.md)
- **`TextPluginData`**: [summary.py_docs.md](./summary.py_docs.md)
- **`_prepare_video`**: [summary.py_docs.md](./summary.py_docs.md)
- **`editor`**: [summary.py_docs.md](./summary.py_docs.md)
- **`google.protobuf`**: [summary.py_docs.md](./summary.py_docs.md)
- **`io`**: [summary.py_docs.md](./summary.py_docs.md)
- **`json`**: [summary.py_docs.md](./summary.py_docs.md)
- **`layout_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`logging`**: [summary.py_docs.md](./summary.py_docs.md)
- **`make_np`**: [summary.py_docs.md](./summary.py_docs.md)
- **`metadata`**: [summary.py_docs.md](./summary.py_docs.md)
- **`moviepy`**: [summary.py_docs.md](./summary.py_docs.md)
- **`moviepy.editor.`**: [summary.py_docs.md](./summary.py_docs.md)
- **`numpy`**: [summary.py_docs.md](./summary.py_docs.md)
- **`struct`**: [summary.py_docs.md](./summary.py_docs.md)
- **`struct_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tempfile`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.compat.proto.summary_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.compat.proto.tensor_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.compat.proto.tensor_shape_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.plugins.custom_scalar`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.plugins.hparams.api_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.plugins.hparams.metadata`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.plugins.hparams.plugin_data_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.plugins.mesh`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.plugins.mesh.plugin_data_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.plugins.pr_curve.plugin_data_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`tensorboard.plugins.text.plugin_data_pb2`**: [summary.py_docs.md](./summary.py_docs.md)
- **`torch`**: [summary.py_docs.md](./summary.py_docs.md)
- **`typing`**: [summary.py_docs.md](./summary.py_docs.md)
- **`wave`**: [summary.py_docs.md](./summary.py_docs.md)


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
