# Keyword Index: `docs/aten/src/ATen/native/quantized/cpu/fbgemm_utils.cpp_docs.md`

## File Information

- **Original File**: [docs/aten/src/ATen/native/quantized/cpu/fbgemm_utils.cpp_docs.md](../../../../../../../../docs/aten/src/ATen/native/quantized/cpu/fbgemm_utils.cpp_docs.md)
- **Documentation**: [`fbgemm_utils.cpp_docs.md_docs.md`](./fbgemm_utils.cpp_docs.md_docs.md)
- **Folder**: `docs/aten/src/ATen/native/quantized/cpu`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Identifiers

- **`A`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`ACLUtils`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`AT_ASSERT`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`AT_DISPATCH_QINT_TYPES`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`AT_MKLDNN_ENABLED`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`AT_PER_OPERATOR_HEADERS`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`ATen`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`C`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`ChannelsLast`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Code`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Common`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Considerations`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Context`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Contiguous`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Conv2dPackedParamsBase`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`ConvParamsSerializationType`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`ConvParamsSerializationTypeV3`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`ConvertConvWeightsToChannelLastTensor`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`CopyToChannelsLast3dTensor`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Currently`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Dependencies`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Detailed`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`DispatchKeySet`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Documentation`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Element`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`EmbeddingPackedParams`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`EmbeddingPackedParamsBase`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Examples`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Expected`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Extension`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`FBCODE_CAFFE2`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`FBGEMM`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`For`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`G`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`High`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`IC`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`IC_G`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Index`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`IsChannelsLast3d`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`JIT`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`KB`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`KH`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`KW`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Keyword`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Level`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Library`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`LinearUnpackImpl`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`MakeEmptyAffineQuantizedChannelsLast3dTensor`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`MakeFbgemmConvParam`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`MakeStridedQTensorCPU`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`N`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`NB`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Namespaces`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`No`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Notes`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`OC`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`OC_G`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`ONEDNN`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Original`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Overview`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`PackedEmbeddingBagWeight`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`PackedLinearWeightsOnednn`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`PackedLinearWeightsQnnp`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`PackedParam`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`PackedParams`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Param`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Patterns`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Pooling`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`QEngine`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`QNNPACK`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`QTensorImpl`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`QnnpackUtils`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`QuantizerPtr`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Repository`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Role`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Safety`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`ScalarType`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Security`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Serialization`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Source`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`StorageImpl`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Structure`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`THW`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`TORCH_API`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`TORCH_CHECK`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Tensor`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`TensorBody`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`TensorOperators`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`TensorOptions`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Tensors`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Test`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Testing`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`This`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`TransposeConvTensorUnpackConversion`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`USE_PYTORCH_QNNPACK`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Unknown`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`UpSampleNearest3d`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Uses`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`Utils`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)
- **`XnnpackUtils`**: [fbgemm_utils.cpp_docs.md_docs.md](./fbgemm_utils.cpp_docs.md_docs.md)


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
