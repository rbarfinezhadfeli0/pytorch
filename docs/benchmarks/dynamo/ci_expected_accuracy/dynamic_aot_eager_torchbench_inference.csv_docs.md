# Documentation: `benchmarks/dynamo/ci_expected_accuracy/dynamic_aot_eager_torchbench_inference.csv`

## File Metadata

- **Path**: `benchmarks/dynamo/ci_expected_accuracy/dynamic_aot_eager_torchbench_inference.csv`
- **Size**: 2,311 bytes (2.26 KB)
- **Type**: Source File (.csv)
- **Extension**: `.csv`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
name,accuracy,graph_breaks



torchrec_dlrm,eager_fail_to_run,0



BERT_pytorch,pass,0



Background_Matting,pass_due_to_skip,0



LearningToPaint,pass,0



Super_SloMo,pass,0



alexnet,pass,0



basic_gnn_edgecnn,pass,0



basic_gnn_gcn,pass,6



basic_gnn_gin,pass,0



basic_gnn_sage,pass,0



cm3leon_generate,pass,4



dcgan,pass,0



demucs,pass,3



densenet121,pass,0



detectron2_fasterrcnn_r_101_c4,eager_fail_to_run,0



detectron2_fasterrcnn_r_101_dc5,eager_fail_to_run,0



detectron2_fasterrcnn_r_101_fpn,eager_fail_to_run,0



detectron2_fasterrcnn_r_50_c4,eager_fail_to_run,0



detectron2_fasterrcnn_r_50_dc5,eager_fail_to_run,0



detectron2_fasterrcnn_r_50_fpn,eager_fail_to_run,0



detectron2_fcos_r_50_fpn,pass,22



detectron2_maskrcnn_r_101_c4,eager_fail_to_run,0



detectron2_maskrcnn_r_101_fpn,eager_fail_to_run,0



detectron2_maskrcnn_r_50_c4,eager_fail_to_run,0



detectron2_maskrcnn_r_50_fpn,eager_fail_to_run,0



dlrm,pass,0



doctr_det_predictor,pass,3



doctr_reco_predictor,pass,1



drq,pass,0



fastNLP_Bert,pass,4



functorch_dp_cifar10,pass,0



functorch_maml_omniglot,pass,0



lennard_jones,pass,0



llama,pass,0



llama_v2_7b_16h,model_fail_to_load,0



llava,model_fail_to_load,0



maml,pass_due_to_skip,0



maml_omniglot,pass,0



microbench_unbacked_tolist_sum,pass,2



mnasnet1_0,pass,0



mobilenet_v2,pass,0



mobilenet_v2_quantized_qat,model_fail_to_load,0



mobilenet_v3_large,pass,0



moco,pass,7



moondream,model_fail_to_load,0



nanogpt,pass,0



nvidia_deeprecommender,pass,0



opacus_cifar10,pass,0



phlippe_densenet,pass,0



phlippe_resnet,pass,0



pyhpc_equation_of_state,pass,0



pyhpc_isoneutral_mixing,pass,0



pyhpc_turbulent_kinetic_energy,pass,0



pytorch_CycleGAN_and_pix2pix,pass,0



pytorch_stargan,pass,0



pytorch_unet,pass,0



resnet152,pass,0



resnet18,pass,0



resnet50,pass,0



resnet50_quantized_qat,model_fail_to_load,0



resnext50_32x4d,pass,0



sam,pass,0



sam_fast,model_fail_to_load,0



shufflenet_v2_x1_0,pass,0



soft_actor_critic,pass,0



speech_transformer,pass,10



squeezenet1_1,pass,0



stable_diffusion_text_encoder,pass,0



stable_diffusion_unet,pass_due_to_skip,0



torch_multimodal_clip,pass,0



tts_angular,pass,2



vgg16,pass,0



vision_maskrcnn,pass,21



yolov3,pass,0

```



## High-Level Overview

This file is part of the PyTorch framework located at `benchmarks/dynamo/ci_expected_accuracy`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/ci_expected_accuracy`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/dynamo/ci_expected_accuracy`):

- [`aot_eager_huggingface_inference.csv_docs.md`](./aot_eager_huggingface_inference.csv_docs.md)
- [`aot_inductor_torchbench_inference.csv_docs.md`](./aot_inductor_torchbench_inference.csv_docs.md)
- [`dynamic_aot_eager_timm_inference.csv_docs.md`](./dynamic_aot_eager_timm_inference.csv_docs.md)
- [`dynamic_cpu_max_autotune_inductor_amp_freezing_huggingface_inference.csv_docs.md`](./dynamic_cpu_max_autotune_inductor_amp_freezing_huggingface_inference.csv_docs.md)
- [`dynamic_inductor_huggingface_training.csv_docs.md`](./dynamic_inductor_huggingface_training.csv_docs.md)
- [`inductor_timm_inference.csv_docs.md`](./inductor_timm_inference.csv_docs.md)
- [`dynamic_cpu_max_autotune_inductor_amp_freezing_torchbench_inference.csv_docs.md`](./dynamic_cpu_max_autotune_inductor_amp_freezing_torchbench_inference.csv_docs.md)
- [`dynamic_aot_eager_timm_training.csv_docs.md`](./dynamic_aot_eager_timm_training.csv_docs.md)
- [`dynamic_cpu_inductor_timm_inference.csv_docs.md`](./dynamic_cpu_inductor_timm_inference.csv_docs.md)
- [`inductor_torchbench_training.csv_docs.md`](./inductor_torchbench_training.csv_docs.md)


## Cross-References

- **File Documentation**: `dynamic_aot_eager_torchbench_inference.csv_docs.md`
- **Keyword Index**: `dynamic_aot_eager_torchbench_inference.csv_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
