# Index: `torch/_inductor/fx_passes/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `torch/_inductor/fx_passes/`

## Subfolders

- [`serialized_patterns/`](./serialized_patterns/index.md) - serialized_patterns module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`README.md`](../../../../torch/_inductor/fx_passes/README.md) | Documentation | [docs](./README.md_docs.md) | [keywords](./README.md_kw.md) |
| [`__init__.py`](../../../../torch/_inductor/fx_passes/__init__.py) | Package initialization | [docs](./__init__.py_docs.md) | [keywords](./__init__.py_kw.md) |
| [`b2b_gemm.py`](../../../../torch/_inductor/fx_passes/b2b_gemm.py) | Source code | [docs](./b2b_gemm.py_docs.md) | [keywords](./b2b_gemm.py_kw.md) |
| [`binary_folding.py`](../../../../torch/_inductor/fx_passes/binary_folding.py) | Source code | [docs](./binary_folding.py_docs.md) | [keywords](./binary_folding.py_kw.md) |
| [`bucketing.py`](../../../../torch/_inductor/fx_passes/bucketing.py) | Source code | [docs](./bucketing.py_docs.md) | [keywords](./bucketing.py_kw.md) |
| [`control_dependencies.py`](../../../../torch/_inductor/fx_passes/control_dependencies.py) | Source code | [docs](./control_dependencies.py_docs.md) | [keywords](./control_dependencies.py_kw.md) |
| [`ddp_fusion.py`](../../../../torch/_inductor/fx_passes/ddp_fusion.py) | Source code | [docs](./ddp_fusion.py_docs.md) | [keywords](./ddp_fusion.py_kw.md) |
| [`decompose_mem_bound_mm.py`](../../../../torch/_inductor/fx_passes/decompose_mem_bound_mm.py) | Source code | [docs](./decompose_mem_bound_mm.py_docs.md) | [keywords](./decompose_mem_bound_mm.py_kw.md) |
| [`dedupe_symint_uses.py`](../../../../torch/_inductor/fx_passes/dedupe_symint_uses.py) | Source code | [docs](./dedupe_symint_uses.py_docs.md) | [keywords](./dedupe_symint_uses.py_kw.md) |
| [`efficient_conv_bn_eval.py`](../../../../torch/_inductor/fx_passes/efficient_conv_bn_eval.py) | Source code | [docs](./efficient_conv_bn_eval.py_docs.md) | [keywords](./efficient_conv_bn_eval.py_kw.md) |
| [`freezing_patterns.py`](../../../../torch/_inductor/fx_passes/freezing_patterns.py) | Source code | [docs](./freezing_patterns.py_docs.md) | [keywords](./freezing_patterns.py_kw.md) |
| [`fsdp.py`](../../../../torch/_inductor/fx_passes/fsdp.py) | Source code | [docs](./fsdp.py_docs.md) | [keywords](./fsdp.py_kw.md) |
| [`fuse_attention.py`](../../../../torch/_inductor/fx_passes/fuse_attention.py) | Source code | [docs](./fuse_attention.py_docs.md) | [keywords](./fuse_attention.py_kw.md) |
| [`graph_view.py`](../../../../torch/_inductor/fx_passes/graph_view.py) | Source code | [docs](./graph_view.py_docs.md) | [keywords](./graph_view.py_kw.md) |
| [`group_batch_fusion.py`](../../../../torch/_inductor/fx_passes/group_batch_fusion.py) | Source code | [docs](./group_batch_fusion.py_docs.md) | [keywords](./group_batch_fusion.py_kw.md) |
| [`joint_graph.py`](../../../../torch/_inductor/fx_passes/joint_graph.py) | Source code | [docs](./joint_graph.py_docs.md) | [keywords](./joint_graph.py_kw.md) |
| [`memory_estimator.py`](../../../../torch/_inductor/fx_passes/memory_estimator.py) | Source code | [docs](./memory_estimator.py_docs.md) | [keywords](./memory_estimator.py_kw.md) |
| [`micro_pipeline_tp.py`](../../../../torch/_inductor/fx_passes/micro_pipeline_tp.py) | Source code | [docs](./micro_pipeline_tp.py_docs.md) | [keywords](./micro_pipeline_tp.py_kw.md) |
| [`misc_patterns.py`](../../../../torch/_inductor/fx_passes/misc_patterns.py) | Source code | [docs](./misc_patterns.py_docs.md) | [keywords](./misc_patterns.py_kw.md) |
| [`mkldnn_fusion.py`](../../../../torch/_inductor/fx_passes/mkldnn_fusion.py) | Source code | [docs](./mkldnn_fusion.py_docs.md) | [keywords](./mkldnn_fusion.py_kw.md) |
| [`node_runtime_estimation.py`](../../../../torch/_inductor/fx_passes/node_runtime_estimation.py) | Source code | [docs](./node_runtime_estimation.py_docs.md) | [keywords](./node_runtime_estimation.py_kw.md) |
| [`numeric_utils.py`](../../../../torch/_inductor/fx_passes/numeric_utils.py) | Source code | [docs](./numeric_utils.py_docs.md) | [keywords](./numeric_utils.py_kw.md) |
| [`overlap_manual_scheduling.py`](../../../../torch/_inductor/fx_passes/overlap_manual_scheduling.py) | Source code | [docs](./overlap_manual_scheduling.py_docs.md) | [keywords](./overlap_manual_scheduling.py_kw.md) |
| [`overlap_preserving_bucketer.py`](../../../../torch/_inductor/fx_passes/overlap_preserving_bucketer.py) | Source code | [docs](./overlap_preserving_bucketer.py_docs.md) | [keywords](./overlap_preserving_bucketer.py_kw.md) |
| [`overlap_scheduling.py`](../../../../torch/_inductor/fx_passes/overlap_scheduling.py) | Source code | [docs](./overlap_scheduling.py_docs.md) | [keywords](./overlap_scheduling.py_kw.md) |
| [`pad_mm.py`](../../../../torch/_inductor/fx_passes/pad_mm.py) | Source code | [docs](./pad_mm.py_docs.md) | [keywords](./pad_mm.py_kw.md) |
| [`post_grad.py`](../../../../torch/_inductor/fx_passes/post_grad.py) | Source code | [docs](./post_grad.py_docs.md) | [keywords](./post_grad.py_kw.md) |
| [`pre_grad.py`](../../../../torch/_inductor/fx_passes/pre_grad.py) | Source code | [docs](./pre_grad.py_docs.md) | [keywords](./pre_grad.py_kw.md) |
| [`quantization.py`](../../../../torch/_inductor/fx_passes/quantization.py) | Source code | [docs](./quantization.py_docs.md) | [keywords](./quantization.py_kw.md) |
| [`reinplace.py`](../../../../torch/_inductor/fx_passes/reinplace.py) | Source code | [docs](./reinplace.py_docs.md) | [keywords](./reinplace.py_kw.md) |
| [`replace_random.py`](../../../../torch/_inductor/fx_passes/replace_random.py) | Source code | [docs](./replace_random.py_docs.md) | [keywords](./replace_random.py_kw.md) |
| [`split_cat.py`](../../../../torch/_inductor/fx_passes/split_cat.py) | Source code | [docs](./split_cat.py_docs.md) | [keywords](./split_cat.py_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
