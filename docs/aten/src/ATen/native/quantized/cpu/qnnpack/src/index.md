# Index: `aten/src/ATen/native/quantized/cpu/qnnpack/src/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/`

## Subfolders

- [`hgemm/`](./hgemm/index.md) - hgemm module
- [`q8avgpool/`](./q8avgpool/index.md) - q8avgpool module
- [`q8conv/`](./q8conv/index.md) - q8conv module
- [`q8dwconv/`](./q8dwconv/index.md) - q8dwconv module
- [`q8gavgpool/`](./q8gavgpool/index.md) - q8gavgpool module
- [`q8gemm/`](./q8gemm/index.md) - q8gemm module
- [`q8gemm_sparse/`](./q8gemm_sparse/index.md) - q8gemm_sparse module
- [`q8vadd/`](./q8vadd/index.md) - q8vadd module
- [`qnnpack/`](./qnnpack/index.md) - qnnpack module
- [`requantization/`](./requantization/index.md) - requantization module
- [`sconv/`](./sconv/index.md) - sconv module
- [`sdwconv/`](./sdwconv/index.md) - sdwconv module
- [`sgemm/`](./sgemm/index.md) - sgemm module
- [`u8clamp/`](./u8clamp/index.md) - u8clamp module
- [`u8lut32norm/`](./u8lut32norm/index.md) - u8lut32norm module
- [`u8maxpool/`](./u8maxpool/index.md) - u8maxpool module
- [`u8rmax/`](./u8rmax/index.md) - u8rmax module
- [`x8lut/`](./x8lut/index.md) - x8lut module
- [`x8zip/`](./x8zip/index.md) - x8zip module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`add.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/add.c) | Source code | [docs](./add.c_docs.md) | [keywords](./add.c_kw.md) |
| [`average-pooling.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/average-pooling.c) | Source code | [docs](./average-pooling.c_docs.md) | [keywords](./average-pooling.c_kw.md) |
| [`channel-shuffle.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/channel-shuffle.c) | Source code | [docs](./channel-shuffle.c_docs.md) | [keywords](./channel-shuffle.c_kw.md) |
| [`clamp.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/clamp.c) | Source code | [docs](./clamp.c_docs.md) | [keywords](./clamp.c_kw.md) |
| [`conv-prepack.cc`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/conv-prepack.cc) | Source code | [docs](./conv-prepack.cc_docs.md) | [keywords](./conv-prepack.cc_kw.md) |
| [`conv-run.cc`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/conv-run.cc) | Source code | [docs](./conv-run.cc_docs.md) | [keywords](./conv-run.cc_kw.md) |
| [`convolution.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/convolution.c) | Source code | [docs](./convolution.c_docs.md) | [keywords](./convolution.c_kw.md) |
| [`deconv-run.cc`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/deconv-run.cc) | Source code | [docs](./deconv-run.cc_docs.md) | [keywords](./deconv-run.cc_kw.md) |
| [`deconvolution.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/deconvolution.c) | Source code | [docs](./deconvolution.c_docs.md) | [keywords](./deconvolution.c_kw.md) |
| [`fc-dynamic-run.cc`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-dynamic-run.cc) | Source code | [docs](./fc-dynamic-run.cc_docs.md) | [keywords](./fc-dynamic-run.cc_kw.md) |
| [`fc-prepack.cc`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-prepack.cc) | Source code | [docs](./fc-prepack.cc_docs.md) | [keywords](./fc-prepack.cc_kw.md) |
| [`fc-run.cc`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-run.cc) | Source code | [docs](./fc-run.cc_docs.md) | [keywords](./fc-run.cc_kw.md) |
| [`fc-unpack.cc`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-unpack.cc) | Source code | [docs](./fc-unpack.cc_docs.md) | [keywords](./fc-unpack.cc_kw.md) |
| [`fully-connected-sparse.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/fully-connected-sparse.c) | Source code | [docs](./fully-connected-sparse.c_docs.md) | [keywords](./fully-connected-sparse.c_kw.md) |
| [`fully-connected.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/fully-connected.c) | Source code | [docs](./fully-connected.c_docs.md) | [keywords](./fully-connected.c_kw.md) |
| [`global-average-pooling.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/global-average-pooling.c) | Source code | [docs](./global-average-pooling.c_docs.md) | [keywords](./global-average-pooling.c_kw.md) |
| [`hardsigmoid.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/hardsigmoid.c) | Source code | [docs](./hardsigmoid.c_docs.md) | [keywords](./hardsigmoid.c_kw.md) |
| [`hardswish.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/hardswish.c) | Source code | [docs](./hardswish.c_docs.md) | [keywords](./hardswish.c_kw.md) |
| [`indirection.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/indirection.c) | Source code | [docs](./indirection.c_docs.md) | [keywords](./indirection.c_kw.md) |
| [`init.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/init.c) | Source code | [docs](./init.c_docs.md) | [keywords](./init.c_kw.md) |
| [`leaky-relu.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/leaky-relu.c) | Source code | [docs](./leaky-relu.c_docs.md) | [keywords](./leaky-relu.c_kw.md) |
| [`max-pooling.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/max-pooling.c) | Source code | [docs](./max-pooling.c_docs.md) | [keywords](./max-pooling.c_kw.md) |
| [`operator-delete.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/operator-delete.c) | Source code | [docs](./operator-delete.c_docs.md) | [keywords](./operator-delete.c_kw.md) |
| [`operator-run.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/operator-run.c) | Source code | [docs](./operator-run.c_docs.md) | [keywords](./operator-run.c_kw.md) |
| [`sigmoid.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/sigmoid.c) | Source code | [docs](./sigmoid.c_docs.md) | [keywords](./sigmoid.c_kw.md) |
| [`softargmax.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/softargmax.c) | Source code | [docs](./softargmax.c_docs.md) | [keywords](./softargmax.c_kw.md) |
| [`tanh.c`](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/tanh.c) | Source code | [docs](./tanh.c_docs.md) | [keywords](./tanh.c_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
