# Index: `torch/csrc/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `torch/csrc/`

## Subfolders

- [`acc/`](./acc/index.md) - acc module
- [`api/`](./api/index.md) - api module
- [`autograd/`](./autograd/index.md) - autograd module
- [`cpu/`](./cpu/index.md) - cpu module
- [`cuda/`](./cuda/index.md) - cuda module
- [`distributed/`](./distributed/index.md) - distributed module
- [`dynamo/`](./dynamo/index.md) - dynamo module
- [`export/`](./export/index.md) - export module
- [`functionalization/`](./functionalization/index.md) - functionalization module
- [`functorch/`](./functorch/index.md) - functorch module
- [`fx/`](./fx/index.md) - fx module
- [`inductor/`](./inductor/index.md) - inductor module
- [`instruction_counter/`](./instruction_counter/index.md) - instruction_counter module
- [`jit/`](./jit/index.md) - jit module
- [`lazy/`](./lazy/index.md) - lazy module
- [`monitor/`](./monitor/index.md) - monitor module
- [`mps/`](./mps/index.md) - mps module
- [`mtia/`](./mtia/index.md) - mtia module
- [`multiprocessing/`](./multiprocessing/index.md) - multiprocessing module
- [`onnx/`](./onnx/index.md) - onnx module
- [`profiler/`](./profiler/index.md) - profiler module
- [`stable/`](./stable/index.md) - stable module
- [`tensor/`](./tensor/index.md) - tensor module
- [`utils/`](./utils/index.md) - utils module
- [`xpu/`](./xpu/index.md) - xpu module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`CudaIPCTypes.cpp`](../../../torch/csrc/CudaIPCTypes.cpp) | Source code | [docs](./CudaIPCTypes.cpp_docs.md) | [keywords](./CudaIPCTypes.cpp_kw.md) |
| [`CudaIPCTypes.h`](../../../torch/csrc/CudaIPCTypes.h) | Source code | [docs](./CudaIPCTypes.h_docs.md) | [keywords](./CudaIPCTypes.h_kw.md) |
| [`DataLoader.cpp`](../../../torch/csrc/DataLoader.cpp) | Source code | [docs](./DataLoader.cpp_docs.md) | [keywords](./DataLoader.cpp_kw.md) |
| [`DataLoader.h`](../../../torch/csrc/DataLoader.h) | Source code | [docs](./DataLoader.h_docs.md) | [keywords](./DataLoader.h_kw.md) |
| [`Device.cpp`](../../../torch/csrc/Device.cpp) | Source code | [docs](./Device.cpp_docs.md) | [keywords](./Device.cpp_kw.md) |
| [`Device.h`](../../../torch/csrc/Device.h) | Source code | [docs](./Device.h_docs.md) | [keywords](./Device.h_kw.md) |
| [`DeviceAccelerator.cpp`](../../../torch/csrc/DeviceAccelerator.cpp) | Source code | [docs](./DeviceAccelerator.cpp_docs.md) | [keywords](./DeviceAccelerator.cpp_kw.md) |
| [`DeviceAccelerator.h`](../../../torch/csrc/DeviceAccelerator.h) | Source code | [docs](./DeviceAccelerator.h_docs.md) | [keywords](./DeviceAccelerator.h_kw.md) |
| [`Dtype.cpp`](../../../torch/csrc/Dtype.cpp) | Source code | [docs](./Dtype.cpp_docs.md) | [keywords](./Dtype.cpp_kw.md) |
| [`Dtype.h`](../../../torch/csrc/Dtype.h) | Source code | [docs](./Dtype.h_docs.md) | [keywords](./Dtype.h_kw.md) |
| [`DynamicTypes.cpp`](../../../torch/csrc/DynamicTypes.cpp) | Source code | [docs](./DynamicTypes.cpp_docs.md) | [keywords](./DynamicTypes.cpp_kw.md) |
| [`DynamicTypes.h`](../../../torch/csrc/DynamicTypes.h) | Source code | [docs](./DynamicTypes.h_docs.md) | [keywords](./DynamicTypes.h_kw.md) |
| [`Event.cpp`](../../../torch/csrc/Event.cpp) | Source code | [docs](./Event.cpp_docs.md) | [keywords](./Event.cpp_kw.md) |
| [`Event.h`](../../../torch/csrc/Event.h) | Source code | [docs](./Event.h_docs.md) | [keywords](./Event.h_kw.md) |
| [`Exceptions.cpp`](../../../torch/csrc/Exceptions.cpp) | Source code | [docs](./Exceptions.cpp_docs.md) | [keywords](./Exceptions.cpp_kw.md) |
| [`Exceptions.h`](../../../torch/csrc/Exceptions.h) | Source code | [docs](./Exceptions.h_docs.md) | [keywords](./Exceptions.h_kw.md) |
| [`Export.h`](../../../torch/csrc/Export.h) | Source code | [docs](./Export.h_docs.md) | [keywords](./Export.h_kw.md) |
| [`Generator.cpp`](../../../torch/csrc/Generator.cpp) | Source code | [docs](./Generator.cpp_docs.md) | [keywords](./Generator.cpp_kw.md) |
| [`Generator.h`](../../../torch/csrc/Generator.h) | Source code | [docs](./Generator.h_docs.md) | [keywords](./Generator.h_kw.md) |
| [`Layout.cpp`](../../../torch/csrc/Layout.cpp) | Source code | [docs](./Layout.cpp_docs.md) | [keywords](./Layout.cpp_kw.md) |
| [`Layout.h`](../../../torch/csrc/Layout.h) | Source code | [docs](./Layout.h_docs.md) | [keywords](./Layout.h_kw.md) |
| [`MemoryFormat.cpp`](../../../torch/csrc/MemoryFormat.cpp) | Source code | [docs](./MemoryFormat.cpp_docs.md) | [keywords](./MemoryFormat.cpp_kw.md) |
| [`MemoryFormat.h`](../../../torch/csrc/MemoryFormat.h) | Source code | [docs](./MemoryFormat.h_docs.md) | [keywords](./MemoryFormat.h_kw.md) |
| [`Module.cpp`](../../../torch/csrc/Module.cpp) | Source code | [docs](./Module.cpp_docs.md) | [keywords](./Module.cpp_kw.md) |
| [`Module.h`](../../../torch/csrc/Module.h) | Source code | [docs](./Module.h_docs.md) | [keywords](./Module.h_kw.md) |
| [`PyInterpreter.cpp`](../../../torch/csrc/PyInterpreter.cpp) | Source code | [docs](./PyInterpreter.cpp_docs.md) | [keywords](./PyInterpreter.cpp_kw.md) |
| [`PyInterpreter.h`](../../../torch/csrc/PyInterpreter.h) | Source code | [docs](./PyInterpreter.h_docs.md) | [keywords](./PyInterpreter.h_kw.md) |
| [`PyInterpreterHooks.cpp`](../../../torch/csrc/PyInterpreterHooks.cpp) | Source code | [docs](./PyInterpreterHooks.cpp_docs.md) | [keywords](./PyInterpreterHooks.cpp_kw.md) |
| [`PyInterpreterHooks.h`](../../../torch/csrc/PyInterpreterHooks.h) | Source code | [docs](./PyInterpreterHooks.h_docs.md) | [keywords](./PyInterpreterHooks.h_kw.md) |
| [`QScheme.cpp`](../../../torch/csrc/QScheme.cpp) | Source code | [docs](./QScheme.cpp_docs.md) | [keywords](./QScheme.cpp_kw.md) |
| [`QScheme.h`](../../../torch/csrc/QScheme.h) | Source code | [docs](./QScheme.h_docs.md) | [keywords](./QScheme.h_kw.md) |
| [`README.md`](../../../torch/csrc/README.md) | Documentation | [docs](./README.md_docs.md) | [keywords](./README.md_kw.md) |
| [`Size.cpp`](../../../torch/csrc/Size.cpp) | Source code | [docs](./Size.cpp_docs.md) | [keywords](./Size.cpp_kw.md) |
| [`Size.h`](../../../torch/csrc/Size.h) | Source code | [docs](./Size.h_docs.md) | [keywords](./Size.h_kw.md) |
| [`Storage.cpp`](../../../torch/csrc/Storage.cpp) | Source code | [docs](./Storage.cpp_docs.md) | [keywords](./Storage.cpp_kw.md) |
| [`Storage.h`](../../../torch/csrc/Storage.h) | Source code | [docs](./Storage.h_docs.md) | [keywords](./Storage.h_kw.md) |
| [`StorageMethods.cpp`](../../../torch/csrc/StorageMethods.cpp) | Source code | [docs](./StorageMethods.cpp_docs.md) | [keywords](./StorageMethods.cpp_kw.md) |
| [`StorageMethods.h`](../../../torch/csrc/StorageMethods.h) | Source code | [docs](./StorageMethods.h_docs.md) | [keywords](./StorageMethods.h_kw.md) |
| [`StorageSharing.cpp`](../../../torch/csrc/StorageSharing.cpp) | Source code | [docs](./StorageSharing.cpp_docs.md) | [keywords](./StorageSharing.cpp_kw.md) |
| [`StorageSharing.h`](../../../torch/csrc/StorageSharing.h) | Source code | [docs](./StorageSharing.h_docs.md) | [keywords](./StorageSharing.h_kw.md) |
| [`Stream.cpp`](../../../torch/csrc/Stream.cpp) | Source code | [docs](./Stream.cpp_docs.md) | [keywords](./Stream.cpp_kw.md) |
| [`Stream.h`](../../../torch/csrc/Stream.h) | Source code | [docs](./Stream.h_docs.md) | [keywords](./Stream.h_kw.md) |
| [`THConcat.h`](../../../torch/csrc/THConcat.h) | Source code | [docs](./THConcat.h_docs.md) | [keywords](./THConcat.h_kw.md) |
| [`THP.h`](../../../torch/csrc/THP.h) | Source code | [docs](./THP.h_docs.md) | [keywords](./THP.h_kw.md) |
| [`TypeInfo.cpp`](../../../torch/csrc/TypeInfo.cpp) | Source code | [docs](./TypeInfo.cpp_docs.md) | [keywords](./TypeInfo.cpp_kw.md) |
| [`TypeInfo.h`](../../../torch/csrc/TypeInfo.h) | Source code | [docs](./TypeInfo.h_docs.md) | [keywords](./TypeInfo.h_kw.md) |
| [`Types.h`](../../../torch/csrc/Types.h) | Source code | [docs](./Types.h_docs.md) | [keywords](./Types.h_kw.md) |
| [`copy_utils.h`](../../../torch/csrc/copy_utils.h) | Source code | [docs](./copy_utils.h_docs.md) | [keywords](./copy_utils.h_kw.md) |
| [`empty.c`](../../../torch/csrc/empty.c) | Source code | [docs](./empty.c_docs.md) | [keywords](./empty.c_kw.md) |
| [`itt.cpp`](../../../torch/csrc/itt.cpp) | Source code | [docs](./itt.cpp_docs.md) | [keywords](./itt.cpp_kw.md) |
| [`itt.h`](../../../torch/csrc/itt.h) | Source code | [docs](./itt.h_docs.md) | [keywords](./itt.h_kw.md) |
| [`itt_wrapper.cpp`](../../../torch/csrc/itt_wrapper.cpp) | Source code | [docs](./itt_wrapper.cpp_docs.md) | [keywords](./itt_wrapper.cpp_kw.md) |
| [`itt_wrapper.h`](../../../torch/csrc/itt_wrapper.h) | Source code | [docs](./itt_wrapper.h_docs.md) | [keywords](./itt_wrapper.h_kw.md) |
| [`python_dimname.cpp`](../../../torch/csrc/python_dimname.cpp) | Source code | [docs](./python_dimname.cpp_docs.md) | [keywords](./python_dimname.cpp_kw.md) |
| [`python_dimname.h`](../../../torch/csrc/python_dimname.h) | Source code | [docs](./python_dimname.h_docs.md) | [keywords](./python_dimname.h_kw.md) |
| [`python_headers.h`](../../../torch/csrc/python_headers.h) | Source code | [docs](./python_headers.h_docs.md) | [keywords](./python_headers.h_kw.md) |
| [`serialization.cpp`](../../../torch/csrc/serialization.cpp) | Source code | [docs](./serialization.cpp_docs.md) | [keywords](./serialization.cpp_kw.md) |
| [`serialization.h`](../../../torch/csrc/serialization.h) | Source code | [docs](./serialization.h_docs.md) | [keywords](./serialization.h_kw.md) |
| [`shim_common.cpp`](../../../torch/csrc/shim_common.cpp) | Source code | [docs](./shim_common.cpp_docs.md) | [keywords](./shim_common.cpp_kw.md) |
| [`shim_conversion_utils.h`](../../../torch/csrc/shim_conversion_utils.h) | Source code | [docs](./shim_conversion_utils.h_docs.md) | [keywords](./shim_conversion_utils.h_kw.md) |
| [`stub.c`](../../../torch/csrc/stub.c) | Source code | [docs](./stub.c_docs.md) | [keywords](./stub.c_kw.md) |
| [`utils.cpp`](../../../torch/csrc/utils.cpp) | Source code | [docs](./utils.cpp_docs.md) | [keywords](./utils.cpp_kw.md) |
| [`utils.h`](../../../torch/csrc/utils.h) | Source code | [docs](./utils.h_docs.md) | [keywords](./utils.h_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
