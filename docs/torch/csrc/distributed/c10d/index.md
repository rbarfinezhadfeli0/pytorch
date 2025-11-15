# Index: `torch/csrc/distributed/c10d/`

## Overview

This folder contains source code, configurations, or documentation for the PyTorch project.

**Location**: `torch/csrc/distributed/c10d/`

## Subfolders

- [`control_collectives/`](./control_collectives/index.md) - control_collectives module
- [`control_plane/`](./control_plane/index.md) - control_plane module
- [`cuda/`](./cuda/index.md) - cuda module
- [`quantization/`](./quantization/index.md) - quantization module
- [`symm_mem/`](./symm_mem/index.md) - symm_mem module

## Files

| File | Description | Documentation | Keywords |
|------|-------------|---------------|----------|
| [`Backend.cpp`](../../../../../torch/csrc/distributed/c10d/Backend.cpp) | Source code | [docs](./Backend.cpp_docs.md) | [keywords](./Backend.cpp_kw.md) |
| [`Backend.hpp`](../../../../../torch/csrc/distributed/c10d/Backend.hpp) | Source code | [docs](./Backend.hpp_docs.md) | [keywords](./Backend.hpp_kw.md) |
| [`Backoff.cpp`](../../../../../torch/csrc/distributed/c10d/Backoff.cpp) | Source code | [docs](./Backoff.cpp_docs.md) | [keywords](./Backoff.cpp_kw.md) |
| [`Backoff.hpp`](../../../../../torch/csrc/distributed/c10d/Backoff.hpp) | Source code | [docs](./Backoff.hpp_docs.md) | [keywords](./Backoff.hpp_kw.md) |
| [`FakeProcessGroup.hpp`](../../../../../torch/csrc/distributed/c10d/FakeProcessGroup.hpp) | Source code | [docs](./FakeProcessGroup.hpp_docs.md) | [keywords](./FakeProcessGroup.hpp_kw.md) |
| [`FileStore.cpp`](../../../../../torch/csrc/distributed/c10d/FileStore.cpp) | Source code | [docs](./FileStore.cpp_docs.md) | [keywords](./FileStore.cpp_kw.md) |
| [`FileStore.hpp`](../../../../../torch/csrc/distributed/c10d/FileStore.hpp) | Source code | [docs](./FileStore.hpp_docs.md) | [keywords](./FileStore.hpp_kw.md) |
| [`FlightRecorder.cpp`](../../../../../torch/csrc/distributed/c10d/FlightRecorder.cpp) | Source code | [docs](./FlightRecorder.cpp_docs.md) | [keywords](./FlightRecorder.cpp_kw.md) |
| [`FlightRecorder.hpp`](../../../../../torch/csrc/distributed/c10d/FlightRecorder.hpp) | Source code | [docs](./FlightRecorder.hpp_docs.md) | [keywords](./FlightRecorder.hpp_kw.md) |
| [`FlightRecorderCuda.cpp`](../../../../../torch/csrc/distributed/c10d/FlightRecorderCuda.cpp) | Source code | [docs](./FlightRecorderCuda.cpp_docs.md) | [keywords](./FlightRecorderCuda.cpp_kw.md) |
| [`FlightRecorderDetail.hpp`](../../../../../torch/csrc/distributed/c10d/FlightRecorderDetail.hpp) | Source code | [docs](./FlightRecorderDetail.hpp_docs.md) | [keywords](./FlightRecorderDetail.hpp_kw.md) |
| [`Functional.cpp`](../../../../../torch/csrc/distributed/c10d/Functional.cpp) | Source code | [docs](./Functional.cpp_docs.md) | [keywords](./Functional.cpp_kw.md) |
| [`Functional.hpp`](../../../../../torch/csrc/distributed/c10d/Functional.hpp) | Source code | [docs](./Functional.hpp_docs.md) | [keywords](./Functional.hpp_kw.md) |
| [`GlooDeviceFactory.cpp`](../../../../../torch/csrc/distributed/c10d/GlooDeviceFactory.cpp) | Source code | [docs](./GlooDeviceFactory.cpp_docs.md) | [keywords](./GlooDeviceFactory.cpp_kw.md) |
| [`GlooDeviceFactory.hpp`](../../../../../torch/csrc/distributed/c10d/GlooDeviceFactory.hpp) | Source code | [docs](./GlooDeviceFactory.hpp_docs.md) | [keywords](./GlooDeviceFactory.hpp_kw.md) |
| [`GroupRegistry.cpp`](../../../../../torch/csrc/distributed/c10d/GroupRegistry.cpp) | Source code | [docs](./GroupRegistry.cpp_docs.md) | [keywords](./GroupRegistry.cpp_kw.md) |
| [`GroupRegistry.hpp`](../../../../../torch/csrc/distributed/c10d/GroupRegistry.hpp) | Source code | [docs](./GroupRegistry.hpp_docs.md) | [keywords](./GroupRegistry.hpp_kw.md) |
| [`HashStore.cpp`](../../../../../torch/csrc/distributed/c10d/HashStore.cpp) | Source code | [docs](./HashStore.cpp_docs.md) | [keywords](./HashStore.cpp_kw.md) |
| [`HashStore.hpp`](../../../../../torch/csrc/distributed/c10d/HashStore.hpp) | Source code | [docs](./HashStore.hpp_docs.md) | [keywords](./HashStore.hpp_kw.md) |
| [`NCCLUtils.cpp`](../../../../../torch/csrc/distributed/c10d/NCCLUtils.cpp) | Source code | [docs](./NCCLUtils.cpp_docs.md) | [keywords](./NCCLUtils.cpp_kw.md) |
| [`NCCLUtils.hpp`](../../../../../torch/csrc/distributed/c10d/NCCLUtils.hpp) | Source code | [docs](./NCCLUtils.hpp_docs.md) | [keywords](./NCCLUtils.hpp_kw.md) |
| [`NanCheck.cu`](../../../../../torch/csrc/distributed/c10d/NanCheck.cu) | Source code | [docs](./NanCheck.cu_docs.md) | [keywords](./NanCheck.cu_kw.md) |
| [`NanCheck.hpp`](../../../../../torch/csrc/distributed/c10d/NanCheck.hpp) | Source code | [docs](./NanCheck.hpp_docs.md) | [keywords](./NanCheck.hpp_kw.md) |
| [`Ops.cpp`](../../../../../torch/csrc/distributed/c10d/Ops.cpp) | Source code | [docs](./Ops.cpp_docs.md) | [keywords](./Ops.cpp_kw.md) |
| [`ParamCommsUtils.cpp`](../../../../../torch/csrc/distributed/c10d/ParamCommsUtils.cpp) | Source code | [docs](./ParamCommsUtils.cpp_docs.md) | [keywords](./ParamCommsUtils.cpp_kw.md) |
| [`ParamCommsUtils.hpp`](../../../../../torch/csrc/distributed/c10d/ParamCommsUtils.hpp) | Source code | [docs](./ParamCommsUtils.hpp_docs.md) | [keywords](./ParamCommsUtils.hpp_kw.md) |
| [`PrefixStore.cpp`](../../../../../torch/csrc/distributed/c10d/PrefixStore.cpp) | Source code | [docs](./PrefixStore.cpp_docs.md) | [keywords](./PrefixStore.cpp_kw.md) |
| [`PrefixStore.hpp`](../../../../../torch/csrc/distributed/c10d/PrefixStore.hpp) | Source code | [docs](./PrefixStore.hpp_docs.md) | [keywords](./PrefixStore.hpp_kw.md) |
| [`ProcessGroup.cpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroup.cpp) | Source code | [docs](./ProcessGroup.cpp_docs.md) | [keywords](./ProcessGroup.cpp_kw.md) |
| [`ProcessGroup.hpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroup.hpp) | Source code | [docs](./ProcessGroup.hpp_docs.md) | [keywords](./ProcessGroup.hpp_kw.md) |
| [`ProcessGroupGloo.cpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupGloo.cpp) | Source code | [docs](./ProcessGroupGloo.cpp_docs.md) | [keywords](./ProcessGroupGloo.cpp_kw.md) |
| [`ProcessGroupGloo.hpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupGloo.hpp) | Source code | [docs](./ProcessGroupGloo.hpp_docs.md) | [keywords](./ProcessGroupGloo.hpp_kw.md) |
| [`ProcessGroupGlooCuda.cpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupGlooCuda.cpp) | Source code | [docs](./ProcessGroupGlooCuda.cpp_docs.md) | [keywords](./ProcessGroupGlooCuda.cpp_kw.md) |
| [`ProcessGroupGlooDetail.hpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupGlooDetail.hpp) | Source code | [docs](./ProcessGroupGlooDetail.hpp_docs.md) | [keywords](./ProcessGroupGlooDetail.hpp_kw.md) |
| [`ProcessGroupMPI.cpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupMPI.cpp) | Source code | [docs](./ProcessGroupMPI.cpp_docs.md) | [keywords](./ProcessGroupMPI.cpp_kw.md) |
| [`ProcessGroupMPI.hpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupMPI.hpp) | Source code | [docs](./ProcessGroupMPI.hpp_docs.md) | [keywords](./ProcessGroupMPI.hpp_kw.md) |
| [`ProcessGroupNCCL.cpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp) | Source code | [docs](./ProcessGroupNCCL.cpp_docs.md) | [keywords](./ProcessGroupNCCL.cpp_kw.md) |
| [`ProcessGroupNCCL.hpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp) | Source code | [docs](./ProcessGroupNCCL.hpp_docs.md) | [keywords](./ProcessGroupNCCL.hpp_kw.md) |
| [`ProcessGroupUCC.cpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupUCC.cpp) | Source code | [docs](./ProcessGroupUCC.cpp_docs.md) | [keywords](./ProcessGroupUCC.cpp_kw.md) |
| [`ProcessGroupUCC.hpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupUCC.hpp) | Source code | [docs](./ProcessGroupUCC.hpp_docs.md) | [keywords](./ProcessGroupUCC.hpp_kw.md) |
| [`ProcessGroupWrapper.cpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupWrapper.cpp) | Source code | [docs](./ProcessGroupWrapper.cpp_docs.md) | [keywords](./ProcessGroupWrapper.cpp_kw.md) |
| [`ProcessGroupWrapper.hpp`](../../../../../torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp) | Source code | [docs](./ProcessGroupWrapper.hpp_docs.md) | [keywords](./ProcessGroupWrapper.hpp_kw.md) |
| [`PyProcessGroup.hpp`](../../../../../torch/csrc/distributed/c10d/PyProcessGroup.hpp) | Source code | [docs](./PyProcessGroup.hpp_docs.md) | [keywords](./PyProcessGroup.hpp_kw.md) |
| [`RankLocal.hpp`](../../../../../torch/csrc/distributed/c10d/RankLocal.hpp) | Source code | [docs](./RankLocal.hpp_docs.md) | [keywords](./RankLocal.hpp_kw.md) |
| [`Store.cpp`](../../../../../torch/csrc/distributed/c10d/Store.cpp) | Source code | [docs](./Store.cpp_docs.md) | [keywords](./Store.cpp_kw.md) |
| [`Store.hpp`](../../../../../torch/csrc/distributed/c10d/Store.hpp) | Source code | [docs](./Store.hpp_docs.md) | [keywords](./Store.hpp_kw.md) |
| [`TCPStore.cpp`](../../../../../torch/csrc/distributed/c10d/TCPStore.cpp) | Source code | [docs](./TCPStore.cpp_docs.md) | [keywords](./TCPStore.cpp_kw.md) |
| [`TCPStore.hpp`](../../../../../torch/csrc/distributed/c10d/TCPStore.hpp) | Source code | [docs](./TCPStore.hpp_docs.md) | [keywords](./TCPStore.hpp_kw.md) |
| [`TCPStoreBackend.cpp`](../../../../../torch/csrc/distributed/c10d/TCPStoreBackend.cpp) | Source code | [docs](./TCPStoreBackend.cpp_docs.md) | [keywords](./TCPStoreBackend.cpp_kw.md) |
| [`TCPStoreBackend.hpp`](../../../../../torch/csrc/distributed/c10d/TCPStoreBackend.hpp) | Source code | [docs](./TCPStoreBackend.hpp_docs.md) | [keywords](./TCPStoreBackend.hpp_kw.md) |
| [`TCPStoreLibUvBackend.cpp`](../../../../../torch/csrc/distributed/c10d/TCPStoreLibUvBackend.cpp) | Source code | [docs](./TCPStoreLibUvBackend.cpp_docs.md) | [keywords](./TCPStoreLibUvBackend.cpp_kw.md) |
| [`TraceUtils.h`](../../../../../torch/csrc/distributed/c10d/TraceUtils.h) | Source code | [docs](./TraceUtils.h_docs.md) | [keywords](./TraceUtils.h_kw.md) |
| [`Types.cpp`](../../../../../torch/csrc/distributed/c10d/Types.cpp) | Source code | [docs](./Types.cpp_docs.md) | [keywords](./Types.cpp_kw.md) |
| [`Types.hpp`](../../../../../torch/csrc/distributed/c10d/Types.hpp) | Source code | [docs](./Types.hpp_docs.md) | [keywords](./Types.hpp_kw.md) |
| [`UCCTracing.cpp`](../../../../../torch/csrc/distributed/c10d/UCCTracing.cpp) | Source code | [docs](./UCCTracing.cpp_docs.md) | [keywords](./UCCTracing.cpp_kw.md) |
| [`UCCTracing.hpp`](../../../../../torch/csrc/distributed/c10d/UCCTracing.hpp) | Source code | [docs](./UCCTracing.hpp_docs.md) | [keywords](./UCCTracing.hpp_kw.md) |
| [`UCCUtils.cpp`](../../../../../torch/csrc/distributed/c10d/UCCUtils.cpp) | Source code | [docs](./UCCUtils.cpp_docs.md) | [keywords](./UCCUtils.cpp_kw.md) |
| [`UCCUtils.hpp`](../../../../../torch/csrc/distributed/c10d/UCCUtils.hpp) | Source code | [docs](./UCCUtils.hpp_docs.md) | [keywords](./UCCUtils.hpp_kw.md) |
| [`UnixSockUtils.hpp`](../../../../../torch/csrc/distributed/c10d/UnixSockUtils.hpp) | Source code | [docs](./UnixSockUtils.hpp_docs.md) | [keywords](./UnixSockUtils.hpp_kw.md) |
| [`Utils.cpp`](../../../../../torch/csrc/distributed/c10d/Utils.cpp) | Source code | [docs](./Utils.cpp_docs.md) | [keywords](./Utils.cpp_kw.md) |
| [`Utils.hpp`](../../../../../torch/csrc/distributed/c10d/Utils.hpp) | Source code | [docs](./Utils.hpp_docs.md) | [keywords](./Utils.hpp_kw.md) |
| [`WinSockUtils.hpp`](../../../../../torch/csrc/distributed/c10d/WinSockUtils.hpp) | Source code | [docs](./WinSockUtils.hpp_docs.md) | [keywords](./WinSockUtils.hpp_kw.md) |
| [`Work.cpp`](../../../../../torch/csrc/distributed/c10d/Work.cpp) | Source code | [docs](./Work.cpp_docs.md) | [keywords](./Work.cpp_kw.md) |
| [`Work.hpp`](../../../../../torch/csrc/distributed/c10d/Work.hpp) | Source code | [docs](./Work.hpp_docs.md) | [keywords](./Work.hpp_kw.md) |
| [`c10d.h`](../../../../../torch/csrc/distributed/c10d/c10d.h) | Source code | [docs](./c10d.h_docs.md) | [keywords](./c10d.h_kw.md) |
| [`comm.cpp`](../../../../../torch/csrc/distributed/c10d/comm.cpp) | Source code | [docs](./comm.cpp_docs.md) | [keywords](./comm.cpp_kw.md) |
| [`comm.hpp`](../../../../../torch/csrc/distributed/c10d/comm.hpp) | Source code | [docs](./comm.hpp_docs.md) | [keywords](./comm.hpp_kw.md) |
| [`debug.cpp`](../../../../../torch/csrc/distributed/c10d/debug.cpp) | Source code | [docs](./debug.cpp_docs.md) | [keywords](./debug.cpp_kw.md) |
| [`debug.h`](../../../../../torch/csrc/distributed/c10d/debug.h) | Source code | [docs](./debug.h_docs.md) | [keywords](./debug.h_kw.md) |
| [`default_comm_hooks.cpp`](../../../../../torch/csrc/distributed/c10d/default_comm_hooks.cpp) | Source code | [docs](./default_comm_hooks.cpp_docs.md) | [keywords](./default_comm_hooks.cpp_kw.md) |
| [`default_comm_hooks.hpp`](../../../../../torch/csrc/distributed/c10d/default_comm_hooks.hpp) | Source code | [docs](./default_comm_hooks.hpp_docs.md) | [keywords](./default_comm_hooks.hpp_kw.md) |
| [`error.h`](../../../../../torch/csrc/distributed/c10d/error.h) | Source code | [docs](./error.h_docs.md) | [keywords](./error.h_kw.md) |
| [`exception.h`](../../../../../torch/csrc/distributed/c10d/exception.h) | Source code | [docs](./exception.h_docs.md) | [keywords](./exception.h_kw.md) |
| [`init.cpp`](../../../../../torch/csrc/distributed/c10d/init.cpp) | Source code | [docs](./init.cpp_docs.md) | [keywords](./init.cpp_kw.md) |
| [`logger.cpp`](../../../../../torch/csrc/distributed/c10d/logger.cpp) | Source code | [docs](./logger.cpp_docs.md) | [keywords](./logger.cpp_kw.md) |
| [`logger.hpp`](../../../../../torch/csrc/distributed/c10d/logger.hpp) | Source code | [docs](./logger.hpp_docs.md) | [keywords](./logger.hpp_kw.md) |
| [`logging.cpp`](../../../../../torch/csrc/distributed/c10d/logging.cpp) | Source code | [docs](./logging.cpp_docs.md) | [keywords](./logging.cpp_kw.md) |
| [`logging.h`](../../../../../torch/csrc/distributed/c10d/logging.h) | Source code | [docs](./logging.h_docs.md) | [keywords](./logging.h_kw.md) |
| [`python_callback_work.cpp`](../../../../../torch/csrc/distributed/c10d/python_callback_work.cpp) | Source code | [docs](./python_callback_work.cpp_docs.md) | [keywords](./python_callback_work.cpp_kw.md) |
| [`python_callback_work.hpp`](../../../../../torch/csrc/distributed/c10d/python_callback_work.hpp) | Source code | [docs](./python_callback_work.hpp_docs.md) | [keywords](./python_callback_work.hpp_kw.md) |
| [`python_comm_hook.cpp`](../../../../../torch/csrc/distributed/c10d/python_comm_hook.cpp) | Source code | [docs](./python_comm_hook.cpp_docs.md) | [keywords](./python_comm_hook.cpp_kw.md) |
| [`python_comm_hook.h`](../../../../../torch/csrc/distributed/c10d/python_comm_hook.h) | Source code | [docs](./python_comm_hook.h_docs.md) | [keywords](./python_comm_hook.h_kw.md) |
| [`reducer.cpp`](../../../../../torch/csrc/distributed/c10d/reducer.cpp) | Source code | [docs](./reducer.cpp_docs.md) | [keywords](./reducer.cpp_kw.md) |
| [`reducer.hpp`](../../../../../torch/csrc/distributed/c10d/reducer.hpp) | Source code | [docs](./reducer.hpp_docs.md) | [keywords](./reducer.hpp_kw.md) |
| [`reducer_cuda.cpp`](../../../../../torch/csrc/distributed/c10d/reducer_cuda.cpp) | Source code | [docs](./reducer_cuda.cpp_docs.md) | [keywords](./reducer_cuda.cpp_kw.md) |
| [`reducer_timer.hpp`](../../../../../torch/csrc/distributed/c10d/reducer_timer.hpp) | Source code | [docs](./reducer_timer.hpp_docs.md) | [keywords](./reducer_timer.hpp_kw.md) |
| [`sequence_num.cpp`](../../../../../torch/csrc/distributed/c10d/sequence_num.cpp) | Source code | [docs](./sequence_num.cpp_docs.md) | [keywords](./sequence_num.cpp_kw.md) |
| [`sequence_num.hpp`](../../../../../torch/csrc/distributed/c10d/sequence_num.hpp) | Source code | [docs](./sequence_num.hpp_docs.md) | [keywords](./sequence_num.hpp_kw.md) |
| [`socket.cpp`](../../../../../torch/csrc/distributed/c10d/socket.cpp) | Source code | [docs](./socket.cpp_docs.md) | [keywords](./socket.cpp_kw.md) |
| [`socket.h`](../../../../../torch/csrc/distributed/c10d/socket.h) | Source code | [docs](./socket.h_docs.md) | [keywords](./socket.h_kw.md) |
| [`socket_fmt.h`](../../../../../torch/csrc/distributed/c10d/socket_fmt.h) | Source code | [docs](./socket_fmt.h_docs.md) | [keywords](./socket_fmt.h_kw.md) |


## Navigation

- **Parent Folder**: [..](../index.md)
- **Folder Documentation**: [doc.md](./doc.md)
- **Keyword Index**: [sub.md](./sub.md)

---

*Generated by PyTorch Repository Documentation System*
