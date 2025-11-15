# Keyword Index: `aten/src/ATen/Context.h`

## File Information

- **Original File**: [aten/src/ATen/Context.h](../../../../aten/src/ATen/Context.h)
- **Documentation**: [`Context.h_docs.md`](./Context.h_docs.md)
- **Folder**: `aten/src/ATen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`CuBLASReductionOption`**: [Context.h_docs.md](./Context.h_docs.md)
- **`TORCH_API`**: [Context.h_docs.md](./Context.h_docs.md)
- **`Tensor`**: [Context.h_docs.md](./Context.h_docs.md)
- **`implements`**: [Context.h_docs.md](./Context.h_docs.md)

### Functions

- **`example_func`**: [Context.h_docs.md](./Context.h_docs.md)
- **`getDeviceFromPtr`**: [Context.h_docs.md](./Context.h_docs.md)
- **`getNumGPUs`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasCKGEMM`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasCKSDPA`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasCUDA`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasCUDART`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasCuBLASLt`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasCuDNN`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasCuSOLVER`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasEigenSparse`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasHIP`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasHPU`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasIPU`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasKleidiAI`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasLAPACK`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasLazy`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasMAGMA`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasMAIA`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasMKL`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasMKLDNN`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasMPS`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasMTIA`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasOpenMP`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasROCM`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasXLA`**: [Context.h_docs.md](./Context.h_docs.md)
- **`hasXPU`**: [Context.h_docs.md](./Context.h_docs.md)
- **`if`**: [Context.h_docs.md](./Context.h_docs.md)
- **`init`**: [Context.h_docs.md](./Context.h_docs.md)
- **`isPinnedPtr`**: [Context.h_docs.md](./Context.h_docs.md)
- **`lazyInitCUDA`**: [Context.h_docs.md](./Context.h_docs.md)
- **`lazyInitDevice`**: [Context.h_docs.md](./Context.h_docs.md)
- **`lazyInitHIP`**: [Context.h_docs.md](./Context.h_docs.md)
- **`lazyInitMTIA`**: [Context.h_docs.md](./Context.h_docs.md)
- **`lazyInitPrivateUse1`**: [Context.h_docs.md](./Context.h_docs.md)
- **`lazyInitXPU`**: [Context.h_docs.md](./Context.h_docs.md)
- **`manual_seed`**: [Context.h_docs.md](./Context.h_docs.md)
- **`versionCUDART`**: [Context.h_docs.md](./Context.h_docs.md)
- **`versionCuDNN`**: [Context.h_docs.md](./Context.h_docs.md)
- **`versionCuDNNFrontend`**: [Context.h_docs.md](./Context.h_docs.md)
- **`versionRuntimeCuDNN`**: [Context.h_docs.md](./Context.h_docs.md)

### Includes

- **`ATen/BlasBackend.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/CPUGeneratorImpl.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/DeviceAccelerator.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/LinalgBackend.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/ROCmFABackend.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/SDPBackend.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/core/ATenGeneral.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/core/DeprecatedTypeProperties.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/core/Generator.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/core/LegacyTypeDispatch.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/AcceleratorHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/CUDAHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/HIPHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/HPUHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/IPUHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/MAIAHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/MPSHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/MTIAHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/PrivateUse1HooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/XLAHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`ATen/detail/XPUHooksInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`c10/core/QEngine.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`c10/core/impl/DeviceGuardImplInterface.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`c10/util/CallOnce.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`c10/util/Exception.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`c10/util/env.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`c10/util/hash.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`c10/util/irange.h`**: [Context.h_docs.md](./Context.h_docs.md)
- **`cstdint`**: [Context.h_docs.md](./Context.h_docs.md)
- **`map`**: [Context.h_docs.md](./Context.h_docs.md)
- **`mutex`**: [Context.h_docs.md](./Context.h_docs.md)
- **`unordered_map`**: [Context.h_docs.md](./Context.h_docs.md)

### Namespaces

- **`at`**: [Context.h_docs.md](./Context.h_docs.md)


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
