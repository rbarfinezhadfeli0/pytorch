# Documentation: `docs/torch/utils/hipify/cuda_to_hip_mappings.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/hipify/cuda_to_hip_mappings.py_docs.md`
- **Size**: 52,739 bytes (51.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/hipify/cuda_to_hip_mappings.py`

## File Metadata

- **Path**: `torch/utils/hipify/cuda_to_hip_mappings.py`
- **Size**: 398,827 bytes (389.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import collections
import os

from .constants import (API_BLAS, API_C10, API_CAFFE2, API_DRIVER, API_FFT,
                        API_PYTORCH, API_RAND, API_ROCTX, API_RTC, API_RUNTIME,
                        API_SPECIAL, API_ROCMSMI, CONV_CACHE, CONV_CONTEXT, CONV_D3D9,
                        CONV_D3D10, CONV_D3D11, CONV_DEF, CONV_DEVICE,
                        CONV_DEVICE_FUNC, CONV_EGL, CONV_ERROR, CONV_EVENT,
                        CONV_EXEC, CONV_GL, CONV_GRAPHICS, CONV_INCLUDE,
                        CONV_INCLUDE_CUDA_MAIN_H, CONV_INIT, CONV_JIT,
                        CONV_MATH_FUNC, CONV_MEM, CONV_MODULE,
                        CONV_NUMERIC_LITERAL, CONV_OCCUPANCY, CONV_OTHER,
                        CONV_PEER, CONV_SPECIAL_FUNC, CONV_STREAM,
                        CONV_SURFACE, CONV_TEX, CONV_THREAD, CONV_TYPE,
                        CONV_VDPAU, CONV_VERSION, HIP_UNSUPPORTED)

""" Mapping of CUDA functions, include files, constants, and types to ROCm/HIP equivalents
This closely follows the implementation in hipify-clang
https://github.com/ROCm/hip/blob/59071b895ed1c86d9698b4c859cefcdd5acda06f/hipify-clang/src/CUDA2HipMap.cpp
and its structure.
There are different maps for fundamental names, include files, identifies, sparse, and
PyTorch specific translations.
Each of the entries in these maps translates a CUDA string to a tuple containing the
ROCm/HIP string, a type and API annotation and - optionally - an annotation if it is not
supported in ROCm/HIP yet.
"""

_IS_FBCODE = os.environ.get("IS_FBCODE", "0") == "1"

# FBCODE compiles against rccl sources instead of an installed rccl package.
# The header location is src/rccl.h versus rccl/rccl.h, respectively.
_RCCL_HEADER = "<rccl.h>" if _IS_FBCODE else "<rccl/rccl.h>"

# List of math functions that should be replaced inside device code only.
MATH_TRANSPILATIONS = collections.OrderedDict(
    [
        ("std::max", ("::max")),
        ("std::min", ("::min")),
        ("std::ceil", ("::ceil")),
        ("std::floor", ("::floor")),
        ("std::exp", ("::exp")),
        ("std::log", ("::log")),
        ("std::pow", ("::pow")),
        ("std::fabs", ("::fabs")),
        ("std::fmod", ("::fmod")),
        ("std::remainder", ("::remainder")),
        ("std::frexp", ("::frexp")),
    ]
)

# pyrefly: ignore [no-matching-overload]
CUDA_TYPE_NAME_MAP = collections.OrderedDict(
    [
        ("CUresult", ("hipError_t", CONV_TYPE, API_DRIVER)),
        ("cudaError_t", ("hipError_t", CONV_TYPE, API_RUNTIME)),
        ("cudaError", ("hipError_t", CONV_TYPE, API_RUNTIME)),
        (
            "CUDA_ARRAY3D_DESCRIPTOR",
            ("HIP_ARRAY3D_DESCRIPTOR", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUDA_ARRAY_DESCRIPTOR", ("HIP_ARRAY_DESCRIPTOR", CONV_TYPE, API_DRIVER)),
        ("CUDA_MEMCPY2D", ("hip_Memcpy2D", CONV_TYPE, API_DRIVER)),
        ("CUDA_MEMCPY3D", ("HIP_MEMCPY3D", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUDA_MEMCPY3D_PEER",
            ("HIP_MEMCPY3D_PEER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_POINTER_ATTRIBUTE_P2P_TOKENS",
            (
                "HIP_POINTER_ATTRIBUTE_P2P_TOKENS",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_RESOURCE_DESC",
            ("HIP_RESOURCE_DESC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_RESOURCE_VIEW_DESC",
            ("HIP_RESOURCE_VIEW_DESC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUipcEventHandle",
            ("hipIpcEventHandle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUipcMemHandle", ("hipIpcMemHandle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUaddress_mode", ("hipAddress_mode", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUarray_cubemap_face",
            ("hipArray_cubemap_face", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUarray_format", ("hipArray_format", CONV_TYPE, API_DRIVER)),
        ("CUcomputemode", ("hipComputemode", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUmem_advise", ("hipMemAdvise", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUmem_range_attribute",
            ("hipMemRangeAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUctx_flags", ("hipCctx_flags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUdevice", ("hipDevice_t", CONV_TYPE, API_DRIVER)),
        ("CUdevice_attribute_enum", ("hipDeviceAttribute_t", CONV_TYPE, API_DRIVER)),
        ("CUdevice_attribute", ("hipDeviceAttribute_t", CONV_TYPE, API_DRIVER)),
        ("CUpointer_attribute", ("hipPointer_attribute", CONV_TYPE, API_DRIVER)),
        ("CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL", ("HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL", CONV_TYPE, API_DRIVER)),
        ("CU_POINTER_ATTRIBUTE_BUFFER_ID", ("HIP_POINTER_ATTRIBUTE_BUFFER_ID", CONV_TYPE, API_DRIVER)),
        ("CUdeviceptr", ("hipDeviceptr_t", CONV_TYPE, API_DRIVER)),
        ("CUarray_st", ("hipArray", CONV_TYPE, API_DRIVER)),
        ("CUarray", ("hipArray *", CONV_TYPE, API_DRIVER)),
        ("CUdevprop_st", ("hipDeviceProp_t", CONV_TYPE, API_DRIVER)),
        ("CUdevprop", ("hipDeviceProp_t", CONV_TYPE, API_DRIVER)),
        ("CUfunction", ("hipFunction_t", CONV_TYPE, API_DRIVER)),
        (
            "CUgraphicsResource",
            ("hipGraphicsResource_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUmipmappedArray",
            ("hipMipmappedArray_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUfunction_attribute",
            ("hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUfunction_attribute_enum",
            ("hipFuncAttribute_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsMapResourceFlags",
            ("hipGraphicsMapFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsMapResourceFlags_enum",
            ("hipGraphicsMapFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsRegisterFlags",
            ("hipGraphicsRegisterFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUgraphicsRegisterFlags_enum",
            ("hipGraphicsRegisterFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUoccupancy_flags",
            ("hipOccupancyFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUoccupancy_flags_enum",
            ("hipOccupancyFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUfunc_cache_enum", ("hipFuncCache", CONV_TYPE, API_DRIVER)),
        ("CUfunc_cache", ("hipFuncCache", CONV_TYPE, API_DRIVER)),
        ("CUipcMem_flags", ("hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUipcMem_flags_enum",
            ("hipIpcMemFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUjit_cacheMode", ("hipJitCacheMode", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUjit_cacheMode_enum",
            ("hipJitCacheMode", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUjit_fallback", ("hipJitFallback", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUjit_fallback_enum",
            ("hipJitFallback", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUjit_option", ("hipJitOption", CONV_JIT, API_DRIVER)),
        ("CUjit_option_enum", ("hipJitOption", CONV_JIT, API_DRIVER)),
        ("CUjit_target", ("hipJitTarget", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUjit_target_enum", ("hipJitTarget", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUjitInputType", ("hipJitInputType", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUjitInputType_enum",
            ("hipJitInputType", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUlimit", ("hipLimit_t", CONV_TYPE, API_DRIVER)),
        ("CUlimit_enum", ("hipLimit_t", CONV_TYPE, API_DRIVER)),
        ("CUmemAccessDesc", ("hipMemAccessDesc", CONV_TYPE, API_DRIVER)),
        ("CUmemAccessDesc_st", ("hipMemAccessDesc", CONV_TYPE, API_DRIVER)),
        ("CUmemAccessDesc_v1", ("hipMemAccessDesc", CONV_TYPE, API_DRIVER)),
        (
            "CUmemAttach_flags",
            ("hipMemAttachFlags_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUmemAttach_flags_enum",
            ("hipMemAttachFlags_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUmemAllocationGranularity_flags", ("hipMemAllocationGranularity_flags", CONV_TYPE, API_DRIVER)),
        ("CUmemAllocationGranularity_flags_enum", ("hipMemAllocationGranularity_flags", CONV_TYPE, API_DRIVER)),
        ("CUmemAllocationHandleType", ("hipMemAllocationHandleType", CONV_TYPE, API_DRIVER)),
        ("CUmemAllocationHandleType_enum", ("hipMemAllocationHandleType", CONV_TYPE, API_DRIVER)),
        ("CUmemAllocationProp", ("hipMemAllocationProp", CONV_TYPE, API_DRIVER)),
        ("CUmemAllocationProp_st", ("hipMemAllocationProp", CONV_TYPE, API_DRIVER)),
        ("CUmemAllocationProp_v1", ("hipMemAllocationProp", CONV_TYPE, API_DRIVER)),
        ("CUmemAllocationType", ("hipMemAllocationType", CONV_TYPE, API_DRIVER)),
        ("CUmemAllocationType_enum", ("hipMemAllocationType", CONV_TYPE, API_DRIVER)),
        ("CUmemGenericAllocationHandle", ("hipMemGenericAllocationHandle_t", CONV_TYPE, API_DRIVER)),
        ("CUmemGenericAllocationHandle_v1", ("hipMemGenericAllocationHandle_t", CONV_TYPE, API_DRIVER)),
        ("CUmemHandleType", ("hipMemHandleType", CONV_TYPE, API_DRIVER)),
        ("CUmemHandleType_enum", ("hipMemHandleType", CONV_TYPE, API_DRIVER)),
        ("CUmemLocation", ("hipMemLocation", CONV_TYPE, API_DRIVER)),
        ("CUmemLocationType", ("hipMemLocationType", CONV_TYPE, API_DRIVER)),
        ("CUmemLocationType_enum", ("hipMemLocationType", CONV_TYPE, API_DRIVER)),
        ("CUmemLocation_st", ("hipMemLocation", CONV_TYPE, API_DRIVER)),
        ("CUmemLocation_v1", ("hipMemLocation", CONV_TYPE, API_DRIVER)),
        ("CUmemOperationType", ("hipMemOperationType", CONV_TYPE, API_DRIVER)),
        ("CUmemOperationType_enum", ("hipMemOperationType", CONV_TYPE, API_DRIVER)),
        ("CUmemPoolHandle_st", ("ihipMemPoolHandle_t", CONV_TYPE, API_DRIVER)),
        ("CUmemPoolProps", ("hipMemPoolProps", CONV_TYPE, API_DRIVER)),
        ("CUmemPoolProps_st", ("hipMemPoolProps", CONV_TYPE, API_DRIVER)),
        ("CUmemPoolProps_v1", ("hipMemPoolProps", CONV_TYPE, API_DRIVER)),
        ("CUmemPoolPtrExportData", ("hipMemPoolPtrExportData", CONV_TYPE, API_DRIVER)),
        ("CUmemPoolPtrExportData_st", ("hipMemPoolPtrExportData", CONV_TYPE, API_DRIVER)),
        ("CUmemPoolPtrExportData_v1", ("hipMemPoolPtrExportData", CONV_TYPE, API_DRIVER)),
        ("CUmemPool_attribute", ("hipMemPoolAttr", CONV_TYPE, API_DRIVER)),
        ("CUmemPool_attribute_enum", ("hipMemPoolAttr", CONV_TYPE, API_DRIVER)),
        ("CUmem_advise_enum", ("hipMemoryAdvise", CONV_TYPE, API_DRIVER)),
        ("CUmem_range_attribute_enum", ("hipMemRangeAttribute", CONV_TYPE, API_DRIVER)),
        ("CUmemoryPool", ("hipMemPool_t", CONV_TYPE, API_DRIVER)),
        ("CUmemorytype", ("hipMemType_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUmemorytype_enum", ("hipMemType_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUresourcetype", ("hipResourceType", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUresourcetype_enum",
            ("hipResourceType", CONV_TEX, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUresourceViewFormat", ("hipResourceViewFormat", CONV_TEX, API_DRIVER)),
        ("CUresourceViewFormat_enum", ("hipResourceViewFormat", CONV_TEX, API_DRIVER)),
        ("CUsharedconfig", ("hipSharedMemConfig", CONV_TYPE, API_DRIVER)),
        ("CUsharedconfig_enum", ("hipSharedMemConfig", CONV_TYPE, API_DRIVER)),
        ("CUcontext", ("hipCtx_t", CONV_TYPE, API_DRIVER)),
        ("CUmodule", ("hipModule_t", CONV_TYPE, API_DRIVER)),
        ("CUstream", ("hipStream_t", CONV_TYPE, API_DRIVER)),
        ("CUstream_st", ("ihipStream_t", CONV_TYPE, API_DRIVER)),
        ("CUstreamCallback", ("hipStreamCallback_t", CONV_TYPE, API_DRIVER)),
        ("CUsurfObject", ("hipSurfaceObject", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUsurfref",
            ("hipSurfaceReference_t", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUtexObject", ("hipTextureObject_t", CONV_TYPE, API_DRIVER)),
        ("CUtexref", ("textureReference", CONV_TYPE, API_DRIVER)),
        ("CUstream_flags", ("hipStreamFlags", CONV_TYPE, API_DRIVER)),
        (
            "CUstreamWaitValue_flags",
            ("hipStreamWaitValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUstreamWriteValue_flags",
            ("hipStreamWriteValueFlags", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUstreamBatchMemOpType",
            ("hipStreamBatchMemOpType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUdevice_P2PAttribute",
            ("hipDeviceP2PAttribute", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUevent", ("hipEvent_t", CONV_TYPE, API_DRIVER)),
        ("CUevent_st", ("ihipEvent_t", CONV_TYPE, API_DRIVER)),
        ("CUevent_flags", ("hipEventFlags", CONV_EVENT, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUfilter_mode", ("hipTextureFilterMode", CONV_TEX, API_DRIVER)),
        ("CUGLDeviceList", ("hipGLDeviceList", CONV_GL, API_DRIVER, HIP_UNSUPPORTED)),
        ("CUGLmap_flags", ("hipGLMapFlags", CONV_GL, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUd3d9DeviceList",
            ("hipD3D9DeviceList", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d9map_flags",
            ("hipD3D9MapFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d9register_flags",
            ("hipD3D9RegisterFlags", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10DeviceList",
            ("hipd3d10DeviceList", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10map_flags",
            ("hipD3D10MapFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d10register_flags",
            ("hipD3D10RegisterFlags", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUd3d11DeviceList",
            ("hipd3d11DeviceList", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUeglStreamConnection_st",
            ("hipEglStreamConnection", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUeglStreamConnection",
            ("hipEglStreamConnection", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "libraryPropertyType_t",
            ("hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "libraryPropertyType",
            ("hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaStreamCallback_t", ("hipStreamCallback_t", CONV_TYPE, API_RUNTIME)),
        ("cudaArray", ("hipArray", CONV_MEM, API_RUNTIME)),
        ("cudaArray_t", ("hipArray_t", CONV_MEM, API_RUNTIME)),
        ("cudaArray_const_t", ("hipArray_const_t", CONV_MEM, API_RUNTIME)),
        ("cudaMipmappedArray_t", ("hipMipmappedArray_t", CONV_MEM, API_RUNTIME)),
        (
            "cudaMipmappedArray_const_t",
            ("hipMipmappedArray_const_t", CONV_MEM, API_RUNTIME),
        ),
        ("cudaArrayDefault", ("hipArrayDefault", CONV_MEM, API_RUNTIME)),
        ("cudaArrayLayered", ("hipArrayLayered", CONV_MEM, API_RUNTIME)),
        (
            "cudaArraySurfaceLoadStore",
            ("hipArraySurfaceLoadStore", CONV_MEM, API_RUNTIME),
        ),
        ("cudaArrayCubemap", ("hipArrayCubemap", CONV_MEM, API_RUNTIME)),
        ("cudaArrayTextureGather", ("hipArrayTextureGather", CONV_MEM, API_RUNTIME)),
        ("cudaMemoryAdvise", ("hipMemoryAdvise", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED)),
        (
            "cudaMemRangeAttribute",
            ("hipMemRangeAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaMemcpyKind", ("hipMemcpyKind", CONV_MEM, API_RUNTIME)),
        ("cudaMemoryType", ("hipMemoryType", CONV_MEM, API_RUNTIME)),
        ("cudaExtent", ("hipExtent", CONV_MEM, API_RUNTIME)),
        ("cudaPitchedPtr", ("hipPitchedPtr", CONV_MEM, API_RUNTIME)),
        ("cudaPos", ("hipPos", CONV_MEM, API_RUNTIME)),
        ("cudaEvent_t", ("hipEvent_t", CONV_TYPE, API_RUNTIME)),
        ("cudaStream_t", ("hipStream_t", CONV_TYPE, API_RUNTIME)),
        ("cudaHostFn_t", ("hipHostFn_t", CONV_TYPE, API_RUNTIME)),
        ("cudaPointerAttributes", ("hipPointerAttribute_t", CONV_TYPE, API_RUNTIME)),
        ("cudaDeviceAttr", ("hipDeviceAttribute_t", CONV_TYPE, API_RUNTIME)),
        ("cudaDeviceProp", ("hipDeviceProp_t", CONV_TYPE, API_RUNTIME)),
        (
            "cudaDeviceP2PAttr",
            ("hipDeviceP2PAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaComputeMode",
            ("hipComputeMode", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaFuncCache", ("hipFuncCache_t", CONV_CACHE, API_RUNTIME)),
        (
            "cudaFuncAttributes",
            ("hipFuncAttributes", CONV_EXEC, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaSharedMemConfig", ("hipSharedMemConfig", CONV_TYPE, API_RUNTIME)),
        ("cudaLimit", ("hipLimit_t", CONV_TYPE, API_RUNTIME)),
        ("cudaOutputMode", ("hipOutputMode", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED)),
        ("cudaTextureReadMode", ("hipTextureReadMode", CONV_TEX, API_RUNTIME)),
        ("cudaTextureFilterMode", ("hipTextureFilterMode", CONV_TEX, API_RUNTIME)),
        ("cudaChannelFormatKind", ("hipChannelFormatKind", CONV_TEX, API_RUNTIME)),
        ("cudaChannelFormatDesc", ("hipChannelFormatDesc", CONV_TEX, API_RUNTIME)),
        ("cudaResourceDesc", ("hipResourceDesc", CONV_TEX, API_RUNTIME)),
        ("cudaResourceViewDesc", ("hipResourceViewDesc", CONV_TEX, API_RUNTIME)),
        ("cudaTextureDesc", ("hipTextureDesc", CONV_TEX, API_RUNTIME)),
        (
            "surfaceReference",
            ("hipSurfaceReference", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaTextureObject_t", ("hipTextureObject_t", CONV_TEX, API_RUNTIME)),
        ("cudaResourceType", ("hipResourceType", CONV_TEX, API_RUNTIME)),
        ("cudaResourceViewFormat", ("hipResourceViewFormat", CONV_TEX, API_RUNTIME)),
        ("cudaTextureAddressMode", ("hipTextureAddressMode", CONV_TEX, API_RUNTIME)),
        (
            "cudaSurfaceBoundaryMode",
            ("hipSurfaceBoundaryMode", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaSurfaceFormatMode",
            ("hipSurfaceFormatMode", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaTextureType1D", ("hipTextureType1D", CONV_TEX, API_RUNTIME)),
        ("cudaTextureType2D", ("hipTextureType2D", CONV_TEX, API_RUNTIME)),
        ("cudaTextureType3D", ("hipTextureType3D", CONV_TEX, API_RUNTIME)),
        ("cudaTextureTypeCubemap", ("hipTextureTypeCubemap", CONV_TEX, API_RUNTIME)),
        (
            "cudaTextureType1DLayered",
            ("hipTextureType1DLayered", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaTextureType2DLayered",
            ("hipTextureType2DLayered", CONV_TEX, API_RUNTIME),
        ),
        (
            "cudaTextureTypeCubemapLayered",
            ("hipTextureTypeCubemapLayered", CONV_TEX, API_RUNTIME),
        ),
        ("cudaIpcEventHandle_t", ("hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME)),
        ("cudaIpcEventHandle_st", ("hipIpcEventHandle_t", CONV_TYPE, API_RUNTIME)),
        ("cudaIpcMemHandle_t", ("hipIpcMemHandle_t", CONV_TYPE, API_RUNTIME)),
        ("cudaIpcMemHandle_st", ("hipIpcMemHandle_t", CONV_TYPE, API_RUNTIME)),
        (
            "cudaGraphicsCubeFace",
            ("hipGraphicsCubeFace", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsMapFlags",
            ("hipGraphicsMapFlags", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGraphicsRegisterFlags",
            ("hipGraphicsRegisterFlags", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaGLDeviceList",
            ("hipGLDeviceList", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cudaGLMapFlags", ("hipGLMapFlags", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED)),
        (
            "cudaD3D9DeviceList",
            ("hipD3D9DeviceList", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9MapFlags",
            ("hipD3D9MapFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D9RegisterFlags",
            ("hipD3D9RegisterFlags", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10DeviceList",
            ("hipd3d10DeviceList", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10MapFlags",
            ("hipD3D10MapFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D10RegisterFlags",
            ("hipD3D10RegisterFlags", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaD3D11DeviceList",
            ("hipd3d11DeviceList", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaEglStreamConnection",
            ("hipEglStreamConnection", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("cublasHandle_t", ("hipblasHandle_t", CONV_TYPE, API_BLAS)),
        ("cublasOperation_t", ("hipblasOperation_t", CONV_TYPE, API_BLAS)),
        ("cublasStatus_t", ("hipblasStatus_t", CONV_TYPE, API_BLAS)),
        ("cublasFillMode_t", ("hipblasFillMode_t", CONV_TYPE, API_BLAS)),
        ("cublasDiagType_t", ("hipblasDiagType_t", CONV_TYPE, API_BLAS)),
        ("cublasSideMode_t", ("hipblasSideMode_t", CONV_TYPE, API_BLAS)),
        ("cublasPointerMode_t", ("hipblasPointerMode_t", CONV_TYPE, API_BLAS)),
        ("cublasGemmAlgo_t", ("hipblasGemmAlgo_t", CONV_TYPE, API_BLAS)),
        (
            "cublasAtomicsMode_t",
            ("hipblasAtomicsMode_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED),
        ),
        (
            "cublasDataType_t",
            ("hipblasDatatype_t", CONV_TYPE, API_BLAS, HIP_UNSUPPORTED),
        ),
        ("curandStatus", ("hiprandStatus_t", CONV_TYPE, API_RAND)),
        ("curandStatus_t", ("hiprandStatus_t", CONV_TYPE, API_RAND)),
        ("curandRngType", ("hiprandRngType_t", CONV_TYPE, API_RAND)),
        ("curandRngType_t", ("hiprandRngType_t", CONV_TYPE, API_RAND)),
        ("curandGenerator_st", ("hiprandGenerator_st", CONV_TYPE, API_RAND)),
        ("curandGenerator_t", ("hiprandGenerator_t", CONV_TYPE, API_RAND)),
        (
            "curandDirectionVectorSet",
            ("hiprandDirectionVectorSet_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDirectionVectorSet_t",
            ("hiprandDirectionVectorSet_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curandOrdering", ("hiprandOrdering_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED)),
        (
            "curandOrdering_t",
            ("hiprandOrdering_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistribution_st",
            ("hiprandDistribution_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2V_st",
            ("hiprandDistribution_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistribution_t",
            ("hiprandDistribution_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2V_t",
            ("hiprandDistribution_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionShift_st",
            ("hiprandDistributionShift_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionShift_t",
            ("hiprandDistributionShift_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionM2Shift_st",
            ("hiprandDistributionM2Shift_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDistributionM2Shift_t",
            ("hiprandDistributionM2Shift_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2_st",
            ("hiprandHistogramM2_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2_t",
            ("hiprandHistogramM2_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2K_st",
            ("hiprandHistogramM2K_st", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandHistogramM2K_t",
            ("hiprandHistogramM2K_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandDiscreteDistribution_st",
            ("hiprandDiscreteDistribution_st", CONV_TYPE, API_RAND),
        ),
        (
            "curandDiscreteDistribution_t",
            ("hiprandDiscreteDistribution_t", CONV_TYPE, API_RAND),
        ),
        ("curandMethod", ("hiprandMethod_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED)),
        ("curandMethod_t", ("hiprandMethod_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED)),
        (
            "curandDirectionVectors32_t",
            ("hiprandDirectionVectors32_t", CONV_TYPE, API_RAND),
        ),
        (
            "curandDirectionVectors64_t",
            ("hiprandDirectionVectors64_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curandStateMtgp32_t", ("hiprandStateMtgp32_t", CONV_TYPE, API_RAND)),
        ("curandStateMtgp32", ("hiprandStateMtgp32_t", CONV_TYPE, API_RAND)),
        (
            "curandStateScrambledSobol64_t",
            ("hiprandStateScrambledSobol64_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandStateSobol64_t",
            ("hiprandStateSobol64_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        (
            "curandStateScrambledSobol32_t",
            ("hiprandStateScrambledSobol32_t", CONV_TYPE, API_RAND, HIP_UNSUPPORTED),
        ),
        ("curandStateSobol32_t", ("hiprandStateSobol32_t", CONV_TYPE, API_RAND)),
        ("curandStateMRG32k3a_t", ("hiprandStateMRG32k3a_t", CONV_TYPE, API_RAND)),
        (
            "curandStatePhilox4_32_10_t",
            ("hiprandStatePhilox4_32_10_t", CONV_TYPE, API_RAND),
        ),
        ("curandStateXORWOW_t", ("hiprandStateXORWOW_t", CONV_TYPE, API_RAND)),
        ("curandState_t", ("hiprandState_t", CONV_TYPE, API_RAND)),
        ("curandState", ("hiprandState_t", CONV_TYPE, API_RAND)),
        ("CUuuid", ("hipUUID", CONV_TYPE, API_RUNTIME)),
        ("cudaGraph_t", ("hipGraph_t", CONV_TYPE, API_RAND)),
        ("cudaGraphNode_t", ("hipGraphNode_t", CONV_TYPE, API_RAND)),
        ("cudaGraphExec_t", ("hipGraphExec_t", CONV_TYPE, API_RAND)),
        ("__nv_bfloat16", ("__hip_bfloat16", CONV_TYPE, API_RUNTIME)),
        ("__nv_bfloat162", ("__hip_bfloat162", CONV_TYPE, API_RUNTIME)),
    ]
)

# pyrefly: ignore [no-matching-overload]
CUDA_INCLUDE_MAP = collections.OrderedDict(
    [
        # since pytorch uses "\b{pattern}\b" as the actual re pattern,
        # patterns listed here have to begin and end with alnum chars
        (
            "include <cuda.h",
            ("include <hip/hip_runtime.h", CONV_INCLUDE_CUDA_MAIN_H, API_DRIVER),
        ),
        (
            'include "cuda.h',
            ('include "hip/hip_runtime.h', CONV_INCLUDE_CUDA_MAIN_H, API_DRIVER),
        ),
        (
            "cuda_runtime.h",
            ("hip/hip_runtime.h", CONV_INCLUDE_CUDA_MAIN_H, API_RUNTIME),
        ),
        ("cuda_runtime_api.h", ("hip/hip_runtime_api.h", CONV_INCLUDE, API_RUNTIME)),
        ("cuda_profiler_api.h", ("hip/hip_runtime_api.h", CONV_INCLUDE, API_RUNTIME)),
        (
            "channel_descriptor.h",
            ("hip/channel_descriptor.h", CONV_INCLUDE, API_RUNTIME),
        ),
        ('include "device_functions.h', ('include "hip/device_functions.h', CONV_INCLUDE, API_RUNTIME)),
        ('include <device_functions.h', ('include <hip/device_functions.h', CONV_INCLUDE, API_RUNTIME)),
        ('include "driver_types.h', ('include "hip/driver_types.h', CONV_INCLUDE, API_RUNTIME)),
        ('include <driver_types.h', ('include <hip/driver_types.h', CONV_INCLUDE, API_RUNTIME)),
        ('include "library_types.h', ('include "hip/library_types.h', CONV_INCLUDE, API_RUNTIME)),
        ('include <library_types.h', ('include <hip/library_types.h', CONV_INCLUDE, API_RUNTIME)),
        ("cuComplex.h", ("hip/hip_complex.h", CONV_INCLUDE, API_RUNTIME)),
        ("cuda_fp16.h", ("hip/hip_fp16.h", CONV_INCLUDE, API_RUNTIME)),
        ("cuda_bf16.h", ("hip/hip_bf16.h", CONV_INCLUDE, API_RUNTIME)),
        (
            "cuda_texture_types.h",
            ("hip/hip_texture_types.h", CONV_INCLUDE, API_RUNTIME),
        ),
        ("cooperative_groups.h", ("hip/hip_cooperative_groups.h", CONV_INCLUDE, API_RUNTIME)),
        ("vector_types.h", ("hip/hip_vector_types.h", CONV_INCLUDE, API_RUNTIME)),
        ("cublas.h", ("hipblas/hipblas.h", CONV_INCLUDE_CUDA_MAIN_H, API_BLAS)),
        ("cublas_v2.h", ("hipblas/hipblas.h", CONV_INCLUDE_CUDA_MAIN_H, API_BLAS)),
        ("cublasLt.h", ("hipblaslt/hipblaslt.h", CONV_INCLUDE_CUDA_MAIN_H, API_BLAS)),
        ("curand.h", ("hiprand/hiprand.h", CONV_INCLUDE_CUDA_MAIN_H, API_RAND)),
        ("curand_kernel.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_discrete.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_discrete2.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_globals.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_lognormal.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_mrg32k3a.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_mtgp32.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_mtgp32_host.h", ("hiprand/hiprand_mtgp32_host.h", CONV_INCLUDE, API_RAND)),
        ("curand_mtgp32_kernel.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        (
            "curand_mtgp32dc_p_11213.h",
            ("rocrand/rocrand_mtgp32_11213.h", CONV_INCLUDE, API_RAND),
        ),
        ("curand_normal.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_normal_static.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_philox4x32_x.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_poisson.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_precalc.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("curand_uniform.h", ("hiprand/hiprand_kernel.h", CONV_INCLUDE, API_RAND)),
        ("cusparse.h", ("hipsparse/hipsparse.h", CONV_INCLUDE, API_RAND)),
        ("cusparseLt.h", ("hipsparselt/hipsparselt.h", CONV_INCLUDE, API_RAND)),
        ("cufft.h", ("hipfft/hipfft.h", CONV_INCLUDE, API_BLAS)),
        ("cufftXt.h", ("hipfft/hipfftXt.h", CONV_INCLUDE, API_BLAS)),
        # PyTorch also has a source file named "nccl.h", so we need to "<"">" to differentiate
        ("<nccl.h>", (_RCCL_HEADER, CONV_INCLUDE, API_RUNTIME)),
        ("nvrtc.h", ("hip/hiprtc.h", CONV_INCLUDE, API_RTC)),
        ("thrust/system/cuda", ("thrust/system/hip", CONV_INCLUDE, API_BLAS)),
        ("cub/util_allocator.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_reduce.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_raking_layout.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/cub.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/config.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/util_ptx.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/util_type.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_run_length_encode.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_load.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_store.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/block/block_scan.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_radix_sort.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_reduce.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_scan.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("cub/device/device_select.cuh", ("hipcub/hipcub.hpp", CONV_INCLUDE, API_BLAS)),
        ("nvtx3/nvtx3.hpp", ("roctracer/roctx.h", CONV_INCLUDE, API_ROCTX)),
        ("nvToolsExt.h", ("roctracer/roctx.h", CONV_INCLUDE, API_ROCTX)),
        ("nvml.h", ("rocm_smi/rocm_smi.h", CONV_INCLUDE, API_ROCMSMI)),
    ]
)

# pyrefly: ignore [no-matching-overload]
CUDA_IDENTIFIER_MAP = collections.OrderedDict(
    [
        ("__CUDACC__", ("__HIPCC__", CONV_DEF, API_RUNTIME)),
        (
            "CUDA_ERROR_INVALID_CONTEXT",
            ("hipErrorInvalidContext", CONV_TYPE, API_DRIVER),
        ),
        (
            "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            ("hipErrorContextAlreadyCurrent", CONV_TYPE, API_DRIVER),
        ),
        (
            "CUDA_ERROR_ARRAY_IS_MAPPED",
            ("hipErrorArrayIsMapped", CONV_TYPE, API_DRIVER),
        ),
        ("CUDA_ERROR_ALREADY_MAPPED", ("hipErrorAlreadyMapped", CONV_TYPE, API_DRIVER)),
        (
            "CUDA_ERROR_ALREADY_ACQUIRED",
            ("hipErrorAlreadyAcquired", CONV_TYPE, API_DRIVER),
        ),
        ("CUDA_ERROR_NOT_MAPPED", ("hipErrorNotMapped", CONV_TYPE, API_DRIVER)),
        (
            "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
            ("hipErrorNotMappedAsArray", CONV_TYPE, API_DRIVER),
        ),
        (
            "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
            ("hipErrorNotMappedAsPointer", CONV_TYPE, API_DRIVER),
        ),
        (
            "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
            ("hipErrorContextAlreadyInUse", CONV_TYPE, API_DRIVER),
        ),
        ("CUDA_ERROR_INVALID_SOURCE", ("hipErrorInvalidSource", CONV_TYPE, API_DRIVER)),
        ("CUDA_ERROR_FILE_NOT_FOUND", ("hipErrorFileNotFound", CONV_TYPE, API_DRIVER)),
        ("CUDA_ERROR_NOT_FOUND", ("hipErrorNotFound", CONV_TYPE, API_DRIVER)),
        (
            "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
            (
                "hipErrorLaunchIncompatibleTexturing",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
            ("hipErrorPrimaryContextActive", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_CONTEXT_IS_DESTROYED",
            ("hipErrorContextIsDestroyed", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NOT_PERMITTED",
            ("hipErrorNotPermitted", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NOT_SUPPORTED",
            ("hipErrorNotSupported", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMissingConfiguration",
            ("hipErrorMissingConfiguration", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorPriorLaunchFailure",
            ("hipErrorPriorLaunchFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidDeviceFunction",
            ("hipErrorInvalidDeviceFunction", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidConfiguration",
            ("hipErrorInvalidConfiguration", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidPitchValue",
            ("hipErrorInvalidPitchValue", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidSymbol",
            ("hipErrorInvalidSymbol", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidHostPointer",
            ("hipErrorInvalidHostPointer", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidDevicePointer",
            ("hipErrorInvalidDevicePointer", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaErrorInvalidTexture",
            ("hipErrorInvalidTexture", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidTextureBinding",
            ("hipErrorInvalidTextureBinding", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidChannelDescriptor",
            (
                "hipErrorInvalidChannelDescriptor",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorInvalidMemcpyDirection",
            ("hipErrorInvalidMemcpyDirection", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorAddressOfConstant",
            ("hipErrorAddressOfConstant", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTextureFetchFailed",
            ("hipErrorTextureFetchFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTextureNotBound",
            ("hipErrorTextureNotBound", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSynchronizationError",
            ("hipErrorSynchronizationError", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidFilterSetting",
            ("hipErrorInvalidFilterSetting", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidNormSetting",
            ("hipErrorInvalidNormSetting", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMixedDeviceExecution",
            ("hipErrorMixedDeviceExecution", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNotYetImplemented",
            ("hipErrorNotYetImplemented", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMemoryValueTooLarge",
            ("hipErrorMemoryValueTooLarge", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInsufficientDriver",
            ("hipErrorInsufficientDriver", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSetOnActiveProcess",
            ("hipErrorSetOnActiveProcess", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorContextIsDestroyed",
            ("hipErrorContextIsDestroyed", CONV_TYPE, API_RUNTIME),
        ),
        (
            "cudaErrorInvalidSurface",
            ("hipErrorInvalidSurface", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateVariableName",
            ("hipErrorDuplicateVariableName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateTextureName",
            ("hipErrorDuplicateTextureName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDuplicateSurfaceName",
            ("hipErrorDuplicateSurfaceName", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorDevicesUnavailable",
            ("hipErrorDevicesUnavailable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorIncompatibleDriverContext",
            (
                "hipErrorIncompatibleDriverContext",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorDeviceAlreadyInUse",
            ("hipErrorDeviceAlreadyInUse", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchMaxDepthExceeded",
            ("hipErrorLaunchMaxDepthExceeded", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFileScopedTex",
            ("hipErrorLaunchFileScopedTex", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFileScopedSurf",
            ("hipErrorLaunchFileScopedSurf", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorSyncDepthExceeded",
            ("hipErrorSyncDepthExceeded", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchPendingCountExceeded",
            (
                "hipErrorLaunchPendingCountExceeded",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "cudaErrorNotPermitted",
            ("hipErrorNotPermitted", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNotSupported",
            ("hipErrorNotSupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorStartupFailure",
            ("hipErrorStartupFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorApiFailureBase",
            ("hipErrorApiFailureBase", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("CUDA_SUCCESS", ("hipSuccess", CONV_TYPE, API_DRIVER)),
        ("cudaSuccess", ("hipSuccess", CONV_TYPE, API_RUNTIME)),
        ("CUDA_ERROR_INVALID_VALUE", ("hipErrorInvalidValue", CONV_TYPE, API_DRIVER)),
        ("cudaErrorInvalidValue", ("hipErrorInvalidValue", CONV_TYPE, API_RUNTIME)),
        (
            "CUDA_ERROR_OUT_OF_MEMORY",
            ("hipErrorMemoryAllocation", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorMemoryAllocation",
            ("hipErrorMemoryAllocation", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_NOT_INITIALIZED",
            ("hipErrorNotInitialized", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorInitializationError",
            ("hipErrorInitializationError", CONV_TYPE, API_RUNTIME),
        ),
        ("CUDA_ERROR_DEINITIALIZED", ("hipErrorDeinitialized", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorCudartUnloading",
            ("hipErrorDeinitialized", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_DISABLED",
            ("hipErrorProfilerDisabled", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorProfilerDisabled",
            ("hipErrorProfilerDisabled", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
            ("hipErrorProfilerNotInitialized", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorProfilerNotInitialized",
            ("hipErrorProfilerNotInitialized", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_ALREADY_STARTED",
            ("hipErrorProfilerAlreadyStarted", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorProfilerAlreadyStarted",
            ("hipErrorProfilerAlreadyStarted", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
            ("hipErrorProfilerAlreadyStopped", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorProfilerAlreadyStopped",
            ("hipErrorProfilerAlreadyStopped", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_NO_DEVICE", ("hipErrorNoDevice", CONV_TYPE, API_DRIVER)),
        ("cudaErrorNoDevice", ("hipErrorNoDevice", CONV_TYPE, API_RUNTIME)),
        ("CUDA_ERROR_INVALID_DEVICE", ("hipErrorInvalidDevice", CONV_TYPE, API_DRIVER)),
        ("cudaErrorInvalidDevice", ("hipErrorInvalidDevice", CONV_TYPE, API_RUNTIME)),
        ("CUDA_ERROR_INVALID_IMAGE", ("hipErrorInvalidImage", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorInvalidKernelImage",
            ("hipErrorInvalidImage", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_MAP_FAILED", ("hipErrorMapFailed", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorMapBufferObjectFailed",
            ("hipErrorMapFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        ("CUDA_ERROR_UNMAP_FAILED", ("hipErrorUnmapFailed", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorUnmapBufferObjectFailed",
            ("hipErrorUnmapFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NO_BINARY_FOR_GPU",
            ("hipErrorNoBinaryForGpu", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorNoKernelImageForDevice",
            ("hipErrorNoBinaryForGpu", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_ECC_UNCORRECTABLE",
            ("hipErrorECCNotCorrectable", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorECCUncorrectable",
            ("hipErrorECCNotCorrectable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_UNSUPPORTED_LIMIT",
            ("hipErrorUnsupportedLimit", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorUnsupportedLimit",
            ("hipErrorUnsupportedLimit", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
            ("hipErrorPeerAccessUnsupported", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessUnsupported",
            ("hipErrorPeerAccessUnsupported", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_PTX",
            ("hipErrorInvalidKernelFile", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorInvalidPtx",
            ("hipErrorInvalidKernelFile", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
            ("hipErrorInvalidGraphicsContext", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorInvalidGraphicsContext",
            ("hipErrorInvalidGraphicsContext", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_NVLINK_UNCORRECTABLE",
            ("hipErrorNvlinkUncorrectable", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorNvlinkUncorrectable",
            ("hipErrorNvlinkUncorrectable", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
            ("hipErrorSharedObjectSymbolNotFound", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorSharedObjectSymbolNotFound",
            (
                "hipErrorSharedObjectSymbolNotFound",
                CONV_TYPE,
                API_RUNTIME,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
            ("hipErrorSharedObjectInitFailed", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorSharedObjectInitFailed",
            ("hipErrorSharedObjectInitFailed", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_OPERATING_SYSTEM",
            ("hipErrorOperatingSystem", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorOperatingSystem",
            ("hipErrorOperatingSystem", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_HANDLE",
            ("hipErrorInvalidResourceHandle", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorInvalidResourceHandle",
            ("hipErrorInvalidResourceHandle", CONV_TYPE, API_RUNTIME),
        ),
        ("CUDA_ERROR_NOT_READY", ("hipErrorNotReady", CONV_TYPE, API_DRIVER)),
        ("cudaErrorNotReady", ("hipErrorNotReady", CONV_TYPE, API_RUNTIME)),
        (
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            ("hipErrorIllegalAddress", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorIllegalAddress",
            ("hipErrorIllegalAddress", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            ("hipErrorLaunchOutOfResources", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorLaunchOutOfResources",
            ("hipErrorLaunchOutOfResources", CONV_TYPE, API_RUNTIME),
        ),
        ("CUDA_ERROR_LAUNCH_TIMEOUT", ("hipErrorLaunchTimeOut", CONV_TYPE, API_DRIVER)),
        (
            "cudaErrorLaunchTimeout",
            ("hipErrorLaunchTimeOut", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
            ("hipErrorPeer
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/hipify`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/hipify`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/utils/hipify`):

- [`version.py_kw.md_docs.md`](./version.py_kw.md_docs.md)
- [`cuda_to_hip_mappings.py_kw.md_docs.md`](./cuda_to_hip_mappings.py_kw.md_docs.md)
- [`version.py_docs.md_docs.md`](./version.py_docs.md_docs.md)
- [`hipify_python.py_docs.md_docs.md`](./hipify_python.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`constants.py_docs.md_docs.md`](./constants.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`constants.py_kw.md_docs.md`](./constants.py_kw.md_docs.md)
- [`hipify_python.py_kw.md_docs.md`](./hipify_python.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cuda_to_hip_mappings.py_docs.md_docs.md`
- **Keyword Index**: `cuda_to_hip_mappings.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
