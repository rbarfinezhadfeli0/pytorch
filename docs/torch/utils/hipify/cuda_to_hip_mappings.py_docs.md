# Documentation: cuda_to_hip_mappings.py

## File Metadata
- **Path**: `torch/utils/hipify/cuda_to_hip_mappings.py`
- **Size**: 398827 bytes
- **Lines**: 9488
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
            ("hipErrorPeerAccessAlreadyEnabled", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessAlreadyEnabled",
            ("hipErrorPeerAccessAlreadyEnabled", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
            ("hipErrorPeerAccessNotEnabled", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorPeerAccessNotEnabled",
            ("hipErrorPeerAccessNotEnabled", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_ASSERT",
            ("hipErrorAssert", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorAssert",
            ("hipErrorAssert", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_TOO_MANY_PEERS",
            ("hipErrorTooManyPeers", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorTooManyPeers",
            ("hipErrorTooManyPeers", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
            ("hipErrorHostMemoryAlreadyRegistered", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorHostMemoryAlreadyRegistered",
            ("hipErrorHostMemoryAlreadyRegistered", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
            ("hipErrorHostMemoryNotRegistered", CONV_TYPE, API_DRIVER),
        ),
        (
            "cudaErrorHostMemoryNotRegistered",
            ("hipErrorHostMemoryNotRegistered", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CUDA_ERROR_HARDWARE_STACK_ERROR",
            ("hipErrorHardwareStackError", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorHardwareStackError",
            ("hipErrorHardwareStackError", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            ("hipErrorIllegalInstruction", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorIllegalInstruction",
            ("hipErrorIllegalInstruction", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_MISALIGNED_ADDRESS",
            ("hipErrorMisalignedAddress", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorMisalignedAddress",
            ("hipErrorMisalignedAddress", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_ADDRESS_SPACE",
            ("hipErrorInvalidAddressSpace", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidAddressSpace",
            ("hipErrorInvalidAddressSpace", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_INVALID_PC",
            ("hipErrorInvalidPc", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorInvalidPc",
            ("hipErrorInvalidPc", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_LAUNCH_FAILED",
            ("hipErrorLaunchFailure", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "cudaErrorLaunchFailure",
            ("hipErrorLaunchFailure", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ERROR_UNKNOWN",
            ("hipErrorUnknown", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("cudaErrorUnknown", ("hipErrorUnknown", CONV_TYPE, API_RUNTIME)),
        (
            "CU_TR_ADDRESS_MODE_WRAP",
            ("HIP_TR_ADDRESS_MODE_WRAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_CLAMP",
            ("HIP_TR_ADDRESS_MODE_CLAMP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_MIRROR",
            ("HIP_TR_ADDRESS_MODE_MIRROR", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TR_ADDRESS_MODE_BORDER",
            ("HIP_TR_ADDRESS_MODE_BORDER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_X",
            ("HIP_CUBEMAP_FACE_POSITIVE_X", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_X",
            ("HIP_CUBEMAP_FACE_NEGATIVE_X", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_Y",
            ("HIP_CUBEMAP_FACE_POSITIVE_Y", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_Y",
            ("HIP_CUBEMAP_FACE_NEGATIVE_Y", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_POSITIVE_Z",
            ("HIP_CUBEMAP_FACE_POSITIVE_Z", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CUBEMAP_FACE_NEGATIVE_Z",
            ("HIP_CUBEMAP_FACE_NEGATIVE_Z", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT8",
            ("HIP_AD_FORMAT_UNSIGNED_INT8", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT16",
            ("HIP_AD_FORMAT_UNSIGNED_INT16", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_UNSIGNED_INT32",
            ("HIP_AD_FORMAT_UNSIGNED_INT32", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT8",
            ("HIP_AD_FORMAT_SIGNED_INT8", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT16",
            ("HIP_AD_FORMAT_SIGNED_INT16", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_AD_FORMAT_SIGNED_INT32",
            ("HIP_AD_FORMAT_SIGNED_INT32", CONV_TYPE, API_DRIVER),
        ),
        ("CU_AD_FORMAT_HALF", ("HIP_AD_FORMAT_HALF", CONV_TYPE, API_DRIVER)),
        ("CU_AD_FORMAT_FLOAT", ("HIP_AD_FORMAT_FLOAT", CONV_TYPE, API_DRIVER)),
        (
            "CU_COMPUTEMODE_DEFAULT",
            ("hipComputeModeDefault", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_EXCLUSIVE",
            ("hipComputeModeExclusive", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_PROHIBITED",
            ("hipComputeModeProhibited", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_COMPUTEMODE_EXCLUSIVE_PROCESS",
            ("hipComputeModeExclusiveProcess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_SET_READ_MOSTLY",
            ("hipMemAdviseSetReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_UNSET_READ_MOSTLY",
            ("hipMemAdviseUnsetReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_SET_PREFERRED_LOCATION",
            (
                "hipMemAdviseSetPreferredLocation",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",
            (
                "hipMemAdviseUnsetPreferredLocation",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_ADVISE_SET_ACCESSED_BY",
            ("hipMemAdviseSetAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ADVISE_UNSET_ACCESSED_BY",
            ("hipMemAdviseUnsetAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",
            ("hipMemRangeAttributeReadMostly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION",
            (
                "hipMemRangeAttributePreferredLocation",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY",
            ("hipMemRangeAttributeAccessedBy", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION",
            (
                "hipMemRangeAttributeLastPrefetchLocation",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_CTX_SCHED_AUTO",
            ("HIP_CTX_SCHED_AUTO", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_SPIN",
            ("HIP_CTX_SCHED_SPIN", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_YIELD",
            ("HIP_CTX_SCHED_YIELD", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_BLOCKING_SYNC",
            ("HIP_CTX_SCHED_BLOCKING_SYNC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_BLOCKING_SYNC",
            ("HIP_CTX_BLOCKING_SYNC", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_SCHED_MASK",
            ("HIP_CTX_SCHED_MASK", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_MAP_HOST",
            ("HIP_CTX_MAP_HOST", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_LMEM_RESIZE_TO_MAX",
            ("HIP_CTX_LMEM_RESIZE_TO_MAX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_CTX_FLAGS_MASK",
            ("HIP_CTX_FLAGS_MASK", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LAUNCH_PARAM_BUFFER_POINTER",
            ("HIP_LAUNCH_PARAM_BUFFER_POINTER", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_LAUNCH_PARAM_BUFFER_SIZE",
            ("HIP_LAUNCH_PARAM_BUFFER_SIZE", CONV_TYPE, API_DRIVER),
        ),
        ("CU_LAUNCH_PARAM_END", ("HIP_LAUNCH_PARAM_END", CONV_TYPE, API_DRIVER)),
        (
            "CU_IPC_HANDLE_SIZE",
            ("HIP_IPC_HANDLE_SIZE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_DEVICEMAP",
            ("HIP_MEMHOSTALLOC_DEVICEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_PORTABLE",
            ("HIP_MEMHOSTALLOC_PORTABLE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTALLOC_WRITECOMBINED",
            ("HIP_MEMHOSTALLOC_WRITECOMBINED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_DEVICEMAP",
            ("HIP_MEMHOSTREGISTER_DEVICEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_IOMEMORY",
            ("HIP_MEMHOSTREGISTER_IOMEMORY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMHOSTREGISTER_PORTABLE",
            ("HIP_MEMHOSTREGISTER_PORTABLE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_PARAM_TR_DEFAULT",
            ("HIP_PARAM_TR_DEFAULT", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_LEGACY",
            ("HIP_STREAM_LEGACY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_STREAM_PER_THREAD",
            ("HIP_STREAM_PER_THREAD", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSA_OVERRIDE_FORMAT",
            ("HIP_TRSA_OVERRIDE_FORMAT", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSF_NORMALIZED_COORDINATES",
            ("HIP_TRSF_NORMALIZED_COORDINATES", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TRSF_READ_AS_INTEGER",
            ("HIP_TRSF_READ_AS_INTEGER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CU_TRSF_SRGB", ("HIP_TRSF_SRGB", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED)),
        (
            "CUDA_ARRAY3D_2DARRAY",
            ("HIP_ARRAY3D_LAYERED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_CUBEMAP",
            ("HIP_ARRAY3D_CUBEMAP", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_DEPTH_TEXTURE",
            ("HIP_ARRAY3D_DEPTH_TEXTURE", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_LAYERED",
            ("HIP_ARRAY3D_LAYERED", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_SURFACE_LDST",
            ("HIP_ARRAY3D_SURFACE_LDST", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CUDA_ARRAY3D_TEXTURE_GATHER",
            ("HIP_ARRAY3D_TEXTURE_GATHER", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxThreadsPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X",
            ("hipDeviceAttributeMaxBlockDimX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y",
            ("hipDeviceAttributeMaxBlockDimY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z",
            ("hipDeviceAttributeMaxBlockDimZ", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X",
            ("hipDeviceAttributeMaxGridDimX", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y",
            ("hipDeviceAttributeMaxGridDimY", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z",
            ("hipDeviceAttributeMaxGridDimZ", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK",
            (
                "hipDeviceAttributeMaxSharedMemoryPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK",
            (
                "hipDeviceAttributeMaxSharedMemoryPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY",
            (
                "hipDeviceAttributeTotalConstantMemory",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_WARP_SIZE",
            ("hipDeviceAttributeWarpSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_PITCH",
            ("hipDeviceAttributeMaxPitch", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxRegistersPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK",
            (
                "hipDeviceAttributeMaxRegistersPerBlock",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CLOCK_RATE",
            ("hipDeviceAttributeClockRate", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT",
            (
                "hipDeviceAttributeTextureAlignment",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP",
            (
                "hipDeviceAttributeAsyncEngineCount",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
            (
                "hipDeviceAttributeMultiprocessorCount",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT",
            (
                "hipDeviceAttributeKernelExecTimeout",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_INTEGRATED",
            ("hipDeviceAttributeIntegrated", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY",
            (
                "hipDeviceAttributeCanMapHostMemory",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE",
            ("hipDeviceAttributeComputeMode", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH",
            (
                "hipDeviceAttributeMaxTexture3DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture3DHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH",
            (
                "hipDeviceAttributeMaxTexture3DDepth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLayeredHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTexture2DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLayeredHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES",
            (
                "hipDeviceAttributeMaxTexture2DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT",
            (
                "hipDeviceAttributeSurfaceAlignment",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS",
            ("hipDeviceAttributeConcurrentKernels", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_ECC_ENABLED",
            ("hipDeviceAttributeEccEnabled", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID",
            ("hipDeviceAttributePciBusId", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID",
            ("hipDeviceAttributePciDeviceId", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TCC_DRIVER",
            ("hipDeviceAttributeTccDriver", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE",
            (
                "hipDeviceAttributeMemoryClockRate",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH",
            ("hipDeviceAttributeMemoryBusWidth", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE",
            ("hipDeviceAttributeL2CacheSize", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR",
            ("hipDeviceAttributeMaxThreadsPerMultiProcessor", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT",
            (
                "hipDeviceAttributeAsyncEngineCount",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING",
            (
                "hipDeviceAttributeUnifiedAddressing",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTexture1DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER",
            (
                "hipDeviceAttributeCanTex2DGather",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DGatherWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DGatherHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DWidthAlternate",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DHeightAlternate",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE",
            (
                "hipDeviceAttributeMaxTexture3DDepthAlternate",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID",
            ("hipDeviceAttributePciDomainId", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT",
            (
                "hipDeviceAttributeTexturePitchAlignment",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH",
            (
                "hipDeviceAttributeMaxTextureCubemapWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxTextureCubemapLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface1DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface2DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface2DHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH",
            (
                "hipDeviceAttributeMaxSurface3DWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface3DHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH",
            (
                "hipDeviceAttributeMaxSurface3DDepth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurface1DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurface1DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurface2DLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT",
            (
                "hipDeviceAttributeMaxSurface2DLayeredHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurface2DLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH",
            (
                "hipDeviceAttributeMaxSurfaceCubemapWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS",
            (
                "hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DLinearWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DLinearWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DLinearHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH",
            (
                "hipDeviceAttributeMaxTexture2DLinearPitch",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT",
            (
                "hipDeviceAttributeMaxTexture2DMipmappedHeight",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
            ("hipDeviceAttributeComputeCapabilityMajor", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
            ("hipDeviceAttributeComputeCapabilityMinor", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH",
            (
                "hipDeviceAttributeMaxTexture1DMipmappedWidth",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED",
            (
                "hipDeviceAttributeStreamPrioritiesSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED",
            (
                "hipDeviceAttributeGlobalL1CacheSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED",
            (
                "hipDeviceAttributeLocalL1CacheSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
            (
                "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",
                CONV_TYPE,
                API_DRIVER,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR",
            (
                "hipDeviceAttributeMaxRegistersPerMultiprocessor",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY",
            ("hipDeviceAttributeManagedMemory", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD",
            ("hipDeviceAttributeIsMultiGpuBoard", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID",
            (
                "hipDeviceAttributeMultiGpuBoardGroupId",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED",
            (
                "hipDeviceAttributeHostNativeAtomicSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
            (
                "hipDeviceAttributeSingleToDoublePrecisionPerfRatio",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
            (
                "hipDeviceAttributePageableMemoryAccess",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS",
            (
                "hipDeviceAttributeConcurrentManagedAccess",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED",
            (
                "hipDeviceAttributeComputePreemptionSupported",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
            (
                "hipDeviceAttributeCanUseHostPointerForRegisteredMem",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_DEVICE_ATTRIBUTE_MAX",
            ("hipDeviceAttributeMax", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_CONTEXT",
            ("hipPointerAttributeContext", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_MEMORY_TYPE",
            ("hipPointerAttributeMemoryType", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_DEVICE_POINTER",
            (
                "hipPointerAttributeDevicePointer",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_POINTER_ATTRIBUTE_HOST_POINTER",
            ("hipPointerAttributeHostPointer", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_P2P_TOKENS",
            ("hipPointerAttributeP2pTokens", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS",
            ("hipPointerAttributeSyncMemops", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_BUFFER_ID",
            ("hipPointerAttributeBufferId", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_POINTER_ATTRIBUTE_IS_MANAGED",
            ("hipPointerAttributeIsManaged", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
            (
                "hipFuncAttributeMaxThreadsPerBlocks",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
            ("hipFuncAttributeSharedSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
            ("hipFuncAttributeMaxDynamicSharedMemorySize", CONV_TYPE, API_RUNTIME),
        ),
        (
            "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
            ("hipFuncAttributeConstSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
            ("hipFuncAttributeLocalSizeBytes", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_NUM_REGS",
            ("hipFuncAttributeNumRegs", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_PTX_VERSION",
            ("hipFuncAttributePtxVersion", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_BINARY_VERSION",
            ("hipFuncAttributeBinaryVersion", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA",
            ("hipFuncAttributeCacheModeCA", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_FUNC_ATTRIBUTE_MAX",
            ("hipFuncAttributeMax", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE",
            ("hipGraphicsMapFlagsNone", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY",
            ("hipGraphicsMapFlagsReadOnly", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD",
            ("hipGraphicsMapFlagsWriteDiscard", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_NONE",
            ("hipGraphicsRegisterFlagsNone", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY",
            (
                "hipGraphicsRegisterFlagsReadOnly",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD",
            (
                "hipGraphicsRegisterFlagsWriteDiscard",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST",
            (
                "hipGraphicsRegisterFlagsSurfaceLoadStore",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER",
            (
                "hipGraphicsRegisterFlagsTextureGather",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_OCCUPANCY_DEFAULT",
            ("hipOccupancyDefault", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE",
            (
                "hipOccupancyDisableCachingOverride",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_FUNC_CACHE_PREFER_NONE",
            ("hipFuncCachePreferNone", CONV_CACHE, API_DRIVER),
        ),
        (
            "CU_FUNC_CACHE_PREFER_SHARED",
            ("hipFuncCachePreferShared", CONV_CACHE, API_DRIVER),
        ),
        ("CU_FUNC_CACHE_PREFER_L1", ("hipFuncCachePreferL1", CONV_CACHE, API_DRIVER)),
        (
            "CU_FUNC_CACHE_PREFER_EQUAL",
            ("hipFuncCachePreferEqual", CONV_CACHE, API_DRIVER),
        ),
        (
            "CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS",
            ("hipIpcMemLazyEnablePeerAccess", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CUDA_IPC_HANDLE_SIZE", ("HIP_IPC_HANDLE_SIZE", CONV_TYPE, API_DRIVER)),
        (
            "CU_JIT_CACHE_OPTION_NONE",
            ("hipJitCacheModeOptionNone", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_CACHE_OPTION_CG",
            ("hipJitCacheModeOptionCG", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_CACHE_OPTION_CA",
            ("hipJitCacheModeOptionCA", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_PREFER_PTX",
            ("hipJitFallbackPreferPtx", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_PREFER_BINARY",
            ("hipJitFallbackPreferBinary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CU_JIT_MAX_REGISTERS", ("hipJitOptionMaxRegisters", CONV_JIT, API_DRIVER)),
        (
            "CU_JIT_THREADS_PER_BLOCK",
            ("hipJitOptionThreadsPerBlock", CONV_JIT, API_DRIVER),
        ),
        ("CU_JIT_WALL_TIME", ("hipJitOptionWallTime", CONV_JIT, API_DRIVER)),
        ("CU_JIT_INFO_LOG_BUFFER", ("hipJitOptionInfoLogBuffer", CONV_JIT, API_DRIVER)),
        (
            "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES",
            ("hipJitOptionInfoLogBufferSizeBytes", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_ERROR_LOG_BUFFER",
            ("hipJitOptionErrorLogBuffer", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES",
            ("hipJitOptionErrorLogBufferSizeBytes", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_OPTIMIZATION_LEVEL",
            ("hipJitOptionOptimizationLevel", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_TARGET_FROM_CUCONTEXT",
            ("hipJitOptionTargetFromContext", CONV_JIT, API_DRIVER),
        ),
        ("CU_JIT_TARGET", ("hipJitOptionTarget", CONV_JIT, API_DRIVER)),
        (
            "CU_JIT_FALLBACK_STRATEGY",
            ("hipJitOptionFallbackStrategy", CONV_JIT, API_DRIVER),
        ),
        (
            "CU_JIT_GENERATE_DEBUG_INFO",
            ("hipJitOptionGenerateDebugInfo", CONV_JIT, API_DRIVER),
        ),
        ("CU_JIT_LOG_VERBOSE", ("hipJitOptionLogVerbose", CONV_JIT, API_DRIVER)),
        (
            "CU_JIT_GENERATE_LINE_INFO",
            ("hipJitOptionGenerateLineInfo", CONV_JIT, API_DRIVER),
        ),
        ("CU_JIT_CACHE_MODE", ("hipJitOptionCacheMode", CONV_JIT, API_DRIVER)),
        ("CU_JIT_NEW_SM3X_OPT", ("hipJitOptionSm3xOpt", CONV_JIT, API_DRIVER)),
        ("CU_JIT_FAST_COMPILE", ("hipJitOptionFastCompile", CONV_JIT, API_DRIVER)),
        ("CU_JIT_NUM_OPTIONS", ("hipJitOptionNumOptions", CONV_JIT, API_DRIVER)),
        (
            "CU_TARGET_COMPUTE_10",
            ("hipJitTargetCompute10", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_11",
            ("hipJitTargetCompute11", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_12",
            ("hipJitTargetCompute12", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_13",
            ("hipJitTargetCompute13", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_20",
            ("hipJitTargetCompute20", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_21",
            ("hipJitTargetCompute21", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_30",
            ("hipJitTargetCompute30", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_32",
            ("hipJitTargetCompute32", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_35",
            ("hipJitTargetCompute35", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_37",
            ("hipJitTargetCompute37", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_50",
            ("hipJitTargetCompute50", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_52",
            ("hipJitTargetCompute52", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_53",
            ("hipJitTargetCompute53", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_60",
            ("hipJitTargetCompute60", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_61",
            ("hipJitTargetCompute61", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_TARGET_COMPUTE_62",
            ("hipJitTargetCompute62", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_CUBIN",
            ("hipJitInputTypeBin", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_PTX",
            ("hipJitInputTypePtx", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_FATBINARY",
            ("hipJitInputTypeFatBinary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_OBJECT",
            ("hipJitInputTypeObject", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_INPUT_LIBRARY",
            ("hipJitInputTypeLibrary", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_JIT_NUM_INPUT_TYPES",
            ("hipJitInputTypeNumInputTypes", CONV_JIT, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_STACK_SIZE",
            ("hipLimitStackSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_PRINTF_FIFO_SIZE",
            ("hipLimitPrintfFifoSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_MALLOC_HEAP_SIZE",
            ("hipLimitMallocHeapSize", CONV_TYPE, API_DRIVER),
        ),
        (
            "CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH",
            ("hipLimitDevRuntimeSyncDepth", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT",
            (
                "hipLimitDevRuntimePendingLaunchCount",
                CONV_TYPE,
                API_DRIVER,
                HIP_UNSUPPORTED,
            ),
        ),
        (
            "CU_LIMIT_STACK_SIZE",
            ("hipLimitStackSize", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_GLOBAL",
            ("hipMemAttachGlobal", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_HOST",
            ("hipMemAttachHost", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEM_ATTACH_SINGLE",
            ("hipMemAttachSingle", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_HOST",
            ("hipMemTypeHost", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_DEVICE",
            ("hipMemTypeDevice", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_ARRAY",
            ("hipMemTypeArray", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        (
            "CU_MEMORYTYPE_UNIFIED",
            ("hipMemTypeUnified", CONV_TYPE, API_DRIVER, HIP_UNSUPPORTED),
        ),
        ("CU_MEMHOSTREGISTER_READ_ONLY", ("hipHostRegisterReadOnly", CONV_TYPE, API_DRIVER)),
        ("CU_MEMPOOL_ATTR_RELEASE_THRESHOLD", ("hipMemPoolAttrReleaseThreshold", CONV_TYPE, API_DRIVER)),
        ("CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT", ("hipMemPoolAttrReservedMemCurrent", CONV_TYPE, API_DRIVER)),
        ("CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH", ("hipMemPoolAttrReservedMemHigh", CONV_TYPE, API_DRIVER)),
        (
            "CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES",
            ("hipMemPoolReuseAllowInternalDependencies", CONV_TYPE, API_DRIVER)
        ),
        ("CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC", ("hipMemPoolReuseAllowOpportunistic", CONV_TYPE, API_DRIVER)),
        (
            "CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES",
            ("hipMemPoolReuseFollowEventDependencies", CONV_TYPE, API_DRIVER)
        ),
        ("CU_MEMPOOL_ATTR_USED_MEM_CURRENT", ("hipMemPoolAttrUsedMemCurrent", CONV_TYPE, API_DRIVER)),
        ("CU_MEMPOOL_ATTR_USED_MEM_HIGH", ("hipMemPoolAttrUsedMemHigh", CONV_TYPE, API_DRIVER)),
        ("CU_MEM_ACCESS_FLAGS_PROT_NONE", ("hipMemAccessFlagsProtNone", CONV_TYPE, API_DRIVER)),
        ("CU_MEM_ACCESS_FLAGS_PROT_READ", ("hipMemAccessFlagsProtRead", CONV_TYPE, API_DRIVER)),
        ("CU_MEM_ACCESS_FLAGS_PROT_READWRITE", ("hipMemAccessFlagsProtReadWrite", CONV_TYPE, API_DRIVER)),
        ("CU_MEM_ALLOCATION_TYPE_INVALID", ("hipMemAllocationTypeInvalid", CONV_TYPE, API_DRIVER)),
        ("CU_MEM_ALLOCATION_TYPE_MAX", ("hipMemAllocationTypeMax", CONV_TYPE, API_DRIVER)),
        ("CU_MEM_ALLOCATION_TYPE_PINNED", ("hipMemAllocationTypePinned", CONV_TYPE, API_DRIVER)),
        ("CU_MEM_ALLOC_GRANULARITY_MINIMUM", ("hipMemAllocationGranularityMinimum", CONV_TYPE, API_DRIVER)),
        ("CU_MEM_ALLOC_GRANULARITY_RECOMMENDED", ("hipM

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough


## Key Components

The file contains 18487 words across 9488 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 398827 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
