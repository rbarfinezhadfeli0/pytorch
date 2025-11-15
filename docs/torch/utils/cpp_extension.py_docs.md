# Documentation: `torch/utils/cpp_extension.py`

## File Metadata

- **Path**: `torch/utils/cpp_extension.py`
- **Size**: 134,172 bytes (131.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import shutil
import setuptools
import subprocess
import sys
import sysconfig
import types
import collections
from pathlib import Path
import errno
import logging

logger = logging.getLogger(__name__)

import torch
import torch._appdirs
from .file_baton import FileBaton
from ._cpp_extension_versioner import ExtensionVersioner
from typing_extensions import deprecated
from torch.torch_version import TorchVersion, Version


from setuptools.command.build_ext import build_ext

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'

_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')


SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)

VersionRange = tuple[tuple[int, ...], tuple[int, ...]]
VersionMap = dict[str, VersionRange]
# The following values were taken from the following GitHub gist that
# summarizes the minimum valid major versions of g++/clang++ for each supported
# CUDA version: https://gist.github.com/ax3l/9489132
# Or from include/crt/host_config.h in the CUDA SDK
# The second value is the exclusive(!) upper bound, i.e. min <= version < max
CUDA_GCC_VERSIONS: VersionMap = {
    '11.0': (MINIMUM_GCC_VERSION, (10, 0)),
    '11.1': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.2': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.3': (MINIMUM_GCC_VERSION, (11, 0)),
    '11.4': ((6, 0, 0), (12, 0)),
    '11.5': ((6, 0, 0), (12, 0)),
    '11.6': ((6, 0, 0), (12, 0)),
    '11.7': ((6, 0, 0), (12, 0)),
}

MINIMUM_CLANG_VERSION = (3, 3, 0)
CUDA_CLANG_VERSIONS: VersionMap = {
    '11.1': (MINIMUM_CLANG_VERSION, (11, 0)),
    '11.2': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.3': (MINIMUM_CLANG_VERSION, (12, 0)),
    '11.4': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.5': (MINIMUM_CLANG_VERSION, (13, 0)),
    '11.6': (MINIMUM_CLANG_VERSION, (14, 0)),
    '11.7': (MINIMUM_CLANG_VERSION, (14, 0)),
}

__all__ = ["get_default_build_root", "check_compiler_ok_for_platform", "get_compiler_abi_compatibility_and_version", "BuildExtension",
           "CppExtension", "CUDAExtension", "SyclExtension", "include_paths", "library_paths", "load", "load_inline", "is_ninja_available",
           "verify_ninja_availability", "remove_extension_h_precompiler_headers", "get_cxx_compiler", "check_compiler_is_gcc"]
# Taken directly from python stdlib < 3.9
# See https://github.com/pytorch/pytorch/issues/48617
def _nt_quote_args(args: list[str] | None) -> list[str]:
    """Quote command-line arguments for DOS/Windows conventions.

    Just wraps every argument which contains blanks in double quotes, and
    returns a new argument list.
    """
    # Cover None-type
    if not args:
        return []
    return [f'"{arg}"' if ' ' in arg else arg for arg in args]

def _find_cuda_home() -> str | None:
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        logger.warning("No CUDA runtime is found, using CUDA_HOME='%s'", cuda_home)
    return cuda_home

def _find_rocm_home() -> str | None:
    """Find the ROCm install path."""
    # Guess #1
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        # Guess #2
        hipcc_path = shutil.which('hipcc')
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(
                os.path.realpath(hipcc_path)))
            # can be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        else:
            # Guess #3
            fallback_path = '/opt/rocm'
            if os.path.exists(fallback_path):
                rocm_home = fallback_path
    if rocm_home and torch.version.hip is None:
        logger.warning("No ROCm runtime is found, using ROCM_HOME='%s'", rocm_home)
    return rocm_home

def _find_sycl_home() -> str | None:
    sycl_home = None
    icpx_path = shutil.which('icpx')
    # Guess 1: for source code build developer/user, we'll have icpx in PATH,
    # which will tell us the SYCL_HOME location.
    if icpx_path is not None:
        sycl_home = os.path.dirname(os.path.dirname(
            os.path.realpath(icpx_path)))

    # Guess 2: for users install Pytorch with XPU support, the sycl runtime is
    # inside intel-sycl-rt, which is automatically installed via pip dependency.
    else:
        try:
            files = importlib.metadata.files('intel-sycl-rt') or []
            for f in files:
                if f.name == "libsycl.so":
                    sycl_home = os.path.dirname(Path(f.locate()).parent.resolve())
                    break
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Trying to find SYCL_HOME from intel-sycl-rt package, but it is not installed.")
    return sycl_home

def _join_rocm_home(*paths) -> str:
    """
    Join paths with ROCM_HOME, or raises an error if it ROCM_HOME is not set.

    This is basically a lazy way of raising an error for missing $ROCM_HOME
    only once we need to get any ROCm-specific path.
    """
    if ROCM_HOME is None:
        raise OSError('ROCM_HOME environment variable is not set. '
                      'Please set it to your ROCm install root.')
    return os.path.join(ROCM_HOME, *paths)

def _join_sycl_home(*paths) -> str:
    """
    Join paths with SYCL_HOME, or raises an error if it SYCL_HOME is not found.

    This is basically a lazy way of raising an error for missing SYCL_HOME
    only once we need to get any SYCL-specific path.
    """
    if SYCL_HOME is None:
        raise OSError('SYCL runtime is not dected. Please setup the pytorch '
                      'prerequisites for Intel GPU following the instruction in '
                      'https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support '
                      'or install intel-sycl-rt via pip.')

    return os.path.join(SYCL_HOME, *paths)



ABI_INCOMPATIBILITY_WARNING = (
    "                               !! WARNING !!"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "Your compiler (%s) may be ABI-incompatible with PyTorch!"
    "Please use a compiler that is ABI-compatible with GCC 5.0 and above."
    "See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html."
    "See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6"
    "for instructions on how to install GCC 5 or higher."
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "                              !! WARNING !!"
)
WRONG_COMPILER_WARNING = (
    "                               !! WARNING !!"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "Your compiler (%s) is not compatible with the compiler Pytorch was"
    "built with for this platform, which is %s on %s. Please"
    "use %s to compile your extension. Alternatively, you may"
    "compile PyTorch from source using %s, and then you can also use"
    "%s to compile your extension."
    "See https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help"
    "with compiling PyTorch from source."
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    "                              !! WARNING !!"
)
CUDA_MISMATCH_MESSAGE = (
    "The detected CUDA version (%s) mismatches the version that was used to compile"
    "PyTorch (%s). Please make sure to use the same CUDA versions."
)
CUDA_MISMATCH_WARN = (
    "The detected CUDA version (%s) has a minor version mismatch with the version that was used to compile PyTorch (%s). Most likely this shouldn't be a problem."
)
CUDA_NOT_FOUND_MESSAGE = (
    "CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH"
    "environment variable or add NVCC to your system PATH. The extension compilation will fail."
)
ROCM_HOME = _find_rocm_home() if (torch.cuda._is_compiled() and torch.version.hip) else None
HIP_HOME = _join_rocm_home('hip') if ROCM_HOME else None
IS_HIP_EXTENSION = bool(ROCM_HOME is not None and torch.version.hip is not None)
ROCM_VERSION = None
if torch.version.hip is not None:
    ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split('.')[:2])

CUDA_HOME = _find_cuda_home() if (torch.cuda._is_compiled() and torch.version.cuda) else None
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
SYCL_HOME = _find_sycl_home() if torch.xpu._is_compiled() else None
WINDOWS_CUDA_HOME = os.environ.get('WINDOWS_CUDA_HOME')  # used for AOTI cross-compilation

# PyTorch releases have the version pattern major.minor.patch, whereas when
# PyTorch is built from source, we append the git commit hash, which gives
# it the below pattern.
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+\w+\+\w+')

COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/wd4624', '/wd4067', '/wd4068', '/EHsc']

MSVC_IGNORE_CUDAFE_WARNINGS = [
    'base_class_has_different_dll_interface',
    'field_without_dll_interface',
    'dll_interface_conflict_none_assumed',
    'dll_interface_conflict_dllexport_assumed'
]

COMMON_NVCC_FLAGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

COMMON_HIP_FLAGS = [
    '-D__HIP_PLATFORM_AMD__=1',
    '-DUSE_ROCM=1',
    '-DHIPBLAS_V2',
]

if not IS_WINDOWS:
    COMMON_HIP_FLAGS.append('-fPIC')

COMMON_HIPCC_FLAGS = [
    '-DCUDA_HAS_FP16=1',
    '-D__HIP_NO_HALF_OPERATORS__=1',
    '-D__HIP_NO_HALF_CONVERSIONS__=1',
    '-DHIP_ENABLE_WARP_SYNC_BUILTINS=1'
]

if IS_WINDOWS:
    # Compatibility flags, similar to those set in cmake/Dependencies.cmake.
    COMMON_HIPCC_FLAGS.append('-fms-extensions')
    # Suppress warnings about dllexport.
    COMMON_HIPCC_FLAGS.append('-Wno-ignored-attributes')


def _get_icpx_version() -> str:
    icpx = 'icx' if IS_WINDOWS else 'icpx'
    compiler_info = subprocess.check_output([icpx, '--version'])
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode().strip())
    version = ['0', '0', '0'] if match is None else list(match.groups())
    version = list(map(int, version))
    if len(version) != 3:
        raise AssertionError("Failed to parse DPC++ compiler version")
    # Aligning version format with what torch.version.xpu() returns
    return f"{version[0]}{version[1]:02}{version[2]:02}"


def _get_sycl_arch_list():
    if 'TORCH_XPU_ARCH_LIST' in os.environ:
        return os.environ.get('TORCH_XPU_ARCH_LIST')
    arch_list = torch.xpu.get_arch_list()
    # Dropping dg2* archs since they lack hardware support for fp64 and require
    # special consideration from the user. If needed these platforms can
    # be requested thru TORCH_XPU_ARCH_LIST environment variable.
    arch_list = [x for x in arch_list if not x.startswith('dg2')]
    return ','.join(arch_list)


# If arch list returned by _get_sycl_arch_list() is empty, then sycl kernels will be compiled
# for default spir64 target and avoid device specific compilations entirely. Further, kernels
# will be JIT compiled at runtime.
def _append_sycl_targets_if_missing(cflags) -> None:
    if any(flag.startswith('-fsycl-targets=') for flag in cflags):
        # do nothing: user has manually specified sycl targets
        return
    if _get_sycl_arch_list() != '':
        # AOT (spir64_gen) + JIT (spir64)
        cflags.append('-fsycl-targets=spir64_gen,spir64')
    else:
        # JIT (spir64)
        cflags.append('-fsycl-targets=spir64')

def _get_sycl_device_flags(cflags):
    # We need last occurrence of -fsycl-targets as it will be the one taking effect.
    # So searching in reversed list.
    flags = [f for f in reversed(cflags) if f.startswith('-fsycl-targets=')]
    if not flags:
        raise AssertionError("bug: -fsycl-targets should have been amended to cflags")

    arch_list = _get_sycl_arch_list()
    if arch_list != '':
        flags += [f'-Xs "-device {arch_list}"']
    return flags

_COMMON_SYCL_FLAGS = [
    '-fsycl',
]

_SYCL_DLINK_FLAGS = [
    *_COMMON_SYCL_FLAGS,
    '-fsycl-link',
    '--offload-compress',
]

JIT_EXTENSION_VERSIONER = ExtensionVersioner()

PLAT_TO_VCVARS = {
    'win32' : 'x86',
    'win-amd64' : 'x86_amd64',
}

min_supported_cpython = "0x030A0000"  # Python 3.10 hexcode

def get_cxx_compiler():
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    return compiler

def _is_binary_build() -> bool:
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)


def _accepted_compilers_for_platform() -> list[str]:
    # gnu-c++ and gnu-cc are the conda gcc compilers
    return ['clang++', 'clang'] if IS_MACOS else ['g++', 'gcc', 'gnu-c++', 'gnu-cc', 'clang++', 'clang']

def _maybe_write(filename, new_content) -> None:
    r'''
    Equivalent to writing the content into the file but will not touch the file
    if it already had the right content (to avoid triggering recompile).
    '''
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.read()

        if content == new_content:
            # The file already contains the right thing!
            return

    with open(filename, 'w') as source_file:
        source_file.write(new_content)

def get_default_build_root() -> str:
    """
    Return the path to the root folder under which extensions will built.

    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.

    This directory is **user-specific** so that multiple users on the same
    machine won't meet permission issues.
    """
    return os.path.realpath(torch._appdirs.user_cache_dir(appname='torch_extensions'))


def check_compiler_ok_for_platform(compiler: str) -> bool:
    """
    Verify that the compiler is the expected one for the current platform.

    Args:
        compiler (str): The compiler executable to check.

    Returns:
        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,
        and always True for Windows.
    """
    if IS_WINDOWS:
        return True
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        return False
    # Use os.path.realpath to resolve any symlinks, in particular from 'c++' to e.g. 'g++'.
    compiler_path = os.path.realpath(compiler_path)
    # Check the compiler name
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True
    # If compiler wrapper is used try to infer the actual compiler by invoking it with -v flag
    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # Don't localize output
    try:
        version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    except subprocess.CalledProcessError:
        # If '-v' fails, try '--version'
        version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        # Check for 'gcc' or 'g++' for sccache wrapper
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            # Clang is also a supported compiler on Linux
            # Though on Ubuntu it's sometimes called "Ubuntu clang version"
            return 'clang version' in version_string
        compiler_path = os.path.realpath(results[0].strip())
        # On RHEL/CentOS c++ is a gcc compiler wrapper
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
            return True
        return any(name in compiler_path for name in _accepted_compilers_for_platform())
    if IS_MACOS:
        # Check for 'clang' or 'clang++'
        return version_string.startswith("Apple clang")
    return False


def get_compiler_abi_compatibility_and_version(compiler) -> tuple[bool, TorchVersion]:
    """
    Determine if the given compiler is ABI-compatible with PyTorch alongside its version.

    Args:
        compiler (str): The compiler executable name to check (e.g. ``g++``).
            Must be executable in a shell process.

    Returns:
        A tuple that contains a boolean that defines if the compiler is (likely) ABI-incompatible with PyTorch,
        followed by a `TorchVersion` string that contains the compiler version separated by dots.
    """
    if not _is_binary_build():
        return (True, TorchVersion('0.0.0'))
    if os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') in ['ON', '1', 'YES', 'TRUE', 'Y']:
        return (True, TorchVersion('0.0.0'))

    # First check if the compiler is one of the expected ones for the particular platform.
    if not check_compiler_ok_for_platform(compiler):
        logger.warning(WRONG_COMPILER_WARNING, compiler, _accepted_compilers_for_platform()[0], sys.platform, _accepted_compilers_for_platform()[0])
        return (False, TorchVersion('0.0.0'))

    if IS_MACOS:
        # There is no particular minimum version we need for clang, so we're good here.
        return (True, TorchVersion('0.0.0'))
    try:
        if IS_LINUX:
            minimum_required_version = MINIMUM_GCC_VERSION
            compiler_info = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
        version = ['0', '0', '0'] if match is None else list(match.groups())
    except Exception:
        _, error, _ = sys.exc_info()
        logger.warning('Error checking compiler version for %s: %s', compiler, error)
        return (False, TorchVersion('0.0.0'))

    # convert alphanumeric string to numeric string
    # amdclang++ returns str like 0.0.0git, others return 0.0.0
    numeric_version = [re.sub(r'\D', '', v) for v in version]

    if tuple(map(int, numeric_version)) >= minimum_required_version:
        return (True, TorchVersion('.'.join(numeric_version)))

    compiler = f'{compiler} {".".join(numeric_version)}'
    logger.warning(ABI_INCOMPATIBILITY_WARNING, compiler)

    return (False, TorchVersion('.'.join(numeric_version)))


def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc.exe' if IS_WINDOWS else 'nvcc')
    if not os.path.exists(nvcc):
        raise FileNotFoundError(f"nvcc not found at '{nvcc}'. Ensure CUDA path '{CUDA_HOME}' is correct.")

    cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
    cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)
    if cuda_version is None:
        return

    cuda_str_version = cuda_version.group(1)
    cuda_ver = Version(cuda_str_version)
    if torch.version.cuda is None:
        return

    torch_cuda_version = Version(torch.version.cuda)
    if cuda_ver != torch_cuda_version:
        # major/minor attributes are only available in setuptools>=49.4.0
        if getattr(cuda_ver, "major", None) is None:
            raise ValueError("setuptools>=49.4.0 is required")
        if cuda_ver.major != torch_cuda_version.major:
            raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
        logger.warning(CUDA_MISMATCH_WARN, cuda_str_version, torch.version.cuda)

    if not (sys.platform.startswith('linux') and
            os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') not in ['ON', '1', 'YES', 'TRUE', 'Y'] and
            _is_binary_build()):
        return

    cuda_compiler_bounds: VersionMap = CUDA_CLANG_VERSIONS if compiler_name.startswith('clang') else CUDA_GCC_VERSIONS

    if cuda_str_version not in cuda_compiler_bounds:
        logger.warning('There are no %s version bounds defined for CUDA version %s', compiler_name, cuda_str_version)
    else:
        min_compiler_version, max_excl_compiler_version = cuda_compiler_bounds[cuda_str_version]
        # Special case for 11.4.0, which has lower compiler bounds than 11.4.1
        if "V11.4.48" in cuda_version_str and cuda_compiler_bounds == CUDA_GCC_VERSIONS:
            max_excl_compiler_version = (11, 0)
        min_compiler_version_str = '.'.join(map(str, min_compiler_version))
        max_excl_compiler_version_str = '.'.join(map(str, max_excl_compiler_version))

        version_bound_str = f'>={min_compiler_version_str}, <{max_excl_compiler_version_str}'

        if compiler_version < TorchVersion(min_compiler_version_str):
            raise RuntimeError(
                f'The current installed version of {compiler_name} ({compiler_version}) is less '
                f'than the minimum required version by CUDA {cuda_str_version} ({min_compiler_version_str}). '
                f'Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).'
            )
        if compiler_version >= TorchVersion(max_excl_compiler_version_str):
            raise RuntimeError(
                f'The current installed version of {compiler_name} ({compiler_version}) is greater '
                f'than the maximum required version by CUDA {cuda_str_version}. '
                f'Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).'
            )


# Specify Visual Studio C runtime library for hipcc
def _set_hipcc_runtime_lib(is_standalone, debug) -> None:
    if is_standalone:
        if debug:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=static_dbg')
        else:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=static')
    else:
        if debug:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=dll_dbg')
        else:
            COMMON_HIP_FLAGS.append('-fms-runtime-lib=dll')

def _append_sycl_std_if_no_std_present(cflags) -> None:
    if not any(flag.startswith('-sycl-std=') for flag in cflags):
        cflags.append('-sycl-std=2020')


def _wrap_sycl_host_flags(cflags):
    host_cxx = get_cxx_compiler()
    host_cflags = [
        f'-fsycl-host-compiler={host_cxx}',
        shlex.quote(f'-fsycl-host-compiler-options={cflags}'),
    ]
    return host_cflags


class BuildExtension(build_ext):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
    C++/CUDA/SYCL compilation (and support for CUDA/SYCL files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages/compilers (the only expected values are ``cxx``, ``nvcc`` or
    ``sycl``) to a list of additional compiler flags to supply to the compiler.
    This makes it possible to supply different flags to the C++, CUDA and SYCL
    compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """

    @classmethod
    def with_options(cls, **options):
        """Return a subclass with alternative constructor that extends any original keyword arguments to the original constructor with the given options."""
        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs) -> None:
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = ('Attempted to use ninja as the BuildExtension backend but '
                   '%s. Falling back to using the slow distutils backend.')
            if not is_ninja_available():
                logger.warning(msg, 'we could not find ninja.')
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        compiler_name, compiler_version = self._check_abi()

        cuda_ext = False
        sycl_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not (cuda_ext and sycl_ext) and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                elif ext == '.sycl':
                    sycl_ext = True

                # This check accounts on a case when cuda and sycl sources
                # are mixed in the same extension. We can stop checking
                # sources if both are found or there is no more sources.
                if cuda_ext and sycl_ext:
                    break

            extension = next(extension_iter, None)

        if sycl_ext:
            if not self.use_ninja:
                raise AssertionError("ninja is required to build sycl extensions.")

        if cuda_ext and not IS_HIP_EXTENSION:
            _check_cuda_version(compiler_name, compiler_version)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx', 'nvcc' and 'sycl' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx', 'nvcc' or 'sycl' is
            # passed to extra_compile_args in CUDAExtension or SyclExtension, i.e.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc', 'sycl']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')

            if IS_HIP_EXTENSION:
                self._hipify_compile_flags(extension)

            if extension.py_limited_api:
                # compile any extension that has passed in py_limited_api to the
                # Extension constructor with the Py_LIMITED_API flag set to our
                # min supported CPython version.
                # See https://docs.python.org/3/c-api/stable.html#c.Py_LIMITED_API
                self._add_compile_flag(extension, f'-DPy_LIMITED_API={min_supported_cpython}')
            self._define_torch_extension_name(extension)

            if 'nvcc_dlink' in extension.extra_compile_args:
                if not self.use_ninja:
                    raise AssertionError(
                        f"With dlink=True, ninja is required to build cuda extension {extension.name}."
                    )

        # Register .cu, .cuh, .hip, .mm and .sycl as valid source extensions.
        # NOTE: At the moment .sycl is not a standard extension for SYCL supported
        # by compiler. Here we introduce a torch level convention that SYCL sources
        # should have .sycl file extension.
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip', '.sycl']
        if torch.backends.mps.is_built():
            self.compiler.src_extensions += ['.mm']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++17'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            cflags = (COMMON_NVCC_FLAGS +
                      ['--compiler-options', "'-fPIC'"] +
                      cflags + _get_cuda_arch_flags(cflags))

            # NVCC does not allow multiple -ccbin/--compiler-bindir to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv("CC")
            if (
                _ccbin is not None
                and not any(flag.startswith(('-ccbin', '--compiler-bindir')) for flag in cflags)
            ):
                cflags.extend(['-ccbin', _ccbin])

            return cflags

        def convert_to_absolute_paths_inplace(paths) -> None:
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = [_join_rocm_home('bin', 'hipcc') if IS_HIP_EXTENSION else _join_cuda_home('bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    if IS_HIP_EXTENSION:
                        cflags = COMMON_HIPCC_FLAGS + cflags + _get_rocm_arch_flags(cflags)
                    else:
                        cflags = unix_cuda_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                if IS_HIP_EXTENSION:
                    cflags = COMMON_HIP_FLAGS + cflags
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def unix_wrap_ninja_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567  # codespell:ignore
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            # pyrefly: ignore [no-matching-overload]
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(_is_cuda_file, sources))
            with_sycl = any(map(_is_sycl_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc/sycl to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = cuda_post_cflags + _get_rocm_arch_flags(cuda_post_cflags)
                    cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                else:
                    cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std17_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]

            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = unix_cuda_flags(extra_postargs['nvcc_dlink'])
                cuda_dlink_post_cflags = [shlex.quote(f) for f in cuda_dlink_post_cflags]
            else:
                cuda_dlink_post_cflags = None

            sycl_post_cflags = None
            sycl_cflags = None
            sycl_dlink_post_cflags = None
            if with_sycl:
                sycl_cflags = extra_cc_cflags + common_cflags + _COMMON_SYCL_FLAGS
                if isinstance(extra_postargs, dict):
                    sycl_post_cflags = extra_postargs['sycl']
                else:
                    sycl_post_cflags = list(extra_postargs)
                _append_sycl_targets_if_missing(sycl_post_cflags)
                append_std17_if_no_std_present(sycl_cflags)
                _append_sycl_std_if_no_std_present(sycl_cflags)
                host_cflags = extra_cc_cflags + common_cflags + post_cflags
                append_std17_if_no_std_present(host_cflags)
                # escaping quoted arguments to pass them thru SYCL compiler
                icpx_version = _get_icpx_version()
                if int(icpx_version) >= 20250200:
                    host_cflags = [item.replace('"', '\\"') for item in host_cflags]
                else:
                    host_cflags = [item.replace('"', '\\\\"') for item in host_cflags]
                host_cflags = ' '.join(host_cflags)
                # Note the order: shlex.quote sycl_flags first, _wrap_sycl_host_flags
                # second. Reason is that sycl host flags are quoted, space containing
                # strings passed to SYCL compiler.
                sycl_cflags = [shlex.quote(f) for f in sycl_cflags]
                sycl_cflags += _wrap_sycl_host_flags(host_cflags)
                sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
                sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_post_cflags)
                sycl_post_cflags = [shlex.quote(f) for f in sycl_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                sycl_cflags=sycl_cflags,
                sycl_post_cflags=sycl_post_cflags,
                sycl_dlink_post_cflags=sycl_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
                with_sycl=with_sycl)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        def win_cuda_flags(cflags):
            return (COMMON_NVCC_FLAGS +
                    cflags + _get_cuda_arch_flags(cflags))

        def win_hip_flags(cflags):
            return (COMMON_HIPCC_FLAGS + COMMON_HIP_FLAGS + cflags + _get_rocm_arch_flags(cflags))

        def win_wrap_single_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')  # codespell:ignore
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        if IS_HIP_EXTENSION:
                            nvcc = _get_hipcc_path()
                        else:
                            nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []

                        if IS_HIP_EXTENSION:
                            cflags = win_hip_flags(cflags)
                        else:
                            cflags = win_cuda_flags(cflags) + ['-std=c++17', '--use-local-env']
                            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                                cflags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cflags
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        cmd = [nvcc, '-c', src, '-o', obj] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                        append_std17_if_no_std_present(cflags)
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = COMMON_MSVC_FLAGS + self.cflags
                        append_std17_if_no_std_present(cflags)
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        def win_wrap_ninja_compile(sources,
                                   output_dir=None,
                                   macros=None,
                                   include_dirs=None,
                                   debug=0,
                                   extra_preargs=None,
                                   extra_postargs=None,
                                   depends=None,
                                   is_standalone=False):
            if not self.compiler.initialized:
                self.compiler.initialize()
            # pyrefly: ignore [no-matching-overload]
            output_dir = os.path.abspath(output_dir)

            # Note [Absolute include_dirs]
            # Convert relative path in self.compiler.include_dirs to absolute path if any.
            # For ninja build, the build location is not local, but instead, the build happens
            # in a script-created build folder. Thus, relative paths lose their correctness.
            # To be consistent with jit extension, we allow user to enter relative include_dirs
            # in setuptools.setup, and we convert the relative path to absolute path here.
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            # Replace space with \ when using hipcc (hipcc passes includes to clang without ""s so clang sees space in include paths as new argument)
            if IS_HIP_EXTENSION:
                pp_opts = ["-I{}".format(s[2:].replace(" ", "\\")) if s.startswith('-I') else s for s in pp_opts]
            common_cflags = extra_preargs or []
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            cflags = cflags + common_cflags + pp_opts + COMMON_MSVC_FLAGS
            if IS_HIP_EXTENSION:
                _set_hipcc_runtime_lib(is_standalone, debug)
                common_cflags.extend(COMMON_HIP_FLAGS)
            else:
                common_cflags.extend(COMMON_MSVC_FLAGS)
            with_cuda = any(map(_is_cuda_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = ['-std=c++17']
                for common_cflag in common_cflags:
                    cuda_cflags.append('-Xcompiler')
                    cuda_cflags.append(common_cflag)
                if not IS_HIP_EXTENSION:
                    cuda_cflags.append('--use-local-env')
                    for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                        cuda_cflags.append('-Xcudafe')
                        cuda_cflags.append('--diag_suppress=' + ignore_warning)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = win_hip_flags(cuda_post_cflags)
                else:
                    cuda_post_cflags = win_cuda_flags(cuda_post_cflags)
            cflags = _nt_quote_args(cflags)
            post_cflags = _nt_quote_args(post_cflags)
            if with_cuda:
                cuda_cflags = _nt_quote_args(cuda_cflags)
                cuda_post_cflags = _nt_quote_args(cuda_post_cflags)
            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = win_cuda_flags(extra_postargs['nvcc_dlink'])
            else:
                cuda_dlink_post_cflags = None

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=cflags,
                post_cflags=post_cflags,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                sycl_cflags=None,
                sycl_post_cflags=None,
                sycl_dlink_post_cflags=None,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
                with_sycl=False)

            # Return *all* object filenames, not just the ones we just built.
            return objects
        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511  # codespell:ignore
        if self.compiler.compiler_type == 'msvc':
            if self.use_ninja:
                self.compiler.compile = win_wrap_ninja_compile
            else:
                self.compiler.compile = win_wrap_single_compile
        else:
            if self.use_ninja:
                self.compiler.compile = unix_wrap_ninja_compile
            else:
                self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split('.')
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename

    def _check_abi(self) -> tuple[str, TorchVersion]:
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()
        _, version = get_compiler_abi_compatibility_and_version(compiler)
        # Warn user if VC env is activated but `DISTUILS_USE_SDK` is not set.
        if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
            msg = ('It seems that the VC environment is activated but DISTUTILS_U
```



## High-Level Overview


This Python file contains 4 class(es) and 83 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BuildExtension`, `cls_with_options`

**Functions defined**: `_nt_quote_args`, `_find_cuda_home`, `_find_rocm_home`, `_find_sycl_home`, `_join_rocm_home`, `_join_sycl_home`, `_get_icpx_version`, `_get_sycl_arch_list`, `_append_sycl_targets_if_missing`, `_get_sycl_device_flags`, `get_cxx_compiler`, `_is_binary_build`, `_accepted_compilers_for_platform`, `_maybe_write`, `get_default_build_root`, `check_compiler_ok_for_platform`, `get_compiler_abi_compatibility_and_version`, `_check_cuda_version`, `_set_hipcc_runtime_lib`, `_append_sycl_std_if_no_std_present`

**Key imports**: copy, glob, importlib, importlib.abc, os, re, shlex, shutil, setuptools, subprocess


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `glob`
- `importlib`
- `importlib.abc`
- `os`
- `re`
- `shlex`
- `shutil`
- `setuptools`
- `subprocess`
- `sys`
- `sysconfig`
- `types`
- `collections`
- `pathlib`: Path
- `errno`
- `logging`
- `torch`
- `torch._appdirs`
- `.file_baton`: FileBaton
- `._cpp_extension_versioner`: ExtensionVersioner
- `typing_extensions`: deprecated
- `torch.torch_version`: TorchVersion, Version
- `setuptools.command.build_ext`: build_ext
- `torch.utils.cpp_extension`: BuildExtension, CppExtension
- `.hipify`: hipify_python


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `cpp_extension.py_docs.md`
- **Keyword Index**: `cpp_extension.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
