# Documentation: `torch/_inductor/cpp_builder.py`

## File Metadata

- **Path**: `torch/_inductor/cpp_builder.py`
- **Size**: 85,757 bytes (83.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# This CPP builder is designed to support both Windows and Linux OS.
# The design document please check this RFC: https://github.com/pytorch/pytorch/issues/124245

import copy
import ctypes
import errno
import functools
import json
import locale
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import warnings
from collections.abc import Sequence
from ctypes import cdll, wintypes
from ctypes.util import find_library
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config, exc
from torch._inductor.cpu_vec_isa import invalid_vec_isa, VecISA
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.torch_version import TorchVersion


if config.is_fbcode():
    from triton.fb.build import _run_build_command, build_paths

    from torch._inductor.fb.utils import (
        log_global_cache_errors,
        log_global_cache_stats,
        log_global_cache_vals,
        use_global_cache,
    )
else:

    def log_global_cache_errors(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def log_global_cache_stats(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def log_global_cache_vals(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def use_global_cache() -> bool:  # type: ignore[misc]
        return False


# Windows need setup a temp dir to store .obj files.
_BUILD_TEMP_DIR = "CxxBuild"
_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
_LINKER_SCRIPT = os.path.join(_TORCH_PATH, "_inductor/script.ld")

# initialize variables for compilation
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"

MINGW_GXX = "x86_64-w64-mingw32-g++"

SUBPROCESS_DECODE_ARGS = (locale.getpreferredencoding(),) if _IS_WINDOWS else ()

log = logging.getLogger(__name__)


# =============================== toolchain ===============================
@functools.lru_cache(1)
def cpp_compiler_search(search: str) -> str:
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT

    for cxx in search:
        try:
            if cxx is None:
                # gxx package is only available for Linux
                # according to https://anaconda.org/conda-forge/gxx/
                if sys.platform != "linux":
                    continue
                # Do not install GXX by default
                if not os.getenv("TORCH_INDUCTOR_INSTALL_GXX"):
                    continue
                from torch.utils._filelock import FileLock

                lock_dir = get_lock_dir()
                lock = FileLock(
                    os.path.join(lock_dir, "g++.lock"), timeout=LOCK_TIMEOUT
                )
                with lock:
                    cxx = install_gcc_via_conda()
            subprocess.check_output([cxx, "--version"])
            return cxx
        except (subprocess.SubprocessError, FileNotFoundError, ImportError):
            continue
    raise exc.InvalidCxxCompiler


def install_gcc_via_conda() -> str:
    """On older systems, this is a quick way to get a modern compiler"""
    prefix = os.path.join(cache_dir(), "gcc")
    cxx_path = os.path.join(prefix, "bin", "g++")
    if not os.path.exists(cxx_path):
        log.info("Downloading GCC via conda")
        conda = os.environ.get("CONDA_EXE", "conda")
        if conda is None:
            conda = shutil.which("conda")
        if conda is not None:
            subprocess.check_call(
                [
                    conda,
                    "create",
                    f"--prefix={prefix}",
                    "--channel=conda-forge",
                    "--quiet",
                    "-y",
                    "python=3.8",
                    "gxx",
                ],
                stdout=subprocess.PIPE,
            )
    return cxx_path


@functools.cache
def check_compiler_exist_windows(compiler: str) -> None:
    """
    Check if compiler is ready, in case end user not activate MSVC environment.
    """
    try:
        subprocess.check_output([compiler, "/help"], stderr=subprocess.STDOUT)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Compiler: {compiler} is not found.") from exc
    except subprocess.SubprocessError:
        # Expected that some compiler(clang, clang++) is exist, but they not support `/help` args.
        pass


class WinPeFileVersionInfo:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.version_dll = ctypes.WinDLL("version.dll")  # type: ignore[attr-defined]
        self._setup_functions()
        self._get_version_info()

    def _setup_functions(self) -> None:
        self.version_dll.GetFileVersionInfoSizeW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPDWORD,
        ]
        self.version_dll.GetFileVersionInfoSizeW.restype = wintypes.DWORD

        self.version_dll.GetFileVersionInfoW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        self.version_dll.GetFileVersionInfoW.restype = wintypes.BOOL

        self.version_dll.VerQueryValueW.argtypes = [
            wintypes.LPCVOID,
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.UINT),
        ]
        self.version_dll.VerQueryValueW.restype = wintypes.BOOL

    def _get_version_info(self) -> None:
        dummy = wintypes.DWORD()
        size = self.version_dll.GetFileVersionInfoSizeW(
            self.file_path, ctypes.byref(dummy)
        )

        if size == 0:
            raise RuntimeError(f"Can't get version info size of {self.file_path}.")

        self.version_info = ctypes.create_string_buffer(size)
        success = self.version_dll.GetFileVersionInfoW(
            self.file_path, 0, size, self.version_info
        )

        if not success:
            raise RuntimeError(f"Can't get version info of {self.file_path}.")

    def get_language_id(self) -> int:
        lp_buffer = ctypes.c_void_p()
        u_len = wintypes.UINT()

        success = self.version_dll.VerQueryValueW(
            self.version_info,
            r"\VarFileInfo\Translation",
            ctypes.byref(lp_buffer),
            ctypes.byref(u_len),
        )

        if not success or u_len.value == 0:
            return 0

        translations = []
        lang_id: int = 0
        if lp_buffer.value is not None:
            for i in range(u_len.value // 4):
                offset = i * 4
                data = ctypes.string_at(lp_buffer.value + offset, 4)
                lang_id = int.from_bytes(data[:2], "little")
                code_page = int.from_bytes(data[2:4], "little")
                translations.append((lang_id, code_page))
        else:
            # Handle the case where lp_buffer.value is None
            print("Buffer is None")

        return lang_id


@functools.cache
def check_msvc_cl_language_id(compiler: str) -> None:
    """
    Torch.compile() is only work on MSVC with English language pack well.
    Check MSVC's language pack: https://github.com/pytorch/pytorch/issues/157673#issuecomment-3051682766
    """

    def get_msvc_cl_path() -> tuple[bool, str]:
        """
        Finds the path to cl.exe using vswhere.exe.
        """
        vswhere_path = os.path.join(
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
            "Microsoft Visual Studio",
            "Installer",
            "vswhere.exe",
        )
        if not os.path.exists(vswhere_path):
            vswhere_path = os.path.join(
                os.environ.get("ProgramFiles", "C:\\Program Files"),
                "Microsoft Visual Studio",
                "Installer",
                "vswhere.exe",
            )
            if not os.path.exists(vswhere_path):
                return False, ""  # vswhere.exe not found

        try:
            # Get the Visual Studio installation path
            cmd = [
                vswhere_path,
                "-latest",
                "-prerelease",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ]
            vs_install_path = subprocess.check_output(
                cmd, text=True, encoding="utf-8"
            ).strip()

            if not vs_install_path:
                return False, ""

            # Find the latest MSVC toolset version within the installation
            msvc_tools_path = os.path.join(vs_install_path, "VC", "Tools", "MSVC")
            if not os.path.exists(msvc_tools_path):
                return False, ""

            # Get the latest toolset version directory
            toolset_versions = [
                d
                for d in os.listdir(msvc_tools_path)
                if os.path.isdir(os.path.join(msvc_tools_path, d))
            ]
            if not toolset_versions:
                return False, ""
            latest_toolset_version = sorted(toolset_versions, reverse=True)[0]

            # Construct the full cl.exe path
            cl_path = os.path.join(
                msvc_tools_path,
                latest_toolset_version,
                "bin",
                "HostX64",
                "x64",
                "cl.exe",
            )
            if os.path.exists(cl_path):
                return True, cl_path
            else:
                # Fallback for older versions or different architectures if needed
                cl_path = os.path.join(
                    msvc_tools_path,
                    latest_toolset_version,
                    "bin",
                    "HostX86",
                    "x86",
                    "cl.exe",
                )
                if os.path.exists(cl_path):
                    return True, cl_path

        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, ""

        return False, ""

    if not _is_msvc_cl(compiler):
        return

    if os.path.exists(compiler):
        # Passed compiler with path.
        cl_exe_path = compiler
    else:
        b_ret, cl_exe_path = get_msvc_cl_path()
        if b_ret is False:
            return

    version_info = WinPeFileVersionInfo(cl_exe_path)
    lang_id = version_info.get_language_id()
    if lang_id != 1033:
        # MSVC English language id is 0x0409, and the DEC value is 1033.
        raise RuntimeError(
            "Torch.compile() is only support MSVC with English language pack,"
            "Please reinstall its language pack to English."
        )


@functools.cache
def check_mingw_win32_flavor(compiler: str) -> str:
    """
    Check if MinGW `compiler` exists and return it's flavor (win32 or posix).
    """
    try:
        out = subprocess.check_output(
            [compiler, "-v"], stderr=subprocess.STDOUT, text=True
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Compiler: {compiler} is not found.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to run {compiler} -v") from e

    flavor: str | None = None
    for line in out.splitlines():
        if "Thread model" in line:
            flavor = line.split(":", 1)[-1].strip().lower()

    if flavor is None:
        raise RuntimeError(
            f"Cannot determine the flavor of {compiler} (win32 or posix). No Thread model found in {compiler} -v"
        )

    if flavor not in ("win32", "posix"):
        raise RuntimeError(
            f"Only win32 and pofix flavor of {compiler} is supported. The flavor is {flavor}"
        )

    return flavor


def get_cpp_compiler() -> str:
    if (
        config.aot_inductor.cross_target_platform == "windows"
        and sys.platform != "win32"
    ):
        # we're doing cross-compilation
        compiler = MINGW_GXX
        if not config.aot_inductor.package_cpp_only:
            check_mingw_win32_flavor(compiler)
        return compiler

    if _IS_WINDOWS:
        compiler = os.environ.get("CXX", "cl")
        compiler = normalize_path_separator(compiler)
        check_compiler_exist_windows(compiler)
        check_msvc_cl_language_id(compiler)
    else:
        if config.is_fbcode():
            return build_paths.cc
        if isinstance(config.cpp.cxx, (list, tuple)):
            search = tuple(config.cpp.cxx)
        else:
            search = (config.cpp.cxx,)
        compiler = cpp_compiler_search(search)
    return compiler


def get_ld_and_objcopy(use_relative_path: bool) -> tuple[str, str]:
    if _IS_WINDOWS:
        raise RuntimeError("Windows is not supported yet.")
    else:
        if config.is_fbcode():
            ld = build_paths.ld
            objcopy = (
                build_paths.objcopy_fallback
                if use_relative_path
                else build_paths.objcopy
            )
        else:
            ld = "ld"
            objcopy = "objcopy"
    return ld, objcopy


def convert_cubin_to_obj(
    cubin_file: str,
    kernel_name: str,
    ld: str,
    objcopy: str,
) -> str:
    obj_file = cubin_file + ".o"
    # Convert .cubin to .o
    cmd = f"{ld} -r -b binary -z noexecstack -o {obj_file} {cubin_file}"
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # Rename .data to .rodata
    cmd = f"{objcopy} --rename-section .data=.rodata,alloc,load,readonly,data,contents {obj_file}"
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # By default objcopy will create *_start, *_size, *_end symbols using the full path
    # Rename to use the unique kernel name
    file_name = re.sub(r"[\W]", "_", cubin_file)
    cmd = (
        objcopy
        + f" --redefine-sym _binary_{file_name}_start=__{kernel_name}_start "
        + f"--redefine-sym _binary_{file_name}_size=__{kernel_name}_size "
        + f"--redefine-sym _binary_{file_name}_end=__{kernel_name}_end "
        + obj_file
    )
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    return obj_file


@functools.cache
def _is_apple_clang(cpp_compiler: str) -> bool:
    version_string = subprocess.check_output([cpp_compiler, "--version"]).decode("utf8")
    return "Apple" in version_string.splitlines()[0]


@functools.cache
def _is_clang(cpp_compiler: str) -> bool:
    # Mac OS apple clang maybe named as gcc, need check compiler info.
    if sys.platform == "darwin":
        return _is_apple_clang(cpp_compiler)
    elif _IS_WINDOWS:
        # clang suite have many compilers, and only clang-cl is supported.
        if re.search(r"((clang$)|(clang\+\+$))", cpp_compiler):
            raise RuntimeError(
                "Please use clang-cl, due to torch.compile only support MSVC-like CLI (compiler flags syntax)."
            )
        return bool(re.search(r"(clang-cl)", cpp_compiler))
    return bool(re.search(r"(clang|clang\+\+)", cpp_compiler))


@functools.cache
def _is_gcc(cpp_compiler: str) -> bool:
    # Since "clang++" ends with "g++", the regex match below would validate on it.
    if _is_clang(cpp_compiler):
        return False
    return bool(re.search(r"(gcc|g\+\+|gnu-c\+\+)", cpp_compiler))


@functools.cache
def _is_msvc_cl(cpp_compiler: str) -> bool:
    if not _IS_WINDOWS:
        return False

    try:
        output_msg = (
            subprocess.check_output([cpp_compiler, "/help"], stderr=subprocess.STDOUT)
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
        return "Microsoft" in output_msg.splitlines()[0]
    except FileNotFoundError:
        return False

    return False


@functools.cache
def _is_intel_compiler(cpp_compiler: str) -> bool:
    def _check_minimal_version(compiler_version: TorchVersion) -> None:
        """
        On Windows: early version icx has `-print-file-name` issue, and can't preload correctly for inductor.
        """
        min_version = "2024.2.1" if _IS_WINDOWS else "0.0.0"
        if compiler_version < TorchVersion(min_version):
            raise RuntimeError(
                f"Intel Compiler error: less than minimal version {min_version}."
            )

    try:
        output_msg = (
            subprocess.check_output(
                [cpp_compiler, "--version"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
        is_intel_compiler = "Intel" in output_msg.splitlines()[0]
        if is_intel_compiler:
            if _IS_WINDOWS:
                if re.search(r"((icx$)|(icx-cc$))", cpp_compiler):
                    raise RuntimeError(
                        "Please use icx-cl, due to torch.compile only support MSVC-like CLI (compiler flags syntax)."
                    )

            # Version check
            icx_ver_search = re.search(r"(\d+[.]\d+[.]\d+[.]\d+)", output_msg)
            if icx_ver_search is not None:
                icx_ver = icx_ver_search.group(1)
                _check_minimal_version(TorchVersion(icx_ver))

        return is_intel_compiler
    except FileNotFoundError:
        return False
    except subprocess.SubprocessError:
        # --version args not support.
        return False

    return False


@functools.cache
def is_gcc() -> bool:
    return _is_gcc(get_cpp_compiler())


@functools.cache
def is_clang() -> bool:
    return _is_clang(get_cpp_compiler())


@functools.cache
def is_intel_compiler() -> bool:
    return _is_intel_compiler(get_cpp_compiler())


@functools.cache
def is_apple_clang() -> bool:
    return _is_apple_clang(get_cpp_compiler())


@functools.cache
def is_msvc_cl() -> bool:
    return _is_msvc_cl(get_cpp_compiler())


@functools.cache
def get_compiler_version_info(compiler: str) -> str:
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # Don't localize output
    try:
        version_string = subprocess.check_output(
            [compiler, "-v"], stderr=subprocess.STDOUT, env=env
        ).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception:
        try:
            version_string = subprocess.check_output(
                [compiler, "--version"], stderr=subprocess.STDOUT, env=env
            ).decode(*SUBPROCESS_DECODE_ARGS)
        except Exception:
            return ""
    # Multiple lines to one line string.
    version_string = version_string.replace("\r", "_")
    version_string = version_string.replace("\n", "_")
    return version_string


# =============================== cpp builder ===============================
def _append_list(dest_list: list[str], src_list: list[str]) -> None:
    dest_list.extend(copy.deepcopy(item) for item in src_list)


def _remove_duplication_in_list(orig_list: list[str]) -> list[str]:
    new_list: list[str] = []
    for item in orig_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def _create_if_dir_not_exist(path_dir: str) -> None:
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError(f"Fail to create path {path_dir}") from exc


def _remove_dir(path_dir: str) -> None:
    if os.path.exists(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
        os.rmdir(path_dir)


def _run_compile_cmd(cmd_line: str, cwd: str) -> None:
    cmd = shlex.split(cmd_line)
    try:
        subprocess.run(
            cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        output = e.stdout.decode(*SUBPROCESS_DECODE_ARGS)
        openmp_problem = "'omp.h' file not found" in output or "libomp" in output
        if openmp_problem and sys.platform == "darwin":
            instruction = (
                "\n\nOpenMP support not found. Please try one of the following solutions:\n"
                "(1) Set the `CXX` environment variable to a compiler other than Apple clang++/g++ "
                "that has builtin OpenMP support;\n"
                "(2) install OpenMP via conda: `conda install llvm-openmp`;\n"
                "(3) install libomp via brew: `brew install libomp`;\n"
                "(4) manually setup OpenMP and set the `OMP_PREFIX` environment variable to point to a path"
                " with `include/omp.h` under it."
            )
            output += instruction
        raise exc.CppCompileError(cmd, output) from e


def run_compile_cmd(cmd_line: str, cwd: str) -> None:
    with dynamo_timed("compile_file"):
        _run_compile_cmd(cmd_line, cwd)


def normalize_path_separator(orig_path: str) -> str:
    if _IS_WINDOWS:
        return orig_path.replace(os.sep, "/")
    return orig_path


class BuildOptionsBase:
    """
    This is the Base class for store cxx build options, as a template.
    Actually, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    """

    def __init__(
        self,
        compiler: str = "",
        definitions: Optional[list[str]] = None,
        include_dirs: Optional[list[str]] = None,
        cflags: Optional[list[str]] = None,
        ldflags: Optional[list[str]] = None,
        libraries_dirs: Optional[list[str]] = None,
        libraries: Optional[list[str]] = None,
        passthrough_args: Optional[list[str]] = None,
        aot_mode: bool = False,
        use_relative_path: bool = False,
        compile_only: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        self._compiler = compiler
        self._definitions: list[str] = definitions or []
        self._include_dirs: list[str] = include_dirs or []
        self._cflags: list[str] = cflags or []
        self._ldflags: list[str] = ldflags or []
        self._libraries_dirs: list[str] = libraries_dirs or []
        self._libraries: list[str] = libraries or []
        # Some args are hard to abstract to OS compatible, passthrough directly.
        self._passthrough_args: list[str] = passthrough_args or []

        # Optionally, the path to a precompiled header which should be included on the
        # build command line.
        self.precompiled_header: Optional[str] = None

        self._aot_mode: bool = aot_mode
        self._use_relative_path: bool = use_relative_path
        self._compile_only: bool = compile_only
        self._precompiling: bool = precompiling
        self._preprocessing: bool = preprocessing

    def _process_compile_only_options(self) -> None:
        if self._compile_only:
            self._libraries_dirs = []
            self._libraries = []

    def _remove_duplicate_options(self) -> None:
        self._definitions = _remove_duplication_in_list(self._definitions)
        self._include_dirs = _remove_duplication_in_list(self._include_dirs)
        self._cflags = _remove_duplication_in_list(self._cflags)
        self._ldflags = _remove_duplication_in_list(self._ldflags)
        self._libraries_dirs = _remove_duplication_in_list(self._libraries_dirs)
        self._libraries = _remove_duplication_in_list(self._libraries)
        self._passthrough_args = _remove_duplication_in_list(self._passthrough_args)

    def _finalize_options(self) -> None:
        self._process_compile_only_options()
        self._remove_duplicate_options()

    def get_compiler(self) -> str:
        return self._compiler

    def get_definitions(self) -> list[str]:
        return self._definitions

    def get_include_dirs(self) -> list[str]:
        return self._include_dirs

    def get_cflags(self) -> list[str]:
        return self._cflags

    def get_ldflags(self) -> list[str]:
        return self._ldflags

    def get_libraries_dirs(self) -> list[str]:
        return self._libraries_dirs

    def get_libraries(self) -> list[str]:
        return self._libraries

    def get_passthrough_args(self) -> list[str]:
        return self._passthrough_args

    def get_aot_mode(self) -> bool:
        return self._aot_mode

    def get_use_relative_path(self) -> bool:
        return self._use_relative_path

    def get_compile_only(self) -> bool:
        return self._compile_only

    def get_precompiling(self) -> bool:
        return self._precompiling

    def get_preprocessing(self) -> bool:
        return self._preprocessing

    def save_flags_to_json(self, file: str) -> None:
        attrs = {
            "compiler": self.get_compiler(),
            "definitions": self.get_definitions(),
            "include_dirs": self.get_include_dirs(),
            "cflags": self.get_cflags(),
            "ldflags": self.get_ldflags(),
            "libraries_dirs": self.get_libraries_dirs(),
            "libraries": self.get_libraries(),
            "passthrough_args": self.get_passthrough_args(),
            "aot_mode": self.get_aot_mode(),
            "use_relative_path": self.get_use_relative_path(),
            "compile_only": self.get_compile_only(),
        }

        with open(file, "w") as f:
            json.dump(attrs, f)


def _get_warning_all_cflag(warning_all: bool = True) -> list[str]:
    if not _IS_WINDOWS:
        return ["Wall"] if warning_all else []
    else:
        return []


def _get_cpp_std_cflag(std_num: str = "c++17") -> list[str]:
    if _IS_WINDOWS:
        """
        On Windows, only c++20 can support `std::enable_if_t`.
        Ref: https://learn.microsoft.com/en-us/cpp/overview/cpp-conformance-improvements-2019?view=msvc-170#checking-for-abstract-class-types # noqa: B950
        Note:
            Only setup c++20 for Windows inductor. I tried to upgrade all project to c++20, but it is failed:
            https://github.com/pytorch/pytorch/pull/131504
        """
        std_num = "c++20"
        return [f"std:{std_num}"]
    else:
        return [f"std={std_num}"]


def _get_os_related_cpp_cflags(cpp_compiler: str) -> list[str]:
    if _IS_WINDOWS:
        cflags = [
            "wd4819",
            "wd4251",
            "wd4244",
            "wd4267",
            "wd4275",
            "wd4018",
            "wd4190",
            "wd4624",
            "wd4067",
            "wd4068",
            "EHsc",
            # For Intel oneAPI, ref: https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus?view=msvc-170
            "Zc:__cplusplus",
            # Enable max compatible to msvc for oneAPI headers.
            # ref: https://github.com/pytorch/pytorch/blob/db38c44ad639e7ada3e9df2ba026a2cb5e40feb0/cmake/public/utils.cmake#L352-L358 # noqa: B950
            "permissive-",
        ]
    else:
        cflags = ["Wno-unused-variable", "Wno-unknown-pragmas"]
        if _is_clang(cpp_compiler):
            ignored_optimization_argument = (
                "Werror=ignored-optimization-argument"
                if config.aot_inductor.raise_error_on_ignored_optimization
                else "Wno-ignored-optimization-argument"
            )
            cflags.append(ignored_optimization_argument)
        if _is_gcc(cpp_compiler):
            # Issue all the warnings demanded by strict ISO C and ISO C++.
            # Ref: https://github.com/pytorch/pytorch/issues/153180#issuecomment-2986676878
            cflags.append("pedantic")
    return cflags


def _get_os_related_cpp_definitions(cpp_compiler: str) -> list[str]:
    os_definitions: list[str] = []
    if _IS_WINDOWS:
        # On Windows, we need disable min/max macro to avoid C2589 error, as PyTorch CMake:
        # https://github.com/pytorch/pytorch/blob/9a41570199155eee92ebd28452a556075e34e1b4/CMakeLists.txt#L1118-L1119
        os_definitions.append("NOMINMAX")
    return os_definitions


def _get_ffast_math_flags() -> list[str]:
    if _IS_WINDOWS:
        flags = []
    else:
        # ffast-math is equivalent to these flags as in
        # https://github.com/gcc-mirror/gcc/blob/4700ad1c78ccd7767f846802fca148b2ea9a1852/gcc/opts.cc#L3458-L3468
        # however gcc<13 sets the FTZ/DAZ flags for runtime on x86 even if we have
        # -ffast-math -fno-unsafe-math-optimizations because the flags for runtime
        # are added by linking in crtfastmath.o. This is done by the spec file which
        # only does globbing for -ffast-math.
        flags = [
            "fno-trapping-math",
            "funsafe-math-optimizations",
            "ffinite-math-only",
            "fno-signed-zeros",
            "fno-math-errno",
        ]

        flags.append("fno-finite-math-only")
        if not config.cpp.enable_unsafe_math_opt_flag:
            flags.append("fno-unsafe-math-optimizations")
        flags.append(f"ffp-contract={config.cpp.enable_floating_point_contract_flag}")

        if is_gcc():
            flags.append("fexcess-precision=fast")

    return flags


def _get_inductor_debug_symbol_cflags() -> tuple[list[str], list[str]]:
    """
    When we turn on generate debug symbol.
    On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
    On Linux, it should create some debug sections in binary file.
    """
    cflags: list[str] = []
    ldflags: list[str] = []

    if _IS_WINDOWS:
        cflags = ["ZI", "_DEBUG"]
        ldflags = ["DEBUG", "ASSEMBLYDEBUG ", "OPT:REF", "OPT:ICF"]
    else:
        cflags.append("g")

    return cflags, ldflags


def _get_optimization_cflags(
    cpp_compiler: str, min_optimize: bool = False
) -> tuple[list[str], list[str]]:
    cflags: list[str] = []
    ldflags: list[str] = []

    should_use_optimized_flags = not (
        config.aot_inductor.debug_compile
        or os.environ.get("TORCHINDUCTOR_DEBUG_COMPILE", "0") == "1"
    )
    should_add_debug_symbol_flags = (
        config.aot_inductor.debug_compile
        or config.aot_inductor.debug_symbols
        or os.environ.get("TORCHINDUCTOR_DEBUG_COMPILE", "0") == "1"
        or os.environ.get("TORCHINDUCTOR_DEBUG_SYMBOL", "0") == "1"
    )
    if should_use_optimized_flags:
        if _IS_WINDOWS:
            cflags += ["O1" if min_optimize else "O2"]
        else:
            cflags += [
                config.aot_inductor.compile_wrapper_opt_level if min_optimize else "O3",
                "DNDEBUG",
            ]
    else:
        if _IS_WINDOWS:
            cflags += ["Od", "Ob0", "Oy-"]
        else:
            cflags += ["O0"]

    if should_add_debug_symbol_flags:
        debug_cflags, debug_ldflags = _get_inductor_debug_symbol_cflags()
        cflags += debug_cflags
        ldflags += debug_ldflags

    cflags += _get_ffast_math_flags()

    if _IS_WINDOWS:
        pass
    else:
        if sys.platform != "darwin":
            # on macos, unknown argument: '-fno-tree-loop-vectorize'
            if _is_gcc(cpp_compiler):
                cflags.append("fno-tree-loop-vectorize")
            # https://stackoverflow.com/questions/65966969/why-does-march-native-not-work-on-apple-m1
            # `-march=native` is unrecognized option on M1
            if not config.is_fbcode():
                if platform.machine() == "ppc64le":
                    cflags.append("mcpu=native")
                elif platform.machine() == "riscv64":
                    cflags.append("march=rv64gc")
                elif platform.machine() == "riscv32":
                    cflags.append("march=rv32gc")
                else:
                    cflags.append("march=native")

        if config.aot_inductor.enable_lto and _is_clang(cpp_compiler):
            cflags.append("flto=thin")

    return cflags, ldflags


def _get_shared_cflags(do_link: bool) -> list[str]:
    if _IS_WINDOWS:
        """
        MSVC `/MD` using python `ucrtbase.dll` lib as runtime.
        https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=msvc-170
        """
        return ["DLL", "MD"]
    if platform.system() == "Darwin" and "clang" in get_cpp_compiler():
        # This causes undefined symbols to behave the same as linux
        return ["shared", "fPIC", "undefined dynamic_lookup"]
    flags = []
    if do_link:
        flags.append("shared")

    flags.append("fPIC")
    return flags


def get_cpp_options(
    cpp_compiler: str,
    do_link: bool,
    warning_all: bool = True,
    extra_flags: Sequence[str] = (),
    min_optimize: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []

    opt_cflags, opt_ldflags = _get_optimization_cflags(cpp_compiler, min_optimize)

    cflags = (
        opt_cflags
        + _get_shared_cflags(do_link)
        + _get_warning_all_cflag(warning_all)
        + _get_cpp_std_cflag()
        + _get_os_related_cpp_cflags(cpp_compiler)
    )

    definitions += _get_os_related_cpp_definitions(cpp_compiler)

    if not _IS_WINDOWS and config.aot_inductor.enable_lto and _is_clang(cpp_compiler):
        ldflags.append("fuse-ld=lld")
        ldflags.append("flto=thin")

    passthrough_args.append(" ".join(extra_flags))

    if config.aot_inductor.cross_target_platform == "windows":
        passthrough_args.extend(["-static-libstdc++", "-static-libgcc"])
        if check_mingw_win32_flavor(MINGW_GXX) == "posix":
            passthrough_args.append("-Wl,-Bstatic -lwinpthread -Wl,-Bdynamic")

    return (
        definitions,
        include_dirs,
        cflags,
        ldflags + opt_ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


class CppOptions(BuildOptionsBase):
    """
    This class is inherited from BuildOptionsBase, and as cxx build options.
    This option need contains basic cxx build option, which contains:
    1. OS related args.
    2. Toolchains related args.
    3. Cxx standard related args.
    Note:
    1. This Options is good for assist modules build, such as x86_isa_help.
    """

    def __init__(
        self,
        compile_only: bool = False,
        warning_all: bool = True,
        extra_flags: Sequence[str] = (),
        use_relative_path: bool = False,
        compiler: str = "",
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            use_relative_path=use_relative_path,
            precompiling=precompiling,
            preprocessing=preprocessing,
        )
        self._compiler = compiler if compiler else get_cpp_compiler()

        (
            definitions,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthrough_args,
        ) = get_cpp_options(
            cpp_compiler=self._compiler,
            do_link=not (compile_only or precompiling or preprocessing),
            extra_flags=extra_flags,
            warning_all=warning_all,
            min_optimize=min_optimize,
        )

        _append_list(self._definitions, definitions)
        _append_list(self._include_dirs, include_dirs)
        _append_list(self._cflags, cflags)
        _append_list(self._ldflags, ldflags)
        _append_list(self._libraries_dirs, libraries_dirs)
        _append_list(self._libraries, libraries)
        _append_list(self._passthrough_args, passthrough_args)
        self._finalize_options()


def _get_torch_cpp_wrapper_definition() -> list[str]:
    return ["TORCH_INDUCTOR_CPP_WRAPPER", "STANDALONE_TORCH_HEADER"]


def _use_custom_generated_macros() -> list[str]:
    return [" C10_USING_CUSTOM_GENERATED_MACROS"]


def _use_fb_internal_macros() -> list[str]:
    if not _IS_WINDOWS:
        if config.is_fbcode():
            fb_internal_macros = [
                "C10_USE_GLOG",
                "C10_USE_MINIMAL_GLOG",
                "C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
            ]
            return fb_internal_macros
        else:
            return []
    else:
        return []


def _setup_standard_sys_libs(
    cpp_compiler: str,
    aot_mode: bool,
    use_relative_path: bool,
) -> tuple[list[str], list[str], list[str]]:
    cflags: list[str] = []
    include_dirs: list[str] = []
    passthrough_args: list[str] = []
    if _IS_WINDOWS:
        return cflags, include_dirs, passthrough_args

    if config.is_fbcode():
        # TODO(T203137008) Can we unify these flags with triton_cc_command?
        cflags.append("nostdinc")
        # Note that the order of include paths do matter, as a result
        # we need to have several branches interleaved here
        include_dirs.append(build_paths.sleef_include)
        include_dirs.append(build_paths.openmp_include)
        include_dirs.append(build_paths.python_include)
        include_dirs.append(build_paths.cc_include)
        include_dirs.append(build_paths.libgcc_include)
        include_dirs.append(build_paths.libgcc_arch_include)
        include_dirs.append(build_paths.libgcc_backward_include)
        include_dirs.append(build_paths.glibc_include)
        include_dirs.append(build_paths.linux_kernel_include)
        include_dirs.append("include")

        if aot_mode and not use_relative_path:
            linker_script = _LINKER_SCRIPT
        else:
            linker_script = os.path.basename(_LINKER_SCRIPT)

        if _is_clang(cpp_compiler):
            passthrough_args.append(" --rtlib=compiler-rt")
            passthrough_args.append(" -fuse-ld=lld")
            passthrough_args.append(f" -Wl,--script={linker_script}")
            passthrough_args.append(" -B" + build_paths.glibc_lib)
            passthrough_args.append(" -L" + build_paths.glibc_lib)

    return cflags, include_dirs, passthrough_args


def _get_build_args_of_chosen_isa(vec_isa: VecISA) -> tuple[list[str], list[str]]:
    macros: list[str] = []
    build_flags: list[str] = []
    if vec_isa != invalid_vec_isa:
        # Add Windows support later.
        macros.extend(copy.deepcopy(x) for x in vec_isa.build_macro())

        build_flags = [vec_isa.build_arch_flags()]

        if config.is_fbcode():
            cap = str(vec_isa).upper()
            macros = [
                f"CPU_CAPABILITY={cap}",
                f"CPU_CAPABILITY_{cap}",
                f"HAVE_{cap}_CPU_DEFINITION",
            ]

    return macros, build_flags


def _get_torch_related_args(
    include_pytorch: bool, aot_mode: bool
) -> tuple[list[str], list[str], list[str]]:
    from torch.utils.cpp_extension import include_paths, TORCH_LIB_PATH

    libraries = []
    include_dirs = include_paths()

    if config.aot_inductor.link_libtorch:
        libraries_dirs = [TORCH_LIB_PATH]
        if sys.platform != "darwin" and not config.is_fbcode():
            libraries.extend(["torch", "torch_cpu"])
            if not aot_mode:
                libraries.append("torch_python")
    else:
        libraries_dirs = []
        if config.aot_inductor.cross_target_platform == "windows":
            aoti_shim_library = config.aot_inductor.aoti_shim_library

            assert aoti_shim_library, (
                "'config.aot_inductor.aoti_shim_library' must be set when 'cross_target_platform' is 'windows'."
            )
            if isinstance(aoti_shim_library, str):
                libraries.append(aoti_shim_library)
            else:
                assert isinstance(aoti_shim_library, list)
                libraries.extend(aoti_shim_library)

    if config.aot_inductor.cross_target_platform == "windows":
        assert config.aot_inductor.aoti_shim_library_path, (
            "'config.aot_inductor.aoti_shim_library_path' must be set to the path of the AOTI shim library",
            " when 'cross_target_platform' is 'windows'.",
        )
        libraries_dirs.append(config.aot_inductor.aoti_shim_library_path)

    if _IS_WINDOWS:
        libraries.append("sleef")

    return include_dirs, libraries_dirs, libraries


def _get_python_include_dirs() -> list[str]:
    include_dir = Path(sysconfig.get_path("include"))
    # On Darwin Python executable from a framework can return
    # non-existing /Library/Python/... include path, in which case
    # one should use Headers folder from the framework
    if not include_dir.exists() and platform.system() == "Darwin":
        std_lib = Path(sysconfig.get_path("stdlib"))
        include_dir = (std_lib.parent.parent / "Headers").absolute()
    if not (include_dir / "Python.h").exists():
        warnings.warn(f"Can't find Python.h in {str(include_dir)}")
    return [str(include_dir)]


def _get_python_related_args() -> tuple[list[str], list[str]]:
    python_include_dirs = _get_python_include_dirs()
    python_include_path = sysconfig.get_path(
        "include", scheme="nt" if _IS_WINDOWS else "posix_prefix"
    )
    if python_include_path is not None:
        python_include_dirs.append(python_include_path)

    if _IS_WINDOWS:
        python_lib_path = [
            str(
                (
                    Path(sysconfig.get_path("include", scheme="nt")).parent / "libs"
                ).absolute()
            )
        ]
    else:
        python_lib_path = [sysconfig.get_config_var("LIBDIR")]

    if config.is_fbcode():
        python_include_dirs.append(build_paths.python_include)

    return python_include_dirs, python_lib_path


@functools.cache
def is_conda_llvm_openmp_installed() -> bool:
    try:
        command = "conda list llvm-openmp --json"
        output = subprocess.check_output(command.split()).decode("utf8")
        return len(json.loads(output)) > 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@functools.cache
def homebrew_libomp() -> tuple[bool, str]:
    try:
        # check if `brew` is installed
        if shutil.which("brew") is None:
            return False, ""
        # get the location of `libomp` if it is installed
        # this is the location that `libomp` **would** be installed
        # see https://github.com/Homebrew/brew/issues/10261#issuecomment-756563567 for details
        libomp_path = (
            subprocess.check_output(["brew", "--prefix", "libomp"])
            .decode("utf8")
            .strip()
        )
        # check if `libomp` is installed
        omp_available = os.path.exists(libomp_path)
        return omp_available, libomp_path
    except subprocess.SubprocessError:
        return False, ""


@functools.cache
def perload_clang_libomp_win(cpp_compiler: str, omp_name: str) -> None:
    try:
        output = subprocess.check_output([cpp_compiler, "-print-file-name=bin"]).decode(
            "utf8"
        )
        omp_path = os.path.join(output.rstrip(), omp_name)
        if os.path.isfile(omp_path):
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            cdll.LoadLibrary(omp_path)
    except subprocess.SubprocessError:
        pass


@functools.cache
def perload_icx_libomp_win(cpp_compiler: str) -> None:
    def _load_icx_built_in_lib_by_name(cpp_compiler: str, lib_name: str) -> bool:
        try:
            output = subprocess.check_output(
                [cpp_compiler, f"-print-file-name={lib_name}"],
                stderr=subprocess.DEVNULL,
            ).decode(*SUBPROCESS_DECODE_ARGS)
            omp_path = output.rstrip()
            if os.path.isfile(omp_path):
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                cdll.LoadLibrary(omp_path)
                return True
        except subprocess.SubprocessError:
            pass
        return False

    """
    Intel Compiler implemented more math libraries than clang, for performance proposal.
    We need preload them like openmp library.
    """
    preload_list = [
        "libiomp5md.dll",  # openmp
        "svml_dispmd.dll",  # svml library
        "libmmd.dll",  # libm
    ]

    for lib_name in preload_list:
        _load_icx_built_in_lib_by_name(cpp_compiler, lib_name)


def _get_openmp_args(
    cpp_compiler: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    cflags: list[str] = []
    ldflags: list[str] = []
    include_dir_paths: list[str] = []
    lib_dir_paths: list[str] = []
    libs: list[str] = []
    passthrough_args: list[str] = []

    if config.aot_inductor.cross_target_platform == "windows":
        return cflags, ldflags, include_dir_paths, lib_dir_paths, libs, passthrough_args
    if _IS_MACOS:
        # Per https://mac.r-project.org/openmp/ right way to pass `openmp` flags to MacOS is via `-Xclang`
        cflags.append("Xclang")
        cflags.append("fopenmp")

        # only Apple builtin compilers (Apple Clang++) require openmp
        omp_available = not _is_apple_clang(cpp_compiler)

        # check the `OMP_PREFIX` environment first
        omp_prefix = os.getenv("OMP_PREFIX")
        if omp_prefix is not None:
            header_path = os.path.join(omp_prefix, "include", "omp.h")
            valid_env = os.path.exists(header_path)
            if valid_env:
                include_dir_paths.append(os.path.join(omp_prefix, "include"))
                lib_dir_paths.append(os.path.join(omp_prefix, "lib"))
            else:
                warnings.warn("environment variable `OMP_PREFIX` is invalid.")
            omp_available = omp_available or valid_env

        if not omp_available:
            libs.append("omp")

        # prefer to use openmp from `conda install llvm-openmp`
        conda_prefix = os.getenv("CONDA_PREFIX")
        if not omp_available and conda_prefix is not None:
            omp_available = is_conda_llvm_openmp_installed()
            if omp_available:
                conda_lib_path = os.path.join(conda_prefix, "lib")
                include_dir_paths.append(os.path.join(conda_prefix, "include"))
                lib_dir_paths.append(conda_lib_path)
                # Prefer Intel OpenMP on x86 machine
                if os.uname().machine == "x86_64" and os.path.exists(
                    os.path.join(conda_lib_path, "libiomp5.dylib")
                ):
                    libs.append("iomp5")

        # next, try to use openmp from `brew install libomp`
        if not omp_available:
            omp_available, libomp_path = homebrew_libomp()
            if omp_available:
                include_dir_paths.append(os.path.join(libomp_path, "include"))
                lib_dir_paths.append(os.path.join(libomp_path, "lib"))

        # if openmp is still not available, we let the compiler to have a try,
        # and raise error together with instructions at compilation error later
    elif _IS_WINDOWS:
        """
        On Windows, `clang` and `icx` have their specific openmp implenmention.
        And the openmp lib is in compiler's some sub-directory.
        For dynamic library(DLL) load, the Windows native APIs are `LoadLibraryA` and `LoadLibraryExA`, and their search
        dependencies have some rules:
        https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa#searching-for-dlls-and-dependencies
        In some case, the rules may not include compiler's sub-directories.
        So, it can't search and load compiler's openmp library correctly.
        And then, the whole application would be broken.

        To avoid the openmp load failed, we can automatic locate the openmp binary and preload it.
        1. For clang, the function is `perload_clang_libomp_win`.
        2. For icx, the function is `perload_icx_libomp_win`.
        """
        if _is_clang(cpp_compiler):
            cflags.append("openmp")
            libs.append("libomp")
            perload_clang_libomp_win(cpp_compiler, "libomp.dll")
        elif _is_intel_compiler(cpp_compiler):
            cflags.append("Qiopenmp")
            libs.append("libiomp5md")
            perload_icx_libomp_win(cpp_compiler)
        else:
            # /openmp, /openmp:llvm
            # llvm on Windows, new openmp: https://devblogs.microsoft.com/cppblog/msvc-openmp-update/
            # msvc openmp: https://learn.microsoft.com/zh-cn/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170
            cflags.append("openmp")
            cflags.append("openmp:experimental")  # MSVC CL
    else:
        if config.is_fbcode():
            include_dir_paths.append(build_paths.openmp_include)

            openmp_lib = build_paths.openmp_lib_so
            fb_openmp_extra_flags = f"-Wp,-fopenmp {openmp_lib}"
            passthrough_args.append(fb_openmp_extra_flags)

            libs.append("omp")
        else:
            if _is_clang(cpp_compiler):
                # TODO: fix issue, can't find omp.h
                cflags.append("fopenmp")
                libs.append("gomp")
            elif _is_intel_compiler(cpp_compiler):
                cflags.append("fiopenmp")
            else:
                cflags.append("fopenmp")
                libs.append("gomp")

    return cflags, ldflags, include_dir_paths, lib_dir_paths, libs, passthrough_args


def _get_libstdcxx_args() -> tuple[list[str], list[str]]:
    """
    For fbcode cpu case, we should link stdc++ instead assuming the binary where dlopen is executed is built with dynamic stdc++.
    """
    lib_dir_paths: list[str] = []
    libs: list[str] = []
    if config.is_fbcode():
        lib_dir_paths = [sysconfig.get_config_var("LIBDIR")]
        libs.append("stdc++")

    return lib_dir_paths, libs


def get_mmap_self_macro(
    use_mmap_weights: bool, use_mmap_weights_external: bool
) -> list[str]:
    macros = []

    if use_mmap_weights 
```



## High-Level Overview


This Python file contains 11 class(es) and 107 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WinPeFileVersionInfo`, `BuildOptionsBase`, `CppOptions`, `CppTorchOptions`, `CppTorchDeviceOptions`, `CppBuilder`

**Functions defined**: `log_global_cache_errors`, `log_global_cache_stats`, `log_global_cache_vals`, `use_global_cache`, `cpp_compiler_search`, `install_gcc_via_conda`, `check_compiler_exist_windows`, `__init__`, `_setup_functions`, `_get_version_info`, `get_language_id`, `check_msvc_cl_language_id`, `get_msvc_cl_path`, `check_mingw_win32_flavor`, `get_cpp_compiler`, `get_ld_and_objcopy`, `convert_cubin_to_obj`, `_is_apple_clang`, `_is_clang`, `_is_gcc`

**Key imports**: copy, ctypes, errno, functools, json, locale, logging, os, platform, re


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `ctypes`
- `errno`
- `functools`
- `json`
- `locale`
- `logging`
- `os`
- `platform`
- `re`
- `shlex`
- `shutil`
- `subprocess`
- `sys`
- `sysconfig`
- `tempfile`
- `textwrap`
- `warnings`
- `collections.abc`: Sequence
- `ctypes.util`: find_library
- `pathlib`: Path
- `typing`: Any, Optional, Union
- `torch`
- `torch._dynamo.utils`: dynamo_timed
- `torch._inductor`: config, exc
- `torch._inductor.cpu_vec_isa`: invalid_vec_isa, VecISA
- `torch._inductor.runtime.runtime_utils`: cache_dir
- `torch.torch_version`: TorchVersion
- `triton.fb.build`: _run_build_command, build_paths


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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `cpp_builder.py_docs.md`
- **Keyword Index**: `cpp_builder.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
