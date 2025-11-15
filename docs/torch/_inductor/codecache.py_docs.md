# Documentation: `torch/_inductor/codecache.py`

## File Metadata

- **Path**: `torch/_inductor/codecache.py`
- **Size**: 174,476 bytes (170.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import importlib.resources
import io
import itertools
import json
import logging
import os
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import warnings
from bisect import bisect_right
from copy import copy
from ctypes import c_void_p, CDLL, cdll
from datetime import timedelta
from functools import lru_cache, partial
from pathlib import Path
from tempfile import _TemporaryFileWrapper
from time import time, time_ns
from types import ModuleType
from typing import Any, cast, Generic, NoReturn, TYPE_CHECKING, TypeVar, Union
from typing_extensions import override, Self

import torch
import torch.distributed as dist
from torch import SymInt, Tensor
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.exc import SkipFrame
from torch._dynamo.utils import (
    CompileEventLogger,
    counters,
    dynamo_timed,
    get_metrics_context,
)
from torch._inductor import config, exc, metrics
from torch._inductor.codegen.common import (
    custom_backend_codegen_configs,
    custom_backend_passes,
    init_backend_registration,
)
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.codegen.rocm.compile_command import (
    rocm_compile_command,
    rocm_compiler,
)
from torch._inductor.compile_worker.utils import in_toplevel_process
from torch._inductor.cpp_builder import (
    _LINKER_SCRIPT,
    _set_gpu_runtime_env,
    _TORCH_PATH,
    _transform_cuda_paths,
    convert_cubin_to_obj,
    CppBuilder,
    CppOptions,
    CppTorchDeviceOptions,
    get_compiler_version_info,
    get_ld_and_objcopy,
    get_name_and_dir_from_output_file_path,
    normalize_path_separator,
    run_asm_build_object,
)
from torch._inductor.cpu_vec_isa import pick_vec_isa
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    CustomGraphPass,
    CustomGraphPassType,
    CustomPartitionerFn,
    CustomPartitionerFnType,
)
from torch._inductor.freezing_utils import has_frozen_params, is_frozen_param
from torch._inductor.runtime.compile_tasks import _reload_python_module
from torch._inductor.runtime.runtime_utils import cache_dir, default_cache_dir
from torch._inductor.utils import (
    ALIGN_BYTES,
    clear_on_fresh_cache,
    determine_aoti_mmap_flags,
    is_linux,
    is_windows,
)
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import (
    extract_tensor_metadata,
    FakeTensor,
    TensorMetadata,
)
from torch._utils_internal import log_cache_bypass
from torch.compiler import config as cconfig
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.export.pt2_archive._package_weights import TensorProperties, Weights
from torch.export.pt2_archive.constants import CUSTOM_OBJ_FILENAME_PREFIX
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv
from torch.utils._ordered_set import OrderedSet

from .output_code import CompiledFxGraph
from .remote_cache import create_cache
from .runtime import autotune_cache
from .runtime.autotune_cache import AutotuneCacheBundler
from .triton_bundler import TritonBundler
from .virtualized import V


if config.is_fbcode():
    from triton.fb.build import build_paths


T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, KeysView, Sequence
    from concurrent.futures import Future

    from .compile_fx import _CompileFxKwargs
    from .cpp_builder import BuildOptionsBase
    from .graph import GraphLowering
    from .ir import ChoiceCaller
    from .output_code import CompiledFxGraphConstants, OutputCode
    from .remote_cache import JsonDataTy, RemoteCache
    from .runtime.hints import HalideInputSpec, HalideMeta
    from .runtime.triton_heuristics import CachingAutotuner
    from .utils import InputType


_IS_WINDOWS = sys.platform == "win32"
LOCK_TIMEOUT = config.file_lock_timeout

output_code_log = torch._logging.getArtifactLogger(__name__, "output_code")
autotuning_log = torch._logging.getArtifactLogger(__name__, "autotuning")
log = logging.getLogger(__name__)


def use_re_build() -> bool:
    """
    Use for CUTLASS compilation only right now.
    """
    if config.is_fbcode() and not cuda_env.nvcc_exist(_cuda_compiler()):
        from triton.fb.re_build_helper import should_build_locally

        return not should_build_locally()
    return False


def get_cpp_wrapper_cubin_path_name() -> str:
    return "cubin_path" if torch.version.hip is None else "hsaco_path"


def get_kernel_bin_format(device: str) -> str:
    if device == "cuda":
        return "cubin" if torch.version.hip is None else "hsaco"
    elif device == "xpu":
        return "spv"
    else:
        return ""


def get_device_information(device_type: str) -> dict[str, str]:
    """
    Gets all the current device information used to compile the .so.
    """
    metadata: dict[str, str] = {
        "AOTI_PLATFORM": sys.platform,
        "AOTI_MACHINE": platform.machine(),
        "AOTI_CPU_ISA": str(torch._inductor.cpu_vec_isa.pick_vec_isa()).upper(),
        "AOTI_COMPUTE_CAPABILITY": str(
            get_interface_for_device(device_type).get_compute_capability()
        ),
    }
    return metadata


class CacheBase:
    @staticmethod
    @functools.cache
    def get_system() -> dict[str, Any]:
        from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

        if HAS_TRITON:
            # Use triton_key instead of triton.__version__ as the version
            # is not updated with each code change
            triton_version = triton_key()
        else:
            triton_version = None

        try:
            system: dict[str, Any] = {
                "device": {"name": None},
                "version": {
                    "triton": triton_version,
                },
            }
            device_properties = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            )
            if torch.version.cuda is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cuda"] = torch.version.cuda
            else:
                system["device"]["name"] = device_properties.gcnArchName
                system["version"]["hip"] = torch.version.hip
        except (AssertionError, RuntimeError):
            # If cuda is not installed, none of the above config is relevant.
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return system

    @staticmethod
    @clear_on_fresh_cache
    @functools.cache
    def get_local_cache_path() -> Path:
        return Path(os.path.join(cache_dir(), "cache", CacheBase.get_system()["hash"]))

    def __init__(self) -> None:
        self.system = CacheBase.get_system()

    def get_local_cache(self) -> dict[str, Any]:
        local_cache_path = self.get_local_cache_path()
        if not local_cache_path.is_file():
            return {}
        with open(local_cache_path) as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache["cache"]

    def update_local_cache(self, local_cache: dict[str, Any]) -> None:
        local_cache_path = self.get_local_cache_path()
        write_atomic(
            str(local_cache_path),
            json.dumps({"system": self.system, "cache": local_cache}, indent=4),
            make_dirs=True,
        )


class LocalCache(CacheBase):
    def lookup(self, *keys: str) -> dict[str, Any] | None:
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys:
            if key in cache:
                sub_cache = cache[key]
            else:
                return None

        return sub_cache

    def set_value(self, *keys: str, value: Any) -> None:
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys[0:-1]:
            sub_cache.setdefault(key, {})
            sub_cache = sub_cache[key]
        sub_cache[keys[-1]] = value

        self.update_local_cache(cache)


class PersistentCache(CacheBase):
    def lookup(
        self,
        choices: list[ChoiceCaller],
        op: str,
        inputs: str,
        benchmark: Callable[[Any], dict[ChoiceCaller, float]] | None,
        hint_override: int | None = None,
    ) -> dict[ChoiceCaller, float]:
        """
        Check to see if we have benchmarked the given choice callers. For each
        choice caller:

            1. Check local_cache[op][inputs][choice][precision], return benchmark if cached.
            2. If benchmark is not None:
                a. `max_autotune_gemm=True`: benchmark the choice, update
                    local_cache[op][inputs][choice], and return the benchmark.
                b. `max_autotune_gemm=False`: don't benchmark the choice, return nothing.
        """
        precision = torch.get_float32_matmul_precision()
        cache_key = f"{inputs}_{hint_override}" if hint_override is not None else inputs

        timings = {}

        def check_cache(cache: dict[str, Any]) -> bool:
            """Check if `cache` contains data for all the choices"""
            hit = True
            for choice in choices:
                choice_hash = choice.hash_key()
                if choice_hash in cache.get(op, {}).get(cache_key, {}).get(
                    precision, {}
                ):
                    # cache hit
                    timings[choice] = cache[op][cache_key][precision][choice_hash]
                else:
                    # cache miss
                    hit = False
                    break
            return hit

        local_cache = self.get_local_cache() if config.autotune_local_cache else {}
        if (not check_cache(local_cache)) and (benchmark is not None):
            # re-benchmark everything to try to get consistent numbers from the same machine
            timings = benchmark(choices)
            assert all(choice in timings for choice in choices)
            local_cache.setdefault(op, {})
            local_cache[op].setdefault(cache_key, {}).setdefault(precision, {})
            for choice, timing in timings.items():
                local_cache[op][cache_key][precision][choice.hash_key()] = timing

            self.update_local_cache(local_cache)

        return timings


def get_lock_dir() -> str:
    lock_dir = os.path.join(cache_dir(), "locks")
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def sha256_hash(data: bytes) -> str:
    # [:51] to strip off the "Q====" suffix common to every hash value.
    return base64.b32encode(hashlib.sha256(data).digest())[:51].decode("utf-8").lower()


def code_hash(code: str | bytes, extra: str | bytes = "") -> str:
    hashing_str = code if isinstance(code, bytes) else code.encode("utf-8")
    if extra:
        extra_b = extra if isinstance(extra, bytes) else extra.encode("utf-8")
        hashing_str = hashing_str + b"||" + extra_b
    return "c" + sha256_hash(hashing_str)


def get_path(
    basename: str, extension: str, specified_dir: str = ""
) -> tuple[str, str, str]:
    if specified_dir:
        if os.path.isabs(specified_dir):
            subdir = specified_dir
        else:
            subdir = os.path.join(cache_dir(), specified_dir)
    else:
        subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{extension}")
    return basename, subdir, path


def get_hash(content: str | bytes, extra: str = "", hash_type: str = "code") -> str:
    if hash_type in {"amdgcn", "code", "ptx", "spv"}:
        return code_hash(content, extra)
    if hash_type in {"cubin", "hsaco", "spv"}:
        return code_hash(repr(content))
    raise AssertionError(f"Unknown hash type {hash_type}")


class WritableTempFile:
    """
    Avoid "Permission denied error" on Windows:
      with tempfile.NamedTemporaryFile("w", suffix=".gv") as temp_file:
        # Not writable on Windows:
        # https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile

    Example:
        with WritableTempFile("w", suffix=".gv") as temp_file:
            tree.to_dotfile(temp_file.name)
    """

    def __init__(
        self, mode: str = "w", *, encoding: Any = None, suffix: Any = None
    ) -> None:
        self.mode = mode
        self.encoding = encoding
        self.suffix = suffix

    def __enter__(self) -> _TemporaryFileWrapper[Any]:
        self.temp_file = tempfile.NamedTemporaryFile(
            self.mode, encoding=self.encoding, suffix=self.suffix, delete=False
        )
        return self.temp_file

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.temp_file.close()
        try:
            os.unlink(self.temp_file.name)
        except OSError as e:
            if _IS_WINDOWS:
                # On Windows, some case temp file is opened and fail to unlink. Need to ignore it.
                pass
            else:
                raise e


def write(
    content: str | bytes,
    extension: str,
    extra: str = "",
    hash_type: str = "code",
    specified_dir: str = "",
    key: str | None = None,
) -> tuple[str, str]:
    if key is None:
        # use striped content to compute hash so we don't end up with different
        # hashes just because the content begins/ends with different number of
        # spaces.
        key = get_hash(content.strip(), extra, hash_type)
    basename, _subdir, path = get_path(key, extension, specified_dir)
    if not os.path.exists(path):
        write_atomic(path, content, make_dirs=True)
    return basename, path


def write_text(text: str) -> str:
    """
    Write the `text` to a file and return the path computed based on the hash.
    """
    return write(text, "txt")[1]


def write_atomic(
    path_: str,
    content: str | bytes,
    make_dirs: bool = False,
    encode_utf_8: bool = False,
) -> None:
    # Write into temporary file first to avoid conflicts between threads
    # Avoid using a named temporary file, as those have restricted permissions
    assert isinstance(content, (str, bytes)), (
        "Only strings and byte arrays can be saved in the cache"
    )
    path = Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode, encoding="utf-8" if encode_utf_8 else None) as f:
        f.write(content)
    try:
        tmp_path.rename(target=path)
    except FileExistsError:
        if not _IS_WINDOWS:
            raise
        # On Windows file exist is expected: https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename
        # Below two lines code is equal to `tmp_path.rename(path)` on non-Windows OS.
        # 1. Copy tmp_file to Target(Dst) file.
        shutil.copy2(src=tmp_path, dst=path)
        # 2. Delete tmp_file.
        os.remove(tmp_path)


@dataclasses.dataclass
class TensorMetadataAndValues:
    """
    TensorMetadata plus the elements as a list of raw values.
    Used for hashing inlined constants.
    """

    tensor_metadata: TensorMetadata
    values: list[Any]


def _ident(x: T) -> T:
    return x


def extract_tensor_metadata_for_cache_key(t: Tensor) -> TensorMetadata:
    """
    Extracts the tensor metadata and removes fields of the TensorMetadata
    that are not needed for caching
    """
    meta = extract_tensor_metadata(t)
    if not hasattr(t, "_is_inductor_static"):
        meta = dataclasses.replace(meta, storage_offset=0, storage_bytes=None)

    return meta


class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        has_user_defined_triton_kernels: bool = False,
    ) -> None:
        """
        Create an FX graph pickler. If include_non_inlined=True, then pickling will
        include the _values_ for all Tensors. (Note that any tensors are constants
        attached as attributes to the GraphModule). Otherwise, pickling will include
        only the metadata for these tensors.
        """
        self._stream = io.BytesIO()
        super().__init__(self._stream)

        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table.update(
            {
                FakeTensor: functools.partial(self._reduce_fake_tensor),
                torch.Tensor: functools.partial(self._reduce_tensor),
                torch.nn.parameter.Parameter: functools.partial(self._reduce_tensor),
                torch.SymInt: functools.partial(self._reduce_symint),
                torch.fx.experimental._backward_state.BackwardState: functools.partial(
                    self._reduce_unsupported
                ),
            }
        )
        if has_user_defined_triton_kernels:
            # Need to use runtime type as GraphModule generates a singleton in __new__ function
            self.dispatch_table[gm.__class__] = functools.partial(
                self._reduce_graph_module
            )

        # Run with pickler.fast so it doesn't intern strings, making the hash result more predictable
        # TODO: pickler.fast is technically deprecated. Will this work on new python versions?
        self.fast = True

    def _reduce_fake_tensor(
        self, t: Tensor
    ) -> tuple[Callable[[T], T], tuple[TensorMetadata]]:
        """
        Custom reducer to pickle FakeTensors.
        """
        metadata = extract_tensor_metadata_for_cache_key(t)
        return (_ident, (metadata,))

    def _reduce_tensor(
        self, t: Tensor
    ) -> tuple[Callable[[T], T], tuple[TensorMetadata | TensorMetadataAndValues]]:
        """
        Custom reducer to pickle Tensors.  If we see tensors, we know they're constants
        stored as attributes on the GraphModule.
        """
        from .graph import GraphLowering

        if t.is_mkldnn:
            # TODO: These tensors don't currently pickle, so we can't cache a compiled
            # graph containing them. Just fail now. If mkldnn tensors get pickling
            # support, we can remove this.
            raise BypassFxGraphCache("mkldnn tensors unpickleable")

        metadata = extract_tensor_metadata_for_cache_key(t)

        # If this is a non-inlined frozen parameter, we consider the metadata only.
        if is_frozen_param(t) and not GraphLowering.can_inline_constant(t):
            return (_ident, (metadata,))

        # Very large tensors will be expensive to copy to cpu and hash. Let's at least
        # report any slowness.
        start = time()
        values = t.tolist()
        elapsed = time() - start
        if elapsed > 1.0:
            warnings.warn(
                f"FX graph cache copying of a large constant took {elapsed:.1}s. "
                "Please file an issue."
            )

        return (_ident, (TensorMetadataAndValues(metadata, values),))

    def _reduce_symint(self, s: SymInt) -> tuple[Callable[[T], T], tuple[str]]:
        """
        Custom reducer to pickle SymInts.
        """
        # For hashing purposes, we only care about the name of the symbol and not the
        # backed value. We evaluate guards stored with a cached graph to ensure a cached
        # entity with SymInt args is safe to reuse.
        return (_ident, (str(s),))

    def _reduce_unsupported(self, s: Any) -> NoReturn:
        """
        Custom reducer to handle any objects that we don't support and therefore
        raise to bypass caching.
        """
        raise BypassFxGraphCache("Reduce unsupported")

    def _reduce_graph_module(
        self, gm: torch.fx.GraphModule
    ) -> tuple[Any, tuple[dict[str, Any], str]]:
        """
        Custom reducer for graph module to handle irrelevant data for user
        defined triton kernels
        Essentially what we are doing here is a huge hack where user defined
        triton kernel contain a dynamo time side table and the arguments to the
        call_function are indices into this side table. These arguments are not
        for hashing purposes since we included the source code into the cache
        key and the numbers are prone to give false negatives due to ordering.
        """
        fn, (data, imports) = gm.__reduce__()
        code = data["_code"]
        code = re.sub(r"kernel_idx = \d+", "", code)
        code = re.sub(r"constant_args_idx = \d+", "", code)
        data["_code"] = code
        return fn, (data, imports)

    def dumps(self, obj: Any) -> bytes:
        """
        Pickle an object and return a byte string.
        """
        try:
            self.dump(obj)
            return self._stream.getvalue()
        except (TypeError, AttributeError, pickle.PicklingError) as e:
            # Some configs options may not pickle.
            log.warning("Failed to pickle cache key", exc_info=True)
            raise BypassFxGraphCache("Failed to pickle cache key") from e
        finally:
            # Reset our stream for the next dump.
            self._stream.seek(0)
            self._stream.truncate(0)

    def get_hash(self, obj: Any) -> str:
        """
        Serialize an object and return a hash of the bytes.
        """
        serialized_data = self.dumps(obj)
        return sha256_hash(serialized_data)

    def debug_lines(self, inp: FxGraphHashDetails) -> list[str]:
        """
        Get a printable string describing in more detail all the attributes
        comprising an object. Useful for debugging when one graph hashes
        to a different value than another.
        """

        def get_str(obj: Any) -> str:
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata_for_cache_key(obj))
            elif isinstance(obj, bytes):
                val = obj.decode("utf-8", errors="replace")
                return val if len(val) <= 1024 else val[:1024] + "..."
            elif type(obj) in self.dispatch_table:
                # Run the reducer on the object
                return str(self.dispatch_table[type(obj)](obj)[1])
            else:
                return str(obj)

        lines = []
        for attr, obj in vars(inp).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = self.get_hash(obj[ii])
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = self.get_hash(v)
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                h = self.get_hash(obj)
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        return lines


def build_code_hash(
    roots: list[str] | None, prefix: str, hasher: hashlib._Hash
) -> None:
    for lib in sorted(pkgutil.iter_modules(roots, prefix), key=lambda x: x.name):
        spec = lib.module_finder.find_spec(lib.name, None)
        assert spec is not None
        module = spec.origin
        assert module is not None
        with open(module, "rb") as f:
            hasher.update(spec.name.encode("utf-8"))
            hasher.update(f.read())
        if lib.ispkg:
            # need to also hash submodules
            build_code_hash(spec.submodule_search_locations, f"{spec.name}.", hasher)


def torch_key_cache(func: Callable[[], bytes]) -> Callable[[], bytes]:
    """
    This function is a reimplementation of functools.lru_cache with a
    set function that allows prepopulating the cache.
    """
    # Use list for reference semantics
    _cache: list[bytes] = []

    def wrapper() -> bytes:
        if len(_cache) == 0:
            _cache.append(func())
        return _cache[0]

    def set_val(val: bytes) -> None:
        assert len(_cache) == 0
        _cache.append(val)

    def clear() -> None:
        _cache.clear()

    wrapper.set = set_val  # type: ignore[attr-defined]
    wrapper.clear = clear  # type: ignore[attr-defined]
    return wrapper


@torch_key_cache
def torch_key() -> bytes:
    """
    Compute a key that contains relevant information about torch source files
    """
    with dynamo_timed("inductor_codecache_torch_key", log_pt2_compile_event=False):
        if not config.is_fbcode():

            def get_code_hash(root: str) -> bytes:
                # This function isn't meant to be used outside of torch_key, just a
                # helper for clarity. Instead, use torch_key() directly when you need
                # a hash representing the state of the source code.
                extra_files = (
                    "codegen/aoti_runtime/interface.cpp",
                    "script.ld",
                )
                inductor_root = os.path.dirname(__file__)
                extra_files = [os.path.join(inductor_root, x) for x in extra_files]
                hasher = hashlib.sha256()
                hasher.update(torch.__version__.encode("utf-8"))
                build_code_hash([root], "", hasher)
                for path in extra_files:
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            hasher.update(f.read())
                return hasher.digest()

            return get_code_hash(_TORCH_PATH)

        from libfb.py import parutil

        return parutil.get_file_contents("torch/src_hash.txt").rstrip().encode("ascii")


def get_inductor_root() -> str:
    return os.path.dirname(__file__)


@dataclasses.dataclass
class OrderedSetHolder:
    """
    See FxGraphHashDetails. Holds a sorted list to support stable hashing
    of set kwargs.
    """

    items: list[Any]


class BypassFxGraphCache(Exception):
    """
    Exception to indicate that the FxGraphCache should be bypassed.
    """


class FxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """

    # Excluded kwargs param that are not stable between runs
    EXCLUDED_KWARGS = ["graph_id"]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[InputType],
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
    ) -> None:
        self.gm = gm
        self.example_inputs = example_inputs
        self.cache_key_tag = cconfig.cache_key_tag

        # Order kwargs so hashing is stable to changes in kwarg order. Although
        # it's technically a _CompileFxKwargs we don't actually need it typed as
        # such since we're just using it to generate a hash.
        self.fx_kwargs: dict[str, object] = {}
        for k, v in sorted(fx_kwargs.items()):
            if k not in self.EXCLUDED_KWARGS:
                if type(v) in (set, OrderedSet):  # noqa: set_linter
                    # Special case to handle set params. Python sets can't be
                    # ordered, so sort the elements and store them in a proxy.
                    self.fx_kwargs[k] = OrderedSetHolder(sorted(v))  # type: ignore[call-overload]
                else:
                    self.fx_kwargs[k] = v

        from torch._higher_order_ops.triton_kernel_wrap import (
            kernel_side_table,
            triton_kernel_wrapper_functional,
            triton_kernel_wrapper_mutation,
        )
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        # Node meta will not be part of gm's reduce function, so lets remember
        # the kernel source code separately
        self.user_defined_triton_source: list[Any] = []
        if gm is not None:
            for module in gm.modules():
                if not isinstance(module, torch.fx.GraphModule):
                    continue
                for node in itertools.chain(
                    module.graph.find_nodes(
                        op="call_function", target=triton_kernel_wrapper_functional
                    ),
                    module.graph.find_nodes(
                        op="call_function", target=triton_kernel_wrapper_mutation
                    ),
                ):
                    from triton.runtime.autotuner import Autotuner

                    kernel = kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
                    configs = None
                    if isinstance(kernel, Autotuner):
                        if kernel.configs:
                            configs = str(
                                sorted(
                                    sorted(str(kv) for kv in c.all_kwargs().items())
                                    for c in kernel.configs
                                )
                            )
                        kernel = kernel.fn

                    kernel_source = (
                        user_defined_triton_kernel_transitive_closure_source_code(
                            kernel
                        )
                    )
                    constant_args = kernel_side_table.get_constant_args(
                        node.kwargs["constant_args_idx"]
                    )
                    self.user_defined_triton_source.append(
                        (kernel_source, constant_args, configs)
                    )

        # Alignment checks
        self.inputs_to_check = inputs_to_check

        no_tensor_inputs = not any(isinstance(x, torch.Tensor) for x in example_inputs)
        # This device index is usually already encoded by the device of the inputs
        # but fx graphs don't necessarily have tensor inputs. If there aren't any,
        # we need to guard on the device index in case we allocate cuda tensors
        if no_tensor_inputs and torch.accelerator.is_available():
            self.default_cuda_device_index = torch.accelerator.current_device_index()

        # 'Deterministic algorithms' can affect codegen via lowering to cuda kernels.
        self.deterministic_algorithms_settings = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
        )

        # Global settings affecting matmul codegen.
        self.cuda_matmul_settings = (
            torch.backends.cuda.matmul.fp32_precision,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )

        # Also hash on various system info (including the triton compiler version).
        self.torch_version = torch_key()
        self.system_info = CacheBase.get_system()
        self.inductor_config = config.save_config_portable(ignore_private_configs=False)
        # Custom post grad passes should provide an ID to hash.
        self.post_grad_custom_pre_pass = self._get_custom_pass_detail(
            config.post_grad_custom_pre_pass
        )
        # TODO: change to more holistic config rather than bundled_autograd_cache
        self.precompile_enabled = torch._functorch.config.bundled_autograd_cache
        self.post_grad_custom_post_pass = self._get_custom_pass_detail(
            config.post_grad_custom_post_pass
        )
        self.joint_custom_pre_pass = self._get_custom_pass_detail(
            config.joint_custom_pre_pass
        )
        self.joint_custom_post_pass = self._get_custom_pass_detail(
            config.joint_custom_post_pass
        )
        self._pre_fusion_custom_pass = self._get_custom_pass_detail_unsafe(
            config._pre_fusion_custom_pass
        )
        self._fuse_ddp_communication_passes = self._get_custom_pass_detail_unsafe(
            config._fuse_ddp_communication_passes
        )

        # Register indcutor backends and custom passes and get their UUIDs.
        init_backend_registration()
        self.custom_backend_passes = tuple(
            map(self._get_custom_pass_detail, custom_backend_passes.values())
        )

        # Save custom inductor codegen configs
        self.custom_backend_codegen_configs = {
            device: custom_config.save_config_portable(ignore_private_configs=False)
            for device, custom_config in custom_backend_codegen_configs.items()
            if custom_config is not None
        }

        # Register the custom partitioner function
        self._custom_partitioner_fn = self._get_custom_partitioner_fn_detail(
            config.custom_partitioner_fn
        )

    # This is mainly added to handle these two inductor configs, which are (unfortunately)
    # sometimes cache safe:
    # - _pre_fusion_custom_pass
    # - _fuse_ddp_communication_passes
    # Their types can be found in `torch/_inductor/config.py`, but:
    # - if they are string names, we can cache them safely (one is by default)
    # - if any of them are set to custom callables, we will need to cache miss
    # Future work is for someone to find any places where these functions are used
    # and force them to be of type CustomGraphPass, so we can guarantee serialization.
    def _get_custom_pass_detail_unsafe(self, custom_pass: Any) -> Any | None:
        if not custom_pass:
            return None
        if isinstance(custom_pass, list):
            return [self._get_custom_pass_detail_unsafe(x) for x in custom_pass]
        if isinstance(custom_pass, str):
            return custom_pass
        if isinstance(custom_pass, CustomGraphPass):
            return custom_pass.uuid()
        if callable(custom_pass):
            # Returning None is safe here because we raise an explicit bypass error
            # later if we detect these passes are set to callables
            return None
        raise AssertionError(f"unknown config type: {str(type(custom_pass))}")

    def _get_custom_pass_detail(
        self, custom_pass: CustomGraphPassType | CustomGraphModulePass
    ) -> Any | None:
        if not custom_pass:
            return None
        assert isinstance(custom_pass, (CustomGraphPass, CustomGraphModulePass))
        return custom_pass.uuid()

    def _get_custom_partitioner_fn_detail(
        self, custom_partitioner_fn: CustomPartitionerFnType
    ) -> Any | None:
        if not custom_partitioner_fn:
            return None
        assert isinstance(custom_partitioner_fn, CustomPartitionerFn)
        return custom_partitioner_fn.uuid()


def compiled_fx_graph_hash(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[InputType],
    fx_kwargs: _CompileFxKwargs,
    inputs_to_check: Sequence[int],
) -> tuple[str, list[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
    has_user_defined_triton_kernels = len(details.user_defined_triton_source) != 0
    pickler = FxGraphCachePickler(gm, has_user_defined_triton_kernels)

    # The prefix distinguishes among the other kinds of objects we
    # cache in this module.
    key = "f" + pickler.get_hash(details)
    debug_lines = pickler.debug_lines(details)
    debug_str = "\n".join(debug_lines)
    log.debug(f"FX graph cache hash details for key {key}:\n{debug_str}")  # noqa: G004
    return key, debug_lines


def add_ephemeral_timeout_increase_for_distributed(time_saved_ns: int) -> int:
    """
    Ephemerally increases the NCCL timeout when compiling for a distributed job
    Returns amount of seconds increased
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0

    increased_timeout_sec = int(time_saved_ns // 1e9)  # convert to seconds

    if config.is_fbcode():
        fudge_factor = torch._utils_internal.justknobs_getval_int(
            "pytorch/remote_cache:ephemeral_timeout_fudge_factor_percentage"
        )
        log.info(
            "Ephemeral NCCL timeout increase fudge factor %d and original increase value %d",
            fudge_factor,
            increased_timeout_sec,
        )
        increased_timeout_sec += int(increased_timeout_sec * fudge_factor / 100)

    log.info("Increasing NCCL timeout by %d", increased_timeout_sec)
    dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
        timedelta(seconds=increased_timeout_sec)
    )
    return increased_timeout_sec


class GuardedCache(Generic[T]):
    """
    Mixin for caches that have guards associated with their entries.
    """

    @classmethod
    def _get_tmp_dir_for_key(cls: type[GuardedCache[T]], _key: str) -> str:
        raise NotImplementedError("Implement _get_tmp_dir_for_key on parent class")

    @classmethod
    def iterate_over_candidates(
        cls: type[GuardedCache[T]],
        local: bool,
        remote_cache: RemoteCache[JsonDataTy] | None,
        key: str,
    ) -> Generator[tuple[T, bytes], None, None]:
        if local:
            subdir = cls._get_tmp_dir_for_key(key)
            if os.path.exists(subdir):
                for path in sorted(os.listdir(subdir)):
                    try:
                        with open(os.path.join(subdir, path), "rb") as f:
                            content = f.read()
                            yield pickle.loads(content), content
                    except Exception:
                        log.warning(
                            "fx graph cache unable to load compiled graph",
                            exc_info=True,
                        )

        if remote_cache:
            try:
                if (cache_data := remote_cache.get(key)) is not None:
                    assert isinstance(cache_data, dict)
                    data = cache_data["data"]
                    assert isinstance(data, (str, bytes))
                    content = base64.b64decode(data)
                    yield pickle.loads(content), content
            except Exception:
                log.warning(
                    "%s unable to load compiled graph", cls.__name__, exc_info=True
                )

    @classmethod
    def find_guarded_entry(
        cls: type[GuardedCache[T]],
        key: str,
        local: bool,
        remote_cache: RemoteCache[JsonDataTy] | None,
        evaluate_guards: Callable[[str, list[int] | list[torch.SymInt]], bool],
        hints: list[int],
    ) -> tuple[T | None, bytes | None, dict[str, str]]:
        """
        Find the first cache entry in iterate_over_candidates that passes `evaluate_guards`.

        Args:
            key: The cache key to look up
            local: Whether to check the local cache
            remote_cache: The remote cache to check, if any
            evaluate_guards: Function that evaluates whether a guard passes the check,
                given a list of hint values and the guard expression.
            hints: List of symint hints paired with evaluate_guards

        Returns:
            A tuple of (graph, pickled_content) if found, or (None, None) if not found
        """
        graph = None
        pickled_content = None
        result_status = "full_miss"
        sample_guards_expr = None

        # Iterate over any entries in the subdir for this key and evaluate
        # guards to determine whether there's a hit.

        for candidate, content in cls.iterate_over_candidates(local, remote_cache, key):
            assert hasattr(candidate, "guards_expr")
            if not candidate.guards_expr:  # type: ignore[attr-defined]
                # No guards to evaluate, so this is a hit.
                graph = candidate
                pickled_content = content
                result_status = "hit"
                break

            # Evaluate the guard expression in the current context.
            # If there's not a cache hit, we don't want the evaluation to
            # affect the current env, e.g., cause the creation of new guards,
            # so we evaluate with the hints instead of the symbols.
            hit = bool(evaluate_guards(candidate.guards_expr, hints))  # type: ignore[attr-defined]
            if hit:
                graph = candidate
                pickled_content = content
                result_status = "hit"
                sample_guards_expr = candidate.guards_expr
                break
            else:
                # At least one guard missed, log this
                result_status = "guard_miss"
                sample_guards_expr = candidate.guards_expr

        info = {"cache_status_detailed": result_status}
        if sample_guards_expr is not None:
            info["cache_status_guard_expr"] = sample_guards_expr
        return graph, pickled_content, info

    @classmethod
    def _filter_backed_symints(
        cls: type[GuardedCache[T]], inputs: Sequence[InputType]
    ) -> list[torch.SymInt]:
        """
        Get the backed SymInt objects from the input list. Note that we can never
        have guards that depend on unbacked symint.
        """
        return [s for s in inputs if isinstance(s, torch.SymInt) and has_hint(s)]

    @classmethod
    def _get_shape_env(cls: type[GuardedCache[T]]) -> ShapeEnv | None:
        """
        Helper to get the shape env from the tracing context.
        """
        ctx = torch._guards.TracingContext.try_get()
        if not ctx or not ctx.fake_mode:
            return None
        return ctx.fake_mode.shape_env


@CacheArtifactFactory.register
class InductorCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None:
        FxGraphCache._write_to_local_cache(self.key, self.content)

    @override
    @staticmethod
    def type() -> str:
        return "inductor"


class FxGraphCache(GuardedCache[CompiledFxGraph]):
    """
    Supports caching and reusing compiled Fx graphs.

    The overall strategy is as follows:
    - This cache stores entries on disk. When saving an entry, we can't
      serialize callables (that could be C++, Triton, etc.), so we serialize
      their own disk cache location. We then recreate the compiled artifact
      after fetching from disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See FxGraphCachePickler.
    - Among the metadata we store, we also include a guards expression that's
      appropriate for validating any symbols for Tensor arguments that have
      symbolic bounds. On cache lookup then, we evaluate those guards in the
      current context to validate that a cached entry can be served.
    - A given graph could have multiple compiled versions, corresponding to
      different sets of guards. Therefore, we store cache entries in the form:
          <temp dir>/<fx graph hash>/<serialized metadata>
    - On lookup, we compute the key from the graph details, iterate over all
      leaf files in the corresponding subdirectory, deserialize the entry, and
      evaluate its guards expression. If the evaluation succeeds, we have a
      cache hit. If it fails, we compile the graph and store a new entry.
    - Finally, on a cache hit, we need to make sure any guards that would
      have been created during compilation are added to the current context.
    """

    # TODO(masnesral): Investigate whether it's beneficial to store compiled graphs
    # in an in-memory cache after loading from disk.
    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), "fxgraph")

    @classmethod
    def _get_tmp_dir_for_key(cls: type[FxGraphCache], key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
        return os.path.join(FxGraphCache._get_tmp_dir(), key[1:3], key)

    @staticmethod
    def cache_hit_post_compile(
        graph: CompiledFxGraph,
        cache_info: dict[str, Any],
        constants: CompiledFxGraphConstants,
    ) -> tuple[CompiledFxGraph | None, dict[str, Any]]:
        """
        Cache specific post compile steps that need to run if we find a graph in the cache
        This includes putting bundled triton artifacts in the right place,
        reloading the PyCodeCache artifact, etc.

        These don't always happen (i.e. on a cache miss, so they are in a separate function from
        CompiledFxGraph.post_compile)
        """
        if bundle := graph._triton_bundle:
            triton_bundler_meta = TritonBundler.read_and_emit(bundle)
            if (meta := triton_bundler_meta) is not None:
                cache_info["triton_bundler_meta"] = str(meta)
                CompileEventLogger.try_add_pt2_compile(
                    "inductor_compile", cached_kernel_names=meta.cached_kernel_names
                )
                CompileEventLogger.try_add_pt2_compile(
                    "AOTAutogradCache.inductor_load",
                    cached_kernel_names=meta.cached_kernel_names,
                )
                if len(meta.cached_kernel_names) > 0:
                    CompileEventLogger.try_(
                        CompileEventLogger.increment_toplevel, "num_triton_bundles"
                    )

        try:
            artifact_path = graph.after_deserialization(constants)

            from .graph import GraphLowering

            # This is used by tests to check the output for specific details.
            if GraphLowering.save_output_code is not None:
                GraphLowering.save_output_code(graph.source_code)

        except OSError:
            # Not expected, but in case the PyCodeCache entry is removed from
            # underneath us, treat it as a cache miss and recompile.
            return None, cache_info

        inductor_meta = autotune_cache.inductor_meta_from_config()
        code = graph.source_code
        AutotuneCacheBundler.begin_compile(inductor_meta, code=code)

        # Increment the cached metrics/counters by the amounts recorded when the FX
        # graph was compiled for this cache entry. Pretending these counters
        # were incremented normally is useful for testing with the cache enabled.
        metrics.CachedMetricsHelper.apply_deltas(graph.metrics_deltas)
        counters["inductor"] += graph.counter_deltas

        output_code_log.debug("Output code: \n%s", code)
        output_code_log.debug("Output code written to: %s", artifact_path)
        # On cache hit, use artifact path as filename
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_runnable",
                "encoding": "string",
            },
            payload_fn=lambda: graph.runnable_graph_str,
        )
        trace_structured(
            "inductor_post_grad_graph",
            payload_fn=lambda: graph.inductor_post_grad_graph_str,
        )
        trace_structured(
            "inductor_output_code",
            lambda: {
                "filename": artifact_path,
                "file_path": os.path.abspath(artifact_path),
            },
            payload_fn=lambda: code,
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_provenance_tracking_node_mappings",
                "encoding": "json",
            },
            payload_fn=lambda: graph.inductor_provenance_mapping_str,
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_provenance_tracking_kernel_stack_traces",
                "encoding": "json",
            },
            payload_fn=lambda: graph.inductor_provenance_stack_traces_str,
        )
        if (
            get_metrics_context().in_progress()
            and graph.inductor_provenance_stack_traces_str
        ):
            get_metrics_context().add_to_set(
                "inductor_provenance", graph.inductor_provenance_stack_traces_str
            )
        return graph, cache_info

    @staticmethod
    def _lookup_graph(
        key: str,
        example_inputs: Sequence[InputType],
        local: bool,
        remote_cache: RemoteCache[JsonDataTy] | None,
        constants: CompiledFxGraphConstants,
        evaluate_guards: Callable[[str, list[int] | list[torch.SymInt]], bool]
        | None = None,
    ) -> tuple[CompiledFxGraph | None, dict[str, Any]]:
        """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledFxGraph object. On a miss, return None.
        `constants` tracks a list of constants, or a way to obtain the list of constants
        associated with a given cache entry
        `evaluate_guards` allows AOTAutogradCache and other callers to customize
        what constitutes a guard success. Normally, a guard hit happens if
        `shape_env.evaluate_guards_expression` returns True.
        """
        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None

        symints = FxGraphCache._filter_backed_symints(example_inputs)
        hints = [hint_int(s) for s in symints]

        # If this config is turned on, everything is a guard hit and we check nothing
        if config.unsafe_skip_cache_dynamic_shape_guards:
            # This also makes it so we don't add anything to the dynamic
            # shape environment
            evaluate_guards = la
```



## High-Level Overview


This Python file contains 33 class(es) and 160 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CacheBase`, `LocalCache`, `PersistentCache`, `WritableTempFile`, `TensorMetadataAndValues`, `FxGraphCachePickler`, `OrderedSetHolder`, `BypassFxGraphCache`, `FxGraphHashDetails`, `GuardedCache`, `InductorCacheArtifact`, `FxGraphCache`, `CudaKernelParamCache`, `AotCodeCompiler`, `SYSTEM_INFO`, `CppCodeCache`, `CppPythonBindingsCodeCache`, `CppWrapperCodeCache`, `HalideCodeCache`, `Out`

**Functions defined**: `use_re_build`, `get_cpp_wrapper_cubin_path_name`, `get_kernel_bin_format`, `get_device_information`, `get_system`, `get_local_cache_path`, `__init__`, `get_local_cache`, `update_local_cache`, `lookup`, `set_value`, `lookup`, `check_cache`, `get_lock_dir`, `sha256_hash`, `code_hash`, `get_path`, `get_hash`, `__init__`, `__enter__`

**Key imports**: annotations, base64, copyreg, dataclasses, functools, hashlib, importlib, importlib.resources, io, itertools


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `base64`
- `copyreg`
- `dataclasses`
- `functools`
- `hashlib`
- `importlib`
- `importlib.resources`
- `io`
- `itertools`
- `json`
- `logging`
- `os`
- `pickle`
- `pkgutil`
- `platform`
- `re`
- `shlex`
- `shutil`
- `struct`
- `subprocess`
- `sys`
- `tempfile`
- `textwrap`
- `threading`
- `warnings`
- `bisect`: bisect_right
- `copy`: copy
- `ctypes`: c_void_p, CDLL, cdll
- `datetime`: timedelta


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data
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

- **File Documentation**: `codecache.py_docs.md`
- **Keyword Index**: `codecache.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
