# Documentation: `docs/torch/_inductor/triton_bundler.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/triton_bundler.py_docs.md`
- **Size**: 19,646 bytes (19.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/triton_bundler.py`

## File Metadata

- **Path**: `torch/_inductor/triton_bundler.py`
- **Size**: 15,987 bytes (15.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import copy
import dataclasses
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from torch._dynamo.utils import counters, dynamo_timed, set_feature_use
from torch._utils_internal import justknobs_check
from torch.utils._filelock import FileLock

from .runtime.runtime_utils import triton_cache_dir
from .utils import _IS_WINDOWS, GPU_KERNEL_BIN_EXTS


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TritonBundleEntry:
    """
    When we have compiled a triton kernel, we take note of that kernel by
    its triton generated hash, its device, and where this kernel is located.
    This is the minimum information we can use to later retrieve this kernel
    from file system.
    """

    kernel_hash: str
    device: int
    directory: str


@dataclasses.dataclass(frozen=True)
class TritonKernelArtifact:
    """
    Artifact for an individual kernel converted to bytes.
    Bytes could be a cubin, json, ttir, or ttgir.
    """

    filename: str
    payload: bytes = dataclasses.field(repr=False)  # Do not display binary


@dataclasses.dataclass(frozen=True)
class StaticallyLaunchedAutotuner:
    """
    Represents a statically compiled CachingAutotuner object that we can
    save directly in the cache. A CachingAutotuner is made up of a list of
    StaticTritonCompileResults, each of which uses the cubin from a TritonKernelArtifact.

    Statically saved here have their cubin files saved by a corresponding TritonBundleEntry.
    """

    cache_key: str
    kernel_name: str
    kernel: "CachingAutotuner"  # type: ignore[name-defined] # noqa: F821


@dataclasses.dataclass(frozen=True)
class TritonKernelArtifacts:
    """
    Collection of artifacts for a particular kernel.
    """

    kernel_hash: str
    device: int
    artifacts: list[TritonKernelArtifact]


@dataclasses.dataclass(frozen=True)
class TritonBundlerMetadata:
    """
    Metadata used for instrumentation
    """

    cached_kernel_names: list[str]
    statically_launched_kernel_names: list[str]


@dataclasses.dataclass(frozen=True)
class TritonBundle:
    """
    Serializable bundle to save into FXGraphCache
    """

    kernel_artifacts: list[TritonKernelArtifacts]
    static_autotuners: list[StaticallyLaunchedAutotuner]


class TritonBundler:
    """
    Lightweight Triton Kernel bundler that notes each time we compile a triton
    kernel. When collect is called, converts all the previously noted kernels and
    their artifacts into a structured bytes blob, and later when write is called
    it writes this structured blob back to file system.

    Intended Life cycle:
    - TritonBundler.begin_compile is called when we start compiling in Inductor
    - TritonBundler.put is called each time a Triton Kernel is compiled
    - TritonBundler.collect is called when a cache entry is being generated
    - TritonBundler.end_compile is called to indicate bundling is completed,
      collect will execute this function as well.
    - TritonBundler.read_and_emit is called when a cache entry is read
    """

    _entries: Optional[list[TritonBundleEntry]] = None
    _static_autotuners: Optional[list[StaticallyLaunchedAutotuner]] = None

    # __grp__kernel_name.json contains metadata with source code paths
    # we use this as sentinel value for search and replace
    _REPLACE_BYTES: bytes = b"[REPLACE]"

    @staticmethod
    def is_enabled() -> bool:
        from torch._inductor import config

        if config.force_disable_caches:
            return False

        if (b := config.bundle_triton_into_fx_graph_cache) is not None:
            return b

        if not config.is_fbcode():
            return False

        return justknobs_check(
            "pytorch/remote_cache:bundle_triton_into_fx_graph_cache_v2"
        )

    @classmethod
    def begin_compile(cls) -> None:
        """
        Initializes the TritonBundler.
        The current TritonBundler bundle is finalized by TritonBundler.collect.
        """
        if not TritonBundler.is_enabled():
            return
        log.debug("TritonBundler.begin_compile is called")
        assert cls._entries is None
        cls._entries = []
        cls._static_autotuners = []

    @classmethod
    def end_compile(cls) -> None:
        """
        Finalizes the TritonBundler. If collect is not yet called, it
        discards the current bundle.
        """
        log.debug("TritonBundler.end_compile is called")
        cls._entries = None
        cls._static_autotuners = None

    @classmethod
    def put(cls, kernel_hash: str, device: int) -> None:
        """
        Lazily observes that we have seen a Triton kernel compilation. Remembers
        it for when collect is later called.
        """
        if (entries := cls._entries) is not None:
            entries.append(
                TritonBundleEntry(kernel_hash, device, triton_cache_dir(device))
            )

    @classmethod
    def put_static_autotuner(cls, key: str, kernel: "CachingAutotuner") -> None:  # type: ignore[name-defined] # noqa: F821
        from torch._inductor import config

        assert config.use_static_cuda_launcher
        if (entries := cls._static_autotuners) is not None:
            # Clear a bunch of unpicklable values and make a copy to save
            # for FXGraphCache
            old_values = kernel.prepare_for_pickle()
            new_kernel = copy.deepcopy(kernel)
            new_kernel.prepare_for_caching()
            new_kernel._reload_kernel = None

            entries.append(
                StaticallyLaunchedAutotuner(
                    key,
                    new_kernel.inductor_meta.get("kernel_name", "unknown_kernel"),
                    new_kernel,
                )
            )

            # Put the values back since we need it to use now
            kernel.restore_after_unpickle(old_values)

    @classmethod
    def collect_static_autotuners(
        cls,
    ) -> tuple[list[StaticallyLaunchedAutotuner], list[str]]:
        if not cls._static_autotuners:
            return [], []
        else:
            log.info(
                "Saving %d statically launchable CachingAutotuners",
                len(cls._static_autotuners),
            )
            static_autotuner_names = [i.kernel_name for i in cls._static_autotuners]
            counters["inductor"]["triton_bundler_save_static_autotuner"] += 1
            return cls._static_autotuners, static_autotuner_names

    @classmethod
    def load_autotuners(
        cls, static_autotuners: Optional[list[StaticallyLaunchedAutotuner]]
    ) -> list[str]:
        """
        Load statically launchable CachingAutotuners into async_compile.CompiledTritonKernels
        cache.
        """
        if not static_autotuners:
            return []

        from torch._inductor.async_compile import CompiledTritonKernels
        from torch._inductor.codecache import StaticAutotunerFuture

        log.info("Loading %d statically launchable autotuners", len(static_autotuners))
        kernel_names = []
        with dynamo_timed("TritonBundler.load_cached_static_autotuners"):
            for result in static_autotuners:
                try:
                    # Make sure the cubin path exists and is valid
                    for compile_result in result.kernel.compile_results:
                        compile_result.reload_cubin_path()
                except RuntimeError:
                    log.warning(
                        "Failed to reload cubin file statically launchable autotuner %s",
                        result.kernel_name,
                        exc_info=True,
                    )
                    continue
                # We make a future instead of returning the kernel here so that
                # kernels that are not statically launchable (i.e. cache miss)
                # can launch a worker without waiting on the blocking step of
                # StaticAutotunerFuture.result().
                CompiledTritonKernels._cache[result.cache_key] = StaticAutotunerFuture(
                    result.kernel
                )
                counters["inductor"]["triton_bundler_load_static_autotuner"] += 1
                kernel_names.append(result.kernel_name)
        return kernel_names

    @classmethod
    def collect(
        cls,
    ) -> tuple[TritonBundle, Optional[TritonBundlerMetadata]]:
        """
        This is the main function called when a cache write happens. This function
        converts all the previously remembered kernels into bundled format so that
        it can be written into a cache entry.
        This function also finalizes the current bundle.
        """
        from torch._inductor import config

        if not TritonBundler.is_enabled():
            cls.end_compile()
            set_feature_use("triton_bundling", False)
            return TritonBundle([], []), None
        set_feature_use("triton_bundling", True)

        with dynamo_timed(key="TritonBundler.collect", log_pt2_compile_event=True):
            entries = cls._entries
            if entries is not None:
                result: list[TritonKernelArtifacts] = []
                kernel_names: list[str] = []
                for entry in entries:
                    artifacts: list[TritonKernelArtifact] = []
                    path = os.path.join(entry.directory, entry.kernel_hash)
                    if not os.path.exists(path):
                        continue
                    for filename in os.listdir(path):
                        filepath = os.path.join(path, filename)
                        try:
                            assert os.path.isfile(filepath)
                            with open(filepath, "rb") as file:
                                payload = file.read()
                                if filepath.endswith(".json"):
                                    # Make sure there's no sentinel value
                                    if TritonBundler._REPLACE_BYTES in payload:
                                        log.warning(
                                            "Bundle contains illegal %s, payload: %s",
                                            TritonBundler._REPLACE_BYTES,
                                            payload,
                                        )
                                        raise AssertionError(
                                            "Bundle contains illegal bytes"
                                        )
                                    # Remove the path from payload
                                    payload = payload.replace(
                                        str.encode(path), TritonBundler._REPLACE_BYTES
                                    )
                                artifacts.append(
                                    TritonKernelArtifact(filename, payload)
                                )
                            counters["inductor"]["triton_bundler_save_kernel"] += 1
                        except Exception:
                            log.debug("failed to collect triton kernel", exc_info=True)
                        extension = os.path.splitext(filename)[1]
                        if extension in GPU_KERNEL_BIN_EXTS.values():
                            # Each kernel has bunch of files like .cubin(for cuda), .spv(for xpu), .json, .ttir
                            # Just append one of them without the extension
                            kernel_names.append(Path(filename).stem)
                    if artifacts:
                        result.append(
                            TritonKernelArtifacts(
                                entry.kernel_hash,
                                entry.device,
                                artifacts,
                            )
                        )
                if config.use_static_cuda_launcher:
                    static_autotuners, static_kernel_names = (
                        cls.collect_static_autotuners()
                    )
                else:
                    static_autotuners = []
                    static_kernel_names = []
                cls.end_compile()
                return TritonBundle(result, static_autotuners), TritonBundlerMetadata(
                    kernel_names, static_kernel_names
                )
            return TritonBundle([], []), None

    @staticmethod
    def read_and_emit(bundle: TritonBundle) -> Optional[TritonBundlerMetadata]:
        """
        This is the main function called when a cache read happens. This function
        converts the bundled format back into individual files and writes them
        to the filesystem.

        NOTE: When we are writing to the filesystem, we assume exclusive access
        to the target directory.
        This means that if the target folder already exists and is non-empty,
        we bail out.
        Exclusive access means that no other process should be writing to
        or reading from the target directory.
        """
        from torch._inductor import config

        if not TritonBundler.is_enabled():
            return None

        with dynamo_timed(
            key="TritonBundler.read_and_emit", log_pt2_compile_event=True
        ):
            kernel_names: list[str] = []

            for artifacts in bundle.kernel_artifacts:
                basedir = triton_cache_dir(artifacts.device)
                directory = os.path.join(basedir, artifacts.kernel_hash)

                if os.path.exists(directory) and len(os.listdir(directory)) != 0:
                    # If directory already exists, we bail out and leave
                    # local disk to take care of caching
                    log.debug(
                        "Bailing out TritonBundler.read_and_emit, %s is non empty",
                        directory,
                    )
                    continue

                Path(basedir).mkdir(parents=True, exist_ok=True)

                # Random ID to avoid any collisions
                rnd_id = str(uuid.uuid4())
                tmp_dir = os.path.join(basedir, f"tmp.{rnd_id}")
                os.makedirs(tmp_dir)

                for artifact in artifacts.artifacts:
                    filepath = os.path.join(tmp_dir, artifact.filename)
                    with open(filepath, "wb") as file:
                        payload = artifact.payload
                        if artifact.filename.endswith(".json"):
                            payload = payload.replace(
                                TritonBundler._REPLACE_BYTES, str.encode(directory)
                            )
                        file.write(payload)
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"] += 1
                    extension = os.path.splitext(artifact.filename)[1]
                    if extension in GPU_KERNEL_BIN_EXTS.values():
                        # Each kernel has bunch of files like .cubin(for cuda), spv(for xpu), .json, .ttir
                        # Just append one of them without the extension
                        kernel_names.append(Path(artifact.filename).stem)

                if _IS_WINDOWS:
                    with FileLock(directory + ".lock"):
                        if os.path.exists(directory):
                            shutil.rmtree(directory)
                        os.replace(tmp_dir, directory)
                else:
                    # Atomic on POSIX systems
                    try:
                        os.replace(tmp_dir, directory)
                    except OSError:
                        log.warning("Directory %s is not empty - skipping!", tmp_dir)

            if config.use_static_cuda_launcher:
                static_kernel_names = TritonBundler.load_autotuners(
                    bundle.static_autotuners
                )
            else:
                static_kernel_names = []
            return TritonBundlerMetadata(kernel_names, static_kernel_names)

```



## High-Level Overview

"""    When we have compiled a triton kernel, we take note of that kernel by    its triton generated hash, its device, and where this kernel is located.    This is the minimum information we can use to later retrieve this kernel    from file system.

This Python file contains 7 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TritonBundleEntry`, `TritonKernelArtifact`, `StaticallyLaunchedAutotuner`, `TritonKernelArtifacts`, `TritonBundlerMetadata`, `TritonBundle`, `TritonBundler`

**Functions defined**: `is_enabled`, `begin_compile`, `end_compile`, `put`, `put_static_autotuner`, `collect_static_autotuners`, `load_autotuners`, `collect`, `read_and_emit`

**Key imports**: copy, dataclasses, logging, os, shutil, uuid, Path, Optional, counters, dynamo_timed, set_feature_use, justknobs_check


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `dataclasses`
- `logging`
- `os`
- `shutil`
- `uuid`
- `pathlib`: Path
- `typing`: Optional
- `torch._dynamo.utils`: counters, dynamo_timed, set_feature_use
- `torch._utils_internal`: justknobs_check
- `torch.utils._filelock`: FileLock
- `.runtime.runtime_utils`: triton_cache_dir
- `.utils`: _IS_WINDOWS, GPU_KERNEL_BIN_EXTS
- `torch._inductor`: config
- `torch._inductor.async_compile`: CompiledTritonKernels
- `torch._inductor.codecache`: StaticAutotunerFuture


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

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

- **File Documentation**: `triton_bundler.py_docs.md`
- **Keyword Index**: `triton_bundler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `triton_bundler.py_docs.md_docs.md`
- **Keyword Index**: `triton_bundler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
