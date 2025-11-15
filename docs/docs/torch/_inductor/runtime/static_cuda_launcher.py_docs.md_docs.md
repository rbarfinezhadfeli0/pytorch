# Documentation: `docs/torch/_inductor/runtime/static_cuda_launcher.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/runtime/static_cuda_launcher.py_docs.md`
- **Size**: 17,446 bytes (17.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/runtime/static_cuda_launcher.py`

## File Metadata

- **Path**: `torch/_inductor/runtime/static_cuda_launcher.py`
- **Size**: 13,041 bytes (12.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import functools
import os
from typing import Any
from typing_extensions import Unpack

from .triton_compat import ASTSource, CompiledKernel, knobs as triton_knobs
from .triton_helpers import get_constexprs


class StaticallyLaunchedCudaKernel:
    """
    Parses the metadata of a CompiledKernel from Triton into a structure that can
    launch the cuda kernel directly. Only works for triton kernels compiled to cubin.

    Doing this avoids C++ codegen and compilation during compile, since we can use a
    statically compiled library to launch the kernel. To avoid mallocing for the arguments,
    we have a launcher for different numbers of arguments up to a max. StaticCudaLauncher
    only supports # of arguments up until 10 for now.

    Workflow:
    Compile time:
    1. Compile a kernel with triton and get a CompiledKernel
    2. Instantiate kernel = StaticallyLaunchedCudaKernel(triton_kernel)
    3. Write to a cubin file: kernel.write_cubin_to_file(filepath)
    4. Call kernel.load_kernel() (CUDA should be initialized by this point) to load the cubin
    Runtime:
    5. Call kernel.run(grid, stream, args) to launch the kernel

    Note that after step 3, StaticallyLaunchedCudaKernel is fully pickleable/serializable.
    This allows it to be cached by FXGraphCache/TritonBundler, as well as sent from the worker
    to the parent process in inductor.

    There are two main versions of triton that we wish to support: 3.3 and 3.2. Triton makes considerable changes
    to how it handles constants in 3.3, so there's some special logic necessary to handle both versions.
    """

    def __init__(self, kernel: CompiledKernel) -> None:
        # pyrefly: ignore [missing-attribute]
        self.name = kernel.src.fn.__name__
        # pyrefly: ignore [missing-attribute]
        if "hsaco" in kernel.asm:
            # pyrefly: ignore [missing-attribute]
            self.cubin_raw = kernel.asm["hsaco"]
            self.is_rocm = True
        # pyrefly: ignore [missing-attribute]
        elif "cubin" in kernel.asm:
            # pyrefly: ignore [missing-attribute]
            self.cubin_raw = kernel.asm["cubin"]
            self.is_rocm = False
        else:
            raise RuntimeError(
                "Expected either 'hsaco' (ROCm) or 'cubin' (CUDA) in kernel.asm"
            )

        # pyrefly: ignore [missing-attribute]
        self.cubin_path = kernel._cubin_path

        # Used by torch.compile to filter constants in older triton versions
        # pyrefly: ignore [missing-attribute]
        self.arg_names = kernel.src.fn.arg_names

        # Const exprs that are declared by the triton kernel directly
        # Used to generate the kernel launcher's def args
        # pyrefly: ignore [missing-attribute]
        self.declared_constexprs = get_constexprs(kernel.src.fn)

        # pyrefly: ignore [missing-attribute]
        self.hash = kernel.hash

        if triton_knobs is None:
            # pyrefly: ignore [missing-attribute]
            launch_enter = kernel.__class__.launch_enter_hook
            # pyrefly: ignore [missing-attribute]
            launch_exit = kernel.__class__.launch_exit_hook
        else:
            launch_enter = triton_knobs.runtime.launch_enter_hook
            launch_exit = triton_knobs.runtime.launch_exit_hook

        def hook_is_empty(hook: Any) -> bool:
            if hook is None:
                return True
            if (
                triton_knobs
                and (HookChain := getattr(triton_knobs, "HookChain", None)) is not None
                and isinstance(hook, HookChain)
            ):
                # Support hooks after https://github.com/triton-lang/triton/pull/7866
                return len(hook.calls) == 0
            return False

        if not hook_is_empty(launch_enter) or not hook_is_empty(launch_exit):
            raise NotImplementedError(
                "We don't support launch enter or launch exit hooks"
            )
        # pyrefly: ignore [missing-attribute]
        self.num_warps = kernel.metadata.num_warps
        self.shared = (
            # pyrefly: ignore [missing-attribute]
            kernel.shared if hasattr(kernel, "shared") else kernel.metadata.shared
        )

        def needs_scratch_arg(scratch_name: str, param_name: str) -> bool:
            # pyrefly: ignore [missing-attribute]
            if hasattr(kernel.metadata, param_name):
                if getattr(kernel.metadata, param_name) > 0:
                    raise NotImplementedError(
                        f"{scratch_name} scratch not yet supported"
                    )
                return True
            return False

        # Newer triton versions pass an extra global scratch parameter to the compiled cuda kernel.
        # Inductor never uses this field or enables it, but we still have to pass
        # an extra None into the set of params if its enabled
        self.has_global_scratch = needs_scratch_arg("Global", "global_scratch_size")
        # same situation for profile scratch - triton-lang/triton#7258
        self.has_profile_scratch = needs_scratch_arg("Profile", "profile_scratch_size")

        # pyrefly: ignore [missing-attribute]
        self.arg_tys = self.arg_ty_from_signature(kernel.src)
        self.function: int | None = None  # Loaded by load_kernel(on the parent process)
        num_ctas = 1
        if hasattr(kernel, "num_ctas"):
            num_ctas = kernel.num_ctas
        elif hasattr(kernel, "metadata"):
            num_ctas = kernel.metadata.num_ctas

        if num_ctas != 1:
            raise NotImplementedError(
                "Static cuda launcher only supports num_ctas == 1"
            )

    def reload_cubin_from_raw(self, filepath: str) -> str:
        """
        If the cubin file triton generated gets deleted under us, we can
        reload it from the raw cubin file.
        """
        if self.cubin_path is None:
            assert self.cubin_raw is not None
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(self.cubin_raw)
                self.cubin_path = filepath
        return self.cubin_path

    def load_kernel(self, device: int) -> None:
        from torch._C import _StaticCudaLauncher

        if self.function is not None:
            return

        assert hasattr(self, "cubin_path")
        assert self.cubin_path is not None
        (self.function, self.n_regs, self.n_spills) = _StaticCudaLauncher._load_kernel(
            self.cubin_path, self.name, self.shared, device
        )
        # Don't need the cubin path anymore now that we've loaded
        self.cubin_path = None
        self.cubin_raw = None

    @staticmethod
    @functools.lru_cache
    def type_mappings() -> dict[str, str]:
        return {
            "i1": "i",
            "i8": "b",
            "i16": "h",
            "i32": "i",
            "i64": "l",
            "u1": "I",
            "u8": "B",
            "u16": "H",
            "u32": "I",
            "u64": "K",
            "fp16": "f",
            "bf16": "f",
            "fp32": "f",
            "f32": "f",
            "fp64": "d",
            # TODO handle nvTmaDesc/CUtensormap
        }

    def extract_type(self, ty: str) -> str:
        """
        Takes a triton type from CompiledKernel.signature and
        converts it into a single char encoding. _StaticCudaLauncher
        will switch on this char to figure out what type the underlying
        value should be passed to the triton kernel as.
        """
        if ty[0] == "*":
            return "O"
        elif ty == "nvTmaDesc":
            raise NotImplementedError("nvTmaDesc kernels are not yet supported")
        return StaticallyLaunchedCudaKernel.type_mappings()[ty]

    def arg_ty_from_signature(self, src: ASTSource) -> str:
        def index_key(i: Any) -> int:
            if isinstance(i, str):
                # pyrefly: ignore [missing-attribute]
                return src.fn.arg_names.index(i)
            elif isinstance(i, tuple):
                # In triton 3.3, src.fn.constants has tuples as a key
                return i[0]
            else:
                return i

        # pyrefly: ignore [missing-attribute]
        signature = {index_key(key): value for key, value in src.signature.items()}
        # Triton uses these as the main way to filter out constants passed to their cubin
        constants = [index_key(key) for key in getattr(src, "constants", dict())]
        # This value is always a superset of kernel.fn.constexprs: kernel.fn.constexprs are
        # constants declared by the triton kernel directly, whereas this list can have
        # constants that are unused by the triton kernel that triton figured out during
        # compilation.
        self.full_constexprs = constants
        # Despite requiring them to be passed in, the triton CUDA launcher
        # completely ignores the constexprs passed into it when generating code.
        # So we can ignore them here too
        params = []

        for i in sorted(signature.keys()):
            ty = signature[i]
            # In newer triton versions, constants are passed in to signature with type `constexpr`
            # In older triton versions, there can be constants in src.constants that are not `constexpr` in signature
            # so we check both here
            if ty == "constexpr" or i in constants:
                pass
            else:
                # pyrefly: ignore [bad-argument-type]
                params.append(self.extract_type(ty))
        return "".join(params)

    def __getstate__(self) -> dict[str, Any]:
        # Remove objects that are no longer valid for pickling
        state = self.__dict__.copy()
        state["function"] = None
        # Cubin paths aren't consistent across processes, so we clear
        # and reload them.
        state["cubin_path"] = None
        return state

    def run(
        self,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        stream: int,
        *args: Unpack[tuple[object, ...]],
    ) -> None:
        """Actually run the kernel at runtime. This function is the hot codepath."""
        from torch._C import _StaticCudaLauncher

        # Assert load_kernel() has been called and args match
        assert self.function is not None

        # TODO: actually, if the args *don't* match, we probably should
        # throw an exception. But if inductor is the only one calling this
        # thing, it should always match.
        # Get rid of constants before passing to cubin launcher

        arg_tys = self.arg_tys

        if self.is_rocm:
            # ROCm/HIP kernel ABI: The Triton HIP backend ALWAYS includes both
            # global_scratch and profile_scratch parameters in the kernel signature,
            # even when the kernel doesn't use them (i.e., when has_*_scratch is False).
            #
            # This differs fundamentally from CUDA, where these parameters are only
            # present in the signature if the corresponding has_*_scratch flag is True.
            #
            # The flags indicate whether memory will be allocated/used:
            # - has_global_scratch: Whether global scratch workspace is needed
            # - has_profile_scratch: Whether profiling instrumentation is enabled
            #
            # However, regardless of flag values, we MUST always pass both parameters
            # to match the HIP kernel ABI. Passing None is safe:
            #
            # - If scratch is not needed (has_*_scratch=False or scratch_size=0):
            #   The None becomes nullptr, which the kernel never dereferences
            #
            # - If scratch is needed (has_*_scratch=True and scratch_size>0):
            #   The None becomes nullptr initially, but the HIP runtime intercepts
            #   the kernel launch, allocates the required scratch memory based on
            #   kernel metadata, and replaces the nullptr with a valid pointer before
            #   the kernel actually executes
            #
            # Not passing both parameters causes segmentation faults because the kernel
            # expects them at specific positions in the argument array.
            arg_tys = arg_tys + "OO"
            args = (*args, None, None)

        else:
            for has_scratch in [self.has_global_scratch, self.has_profile_scratch]:
                if has_scratch:
                    arg_tys = arg_tys + "O"
                    args = (*args, None)
        # pyrefly: ignore [bad-argument-type]
        assert len(args) == len(arg_tys)

        # TODO: can handle grid functions here or in C++, so
        # that we don't need the grid handler above.
        _StaticCudaLauncher._launch_kernel(
            self.function,
            grid_x,
            grid_y,
            grid_z,
            self.num_warps,
            self.shared,
            arg_tys,
            # pyrefly: ignore [bad-argument-type]
            args,
            stream,
        )

```



## High-Level Overview

"""    Parses the metadata of a CompiledKernel from Triton into a structure that can    launch the cuda kernel directly. Only works for triton kernels compiled to cubin.    Doing this avoids C++ codegen and compilation during compile, since we can use a    statically compiled library to launch the kernel. To avoid mallocing for the arguments,    we have a launcher for different numbers of arguments up to a max. StaticCudaLauncher    only supports # of arguments up until 10 for now.    Workflow:    Compile time:    1. Compile a kernel with triton and get a CompiledKernel    2. Instantiate kernel = StaticallyLaunchedCudaKernel(triton_kernel)    3. Write to a cubin file: kernel.write_cubin_to_file(filepath)    4. Call kernel.load_kernel() (CUDA should be initialized by this point) to load the cubin    Runtime:    5. Call kernel.run(grid, stream, args) to launch the kernel    Note that after step 3, StaticallyLaunchedCudaKernel is fully pickleable/serializable.    This allows it to be cached by FXGraphCache/TritonBundler, as well as sent from the worker    to the parent process in inductor.    There are two main versions of triton that we wish to support: 3.3 and 3.2. Triton makes considerable changes    to how it handles constants in 3.3, so there's some special logic necessary to handle both versions.

This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StaticallyLaunchedCudaKernel`

**Functions defined**: `__init__`, `hook_is_empty`, `needs_scratch_arg`, `reload_cubin_from_raw`, `load_kernel`, `type_mappings`, `extract_type`, `arg_ty_from_signature`, `index_key`, `__getstate__`, `run`

**Key imports**: functools, os, Any, Unpack, ASTSource, CompiledKernel, knobs as triton_knobs, get_constexprs, _StaticCudaLauncher, _StaticCudaLauncher


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `os`
- `typing`: Any
- `typing_extensions`: Unpack
- `.triton_compat`: ASTSource, CompiledKernel, knobs as triton_knobs
- `.triton_helpers`: get_constexprs
- `torch._C`: _StaticCudaLauncher


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/_inductor/runtime`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`hints.py_docs.md`](./hints.py_docs.md)
- [`coordinate_descent_tuner.py_docs.md`](./coordinate_descent_tuner.py_docs.md)
- [`autotune_cache.py_docs.md`](./autotune_cache.py_docs.md)
- [`triton_heuristics.py_docs.md`](./triton_heuristics.py_docs.md)
- [`debug_utils.py_docs.md`](./debug_utils.py_docs.md)
- [`compile_tasks.py_docs.md`](./compile_tasks.py_docs.md)
- [`triton_compat.py_docs.md`](./triton_compat.py_docs.md)
- [`cache_dir_utils.py_docs.md`](./cache_dir_utils.py_docs.md)


## Cross-References

- **File Documentation**: `static_cuda_launcher.py_docs.md`
- **Keyword Index**: `static_cuda_launcher.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/_inductor/runtime`):

- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`hints.py_kw.md_docs.md`](./hints.py_kw.md_docs.md)
- [`cache_dir_utils.py_kw.md_docs.md`](./cache_dir_utils.py_kw.md_docs.md)
- [`cache_dir_utils.py_docs.md_docs.md`](./cache_dir_utils.py_docs.md_docs.md)
- [`halide_helpers.py_docs.md_docs.md`](./halide_helpers.py_docs.md_docs.md)
- [`debug_utils.py_docs.md_docs.md`](./debug_utils.py_docs.md_docs.md)
- [`runtime_utils.py_kw.md_docs.md`](./runtime_utils.py_kw.md_docs.md)
- [`static_cuda_launcher.py_kw.md_docs.md`](./static_cuda_launcher.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `static_cuda_launcher.py_docs.md_docs.md`
- **Keyword Index**: `static_cuda_launcher.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
