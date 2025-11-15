# Documentation: `torch/_dynamo/backends/tvm.py`

## File Metadata

- **Path**: `torch/_dynamo/backends/tvm.py`
- **Size**: 8,283 bytes (8.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
This module provides TVM backend integration for TorchDynamo.

Apache TVM is a deep learning compiler framework that can optimize and execute
models on various hardware backends. This module enables:

- Compilation of PyTorch models to TVM's computation graphs
- Multiple scheduling options:
  - Default scheduler
  - Auto-scheduler for automatic optimization
  - Meta-schedule for evolutionary search-based tuning
- Hardware-specific optimizations:
  - CUDA GPU support
  - CPU support with LLVM targeting and architecture-specific tuning
  - Automatic detection of CPU capabilities (AVX2, AVX512)
- Tensor conversion utilities between PyTorch and TVM formats
- Configurable optimization levels and tuning trials

The backend can be used with torch.compile():
    model = torch.compile(model, backend="tvm")
"""

import functools
import importlib
import logging
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import torch
from torch import fx

from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend


log = logging.getLogger(__name__)


@register_backend
@fake_tensor_unsupported  # type: ignore[arg-type]
def tvm(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: Optional[MappingProxyType[str, Any]] = None,
) -> Callable[..., Any]:
    if options is None:
        options = MappingProxyType({"scheduler": None, "trials": 20000, "opt_level": 3})
    assert options is not None
    import tvm  # type: ignore[import]
    from tvm import relay  # type: ignore[import]
    from tvm.contrib import graph_executor  # type: ignore[import]

    jit_mod = torch.jit.trace(gm, example_inputs)
    device = device_from_inputs(example_inputs)
    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    example_outputs = gm(*example_inputs)
    if len(example_outputs) == 0:
        log.warning("Explicitly fall back to eager due to zero output")
        return gm.forward
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)
    if device.type == "cuda":
        dev = tvm.cuda(device.index)
        target = tvm.target.cuda()
    else:
        dev = tvm.cpu(0)
        target = tvm.target.Target(llvm_target())

    scheduler = options.get("scheduler", None)
    if scheduler is None:
        scheduler = os.environ.get("TVM_SCHEDULER", None)

    trials = options.get("trials", 20000)
    opt_level = options.get("opt_level", 3)

    if scheduler == "auto_scheduler":
        # pyrefly: ignore [import-error]
        from tvm import auto_scheduler

        log_file = tempfile.NamedTemporaryFile()

        # pyrefly: ignore [bad-argument-type]
        if not os.path.exists(log_file):
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod["main"], params, target
            )
            if len(tasks) != 0:
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                # pyrefly: ignore [bad-argument-type]
                if not os.path.exists(log_file):
                    assert trials > 0
                    tune_option = auto_scheduler.TuningOptions(
                        num_measure_trials=trials,
                        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                        early_stopping=2000,
                    )
                    try:
                        tuner.tune(tune_option)
                    except Exception:
                        # pyrefly: ignore [bad-argument-type]
                        if os.path.exists(log_file):
                            # pyrefly: ignore [bad-argument-type]
                            os.unlink(log_file)
                        raise

        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=opt_level, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)
    elif scheduler == "meta_schedule":
        # pyrefly: ignore [import-error]
        from tvm import meta_schedule as ms

        with tempfile.TemporaryDirectory() as work_dir:
            if device.type != "cuda":
                # meta_schedule needs num-cores to be specified
                # here we use the maximum core count
                target = tvm.target.Target(
                    f"{llvm_target()} --num-cores {ms.utils.cpu_count(logical=False)}"
                )
            # TODO(shingjan): This could be replaced by tvm.contrib.torch.optimize_torch
            # once USE_PT_TVMDSOOP is updated and turned on by default in TVM.
            assert trials > 0
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=trials,
                num_trials_per_iter=64,
                params=params,
                strategy="evolutionary",
                opt_level=opt_level,
            )
            lib = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
                opt_level=opt_level,
            )
    elif scheduler == "default" or not scheduler:
        # no autotuning
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target=target, params=params)
    else:
        raise NotImplementedError(
            "This tuning option is invalid/not implemented for torchdynamo's TVM-related backend. "
            "There are three available options: default, auto_scheduler and meta_schedule."
        )
    m = graph_executor.GraphModule(lib["default"](dev))

    def to_torch_tensor(nd_tensor: tvm.nd.array) -> torch.Tensor:
        """A helper function to transfer a NDArray to torch.tensor."""
        if nd_tensor.dtype == "bool":
            # DLPack does not support boolean so it can't be handled by
            # torch.utils.dlpack.from_pack. Workaround by going through
            # numpy, although this brings additional data copy overhead.
            return torch.from_numpy(nd_tensor.numpy())
        return torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())

    def to_tvm_tensor(torch_tensor: torch.Tensor) -> tvm.nd.array:
        """A helper function to transfer a torch.tensor to NDArray."""
        if torch_tensor.dtype == torch.bool:
            # same reason as above, fallback to numpy conversion which
            # could introduce data copy overhead
            return tvm.nd.array(torch_tensor.cpu().numpy())
        return tvm.nd.from_dlpack(torch_tensor)

    def exec_tvm(*i_args: torch.Tensor) -> list[torch.Tensor]:
        args = [a.contiguous() for a in i_args]
        shape_info, _ = m.get_input_info()
        active_inputs = {name for name, _ in shape_info.items()}
        for idx, arg in enumerate(args, 0):
            if arg.dim() != 0:
                if arg.requires_grad:
                    arg = arg.detach()
                inp_name = f"inp_{idx}"
                if inp_name not in active_inputs:
                    log.warning(
                        "input %s skipped as not found in tvm's runtime library",
                        inp_name,
                    )
                    continue
                m.set_input(
                    inp_name,
                    to_tvm_tensor(arg),
                )
        m.run()
        return [to_torch_tensor(m.get_output(i)) for i in range(m.get_num_outputs())]

    return exec_tvm


tvm_meta_schedule = functools.partial(tvm, scheduler="meta_schedule")
tvm_auto_scheduler = functools.partial(tvm, scheduler="auto_scheduler")


def has_tvm() -> bool:
    try:
        importlib.import_module("tvm")
        return True
    except ImportError:
        return False


@functools.cache
def llvm_target() -> str:
    if sys.platform == "linux":
        cpuinfo = Path("/proc/cpuinfo").read_text()
        if "avx512" in cpuinfo:
            return "llvm -mcpu=skylake-avx512"
        elif "avx2" in cpuinfo:
            return "llvm -mcpu=core-avx2"
    return "llvm"

```



## High-Level Overview

"""This module provides TVM backend integration for TorchDynamo.Apache TVM is a deep learning compiler framework that can optimize and executemodels on various hardware backends. This module enables:- Compilation of PyTorch models to TVM's computation graphs- Multiple scheduling options:  - Default scheduler  - Auto-scheduler for automatic optimization  - Meta-schedule for evolutionary search-based tuning- Hardware-specific optimizations:  - CUDA GPU support  - CPU support with LLVM targeting and architecture-specific tuning  - Automatic detection of CPU capabilities (AVX2, AVX512)- Tensor conversion utilities between PyTorch and TVM formats- Configurable optimization levels and tuning trialsThe backend can be used with torch.compile():    model = torch.compile(model, backend="tvm")

This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `tvm`, `to_torch_tensor`, `to_tvm_tensor`, `exec_tvm`, `has_tvm`, `llvm_target`

**Key imports**: functools, importlib, logging, os, sys, tempfile, Callable, Path, MappingProxyType, Any, Optional


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/backends`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `importlib`
- `logging`
- `os`
- `sys`
- `tempfile`
- `collections.abc`: Callable
- `pathlib`: Path
- `types`: MappingProxyType
- `typing`: Any, Optional
- `torch`
- `.common`: device_from_inputs, fake_tensor_unsupported
- `.registry`: register_backend
- `tvm  `
- `tvm`: relay  
- `tvm.contrib`: graph_executor  


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_dynamo/backends`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`onnxrt.py_docs.md`](./onnxrt.py_docs.md)
- [`cudagraphs.py_docs.md`](./cudagraphs.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`debugging.py_docs.md`](./debugging.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`torchxla.py_docs.md`](./torchxla.py_docs.md)
- [`tensorrt.py_docs.md`](./tensorrt.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)


## Cross-References

- **File Documentation**: `tvm.py_docs.md`
- **Keyword Index**: `tvm.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
