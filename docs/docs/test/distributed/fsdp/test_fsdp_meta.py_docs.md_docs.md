# Documentation: `docs/test/distributed/fsdp/test_fsdp_meta.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_fsdp_meta.py_docs.md`
- **Size**: 18,646 bytes (18.21 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_fsdp_meta.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_meta.py`
- **Size**: 14,777 bytes (14.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import itertools
import sys
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import (
    always_wrap_policy as always_wrap,
    enable_wrap,
    ModuleWrapPolicy,
    wrap,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init
except ImportError:
    _TORCHDISTX_AVAIL = False


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def _reset_params_if_meta(is_meta: bool, model: nn.Module):
    # For torchdistX init, we don't need to call reset_params, as
    # deferred_init(model).materialize() is equivalent to model().
    if is_meta:
        for module in model.modules():
            # Assume that a module has `reset_parameters()` iff it has directly
            # managed parameters or buffers
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class MyLinear(nn.Linear):
    """
    Linear layer with deterministic reset_parameters for testing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self, *args, **kwargs):
        torch.manual_seed(42)
        with torch.no_grad():
            # Use an initialization method that depends on shape
            torch.nn.init.xavier_uniform_(self.weight, 1.0)


class MyBuffer(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.buf = torch.nn.Buffer(torch.empty((3, 3), device=device))

    def reset_parameters(self, *args, **kwargs):
        torch.manual_seed(42)
        # Use an initialization method that depends on shape
        torch.nn.init.xavier_uniform_(self.buf, 0.5)


class MyModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.lin1 = MyLinear(2, 2, bias=False, device=device)
        self.lin2 = MyLinear(2, 2, bias=False, device=device)
        self.buf_mod = MyBuffer(device)

    def forward(self, x):
        return self.lin2(self.lin1(x))


class NestedModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lin1 = MyLinear(2, 2, bias=False, device=device)
        self.lin1 = wrap(self.lin1)
        self.lin2 = MyLinear(2, 2, bias=False, device=device)
        self.l3 = MyModel(device=device)
        self.l3 = wrap(self.l3)

    def forward(self, x):
        return self.l3(self.lin2(self.lin1(x)))


def _init_with_reset_params(module: nn.Module):
    """
    to_empty + reset_parameters() init function example for modules
    initialized with device="meta"
    """
    has_meta_states = any(
        t.is_meta
        for t in itertools.chain(
            module.parameters(recurse=False), module.buffers(recurse=False)
        )
    )
    if has_meta_states:
        device = torch.device(device_type, torch.accelerator.current_device_index())
        module.to_empty(device=device, recurse=False)
        module.reset_parameters()


def _init_with_torchdistX(module: nn.Module):
    """
    torchdistX-based deferred module initialization function example
    using ``materialize_module``.
    """
    assert _TORCHDISTX_AVAIL

    def check_fn(k):
        return not isinstance(k, FSDP)

    deferred_init.materialize_module(module, check_fn=check_fn)


class TestFSDPWithMetaDevice(FSDPTest):
    @property
    def world_size(self):
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    def _compare_fsdp(self, fsdp1, fsdp2):
        with FSDP.summon_full_params(fsdp1):
            with FSDP.summon_full_params(fsdp2):
                for p1, p2 in zip(fsdp1.parameters(), fsdp2.parameters()):
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    def _test_simple_model_with_meta_device(self, meta_module_fn, init_fn=None):
        # Create model on meta device and wrap with FSDP.
        model = meta_module_fn()
        is_meta = next(model.parameters()).is_meta
        fsdp_meta = FSDP(
            model,
            auto_wrap_policy=always_wrap,
            param_init_fn=init_fn,
        )

        meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)

        # Test to make sure it is the same model parameters as regular FSDP
        # approach.
        regular = MyModel(device=device_type)
        _reset_params_if_meta(is_meta, regular)
        fsdp_regular = FSDP(regular, auto_wrap_policy=always_wrap)
        regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)

        self._compare_fsdp(fsdp_meta, fsdp_regular)
        inp = torch.randn(10, 2, device=device_type)
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        meta_opt.step()
        regular_opt.step()
        self._compare_fsdp(fsdp_meta, fsdp_regular)

        # Test that meta init works if all submodules are contained in only a
        # single FSDP unit.
        model = meta_module_fn()
        fsdp_meta = FSDP(model, param_init_fn=init_fn)
        meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)
        regular = MyModel(device=device_type)
        _reset_params_if_meta(is_meta, regular)
        fsdp_regular = FSDP(regular, auto_wrap_policy=always_wrap)
        regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)

        # Run a forward + backward pass + optimizer step
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        meta_opt.step()
        regular_opt.step()
        self._compare_fsdp(fsdp_meta, fsdp_regular)

    @skip_if_lt_x_gpu(2)
    def test_simple_model_with_meta_device_reset_params(self):
        def meta_module_fn():
            return MyModel(device="meta")

        self._test_simple_model_with_meta_device(
            meta_module_fn, _init_with_reset_params
        )

    @skip_if_lt_x_gpu(2)
    def test_simple_model_with_meta_device_default_init(self):
        def meta_module_fn():
            return MyModel(device="meta")

        self._test_simple_model_with_meta_device(meta_module_fn)

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    def test_simple_model_with_torchdistX_default_init(self):
        def meta_module_fn():
            return deferred_init.deferred_init(MyModel, device=device_type)

        self._test_simple_model_with_meta_device(meta_module_fn)

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    def test_simple_model_with_torchdistX_init_fn(self):
        def meta_module_fn():
            return deferred_init.deferred_init(MyModel, device=device_type)

        self._test_simple_model_with_meta_device(
            meta_module_fn, init_fn=_init_with_torchdistX
        )

    def _test_nested_model_with_meta_device(
        self, auto_wrap, meta_module_fn, init_fn=None
    ):
        if auto_wrap:
            module = meta_module_fn()
            is_meta = (
                next(module.parameters()).is_meta or next(module.buffers()).is_meta
            )
            fsdp_meta = FSDP(
                module,
                auto_wrap_policy=always_wrap,
                param_init_fn=init_fn,
            )
            meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)
            module_regular = NestedModel(device=device_type)
            _reset_params_if_meta(is_meta, module_regular)
            fsdp_regular = FSDP(
                module_regular,
                auto_wrap_policy=always_wrap,
            )
            regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)
        else:
            with enable_wrap(
                wrapper_cls=FSDP,
                param_init_fn=init_fn,
            ):
                module = meta_module_fn()
                is_meta = next(module.parameters()).is_meta
                # Non FSDP modules will still be initialized because they bubble up
                # to be part of a larger FSDP unit.
                fsdp_meta = wrap(module)
                meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)

            # Init and reset parameters before wrapping so that reset_params
            # matches up with meta device's initialization.
            module_regular = NestedModel(device=device_type)
            _reset_params_if_meta(is_meta, module_regular)
            with enable_wrap(wrapper_cls=FSDP):
                module_regular.lin1 = wrap(module_regular.lin1)
                module_regular.l3 = wrap(module_regular.l3)
                fsdp_regular = wrap(module_regular)
                regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)

        # Compare it before training
        self._compare_fsdp(fsdp_meta, fsdp_regular)
        inp = torch.randn(10, 2, device=device_type)
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        meta_opt.step()
        regular_opt.step()
        self._compare_fsdp(fsdp_meta, fsdp_regular)

    @skip_if_lt_x_gpu(2)
    @parametrize("auto_wrap", [True, False])
    def test_nested_model_with_meta_device_reset_params(self, auto_wrap):
        def meta_module_fn():
            return NestedModel(device="meta")

        self._test_nested_model_with_meta_device(
            auto_wrap=auto_wrap,
            meta_module_fn=meta_module_fn,
            init_fn=_init_with_reset_params,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("auto_wrap", [True, False])
    def test_nested_model_with_meta_device_default_init(self, auto_wrap):
        def meta_module_fn():
            return NestedModel(device="meta")

        self._test_nested_model_with_meta_device(
            auto_wrap=auto_wrap,
            meta_module_fn=meta_module_fn,
        )

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    @parametrize("auto_wrap", [True, False])
    def test_nested_model_with_torchdistX_default_init(self, auto_wrap):
        def meta_module_fn():
            return deferred_init.deferred_init(NestedModel, device=device_type)

        self._test_nested_model_with_meta_device(
            auto_wrap=auto_wrap, meta_module_fn=meta_module_fn
        )

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    @parametrize("auto_wrap", [True, False])
    def test_nested_model_with_torchdistX_init_fn(self, auto_wrap):
        def meta_module_fn():
            return deferred_init.deferred_init(NestedModel, device=device_type)

        self._test_nested_model_with_meta_device(
            auto_wrap=auto_wrap,
            meta_module_fn=meta_module_fn,
            init_fn=_init_with_torchdistX,
        )

    def _test_bad_arg(self, meta_module_fn):
        mod = meta_module_fn()
        with self.assertRaisesRegex(ValueError, "to be callable"):
            FSDP(mod, param_init_fn=42)

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(
        not _TORCHDISTX_AVAIL,
        "Test requires torchdistX: https://github.com/pytorch/torchdistX",
    )
    def test_bad_arg_torchdistx(self):
        def meta_module_fn():
            return deferred_init.deferred_init(NestedModel, device_type)

        self._test_bad_arg(meta_module_fn)

    @skip_if_lt_x_gpu(2)
    def test_bad_arg_meta(self):
        def meta_module_fn():
            return NestedModel(device="meta")

        self._test_bad_arg(meta_module_fn)

    @skip_if_lt_x_gpu(2)
    def test_meta_device_with_mixed_precision(self):
        """
        Tests meta device initialization with a ``param_init_fn`` when
        specifying mixed precision with ``param_dtype=torch.float32``.
        """

        class FakeLinear(nn.Module):
            def __init__(
                self, in_dim: int, out_dim: int, device: Union[torch.device, str]
            ) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn((in_dim, out_dim), device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(5, 5, device="meta")
                self.lin2 = FakeLinear(5, 5, device="meta")
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin2(self.relu(self.lin1(x)))

            def _module_init_fn(self, module: nn.Module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

        def _param_init_fn(module: nn.Module) -> None:
            # TODO: `module.to_empty()` is not generally correct for meta
            # device initialization.
            # https://github.com/pytorch/pytorch/issues/90465
            module.to_empty(device=torch.device(device_type))
            module.apply(model._module_init_fn)

        model = Model()
        # Wrap `lin1` and the top level `model` to create nested FSDP instances
        # where each instance has parameters
        FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
            mixed_precision=MixedPrecision(
                param_dtype=torch.float32, reduce_dtype=torch.float16
            ),
            param_init_fn=_param_init_fn,
            device_id=torch.accelerator.current_device_index(),
        )


instantiate_parametrized_tests(TestFSDPWithMetaDevice)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 7 class(es) and 45 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyLinear`, `MyBuffer`, `MyModel`, `NestedModel`, `TestFSDPWithMetaDevice`, `FakeLinear`, `Model`

**Functions defined**: `_reset_params_if_meta`, `__init__`, `reset_parameters`, `__init__`, `reset_parameters`, `__init__`, `forward`, `__init__`, `forward`, `_init_with_reset_params`, `_init_with_torchdistX`, `check_fn`, `world_size`, `process_group`, `_compare_fsdp`, `_test_simple_model_with_meta_device`, `test_simple_model_with_meta_device_reset_params`, `meta_module_fn`, `test_simple_model_with_meta_device_default_init`, `meta_module_fn`

**Key imports**: itertools, sys, Union, torch, torch.distributed as dist, torch.nn as nn, FullyShardedDataParallel as FSDP, MixedPrecision, skip_if_lt_x_gpu, FSDPTest, deferred_init


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `sys`
- `typing`: Union
- `torch`
- `torch.distributed as dist`
- `torch.nn as nn`
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP, MixedPrecision
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_fsdp`: FSDPTest
- `torchdistx`: deferred_init


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/fsdp/test_fsdp_meta.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/fsdp`):

- [`test_fsdp_memory.py_docs.md`](./test_fsdp_memory.py_docs.md)
- [`test_fsdp_mixed_precision.py_docs.md`](./test_fsdp_mixed_precision.py_docs.md)
- [`test_fsdp_uneven.py_docs.md`](./test_fsdp_uneven.py_docs.md)
- [`test_fsdp_dtensor_state_dict.py_docs.md`](./test_fsdp_dtensor_state_dict.py_docs.md)
- [`test_fsdp_tp_integration.py_docs.md`](./test_fsdp_tp_integration.py_docs.md)
- [`test_distributed_checkpoint.py_docs.md`](./test_distributed_checkpoint.py_docs.md)
- [`test_fsdp_multiple_forward.py_docs.md`](./test_fsdp_multiple_forward.py_docs.md)
- [`test_checkpoint_wrapper.py_docs.md`](./test_checkpoint_wrapper.py_docs.md)
- [`test_fsdp_clip_grad_norm.py_docs.md`](./test_fsdp_clip_grad_norm.py_docs.md)
- [`test_fsdp_use_orig_params.py_docs.md`](./test_fsdp_use_orig_params.py_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_meta.py_docs.md`
- **Keyword Index**: `test_fsdp_meta.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/fsdp/test_fsdp_meta.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/fsdp`):

- [`test_fsdp_grad_acc.py_docs.md_docs.md`](./test_fsdp_grad_acc.py_docs.md_docs.md)
- [`test_fsdp_ignored_modules.py_kw.md_docs.md`](./test_fsdp_ignored_modules.py_kw.md_docs.md)
- [`test_fsdp_meta.py_kw.md_docs.md`](./test_fsdp_meta.py_kw.md_docs.md)
- [`test_fsdp_apply.py_docs.md_docs.md`](./test_fsdp_apply.py_docs.md_docs.md)
- [`test_fsdp_tp_integration.py_kw.md_docs.md`](./test_fsdp_tp_integration.py_kw.md_docs.md)
- [`test_fsdp_fx.py_docs.md_docs.md`](./test_fsdp_fx.py_docs.md_docs.md)
- [`test_fsdp_memory.py_kw.md_docs.md`](./test_fsdp_memory.py_kw.md_docs.md)
- [`test_fsdp_apply.py_kw.md_docs.md`](./test_fsdp_apply.py_kw.md_docs.md)
- [`test_fsdp_tp_integration.py_docs.md_docs.md`](./test_fsdp_tp_integration.py_docs.md_docs.md)
- [`test_fsdp_multiple_forward.py_kw.md_docs.md`](./test_fsdp_multiple_forward.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_meta.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_meta.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
