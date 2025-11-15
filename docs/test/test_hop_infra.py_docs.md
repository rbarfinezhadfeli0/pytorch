# Documentation: `test/test_hop_infra.py`

## File Metadata

- **Path**: `test/test_hop_infra.py`
- **Size**: 3,124 bytes (3.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: higher order operators"]
import importlib
import pkgutil

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.testing._internal.hop_db import (
    FIXME_hop_that_doesnt_have_opinfo_test_allowlist,
    hop_db,
)


def do_imports():
    for mod in pkgutil.walk_packages(
        torch._higher_order_ops.__path__, "torch._higher_order_ops."
    ):
        modname = mod.name
        importlib.import_module(modname)


do_imports()


@skipIfTorchDynamo("not applicable")
class TestHOPInfra(TestCase):
    def test_all_hops_have_opinfo(self):
        """All HOPs should have an OpInfo in torch/testing/_internal/hop_db.py"""
        from torch._ops import _higher_order_ops

        hops_that_have_op_info = {k.name for k in hop_db}
        all_hops = _higher_order_ops.keys()

        missing_ops = set()

        for op in all_hops:
            if (
                op not in hops_that_have_op_info
                and op not in FIXME_hop_that_doesnt_have_opinfo_test_allowlist
            ):
                missing_ops.add(op)

        self.assertTrue(
            len(missing_ops) == 0,
            f"Missing hop_db OpInfo entries for {missing_ops}, please add them to torch/testing/_internal/hop_db.py",
        )

    def test_all_hops_are_imported(self):
        """All HOPs should be listed in torch._higher_order_ops.__all__

        Some constraints (see test_testing.py::TestImports)
        - Sympy must be lazily imported
        - Dynamo must be lazily imported
        """
        imported_hops = torch._higher_order_ops.__all__
        registered_hops = torch._ops._higher_order_ops.keys()

        # Please don't add anything here.
        # We want to ensure that all HOPs are imported at "import torch" time.
        # It is bad if someone tries to access torch.ops.higher_order.cond
        # and it doesn't exist (this may happen if your HOP isn't imported at
        # "import torch" time).
        FIXME_ALLOWLIST = {
            "autograd_function_apply",
            "run_with_rng_state",
            "graphsafe_run_with_rng_state",
            "map_impl",
            "_export_tracepoint",
            "run_and_save_rng_state",
            "map",
            "custom_function_call",
            "trace_wrapped",
            "triton_kernel_wrapper_functional",
            "triton_kernel_wrapper_mutation",
            "wrap",  # Really weird failure -- importing this causes Dynamo to choke on checkpoint
        }
        not_imported_hops = registered_hops - imported_hops
        not_imported_hops = not_imported_hops - FIXME_ALLOWLIST
        self.assertEqual(
            not_imported_hops,
            set(),
            msg="All HOPs must be listed under torch/_higher_order_ops/__init__.py's __all__.",
        )

    def test_imports_from_all_work(self):
        """All APIs listed in torch._higher_order_ops.__all__ must be importable"""
        stuff = torch._higher_order_ops.__all__
        for attr in stuff:
            getattr(torch._higher_order_ops, attr)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""All HOPs should have an OpInfo in torch/testing/_internal/hop_db.py"""        from torch._ops import _higher_order_ops        hops_that_have_op_info = {k.name for k in hop_db}        all_hops = _higher_order_ops.keys()        missing_ops = set()        for op in all_hops:            if (                op not in hops_that_have_op_info                and op not in FIXME_hop_that_doesnt_have_opinfo_test_allowlist            ):                missing_ops.add(op)        self.assertTrue(            len(missing_ops) == 0,            f"Missing hop_db OpInfo entries for {missing_ops}, please add them to torch/testing/_internal/hop_db.py",        )    def test_all_hops_are_imported(self):

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestHOPInfra`

**Functions defined**: `do_imports`, `test_all_hops_have_opinfo`, `test_all_hops_are_imported`, `test_imports_from_all_work`

**Key imports**: importlib, pkgutil, torch, run_tests, skipIfTorchDynamo, TestCase, _higher_order_ops, torch, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `importlib`
- `pkgutil`
- `torch`
- `torch.testing._internal.common_utils`: run_tests, skipIfTorchDynamo, TestCase
- `torch._ops`: _higher_order_ops


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_hop_infra.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_hop_infra.py_docs.md`
- **Keyword Index**: `test_hop_infra.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
