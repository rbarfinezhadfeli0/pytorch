# Documentation: `docs/test/inductor/test_fuzzer.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_fuzzer.py_docs.md`
- **Size**: 11,590 bytes (11.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_fuzzer.py`

## File Metadata

- **Path**: `test/inductor/test_fuzzer.py`
- **Size**: 7,681 bytes (7.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import unittest
from typing import Literal
from unittest.mock import MagicMock, patch

import torch
from torch._dynamo import config as dynamo_config
from torch._inductor import config as inductor_config
from torch._inductor.fuzzer import ConfigFuzzer, MODULE_DEFAULTS, SamplingMethod, Status
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal import fake_config_module as fake_config
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


def create_simple_test_model_cpu():
    def test_fn() -> bool:
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
        )

        x = torch.randn(32, 10)
        model(x)
        return True

    return test_fn


def create_simple_test_model_gpu():
    batch_size = 32
    seq_length = 50
    hidden_size = 768

    inp = torch.randn(batch_size, seq_length, hidden_size, device=GPU_TYPE)
    weight = torch.randn(hidden_size, hidden_size, device=GPU_TYPE)

    def test_fn() -> bool:
        matmul_output = inp @ weight
        torch.nn.LayerNorm(hidden_size, device=GPU_TYPE)(matmul_output)
        return True

    return test_fn


class TestConfigFuzzer(TestCase):
    def test_sampling_method_toggle(self):
        toggle = SamplingMethod.dispatch(SamplingMethod.TOGGLE)
        self.assertEqual(toggle("", bool, False), True)
        self.assertEqual(toggle("", bool, True), False)
        self.assertEqual(toggle("", Literal["foo", "bar"], "foo"), "bar")
        self.assertEqual(toggle("", Literal["foo", "bar"], "bar"), "foo")
        self.assertTrue("bar" in toggle("", list[Literal["foo", "bar"]], ["foo"]))
        self.assertTrue("foo" in toggle("", list[Literal["foo", "bar"]], ["bar"]))

    def test_sampling_method_random(self):
        random = SamplingMethod.dispatch(SamplingMethod.RANDOM)
        samp = [random("", bool, False) for i in range(1000)]
        self.assertTrue(not all(samp))

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_config_fuzzer_inductor_gpu(self):
        fuzzer = ConfigFuzzer(inductor_config, create_simple_test_model_gpu, seed=30)
        self.assertIsNotNone(fuzzer.default)
        fuzzer.reproduce([{"max_fusion_size": 1}])

    def test_config_fuzzer_inductor_cpu(self):
        fuzzer = ConfigFuzzer(inductor_config, create_simple_test_model_cpu, seed=100)
        self.assertIsNotNone(fuzzer.default)
        fuzzer.reproduce([{"max_fusion_size": 1}])

    def test_config_fuzzer_bisector_exception(self):
        key_1 = {"e_bool": False, "e_optional": None}

        class MyException(Exception):
            pass

        def create_key_1():
            def myfn():
                if not fake_config.e_bool and fake_config.e_optional is None:
                    raise MyException("hi")
                return True

            return myfn

        fuzzer = ConfigFuzzer(fake_config, create_key_1, seed=100, default={})
        results = fuzzer.bisect(num_attempts=2, p=1.0)
        self.assertEqual(len(results), 2)
        for res in results:
            self.assertEqual(res, key_1)

    def test_config_fuzzer_bisector_boolean(self):
        key_1 = {"e_bool": False, "e_optional": None}

        def create_key_1():
            def myfn():
                if not fake_config.e_bool and fake_config.e_optional is None:
                    return False
                return True

            return myfn

        fuzzer = ConfigFuzzer(fake_config, create_key_1, seed=100, default={})
        num_attempts = 2
        results = fuzzer.bisect(num_attempts=num_attempts, p=1.0)
        self.assertEqual(len(results), num_attempts)
        for res in results:
            self.assertEqual(res, key_1)

    def test_config_fuzzer_n_tuple(self):
        key_1 = {"e_bool": False, "e_optional": None}

        def create_key_1():
            def myfn():
                if not fake_config.e_bool and fake_config.e_optional is None:
                    return False
                return True

            return myfn

        fuzzer = ConfigFuzzer(fake_config, create_key_1, seed=100, default={})
        max_combo = 100
        results = fuzzer.fuzz_n_tuple(2, max_combinations=max_combo)
        self.assertEqual(results.num_ran(), max_combo)
        self.assertEqual(results.lookup(tuple(key_1.keys())), Status.FAILED_RUN_RETURN)

    def test_config_fuzzer_inductor_bisect(self):
        # these values just chosen randomly, change to different ones if necessary
        key_1 = {"split_reductions": False, "compute_all_bounds": True}

        def create_key_1():
            def myfn():
                if (
                    not inductor_config.split_reductions
                    and inductor_config.compute_all_bounds
                ):
                    return False
                return True

            return myfn

        fuzzer = ConfigFuzzer(inductor_config, create_key_1, seed=100)
        num_attempts = 2
        results = fuzzer.bisect(num_attempts=num_attempts, p=1.0)
        self.assertEqual(len(results), num_attempts)
        for res in results:
            self.assertEqual(res, key_1)

        new_results = fuzzer.reproduce(results)
        self.assertEqual(len(new_results), 1)
        self.assertEqual(
            set(key_1.keys()),
            {j for i in new_results.keys() for j in i}
            - set(MODULE_DEFAULTS["torch._inductor.config"].keys()),
        )

    @unittest.skipIf(not IS_LINUX, "PerfCounters are only supported on Linux")
    @unittest.skip(
        "Need default values for dynamo flags - https://github.com/pytorch/pytorch/issues/164062"
    )
    def test_config_fuzzer_dynamo_bisect(self):
        # these values just chosen randomly, change to different ones if necessary
        key_1 = {"dead_code_elimination": False, "specialize_int": True}

        def create_key_1():
            def myfn():
                if (
                    not dynamo_config.dead_code_elimination
                    and dynamo_config.specialize_int
                ):
                    return False
                return True

            return myfn

        fuzzer = ConfigFuzzer(dynamo_config, create_key_1, seed=10)
        num_attempts = 2
        results = fuzzer.bisect(num_attempts=num_attempts, p=1.0)
        self.assertEqual(len(results), num_attempts)
        for res in results:
            self.assertEqual(res, key_1)

        new_results = fuzzer.reproduce(results)
        self.assertEqual(len(new_results), 1)
        self.assertEqual(
            set(key_1.keys()),
            {j for i in new_results.keys() for j in i}
            - set(MODULE_DEFAULTS["torch._dynamo.config"].keys()),
        )

    @patch("torch.compile")
    def test_fuzzer_inductor_calling_compile(self, compile):
        def create_key_1():
            def myfn():
                return True

            return myfn

        fuzzer = ConfigFuzzer(inductor_config, create_key_1, seed=100)
        num_attempts = 3
        fuzzer.bisect(num_attempts=num_attempts, p=0.5)
        self.assertEqual(compile.call_count, num_attempts)

    def test_fuzzer_running_test(self):
        def create_key_1():
            def myfn():
                return True

            return myfn

        fuzzer = ConfigFuzzer(inductor_config, create_key_1, seed=100)
        fuzzer.test_config = MagicMock(return_value=Status.PASSED)
        num_attempts = 20
        fuzzer.bisect(num_attempts=num_attempts, p=0.5)
        self.assertEqual(fuzzer.test_config.call_count, num_attempts)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 29 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestConfigFuzzer`, `MyException`

**Functions defined**: `create_simple_test_model_cpu`, `test_fn`, `create_simple_test_model_gpu`, `test_fn`, `test_sampling_method_toggle`, `test_sampling_method_random`, `test_config_fuzzer_inductor_gpu`, `test_config_fuzzer_inductor_cpu`, `test_config_fuzzer_bisector_exception`, `create_key_1`, `myfn`, `test_config_fuzzer_bisector_boolean`, `create_key_1`, `myfn`, `test_config_fuzzer_n_tuple`, `create_key_1`, `myfn`, `test_config_fuzzer_inductor_bisect`, `create_key_1`, `myfn`

**Key imports**: unittest, Literal, MagicMock, patch, torch, config as dynamo_config, config as inductor_config, ConfigFuzzer, MODULE_DEFAULTS, SamplingMethod, Status, run_tests, TestCase, fake_config_module as fake_config, IS_LINUX


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `typing`: Literal
- `unittest.mock`: MagicMock, patch
- `torch`
- `torch._dynamo`: config as dynamo_config
- `torch._inductor`: config as inductor_config
- `torch._inductor.fuzzer`: ConfigFuzzer, MODULE_DEFAULTS, SamplingMethod, Status
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal`: fake_config_module as fake_config
- `torch.testing._internal.common_utils`: IS_LINUX
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_fuzzer.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_fuzzer.py_docs.md`
- **Keyword Index**: `test_fuzzer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/inductor/test_fuzzer.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_split_cat_fx_passes.py_kw.md_docs.md`](./test_split_cat_fx_passes.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fuzzer.py_docs.md_docs.md`
- **Keyword Index**: `test_fuzzer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
