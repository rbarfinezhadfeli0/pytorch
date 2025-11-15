# Documentation: `test/ao/sparsity/test_sparsity_utils.py`

## File Metadata

- **Path**: `test/ao/sparsity/test_sparsity_utils.py`
- **Size**: 5,854 bytes (5.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: sparse"]


import logging

import torch
from torch.ao.pruning.sparsifier.utils import (
    fqn_to_module,
    get_arg_info_from_tensor_fqn,
    module_to_fqn,
)
from torch.testing._internal.common_quantization import (
    ConvBnReLUModel,
    ConvModel,
    FunctionalLinear,
    LinearAddModel,
    ManualEmbeddingBagLinear,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
)
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

model_list = [
    ConvModel,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
    LinearAddModel,
    ConvBnReLUModel,
    ManualEmbeddingBagLinear,
    FunctionalLinear,
]


class TestSparsityUtilFunctions(TestCase):
    def test_module_to_fqn(self):
        """
        Tests that module_to_fqn works as expected when compared to known good
        module.get_submodule(fqn) function
        """
        for model_class in model_list:
            model = model_class()
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                fqn = module_to_fqn(model, module)
                check_module = model.get_submodule(fqn)
                self.assertEqual(module, check_module)

    def test_module_to_fqn_fail(self):
        """
        Tests that module_to_fqn returns None when an fqn that doesn't
        correspond to a path to a node/tensor is given
        """
        for model_class in model_list:
            model = model_class()
            fqn = module_to_fqn(model, torch.nn.Linear(3, 3))
            self.assertEqual(fqn, None)

    def test_module_to_fqn_root(self):
        """
        Tests that module_to_fqn returns '' when model and target module are the same
        """
        for model_class in model_list:
            model = model_class()
            fqn = module_to_fqn(model, model)
            self.assertEqual(fqn, "")

    def test_fqn_to_module(self):
        """
        Tests that fqn_to_module operates as inverse
        of module_to_fqn
        """
        for model_class in model_list:
            model = model_class()
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                fqn = module_to_fqn(model, module)
                check_module = fqn_to_module(model, fqn)
                self.assertEqual(module, check_module)

    def test_fqn_to_module_fail(self):
        """
        Tests that fqn_to_module returns None when it tries to
        find an fqn of a module outside the model
        """
        for model_class in model_list:
            model = model_class()
            fqn = "foo.bar.baz"
            check_module = fqn_to_module(model, fqn)
            self.assertEqual(check_module, None)

    def test_fqn_to_module_for_tensors(self):
        """
        Tests that fqn_to_module works for tensors, actually all parameters
        of the model. This is tested by identifying a module with a tensor,
        and generating the tensor_fqn using module_to_fqn on the module +
        the name of the tensor.
        """
        for model_class in model_list:
            model = model_class()
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                module_fqn = module_to_fqn(model, module)
                for tensor_name, tensor in module.named_parameters(recurse=False):
                    tensor_fqn = (  # string manip to handle tensors on root
                        module_fqn + ("." if module_fqn != "" else "") + tensor_name
                    )
                    check_tensor = fqn_to_module(model, tensor_fqn)
                    self.assertEqual(tensor, check_tensor)

    def test_get_arg_info_from_tensor_fqn(self):
        """
        Tests that get_arg_info_from_tensor_fqn works for all parameters of the model.
        Generates a tensor_fqn in the same way as test_fqn_to_module_for_tensors and
        then compares with known (parent) module and tensor_name as well as module_fqn
        from module_to_fqn.
        """
        for model_class in model_list:
            model = model_class()
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                module_fqn = module_to_fqn(model, module)
                for tensor_name, _ in module.named_parameters(recurse=False):
                    tensor_fqn = (
                        module_fqn + ("." if module_fqn != "" else "") + tensor_name
                    )
                    arg_info = get_arg_info_from_tensor_fqn(model, tensor_fqn)
                    self.assertEqual(arg_info["module"], module)
                    self.assertEqual(arg_info["module_fqn"], module_fqn)
                    self.assertEqual(arg_info["tensor_name"], tensor_name)
                    self.assertEqual(arg_info["tensor_fqn"], tensor_fqn)

    def test_get_arg_info_from_tensor_fqn_fail(self):
        """
        Tests that get_arg_info_from_tensor_fqn works as expected for invalid tensor_fqn
        inputs. The string outputs still work but the output module is expected to be None.
        """
        for model_class in model_list:
            model = model_class()
            tensor_fqn = "foo.bar.baz"
            arg_info = get_arg_info_from_tensor_fqn(model, tensor_fqn)
            self.assertEqual(arg_info["module"], None)
            self.assertEqual(arg_info["module_fqn"], "foo.bar")
            self.assertEqual(arg_info["tensor_name"], "baz")
            self.assertEqual(arg_info["tensor_fqn"], "foo.bar.baz")


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")

```



## High-Level Overview

"""        Tests that module_to_fqn works as expected when compared to known good        module.get_submodule(fqn) function

This Python file contains 9 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSparsityUtilFunctions`

**Functions defined**: `test_module_to_fqn`, `test_module_to_fqn_fail`, `test_module_to_fqn_root`, `test_fqn_to_module`, `test_fqn_to_module_fail`, `test_fqn_to_module_for_tensors`, `test_get_arg_info_from_tensor_fqn`, `test_get_arg_info_from_tensor_fqn_fail`

**Key imports**: logging, torch, raise_on_run_directly, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/ao/sparsity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/ao/sparsity/test_sparsity_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/ao/sparsity`):

- [`test_kernels.py_docs.md`](./test_kernels.py_docs.md)
- [`test_activation_sparsifier.py_docs.md`](./test_activation_sparsifier.py_docs.md)
- [`test_data_scheduler.py_docs.md`](./test_data_scheduler.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)
- [`test_scheduler.py_docs.md`](./test_scheduler.py_docs.md)
- [`test_data_sparsifier.py_docs.md`](./test_data_sparsifier.py_docs.md)
- [`test_structured_sparsifier.py_docs.md`](./test_structured_sparsifier.py_docs.md)
- [`test_qlinear_packed_params.py_docs.md`](./test_qlinear_packed_params.py_docs.md)
- [`test_sparsifier.py_docs.md`](./test_sparsifier.py_docs.md)


## Cross-References

- **File Documentation**: `test_sparsity_utils.py_docs.md`
- **Keyword Index**: `test_sparsity_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
