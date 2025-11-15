# Documentation: `test/nn/test_parametrization.py`

## File Metadata

- **Path**: `test/nn/test_parametrization.py`
- **Size**: 83,997 bytes (82.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: nn"]
import pickle
from copy import deepcopy
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.parametrize as parametrize
from torch import Tensor
from torch.__future__ import get_swap_module_params_on_conversion
from torch.nn import Buffer, Parameter
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    gradcheck,
    instantiate_parametrized_tests,
    run_tests,
    set_default_dtype,
    skipIfNoLapack,
    skipIfTorchDynamo,
    swap,
    TemporaryFileName,
)
from torch.testing._internal.two_tensor import TwoTensor


class TestNNParametrization(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    # torch/nn/utils/parametrize
    @skipIfNoLapack
    @swap([True, False])
    def test_register_and_remove_parametrization(self):
        r"""Test that it is possible to add a few parametrizations
        on a parameter or a buffer and that removing them restores the initial state
        It also tests that backpropagating through them works as expected
        """

        # Define a couple matrix parametrizations
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                # Cayley map
                # If X is skew-symmetric it returns an orthogonal matrix
                Id = torch.eye(X.size(0), device=X.device)
                # We call contiguous because solve returns a tensor with strides that are Fortran-contiguous
                # and autograd raises a performance warning.
                # This happens when we remove the parametrization with leave_parametrized=True,
                # which does a set_ with a non-contiguous tensor while the gradient is contiguous
                return torch.linalg.solve(Id + X, Id - X).contiguous()

        class Resize(nn.Module):
            def forward(self, X):
                return X[[0]]

        class NoResize(nn.Module):
            def forward(self, X):
                return X

        # Define a couple vector parametrizations
        class FirstZero(nn.Module):
            def forward(self, x):
                return torch.cat([x.new_zeros(1), x[1:]])

        class LastZero(nn.Module):
            def forward(self, x):
                return torch.cat([x[:-1], x.new_zeros(1)])

        model = nn.Linear(8, 8)
        initial_weight_id = id(model.weight)
        initial_bias_id = id(model.bias)
        initial_model = deepcopy(model)

        # Test unsafe flag
        with self.assertRaisesRegex(
            ValueError,
            "Registering a parametrization may not change the shape of the tensor",
        ):
            parametrize.register_parametrization(
                model, "weight", Resize()
            )  # default unsafe = False
            model(torch.ones(8, 8))

        # One parametrization with unsafe=True
        parametrize.register_parametrization(model, "weight", Resize(), unsafe=True)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        self.assertTrue(model.weight.shape[0] == 1)
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Two parametrizations with unsafe=True
        parametrize.register_parametrization(model, "weight", Resize(), unsafe=True)
        parametrize.register_parametrization(model, "weight", NoResize(), unsafe=False)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        self.assertTrue(model.weight.shape[0] == 1)
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test unsafe flag doesn't change expected behavior
        parametrize.register_parametrization(model, "weight", Skew(), unsafe=True)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del A
        # Remove and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test one parametrization
        parametrize.register_parametrization(model, "weight", Skew())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del A
        # Remove and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test two parametrizations at the same time and removing them
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())
        # Result should be orthogonal
        X = model.weight
        Id = torch.eye(X.size(0), device=X.device)
        self.assertEqual(X.T @ X, Id)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del X
        # Structure tests
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertIn("weight", model.parametrizations)
        self.assertNotIn("weight", model._parameters)
        # Remove
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)

        # Add everything
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())
        parametrize.register_parametrization(model, "bias", FirstZero())
        parametrize.register_parametrization(model, "bias", LastZero())

        # Basic tests
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertTrue(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.0)
        self.assertEqual(model.bias[-1].item(), 0.0)
        self.assertEqual(
            len(list(model.parameters())), 2
        )  # Nothing weird has happpened
        # Should not throw

        sgd = torch.optim.SGD(model.parameters(), lr=0.01)

        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)

        # Remove first parametrization.
        # Check that the model is still parametrized and so is the second parameter
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertTrue(parametrize.is_parametrized(model))  # Still parametrized
        self.assertFalse(
            parametrize.is_parametrized(model, "weight")
        )  # Parametrization removed
        self.assertTrue(
            parametrize.is_parametrized(model, "bias")
        )  # Still parametrized
        self.assertEqual(model.bias[0].item(), 0.0)  # Still parametrized
        self.assertEqual(model.bias[-1].item(), 0.0)  # Still parametrized
        self.assertNotEqual(model.weight, initial_model.weight)  # Has been updated
        self.assertEqual(id(model.weight), initial_weight_id)  # Keeps the same id
        self.assertEqual(len(list(model.parameters())), 2)  # Nothing weird has happened
        # Should not throw
        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)

        # Remove the second parametrization.
        # Check that the module is not parametrized
        parametrize.remove_parametrizations(model, "bias", leave_parametrized=False)
        self.assertFalse(parametrize.is_parametrized(model))  # Not parametrized
        self.assertNotEqual(model.bias, initial_model.bias)  # Has been updated
        self.assertNotEqual(model.bias[0].item(), 0.0)  # Not parametrized
        self.assertNotEqual(model.bias[-1].item(), 0.0)  # Not parametrized
        self.assertEqual(id(model.bias), initial_bias_id)  # Keeps the same id
        self.assertFalse(
            hasattr(model, "parametrizations")
        )  # Not parametrized the module
        self.assertEqual(model.__class__, nn.Linear)  # Resores the previous class
        self.assertEqual(len(list(model.parameters())), 2)  # Nothing weird has happeed

        # Should not throw things are updated
        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del weight_copy, bias_copy

        # Test leave_parametrized=True
        for _ in range(2):
            parametrize.register_parametrization(model, "weight", Skew())
            parametrize.register_parametrization(model, "weight", Orthogonal())
            parametrize.remove_parametrizations(
                model, "weight", leave_parametrized=True
            )
            # We didn't change the dtype nor had multiple inputs, so the id should be the same
            self.assertEqual(id(model.weight), initial_weight_id)
            self.assertEqual(id(model.bias), initial_bias_id)

            # Should not throw. Things are updated
            weight_copy = model.weight.clone()
            bias_copy = model.bias.clone()
            sgd.zero_grad()
            (model.weight.T @ model.bias).sum().backward()
            sgd.step()
            self.assertNotEqual(model.weight, weight_copy)
            self.assertNotEqual(model.bias, bias_copy)
            if get_swap_module_params_on_conversion():
                # When using the swap_tensors path, this is needed so that the autograd
                # graph is not alive anymore.
                del weight_copy, bias_copy

    @swap([True, False])
    def test_register_and_remove_nested_parametrization(self):
        r"""Test that it is possible to nest the parametrizations
        meaning that the original param is parametrized again
        """

        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        model = nn.Linear(8, 8)
        # Add top level parametrization
        parametrize.register_parametrization(model, "weight", Skew())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del A

        # Add nested parametrization
        param_mod = model.parametrizations.weight
        self.assertFalse(hasattr(param_mod, "parametrizations"))
        self.assertFalse(parametrize.is_parametrized(param_mod))
        self.assertFalse(parametrize.is_parametrized(param_mod, "original"))

        parametrize.register_parametrization(param_mod, "original", Skew())
        self.assertTrue(hasattr(param_mod, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(param_mod))
        self.assertTrue(parametrize.is_parametrized(param_mod, "original"))
        self.assertNotIn("original", param_mod._parameters)
        # Result should be skew-symmetric
        A = param_mod.original
        self.assertEqual(A, -A.T)

        # Remove nested param and check consistency
        parametrize.remove_parametrizations(
            param_mod, "original", leave_parametrized=False
        )
        self.assertFalse(hasattr(param_mod, "parametrizations"))
        self.assertEqual(param_mod.__class__, parametrize.ParametrizationList)

        # Remove top level and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)

    @swap([True, False])
    def test_register_and_remove_buffer_parametrization(self):
        r"""Test that it is possible to add and remove parametrizations on buffers"""

        # Define a couple vector parametrizations
        class FirstZero(nn.Module):
            def forward(self, x):
                return torch.cat([x.new_zeros(1), x[1:]])

        class LastZero(nn.Module):
            def forward(self, x):
                return torch.cat([x[:-1], x.new_zeros(1)])

        model = nn.Linear(8, 8)

        # Instantiate parametrizations on buffers. It should work as expected
        delattr(model, "bias")
        model.bias = Buffer(torch.ones(8))
        parametrize.register_parametrization(model, "bias", FirstZero())
        parametrize.register_parametrization(model, "bias", LastZero())
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.0)
        self.assertEqual(model.bias[-1].item(), 0.0)
        self.assertTrue((model.bias[1:-1] == torch.ones(6)).all())
        self.assertEqual(len(list(model.parameters())), 1)

        # Remove parametrizations on buffers. It should work as expected
        parametrize.remove_parametrizations(model, "bias", leave_parametrized=True)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.0)
        self.assertEqual(model.bias[-1].item(), 0.0)
        self.assertTrue((model.bias[1:-1] == torch.ones(6)).all())
        self.assertEqual(len(list(model.parameters())), 1)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @skipIfTorchDynamo(
        "Not applicable; see https://github.com/pytorch/pytorch/issues/127738"
    )
    @swap([True, False])
    def test_serialization_parametrization(self):
        r"""Test that it is possible to serialize a parametrized model via state_dict"""

        # A stateful parametrization
        class Orthogonal(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.id = Buffer(torch.eye(n))
                self.B = Buffer(torch.empty(n, n))
                init.orthogonal_(self.B)

            def forward(self, X):
                A = X.triu(1)
                A = A - A.T
                return self.B @ torch.linalg.solve(self.id + A, self.id - A)

        def get_model():
            model = torch.nn.Sequential(
                torch.nn.Linear(5, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1),
            )

            parametrize.register_parametrization(model[0], "weight", Orthogonal(5))
            return model

        model = get_model()

        prev_weight = model[0].weight
        prev_B = model[0].parametrizations.weight[0].B

        new_model = get_model()
        with TemporaryFileName() as fname:
            torch.save(model.state_dict(), fname)
            new_model.load_state_dict(torch.load(fname))

        # Integrity tests
        self.assertTrue(parametrize.is_parametrized(new_model[0], "weight"))
        self.assertEqual(prev_weight, new_model[0].weight)
        self.assertEqual(prev_B, new_model[0].parametrizations.weight[0].B)

        # Trying to save the whole parametrized model raises
        with self.assertRaisesRegex(RuntimeError, "state_dict"):
            with TemporaryFileName() as fname:
                torch.save(model, fname)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_initialization_parametrization(self):
        r"""Test that it is possible to initialize a parametrization when it
        implements a `right_inverse` method
        """

        class Skew(nn.Module):
            def forward(self, X):
                A = X.triu(1)
                return A - A.T

            def is_skew(self, A):
                return torch.allclose(A, -A.T, atol=1e-6)

            def right_inverse(self, X):
                if not self.is_skew(X):
                    raise ValueError("The matrix is not skew-symmetric.")
                return X.triu(1)

        # Implements a Cayley map where right_inverse is not quite the inverse of forward
        class Orthogonal(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.B = Buffer(torch.eye(n))

            def forward(self, X):
                Id = torch.eye(X.size(0))
                return self.B @ torch.linalg.solve(Id + X, Id - X)

            def is_orthogonal(self, X):
                Id = torch.eye(X.size(0))
                return torch.allclose(X.T @ X, Id, atol=1e-4)

            def right_inverse(self, X):
                if not self.is_orthogonal(X):
                    raise ValueError("The input is not orthogonal.")
                # cayley(0) == Id, so B @ cayley(0) == B
                self.B = X
                return torch.zeros_like(X)

        N = 5
        model = nn.Linear(N, N)
        # Register the skew-symmetric constraint. The result is now skew-symmetric
        skew = Skew()
        # Make the weight skew-symmetric before registering the parametrization
        with torch.no_grad():
            model.weight.set_(skew(model.weight))
        parametrize.register_parametrization(model, "weight", skew)
        X = torch.rand(N, N)
        # X is not skew-symmetric, so it throws an error
        with self.assertRaises(ValueError):
            model.weight = X
        # Make X skew-symmetric
        X = X - X.T
        model.weight = X
        self.assertEqual(model.parametrizations.weight.original, X.triu(1))
        self.assertEqual(model.weight, X)

        # Having several parametrizations registered should work in the same way
        parametrize.register_parametrization(model, "weight", Orthogonal(N))
        # Register now the Cayley map. The result is now orthogonal
        X = torch.rand(N, N)
        # X is not orthogonal, so it throws an error
        with self.assertRaises(ValueError):
            model.weight = X
        init.orthogonal_(X)
        model.weight = X
        self.assertEqual(model.weight, X)
        self.assertEqual(model.parametrizations.weight.original, torch.zeros_like(X))

    @swap([True, False])
    def test_errors_unparametrized_tensor_parametrization(self):
        # Test errors when registering a parametrization on an unparametrized tensor
        module = nn.Linear(3, 4)
        weight_init = module.weight.clone()

        class Identity(nn.Module):
            def forward(self, x):
                return x

        # Register a parametrization on a non-existing parameter throws
        with self.assertRaisesRegex(ValueError, "does not have a parameter"):
            parametrize.register_parametrization(module, "foo", Identity())
        self.assertFalse(parametrize.is_parametrized(module))

        # Removing parametrizations from an unparametrized tensor throws
        with self.assertRaisesRegex(ValueError, "does not have a parametrization"):
            parametrize.remove_parametrizations(module, "bias")
        self.assertFalse(parametrize.is_parametrized(module))

        # A correct parametrization with several outputs
        class Sum(nn.Module):
            def forward(self, x, y):
                return x + y

            def right_inverse(self, z):
                return z, torch.zeros_like(z)

        parametrize.register_parametrization(module, "weight", Sum())
        # Cannot remove a parametrization with several outputs with `leave_parametrized=False`
        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=False
            )
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        # A parametrization with an incorrect number of outputs
        class WrongNumberParams(nn.Module):
            def forward(self, x, y, z):
                return x + y + z

            def right_inverse(self, w):
                return w, torch.zeros_like(w)

        # Makes param(*param.right_inverse(X)) fail
        with self.assertRaisesRegex(TypeError, "positional argument"):
            parametrize.register_parametrization(module, "weight", WrongNumberParams())
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization with a right_inverse that does not return a Tensor or Sequence[Tensor]
        class WrongRightInverse(Identity):
            def right_inverse(self, z):
                return None

        # right_inverse should return a Tensor or a Sequence[Tensor]
        with self.assertRaisesRegex(ValueError, "Tensor or a Sequence of"):
            parametrize.register_parametrization(module, "weight", WrongRightInverse())
        self.assertFalse(parametrize.is_parametrized(module))

        # If it's a sequence, it must to be a sequence of tensors
        class WrongRightInverseSequence(nn.Module):
            def forward(self, x, y):
                return x

            def right_inverse(self, z):
                return None, z

        with self.assertRaisesRegex(ValueError, "of the sequence with type"):
            parametrize.register_parametrization(
                module, "weight", WrongRightInverseSequence()
            )
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization from one tensor to one tensor that changes the dtype
        class ChangeDtypeInverse(nn.Module):
            def forward(self, x):
                return x.float()

            def right_inverse(self, w):
                return w.bool()

        # For parametrizations that return one tensor, right_inverse may not change the dtype
        with self.assertRaisesRegex(
            ValueError, "outputs one tensor, it may not change the dtype"
        ):
            parametrize.register_parametrization(module, "weight", ChangeDtypeInverse())
        self.assertFalse(parametrize.is_parametrized(module))

        class ChangeDeviceInverse(nn.Module):
            def forward(self, x):
                return x.float()

            def right_inverse(self, w):
                return w.to(torch.device("meta"))

        # For parametrizations that return one tensor, right_inverse may not change the device
        with self.assertRaisesRegex(
            ValueError, "outputs one tensor, it may not change the device"
        ):
            parametrize.register_parametrization(
                module, "weight", ChangeDeviceInverse()
            )
        self.assertFalse(parametrize.is_parametrized(module))

        # Doesn't return a tensor
        class NotTensor(nn.Module):
            def forward(self, x):
                return 2

        # Forward must return a tensor
        with self.assertRaisesRegex(ValueError, "must return a tensor"):
            parametrize.register_parametrization(module, "weight", NotTensor())
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization from one tensor to one tensor that changes the dtype
        class ChangeDtype(nn.Module):
            def forward(self, x):
                return x.bool()

        # forward should not change the initial dtype
        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtype())
        self.assertFalse(parametrize.is_parametrized(module))

        # Change shape
        class ChangeShape(nn.Module):
            def forward(self, x):
                return x[:-1]

        # forward should not change the original shape
        with self.assertRaisesRegex(ValueError, "may not change the shape"):
            parametrize.register_parametrization(module, "weight", ChangeShape())
        self.assertFalse(parametrize.is_parametrized(module))

        # Many to one that changes dtype
        class ChangeDtypeMulti(nn.Module):
            def forward(self, x, y):
                return (x + y).bool()

            def right_inverse(self, w):
                return w, w + 1

        # forward should not change the original shape even for parametrizations with many inputs
        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtypeMulti())
        self.assertFalse(parametrize.is_parametrized(module))

        # Returning a sequence of size one, although weird, it's correct
        class SequenceLen1(nn.Module):
            def forward(self, x):
                return x

            def right_inverse(self, w):
                return (w,)

        parametrize.register_parametrization(module, "weight", SequenceLen1())
        self.assertTrue(hasattr(module.parametrizations.weight, "original0"))
        self.assertFalse(hasattr(module.parametrizations.weight, "original1"))
        _ = module.weight  # Does not throw
        self.assertTrue(parametrize.is_parametrized(module))
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        # None of the operations above should have altered the weight
        self.assertFalse(parametrize.is_parametrized(module))
        self.assertEqual(module.weight, weight_init)

    @swap([True, False])
    def test_errors_parametrized_tensor_parametrization(self):
        # Test errors when registering a parametrization on a parametrized tensor

        class Identity(nn.Module):
            def forward(self, x):
                return x

        module = nn.Linear(3, 4)
        parametrize.register_parametrization(module, "weight", Identity())

        # Has to return a tensor
        class WrongReturn(nn.Module):
            def forward(self, x):
                return x, x

        with self.assertRaisesRegex(ValueError, "must return a tensor"):
            parametrize.register_parametrization(module, "weight", WrongReturn())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change dtype
        class ChangeDtype(nn.Module):
            def forward(self, x):
                return x.bool()

        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtype())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change shape
        class ChangeShape(nn.Module):
            def forward(self, x):
                return x[:-1]

        with self.assertRaisesRegex(ValueError, "may not change the shape"):
            parametrize.register_parametrization(module, "weight", ChangeShape())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # The following checks are mostly due to bugs in the code of the parametrization

        # right_inverse has to return a tensor
        class WrongReturnInverse(Identity):
            def right_inverse(self, x):
                return x, x

        with self.assertRaisesRegex(ValueError, "right_inverse must return a tensor"):
            parametrize.register_parametrization(module, "weight", WrongReturnInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change dtype
        class ChangeDtypeInverse(Identity):
            def right_inverse(self, x):
                return x.bool()

        with self.assertRaisesRegex(ValueError, "must have the same dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtypeInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change shape
        class ChangeShapeInverse(Identity):
            def right_inverse(self, x):
                return x[:-1]

        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            parametrize.register_parametrization(module, "weight", ChangeShapeInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_multiple_inputs_parametrization(self):
        # A parametrization with several outputs
        class RankOne(nn.Module):
            def forward(self, x, y):
                # Form a rank-1 matrix from a pair of vectors
                return x.unsqueeze(-1) @ y.unsqueeze(-2)

            def right_inverse(self, Y):
                # We project the given matrix onto the rank 1 matrices
                U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                # S is ordered in a decreasing way.
                s0_sqrt = S[0].sqrt().unsqueeze(-1)
                return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt

        # Simple parametrisation
        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, w):
                return 0.5 * w

        model = nn.Linear(3, 3)
        # Test one parametrization
        parametrize.register_parametrization(model, "weight", RankOne())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertTrue(hasattr(model.parametrizations.weight, "original0"))
        self.assertIn("original0", model.parametrizations.weight._parameters)
        self.assertTrue(hasattr(model.parametrizations.weight, "original1"))
        self.assertIn("original1", model.parametrizations.weight._parameters)
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be rank 1
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)

        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            # Cannot remove a parametrization with multiple inputs and not leave it parametrized
            parametrize.remove_parametrizations(
                model, "weight", leave_parametrized=False
            )
        # Remove parametrization and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=True)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)
        self.assertIn("weight", model._parameters)

        # Registering parametrizations with one input on top of one with multiple inputs should work
        init_weight = model.weight.clone()
        parametrize.register_parametrization(model, "weight", RankOne())
        # Projecting a rank 1 matrix onto the matrices of rank one does not change the matrix
        self.assertEqual(init_weight, model.weight)
        parametrize.register_parametrization(model, "weight", Double())
        # The matrix now is twice the initial matrix
        self.assertEqual(2.0 * init_weight, model.weight)
        # Multiplying by a scalar does not change the rank
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)

        # The model has now three parameters
        self.assertEqual(len(list(model.parameters())), 3)

        sgd = torch.optim.SGD(model.parameters(), lr=0.1)

        # Test backward. Should not throw
        for _ in range(2):
            sgd.zero_grad()
            loss = (model.weight.T @ model.bias).sum()
            loss.backward()
            sgd.step()

        # Same drill as before, removing should work as expected
        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            # Cannot remove a parametrization with multiple inputs and not leave it parametrized
            parametrize.remove_parametrizations(
                model, "weight", leave_parametrized=False
            )
        # Remove parametrization and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=True)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)
        self.assertIn("weight", model._parameters)

        # The model has now two parameters
        self.assertEqual(len(list(model.parameters())), 2)

        # Test backward. Should not throw
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(2):
            sgd.zero_grad()
            loss = (model.weight.T @ model.bias).sum()
            loss.backward()
            sgd.step()

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_caching_parametrization(self):
        r"""Test the caching system of a parametrization"""

        # Define a couple matrix parametrizations
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                Id = torch.eye(X.size(0), device=X.device)
                return torch.linalg.solve(Id + X, Id - X)

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())

        # Test that the caching system works
        with parametrize.cached():
            X = model.weight
            Y = model.weight
            self.assertEqual(id(X), id(Y))

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_caching_parametrization_with_transfer_parametrizations_and_params(self):
        r"""Test that transferring parametrizations doesn't cause issues with caching"""

        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                Id = torch.eye(X.size(0), device=X.device)
                return torch.linalg.solve(Id + X, Id - X)

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())

        to_model = nn.Linear(5, 5)
        parametrize.transfer_parametrizations_and_params(model, to_model)

        with parametrize.cached():
            X = model.weight
            Y = model.weight
            self.assertEqual(id(X), id(Y))

            A = to_model.weight
            B = to_model.weight
            self.assertEqual(id(A), id(B))

            # test that the results are distinct objects for each module
            self.assertNotEqual(id(A), id(X))

    @swap([True, False])
    def test_parametrization_same_training_mode(self):
        r"""Test training mode updated on parametrization registration"""

        class Identity(nn.Module):
            def forward(self, X):
                return X

        module = nn.Linear(4, 4)
        module.eval()
        parametrize.register_parametrization(module, "weight", Identity())
        self.assertFalse(module.parametrizations.weight[0].training)
        module.train()
        parametrize.register_parametrization(module, "weight", Identity().eval())
        self.assertTrue(module.parametrizations.weight[0].training)
        self.assertTrue(module.parametrizations.weight[1].training)

    @swap([True, False])
    def test_type_before_parametrizations(self):
        r"""Test that type_before_parametrizations always retrieves original type"""

        class Identity(nn.Module):
            def forward(self, X):
                return X

        model = nn.Linear(5, 5)
        original_type = type(model)
        self.assertTrue(
            parametrize.type_before_parametrizations(model) == original_type
        )
        parametrize.register_parametrization(model, "weight", Identity())
        self.assertTrue(
            parametrize.type_before_parametrizations(model) == original_type
        )

    @skipIfTorchDynamo(
        "Not applicable; see https://github.com/pytorch/pytorch/issues/127738"
    )
    @swap([True, False])
    def test_deepcopy_after_parametrization(self):
        r"""Test that we are able to create a deepcopy of the module when it's parametrized."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class ModelWithoutDeepcopy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.tensor([1.0, 1.0, 1.0, 1.0]), requires_grad=True
                )
                self.bias = nn.Parameter(
                    torch.tensor([0.0, 0.0, 0.0, 0.0]), requires_grad=True
                )
                self.attr = [1.0, 2.0, 3.0, 4.0]

        class ActualModel(ModelWithoutDeepcopy):
            # Emulate custom implementation of the deepcopying.
            def __deepcopy__(self, memo):
                result = self.__new__(self.__class__)
                memo[id(self)] = result
                result.__dict__ = deepcopy(self.__dict__, memo)
                return result

        def check_deepcopy(m1: nn.Module, m2: nn.Module):
            w1 = m1.parametrizations.weight.original
            w2 = m2.parametrizations.weight.original
            b1 = (
                m1.parametrizations.bias.original
                if parametrize.is_parametrized(m1, "bias")
                else m1.bias
            )
            b2 = (
                m2.parametrizations.bias.original
                if parametrize.is_parametrized(m2, "bias")
                else m2.bias
            )
            # Weights, biases and attributes should be equal but they must be different objects.
            self.assertEqual(m1.__dict__.keys(), m2.__dict__.keys())
            self.assertIsNot(m1, m2)
            self.assertEqual(w1, w2)
            self.assertIsNot(w1, w2)
            self.assertEqual(b1, b2)
            self.assertIsNot(b1, b2)
            self.assertEqual(m1.attr, m2.attr)
            self.assertIsNot(m1.attr, m2.attr)

        for model in (ModelWithoutDeepcopy(), ActualModel()):
            # General check that we are able to create deepcopy.
            parametrize.register_parametrization(model, "weight", AddOne())
            check_deepcopy(model, deepcopy(model))
            # Check that this works on models with several parametrized tensors.
            parametrize.register_parametrization(model, "bias", AddOne())
            check_deepcopy(model, deepcopy(model))
            # Check that this works on models where tensors have more than one parametrization.
            parametrize.register_parametrization(model, "weight", AddOne())
            check_deepcopy(model, deepcopy(model))

    @swap([True, False])
    def test_transfer_parametrizations_and_params(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, x):
                return 0.5 * x

        class MinusOne(nn.Module):
            def forward(self, x):
                return x - 1.0

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", AddOne())
        parametrize.register_parametrization(model, "weight", Double())
        parametrize.register_parametrization(model, "weight", MinusOne())
        hold_weight = model.weight

        to_model = torch.ao.nn.qat.Linear(
            5, 5, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # checks that final and original value are correct and the to_model is parametrized
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )

        # check that the transfer didn't affect the original value
        self.assertEqual(hold_weight, model.weight)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del hold_weight

        # testing that changes to one set of parametrizations do not affect the other
        parametrize.remove_parametrizations(to_model, "weight")
        self.assertFalse(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(model, "weight"))

        # also test that parameters that don't exist in to_model get transferred
        model.test_param = Parameter(torch.randn(5, 5))

        self.assertTrue(not hasattr(to_model, "test_param"))
        parametrize.register_parametrization(model, "test_param", Double())
        hold_test_param = model.test_param
        parametrize.transfer_parametrizations_and_params(model, to_model, "test_param")

        # check that previously missing params got transferred correctly
        self.assertEqual(model.test_param, to_model.test_param)
        self.assertEqual(
            model.parametrizations.test_param.original,
            to_model.parametrizations.test_param.original,
        )

        # check that the new transfer didn't change the value for the from_module
        self.assertEqual(hold_test_param, model.test_param)

    @swap([True, False])
    def test_transfer_parametrizations_and_params_right_inverse(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, x):
                return 0.5 * x

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Double())
        hold_weight = model.weight

        to_model = torch.ao.nn.qat.Linear(
            5, 5, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # check that transfer occurs successfully
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )

        # check that transfer doesn't affect the from_model weight
        self.assertEqual(hold_weight, model.weight)

    @swap([True, False])
    def test_transfer_parametrizations_and_params_single_param(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

        class MinusOne(nn.Module):
            def forward(self, x):
                return x - 1.0

        model = nn.Linear(5, 5, bias=True)
        parametrize.register_parametrization(model, "weight", AddOne())
        parametrize.register_parametrization(model, "weight", Double())
        parametrize.register_parametrization(model, "weight", MinusOne())
        parametrize.register_parametrization(model, "bias", AddOne())
        parametrize.register_parametrization(model, "bias", Double())
        parametrize.register_parametrization(model, "bias", MinusOne())

        to_model = torch.ao.nn.qat.Linear(
            5, 5, bias=True, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model, "weight")

        # check that weight and only weight was transferred
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )
        self.assertTrue("bias" not in to_model.parametrizations)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    # and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    
```



## High-Level Overview

r"""Test that it is possible to add a few parametrizations        on a parameter or a buffer and that removing them restores the initial state        It also tests that backpropagating through them works as expected

This Python file contains 58 class(es) and 116 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNNParametrization`, `Skew`, `Orthogonal`, `Resize`, `NoResize`, `FirstZero`, `LastZero`, `Skew`, `FirstZero`, `LastZero`, `Orthogonal`, `Skew`, `Orthogonal`, `Identity`, `Sum`, `WrongNumberParams`, `WrongRightInverse`, `WrongRightInverseSequence`, `ChangeDtypeInverse`, `ChangeDeviceInverse`

**Functions defined**: `test_register_and_remove_parametrization`, `forward`, `forward`, `forward`, `forward`, `forward`, `forward`, `test_register_and_remove_nested_parametrization`, `forward`, `test_register_and_remove_buffer_parametrization`, `forward`, `forward`, `test_serialization_parametrization`, `__init__`, `forward`, `get_model`, `test_initialization_parametrization`, `forward`, `is_skew`, `right_inverse`

**Key imports**: pickle, deepcopy, product, torch, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init, torch.nn.utils.parametrize as parametrize, Tensor, get_swap_module_params_on_conversion


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/nn`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `pickle`
- `copy`: deepcopy
- `itertools`: product
- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.nn.init as init`
- `torch.nn.utils.parametrize as parametrize`
- `torch.__future__`: get_swap_module_params_on_conversion
- `torch.nn`: Buffer, Parameter
- `torch.testing._internal.common_cuda`: TEST_MULTIGPU
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_nn`: NNTestCase
- `torch.testing._internal.two_tensor`: TwoTensor


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/nn/test_parametrization.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/nn`):

- [`test_dropout.py_docs.md`](./test_dropout.py_docs.md)
- [`test_convolution.py_docs.md`](./test_convolution.py_docs.md)
- [`test_lazy_modules.py_docs.md`](./test_lazy_modules.py_docs.md)
- [`test_packed_sequence.py_docs.md`](./test_packed_sequence.py_docs.md)
- [`test_embedding.py_docs.md`](./test_embedding.py_docs.md)
- [`test_multihead_attention.py_docs.md`](./test_multihead_attention.py_docs.md)
- [`test_init.py_docs.md`](./test_init.py_docs.md)
- [`test_module_hooks.py_docs.md`](./test_module_hooks.py_docs.md)
- [`test_pruning.py_docs.md`](./test_pruning.py_docs.md)


## Cross-References

- **File Documentation**: `test_parametrization.py_docs.md`
- **Keyword Index**: `test_parametrization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
