# Documentation: `test/nn/test_embedding.py`

## File Metadata

- **Path**: `test/nn/test_embedding.py`
- **Size**: 68,364 bytes (66.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: nn"]
import itertools
import random
import unittest
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyNativeDeviceTypes,
    onlyOn,
    skipCUDAIf,
    skipMeta,
    skipXPUIf,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    _assertGradAndGradgradChecks,
    dtype2prec_DONTUSE,
    instantiate_parametrized_tests,
    IS_JETSON,
    parametrize as parametrize_test,
    run_tests,
    set_default_dtype,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_XPU,
)


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class TestEmbeddingNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA/XPU unavailable")
    def test_embedding_max_norm_unsorted_repeating_indices(self):
        def create_embedding(device):
            # Seed RNG so we get the same Embedding each time
            torch.manual_seed(0)
            return torch.nn.Embedding(
                num_embeddings=20, embedding_dim=64, max_norm=1.0
            ).to(device)

        ix = torch.arange(2, device="cpu", dtype=torch.long).repeat(2000)
        out_cpu = create_embedding("cpu")(ix)

        ix = ix.to(device_type)
        out = create_embedding(device_type)(ix)
        self.assertEqual(out.cpu(), out_cpu)

    def test_embedding_sparse_basic(self):
        embedding = nn.Embedding(10, 20, sparse=True)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long)
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

    def test_embedding_sparse_empty_tensor(self):
        embedding = nn.Embedding(0, 0, sparse=True)
        input = torch.tensor([], dtype=torch.int64)
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

        embedding = nn.Embedding(10, 0, sparse=True)
        input = torch.LongTensor([[0, 2, 4, 5], [4, 3, 0, 9]])
        embedding(input).sum().backward()
        self.assertTrue(embedding.weight.grad.is_sparse)
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

    def test_move_sparse_half_embedding(self):
        embedding = nn.Embedding(10, 3, sparse=True)
        self.assertEqual(embedding.weight.device.type, "cpu")
        self.assertEqual(embedding.weight.dtype, torch.get_default_dtype())
        embedding.to(torch.float16)
        self.assertEqual(embedding.weight.dtype, torch.float16)
        self.assertEqual(embedding.embedding_dim, 3)
        self.assertEqual(embedding.num_embeddings, 10)

        if not torch.accelerator.is_available():
            embedding.to(device_type)
            self.assertEqual(embedding.weight.device.type, device_type)
            embedding.to("cpu")
            self.assertEqual(embedding.weight.device.type, "cpu")

    def test_embedding_max_norm(self):
        embedding = nn.Embedding(22, 5, max_norm=1.0)
        input = torch.tensor([2, 8, 8, 6], dtype=torch.long)
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    @parametrize_test(
        "dtype",
        (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float,
            torch.double,
        ),
    )
    def test_embedding_from_pretrained(self, dtype):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        embedding = nn.Embedding.from_pretrained(a)
        self.assertEqual(a, embedding.weight.data)

        input = torch.LongTensor([0, 1])
        output = embedding(input)
        self.assertEqual(a, output)

    def test_embedding_bag_from_pretrained(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        embedding = nn.EmbeddingBag.from_pretrained(a)
        self.assertEqual(a, embedding.weight)

        input = torch.tensor([0, 1], dtype=torch.long)
        output = embedding(input, torch.arange(input.size(0)))
        self.assertEqual(a, output)

    def test_embedding_from_pretrained_padding_idx(self):
        padding_idx = 2
        padding_vec = torch.ones(3) * 7
        embeddings = torch.rand(4, 3, requires_grad=True)
        with torch.no_grad():
            embeddings[padding_idx] = padding_vec
        embedding_nn = nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx)
        self.assertEqual(embedding_nn.weight[padding_idx], padding_vec)

    def test_embedding_bag_from_pretrained_padding_idx(self):
        padding_idx = 2
        embeddings = torch.rand(4, 3, requires_grad=True)
        embedding_nn = nn.EmbeddingBag.from_pretrained(
            embeddings, padding_idx=padding_idx
        )
        self.assertEqual(embedding_nn.weight, embeddings)

    def test_embedding_from_pretrained_options(self):
        with set_default_dtype(torch.double):
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            opts = {
                "max_norm": 2.0,
                "norm_type": 0.5,
                "scale_grad_by_freq": False,
                "sparse": True,
            }
            embedding = nn.Embedding.from_pretrained(a, **opts)
            input = torch.LongTensor([0, 1])
            output = embedding(input)
            # test output and that weight matrix was renormalized
            self.assertEqual(a, output)
            self.assertTrue(a.ne(torch.arange(1, 7, dtype=a.dtype).view(2, 3)).all())
            self.assertTrue(
                output.data.norm(p=opts["norm_type"], dim=1).le(opts["max_norm"]).all()
            )

    def test_embedding_functional(self):
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        embeddings = torch.rand(4, 3, requires_grad=True)

        embed_old = torch.nn.Embedding(4, 3)
        embed_old.weight.data = embeddings.data
        # A silly test for eager, this test is useful for when we run under PYTORCH_TEST_WITH_DYNAMO=1
        # as it ensures that setattr correctly works.
        self.assertEqual(embed_old.weight.data, embeddings.data)
        res_old = embed_old(a)

        res_F = F.embedding(a, embeddings)
        self.assertEqual(res_old, res_F)

        embed_old = torch.nn.Embedding(4, 3)
        embed_old = embed_old.from_pretrained(embeddings, padding_idx=2)
        res_old = embed_old(a)
        res_F = F.embedding(a, embeddings, padding_idx=2)

        self.assertEqual(res_old, res_F)

    # https://github.com/pytorch/pytorch/issues/130806
    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA/XPU not available")
    @largeTensorTest("40GB", device=device_type)
    def test_large_tensors(self):
        input = torch.randint(low=0, high=16032, size=[131072], device=device_type)
        w = torch.randn([16032, 16384], device=device_type)
        out = torch.nn.functional.embedding(input, w)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(out.numel(), 2147483648)

    def test_embedding_bag_functional(self):
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        embeddings = torch.rand(4, 3, requires_grad=True)

        embed_old = torch.nn.EmbeddingBag(4, 3)
        embed_old.weight = torch.nn.Parameter(embeddings)
        res_old = embed_old(a)

        res_F = F.embedding_bag(a, embeddings)
        self.assertEqual(res_old, res_F)

        embed_old = torch.nn.EmbeddingBag(4, 3)
        embed_old = embed_old.from_pretrained(embeddings, padding_idx=2)
        res_old = embed_old(a)
        res_F = F.embedding_bag(a, embeddings, padding_idx=2)

        self.assertEqual(res_old, res_F)

    # Make sure that error is thrown if padding_idx is out of bounds
    def test_embedding_bag_padding_idx_error(self):
        a = torch.tensor([[1, 3, 2], [0, 2, 1]], dtype=torch.long)
        num_embeddings = 4
        num_features = 3
        embeddings = torch.rand(num_embeddings, num_features, requires_grad=True)

        functional_err_msg = r"padding_idx must be within the number of embeddings"
        module_err_msg = r"padding_idx must be within num_embeddings"

        for padding_idx in range(-(num_embeddings + 2), (num_embeddings + 2)):
            if (padding_idx < -num_embeddings) or (padding_idx >= num_embeddings):
                with self.assertRaisesRegex(RuntimeError, functional_err_msg):
                    F.embedding_bag(a, embeddings, padding_idx=padding_idx)
                with self.assertRaisesRegex(AssertionError, module_err_msg):
                    torch.nn.EmbeddingBag(
                        num_embeddings, num_features, padding_idx=padding_idx
                    )
            else:
                F.embedding_bag(a, embeddings, padding_idx=padding_idx)
                torch.nn.EmbeddingBag(
                    num_embeddings, num_features, padding_idx=padding_idx
                )

    def test_embeddingbag_from_pretrained(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        embeddingbag = nn.EmbeddingBag.from_pretrained(a)
        self.assertEqual(a, embeddingbag.weight.data)

        input = torch.LongTensor([[0, 1]])
        output = embeddingbag(input)
        self.assertEqual(a.mean(0, keepdim=True), output)

    def test_embeddingbag_from_pretrained_options(self):
        with set_default_dtype(torch.double):
            a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            opts = {
                "max_norm": 2.0,
                "norm_type": 0.5,
                "scale_grad_by_freq": False,
                "mode": "max",
                "sparse": False,
            }
            embeddingbag = nn.EmbeddingBag.from_pretrained(a, **opts)

            input = torch.LongTensor([[0, 1]])
            output = embeddingbag(input)
            self.assertEqual(a.max(0, keepdim=True)[0], output)
            self.assertTrue(a.ne(torch.arange(1, 7, dtype=a.dtype).view(2, 3)).all())
            self.assertTrue(
                a.norm(p=opts["norm_type"], dim=1).le(opts["max_norm"]).all()
            )

    def test_embeddingbag_include_last_offset(self):
        # Test case from https://github.com/pytorch/pytorch/issues/89677
        embeddingbag = nn.EmbeddingBag(100, 3, include_last_offset=True, padding_idx=61)
        input = torch.tensor([0, 1, 2, 3])
        out = embeddingbag(input, torch.tensor([0, 3, 3]))
        out2 = embeddingbag(input, torch.tensor([0, 3, 4]))

        weight = embeddingbag.weight
        row0 = weight[0:3].mean(0)
        row1 = weight[3]
        ref_out = torch.stack([row0, row1])

        self.assertEqual(ref_out, out)
        self.assertEqual(ref_out, out2)


class TestEmbeddingNNDeviceType(NNTestCase):
    def test_embedding_dense_grad(self, device):
        with set_default_dtype(torch.double):
            embd = nn.Embedding(20, 20).to(device)
            weight = embd.weight

            def fn_wrapper(device):
                def fn(weight):
                    inp = torch.tensor(
                        [[0, 1, 1, 2], [3, 5, 7, 11]], dtype=torch.long
                    ).to(device)
                    return torch.nn.functional.embedding(inp, weight)

                return fn

            fn = fn_wrapper(device)
            _assertGradAndGradgradChecks(self, fn, (weight,))

    def test_embedding_scalar_weight_error(self, device):
        indices = torch.rand(2, 2, device=device).long()
        weights = [
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device).reshape(1, 1, 1),
        ]

        for weight in weights:
            with self.assertRaisesRegex(RuntimeError, "'weight' must be 2-D"):
                torch.nn.functional.embedding(indices, weight)

    @dtypesIfCUDA(torch.float16, torch.float64)
    @dtypesIfXPU(torch.float16, torch.float64)
    @dtypes(torch.float64)
    def test_embedding_backward(self, device, dtype):
        embedding = nn.Embedding(10, 3, sparse=True)
        tensor = torch.tensor([[7, 1, 3]])
        ones = torch.tensor(1.0, dtype=dtype).expand(3, 3)
        tensorTwice = tensor.repeat(1, 2)
        onesTwice = torch.cat((ones, ones))

        embedding = embedding.to(dtype=dtype).to(device)
        tensor = tensor.to(device)
        ones = ones.to(device)
        tensorTwice = tensorTwice.to(device)
        onesTwice = onesTwice.to(device)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensor)
        self.assertEqual(embedding.weight.grad._values(), ones)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        embedding(tensor[0]).sum().backward()
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

        embedding.zero_grad()
        embedding(tensor[0]).sum().backward()
        tensor[0, 0] = 8
        embedding(tensor[0]).sum().backward()
        tensorTwice[0, 3] = 8
        self.assertEqual(embedding.weight.grad._indices(), tensorTwice)
        self.assertEqual(embedding.weight.grad._values(), onesTwice)

    @dtypesIfCUDA(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    @dtypesIfXPU(torch.float32, torch.double, torch.half)
    @dtypes(torch.float32)
    def test_embedding_max_norm_backward(self, device, dtype):
        # can't use gradcheck since in place renorm makes analytical gradients different from produced ones
        weight = torch.randn((4, 4), device=device, dtype=dtype) * 2
        weight.requires_grad_()
        inp_list = [0, 1, 2, 2]
        inp = torch.tensor(inp_list, device=device)
        out = nn.functional.embedding(inp, weight, max_norm=1.0).sum()
        out.backward()

        expected_grad = (
            torch.tensor([[1.0, 1.0, 2.0, 0.0]], device=device, dtype=dtype)
            .transpose(0, 1)
            .expand(4, 4)
        )
        self.assertEqual(weight.grad, expected_grad)

    @dtypesIfCUDA(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    @dtypesIfXPU(torch.float32, torch.double, torch.half)
    @dtypes(torch.float32)
    def test_embedding_max_norm_fwd_AD(self, device, dtype):
        if torch.device(device).type == "xla":
            self.skipTest("forward AD doesn't work on xla")

        # can't use gradcheck since in place renorm makes analytical gradients different from produced ones
        weight = torch.randn((4, 4), device=device, dtype=dtype) * 2
        tangent = torch.ones((4, 4), device=device, dtype=dtype)
        inp = torch.tensor([[0, 1], [2, 2]], device=device)
        with torch.autograd.forward_ad.dual_level():
            dual_weight = torch.autograd.forward_ad.make_dual(weight, tangent)
            out = nn.functional.embedding(inp, dual_weight, max_norm=1.0)
            jvp = torch.autograd.forward_ad.unpack_dual(out).tangent

        expected_grad = torch.ones((2, 2, 4), device=device, dtype=dtype)
        self.assertEqual(jvp, expected_grad)

    @dtypesIfCUDA(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    @dtypesIfXPU(torch.float32, torch.double, torch.half)
    @dtypes(torch.float32)
    def test_embedding_padding_idx(self, device, dtype):
        embedding = nn.Embedding(10, 20, padding_idx=0).to(device, dtype)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=0, sparse=True).to(device, dtype)
        input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][0].sum(), 0)
        self.assertEqual(output[1][2].sum(), 0)

        # negative indexing check for padding_idx
        # padding_idx=-2, num_embeddings=10 ==> index 8 padded
        embedding = nn.Embedding(10, 20, padding_idx=-2).to(device, dtype)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        embedding = nn.Embedding(10, 20, padding_idx=-2, sparse=True).to(device, dtype)
        input = torch.tensor([[0, 2, 8, 5], [4, 8, 0, 9]], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[0][2].sum(), 0)
        self.assertEqual(output[1][1].sum(), 0)

        # change padding vector
        padding_vector = torch.ones(20, dtype=dtype, device=device)
        embedding = nn.Embedding(10, 20, padding_idx=2, sparse=True).to(device, dtype)
        with torch.no_grad():
            embedding.weight[2] = padding_vector
        input = torch.tensor([0, 2], dtype=torch.long).to(device)
        output = embedding(input)
        self.assertEqual(output[1], padding_vector)

        # out of bounds check for padding_idx
        self.assertRaises(
            AssertionError,
            nn.Embedding,
            num_embeddings=10,
            embedding_dim=20,
            padding_idx=25,
        )
        self.assertRaises(
            AssertionError,
            nn.Embedding,
            num_embeddings=10,
            embedding_dim=20,
            padding_idx=-25,
        )

        padding_idx = 0
        embedding = nn.Embedding(5, 2, padding_idx=padding_idx).to(device, dtype)
        for n in (
            1,
            2,
            1000,
        ):  # Need large N to trigger all the methods we have implemented
            for other_indices in ([], [1, 3], [2]):
                indices = torch.tensor(
                    other_indices + [padding_idx] * n, dtype=torch.long
                ).to(device)
                pre = embedding.weight[padding_idx].clone()
                embedding(indices).sum().backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

                # test double backward
                emb_sum = embedding(indices).sum()
                emb_grad = torch.autograd.grad(
                    outputs=emb_sum,
                    inputs=list(embedding.parameters()),
                    retain_graph=True,
                )
                scalar = emb_grad[0].sum() + emb_sum
                scalar.backward()
                after = (embedding.weight + embedding.weight.grad)[padding_idx]
                embedding.zero_grad()
                self.assertEqual(after, pre)

    # Check correctness of torch.nn.functional.embedding_bag forward and
    # backward functions with padding_idx, given a 1D input separated into bags
    # with an offset array. Compare against an equivalent 2D input that uses
    # padding indices to fill in the gaps indicated by the offset array

    @skipIfTorchDynamo("see https://github.com/pytorch/pytorch/pull/95621")
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.half, torch.bfloat16)
    @dtypesIfXPU(torch.half, torch.bfloat16)
    def test_embedding_bag_1D_padding_idx(self, device, dtype):
        num_features = 3
        max_indices_per_bag = 10
        num_bags = 10
        num_words = 100

        def gen_1D_indices_offsets(include_last_offset, allpad):
            indices = []
            offsets = []
            cur_offset = 0

            # Make one bag full and one bag empty, for extra coverage
            empty_bag = random.randint(0, num_bags - 1)
            full_bag = empty_bag
            while full_bag == empty_bag:
                full_bag = random.randint(0, num_bags - 1)

            for bag in range(num_bags):
                offsets.append(cur_offset)
                if bag == full_bag:
                    bag_size = max_indices_per_bag
                elif bag == empty_bag:
                    bag_size = 0
                else:
                    bag_size = random.randint(1, max_indices_per_bag - 1)
                indices += [
                    1 if allpad else random.randint(0, num_words - 1)
                    for _ in range(bag_size)
                ]
                cur_offset += bag_size

            # embedding_bag requires first entry of offsets to be 0
            assert offsets[0] == 0

            indices = torch.tensor(indices, device=device)

            if include_last_offset:
                offsets.append(indices.size(0))

            offsets = torch.tensor(offsets, device=device)

            return indices, offsets

        # Convert a 1-D indices-offsets representation into 2-D. Fill any empty
        # indices with padding_idx
        def gen_2D_indices_from_1D(
            indices_1D, offsets, include_last_offset, padding_idx
        ):
            assert offsets[0] == 0
            if include_last_offset:
                offsets = offsets[:-1]
            indices_2D = torch.empty(
                num_bags, max_indices_per_bag, device=device, dtype=torch.long
            )
            for bag in range(num_bags):
                # Determine the start and end position of the bag within indices_1D
                start = offsets[bag]
                end = len(indices_1D) if bag + 1 == num_bags else offsets[bag + 1]
                end = min(len(indices_1D), end)

                # Pull out the bag's indices from indices_1D, and fill any
                # remaining space with padding indices
                indices_in_bag = []
                for item_pos in range(max_indices_per_bag):
                    if (start + item_pos) < end:
                        indices_in_bag.append(indices_1D[start + item_pos])
                    else:
                        indices_in_bag.append(padding_idx)
                indices_2D[bag] = torch.tensor(indices_in_bag, device=device)

            return indices_2D

        test_cases = product(
            ["max", "mean", "sum"], [False, True], [False, True], [False, True]
        )

        for mode, sparse, include_last_offset, allpad in test_cases:
            # Max sparse and bfloat16 are not supported
            if mode == "max":
                if sparse or (dtype == torch.bfloat16):
                    continue
            indices_1D, offsets = gen_1D_indices_offsets(include_last_offset, allpad)
            for padding_idx_1D in list(set(indices_1D.tolist())) + [None]:
                msg = (
                    f"mode: '{mode}', sparse: {sparse}, include_last_offset: {include_last_offset}, "
                    f"padding_idx_1D: {padding_idx_1D}"
                )

                # If 1D input does not use a padding index, we still need one for the 2D input,
                # so we can add one dummy word to the weights to act as the padded word
                padding_idx_2D = (
                    padding_idx_1D if padding_idx_1D is not None else num_words
                )
                num_words_with_padding = (
                    num_words if padding_idx_1D is not None else num_words + 1
                )

                indices_2D = gen_2D_indices_from_1D(
                    indices_1D, offsets, include_last_offset, padding_idx_2D
                )

                weights = torch.randn(
                    num_words_with_padding,
                    num_features,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                weights_check = weights.detach().clone().requires_grad_(True)

                bag = torch.nn.functional.embedding_bag(
                    indices_1D,
                    weights,
                    offsets,
                    padding_idx=padding_idx_1D,
                    mode=mode,
                    sparse=sparse,
                    include_last_offset=include_last_offset,
                )

                bag_check = torch.nn.functional.embedding_bag(
                    indices_2D,
                    weights_check,
                    padding_idx=padding_idx_2D,
                    mode=mode,
                    sparse=sparse,
                )
                self.assertEqual(bag, bag_check, msg=msg)

                bag.sum().backward()
                bag_check.sum().backward()

                # Sometimes, half dtype gradients mismatch by a greater amount
                # than other dtypes
                if dtype in [torch.half, torch.bfloat16]:
                    atol = 0.01
                    rtol = 0.01
                else:
                    atol = None
                    rtol = None
                self.assertEqual(
                    weights.grad, weights_check.grad, msg=msg, atol=atol, rtol=rtol
                )

    @onlyOn(["cuda", "xpu"])
    @dtypes(
        torch.bfloat16,
    )
    @largeTensorTest("80GB", device="cuda")
    @largeTensorTest("80GB", device="xpu")
    def test_embedding_backward_large_batch_overflow(self, device, dtype):
        """
        Test that embedding_dense_backward handles large batches that exceed INT32_MAX thread IDs.

        This reproduces the bug where gid = blockIdx.x * blockDim.x + threadIdx.x overflows
        when declared as int32, causing negative indices and illegal memory access.
        """
        # Parameters chosen to GUARANTEE int32 overflow
        num_indices = 8_214_880
        embedding_dim = 4096
        num_weights = 1280
        padding_idx = -1
        scale_grad_by_freq = False

        # Verify parameters guarantee overflow
        NROWS_PER_THREAD = 10
        max_segments = min(num_indices, num_weights)
        min_partial_for_overflow = (2**31) // 4096
        required_indices = (min_partial_for_overflow - max_segments) * NROWS_PER_THREAD

        assert num_indices > required_indices, (
            f"Test bug: num_indices={num_indices:,} too small! Need >{required_indices:,}"
        )

        # Generate indices that create many partial segments
        # Strategy: ~950 unique indices, each appearing many times
        num_unique = 954
        unique_indices = torch.randint(
            2, 1276, (num_unique,), dtype=torch.int64, device=device
        )
        counts = torch.randint(
            5000, 12000, (num_unique,), dtype=torch.int64, device=device
        )

        # Normalize to exactly num_indices
        counts = (counts.float() / counts.float().sum() * num_indices).long()
        counts[-1] = num_indices - counts[:-1].sum()

        indices = torch.repeat_interleave(unique_indices, counts)
        assert indices.numel() == num_indices

        # Verify we'll trigger overflow
        approx_partial_segments = num_indices // NROWS_PER_THREAD + max_segments
        stride_warped = ((embedding_dim + 31) // 32) * 32
        total_threads = approx_partial_segments * stride_warped

        assert total_threads > 2**31 - 1, (
            f"Test bug: threads={total_threads:,} <= INT32_MAX, won't trigger overflow!"
        )

        # Create gradient output
        grad_output = torch.randn(
            num_indices, embedding_dim, dtype=dtype, device=device
        )

        # This should complete without error (after fix)
        # Before fix: RuntimeError with "illegal memory access"
        grad_weight = torch.ops.aten.embedding_dense_backward(
            grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
        )

        # Verify output shape
        assert grad_weight.shape == (num_weights, embedding_dim)
        assert grad_weight.dtype == torch.bfloat16

    # Check correctness of torch.nn.functional.embedding_bag forward and
    # backward functions with padding_idx, given a 2D indices input. Compare
    # against torch.nn.functional.embedding followed by a reduction.
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.half, torch.bfloat16)
    @dtypesIfXPU(torch.half, torch.bfloat16)
    def test_embedding_bag_2D_padding_idx(self, device, dtype):
        # Use a Python implementation of embedding_bag with padding_idx support
        # to check torch.nn.functional.embedding_bag correctness
        def embedding_bag_check(indices, weights, mode, sparse, padding_idx):
            assert padding_idx is not None
            embedding = torch.nn.functional.embedding(
                indices, weights, padding_idx=padding_idx, sparse=sparse
            )

            reduction_dim = indices.dim() - 1

            if mode == "sum" or mode == "mean":
                # We must avoid including elements at padding_idx in the
                # sum/mean, so multiply those elements by 0, and multiply
                # all other elements by 1
                per_sample_weights = indices.ne(padding_idx).to(dtype).unsqueeze(-1)
                res = embedding.mul(per_sample_weights).sum(dim=reduction_dim)

                if mode == "mean":
                    weights_sum = per_sample_weights.sum(dim=reduction_dim)
                    res = res.div(weights_sum)

            elif mode == "max":
                # We must avoid allowing elements at padding_idx to be chosen
                # as the max, so set those elements to negative infinity
                res = embedding.masked_fill(
                    indices.unsqueeze(-1) == padding_idx, -float("inf")
                ).amax(dim=reduction_dim)

            else:
                raise RuntimeError(f"mode '{mode}' is not available")

            # If a row is all padding, set its corresponding result row to 0.
            # This is needed because the above mean and max mode
            # implementations set these elements to nan and -inf, respectively
            if mode in ["mean", "max"]:
                res = res.masked_fill(
                    indices.eq(padding_idx).all(dim=-1).unsqueeze(-1), 0
                )

            return res

        num_features = 3
        num_words = 10
        indices_dim1 = 10

        for mode, sparse, allpad, indices_dim0 in product(
            ["max", "mean", "sum"], [False, True], [False, True], [1, 10]
        ):
            # Max sparse and bfloat16 are not supported
            if mode == "max":
                if sparse or (dtype == torch.bfloat16):
                    continue

            if allpad:
                indices = torch.empty(
                    indices_dim0, indices_dim1, dtype=torch.long, device=device
                ).fill_(1)
            else:
                indices = torch.randint(
                    0, num_words, (indices_dim0, indices_dim1), device=device
                )

                if indices_dim0 > 1:
                    # Fill one row with duplicate index so we can test with a fully
                    # padded row
                    duplicate_row = random.randint(0, indices_dim0 - 1)
                    indices[duplicate_row] = indices[duplicate_row][0]

            for padding_idx in list(set(indices.flatten(0, -1).tolist())):
                weights = torch.randn(
                    num_words,
                    num_features,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                weights_check = weights.detach().clone().requires_grad_(True)

                msg = (
                    f"mode: '{mode}', sparse: {sparse}, padding_idx: {padding_idx}, "
                    f"allpad: {allpad}, indices.size(): {indices.size()}"
                )

                # Check forward with a Python implementation of padding_idx embedding_bag
                bag_check = embedding_bag_check(
                    indices, weights_check, mode, sparse, padding_idx
                )
                bag = torch.nn.functional.embedding_bag(
                    indices, weights, padding_idx=padding_idx, mode=mode, sparse=sparse
                )

                self.assertEqual(bag, bag_check, msg=msg)

                bag_check.sum().backward()
                grad_check = weights_check.grad

                bag.sum().backward()
                grad = weights.grad

                # Sometimes, half dtype gradients mismatch by a greater amount
                # than other dtypes
                if dtype in [torch.half, torch.bfloat16]:
                    atol = 0.01
                    rtol = 0.01
                else:
                    atol = None
                    rtol = None
                self.assertEqual(grad, grad_check, msg=msg, atol=atol, rtol=rtol)

    @onlyOn(["cuda", "xpu"])
    @dtypes(
        *(
            (torch.float, torch.double, torch.bfloat16, torch.half)
            if TEST_WITH_ROCM
            else (torch.float, torch.double, torch.half)
        )
    )
    def test_embedding_max_norm_device(self, device, dtype):
        embedding = nn.Embedding(22, 5, max_norm=1.0).to(device, dtype=dtype)
        # nn.Embedding only takes LongTensor as input
        input = torch.tensor([2, 8, 8, 6], device=device, dtype=torch.long)
        output = embedding(input)
        self.assertEqual(output[1], output[2])
        self.assertTrue(output.data.norm(p=2, dim=1).le(1).all())

    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_embedding_bag_empty_input(self, device, dtypes):
        m = 4
        n = 3
        x = torch.tensor([], device=device, dtype=dtypes[0])
        for sparse in [True, False]:
            Embed = torch.nn.EmbeddingBag(m, n, sparse=sparse)
            Embed.to(device)

            output = Embed(
                input=x, offsets=torch.tensor([0], device=device, dtype=dtypes[1])
            )
            self.assertEqual(output, torch.zeros_like(output))

            output = Embed(
                input=x, offsets=torch.tensor([0, 0], device=device, dtype=dtypes[1])
            )
            self.assertEqual(output, torch.zeros_like(output))

    @skipCUDAIf(True, "no out-of-bounds check on CUDA for perf.")
    @skipXPUIf(True, "no out-of-bounds check on XPU for perf.")
    @dtypes(*itertools.product((torch.float, torch.double), (torch.int, torch.long)))
    @parametrize_test("padding_idx", [None, 0])
    @parametrize_test("mode", ["sum", "mean", "max"])
    def test_embedding_bag_out_of_bounds_idx(self, device, dtypes, padding_idx, mode):
        padding_idx = 0
        w_dtype, idx_dtype = dtypes
        # negative out-of-bound
        idx1 = torch.tensor([[-1, 1]], device=device, dtype=idx_dtype)
        # positive out-of-bound
        idx2 = torch.tensor([[11, 8]], device=device, dtype=idx_dtype)
        weight = torch.randn(10, 2, device=device, dtype=w_dtype)
        if mode == "sum":
            # Only `sum` supports per_sample_weight
            per_sample_weights = (
                None,
                torch.randn_like(idx1, device=device, dtype=w_dtype),
            )
        else:
            per_sample_weights = (None,)

        for p_s_weights, idx in itertools.product(per_sample_weights, (idx1, idx2)):
            msg = "Expected idx >= 0 && idx < num_embeddings"
            with self.assertRaisesRegex(RuntimeError, msg):
                torch.nn.functional.embedding_bag(
                    idx,
                    weight,
                    per_sample_weights=p_s_weights,
                    padding_idx=padding_idx,
                    mode=mode,
                )

    def test_embedding_bag_dimension_errors(self, device):
        funcs = (
            lambda x, y, z: torch.nn.functional.embedding_bag(y, x, z),
            torch.embedding_bag,
            torch._embedding_bag,
            torch._embedding_bag_forward_only,
        )
        for i, f in enumerate(funcs):
            err_type = (ValueError, RuntimeError) if i == 0 else RuntimeError

            weight = torch.full(
                (
                    2,
                    6,
                ),
                0,
                dtype=torch.float64,
                device=device,
            )
            indices = torch.full(
                (
                    2,
                    0,
                    0,
                    6,
                    6,
                ),
                2,
                dtype=torch.int64,
                device=device,
            )
            offsets = torch.full((2, 0, 0, 6, 6), 0, dtype=torch.int64, device=device)

            if i == 0:
                error_msg = "input has to be 1D or 2D Tensor"
            else:
                error_msg = "input has to be a 1D or 2D Tensor"
            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type, error_msg, lambda: f(weight, indices, offsets)
            )

            weight = torch.full((2, 2), 0, dtype=torch.float64, device=device)
            indices = torch.full((2,), 1, dtype=torch.int64, device=device)

            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type,
                "offsets has to be a 1D Tensor",
                lambda: f(weight, indices, offsets),
            )

            weight = torch.full((2, 2, 2), 0, dtype=torch.float64, device=device)
            indices = torch.full((2,), 2, dtype=torch.int64, device=device)
            offsets = torch.full((2,), 0, dtype=torch.int64, device=device)

            torch._dynamo.disable(self.assertRaisesRegex)(
                err_type,
                "weight has to be a 2D Tensor",
                lambda: f(weight, indices, offsets),
            )

    @dtypes(*itertools.product((torch.int, torch.long), (torch.int, torch.long)))
    def test_EmbeddingBag_per_sample_weights_failures(self, device, dtypes):
        # Failure 1: mismatched embeddings / per_sample_weights dtype (only on CPU device)
        es = nn.EmbeddingBag(5, 2, mode="sum").to(dtype=torch.float, device=device)
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=dtypes[1], device=device)
        per_sample_weights = torch.randn_like(input, dtype=torch.double, device=device)
        if device == "cpu":
            with self.assertRaisesRegex(RuntimeError, "have the same type as"):
                es(input, offsets, per_sample_weights)

        # Failure 2.1: input/per_sample_weights have different sizes (1d input)
        input = torch.tensor([3, 1, 1, 1, 4, 0], dtype=dtypes[0], device=device)
        offsets = torch.tensor([0, 0, 3, 3, 6], dtype=dtypes[1], device=device)
        per_sample_weights = torch.randn(5, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, "same shape as the input"):
            es(input, offsets, per_sample_weights)

        # Failure 2.2: input/per_sample_weights have different sizes (2d input)
        input = torch.randint(5, (7, 3), dtype=dtypes[0], device=device)
        offsets = None
        per_sample_weights = torch.randn(7 * 3, dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, "same shape as the input"):
            es(input, offsets, per_sample_weights)

        # Failure 3: Unsupported per_sample_weights and mode=('max', 'mean')
        for unsupported_mode in ("max", "mean"):
            es = nn.EmbeddingBag(5, 2, mode=unsupported_mode).to(
                dtype=torch.float, device=device
            )
            input = torch.randint(5, (7, 3), dtype=dtypes[0], device=device)
            offsets = None
            per_sample_weights = torch.randn(7, 3, dtype=torch.float, device=device)
            with self.assertRaisesRegex(
                NotImplementedError, "only supported for mode='sum'"
            ):
                es(input, offsets, per_sample_weights)

    def _embedding_bag_reference_impl(
        self,
        input,
        weight,
        offsets=None,
        mode="sum",
        per_sample_weights=None,
        include_last_offset=False,
    ):
        assert mode == "sum" or per_sample_weights is None
        assert offsets is not None
        if per_sample_weights is None:
            per_sample_weights = torch.ones(input.size()).to(
                dtype=weight.dtype, device=weight.device
            )
        assert input.numel() == per_sample_weights.numel()

        bags = []
        long_input = input.to(torch.long)
        embeddings = weight.index_select(0, long_input) * per_sample_weights.unsqueeze(
            1
        )
        if include_last_offset:
            for index in range(len(offsets) - 1):
                offset = offsets[index]
                next_offset = offsets[index + 1]
                length = next_offset - offset
                if length == 0:
                    bags.append(
                        torch.tensor([0] * weight.size(1)).to(
                            dtype=embeddings.dtype, device=embeddings.device
                        )
                    )
                else:
                    if mode == "sum":
                        bags.append(embeddings.narrow(0, offset, length).sum(0))
                    elif mode == "mean":
                        bags.append(
                            embeddings.narrow(0, offset, length).sum(0).div(length)
                        )
                    else:
                        assert mode == "max"
                        bags.append(embeddings.narrow(0, offset, length).max(0)[0])
        else:
            for index, offset in enumerate(offsets):
                if index + 1 < len(offsets):
                    next_offset = offsets[index + 1]
                else:
                    next_offset = len(long_input)
                length = next_offset - offset
                if length == 0:
                    bags.append(
                        torch.tensor([0] * weight.size(1)).to(
                            dtype=embeddings.dtype, device=embeddings.device
                        )
                    )
                else:
                    if mode == "sum":
                        bags.append(embeddings.narrow(0, offset, length).sum(0))
                    elif mode == "mean":
                        bags.append(
                            embeddings.narrow(0, offset, length).sum(0).div(length)
                        )
                    else:
                        assert mode == "max"
                        bags.append(embeddings.narrow(0, offset, length).max(0)[0])
        return torch.stack(bags)

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.half, torch.bfloat16, torch.float, torch.double),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    @dtypesIfXPU(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float32, torch.double, torch.half),
        )
    )
    def test_EmbeddingBag_empty_per_sample_weights_and_offsets(self, device, dtypes):
        # Test empty input and per sample weight, and backward pass. There was a CUDA
        # invalid configuration bug (more context in #46572)
        def test_per_sample_weights(mode, trainable_scale):
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            input = torch.tensor([], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 0, 0, 0], device=device, dtype=dtypes[1])
            per_sample_weights = torch.randn_like(
                input, dtype=dtypes[2]
            ).requires_grad_(trainable_scale)
            ref_per_sample_weights = per_sample_weights.detach().requires_grad_(
                trainable_scale
            )
            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights
            )
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(
                result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0
            )

            grad = torch.randn_like(expected)
            result.backward(grad)
            # the reference impl doesn't have grad fn for empty input; but the grad should
            # simply be a zero tensor
            ref_weights_grad = torch.zeros_like(es.weight)
            self.assertEqual(
                es.weight.grad,
                ref_weights_grad,
                atol=dtype2prec_DONTUSE[dtypes[2]],
                rtol=0,
            )
            if trainable_scale:
                ref_per_sample_weights_grad = torch.empty_like(per_sample_weights)
                self.assertEqual(
                    per_sample_weights.grad,
                    ref_per_sample_weights_grad,
                    atol=dtype2prec_DONTUSE[dtypes[2]],
                    rtol=0,
                )

        modes = ("sum",)
        trainable_scale = (True, False)
        for mode, trainable in itertools.product(modes, trainable_scale):
            test_per_sample_weights(mode, trainable)

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half, torch.bfloat16),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    @dtypesIfXPU(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float32, torch.double, torch.half),
        )
    )
    def test_EmbeddingBag_per_sample_weights_and_offsets(self, device, dtypes):
        def test_per_sample_weights(mode, trainable_scale):
            es = nn.EmbeddingBag(5, 2, mode=mode).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=dtypes[1])
            per_sample_weights = torch.randn_like(
                input, dtype=dtypes[2]
            ).requires_grad_(trainable_scale)
            ref_per_sample_weights = per_sample_weights.detach().requires_grad_(
                trainable_scale
            )
            reference_weights = es.weight.detach().requires_grad_()

            expected = self._embedding_bag_reference_impl(
                input, reference_weights, offsets, mode, ref_per_sample_weights
            )
            result = es(input, offsets, per_sample_weights)
            self.assertEqual(
                result, expected, atol=dtype2prec_DONTUSE[dtypes[2]], rtol=0
            )

            grad = torch.randn_like(expected).to(dtype=dtypes[2], device=device)
            result.backward(grad)
            expected.backward(grad)
            self.assertEqual(
                es.weight.grad,
                reference_weights.grad,
                atol=dtype2prec_DONTUSE[dtypes[2]],
                rtol=0,
            )
            if trainable_scale:
                self.assertEqual(
                    per_sample_weights.grad,
                    ref_per_sample_weights.grad,
                    atol=dtype2prec_DONTUSE[dtypes[2]],
                    rtol=0,
                )

        modes = ("sum",)
        trainable_scale = (True, False)
        for mode, trainable in itertools.product(modes, trainable_scale):
            test_per_sample_weights(mode, trainable)

    @skipMeta
    @dtypes(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half, torch.bfloat16),
        )
    )
    @dtypesIfCUDA(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float, torch.double, torch.half),
        )
    )
    @dtypesIfXPU(
        *itertools.product(
            (torch.int, torch.long),
            (torch.int, torch.long),
            (torch.float32, torch.double, torch.half),
        )
    )
    def test_EmbeddingBag_per_sample_weights_and_new_offsets(self, device, dtypes):
        def test_per_sample_weights_new_offsets(
            mode, trainable_scale, include_last_offset, has_weight=True
        ):
            es = nn.EmbeddingBag(
                5, 2, mode=mode, include_last_offset=include_last_offset
            ).to(dtype=dtypes[2], device=device)
            es.weight.data.copy_(
                torch.arange(1, 11, device=device).view_as(es.weight).to(dtypes[2])
            )
            input = torch.tensor([3, 1, 1, 1, 4, 0], device=device, dtype=dtypes[0])
            offsets = torch.tensor([0, 0, 3, 3, 6], device=device, dtype=dt
```



## High-Level Overview


This Python file contains 2 class(es) and 53 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestEmbeddingNN`, `TestEmbeddingNNDeviceType`

**Functions defined**: `test_embedding_max_norm_unsorted_repeating_indices`, `create_embedding`, `test_embedding_sparse_basic`, `test_embedding_sparse_empty_tensor`, `test_move_sparse_half_embedding`, `test_embedding_max_norm`, `test_embedding_from_pretrained`, `test_embedding_bag_from_pretrained`, `test_embedding_from_pretrained_padding_idx`, `test_embedding_bag_from_pretrained_padding_idx`, `test_embedding_from_pretrained_options`, `test_embedding_functional`, `test_large_tensors`, `test_embedding_bag_functional`, `test_embedding_bag_padding_idx_error`, `test_embeddingbag_from_pretrained`, `test_embeddingbag_from_pretrained_options`, `test_embeddingbag_include_last_offset`, `test_embedding_dense_grad`, `fn_wrapper`

**Key imports**: itertools, random, unittest, product, torch, torch.nn as nn, torch.nn.functional as F, NNTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/nn`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `random`
- `unittest`
- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.testing._internal.common_nn`: NNTestCase


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/nn/test_embedding.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/nn`):

- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)
- [`test_dropout.py_docs.md`](./test_dropout.py_docs.md)
- [`test_convolution.py_docs.md`](./test_convolution.py_docs.md)
- [`test_lazy_modules.py_docs.md`](./test_lazy_modules.py_docs.md)
- [`test_packed_sequence.py_docs.md`](./test_packed_sequence.py_docs.md)
- [`test_multihead_attention.py_docs.md`](./test_multihead_attention.py_docs.md)
- [`test_init.py_docs.md`](./test_init.py_docs.md)
- [`test_module_hooks.py_docs.md`](./test_module_hooks.py_docs.md)
- [`test_pruning.py_docs.md`](./test_pruning.py_docs.md)


## Cross-References

- **File Documentation**: `test_embedding.py_docs.md`
- **Keyword Index**: `test_embedding.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
