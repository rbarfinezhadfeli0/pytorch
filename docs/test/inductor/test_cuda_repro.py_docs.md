# Documentation: `test/inductor/test_cuda_repro.py`

## File Metadata

- **Path**: `test/inductor/test_cuda_repro.py`
- **Size**: 100,035 bytes (97.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
# ruff: noqa: F841

import copy
import functools
import gc
import math
import os
import sys
import unittest

import torch
import torch._dynamo.config as dynamo_config
import torch.backends.cuda
import torch.nn.functional as F
from torch import nn
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import (
    run_and_get_code,
    run_and_get_graph_lowering,
    run_fw_bw_and_get_code,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM80OrLater,
    SM90OrLater,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    MI350_ARCH,
    parametrize,
    skipIfRocmArch,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    xfailIfPy312Plus,
)
from torch.testing._internal.inductor_utils import IS_BIG_GPU


if TEST_WITH_ROCM:
    config.force_layout_optimization = 1
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)
from torch.testing._internal.inductor_utils import skipCUDAIf


try:
    try:
        import triton  # @manual
        from triton import language as tl  # @manual
    except ImportError:
        raise unittest.SkipTest("requires triton")  # noqa: B904

    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


TestCase = test_torchinductor.TestCase
ToTuple = test_torchinductor.ToTuple
check_model_cuda = test_torchinductor.check_model_cuda
aten = torch.ops.aten


@instantiate_parametrized_tests
class CudaReproTests(TestCase):
    device = "cuda"
    common = check_model_cuda

    def test_mm_out_dtype_compile(self):
        a = torch.randn(1, 3, device="cuda", dtype=torch.float16)
        b = torch.randn(3, 2, device="cuda", dtype=torch.float16)

        def fn(x, y):
            return torch.mm(x, y, out_dtype=torch.float32)

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result, expected)

    def test_index_put_issue(self):
        def forward(
            self,
            arg76_1,
            expand_default,
            full_like_default,
            _to_copy_default_67,
            zeros,
        ):
            sum_sym_int_19 = torch.ops.aten.sum(_to_copy_default_67, [0], True)
            view_default_57 = torch.ops.aten.view.default(sum_sym_int_19, [512, 768])
            where_self = torch.ops.aten.where.self(
                expand_default, view_default_57, full_like_default
            )
            clone_default_12 = torch.ops.aten.clone.default(zeros)
            index_put__default = torch.ops.aten.index_put_.default(
                clone_default_12, [arg76_1], where_self, True
            )
            return (index_put__default,)

        inps = [
            (torch.Size([512]), torch.int64),
            (torch.Size([512, 768]), torch.bool),
            (torch.Size([512, 768]), torch.float16),
            (torch.Size([4, 512, 768]), torch.float16),
            (torch.Size([512, 768]), torch.float16),
        ]
        inps = [torch.zeros(())] + [
            torch.ones(shape, dtype=dtype, device="cuda") for (shape, dtype) in inps
        ]
        mod = make_fx(forward)(*inps)
        compiled = compile_fx_inner(mod, inps)
        compiled(inps)

    def test_view_replay_padding_issue_163328(self):
        class ReproModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_points_out = 120
                self.lc_num = 2
                input_channels = 16
                self.linear_main = nn.Linear(input_channels, self.num_points_out * 2)
                self.linear_lc = nn.Linear(input_channels, self.num_points_out * 2)

            def forward(self, x: torch.Tensor):
                bs, num_lat, num_lon, channels = x.shape
                index = num_lat - self.lc_num

                main_x = x[:, :index].reshape(bs * index * num_lon, channels)
                lc_x = x[:, index:].reshape(bs * self.lc_num * num_lon, channels)

                refline = self.linear_main(main_x).reshape(bs, index, num_lon, -1)
                lc_refline = self.linear_lc(lc_x).reshape(bs, self.lc_num, num_lon, -1)

                base = torch.cat([refline, lc_refline], dim=1).contiguous()
                out0 = base.reshape(bs, num_lat, num_lon, self.num_points_out, 2)
                out1 = base.reshape(bs, num_lat * num_lon, self.num_points_out * 2)
                return {"ten0": out0, "ten1": out1}

        torch.manual_seed(0)
        model = ReproModule().cuda()
        inputs = torch.randn(36, 9, 7, 16, device="cuda", requires_grad=True)

        eager_out = model(inputs)
        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
        )
        compiled_out = compiled_model(inputs)

        self.assertEqual(compiled_out["ten0"], eager_out["ten0"])
        self.assertEqual(compiled_out["ten1"], eager_out["ten1"])

    def test_effn_attn_bias_padding(self):
        batch_size, num_heads, seq_len, head_dim = 2, 32, 512, 128

        def fn(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            input_tensor: torch.Tensor,  # This will be our starting point
        ):
            # Input tensor should be [2, 1, 8192, 1] with appropriate strides
            bias = torch.ops.aten.expand(
                input_tensor, [2, 32, seq_len, seq_len]
            )  # Expands with stride pattern [65536, 0, 8, 0]

            return torch.ops.aten._scaled_dot_product_efficient_attention(
                query,
                key,
                value,
                bias,
                compute_log_sumexp=True,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            )

        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")

        input_tensor = torch.rand([2, 1, seq_len, 1], device="cuda")

        out, code = run_and_get_code(torch.compile(fn), query, key, value, input_tensor)

        input_tensor2 = torch.rand([2, 32, seq_len, seq_len], device="cuda").copy_(
            input_tensor
        )
        # even though the last dim is broadcasted, needs stride 1 for alignment
        # but dim 1 stride can be 0
        FileCheck().check("buf0").check("(262144, 0, 512, 1").run(code[0])

        # dont check rng state
        self.assertEqual(out[:2], fn(query, key, value, input_tensor2)[:2])

    @skipIfRocmArch(MI350_ARCH)
    def test_effn_attn_bias_padding_misaligned(self):
        seqlen_start = 1008

        for offset in range(-1, 2):
            seqlen = seqlen_start + offset
            torch._dynamo.reset()

            bsz = 32
            q = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16, device="cuda")
            mask = torch.ones([bsz, 1, seqlen, seqlen], dtype=torch.bool, device="cuda")
            inputs = [q, k, v, mask]

            def f(q, k, v, mask):
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return F.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, dropout_p=0.0
                    )

            f_compiled = torch.compile(f)

            out, code = run_and_get_code(f_compiled, *inputs)
            # padded bias should have an expanded dim
            FileCheck().check("buf0 =").check_same(", 0, ").run(code[0])
            # single fused padded kernel
            FileCheck().check_count("empty_strided_cuda(", 1, exactly=True).check(
                "return"
            ).run(code[0])

            self.assertEqual(out, f(*inputs))

    def test_input_channels_last(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
            ToTuple(),
        ).cuda()
        inp = torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last).cuda()

        self.common(
            m,
            (inp,),
            check_lowp=False,
        )

        @torch.compile()
        def foo(m, inp):
            return m(inp)

        self.assertTrue(foo(m, inp)[0].is_contiguous(memory_format=torch.channels_last))

    # https://github.com/pytorch/torchdynamo/issues/1681#issuecomment-1283433527
    def test_unspec_inputs_interop(self):
        class Repro(torch.nn.Module):
            def forward(self, x, y):
                unsqueeze = torch.ops.aten.unsqueeze.default(x, 4)
                permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 2, 4, 3])
                add = torch.ops.aten.add.Tensor(y, 1)
                return [permute, add]

        inps = [
            rand_strided((12, 3, 512, 64), (64, 196608, 768, 1), torch.float32, "cuda"),
            rand_strided((), (), torch.int64, "cpu"),
        ]
        mod = make_fx(Repro().to(device="cuda"))(*inps)
        compiled = compile_fx_inner(mod, inps)
        compiled(inps)

    @unittest.skipIf(
        IS_FBCODE, "RuntimeError: Triton Error [CUDA]: invalid device context"
    )
    def test_backward_context(self):
        def fn(x):
            return x * 3

        x = torch.randn(4, device="cuda", requires_grad=True)
        gO = torch.rand_like(x)
        opt_fn = torch.compile(fn)
        out = opt_fn(x)
        out.backward(gO)

    @config.patch(fallback_random=True)
    def test_dtype_factory_issue(self):
        def forward():
            randn = torch.ops.aten.randn.default(
                [12, 64, 1, 64],
                dtype=torch.float32,
                device=torch.device(type="cuda", index=0),
                pin_memory=False,
            )
            unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(randn, -1)
            return (unsqueeze_default_2,)

        mod = make_fx(forward)()
        compiled = compile_fx_inner(mod, ())
        assert compiled([])[0].device.type == "cuda"

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_no_device_idx_repro_cudagraphs(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self):
                full = torch.ops.aten.full.default(
                    [8, 512],
                    1,
                    dtype=torch.float32,
                    layout=torch.strided,
                    device=torch.device(type="cuda", index=0),
                    pin_memory=False,
                )
                full_1 = torch.ops.aten.full.default(
                    [8, 512],
                    0,
                    dtype=torch.int64,
                    layout=torch.strided,
                    device=torch.device(type="cuda", index=0),
                    pin_memory=False,
                )
                return (full_1, full)

        self.common(Repro(), ())

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs(self):
        @torch.compile(backend="inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(
        automatic_dynamic_shapes=True,
        assume_static_by_default=False,
    )
    def test_dynamic_to_static_cudagraphs(self):
        for b in [False, True]:
            with config.patch({"triton.cudagraph_trees": b}):

                @torch.compile(backend="inductor")
                def fn(x, y):
                    r = x + y
                    return r, r.size(0)

                inputs = (
                    torch.randn((5, 5), device="cuda"),
                    torch.randn((5, 5), device="cuda"),
                )
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 5)))

                inputs = (
                    torch.randn((6, 6), device="cuda"),
                    torch.randn((6, 6), device="cuda"),
                )
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 6)))

    def _test_split_reduction_impl(self, x):
        def max(x):
            return torch.max(x)

        max_c = torch.compile(max)

        out, code = run_and_get_code(max_c, x)
        self.assertEqual(out, max(x))

        if DO_PERF_TEST:
            ms_c = benchmarker.benchmark_gpu(lambda: max_c(x))
            ms_eager = benchmarker.benchmark_gpu(lambda: max(x))
            print(f"compile {ms_c=:.03f}, eager {ms_eager=:.03f}")

    def test_split_reduction_transposed(self):
        x = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")
        x = x.t().contiguous().t()

        self._test_split_reduction_impl(x)

    def test_split_reduction_channels_last(self):
        x = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")
        x = x.reshape([256, 256, 256, 2]).to(memory_format=torch.channels_last)

        self._test_split_reduction_impl(x)

    @config.patch({"emulate_precision_casts": True})
    def test_bool_emulate_low_precision(self):
        from torch import device

        inf = float("inf")

        def forward():
            full_1 = torch.ops.aten.full.default(
                [6, 6],
                1,
                dtype=torch.float32,
                layout=torch.strided,
                device=device(type="cpu"),
                pin_memory=False,
            )
            device_put_3 = torch.ops.prims.device_put.default(
                full_1, device(type="cuda", index=0)
            )
            full_1 = None

            convert_element_type_40 = torch.ops.prims.convert_element_type.default(
                device_put_3, torch.bool
            )
            device_put_3 = None
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(convert_element_type_40, 1)
            convert_element_type_40 = None
            unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
            unsqueeze_4 = None
            expand = torch.ops.aten.expand.default(unsqueeze_5, [-1, 256, -1, 256])
            unsqueeze_5 = None
            clone = torch.ops.aten.clone.default(
                expand, memory_format=torch.contiguous_format
            )
            expand = None
            view_15 = torch.ops.aten.reshape.default(clone, [1536, 1536])
            clone = None
            scalar_tensor = torch.ops.aten.scalar_tensor.default(
                -inf, dtype=torch.float16, device=device(type="cuda", index=0)
            )
            scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(
                0.0,
                dtype=torch.float16,
                layout=torch.strided,
                device=device(type="cuda", index=0),
            )
            where = torch.ops.aten.where.self(view_15, scalar_tensor_1, scalar_tensor)
            view_15 = scalar_tensor_1 = scalar_tensor = None
            return where

        from torch._inductor import config

        config.emulate_precision_casts = True
        self.assertEqual(torch.compile(forward)(), forward())

    @config.patch({"emulate_precision_casts": True})
    def test_emulate_low_precision(self):
        def foo(x):
            return torch.nn.functional.gelu(x) * 10.0

        inp = torch.rand([32], device="cuda", requires_grad=True, dtype=torch.bfloat16)
        out, codes = run_fw_bw_and_get_code(lambda: torch.compile(foo)(inp))

        # fwd, backward
        for code in codes:
            f = FileCheck()
            # in eager, there are two down casts
            for _ in range(2):
                f.check(".to(tl.bfloat16)").check_next(".to(tl.float32)")
            f.run(code)

        self.assertEqual(foo(inp), out)

    # TODO: Abstract this out, test more extensively
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic_shapes(self):
        torch._dynamo.reset()  # Needed since everywhere else uses "inductor"

        def f(x):
            return x.cos().view(x.shape).sin()

        cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        f2 = torch.compile(f, backend=cnts)

        f2(torch.randn(32))

        inp = torch.randn(16)
        real_out = f(inp)
        compiled_out = f2(inp)

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(real_out, compiled_out)
        torch._dynamo.reset()

    @config.patch({"triton.cudagraphs": True, "size_asserts": False})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs_no_size_asserts(self):
        @torch.compile(backend="inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraph_trees": False})
    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_inplace_updates_cudagraphs(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = torch.nn.Parameter(
                    torch.randn(10, 20, requires_grad=True)
                )

            def forward(self, x):
                x = torch.matmul(x, self.weight1)
                return x

        from copy import deepcopy

        model = Repro().cuda()
        model_ref = deepcopy(model)
        model_opt = torch.compile(model, backend="inductor")

        input = torch.randn(10, 10, device="cuda", requires_grad=True)

        for _ in range(2):
            output_ref = model_ref(input)
            output_res = model_opt(input)
            output_ref.sum().backward()
            output_res.sum().backward()
            for p_ref, p_res in zip(model_ref.parameters(), model_opt.parameters()):
                self.assertEqual(p_ref.grad, p_res.grad)
            with torch.no_grad():
                for param in model_ref.parameters():
                    param.add_(1.0)
                for param in model_opt.parameters():
                    param.add_(1.0)

    # https://github.com/pytorch/torchdynamo/issues/1850
    def test_inductor_output_aliases_intermediate(self):
        def foo(x):
            out = x + x
            return out.t()

        foo_opt = torch.compile(foo, backend="inductor")

        inpt = torch.randn(10, 10, device="cuda", requires_grad=True)
        # TODO: this is broken, fix later
        # out = foo_opt(inpt)
        # out.add_(2)

        out_ref = foo(inpt)
        out_ref.add_(2)
        # self.assertEqual(out_ref, out)

    def test_accuracy_issue1(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features=768, out_features=2, bias=True
                )

            def forward(self, start_positions: torch.Tensor, x: torch.Tensor):
                linear = self.linear(x)
                split = linear.split(1, dim=-1)
                getitem = split[0]
                squeeze = getitem.squeeze(-1)
                clamp = start_positions.clamp(0, 128)
                cross_entropy = torch.nn.functional.cross_entropy(
                    squeeze, clamp, None, None, 128, None, "mean", 0.0
                )
                return cross_entropy

        mod = Repro().cuda()
        opt_mod = torch.compile(mod, backend="inductor")
        mod.eval()
        opt_mod.eval()

        args = [
            ((1,), (1,), torch.int64, "cuda", False),
            ((1, 128, 768), (98304, 768, 1), torch.float32, "cuda", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        with torch.cuda.amp.autocast(enabled=False):
            assert same_two_models(mod, opt_mod, args), "Dynamo failed"

    @config.patch(allow_buffer_reuse=False)
    def test_issue103461(self):
        def forward(add_1):
            var_mean = torch.ops.aten.var_mean.correction(
                add_1, [2], correction=0, keepdim=True
            )
            getitem_1 = var_mean[1]
            return getitem_1

        x = torch.randn(1, 8, 768, device="cuda")
        correct = forward(x)
        actual = torch.compile(forward, fullgraph=True)(x)
        self.assertEqual(actual, correct)

    def test_full_copy(self):
        def forward(x):
            full_10 = torch.ops.aten.full.default(
                [204, 204, 28],
                0,
                dtype=torch.float64,
                layout=torch.strided,
                device="cuda",
                pin_memory=False,
            )
            return x + full_10.to("cpu")

        o = torch.randn([204, 204, 28], dtype=torch.float64)
        correct = forward(o)
        actual = torch.compile(forward, fullgraph=True)(o)
        self.assertEqual(actual, correct)

    def test_autotune_inplace_kernel(self):
        """
        This UT tests autotune on an inplace kernel. The autotune should not contaminate
        the input buffers when tuning with multiple configs. For more details, refer to
        https://github.com/triton-lang/triton/issues/781
        https://github.com/pytorch/torchdynamo/issues/1670
        """
        from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
        from torch._inductor.runtime.hints import AttrsDescriptorWrapper, HeuristicType
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from torch._inductor.utils import triton_version_uses_attrs_dict

        def autotune(configs, meta):
            def decorator(fn):
                if triton_version_uses_attrs_dict():
                    # Newer versions of Triton puts constexpr in signature
                    # Ref: https://github.com/pytorch/pytorch/pull/145051
                    meta["signature"]["XBLOCK"] = "constexpr"

                return CachingAutotuner(
                    # force autotune by setting save_cache_hook to False
                    fn,
                    triton_meta=meta,
                    configs=configs,
                    save_cache_hook=False,
                    mutated_arg_names=["in_out_ptr0"],
                    reset_to_zero_arg_names=[],
                    optimize_mem=True,
                    heuristic_type=HeuristicType.POINTWISE,
                    inductor_meta={"grid_type": "Grid1D"},
                )

            return decorator

        @autotune(
            configs=[
                triton.Config({"XBLOCK": 1}),
                triton.Config({"XBLOCK": 2}),
            ],
            meta={
                "signature": {
                    "in_out_ptr0": "*fp32",
                    "in_ptr0": "*fp32",
                    "xnumel": "i32",
                },
                "device": DeviceProperties.create(torch.device("cuda")),
                "configs": [
                    AttrsDescriptorWrapper(divisible_by_16=(0, 1), equal_to_1=())
                ],
                "constants": {},
            },
        )
        @triton.jit
        def kernel(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * XBLOCK
            offsets = block_start + tl.arange(0, XBLOCK)
            mask = offsets < xnumel
            x = tl.load(in_out_ptr0 + offsets, mask=mask, other=0.0)
            y = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)
            output = x + y
            tl.store(in_out_ptr0 + offsets, output, mask=mask)

        xnumel = 384
        in0 = rand_strided((xnumel,), (1,), device="cuda", dtype=torch.float32)
        inout1 = rand_strided((xnumel,), (1,), device="cuda", dtype=torch.float32)
        inout2 = inout1.clone()

        stream0 = get_cuda_stream(0)
        kernel.run(inout1, in0, xnumel, stream=stream0)
        kernel.run(inout2, in0, xnumel, stream=stream0)

        assert same(inout1, inout2, tol=0.001, equal_nan=True), (
            "failed autotune with inplace kernel"
        )

    def test_sort_stride_issue(self):
        # This minified testcase comes from detectron2_maskrcnn_r_50_fpn
        # There was a false error from our size_assert code
        @torch.compile(fullgraph=True)
        def forward(pred_objectness_logits_3_: torch.Tensor):
            sort_3 = pred_objectness_logits_3_.sort(descending=True, dim=1)
            getitem_12 = sort_3[0]
            return getitem_12

        args = [((1, 100), (0, 1), torch.float16, "cuda", False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        result = forward(*args)
        assert same(result, torch.sort(args[0], descending=True, dim=1)[0])

    def test_scalar_triton_index(self):
        # The indirect indexing via a scalar like below used to lead to
        # bad triton code that made triton segfault when compiling.
        # See https://github.com/pytorch/torchdynamo/issues/1515
        def fn(a):
            zero = torch.zeros((16,), device=a.device, dtype=torch.int64)
            return (a[zero],)

        a = torch.randn((8,), dtype=torch.float32, device="cuda")

        fn_optimized = torch.compile(fn, backend="inductor")
        assert same(fn(a), fn_optimized(a))

    def test_indirect_indexing_dense_mask(self):
        def fn(x, y):
            ne = torch.ops.aten.ne.Scalar(x, 1)
            sum_1 = torch.ops.aten.sum.dim_IntList(ne, [1])
            sub = torch.ops.aten.sub.Tensor(sum_1, 1)
            unsqueeze = torch.ops.aten.unsqueeze.default(sub, -1)
            gather = torch.ops.aten.gather.default(x, 1, unsqueeze)
            squeeze = torch.ops.aten.squeeze.default(gather)
            out = torch.ops.aten.multiply(y, squeeze)
            return (out,)

        a = torch.zeros((1, 128), dtype=torch.int64, device="cuda")
        b = torch.zeros((1, 128), dtype=torch.int64, device="cuda")

        fn_optimized = torch.compile(fn, backend="inductor")
        assert same(fn(a, b), fn_optimized(a, b))

    def test_simplify_dims(self):
        def fn(a):
            return (a + 1,)

        self.common(fn, (torch.randn(2, 3, 10, 5, 6, device="cuda")[:, :, 2::2, :, :],))

    @config.patch(permute_fusion=True)
    def test_permute_fusion(self):
        class Repro(torch.nn.Module):
            def forward(self, view, reshape_2):
                permute = view.permute(0, 2, 1)
                view = None
                reshape = torch.reshape(permute, (-1, 642))
                bmm = torch.bmm(permute, reshape_2)
                return (bmm,)

        args = [
            ((1024, 642, 160), (102720, 160, 1), torch.float32, "cuda", True),
            ((1024, 642, 20), (12840, 20, 1), torch.float32, "cuda", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        mod = Repro()
        opt_mod = torch.compile(mod, backend="inductor")

        ref = mod(*args)
        res = opt_mod(*args)
        self.assertTrue(same(ref, res))

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_add_alpha_autotune(self):
        def fn(x, y):
            aten.add_.Tensor(x, y, alpha=0.55)
            return (x,)

        x1 = torch.zeros(2, 3, 4, 10, device="cuda")
        x2 = torch.zeros(2, 3, 4, 10, device="cuda")
        x3 = torch.zeros(2, 3, 4, 10, device="cuda")
        y = torch.randn(2, 3, 4, 10, device="cuda").to(
            memory_format=torch.channels_last
        )
        fn_fx = make_fx(fn)(x1, y)
        fn_compiled = compile_fx_inner(fn_fx, [x1, y])
        fn(x2, y)
        fn_compiled([x3, y])
        assert same(x2, x3)

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_buffer_autotune(self):
        def foo(x, y, z):
            a = x @ y
            return a.unsqueeze(0).unsqueeze(0) + z

        x = torch.zeros(5, 5, device="cuda")
        y = torch.zeros(5, 5, device="cuda")
        z = torch.zeros(1, 1, 5, 5, device="cuda").to(memory_format=torch.channels_last)
        self.common(
            foo,
            (x, y, z),
            check_lowp=False,
        )

    def test_memory_history_inductor(self):
        def called_inside_compile(x, w, b):
            a = x @ w + b
            return torch.sigmoid(a)

        @torch.compile
        def fn(x, w, b):
            x = called_inside_compile(x, w, b)
            return called_inside_compile(x, w, b)

        w = torch.rand(3, 3, device="cuda")
        b = torch.rand(3, device="cuda")
        x = torch.rand(3, device="cuda")
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history(True)
            r = fn(x, w, b)
        finally:
            torch.cuda.memory._record_memory_history(False)
        snapshot = str(torch.cuda.memory._snapshot())
        self.assertTrue("called_inside_compile" in snapshot)

    def test_negative_arange_dynamic_shapes(self):
        # Repro from alibi relative encodings
        def sign(x):
            return (x > 0) - (x < 0)

        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                nheads = 16
                start = math.log2(0.5)
                end = math.log2(1 / (2**8))

                self.scales = nn.Buffer(
                    2
                    ** torch.arange(
                        start,
                        end + 1e-6 * sign(end - start),
                        (end - start) / (nheads - 1),
                    ).view(1, nheads, 1, 1),
                )
                self.emb = nn.Embedding(1024, 256)
                self.dec_layer = nn.TransformerDecoderLayer(
                    256, 16, 512, batch_first=True, norm_first=True
                )
                self.head = nn.Linear(256, 1024)

            def forward(self, enc_out: torch.Tensor, dec_in: torch.Tensor):
                padmask = dec_in == 0
                dec_mask = padmask.unsqueeze(-1) == padmask.unsqueeze(-2)
                dec_mask = dec_mask.to(dtype=torch.float32)
                dec_mask = dec_mask.tril(diagonal=0).cuda()

                q_pos = torch.arange(dec_in.size(1), dtype=torch.long, device="cuda")
                k_pos = torch.arange(dec_in.size(1), dtype=torch.long, device="cuda")
                rel_pos = k_pos[None, :] - q_pos[:, None]
                values = rel_pos.abs().neg().unsqueeze(0).unsqueeze(0)
                dec_bias = values * self.scales
                dec_bias.tril_(diagonal=0)

                dec_mask = dec_mask + dec_bias[0]
                out = self.emb(dec_in)
                out = self.dec_layer(out, enc_out, tgt_mask=dec_mask)
                return self.head(out)

        mod = Repro().cuda()
        opt_mod = torch.compile(mod, backend="inductor", dynamic=True)
        mod.eval()
        opt_mod.eval()

        enc_out = torch.rand(1, 512, 256).cuda()
        dec_inputs = [
            torch.randint(0, 512, (1, i + 1), dtype=torch.long).cuda() for i in range(8)
        ]

        for dec_inp in dec_inputs:
            assert same_two_models(mod, opt_mod, [enc_out, dec_inp], only_fwd=True), (
                "Inductor with dynamic shapes failed"
            )

    def test_issue97695_1input(self):
        def fn(arg3_1, relu, permute_1):
            addmm_1 = torch.ops.aten.addmm.default(arg3_1, relu, permute_1)
            cat_2 = torch.ops.aten.cat.default([addmm_1], 1)
            return (cat_2,)

        args = [
            ((96,), (1,), torch.float32, "cuda"),
            ((10, 256), (256, 1), torch.float32, "cuda"),
            ((256, 96), (1, 256), torch.float32, "cuda"),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        correct = fn(*args)

        mod = make_fx(fn, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        ref = compiled(list(args))
        assert same(ref, correct)

        ref = torch.compile(fn, fullgraph=True)(*args)
        assert same(ref, correct)

    def test_issue_103924(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temperature = 1
                self.layer = torch.nn.Softmax(dim=1)

            def forward(self, x):
                n_samples, _ = x.shape
                y = 1.0 * torch.ones(n_samples, dtype=x.dtype, device=x.device)
                inp = x / y[..., None]
                return self.layer(inp)

        x = torch.rand([4, 4], device="cuda")
        m = MyModule()
        opt_m = torch.compile(backend="inductor")(m)
        self.assertEqual(opt_m(x), m(x))

    def test_issue97695_2input(self):
        def fn(arg3_1, arg3_2, relu, permute_1):
            addmm_1 = torch.ops.aten.addmm.default(arg3_1, relu, permute_1)
            addmm_2 = torch.ops.aten.addmm.default(arg3_2, relu, permute_1)
            cat_2 = torch.ops.aten.cat.default([addmm_1, addmm_2], 1)
            return (cat_2,)

        args = [
            ((96,), (1,), torch.float32, "cuda"),
            ((96,), (1,), torch.float32, "cuda"),
            ((10, 256), (256, 1), torch.float32, "cuda"),
            ((256, 96), (1, 256), torch.float32, "cuda"),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        correct = fn(*args)

        ref = torch.compile(fn, fullgraph=True)(*args)
        assert same(ref, correct)

    def test_scatter_index_not_wrapped(self):
        src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.device)
        index = torch.tensor([0, 1, 0, 1, 2, 0], device=self.device)
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        compiled_sr = torch.compile(torch.scatter_reduce)

        input_orig = input.clone()
        out, code = run_and_get_code(compiled_sr, input, 0, index, src, "sum")
        # tmp0 - not wrapping of negative numbers
        FileCheck().check("tl.device_assert(((0 <= tmp0) & (tmp0 < 4))").check_next(
            "atomic_add"
        ).run(code[0])
        self.assertEqual(
            out, torch.scatter_reduce(input_orig.clone(), 0, index, src, "sum")
        )

    def test_normalize_norm_leq_one(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(x, dim=-1)

        inp = torch.tensor([[3.799999, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        out = compiled(inp)
        norm = out.norm(dim=-1)
        self.assertTrue(
            torch.all(norm <= 1.0), f"expected norm <= 1.0 but got {norm.item()}"
        )

    def test_libdevice_routing(self):
        def foo(x):
            return x.exp()

        inp = torch.ones(64, device="cuda").to(torch.float64)

        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

        inp = inp.to(torch.float)
        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check_not("tl_math.exp").check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

        def foo(x):
            return x.sigmoid()

        inp = torch.ones(64, device="cuda").to(torch.float64)
        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

    def test_uint_view_copy(self):
        @torch.compile
        def view_copy(target, source):
            assert target.dtype == torch.bfloat16
            assert source.dtype == torch.uint16
            target.view(torch.uint16).copy_(source)

        target = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
        source = torch.full_like(target, 4, dtype=torch.uint16)

        out = target.view(torch.uint16).copy_(source).clone()
        view_copy(target, source)
        self.assertEqual(out, target.view(torch.uint16))

    def test_embedding_var_mean(self):
        def forward(arg0_1):
            full = torch.ops.aten.full.default(
                [1, 2048],
                1,
                dtype=torch.float32,
                layout=torch.strided,
                device=torch.device(type="cuda", index=0),
                pin_memory=False,
            )
            convert_element_type_1 = torch.ops.prims.convert_element_type.default(
                full, torch.int64
            )
            cumsum = torch.ops.aten.cumsum.default(convert_element_type_1, 1)
            mul = torch.ops.aten.mul.Tensor(cumsum, convert_element_type_1)
            sub_1 = torch.ops.aten.sub.Tensor(mul, 1)
            slice_5 = torch.ops.aten.slice.Tensor(sub_1, 0, 0, 9223372036854775807)
            slice_6 = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807)
            add_2 = torch.ops.aten.add.Tensor(slice_6, 2)
            embedding_1 = torch.ops.aten.embedding.default(arg0_1, add_2)
            var_mean = torch.ops.aten.var_mean.correction(
                embedding_1, [2], correction=0, keepdim=True
            )
            return [var_mean[0], var_mean[1], add_2]

        emb = torch.randn([2050, 768], device="cuda")
        gm = make_fx(forward)(emb)
        opt = torch._inductor.compile_fx.compile_fx_inner(gm, [emb])
        opt([emb])
        torch.cuda.synchronize()

    def test_deterministic_algorithms(self):
        N = 10000

        @torch.compile
        def fn(idx, values):
            x = torch.zeros(1, device="cuda")
            x[idx] += values
            return x

        idx = torch.zeros(N, dtype=torch.int64, device="cuda")
        values = torch.randn(N, device="cuda")

        r0 = fn(idx, values)
        with DeterministicGuard(True):
            r1 = fn(idx, values)
            for _ in range(10):
                rn = fn(idx, values)
                self.assertEqual(r1, rn, atol=0, rtol=0)

    # https://github.com/pytorch/pytorch/issues/96406
    def test_linear_cpu_input(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, data):
                data = data.to("cuda")
                return self.linear(data)

        mod = Model().cuda().eval()
        with torch.no_grad():
            self.common(mod, (torch.randn(4, 4),))

    @config.patch({"fallback_random": True, "triton.cudagraphs": True})
    def test_xlnet_lm_stride_repro(self):
        class Repro(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = nn.Dropout(p=0.1, inplace=False)

            def forward(self, x):
                y = torch._C._nn.gelu(x)
                return self.dropout(y)

        mod = Repro()
        x = torch.randn((512, 1, 4096), requires_grad=True, device="cuda")
        y = torch.compile(mod)(x)
        # Inductor claims the output layout of gelu's saved variable for
        # backwards will be (4096, 4096, 1) but in actuality it is (4096,
        # 2097152, 1).  Fortunately this doesn't actually matter in practice.
        y.sum().backward()

    def test_lookup_seed_backward(self):
        @torch.compile(fullgraph=True)
        def forward(inductor_seeds, mul_4, view_15):
            inductor_lookup_seed_2 = torch.ops.prims.inductor_lookup_seed.default(
                inductor_seeds, 2
            )
            inductor_random_2 = torch.ops.prims.inductor_random.default(
                [2, 512, 768], inductor_lookup_seed_2, "rand"
            )
            gt_2 = torch.ops.aten.gt.Scalar(inductor_random_2, 0.1)
            mul_7 = torch.ops.aten.mul.Tensor(gt_2, view_15)
            mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112)
            add_5 = torch.ops.aten.add.Tensor(mul_8, mul_4)
            var_mean_1 = torch.ops.aten.var_mean.correction(
                add_5, [2], correction=0, keepdim=True
            )
            getitem_3 = var_mean_1[1]
            sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3)
            return (sub_3,)

        buf0 = torch.zeros((37,), dtype=torch.int64, device="cuda")
        buf1 = torch.zeros((2, 512, 768), device="cuda")
        buf2 = torch.zeros((2, 512, 768), device="cuda")
        forward(buf0, buf1, buf2)

    def test_issue100806(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 30)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = torch.cat((x, x), dim=1)
                x = x.view(-1, 2, 30)
                x = x[:, 1, :]
                x = self.relu(x)
                return x

        device = "cuda"
        batch_size = 2
        x = torch.randn(batch_size, 10).to(device)
        func = Model().to(device)

        with torch.no_grad():
            func.train(False)
            jit_func = torch.compile(func)

            res1 = func(x)
            res2 = jit_func(x)
            self.assertEqual(res1, res2)

    def test_issue103481(self):
        def fn(x, y):
            # NOTE: 6 dimensions is important! does not fail for 5 dimensions
            mean = torch.mean(x, [2, 3, 4, 5], keepdim=True)
            add = mean + y
            return add

        x = torch.rand(4, 4, 4, 4, 4, 4, device="cuda")
        y = torch.rand((), device="cuda")
        expect = fn(x, y)

        opt_fn = torch.compile(fn)
        actual = opt_fn(x, y)

        self.assertEqual(expect, actual)

    @config.patch({"triton.dense_indexing": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_bucketize_dynamic_dense(self):
        """
        Make sure that ops.bucketize() can handle dense_indexing, which previously
        caused issues due to incorrect handling of the size of offsets.
        """

        def fn(values, offsets):
            return torch.bucketize(values, offsets)

        values = torch.rand((64, 64), device="cuda")
        offsets = torch.tensor([0.05, 0.1, 0.5, 0.8, 0.85, 0.95], device="cuda")

        expect = fn(values, offsets)

        opt_fn = torch.compile(fn, dynamic=True)
        actual = opt_fn(values, offsets)

        self.assertEqual(expect, actual)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @config.patch(
        {
            "max_autotune_gemm_backends": "TRITON",
            "triton.disallow_failing_autotune_kernels_TESTING_ONLY": True,
            "compile_threads": 1,
        }
    )
    def test_bucketize_epilogue(self):
        """
        See https://github.com/pytorch/pytorch/issues/148764.
        Make sure that when torch.bucketize appears as an epilogue, the codegen is valid.

        Note: during autotuning, there's also the option to _not_ do the fusion.
        So if you run the test with standard configs, the fused kernel would fail during
        autotuning, and another non-fused kernel would be selected (and Inductor would
        throw some errors, but the test would pass)

        So we set disallow_failing_autotune_kernels_TESTING_ONLY=True to prevent the
        autotuner from catching failures. And set compile_threads=1 so that compile
        failures aren't caught by the asyn runner infra.
        """

        def fn(x: torch.Tensor, y: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
            z = torch.mm(x, y)
            return torch.bucketize(z, buckets)

        buckets = torch.arange(-100, 100, 10, device="cuda")
        x = torch.randn(64, 64, device="cuda").clamp(-99, 99)
        y = torch.randn(64, 64, device="cuda").clamp(-99, 99)

        opt_fn = torch.compile(fn, mode="max-autotune")

        expected = fn(x, y, buckets)
        actual = opt_fn(x, y, buckets)

        self.assertEqual(expected, actual)

    def test_float64_constants(self):
        def fn():
            # NOTE: tensors of all the same value are constant folded, so we
            # need a tensor with two distinct values
            a = torch.tensor([1 / 10, 2 / 10], dtype=torch.float64, device="cuda")
            return a * 2e50

        cfn = torch.compile(fn)
        expect = fn()
        actual = cfn()
        self.assertEqual(expect, actual, atol=0, rtol=0)

    def test_issue104759(self):
        def fn(arg7_1, add_1, permute_2, select_scatter, slice_8):
            slice_scatter_4 = torch.ops.aten.slice_scatter.default(
                permute_2, select_scatter, 0, 1, 9223372036854775807
            )
            permute_3 = torch.ops.aten.permute.default(slice_scatter_4, [1, 3, 0, 2, 4])
            view_6 = torch.ops.aten.view.default(permute_3, [1, 1000, 48])
            view_7 = torch.ops.aten.view.default(view_6, [1000, 48])
            view_8 = torch.ops.aten.view.default(view_7, [1, 1000, 48])
            view_9 = torch.ops.aten.view.default(view_8, [1, 1000, 3, 4, 4])
            permute_4 = torch.ops.aten.permute.default(view_9, [2, 0, 3, 1, 4])
            slice_7 = torch.ops.aten.slice.Tensor(permute_4, 0, 1, 9223372036854775807)
            slice_scatter_5 = torch.ops.aten.slice_scatter.default(
                slice_8, slice_7, 4, 0, 9223372036854775807
            )
            slice_scatter_6 = torch.ops.aten.slice_scatter.default(
                arg7_1, slice_scatter_5, 3, 0, 1000
            )
            mul_8 = torch.ops.aten.mul.Scalar(add_1, 0.7071067811865476)
            slice_9 = torch.ops.aten.slice.Tensor(slice_scatter_6, 3, 0, 1000)
            slice_10 = torch.ops.aten.slice.Tensor(slice_9, 4, 0, 9223372036854775807)
            select_2 = torch.ops.aten.select.int(slice_10, 0, 0)
            permute_5 = torch.ops.aten.permute.default(select_2, [0, 1, 3, 2])
            mul_9 = torch.ops.aten.mul.Scalar(permute_5, 0.7071067811865476)
            expand = torch.ops.aten.expand.default(mul_8, [1, 4, 1000, 4])
            view_10 = torch.ops.aten.view.default(expand, [4, 1000, 4])
            expand_1 = torch.ops.aten.expand.default(mul_9, [1, 4, 4, 1000])
            view_11 = torch.ops.aten.view.default(expand_1, [4, 4, 1000])
            bmm = torch.ops.aten.bmm.default(view_10, view_11)
            return (bmm,)

        args = []
        args.append(torch.randn((2, 1, 4, 1200, 4), dtype=torch.float16, device="cuda"))
        args.append(
            rand_strided(
                (1, 4, 1000, 4), (16000, 4, 16, 1), dtype=torch.float16, device="cuda"
            )
        )
        args.append(
            rand_strided(
                (3, 1, 4, 1000, 4),
                (16, 48000, 4, 48, 1),
                dtype=torch.float16,
                device="cuda",
            )
        )
        args.append(
            rand_strided(
                (2, 1, 4, 1000, 4),
                (16, 48000, 4, 48, 1),
                dtype=torch.float16,
                device="cuda",
            )
        )
        args.append(
            rand_strided(
                (2, 1, 4, 1000, 4),
                (19200, 19200, 4800, 4, 1),
                dtype=torch.float16,
                device="cuda",
            )
        )

        correct = fn(*args)
        mod = make_fx(fn, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        ref = compiled(list(args))
        assert same(ref, correct)

    @config.patch({"triton.cudagraphs": True})
    def test_index_put_inplace_cudagraph(self):
        def fn(x, y, z):
            x = torch.zeros_like(x)
            return x.index_put_([y], z, True)

        x = torch.zeros((512, 512), device="cuda", dtype=torch.bool)
        y = torch.zeros((512,), device="cuda", dtype=torch.int64)
        z = torch.ones((512, 512), device="cuda", dtype=torch.bool)

        opt_fn = torch.compile(fn, backend="inductor")

        ref = fn(x, y, z)

        # run it twice to test cuda graph issue
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        self.assertEqual(ref, res)

    @config.patch({"triton.cudagraphs"
```



## High-Level Overview


This Python file contains 18 class(es) and 199 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CudaReproTests`, `ReproModule`, `Repro`, `Repro`, `Repro`, `Repro`, `Repro`, `Repro`, `MyModule`, `Model`, `Repro`, `Model`, `Model`, `SelfAttention`, `ToyModel`, `ToyModel`, `Model`, `Foo`

**Functions defined**: `test_mm_out_dtype_compile`, `fn`, `test_index_put_issue`, `forward`, `test_view_replay_padding_issue_163328`, `__init__`, `forward`, `test_effn_attn_bias_padding`, `fn`, `test_effn_attn_bias_padding_misaligned`, `f`, `test_input_channels_last`, `foo`, `test_unspec_inputs_interop`, `forward`, `test_backward_context`, `fn`, `test_dtype_factory_issue`, `forward`, `test_no_device_idx_repro_cudagraphs`

**Key imports**: copy, functools, gc, math, os, sys, unittest, torch, torch._dynamo.config as dynamo_config, torch.backends.cuda


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `functools`
- `gc`
- `math`
- `os`
- `sys`
- `unittest`
- `torch`
- `torch._dynamo.config as dynamo_config`
- `torch.backends.cuda`
- `torch.nn.functional as F`
- `torch._dynamo.debug_utils`: same_two_models
- `torch._dynamo.testing`: rand_strided
- `torch._dynamo.utils`: same
- `torch._inductor`: config
- `torch._inductor.compile_fx`: compile_fx_inner
- `torch._inductor.runtime.benchmarking`: benchmarker
- `torch._inductor.runtime.hints`: DeviceProperties
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.nn.attention`: sdpa_kernel, SDPBackend
- `torch.testing`: FileCheck
- `torch.testing._internal.inductor_utils`: IS_BIG_GPU
- `triton  `
- `triton`: language as tl  
- `.`: test_torchinductor
- `test_torchinductor  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_cuda_repro.py
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

- **File Documentation**: `test_cuda_repro.py_docs.md`
- **Keyword Index**: `test_cuda_repro.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
