# Documentation: `test/distributed/_shard/sharded_tensor/test_sharded_tensor.py`

## File Metadata

- **Path**: `test/distributed/_shard/sharded_tensor/test_sharded_tensor.py`
- **Size**: 127,250 bytes (124.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import copy
import io
import itertools
import math
import pickle
import sys

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d, rpc
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.api import (
    _collect_local_shard,
    _reshard_output,
    _shard_tensor,
    load_with_process_group,
    shard_parameter,
)
from torch.distributed._shard.sharded_tensor import (
    custom_sharded_op_impl,
    pre_load_state_dict_hook,
    Shard,
    ShardedTensor,
    ShardedTensorBase,
    ShardedTensorMetadata,
    state_dict_hook,
)
from torch.distributed._shard.sharded_tensor.api import (
    _create_tensor_from_params,
    TensorProperties,
)
from torch.distributed._shard.sharded_tensor.utils import (
    _parse_and_validate_remote_device,
)
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)
from torch.distributed.remote_device import _remote_device
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
    spawn_threads_and_init_comms,
    tp_transports,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
    MyShardedModel1,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorMetadata(TestCase):
    def test_serialize_and_deserialize(self):
        shard_metadatas = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="rank:0/cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 5],
                shard_sizes=[5, 5],
                placement="rank:1/cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="rank:2/cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="rank:3/cuda:3",
            ),
        ]

        dtypes = [
            torch.float,
            torch.double,
            torch.cfloat,
            torch.cdouble,
            torch.half,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.short,
            torch.int,
            torch.long,
            torch.bool,
        ]

        layouts = [torch.strided, torch.sparse_coo]
        requires_grads = [True, False]
        memory_formats = [
            torch.contiguous_format,
            torch.channels_last,
            torch.preserve_format,
        ]
        pin_memories = [True, False]

        for tensor_properties_input in itertools.product(
            dtypes, layouts, requires_grads, memory_formats, pin_memories
        ):
            (
                dtype,
                layout,
                requires_grad,
                memory_format,
                pin_memory,
            ) = tensor_properties_input

            expected_st_metadata = sharded_tensor.ShardedTensorMetadata(
                shard_metadatas,
                (10, 10),
                TensorProperties(
                    dtype, layout, requires_grad, memory_format, pin_memory
                ),
            )

            pickled_obj = pickle.dumps(expected_st_metadata)
            st_metadata = pickle.loads(pickled_obj)
            self.assertEqual(expected_st_metadata, st_metadata)


class TestCreateTensorFromParams(TestCase):
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "CUDA GPU is needed")
    def test_empty(self):
        expected_dtype = torch.double
        tensor_properties = TensorProperties(
            dtype=expected_dtype,
            layout=torch.strided,
            requires_grad=False,
            pin_memory=False,
            memory_format=torch.contiguous_format,
        )
        local_device = torch.device("cuda:0")
        local_tensor = _create_tensor_from_params(
            5, 10, local_device=local_device, tensor_properties=tensor_properties
        )
        self.assertEqual(local_device, local_tensor.device)
        self.assertEqual(expected_dtype, local_tensor.dtype)
        self.assertEqual(torch.strided, local_tensor.layout)
        self.assertEqual(False, local_tensor.requires_grad)


class TestShardParameter(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_parameter(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        fc = torch.nn.Linear(12, 12).cuda(self.rank)
        weight_og = fc.weight.clone()
        shard_parameter(fc, "weight", spec)

        # Verify.
        self.assertTrue(isinstance(fc.weight, ShardedTensor))
        local_shards = fc.weight.local_shards()
        self.assertEqual(1, len(local_shards))
        self.assertEqual(torch.Size([3, 12]), local_shards[0].tensor.size())
        self.assertEqual(3, local_shards[0].tensor.size(0))
        self.assertEqual(12, local_shards[0].tensor.size(1))
        self.assertEqual(
            torch.narrow(weight_og, 0, 3 * self.rank, 3), local_shards[0].tensor
        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_parameter_errors(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        fc = torch.nn.Linear(12, 12).cuda(self.rank)
        with self.assertRaisesRegex(ValueError, "does not match with src_rank"):
            shard_parameter(fc, "weight", spec, src_rank=self.rank)

        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            shard_parameter(fc, "foo", spec)

        with self.assertRaisesRegex(
            ValueError, "Expected Linear.bias to be a Tensor, but found str"
        ):
            del fc.bias
            fc.bias = "foo"
            shard_parameter(fc, "bias", spec)

        with self.assertRaisesRegex(ValueError, "not a contiguous Tensor"):
            fc.bias = torch.rand(10, 10).cuda(self.rank).t()
            shard_parameter(fc, "bias", spec)

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{self.rank}/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        with self.assertRaisesRegex(ValueError, "does not match with sharding_spec"):
            shard_parameter(fc, "weight", spec)

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet!"):
            shard_parameter(fc, "weight", spec)


class TestShardTensor(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_tensor(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        tensor = torch.rand(12, 12).cuda(self.rank)
        st = _shard_tensor(tensor, spec)

        # Verify.
        self.assertTrue(isinstance(st, sharded_tensor.ShardedTensor))
        local_shard = st.local_tensor()
        self.assertEqual(1, len(st.local_shards()))
        self.assertEqual(torch.Size([3, 12]), local_shard.size())
        self.assertEqual(torch.narrow(tensor, 0, 3 * self.rank, 3), local_shard)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_tensor_with_empty_shard(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        tensor = torch.rand(9, 12).cuda(self.rank)
        st = _shard_tensor(tensor, spec)

        # Verify.
        self.assertTrue(isinstance(st, sharded_tensor.ShardedTensor))
        sms = st.metadata().shards_metadata
        self.assertEqual(len(sms), 4)
        for sm in sms:
            self.assertTrue(sm.shard_offsets[0] + sm.shard_sizes[0] <= tensor.size(0))

        local_shard = st.local_tensor()
        self.assertEqual(1, len(st.local_shards()))
        if dist.get_rank() < 3:
            self.assertEqual(torch.Size([3, 12]), local_shard.size())
            self.assertEqual(torch.narrow(tensor, 0, 3 * self.rank, 3), local_shard)
        else:
            self.assertEqual(torch.Size([0, 12]), local_shard.size())

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_tensor_errors(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        tensor = torch.rand(12, 12).cuda(self.rank)

        with self.assertRaisesRegex(ValueError, "does not match with src_rank"):
            _shard_tensor(tensor, spec, src_rank=self.rank)

        with self.assertRaisesRegex(ValueError, "not a contiguous Tensor"):
            tensor_t = torch.rand(12, 12).cuda(self.rank).t()
            _shard_tensor(tensor_t, spec)

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{self.rank}/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        with self.assertRaisesRegex(ValueError, "does not match with sharding_spec"):
            _shard_tensor(tensor, spec)

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet!"):
            _shard_tensor(tensor, spec)


class TestModuleHookApi(ShardedTensorTestBase):
    class DummyNNModule(torch.nn.Module):
        def __init__(self, spec, tensor_size):
            super().__init__()
            self.st = sharded_tensor.rand(spec, *tensor_size)

        def forward(self):
            return self.st

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_reshard_output(self):
        specs = _chunk_sharding_specs_list_for_test([0, 1], seed=5)
        spec, reshard_spec = specs[0], specs[1]
        test_module = self.DummyNNModule(spec, [24, 12])
        st = test_module()
        local_shard = st.local_tensor()
        pg = dist.distributed_c10d._get_default_group()
        st_compare = ShardedTensor._init_from_local_shards(
            copy.deepcopy(st.local_shards()),
            st.size(),
            process_group=pg,
        )
        st_compare._sharding_spec = copy.deepcopy(spec)
        st_compare.reshard(reshard_spec)
        test_module = _reshard_output(test_module, reshard_spec)
        st = test_module()
        local_shard = st.local_tensor()
        local_shard_compare = st_compare.local_tensor()
        self.assertEqual(local_shard, local_shard_compare)
        self.assertEqual(local_shard.size(0), 24)
        self.assertEqual(local_shard.size(1), 3)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_collect_local_shard(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=5)
        spec = specs[0]
        test_module = self.DummyNNModule(spec, [23, 15])
        st = test_module()
        local_shard = st.local_tensor()
        test_module = _collect_local_shard(test_module)
        output = test_module()
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(local_shard, output)


class TestLocalTensor(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_local_tensor(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, 24, 12)
        local_shard = st.local_tensor()
        self.assertEqual(torch.Size([6, 12]), local_shard.size())
        self.assertEqual(st.local_tensor(), local_shard)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_local_tensor_error(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:2/cuda:2",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, 24, 12)
        with self.assertRaisesRegex(
            NotImplementedError, "Only single local shard is supported."
        ):
            st.local_tensor()


class TestShardedTensorChunked(ShardedTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_metadata(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
        st_metadata = st.metadata()
        self.assertEqual(torch.Size([10, 20]), st_metadata.size)
        self.assertEqual(torch.Size([10, 20]), st.size())
        self.assertEqual(torch.float, st.dtype)
        self.assertEqual(torch.strided, st.layout)
        self.assertEqual(False, st.requires_grad)
        self.assertTrue(st.is_contiguous())
        self.assertFalse(st.is_pinned())

        st = sharded_tensor.empty(spec, 10, 20, requires_grad=True, init_rrefs=True)
        self.assertEqual(True, st.requires_grad)

        st = sharded_tensor.empty(spec, 10, 20, dtype=torch.double, init_rrefs=True)
        self.assertEqual(torch.double, st.dtype)

        # Need CPU for pin_memory
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )

        st = sharded_tensor.empty(spec, 10, 20, pin_memory=True, init_rrefs=True)
        self.assertEqual(True, st.is_pinned())

        # test read only properties, they're read only as we can't simply change
        # the global metadata without changing the underlying shard's properties
        with self.assertRaisesRegex(RuntimeError, "torch function '__set__'"):
            st.requires_grad = True

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_complete_world_size(self):
        for dim in [0, -2]:
            spec = ChunkShardingSpec(
                dim=dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )
            st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

            # Validate local shard.
            local_shards = st.local_shards()
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            if self.rank == 3:
                self.assertEqual((1, 20), local_shard.size())
            else:
                self.assertEqual((3, 20), local_shard.size())

            # Validate global metadata.
            st_metadata = st.metadata()
            shards_metadata = st_metadata.shards_metadata
            self.assertEqual(4, len(shards_metadata))

            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual([rank * 3, 0], shard_metadata.shard_offsets)
                if rank == 3:
                    self.assertEqual([1, 20], shard_metadata.shard_sizes)
                else:
                    self.assertEqual([3, 20], shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement)
                )

            # Validate remote shards.
            remote_shards = st.remote_shards()
            self.assertEqual(3, len(remote_shards))

            for rpc_rank, shards in remote_shards.items():
                self.assertEqual(1, len(shards))
                for remote_shard in shards:
                    self.assertEqual(rpc_rank, remote_shard.owner().id)
                    shard = remote_shard.to_here()
                    self.assertEqual(
                        f"rank:{rpc_rank}/cuda:{rpc_rank}",
                        str(shard.metadata.placement),
                    )
                    if rpc_rank == 3:
                        self.assertEqual((1, 20), shard.tensor.size())
                    else:
                        self.assertEqual((3, 20), shard.tensor.size())

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_ones(self):
        """Test sharded_tensor.ones(...)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        st = sharded_tensor.ones(spec, h, w)

        # Validate local shard is initialized with torch.ones
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        # The split: for rank!=3 ceil(h/4)=3  for rank=3 1
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(local_shard, torch.ones(expected_h, w))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_even(self) -> None:
        """Test _sharded_tensor.gather(...) with evenly distributed._shards"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        st = sharded_tensor.ones(spec, h, w)

        full_tensor = None
        dst = 1
        if self.rank == dst:
            full_tensor = torch.zeros(
                h,
                w,
                device=torch.device(f"cuda:{dst}"),
            )
        st.gather(dst, full_tensor)

        if self.rank == dst:
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            self.assertIsNone(full_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_uneven(self) -> None:
        """Test _sharded_tensor.gather(...) with unevenly distributed._shards"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
            ],
        )
        h, w = 10, 20
        st = sharded_tensor.ones(spec, h, w)

        full_tensor = None
        dst = 1
        if self.rank == dst:
            full_tensor = torch.zeros(
                h,
                w,
                device=torch.device(f"cuda:{dst}"),
            )
        st.gather(dst, full_tensor)

        if self.rank == dst:
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            self.assertIsNone(full_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_zeros(self):
        """Test sharded_tensor.zeros(...)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        st = sharded_tensor.zeros(spec, h, w)

        # Validate local shard is initialized with torch.zeros
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        # The split: for rank!=3 ceil(h/4)=3  for rank=3 1
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(local_shard, torch.zeros(expected_h, w))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_rand(self):
        """Test sharded_tensor.rand(...)/randn(...)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        seed = 1234

        expected_h = 2
        expected_device = torch.device(f"cuda:{self.rank}")
        dtype = torch.double
        torch.manual_seed(seed)
        # Test sharded_tensor.rand creation
        expected = torch.rand(expected_h, w, device=expected_device, dtype=dtype)
        # reset seed to ensure the same random numbers are generated
        torch.manual_seed(seed)
        st = sharded_tensor.rand(spec, h, w, dtype=dtype)

        # Validate local shard is initialized with torch.rand
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(expected_device, local_shard.device)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(expected, local_shard)

        # Test sharded_tensor.randn creation
        torch.manual_seed(seed)
        expected_randn = torch.randn(expected_h, w, device=expected_device, dtype=dtype)
        # reset seed to ensure the same random numbers are generated
        torch.manual_seed(seed)
        st_randn = sharded_tensor.randn(spec, h, w, dtype=dtype)

        # Validate local shard is initialized with torch.randn
        local_shards = st_randn.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(expected_device, local_shard.device)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(expected_randn, local_shard)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_full(self):
        """Test sharded_tensor.full(...)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        fill_value = 1234
        st = sharded_tensor.full(
            spec, size=(h, w), fill_value=fill_value, dtype=torch.int32
        )

        # Validate local shard is initialized with torch.full
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        # The split: for rank!=3 ceil(h/4)=3  for rank=3 1
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(
            local_shard,
            torch.full(size=(expected_h, w), fill_value=fill_value, dtype=torch.int32),
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_like(self):
        """Test tensor like methods, i.e. torch.zeros_like(...), torch.full_like, etc."""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 8
        expected_h = 2
        seed = 1234
        dtype = torch.double
        expected_device = torch.device(f"cuda:{self.rank}")
        st = sharded_tensor.rand(spec, (h, w), dtype=dtype)
        tensor_like_ops = {
            torch.zeros_like: torch.zeros,
            torch.ones_like: torch.ones,
            torch.rand_like: torch.rand,
            torch.randn_like: torch.randn,
            torch.empty_like: torch.empty,
            torch.full_like: torch.full,
        }
        for op, expect_local_op in tensor_like_ops.items():
            if op == torch.full_like:
                # special handle full/full_like as it needs to have additional fill_value arg
                expect_tensor = expect_local_op(
                    (expected_h, w), 8.8, device=expected_device, dtype=dtype
                )
                new_op_st = op(st, 8.8, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor(), expect_tensor)
            elif op == torch.empty_like:
                # empty/empty_like we only compare the shape
                expect_tensor = expect_local_op(
                    expected_h, w, device=expected_device, dtype=dtype
                )
                new_op_st = op(st, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor().shape, expect_tensor.shape)
            else:
                torch.manual_seed(seed)
                expect_tensor = expect_local_op(
                    expected_h, w, device=expected_device, dtype=dtype
                )
                torch.manual_seed(seed)
                new_op_st = op(st, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor(), expect_tensor)

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_partial_world_size(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

        # Validate local shard.
        local_shards = st.local_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((5, 20), local_shard.size())
        else:
            self.assertEqual(0, len(local_shards))

        # Validate global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(2, len(shards_metadata))

        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_rank * 5, 0], shard_metadata.shard_offsets)
            self.assertEqual([5, 20], shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{shard_rank + 2}/cuda:{shard_rank + 2}",
                str(shard_metadata.placement),
            )

        # Validate remote shards.
        remote_shards = st.remote_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(remote_shards))
        else:
            self.assertEqual(2, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual(
                    f"rank:{rpc_rank}/cuda:{rpc_rank}", str(shard.metadata.placement)
                )
                self.assertEqual((5, 20), shard.tensor.size())

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_new_group(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        pg = dist.new_group(ranks=[1, 2, 3])
        st = sharded_tensor.empty(spec, 10, 20, process_group=pg, init_rrefs=True)

        # Validate local shard.
        local_shards = st.local_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((5, 20), local_shard.size())
        else:
            self.assertEqual(0, len(local_shards))

        # Validate global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(2, len(shards_metadata))

        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_rank * 5, 0], shard_metadata.shard_offsets)
            self.assertEqual([5, 20], shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{shard_rank + 2}/cuda:{shard_rank + 2}",
                str(shard_metadata.placement),
            )

        # Validate remote shards.
        remote_shards = st.remote_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(remote_shards))
        else:
            self.assertEqual(2, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                shard = remote_shard.to_here()
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                self.assertEqual(
                    f"rank:{rpc_rank}/cuda:{rpc_rank}", str(shard.metadata.placement)
                )
                self.assertEqual((5, 20), shard.tensor.size())

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_multiple_local_shards(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.empty(spec, 16, 20, init_rrefs=True)

        # Validate local shards.
        local_shards = st.local_shards()
        self.assertEqual(2, len(local_shards))
        for local_shard in local_shards:
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((2, 20), local_shard.tensor.size())

        # Validate global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(8, len(shards_metadata))

        for shard_idx, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_idx * 2, 0], shard_metadata.shard_offsets)
            self.assertEqual([2, 20], shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{shard_idx % 4}/cuda:{shard_idx % 4}",
                str(shard_metadata.placement),
            )

        # Validate remote shards.
        remote_shards = st.remote_shards()
        self.assertEqual(3, len(remote_shards))
        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(2, len(shards))
            for remote_shard in shards:
                shard = remote_shard.to_here()
                self.assertEqual((2, 20), shard.tensor.size())
                self.assertEqual(rpc_rank, remote_shard.owner().id)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharding_columns(self):
        self.init_pg()

        for dim in [1, -1]:
            spec = ChunkShardingSpec(
                dim=dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )

            st = sharded_tensor.empty(spec, 10, 32)

            # Validate local shard.
            local_shards = st.local_shards()
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((10, 8), local_shard.size())

            # Validate global metadata.
            st_metadata = st.metadata()
            shards_metadata = st_metadata.shards_metadata
            self.assertEqual(4, len(shards_metadata))

            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual([0, rank * 8], shard_metadata.shard_offsets)
                self.assertEqual([10, 8], shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement)
                )

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_invalid_sharding(self):
        self.init_pg()

        with self.assertRaisesRegex(
            NotImplementedError, "does not support named dimension"
        ):
            spec = ChunkShardingSpec(dim="H", placements=["rank:1/cuda:1"])
            sharded_tensor.empty(spec, 10, 20)

        for dim in [2, 3, 4, -3, -4, -5]:
            spec = ChunkShardingSpec(dim=dim, placements=["rank:1/cuda:1"])
            with self.assertRaisesRegex(ValueError, "Invalid sharding dim"):
                sharded_tensor.empty(spec, 10, 20)

        spec = ChunkShardingSpec(dim=0, placements=["rank:5/cuda:1"])
        with self.assertRaisesRegex(
            ValueError, "Global rank 5 does not exist in input process group"
        ):
            sharded_tensor.empty(spec, 10, 20)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        st = sharded_tensor.empty(spec, 10, 20)
        tensor = torch.empty(10, 20)
        with self.assertRaisesRegex(
            RuntimeError, r".*not supported for ShardedTensor!$"
        ):
            torch.add(st, tensor)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            ValueError, "Only torch.strided layout is currently supported"
        ):
            sharded_tensor.empty(spec, 10, 20, layout=torch.sparse_coo)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            ValueError,
            "Only torch.contiguous_format memory_format is currently supported",
        ):
            sharded_tensor.empty(spec, 10, 20, memory_format=torch.channels_last)

        spec = ChunkShardingSpec(dim=0, placements=["worker0/cuda:1"])
        with self.assertRaisesRegex(
            RuntimeError, "RPC framework needs to be initialized"
        ):
            sharded_tensor.empty(spec, 10, 20)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            RuntimeError, "RPC Framework needs to be initialized"
        ):
            st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

        with self.assertRaisesRegex(
            RuntimeError, "ShardedTensor created with init_rrefs=False"
        ):
            st = sharded_tensor.empty(spec, 10, 20)
            st.remote_shards()

        self.init_rpc()
        spec = ChunkShardingSpec(dim=0, placements=["workerfoo/cuda:1"])
        with self.assertRaisesRegex(ValueError, "Invalid worker name"):
            sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_invalid_pg_rpc_ranks(self):
        self.init_pg()

        # Init RPC with different ranks.
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            _transports=tp_transports()
        )
        rpc_backend_options.init_method = f"file://{self.file_name}"
        rank = (self.rank + 1) % self.world_size
        rpc.init_rpc(
            name=f"worker{rank}",
            rank=rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        spec = ChunkShardingSpec(dim=0, placements=["rank:1/cuda:1"])
        with self.assertRaisesRegex(
            ValueError, "Default ProcessGroup and RPC ranks must be the same"
        ):
            sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_insufficient_sharding_dims(self):
        self.init_pg()

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.empty(spec, 2, 20)

        # Validate local shard.
        local_shards = st.local_shards()
        if self.rank <= 1:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((1, 20), local_shard.size())
        else:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual(local_shard.numel(), 0)

        # Validate global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(4, len(shards_metadata))

        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_rank, 0], shard_metadata.shard_offsets)
            self.assertEqual(
                f"rank:{shard_rank}/cuda:{shard_rank}", str(shard_metadata.placement)
            )
            if shard_rank <= 1:
                self.assertEqual([1, 20], shard_metadata.shard_sizes)
            else:
                self.assertEqual([0, 20], shard_metadata.shard_sizes)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_sizes(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # Test with *args
        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
        self.assertEqual(torch.Size([10, 20]), st.size())

        # Test with single *args
        st = sharded_tensor.empty(spec, 10, init_rrefs=True)
        self.assertEqual(torch.Size([10]), st.size())

        # Test with list
        st = sharded_tensor.empty(spec, [10, 20], init_rrefs=True)
        self.assertEqual(torch.Size([10, 20]), st.size())

        # Test with tuple
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        self.assertEqual(torch.Size([10, 20]), st.size())

        # Test with row size
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        self.assertEqual(st.size(0), 10)

        # Test with col size
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        self.assertEqual(st.size(1), 20)

        # Test with negative indexed size
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        self.assertEqual(st.size(-1), 20)

        # Test with dim/ndim
        self.assertEqual(st.dim(), 2)
        self.assertEqual(st.ndim, 2)
        # Test with invalid input
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            st.size(-3)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            st.size(2)

        with self.assertRaises(TypeError):
            st = sharded_tensor.empty(spec, "foo")

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_state_dict(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        m = MyShardedModel1(spec)

        # Test save
        m._register_state_dict_hook(state_dict_hook)
        buffer = io.BytesIO()
        mod_state_dict = m.state_dict()
        mod_state_keys = mod_state_dict.keys()
        self.assertTrue("sharded_tensor1" in mod_state_keys)
        self.assertTrue("submodule.sharded_tensor2" in mod_state_keys)
        torch.save(mod_state_dict, buffer)

        # Test load.
        module_load = MyShardedModel1()
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        buffer.seek(0)
        # weights_only=False as ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
        state_dict_deser = torch.load(buffer, weights_only=False)
        module_load.load_state_dict(state_dict_deser, strict=False)

        module_load._register_state_dict_hook(state_dict_hook)
        loaded_dict_keys = module_load.state_dict().keys()
        self.assertTrue("sharded_tensor1" in loaded_dict_keys)
        self.assertTrue("submodule.sharded_tensor2" in loaded_dict_keys)
        # Verify after load.
        self.assertTrue(torch.equal(m.sharded_tensor1, module_load.sharded_tensor1))
        self.assertTrue(
            torch.equal(
                m.submodule.sharded_tensor2, module_load.submodule.sharded_tensor2
            )
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_state_dict_new_group(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:0",
                "rank:3/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        pg = dist.new_group([2, 3])

        m = MyShardedModel1(spec, pg)

        # Test save
        m._register_state_dict_hook(state_dict_hook)
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)

        # Test load.
        module_load = MyShardedModel1(spec=None, group=pg)
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        buffer.seek(0)
        with load_with_process_group(pg):
            # ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
            state_dict_deser = torch.load(buffer, weights_only=False)
            module_load.load_state_dict(state_dict_deser, strict=False)

        # Verify after load.
        self.assertTrue(torch.equal(m.sharded_tensor1, module_load.sharded_tensor1))
        self.assertTrue(
            torch.equal(
                m.submodule.sharded_tensor2, module_load.submodule.sharded_tensor2
            )
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_state_dict_no_sharded_tensors(self):
        # Verify hooks don't affect modules with no ShardedTensors.
        m = torch.nn.Linear(10, 10)

        # Test save
        state_dict_before = m.state_dict()
        m._register_state_dict_hook(state_dict_hook)
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)
        self.assertEqual(state_dict_before, m.state_dict())

        # Test load.
        module_load = torch.nn.Linear(10, 10)
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        buffer.seek(0)
        state_dict_deser = torch.load(buffer)
        module_load.load_state_dict(state_dict_deser, strict=False)

        # Verify after load.
        self.assertEqual(m.weight, module_load.weight)
        self.assertEqual(m.bias, module_load.bias)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_load_state_dict_errors(self):
        self.init_rpc()

        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        m = MyShardedModel1(spec)

        # Test save
        m._register_state_dict_hook(state_dict_hook)
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)

        pg = dist.new_group(ranks=[0, 2, 3])

        buffer.seek(0)
        if self.rank != 0:
            with self.assertRaisesRegex(RuntimeError, "Local rank at save time was"):
                with load_with_process_group(pg):
                    # ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
                    torch.load(buffer, weights_only=False)
        else:
            with self.assertRaisesRegex(
                RuntimeError, "Local world size at save time was"
            ):
                with load_with_process_group(pg):
                    # ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
                    torch.load(buffer, weights_only=False)

        dist.destroy_process_group()
        buffer.seek(0)
        with self.assertRaisesRegex(
            RuntimeError, "Need to initialize default process group"
        ):
            # ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
            torch.load(buffer, weights_only=False)
        rpc.shutdown()

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_cleanup(self):
        def create_tensors():
            spec = ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )
            sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
            sharded_tensor.empty(spec, 10, 20)

        create_tensors()
        self.assertEqual(0, len(sharded_tensor.api._sharded_tensor_map))


class TestShardedTensorEnumerable(ShardedTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_metadata(self):
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        st = sharded_tensor.empty(spec, 10, 10, init_rrefs=True)
        st_metadata = st.metadata()
        self.assertEqual(torch.Size([10, 10]), st_metadata.size)
        self.assertEqual(torch.float, st.dtype)
        self.assertEqual(torch.strided, st.layout)
        self.assertEqual(False, st.requires_grad)
        self.assertTrue(st.is_contiguous())
        self.assertFalse(st.is_pinned())

        st = sharded_tensor.empty(spec, 10, 10, requires_grad=True, init_rrefs=True)
        self.assertEqual(True, st.requires_grad)

        st = sharded_tensor.empty(spec, 10, 10, dtype=torch.double, init_rrefs=True)
        sel
```



## High-Level Overview


This Python file contains 15 class(es) and 84 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestShardedTensorMetadata`, `TestCreateTensorFromParams`, `TestShardParameter`, `TestShardTensor`, `TestModuleHookApi`, `DummyNNModule`, `TestLocalTensor`, `TestShardedTensorChunked`, `TestShardedTensorEnumerable`, `TestShardedTensorFromLocalTensor`, `TestShardedTensorFromLocalShards`, `TestShardedTensorCustomOps`, `TestShardMetadata`, `TestShardedTensorSubGroupInit`, `TestCreateTensorNoProcessGroupMode`

**Functions defined**: `test_serialize_and_deserialize`, `test_empty`, `test_shard_parameter`, `test_shard_parameter_errors`, `test_shard_tensor`, `test_shard_tensor_with_empty_shard`, `test_shard_tensor_errors`, `__init__`, `forward`, `test_reshard_output`, `test_collect_local_shard`, `test_local_tensor`, `test_local_tensor_error`, `test_sharded_tensor_metadata`, `test_complete_world_size`, `test_create_sharded_tensor_with_ones`, `test_gather_even`, `test_gather_uneven`, `test_create_sharded_tensor_with_zeros`, `test_create_sharded_tensor_with_rand`

**Key imports**: copy, io, itertools, math, pickle, sys, torch, torch.distributed as dist, distributed_c10d, rpc, sharded_tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_shard/sharded_tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `io`
- `itertools`
- `math`
- `pickle`
- `sys`
- `torch`
- `torch.distributed as dist`
- `torch.distributed`: distributed_c10d, rpc
- `torch.distributed._shard`: sharded_tensor
- `torch.distributed.remote_device`: _remote_device
- `torch.distributed._shard.sharding_spec.api`: custom_sharding_spec_op


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/_shard/sharded_tensor/test_sharded_tensor.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_shard/sharded_tensor`):

- [`test_logger.py_docs.md`](./test_logger.py_docs.md)
- [`test_sharded_tensor_reshard.py_docs.md`](./test_sharded_tensor_reshard.py_docs.md)


## Cross-References

- **File Documentation**: `test_sharded_tensor.py_docs.md`
- **Keyword Index**: `test_sharded_tensor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
