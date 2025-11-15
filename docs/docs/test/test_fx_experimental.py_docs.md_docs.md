# Documentation: `docs/test/test_fx_experimental.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_fx_experimental.py_docs.md`
- **Size**: 54,541 bytes (53.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_fx_experimental.py`

## File Metadata

- **Path**: `test/test_fx_experimental.py`
- **Size**: 80,121 bytes (78.24 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]
# ruff: noqa: F841

import functools
import math
import numbers
import operator
import pickle
import sys
import sympy
import tempfile
import typing
import unittest
from types import BuiltinFunctionType
from typing import NamedTuple, Optional, Union
from collections.abc import Callable

import torch
import torch.fx.experimental.meta_tracer
import torch.fx.experimental.optimization as optimization
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.experimental import merge_matmul
from torch.fx.experimental.accelerator_partitioner import Partitioner
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.normalize import NormalizeArgs, NormalizeOperators
from torch.fx.experimental.partitioner_utils import (
    Device,
    get_latency_of_partitioned_graph,
    get_partition_to_latency_mapping,
    NodeLatency,
    PartitionerConfig,
    PartitionMode,
)
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.experimental.schema_type_annotation import AnnotateTypesWithSchema
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.operator_schemas import (
    _torchscript_type_to_python_type,
    create_type_hint,
    normalize_function,
    normalize_module,
    type_matches,
)
from torch.fx.passes import graph_manipulation
from torch.fx.passes.param_fetch import lift_lowering_attrs_to_nodes
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.split_module import split_module
from torch.fx.passes.annotate_getitem_nodes import annotate_getitem_nodes
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_nn import module_tests, get_new_module_tests
from torch.testing._internal.common_utils import TEST_Z3, run_tests, TestCase, TEST_WITH_CROSSREF
from torch.testing._internal.jit_utils import JitTestCase
import torch.utils._pytree as pytree

try:
    import torchvision.models
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")
skipIfNoMkldnn = unittest.skipIf(
    not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()),
    "no MKLDNN",
)


def symbolic_trace_with_rewrite(root: Union[torch.nn.Module, Callable]) -> GraphModule:
    return GraphModule(
        root if isinstance(root, torch.nn.Module) else torch.nn.Module(),
        RewritingTracer().trace(root),
    )


class TestFXExperimental(JitTestCase):
    def test_find_single_partition(self):
        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(1)
        b = torch.rand(1)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 150, 1),
            Device("dev_2", 125, 2),
        ]
        partitioner_config = PartitionerConfig(devices)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        assert dag.nodes[0].logical_device_ids == [1]

    def test_lack_of_devices(self):
        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        b = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [Device("dev_0", 4, 0), Device("dev_1", 4, 1)]
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        catch_runtime_error = False
        try:
            ret = partitioner.partition_graph(traced, m, partitioner_config)
        except RuntimeError:
            catch_runtime_error = True
        assert catch_runtime_error

    def test_large_node_error(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                linear = self.linear(a)
                add = linear + a
                return add

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        partitioner = Partitioner()
        devices = [
            Device("dev_0", 40, 0),
            Device("dev_1", 40, 0),
            Device("dev_2", 40, 0),
            Device("dev_3", 40, 0),
            Device("dev_4", 40, 0),
        ]
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        catch_runtime_error = False
        try:
            ret = partitioner.partition_graph(traced, m, partitioner_config)
        except RuntimeError:
            catch_runtime_error = True
        assert catch_runtime_error

    def test_partition_node_manipulation(self):
        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                add_1 = a + b
                add_2 = add_1 + torch.rand(4)
                add_3 = add_2 + torch.rand(4)
                return add_3

        m = TestModule()
        traced = symbolic_trace(m)
        a, b = torch.rand(4), torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [Device("dev_0", 1000, 0)]
        partitioner_config = PartitionerConfig(devices)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        partition = partitioner.partitions[0]
        assert partition.used_mem_bytes == 112
        # Select add_2 node to remove
        selected_node = None
        for node in partition.nodes:
            if node.name == "add_2":
                selected_node = node
        partition.remove_node(selected_node)
        assert partition.used_mem_bytes == 80

    def test_size_based_partition(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.c = torch.rand(4)

            def forward(self, a, b):
                add_1 = a + b
                linear = self.linear(add_1)
                add_2 = linear + self.c
                return add_2

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        b = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 125, 1),
            Device("dev_2", 125, 2),
        ]
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        for i, node in enumerate(dag.nodes):
            assert node.logical_device_ids == [i]

    def test_partition_device_mapping(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                b = torch.rand(4)
                add_1 = a + b
                linear_1 = self.linear(add_1)
                add_2 = torch.rand(4) + a
                add_3 = add_2 + linear_1
                return add_3

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        partitioner = Partitioner()
        devices = [Device("dev_0", 120, 0), Device("dev_1", 160, 1)]
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a), module_with_submodules(a))
        for i, node in enumerate(dag.nodes):
            if i == 1:
                assert node.logical_device_ids == [1]
            else:
                assert node.logical_device_ids == [0]

    def test_sparse_nn_partition(self):
        class MyRecommendationModule(torch.nn.Module):
            def create_mlp(self, num_of_layers: int, input_size: int, output_size: int):
                layers = torch.nn.ModuleList()
                for _ in range(num_of_layers):
                    ll = torch.nn.Linear(input_size, output_size)
                    layers.append(ll)
                    layers.append(torch.nn.ReLU())
                return layers

            def __init__(self) -> None:
                super().__init__()
                layers = self.create_mlp(4, 4, 4)
                self.bottom_layers = torch.nn.Sequential(*layers)
                layers = self.create_mlp(3, 24, 24)
                self.top_layers = torch.nn.Sequential(*layers)
                self.embedding_layers = torch.nn.ModuleList()
                el = torch.nn.EmbeddingBag(500000, 4, mode="sum", sparse=True)
                self.embedding_layers.append(el)
                for _ in range(3):
                    el = torch.nn.EmbeddingBag(1000000, 4, mode="sum", sparse=True)
                    self.embedding_layers.append(el)
                el = torch.nn.EmbeddingBag(500000, 4, mode="sum", sparse=True)
                self.embedding_layers.append(el)

            def forward(self, a, b, offset):
                x = self.bottom_layers(a)
                y = []
                c = []
                for _ in range(len(self.embedding_layers)):
                    temp = torch.randint(10, (8,))
                    c.append(temp + b)
                for i in range(len(self.embedding_layers)):
                    if i % 2 == 0:
                        y.append(self.embedding_layers[i](c[i], offset))
                    else:
                        y.append(
                            self.embedding_layers[i](torch.randint(10, (8,)), offset)
                        )
                z = torch.cat([x] + y, dim=1)
                p = self.top_layers(z)
                return p

        m = MyRecommendationModule()
        a = torch.rand(2, 4)
        b = torch.randint(10, (8,))
        offset = torch.randint(1, (2,))
        traced = symbolic_trace(m)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b, offset])
        devices = [
            Device("dev_0", 33000000, 0),
            Device("dev_1", 33000000, 1),
            Device("dev_2", 33000000, 2),
        ]
        partitioner_config = PartitionerConfig(devices, PartitionMode.sparse_nn)
        partitioner = Partitioner()
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b, offset), module_with_submodules(a, b, offset))
        assert len(module_with_submodules.graph.nodes) == 24

    def test_partition_latency(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + linear_1
                add_4 = add_2 + add_3
                return add_4

        def get_node_to_latency_mapping(fx_module: GraphModule):
            """Given a fx module, generate node latency for each node
            based on the size of each node
            """
            node_to_latency_mapping: dict[Node, NodeLatency] = {}
            for node in fx_module.graph.nodes:
                if node.op not in {"output", "placeholder", "get_attr"}:
                    if node.size_bytes.total_size == node.size_bytes.output_size:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, 2.0 * node.size_bytes.total_size
                        )
                    else:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, node.size_bytes.output_size
                        )
            return node_to_latency_mapping

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        node_to_latency_mapping = get_node_to_latency_mapping(traced)
        devices = [Device("dev_0", 200, 0), Device("dev_1", 200, 1)]
        partitioner = Partitioner()
        partitioner_config = PartitionerConfig(devices)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        self.assertEqual(traced(a), module_with_submodules(a))
        partitions = partitioner.partitions
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            partitions, node_to_latency_mapping
        )
        for p in partition_to_latency_mapping:
            if p.partition_id == 0:
                assert partition_to_latency_mapping[p] == (128.0, 80.0, 160.0)
            else:
                assert partition_to_latency_mapping[p] == (16.0, 32.0, 32.0)
        transfer_rate_bytes_per_sec = 2
        critical_path_latency_sec = get_latency_of_partitioned_graph(
            partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec
        )
        assert critical_path_latency_sec == 208.0

    def test_cost_aware_partition(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + torch.rand(4)
                add_4 = add_2 + linear_1
                add_5 = add_3 + add_4
                return add_5

        def get_node_to_latency_mapping(fx_module: GraphModule):
            node_to_latency_mapping: dict[Node, NodeLatency] = {}
            for node in fx_module.graph.nodes:
                if node.op not in {"output", "placeholder", "get_attr"}:
                    if node.size_bytes.total_size == node.size_bytes.output_size:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, 1
                        )
                    else:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, node.size_bytes.output_size
                        )
            return node_to_latency_mapping

        m = MyModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 125, 1),
            Device("dev_2", 125, 2),
            Device("dev_3", 125, 3),
        ]
        node_to_latency_mapping = get_node_to_latency_mapping(traced)
        partitioner_config = PartitionerConfig(
            devices,
            mode=PartitionMode.cost_aware,
            transfer_rate_bytes_per_sec=2,
            node_to_latency_mapping=node_to_latency_mapping,
        )
        partitioner = Partitioner()
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a), module_with_submodules(a))
        partitions = partitioner.partitions
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            partitions, node_to_latency_mapping
        )
        critical_path_latency_sec = get_latency_of_partitioned_graph(
            partitions,
            partition_to_latency_mapping,
            partitioner_config.transfer_rate_bytes_per_sec,
        )
        assert critical_path_latency_sec == 160.0

    def test_aot_based_partition(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = torch.rand(4)
                self.c = torch.rand(4)

            def forward(self, a):
                add_1 = a + self.b
                add_2 = self.c + add_1
                return add_2

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        node_to_partition_id = {}
        partition_to_logical_devices = {}
        count = 0
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        for node in traced.graph.nodes:
            if node.op not in {"placeholder", "get_attr", "output"}:
                node_to_partition_id[node] = count
                partition_to_logical_devices[count] = [0]
                count += 1
        devices = [Device("dev_0", 200, 0)]
        partitioner_config = PartitionerConfig(
            devices=devices,
            mode=PartitionMode.aot_based,
            node_to_partition_mapping=node_to_partition_id,
            partition_to_logical_device_mapping=partition_to_logical_devices,
        )
        partitioner = Partitioner()
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(module_with_submodules(a), traced(a))
        for node in dag.nodes:
            assert node.size_bytes == 48
            assert node.logical_device_ids == [0]

    def test_replace_target_nodes_with(self):
        class testModule(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = testModule()
        traced = symbolic_trace(m)
        input1 = torch.randn(1)
        input2 = torch.randn(1)
        assert (input1 + input2) == traced(input1, input2)
        graph_manipulation.replace_target_nodes_with(
            fx_module=traced,
            old_op="call_function",
            old_target=operator.add,
            new_op="call_function",
            new_target=operator.mul,
        )
        assert (input1 * input2) == traced(input1, input2)

    def test_saturate_host(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + linear_1
                add_4 = add_2 + add_3
                return add_4

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        devices = [
            Device("dev_0", 200, 0),
            Device("dev_1", 200, 1),
            Device("dev_2", 100, 2),
            Device("dev_3", 100, 3),
            Device("dev_4", 200, 4),
            Device("dev_5", 100, 5),
        ]
        partitioner = Partitioner()
        # Without host saturation, the model will be split into two partitions.
        # dev_0 holds partition 0 of 192 bytes and dev_1 holds partition 1 of 48 bytes.
        partitioner_config = PartitionerConfig(devices, saturate_host=True)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        self.assertEqual(traced(a), module_with_submodules(a))

        partitions = partitioner.partitions
        self.assertEqual(len(partitions), 2)
        # With host saturation, partition 1 will be replicated to dev_4, and partition 2
        # will be replicated to dev_2.
        self.assertEqual(partitions[0].logical_device_ids, [0, 4])
        self.assertEqual(partitions[1].logical_device_ids, [1, 2])

    @skipIfNoTorchVision
    def test_conv_bn_fusion(self):
        rn18 = resnet18().eval()
        traced = symbolic_trace(rn18)
        fused = optimization.fuse(traced)

        self.assertTrue(
            all(not isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )

        N, C, H, W = 20, 3, 224, 224
        inp = torch.randn(N, C, H, W)

        self.assertEqual(fused(inp), rn18(inp))

    def test_conv_bn_fusion_not_running_state(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(32, 64, 3, stride=2)
                self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        model = M().eval()

        traced = symbolic_trace(model)
        fused = optimization.fuse(traced)
        inp = torch.randn([1, 32, 50, 50])

        # bn need not be folded in conv
        self.assertTrue(
            any(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )
        self.assertEqual(fused(inp), model(inp))

    def test_conv_bn_fusion_mixed_dtype(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, dtype=torch.bfloat16)
                self.bn = torch.nn.BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        model = M().eval()

        traced = symbolic_trace(model)
        fused = optimization.fuse(traced)
        inp = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)

        self.assertTrue(
            all(not isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )
        self.assertEqual(fused(inp), model(inp))

    def test_call_to_assert_no_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint()

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_meta_tracer(self):
        class MetaTracerTestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=42, embedding_dim=16)
                self.layernorm = torch.nn.LayerNorm(16)

            def forward(self, x):
                emb = self.emb(x)
                emb = emb + torch.arange(emb.shape[-1], dtype=torch.float, device=emb.device)
                lol = self.layernorm(emb)
                return torch.relu(lol) if lol.shape[0] < 30 else torch.sigmoid(lol)

        mttm = MetaTracerTestModule()
        for BS in [15, 35]:
            x = torch.zeros(BS, dtype=torch.long).random_(42)
            meta_args = {'x' : x.to(device='meta')}
            gm = torch.fx.experimental.meta_tracer.symbolic_trace(mttm, meta_args=meta_args)
            torch.testing.assert_close(gm(x), mttm(x))

            # Test serialization/deserialization
            with tempfile.TemporaryDirectory() as tmp_dir:
                with open(f'{tmp_dir}/meta_module.pkl', 'wb') as f:
                    pickle.dump(gm, f)

                with open(f'{tmp_dir}/meta_module.pkl', 'rb') as f:
                    loaded = pickle.load(f)

                torch.testing.assert_close(loaded(x), mttm(x))


    def test_call_to_assert_with_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b, "test message"
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint()

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, "test message"):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_call_to_assert_with_empty_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b, ""
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint()

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_call_to_assert_with_multiline_message(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                error_msg = """
An error message with
terrible spacing
                """
                assert a == b, error_msg
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint()

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target == torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        error_msg = """
An error message with
terrible spacing
    """
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, error_msg):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_subgraph_creation(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x, y):
                z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
                w = self.linear(y).clamp(min=0.0, max=1.0)
                return z + w

        # symbolically trace model
        my_module = MyModule()
        my_module_traced = symbolic_trace(my_module)

        # random mod partitioning
        partition_counter = 0
        NPARTITIONS = 3

        # Add some random meta info to make sure it is kept around.
        for node in my_module_traced.graph.nodes:
            if node.op != "output":
                node.meta["test_meta_info"] = True

        def mod_partition(node: Node):
            nonlocal partition_counter
            partition = partition_counter % NPARTITIONS
            partition_counter = (partition_counter + 1) % NPARTITIONS
            return partition

        # split module in module with submodules
        module_with_submodules = split_module(
            my_module_traced, my_module, mod_partition
        )

        # Check that test_meta_info was still on all nodes.
        submodules = dict(module_with_submodules.named_modules())
        for node in module_with_submodules.graph.nodes:
            if node.op == "call_module":
                submod = submodules[node.target]
                self.assertTrue(isinstance(submod, torch.fx.GraphModule))
                for submod_node in submod.graph.nodes:
                    if submod_node.op != "output":
                        stored_op = submod_node.meta.get("test_meta_info")
                        self.assertTrue(stored_op is not None and stored_op)

        x = torch.rand(3, 4)
        y = torch.rand(3, 4)

        orig_out = my_module_traced(x, y)
        submodules_out = module_with_submodules(x, y)

        self.assertEqual(orig_out, submodules_out)

    def test_split_module_input_names(self):
        class Mod(torch.nn.Module):
            def forward(self, x, a0, a1, b0, b1, c0, c1):
                x = x + (a0 ** 2) + (a1 / 2)
                x = x + (b0 ** 2) + (b1 / 2)
                x = x + (c0 ** 2) + (c1 / 2)
                return x

        mod = Mod()
        traced = torch.fx.symbolic_trace(mod)

        seen = 0

        def split(n):
            nonlocal seen
            result = seen // 4
            seen += 1
            return result

        split = split_module(traced, mod, split, keep_original_input_name=False)

        # All the submodules should take in the inputs in the same order.
        args = [torch.tensor(2.), torch.tensor(3.), torch.tensor(4.)]
        output0 = split.submod_0(*args)
        output1 = split.submod_1(*args)
        output2 = split.submod_2(*args)
        self.assertEqual(output0, output1)
        self.assertEqual(output1, output2)

        # Each submodule should have normalized input names
        def check_ph(gm):
            nodes = list(gm.graph.nodes)
            self.assertEqual(nodes[0].target, "arg_0")
            self.assertEqual(nodes[1].target, "arg_1")
            self.assertEqual(nodes[2].target, "arg_2")

        check_ph(split.submod_0)
        check_ph(split.submod_1)
        check_ph(split.submod_2)

    def test_split_module_dead_code(self):
        class ModWithDeadCode(torch.nn.Module):
            def forward(self, x):
                output = x * 2  # we want this
                dead_line = x + 2  # this is dead
                return output

        mod = ModWithDeadCode()
        traced = torch.fx.symbolic_trace(mod)

        # split into before (0), target (1), and after(2)
        saw_mul = False

        def split_callback(n):
            nonlocal saw_mul
            if n.target == operator.mul:
                saw_mul = True
                return 1

            if not saw_mul:
                return 0
            if saw_mul:
                return 2

        split = split_module(traced, mod, split_callback)

        x = torch.randn((5,))
        torch.testing.assert_close(
            split(x), traced(x)
        )

    def test_split_module_return_node(self):
        def foo(x):
            x.add_(1)

        gm = make_fx(foo, tracing_mode="fake")(torch.randn(3,))

        def cb(_):
            return 1

        sp_gm = split_module(gm, None, cb)
        submod_gm = sp_gm.submod_1
        for node in submod_gm.graph.nodes:
            if node.op == "output":
                break
        else:
            raise RuntimeError("Expected the subgraph to have an output node.")


    def test_split_module_kwargs_expansion(self):
        class ModuleWithKwargsExpansion(torch.nn.Module):
            def forward(self, x, **kwargs):
                return x + kwargs['foo']

        mod = ModuleWithKwargsExpansion()
        traced = torch.fx.symbolic_trace(mod)

        seen_getitem = False

        def split_callback(n):
            nonlocal seen_getitem
            split_idx = int(seen_getitem)
            if n.target == operator.getitem:
                seen_getitem = True
            return split_idx

        split = split_module(traced, mod, split_callback)

        x = torch.randn(5, 3)
        foo = torch.randn(5, 3)
        torch.testing.assert_close(split(x, foo=foo), traced(x, foo=foo))

    @skipIfNoTorchVision
    def test_subgraph_trivial_resnet(self):
        # Smoke test trivially splitting resnet into 1 partition works
        # There was an issue before causing submodule names to be aliased
        m = resnet18()
        traced = symbolic_trace(m)
        a = torch.rand(64, 3, 7, 7)
        module_with_submodules = split_module(traced, m, lambda node: 0)
        module_with_submodules(a)

    def test_split_module_default_arg(self):
        class ModelToTrace(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(512, 512)

            def forward(self, x, targets=None):
                x = self.lin(x)

                if targets is not None:
                    x = x + targets

                return x

        mtt = ModelToTrace()
        traced = torch.fx.symbolic_trace(mtt, concrete_args={'targets': None})

        split = split_module(traced, mtt, lambda node: 0)

        x = torch.randn(50, 512)
        torch.testing.assert_close(split(x), traced(x))

    def test_split_module_keep_original_order_and_noop_graph(self):
        # Verify that split_module returns a similar no-op graph
        # for `keep_original_order={True|False}`.
        def fn(x):
            return (x,)

        g = make_fx(fn, tracing_mode="fake")(torch.randn(3, 3))

        # g.graph.print_tabular()
        # opcode       name    target    args       kwargs
        # -----------  ------  --------  ---------  --------
        # placeholder  x_1     x_1       ()         {}
        # output       output  output    ((x_1,),)  {}

        def _test_split_graph(split_gm):
            # Verify that the split_gm has same structure as original
            self.assertEqual(len(split_gm.graph.nodes), 2)

            nodes = list(split_gm.graph.nodes)
            self.assertEqual(nodes[0].op, "placeholder")
            self.assertEqual(nodes[1].op, "output")

        # `keep_original_order=False`
        _test_split_graph(split_module(g, None, split_callback=lambda _ : 0, keep_original_order=False))

        # `keep_original_order=True`
        _test_split_graph(split_module(g, None, split_callback=lambda _ : 0, keep_original_order=True))

    @unittest.skipIf(TEST_WITH_CROSSREF, "See https://github.com/pytorch/pytorch/issues/160077")
    def test_split_module_symint_dependency_handling(self):
        # Based on the code from - transformers/models/granitemoe/modeling_granitemoe.py
        class GraniteMoeTopKGating(torch.nn.Module):
            def __init__(self, input_size: int, num_experts: int, top_k: int):
                super().__init__()

                self.num_experts = num_experts
                self.input_size = input_size
                self.top_k = top_k

                self.layer = torch.nn.Linear(input_size, num_experts, bias=False)

            def forward(self, hidden_states):
                # compute the top_k routing decision
                logits = self.layer(hidden_states).float()  # [batch_size x seq_len, num_experts]
                top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)  # [num_tokens, top_k]
                top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)  # [num_tokens, top_k]

                # compute number of input given to each expert
                zeros = torch.zeros(
                    [top_k_gates.size(0), self.num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device
                )  # [num_tokens, num_experts]
                gates = zeros.scatter(1, top_k_indices, 1)  # [num_tokens, num_experts]
                expert_size = gates.long().sum(0)  # [num_experts,]
                expert_size = expert_size.tolist()

                # sort and group input tokens according to expert assignment
                top_k_experts = top_k_indices.flatten()  # [num_tokens * top_k]
                _, index_sorted_experts = top_k_experts.sort(0)  # [num_tokens * top_k]
                batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")  # [num_tokens * top_k]

                # gather the gate values for grouped input tokens
                top_k_gates = top_k_gates.flatten()  # [num_tokens * top_k]
                batch_gates = top_k_gates[index_sorted_experts]  # [num_tokens * top_k]

                return index_sorted_experts, batch_index, batch_gates, expert_size, logits

        class GraniteMoeMoE(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.input_size = 32
                self.num_local_experts = 4

                num_experts_per_tok = 2
                self.router = GraniteMoeTopKGating(
                    input_size=self.input_size,
                    num_experts=self.num_local_experts,
                    top_k=num_experts_per_tok,
                )

            def forward(self, layer_input):
                _, batch_index, _, expert_size, _ = self.router(layer_input)
                expert_inputs = layer_input[batch_index]
                return expert_inputs.split(expert_size, dim=0)

        moe = GraniteMoeMoE()
        inp = torch.randn([32, 32])

        expected = moe(inp)

        PARTITION_ID = 0
        PARTITION_OPS_CTR = 0
        NODE_PARTITION_MAP = {}

        # `callback` is called multiple times with same `node` in `split_module`.
        # Cache the result such that partition id is consistent across calls.
        def callback(node) -> int:
            nonlocal PARTITION_ID, PARTITION_OPS_CTR, NODE_PARTITION_MAP
            if node in NODE_PARTITION_MAP:
                return NODE_PARTITION_MAP[node]

            if PARTITION_OPS_CTR % 5 == 0:
                PARTITION_ID += 1

            PARTITION_OPS_CTR += 1

            NODE_PARTITION_MAP[node] = PARTITION_ID
            return PARTITION_ID

        def backend(gm, inps):
            split_gm = split_module(gm, root_m=None, split_callback=callback,
                                    keep_original_order=True, keep_original_node_name=True)
            return split_gm

        actual = torch.compile(moe, backend=backend)(inp)
        torch.testing.assert_close(actual, expected)

    def test_normalize_binary_operators(self):
        ops_to_test = {
            torch.add,
            torch.mul,
            torch.sub,
            torch.div,
            torch.floor_divide,
            torch.remainder,
            torch.eq,
            torch.ne,
            torch.lt,
            torch.le,
            torch.gt,
            torch.ge,
        }

        # Test Tensor/Tensor callsite
        for op in ops_to_test:

            class WrapperMod(torch.nn.Module):
                def forward(self, x, y):
                    return op(x, y)

            traced = symbolic_trace(WrapperMod())
            normalized = NormalizeOperators(traced).transform()
            x, y = torch.randn(3, 4), torch.randn(3, 4)
            torch.testing.assert_close(traced(x, y), normalized(x, y))
            self.assertFalse(
                any(n.target in ops_to_test for n in normalized.graph.nodes)
            )

        # Test Tensor/scalar callsite
        for op in ops_to_test:

            class WrapperMod(torch.nn.Module):
                def forward(self, x):
                    return op(x, 42)

            traced = symbolic_trace(WrapperMod())
            normalized = NormalizeOperators(traced).transform()
            x = torch.randn(3, 4)
            torch.testing.assert_close(traced(x), normalized(x))
            self.assertFalse(
                any(n.target in ops_to_test for n in normalized.graph.nodes)
            )

    @skipIfNoTorchVision
    def test_normalize_args(self):
        m = resnet18()

        class FunctionalTracer(torch.fx.Tracer):
            def is_leaf_module(
                self, m: torch.nn.Module, module_qualified_name: str
            ) -> bool:
                # `leaves` contains the set of standard `nn.Modules` that are not
                # currently symbolically traceable. Ideally this set would be empty
                leaves = {torch.nn.BatchNorm2d}
                return type(m) in leaves

        traced = torch.fx.GraphModule(m, FunctionalTracer().trace(m))

        input = torch.randn(5, 3, 224, 224)
        ref_outs = traced(input)

        ShapeProp(traced).propagate(input)
        traced = NormalizeArgs(traced).transform()

        modules = dict(traced.named_modules())

        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target != operator.add:
                self.assertEqual(len(node.args), 0)
            elif node.op == "call_module":
                submod_class = modules[node.target].__class__
                nn_class = getattr(torch.nn, submod_class.__name__)
                if submod_class == nn_class:
                    self.assertEqual(len(node.args), 0)
        traced(input)
        self.assertEqual(traced(input), ref_outs)

    def test_normalize_modules_exhaustive(self):
        """
        Exhaustively test `Node.normalized_arguments` on all standard
        torch.nn Module classes
        """
        for test_params in module_tests + get_new_module_tests():
            if "constructor" not in test_params:
                constructor = getattr(torch.nn, test_params["module_name"])
            else:
                constructor = test_params["constructor"]

            if "constructor_args" not in test_params:
                args = ()
            else:
                args = test_params["constructor_args"]

            mod = constructor(*args)
            # Skip modules that are not standard `torch.nn`
            # instances, including functionals. (functionals
            # are tested in test_normalize_args)
            if mod.__class__.__name__ not in dir(torch.nn):
                continue

            if "input_fn" not in test_params:
                inputs = torch.randn(test_params["input_size"])
            else:
                inputs = test_params["input_fn"]()

            if not isinstance(inputs, (tuple, list)):
                inputs = (inputs,)

            params = ", ".join(f"v{i}" for i in range(len(inputs)))

            # Generate a class to wrap this standard `nn.Module` instance
            test_classname = f"Test{mod.__class__.__name__}"
            test_mod_code = f"""
class {test_classname}(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, {params}):
        return self.mod({params})
            """

            gbls = {"torch": torch}
            exec(test_mod_code, gbls)

            test_instance = gbls[test_classname](mod)
            traced = symbolic_trace(test_instance)

            # Use `Node.normalized_arguments` to get a new set of arguments
            # to feed to the Module. Then, rewrite the node to only take
            # in those arguments as kwargs
            modules = dict(traced.named_modules())
            for node in traced.graph.nodes:
                if node.op == "call_module":
                    submod_class = modules[node.target].__class__
                    nn_class = getattr(torch.nn, submod_class.__name__)
                    if submod_class == nn_class:
                        normalized_args = node.normalized_arguments(traced)
                        normalized_args2 = normalize_module(
                            traced, node.target, node.args, node.kwargs
                        )
                        assert normalized_args == normalized_args2
                        assert normalized_args
                        node.args = normalized_args.args
                        node.kwargs = normalized_args.kwargs

            traced.recompile()

            # These Modules have an RNG in their forward, so testing
            # correctness by comparing outputs is not correct. Skip that
            # check for these
            stochastic_modules = {"FractionalMaxPool2d", "FractionalMaxPool3d", "RReLU"}

            if mod.__class__.__name__ not in stochastic_modules:
                self.assertEqual(traced(*inputs), mod(*inputs))

            traced = NormalizeArgs(symbolic_trace(test_instance)).transform()
            modules = dict(traced.named_modules())
            for node in traced.graph.nodes:
                if node.op == "call_module":
                    submod_class = modules[node.target].__class__
                    nn_class = getattr(torch.nn, submod_class.__name__)
                    if submod_class == nn_class:
                        self.assertEqual(len(node.args), 0)

    def test_normalize_args_preserve_meta(self):
        class MyModule(torch.nn.Module):
            def forward(self, a):
                return torch.add(a, 3)

        m = MyModule()
        traced = symbolic_trace(m)

        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target == torch.add:
                node.meta["my_key"] = 7
                break
        else:
            self.fail("Didn't find call_function torch.add")

        input = torch.randn(2, 3)
        ShapeProp(traced).propagate(input)
        traced = NormalizeArgs(traced).transform()

        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target == torch.add:
                self.assertTrue("my_key" in node.meta)
                self.assertEqual(node.meta["my_key"], 7)
                break
        else:
            self.fail("Didn't find call_function torch.add")

    def test_normalize_args_perserve_type(self):
        class MyModule(torch.nn.Module):
            def forward(self, a: list[torch.Tensor]):
                return torch.add(a[0], a[1])

        m = MyModule()
        traced = symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()

        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(node.type, list[torch.Tensor])

    @skipIfNoTorchVision
    def test_annotate_returns_with_schema(self):
        m = resnet18()

        traced_modules = symbolic_trace(m)
        traced_modules_annotated = AnnotateTypesWithSchema(traced_modules).transform()
        for node in traced_modules_annotated.graph.nodes:
            if node.type is None:
                check = (node.op, node.target)
                self.assertIn(
                    check,
                    {
                        ("placeholder", "x"),
                        ("call_module", "maxpool"),
                        ("call_function", operator.add),
                        ("call_function", torch.flatten),
                        ("output", "output"),
                    }
                )

        # Smoke test torchscript compilation since now we're emitting type annotations
        torch.jit.script(traced_modules_annotated)

        class FunctionalTracer(torch.fx.Tracer):
            def is_leaf_module(
                self, m: torch.nn.Module, module_qualified_name: str
            ) -> bool:
                # `leaves` contains the set of standard `nn.Modules` that are not
                # currently symbolically traceable. Ideally this set would be empty
                leaves = {torch.nn.BatchNorm2d}
                return type(m) in leaves

        traced_functionals = torch.fx.GraphModule(m, FunctionalTracer().trace(m))

        traced_functionals_annotated = AnnotateTypesWithSchema(
            traced_functionals
        ).transform()
        for node in traced_functionals_annotated.graph.nodes:
            if node.type is None:
                check = (node.op, node.target)
                excluded_nodes = {
                    ("placeholder", "x"),
                    # Return type differs based on boolean dispatch :(
                    ("call_function", torch.nn.functional.max_pool2d),
                    ("output", "output"),
                }
                # AnnotateTypesWithSchema doesn't work with bound C++ functions
                if not isinstance(node.target, BuiltinFunctionType):
                    self.assertIn(check, excluded_nodes)

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



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

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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
python docs/test/test_fx_experimental.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_fx_experimental.py_docs.md_docs.md`
- **Keyword Index**: `test_fx_experimental.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
