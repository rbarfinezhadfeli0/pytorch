# Documentation: `torch/_inductor/codegen/cpp_wrapper_cpu.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/cpp_wrapper_cpu.py`
- **Size**: 129,255 bytes (126.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import ctypes
import functools
import math
import os
import sys
import textwrap
from itertools import chain, count
from typing import Any, Optional, Protocol, TYPE_CHECKING, Union

import sympy

import torch
import torch._higher_order_ops.torchbind
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch._ops
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch.fx.experimental.symbolic_shapes import ConvertIntKey, DivideByKey, SymTypes
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config, cpp_builder, ir
from ..ir import ExternKernel
from ..utils import _align, DeferredLineBase, LineContext, normalize_name
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import get_device_op_overrides, IndentedBuffer, Kernel
from .cpp_utils import cexpr, DEVICE_TO_ATEN, DEVICE_TO_INT, DTYPE_TO_ATEN, DTYPE_TO_CPP
from .wrapper import (
    EnterSubgraphLine,
    ExitSubgraphLine,
    PythonWrapperCodegen,
    SymbolicCallArg,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..graph import GraphLowering

    # At most, the list nesting can go one layer deep.
    _OUTPUT_ARGS_TYPE = list[Union[Optional[str], list[Optional[str]]]]

    from ..scheduler import BaseSchedulerNode


class HasWriteLine(Protocol):
    def writeline(self, line: Union[LineContext, DeferredLineBase, str]) -> None: ...


class CppWrapperCpu(PythonWrapperCodegen):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        if not hasattr(self, "device"):
            self.device = "cpu"
        # must be initialized prior to calling super().__init__()
        self.included_devices: OrderedSet[str] = OrderedSet()
        self.model_class_name_suffix = (
            ""
            if config.aot_inductor.dynamic_linkage
            else config.aot_inductor.model_name_for_generated_files
        )
        self.aoti_model_class_name = f"AOTInductorModel{self.model_class_name_suffix}"

        super().__init__()

        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.comment = "//"
        self.none_str = "nullptr"
        self.supports_intermediate_hooks = False
        self.kernel_callsite_id = count()
        self.int_array_id = count()  # for int array local variable declarations
        self.declared_int_array_vars: OrderedSet[str] = OrderedSet()
        self.tmp_tensor_id = count()  # for tmp tensor local variable declarations
        self.arg_var_id = count()
        self.used_cached_devices: OrderedSet[str] = OrderedSet()
        self.used_cached_dtypes: OrderedSet[str] = OrderedSet()
        self.used_cached_layouts: OrderedSet[str] = OrderedSet()
        self.used_cached_memory_formats: OrderedSet[str] = OrderedSet()
        self.used_cond_predicate: OrderedSet[str] = OrderedSet()
        self.cached_output_id = count()
        self.scalar_to_tensor_id = count()
        self.custom_op_wrapper_loaded = False
        # For GEMM kernels that must be initialized and are resolved at linking.
        self.initialized_kernels: dict[str, Kernel] = {}
        self.device_codegen = get_device_op_overrides(self.device)
        # only need to include each header once
        self.include_extra_header = functools.lru_cache(None)(  # type: ignore[method-assign]
            self._include_extra_header
        )

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperCpu()

    @staticmethod
    def _generate_temporary_array_pointer(
        c_type: str, elements: Sequence[str], *, force_mutable: bool = False
    ) -> str:
        """Get a pointer to an array that only exists for the duration of the C++
        statement it's used in."""
        # If the c_type is already a pointer, return a mutable pointer to the array.
        # Otherwise, return a const pointer.  In the C-shim API, pointer types are only
        # const-qualified with respect to the underlying value, not any nested pointers.
        # e.g. const double** is possible, but not const double* const*.  This means
        # that an array containing pointers must _already_ be properly const-qualified
        # by the c_type, and not add additional const-ness.
        # MSVC does not support implicitly converting a const iterator to a const pointer.
        ptr_call = (
            "data()"
            if force_mutable or c_type.endswith("*") or cpp_builder.is_msvc_cl()
            else "cbegin()"
        )
        return (
            f"std::array<{c_type}, {len(elements)}>{{{', '.join(elements)}}}.{ptr_call}"
        )

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        graph_name="",
        original_fxnode_name=None,
    ):
        """
        Generates kernel call code.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        assert arg_types is not None and len(call_args) == len(arg_types), (
            "Mismatch call_args and arg_types in generate_kernel_call:\n"
            f"call_args: {call_args}\n"
            f"arg_types: {arg_types}"
        )
        new_args = []
        for idx, arg in enumerate(call_args):
            if "*" in arg_types[idx]:
                new_args.append(f"({arg_types[idx]})({arg}.data_ptr())")
            else:
                # arg is a scalar
                new_args.append(arg)
        # debug printer related logic for cpp kernel type.
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args,
            kernel_name,
            None,
            None,
            "cpp",
        )
        with debug_printer_manager:
            self.writeline(self.wrap_kernel_call(kernel_name, new_args))

    def write_constant(self, name, hashed):
        # include a hash so our code cache gives different constants different files
        self.header.writeline(f"// {name} {hashed}")

    @staticmethod
    def get_device_include_path(device: str) -> str:
        if V.graph.aot_mode:
            return f"#include <torch/csrc/inductor/aoti_include/{device}.h>"
        return f"#include <torch/csrc/inductor/cpp_wrapper/{device}.h>"

    def add_device_include(self, device: str) -> None:
        if device in self.included_devices:
            return

        self.included_devices.add(device)

        # Add the default header for this device, plus any C-shim extensions that are
        # present.
        self.header.splice(self.get_device_include_path(device))
        extend_aoti_c_shim_include = (
            f"torch/csrc/inductor/aoti_torch/generated/extend/c_shim_{self.device}.h"
        )
        extend_aoti_c_shim_path = os.path.join(
            os.path.dirname(torch.__file__),
            "include",
            extend_aoti_c_shim_include,
        )
        if os.path.exists(extend_aoti_c_shim_path):
            self.header.splice(f"#include <{extend_aoti_c_shim_include}>")

    def write_header(self):
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        if not V.graph.aot_mode:
            self.header.splice(
                """
                import torch
                from torch._inductor.codecache import CppWrapperCodeCache

                cpp_wrapper_src = (
                r'''
                """
            )

        for device in V.graph.device_types:
            if device != "meta":
                self.add_device_include(device)

        if V.graph.aot_mode:
            if config.aot_inductor.dynamic_linkage:
                with open(
                    os.path.join(
                        os.path.dirname(__file__), "aoti_runtime", "interface.cpp"
                    )
                ) as f:
                    self.header.splice(f.read())
            else:
                # we produce a separate model header for each model in static linkage
                self.header.splice(f"""#include \"{self.model_class_name_suffix}.h\"""")
            self.header.splice("\n")

        if config.cpp.enable_kernel_profile:
            self.header.splice(
                "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
            )
            self.header.splice(
                """
                namespace torch::aot_inductor {
                thread_local KernelContext* tls_kernel_context = nullptr;
                }
                """
            )

    def _include_extra_header(self, header: str):
        # This is needed for cpp to python dtype conversion
        self.header.splice(f"#include <{header}>")

    def mark_output_type(self):
        # mark output type to unwrap tensor back to python scalar
        from ..ir import ShapeAsConstantBuffer

        output_is_tensor = {}
        for idx, x in enumerate(V.graph.graph_outputs):
            if isinstance(x, ShapeAsConstantBuffer):
                output_is_tensor[idx] = False
            else:
                output_is_tensor[idx] = True

        self.output_is_tensor = output_is_tensor

    def write_prefix(self):
        if V.graph.is_const_graph:
            # We do not write prefix for constant graph, it will be written by main module.
            return
        if config.aot_inductor.custom_ops_to_c_shims:
            # custom_ops_to_c_shims contains declaration of custom ops with C shim.
            # TODO: this could be auto-generated from a passed-in custom op schema
            custom_c_shims = list(
                chain(*config.aot_inductor.custom_ops_to_c_shims.values())
            )
            declarations = "\n".join(
                [f"extern {textwrap.dedent(shim)};" for shim in custom_c_shims]
            )
            self.prefix.splice(
                f"""
                extern "C" {{
                    {declarations}
                }}
                """
            )
        if V.graph.aot_mode:
            self.prefix.writeline("namespace torch::aot_inductor {")

    def write_input_output_info(
        self,
        info_kind: str,
        idx: int,
        name: str,
    ):
        self.prefix.writeline(f"""{info_kind}[{idx}].name = "{name}";""")

    def codegen_input_symbol_assignment(
        self,
        name: str,
        value: ir.TensorBox,
        bound_vars: OrderedSet[sympy.Symbol],
    ):
        code = self.prefix

        @functools.cache
        def sizeof(name):
            self.codegen_input_size_var_decl(code, name)
            return f"{name}_size"

        @functools.cache
        def strideof(name):
            self.codegen_input_stride_var_decl(code, name)
            return f"{name}_stride"

        def codegen_symbol(
            sym_or_exp: Union[sympy.Symbol, sympy.Expr],
            base_name: str,
            name_fn: Callable[[str], str],
            dim: int,
        ):
            if isinstance(sym_or_exp, sympy.Symbol):
                if sym_or_exp in bound_vars:
                    return
                code.writeline(f"int64_t {sym_or_exp} = {name_fn(base_name)}[{dim}];")
                bound_vars.add(sym_or_exp)
            elif isinstance(sym_or_exp, sympy.Expr):
                undefined_symbols = [
                    sym for sym in sym_or_exp.free_symbols if sym not in bound_vars
                ]
                if len(undefined_symbols) != 1:
                    # Skip if expression contains no symbols or if multiple
                    # symbols exists since we assume each base symbol is defined
                    # by other codegen_symbol calls.
                    return

                from torch.utils._sympy.solve import try_solve

                free_symbol = undefined_symbols.pop()
                base_name = name_fn(base_name)
                # Use a size symbol to solve the free symbol
                size_symbol = sympy.Symbol(f"{base_name}_{dim}", integer=True)
                code.writeline(f"int64_t {size_symbol} = {base_name}[{dim}];")
                solution = try_solve(sympy.Eq(sym_or_exp, size_symbol), free_symbol)
                if solution is not None:
                    code.writeline(f"int64_t {free_symbol} = {cexpr(solution[1])};")
                    bound_vars.add(free_symbol)
                else:
                    raise AssertionError(
                        str(sympy.Eq(sym_or_exp, size_symbol)) + " is not solvable"
                    )

        if isinstance(value, sympy.Expr):
            if not isinstance(value, sympy.Symbol) or value in bound_vars:
                return
            if value.is_integer:
                decl = "int64_t"
            elif value.is_float:
                decl = "double"
            else:
                raise AssertionError("Unexpected symbol type")
            code.writeline(f"{decl} {value} = {name};")
            bound_vars.add(value)
        elif isinstance(value, ir.TensorBox):
            for dim, size in enumerate(value.get_size()):
                codegen_symbol(size, name, sizeof, dim)
            for dim, stride in enumerate(value.get_stride()):
                codegen_symbol(stride, name, strideof, dim)
        elif isinstance(value, ir.TorchBindObject):
            # torchbind objects are loaded in proxy executor
            pass
        else:
            raise AssertionError(f"Unknown value type: {type(value)}")

    def generate_input_output_runtime_checks(self):
        """
        In debug_compile mode, we generate checks to ensure the dtype/shape/stride/device of each
        real input/output tensor match ones provided at compile time via sample
        input/output.
        """

        def gen_check(handle_kind, idx, name, tensor):
            # Wrap AtenTensorHandle with ConstantHandle for cleaner utility function access
            self.prefix.writeline(
                f"ConstantHandle {name} = ConstantHandle({handle_kind}[{idx}]);"
            )
            self.codegen_tensor_dtype_var_decl(self.prefix, name)
            expected_dtype_name = DTYPE_TO_ATEN[tensor.dtype]
            dtype_str = str(tensor.dtype).split(".")[-1]
            self.prefix.splice(
                f"""
                    int32_t {name}_expected_dtype = aoti_torch_dtype_{dtype_str}();
                    if ({name}_expected_dtype != {name}_dtype) {{
                        std::stringstream ss;
                        ss << "{handle_kind}[{idx}]: unmatched dtype, "
                           << "expected: " << {name}_expected_dtype << "({expected_dtype_name}), "
                           << "but got: " << {name}_dtype << "\\n";
                        throw std::runtime_error(ss.str());
                    }}
                """
            )
            self.codegen_input_size_var_decl(self.prefix, name)
            for dim_idx, d in enumerate(tensor.get_size()):
                if isinstance(d, (int, sympy.Integer)):
                    self.prefix.splice(
                        f"""
                            if ({d} != {name}_size[{dim_idx}]) {{
                                std::stringstream ss;
                                ss << "{handle_kind}[{idx}]: unmatched dim value at {dim_idx}, "
                                   << "expected: {d}, " << "but got: " << {name}_size[{dim_idx}]
                                   << "\\n";
                                throw std::runtime_error(ss.str());
                            }}
                        """
                    )
                else:
                    from torch.utils._sympy.value_ranges import bound_sympy

                    sym_range = bound_sympy(d, V.graph.sizevars.shape_env.var_to_range)
                    if not math.isinf(sym_range.lower):
                        self.prefix.splice(
                            f"""
                                if ({name}_size[{dim_idx}] < {sym_range.lower}) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: dim value is too small at {dim_idx}, "
                                       << "expected it to be >= {sym_range.lower}, " << "but got: "
                                       << {name}_size[{dim_idx}] << "\\n";
                                    throw std::runtime_error(ss.str());
                                }}
                            """
                        )
                    if not math.isinf(sym_range.upper):
                        # Limit upper bound to max C long long value (2^63 - 1)
                        max_long_long = ctypes.c_longlong(2**63 - 1).value
                        upper_bound = min(sym_range.upper, max_long_long)
                        self.prefix.splice(
                            f"""
                                if ({name}_size[{dim_idx}] > {upper_bound}) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: dim value is too large at {dim_idx}, "
                                       << "expected to be <= {upper_bound}, " << "but got: "
                                       << {name}_size[{dim_idx}] << "\\n";
                                    throw std::runtime_error(ss.str());
                                }}
                            """
                        )

            self.codegen_input_stride_var_decl(self.prefix, name)
            for stride_idx, s in enumerate(tensor.get_stride()):
                if not isinstance(s, (int, sympy.Integer)):
                    continue
                self.prefix.splice(
                    f"""
                        if ({s} != {name}_stride[{stride_idx}]) {{
                            std::stringstream ss;
                            ss << "{handle_kind}[{idx}]: unmatched stride value at {stride_idx}, "
                               << "expected: {s}, " << "but got: " << {name}_stride[{stride_idx}]
                               << "\\n";
                            throw std::runtime_error(ss.str());
                        }}
                    """
                )

            # check input device type
            if isinstance(tensor, ir.TensorBox):
                tensor_device = tensor.get_device()
                if tensor_device is not None:
                    expected_device_type = DEVICE_TO_INT.get(tensor_device.type)
                    if expected_device_type is not None:
                        self.codegen_input_device_type_var_decl(self.prefix, name)
                        device_type_str = str(tensor_device.type)
                        self.prefix.splice(
                            f"""
                                int32_t {name}_expected_device_type = {expected_device_type};
                                if ({name}_expected_device_type != {name}_device_type) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: unmatched device type, "
                                    << "expected: " << {name}_expected_device_type << "{expected_device_type}({device_type_str}), "
                                    << "but got: " << {name}_device_type << "\\n";
                                    throw std::runtime_error(ss.str());
                                }}
                            """
                        )

        # Create a separate function for each input check to avoid "too big to optimize" error
        for idx, (name, tensor) in enumerate(V.graph.graph_inputs.items()):
            self.prefix.splice(
                f"""
                AOTI_NOINLINE static void check_input_{idx}(
                    AtenTensorHandle* input_handles
                ) {{
                """
            )
            with self.prefix.indent():
                gen_check("input_handles", idx, name, tensor)
            self.prefix.writeline("}")

        # force noinline to avoid any potential compilation slowdown due to aggressive
        # inline done by the host compiler
        self.prefix.splice(
            """
            static bool _check_aoti_runtime_check_inputs_env() {
                const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
                const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
                return result;
            }

            AOTI_NOINLINE static void __check_inputs_outputs(
                AtenTensorHandle* input_handles,
                AtenTensorHandle* output_handles) {
                if (!_check_aoti_runtime_check_inputs_env()){
                    return;
                }
            """
        )
        with self.prefix.indent():
            for idx in range(len(V.graph.graph_inputs)):
                self.prefix.writeline(f"check_input_{idx}(input_handles);")
        self.prefix.writeline("}")

    def write_wrapper_decl(self):
        inputs_len = len(V.graph.graph_inputs.keys())
        if V.graph.aot_mode:
            self.codegen_additional_funcs()

            if V.graph.const_module:
                self.header.splice(V.graph.const_module.wrapper_code.header)

                assert V.graph.const_wrapper_code is not None
                self.prefix.splice(V.graph.const_wrapper_code)

                assert V.graph.const_kernel_code is not None
                self.kernel_declarations.splice(V.graph.const_kernel_code)

            if V.graph.is_const_graph:
                self.prefix.splice(
                    f"""
                    void {self.aoti_model_class_name}::_const_run_impl(
                        std::vector<AtenTensorHandle>& output_handles,
                        DeviceStreamType stream,
                        AOTIProxyExecutorHandle proxy_executor
                    ) {{
                    """
                )
            else:
                if not config.aot_inductor.use_runtime_constant_folding:
                    # If we do not split the constant graph, we'll just create
                    # an empty implementation when wrapping the main module.
                    self.prefix.splice(
                        f"""
                        void {self.aoti_model_class_name}::_const_run_impl(
                            std::vector<AtenTensorHandle>& output_handles,
                            DeviceStreamType stream,
                            AOTIProxyExecutorHandle proxy_executor
                        ) {{}}

                        """
                    )

                run_impl_proto = f"""
                    void {self.aoti_model_class_name}::run_impl(
                        AtenTensorHandle*
                            input_handles, // array of input AtenTensorHandle; handles
                                            // are stolen; the array itself is borrowed
                        AtenTensorHandle*
                            output_handles, // array for writing output AtenTensorHandle; handles
                                            // will be stolen by the caller; the array itself is
                                            // borrowed
                        DeviceStreamType stream,
                        AOTIProxyExecutorHandle proxy_executor
                    ) {{
                        __check_inputs_outputs(input_handles, output_handles);
                    """

                self.generate_input_output_runtime_checks()
                self.prefix.splice(run_impl_proto)
        else:
            # cpp entry function for JIT with cpp wrapper
            self.prefix.splice(
                """
                void inductor_entry_impl(
                    AtenTensorHandle*
                        input_handles, // array of input AtenTensorHandle; handles
                                        // are stolen; the array itself is borrowed
                    AtenTensorHandle*
                        output_handles  // array for writing output AtenTensorHandle; handles
                                        // will be stolen by the caller; the array itself is
                                        // borrowed)
                ) {
                """
            )
        with self.prefix.indent():
            # assign inputs and outputs in both cases so the later codegen can be simplified
            if not V.graph.is_const_graph:
                if V.graph.aot_mode:
                    num_args = len(V.graph.graph_inputs)
                else:
                    # Weights are promoted in the JIT mode
                    num_args = len(V.graph.graph_inputs) + len(V.graph.constants)
                    # release GIL to support multiple instances inference (in different threads of the same process)
                    self.prefix.splice("py::gil_scoped_release_simple release;")

                self.prefix.splice(
                    f"""
                        auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, {num_args});
                    """
                )

            if inputs_len != 0:
                for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
                    # unwrap input tensor back to scalar
                    if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                        from ..graph import may_get_constant_buffer_dtype

                        dtype = may_get_constant_buffer_dtype(
                            V.graph.graph_inputs[input_key]  # type: ignore[arg-type]
                        )
                        assert dtype is not None, (
                            "Fails to get the dtype of the sympy.Expr"
                        )
                        self.codegen_tensor_item(
                            dtype, f"inputs[{idx}]", input_key, self.prefix
                        )
                    else:
                        self.prefix.writeline(
                            f"auto {input_key} = std::move(inputs[{idx}]);"
                        )
                # debug printing for all input args to AOTI model
                debug_printer_manager = V.graph.wrapper_code.debug_printer
                debug_printer_manager.codegen_model_inputs_value_print(
                    input_args_to_print=[
                        input_key
                        for input_key in V.graph.graph_inputs
                        if input_key.startswith("arg")
                    ]
                )

            assert all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ), "Expect all constants to be Tensor"
            for idx, constants_key in enumerate(V.graph.constants.keys()):
                if V.graph.aot_mode:
                    # Weights are stored in constants_ and owned by ConstantHandle there.
                    # Don't call std::move here because it will cause constants_ to lose the ownership.
                    self.prefix.writeline(
                        f"""[[maybe_unused]] auto& {constants_key} = constants_->at({idx});"""
                    )
                else:
                    # Append constants as inputs to the graph
                    constants_idx = inputs_len + idx
                    self.prefix.writeline(
                        f"[[maybe_unused]] auto {constants_key} = std::move(inputs[{constants_idx}]);"
                    )

            self.codegen_inputs()

            if V.graph.aot_mode:
                if not V.graph.is_const_graph:
                    self.prefix.writeline("inputs.clear();")
                self.prefix.writeline(
                    "[[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());"
                )

    def codegen_tensor_dtype_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"int32_t {name}_dtype;")
        code.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype({name}, &{name}_dtype));"
        )

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"auto {name}_size = {name}.sizes();")

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"auto {name}_stride = {name}.strides();")

    def codegen_input_device_type_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"int32_t {name}_device_type;")
        code.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type({name}, &{name}_device_type));"
        )

    def codegen_additional_funcs(self):
        pass

    def codegen_model_kernels(self):
        self.prefix.writeline("namespace {")

        # Tell compiler we need to link with the non-mangled symbols
        for kernel in self.initialized_kernels.values():
            assert hasattr(kernel, "get_signature"), (
                f"{kernel} must have get_signature implemented"
            )
            signature = kernel.get_signature()
            self.prefix.writeline(f'extern "C" {signature};')

        self.prefix.writeline(
            "class AOTInductorModelKernels : public AOTInductorModelKernelsBase {"
        )
        self.prefix.writeline("  public:")
        declare_kernel = OrderedSet(self.src_to_kernel.values()) - OrderedSet(
            self.initialized_kernels.keys()
        )
        declare_kernel.update(
            entry[0] for entry in self.user_defined_kernel_cache.values()
        )
        if V.graph.const_module:
            declare_kernel.update(
                V.graph.const_module.wrapper_code.src_to_kernel.values()
            )
        for kernel in sorted(declare_kernel):
            self.prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"    {self.device_codegen.cpp_kernel_type()} {kernel}{{nullptr}};"
                )
            )
        for name, kernel in self.initialized_kernels.items():
            assert hasattr(kernel, "get_signature"), (
                f"{kernel} must have get_signature implemented"
            )
            kernel_ptr = f"(*{name})"
            signature = kernel.get_signature().replace(name, kernel_ptr)
            self.prefix.writeline(f"    {signature} = torch::aot_inductor::{name};")
        self.prefix.writeline("};")
        self.prefix.writeline("}  // namespace\n\n")

        if config.aot_inductor.embed_kernel_binary:
            self.prefix.writeline('extern "C" {')
            for name in sorted(declare_kernel):
                self.prefix.writeline(
                    f"    extern const unsigned char __{name}_start[];"
                )
                if torch.xpu.is_available():
                    self.prefix.writeline(
                        f"    extern const unsigned char __{name}_end[];"
                    )
            self.prefix.writeline("}")

    # MSVC string was longer than the limit of 16380 single-byte characters.
    # https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026
    MSVC_C2026_MAX_STRING_LENGTH = 16000

    def codegen_write_arg_with_large_length_string(
        self,
        arg_name: str,
        arg_str_val: str,
        max_truncate_length: int = MSVC_C2026_MAX_STRING_LENGTH,
    ):
        def truncate_string(s: str, length: int) -> list[str]:
            return [s[i : i + length] for i in range(0, len(s), length)]

        if len(arg_str_val) > max_truncate_length:
            truncated_strs = truncate_string(arg_str_val, max_truncate_length)
            self.prefix.writeline(f"{arg_name} =")
            for truncate_str in truncated_strs:
                self.prefix.writeline(f'R"({truncate_str})"')
            self.prefix.writeline(";")
        else:
            self.prefix.writeline(f'{arg_name} = R"({arg_str_val})";')

    def codegen_model_constructor(self):
        """
        // Generated code example
        AOTInductorModel::AOTInductorModel()
            : AOTInductorModelBase(4, 1) {
        inputs_info_[0].name = "input0";
        inputs_info_[0].dtype = "torch.float16";
        ...
        constants_info_[0].name = "L__self___weight";
        constants_info_[0].dtype = at::kFloat;
        constants_info_[0].offset = 0;
        constants_info_[0].data_size = 8192;
        constants_info_[0].shape = {64, 32};
        constants_info_[0].stride = {32, 1};
        ...
        outputs_info_[0].name = "output0";
        outputs_info_[0].dtype = "torch.float16";
        }
        """

        num_inputs = len(V.graph.graph_inputs)
        num_outputs = len(V.graph.graph_outputs)
        num_constants = len(V.graph.constants)
        include_weights = (
            "true"
            if config.aot_inductor.package_constants_in_so
            and config.aot_inductor.package_constants_on_disk_format != "binary_blob"
            else "false"
        )
        self.prefix.splice(
            f"""
            {self.aoti_model_class_name}::{self.aoti_model_class_name}(std::shared_ptr<ConstantMap> constants_map,
                                               std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                               const std::string& device_str,
                                               std::optional<std::string> cubin_dir)
                : AOTInductorModelBase({num_inputs},
                                       {num_outputs},
                                       {num_constants},
                                       device_str,
                                       std::move(cubin_dir),
                                       {include_weights}) {{
            """
        )

        with self.prefix.indent():
            for idx, (name, inp) in enumerate(V.graph.graph_inputs.items()):
                assert not isinstance(inp, sympy.Expr), (
                    f"input {name=} cannot be symbolic"
                )
                self.write_input_output_info("inputs_info_", idx, name)

            all_cuda = all(
                V.graph.get_original_value_of_constant(name).is_cuda
                for name in V.graph.constants
                if name not in V.graph.folded_constants
            )
            for idx, name in enumerate(V.graph.constants.keys()):
                tensor = V.graph.get_original_value_of_constant(name)
                assert isinstance(tensor, torch.Tensor)
                self.prefix.writeline(f"""constants_info_[{idx}].name = "{name}";""")
                self.prefix.writeline(
                    f"constants_info_[{idx}].dtype = static_cast<int32_t>({self.codegen_dtype(tensor.dtype)});"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].offset = {tensor.storage_offset()};"
                )

                # If constants to serialize contain cpu tensors, we always align data_size it to 64.
                # When loading the constants, the valid data will depends on the size
                # not the data_size so there won't be correctness issue.
                data_size = (
                    torch.ops.mkldnn._nbytes(tensor)
                    if tensor.is_mkldnn
                    else tensor.untyped_storage().nbytes()
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].data_size = {data_size if all_cuda else _align(data_size)};"
                )

                from_folded = "true" if name in V.graph.folded_constants else "false"
                self.prefix.writeline(
                    f"constants_info_[{idx}].from_folded = {from_folded};"
                )

                if name in V.graph.folded_constants:
                    constant_type_str = "FoldedConstant"
                elif name.startswith("_tensor_constant"):
                    constant_type_str = "TensorConstant"
                elif any(
                    name == normalize_name(parameter_name)
                    for parameter_name in V.graph.named_parameters
                ):
                    constant_type_str = "Parameter"
                elif any(
                    name == normalize_name(buffer_name)
                    for buffer_name in V.graph.named_buffers
                ):
                    constant_type_str = "Buffer"
                else:
                    constant_type_str = "Unknown"
                self.prefix.writeline(
                    f"constants_info_[{idx}].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::{constant_type_str});"
                )

                size_str = ", ".join([str(s) for s in tensor.size()])
                self.prefix.writeline(f"constants_info_[{idx}].shape = {{{size_str}}};")

                stride_str = ", ".join([str(s) for s in tensor.stride()])
                self.prefix.writeline(
                    f"constants_info_[{idx}].stride = {{{stride_str}}};"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].layout = static_cast<int32_t>({self.codegen_layout(tensor.layout)});"
                )

                if tensor.is_mkldnn:
                    opaque_metadata_tensor = torch.ops.mkldnn._get_mkldnn_serialized_md(
                        tensor
                    )
                    assert opaque_metadata_tensor.dim() == 1, (
                        "Expect opaque_metadata_tensor to be 1-D"
                    )

                    opaque_metadata_list = opaque_metadata_tensor.tolist()
                    opaque_metadata_str = self.codegen_shape_tuple(opaque_metadata_list)
                    self.prefix.writeline(
                        f"constants_info_[{idx}].opaque_metadata = {opaque_metadata_str};"
                    )
                if name in V.graph.dynamo_flat_name_to_original_fqn:
                    original_fqn = V.graph.dynamo_flat_name_to_original_fqn.get(
                        name, name
                    )
                elif name in V.graph.allocated_constant_name:
                    original_fqn = V.graph.allocated_constant_name[name]
                else:
                    raise AssertionError("original_fqn must be set for constant")
                self.prefix.writeline(
                    f"""constants_info_[{idx}].original_fqn = "{original_fqn}";"""
                )
            self.prefix.writeline("update_constants_map(std::move(constants_map));")
            self.prefix.writeline("update_constants_array(std::move(constants_array));")

            def escape_string(x):
                return (
                    x.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                )

            # Origin code: self.prefix.writeline(f'in_spec_ = R"({config.aot_inductor.serialized_in_spec})";')
            # Fix msvc C2026 error via codegen_write_arg_with_large_length_string
            self.codegen_write_arg_with_large_length_string(
                arg_name="in_spec_", arg_str_val=config.aot_inductor.serialized_in_spec
            )
            # Origin code: self.prefix.writeline(f'out_spec_ = R"({config.aot_inductor.serialized_out_spec})";')
            # Fix msvc C2026 error via codegen_write_arg_with_large_length_string
            self.codegen_write_arg_with_large_length_string(
                arg_name="out_spec_",
                arg_str_val=config.aot_inductor.serialized_out_spec,
            )

            for idx, output in enumerate(V.graph.graph_outputs):
                assert not isinstance(output, sympy.Expr), (
                    f"output {name=} cannot be symbolic"
                )
                name = f"output{idx}"
                self.write_input_output_info("outputs_info_", idx, name)

            self.prefix.writeline(
                "this->kernels_ = std::make_unique<AOTInductorModelKernels>();"
            )

        self.prefix.writeline("}")

    def codegen_const_run_driver(self):
        """
        // Generated code example
        std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
            DeviceStreamType stream,
            AOTIProxyExecutorHandle proxy_executor,
            bool initialization
        ) {
            std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
            std::vector<AtenTensorHandle> output_handles;
            // build up output_handles over here.
            _const_run_impl(output_handles, stream, proxy_executor);
            // build up folded_constants_map
            return folded_constants_map;
        }
        """

        self.prefix.splice(
            f"""
            std::unordered_map<std::string, AtenTensorHandle> {self.aoti_model_class_name}::const_run_impl(
                DeviceStreamType stream,
                AOTIProxyExecutorHandle proxy_executor,
                bool initialization
            ) {{
            """
        )
        if not config.aot_inductor.use_runtime_constant_folding:
            self.prefix.splice(
                """
                    if (!initialization) {
                        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                                  << "aot_inductor.use_runtime_constant_folding=False\\n";
                    }
                    return {};
                }
                """
            )
            return

        with self.prefix.indent():
            # This is a mapping to the index of constant folding graph's output
            const_index_mapping: list[Optional[tuple[int, str]]] = [None] * len(
                V.graph.const_output_index
            )
            for idx, (name, _) in enumerate(V.graph.constants.items()):
                if name in V.graph.const_output_index:
                    const_index_mapping[V.graph.const_output_index[name]] = (idx, name)  # type: ignore[call-overload]
            assert None not in const_index_mapping, (
                "Not all constant gets mapped for constant folding graph."
            )

            self.prefix.writeline(
                f"""
                std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
                folded_constants_map.reserve({len(const_index_mapping)});
                std::vector<AtenTensorHandle> output_handles({len(const_index_mapping)});
                """
            )

            self.prefix.splice(
                """
                // The below assignment of output_handles to constants is not used directly.
                // It's only used to memo the correspondence of handle and constants.
                """
            )

            for output_idx, (const_idx, _) in enumerate(const_index_mapping):  # type: ignore[misc]
                self.prefix.writeline(
                    f"output_handles[{output_idx}] = constants_->at({const_idx});"
                )

            self.prefix.writeline(
                "_const_run_impl(output_handles, stream, proxy_executor);"
            )

            for output_idx, (_, const_name) in enumerate(const_index_mapping):  # type: ignore[misc]
                self.prefix.writeline(
                    f'folded_constants_map["{const_name}"] = output_handles[{output_idx}];'
                )
            self.prefix.writeline("return folded_constants_map;")

        self.prefix.writeline("}")

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperCpu.generate", log_pt2_compile_event=True):
            self.write_wrapper_decl()
            return super().generate(is_inference)

    def finalize_prefix(self):
        prior = self.prefix
        self.prefix = aot_mode_decls = IndentedBuffer()
        if V.graph.aot_mode and not V.graph.is_const_graph:
            aot_mode_decls.writeline("namespace torch::aot_inductor {")
            self.codegen_model_kernels()
            self.codegen_model_constructor()
            self.codegen_const_run_driver()
            aot_mode_decls.writeline("} // namespace torch::aot_inductor")
            aot_mode_decls.writeline("using namespace torch::aot_inductor;")

        self.prefix = cache_decls = IndentedBuffer()
        for dtype in self.used_cached_dtypes:
            cache_decls.writeline(f"CACHE_TORCH_DTYPE({dtype});")
        for device in self.used_cached_devices:
            cache_decls.writeline(f"CACHE_TORCH_DEVICE({device});")
        for layout in self.used_cached_layouts:
            cache_decls.writeline(f"CACHE_TORCH_LAYOUT({layout});")
        for memory_format in self.used_cached_memory_formats:
            cache_decls.writeline(f"CACHE_TORCH_MEMORY_FORMAT({memory_format});")

        self.prefix.splice(aot_mode_decls)
        self.prefix.splice(prior)

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu: bool = False,
        cpp_definition: Optional[str] = None,
    ):
        if cpp_definition is not None:
            self.header.splice(cpp_definition)
            self.kernel_declarations.splice(f"\n{kernel_body}\n")
        else:
            self.header.splice(f"\n{kernel_body}\n")

    def codegen_scalar_to_tensor(self, output: str):
        name = f"scalar_to_tensor_{next(self.scalar_to_tensor_id)}"
        self.wrapper_call.writeline(
            f"RAIIAtenTensorHandle {name} = scalar_to_tensor_handle({output});"
        )
        return name

    def codegen_tensor_item(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        dtype_str = str(dtype).split(".")[-1]
        writer = indented_buffer or self

        if dtype == torch.float16 or dtype == torch.bfloat16:
            scalar_tmp = f"{scalar}_tmp"
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar_tmp};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar_tmp}));"
            )
            writer.writeline(f"float {scalar} = float({scalar_tmp});")
        else:
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )

    def generate_return(self, output_refs: list[str]):
        cst_names = V.graph.constants.keys()
        output2idx: dict[str, int] = {}

        # If any output ref represents an rvalue tensor, materialize it to an lvalue
        # RAIIAtenTensorHandle first.  This prevents situations where the code for the
        # rvalue tensor references tensor handles whose contents are modified below.
        output_refs = [
            self.create_tmp_raii_handle_var_if_needed(o, self.wrapper_call)
            for o in output_refs
        ]

        for idx, output in enumerate(output_refs):
            if output == "nullptr":
                continue

            is_constant_buffer = output in cst_names
            output_buffer = V.graph.graph_outputs[idx]
            if isinstance(output_buffer, ir.BaseView):
                output_storage = output_buffer.unwrap_view()
                assert isinstance(output_storage, (ir.BaseView, ir.MutableBox))
                if isinstance(output_storage.data, ir.ConstantBuffer):
                    is_constant_buffer = True

            if isinstance(output_buffer, ir.ShapeAsConstantBuffer):
                # Need to wrap scalar into tensor as the main function returns a vector of tensors
                output_tensor = self.codegen_scalar_to_tensor(output)
                self.wrapper_call.writeline(
                    f"output_handles[{idx}] = {output_tensor}.release();"
                )
                continue

            if is_constant_buffer:
                # See NOTE(return_constant) above.
                self.wrapper_call.writeline(
                    f"aoti_torch_clone({output}, &output_handles[{idx}]);"
                )
            else:
                if output in output2idx:
                    src_idx = output2idx[output]
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = output_handles[{src_idx}];"
                    )
                else:
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = {output}.release();"
                    )

            if output not in output2idx:
                output2idx[output] = idx

    def generate_before_suffix(self, result):
        if not V.graph.is_const_graph:
            if V.graph.aot_mode:
                result.writeline(f"}} // {self.aoti_model_class_name}::run_impl")
            else:
                result.writeline("} // inductor_entry_impl")

    def generate_end(self, result):
        """Generates the end of the code block, and any code needed to call it."""
        if V.graph.aot_mode:
            if V.graph.is_const_graph:
                result.writeline(f"}} // {self.aoti_model_class_name}::_const_run_impl")
            else:
                result.writeline("} // namespace torch::aot_inductor\n\n\n")
            return

        if config.cpp_wrapper_build_separate:

```



## High-Level Overview


This Python file contains 3 class(es) and 115 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `HasWriteLine`, `CppWrapperCpu`

**Functions defined**: `writeline`, `__init__`, `create`, `_generate_temporary_array_pointer`, `_generate_kernel_call_helper`, `write_constant`, `get_device_include_path`, `add_device_include`, `write_header`, `_include_extra_header`, `mark_output_type`, `write_prefix`, `write_input_output_info`, `codegen_input_symbol_assignment`, `sizeof`, `strideof`, `codegen_symbol`, `generate_input_output_runtime_checks`, `gen_check`, `write_wrapper_decl`

**Key imports**: annotations, ctypes, functools, math, os, sys, textwrap, chain, count, Any, Optional, Protocol, TYPE_CHECKING, Union, sympy


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `ctypes`
- `functools`
- `math`
- `os`
- `sys`
- `textwrap`
- `itertools`: chain, count
- `typing`: Any, Optional, Protocol, TYPE_CHECKING, Union
- `sympy`
- `torch`
- `torch._higher_order_ops.torchbind`
- `torch._inductor.async_compile  `
- `torch._ops`
- `torch._inductor.runtime.runtime_utils`: dynamo_timed
- `torch.fx.experimental.symbolic_shapes`: ConvertIntKey, DivideByKey, SymTypes
- `torch.utils._ordered_set`: OrderedSet
- `torch.utils._sympy.symbol`: symbol_is_type, SymT
- `..`: config, cpp_builder, ir
- `..ir`: ExternKernel
- `..utils`: _align, DeferredLineBase, LineContext, normalize_name
- `..virtualized`: V
- `.aoti_hipify_utils`: maybe_hipify_code_wrapper
- `.common`: get_device_op_overrides, IndentedBuffer, Kernel
- `.cpp_utils`: cexpr, DEVICE_TO_ATEN, DEVICE_TO_INT, DTYPE_TO_ATEN, DTYPE_TO_CPP
- `collections.abc`: Callable, Sequence
- `..graph`: GraphLowering
- `..scheduler`: BaseSchedulerNode
- `torch._inductor.codecache`: CppWrapperCodeCache


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor/codegen`):

- [`cpp_wrapper_mps.py_docs.md`](./cpp_wrapper_mps.py_docs.md)
- [`wrapper_fxir.py_docs.md`](./wrapper_fxir.py_docs.md)
- [`cpp_flex_attention_template.py_docs.md`](./cpp_flex_attention_template.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`simd_kernel_features.py_docs.md`](./simd_kernel_features.py_docs.md)
- [`block_analysis.py_docs.md`](./block_analysis.py_docs.md)
- [`cpp_wrapper_cpu_array_ref.py_docs.md`](./cpp_wrapper_cpu_array_ref.py_docs.md)
- [`cpp_bmm_template.py_docs.md`](./cpp_bmm_template.py_docs.md)
- [`python_wrapper_mtia.py_docs.md`](./python_wrapper_mtia.py_docs.md)
- [`cpp_template.py_docs.md`](./cpp_template.py_docs.md)


## Cross-References

- **File Documentation**: `cpp_wrapper_cpu.py_docs.md`
- **Keyword Index**: `cpp_wrapper_cpu.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
