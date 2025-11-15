# Documentation: cpp_wrapper_cpu.py

## File Metadata
- **Path**: `torch/_inductor/codegen/cpp_wrapper_cpu.py`
- **Size**: 129255 bytes
- **Lines**: 2974
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
            # Close the wrapper code block, then write any kernel definitions.
            result.splice("'''\n)")
            if self.kernel_declarations:
                result.splice("\nkernel_src = (\nr'''")
                result.splice(self.kernel_declarations.getvalue())
                result.splice("'''\n)")
            else:
                result.splice(
                    """
                    kernel_src = ''
                    """
                )
        else:
            # Merge main code and kernel code
            result.splice(self.kernel_declarations.getvalue())
            self.kernel_declarations.clear()
            # Close the wrapper code block
            result.splice("'''\n)")

        kernel_code = "kernel_src" if config.cpp_wrapper_build_separate else "None"
        # Cpp entry function for JIT with cpp wrapper
        result.splice(
            f"""
            inductor_entry = CppWrapperCodeCache.load_pybinding(
                argtypes=["std::vector<AtenTensorHandle>"],
                main_code=cpp_wrapper_src,
                device_type="{self.device}",
                num_outputs={len(V.graph.graph_outputs)},
                kernel_code={kernel_code},
            )
            """
        )

        wrapper_body = "input_tensors = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg, device='cpu') for arg in args]"
        if V.graph.constants:
            # Append constants to the input args for cpp wrapper.
            # Python wrapper directly gets the value inside the wrapper call
            # as a global variable passed when calling exec(code, mod.__dict__, mod.__dict__).
            # For cpp wrapper, we need to pass this python value to the inductor_entry_impl function explicitly.
            assert all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ), "Expect all constants to be Tensor"
            constants_str = f"[{', '.join(V.graph.constants.keys())}]"
            wrapper_body += f"""
                    constants_tensor = {constants_str}
                    input_tensors.extend(constants_tensor)
            """
        # Convert vector of at::Tensor to vector of AtenTensorHandle.
        # If we pass at::Tensor, the compilation will be too slow.
        wrapper_body += """
                    input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)
        """
        # Release the inputs for memory reuse.
        wrapper_body += """
                    args.clear()
                    del input_tensors
        """

        # unwrap output tensor back to python scalar
        if all(x for x in self.output_is_tensor.values()):
            # If no ShapeAsConstantBuffer in the output, directly return the output as tensors
            outputs_str = "output_tensors"
        else:
            outputs = [
                (
                    f"output_tensors[{i}]"
                    if self.output_is_tensor[i]
                    else f"output_tensors[{i}].item()"
                )
                for i in range(len(V.graph.graph_outputs))
            ]
            outputs_str = f"[{', '.join(outputs)}]"
        wrapper_body += f"""
                    output_handles = f(input_handles)
                    output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
                    return {outputs_str}
        """

        # Wrap the func to support setting result._boxed_call = True
        result.splice(
            f"""
            def _wrap_func(f):
                def g(args):
                    {wrapper_body}
                return g

            call = _wrap_func(inductor_entry)
            """
        )

    @staticmethod
    def get_c_shim_func_name(kernel: str, device: str) -> str:
        if kernel.startswith("aoti_torch_"):
            return kernel

        assert "::" in kernel, "Cpp kernel name: " + kernel + " does not contain '::'"
        kernel_tokens = kernel.split("::")
        kernel_suffix = kernel_tokens[-1]
        if kernel_suffix == "call":
            kernel_suffix = kernel_tokens[-2]

        shim_fn = f"aoti_torch_{device}_{kernel_suffix}"
        return shim_fn

    def generate_c_shim_extern_kernel_call(
        self,
        kernel: str,
        args: list[str],
        device: str,
        *,
        debug_args: Optional[list[str]] = None,
        stack_traces: Optional[OrderedSet[str]] = None,
    ) -> None:
        """debug_args kwarg allows CppWrapperCpuArrayRef to pass in wrapped arguments in
        place of args while preserving debug printer output."""
        # We can do this unconditionally, since we cache this call.
        self.add_device_include(device)

        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            debug_args if debug_args is not None else args, kernel, None, None, "extern"
        )
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        with debug_printer_manager:
            shim_fn = self.get_c_shim_func_name(kernel, device)
            shim_fn_codes = [
                f"AOTI_TORCH_ERROR_CODE_CHECK({shim_fn}({', '.join(args)}));"
            ]
            if enable_kernel_profile:
                stack_trace_str = 'R"('
                if stack_traces:
                    for stack_trace in stack_traces:
                        for line in stack_trace.split("\n"):
                            stack_trace_str += f"\n{line}"
                        stack_trace_str += "\n"
                stack_trace_str += ')"'

                shim_fn_codes = [
                    "{",
                    f"""KernelContextGuard _ctx("{shim_fn}", {stack_trace_str});""",
                    f"""RAIIAtenRecordFunctionHandle record_{shim_fn}_("{shim_fn}", nullptr);""",
                    shim_fn_codes[0],
                    "}",
                ]
            self.writelines(shim_fn_codes)

    def generate_c_shim_extern_kernel_alloc(
        self, extern_kernel: ir.ExternKernelAlloc, args: list[str]
    ) -> None:
        # registered output buffer name
        name = extern_kernel.name
        output_handle_name = f"{name}_handle"
        is_inplace = (
            isinstance(extern_kernel.op_overload, torch._ops.OpOverload)
            and torch.Tag.inplace_view in extern_kernel.op_overload.tags
        )

        if not is_inplace:
            self.writeline(f"AtenTensorHandle {output_handle_name};")
            args = [*args, f"&{output_handle_name}"]

        device = d.type if (d := extern_kernel.get_device()) else self.device

        self.generate_c_shim_extern_kernel_call(
            extern_kernel.get_kernel_name(), args, device
        )

        if extern_kernel.python_kernel_name in (
            "torch.ops._c10d_functional.all_reduce_.default",
            "torch.ops._c10d_functional.wait_tensor.default",
        ):
            # all_reduce_ is an inplace op and its returned tensor is not used anywhere.
            # wait_tensor returns its input without any modification and the returned tensor is not used anywhere.
            # In both cases, we can immediately delete the returned AtenTensorHandle to reduce its lifetime.
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object({output_handle_name}));"
            )
        elif not is_inplace:
            self.writeline(f"RAIIAtenTensorHandle {name}({output_handle_name});")

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        if getattr(extern_kernel, "outputs", None):
            # ir.ExternKernelAlloc may have outputs if it returns a tuple
            self.generate_c_shim_fallback_kernel(extern_kernel, args)
        else:
            self.generate_c_shim_extern_kernel_alloc(extern_kernel, args)

    def generate_c_shim_fallback_kernel(
        self, fallback_kernel: ir.FallbackKernel, args: list[str]
    ) -> None:
        output_args = []
        output_raii_handles = []
        output_name_base = fallback_kernel.get_name()
        for idx, output in enumerate(fallback_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
                # TODO: handle integer output (e.g., as in attention)
                name = f"{output.get_name()}"
                output_handle_name = f"{name}_handle"
                if output.indices:
                    assert output.indices[0][1] == idx, (
                        f"expected {output.indices[0][1]=} == {idx=} for {output_name_base=}"
                    )
                self.writeline(f"AtenTensorHandle {output_handle_name};")
                output_args.append(f"&{output_handle_name}")
                output_raii_handles.append(
                    f"RAIIAtenTensorHandle {name}({output_handle_name});"
                )
            elif isinstance(output, int):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"int64_t {output_name} = {output};")
                output_args.append(f"&{output_name}")
            elif isinstance(output, sympy.Expr):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"auto {output_name} = {cexpr(output)};")
                output_args.append(f"&{output_name}")
            elif output is None:
                output_args.append("nullptr")
            else:
                raise NotImplementedError(f"unsupported type of {output=}")
        args = args + output_args
        device = d.type if (d := fallback_kernel.get_device()) else self.device

        self.generate_c_shim_extern_kernel_call(
            fallback_kernel.cpp_kernel_name,  # type: ignore[arg-type]
            args,
            device,
        )
        for raii_handle in output_raii_handles:
            self.writeline(raii_handle)

    def _generate_extern_kernel_out_helper(
        self,
        kernel: str,
        out: str,
        out_view: Optional[str],
        args: list[str],
        device: str,
        stack_traces: Optional[OrderedSet[str]] = None,
    ) -> None:
        if out_view:
            out_name = f"{out}_as_strided"
            self.writeline(f"auto {out_name} = {out_view};")
            args.insert(0, out_name)
        else:
            args.insert(0, out)

        self.generate_c_shim_extern_kernel_call(
            kernel, args, device, stack_traces=stack_traces
        )

    def _get_scatter_reduce_enum(self, reduce):
        # Follow aten/src/ATen/native/ReductionType.h:get_operator_enum
        get_operator_enum = {"add": "sum", "multiply": "prod"}
        if reduce in get_operator_enum:
            reduce = get_operator_enum[reduce]

        return reduce

    def _generate_scatter_fallback(
        self,
        output,
        inputs,
        cpp_kernel_name,
        python_kernel_name,
        src_is_tensor,
        reduce,
        kwargs,
        device,
    ):
        reduce = self._get_scatter_reduce_enum(reduce)

        # call the ABI shim function instead of the ATen one
        self.add_device_include(device)
        cpp_kernel_name = self.get_c_shim_func_name(cpp_kernel_name, device)
        # TODO: consider remove "_out" and add missing inplace variants to fallback_ops.py
        cpp_kernel_name = cpp_kernel_name.replace("__", "_") + "_out"
        inputs_wrapped = [str(x) for x in inputs]
        line = f"{cpp_kernel_name}({output}, {','.join(inputs_wrapped)}"

        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += f", {','.join(kwargs)}"
        else:
            if src_is_tensor:
                if reduce:
                    line += f", {V.graph.wrapper_code.val_to_arg_str(reduce)}"
            else:
                assert reduce is None, (
                    "Expect reduce to be None for aten.scatter_ with scalar src"
                )
        line += ");"
        self.writeline(line)

    def _generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        # TODO: update aoti_torch_index_put_out in ir.py to use autogen out version
        # See the comment in codegen_reinterpret_view about why having something like
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) in a tmp array can cause the corresponding
        # tensor prematurely deallocated, thus the temporary array trick here.
        indices_str = self._generate_temporary_array_pointer(
            "AtenTensorHandle", indices
        )
        args = [
            x,
            indices_str,
            str(len(indices)),
            values,
            accumulate,
        ]
        args.insert(0, x)  # set x as the output tensor, this fallback mutates x.
        self.writeline(self.wrap_kernel_call(kernel, args))

    def add_benchmark_harness(self, output):
        if V.graph.aot_mode:
            return
        super().add_benchmark_harness(output)

    def codegen_cpp_sizevar(self, x: sympy.Expr, *, simplify: bool = True) -> str:
        return cexpr(V.graph.sizevars.simplify(x) if simplify else x)

    def codegen_sizevar(self, x: sympy.Expr) -> str:
        return self.codegen_cpp_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        # in the abi_compatible mode, outputs are returned via arguments
        return name

    def codegen_shape_tuple(self, shape: Sequence[sympy.Expr]) -> str:
        parts = [*map(self.codegen_sizevar, shape)]
        if len(parts) == 0:
            return "{}"
        if len(parts) == 1:
            return f"{{{parts[0]}, }}"
        return f"{{{', '.join(parts)}}}"

    def ensure_size_computed(self, sym: sympy.Symbol):
        if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
            if sym in self.computed_sizes:
                return
            self.computed_sizes.add(sym)
            expr = V.graph.sizevars.inv_precomputed_replacements[sym]
            self.writeline(f"int64_t {sym} = {cexpr(expr)};")

    def _generate_symbolic_call_arg_helper(
        self, arg: SymbolicCallArg, graph: GraphLowering
    ) -> None:
        if (arg.inner, graph) not in self.kernel_numel_expr:
            # declare expr once in each graph (scope)
            self.kernel_numel_expr.add((arg.inner, graph))
            self.writeline(f"int64_t {arg.inner} = {cexpr(arg.inner_expr)};")
        else:
            self.writeline(f"{arg.inner} = {cexpr(arg.inner_expr)};")

    def _codegen_dynamic_scalar(self, node):
        (data,) = (t.codegen_reference() for t in node.inputs)
        self.codegen_tensor_item(node.inputs[0].get_dtype(), data, f"{node.sym}_raw")

        if len(node.keypath) == 0:
            self.writeline(f"auto {node.sym} = {node.sym}_raw;")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            self.writeline(f"int64_t {node.sym} = {node.sym}_raw ? 1 : 0;")
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            # TODO: assert divisibility here
            self.writeline(
                f"int64_t {node.sym} = {node.sym}_raw / {node.keypath[0].divisor};"
            )
        else:
            raise AssertionError(f"unrecognized keypath {node.keypath}")

        # record in unbacked_symbol_decls so we won't generate a declaration of the symbol again
        self.unbacked_symbol_decls.add(str(node.sym))

    def codegen_dynamic_select_index(self, node, clamp):
        index_cpp_str = self.val_to_arg_str_for_prim_type(node.index, int)
        size_cpp_str = self.val_to_arg_str_for_prim_type(node.size, int)

        # codegen index
        sym = node.unbacked_offset_symbol
        index_str = (
            f"{index_cpp_str} < 0 ? {index_cpp_str} + "
            f"{self.val_to_arg_str_for_prim_type(node.size, int)}: {index_cpp_str}"
        )
        self.writeline(f"auto {sym}_index = {index_str};")
        index_str_clamped = (
            f"{sym}_index < 0 ? 0 : ({sym}_index > {size_cpp_str} ? {size_cpp_str} : {sym}_index)"
            if clamp
            else f"{sym}_index"
        )
        self.writeline(f"auto {sym}_index_clamped = {index_str_clamped};")
        self.writeline(
            f"auto {sym} = {self.val_to_arg_str_for_prim_type(node.base_offset, int)} + "
            f"{self.val_to_arg_str_for_prim_type(node.base_dim_stride, int)} * {sym}_index_clamped;"
        )
        # record in unbacked_symbol_decls so we won't generate a declaration of the symbol again
        self.unbacked_symbol_decls.add(str(sym))

    def codegen_dynamic_slice_size(self, node):
        start_cpp_str = self.val_to_arg_str_for_prim_type(node.start, int)
        end_cpp_str = self.val_to_arg_str_for_prim_type(node.end, int)
        size_cpp_str = self.val_to_arg_str_for_prim_type(node.size, int)
        step_cpp_str = self.val_to_arg_str_for_prim_type(node.step, int)
        sym = node.unbacked_size_symbol

        def codegen_clamp(index_str, start=True):
            suf = "st" if start else "en"
            index_ = f"{sym}_{suf}_index"
            self.writeline(
                f"int64_t {index_} = {index_str} < 0 ? {index_str} + {size_cpp_str} : {index_str};"
            )
            self.writeline(
                f"int64_t {sym}_{suf}_cl = {index_} < 0 ? 0 : ({index_} > {size_cpp_str} ? {size_cpp_str} : {index_});"
            )

        codegen_clamp(start_cpp_str, start=True)
        codegen_clamp(end_cpp_str, start=False)
        if node.step == 1:
            step_str = f"{sym}_en_cl - {sym}_st_cl"
        else:
            step_str = (
                f"({sym}_en_cl - {sym}_st_cl + {step_cpp_str} - 1) / {step_cpp_str}"
            )
        self.writeline(f"int64_t {sym}_with_step = {step_str};")
        self.writeline(f"int64_t {sym} = {sym}_with_step < 0 ? 0 : {sym}_with_step;")
        self.unbacked_symbol_decls.add(str(sym))

    def make_buffer_free(self, buffer):
        return (
            ""
            if isinstance(buffer.get_output_spec(), ir.MultiOutputLayout)
            or isinstance(buffer, ir.TMADescriptor)
            else f"{buffer.get_name()}.reset();"
        )

    def make_free_by_names(self, names_to_del: list[str]):
        return " ".join(f"{name}.reset();" for name in names_to_del)

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f"auto {new_name} = std::move({old_name});  // reuse"

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline(
            'RAIIAtenRecordFunctionHandle record_inductor_wrapper_call_("inductor_wrapper_call", nullptr);'
        )

    def generate_start_graph(self):
        pass

    def generate_end_graph(self):
        pass

    def generate_inf_and_nan_checker(self, nodes):
        for buf in nodes.get_names():
            # TODO: Add buf name directly into check_inf_and_nan.
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan({buf}));"
            )

    def codegen_device(self, device):
        assert device.type in DEVICE_TO_ATEN, (
            device.type + " not found in DEVICE_TO_ATEN"
        )
        device_str = DEVICE_TO_ATEN[device.type][5:].lower()  # remove "at::k"
        self.used_cached_devices.add(device_str)
        return f"cached_torch_device_type_{device_str}, {device.index if device.index else 0}"

    def codegen_dtype(self, dtype):
        dtype_str = str(dtype).split(".")[-1]
        self.used_cached_dtypes.add(dtype_str)
        return f"cached_torch_dtype_{dtype_str}"

    def codegen_layout(self, layout):
        layout_str = str(layout).split(".")[-1]
        self.used_cached_layouts.add(layout_str)
        return f"cached_torch_layout_{layout_str}"

    def codegen_memory_format(self, memory_format):
        memory_format_str = str(memory_format).split(".")[-1]
        self.used_cached_memory_formats.add(memory_format_str)
        return f"cached_torch_memory_format_{memory_format_str}"

    @functools.cache  # noqa: B019
    def codegen_int_array_var(
        self,
        int_array: str,
        writeline: Callable[..., None],
        known_statically=False,
        graph=None,  # for per-graph caching
    ):
        # Used for size/stride declaration
        #
        # Because the memory planning is done in two passes (see the implementation
        # of self.generate), the writeline behavior is different in the two passes.
        # As a result, the emitted int array declarations may appear in a later
        # position of the generated code, so the second pass codegen should not
        # reuse int array declarations generated in the first pass.
        # This is why writeline needs to explicitly passed in as a parameter.
        var = f"int_array_{next(self.int_array_id)}"
        ctype = "int64_t"
        if int_array == "{}":
            #  An array of unknown bound cannot be initialized with {}.
            if known_statically:
                if config.cpp.use_constexpr_for_int_array:
                    writeline(f"static constexpr {ctype} *{var}=nullptr;")
                else:
                    writeline(f"static const {ctype} *{var}=nullptr;")
            else:
                writeline(f"const {ctype} *{var}=nullptr;")
        else:
            if var not in self.declared_int_array_vars:
                self.declared_int_array_vars.add(var)
                if known_statically:
                    if config.cpp.use_constexpr_for_int_array:
                        writeline(f"static constexpr {ctype} {var}[] = {int_array};")
                    else:
                        writeline(f"static const {ctype} {var}[] = {int_array};")
                else:
                    writeline(f"const {ctype} {var}[] = {int_array};")
        return var

    def make_buffer_allocation(self, buffer):
        return self.make_allocation(
            buffer.get_name(),
            buffer.get_device(),
            buffer.get_dtype(),
            buffer.get_size(),
            buffer.get_stride(),
            V.graph.get_allocation_size(buffer),
            buffer.get_is_pinned(),
        )

    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False
    ):
        if allocation_shape is None:
            allocation_shape = shape

        orig_stride = stride
        device_str = self.codegen_device(device)
        dtype_code = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        allocation_size = self.codegen_shape_tuple(allocation_shape)
        stride = self.codegen_shape_tuple(orig_stride)

        size_array_var = self.codegen_int_array_var(
            size,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(shape),
            graph=self.get_codegened_graph(),
        )

        if allocation_size != size:
            allocation_size_array_var = self.codegen_int_array_var(
                allocation_size,
                self.wrapper_call.writeline,
                known_statically=self.is_statically_known_list_of_ints(
                    allocation_shape
                ),
                graph=self.get_codegened_graph(),
            )
        else:
            allocation_size_array_var = size_array_var

        stride_array_var = self.codegen_int_array_var(
            stride,
            self.wrapper_call.writeline,
            known_statically=self.is_statically_known_list_of_ints(orig_stride),
            graph=self.get_codegened_graph(),
        )
        device_type, device_id = device_str.split(",")
        device_idx = "this->device_idx_" if V.graph.aot_mode else device_id

        handle_name = f"{name}_handle"
        args = [
            str(len(shape)),
            allocation_size_array_var,
            stride_array_var,
            dtype_code,
            device_type,
            device_idx,
            f"&{handle_name}",
        ]

        self.wrapper_call.writeline(f"AtenTensorHandle {handle_name};")
        pinned_str = "_pinned" if is_pinned else ""
        self.wrapper_call.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided{pinned_str}({', '.join(args)}));"
        )

        if allocation_size != size:
            old_handle_name, handle_name = handle_name, f"{name}_handle_restrided"
            self.wrapper_call.writeline(f"AtenTensorHandle {handle_name};")
            args = [
                old_handle_name,
                size_array_var,
                stride_array_var,
                f"&{handle_name}",
            ]
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_as_strided({', '.join(args)}));"
            )
            self.wrapper_call.writeline(
                f"wrap_with_raii_handle_if_needed({old_handle_name});"
            )

        return f"RAIIAtenTensorHandle {name}({handle_name});"

    def codegen_alloc_from_pool(
        self, name, offset, dtype, shape, stride
    ) -> tuple[str, list[str]]:
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(stride)
        tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
        args = [
            name,
            cexpr(offset),  # bytes not numel
            self.codegen_dtype(dtype),
            str(len(shape)),
            self.codegen_int_array_var(
                size, self.wrapper_call.writeline, graph=self.get_codegened_graph()
            ),
            self.codegen_int_array_var(
                stride, self.wrapper_call.writeline, graph=self.get_codegened_graph()
            ),
            f"&{tmp_name}",
        ]
        # We return the lines instead of writing here because writing here is bug prune.
        # If you write aoti_torch__alloc_from_pool lines, you must write the RAIIAtenTensorHandle
        # as well, otherwise you get memory leaks
        allocations_to_write = [
            f"AtenTensorHandle {tmp_name};",
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool({', '.join(args)}));",
        ]
        return f"RAIIAtenTensorHandle({tmp_name})", allocations_to_write

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline: Callable[..., None],
        dtype=None,
    ) -> str:
        """Returns a newly-created, temporary RAII tensor handle containing the
        reinterpreted tensor data.  Callers of this function are responsible for saving
        the handle if persistent access is needed."""
        dim = str(len(size))
        original_offset = offset
        offset = self.codegen_sizevar(offset)
        call_strs = []
        final_tensor_str = None

        def create_reinterpret_call() -> str:
            args = [
                f"{data.get_name()}",
                dim,
                self.codegen_int_array_var(
                    self.codegen_shape_tuple(size),
                    writeline,
                    known_statically=self.is_statically_known_list_of_ints(size),
                    graph=self.get_codegened_graph(),
                ),
                self.codegen_int_array_var(
                    self.codegen_shape_tuple(stride),
                    writeline,
                    known_statically=self.is_statically_known_list_of_ints(stride),
                    graph=self.get_codegened_graph(),
                ),
                offset,
            ]
            return f"wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper({', '.join(args)}))"

        def create_dtypeview_call(reinterpret_call: str) -> tuple[str, list[str]]:
            tmp_AtenTensorHandle = f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}"
            tmp_call_strs = [f"AtenTensorHandle {tmp_AtenTensorHandle};"]
            device_name = data.layout.device.type
            dtypeview_function = f"aoti_torch_{device_name}_view_dtype"
            tmp_call_strs.append(
                f"AOTI_TORCH_ERROR_CODE_CHECK({dtypeview_function}"
                f"({reinterpret_call}, {self.codegen_dtype(dtype)}, &{tmp_AtenTensorHandle}));"
            )
            return f"RAIIAtenTensorHandle({tmp_AtenTensorHandle})", tmp_call_strs

        def create_new_tensor_handle() -> tuple[str, list[str]]:
            tmp_AtenTensorHandle = f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}"
            tmp_call_strs = [
                f"AtenTensorHandle {tmp_AtenTensorHandle};",
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle({data.get_name()}, &{tmp_AtenTensorHandle}));",
            ]
            return f"RAIIAtenTensorHandle({tmp_AtenTensorHandle})", tmp_call_strs

        if (
            size == data.layout.size
            and stride == data.layout.stride
            and original_offset == data.layout.offset
        ):
            # pure dtypeview
            if dtype is not None and dtype != data.dtype:
                final_tensor_str, tmp_call_strs = create_dtypeview_call(data.get_name())
            else:
                final_tensor_str, tmp_call_strs = create_new_tensor_handle()
            call_strs.extend(tmp_call_strs)
        else:
            # firstly create reinterpretview
            final_tensor_str = create_reinterpret_call()

            if dtype is not None and dtype != data.dtype:
                # wrap it with dtypeview
                final_tensor_str, tmp_call_strs = create_dtypeview_call(
                    final_tensor_str
                )
                call_strs.extend(tmp_call_strs)

        for line in call_strs:
            writeline(line)

        # NB, the return handle here represents a temporary tensor, which will be automatically
        # released.
        # Here's a sample usage in the cpp wrapper code:
        # ```
        # aoti_torch_addmm_out(
        #     buf1,
        #     arg1_1,
        #     RAIIAtenTensorHandle(tmp_tensor_handle_0),
        #     buf0,
        #     1L,
        #     1L));
        # ```
        # RAIIAtenTensorHandle(tmp_tensor_handle_0) will be released after the call to addmm_out.
        # This could be problematic when it's used in a different pattern, for example:
        # ````
        # AtenTensorHandle tensor_args[] = {RAIIAtenTensorHandle(tmp_tensor_handle_2), buf5, buf6};
        # aoti_torch_proxy_executor_call_function(..., tensor_args);
        # ````
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) will be invalid when it's used in the latter
        # kernel call.
        #
        # This is solved by updating the proxy_executor invocation to
        # ```
        # aoti_torch_proxy_executor_call_function(...,
        #     std::array<AtenTensorHandle, 3>{
        #         RAIIAtenTensorHandle(tmp_tensor_handle_2), buf5, buf6
        #     }.cbegin()
        # );
        # ```
        return final_tensor_str

    def codegen_device_copy(self, src, dst, non_blocking: Union[bool, str]):
        """This function is overridden by cpp_wrapper_cpu_array_ref, so we don't need to
        handle cases where dst is not an AtenTensorHandle."""
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_({dst}, {src}, {non_blocking}));"
        )

    def codegen_multi_output(self, node: ir.MultiOutput):
        # in the abi_compatible mode, outputs are retrieved by passing
        # output pointers, so we skip its codegen here.
        pass

    def codegen_subgraph_prefix(self, subgraph, outer_inputs, outer_outputs):
        assert len(subgraph.graph.graph_inputs) == len(outer_inputs)

        for (inner_input, inner_input_val), outer_input in zip(
            subgraph.graph.graph_inputs.items(), outer_inputs
        ):
            if not isinstance(inner_input_val, ir.TensorBox):
                continue

            # in ABI-compatible mode, we copy the underlying at::Tensor of the conditional
            # input (outer_input) into another at::Tensor to be used as a subgraph input
            # (inner_input) in the nested scope. we can't std::move here, as the codegened
            # outer input may be an expression / rvalue (e.g., reinterpret_view(x)), so we
            # can't necessarily std::move it back to the origin (x).
            self.writeline(f"AtenTensorHandle {inner_input}_handle;")
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({outer_input}, &{inner_input}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {inner_input}({inner_input}_handle);")

    def codegen_subgraph_suffix(self, subgraph, outer_inputs, outer_outputs):
        for inner_output, outer_output in zip(
            subgraph.graph.graph_outputs, outer_outputs
        ):
            src = inner_output.codegen_reference()
            if not isinstance(inner_output, ir.ShapeAsConstantBuffer):
                # in ABI-compatible mode, we need to std::move subgraph output (inner_output)
                # to the conditional output (outer_output), as RAIIAtenTensorHandle's copy
                # constructor is deleted.
                src = f"std::move({src})"
                # in case the outer_output carried a value
                # before (e.g., in the while_loop codegen)
                self.writeline(f"{outer_output}.reset();")
            self.writeline(f"{outer_output} = {src};")

    def codegen_invoke_subgraph(self, invoke_subgraph):
        raise NotImplementedError(
            "codegen invoke_subgraph is not implemented for cpp wrapper"
        )

    def codegen_conditional(self, conditional):
        outer_inputs = [f"{buf.codegen_reference()}" for buf in conditional.operands]
        outer_outputs = []
        for out in conditional.outputs:
            # in ABI-compatible mode, ir.MultiOutput is not codegened,
            # hence pre-declare output variables directly and separately
            self.writeline(f"RAIIAtenTensorHandle {out.get_name()};")
            outer_outputs.append(out.get_name())

        if not isinstance(conditional.predicate, ir.ShapeAsConstantBuffer):
            # in ABI-compatible mode, we need to use the ABI shim function
            # to extract a C++ bool from the underlying scalar bool Tensor
            predicate = f"{conditional.predicate.get_name()}_scalar"
            if predicate not in self.used_cond_predicate:
                self.codegen_tensor_item(
                    torch.bool,
                    conditional.predicate.codegen_reference(),
                    predicate,
                )
                self.used_cond_predicate.add(predicate)
        else:
            # the predicate is not a Tensor: SymBool or Python bool
            predicate = conditional.predicate.codegen_reference()

        self.writeline(f"if ({predicate}) {{")
        self.writeline(EnterSubgraphLine(self, conditional.true_subgraph.graph))
        self.codegen_subgraph(conditional.true_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))
        self.writeline("} else {")
        self.writeline(EnterSubgraphLine(self, conditional.false_subgraph.graph))
        self.codegen_subgraph(conditional.false_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))
        self.writeline("}")

    def codegen_subgraph(self, subgraph, outer_inputs, outer_outputs):
        # TODO (desertfire) - This function is the old way of supporting
        # subgraph codegen by inlining subgraphs in the output code. For python
        # wrapper, we have moved to lifting subgraphs as functions, supported by
        # PythonWrapperCode `codegen_subgraph` function. We should perhaps
        # support lifting of subgraphs as functions for cpp wrapper as well.
        try:
            self.push_codegened_graph(subgraph.graph)
            self.writeline(f"// subgraph: {subgraph.name}")
            self.codegen_subgraph_prefix(subgraph, outer_inputs, outer_outputs)
            parent_graph = V.graph
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.codegen_subgraph(
                    parent_graph=parent_graph,
                )
            self.codegen_subgraph_suffix(subgraph, outer_inputs, outer_outputs)
        finally:
            self.pop_codegened_graph()

    def codegen_while_loop(self, while_loop, stack_output=False):
        if stack_output:
            raise NotImplementedError("NYI cpp wrapper for while_loop_stack_output")
        is_bool_pred = isinstance(
            while_loop.cond_subgraph.graph.graph_outputs[0], ir.ShapeAsConstantBuffer
        )
        name = while_loop.get_name()
        outer_carried_inputs = [
            buf.codegen_reference() for buf in while_loop.carried_inputs
        ]
        outer_additional_inputs = [
            buf.codegen_reference() for buf in while_loop.additional_inputs
        ]
        cond_result_name = f"{name}_cond_result"
        if is_bool_pred:
            self.writeline(f"bool {cond_result_name};")
        else:
            self.writeline(f"RAIIAtenTensorHandle {cond_result_name};")

        cond_outer_inputs = []
        for inp, out in zip(outer_carried_inputs, while_loop.outputs):
            # in ABI-compatible mode, the carried inputs are codegened
            # as buffers outside the while loop and set to the initial
            # values. at the end of each while_loop iteration, they
            # will be assigned the carried values.
            out_name = out.get_name()
            self.writeline(f"AtenTensorHandle {out_name}_handle;")
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({inp}, &{out_name}_handle));"
            )
            self.writeline(f"RAIIAtenTensorHandle {out_name}({out_name}_handle);")
            cond_outer_inputs.append(out_name)

        # additional inputs will be assigned within the while_loop
        # iteration directly from the corresponding outer graph buffers
        cond_outer_inputs.extend(outer_additional_inputs)

        cond_outer_outputs = [cond_result_name]
        body_outer_inputs = list(cond_outer_inputs)
        body_outer_outputs = body_outer_inputs[: len(outer_carried_inputs)]

        self.writeline("while (1) {")
        self.writeline(EnterSubgraphLine(self, while_loop.cond_subgraph.graph))
        self.codegen_subgraph(
            while_loop.cond_subgraph, cond_outer_inputs, cond_outer_outputs
        )

        if is_bool_pred:
            cond_result = f"{cond_result_name}"
        else:
            cond_result = f"{cond_result_name}_scalar"
            self.codegen_tensor_item(torch.bool, cond_result_name, cond_result)
        self.writeline(f"if (!{cond_result}) break;")

        self.writeline(ExitSubgraphLine(self))
        self.writeline(EnterSubgraphLine(self, while_loop.body_subgraph.graph))
        self.codegen_subgraph(
            while_loop.body_subgraph, body_outer_inputs, body_outer_outputs
        )
        self.writeline(ExitSubgraphLine(self))
        self.writeline("}")

    def generate_extern_kernel_args_decl_if_needed(
        self,
        op_overload: Union[torch._ops.OpOverload, torch._ops.HigherOrderOperator],
        raw_args: Sequence[Any],
        output_args: _OUTPUT_ARGS_TYPE,
        raw_outputs: Sequence[ir.Buffer],
    ):
        """
        Generates declarations for external kernel arguments if needed, based on the provided
        operator and its arguments. It processes both input and output arguments, categorizing
        them into tensor and integer arguments for further code generation.
        """
        schema = None
        if isinstance(op_overload, torch._higher_order_ops.torchbind.CallTorchBind):
            obj = raw_args[0]
            method = raw_args[1]
            schema = op_overload.schema(obj, method)
        else:
            assert isinstance(op_overload, torch._ops.OpOverload), type(op_overload)
            schema = op_overload._schema
        assert schema is not None
        arg_types = [x.real_type for x in schema.arguments]
        return_types = [x.type for x in schema.returns]

        new_tensor_args = []
        new_int_args = []

        def fill_args(arg, arg_type):
            static_arg_types = (
                torch.FloatType,
                torch.BoolType,
                torch.StringType,
                torch.Type,
                torch.DeviceObjType,
            )
            inductor_tensor_buffers = (
                ir.Buffer,
                ir.ReinterpretView,
            )

            if isinstance(arg_type, torch.TensorType):
                assert isinstance(arg, inductor_tensor_buffers), f"got {type(arg)}"
                new_tensor_args.append(f"{arg.codegen_reference()}")
            elif isinstance(arg_type, torch.IntType):
                # int
                new_int_args.append(str(arg))
            elif isinstance(arg_type, torch.SymIntType):
                # SymInt
                expr = arg.node.expr if isinstance(arg, torch.SymInt) else arg
                new_int_args.append(cexpr(expr))
            elif isinstance(arg_type, torch.NumberType):
                # Scalar of type int
                assert isinstance(arg, (int, float, bool))
                # Only treat int Scalar as dynamic
                if isinstance(arg, int):
                    new_int_args.append(str(arg))
            elif isinstance(arg, ir.TorchBindObject):
                # torchbind objects are loaded in proxy executor
                pass
            elif isinstance(arg_type, torch.ListType):
                assert isinstance(arg, (list, tuple))

                # List[Tensor]
                if isinstance(arg_type.getElementType(), torch.TensorType):
                    new_tensor_args.extend([f"{a.codegen_reference()}" for a in arg])
                # List[Optional[Tensor]]
                elif isinstance(
                    arg_type.getElementType(), torch.OptionalType
                ) and isinstance(
                    arg_type.getElementType().getElementType(), torch.TensorType
                ):
                    new_tensor_args.extend(
                        [f"{a.codegen_reference()}" for a in arg if a is not None]
                    )
                # List[int]
                elif isinstance(arg_type.getElementType(), torch.IntType):
                    new_int_args.extend([str(a) for a in arg])
                # List[SymInt]
                elif isinstance(arg_type.getElementType(), torch.SymIntType):
                    expressions = [
                        a.node.expr if isinstance(a, torch.SymInt) else a for a in arg
                    ]
                    new_int_args.extend([cexpr(expr) for expr in expressions])
                # List[Scalar]
                elif isinstance(arg_type.getElementType(), torch.NumberType):
                    # Only treat int Scalar as dynamic
                    is_int_type = [isinstance(a, int) for a in arg]
                    if any(is_int_type):
                        assert all(is_int_type), (
                            "AOTInductor only supports int scalars of the same type"
                        )
                        new_int_args.extend([str(a) for a in arg])
                else:
                    assert isinstance(
                        arg_type.getElementType(),
                        static_arg_types,  # type: ignore[arg-type]
                    ), (
                        f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"
                    )
            else:
                assert isinstance(
                    arg_type,
                    static_arg_types,  # type: ignore[arg-type]
                ), (
                    f"Fall through arguments must be one of static_arg_types, got {type(arg_type)}"
                )

        for arg, arg_type in zip(raw_args, arg_types):
            if arg is not None:
                if isinstance(arg_type, torch.OptionalType):
                    fill_args(arg, arg_type.getElementType())
                else:
                    fill_args(arg, arg_type)

        def fill_output_arg(
            arg: str, return_type: torch.JitType, is_mutated_output: bool
        ) -> None:
            if isinstance(return_type, torch.TensorType):
                if not is_mutated_output:
                    self.writeline(f"AtenTensorHandle {arg}_handle;  // output buffer")
                    self.writeline(
                        f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{arg}_handle));"
                    )
                    self.writeline(f"RAIIAtenTensorHandle {arg}({arg}_handle);")
                new_tensor_args.append(f"{arg}")
            elif isinstance(return_type, torch.SymIntType):
                raise NotImplementedError("NYI support for return type: SymInt")
            elif isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.SymIntType
            ):
                raise NotImplementedError("NYI support for return type: List[SymInt]")
            else:
                raise AssertionError(f"Unsupported return type found: {return_type}")

        # TODO: Only support None and tensor(s) returns for now, SymInt is not implemented yet
        for return_type in return_types:
            if isinstance(
                return_type, (torch.TensorType, torch.NoneType, torch.IntType)
            ):
                pass
            elif isinstance(return_type, torch.OptionalType):
                assert isinstance(return_type.getElementType(), torch.TensorType)
            elif isinstance(return_type, torch.ListType):
                assert isinstance(return_type.getElementType(), torch.TensorType)
            else:
                raise NotImplementedError(
                    f"return type {return_type} is not yet supported."
                )

        for output_arg, raw_output_arg in zip(output_args, raw_outputs):  # type: ignore[arg-type]
            # None output is supported, but Optional return types are not yet supported
            if output_arg is None:
                continue
            elif isinstance(raw_output_arg, int):
                new_int_args.append(str(raw_output_arg))
            elif isinstance(output_arg, list):
                for out in output_arg:
                    assert out is not None, out
                    fill_output_arg(
                        out,
                        torch.TensorType.get(),
                        isinstance(raw_output_arg, ir.MutationOutput),
                    )
            else:
                fill_output_arg(
                    output_arg,
                    torch.TensorType.get(),
                    isinstance(raw_output_arg, ir.MutationOutput),
                )

        return new_tensor_args, new_int_args

    @staticmethod
    def _compatible_with_stableivalue(op: torch._ops.OpOverload) -> bool:
        """Returns true if op_overload._schema only utilizes types supported by the AOT
        C-shim *internal* function to_ivalue.  to_ivalue is an implementation detail, so
        these types are not guaranteed to be supported long-term.  When generating code
        for cpp_wrapper mode, we don't have to be forward-compatible, so changing this
        function's implementation in future is fine."""
        supported_types = (
            torch.BoolType,
            torch.DeviceObjType,
            torch.FloatType,
            # ScalarTypeType, LayoutType, and MemoryFormatType are seen as IntType
            # when queried via torch.JitType.type.
            torch.IntType,
            torch.TensorType,
        )

        def type_supported(t: torch.JitType) -> bool:
            if isinstance(t, torch.OptionalType):
                return type_supported(t.getElementType())
            return isinstance(t, supported_types)

        return all(
            type_supported(a.type)
            for a in chain(op._schema.arguments, op._schema.returns)
        )

    def generate_fallback_kernel_with_runtime_lookup(
        self,
        buf_name: str,
        python_kernel_name: str,
        get_args: Callable[[], Sequence[str]],
        op_overload: Union[torch._ops.OpOverload, torch._ops.HigherOrderOperator],
        raw_args: Sequence[Any],
        outputs: Sequence[ir.Buffer],
    ) -> None:
        """Generate a call to a kernel not contained in the C-shim.  This results in
        different code paths for AOT Inductor vs cpp_wrapper Inductor mode."""

        def extract_output_name(
            out: Optional[Union[ir.Buffer, Sequence[ir.Buffer]]],
        ) -> Union[Optional[str], _OUTPUT_ARGS_TYPE]:
            if out is None:
                return None
            if isinstance(out, (ir.MultiOutput, ir._CollectiveKernel)):
                return out.get_name()
            if isinstance(out, ir.MutationOutput):
                mutated_buf_names = out.get_mutation_names()
                assert (
                    isinstance(mutated_buf_names, list) and len(mutated_buf_names) == 1
                ), "Expect only one mutated buffer in MutationOutput"
                return mutated_buf_names[0]
            if isinstance(out, (list, tuple)):
                return [extract_output_name(o) for o in out]  # type: ignore[misc]
            if isinstance(out, int):
                return str(out)
            raise AssertionError(f"Unexpected output: {type(out)}")

        if isinstance(op_overload, torch._ops.HigherOrderOperator):
            assert isinstance(
                op_overload, torch._higher_order_ops.torchbind.CallTorchBind
            ), type(op_overload)
            assert len(raw_args) > 1
            obj = raw_args[0]
            method = raw_args[1]
            return_schema = op_overload.schema(obj, method).returns
        else:
          

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 3 class(es): HasWriteLine, CppWrapperCpu, AOTInductorModelKernels

### Functions
This file defines 115 function(s): writeline, __init__, create, _generate_temporary_array_pointer, _generate_kernel_call_helper, write_constant, get_device_include_path, add_device_include, write_header, _include_extra_header, mark_output_type, write_prefix, write_input_output_info, codegen_input_symbol_assignment, sizeof, strideof, codegen_symbol, generate_input_output_runtime_checks, gen_check, write_wrapper_decl, codegen_tensor_dtype_var_decl, codegen_input_size_var_decl, codegen_input_stride_var_decl, codegen_input_device_type_var_decl, codegen_additional_funcs, codegen_model_kernels, codegen_write_arg_with_large_length_string, truncate_string, codegen_model_constructor, escape_string


## Key Components

The file contains 9009 words across 2974 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 129255 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
