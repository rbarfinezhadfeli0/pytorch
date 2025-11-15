# Documentation: `docs/torch/csrc/jit/runtime/static/ops.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/static/ops.cpp_docs.md`
- **Size**: 53,421 bytes (52.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/static/ops.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/static/ops.cpp`
- **Size**: 102,147 bytes (99.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorConversions.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/processed_node_wrapper.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <iterator>

#include <ATen/CompositeExplicitAutogradFunctions.h>

// clang-format off
C10_DEFINE_bool(
    static_runtime_enable_fast_math,
    true,
    "If on, static runtime may use use optimizations that cause accuracy loss "
    "vs the jit interpreter")

namespace at::native {
static void repeat_out(
    at::Tensor& result,
    const Tensor& self,
    IntArrayRef repeats) {
  TORCH_CHECK(
      repeats.size() >= static_cast<size_t>(self.dim()),
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for (const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  // return an empty tensor if one of the repeat dimensions is zero
  at::native::resize_(result, target_size, std::nullopt);
  if (zero_tensor) {
    return;
  }

  Tensor xtensor = at::compositeexplicitautograd::expand(self, padded_size);
  Tensor urtensor = at::native::alias(result);
  for (const auto i : c10::irange(xtensor.dim())) {
    // can't unfold with step 0, so make sure step is at least 1
    // (it doesn't matter what it is in that case, because the size is 0).
    urtensor = urtensor.unfold(
        i, xtensor.size(i), std::max<int64_t>(xtensor.size(i), 1));
  }

  at::native::copy_(urtensor, xtensor.expand_as(urtensor));
}

// copy version of view ops
at::Tensor& reshape_copy_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::DimVector& proposed_shape,
    bool infer_size) {
  const auto& shape = infer_size
      ? at::infer_size_dv(proposed_shape, self.numel())
      : proposed_shape;
  at::native::resize_(out, shape, std::nullopt);

  auto self_contig = self.expect_contiguous();

  size_t nbytes = self.nbytes();
  if (nbytes == 0) {
    return out;
  }

  const void* self_data = self_contig->const_data_ptr();
  void* out_data = out.mutable_data_ptr();
  memcpy(out_data, self_data, nbytes);

  return out;
}

static at::Tensor& flatten_copy_out(
    at::Tensor& out,
    const at::Tensor& self,
    int64_t start_dim,
    int64_t end_dim) {
  start_dim =
      start_dim < 0 ? c10::maybe_wrap_dim(start_dim, self.dim()) : start_dim;
  end_dim = end_dim < 0 ? c10::maybe_wrap_dim(end_dim, self.dim()) : end_dim;
  TORCH_CHECK(
      start_dim <= end_dim,
      "flatten() has invalid args: start_dim cannot come after end_dim");

  if (self.dim() == 0) {
    return reshape_copy_out(out, self, at::DimVector{1}, false);
  }

  if (start_dim == end_dim) {
    auto shape = at::DimVector{self.sizes()};
    return reshape_copy_out(out, self, shape, false);
  }

  // We don't want to infer_size on the entire shape, because that can give us
  // an extra degree of freedom we don't want; for example, consider shape [0,
  // 1, 3, 0], with start_dim=1, end_dim=2. It's clear we want result shape
  // [0, 3, 0] but passing [0, -1, 0] to infer_size means the -1 can take on
  // any value and satisfy the constraints.
  auto iter = self.sizes().data();
  auto slice_numel = std::accumulate(
      iter + start_dim,
      iter + end_dim + 1,
      static_cast<int64_t>(1),
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multiplies<int64_t>());

  at::DimVector shape;
  shape.reserve(self.dim() - end_dim + start_dim);
  for (const auto i : c10::irange(start_dim)) {
    shape.push_back(self.sizes()[i]);
  }
  shape.push_back(slice_numel);
  for (int64_t i = end_dim + 1; i < self.dim(); i++) {
    shape.push_back(self.sizes()[i]);
  }
  return reshape_copy_out(out, self, shape, false);
}

namespace {

// This is annoying and sily, but it's solving a real problem: the
// _MSC_VER version causes an ICE on our old clang5 builds. The
// non-_MSC_VER version is a syntax error according to MSVC. Use the
// appropriate version depending on if we're MSVC or not.

#define TO_COPY_OUT_FAST_PATH_LOGIC(out, self, self_t)             \
  do {                                                             \
    const auto N = self.numel();                                   \
    const auto self_data = self.const_data_ptr<self_t>();          \
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(                        \
        kHalf,                                                     \
        kBFloat16,                                                 \
        kBool,                                                     \
        out.scalar_type(),                                         \
        "to_copy_out_inner_loop",                                  \
        [&]() {                                                    \
          const auto out_data = out.mutable_data_ptr<scalar_t>();  \
          for (const auto idx : c10::irange(N)) {                  \
            /* NOLINTNEXTLINE(bugprone-signed-char-misuse) */      \
            out_data[idx] = static_cast<scalar_t>(self_data[idx]); \
          }                                                        \
        });                                                        \
  } while (0)

#ifdef _MSC_VER
template <typename T>
void to_copy_out_fast_path(Tensor& out, const Tensor& self) {
  TO_COPY_OUT_FAST_PATH_LOGIC(out, self, T);
}

#define TO_COPY_OUT_FAST_PATH_BODY(out, self) \
  to_copy_out_fast_path<scalar_t>(out, self)
#else
#define TO_COPY_OUT_FAST_PATH_BODY(out, self) \
  using self_t = scalar_t;                    \
  TO_COPY_OUT_FAST_PATH_LOGIC(out, self, self_t)
#endif
} // namespace

at::Tensor& to_copy_out(
    Tensor& out,
    const Tensor& self,
    bool non_blocking,
    bool copy_strides,
    std::optional<MemoryFormat> memory_format) {
  if (copy_strides) {
    at::native::resize_impl_cpu_(
        out.unsafeGetTensorImpl(), self.sizes(), self.strides());
  } else {
    at::native::resize_(out, self.sizes(), std::nullopt);
  }
  auto is_unsupported_dtype = [](ScalarType t) {
#define TORCH_OPS_UNSUPPORTED_TYPE(_, type) \
  case k##type:                             \
    return true;
    switch (t) {
      AT_FORALL_QINT_TYPES(TORCH_OPS_UNSUPPORTED_TYPE)
      AT_FORALL_COMPLEX_TYPES(TORCH_OPS_UNSUPPORTED_TYPE)
      default:
        return false;
    }
#undef TORCH_OPS_UNSUPPORTED_TYPE
  };
  // Fast path: can we just copy the data ourselves? Avoids creating a
  // TensorIterator in at::native::copy_, which is relatively
  // expensive.
  if (self.is_contiguous() && !non_blocking &&
      // Did the user request us to make a copy that isn't contiguous?
      (memory_format == std::nullopt ||
       memory_format == c10::MemoryFormat::Preserve ||
       memory_format == c10::MemoryFormat::Contiguous) &&
      // CopyKernel.cpp handles this case specially, so let's not mess
      // with it.
      !self.is_neg() && !is_unsupported_dtype(self.dtype().toScalarType()) &&
      !is_unsupported_dtype(out.dtype().toScalarType()) &&
      !(
          // FBGEMM optimization might kick in, don't interfere with
          // that.
          (self.dtype() == kFloat && out.dtype() == kHalf) ||
          (self.dtype() == kHalf && out.dtype() == kFloat))) {
    AT_DISPATCH_ALL_TYPES_AND3(
        kHalf, kBFloat16, kBool, self.scalar_type(), "to_copy_out", [&]() {
          TO_COPY_OUT_FAST_PATH_BODY(out, self);
        });
    return out;
  }
  at::native::copy_(out, self, non_blocking);
  return out;
}

static Tensor& linear_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt) {
  TORCH_CHECK(!input.is_mkldnn());

  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(std::in_place);

  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::cpu::addmm_out(output, *bias, input, weight.t());
  }
  at::native::matmul_out(input, weight.t(), output);
  if (bias->defined()) {
    at::cpu::add_(output, *bias);
  }
  return output;
}

static Tensor& c2_argmin_out(
    Tensor& output,
    const Tensor& input,
    const int64_t dim,
    const bool keepdim) {
  const auto ndim = input.dim();
  int64_t dim_ = maybe_wrap_dim(dim, ndim);
  TORCH_CHECK(dim_ >= 0 && dim_ < ndim);

  const auto in_dims = input.sizes();

  c10::SmallVector<int64_t, 5> out_dims;
  out_dims.reserve(ndim);
  int prev_size = 1;
  int next_size = 1;
  for (int i = 0; i < dim_; ++i) {
    out_dims.push_back(in_dims[i]);
    prev_size *= in_dims[i];
  }
  if (keepdim) {
    out_dims.push_back(1);
  }
  for (auto i = dim_ + 1; i < ndim; ++i) {
    out_dims.push_back(in_dims[i]);
    next_size *= in_dims[i];
  }
  at::native::resize_(output, out_dims, std::nullopt);

  const auto n = in_dims[dim_];

  if (next_size == 1) {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
          const auto in_ptr = input.const_data_ptr<scalar_t>();
          const auto out_ptr = output.mutable_data_ptr<int64_t>();
          // input is a [prev_size, n] tensor.
          // output is a [prev_size,] tensor.
          // Thus, access is contiguous/coalesced.
          for (int i = 0; i < prev_size; ++i) {
            auto v = std::min_element(
                in_ptr + i * n,
                in_ptr + (i + 1) * n,
                [](scalar_t a, scalar_t b) {
                  // if a is nan, then a is *less* than b with LessOrNan
                  // semantics
                  if (at::_isnan(a)) {
                    return true;
                  }
                  // if a is not nan and b is nan, then a is not less than b
                  // with LessOrNan semantics otherwise, act normally. If `b`
                  // is NaN then a < b will always return false, so this is
                  // equivalent to the first snippet.
                  return a < b;
                });
            out_ptr[i] = std::distance(in_ptr + i * n, v);
          }
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
          const auto less_or_nan = native::detail::LessOrNan<scalar_t>{};

          const auto in_ptr = input.const_data_ptr<scalar_t>();
          const auto out_ptr = output.mutable_data_ptr<int64_t>();

          std::memset(out_ptr, 0, prev_size * next_size * sizeof(int64_t));

          for (int i = 0; i < prev_size; ++i) {
            const scalar_t* cur_in_ptr = in_ptr + i * n * next_size + next_size;
            for (int k = 1; k < n; ++k) {
              for (int j = 0; j < next_size; ++j) {
                int64_t* cur_out_ptr = out_ptr + i * next_size + j;
                if (less_or_nan(
                        *cur_in_ptr,
                        in_ptr
                            [i * n * next_size + *cur_out_ptr * next_size + j],
                        *cur_out_ptr,
                        k)) {
                  *cur_out_ptr = k;
                }
                ++cur_in_ptr;
              }
            }
          }
        });
  }
  return output;
}

static at::Tensor& dequantize_copy_out(Tensor& out, const Tensor& self) {
  if (C10_UNLIKELY(!self.is_quantized())) {
    // fallback to dequantize_cpu equivalent case: make sure out is at::kFloat
    DCHECK(out.scalar_type() == kFloat);
    return at::native::to_copy_out(out, self, false, false, std::nullopt);
  }
  return get_qtensorimpl(self)->quantizer()->dequantize_out(out, self);
}
} // namespace at::native

namespace torch::jit {

C10_DEFINE_REGISTRY(SROperatorRegistry, SROperatorFunctor)

bool opIsRegistered(const c10::Symbol& op_name) {
  const std::string name(op_name.toQualString());
  return SROperatorRegistry()->Has(name);
}

static bool disableUnsafeMathOp(const char* op_name) {
  if (FLAGS_static_runtime_enable_fast_math) {
    return false;
  }
  // This list contains ops that use caffe2 math library or use NNC that does
  // not guarantee bit exactness vs the jit interpreter. Note aten::relu is not
  // included even though it uses NNC because the results of relu should always
  // match.
  static const c10::FastSet<std::string> fast_ops{
      "aten::add", "aten::tanh", "aten::sigmoid", "aten::logit"};
  return fast_ops.count(op_name) > 0;
}

SROperator getOutOfPlaceOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SROperatorRegistry()->Has(op_name) && !disableUnsafeMathOp(op_name)) {
    return SROperatorRegistry()->Create(op_name)->Generate(n);
  }

  return nullptr;
}

// Returns true if the node represents an op with variadic arguments.
bool hasVarArgs(Node* n) {
  if (n->kind() == prim::VarConcat || n->kind() == prim::VarStack) {
    return true;
  }
  return false;
}

bool canReuseInputsOutputs(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  auto it = node_has_out_variant.find(n);
  if (it != node_has_out_variant.end()) {
    return it->second;
  }
  return getOutOfPlaceOperation(n) != nullptr;
}

// returns true if the producers of the inputs
// to this operations are out of place.
// This means the IValues will not change run to run
static bool inputsCanRunOutOfPlace(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  for (auto* input : n->inputs()) {
    if (!canReuseInputsOutputs(input->node(), node_has_out_variant)) {
      return false;
    }
  }
  return true;
}

bool isOptimizableContainerType(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  const auto& type = n->output()->type();
  bool is_supported_type = false;
  if (type->kind() == TypeKind::ListType) {
    const auto& list_type = type->expectRef<ListType>();
    is_supported_type =
        list_type.getElementType()->kind() == TypeKind::TensorType;
  } else if (type->kind() == TypeKind::TupleType) {
    const auto& tuple_type = type->expectRef<TupleType>();
    auto types = tuple_type.containedTypes();
    const auto& iter =
        std::find_if(types.begin(), types.end(), [](const TypePtr& elem) {
          return elem->kind() == TypeKind::TensorType;
        });
    is_supported_type = iter != types.end();
  }
  return is_supported_type && inputsCanRunOutOfPlace(n, node_has_out_variant);
}

static void listConstructSlowPath(
    const ListType& list_type,
    const size_t size,
    ProcessedNode* p_node) {
  c10::List<IValue> vals(list_type.getElementType());
  vals.reserve(size);
  for (const auto i : c10::irange(size)) {
    vals.push_back(p_node->Input(i));
  }
  p_node->Output(0) = vals;
}

bool sr_schema_check_kind(torch::jit::Node* node, c10::Symbol node_kind) {
  auto is_match = node->kind() == node_kind;
  if (!is_match) {
    torch::jit::LogAndDumpSchema(node);
  }
  return is_match;
}

REGISTER_OPERATOR_FUNCTOR(
    prim::ListConstruct,
    prim_ListConstruct,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::ListConstruct)) {
        return nullptr;
      }
      const bool can_optimize =
          isOptimizableContainerType(n, c10::FastMap<Node*, bool>());
      const auto& type = n->output()->type()->expectRef<ListType>();
      const size_t size = n->inputs().size();
      if (!can_optimize) {
        return [&type, size](ProcessedNode* p_node) {
          DCHECK(p_node->num_inputs() == size);
          listConstructSlowPath(type, size, p_node);
        };
      }
      return [&type, size](ProcessedNode* p_node) {
        DCHECK(p_node->num_inputs() == size);
        const auto& out_l = p_node->Output(0);
        if (!out_l.isNone()) {
          return;
        }
        listConstructSlowPath(type, size, p_node);
      };
    })

static void tupleConstructSlowPath(const size_t size, ProcessedNode* p_node) {
  // prepare inputs
  switch (size) {
    case 1:
      p_node->Output(0) = c10::ivalue::Tuple::create(p_node->Input(0));
      break;
    case 2:
      p_node->Output(0) =
          c10::ivalue::Tuple::create(p_node->Input(0), p_node->Input(1));
      break;
    case 3:
      p_node->Output(0) = c10::ivalue::Tuple::create(
          p_node->Input(0), p_node->Input(1), p_node->Input(2));
      break;
    default: {
      std::vector<IValue> vals;
      vals.reserve(size);
      for (const auto i : c10::irange(size)) {
        vals.push_back(p_node->Input(i));
      }
      p_node->Output(0) = c10::ivalue::Tuple::create(std::move(vals));
      break;
    }
  }
}

REGISTER_OPERATOR_FUNCTOR(
    prim::TupleConstruct,
    prim_TupleConstruct,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TupleConstruct)) {
        return nullptr;
      }
      const bool can_optimize =
          isOptimizableContainerType(n, c10::FastMap<Node*, bool>());
      const size_t size = n->inputs().size();
      if (!can_optimize) {
        return [size](ProcessedNode* p_node) {
          DCHECK(p_node->num_inputs() == size);
          tupleConstructSlowPath(size, p_node);
        };
      }
      return [size](ProcessedNode* p_node) {
        DCHECK(p_node->num_inputs() == size);
        const auto& out_l = p_node->Output(0);
        if (!out_l.isNone()) {
          return;
        }
        tupleConstructSlowPath(size, p_node);
      };
    })

REGISTER_OPERATOR_FUNCTOR(aten::abs, aten_abs, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::abs(Tensor self) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::abs(in0_t);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::abs_out(in0_t, out_t);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::mul, aten_mul, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }

  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::mul(in0_t, in1_t);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::mul_out(out_t, in0_t, in1_t);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::addmm, aten_addmm, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    const auto& in2_t = p_node->Input(2).toTensor();
    const auto in3_s = p_node->Input(3).toScalar();
    const auto in4_s = p_node->Input(4).toScalar();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::addmm(in0_t, in1_t, in2_t, in3_s, in4_s);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::addmm_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
  };
})

#ifdef FBCODE_CAFFE2
// Disable externally to avoid MSVC errors in open-source CI

REGISTER_OPERATOR_FUNCTOR(
    static_runtime::clamp_nan_to_num,
    static_runtime_clamp_nan_to_num,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(
              n,
              "static_runtime::clamp_nan_to_num(Tensor input, Scalar? min, Scalar? max, float? nan, float? posinf, float? posinf) -> Tensor")) {
        return nullptr;
      }
      auto clamp_min_ival_opt = toIValue(n->input(1));
      auto clamp_max_ival_opt = toIValue(n->input(2));
      TORCH_CHECK(
          clamp_min_ival_opt.has_value() && clamp_max_ival_opt.has_value());

      auto clamp_min_opt = clamp_min_ival_opt->toOptional<at::Scalar>();
      auto clamp_max_opt = clamp_max_ival_opt->toOptional<at::Scalar>();
      TORCH_CHECK(clamp_min_opt.has_value() && clamp_max_opt.has_value());

      return [te = createClampNanToNum(),
              clamp_min = clamp_min_opt->to<float>(),
              clamp_max =
                  clamp_max_opt->to<float>()](ProcessedNode* p_node) mutable {
        const auto& in0_t = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(in0_t);
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        auto in3_s = p_node->Input(3).toOptional<double>();

        if (!te || !te->checkInput<float>(in0_t)) {
          at::cpu::nan_to_num_out(
              out_t,
              at::cpu::clamp(in0_t, clamp_min, clamp_max),
              in3_s,
              std::nullopt,
              std::nullopt);
          return;
        }
        at::native::resize_(out_t, in0_t.sizes(), std::nullopt);

        auto output_size = in0_t.numel();

        // This might be UB if in3_s is absurdly large, but most implementations
        // just turn it into `inf` in that case. The PyTorch core nan_to_num
        // kernel just static_cast()s the limits to the destination type, so
        // we'll ignore overflow issues here as well.
        auto nan = in3_s.has_value() ? static_cast<float>(*in3_s) : 0.f;

        te->call(
            {out_t.data_ptr(),
             in0_t.data_ptr(),
             &clamp_min,
             &clamp_max,
             &nan,
             &output_size});
      };
    })

#endif

REGISTER_OPERATOR_FUNCTOR(aten::clamp, aten_clamp, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"))) {
    return [te = createClamp()](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
      auto in2_s = p_node->Input(2).toOptional<at::Scalar>();
      if (!te->checkInput<float>(in0_t)) {
        at::cpu::clamp_out(out_t, in0_t, in1_s, in2_s);
        return;
      }
      at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
      auto output_size = in0_t.numel();
      auto min = in1_s.has_value() ? in1_s->toFloat()
                                   : -std::numeric_limits<float>::infinity();
      auto max = in2_s.has_value() ? in2_s->toFloat()
                                   : std::numeric_limits<float>::infinity();
      te->call({out_t.data_ptr(), in0_t.data_ptr(), &min, &max, &output_size});
    };
  }
  if (n->matches(
          "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor")) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      auto in1_t = p_node->Input(1).toOptional<at::Tensor>();
      auto in2_t = p_node->Input(2).toOptional<at::Tensor>();
      at::cpu::clamp_out(out_t, in0_t, in1_t, in2_t);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::bmm, aten_bmm, [](Node* n) -> SROperator {
  if (!n->matches(
          torch::schema("aten::bmm(Tensor self, Tensor mat2) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::bmm_out(out_t, in0_t, in1_t);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::nan_to_num, aten_nan_to_num, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_d = p_node->Input(1).toOptional<double>();
    const auto in2_d = p_node->Input(2).toOptional<double>();
    const auto in3_d = p_node->Input(3).toOptional<double>();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::nan_to_num(in0_t, in1_d, in2_d, in3_d);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::nan_to_num_out(in0_t, in1_d, in2_d, in3_d, out_t);
  };
})

namespace {

void varStackSerialOut(
    at::Tensor& result,
    int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  auto result_sizes = inputs[0].sizes().vec();
  result_sizes.insert(result_sizes.begin() + dim, inputs.size());
  at::native::resize_(result, result_sizes);

  AT_DISPATCH_FLOATING_TYPES(
      result.scalar_type(), "varstack_serial_kernel", [&]() {
        at::native::detail::
            stack_serial_kernel_impl<scalar_t, ProcessedNodeInputWrapper>(
                result, inputs, dim);
      });
}

std::vector<at::Tensor> unsqueezeVarStackInputs(
    const ProcessedNodeInputWrapper& inputs,
    const int64_t dim) {
  std::vector<at::Tensor> result;
  result.reserve(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    result.push_back(at::native::unsqueeze(inputs[i], dim));
  }
  return result;
}

void varstackNonserialOut(
    at::Tensor& result,
    const int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  std::vector<at::Tensor> inputs_unsqueezed =
      unsqueezeVarStackInputs(inputs, dim);
  fastResizeToZero(result);
  at::cpu::cat_outf(inputs_unsqueezed, dim, result);
}

void varStackFastOut(
    at::Tensor& out,
    int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  DCHECK(out.is_contiguous());
  const auto num_inputs = static_cast<int64_t>(inputs.size());
  TORCH_CHECK(num_inputs > 0, "stack expects a non-empty list of tensors");

  const auto first_tensor_shape = inputs[0].sizes();
  for (const auto i : c10::irange(1, num_inputs)) {
    const auto shape = inputs[i].sizes();
    TORCH_CHECK(
        shape == first_tensor_shape,
        "Stack expects each tensor to be the same size, but got ",
        first_tensor_shape,
        " at position 0 and ",
        shape,
        " at position ",
        i);
  }

  const std::array<int64_t, 2> output_size = (dim == 0 || dim == -2)
      ? std::array<int64_t, 2>{num_inputs, 1}
      : std::array<int64_t, 2>{1, num_inputs};

  at::native::resize_(out, output_size, std::nullopt);

  AT_DISPATCH_ALL_TYPES(out.scalar_type(), "varStackFastOut", [&]() {
    auto* out_data = out.mutable_data_ptr<scalar_t>();
    for (const auto i : c10::irange(num_inputs)) {
      auto& tensor = inputs[i];
      auto* input_ptr = tensor.const_data_ptr<scalar_t>();
      out_data[i] = *input_ptr;
    }
  });
}

bool inputsAreScalars(const ProcessedNodeInputWrapper& inputs) {
  // All stack inputs should have the same size, so we only check
  // the first one. If this isn't true, an exception will be thrown
  // in the VarStack implementation
  const auto& first_tensor = inputs[0];
  return first_tensor.sizes()[0] == 1 && first_tensor.dim() == 1;
}

void varStackOut(ProcessedNode& pnode, int64_t dim) {
  const auto num_inputs = pnode.num_inputs();
  TORCH_CHECK(num_inputs > 1, "stack expects a non-empty list of tensors");
  dim = c10::maybe_wrap_dim(dim, pnode.Input(0).toTensor().dim() + 1);

  auto inputs = ProcessedNodeInputWrapper(pnode);
  auto& output = pnode.Output(0).toTensor();

  if (output.is_contiguous() && inputsAreScalars(inputs)) {
    varStackFastOut(output, dim, inputs);
    return;
  }

  bool can_use_serial = at::native::detail::CanUseNativeSerialStack<
      ProcessedNodeInputWrapper,
      /*skip_overlap_check*/ true>::call(output, inputs, dim);

  if (can_use_serial) {
    varStackSerialOut(output, dim, inputs);
    return;
  }
  varstackNonserialOut(output, dim, inputs);
}

} // namespace

// Split out into a function to appease MSVC's pre-processor
static SROperator aten_stack(Node* n) {
  if (!n->matches(torch::schema(
          "aten::stack(Tensor[] tensors, int dim=0) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto inputs = p_node->Input(0).toTensorVector();
    TORCH_CHECK(!inputs.empty(), "stack expects non-empty tensor list");
    const auto dim = p_node->Input(1).toInt();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::_stack_cpu(inputs, dim);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::_stack_out_cpu(inputs, dim, out_t);
  };
}

REGISTER_OPERATOR_FUNCTOR(aten::stack, aten_stack, aten_stack)

REGISTER_OPERATOR_FUNCTOR(
    prim::VarStack,
    prim_VarStack,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::VarStack)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const size_t num_inputs = p_node->num_inputs();
        const auto dim = p_node->Input(num_inputs - 1).toInt();

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(p_node->Input(0).toTensor());
        }
        varStackOut(*p_node, dim);
      };
    })

REGISTER_OPERATOR_FUNCTOR(aten::leaky_relu, aten_leaky_relu, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_s = p_node->Input(1).toScalar();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::leaky_relu(in0_t, in1_s);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    at::cpu::leaky_relu_out(out_t, in0_t, in1_s);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::relu, aten_relu, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::relu(Tensor self) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  auto te = createRelu();
  return [te](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    if (!te->checkInput<float>(in0_t)) {
      fastResizeToZero(out_t);
      at::cpu::threshold_out(out_t, in0_t, 0, 0);
      return;
    }
    at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
    int64_t nn = in0_t.numel();
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::tanh, aten_tanh, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::tanh(Tensor self) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  auto te = createTanh();
  return [te](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    if (!te->checkInput<float>(in0_t)) {
      fastResizeToZero(out_t);
      at::cpu::tanh_out(out_t, in0_t);
      return;
    }
    at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
    int64_t nn = in0_t.numel();
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
  };
})

REGISTER_OPERATOR_FUNCTOR(
    prim::TensorExprDynamicGroup,
    prim_TensorExprDynamicGroup,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TensorExprDynamicGroup)) {
        return nullptr;
      }
      auto graph = n->g(attr::Subgraph);
      Code code(graph, "");
      return [code](ProcessedNode* p_node) {
        auto num_outputs = p_node->num_outputs();
        Stack stack;
        if (p_node->Output(0).isNone()) {
          stack.reserve(p_node->num_inputs());
        } else {
          stack.reserve(p_node->num_inputs() + num_outputs);
          for (const auto& o : p_node->outputs()) {
            stack.emplace_back(o);
          }
        }
        for (auto i : c10::irange(p_node->num_inputs())) {
          stack.emplace_back(p_node->Input(i));
        }
        runTensorExprDynamicGroup(code, stack);
        if (p_node->Output(0).isNone()) {
          TORCH_INTERNAL_ASSERT(
              stack.size() == num_outputs,
              "Unexpected # of outputs on stack after executing TensorExprDynamicGroup");
          for (auto i : c10::irange(num_outputs)) {
            p_node->Output(i) = std::move(stack[i]);
          }
        }
      };
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::sigmoid,
    aten_sigmoid,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema("aten::sigmoid(Tensor self) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      auto te = createSigmoid();
      return [te](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(in0_t);
        }
        auto& out_t = p_node->Output(0).toTensor();
        if (!te->checkInput<float>(in0_t)) {
          fastResizeToZero(out_t);
          at::cpu::sigmoid_out(out_t, in0_t);
          return;
        }
        at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
        int64_t nn = in0_t.numel();
        te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
      };
    })

REGISTER_OPERATOR_FUNCTOR(aten::logit, aten_logit, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::logit(Tensor self, float? eps=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  std::optional<double> clamp = std::nullopt;
  if (n->inputs()[1]->node()->kind() == prim::Constant) {
    auto clamp_d = toIValue(n->inputs()[1])->toOptional<double>();
    clamp = clamp_d;
  }
  auto te = clamp ? createLogit() : nullptr;
  return [te, clamp](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    if (!te || !te->checkInput<float>(in0_t)) {
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto in1_d = p_node->Input(1).toOptional<double>();
      fastResizeToZero(out_t);
      at::native::logit_out(in0_t, in1_d, out_t);
      return;
    }
    at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
    int64_t nn = in0_t.numel();
    float c = clamp.value() ? static_cast<float>(clamp.value()) : 0;
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn, &c});
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::clone, aten_clone, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) ->Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& src = p_node->Input(0).toTensor();
    const auto& optional_memory_format =
        p_node->Input(1).toOptional<c10::MemoryFormat>();
    auto memory_format =
        optional_memory_format.value_or(c10::MemoryFormat::Preserve);
    /*
      disable out_variant of clone for case with stride = 0 and
      memory formats other than preserve. Perform dynamic allocation
      instead of memory reuse for simpler implementation. We could,
      in principle, figure out copy of strides.
    */
    if ((at::has_internal_overlap(src.unsafeGetTensorImpl()) ==
         at::MemOverlap::Yes) ||
        (memory_format != c10::MemoryFormat::Preserve)) {
      p_node->Output(0) = at::native::clone(src, memory_format);
      return;
    }
    if (p_node->Output(0).isNone()) {
      if (src.is_non_overlapping_and_dense()) {
        // Copy all strides
        p_node->Output(0) =
            at::empty_strided(src.sizes(), src.strides(), src.options());
      } else {
        memory_format = src.suggest_memory_format();
        p_node->Output(0) = create_empty_from(src, memory_format);
      }
    }
    auto& out_t = p_node->Output(0).toTensor();
    at::native::resize_impl_cpu_(
        out_t.unsafeGetTensorImpl(), src.sizes(), src.strides());
    at::native::copy_(out_t, src, false);
  };
})

REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_byte_rowwise_offsets,
    quantized_embedding_bag_byte_rowwise_offsets,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_byte_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& weight = p_node->Input(0).toTensor();
        const auto& indices = p_node->Input(1).toTensor();
        const auto offsets = p_node->Input(2).toOptional<at::Tensor>();
        const auto pruned_weights = p_node->Input(5).toBool();
        const auto per_sample_weights =
            p_node->Input(6).toOptional<at::Tensor>();
        const auto compressed_indices_mapping =
            p_node->Input(7).toOptional<at::Tensor>();
        const auto include_last_offset = p_node->Input(8).toBool();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(weight, at::kFloat);
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        return at::native::embedding_bag_byte_rowwise_offsets_out(
            out_t,
            weight,
            indices,
            offsets,
            false, // unused scale_grad_by_freq
            0, // unused mode
            pruned_weights,
            per_sample_weights,
            compressed_indices_mapping,
            include_last_offset);
      };
    })

REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_4bit_rowwise_offsets,
    embedding_bag_4bit_rowwise_offsets,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_4bit_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& weight = p_node->Input(0).toTensor();
        const auto& indices = p_node->Input(1).toTensor();
        const auto offsets = p_node->Input(2).toOptional<at::Tensor>();
        const auto pruned_weights = p_node->Input(5).toBool();
        const auto per_sample_weights =
            p_node->Input(6).toOptional<at::Tensor>();
        const auto compressed_indices_mapping =
            p_node->Input(7).toOptional<at::Tensor>();
        const auto include_last_offset = p_node->Input(8).toBool();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(weight, at::kFloat);
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        return at::native::embedding_bag_4bit_rowwise_offsets_out(
            out_t,
            weight,
            indices,
            offsets,
            false, // unused scale_grad_by_freq
            0, // unused mode
            pruned_weights,
            per_sample_weights,
            compressed_indices_mapping,
            include_last_offset);
      };
    })

REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_byte_prepack,
    embedding_bag_byte_prepack,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_byte_prepack(Tensor weight) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& weight = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::qembeddingbag_byte_prepack(weight);
          return;
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        at::native::qembeddingbag_byte_prepack_out(out_t, weight);
      };
    })

// The out variant takes precedence over native
REGISTER_OPERATOR_FUNCTOR(aten::narrow_copy, aten_narrow_copy, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& self = p_node->Input(0).toTensor(); // self
    const auto dim = p_node->Input(1).toInt(); // dim
    int64_t start = 0;
    if (p_node->Input(2).isScalar()) {
      start = p_node->Input(2).toInt();
    } else {
      auto& t = p_node->Input(2).toTensor();
      start = t.item<int64_t>();
    }
    auto length = p_node->Input(3).toInt(); // length

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) =
          at::native::narrow_copy_dense_cpu(self, dim, start, length);
      return;
    }
    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);
    at::native::narrow_copy_dense_cpu_out(self, dim, start, length, output);
  };
})
REGISTER_OPERATOR_FUNCTOR(aten::index, aten_index, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_l =
        at::native::toListOfOptionalTensors(p_node->Input(1).toListRef());
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::index(in0_t, in1_l);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::index_out(out_t, in0_t, in1_l);
  };
})

REGISTER_OPERATOR_FUNCTOR(
    aten::index_select,
    aten_index_select,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        const auto dim = p_node->Input(1).toInt();
        const auto& index = p_node->Input(2).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::index_select_cpu_(self, dim, index);
          return;
        }
        auto& out = p_node->Output(0).toTensor();
        fastResizeToZero(out);
        at::native::index_select_out_cpu_(self, dim, index, out);
      };
    })

REGISTER_OPERATOR_FUNCTOR(aten::pow, aten_pow, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      if (p_node->Output(0).isNone()) {
        const auto& in0_t = p_node->Input(0).toTensor();
        auto dtype =
            at::native::result_type(in0_t, p_node->Input(1).toTensor());
        p_node->Output(0) = create_empty_from(in0_t, dtype);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::pow_out(
          out_t, p_node->Input(0).toTensor(), p_node->Input(1).toTensor());
    };
  }
  if (n->matches(torch::schema(
          "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      if (p_node->Output(0).isNone()) {
        const auto& in1_t = p_node->Input(1).toTensor();
        auto dtype =
            at::native::result_type(p_node->Input(0).toScalar(), in1_t);
        p_node->Output(0) = at::native::empty_like(
            in1_t,
            dtype,
            in1_t.options().layout_opt(),
            in1_t.options().device_opt(),
            in1_t.options().pinned_memory_opt(),
            at::MemoryFormat::Preserve);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::pow_out(
          out_t, p_node->Input(0).toScalar(), p_node->Input(1).toTensor());
    };
  }
  if (n->matches(torch::schema(
          "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      if (p_node->Output(0).isNone()) {
        const auto& in0_t = p_node->Input(0).toTensor();
        auto dtype =
            at::native::result_type(in0_t, p_node->Input(1).toScalar());
        p_node->Output(0) = at::native::empty_like(
            in0_t,
            dtype,
            in0_t.options().layout_opt(),
            in0_t.options().device_opt(),
            in0_t.options().pinned_memory_opt(),
            at::MemoryFormat::Preserve);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::pow_out(
          out_t, p_node->Input(0).toTensor(), p_node->Input(1).toScalar());
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

namespace {

struct ToArgs {
  std::optional<at::ScalarType> dtype;
  c10::Layout layout;
  bool know_to_will_alias = false;
  std::optional<c10::MemoryFormat> memory_format;
};

template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
ToArgs extract_to_args(ProcessedNode* p_node) {
  ToArgs result;
  if (!has_constant_non_tensor_dtype_and_flags && p_node->Input(1).isTensor()) {
    const auto& other = p_node->Input(1).toTensor();
    result.dtype = other.scalar_type();
    result.layout = other.layout();
    TORCH_DCHECK_EQ(other.device().type(), c10::DeviceType::CPU);
  } else {
    const auto& self = p_node->Input(0).toTensor();
    result.dtype = p_node->Input(1).toOptional<at::ScalarType>();
    result.layout = self.layout();
    // Static runtime only works with CPU tensors; don't need to read this.
    TORCH_DCHECK_EQ(self.device().type(), c10::DeviceType::CPU);
    result.know_to_will_alias = has_constant_non_tensor_dtype_and_flags &&
        (!result.dtype.has_value() ||
         result.dtype.value() == self.dtype().toScalarType());
  }
  if (has_memory_format) {
    TORCH_DCHECK_EQ(p_node->num_inputs(), 5);
    result.memory_format = p_node->Input(4).toOptional<c10::MemoryFormat>();
    result.know_to_will_alias = result.know_to_will_alias &&
        (result.memory_format.value_or(c10::MemoryFormat::Preserve) ==
         c10::MemoryFormat::Preserve);
  }

  return result;
}

template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
struct CheckToWillAlias {
  static bool call(
      ProcessedNode* p_node,
      const at::Tensor& self,
      const ToArgs& to_args) {
    // The to_maybe_copy_out operator functor should have detected a
    // constant true `copy` argument and used to_copy instead.
    bool copy = false;
    if (has_constant_non_tensor_dtype_and_flags) {
      DCHECK(!p_node->Input(3).toBool());
      copy = false;
    } else {
      copy = p_node->Input(3).toBool();
    }
    return !copy &&
        (to_args.know_to_will_alias ||
         at::native::to_will_alias(
             self,
             to_args.dtype,
             to_args.layout,
             c10::Device{c10::DeviceType::CPU},
             copy,
             has_memory_format ? to_args.memory_format
      
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime/static`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime/static`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/runtime/static`):

- [`fusion.h_kw.md_docs.md`](./fusion.h_kw.md_docs.md)
- [`ProcessedNodeInputs.cpp_docs.md_docs.md`](./ProcessedNodeInputs.cpp_docs.md_docs.md)
- [`impl.h_docs.md_docs.md`](./impl.h_docs.md_docs.md)
- [`memory_planner.cpp_kw.md_docs.md`](./memory_planner.cpp_kw.md_docs.md)
- [`te_wrapper.cpp_kw.md_docs.md`](./te_wrapper.cpp_kw.md_docs.md)
- [`generated_ops.cpp_kw.md_docs.md`](./generated_ops.cpp_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`te_wrapper.h_docs.md_docs.md`](./te_wrapper.h_docs.md_docs.md)
- [`te_wrapper.cpp_docs.md_docs.md`](./te_wrapper.cpp_docs.md_docs.md)
- [`ProcessedNodeInputs.h_kw.md_docs.md`](./ProcessedNodeInputs.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ops.cpp_docs.md_docs.md`
- **Keyword Index**: `ops.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
