# Documentation: `docs/torch/csrc/jit/runtime/static/generated_ops.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/static/generated_ops.cpp_docs.md`
- **Size**: 53,285 bytes (52.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/static/generated_ops.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/static/generated_ops.cpp`
- **Size**: 178,690 bytes (174.50 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
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
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch::jit {

REGISTER_OPERATOR_FUNCTOR(
    aten::absolute,
    aten_absolute,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::absolute(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::absolute(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::absolute_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::angle, aten_angle, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::angle(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::angle(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::angle_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::sgn, aten_sgn, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::sgn(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sgn(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::sgn_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::acos, aten_acos, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::acos(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::acos(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::acos_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::arccos, aten_arccos, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::arccos(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::arccos(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::arccos_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::_add_relu, aten__add_relu, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      const auto alpha = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::add_relu(self, other, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::add_relu_out(self, other, alpha, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::addmv, aten_addmv, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& mat = p_node->Input(1).toTensor();
      const auto& vec = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::addmv(self, mat, vec, beta, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::addmv_out(out, self, mat, vec, beta, alpha);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::addr, aten_addr, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& vec1 = p_node->Input(1).toTensor();
      const auto& vec2 = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::addr(self, vec1, vec2, beta, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::addr_out(self, vec1, vec2, beta, alpha, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::_test_functorch_fallback,
    aten__test_functorch_fallback,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_test_functorch_fallback(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::_test_functorch_fallback(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::_test_functorch_fallback_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::argmax, aten_argmax, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toOptional<int64_t>();
      const auto keepdim = p_node->Input(2).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::argmax(self, dim, keepdim);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::argmax_out(out, self, dim, keepdim);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::acosh, aten_acosh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::acosh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::acosh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::acosh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::asinh, aten_asinh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::asinh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::asinh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::asinh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::arcsinh,
    aten_arcsinh,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::arcsinh(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::arcsinh(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::arcsinh_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::atanh, aten_atanh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::atanh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::atanh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::atanh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::arctanh,
    aten_arctanh,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::arctanh(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::arctanh(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::arctanh_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::asin, aten_asin, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::asin(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::asin(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::asin_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::arcsin, aten_arcsin, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::arcsin(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::arcsin(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::arcsin_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::atan, aten_atan, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::atan(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::atan(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::atan_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::arctan, aten_arctan, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::arctan(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::arctan(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::arctan_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::baddbmm, aten_baddbmm, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& batch1 = p_node->Input(1).toTensor();
      const auto& batch2 = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::baddbmm(self, batch1, batch2, beta, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::baddbmm_out(out, self, batch1, batch2, beta, alpha);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_not,
    aten_bitwise_not,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::bitwise_not(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_not(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::bitwise_not_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::copysign,
    aten_copysign,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::copysign(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::copysign_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_not,
    aten_logical_not,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::logical_not(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_not(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_not_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_xor,
    aten_logical_xor,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logical_xor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_xor(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_xor_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_and,
    aten_logical_and,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logical_and(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_and(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_and_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_or,
    aten_logical_or,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logical_or(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_or(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_or_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::ceil, aten_ceil, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::ceil(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ceil(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::ceil_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::clamp_max,
    aten_clamp_max,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::clamp_max(Tensor self, Scalar max) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto max = p_node->Input(1).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::clamp_max(self, max);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::clamp_max_out(out, self, max);
        };
      }

      if (n->matches(torch::schema(
              "aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& max = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::clamp_max(self, max);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::clamp_max_out(out, self, max);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::clip, aten_clip, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto min = p_node->Input(1).toOptional<at::Scalar>();
      const auto max = p_node->Input(2).toOptional<at::Scalar>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::clip(self, min, max);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::clip_out(self, min, max, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::complex,
    aten_complex,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::complex(Tensor real, Tensor imag) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& real = p_node->Input(0).toTensor();
          const auto& imag = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::complex(real, imag);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::complex_out(real, imag, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::polar, aten_polar, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::polar(Tensor abs, Tensor angle) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& abs = p_node->Input(0).toTensor();
      const auto& angle = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::polar(abs, angle);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::polar_out(abs, angle, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::cos, aten_cos, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::cos(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::cos(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::cos_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::cosh, aten_cosh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::cosh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::cosh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::cosh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::cumprod, aten_cumprod, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto dtype = p_node->Input(2).toOptional<at::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::cumprod(self, dim, dtype);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::cumprod_out(out, self, dim, dtype);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::diff, aten_diff, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto n = p_node->Input(1).toInt();
      const auto dim = p_node->Input(2).toInt();
      const auto prepend = p_node->Input(3).toOptional<at::Tensor>();
      const auto append = p_node->Input(4).toOptional<at::Tensor>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::diff(self, n, dim, prepend, append);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::diff_out(self, n, dim, prepend, append, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::divide, aten_divide, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::divide.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::divide(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::divide_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::true_divide,
    aten_true_divide,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::true_divide(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::true_divide_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::dot, aten_dot, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::dot(Tensor self, Tensor tensor) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& tensor = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::dot(self, tensor);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::dot_out(self, tensor, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::vdot, aten_vdot, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::vdot(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::vdot(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::vdot_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::erf, aten_erf, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::erf(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::erf(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::erf_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::erfc, aten_erfc, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::erfc(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::erfc(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::erfc_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::exp, aten_exp, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::exp(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::exp(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::exp_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::exp2, aten_exp2, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::exp2(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::exp2(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::exp2_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::expm1, aten_expm1, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::expm1(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::expm1(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::expm1_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::floor, aten_floor, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::floor(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::floor(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::floor_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::frac, aten_frac, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::frac(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::frac(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::frac_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::gcd, aten_gcd, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::gcd(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gcd(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::gcd_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::lcm, aten_lcm, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::lcm(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lcm(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lcm_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::index_copy, aten_index_copy, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& source = p_node->Input(3).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::index_copy(self, dim, index, source);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::index_copy_out(out, self, dim, index, source);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::isin, aten_isin, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& elements = p_node->Input(0).toTensor();
      const auto& test_elements = p_node->Input(1).toTensor();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(elements, test_elements, assume_unique, invert);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, elements, test_elements, assume_unique, invert);
    };
  }

  if (n->matches(torch::schema(
          "aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& elements = p_node->Input(0).toTensor();
      const auto test_element = p_node->Input(1).toScalar();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(elements, test_element, assume_unique, invert);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, elements, test_element, assume_unique, invert);
    };
  }

  if (n->matches(torch::schema(
          "aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto element = p_node->Input(0).toScalar();
      const auto& test_elements = p_node->Input(1).toTensor();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(element, test_elements, assume_unique, invert);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, element, test_elements, assume_unique, invert);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::kron, aten_kron, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::kron(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::kron(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::kron_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::ldexp, aten_ldexp, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::ldexp(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::ldexp_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::log10, aten_log10, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::log10(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log10(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::log10_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::log1p, aten_log1p, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::log1p(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log1p(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::log1p_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::log2, aten_log2, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::log2(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log2(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::log2_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::logaddexp,
    aten_logaddexp,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logaddexp(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::logaddexp(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::logaddexp_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::logaddexp2,
    aten_logaddexp2,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logaddexp2(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::logaddexp2(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::logaddexp2_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::xlogy, aten_xlogy, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::xlogy(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::xlogy_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::_log_softmax,
    aten__log_softmax,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          const auto half_to_float = p_node->Input(2).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_log_softmax(self, dim, half_to_float);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_log_softmax_out(out, self, dim, half_to_float);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::_log_softmax_backward_data,
    aten__log_softmax_backward_data,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toInt();
          const auto input_dtype = p_node->Input(3).toScalarType();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_log_softmax_backward_data(
                grad_output, output, dim, input_dtype);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_log_softmax_backward_data_out(
              out, grad_output, output, dim, input_dtype);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::_logcumsumexp,
    aten__logcumsumexp,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_logcumsumexp(Tensor self, int dim) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::_logcumsumexp_cpu(self, dim);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::_logcumsumexp_out_cpu(self, dim, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::logcumsumexp,
    aten_logcumsumexp,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logcumsumexp(Tensor self, int dim) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logcumsumexp(self, dim);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logcumsumexp_out(self, dim, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::matrix_power,
    aten_matrix_power,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::matrix_power(Tensor self, int n) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto n = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::matrix_power(self, n);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::matrix_power_out(self, n, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::mm, aten_mm, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::mm(Tensor self, Tensor mat2) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& mat2 = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::mm(self, mat2);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::mm_out(out, self, mat2);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::multiply,
    aten_multiply,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::multiply(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::multiply_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::mv, aten_mv, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::mv(Tensor self, Tensor vec) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& vec = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::mv(self, vec);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::mv_out(self, vec, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    aten::mvlgamma,
    aten_mvlgamma,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::mvlgamma(Tensor self, int p) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto p = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::mvlgamma(self, p);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::mvlgamma_out(self, p, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::rad2deg,
    aten_rad2deg,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::rad2deg(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::rad2deg(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::rad2deg_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::deg2rad,
    aten_deg2rad,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::deg2rad(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::deg2rad(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::deg2rad_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    aten::reciprocal,
    aten_reciprocal,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::reciprocal(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::reciprocal(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::reciprocal_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::neg, aten_neg, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::neg(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::neg(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::neg_out
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

- **File Documentation**: `generated_ops.cpp_docs.md_docs.md`
- **Keyword Index**: `generated_ops.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
