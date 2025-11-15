# Documentation: `torch/csrc/jit/runtime/register_prim_ops.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/register_prim_ops.cpp`
- **Size**: 135,140 bytes (131.97 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/autocast_mode.h>
#include <ATen/core/Generator.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
#include <torch/library.h>
#include <optional>

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch::jit {

namespace {

std::string stringSlice(
    std::string string,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  int64_t start_val = start.has_value() ? start.value() : INT64_MAX;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;

  const int64_t num_vals =
      slice_indices_adjust(string.size(), &start_val, &end_val, step);

  int64_t i = start_val;
  std::string result;
  for ([[maybe_unused]] const auto j : c10::irange(num_vals)) {
    result += string[i];
    i += step;
  }

  return result;
}

// consecutive whitespace are regarded as a single separator,
// the result will contain no empty strings at the start or end
// if the string has leading or trailing whitespace.
c10::List<std::string> splitNoneSeparator(const std::string& string) {
  c10::List<std::string> splits;
  // whitespaces includes tab, space and
  // the delimiters defined in the implementation of splitlines
  std::string whitespaces =
      " \t\n\r\r\n\v\x0b\f\x0c\x1c\x1d\x1e\x85\u2028\u2029";
  std::string::size_type prev_pos = 0;
  std::string::size_type pos = 0;

  while ((pos = string.find_first_of(whitespaces, pos)) != std::string::npos) {
    auto substr = string.substr(prev_pos, pos - prev_pos);
    // skip the whitespaces as the Python split() method
    if (!substr.empty()) {
      splits.emplace_back(substr);
    }
    pos++;
    prev_pos = pos;
  }
  if (prev_pos != string.size()) {
    splits.emplace_back(string.substr(prev_pos));
  }
  return splits;
}

bool isSortableTupleType(
    const TupleTypePtr& tuple_type,
    std::stringstream& why_not) {
  for (const TypePtr& ele_type : tuple_type->containedTypes()) {
    switch (ele_type->kind()) {
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::StringType:
      case TypeKind::TensorType:
        continue;
      case TypeKind::TupleType:
        if (!isSortableTupleType(ele_type->expect<TupleType>(), why_not)) {
          return false;
        }
        continue;
      case TypeKind::ClassType:
        if (!c10::checkObjectSortSchema(
                ele_type->expect<ClassType>(), why_not)) {
          return false;
        }
        continue;
      default:
        why_not << "Contained elements in " << *tuple_type
                << " are not sortable. Only Int, Bool, Float, String, Tensor, "
                << "a User Defined Class with __lt__ method defined or Tuples "
                << "of aforementionted types can be sorted.";
        return false;
    }
  }

  return true;
}

bool isSortableListOfObjectsOrTuples(
    c10::List<IValue>& ivalues,
    std::stringstream& why_not) {
  if (ivalues.empty()) {
    return true;
  }

  auto type = ivalues.get(0).type();
  // We assume lists have homogeneous types, use first element to determine
  // best sorting methods. If in the future we need to support heterogeneous
  // types inside list, then sorting needs to have runtime sortable checks.
  const size_t n = ivalues.size();
  for (const auto i : c10::irange(n)) {
    const IValue& v = ivalues.get(i);
    auto curr_type = v.type();
    if (*curr_type != *type) {
      why_not << "Only values of same type can be compared. "
              << "Found " << type->repr_str() << " and "
              << curr_type->repr_str();
      return false;
    }
  }

  if (auto tuple_type = type->cast<TupleType>()) {
    return isSortableTupleType(tuple_type, why_not);
  }

  if (auto class_type = type->cast<ClassType>()) {
    return c10::checkObjectSortSchema(class_type, why_not) != nullptr;
  }

  // Basic types like tensors/ints/floats/bools/strs are not checked in this
  // method because they should have been schema matched to specialized
  // aten::sort kernels using listSort<T>.
  why_not << "Only list of Tensors, ints, floats, bools, strs, "
          << "a User Defined Class that defines the __lt__ compare method "
          << "or Tuples of aforementioned types can be sorted, got list of "
          << type->repr_str() << "\n";
  return false;
}

template <bool has_reverse_arg, bool copy_return_list>
void sort_op(Stack& stack) {
  bool reverse = has_reverse_arg ? pop(stack).toBool() : false;
  auto g_list = pop(stack).toList();

  if (copy_return_list) {
    g_list = g_list.copy();
  }

  if (!g_list.empty()) {
    std::stringstream error_str;
    TORCH_CHECK(
        isSortableListOfObjectsOrTuples(g_list, error_str), error_str.str());

    c10::IValueComparator comparator;
    if (reverse) {
      comparator = c10::getGreaterThanComparator(g_list.get(0));
    } else {
      comparator = c10::getLessThanComparator(g_list.get(0));
    }
    std::sort(g_list.begin(), g_list.end(), comparator);
  }

  if (copy_return_list) {
    push(stack, g_list);
  }
}

template <typename T, typename U>
auto powWrapper(T a, U b) {
  TORCH_CHECK(
      !(a == 0.0 && b < 0.0), "0.0 cannot be raised to a negative power")
  return pow(a, b);
}

static const std::vector<OperatorGeneratorArgs> opGenArgs{
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::str(t elem) -> str"),
        [](Stack& stack) {
          std::stringstream ss;
          ss << pop(stack);
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::list(str t) -> str[]"),
        [](Stack& stack) {
          auto str = pop(stack).toStringRef();
          c10::List<std::string> chars;
          chars.reserve(str.size());
          for (auto c : str) {
            chars.push_back(std::string(1, c));
          }
          push(stack, std::move(chars));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::cpu(Tensor(a) self) -> Tensor(a|b)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.cpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::numpy_T.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.numpy_T());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::matrix_H.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.matrix_H());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mT.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.mT());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mH.a(Tensor(a) self) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.mH());
        },
        aliasAnalysisFromSchema()),

    // only used internally in range() translation
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__range_length(int lo, int hi, int step) -> int"),
        [](Stack& stack) {
          int64_t lo = 0, hi = 0, step = 0;
          pop(stack, lo, hi, step);
          // error handling when step_val = 0 during runtime
          TORCH_CHECK(step != 0, "range() arg 3 must not be zero");
          if (step > 0 && lo < hi) {
            push(stack, 1 + (hi - 1 - lo) / step);
          } else if (step < 0 && lo > hi) {
            push(stack, 1 + (lo - 1 - hi) / (0 - step));
          } else {
            push(stack, 0);
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__derive_index(int index, int start, int step) -> int"),
        [](Stack& stack) {
          int64_t index = 0, start = 0, step = 0;
          pop(stack, index, start, step);
          push(stack, start + index * step);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::TupleUnpack(Any tup) -> ..."),
        [](Stack& stack) { tupleUnpack(stack); },
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::unchecked_cast(t x) -> t"),
        noop,
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::IntImplicit(Tensor a) -> int"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ true);
          push(stack, a.item<int64_t>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ComplexImplicit(Tensor a) -> complex"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ false);
          push(stack, a.item<c10::complex<double>>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::FloatImplicit(Tensor a) -> float"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ false);
          push(stack, a.item<double>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ScalarImplicit(Tensor a) -> Scalar"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          checkImplicitTensorToNum(a, /*to int*/ false);
          push(stack, a.item());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.Tensor(Tensor a) -> bool"),
        boolTensor,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.int(int a) -> bool"),
        [](Stack& stack) {
          int64_t i = 0;
          pop(stack, i);
          push(stack, (bool)i);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Bool.float(float a) -> bool"),
        [](Stack& stack) {
          double d = 0;
          pop(stack, d);
          push(stack, (bool)d);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.Tensor(Tensor a) -> int"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.item<int64_t>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.bool(bool a) -> int"),
        [](Stack& stack) {
          bool b = false;
          pop(stack, b);
          push(stack, static_cast<int64_t>(b));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.float(float a) -> int"),
        [](Stack& stack) {
          double d = 0;
          pop(stack, d);
          push(stack, static_cast<int64_t>(d));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.Scalar(Scalar a) -> int"),
        [](Stack& stack) {
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isInt()) {
            push(stack, std::move(scalar));
          } else {
            // toScalar() needed to avoid strict type check in IValue::toInt.
            push(stack, static_cast<int64_t>(scalar.toScalar().toInt()));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Int.str(str a) -> int"),
        [](Stack& stack) {
          auto s = pop(stack).toString();
          std::string::size_type sz = 0;
          int64_t val = static_cast<int64_t>(std::stoll(s->string(), &sz));
          TORCH_CHECK(
              sz == s->string().size(),
              "invalid literal for int() ",
              "with base 10: '",
              s->string(),
              "'");
          push(stack, val);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.Tensor(Tensor a) -> float"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.item<double>());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.Scalar(Scalar a) -> float"),
        [](Stack& stack) {
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isDouble()) {
            push(stack, std::move(scalar));
          } else if (scalar.isComplexDouble()) {
            push(stack, scalar.toComplexDouble().real());
          } else {
            push(stack, static_cast<double>(scalar.toInt()));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.int(int a) -> float"),
        [](Stack& stack) {
          int64_t i = 0;
          pop(stack, i);
          push(stack, (float)i);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.bool(bool a) -> float"),
        [](Stack& stack) {
          bool b = false;
          pop(stack, b);
          push(stack, (float)b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Float.str(str a) -> float"),
        [](Stack& stack) {
          auto s = pop(stack).toString();
          std::string::size_type sz = 0;
          double b = std::stod(s->string(), &sz);
          TORCH_CHECK(
              sz == s->string().size(),
              "could not convert string ",
              "to float: '",
              s->string(),
              "'");
          push(stack, b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Complex.Scalar(Scalar a) -> complex"),
        [](Stack& stack) {
          IValue scalar;
          pop(stack, scalar);
          if (scalar.isComplexDouble()) {
            push(stack, std::move(scalar));
          } else if (scalar.isDouble()) {
            push(stack, c10::complex<double>(scalar.toDouble(), 0));
          } else {
            push(stack, c10::complex<double>(scalar.toInt(), 0));
          }
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::Complex.Tensor_Tensor(Tensor a, Tensor b) -> complex"),
        [](Stack& stack) {
          at::Tensor a, b;
          pop(stack, a, b);
          push(stack, c10::complex<double>(a.item<double>(), b.item<double>()));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::format(str self, ...) -> str"),
        [](Stack& stack) { aten_format(stack); },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::einsum.sublist(Tensor a, ...) -> Tensor"),
        [](Stack& stack) {
          size_t num_inputs = pop(stack).toInt();
          einsum(stack, num_inputs);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::NumToTensor.Scalar(Scalar a) -> Tensor"),
        numToTensorScalar,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::RaiseException(str msg, str? cls=None) -> ()"),
        raiseException,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Size(int[] sizes) -> int[]"),
        [](Stack& stack) {},
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::size(Tensor self) -> int[]"),
        size,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sym_size(Tensor self) -> SymInt[]"),
        sym_size,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::stride(Tensor self) -> int[]"),
        [](Stack& stack) {
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.strides());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::sym_stride(Tensor self) -> SymInt[]"),
        sym_stride,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumName(AnyEnumType enum) -> str"),
        [](Stack& stack) {
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->name());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumValue.int(AnyEnumType enum) -> int"),
        [](Stack& stack) {
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::EnumValue.float(AnyEnumType enum) -> float"),
        [](Stack& stack) {
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::EnumValue.str(AnyEnumType enum) -> str"),
        [](Stack& stack) {
          IValue e = pop(stack);
          push(stack, e.toEnumHolder()->value());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // note the compiler knows to type TupleIndex more accurately than it
        // is listed here.
        TORCH_SELECTIVE_SCHEMA("prim::TupleIndex(Any tup, int i) -> Any"),
        tupleIndex,
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.int_list(int[] a, int[] b) -> bool"),
        listNe<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::unchecked_unwrap_optional(t(a)? optional) -> t(a)"),
        noop,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::device(Tensor a) -> Device"),
        device,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::dtype(Tensor a) -> int"),
        dtype,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::layout(Tensor a) -> Layout"),
        layout,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__not__(bool self) -> bool"),
        _not,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__is__(t1 self, t2 obj) -> bool"),
        is,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::__isnot__(t1 self, t2 obj) -> bool"),
        isNot,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::element_size(Tensor self) -> int"),
        [](Stack& stack) {
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.element_size());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::numel(Tensor self) -> int"),
        [](Stack& stack) {
          at::Tensor arg = pop(stack).toTensor();
          push(stack, arg.numel());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dim(Tensor self) -> int"),
        dim,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::get_device(Tensor self) -> int"),
        [](Stack& stack) {
          RECORD_FUNCTION("get_device", c10::ArrayRef<const c10::IValue>{});
          auto result =
              at::get_device((std::move(peek(stack, 0, 1))).toTensor());
          drop(stack, 1);
          pack(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::storage_offset(Tensor self) -> int"),
        [](Stack& stack) {
          RECORD_FUNCTION("storage_offset", c10::ArrayRef<const c10::IValue>{});
          auto result =
              ((std::move(peek(stack, 0, 1))).toTensor()).storage_offset();
          drop(stack, 1);
          pack(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_contiguous(Tensor self) -> bool"),
        [](Stack& stack) {
          RECORD_FUNCTION("is_contiguous", c10::ArrayRef<const c10::IValue>{});
          auto result =
              ((std::move(peek(stack, 0, 1))).toTensor()).is_contiguous();
          drop(stack, 1);
          pack(stack, result);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_contiguous.memory_format(Tensor self, MemoryFormat memory_format) -> bool"),
        [](Stack& stack) {
          auto memory_format = pop(stack).toMemoryFormat();
          auto t = pop(stack).toTensor();
          push(stack, t.is_contiguous(memory_format));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        // NB: intentionally suffixed with extra _format to prevent tests for
        // "_like" suffix from triggering on this
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_strides_like_format(Tensor self, MemoryFormat memory_format) -> bool"),
        [](Stack& stack) {
          auto memory_format = pop(stack).toMemoryFormat();
          auto t = pop(stack).toTensor();
          push(stack, t.unsafeGetTensorImpl()->is_strides_like(memory_format));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::is_non_overlapping_and_dense(Tensor self) -> bool"),
        [](Stack& stack) {
          auto t = pop(stack).toTensor();
          push(stack, t.unsafeGetTensorImpl()->is_non_overlapping_and_dense());
        },
        aliasAnalysisFromSchema()),
    // these ops are generic over the list element type.
    // CREATING GENERIC_LIST_OPS
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::select.t(t[](a) list, int idx) -> t(*)"),
        listSelect,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__getitem__.t(t[](a) list, int idx) -> t(*)"),
        listSelect,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)"),
        listAppend,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::reverse.t(t[](a!) self) -> ()"),
        listReverse,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::extend.t(t[](a!) self, t[] other) -> ()"),
        listExtend,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::copy.t(t[](a) self) -> t[]"),
        listCopy,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_set_item.t(t [](a!) l, int idx, t(b -> *) el) -> t[](a!)"),
        listSetItem,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::clear.t(t[](a!) self) -> ()"),
        listClear,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::Delete.t(t[](a!) self, int idx) -> ()"),
        listDelete,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::insert.t(t[](a!) self, int idx, t(b -> *) el) -> ()"),
        listInsert,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::pop.t(t[](a!) self, int idx=-1) -> t(*)"),
        listPop,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::add.t(t[] a, t[] b) -> t[]"),
        listAdd,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::add_.t(t[](a!) self, t[] b) -> t[]"),
        listInplaceAdd,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> t[]"),
        listSlice,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::list.t(t[] l) -> t[]"),
        listList,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul.left_t(t[] l, int n) -> t[]"),
        listMulIntLeft,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul.right_(int n, t[] l) -> t[]"),
        listMulIntRight,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::mul_.t(t[](a!) l, int n) -> t[](a!)"),
        listMulIntLeftInPlace,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.t(t[] a) -> int"),
        listLen,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.int_list(int[] a, int[] b) -> bool"),
        listEq<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.device(Device a, Device b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack).toDevice();
          auto b = pop(stack).toDevice();
          push(stack, a == b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.device(Device a, Device b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack).toDevice();
          auto b = pop(stack).toDevice();
          push(stack, a != b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::eq.bool(bool a, bool b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, a == b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ne.bool(bool a, bool b) -> bool"),
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, a != b);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_autocast_enabled() -> bool"),
        [](Stack& stack) {
#if defined BUILD_LITE_INTERPRETER || defined C10_MOBILE
          bool enabled = false;
#else
          bool enabled = at::autocast::is_autocast_enabled(at::kCUDA);
#endif
          push(stack, enabled);
        },
        aliasAnalysisConservative()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::is_autocast_cpu_enabled() -> bool"),
        [](Stack& stack) {
#if defined BUILD_LITE_INTERPRETER || defined C10_MOBILE
          bool enabled = false;
#else
          bool enabled = at::autocast::is_autocast_enabled(at::kCPU);
#endif
          push(stack, enabled);
        },
        aliasAnalysisConservative()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::get_autocast_dtype(str device_type) -> ScalarType"),
        [](Stack& stack) {
#if defined BUILD_LITE_INTERPRETER || defined C10_MOBILE
          // autocast is not supported.
          at::ScalarType dtype = at::ScalarType::Undefined;
#else
          at::DeviceType device_type =
              at::Device(pop(stack).toStringRef()).type();
          at::ScalarType dtype = at::autocast::get_autocast_dtype(device_type);
#endif
          push(stack, dtype);
        },
        aliasAnalysisConservative()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::Uninitialized() -> Any"),
        unInitialized,
        aliasAnalysisSpecialCase()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::Print(...) -> ()"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          std::stringstream ss;
          bool first = true;
          for (const IValue& i : last(stack, num_inputs)) {
            if (!first)
              ss << " ";
            first = false;
            ss << i;
          }
          drop(stack, num_inputs);
          ss << '\n';
          auto* handler = getPrintHandler();
          TORCH_INTERNAL_ASSERT(handler);
          handler(ss.str());
        },
        aliasAnalysisSpecialCase()),
    // This is an alternative to aten::cat op that takes variable number of
    // parameters as input.
    // Format:
    //    prim::VarConcat(Tensors..., dim) -> Tensor
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::VarConcat(...) -> Tensor"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          auto dim = pop(stack).toInt();
          std::vector<at::Tensor> inputs(num_inputs - 1);
          for (int i = 0; i < num_inputs - 1; ++i) {
            inputs[num_inputs - 2 - i] = pop(stack).toTensor();
          }
          push(stack, at::cat(inputs, dim));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::VarStack(...) -> Tensor"),
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          auto dim = pop(stack).toInt();
          std::vector<at::Tensor> inputs(num_inputs - 1);
          for (int i = 0; i < num_inputs - 1; ++i) {
            inputs[num_inputs - 2 - i] = pop(stack).toTensor();
          }
          push(stack, at::stack(inputs, dim));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "prim::IfThenElse(bool cond, Any(a) x, Any(b) y) -> Any(a|b)"),
        [](Stack& stack) {
          const auto cond = stack[stack.size() - 3].toBool();
          stack[stack.size() - 3] =
              std::move(stack[stack.size() - (cond ? 2 : 1)]);
          stack.pop_back();
          stack.pop_back();
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::eq.enum(AnyEnumType a, AnyEnumType b) -> bool"),
        [](Stack& stack) {
          IValue x = pop(stack);
          IValue y = pop(stack);
          push(stack, x == y);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::ne.enum(AnyEnumType a, AnyEnumType b) -> bool"),
        [](Stack& stack) {
          IValue x = pop(stack);
          IValue y = pop(stack);
          push(stack, x != y);
        },
        aliasAnalysisFromSchema()),
    // We define aten::dequantize in both native_functions.yaml and here,
    // however, aten::dequantize.any defined here overrides
    // aten::dequantize.tensors in native_functions.yaml. The variants here
    // are only for graph mode quantization, and they should be removed once
    // we deprecate graph mode quantization, and use the variants in
    // native_functions.yaml.
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::dequantize.tensor(Tensor qtensor) -> Tensor"),
        [](Stack& stack) {
          at::Tensor qtensor;
          pop(stack, qtensor);
          push(stack, at::dequantize(qtensor));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::dequantize.list(Tensor[] qtensors) -> Tensor[]"),
        [](Stack& stack) {
          auto qtensors = pop(stack).toTensorVector();
          push(stack, at::dequantize(qtensors));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dequantize.any(Any tensors) -> Any"),
        [](Stack& stack) { dequantize(stack); },
        aliasAnalysisFromSchema()),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::log, std::log(a), float, float),
    DEFINE_STRING_OP(aten::add, a + b, str),
    DEFINE_COMPARISON_OP_WITH_COMPLEX(aten::eq, a == b),
    DEFINE_COMPARISON_OP_WITH_COMPLEX(aten::ne, a != b),
    DEFINE_GENERIC_OP(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        complex,
        complex),
    DEFINE_INT_FLOAT_OP(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        complex),
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::polar,
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        c10::polar(static_cast<double>(a), static_cast<double>(b)),
        Scalar),
    DEFINE_COMPARISON_OP(aten::lt, a < b),
    DEFINE_COMPARISON_OP(aten::gt, a > b),
    DEFINE_COMPARISON_OP(aten::le, a <= b),
    DEFINE_COMPARISON_OP(aten::ge, a >= b),
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::add, a + b),
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::sub, a - b),
    DEFINE_BINARY_OP_WITH_COMPLEX(aten::mul, a* b),
    DEFINE_BOOL_OP(aten::__and__, a&& b),
    DEFINE_BOOL_OP(aten::__or__, a || b),
    DEFINE_BOOL_OP(aten::__xor__, a != b),
    DEFINE_UNARY_OP(aten::round, round_to_even(a), float, float),
    DEFINE_UNARY_OP(aten::floor, floor(a), int, int),
    DEFINE_UNARY_OP(aten::ceil, ceil(a), int, int),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::neg, -a, int, float),
    DEFINE_UNARY_OP_WITH_COMPLEX(aten::exp, std::exp(a), float, float),
    // Pass in two ops for handling int and float separately as % in C++ only
    // works for int The modulus calculation is different between C++ and
    // Python (on negative), we preserve the python behavior as it's more
    // common and match python syntax, hence the conversion.
    DEFINE_GENERIC_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        int,
        float),
    DEFINE_INT_FLOAT_OP(aten::remainder, fmod((b + fmod(a, b)), b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        Scalar),
    // NB: This is the python truediv operation
    DEFINE_GENERIC_OP_WITH_COMPLEX(
        aten::div,
        static_cast<double>(a) / static_cast<double>(b),
        a / b,
        a / b,
        float,
        float,
        complex),
    DEFINE_SCALAR_BINARY_OP(
        aten::div,
        static_cast<double>(a) / static_cast<double>(b),
        a / b,
        float),
    DEFINE_GENERIC_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        int,
        float),
    DEFINE_INT_FLOAT_OP(aten::floordiv, std::floor(a / b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        Scalar),
    // int ** int produces a float, because negative exponents produce float
    // results
    DEFINE_GENERIC_OP_WITH_COMPLEX(
        aten::pow,
        static_cast<double>(powWrapper(a, b)),
        static_cast<double>(powWrapper(a, b)),
        static_cast<c10::complex<double>>(pow(a, b)),
        float,
        float,
        complex),
    DEFINE_INT_FLOAT_OP(
        aten::pow,
        static_cast<double>(powWrapper(a, b)),
        float),
    DEFINE_FLOAT_COMPLEX_OP(aten::pow, pow(a, b), complex),
    DEFINE_SCALAR_BINARY_OP_AVOID_COLLISION(
        aten::pow,
        static_cast<double>(pow(a, b)),
        static_cast<double>(pow(a, b)),
        float),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::pow.int_to_int(int a, int b) -> int"),
        [](Stack& stack) {
          int64_t a = 0, b = 0;
          pop(stack, a, b);
          push(stack, powWrapper(a, b));
        },
        aliasAnalysisFromSchema()),
    // min and max are in prim:: because there is a difference between
    // the python builtin 'min' and 'torch.min'
    DEFINE_BINARY_OP(prim::min, a < b ? a : b),
    DEFINE_BINARY_OP(prim::max, a > b ? a : b),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::type(Device self) -> str"),
        [](Stack& stack) {
          auto d = pop(stack);
          push(
              stack, DeviceTypeName(d.toDevice().type(), /* lower_case=*/true));
        },
        aliasAnalysisFromSchema()),
    // tensor length op (size of 1st dimension)
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.Tensor(Tensor t) -> int"),
        [](Stack& stack) {
          at::Tensor t = pop(stack).toTensor();
          if (t.dim() == 0) {
            TORCH_CHECK(false, "len() of a 0-d tensor");
          }
          push(stack, t.sizes()[0]);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::ord(str string) -> int"),
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          TORCH_CHECK(
              string.size() == 1,
              "String for ord() must be 1 character, found ",
              string.size());
          uint8_t ord = string.at(0);
          push(stack, int64_t(ord));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::lower(str self) -> str"),
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          std::stringstream ss;
          for (char c : string) {
            ss << static_cast<char>(::tolower(c));
          }
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__contains__.int_list(int[] l, int item) -> bool"),
        listContains<int64_t>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__contains__.str_list(str[] l, str item) -> bool"),
        listContains<std::string>,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::len.str(str s) -> int"),
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          push(stack, static_cast<int64_t>(string.size()));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("aten::dict() -> Dict(str, Tensor)"),
        [](Stack& stack) {
          auto dict =
              c10::impl::GenericDict(StringType::get(), TensorType::get());
          push(stack, dict);
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::__getitem__.str(str s, int index) -> str"),
        [](Stack& stack) {
          auto index = pop(stack).toInt();
          auto string = pop(stack).toStringRef();
          auto norm_index = normalizeIndex(index, string.size());
          char c = string.at(norm_index);
          push(stack, std::string(&c, 1));
        },
        aliasAnalysisFromSchema()),
#define CREATE_COPY_OP(other_type, c_type)                               \
  OperatorGeneratorArgs(                                                 \
      TORCH_SELECTIVE_SCHEMA("aten::copy_." #other_type                  \
                             "(Tensor(a!) self, " #other_type            \
                             " other) -> Tensor(a!)"),                   \
      [](Stack& stack) {                                                 \
        at::Tensor t;                                                    \
        c_type other;                                                    \
        pop(stack, t, other);                                            \
        std::move(t) = other; /* NOLINT(bugprone-use-after-move) */      \
        push(stack, std::move(t)); /* NOLINT(bugprone-use-after-move) */ \
      },                                                                 \
      aliasAnalysisFromSchema())

    CREATE_COPY_OP(Tensor, at::Tensor),
    CREATE_COPY_OP(int, int64_t),
    CREATE_COPY_OP(float, double),
#undef CREATE_COPY_OP
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::backward(Tensor self, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()"),
        [](Stack& stack) {
          bool create_graph = pop(stack).toBool();
          auto retain_graph = pop(stack).toOptional<bool>();
          IValue gradient_ivalue = pop(stack);
          at::Tensor gradient = gradient_ivalue.isNone()
              ? at::Tensor()
              : gradient_ivalue.toTensor();
          at::Tensor self = pop(stack).toTensor();
          bool keep_graph = retain_graph ? retain_graph.value() : create_graph;
          self.backward(gradient, keep_graph, create_graph);
        },
        aliasAnalysisConservative()),
    //
    // create a clone of these declarations with a _hacked_twin overload name
    // and nullability scrubbed from TensorList arg types
    // TODO find out why this exists and how to do it without the hack
    //
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor"),
        [](Stack& stack) {
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result = at::index(self, opt_list_indices);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_unsafe_index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor"),
        [](Stack& stack) {
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result = at::_unsafe_index(self, opt_list_indices);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_index_put_impl_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)"),
        [](Stack& stack) {
          auto unsafe = pop(stack).toBool();
          auto accumulate = pop(stack).toBool();
          auto values = pop(stack).toTensor();
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result = at::_index_put_impl_(
              self, opt_list_indices, values, accumulate, unsafe);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index_put_.hacked_twin(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)"),
        [](Stack& stack) {
          auto accumulate = pop(stack).toBool();
          auto values = pop(stack).toTensor();
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result =
              at::index_put_(self, opt_list_indices, values, accumulate);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor"),
        [](Stack& stack) {
          auto accumulate = pop(stack).toBool();
          auto values = pop(stack).toTensor();
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result =
              at::index_put(self, opt_list_indices, values, accumulate);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::_unsafe_index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor"),
        [](Stack& stack) {
          auto accumulate = pop(stack).toBool();
          auto values = pop(stack).toTensor();
          auto indices = pop(stack).to<c10::List<at::Tensor>>();
          c10::List<std::optional<at::Tensor>> opt_list_indices;
          opt_list_indices.reserve(indices.size());
          for (const auto& ten : indices) {
            opt_list_indices.push_back(ten);
          }
          auto self = pop(stack).toTensor();
          auto result =
              at::_unsafe_index_put(self, opt_list_indices, values, accumulate);
          push(stack, std::move(result));
        },
        aliasAnalysisFromSchema()),
    // reference function parse_to_conversion in python_arg_parsing.h
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
        [](Stack& stack) {
          bool non_blocking = false;
          bool copy = false;
          pop(stack, non_blocking, copy);
          std::optional<at::ScalarType> scalarType =
              pop(stack).toOptional<at::ScalarType>();
          std::optional<c10::Device> device =
              pop(stack).toOptional<c10::Device>();
          at::Tensor self = pop(stack).toTensor();
          push(
              stack, to_dispatch(self, device, scalarType, non_blocking, copy));
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA(
            "aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"),
        toPrimDType,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_cuda(Tensor a) -> bool"),
        isCuda,
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_cpu(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_cpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_xla(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_xla());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_mtia(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_mtia());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::is_xpu(Tensor a) -> bool"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.is_xpu());
        },
        aliasAnalysisFromSchema()),
    OperatorGeneratorArgs(
        TORCH_SELECTIVE_SCHEMA("prim::data(Tensor(a) a) -> Tensor(a)"),
        [](Stack& stack) {
          at::Tensor a;
          pop(stack, a);
          push(stack, a.variable_data());
        },
        aliasAnalysisFromSchema()),
// these ops are not defined for Tensor
#define CREATE_COMPARATOR_LIST_OPS_SPECIALIZED(decl_type, value_type)        \
  OperatorGeneratorArgs(                                                     \
      TORCH_SELECTIVE_SCHEMA("prim::min." decl_type "_list(" decl_type       \
                             "[] l, " decl_type "[] r) -> " decl_type "[]"), \
      minList<value_type>,                                                   \
      aliasAnalysisFromSchema()),                                            \
      OperatorGeneratorArgs(                                                 \
          TORCH_SELECTIVE_SCHEMA("prim::max." decl_type "_list(" decl_type   \
                                 "[] l, " decl_type "[] r) -> " decl_type    \
                                 "[]"),                                      \
          maxList<value_type>,                                               \
          aliasAnalysisFromSchema()),                      
```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 85 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `because`, `torch`

**Classes/Structs**: `TORCH_SELECTIVE_SCHEMA`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/autocast_mode.h`
- `ATen/core/Generator.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`
- `torch/csrc/jit/mobile/promoted_prim_ops.h`
- `torch/csrc/jit/runtime/custom_operator.h`
- `torch/csrc/jit/runtime/operator.h`
- `torch/csrc/jit/runtime/register_ops_utils.h`
- `torch/csrc/jit/runtime/slice_indices_adjust.h`
- `torch/library.h`
- `optional`
- `algorithm`
- `bitset`
- `cctype`
- `cmath`
- `iostream`
- `memory`
- `ostream`
- `stdexcept`
- `string`
- `utility`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `register_prim_ops.cpp_docs.md`
- **Keyword Index**: `register_prim_ops.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
