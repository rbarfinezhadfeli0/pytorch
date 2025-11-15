# Documentation: `torch/csrc/jit/runtime/serialized_shape_function_registry.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/serialized_shape_function_registry.cpp`
- **Size**: 99,973 bytes (97.63 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp

/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python
 * torchgen/shape_functions/gen_jit_shape_functions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/serialized_shape_function_registry.h>

// clang-format off

namespace torch::jit {


static std::string shape_funcs = ""
+ std::string(R"=====(
def unary(self: List[int]) -> List[int]:
  out = annotate(List[int], [])
  for _0 in range(torch.len(self)):
    elem = self[_0]
    _1 = torch.append(out, elem)
  return out

def adaptive_avg_pool2d(self: List[int],
    out: List[int]) -> List[int]:
  if torch.eq(torch.len(out), 2):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(self), 3):
    _0 = True
  else:
    _0 = torch.eq(torch.len(self), 4)
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _1 = torch.__range_length(1, torch.len(self), 1)
  for _2 in range(_1):
    i = torch.__derive_index(_2, 1, 1)
    if torch.ne(self[i], 0):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  shape = annotate(List[int], [])
  _3 = torch.__range_length(0, torch.sub(torch.len(self), 2), 1)
  for _4 in range(_3):
    i0 = torch.__derive_index(_4, 0, 1)
    _5 = torch.append(shape, self[i0])
  for _6 in range(torch.len(out)):
    elem = out[_6]
    _7 = torch.append(shape, elem)
  return shape

def zero_dim_tensor(input: Any) -> List[int]:
  return annotate(List[int], [])

def arange_end(end: Union[float, int],
    inp0: Any,
    inp1: Any,
    inp2: Any,
    inp3: Any) -> List[int]:
  if torch.ge(end, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return [int(torch.ceil(end))]

def arange_start(start: Union[float, int],
    end: Union[float, int],
    inp0: Any,
    inp1: Any,
    inp2: Any,
    inp3: Any) -> List[int]:
  if torch.ge(end, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(end, start):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _0 = int(torch.ceil(torch.sub(end, start)))
  return [_0]

)=====")
+ std::string(R"=====(def arange_start_step(start: Union[float, int],
    end: Union[float, int],
    step: Union[float, int],
    inp0: Any,
    inp1: Any,
    inp2: Any,
    inp3: Any) -> List[int]:
  if torch.ne(step, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(step, 0):
    if torch.ge(start, end):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  else:
    if torch.ge(end, start):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  _0 = torch.div(torch.sub(end, start), step)
  return [torch.ceil(_0)]

def squeeze_nodim(li: List[int]) -> List[int]:
  out = annotate(List[int], [])
  for i in range(torch.len(li)):
    if torch.ne(li[i], 1):
      _0 = torch.append(out, li[i])
    else:
      pass
  return out

def squeeze(li: List[int],
    dim: int) -> List[int]:
  out = annotate(List[int], [])
  _0 = torch.len(li)
  if torch.le(_0, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = _0
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _1 = True
  else:
    _1 = torch.gt(dim, max)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    wrapped_dim = torch.add(dim, dim_post_expr)
  else:
    wrapped_dim = dim
  for i in range(torch.len(li)):
    if torch.eq(i, wrapped_dim):
      if torch.ne(li[i], 1):
        _2 = torch.append(out, li[i])
      else:
        pass
    else:
      _3 = torch.append(out, li[i])
  return out

)=====")
+ std::string(R"=====(def squeeze_dims(li: List[int],
    dims: List[int]) -> List[int]:
  if torch.eq(torch.len(dims), 0):
    _0 = li
  else:
    wrapped_dims = annotate(List[int], [])
    for _1 in range(torch.len(dims)):
      elem = dims[_1]
      _2 = torch.append(wrapped_dims, elem)
    for i in range(torch.len(dims)):
      _3 = wrapped_dims[i]
      _4 = torch.len(li)
      if torch.le(_4, 0):
        dim_post_expr = 1
      else:
        dim_post_expr = _4
      min = torch.neg(dim_post_expr)
      max = torch.sub(dim_post_expr, 1)
      if torch.lt(_3, min):
        _5 = True
      else:
        _5 = torch.gt(_3, max)
      if torch.__not__(_5):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.lt(_3, 0):
        dim = torch.add(_3, dim_post_expr)
      else:
        dim = _3
      _6 = torch._set_item(wrapped_dims, i, dim)
    result = annotate(List[int], [])
    for i0 in range(torch.len(li)):
      if torch.eq(li[i0], 1):
        _7 = torch.__contains__(wrapped_dims, i0)
        if torch.__not__(_7):
          _8 = torch.append(result, li[i0])
        else:
          pass
      else:
        _9 = torch.append(result, li[i0])
    _0 = result
  return _0

def unsqueeze(li: List[int],
    dim: int) -> List[int]:
  _0 = torch.add(torch.len(li), 1)
  if torch.le(_0, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = _0
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _1 = True
  else:
    _1 = torch.gt(dim, max)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    dim0 = torch.add(dim, dim_post_expr)
  else:
    dim0 = dim
  out = annotate(List[int], [])
  for _2 in range(torch.len(li)):
    elem = li[_2]
    _3 = torch.append(out, elem)
  torch.insert(out, dim0, 1)
  return out

)=====")
+ std::string(R"=====(def slice(self: List[int],
    dim: int,
    start: Optional[int],
    end: Optional[int],
    step: int) -> List[int]:
  ndim = torch.len(self)
  if torch.ne(ndim, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.le(ndim, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = ndim
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _0 = True
  else:
    _0 = torch.gt(dim, max)
  if torch.__not__(_0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    dim0 = torch.add(dim, dim_post_expr)
  else:
    dim0 = dim
  if torch.__isnot__(start, None):
    start_val = unchecked_cast(int, start)
  else:
    start_val = 0
  if torch.__isnot__(end, None):
    end_val = unchecked_cast(int, end)
  else:
    end_val = 9223372036854775807
  if torch.gt(step, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _1 = torch.eq(start_val, 9223372036854775807)
  if _1:
    start_val0 = 0
  else:
    start_val0 = start_val
  if torch.lt(start_val0, 0):
    start_val1 = torch.add(start_val0, self[dim0])
  else:
    start_val1 = start_val0
  if torch.lt(end_val, 0):
    end_val0 = torch.add(end_val, self[dim0])
  else:
    end_val0 = end_val
  if torch.lt(start_val1, 0):
    start_val2 = 0
  else:
    if torch.gt(start_val1, self[dim0]):
      start_val3 = self[dim0]
    else:
      start_val3 = start_val1
    start_val2 = start_val3
  if torch.lt(end_val0, start_val2):
    end_val1 = start_val2
  else:
    if torch.ge(end_val0, self[dim0]):
      end_val2 = self[dim0]
    else:
      end_val2 = end_val0
    end_val1 = end_val2
  slice_len = torch.sub(end_val1, start_val2)
  out = annotate(List[int], [])
  for _2 in range(torch.len(self)):
    elem = self[_2]
    _3 = torch.append(out, elem)
  _4 = torch.sub(torch.add(slice_len, step), 1)
  _5 = torch._set_item(out, dim0, torch.floordiv(_4, step))
  return out

)=====")
+ std::string(R"=====(def select(self: List[int],
    dim: int,
    index: int) -> List[int]:
  ndim = torch.len(self)
  if torch.ne(ndim, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.le(ndim, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = ndim
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _0 = True
  else:
    _0 = torch.gt(dim, max)
  if torch.__not__(_0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    dim0 = torch.add(dim, dim_post_expr)
  else:
    dim0 = dim
  size = self[dim0]
  if torch.lt(index, torch.neg(size)):
    _1 = True
  else:
    _1 = torch.ge(index, size)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  out = annotate(List[int], [])
  for i in range(ndim):
    if torch.ne(i, dim0):
      _2 = torch.append(out, self[i])
    else:
      pass
  return out

)=====")
+ std::string(R"=====(def index_select(self: List[int],
    dim: int,
    index: List[int]) -> List[int]:
  _0 = torch.len(self)
  if torch.le(_0, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = _0
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _1 = True
  else:
    _1 = torch.gt(dim, max)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    dim0 = torch.add(dim, dim_post_expr)
  else:
    dim0 = dim
  numel = 1
  for _2 in range(torch.len(index)):
    elem = index[_2]
    numel = torch.mul(numel, elem)
  if torch.le(torch.len(index), 1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(dim0, 0):
    _3 = True
  else:
    _3 = torch.lt(dim0, torch.len(self))
  if _3:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  result_size = annotate(List[int], [])
  for i in range(torch.len(self)):
    if torch.eq(dim0, i):
      _4 = torch.append(result_size, numel)
    else:
      _5 = torch.append(result_size, self[i])
  return result_size

)=====")
+ std::string(R"=====(def embedding(weight: List[int],
    indices: List[int],
    padding_idx: int=-1,
    scale_grad_by_freq: bool=False,
    sparse: bool=False) -> List[int]:
  if torch.eq(torch.len(weight), 2):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(indices), 1):
    _1 = torch.len(weight)
    if torch.le(_1, 0):
      dim_post_expr = 1
    else:
      dim_post_expr = _1
    min = torch.neg(dim_post_expr)
    max = torch.sub(dim_post_expr, 1)
    if torch.lt(0, min):
      _2 = True
    else:
      _2 = torch.gt(0, max)
    if torch.__not__(_2):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    numel = 1
    for _3 in range(torch.len(indices)):
      elem = indices[_3]
      numel = torch.mul(numel, elem)
    if torch.le(torch.len(indices), 1):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    result_size = annotate(List[int], [])
    for i in range(torch.len(weight)):
      if torch.eq(0, i):
        _4 = torch.append(result_size, numel)
      else:
        _5 = torch.append(result_size, weight[i])
    _0 = result_size
  else:
    size = annotate(List[int], [])
    for _6 in range(torch.len(indices)):
      elem0 = indices[_6]
      _7 = torch.append(size, elem0)
    _8 = torch.append(size, weight[1])
    _0 = size
  return _0

def mm(self: List[int],
    mat2: List[int]) -> List[int]:
  _0 = "AssertionError: self must be a matrix"
  _1 = "AssertionError: mat2 must be a matrix"
  if torch.eq(torch.len(self), 2):
    pass
  else:
    ops.prim.RaiseException(_0)
  if torch.eq(torch.len(mat2), 2):
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(self[1], mat2[0]):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return [self[0], mat2[1]]

)=====")
+ std::string(R"=====(def dot(self: List[int],
    tensor: List[int]) -> List[int]:
  if torch.eq(torch.len(self), 1):
    _0 = torch.eq(torch.len(tensor), 1)
  else:
    _0 = False
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(self[0], tensor[0]):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return annotate(List[int], [])

def mv(self: List[int],
    vec: List[int]) -> List[int]:
  if torch.eq(torch.len(self), 2):
    _0 = torch.eq(torch.len(vec), 1)
  else:
    _0 = False
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(self[1], vec[0]):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return [self[0]]

)=====")
+ std::string(R"=====(def matmul(tensor1: List[int],
    tensor2: List[int]) -> List[int]:
  _0 = "AssertionError: self must be a matrix"
  _1 = "AssertionError: mat2 must be a matrix"
  _2 = "AssertionError: self must be a matrix"
  _3 = "AssertionError: mat2 must be a matrix"
  _4 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  _5 = "AssertionError: both  arguments to matmul need to be at least 1D"
  _6 = uninitialized(List[int])
  dim_tensor1 = torch.len(tensor1)
  dim_tensor2 = torch.len(tensor2)
  if torch.eq(dim_tensor1, 1):
    _7 = torch.eq(dim_tensor2, 1)
  else:
    _7 = False
  if _7:
    if torch.eq(torch.len(tensor1), 1):
      _9 = torch.eq(torch.len(tensor2), 1)
    else:
      _9 = False
    if _9:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if torch.eq(tensor1[0], tensor2[0]):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    _8 = annotate(List[int], [])
  else:
    if torch.eq(dim_tensor1, 2):
      _10 = torch.eq(dim_tensor2, 1)
    else:
      _10 = False
    if _10:
      if torch.eq(torch.len(tensor1), 2):
        _12 = torch.eq(torch.len(tensor2), 1)
      else:
        _12 = False
      if _12:
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.eq(tensor1[1], tensor2[0]):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      _11 = [tensor1[0]]
    else:
      if torch.eq(dim_tensor1, 1):
        _13 = torch.eq(dim_tensor2, 2)
      else:
        _13 = False
      if _13:
        _15 = torch.add(torch.len(tensor1), 1)
        if torch.le(_15, 0):
          dim_post_expr = 1
        else:
          dim_post_expr = _15
        min = torch.neg(dim_post_expr)
        max = torch.sub(dim_post_expr, 1)
        if torch.lt(0, min):
          _16 = True
        else:
          _16 = torch.gt(0, max)
        if torch.__not__(_16):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        out = annotate(List[int], [])
        for _17 in range(torch.len(tensor1)):
          elem = tensor1[_17]
          _18 = torch.append(out, elem)
        torch.insert(out, 0, 1)
        if torch.eq(torch.len(out), 2):
          pass
        else:
          ops.prim.RaiseException(_0)
        if torch.eq(torch.len(tensor2), 2):
          pass
        else:
          ops.prim.RaiseException(_1)
        if torch.eq(out[1], tensor2[0]):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        _19 = [out[0], tensor2[1]]
        out0 = annotate(List[int], [])
        for i in range(2):
          if torch.eq(i, 0):
            if torch.ne(_19[i], 1):
              _20 = torch.append(out0, _19[i])
            else:
              pass
          else:
            _21 = torch.append(out0, _19[i])
        _14 = out0
      else:
        if torch.eq(dim_tensor1, 2):
          _22 = torch.eq(dim_tensor2, 2)
        else:
          _22 = False
        if _22:
          _24 = torch.eq(torch.len(tensor1), 2)
          if _24:
            pass
          else:
            ops.prim.RaiseException(_2)
          _25 = torch.eq(torch.len(tensor2), 2)
          if _25:
            pass
          else:
            ops.prim.RaiseException(_3)
          _26 = torch.eq(tensor1[1], tensor2[0])
          if _26:
            pass
          else:
            ops.prim.RaiseException("AssertionError: ")
          _23 = [tensor1[0], tensor2[1]]
        else:
          if torch.ge(dim_tensor1, 1):
            _27 = torch.ge(dim_tensor2, 1)
          else:
            _27 = False
          if _27:
            if torch.gt(dim_tensor1, 1):
              n = tensor1[-2]
            else:
              n = 1
            batch_tensor1 = annotate(List[int], [])
            for i0 in range(torch.sub(dim_tensor1, 2)):
              _29 = torch.append(batch_tensor1, tensor1[i0])
            p = tensor2[-1]
            batch_tensor2 = annotate(List[int], [])
            for i1 in range(torch.sub(dim_tensor2, 2)):
              _30 = torch.append(batch_tensor2, tensor2[i1])
            dimsA = torch.len(batch_tensor1)
            dimsB = torch.len(batch_tensor2)
            ndim = ops.prim.max(dimsA, dimsB)
            expand_batch_portion = annotate(List[int], [])
            for i2 in range(ndim):
              offset = torch.sub(torch.sub(ndim, 1), i2)
              dimA = torch.sub(torch.sub(dimsA, 1), offset)
              dimB = torch.sub(torch.sub(dimsB, 1), offset)
              if torch.ge(dimA, 0):
                sizeA = batch_tensor1[dimA]
              else:
                sizeA = 1
              if torch.ge(dimB, 0):
                sizeB = batch_tensor2[dimB]
              else:
                sizeB = 1
              if torch.ne(sizeA, sizeB):
                _31 = torch.ne(sizeA, 1)
              else:
                _31 = False
              if _31:
                _32 = torch.ne(sizeB, 1)
              else:
                _32 = False
              if _32:
                _33 = torch.format(_4, sizeA, sizeB, i2)
                _34 = torch.add("AssertionError: ", _33)
                ops.prim.RaiseException(_34)
              else:
                pass
              if torch.eq(sizeA, 1):
                _35 = sizeB
              else:
                _35 = sizeA
              _36 = torch.append(expand_batch_portion, _35)
            if torch.gt(dim_tensor1, 1):
              _37 = torch.append(expand_batch_portion, n)
            else:
              pass
            if torch.gt(dim_tensor2, 1):
              _38 = torch.append(expand_batch_portion, p)
            else:
              pass
            _28 = expand_batch_portion
          else:
            ops.prim.RaiseException(_5)
            _28 = _6
          _23 = _28
        _14 = _23
      _11 = _14
    _8 = _11
  return _8

)=====")
+ std::string(R"=====(def linear(input: List[int],
    weight: List[int],
    bias: Optional[List[int]]) -> List[int]:
  _0 = "AssertionError: self must be a matrix"
  _1 = "AssertionError: mat2 must be a matrix"
  _2 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  _3 = "AssertionError: both  arguments to matmul need to be at least 1D"
  _4 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  if torch.le(torch.len(weight), 2):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  self_len = torch.len(weight)
  if torch.eq(self_len, 0):
    _5 = annotate(List[int], [])
  else:
    if torch.eq(self_len, 1):
      _6 = [weight[0]]
    else:
      _6 = [weight[1], weight[0]]
    _5 = _6
  _7 = uninitialized(List[int])
  dim_tensor1 = torch.len(input)
  dim_tensor2 = torch.len(_5)
  if torch.eq(dim_tensor1, 1):
    _8 = torch.eq(dim_tensor2, 1)
  else:
    _8 = False
  if _8:
    if torch.eq(torch.len(input), 1):
      _9 = torch.eq(torch.len(_5), 1)
    else:
      _9 = False
    if _9:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if torch.eq(input[0], _5[0]):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    out = annotate(List[int], [])
  else:
    if torch.eq(dim_tensor1, 2):
      _10 = torch.eq(dim_tensor2, 1)
    else:
      _10 = False
    if _10:
      if torch.eq(torch.len(input), 2):
        _12 = torch.eq(torch.len(_5), 1)
      else:
        _12 = False
      if _12:
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.eq(input[1], _5[0]):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      _11 = [input[0]]
    else:
      if torch.eq(dim_tensor1, 1):
        _13 = torch.eq(dim_tensor2, 2)
      else:
        _13 = False
      if _13:
        _15 = torch.add(torch.len(input), 1)
        if torch.le(_15, 0):
          dim_post_expr = 1
        else:
          dim_post_expr = _15
        min = torch.neg(dim_post_expr)
        max = torch.sub(dim_post_expr, 1)
        if torch.lt(0, min):
          _16 = True
        else:
          _16 = torch.gt(0, max)
        if torch.__not__(_16):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        out0 = annotate(List[int], [])
        for _17 in range(torch.len(input)):
          elem = input[_17]
          _18 = torch.append(out0, elem)
        torch.insert(out0, 0, 1)
        if torch.eq(torch.len(out0), 2):
          pass
        else:
          ops.prim.RaiseException(_0)
        if torch.eq(torch.len(_5), 2):
          pass
        else:
          ops.prim.RaiseException(_1)
        if torch.eq(out0[1], _5[0]):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        _19 = [out0[0], _5[1]]
        out1 = annotate(List[int], [])
        for i in range(2):
          if torch.eq(i, 0):
            if torch.ne(_19[i], 1):
              _20 = torch.append(out1, _19[i])
            else:
              pass
          else:
            _21 = torch.append(out1, _19[i])
        _14 = out1
      else:
        if torch.eq(dim_tensor1, 2):
          _22 = torch.eq(dim_tensor2, 2)
        else:
          _22 = False
        if _22:
          if torch.eq(torch.len(input), 2):
            pass
          else:
            ops.prim.RaiseException(_0)
          if torch.eq(torch.len(_5), 2):
            pass
          else:
            ops.prim.RaiseException(_1)
          if torch.eq(input[1], _5[0]):
            pass
          else:
            ops.prim.RaiseException("AssertionError: ")
          _23 = [input[0], _5[1]]
        else:
          if torch.ge(dim_tensor1, 1):
            _24 = torch.ge(dim_tensor2, 1)
          else:
            _24 = False
          if _24:
            if torch.gt(dim_tensor1, 1):
              n = input[-2]
            else:
              n = 1
            batch_tensor1 = annotate(List[int], [])
            for i0 in range(torch.sub(dim_tensor1, 2)):
              _26 = torch.append(batch_tensor1, input[i0])
            p = _5[-1]
            batch_tensor2 = annotate(List[int], [])
            for i1 in range(torch.sub(dim_tensor2, 2)):
              _27 = torch.append(batch_tensor2, _5[i1])
            dimsA = torch.len(batch_tensor1)
            dimsB = torch.len(batch_tensor2)
            ndim = ops.prim.max(dimsA, dimsB)
            expand_batch_portion = annotate(List[int], [])
            for i2 in range(ndim):
              offset = torch.sub(torch.sub(ndim, 1), i2)
              dimA = torch.sub(torch.sub(dimsA, 1), offset)
              dimB = torch.sub(torch.sub(dimsB, 1), offset)
              if torch.ge(dimA, 0):
                sizeA = batch_tensor1[dimA]
              else:
                sizeA = 1
              if torch.ge(dimB, 0):
                sizeB = batch_tensor2[dimB]
              else:
                sizeB = 1
              if torch.ne(sizeA, sizeB):
                _28 = torch.ne(sizeA, 1)
              else:
                _28 = False
              if _28:
                _29 = torch.ne(sizeB, 1)
              else:
                _29 = False
              if _29:
                _30 = torch.format(_2, sizeA, sizeB, i2)
                _31 = torch.add("AssertionError: ", _30)
                ops.prim.RaiseException(_31)
              else:
                pass
              if torch.eq(sizeA, 1):
                _32 = sizeB
              else:
                _32 = sizeA
              _33 = torch.append(expand_batch_portion, _32)
            if torch.gt(dim_tensor1, 1):
              _34 = torch.append(expand_batch_portion, n)
            else:
              pass
            if torch.gt(dim_tensor2, 1):
              _35 = torch.append(expand_batch_portion, p)
            else:
              pass
            _25 = expand_batch_portion
          else:
            ops.prim.RaiseException(_3)
            _25 = _7
          _23 = _25
        _14 = _23
      _11 = _14
    out = _11
  if torch.__isnot__(bias, None):
    bias0 = unchecked_cast(List[int], bias)
    dimsA0 = torch.len(bias0)
    dimsB0 = torch.len(out)
    ndim0 = ops.prim.max(dimsA0, dimsB0)
    expandedSizes = annotate(List[int], [])
    for i3 in range(ndim0):
      offset0 = torch.sub(torch.sub(ndim0, 1), i3)
      dimA0 = torch.sub(torch.sub(dimsA0, 1), offset0)
      dimB0 = torch.sub(torch.sub(dimsB0, 1), offset0)
      if torch.ge(dimA0, 0):
        sizeA0 = bias0[dimA0]
      else:
        sizeA0 = 1
      if torch.ge(dimB0, 0):
        sizeB0 = out[dimB0]
      else:
        sizeB0 = 1
      if torch.ne(sizeA0, sizeB0):
        _36 = torch.ne(sizeA0, 1)
      else:
        _36 = False
      if _36:
        _37 = torch.ne(sizeB0, 1)
      else:
        _37 = False
      if _37:
        _38 = torch.format(_4, sizeA0, sizeB0, i3)
        _39 = torch.add("AssertionError: ", _38)
        ops.prim.RaiseException(_39)
      else:
        pass
      if torch.eq(sizeA0, 1):
        _40 = sizeB0
      else:
        _40 = sizeA0
      _41 = torch.append(expandedSizes, _40)
    if torch.eq(expandedSizes, out):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  else:
    pass
  return out

)=====")
+ std::string(R"=====(def max_pool2d(input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool) -> List[int]:
  _0 = "AssertionError: max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
  _1 = "AssertionError: max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
  _2 = "AssertionError: max_pool2d: padding must either be a single int, or a tuple of two ints"
  _3 = "AssertionError: max_pool2d: dilation must be either a single int, or a tuple of two ints"
  _4 = "AssertionError: stride should not be zeero"
  _5 = "AssertionError: stride should not be zeero"
  if torch.eq(torch.len(kernel_size), 1):
    _6 = True
  else:
    _6 = torch.eq(torch.len(kernel_size), 2)
  if _6:
    pass
  else:
    ops.prim.RaiseException(_0)
  kH = kernel_size[0]
  if torch.eq(torch.len(kernel_size), 1):
    kW = kH
  else:
    kW = kernel_size[1]
  if torch.eq(torch.len(stride), 0):
    _7 = True
  else:
    _7 = torch.eq(torch.len(stride), 1)
  if _7:
    _8 = True
  else:
    _8 = torch.eq(torch.len(stride), 2)
  if _8:
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(torch.len(stride), 0):
    dH = kH
  else:
    dH = stride[0]
  if torch.eq(torch.len(stride), 0):
    dW = kW
  else:
    if torch.eq(torch.len(stride), 1):
      dW0 = dH
    else:
      dW0 = stride[1]
    dW = dW0
  if torch.eq(torch.len(padding), 1):
    _9 = True
  else:
    _9 = torch.eq(torch.len(padding), 2)
  if _9:
    pass
  else:
    ops.prim.RaiseException(_2)
  padH = padding[0]
  if torch.eq(torch.len(padding), 1):
    padW = padH
  else:
    padW = padding[1]
  if torch.eq(torch.len(dilation), 1):
    _10 = True
  else:
    _10 = torch.eq(torch.len(dilation), 2)
  if _10:
    pass
  else:
    ops.prim.RaiseException(_3)
  dilationH = dilation[0]
  if torch.eq(torch.len(dilation), 1):
    dilationW = dilationH
  else:
    dilationW = dilation[1]
  if torch.eq(torch.len(input), 3):
    _11 = True
  else:
    _11 = torch.eq(torch.len(input), 4)
  if _11:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 4):
    nbatch = input[-4]
  else:
    nbatch = 1
  nInputPlane = input[-3]
  inputHeight = input[-2]
  inputWidth = input[-1]
  if torch.ne(dH, 0):
    pass
  else:
    ops.prim.RaiseException(_4)
  _12 = torch.add(torch.add(inputHeight, padH), padH)
  _13 = torch.mul(dilationH, torch.sub(kH, 1))
  _14 = torch.sub(torch.sub(_12, _13), 1)
  if ceil_mode:
    _15 = torch.sub(dH, 1)
  else:
    _15 = 0
  _16 = torch.floordiv(torch.add(_14, _15), dH)
  outputSize = torch.add(_16, 1)
  if ceil_mode:
    _17 = torch.ge(torch.mul(_16, dH), torch.add(inputHeight, padH))
    if _17:
      outputSize0 = _16
    else:
      outputSize0 = outputSize
    outputHeight = outputSize0
  else:
    outputHeight = outputSize
  if torch.ne(dW, 0):
    pass
  else:
    ops.prim.RaiseException(_5)
  _18 = torch.add(torch.add(inputWidth, padW), padW)
  _19 = torch.mul(dilationW, torch.sub(kW, 1))
  _20 = torch.sub(torch.sub(_18, _19), 1)
  if ceil_mode:
    _21 = torch.sub(dW, 1)
  else:
    _21 = 0
  _22 = torch.floordiv(torch.add(_20, _21), dW)
  outputSize1 = torch.add(_22, 1)
  if ceil_mode:
    _23 = torch.ge(torch.mul(_22, dW), torch.add(inputWidth, padW))
    if _23:
      outputSize2 = _22
    else:
      outputSize2 = outputSize1
    outputWidth = outputSize2
  else:
    outputWidth = outputSize1
  ndim = torch.len(input)
  if torch.gt(kW, 0):
    _24 = torch.gt(kH, 0)
  else:
    _24 = False
  if _24:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.gt(dW, 0):
    _25 = torch.gt(dH, 0)
  else:
    _25 = False
  if _25:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.gt(dilationH, 0):
    _26 = torch.gt(dilationW, 0)
  else:
    _26 = False
  if _26:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ne(input[1], 0):
    valid_dims = torch.ne(input[2], 0)
  else:
    valid_dims = False
  if torch.eq(ndim, 3):
    _27 = torch.ne(input[0], 0)
  else:
    _27 = False
  if _27:
    _28 = valid_dims
  else:
    _28 = False
  if _28:
    _29 = True
  else:
    if torch.eq(ndim, 4):
      _30 = valid_dims
    else:
      _30 = False
    if _30:
      _31 = torch.ne(input[3], 0)
    else:
      _31 = False
    _29 = _31
  if _29:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(torch.floordiv(kW, 2), padW):
    _33 = torch.ge(torch.floordiv(kH, 2), padH)
    _32 = _33
  else:
    _32 = False
  if _32:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(outputWidth, 1):
    _34 = torch.ge(outputHeight, 1)
  else:
    _34 = False
  if _34:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 3):
    _36 = [nInputPlane, outputHeight, outputWidth]
    _35 = _36
  else:
    _37 = [nbatch, nInputPlane, outputHeight, outputWidth]
    _35 = _37
  return _35

)=====")
+ std::string(R"=====(def max_pool2d_with_indices(input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool) -> Tuple[List[int], List[int]]:
  _0 = "AssertionError: max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
  _1 = "AssertionError: max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
  _2 = "AssertionError: max_pool2d: padding must either be a single int, or a tuple of two ints"
  _3 = "AssertionError: max_pool2d: dilation must be either a single int, or a tuple of two ints"
  _4 = "AssertionError: stride should not be zeero"
  if torch.eq(torch.len(kernel_size), 1):
    _5 = True
  else:
    _5 = torch.eq(torch.len(kernel_size), 2)
  if _5:
    pass
  else:
    ops.prim.RaiseException(_0)
  kH = kernel_size[0]
  if torch.eq(torch.len(kernel_size), 1):
    kW = kH
  else:
    kW = kernel_size[1]
  if torch.eq(torch.len(stride), 0):
    _6 = True
  else:
    _6 = torch.eq(torch.len(stride), 1)
  if _6:
    _7 = True
  else:
    _7 = torch.eq(torch.len(stride), 2)
  if _7:
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(torch.len(stride), 0):
    dH = kH
  else:
    dH = stride[0]
  if torch.eq(torch.len(stride), 0):
    dW = kW
  else:
    if torch.eq(torch.len(stride), 1):
      dW0 = dH
    else:
      dW0 = stride[1]
    dW = dW0
  if torch.eq(torch.len(padding), 1):
    _8 = True
  else:
    _8 = torch.eq(torch.len(padding), 2)
  if _8:
    pass
  else:
    ops.prim.RaiseException(_2)
  padH = padding[0]
  if torch.eq(torch.len(padding), 1):
    padW = padH
  else:
    padW = padding[1]
  if torch.eq(torch.len(dilation), 1):
    _9 = True
  else:
    _9 = torch.eq(torch.len(dilation), 2)
  if _9:
    pass
  else:
    ops.prim.RaiseException(_3)
  dilationH = dilation[0]
  if torch.eq(torch.len(dilation), 1):
    dilationW = dilationH
  else:
    dilationW = dilation[1]
  if torch.eq(torch.len(input), 3):
    _10 = True
  else:
    _10 = torch.eq(torch.len(input), 4)
  if _10:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 4):
    nbatch = input[-4]
  else:
    nbatch = 1
  nInputPlane = input[-3]
  inputHeight = input[-2]
  inputWidth = input[-1]
  if torch.ne(dH, 0):
    pass
  else:
    ops.prim.RaiseException(_4)
  _11 = torch.add(torch.add(inputHeight, padH), padH)
  _12 = torch.mul(dilationH, torch.sub(kH, 1))
  _13 = torch.sub(torch.sub(_11, _12), 1)
  if ceil_mode:
    _14 = torch.sub(dH, 1)
  else:
    _14 = 0
  _15 = torch.floordiv(torch.add(_13, _14), dH)
  outputSize = torch.add(_15, 1)
  if ceil_mode:
    _16 = torch.ge(torch.mul(_15, dH), torch.add(inputHeight, padH))
    if _16:
      outputSize0 = _15
    else:
      outputSize0 = outputSize
    outputHeight = outputSize0
  else:
    outputHeight = outputSize
  if torch.ne(dW, 0):
    pass
  else:
    ops.prim.RaiseException(_4)
  _17 = torch.add(torch.add(inputWidth, padW), padW)
  _18 = torch.mul(dilationW, torch.sub(kW, 1))
  _19 = torch.sub(torch.sub(_17, _18), 1)
  if ceil_mode:
    _20 = torch.sub(dW, 1)
  else:
    _20 = 0
  _21 = torch.floordiv(torch.add(_19, _20), dW)
  outputSize1 = torch.add(_21, 1)
  if ceil_mode:
    _22 = torch.ge(torch.mul(_21, dW), torch.add(inputWidth, padW))
    if _22:
      outputSize2 = _21
    else:
      outputSize2 = outputSize1
    outputWidth = outputSize2
  else:
    outputWidth = outputSize1
  ndim = torch.len(input)
  if torch.gt(kW, 0):
    _23 = torch.gt(kH, 0)
  else:
    _23 = False
  if _23:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.gt(dW, 0):
    _24 = torch.gt(dH, 0)
  else:
    _24 = False
  if _24:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.gt(dilationH, 0):
    _25 = torch.gt(dilationW, 0)
  else:
    _25 = False
  if _25:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ne(input[1], 0):
    valid_dims = torch.ne(input[2], 0)
  else:
    valid_dims = False
  if torch.eq(ndim, 3):
    _26 = torch.ne(input[0], 0)
  else:
    _26 = False
  if _26:
    _27 = valid_dims
  else:
    _27 = False
  if _27:
    _28 = True
  else:
    if torch.eq(ndim, 4):
      _29 = valid_dims
    else:
      _29 = False
    if _29:
      _30 = torch.ne(input[3], 0)
    else:
      _30 = False
    _28 = _30
  if _28:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(torch.floordiv(kW, 2), padW):
    _32 = torch.ge(torch.floordiv(kH, 2), padH)
    _31 = _32
  else:
    _31 = False
  if _31:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(outputWidth, 1):
    _33 = torch.ge(outputHeight, 1)
  else:
    _33 = False
  if _33:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 3):
    _34 = [nInputPlane, outputHeight, outputWidth]
    out = _34
  else:
    _35 = [nbatch, nInputPlane, outputHeight, outputWidth]
    out = _35
  return (out, out)

)=====")
+ std::string(R"=====(def t(self: List[int]) -> List[int]:
  if torch.le(torch.len(self), 2):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  self_len = torch.len(self)
  if torch.eq(self_len, 0):
    _0 = annotate(List[int], [])
  else:
    if torch.eq(self_len, 1):
      _1 = [self[0]]
    else:
      _1 = [self[1], self[0]]
    _0 = _1
  return _0

def transpose(self: List[int],
    dim0: int,
    dim1: int) -> List[int]:
  ndims = torch.len(self)
  if torch.le(ndims, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = ndims
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim0, min):
    _0 = True
  else:
    _0 = torch.gt(dim0, max)
  if torch.__not__(_0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim0, 0):
    dim00 = torch.add(dim0, dim_post_expr)
  else:
    dim00 = dim0
  if torch.le(ndims, 0):
    dim_post_expr0 = 1
  else:
    dim_post_expr0 = ndims
  min0 = torch.neg(dim_post_expr0)
  max0 = torch.sub(dim_post_expr0, 1)
  if torch.lt(dim1, min0):
    _1 = True
  else:
    _1 = torch.gt(dim1, max0)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim1, 0):
    dim10 = torch.add(dim1, dim_post_expr0)
  else:
    dim10 = dim1
  if torch.eq(dim00, dim10):
    out = annotate(List[int], [])
    for _3 in range(torch.len(self)):
      elem = self[_3]
      _4 = torch.append(out, elem)
    _2 = out
  else:
    out0 = annotate(List[int], [])
    for i in range(ndims):
      if torch.eq(i, dim00):
        _5 = torch.append(out0, self[dim10])
      else:
        if torch.eq(i, dim10):
          _6 = torch.append(out0, self[dim00])
        else:
          _7 = torch.append(out0, self[i])
    _2 = out0
  return _2

)=====")
+ std::string(R"=====(def conv1d(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int) -> List[int]:
  if torch.eq(torch.len(weight), 3):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 3):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  k = torch.len(input)
  weight_dim = torch.len(weight)
  non_negative = False
  for _0 in range(torch.len(padding)):
    val = padding[_0]
    if torch.lt(val, 0):
      non_negative0 = True
    else:
      non_negative0 = non_negative
    non_negative = non_negative0
  if torch.__not__(non_negative):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  non_negative1 = False
  for _1 in range(torch.len(stride)):
    val0 = stride[_1]
    if torch.lt(val0, 0):
      non_negative2 = True
    else:
      non_negative2 = non_negative1
    non_negative1 = non_negative2
  if torch.__not__(non_negative1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(weight_dim, k):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(weight[0], groups):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _2 = torch.eq(torch.remainder(weight[0], groups), 0)
  if _2:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _3 = torch.eq(input[1], torch.mul(weight[1], groups))
  if _3:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.__is__(bias, None):
    _4 = True
  else:
    bias0 = unchecked_cast(List[int], bias)
    if torch.eq(torch.len(bias0), 1):
      _5 = torch.eq(bias0[0], weight[0])
    else:
      _5 = False
    _4 = _5
  if _4:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  for _6 in range(torch.__range_length(2, k, 1)):
    i = torch.__derive_index(_6, 2, 1)
    _7 = input[i]
    _8 = torch.mul(padding[torch.sub(i, 2)], 2)
    _9 = torch.add(_7, _8)
    _10 = torch.mul(dilation[torch.sub(i, 2)], torch.sub(weight[i], 1))
    if torch.ge(_9, torch.add(_10, 1)):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  has_dilation = torch.gt(torch.len(dilation), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  _11 = torch.append(output_size, input[0])
  _12 = torch.append(output_size, weight[0])
  for _13 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_13, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    _14 = torch.mul(dilation_, torch.sub(weight[d], 1))
    kernel = torch.add(_14, 1)
    _15 = input[d]
    _16 = torch.mul(padding[torch.sub(d, 2)], 2)
    _17 = torch.sub(torch.add(_15, _16), kernel)
    _18 = torch.floordiv(_17, stride[torch.sub(d, 2)])
    _19 = torch.append(output_size, torch.add(_18, 1))
  return output_size

)=====")
+ std::string(R"=====(def conv2d(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int) -> List[int]:
  if torch.eq(torch.len(weight), 4):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 4):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  k = torch.len(input)
  weight_dim = torch.len(weight)
  non_negative = False
  for _0 in range(torch.len(padding)):
    val = padding[_0]
    if torch.lt(val, 0):
      non_negative0 = True
    else:
      non_negative0 = non_negative
    non_negative = non_negative0
  if torch.__not__(non_negative):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  non_negative1 = False
  for _1 in range(torch.len(stride)):
    val0 = stride[_1]
    if torch.lt(val0, 0):
      non_negative2 = True
    else:
      non_negative2 = non_negative1
    non_negative1 = non_negative2
  if torch.__not__(non_negative1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(weight_dim, k):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(weight[0], groups):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _2 = torch.eq(torch.remainder(weight[0], groups), 0)
  if _2:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _3 = torch.eq(input[1], torch.mul(weight[1], groups))
  if _3:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.__is__(bias, None):
    _4 = True
  else:
    bias0 = unchecked_cast(List[int], bias)
    if torch.eq(torch.len(bias0), 1):
      _5 = torch.eq(bias0[0], weight[0])
    else:
      _5 = False
    _4 = _5
  if _4:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  for _6 in range(torch.__range_length(2, k, 1)):
    i = torch.__derive_index(_6, 2, 1)
    _7 = input[i]
    _8 = torch.mul(padding[torch.sub(i, 2)], 2)
    _9 = torch.add(_7, _8)
    _10 = torch.mul(dilation[torch.sub(i, 2)], torch.sub(weight[i], 1))
    if torch.ge(_9, torch.add(_10, 1)):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  has_dilation = torch.gt(torch.len(dilation), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  _11 = torch.append(output_size, input[0])
  _12 = torch.append(output_size, weight[0])
  for _13 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_13, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    _14 = torch.mul(dilation_, torch.sub(weight[d], 1))
    kernel = torch.add(_14, 1)
    _15 = input[d]
    _16 = torch.mul(padding[torch.sub(d, 2)], 2)
    _17 = torch.sub(torch.add(_15, _16), kernel)
    _18 = torch.floordiv(_17, stride[torch.sub(d, 2)])
    _19 = torch.append(output_size, torch.add(_18, 1))
  return output_size

)=====")
+ std::string(R"=====(def batch_norm(input: List[int],
    weight: Optional[List[int]],
    bias: Optional[List[int]],
    running_mean: Optional[List[int]],
    running_var: Optional[List[int]],
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool) -> List[int]:
  out = annotate(List[int], [])
  for _0 in range(torch.len(input)):
    elem = input[_0]
    _1 = torch.append(out, elem)
  return out

)=====")
+ std::string(R"=====(def conv3d(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int) -> List[int]:
  if torch.eq(torch.len(weight), 5):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 5):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  k = torch.len(input)
  weight_dim = torch.len(weight)
  non_negative = False
  for _0 in range(torch.len(padding)):
    val = padding[_0]
    if torch.lt(val, 0):
      non_negative0 = True
    else:
      non_negative0 = non_negative
    non_negative = non_negative0
  if torch.__not__(non_negative):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  non_negative1 = False
  for _1 in range(torch.len(stride)):
    val0 = stride[_1]
    if torch.lt(val0, 0):
      non_negative2 = True
    else:
      non_negative2 = non_negative1
    non_negative1 = non_negative2
  if torch.__not__(non_negative1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(weight_dim, k):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(weight[0], groups):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _2 = torch.eq(torch.remainder(weight[0], groups), 0)
  if _2:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _3 = torch.eq(input[1], torch.mul(weight[1], groups))
  if _3:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.__is__(bias, None):
    _4 = True
  else:
    bias0 = unchecked_cast(List[int], bias)
    if torch.eq(torch.len(bias0), 1):
      _5 = torch.eq(bias0[0], weight[0])
    else:
      _5 = False
    _4 = _5
  if _4:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  for _6 in range(torch.__range_length(2, k, 1)):
    i = torch.__derive_index(_6, 2, 1)
    _7 = input[i]
    _8 = torch.mul(padding[torch.sub(i, 2)], 2)
    _9 = torch.add(_7, _8)
    _10 = torch.mul(dilation[torch.sub(i, 2)], torch.sub(weight[i], 1))
    if torch.ge(_9, torch.add(_10, 1)):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  has_dilation = torch.gt(torch.len(dilation), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  _11 = torch.append(output_size, input[0])
  _12 = torch.append(output_size, weight[0])
  for _13 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_13, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    _14 = torch.mul(dilation_, torch.sub(weight[d], 1))
    kernel = torch.add(_14, 1)
    _15 = input[d]
    _16 = torch.mul(padding[torch.sub(d, 2)], 2)
    _17 = torch.sub(torch.add(_15, _16), kernel)
    _18 = torch.floordiv(_17, stride[torch.sub(d, 2)])
    _19 = torch.append(output_size, torch.add(_18, 1))
  return output_size

)=====")
+ std::string(R"=====(def conv_backwards(grad_output: List[int],
    input: List[int],
    weight: List[int],
    biases: Optional[List[int]]) -> Tuple[List[int], List[int], List[int]]:
  out = annotate(List[int], [])
  for _0 in range(torch.len(input)):
    elem = input[_0]
    _1 = torch.append(out, elem)
  out0 = annotate(List[int], [])
  for _2 in range(torch.len(weight)):
    elem0 = weight[_2]
    _3 = torch.append(out0, elem0)
  return (out, out0, [grad_output[1]])

)=====")
+ std::string(R"=====(def conv_forwards(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int) -> List[int]:
  has_dilation = torch.gt(torch.len(dilation), 0)
  has_output_padding = torch.gt(torch.len(output_padding), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  if transposed:
    weight_output_channels_dim = 1
  else:
    weight_output_channels_dim = 0
  _0 = torch.append(output_size, input[0])
  if transposed:
    _1 = torch.mul(weight[weight_output_channels_dim], groups)
    _2 = torch.append(output_size, _1)
  else:
    _3 = torch.append(output_size, weight[weight_output_channels_dim])
  for _4 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_4, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    if has_output_padding:
      output_padding_ = output_padding[torch.sub(d, 2)]
    else:
      output_padding_ = 0
    if transposed:
      kernel = torch.mul(dilation_, torch.sub(weight[d], 1))
      _5 = torch.mul(torch.sub(input[d], 1), stride[torch.sub(d, 2)])
      _6 = torch.mul(padding[torch.sub(d, 2)], 2)
      _7 = torch.add(torch.sub(_5, _6), kernel)
      _8 = torch.add(torch.add(_7, output_padding_), 1)
      _9 = torch.append(output_size, _8)
    else:
      _10 = torch.mul(dilation_, torch.sub(weight[d], 1))
      kernel0 = torch.add(_10, 1)
      _11 = input[d]
      _12 = torch.mul(padding[torch.sub(d, 2)], 2)
      _13 = torch.sub(torch.add(_11, _12), kernel0)
      _14 = torch.floordiv(_13, stride[torch.sub(d, 2)])
      _15 = torch.append(output_size, torch.add(_14, 1))
  return output_size

)=====")
+ std::string(R"=====(def _conv_forwards(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
    cudnn_enabled: bool,
    allow_tf32: bool) -> List[int]:
  has_dilation = torch.gt(torch.len(dilation), 0)
  has_output_padding = torch.gt(torch.len(output_padding), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  if transposed:
    weight_output_channels_dim = 1
  else:
    weight_output_channels_dim = 0
  _0 = torch.append(output_size, input[0])
  if transposed:
    _1 = torch.mul(weight[weight_output_channels_dim], groups)
    _2 = torch.append(output_size, _1)
  else:
    _3 = torch.append(output_size, weight[weight_output_channels_dim])
  for _4 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_4, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 184 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/inliner.h`
- `torch/csrc/jit/runtime/operator.h`
- `torch/csrc/jit/runtime/serialized_shape_function_registry.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `serialized_shape_function_registry.cpp_docs.md`
- **Keyword Index**: `serialized_shape_function_registry.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
