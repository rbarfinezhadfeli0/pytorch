# Documentation: TestSparseCPU.test_print_uncoalesced_cpu_float64.expect

## File Metadata
- **Path**: `test/expect/TestSparseCPU.test_print_uncoalesced_cpu_float64.expect`
- **Size**: 8558 bytes
- **Lines**: 253
- **Extension**: .expect
- **Type**: Regular file

## Original Source

```expect
# shape: torch.Size([])
# nnz: 2
# sparse_dim: 0
# indices shape: torch.Size([0, 2])
# values shape: torch.Size([2])
########## torch.int32 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 2)),
       values=tensor([0, 1]),
       size=(), nnz=2, dtype=torch.int32, layout=torch.sparse_coo)
# _indices
tensor([], size=(0, 2), dtype=torch.int64)
# _values
tensor([0, 1], dtype=torch.int32)
########## torch.float32 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 2)),
       values=tensor([0., 1.]),
       size=(), nnz=2, layout=torch.sparse_coo)
# after requires_grad_
tensor(indices=tensor([], size=(0, 2)),
       values=tensor([0., 1.]),
       size=(), nnz=2, layout=torch.sparse_coo, requires_grad=True)
# after addition
tensor(indices=tensor([], size=(0, 2)),
       values=tensor([0., 2.]),
       size=(), nnz=2, layout=torch.sparse_coo, grad_fn=<AddBackward0>)
# _indices
tensor([], size=(0, 2), dtype=torch.int64)
# _values
tensor([0., 1.])

# shape: torch.Size([0])
# nnz: 10
# sparse_dim: 0
# indices shape: torch.Size([0, 10])
# values shape: torch.Size([10, 0])
########## torch.int32 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 10)),
       values=tensor([], size=(10, 0)),
       size=(0,), nnz=10, dtype=torch.int32, layout=torch.sparse_coo)
# _indices
tensor([], size=(0, 10), dtype=torch.int64)
# _values
tensor([], size=(10, 0), dtype=torch.int32)
########## torch.float64 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 10)),
       values=tensor([], size=(10, 0)),
       size=(0,), nnz=10, dtype=torch.float64, layout=torch.sparse_coo)
# after requires_grad_
tensor(indices=tensor([], size=(0, 10)),
       values=tensor([], size=(10, 0)),
       size=(0,), nnz=10, dtype=torch.float64, layout=torch.sparse_coo,
       requires_grad=True)
# after addition
tensor(indices=tensor([], size=(0, 10)),
       values=tensor([], size=(10, 0)),
       size=(0,), nnz=10, dtype=torch.float64, layout=torch.sparse_coo,
       grad_fn=<AddBackward0>)
# _indices
tensor([], size=(0, 10), dtype=torch.int64)
# _values
tensor([], size=(10, 0), dtype=torch.float64)

# shape: torch.Size([2])
# nnz: 3
# sparse_dim: 0
# indices shape: torch.Size([0, 3])
# values shape: torch.Size([3, 2])
########## torch.int32 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 3)),
       values=tensor([[0, 0],
                      [0, 1],
                      [1, 1]]),
       size=(2,), nnz=3, dtype=torch.int32, layout=torch.sparse_coo)
# _indices
tensor([], size=(0, 3), dtype=torch.int64)
# _values
tensor([[0, 0],
        [0, 1],
        [1, 1]], dtype=torch.int32)
########## torch.float32 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 3)),
       values=tensor([[0.0000, 0.3333],
                      [0.6667, 1.0000],
                      [1.3333, 1.6667]]),
       size=(2,), nnz=3, layout=torch.sparse_coo)
# after requires_grad_
tensor(indices=tensor([], size=(0, 3)),
       values=tensor([[0.0000, 0.3333],
                      [0.6667, 1.0000],
                      [1.3333, 1.6667]]),
       size=(2,), nnz=3, layout=torch.sparse_coo, requires_grad=True)
# after addition
tensor(indices=tensor([], size=(0, 3)),
       values=tensor([[0.0000, 0.6667],
                      [1.3333, 2.0000],
                      [2.6667, 3.3333]]),
       size=(2,), nnz=3, layout=torch.sparse_coo, grad_fn=<AddBackward0>)
# _indices
tensor([], size=(0, 3), dtype=torch.int64)
# _values
tensor([[0.0000, 0.3333],
        [0.6667, 1.0000],
        [1.3333, 1.6667]])

# shape: torch.Size([100, 3])
# nnz: 3
# sparse_dim: 1
# indices shape: torch.Size([1, 3])
# values shape: torch.Size([3, 3])
########## torch.int32 ##########
# sparse tensor
tensor(indices=tensor([[0, 1, 0]]),
       values=tensor([[0, 0, 0],
                      [0, 0, 1],
                      [1, 1, 1]]),
       size=(100, 3), nnz=3, dtype=torch.int32, layout=torch.sparse_coo)
# _indices
tensor([[0, 1, 0]])
# _values
tensor([[0, 0, 0],
        [0, 0, 1],
        [1, 1, 1]], dtype=torch.int32)
########## torch.float64 ##########
# sparse tensor
tensor(indices=tensor([[0, 1, 0]]),
       values=tensor([[0.0000, 0.2222, 0.4444],
                      [0.6667, 0.8889, 1.1111],
                      [1.3333, 1.5556, 1.7778]]),
       size=(100, 3), nnz=3, dtype=torch.float64, layout=torch.sparse_coo)
# after requires_grad_
tensor(indices=tensor([[0, 1, 0]]),
       values=tensor([[0.0000, 0.2222, 0.4444],
                      [0.6667, 0.8889, 1.1111],
                      [1.3333, 1.5556, 1.7778]]),
       size=(100, 3), nnz=3, dtype=torch.float64, layout=torch.sparse_coo,
       requires_grad=True)
# after addition
tensor(indices=tensor([[0, 1, 0]]),
       values=tensor([[0.0000, 0.4444, 0.8889],
                      [1.3333, 1.7778, 2.2222],
                      [2.6667, 3.1111, 3.5556]]),
       size=(100, 3), nnz=3, dtype=torch.float64, layout=torch.sparse_coo,
       grad_fn=<AddBackward0>)
# _indices
tensor([[0, 1, 0]])
# _values
tensor([[0.0000, 0.2222, 0.4444],
        [0.6667, 0.8889, 1.1111],
        [1.3333, 1.5556, 1.7778]], dtype=torch.float64)

# shape: torch.Size([100, 20, 3])
# nnz: 0
# sparse_dim: 2
# indices shape: torch.Size([2, 0])
# values shape: torch.Size([0, 3])
########## torch.int32 ##########
# sparse tensor
tensor(indices=tensor([], size=(2, 0)),
       values=tensor([], size=(0, 3)),
       size=(100, 20, 3), nnz=0, dtype=torch.int32, layout=torch.sparse_coo)
# _indices
tensor([], size=(2, 0), dtype=torch.int64)
# _values
tensor([], size=(0, 3), dtype=torch.int32)
########## torch.float32 ##########
# sparse tensor
tensor(indices=tensor([], size=(2, 0)),
       values=tensor([], size=(0, 3)),
       size=(100, 20, 3), nnz=0, layout=torch.sparse_coo)
# after requires_grad_
tensor(indices=tensor([], size=(2, 0)),
       values=tensor([], size=(0, 3)),
       size=(100, 20, 3), nnz=0, layout=torch.sparse_coo, requires_grad=True)
# after addition
tensor(indices=tensor([], size=(2, 0)),
       values=tensor([], size=(0, 3)),
       size=(100, 20, 3), nnz=0, layout=torch.sparse_coo, grad_fn=<AddBackward0>)
# _indices
tensor([], size=(2, 0), dtype=torch.int64)
# _values
tensor([], size=(0, 3))

# shape: torch.Size([10, 0, 3])
# nnz: 3
# sparse_dim: 0
# indices shape: torch.Size([0, 3])
# values shape: torch.Size([3, 10, 0, 3])
########## torch.int32 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 3)),
       values=tensor([], size=(3, 10, 0, 3)),
       size=(10, 0, 3), nnz=3, dtype=torch.int32, layout=torch.sparse_coo)
# _indices
tensor([], size=(0, 3), dtype=torch.int64)
# _values
tensor([], size=(3, 10, 0, 3), dtype=torch.int32)
########## torch.float64 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 3)),
       values=tensor([], size=(3, 10, 0, 3)),
       size=(10, 0, 3), nnz=3, dtype=torch.float64, layout=torch.sparse_coo)
# after requires_grad_
tensor(indices=tensor([], size=(0, 3)),
       values=tensor([], size=(3, 10, 0, 3)),
       size=(10, 0, 3), nnz=3, dtype=torch.float64, layout=torch.sparse_coo,
       requires_grad=True)
# after addition
tensor(indices=tensor([], size=(0, 3)),
       values=tensor([], size=(3, 10, 0, 3)),
       size=(10, 0, 3), nnz=3, dtype=torch.float64, layout=torch.sparse_coo,
       grad_fn=<AddBackward0>)
# _indices
tensor([], size=(0, 3), dtype=torch.int64)
# _values
tensor([], size=(3, 10, 0, 3), dtype=torch.float64)

# shape: torch.Size([10, 0, 3])
# nnz: 0
# sparse_dim: 0
# indices shape: torch.Size([0, 0])
# values shape: torch.Size([0, 10, 0, 3])
########## torch.int32 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 0)),
       values=tensor([], size=(0, 10, 0, 3)),
       size=(10, 0, 3), nnz=0, dtype=torch.int32, layout=torch.sparse_coo)
# _indices
tensor([], size=(0, 0), dtype=torch.int64)
# _values
tensor([], size=(0, 10, 0, 3), dtype=torch.int32)
########## torch.float32 ##########
# sparse tensor
tensor(indices=tensor([], size=(0, 0)),
       values=tensor([], size=(0, 10, 0, 3)),
       size=(10, 0, 3), nnz=0, layout=torch.sparse_coo)
# after requires_grad_
tensor(indices=tensor([], size=(0, 0)),
       values=tensor([], size=(0, 10, 0, 3)),
       size=(10, 0, 3), nnz=0, layout=torch.sparse_coo, requires_grad=True)
# after addition
tensor(indices=tensor([], size=(0, 0)),
       values=tensor([], size=(0, 10, 0, 3)),
       size=(10, 0, 3), nnz=0, layout=torch.sparse_coo, grad_fn=<AddBackward0>)
# _indices
tensor([], size=(0, 0), dtype=torch.int64)
# _values
tensor([], size=(0, 10, 0, 3))

```

## High-Level Overview

This file is part of the PyTorch repository. It is a source or configuration file.

## Detailed Walkthrough


## Key Components

The file contains 813 words across 253 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 8558 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
