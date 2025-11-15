# Documentation: `benchmarks/fastrnns/cells.py`

## File Metadata

- **Path**: `benchmarks/fastrnns/cells.py`
- **Size**: 3,641 bytes (3.56 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
import torch
from torch import Tensor


def milstm_cell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())

    # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
    gates = alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias

    # Same as LSTMCell after this point
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()

    return hy, cy


def lstm_cell(
    input: Tensor,
    hidden: tuple[Tensor, Tensor],
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def flat_lstm_cell(
    input: Tensor,
    hx: Tensor,
    cx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def premul_lstm_cell(
    igates: Tensor,
    hidden: tuple[Tensor, Tensor],
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = igates + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def premul_lstm_cell_no_bias(
    igates: Tensor, hidden: tuple[Tensor, Tensor], w_hh: Tensor, b_hh: Tensor
) -> tuple[Tensor, Tensor]:
    hx, cx = hidden
    gates = igates + torch.mm(hx, w_hh.t()) + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def gru_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    gi = torch.mm(input, w_ih.t()) + b_ih
    gh = torch.mm(hidden, w_hh.t()) + b_hh
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def rnn_relu_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    igates = torch.mm(input, w_ih.t()) + b_ih
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    return torch.relu(igates + hgates)


def rnn_tanh_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    igates = torch.mm(input, w_ih.t()) + b_ih
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    return torch.tanh(igates + hgates)

```



## High-Level Overview


This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `milstm_cell`, `lstm_cell`, `flat_lstm_cell`, `premul_lstm_cell`, `premul_lstm_cell_no_bias`, `gru_cell`, `rnn_relu_cell`, `rnn_tanh_cell`

**Key imports**: torch, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/fastrnns`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`benchmarks/fastrnns`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`bench.py_docs.md`](./bench.py_docs.md)
- [`conftest.py_docs.md`](./conftest.py_docs.md)
- [`profile.py_docs.md`](./profile.py_docs.md)
- [`custom_lstms.py_docs.md`](./custom_lstms.py_docs.md)
- [`factory.py_docs.md`](./factory.py_docs.md)
- [`fuser.py_docs.md`](./fuser.py_docs.md)
- [`runner.py_docs.md`](./runner.py_docs.md)


## Cross-References

- **File Documentation**: `cells.py_docs.md`
- **Keyword Index**: `cells.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
