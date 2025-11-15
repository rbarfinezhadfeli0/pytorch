# Documentation: `torch/csrc/profiler/unwind/eh_frame_hdr.h`

## File Metadata

- **Path**: `torch/csrc/profiler/unwind/eh_frame_hdr.h`
- **Size**: 2,680 bytes (2.62 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <cstdint>
#include <ostream>

#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>

// Overview of the format described in
// https://refspecs.linuxfoundation.org/LSB_1.3.0/gLSB/gLSB/ehframehdr.html
namespace torch::unwind {

struct EHFrameHdr {
  EHFrameHdr(void* base) : base_(base) {
    Lexer L(base, base);
    version_ = L.read<uint8_t>();
    eh_frame_ptr_enc_ = L.read<uint8_t>();
    fde_count_enc_ = L.read<uint8_t>();
    table_enc_ = L.read<uint8_t>();
    if (table_enc_ == DW_EH_PE_omit) {
      table_size_ = 0;
    } else {
      switch (table_enc_ & 0xF) {
        case DW_EH_PE_udata2:
        case DW_EH_PE_sdata2:
          table_size_ = 2;
          break;
        case DW_EH_PE_udata4:
        case DW_EH_PE_sdata4:
          table_size_ = 4;
          break;
        case DW_EH_PE_udata8:
        case DW_EH_PE_sdata8:
          table_size_ = 8;
          break;
        case DW_EH_PE_uleb128:
        case DW_EH_PE_sleb128:
          throw UnwindError("uleb/sleb table encoding not supported");
          break;
        default:
          throw UnwindError("unknown table encoding");
      }
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    eh_frame_ = (void*)L.readEncodedOr(eh_frame_ptr_enc_, 0);
    fde_count_ = L.readEncodedOr(fde_count_enc_, 0);
    table_start_ = L.loc();
  }
  size_t nentries() const {
    return fde_count_;
  }

  uint64_t lowpc(size_t i) const {
    return Lexer(table_start_, base_)
        .skip(2 * i * table_size_)
        .readEncoded(table_enc_);
  }
  void* fde(size_t i) const {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return (void*)Lexer(table_start_, base_)
        .skip((2 * i + 1) * table_size_)
        .readEncoded(table_enc_);
  }

  void* entryForAddr(uint64_t addr) const {
    if (!table_size_ || !nentries()) {
      throw UnwindError("search table not present");
    }
    uint64_t low = 0;
    uint64_t high = nentries();
    while (low + 1 < high) {
      auto mid = (low + high) / 2;
      if (addr < lowpc(mid)) {
        high = mid;
      } else {
        low = mid;
      }
    }
    return fde(low);
  }

  friend std::ostream& operator<<(std::ostream& out, const EHFrameHdr& self) {
    out << "EHFrameHeader(version=" << self.version_
        << ",table_size=" << self.table_size_
        << ",fde_count=" << self.fde_count_ << ")";
    return out;
  }

 private:
  void* base_;
  void* table_start_;
  uint8_t version_;
  uint8_t eh_frame_ptr_enc_;
  uint8_t fde_count_enc_;
  uint8_t table_enc_;
  void* eh_frame_ = nullptr;
  int64_t fde_count_;
  uint32_t table_size_;
};

} // namespace torch::unwind

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `EHFrameHdr`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler/unwind`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `ostream`
- `torch/csrc/profiler/unwind/lexer.h`
- `torch/csrc/profiler/unwind/unwind_error.h`


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

Files in the same folder (`torch/csrc/profiler/unwind`):

- [`unwind_fb.cpp_docs.md`](./unwind_fb.cpp_docs.md)
- [`unwind.cpp_docs.md`](./unwind.cpp_docs.md)
- [`dwarf_symbolize_enums.h_docs.md`](./dwarf_symbolize_enums.h_docs.md)
- [`fde.h_docs.md`](./fde.h_docs.md)
- [`sections.h_docs.md`](./sections.h_docs.md)
- [`unwind.h_docs.md`](./unwind.h_docs.md)
- [`debug_info.h_docs.md`](./debug_info.h_docs.md)
- [`action.h_docs.md`](./action.h_docs.md)
- [`lexer.h_docs.md`](./lexer.h_docs.md)
- [`unwind_error.h_docs.md`](./unwind_error.h_docs.md)


## Cross-References

- **File Documentation**: `eh_frame_hdr.h_docs.md`
- **Keyword Index**: `eh_frame_hdr.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
