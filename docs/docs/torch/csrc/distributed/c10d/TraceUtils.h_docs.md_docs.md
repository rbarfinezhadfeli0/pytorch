# Documentation: `docs/torch/csrc/distributed/c10d/TraceUtils.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/TraceUtils.h_docs.md`
- **Size**: 11,839 bytes (11.56 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/TraceUtils.h`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/TraceUtils.h`
- **Size**: 9,086 bytes (8.87 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/combined_traceback.h>

#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/ostream.h> // optional, for ostream fallback
#include <fmt/ranges.h> // for fmt::join

#include <sys/types.h>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

namespace c10d {

inline std::string getTraceStartKey(const std::string& pgName, int rank) {
  return fmt::format(FMT_COMPILE("{}_{}_trace_start"), pgName, rank);
}

inline std::string getTraceEndKey(const std::string& pgName, int rank) {
  return fmt::format(FMT_COMPILE("{}_{}_trace_end"), pgName, rank);
}

inline bool traceUpdate(
    c10::intrusive_ptr<Store>& store,
    const std::string& key,
    uint64_t seq,
    const std::string& col) {
  std::vector<uint8_t> value(col.size() + sizeof(seq) + 1);
  std::memcpy(value.data(), &seq, sizeof(seq));
  std::memcpy(value.data() + sizeof(seq), col.data(), col.size());
  try {
    store->set(key, value);
    return true;
  } catch (...) {
    LOG(ERROR) << "Store is down while updating #" << seq << " with key "
               << key;
    return false;
  }
  return true;
}

enum TraceDebugEvent {
  kEventStart,
  kEventEnd,
};
// <seq, <rank, <col, start/end>>>
using TraceMap =
    std::map<uint64_t, std::map<int, std::pair<std::string, TraceDebugEvent>>>;

inline std::string ranksToString(const std::vector<int>& ranks) {
  return fmt::to_string(fmt::join(ranks, ", "));
}

inline std::string ranksFromTrace(
    const std::vector<std::pair<int, std::string>>& items) {
  fmt::memory_buffer buf;
  bool first = true;
  for (const auto& [rank, _] : items) {
    if (!first) {
      fmt::format_to(std::back_inserter(buf), ", ");
    }
    fmt::format_to(std::back_inserter(buf), "{}", rank);
    first = false;
  }
  return fmt::to_string(buf);
}

inline std::string analyzeMissingRanks(const std::vector<int>& missingRanks) {
  return c10::str(
      "\n\t - To our best knowledge, ranks [",
      ranksToString(missingRanks),
      "] are the lagging ranks that caused this timeout. "
      "They never joined any collectives");
}

inline std::string analyzeLaggingRanks(const TraceMap& traceMap) {
  uint64_t lagSeq = traceMap.begin()->first;
  std::vector<int> startRanks;
  std::vector<int> endRanks;
  for (auto& p : traceMap.begin()->second) {
    if (p.second.second == kEventStart) {
      startRanks.push_back(p.first);
    } else {
      endRanks.push_back(p.first);
    }
  }
  std::string report =
      "\n\t - To our best knowledge, the lagging/dead/mismatched ranks "
      "that caused the desync are:";
  if (!startRanks.empty()) {
    report += c10::str(
        "\n\t   - [",
        ranksToString(startRanks),
        "] joined but didn't finish collective #",
        lagSeq,
        " (count from 1)");
  }
  if (!endRanks.empty()) {
    report += c10::str(
        "\n\t     [",
        ranksToString(endRanks),
        "] finished collective #",
        lagSeq,
        ", but didn't join collective #",
        lagSeq + 1,
        " (count from 1)");
  }
  return report;
}

inline std::string dumpSnapshot(TraceMap& traceMap) {
  std::string report = "\n\t - Snapshot of ranks' latest states:";
  for (auto& tracePair : traceMap) {
    uint64_t seq = tracePair.first;
    std::map<int, std::pair<std::string, TraceDebugEvent>>& subMap =
        tracePair.second;

    std::unordered_map<std::string, std::vector<int>> collectivesStart;
    std::unordered_map<std::string, std::vector<int>> collectivesEnd;
    for (const auto& p : subMap) {
      int rank = p.first;
      const std::string& col = p.second.first;
      if (p.second.second == kEventStart) {
        collectivesStart[col].push_back(rank);
      } else {
        collectivesEnd[col].push_back(rank);
      }
    }

    if (!collectivesStart.empty()) {
      report += c10::str("\n\t   #", seq, " started ranks:");
      for (auto& mapPair : collectivesStart) {
        report += c10::str(
            "\n\t     [",
            ranksToString(mapPair.second),
            "] started ",
            mapPair.first);
      }
    }
    if (!collectivesEnd.empty()) {
      report += c10::str("\n\t   #", seq, " finished ranks:");
      for (auto& mapPair : collectivesEnd) {
        report += c10::str(
            "\n\t     [",
            ranksToString(mapPair.second),
            "] finished ",
            mapPair.first);
      }
    }
  }
  return report;
}

inline bool parseTraceValue(
    c10::intrusive_ptr<Store>& store,
    const std::string& key,
    uint64_t& seq,
    std::string& col) {
  try {
    std::vector<uint8_t> traceValue = store->get(key);
    std::memcpy(&seq, traceValue.data(), sizeof(seq));
    std::string colName((char*)traceValue.data() + sizeof(seq));
    col = colName;
    return true;
  } catch (...) {
    LOG(ERROR) << "Store is down while getting key " << key;
    return false;
  }
  return true;
}

inline std::string retrieveDesyncReport(
    c10::intrusive_ptr<Store>& store,
    const std::string& pgName,
    int myRank,
    int worldSize) {
  std::string report;

  uint64_t thisSeq = 0;
  std::string thisCol;

  std::vector<int> missingRanks;
  TraceMap traceMap;

  for (const auto rank : c10::irange(worldSize)) {
    // Build traceMapStart.
    uint64_t seqStart = 0;
    {
      std::string traceKeyStart = getTraceStartKey(pgName, rank);
      if (!store->check({traceKeyStart})) {
        missingRanks.push_back(rank);
        continue;
      }
      std::string col;
      if (!parseTraceValue(store, traceKeyStart, seqStart, col)) {
        return report;
      }
      traceMap[seqStart].emplace(rank, std::make_pair(col, kEventStart));
      if (rank == myRank) {
        thisSeq = seqStart;
        thisCol = std::move(col);
      }
    }

    // Build traceMapEnd.
    {
      std::string traceKeyEnd = getTraceEndKey(pgName, rank);
      if (!store->check({traceKeyEnd})) {
        continue;
      }
      uint64_t seq = 0;
      std::string col;
      if (!parseTraceValue(store, traceKeyEnd, seq, col)) {
        return report;
      }
      if (seq == seqStart) {
        traceMap[seq][rank].second = kEventEnd;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      !missingRanks.empty() || !traceMap.empty(),
      "Trace shouldn't be empty while enabled GLOO_ASYNC_TIMEOUT_DEBUG");
  TORCH_INTERNAL_ASSERT(
      !thisCol.empty(),
      "Timeout rank [",
      myRank,
      "] must have collective tracking iteam in c10::Store trace");
  TORCH_INTERNAL_ASSERT(
      traceMap[thisSeq][myRank].second == kEventStart,
      "Timeout rank [",
      myRank,
      "] last trace item must be kEventStart. thisSeq = ",
      thisSeq,
      ", col = ",
      thisCol);

  report += c10::str(
      "\n\t - [", myRank, "] Timeout at collective: ", thisCol, ", #", thisSeq);

  if (!missingRanks.empty()) {
    report += analyzeMissingRanks(missingRanks);
  } else {
    report += analyzeLaggingRanks(traceMap);
    report += dumpSnapshot(traceMap);
  }

  return report;
}

inline std::string pickle_str(const c10::IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    torch::jit::Pickler pickler(
        writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}

inline std::string get_python_cpp_trace() {
  // usage:
  // LOG(INFO) << "stacktrace: "
  //           << get_python_cpp_trace();
  // warn: might be slow in getting cpp traces
  // because of slow/broken addr2line
  // in different system libs
  std::shared_ptr<torch::CapturedTraceback> tb =
      torch::CapturedTraceback::gather(
          /*python=*/true, /*script=*/true, /*cpp=*/true);
  torch::SymbolizedTracebacks s_tbs = torch::symbolize({tb.get()});
  const auto& s_tb = s_tbs.tracebacks.at(0);
  constexpr auto TB_FMT_CSTR = FMT_COMPILE("#{} {} from {}:{}\n");
  fmt::memory_buffer buf;
  auto buf_iter = std::back_inserter(buf);
  for (auto idx : c10::irange(s_tb.size())) {
    auto frame_id = s_tb[idx];
    const auto& frame = s_tbs.all_frames.at(frame_id);
    fmt::format_to(
        buf_iter,
        TB_FMT_CSTR,
        idx,
        frame.funcname,
        frame.filename,
        frame.lineno);
  }
  return fmt::to_string(buf);
}

inline c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(
      c10::AnyType::get(), c10::AnyType::get());
}

inline c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

inline std::string ranks_str(const std::vector<uint64_t>& ranks) {
  return fmt::format("[{}]", fmt::join(ranks, ", "));
}

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/ScalarType.h`
- `c10/util/ApproximateClock.h`
- `c10/util/irange.h`
- `torch/csrc/distributed/c10d/Store.hpp`
- `torch/csrc/distributed/c10d/Types.hpp`
- `torch/csrc/distributed/c10d/Utils.hpp`
- `torch/csrc/jit/serialization/pickler.h`
- `torch/csrc/profiler/combined_traceback.h`
- `fmt/compile.h`
- `fmt/core.h`
- `fmt/ostream.h`
- `fmt/ranges.h`
- `sys/types.h`
- `cstdlib`
- `cstring`
- `iterator`
- `string`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/distributed/c10d`):

- [`Utils.hpp_docs.md`](./Utils.hpp_docs.md)
- [`Ops.cpp_docs.md`](./Ops.cpp_docs.md)
- [`Store.hpp_docs.md`](./Store.hpp_docs.md)
- [`WinSockUtils.hpp_docs.md`](./WinSockUtils.hpp_docs.md)
- [`FakeProcessGroup.hpp_docs.md`](./FakeProcessGroup.hpp_docs.md)
- [`Work.cpp_docs.md`](./Work.cpp_docs.md)
- [`PrefixStore.hpp_docs.md`](./PrefixStore.hpp_docs.md)
- [`PyProcessGroup.hpp_docs.md`](./PyProcessGroup.hpp_docs.md)
- [`debug.h_docs.md`](./debug.h_docs.md)
- [`exception.h_docs.md`](./exception.h_docs.md)


## Cross-References

- **File Documentation**: `TraceUtils.h_docs.md`
- **Keyword Index**: `TraceUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/distributed/c10d`):

- [`ProcessGroupWrapper.cpp_docs.md_docs.md`](./ProcessGroupWrapper.cpp_docs.md_docs.md)
- [`c10d.h_kw.md_docs.md`](./c10d.h_kw.md_docs.md)
- [`TCPStoreLibUvBackend.cpp_kw.md_docs.md`](./TCPStoreLibUvBackend.cpp_kw.md_docs.md)
- [`ProcessGroupGlooCuda.cpp_docs.md_docs.md`](./ProcessGroupGlooCuda.cpp_docs.md_docs.md)
- [`NanCheck.cu_docs.md_docs.md`](./NanCheck.cu_docs.md_docs.md)
- [`python_callback_work.hpp_kw.md_docs.md`](./python_callback_work.hpp_kw.md_docs.md)
- [`sequence_num.hpp_kw.md_docs.md`](./sequence_num.hpp_kw.md_docs.md)
- [`Functional.hpp_kw.md_docs.md`](./Functional.hpp_kw.md_docs.md)
- [`TCPStoreBackend.cpp_kw.md_docs.md`](./TCPStoreBackend.cpp_kw.md_docs.md)
- [`ProcessGroupUCC.cpp_kw.md_docs.md`](./ProcessGroupUCC.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `TraceUtils.h_docs.md_docs.md`
- **Keyword Index**: `TraceUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
