# Documentation: `docs/test/cpp/monitor/test_counters.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/monitor/test_counters.cpp_docs.md`
- **Size**: 8,782 bytes (8.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/monitor/test_counters.cpp`

## File Metadata

- **Path**: `test/cpp/monitor/test_counters.cpp`
- **Size**: 6,719 bytes (6.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <thread>

#include <torch/csrc/monitor/counters.h>
#include <torch/csrc/monitor/events.h>

using namespace torch::monitor;

TEST(MonitorTest, CounterDouble) {
  Stat<double> a{
      "a",
      {Aggregation::MEAN, Aggregation::COUNT},
      std::chrono::milliseconds(100000),
      2,
  };
  a.add(5.0);
  ASSERT_EQ(a.count(), 1);
  a.add(6.0);
  ASSERT_EQ(a.count(), 0);

  auto stats = a.get();
  std::unordered_map<Aggregation, double, AggregationHash> want = {
      {Aggregation::MEAN, 5.5},
      {Aggregation::COUNT, 2.0},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Sum) {
  Stat<int64_t> a{
      "a",
      {Aggregation::SUM},
      std::chrono::milliseconds(100000),
      2,
  };
  a.add(5);
  a.add(6);
  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
      {Aggregation::SUM, 11},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Value) {
  Stat<int64_t> a{
      "a",
      {Aggregation::VALUE},
      std::chrono::milliseconds(100000),
      2,
  };
  a.add(5);
  a.add(6);
  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
      {Aggregation::VALUE, 6},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64Mean) {
  Stat<int64_t> a{
      "a",
      {Aggregation::MEAN},
      std::chrono::milliseconds(100000),
      2,
  };
  {
    // zero samples case
    auto stats = a.get();
    std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
        {Aggregation::MEAN, 0},
    };
    ASSERT_EQ(stats, want);
  }

  a.add(0);
  a.add(10);

  {
    auto stats = a.get();
    std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
        {Aggregation::MEAN, 5},
    };
    ASSERT_EQ(stats, want);
  }
}

TEST(MonitorTest, CounterInt64Count) {
  Stat<int64_t> a{
      "a",
      {Aggregation::COUNT},
      std::chrono::milliseconds(100000),
      2,
  };
  ASSERT_EQ(a.count(), 0);
  a.add(0);
  ASSERT_EQ(a.count(), 1);
  a.add(10);
  ASSERT_EQ(a.count(), 0);

  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
      {Aggregation::COUNT, 2},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64MinMax) {
  Stat<int64_t> a{
      "a",
      {Aggregation::MIN, Aggregation::MAX},
      std::chrono::milliseconds(100000),
      6,
  };
  {
    auto stats = a.get();
    std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
        {Aggregation::MAX, 0},
        {Aggregation::MIN, 0},
    };
    ASSERT_EQ(stats, want);
  }

  a.add(0);
  a.add(5);
  a.add(-5);
  a.add(-6);
  a.add(9);
  a.add(2);
  {
    auto stats = a.get();
    std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
        {Aggregation::MAX, 9},
        {Aggregation::MIN, -6},
    };
    ASSERT_EQ(stats, want);
  }
}

TEST(MonitorTest, CounterInt64WindowSize) {
  Stat<int64_t> a{
      "a",
      {Aggregation::COUNT, Aggregation::SUM},
      std::chrono::milliseconds(100000),
      /*windowSize=*/3,
  };
  a.add(1);
  a.add(2);
  ASSERT_EQ(a.count(), 2);
  a.add(3);
  ASSERT_EQ(a.count(), 0);

  // after logging max for window, should be zero
  a.add(4);
  ASSERT_EQ(a.count(), 0);

  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
      {Aggregation::COUNT, 3},
      {Aggregation::SUM, 6},
  };
  ASSERT_EQ(stats, want);
}

TEST(MonitorTest, CounterInt64WindowSizeHuge) {
  Stat<int64_t> a{
      "a",
      {Aggregation::COUNT, Aggregation::SUM},
      std::chrono::hours(24 * 365 * 10), // 10 years
      /*windowSize=*/3,
  };
  a.add(1);
  a.add(2);
  ASSERT_EQ(a.count(), 2);
  a.add(3);
  ASSERT_EQ(a.count(), 0);

  // after logging max for window, should be zero
  a.add(4);
  ASSERT_EQ(a.count(), 0);

  auto stats = a.get();
  std::unordered_map<Aggregation, int64_t, AggregationHash> want = {
      {Aggregation::COUNT, 3},
      {Aggregation::SUM, 6},
  };
  ASSERT_EQ(stats, want);
}

template <typename T>
struct TestStat : public Stat<T> {
  uint64_t mockWindowId{1};

  TestStat(
      std::string name,
      std::initializer_list<Aggregation> aggregations,
      std::chrono::milliseconds windowSize,
      int64_t maxSamples = std::numeric_limits<int64_t>::max())
      : Stat<T>(name, aggregations, windowSize, maxSamples) {}

  uint64_t currentWindowId() const override {
    return mockWindowId;
  }
};

struct AggregatingEventHandler : public EventHandler {
  std::vector<Event> events;

  void handle(const Event& e) override {
    events.emplace_back(e);
  }
};

template <typename T>
struct HandlerGuard {
  std::shared_ptr<T> handler;

  HandlerGuard() : handler(std::make_shared<T>()) {
    registerEventHandler(handler);
  }

  ~HandlerGuard() {
    unregisterEventHandler(handler);
  }
};

TEST(MonitorTest, Stat) {
  HandlerGuard<AggregatingEventHandler> guard;

  Stat<int64_t> a{
      "a",
      {Aggregation::COUNT, Aggregation::SUM},
      std::chrono::milliseconds(1),
  };
  ASSERT_EQ(guard.handler->events.size(), 0);

  a.add(1);
  ASSERT_LE(a.count(), 1);

  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  a.add(2);
  ASSERT_LE(a.count(), 1);

  ASSERT_GE(guard.handler->events.size(), 1);
  ASSERT_LE(guard.handler->events.size(), 2);
}

TEST(MonitorTest, StatEvent) {
  HandlerGuard<AggregatingEventHandler> guard;

  TestStat<int64_t> a{
      "a",
      {Aggregation::COUNT, Aggregation::SUM},
      std::chrono::milliseconds(1),
  };
  ASSERT_EQ(guard.handler->events.size(), 0);

  a.add(1);
  ASSERT_EQ(a.count(), 1);
  a.add(2);
  ASSERT_EQ(a.count(), 2);
  ASSERT_EQ(guard.handler->events.size(), 0);

  a.mockWindowId = 100;

  a.add(3);
  ASSERT_LE(a.count(), 1);

  ASSERT_EQ(guard.handler->events.size(), 1);
  Event e = guard.handler->events.at(0);
  ASSERT_EQ(e.name, "torch.monitor.Stat");
  ASSERT_NE(e.timestamp, std::chrono::system_clock::time_point{});
  std::unordered_map<std::string, data_value_t> data{
      {"a.sum", 3L},
      {"a.count", 2L},
  };
  ASSERT_EQ(e.data, data);
}

TEST(MonitorTest, StatEventDestruction) {
  HandlerGuard<AggregatingEventHandler> guard;

  {
    TestStat<int64_t> a{
        "a",
        {Aggregation::COUNT, Aggregation::SUM},
        std::chrono::hours(10),
    };
    a.add(1);
    ASSERT_EQ(a.count(), 1);
    ASSERT_EQ(guard.handler->events.size(), 0);
  }
  ASSERT_EQ(guard.handler->events.size(), 1);

  Event e = guard.handler->events.at(0);
  ASSERT_EQ(e.name, "torch.monitor.Stat");
  ASSERT_NE(e.timestamp, std::chrono::system_clock::time_point{});
  std::unordered_map<std::string, data_value_t> data{
      {"a.sum", 1L},
      {"a.count", 1L},
  };
  ASSERT_EQ(e.data, data);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TestStat`, `AggregatingEventHandler`, `HandlerGuard`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/monitor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `thread`
- `torch/csrc/monitor/counters.h`
- `torch/csrc/monitor/events.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/monitor/test_counters.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/monitor`):

- [`test_events.cpp_docs.md`](./test_events.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_counters.cpp_docs.md`
- **Keyword Index**: `test_counters.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/monitor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/monitor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/monitor/test_counters.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/monitor`):

- [`test_counters.cpp_kw.md_docs.md`](./test_counters.cpp_kw.md_docs.md)
- [`test_events.cpp_docs.md_docs.md`](./test_events.cpp_docs.md_docs.md)
- [`test_events.cpp_kw.md_docs.md`](./test_events.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_counters.cpp_docs.md_docs.md`
- **Keyword Index**: `test_counters.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
