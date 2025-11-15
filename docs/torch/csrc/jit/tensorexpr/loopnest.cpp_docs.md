# Documentation: `torch/csrc/jit/tensorexpr/loopnest.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/loopnest.cpp`
- **Size**: 104,226 bytes (101.78 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/tensorexpr/loopnest.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <c10/util/Logging.h>
#include <c10/util/irange.h>

#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_cloner.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch::jit::tensorexpr {

LoopNest::LoopNest(const LoopNest& other)
    : root_stmt_(Stmt::clone(other.root_stmt_)),
      output_bufs_(other.output_bufs_) {
  GRAPH_DEBUG("Origin Stmt in LoopNest:\n", std::to_string(root_stmt_));
  verify(root_stmt_);
}

LoopNest::LoopNest(StmtPtr stmt, std::unordered_set<BufPtr> output_bufs)
    : root_stmt_(std::move(stmt)), output_bufs_(std::move(output_bufs)) {
  GRAPH_DEBUG("Origin Stmt in LoopNest:\n", std::to_string(root_stmt_));
  verify(root_stmt_);
}

LoopNest::LoopNest(
    const std::vector<Tensor>& output_tensors,
    const std::vector<Tensor>& tensors_to_compute) {
  initialize(output_tensors, tensors_to_compute);
  GRAPH_DEBUG("Origin Stmt in LoopNest:\n", std::to_string(root_stmt_));
  verify(root_stmt_);
}

LoopNest::LoopNest(const std::vector<Tensor>& output_tensors) {
  initialize(output_tensors, output_tensors);
  GRAPH_DEBUG("Origin Stmt in LoopNest:\n", std::to_string(root_stmt_));
  verify(root_stmt_);
}

std::vector<BufPtr> LoopNest::getIntermediateBufs() const {
  std::vector<BufPtr> result;
  std::unordered_set<BufPtr> result_set;
  auto input_bufs = getInputBufs();
  auto bufs = NodeFinder<Buf>::find(root_stmt_);
  for (const auto& buf : bufs) {
    if (!output_bufs_.count(buf) && !input_bufs.count(buf) &&
        !result_set.count(buf)) {
      result.push_back(buf);
      result_set.insert(buf);
    }
  }
  return result;
}

const std::unordered_set<BufPtr> LoopNest::getInputBufs() const {
  std::unordered_set<BufPtr> result;
  auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);
  for (auto& kv : buf_load_store_uses) {
    bool has_store = false;
    for (auto& use : kv.second) {
      if (use.isStore) {
        has_store = true;
        break;
      }
    }
    if (!has_store) {
      result.insert(kv.first);
    }
  }
  return result;
}

class IndexFlattener : public IRMutator {
 public:
  StmtPtr flatten(const StmtPtr& s) {
    return s->accept_mutator(this);
  }

  ExprPtr mutate(const LoadPtr& v) override {
    if (v->indices().size() == 1) {
      return v;
    }
    return alloc<Load>(
        v->dtype(),
        v->buf(),
        std::vector<ExprPtr>({flatten_index(
            v->buf()->dims(), v->indices(), v->buf()->strides())}));
  }

  StmtPtr mutate(const StorePtr& v) override {
    ExprPtr value = v->value();
    ExprPtr new_value = value->accept_mutator(this);
    if (v->indices().size() == 1 && value == new_value) {
      return v;
    }
    std::vector<ExprPtr> indices = {
        flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides())};
    v->set_indices(indices);
    v->set_value(new_value);
    return v;
  }
};

static bool isValidIdentifierChar(char c, size_t pos) {
  return islower(c) || isupper(c) || c == '_' || (pos > 0 && isdigit(c));
}

// replaces all invalid characters with underscore
std::string sanitizeName(const std::string& input_name) {
  std::stringstream sanitized_name;
  for (size_t i = 0; i < input_name.size(); ++i) {
    if (isValidIdentifierChar(input_name[i], i)) {
      sanitized_name << input_name[i];
    } else {
      if (i == 0) {
        // Don't start names with underscore
        sanitized_name << "v";
      }
      sanitized_name << "_";
    }
  }
  return sanitized_name.str();
}

class VarNameSanitizer : public IRMutator {
 public:
  ExprPtr mutate(const BufPtr& v) override {
    if (seen_bufs_.count(v)) {
      return v;
    }
    const std::string& name = v->name_hint();
    auto new_name = sanitizeName(name);
    if (taken_names_.count(new_name)) {
      new_name = getNextAvailableName(new_name);
    }
    v->set_name_hint(new_name);
    taken_names_.insert(new_name);
    seen_bufs_.insert(v);
    return v;
  }

  ExprPtr mutate(const VarPtr& v) override {
    if (seen_vars_.count(v)) {
      return v;
    }
    const std::string& name = v->name_hint();
    auto new_name = sanitizeName(name);
    if (taken_names_.count(new_name)) {
      new_name = getNextAvailableName(new_name);
    }
    v->set_name_hint(new_name);
    taken_names_.insert(new_name);
    seen_vars_.insert(v);
    return v;
  }

  StmtPtr mutate(const ForPtr& v) override {
    auto new_name = getNextAvailableName(getIndexVarNameAtLevel(level_));
    if (seen_index_vars_.count(v->var())) {
      auto new_var = alloc<Var>("", v->var()->dtype());
      Substitute(v, {{v->var(), new_var}});
    }
    v->var()->set_name_hint(new_name);
    seen_index_vars_.insert(v->var());
    seen_vars_.insert(v->var());
    taken_names_.insert(new_name);
    level_++;
    v->body()->accept_mutator(this);
    level_--;
    v->start()->accept_mutator(this);
    v->stop()->accept_mutator(this);
    return v;
  }

  std::string getIndexVarNameAtLevel(int level_) {
    auto names_num = index_var_names_.size();
    auto counter = level_ / names_num;
    if (counter == 0) {
      return index_var_names_[level_ % names_num];
    } else {
      return index_var_names_[level_ % names_num] + std::to_string(counter);
    }
  }
  std::string getNextAvailableName(const std::string& base_name) {
    std::string name = base_name;
    int counter = 0;
    while (taken_names_.count(name)) {
      counter++;
      name = base_name + "_" + std::to_string(counter);
    }
    return name;
  }

 private:
  std::vector<std::string> index_var_names_ =
      {"i", "j", "k", "l", "m", "n", "o", "p"};
  std::unordered_set<std::string> taken_names_;
  std::unordered_set<VarPtr> seen_index_vars_;
  std::unordered_set<VarPtr> seen_vars_;
  std::unordered_set<BufPtr> seen_bufs_;
  int level_ = 0;
};

StmtPtr LoopNest::sanitizeNames(StmtPtr s) {
  VarNameSanitizer r;
  s->accept_mutator(&r);
  return s;
}

class Vectorizer : public IRMutator {
 public:
  StmtPtr vectorize(ForPtr v) {
    StmtPtr body = v->body();
    VarPtr var = v->var();
    ExprPtr start = v->start();
    ExprPtr stop = v->stop();

    auto start_imm = intValue(start);
    auto stop_imm = intValue(stop);
    if (!start_imm) {
      // Can't vectorize due to non-constant loop start!
      success_ = false;
      return v;
    }

    if (!stop_imm) {
      // Can't vectorize due to non-constant loop stop!
      success_ = false;
      return v;
    }

    var_ = var;
    start_ = immLike(start, *start_imm);
    lanes_ = *stop_imm;

    StmtPtr new_body = body->accept_mutator(this);
    if (new_body == body) {
      // Vectorization failed!
      success_ = false;
      return v;
    }

    return new_body;
  }

  bool success() const {
    return success_;
  }

  ExprPtr mutate(const AddPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) + ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const SubPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) - ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const MulPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) * ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const DivPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) / ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const ModPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) % ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const AndPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) & ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const OrPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) | ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const XorPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) ^ ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const LshiftPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) << ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const RshiftPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(inputs[0]) >> ExprHandle(inputs[1]);
    });
  }

  ExprPtr mutate(const MaxPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return Max::make(
          ExprHandle(inputs[0]), ExprHandle(inputs[1]), v->propagate_nans());
    });
  }

  ExprPtr mutate(const MinPtr& v) override {
    std::vector<ExprPtr> inputs = {v->lhs(), v->rhs()};
    return try_vectorize(v, inputs, [&]() {
      return Min::make(
          ExprHandle(inputs[0]), ExprHandle(inputs[1]), v->propagate_nans());
    });
  }

  ExprPtr mutate(const CompareSelectPtr& v) override {
    std::vector<ExprPtr> inputs = {
        v->lhs(), v->rhs(), v->ret_val1(), v->ret_val2()};
    return try_vectorize(v, inputs, [&]() {
      return CompareSelect::make(
          ExprHandle(inputs[0]),
          ExprHandle(inputs[1]),
          ExprHandle(inputs[2]),
          ExprHandle(inputs[3]),
          v->compare_select_op(),
          v->bias());
    });
  }

  ExprPtr mutate(const BitCastPtr& v) override {
    std::vector<ExprPtr> inputs = {v->src_value()};
    return try_vectorize(v, inputs, [&]() {
      return BitCast::make(
          Dtype(v->dtype().scalar_type(), lanes_), ExprHandle(inputs[0]));
    });
  }

  ExprPtr mutate(const CastPtr& v) override {
    std::vector<ExprPtr> inputs = {v->src_value()};
    return try_vectorize(v, inputs, [&]() {
      return Cast::make(
          Dtype(v->dtype().scalar_type(), lanes_), ExprHandle(inputs[0]));
    });
  }

  ExprPtr mutate(const VarPtr& v) override {
    if (v == var_) {
      return Ramp::make(
                 ExprHandle(start_), ExprHandle(immLike(start_, 1)), lanes_)
          .node();
    }

    return v;
  }

  ExprPtr mutate(const RampPtr& v) override {
    ExprPtr base = v->base();
    ExprPtr stride = v->stride();

    ExprPtr base_new = base->accept_mutator(this);
    ExprPtr stride_new = stride->accept_mutator(this);

    if (base_new == base && stride_new == stride) {
      return v;
    }

    // Can't vectorize a Ramp!
    success_ = false;
    return v;
  }

  ExprPtr mutate(const LoadPtr& v) override {
    Dtype dtype(v->dtype().scalar_type(), lanes_);
    BufPtr buf = v->buf();
    std::vector<ExprPtr> inputs = {v->flat_index()};
    return try_vectorize(v, inputs, [&]() {
      return Load::make(dtype, BufHandle(buf), {ExprHandle(inputs[0])});
    });
  }

  ExprPtr mutate(const ReduceOpPtr& v) override {
    Dtype dtype(v->dtype().scalar_type(), lanes_);

    std::vector<ExprPtr> inputs = {v->body()};

    auto out = try_vectorize(v, inputs, [&]() {
      return ExprHandle(
          alloc<ReduceOp>(inputs[0], v->reduce_args(), v->reducer()));
    });
    return out;
  }

  ExprPtr mutate(const BroadcastPtr& v) override {
    ExprPtr val = v->value();
    ExprPtr new_val = val->accept_mutator(this);
    if (new_val == val) {
      return v;
    }

    // Can't vectorize a Broadcast!
    success_ = false;
    return v;
  }

  ExprPtr mutate(const IfThenElsePtr& v) override {
    ExprPtr condition = v->condition();
    ExprPtr new_condition = condition->accept_mutator(this);
    if (new_condition != condition) {
      // Can't vectorize an IfThenElse condition!
      success_ = false;
      return v;
    }

    std::vector<ExprPtr> inputs = {v->true_value(), v->false_value()};
    return try_vectorize(v, inputs, [&]() {
      return IfThenElse::make(
          ExprHandle(condition), ExprHandle(inputs[0]), ExprHandle(inputs[1]));
    });
  }

  ExprPtr mutate(const IntrinsicsPtr& v) override {
    std::vector<ExprPtr> inputs = v->params();
    return try_vectorize(v, inputs, [&]() {
      return ExprHandle(alloc<Intrinsics>(v->op_type(), inputs));
    });
  }

  StmtPtr mutate(const StorePtr& v) override {
    BufPtr buf = v->buf();
    std::vector<ExprPtr> inputs = {v->flat_index(), v->value()};
    return try_vectorize(v, inputs, [&]() {
      return Store::make(
          BufHandle(buf), {ExprHandle(inputs[0])}, ExprHandle(inputs[1]));
    });
  }

  StmtPtr mutate(const ForPtr& v) override {
    VarPtr var = v->var();
    ExprPtr start = v->start();
    ExprPtr stop = v->stop();
    LoopOptions loop_options = v->loop_options();

    ExprPtr new_start = start->accept_mutator(this);
    ExprPtr new_stop = stop->accept_mutator(this);

    if (new_start != start || new_stop != stop) {
      // Can't vectorize nested For with dependent loop bounds!
      success_ = false;
      return v;
    }

    StmtPtr body = v->body();
    StmtPtr new_body = body->accept_mutator(this);

    if (new_body == body) {
      return (ForPtr)v;
    }

    return alloc<For>(var, new_start, new_stop, new_body, loop_options);
  }

  StmtPtr mutate(const BlockPtr& v) override {
    // IRMutator does in-place mutations. But the logic in vectorization checks
    // for success by looking for a new stmt. So, we override the in-place
    // mutations and create a clone here if any of its statements change.
    // TODO: Can we change the logic of vectorizer so that we don't need this?
    bool any_change = false;
    std::vector<StmtPtr> stmts;
    for (const StmtPtr& stmt : *v) {
      StmtPtr stmt_new = stmt->accept_mutator(this);
      if (stmt != stmt_new) {
        any_change = true;
      } else {
        stmt_new = Stmt::clone(stmt);
      }
      if (stmt_new) {
        stmts.push_back(stmt_new);
      }
    }
    if (any_change) {
      return alloc<Block>(stmts);
    }
    return v;
  }

  template <typename T>
  ExprPtr try_vectorize(ExprPtr e, std::vector<ExprPtr>& inputs, T&& vec_ctor) {
    bool vectorize = vectorize_inputs(inputs);
    if (vectorize) {
      return vec_ctor().node();
    }

    return e;
  }

  template <typename T>
  StmtPtr try_vectorize(StmtPtr s, std::vector<ExprPtr>& inputs, T&& vec_ctor) {
    bool vectorize = vectorize_inputs(inputs);
    if (vectorize) {
      return vec_ctor();
    }

    return s;
  }

  bool vectorize_inputs(std::vector<ExprPtr>& inputs) {
    bool any_vectorized = false;
    std::vector<ExprPtr> new_inputs;

    // Attempt to vectorize each input.
    for (ExprPtr& in : inputs) {
      ExprPtr new_in = in->accept_mutator(this);
      new_inputs.push_back(new_in);
      if (new_in != in) {
        any_vectorized = true;
      }
    }

    // If none of them vectorized, then don't vectorize this.
    if (!any_vectorized) {
      return false;
    }

    // Insert broadcasts for any inputs that weren't vectorized.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == new_inputs[i]) {
        inputs[i] = Broadcast::make(ExprHandle(inputs[i]), lanes_).node();
      } else {
        inputs[i] = new_inputs[i];
      }
    }

    // And then vectorize this node.
    return true;
  }

  VarPtr var_ = nullptr;
  int64_t lanes_ = 0;
  ExprPtr start_ = nullptr;
  bool success_ = true;
};

bool LoopNest::vectorize(const ForPtr& f) {
  BlockPtr b = to<Block>(f->get_parent());
  if (!b) {
    return false;
  }

  // Can't vectorize reduction axes.
  auto reductions = NodeFinder<ReduceOp>::find(f);
  for (const auto& r : reductions) {
    if (std::find(r->reduce_args().begin(), r->reduce_args().end(), f->var()) !=
        r->reduce_args().end()) {
      return false;
    }
  }

  Vectorizer v;
  StmtPtr new_f = nullptr;
  new_f = Stmt::clone(f);
  normalize(to<For>(new_f));
  new_f = FlattenIndexes(new_f);
  new_f = v.vectorize(to<For>(new_f));
  if (!v.success()) {
    // We clone f before vectorizing. So, any partial vectorization will
    // have modified the clone. In case of an exception, we can continue
    // using f.
    new_f = f;
  }

  if (new_f != f) {
    b->replace_stmt(f, IRSimplifier::simplify(new_f));
    return true;
  }

  // Vectorization was not successful.
  return false;
}

void LoopNest::initialize(
    const std::vector<Tensor>& output_tensors,
    const std::vector<Tensor>& tensors_to_compute) {
  for (const auto& t : output_tensors) {
    output_bufs_.insert(t.buf());
  }

  std::vector<StmtPtr> loops;
  for (const Tensor& t : tensors_to_compute) {
    StmtPtr loop = t.stmt();
    if (loop->get_parent()) {
      std::cerr << "Error: creating a loopnest from already used Tensors\n";
      loops = {};
      break;
    }
    // Flatten initializers.
    if (BlockPtr block = to<Block>(loop)) {
      for (const auto& s : block->stmts()) {
        block->remove_stmt(s);
        loops.push_back(s);
      }
    } else {
      loops.push_back(loop);
    }
  }

  root_stmt_ = alloc<Block>(loops);
}

class FunctionInliner : public IRMutator {
 public:
  FunctionInliner(StorePtr producer, std::unordered_set<BufPtr> outputs)
      : buf_(producer->buf()),
        producer_(std::move(producer)),
        outputs_(std::move(outputs)) {
    for (const auto& i : producer_->indices()) {
      if (auto index_var = to<Var>(i)) {
        index_vars_.insert(index_var);
        producer_index_vars_.push_back(index_var);
      } else {
        // If the index can be a constant, then that dimension must have size 1
        // (since we don't support in-place writes). Resolves issue 52581.
        auto index_val = evalInt(i);
        if (!index_val || *index_val != 0) {
          success_ = false;
          break;
        }
        producer_index_vars_.push_back(nullptr);
      }
    }
  }

  bool success() const {
    return success_;
  }

 private:
  ExprPtr mutate_loads(const BufPtr& buf, std::vector<ExprPtr> dims) {
    std::vector<VarPtr> index_vars;
    if (buf->ndim() != producer_index_vars_.size()) {
      // Dimensions of producer and consumer expressions do not match in inliner
      // in the fuser
      success_ = false;
      return nullptr;
    }
    for (const auto i : c10::irange(buf->ndim())) {
      VarPtr func_callee_arg = producer_index_vars_.at(i);
      ExprPtr func_caller_param = dims.at(i);
      if (func_callee_arg == nullptr) {
        continue;
      }
      auto iter = inline_mapping_.find(func_callee_arg);
      if (iter != inline_mapping_.end()) {
        // Duplicated variables
        success_ = false;
        return nullptr;
      }
      // Add a mapping for each function parameter to it's source name.
      inline_mapping_[func_callee_arg] = func_caller_param;
      GRAPH_DEBUG(
          "ComputeInline: Inline mapping: ",
          std::to_string(func_callee_arg),
          " -> ",
          std::to_string(func_caller_param));
      index_vars.push_back(func_callee_arg);
    }

    // Call the actual replacement.
    ExprPtr body = producer_->value();
    GRAPH_DEBUG("ComputeInline: Before rewriting body: ", std::to_string(body));
    ExprPtr result = Expr::clone(body)->accept_mutator(this);
    GRAPH_DEBUG(
        "ComputeInline: After rewriting body: ", std::to_string(result));

    // Remove the mappings we created for this function parameters.
    for (const auto& v : index_vars) {
      for (auto& pair : random_bindings_) {
        if (pair.second.erase(v)) {
          ExprPtr inlined = inline_mapping_[v];
          for (const auto& nv : VarFinder::find(inlined)) {
            pair.second.insert(nv);
          }
        }
      }
      GRAPH_DEBUG("ComputeInline: Inline mapping: erasing", std::to_string(v));
      inline_mapping_.erase(v);
    }
    return result;
  }

  ExprPtr mutate(const LoadPtr& v) override {
    if (!success()) {
      return v;
    }
    BufPtr buf = v->buf();
    if (buf != buf_) {
      return IRMutator::mutate(v);
    }

    if (v->indices().size() != buf->ndim()) {
      // Number of indices doesn't match buf rank in the fuser
      success_ = false;
      return v;
    }
    auto result = mutate_loads(buf, v->indices());
    if (!result) {
      // If we don't inline successfully return the given load.
      success_ = false;
      return v;
    }
    return result;
  }

  // Replace the target variable with the caller expressions.
  ExprPtr mutate(const VarPtr& v) override {
    if (!success()) {
      return v;
    }
    auto iter = inline_mapping_.find(v);
    if (iter == inline_mapping_.end()) {
      return v;
    } else {
      ExprPtr expr = iter->second;
      // Continue to transform the value from the lookup table.
      return expr->accept_mutator(this);
    }
  }

  // Handle random intrinsics which should be cached.
  ExprPtr mutate(const IntrinsicsPtr& v) override {
    if (!success()) {
      return v;
    }
    if (!in_producer_ || v->op_type() != kRand) {
      return IRMutator::mutate(v);
    }

    // Create a new Let Statement for the random variable, which we can refer
    // to multiple times and resolve the same value (ie. store it in a scalar
    // rather than the Tensor).
    const std::string& name = buf_->name_hint();
    VarPtr new_var = alloc<Var>(name, v->dtype());
    random_bindings_[alloc<Let>(new_var, v)] = index_vars_;
    GRAPH_DEBUG(
        "ComputeInline: created random bindings for ", std::to_string(new_var));
    return new_var;
  }

  // Remove the buffer write from the inlined function.
  StmtPtr mutate(const StorePtr& v) override {
    if (!success()) {
      return v;
    }
    // If the buf_ is in the outputs set, keep its statement intact. Otherwise,
    // remove it.
    if (v == producer_ && !outputs_.count(buf_)) {
      in_producer_ = true;
      producer_ = to<Store>(IRMutator::mutate(v));
      if (!producer_) {
        // Producer statement for output buf should remain non-null in the fuser
        success_ = false;
        return v;
      }
      in_producer_ = false;
      return nullptr;
    } else {
      return IRMutator::mutate(v);
    }
  }

  // Any Random Intrinsics that were turned into vars must be inserted here.
  StmtPtr mutate(const BlockPtr& v) override {
    if (!success()) {
      return v;
    }
    std::vector<StmtPtr> stmts;
    for (const StmtPtr& stmt : *v) {
      StmtPtr stmt_new = stmt->accept_mutator(this);
      if (!stmt_new) {
        continue;
      }

      if (stmt == stmt_new) {
        stmt_new = Stmt::clone(stmt);
      }

      stmts.push_back(stmt_new);
    }

    return Block::make(stmts);
  }

  StmtPtr mutate(const ForPtr& v) override {
    if (!success()) {
      return v;
    }
    ForPtr res = to<For>(IRMutator::mutate(v));
    if (!res) {
      return nullptr;
    }

    // Find any random bindings that should be defined in this loops body.
    std::vector<LetPtr> bindings_this_loop;
    VarPtr fv = v->var();
    for (auto& pair : random_bindings_) {
      auto& index_var = pair.second;
      if (index_var.erase(fv)) {
        bindings_this_loop.push_back(pair.first);
      }
    }

    for (const auto& l : bindings_this_loop) {
      res->body()->prepend_stmt(l);
      random_bindings_.erase(l);
    }
    return res;
  }

 private:
  BufPtr buf_;
  StorePtr producer_;

  // Index Vars present in the producer.
  std::unordered_set<VarPtr> index_vars_;
  std::vector<VarPtr> producer_index_vars_;

  std::unordered_map<VarPtr, ExprPtr> inline_mapping_;

  // In the producer's scope - we need to bind any calls to rand().
  bool in_producer_ = false;
  std::unordered_map<LetPtr, std::unordered_set<VarPtr>> random_bindings_;
  std::unordered_set<BufPtr> outputs_;
  bool success_ = true;
};

static StmtPtr computeInlineImpl(
    const BufPtr& b,
    const StmtPtr& stmt,
    const std::unordered_set<BufPtr>& output_bufs) {
  // If buf is used or defined in an ExternalCall, we cannot inline it
  auto buf_load_store_uses = findLoadOrStoreUses(stmt);
  if (!buf_load_store_uses.count(b)) {
    return nullptr;
  }
  for (auto& use : buf_load_store_uses.at(b)) {
    StmtPtr s = use.s;
    if (to<ExternalCall>(s) || to<ExternalCallWithAlloc>(s)) {
      return nullptr;
    }
  }

  // Find producers.
  StorePtr relevant_store{nullptr};
  auto stores = NodeFinder<Store>::find(stmt);
  for (const auto& s : stores) {
    if (s->buf() == b) {
      auto reductions = NodeFinder<ReduceOp>::find(s);
      if (!reductions.empty()) {
        // Cannot inline a reduction computation
        return nullptr;
      }
      if (relevant_store != nullptr) {
        // Cannot inline Buf with multiple Tensors
        return nullptr;
      }
      relevant_store = s;
    }
  }

  if (!relevant_store) {
    // Cannot find a relevant store to inline a buf in the fuser
    return nullptr;
  }

  GRAPH_DEBUG("ComputeInline: Def: ", std::to_string(relevant_store));
  FunctionInliner inliner(relevant_store, output_bufs);
  auto result = stmt->accept_mutator(&inliner);
  if (inliner.success()) {
    return result;
  }
  return nullptr;
}

bool LoopNest::computeInline(const BufPtr& b) {
  // Inlining may not always be successful. Since all mutations now happen
  // in-place, an unsuccessful inlining transformation might leave the IR
  // in an invalid state. To get around this problem, we clone the root stmt,
  // try inlining on the clone, and if it succeeds, we proceed to perform
  // inlining on the actual root stmt. This way the root stmt will always be
  // in a valid state.
  auto stmt_copy = Stmt::clone(root_stmt_);
  auto try_inline = computeInlineImpl(b, stmt_copy, output_bufs_);
  if (!try_inline) {
    return false;
  }
  root_stmt_ = computeInlineImpl(b, root_stmt_, output_bufs_);
  return true;
}

bool LoopNest::computeInline(const StmtPtr& s) {
  auto s_store = to<Store>(s);
  if (s_store == nullptr) {
    // Could not find buffer producer to inline
    return false;
  }
  return computeInline(s_store->buf());
}

// inlining buffers with multiple uses can create duplicated work, which can
// slow down cpu code generation but is enabled on gpu because it avoids
// difficult synchronization logic across blocks. Inlining trivial reads does
// not duplicate work
void LoopNest::inlineIntermediateBufs(bool allow_duplicated_work) {
  std::unordered_set<BufPtr> bufs_to_inline;

  auto intermediate_bufs = getIntermediateBufs();
  if (allow_duplicated_work) {
    bufs_to_inline.insert(intermediate_bufs.begin(), intermediate_bufs.end());
  } else {
    auto buf_load_store_uses = findLoadOrStoreUses(root_stmt_);
    auto input_bufs = getInputBufs();

    for (const auto& buf : intermediate_bufs) {
      TORCH_INTERNAL_ASSERT(
          buf_load_store_uses.count(buf),
          buildErrorMessage(
              "Could not find uses of buf '" + buf->name_hint() +
              "' in the fuser."));
      std::vector<BufLoadOrStoreUse>& uses = buf_load_store_uses[buf];
      auto stores = c10::filter(
          uses, [](const BufLoadOrStoreUse& use) { return use.isStore; });

      // if the intermediate is the buffer formed from reading in the input
      // tensors, always inline, bc we are not duplicating any work
      // and avoiding an intermediary buffer
      if (stores.size() == 1) {
        if (auto store = to<Store>(stores[0].s)) {
          auto input_as_load = to<Load>(store->value());
          if (input_as_load && input_bufs.count(input_as_load->buf())) {
            bufs_to_inline.insert(buf);
            continue;
          }
        } else {
          // If S is not a store, it must be an ExternalCall.
          TORCH_INTERNAL_ASSERT(
              to<ExternalCall>(stores[0].s) ||
                  to<ExternalCallWithAlloc>(stores[0].s),
              buildErrorMessage(
                  "Expected stmt: " + std::to_string(stores[0].s) +
                  "\nto be either a Store or an ExternalCall in the fuser."));
        }
      }

      // all bufs will have at least one store (if they have > 1 they can't be
      // inlined anyway)
      size_t reads = uses.size() - 1;
      // if only one read, we can inline it without duplicating work
      if (reads <= 1) {
        bufs_to_inline.insert(buf);
      }
    }
  }

  if (allow_duplicated_work) {
    bufs_to_inline.insert(output_bufs_.begin(), output_bufs_.end());
  }

  for (const auto& b : bufs_to_inline) {
    computeInline(b);
  }
}

// TODO: Unify with DepTracker
class LoadOrStoreUseFinder : public IRVisitor {
 public:
  std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> findUses(
      const StmtPtr& s) {
    uses_.clear();
    s->accept(this);
    return uses_;
  }

 private:
  void visit(const StorePtr& v) override {
    if (stores_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({(StmtPtr)v, true});
    }
    last_stmt_ = (StmtPtr)v;
    IRVisitor::visit(v);
  }

  void visit(const ExternalCallPtr& v) override {
    if (stores_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({(StmtPtr)v, true});
    }
    last_stmt_ = (StmtPtr)v;

    for (const BufPtr& input_buf : v->buf_args()) {
      if (loads_[input_buf].insert(last_stmt_).second) {
        uses_[input_buf].push_back({last_stmt_, false});
      }
    }

    IRVisitor::visit(v);
  }

  void visit(const ExternalCallWithAllocPtr& v) override {
    for (const auto& out_buf : v->buf_out_args()) {
      if (stores_[out_buf].insert(last_stmt_).second) {
        uses_[out_buf].push_back({(StmtPtr)v, true});
      }
    }
    last_stmt_ = (StmtPtr)v;

    for (const auto& input_buf : v->buf_args()) {
      if (loads_[input_buf].insert(last_stmt_).second) {
        uses_[input_buf].push_back({last_stmt_, false});
      }
    }

    IRVisitor::visit(v);
  }

  void visit(const LoadPtr& v) override {
    if (loads_[v->buf()].insert(last_stmt_).second) {
      uses_[v->buf()].push_back({last_stmt_, false});
    }
    IRVisitor::visit(v);
  }

  StmtPtr last_stmt_ = nullptr;
  std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> uses_;

  // Sets of loads and stores in order to keep the results unique
  std::unordered_map<BufPtr, std::unordered_set<StmtPtr>> loads_;
  std::unordered_map<BufPtr, std::unordered_set<StmtPtr>> stores_;
};

std::unordered_map<BufPtr, std::vector<BufLoadOrStoreUse>> findLoadOrStoreUses(
    const StmtPtr& s) {
  LoadOrStoreUseFinder uf;
  return uf.findUses(s);
}

class ContainedStmtsFinder : public IRVisitor {
 public:
  // Simply list all Stores and Block that are children of the given stmt
  const std::unordered_set<StmtPtr>& findContainedStmts(const StmtPtr& s) {
    contained_.clear();
    s->accept(this);
    return contained_;
  }

 private:
  void visit(const StorePtr& v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }
  void visit(const ExternalCallPtr& v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }
  void visit(const ExternalCallWithAllocPtr& v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }
  void visit(const BlockPtr& v) override {
    contained_.insert((StmtPtr)v);
    IRVisitor::visit(v);
  }

  std::unordered_set<StmtPtr> contained_;
};

class StmtDeleter : public IRMutator {
 public:
  StmtDeleter(const std::unordered_set<StmtPtr>& targets) : targets_(targets) {}

 private:
  StmtPtr mutate(const BlockPtr& v) override {
    std::vector<StmtPtr> stmts;

    for (const auto& s : v->stmts()) {
      if (targets_.count(s) == 0) {
        StmtPtr ns = s->accept_mutator(this);
        if (ns) {
          stmts.push_back(Stmt::clone(ns));
        }
      }
    }

    return Block::make(stmts);
  }

  const std::unordered_set<StmtPtr>& targets_;
};

void LoopNest::eliminateDeadStores() {
  using namespace analysis;
  MemDependencyChecker checker(getInputBufs(), getOutputBufs());
  root_stmt_->accept(&checker);

  std::unordered_set<StmtPtr> deadStores;
  std::vector<std::shared_ptr<AccessInfo>> outputAccesses;
  for (const auto& o : getOutputBufs()) {
    outputAccesses.push_back(checker.output(o));
  }

  for (auto& info : checker.getHistory()) {
    if (!info->isWrite()) {
      continue;
    }
    bool found = false;

    for (auto& output : outputAccesses) {
      if (checker.dependsIndirectly(output, info)) {
        found = true;
        break;
      }
    }

    if (!found) {
      deadStores.insert(info->stmt());
    }
  }

  StmtDeleter deleter(deadStores);
  root_stmt_ = root_stmt_->accept_mutator(&deleter);
}

void LoopNest::prepareForCodegen() {
  // Expand reduction ops.
  ReductionExpander reduceExpander;
  root_stmt_ = reduceExpander.expand(root_stmt_);

  root_stmt_ = FlattenIndexes(root_stmt_);
}

namespace {

// This is extended from IRCloner instead of IRMutator because we want all
// the rest of the IR nodes (the ones not touched directly) to be cloned.
class IfThenElseReplacer : public IRCloner {
 public:
  IfThenElseReplacer(IfThenElsePtr to_replace, ExprPtr new_expr)
      : to_replace_(std::move(to_replace)), new_expr_(std::move(new_expr)) {}

  ExprPtr mutate(const IfThenElsePtr& i) override {
    if (i == to_replace_) {
      return new_expr_;
    }
    return IRCloner::mutate(i);
  }

 private:
  IfThenElsePtr to_replace_;
  ExprPtr new_expr_;
};

// Check if the given condition is optimizable.
// Specifically, this function looks for the following pattern:
//    "var < expr"
//
// If this pattern is found, then this function:
//   * sets `cond_var` to `var`,
//   * sets `compared_value` to `expr`, and
//   * returns true.
bool isConditionOptimizable(
    const ExprPtr& condition,
    VarPtr* cond_var,
    ExprPtr* compared_value) {
  auto cs = to<CompareSelect>(condition);
  if (cs && cs->compare_select_op() == kLT) {
    auto var = to<Var>(cs->lhs());
    if (var) {
      *cond_var = var;
      *compared_value = cs->rhs();
      return true;
    }
  }
  return false;
}

// Checks if the given if-then-else expression is a conditional that is
// generated from `aten::cat`.
//
// The expected format of conditionals is:
//     IfThenElse(var < val1? 1 : 0,
//       IfThenElse (var < val2? 1 : 0,
//         IfThenElse (var < val3? 1 : 0,
//           sub-expr1,
//           sub-expr2),
//         sub-expr3),
//       sub-expr4)
//
// If such a conditional is found, this function also sets:
//   * cond_var to the condition variable found in this expression.
//   * comp_values to the list of compared values in the condition expressions.
//   * sub_exprs to the list of sub-expressions that are the result of this
//     if-then-else expression.
bool isConditionalFromCat(
    const IfThenElsePtr& ite,
    VarPtr* cond_var,
    std::vector<ExprPtr>* comp_values,
    std::vector<ExprPtr>* sub_exprs) {
  VarPtr var = nullptr;
  ExprPtr comp_value;
  if (isConditionOptimizable(ite->condition(), &var, &comp_value)) {
    if (*cond_var == nullptr) {
      *cond_var = var;
    } else if (*cond_var != var) {
      // Different condition variables found in nested if-then-else
      // expressions. Can not optimize such cases.
      return false;
    }
    auto true_ite = to<IfThenElse>(ite->true_value());
    if (true_ite) {
      if (!isConditionalFromCat(true_ite, cond_var, comp_values, sub_exprs)) {
        return false;
      }
    } else {
      sub_exprs->push_back(ite->true_value());
    }
    auto false_ite = to<IfThenElse>(ite->false_value());
    if (false_ite) {
      return false;
    }
    comp_values->push_back(comp_value);
    sub_exprs->push_back(ite->false_value());
    return true;
  }
  return false;
}

bool areConstantsAndSorted(const std::vector<ExprPtr>& comp_values) {
  std::vector<int> comp_consts;
  comp_consts.reserve(comp_values.size());
  for (const auto& c : comp_values) {
    if (!c->isConstant()) {
      return false;
    }
    comp_consts.push_back(immediateAs<int>(c));
  }
  return std::is_sorted(comp_consts.begin(), comp_consts.end());
}

} // namespace

bool LoopNest::optimizeConditionals() {
  // Consider every store in the root_stmt_ and try to optimize the
  // conditionals in that store.
  auto stores = NodeFinder<Store>::find(root_stmt_);
  std::unordered_set<ForPtr> split_fors;
  for (const auto& store : stores) {
    VarPtr cond_var = nullptr;
    // `comp_values` represent the list of compared values that will be
    // collected as we check for the expected pattern. Since that will
    // only include the RHS of the conditions in the if-then-else expressions
    // we need to start with `0` which is the initial bound, given that we
    // only handle normalized loops (check for this is done below).
    std::vector<ExprPtr> comp_values;
    std::vector<ExprPtr> sub_exprs;
    auto ifthenelse_exprs = NodeFinder<IfThenElse>::find(store);
    if (ifthenelse_exprs.empty()) {
      continue;
    }
    // We only check if the first if-then-else expression in this store
    // corresponds to a conditional of the required format. If there are more
    // than one such conditional, optimizing them requires checking if the
    // conditions are exactly the same across them and handling all of them
    // together. Currently, this is not handled.
    if (!isConditionalFromCat(
            ifthenelse_exprs.front(), &cond_var, &comp_values, &sub_exprs)) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        !comp_values.empty(),
        buildErrorMessage(
            "Expected at least one expression in optimizeConditional in the fuser."));
    comp_values.insert(comp_values.begin(), immLike(comp_values[0], 0));

    auto fors = getLoopStmtsFor(store);
    if (cond_var != fors.back()->var()) {
      // Currently, we only handle the case where the condition variable
      // is the same as the inner-most loop variable.
      // TODO: Handle all other cases here.
      //
      // In order to handle all other cases, the method `clone_and_replace`
      // called below to clone the body of the loop with a new store needs
      // to recursively handle cloning of the loops and other blocks it
      // contains.
      continue;
    }

    auto for_to_split = fors.back();
    if (!LoopNest::isNormalized(for_to_split)) {
      // Do not optimize this conditional since the condition variable
      // refers to a loop that is not normalized.
      continue;
    }
    if (split_fors.count(for_to_split)) {
      // This loop has already been split while optimizing conditionals
      // earlier.
      //
      // Optimizing multiple conditionals that require splitting the same loop
      // is tricky. It requires checking if the conditions are exactly the same
      // across them and handling all of them together by splitting the loop
      // exactly once.
      //
      // Currently, this case is not supported.
      continue;
    }
    split_fors.insert(for_to_split);

    // `comp_values` needs to include the end bound, which is `for_to_split`
    // stop value.
    comp_values.push_back(for_to_split->stop());

    // Check if all `comp_values` are constants and they are sorted.
    if (!areConstantsAndSorted(comp_values)) {
      continue;
    }

    // Remove all the if-then-else expressions from this store and create
    // one loop per sub-expression.
    std::vector<StmtPtr> split_loops;
    auto cond_to_replace = ifthenelse_exprs.front();
    for (size_t i = 0; i < sub_exprs.size(); ++i) {
      IfThenElseReplacer ifthenelseReplacer(cond_to_replace, sub_exprs[i]);
      auto new_store = store->accept_mutator(&ifthenelseReplacer);
      auto new_for_body =
          for_to_split->body()->clone_and_replace(store, new_store);
      auto new_for = alloc<For>(
          for_to_split->var(),
          comp_values[i],
          comp_values[i + 1],
          new_for_body);
      LoopNest::normalize(new_for);
      split_loops.push_back(new_for);
    }
    auto par = to<Block>(for_to_split->get_parent());
    par->replace_stmt(for_to_split, alloc<Block>(split_loops));
  }
  root_stmt_ = IRSimplifier::simplify(root_stmt_);
  return true;
}

void LoopNest::vectorizeInnerLoops() {
  std::vector<ForPtr> innerLoops;
  std::vector<ForPtr> worklist;

  // Find outer-most For loops
  if (ForPtr rootF = to<For>(root_stmt_)) {
    worklist.push_back(rootF);
  } else if (BlockPtr body = to<Block>(root_stmt_)) {
    std::vector<BlockPtr> blocks = {body};
    while (!blocks.empty()) {
      BlockPtr b = blocks.back();
      blocks.pop_back();

      for (const StmtPtr& s : *b) {
        if (const ForPtr& f = to<For>(s)) {
          worklist.push_back(f);
        } else if (BlockPtr b2 = to<Block>(s)) {
          blocks.push_back(b2);
        }
      }
    }
  }

  // Traverse the For loop nest find inner-most loops, which are
  // vectorization candidates.
  while (!worklist.empty()) {
    ForPtr f = worklist.back();
    worklist.pop_back();

    bool containsSubLoops = false;
    if (BlockPtr body = to<Block>(f->body())) {
      for (const StmtPtr& s2 : *body) {
        if (const ForPtr& f2 = to<For>(s2)) {
          containsSubLoops = true;
          worklist.push_back(f2);
        }
      }
    }

    if (!containsSubLoops) {
      innerLoops.push_back(f);
    }
  }

  // vectorize inner loops.
  for (const ForPtr& loop : innerLoops) {
    ForPtr split1;
    ForPtr tail1;

    static const int kBodyVectorWidth = 8;
    splitWithTail(loop, kBodyVectorWidth, &split1, &tail1);
    vectorize(split1);

    if (tail1) {
      ForPtr split2;
      ForPtr tail2;
      static const int kTailVectorWidth = 4;
      splitWithTail(tail1, kTailVectorWidth, &split2, &tail2);
      vectorize(split2);
    }
  }
}

void LoopNest::sliceHead(
    const ForPtr& f,
    int factor,
    ForPtr* head,
    ForPtr* tail) {
  if (intValue(f->start()) && intValue(f->stop())) {
    auto start_val = *intValue(f->start());
    auto stop_val = *intValue(f->stop());
    auto size_val = stop_val - start_val;
    if (factor >= size_val) {
      *head = f;
      *tail = nullptr;
      return;
    }
  }

  if (!f) {
    throw malformed_input("sliceHead attempted on null loop");
  }

  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    throw malformed_input("sliceHead attempted on loop with no parent");
  }

  ExprPtr head_end = alloc<Min>(
      alloc<Add>(f->start(), immLike(f->stop(), factor)), f->stop(), true);
  *head = alloc<For>(f->var(), f->start(), head_end, Stmt::clone(f->body()));
  p->insert_stmt_before(*head, f);

  f->set_start(head_end);
  *tail = f;

  if (f->loop_options().is_gpu_block_index() ||
      f->loop_options().is_gpu_thread_index()) {
    LoopNest::normalize(*tail);
  }
}
void LoopNest::sliceHead(const ForPtr& f, int factor) {
  ForPtr head, tail;
  sliceHead(f, factor, &head, &tail);
}

void LoopNest::sliceTail(
    const ForPtr& f,
    int factor,
    ForPtr* head,
    ForPtr* tail) {
  if (intValue(f->start()) && intValue(f->stop())) {
    auto start_val = *intValue(f->start());
    auto stop_val = *intValue(f->stop());
    auto size_val = stop_val - start_val;
    if (factor >= size_val) {
      *head = nullptr;
      *tail = f;
      return;
    }
  }

  if (!f) {
    throw malformed_input("sliceTail attempted on null loop");
  }

  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    throw malformed_input("sliceTail attempted on loop with no parent");
  }

  ExprPtr tail_start = alloc<Max>(
      f->start(), alloc<Sub>(f->stop(), immLike(f->stop(), factor)), true);
  *tail = alloc<For>(f->var(), tail_start, f->stop(), Stmt::clone(f->body()));
  p->insert_stmt_after(*tail, f);

  f->set_stop(tail_start);
  *head = f;

  if (f->loop_options().is_gpu_block_index() ||
      f->loop_options().is_gpu_thread_index()) {
    LoopNest::normalize(*head);
  }
}
void LoopNest::sliceTail(const ForPtr& f, int factor) {
  ForPtr head, tail;
  sliceTail(f, factor, &head, &tail);
}

void LoopNest::splitWithTail(const ForPtr& f, int factor) {
  ForPtr inner, tail;
  splitWithTail(f, factor, &inner, &tail);
}

void LoopNest::splitWithTail(
    const ForPtr& f,
    int factor,
    ForPtr* inner,
    ForPtr* tail) {
  if (!f) {
    throw malformed_input("splitWithTail attempted on null loop");
  }

  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    throw malformed_input("splitWithTail attempted on loop with no parent");
  }

  // Normalize the loop to simplify start and stop bound computation
  normalize(f);

  bool tail_is_needed = true;
  if (intValue(f->start()) && intValue(f->stop())) {
    auto const start_val = *intValue(f->start());
    auto const stop_val = *intValue(f->stop());
    auto const size_val = stop_val - start_val;
    auto const tail_size = size_val % factor;
    if (tail_size == 0) {
      tail_is_needed = false;
    }
  }

  ExprPtr factor_expr = immLike(f->stop(), factor);
  ExprPtr size = alloc<Sub>(f->stop(), f->start());
  ExprPtr split_count = alloc<Div>(size, factor_expr);
  ExprPtr tail_size = alloc<Mod>(size, factor_expr);

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  VarPtr i_inner = alloc<Var>(loop_var_name + "_inner", loop_var_dtype);
  VarPtr i_outer = alloc<Var>(loop_var_name + "_outer", loop_var_dtype);

  // x -> x.outer * inner.size + x.inner
  ExprPtr combined_index1 =
      alloc<Add>(alloc<Mul>(i_outer, factor_expr), i_inner);

  if (tail_is_needed) {
    VarPtr i_tail = alloc<Var>(loop_var_name + "_tail", loop_var_dtype);
    // x -> x.tail + outer.size * inner.size
    ExprPtr combined_index2 =
        alloc<Add>(i_tail, alloc<Mul>(split_count, factor_expr));

    StmtPtr body_tail =
        SubstituteInClone(f->body(), {{f->var(), combined_index2}});
    *tail = alloc<For>(i_tail, immLike(tail_size, 0), tail_size, body_tail);

    p->insert_stmt_after(*tail, f);
  } else {
    *tail = nullptr;
  }

  StmtPtr body_inner =
      Substitute(f->removeBody(), {{f->var(), combined_index1}});

  *inner =
      alloc<For>(i_inner, immLike(factor_expr, 0), factor_expr, body_inner);
  // The input loop `f` will be the outer loop after split.
  f->set_var(i_outer);
  f->set_start(immLike(split_count, 0));
  f->set_stop(split_count);
  f->set_body(*inner);
}

void LoopNest::splitWithMask(const ForPtr& f, int factor) {
  ForPtr inner;
  splitWithMask(f, factor, &inner);
}

void LoopNest::splitWithMask(const ForPtr& f, int factor, ForPtr* inner) {
  BlockPtr p = to<Block>(f->get_parent());
  if (!p) {
    std::cerr << "Parent is not a Block!\n";
    return;
  }

  bool tail_is_needed = true;
  ExprPtr start = IRSimplifier::simplify(f->start());
  ExprPtr stop = IRSimplifier::simplify(f->stop());
  if (start->isConstant() && stop->isConstant()) {
    auto start_val = *intValue(start);
    auto stop_val = *intValue(stop);
    auto size_val = stop_val - start_val;
    auto tail_size = size_val % factor;
    if (tail_size == 0) {
      tail_is_needed = false;
    }
  }

  auto factor_expr = immLike(f->stop(), factor);
  ExprPtr size = alloc<Sub>(f->stop(), f->start());
  // split_count = (size + factor - 1) / factor
  ExprPtr split_count = alloc<Div>(
      alloc<Sub>(alloc<Add>(size, factor_expr), immLike(size, 1)), factor_expr);

  const std::string& loop_var_name = f->var()->name_hint();
  Dtype loop_var_dtype = f->var()->dtype();

  VarPtr i_inner = alloc<Var>(loop_var_name + "_inner", loop_var_dtype);
  VarPtr i_outer = alloc<Var>(loop_var_name + "_outer", loop_var_dtype);

  // x -> x.outer * inner.size + x.inner
  ExprPtr combined_index =
      alloc<Add>(alloc<Mul>(i_outer, factor_expr), i_inner);

  StmtPtr body_inner = f->removeBody();
  // TODO: is it ok that we're doing it eagerly? In the other implementation we
  // are only materializing predicates at the last, lowering, step.
  if (tail_is_needed) {
    auto start = intValue(f->start());
    if (!start || *start != 0) {
      throw unimplemented_lowering();
    }

    ExprPtr predicate =
        CompareSelect::make(ExprHandle(f->var()), ExprHandle(f->stop()), kLT)
            .node();
    body_inner = Cond::make(ExprHandle(predicate), body_inner, nullptr);
  }
  body_inner = Substitute(body_inner, {{f->var(), combined_index}});

  *inner =
      alloc<For>(i_inner, immLike(factor_expr, 0), factor_expr, body_inner);
  // The input loop `f` will be the outer loop after split.
  f->set_var(i_outer);
  f->set_start(immLike(split_count, 0));
  f->set_stop(split_count);
  f->set_body(*inner);
}

std::vector<ForPtr> LoopNest::distributeLoop(
    const ForPtr& loop,
    const std::unordered_set<StmtPtr>& pivots) {
  TORCH_INTERNAL_ASSERT(
      loop,
      buildErrorMessage(
          "Expected non-null loop in distributeLoop in the fuser."));
  auto root = loop->get_parent();
  if (root == nullptr) {
    throw malformed_input("Loop without parent: ", loop);
  }
  auto root_block = to<Block>(root);
  if (root_block == nullptr) {
    throw malformed_input(
        "Loop's parent must be a Block, instead found ", root);
  }

  // Extract bodies for all the loops after distribution.
  std::vector<BlockPtr> new_loop_bodies;
  auto new_loop_body = alloc<Block>(std::vector<StmtPtr>({}));
  whil
```



## High-Level Overview


This C++ file contains approximately 12 class(es)/struct(s) and 181 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `analysis`, `bool`

**Classes/Structs**: `IndexFlattener`, `VarNameSanitizer`, `Vectorizer`, `FunctionInliner`, `LoadOrStoreUseFinder`, `ContainedStmtsFinder`, `StmtDeleter`, `IfThenElseReplacer`, `for`, `LoopComputeAtRewriter`, `CacheReplacer`, `the`, `the`, `RfactorStoreRewriter`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/tensorexpr/loopnest.h`
- `algorithm`
- `iostream`
- `stdexcept`
- `unordered_map`
- `unordered_set`
- `utility`
- `vector`
- `c10/util/Logging.h`
- `c10/util/irange.h`
- `ATen/core/functional.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/tensorexpr/analysis.h`
- `torch/csrc/jit/tensorexpr/bounds_inference.h`
- `torch/csrc/jit/tensorexpr/eval.h`
- `torch/csrc/jit/tensorexpr/expr.h`
- `torch/csrc/jit/tensorexpr/ir.h`
- `torch/csrc/jit/tensorexpr/ir_cloner.h`
- `torch/csrc/jit/tensorexpr/ir_mutator.h`
- `torch/csrc/jit/tensorexpr/ir_printer.h`
- `torch/csrc/jit/tensorexpr/ir_simplifier.h`
- `torch/csrc/jit/tensorexpr/ir_verifier.h`
- `torch/csrc/jit/tensorexpr/tensor.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `loopnest.cpp_docs.md`
- **Keyword Index**: `loopnest.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
