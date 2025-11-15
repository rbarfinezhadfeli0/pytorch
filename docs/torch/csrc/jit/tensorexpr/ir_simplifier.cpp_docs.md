# Documentation: `torch/csrc/jit/tensorexpr/ir_simplifier.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/ir_simplifier.cpp`
- **Size**: 94,781 bytes (92.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

#include <utility>

namespace torch::jit::tensorexpr {

// Creates a new Expr of the given type with the provided lhs and rhs.
static inline ExprPtr newBinaryOpOfType(
    IRNodeType expr_type,
    const ExprPtr& lhs,
    const ExprPtr& rhs,
    bool option) {
  switch (expr_type) {
    case IRNodeType::kAdd:
      return alloc<Add>(lhs, rhs);
    case IRNodeType::kSub:
      return alloc<Sub>(lhs, rhs);
    case IRNodeType::kMul:
      return alloc<Mul>(lhs, rhs);
    case IRNodeType::kDiv:
      return alloc<Div>(lhs, rhs);
    case IRNodeType::kMod:
      return alloc<Mod>(lhs, rhs);
    case IRNodeType::kMax:
      return alloc<Max>(lhs, rhs, option);
    case IRNodeType::kMin:
      return alloc<Min>(lhs, rhs, option);
    case IRNodeType::kAnd:
      return alloc<And>(lhs, rhs);
    case IRNodeType::kXor:
      return alloc<Xor>(lhs, rhs);
    case IRNodeType::kLshift:
      return alloc<Lshift>(lhs, rhs);
    case IRNodeType::kRshift:
      return alloc<Rshift>(lhs, rhs);
    default:
      LOG(FATAL) << "unsupported expr_type: " << static_cast<int>(expr_type);
      return nullptr;
  }
}

template <
    typename Op,
    std::enable_if_t<std::is_same_v<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>>* = nullptr>
static ExprPtr mutateBinaryOp(
    NodePtr<Op> v,
    IRMutator* mutator,
    bool option = false) {
  ExprPtr lhs = v->lhs();
  ExprPtr rhs = v->rhs();
  ExprPtr lhs_new = lhs->accept_mutator(mutator);
  ExprPtr rhs_new = rhs->accept_mutator(mutator);

  ExprPtr node = v;

  if (lhs != lhs_new || rhs != rhs_new) {
    node = newBinaryOpOfType(v->expr_type(), lhs_new, rhs_new, option);
  }

  // Can only fold if both sides are constant.
  if (!lhs_new->isConstant() || !rhs_new->isConstant()) {
    return node;
  }

  return evaluateOp(node);
}

// Simple recursive GCD.
template <typename T>
static T gcd(T a, T b) {
  if (b == 0) {
    return a;
  }
  return gcd(b, a % b);
}

// Helper for determining if an Expr is a multi-lane primitive (e.g. Broadcast
// or Ramp).
static bool isMultilanePrimitive(const ExprPtr& e) {
  return to<Broadcast>(e) || to<Ramp>(e);
}

SimplifierHashType Term::hashVars() const {
  SimplifierHashType hash;
  for (const auto& v : variables_) {
    hash = hasher_.hash_combine(hash, hasher_.hash(v));
  }

  return hash;
}

void Term::sort() {
  // order of ops important for float
  if (dtype().is_floating_point()) {
    throw std::logic_error("reordering FP ops");
  }
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(
      variables_.begin(),
      variables_.end(),
      [&](const ExprPtr& a, const ExprPtr& b) {
        if (!str_repr_cache.count(a)) {
          str_repr_cache[a] = std::to_string(a);
        }
        if (!str_repr_cache.count(b)) {
          str_repr_cache[b] = std::to_string(b);
        }
        return str_repr_cache.at(a) < str_repr_cache.at(b);
      });
}

SimplifierHashType Polynomial::hashVars() const {
  SimplifierHashType hash;
  for (const auto& v : variables_) {
    hash = hasher_.hash_combine(hash, hasher_.hash(v));
  }
  return hash;
}

void Polynomial::sort() {
  if (dtype().is_floating_point()) {
    throw std::logic_error("reordering FP ops");
  }
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(
      variables_.begin(),
      variables_.end(),
      [&](const ExprPtr& a, const ExprPtr& b) {
        if (!str_repr_cache.count(a)) {
          str_repr_cache[a] = std::to_string(a);
        }
        if (!str_repr_cache.count(b)) {
          str_repr_cache[b] = std::to_string(b);
        }
        return str_repr_cache.at(a) < str_repr_cache.at(b);
      });
}

void MaxTerm::uniquefy() {
  std::sort(
      variables_.begin(),
      variables_.end(),
      [&](const ExprPtr& a, const ExprPtr& b) {
        return hasher_.hash(a) < hasher_.hash(b);
      });
  auto it = std::unique(
      variables_.begin(),
      variables_.end(),
      [&](const ExprPtr& a, const ExprPtr& b) {
        return hasher_.hash(a) == hasher_.hash(b);
      });
  variables_.resize(std::distance(variables_.begin(), it));

  // Once we removed duplicates, sort terms alphabetically for stability.
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(
      variables_.begin(),
      variables_.end(),
      [&](const ExprPtr& a, const ExprPtr& b) {
        if (!str_repr_cache.count(a)) {
          str_repr_cache[a] = std::to_string(a);
        }
        if (!str_repr_cache.count(b)) {
          str_repr_cache[b] = std::to_string(b);
        }
        return str_repr_cache.at(a) < str_repr_cache.at(b);
      });
}

void MinTerm::uniquefy() {
  std::sort(
      variables_.begin(),
      variables_.end(),
      [&](const ExprPtr& a, const ExprPtr& b) {
        return hasher_.hash(a) < hasher_.hash(b);
      });
  auto it = std::unique(
      variables_.begin(),
      variables_.end(),
      [&](const ExprPtr& a, const ExprPtr& b) {
        return hasher_.hash(a) == hasher_.hash(b);
      });
  variables_.resize(std::distance(variables_.begin(), it));

  // Once we removed duplicates, sort terms alphabetically for stability.
  std::unordered_map<ExprPtr, std::string> str_repr_cache;
  std::sort(
      variables_.begin(),
      variables_.end(),
      [&](const ExprPtr& a, const ExprPtr& b) {
        if (!str_repr_cache.count(a)) {
          str_repr_cache[a] = std::to_string(a);
        }
        if (!str_repr_cache.count(b)) {
          str_repr_cache[b] = std::to_string(b);
        }
        return str_repr_cache.at(a) < str_repr_cache.at(b);
      });
}

// Handles optimization cases for Broadcast/Ramp +/- Broadcast/Ramp
template <class Op>
static ExprPtr combineMultilane(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (BroadcastPtr bc = to<Broadcast>(lhs)) {
    if (BroadcastPtr bcother = to<Broadcast>(rhs)) {
      if (bc->lanes() != bcother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Broadcast>(
          alloc<Op>(bc->value(), bcother->value()), bc->lanes());
      return ret;
    }

    if (RampPtr r = to<Ramp>(rhs)) {
      if (bc->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(bc->value(), r->base()), r->stride(), r->lanes());
      return ret;
    }
  } else if (RampPtr ramp = to<Ramp>(lhs)) {
    if (RampPtr rother = to<Ramp>(rhs)) {
      if (ramp->lanes() != rother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(ramp->base(), rother->base()),
          alloc<Op>(ramp->stride(), rother->stride()),
          ramp->lanes());
      return ret;
    }

    if (BroadcastPtr bc = to<Broadcast>(rhs)) {
      if (ramp->lanes() != bc->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }
      ExprPtr ret = alloc<Ramp>(
          alloc<Op>(ramp->base(), bc->value()), ramp->stride(), ramp->lanes());
      return ret;
    }
  }

  return nullptr;
}

// Handles optimization cases for Broadcast/Ramp * Broadcast/Ramp
static ExprPtr mulMultilane(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (BroadcastPtr bc = to<Broadcast>(lhs)) {
    if (BroadcastPtr bcother = to<Broadcast>(rhs)) {
      if (bc->lanes() != bcother->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Broadcast>(
          alloc<Mul>(bc->value(), bcother->value()), bc->lanes());
      return ret;
    }

    if (RampPtr r = to<Ramp>(rhs)) {
      if (bc->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(bc->value(), r->base()),
          alloc<Mul>(bc->value(), r->stride()),
          r->lanes());
      return ret;
    }
  } else if (RampPtr ramp = to<Ramp>(lhs)) {
    if (RampPtr r = to<Ramp>(rhs)) {
      if (ramp->lanes() != r->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(ramp->base(), r->base()),
          alloc<Mul>(ramp->stride(), r->stride()),
          r->lanes());
      return ret;
    }

    if (BroadcastPtr bc = to<Broadcast>(rhs)) {
      if (ramp->lanes() != bc->lanes()) {
        throw malformed_input("multilane lane mismatch");
      }

      ExprPtr ret = alloc<Ramp>(
          alloc<Mul>(bc->value(), ramp->base()),
          alloc<Mul>(bc->value(), ramp->stride()),
          ramp->lanes());
      return ret;
    }
  }

  return nullptr;
}

void PolynomialTransformer::addOrUpdateTerm(
    std::unordered_map<SimplifierHashType, TermPtr>& varmap,
    const TermPtr& term) {
  SimplifierHashType hash = term->hashVars();
  auto insertRes = varmap.emplace(hash, term);
  if (insertRes.second == false) {
    TermPtr lt = insertRes.first->second;
    ExprPtr termScalar = evaluateOp(alloc<Add>(lt->scalar(), term->scalar()));

    // If the term is canceled out, remove from the map.
    if (immediateEquals(termScalar, 0)) {
      varmap.erase(hash);
      return;
    }

    varmap[hash] = alloc<Term>(hasher_, termScalar, lt->variables());
  }
}

ExprPtr PolynomialTransformer::addPolynomials(
    const PolynomialPtr& lhs,
    const PolynomialPtr& rhs) {
  // simplify common components
  // The key here is the variable hash, not the term's hash since we do want
  // to combine terms that have the same vars but different scalar components.
  std::unordered_map<SimplifierHashType, TermPtr> varmap;

  for (const auto& lt : lhs->variables()) {
    addOrUpdateTerm(varmap, lt);
  }
  for (const auto& rt : rhs->variables()) {
    addOrUpdateTerm(varmap, rt);
  }

  ExprPtr newScalar = evaluateOp(alloc<Add>(lhs->scalar(), rhs->scalar()));
  return alloc<Polynomial>(hasher_, newScalar, varmap);
}

// Insert a new Term into the provided polynomial. If the new term has common
// variables to an existing term it is combined.
ExprPtr PolynomialTransformer::insertTerm(
    const PolynomialPtr& poly,
    const TermPtr& term) {
  SimplifierHashType tHash = term->hashVars();
  std::vector<TermPtr> newVars;

  bool found = false;
  for (const auto& v : poly->variables()) {
    if (v->hashVars() == tHash) {
      ExprPtr newScalar = evaluateOp(alloc<Add>(term->scalar(), v->scalar()));
      found = true;
      // Skip this term if we cancelled it out.
      if (immediateEquals(newScalar, 0)) {
        continue;
      }
      auto term = alloc<Term>(hasher_, newScalar, v->variables());
      newVars.push_back(term);
    } else {
      newVars.push_back(v);
    }
  }

  if (!found) {
    newVars.push_back(term);
  }

  if (newVars.empty()) {
    return poly->scalar();
  }

  auto Poly = alloc<Polynomial>(hasher_, poly->scalar(), newVars);
  return Poly;
}

ExprPtr PolynomialTransformer::mutate(const AddPtr& v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    ExprPtr result = evaluateOp(alloc<Add>(lhs_new, rhs_new));
    return result;
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto ret = combineMultilane<Add>(lhs_new, rhs_new)) {
      return ret->accept_mutator(this);
    }
  }

  ExprPtr scalar = nullptr;
  ExprPtr variable = nullptr;
  if (lhs_new->isConstant()) {
    scalar = evaluateOp(lhs_new);
    variable = rhs_new;
  } else if (rhs_new->isConstant()) {
    scalar = evaluateOp(rhs_new);
    variable = lhs_new;
  }

  // If there is a scalar, and it's zero: short circuit and return the other
  // side.
  if (scalar && immediateEquals(scalar, 0)) {
    auto c = alloc<Cast>(v->dtype(), variable);
    return c->accept_mutator(this);
  }

  // If this is a floating point Add then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Add>(lhs_new, rhs_new);
  }

  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);

  if (lhsPoly && rhsPoly) {
    return addPolynomials(lhsPoly, rhsPoly);
  }

  TermPtr lhsTerm = to<Term>(lhs_new);
  TermPtr rhsTerm = to<Term>(rhs_new);

  if (lhsPoly && rhsTerm) {
    return insertTerm(lhsPoly, rhsTerm);
  }

  if (rhsPoly && lhsTerm) {
    return insertTerm(rhsPoly, lhsTerm);
  }

  if (lhsTerm && rhsTerm) {
    // If the terms refer to the same variables: combine them.
    if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
      ExprPtr newScalar =
          evaluateOp(alloc<Add>(lhsTerm->scalar(), rhsTerm->scalar()));

      // If the terms cancelled out, return zero.
      if (immediateEquals(newScalar, 0)) {
        return newScalar->accept_mutator(this);
      }

      return alloc<Term>(hasher_, newScalar, lhsTerm->variables());
    }

    // Otherwise this is a new polynomial with no scalar and two variable
    // terms.
    return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);
  }

  // Adds are commutative.
  PolynomialPtr poly = lhsPoly ? lhsPoly : rhsPoly;

  // Add to Polynomial->scalar().
  if (scalar && poly) {
    ExprPtr newScalar = evaluateOp(alloc<Add>(scalar, poly->scalar()));
    return alloc<Polynomial>(hasher_, newScalar, poly->variables());
  }

  // Simple Polynomial with a scalar and Term.
  TermPtr term = lhsTerm ? lhsTerm : rhsTerm;
  if (scalar && term) {
    return alloc<Polynomial>(hasher_, scalar, term);
  }

  // Simple Term with a scalar and variable type.
  if (scalar) {
    return alloc<Polynomial>(
        hasher_, scalar, alloc<Term>(hasher_, immLike(v, 1), variable));
  }

  // If LHS is neither Term not Polynomial, wrap it in a Term.
  if (!lhsTerm && !lhsPoly) {
    lhsTerm = alloc<Term>(hasher_, immLike(v, 1), lhs_new);
  }

  // Same for RHS.
  if (!rhsTerm && !rhsPoly) {
    rhsTerm = alloc<Term>(hasher_, immLike(v, 1), rhs_new);
  }

  // If we now have a poly and a term, we can insert.
  if (poly) {
    return insertTerm(poly, lhsTerm ? lhsTerm : rhsTerm);
  }

  if (lhsTerm->hashVars() == rhsTerm->hashVars()) {
    return alloc<Term>(
        hasher_,
        evaluateOp(alloc<Add>(lhsTerm->scalar(), rhsTerm->scalar())),
        lhsTerm->variables());
  }

  // If all else fails we have a new Polynomial with two new variable Terms.
  return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);
}

ExprPtr PolynomialTransformer::subTerms(
    const TermPtr& lhs,
    TermPtr rhs,
    bool negated) {
  // If RHS not already negated, negate it.
  if (!negated) {
    ExprPtr minusOne = immLike(rhs, -1);
    ExprPtr negateScalar = evaluateOp(alloc<Mul>(minusOne, rhs->scalar()));
    rhs = alloc<Term>(hasher_, negateScalar, rhs->variables());
  }

  if (lhs->hashVars() == rhs->hashVars()) {
    ExprPtr newScalar = evaluateOp(alloc<Add>(lhs->scalar(), rhs->scalar()));

    // If the terms cancel out, return zero.
    if (immediateEquals(newScalar, 0)) {
      return newScalar;
    }

    return alloc<Term>(hasher_, newScalar, lhs->variables());
  }

  return alloc<Polynomial>(
      hasher_,
      getImmediateByType(promoteTypes(lhs->dtype(), rhs->dtype()), 0),
      lhs,
      rhs);
}

// Subtract the RHS Polynomial from the LHS Polynomial, cancelling out where
// possible.
ExprPtr PolynomialTransformer::subPolynomials(
    const PolynomialPtr& lhs,
    const PolynomialPtr& rhs) {
  // simplify common components
  // The key here is the variable hash, not the term's hash since we do want
  // to combine terms that have the same vars but different scalar components.
  std::unordered_map<SimplifierHashType, TermPtr> varmap;

  for (const auto& lt : lhs->variables()) {
    addOrUpdateTerm(varmap, lt);
  }

  for (const auto& rt : rhs->variables()) {
    // Polynomials add their terms, so negate the RHS's Terms.
    ExprPtr negated = evaluateOp(alloc<Mul>(immLike(rt, -1), rt->scalar()));
    TermPtr newRHS = alloc<Term>(hasher_, negated, rt->variables());
    addOrUpdateTerm(varmap, newRHS);
  }

  ExprPtr newScalar = evaluateOp(alloc<Sub>(lhs->scalar(), rhs->scalar()));

  // No vars means this cancelled out to a scalar, return it unwrapped.
  if (varmap.empty()) {
    return newScalar;
  }

  // If there is no scalar and zero or one terms, don't wrap.
  if (immediateEquals(newScalar, 0)) {
    if (varmap.empty()) {
      return nullptr;
    }
    if (varmap.size() == 1) {
      return varmap.begin()->second;
    }
  }

  // Wrap new variables in a Polynomial.
  return alloc<Polynomial>(hasher_, newScalar, varmap);
}

ExprPtr PolynomialTransformer::mutate(const SubPtr& v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    ExprPtr result = evaluateOp(alloc<Sub>(lhs_new, rhs_new));
    return result;
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto ret = combineMultilane<Sub>(lhs_new, rhs_new)) {
      return ret->accept_mutator(this);
    }
  }

  if (rhs_new->isConstant() && immediateEquals(rhs_new, 0)) {
    auto c = alloc<Cast>(v->dtype(), lhs_new);
    return c->accept_mutator(this);
  }

  // If this is a floating point Sub then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Sub>(lhs_new, rhs_new);
  }

  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);

  if (lhsPoly && rhsPoly) {
    auto ret = subPolynomials(lhsPoly, rhsPoly);
    if (!ret) {
      // Cancelled out completely.
      return immLike(v, 0);
    }
    return ret;
  }

  TermPtr lhsTerm = to<Term>(lhs_new);
  TermPtr rhsTerm = to<Term>(rhs_new);

  // Polynomial - Term.
  if (lhsPoly && rhsTerm) {
    // Negate the term.
    ExprPtr negate =
        evaluateOp(alloc<Mul>(immLike(rhsTerm, -1), rhsTerm->scalar()));
    TermPtr newTerm = alloc<Term>(hasher_, negate, rhsTerm->variables());
    return insertTerm(lhsPoly, newTerm);
  }

  // Term - Polynomial.
  if (rhsPoly && lhsTerm) {
    // Negate every part of the Polynomial.
    ExprPtr minusOne = immLike(lhsTerm, -1);
    ExprPtr negateScalar = evaluateOp(alloc<Mul>(minusOne, rhsPoly->scalar()));

    std::vector<TermPtr> variables;
    for (const auto& t : rhsPoly->variables()) {
      ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
      variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
    }

    PolynomialPtr newPoly = alloc<Polynomial>(hasher_, negateScalar, variables);
    return insertTerm(newPoly, lhsTerm);
  }

  if (lhsTerm && rhsTerm) {
    return subTerms(lhsTerm, rhsTerm, false);
  }

  bool lhsScalar = lhs_new->isConstant();
  bool rhsScalar = rhs_new->isConstant();

  if (lhsPoly && rhsScalar) {
    // Easy path, just sub the scalar component.
    ExprPtr newScalar = evaluateOp(alloc<Sub>(lhsPoly->scalar(), rhs_new));
    return alloc<Polynomial>(hasher_, newScalar, lhsPoly->variables());
  }

  if (lhsScalar && rhsPoly) {
    // Sub the scalar component.
    ExprPtr newScalar = evaluateOp(alloc<Sub>(lhs_new, rhsPoly->scalar()));

    // Negate each term in the Polynomial RHS.
    ExprPtr minusOne = immLike(rhsPoly, -1);
    std::vector<TermPtr> variables;
    for (const auto& t : rhsPoly->variables()) {
      ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
      variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
    }

    return alloc<Polynomial>(hasher_, newScalar, variables);
  }

  if (lhsTerm && rhsScalar) {
    // Negate the constant.
    ExprPtr negate = evaluateOp(alloc<Mul>(immLike(rhs_new, -1), rhs_new));
    return alloc<Polynomial>(hasher_, negate, lhsTerm);
  }

  if (lhsScalar && rhsTerm) {
    // Negate the RHS Term.
    ExprPtr negate = evaluateOp(
        alloc<Mul>(immLike(rhsTerm->scalar(), -1), rhsTerm->scalar()));

    return alloc<Polynomial>(
        hasher_, lhs_new, alloc<Term>(hasher_, negate, rhsTerm->variables()));
  }

  // simple term with a scalar and variable type.
  if (lhsScalar) {
    // Create a negated term.
    return alloc<Polynomial>(
        hasher_, lhs_new, alloc<Term>(hasher_, immLike(v, -1), rhs_new));
  }

  if (rhsScalar) {
    // Negate the scalar.
    ExprPtr negate = evaluateOp(alloc<Mul>(immLike(rhs_new, -1), rhs_new));
    return alloc<Polynomial>(
        hasher_, negate, alloc<Term>(hasher_, immLike(v, 1), lhs_new));
  }

  // no scalar...
  if (!lhsTerm && !lhsPoly) {
    lhsTerm = alloc<Term>(hasher_, immLike(v, 1), lhs_new);
  }

  bool createdRHSnegated = false;
  if (!rhsTerm && !rhsPoly) {
    rhsTerm = alloc<Term>(hasher_, immLike(v, -1), rhs_new);
    createdRHSnegated = true;
  }

  if (lhsTerm && rhsTerm) {
    return subTerms(lhsTerm, rhsTerm, createdRHSnegated);
  }

  // Insert wrapped Term into LHS Polynomial.
  if (lhsPoly) {
    CHECK(rhsTerm);
    return insertTerm(lhsPoly, rhsTerm);
  }

  // Insert wrapper Term into negated RHS Poly.
  if (rhsPoly) {
    CHECK(lhsTerm);
    ExprPtr minusOne = immLike(rhsPoly, -1);
    ExprPtr newScalar = evaluateOp(alloc<Mul>(minusOne, rhsPoly->scalar()));

    // Negate each term in the Polynomial RHS.
    std::vector<TermPtr> variables;
    for (const auto& t : rhsPoly->variables()) {
      ExprPtr negate = evaluateOp(alloc<Mul>(minusOne, t->scalar()));
      variables.push_back(alloc<Term>(hasher_, negate, t->variables()));
    }

    auto poly = alloc<Polynomial>(hasher_, newScalar, variables);
    return insertTerm(poly, lhsTerm);
  }

  return alloc<Polynomial>(hasher_, immLike(v, 0), lhsTerm, rhsTerm);
}

// Multiply two terms together, usually creating a new term with the variable
// lists concatenated.
TermPtr PolynomialTransformer::mulTerms(
    const TermPtr& lhs,
    const TermPtr& rhs) {
  ExprPtr scalar = evaluateOp(alloc<Mul>(lhs->scalar(), rhs->scalar()));
  if (immediateEquals(scalar, 0)) {
    return nullptr;
  }

  // Can reorder here since floating point ops don't get put into Terms.
  std::vector<ExprPtr> variables;
  std::vector<ExprPtr> multilaneVariables;
  // For now don't handle exponents.
  for (const auto& c : lhs->variables()) {
    if (isMultilanePrimitive(c)) {
      multilaneVariables.push_back(c);
    } else {
      variables.push_back(c);
    }
  }
  for (const auto& c : rhs->variables()) {
    if (isMultilanePrimitive(c)) {
      multilaneVariables.push_back(c);
    } else {
      variables.push_back(c);
    }
  }

  // Merge all the multilane vars:
  ExprPtr lastNode{nullptr};
  for (const auto& node : multilaneVariables) {
    if (lastNode == nullptr) {
      lastNode = node;
    } else {
      if (auto next = mulMultilane(lastNode, node)) {
        lastNode = next->accept_mutator(this);
      } else {
        variables.push_back(lastNode);
        lastNode = node;
      }
    }
  }
  if (lastNode) {
    variables.push_back(lastNode);
  }

  return alloc<Term>(hasher_, scalar, variables);
}

// Multiply a Polynomial by a Term.
ExprPtr PolynomialTransformer::polyByTerm(
    const PolynomialPtr& poly,
    const TermPtr& term) {
  // poly * term
  //    = (poly_terms + poly_scalar) * term
  //    = poly_terms * term + poly_scalar * term

  // First, multiply all variables (terms) in the polynomial by the input
  // term.
  std::vector<TermPtr> newTerms;
  for (const auto& var : poly->variables()) {
    TermPtr newTerm = mulTerms(var, term);
    if (newTerm) {
      newTerms.push_back(newTerm);
    }
  }

  // If the scalar in poly is not 0, it must be multiplied by term.
  // If there are no variables in term, this becomes the scalar in the result
  // polynomial. If there are variables in term, this becomes a new term in
  // the result polynomial.
  if (!immediateEquals(poly->scalar(), 0)) {
    ExprPtr scalar = evaluateOp(alloc<Mul>(poly->scalar(), term->scalar()));
    if (term->variables().empty()) {
      return alloc<Polynomial>(hasher_, scalar, newTerms);
    }
    newTerms.push_back(alloc<Term>(hasher_, scalar, term->variables()));
  }

  // The only case when the result polynomial has a scalar is when the input
  // term does not have any variables and the input polynomial has a non-zero
  // scalar. That case is handled above. So, at this point, we do not have any
  // scalars in the result polynomial.
  return alloc<Polynomial>(hasher_, std::move(newTerms));
}

// Does multiplying these two expressions make a Rounding Off operation.
// e.g. LHS = (x/y),  RHS = y => (x / y) * y => RoundOff(x, y).
ExprPtr PolynomialTransformer::isRoundOff(
    const ExprPtr& lhs,
    const ExprPtr& rhs) {
  DivPtr div{nullptr};
  ExprPtr other{nullptr};

  if ((div = to<Div>(lhs))) {
    other = rhs;
  } else if ((div = to<Div>(rhs))) {
    other = lhs;
  } else {
    return nullptr;
  }

  ExprPtr denom = div->rhs();

  if (TermPtr denomTerm = to<Term>(denom)) {
    if (immediateEquals(denomTerm->scalar(), 1) &&
        denomTerm->variables().size() == 1) {
      denom = denomTerm->variables()[0];
    }
  }

  if (hasher_.hash(denom) == hasher_.hash(other)) {
    // If the denominator is equal to the other, then yes it's a RoundOff.
    return alloc<RoundOff>(div->lhs(), div->rhs());
  }

  if (denom->isConstant() && other->isConstant()) {
    if (immediateEquals(denom, 0) || immediateEquals(other, 0)) {
      return nullptr;
    }
    // If they are both scalar we may be able to find a common factor.
    if (immediateEquals(evaluateOp(alloc<Mod>(other, denom)), 0)) {
      ExprPtr scalar = evaluateOp(alloc<Div>(other, denom));
      ExprPtr newDenom = evaluateOp(alloc<Div>(other, scalar));
      return alloc<Term>(
          hasher_, scalar, alloc<RoundOff>(div->lhs(), newDenom));
    }
  }

  return nullptr;
}

// Inserts a new component into a term, looking for opportunities to simplify.
ExprPtr PolynomialTransformer::insertIntoTerm(
    const TermPtr& term,
    const ExprPtr& expr) {
  std::vector<ExprPtr> vars;

  // Search for RoundOffs.
  bool merged{false};
  for (const auto& component : term->variables()) {
    if (auto roundoff = isRoundOff(component, expr)) {
      vars.push_back(std::move(roundoff));
      merged = true;
    } else {
      vars.push_back(component);
    }
  }

  if (!merged) {
    vars.push_back(expr);
  }

  if (vars.size() == 1 && immediateEquals(term->scalar(), 1)) {
    return std::move(vars[0]);
  }

  return alloc<Term>(hasher_, term->scalar(), std::move(vars));
}

ExprPtr PolynomialTransformer::mutate(const MulPtr& v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Mul>(lhs_new, rhs_new));
  }

  // Multilane folding.
  if (isMultilanePrimitive(lhs_new)) {
    if (auto ret = mulMultilane(lhs_new, rhs_new)) {
      return ret->accept_mutator(this);
    }
  }

  // Order doesn't matter.
  ExprPtr scalar = nullptr;
  ExprPtr variable = nullptr;
  if (lhs_new->isConstant()) {
    scalar = lhs_new;
    variable = rhs_new;
  } else if (rhs_new->isConstant()) {
    scalar = rhs_new;
    variable = lhs_new;
  }

  // Handle special case mul by 1 since that's safe for floating point, even if
  // it's Nan/Inf.
  if (scalar && immediateEquals(scalar, 1)) {
    auto c = alloc<Cast>(v->dtype(), variable);
    return c->accept_mutator(this);
  }

  // If this is a floating point Mul then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Mul>(lhs_new, rhs_new);
  }

  // Handle special case mul by 0.
  if (scalar && immediateEquals(scalar, 0)) {
    return immLike(v, 0);
  }

  // Catch cases of rounding (Div(A/B) * B).
  if (auto ret = isRoundOff(lhs_new, rhs_new)) {
    return ret;
  } else if (auto ret = isRoundOff(v->lhs(), v->rhs())) {
    // We can break the Round + Mod pattern via factorization of the Div, so
    // check whether it would have worked on the unsimplified tree. If so, we
    // need to simplify again.
    return ret->accept_mutator(this);
  }

  PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
  PolynomialPtr rhsPoly = to<Polynomial>(rhs_new);

  if (lhsPoly && rhsPoly) {
    // This expands to more terms that we can't generally fix without variable
    // factorization, it's more efficient to just leave these as Muls.
    return alloc<Mul>(lhsPoly, rhsPoly);
  }

  TermPtr lhsTerm = to<Term>(lhs_new);
  TermPtr rhsTerm = to<Term>(rhs_new);

  if (lhsPoly && rhsTerm) {
    return polyByTerm(lhsPoly, rhsTerm);
  }

  if (rhsPoly && lhsTerm) {
    return polyByTerm(rhsPoly, lhsTerm);
  }

  if (lhsTerm && rhsTerm) {
    return mulTerms(lhsTerm, rhsTerm);
  }

  if (scalar && lhsTerm) {
    ExprPtr newScalar = evaluateOp(alloc<Mul>(scalar, lhsTerm->scalar()));
    return alloc<Term>(hasher_, newScalar, lhsTerm->variables());
  }

  if (scalar && rhsTerm) {
    ExprPtr newScalar = evaluateOp(alloc<Mul>(scalar, rhsTerm->scalar()));
    return alloc<Term>(hasher_, newScalar, rhsTerm->variables());
  }

  // If this is a scalar * a Polynomial, push the scalar term down.
  // We can wrap the scalar with a Term and use polyByTerm.
  if (scalar && lhsPoly) {
    return polyByTerm(lhsPoly, alloc<Term>(hasher_, scalar));
  }
  if (scalar && rhsPoly) {
    return polyByTerm(rhsPoly, alloc<Term>(hasher_, scalar));
  }

  // simple term with a scalar and variable type.
  if (scalar) {
    return alloc<Term>(hasher_, scalar, variable);
  }

  // Multiplying Polynomial by variable can be wrapped in a term and handled
  // by polyByTerm also.
  if (lhsPoly) {
    auto term = alloc<Term>(hasher_, immLike(rhs_new, 1), rhs_new);
    return polyByTerm(lhsPoly, term);
  }
  if (rhsPoly) {
    auto term = alloc<Term>(hasher_, immLike(lhs_new, 1), lhs_new);
    return polyByTerm(rhsPoly, term);
  }

  // Multiplying Term by a variable is equivalent to adding the variable to
  // the term's list of vars.
  if (lhsTerm) {
    return insertIntoTerm(lhsTerm, rhs_new);
  }
  if (rhsTerm) {
    return insertIntoTerm(rhsTerm, lhs_new);
  }

  // Two variables, create a new Term.
  return alloc<Term>(hasher_, immLike(v, 1), lhs_new, rhs_new);
}

static ExprPtr factorizeDivision(ExprPtr lhs_new, ExprPtr rhs_new) {
  if (!lhs_new || !rhs_new) {
    return nullptr;
  }

  ExprPtr leftScalar = lhs_new->isConstant() ? lhs_new : nullptr;
  ExprPtr rightScalar = rhs_new->isConstant() ? rhs_new : nullptr;

  auto lhsTerm = to<Term>(lhs_new);
  auto rhsTerm = to<Term>(rhs_new);
  if (lhsTerm) {
    leftScalar = lhsTerm->scalar();
  }

  if (rhsTerm) {
    rightScalar = rhsTerm->scalar();
  }

  if (!leftScalar || !rightScalar) {
    return nullptr;
  }

  long left = immediateAs<long>(leftScalar);
  long right = immediateAs<long>(rightScalar);

  long GCD = gcd<long>(left, right);
  if (GCD <= 1) {
    return nullptr;
  }

  leftScalar = evaluateOp(alloc<Div>(leftScalar, immLike(leftScalar, GCD)));
  rightScalar = evaluateOp(alloc<Div>(rightScalar, immLike(rightScalar, GCD)));

  if (lhsTerm) {
    lhs_new = alloc<Term>(lhsTerm->hasher(), leftScalar, lhsTerm->variables());
  } else {
    lhs_new = leftScalar;
  }

  if (rhsTerm) {
    rhs_new = alloc<Term>(rhsTerm->hasher(), rightScalar, rhsTerm->variables());
  } else {
    rhs_new = rightScalar;
  }

  return alloc<Div>(lhs_new, rhs_new);
}

ExprPtr PolynomialTransformer::mutate(const DivPtr& v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Div>(lhs_new, rhs_new));
  }

  // If this is a floating point Div then order of operations is important, we
  // dont want to combine ops.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<Div>(lhs_new, rhs_new);
  }

  // If the numerator is zero, so is the result.
  if (lhs_new->isConstant() && immediateEquals(lhs_new, 0)) {
    return lhs_new;
  }

  // If the denominator is one, return numerator.
  if (rhs_new->isConstant() && immediateEquals(rhs_new, 1)) {
    return lhs_new;
  }

  // If numerator and denominator are equal the result is 1.
  // Unless the denominator could be zero.
  // if (hasher_.hash(lhs_new) == hasher_.hash(rhs_new)) {
  //   return getImmediateByType(v->dtype(), 1);
  // }

  if (auto ret = factorizeDivision(lhs_new, rhs_new)) {
    return ret->accept_mutator(this);
  }

  return alloc<Div>(lhs_new, rhs_new);
}

ExprPtr PolynomialTransformer::mutate(const ModPtr& v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Mod>(lhs_new, rhs_new));
  }

  // 0 % x => 0.
  if (lhs_new->isConstant() && immediateEquals(lhs_new, 0)) {
    return lhs_new;
  }

  // x % 1 == 0.
  if (rhs_new->isConstant() && immediateEquals(rhs_new, 1)) {
    return immLike(v, 0);
  }

  // x % x => 0.
  if (hasher_.hash(lhs_new) == hasher_.hash(rhs_new)) {
    return immLike(v, 0);
  }

  TermPtr lhsTerm = to<Term>(lhs_new);
  if (!lhsTerm) {
    PolynomialPtr lhsPoly = to<Polynomial>(lhs_new);
    if (lhsPoly) {
      // Can still optimize this out if we can factorize the polynomial.
      lhsTerm = factorizePolynomial(lhsPoly);
    }
  }

  if (lhsTerm) {
    // ((C1 * C2) * x) % C1 => 0.
    if (rhs_new->isConstant() &&
        immediateEquals(
            evaluateOp(alloc<Mod>(lhsTerm->scalar(), rhs_new)), 0)) {
      return immLike(v, 0);
    }

    // (x * y * z) % x => 0.
    for (const auto& component : lhsTerm->variables()) {
      if (hasher_.hash(component) == hasher_.hash(rhs_new)) {
        return immLike(v, 0);
      }
    }

    // (6 * x * y) % (3 * x * y) => 0.
    // also, (x * y * z) % (z * y) => 0.
    // This requires all variable terms found in the RHS to be present in the
    // LHS.
    TermPtr rhsTerm = to<Term>(rhs_new);
    if (rhsTerm) {
      auto& lVars = lhsTerm->variables();
      auto& rVars = rhsTerm->variables();
      size_t rLeft = rVars.size();

      auto rIt = rVars.begin();

      for (auto lIt = lVars.begin(); lIt != lVars.end() && !rVars.empty();
           ++lIt) {
        auto lHash = hasher_.hash(*lIt);
        for (; rIt != rVars.end(); ++rIt) {
          auto rHash = hasher_.hash(*rIt);
          if (lHash == rHash) {
            --rLeft;
            break;
          } else if (lHash < rHash) {
            break;
          }
        }
      }

      if (rLeft == 0 &&
          immediateEquals(
              evaluateOp(alloc<Mod>(lhsTerm->scalar(), rhsTerm->scalar())),
              0)) {
        return immLike(v, 0);
      }
    }
  }

  return alloc<Mod>(lhs_new, rhs_new);
}

namespace {

// Combines two MinTerm / MaxTerm expressions into one.
// The first type on the template refers to the op, as in Min or Max and the
// second type refers to the corresponding term, as in MinTerm or MaxTerm.
template <class Op, class OpTerm>
ExprPtr combineMinMaxTerms(
    ExprPtr lhs,
    ExprPtr rhs,
    bool propagate_nans,
    HashProvider& hasher) {
  auto combine_scalars = [&](ExprPtr c1, ExprPtr c2) -> ExprPtr {
    if (c1 && c2) {
      return evaluateOp(alloc<Op>(c1, c2, propagate_nans));
    }
    if (c1) {
      return c1;
    }
    return c2;
  };

  auto combine_opterms = [&](NodePtr<OpTerm> m1, NodePtr<OpTerm> m2) {
    ExprPtr scalar = combine_scalars(m1->scalar(), m2->scalar());
    std::vector<ExprPtr> variables;
    for (const auto& v : m1->variables()) {
      variables.push_back(v);
    }
    for (const auto& v : m2->variables()) {
      variables.push_back(v);
    }
    return alloc<OpTerm>(hasher, scalar, propagate_nans, std::move(variables));
  };

  auto add_expr_to_opterm = [&](ExprPtr expr, NodePtr<OpTerm> opterm) {
    ExprPtr scalar = nullptr;
    std::vector<ExprPtr> variables;
    if (opterm) {
      scalar = opterm->scalar();
      variables = opterm->variables();
    }
    if (expr->isConstant()) {
      scalar = combine_scalars(scalar, expr);
    } else {
      variables.push_back(expr);
    }
    return alloc<OpTerm>(hasher, scalar, propagate_nans, std::move(variables));
  };

  auto lhs_opterm = to<OpTerm>(lhs);
  auto rhs_opterm = to<OpTerm>(rhs);
  if (lhs_opterm && lhs_opterm->propagate_nans() != propagate_nans) {
    return alloc<Op>(lhs, rhs, propagate_nans);
  }
  if (rhs_opterm && rhs_opterm->propagate_nans() != propagate_nans) {
    return alloc<Op>(lhs, rhs, propagate_nans);
  }

  if (lhs_opterm && rhs_opterm) {
    return combine_opterms(lhs_opterm, rhs_opterm);
  } else if (lhs_opterm) {
    return add_expr_to_opterm(rhs, lhs_opterm);
  } else if (rhs_opterm) {
    return add_expr_to_opterm(lhs, rhs_opterm);
  }
  return add_expr_to_opterm(rhs, add_expr_to_opterm(lhs, nullptr));
}

// Returns true if op is one of the 2 operands in opterm and also returns
// the other op of opterm in other_op.
template <class OpTerm>
bool isOperandInMinMaxTerm(
    NodePtr<OpTerm> opterm,
    ExprPtr op,
    HashProvider& hasher,
    ExprPtr* other_op) {
  if (opterm->variables().size() != 2) {
    return false;
  }
  auto lhs = opterm->variables()[0];
  auto rhs = opterm->variables()[1];
  auto op_hash = hasher.hash(std::move(op));
  if (hasher.hash(lhs) == op_hash) {
    *other_op = rhs;
    return true;
  } else if (hasher.hash(rhs) == op_hash) {
    *other_op = lhs;
    return true;
  }
  return false;
}

// Simplifies the nested min-max pattern like:
//   * Max(Min(x, y), Min(x, z)) => Min(x, Max(y, z))
//   * Min(Max(x, y), Max(x, z)) => Max(x, Min(y, z))
// This function is called while processing the outer Min / Max ops.
// At that point the inner Min / Max ops would have been converted to
// MinTerm / MaxTerm as appropriate. So, this function checks for those
// term expressions in the given lhs and rhs.
//
// The first type of the template must be the term type corresponding to the
// outer op (e.g. MaxTerm) and the second type of the template must be the term
// type corresponding to the expected inner op (e.g. MinTerm).
template <class OpTerm, class OtherOpTerm>
bool simplifyNestedMinMax(
    ExprPtr lhs,
    ExprPtr rhs,
    bool propagate_nans,
    HashProvider& hasher,
    ExprPtr* new_op) {
  auto lhs_opterm = to<OtherOpTerm>(lhs);
  auto rhs_opterm = to<OtherOpTerm>(rhs);
  if (lhs_opterm && rhs_opterm &&
      lhs_opterm->propagate_nans() == propagate_nans &&
      rhs_opterm->propagate_nans() == propagate_nans) {
    if (!lhs_opterm->scalar() && !rhs_opterm->scalar()) {
      if (lhs_opterm->variables().size() == 2 &&
          rhs_opterm->variables().size() == 2) {
        auto rhs_v1 = rhs_opterm->variables()[0];
        auto rhs_v2 = rhs_opterm->variables()[1];
        ExprPtr new_op_lhs;
        if (isOperandInMinMaxTerm<OtherOpTerm>(
                lhs_opterm, rhs_v1, hasher, &new_op_lhs)) {
          auto inner_op = alloc<OpTerm>(
              hasher, nullptr, propagate_nans, new_op_lhs, rhs_v2);
          *new_op = alloc<OtherOpTerm>(
              hasher, nullptr, propagate_nans, rhs_v1, inner_op);
          return true;
        }
        if (isOperandInMinMaxTerm<OtherOpTerm>(
                lhs_opterm, rhs_v2, hasher, &new_op_lhs)) {
          auto inner_op = alloc<OpTerm>(
              hasher, nullptr, propagate_nans, new_op_lhs, rhs_v1);
          *new_op = alloc<OtherOpTerm>(
              hasher, nullptr, propagate_nans, rhs_v2, inner_op);
          return true;
        }
      }
    }
  }
  return false;
}

} // namespace

ExprPtr PolynomialTransformer::mutate(const MaxPtr& v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Max>(lhs_new, rhs_new, v->propagate_nans()));
  }

  // If diff is constant, return the appropriate operand.
  ExprPtr diff = alloc<Sub>(lhs_new, rhs_new);
  diff = diff->accept_mutator(this);
  if (diff->isConstant()) {
    if (immediateAs<int>(diff) > 0) {
      return lhs_new;
    }
    return rhs_new;
  }

  // Max(Min(x, y), Min(x, z)) => Min(x, Max(y, z))
  ExprPtr new_op;
  if (simplifyNestedMinMax<MaxTerm, MinTerm>(
          lhs_new, rhs_new, v->propagate_nans(), hasher_, &new_op)) {
    return new_op;
  }

  return combineMinMaxTerms<Max, MaxTerm>(
      lhs_new, rhs_new, v->propagate_nans(), hasher_);
}

ExprPtr PolynomialTransformer::mutate(const MinPtr& v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant()) {
    return evaluateOp(alloc<Min>(lhs_new, rhs_new, v->propagate_nans()));
  }

  // If diff is constant, return the appropriate operand.
  ExprPtr diff = alloc<Sub>(lhs_new, rhs_new);
  diff = diff->accept_mutator(this);
  if (diff->isConstant()) {
    if (immediateAs<int>(diff) < 0) {
      return lhs_new;
    }
    return rhs_new;
  }

  // Min(Max(x, y), Max(x, z)) => Max(x, Min(y, z))
  ExprPtr new_op;
  if (simplifyNestedMinMax<MinTerm, MaxTerm>(
          lhs_new, rhs_new, v->propagate_nans(), hasher_, &new_op)) {
    return new_op;
  }

  return combineMinMaxTerms<Min, MinTerm>(
      lhs_new, rhs_new, v->propagate_nans(), hasher_);
}

ExprPtr PolynomialTransformer::mutate(const CompareSelectPtr& v) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);
  ExprPtr true_branch = v->ret_val1()->accept_mutator(this);
  ExprPtr false_branch = v->ret_val2()->accept_mutator(this);

  // Constant Folding.
  if (lhs_new->isConstant() && rhs_new->isConstant() &&
      true_branch->isConstant() && false_branch->isConstant()) {
    ExprPtr v_new = alloc<CompareSelect>(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        v->compare_select_op(),
        v->bias());
    return evaluateOp(v_new);
  }

  // If the comparison is done in float, don't attempt diff simplification,
  // since we can't correctly handle NaN.
  if (lhs_new->dtype().is_floating_point() ||
      rhs_new->dtype().is_floating_point()) {
    return alloc<CompareSelect>(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        v->compare_select_op(),
        v->bias());
  }

  // If diff is constant, we can determine it.
  ExprPtr diff = alloc<Sub>(rhs_new, lhs_new);
  diff = diff->accept_mutator(this);

  if (!diff->isConstant()) {
    return alloc<CompareSelect>(
        lhs_new,
        rhs_new,
        true_branch,
        false_branch,
        v->compare_select_op(),
        v->bias());
  }

  bool equal = immediateEquals(diff, 0);
  bool lhsSmaller = !equal && !immediateIsNegative(diff);

  switch (v->compare_select_op()) {
    case CompareSelectOperation::kEQ:
      return equal ? true_branch : false_branch;
    case CompareSelectOperation::kGT:
      return (lhsSmaller || equal) ? false_branch : true_branch;
    case CompareSelectOperation::kGE:
      return lhsSmaller ? false_branch : true_branch;
    case CompareSelectOperation::kLT:
      return lhsSmaller ? true_branch : false_branch;
    case CompareSelectOperation::kLE:
      return (lhsSmaller || equal) ? true_branch : false_branch;
    case CompareSelectOperation::kNE:
      return equal ? false_branch : true_branch;
  }

  // should not be possible but just in case.
  return alloc<CompareSelect>(
      lhs_new,
      rhs_new,
      true_branch,
      false_branch,
      v->compare_select_op(),
      v->bias());
}

ExprPtr PolynomialTransformer::mutate(const IntrinsicsPtr& v) {
  std::vector<ExprPtr> new_params;
  bool changed = false;
  bool allConstant = true;
  for (const auto& p : v->params()) {
    ExprPtr new_child = p->accept_mutator(this);
    new_params.push_back(new_child);

    changed |= p != new_child;
    allConstant &= new_child->isConstant();
  }

  ExprPtr node = v;
  if (changed) {
    node = alloc<Intrinsics>(v->op_type(), new_params);
  }

  if (!allConstant || !v->isPure()) {
    return node;
  }

  // we're evaluating, but the evaluator only supports float intrinsics.
  std::vector<ExprPtr> const_params;
  changed = false;
  for (const auto& p : new_params) {
    if (p->dtype().scalar_type() == ScalarType::Float) {
      const_params.push_back(p);
    } else {
      const_params.push_back(
          alloc<Cast>(Dtype(ScalarType::Float, p->dtype().lanes()), p));
      changed = true;
    }
  }

  if (changed) {
    node = alloc<Intrinsics>(v->op_type(), const_params);
  }
  return evaluateOp(node);
}

ExprPtr PolynomialTransformer::mutate(const CastPtr& v) {
  ExprPtr node = v->src_value()->accept_mutator(this);
  if (node->isConstant()) {
    return evaluateOp(alloc<Cast>(v->dtype(), node));
  }

  if (v->dtype() == node->dtype()) {
    return node;
  }

  return alloc<Cast>(v->dtype(), node);
}

ExprPtr PolynomialTransformer::mutate(const IfThenElsePtr& v) {
  ExprPtr condition = v->condition();
  ExprPtr true_value = v->true_value();
  ExprPtr false_value = v->false_value();
  ExprPtr condition_new = condition->accept_mutator(this);
  ExprPtr true_value_new = true_value->accept_mutator(this);
  ExprPtr false_value_new = false_value->accept_mutator(this);

  // If the condition is constant then we can choose the right branch now.
  if (condition_new->isConstant()) {
    if (!immediateEquals(condition_new, 0)) {
      return true_value_new;
    } else {
      return false_value_new;
    }
  }

  // If both branches are the same then don't do the condition.
  if (hasher_.hash(true_value_new) == hasher_.hash(false_value_new)) {
    return true_value_new;
  }

  if (condition == condition_new && true_value == true_value_new &&
      false_value == false_value_new) {
    return v;
  }

  return alloc<IfThenElse>(condition_new, true_value_new, false_value_new);
}

ExprPtr PolynomialTransformer::mutate(const AndPtr& v) {
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(const XorPtr& v) {
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(const LshiftPtr& v) {
  return mutateBinaryOp(v, this);
}

ExprPtr PolynomialTransformer::mutate(const RshiftPtr& v) {
  return mutateBinaryOp(v, this);
}

StmtPtr PolynomialBase::mutate(const CondPtr& v) {
  ExprPtr cond_old = v->condition();
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();

  ExprPtr cond_new = cond_old->accept_mutator(this);
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;

  // If the condition is constant then we can choose the right branch now.
  if (cond_new->isConstant()) {
    if (!immediateEquals(cond_new, 0)) {
      return true_new;
    } else {
      return false_new;
    }
  }

  // If both branches are the same then don't do the condition.
  if (true_new && false_new &&
      hasher_.hash(true_new) == hasher_.hash(false_new)) {
    return true_new;
  }

  BlockPtr true_block = to<Block>(true_new);
  BlockPtr false_block = to<Block>(false_new);
  bool true_empty = !true_new || (true_block && true_block->nstmts() == 0);
  bool false_empty = !false_new || (false_block && false_block->nstmts() == 0);

  if (true_empty && false_empty) {
    return alloc<Block>(std::vector<StmtPtr>({}));
  }
  if (cond_old != cond_new) {
    v->set_condition(cond_new);
  }
  if (true_old != true_new) {
    v->set_true_stmt(true_new);
  }
  if (false_old != false_new) {
    v->set_false_stmt(false_new);
  }
  return v;
}

static StmtPtr handleForCondReordering(
    const ForPtr& loop,
    const CondPtr& cond) {
  if (cond->false_stmt()) {
    return nullptr;
  }

  auto condition_vars = VarFinder::find(cond->condition());
  for (const auto& v : condition_vars) {
    // If the condition depends on a Var that is modified in the loop body, it
    // may not be safe to reorder.
    if (ModifiesVarChecker::check(loop, v)) {
      return nullptr;
    }
  }

  ForPtr new_f = loop->cloneWithNewBody(Stmt::clone(cond->true_stmt()));
  return cond->cloneWithNewBody(new_f);
}

StmtPtr PolynomialBase::mutate(const ForPtr& v) {
  ExprPtr var = v->var();
  ExprPtr start = v->start();
  ExprPtr stop = v->stop();
  StmtPtr body = v->body();
  LoopOptions loop_options = v->loop_options();
  ExprPtr var_new_expr = var->accept_mutator(this);
  VarPtr var_new = to<Var>(var_new_expr);
  ExprPtr start_new = start->accept_mutator(this);
  ExprPtr stop_new = stop->accept_mutator(this);
  StmtPtr body_new = body;

  ExprPtr loops = alloc<Sub>(stop_new, start_new);
  loops = loops->accept_mutator(this);
  if (loop_options.isDefault() && loops->isConstant()) {
    if (immediateEquals(loops, 0)) {
      return alloc<Block>(std::vector<StmtPtr>({}));
    } else if (immediateEquals(loops, 1)) {
      body_new = Substitute(body, {{var_new, start_new}});
      body_new = body_new->accept_mutator(this);
      return body_new;
    }
  }

  body_new = body_new->accept_mutator(this);
  if (!body_new) {
    return alloc<Block>(std::vector<StmtPtr>({}));
  }

  if (auto block = to<Block>(body_new)) {
    if (block->nstmts() == 0) {
      return alloc<Block>(std::vector<StmtPtr>({}));
    }

    if (block->nstmts() == 1) {
      if (auto cond = to<Cond>(block->front())) {
        StmtPtr reordered = handleForCondReordering(v, cond);
        if (reordered) {
          return reordered->accept_mutator(this);
        }
      }
    }
  }

  if (var != var_new) 
```



## High-Level Overview


This C++ file contains approximately 8 class(es)/struct(s) and 115 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ExprPtr`, `torch`

**Classes/Structs**: `Op`, `Op`, `OpTerm`, `OpTerm`, `OpTerm`, `OtherOpTerm`, `ModRound`, `ModRound`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/tensorexpr/bounds_overlap.h`
- `torch/csrc/jit/tensorexpr/ir_printer.h`
- `torch/csrc/jit/tensorexpr/ir_simplifier.h`
- `utility`


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

- **File Documentation**: `ir_simplifier.cpp_docs.md`
- **Keyword Index**: `ir_simplifier.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
