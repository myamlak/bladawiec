#pragma once

#include "qcx/memory/sparsity_pattern.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace qcx::memory {

/// Tag: the tensor carries no symmetry structure.
/// \ingroup qcx-memory
struct NoSymmetry {};

/// Tag: the tensor is dense (no sparsity structure).
/// \ingroup qcx-memory
struct Dense {};

/// Tag: the tensor's nonzero structure is described by an associated
/// SparsityPattern<Backend> rather than being fully
/// dense. Contrast with Dense (the original sparsity tag).
/// \ingroup qcx-memory
struct SparsityPatternBacked {};

/// Contract dispatch, specialized per (SymmetryPolicy, SparsityPolicy) pair.
///
/// Contract is specialized per policy pair (an enumerable set) -
/// deliberately not one generic function. The Tensor
/// policy axes are orthogonal and composable, so every combination that is
/// actually used gets its own, specifically-correct implementation, and a
/// combination without a specialization fails to COMPILE rather than
/// silently falling back to something generically-right-but-structurally-
/// wrong (e.g. densifying a sparse structure and calling the dense path -
/// numerically correct, silently much slower).
///
/// Only Dense x NoSymmetry (today's existing behavior, a thin pass-through -
/// the Fock/RI builders keep their own arithmetic) and
/// SparsityPatternBacked x NoSymmetry (the genuinely new case) have
/// real bodies today. A future symmetry tag (beyond NoSymmetry) or a third
/// sparsity tag adds a specialization without
/// reworking the existing ones - the dispatch is the forward-compatible seam.
/// \ingroup qcx-memory
/// \tparam SymmetryPolicy The Tensor's symmetry tag (NoSymmetry today).
/// \tparam SparsityPolicy The Tensor's sparsity tag (Dense or
/// SparsityPatternBacked today).
template <typename SymmetryPolicy, typename SparsityPolicy> struct ContractionPolicy;

/// The dense x no-symmetry contraction: the plain matrix product - exactly
/// what every existing Fock-build/RI call site already computes directly.
/// A thin, behavior-preserving wrapper, not a
/// rewrite of working arithmetic; its value is the generic dispatch
/// mechanism's completeness - a caller generic over SparsityPolicy has a
/// well-defined Dense case to fall through to.
/// \ingroup qcx-memory
template <> struct ContractionPolicy<NoSymmetry, Dense> {
    /// Contracts two dense operands.
    /// \tparam Matrix Any matrix type with operator* (Eigen::MatrixXd at
    /// the existing call sites).
    /// \param a The left operand.
    /// \param b The right operand.
    /// \returns a * b.
    template <typename Matrix> static Matrix Contract(const Matrix& a, const Matrix& b) {
        return a * b;
    }
};

/// The sparse x no-symmetry contraction: structural iteration over the
/// pattern's covered (row, entry) pairs. Contracting
/// against a SparsityPatternBacked tensor means touching only the index
/// pairs the pattern actually lists - the pattern replaces the task-list
/// bookkeeping, not the numerical kernel (the Fock path's AccumulateBlock
/// arithmetic is reused unchanged, wrapped by this iteration).
/// \ingroup qcx-memory
template <> struct ContractionPolicy<NoSymmetry, SparsityPatternBacked> {
    /// Folds \p acc over every (row, packedIndex, entry) pair the pattern
    /// covers, in CSR order. The packed position is passed alongside the
    /// entry so payload arrays aligned with the pattern (one value per
    /// packed index) can be looked up directly - the same enumeration
    /// SparsityPattern::Contract defines, reached through the policy
    /// dispatch so a caller generic over SparsityPolicy has one surface.
    /// \tparam Backend The pattern's execution backend.
    /// \tparam T The accumulator type.
    /// \tparam F fold(acc, row, packedIndex, entry) -> acc.
    /// \param pattern The sparse structure selecting the covered pairs.
    /// \param acc The initial accumulator (e.g. 0.0, or a Fock matrix).
    /// \param fold The per-pair fold, called once per covered pair.
    /// \returns The folded accumulator.
    template <typename Backend, typename T, typename F>
    static T Contract(const SparsityPattern<Backend>& pattern, T acc, F fold) {
        return pattern.Contract(std::move(acc), std::move(fold));
    }

    /// Scatters a payload aligned with the pattern's packed indices into a
    /// dense operand, touching only the (row, col) pairs the pattern lists:
    /// result(row, col) = dense(row, col) + sparseValues[packedIndex] for
    /// every covered pair, result = dense everywhere else. The structural
    /// meaning of a sparse "contraction" for the shell-pair-indexed shapes -
    /// the mask selects which pairs the accumulation touches.
    /// \tparam Backend The pattern's execution backend.
    /// \tparam Matrix Any matrix type with a mutable operator()(row, col)
    /// (Eigen::MatrixXd at the Fock call sites).
    /// \param dense The dense operand to scatter into (unchanged on the
    /// uncovered entries).
    /// \param pattern The sparse structure selecting the covered pairs.
    /// \param sparseValues The payload; one entry per packed index, so it
    /// must hold at least pattern.IndexCount() values (a documented
    /// precondition, not checked).
    /// \returns The dense operand with the payload added at the covered
    /// (row, col) pairs.
    template <typename Backend, typename Matrix>
    static Matrix Contract(const Matrix& dense,
                           const SparsityPattern<Backend>& pattern,
                           const std::vector<double>& sparseValues) {
        Matrix result = dense;
        const auto& offsets = pattern.RowOffsets().HostView();
        const auto& indices = pattern.Indices().HostView();

        for (std::size_t row = 0; row < pattern.RowCount(); ++row)
        {
            for (std::size_t k = offsets[row]; k < offsets[row + 1]; ++k)
            {
                result(static_cast<std::ptrdiff_t>(row), static_cast<std::ptrdiff_t>(indices[k])) +=
                    sparseValues[k];
            }
        }

        return result;
    }
};

} // namespace qcx::memory
