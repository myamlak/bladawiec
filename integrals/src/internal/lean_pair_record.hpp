#pragma once

/// \file
/// The lean-direct builder's irreducible per-pair state (the
/// 2026-09-08 lean-direct reframe): ONE immutable record per canonical
/// shell pair - no task records, no computed flags, no per-class charges,
/// no quartet offsets, no admission record, no retained workspace per
/// pair. Built once at Create from the counted Schwarz screening
/// machinery's inputs (ComputeSchwarzBounds), then treated as immutable in
/// the lean path. Internal: the record layout is the lean path's contract
/// with its tests (the 24-byte bound and the absence of heap members are
/// static-asserted in lean_fock_build.cpp). When the builder carries an
/// Abelian point-group reduction (LeanFockBuildOptions::symmetryReduction)
/// the record also carries the pair's derived CLASSIFICATION, built at the
/// same Create-time moment as the record itself and packed into the record's
/// existing classAndSymmetry word - no new member, no layout growth, no
/// separate per-pair table.

#include "qcx/error.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <type_traits>

namespace qcx::integrals::internal {

/// The bit width of the per-pair point-group classification: one bit per group
/// element, TWO such masks packed into the pair record's upper two bytes (see
/// LeanShellPairRecord::classAndSymmetry). The eight Abelian computational
/// groups a reduction realizes have at most 2^3 = 8 elements, so eight bits
/// hold the classification exactly - and eight is also the widest group the
/// orbit action's tables carry (lean_orbit_action.hpp), which is why a wider
/// reduction is refused rather than silently truncated.
constexpr std::size_t kSymmetryMaskBits = 8;

/// Validates the shape of one Abelian reduction: the checks every point-group
/// consumer of the reduction shares (the per-pair classification masks this
/// header's record carries, and the orbit action, lean_orbit_action.hpp).
/// Each check is a precondition of the signed-permutation reading itself, so a
/// malformed reduction is refused here rather than believed downstream.
///
/// The checks, in order (the first failure is the reported one): the group
/// order fits the eight-element mask width; the permutation and sign tables
/// carry one row per element; every row spans the function count; every image
/// index is in range; every sign is +1 or -1; and the first element is the
/// identity - all functions fixed, every sign +1 - the invariant both the
/// zero test and every orbit computation rest on.
/// \param reduction The reduction to validate.
/// \param functionCount The basis function count the rows must span.
/// \returns Nothing, or kInvalidArgument naming the first failed check.
inline qcx::Result<void> ValidateSymmetryReductionShape(const SymmetryReduction& reduction,
                                                        std::size_t functionCount) {
    const std::size_t order = reduction.groupOrder;

    if (order > kSymmetryMaskBits)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the symmetry reduction is wider than the classification's eight elements"});
    }

    if (reduction.permutation.size() != order || reduction.sign.size() != order)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the symmetry reduction's element count does not match its group order"});
    }

    for (std::size_t g = 0; g < order; ++g)
    {
        if (reduction.permutation[g].size() != functionCount ||
            reduction.sign[g].size() != functionCount)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the symmetry reduction's rows do not match the function count"});
        }

        for (std::size_t f = 0; f < functionCount; ++f)
        {
            if (reduction.permutation[g][f] >= functionCount)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "the symmetry reduction maps a function out of range"});
            }

            if (reduction.sign[g][f] != 1 && reduction.sign[g][f] != -1)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "the symmetry reduction carries a sign that is neither +1 nor -1"});
            }
        }
    }

    for (std::size_t f = 0; f < functionCount; ++f)
    {
        if (reduction.permutation[0][f] != f || reduction.sign[0][f] != 1)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the symmetry reduction's first element is not the identity"});
        }
    }

    return qcx::Result<void>{};
}

/// One immutable shell-pair record. The record's POSITION in the records
/// vector is the canonical pair index into ShellPairList::pairs (the
/// records are built in pair-index order), so no pair index is stored.
/// The record is a plain POD aggregate: trivially copyable, no pointers,
/// no heap members - 24 bytes (static_asserted).
struct LeanShellPairRecord {
    std::uint32_t shellM; ///< First shell index of the canonical pair row.
    std::uint32_t shellN; ///< Second shell index (shellN >= shellM).
    double qSchwarz; ///< The pair's Schwarz bound Q_MN = sqrt((MN|MN)_max).
    /// The pair's angular-momentum class (la + lb) in the low byte and, when
    /// the builder carries an Abelian reduction, the derived point-group
    /// classification in the two bytes above it: bits 8..15 are the group
    /// elements that map BOTH of the pair's shells onto themselves
    /// function-for-function with sign product +1, bits 16..23 the elements
    /// that do so with sign product -1 (bit g = element g; the eight Abelian
    /// computational groups fit eight bits). Without a reduction both bytes
    /// stay zero, so no cell ever tests as provably zero - which is why the
    /// row walk needs no "is the filter engaged" branch of its own. See
    /// LeanPairCellIsProvablyZero for what the two masks mean together.
    std::uint32_t classAndSymmetry;
    /// Offset to the pair's first AO pair in the packed function-pair
    /// triangle (the PairIndexOf formula over global function indices of
    /// shellM/shellN's first functions).
    std::uint32_t packedAoPairOffset;
};

/// The record's byte size: 4 + 4 + 8 + 4 + 4 = 24, no padding (double
/// aligned at 8). Asserted here so a layout change cannot slip silently.
static_assert(sizeof(LeanShellPairRecord) == 24, "the lean pair record must stay 24 bytes");
static_assert(alignof(LeanShellPairRecord) == 8, "the lean pair record must stay 8-aligned");
static_assert(std::is_trivially_copyable_v<LeanShellPairRecord>,
              "the lean pair record must stay a trivial POD aggregate");

/// The shift of the pair record's "+1 classification" byte (see
/// LeanShellPairRecord::classAndSymmetry).
constexpr std::uint32_t kLeanPairPlusShift = 8;
/// The shift of the pair record's "-1 classification" byte.
constexpr std::uint32_t kLeanPairMinusShift = 16;

/// The row walk's point-group zero test over two pair records.
///
/// True when the cell (bra pair, ket pair) is a provable exact-zero ERI
/// block: some group element fixes both pairs - maps every function of all
/// four shells onto itself - and the two pairs' sign products are OPPOSITE
/// under it, so the invariance relation (g mu g nu | g lambda g sigma) =
/// s(mu)s(nu)s(lambda)s(sigma) (mu nu | lambda sigma) reads A = -A element
/// by element. Such a cell contributes exact zeros to every contraction, so
/// dropping it cannot move a byte - the acceptance invariant the point-group
/// tests pin. Records built without a reduction carry zero masks and never
/// test true.
/// \param bra The bra pair's record.
/// \param ket The ket pair's record.
/// \returns True when the cell's ERI block is provably all-zero.
inline bool LeanPairCellIsProvablyZero(const LeanShellPairRecord& bra,
                                       const LeanShellPairRecord& ket) noexcept {
    const std::uint32_t braPlus = (bra.classAndSymmetry >> kLeanPairPlusShift) & 0xFFu;
    const std::uint32_t braMinus = (bra.classAndSymmetry >> kLeanPairMinusShift) & 0xFFu;
    const std::uint32_t ketPlus = (ket.classAndSymmetry >> kLeanPairPlusShift) & 0xFFu;
    const std::uint32_t ketMinus = (ket.classAndSymmetry >> kLeanPairMinusShift) & 0xFFu;

    return ((braPlus & ketMinus) | (braMinus & ketPlus)) != 0u;
}

} // namespace qcx::integrals::internal
