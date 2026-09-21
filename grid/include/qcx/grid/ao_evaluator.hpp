#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace qcx::grid {

/// One shell's AO slot range, in the evaluator's own ordering.
///
/// A screened consumer assembles over a SUBSET of shells, and needs to know
/// which AO slots that subset covers: the selection passed to the evaluator is
/// a list of shell indices, and this is the range each index stands for.
///
/// The range is the shell's WHOLE block: a general contraction keeps its
/// coefficient rows in one shell, and the evaluator lays their function sets
/// out contiguously from aoOffset, so the block spans every row. A range that
/// covered only the first row would silently lose the others' slots. The
/// ranges tile [0, AOCount()) in order.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct ShellRange {
    std::size_t aoOffset = 0; ///< Index of the shell's first AO.
    std::size_t functionCount = 0; ///< AOs the shell occupies, all contraction rows.
};

/// Evaluates every contracted AO basis function at an arbitrary point.
///
/// Each AO is a contracted Gaussian:
///     phi(r) = sum_i d_i N_l(zeta_i) S_lm(r - R) e^{-zeta_i |r - R|^2}
/// with N_l(zeta) = 2^(l+5/4) zeta^(l/2+3/4) / sqrt(2 pi^(3/2) (2l-1)!!)
/// the integrals module's per-primitive normalization (unit self-overlap
/// for every l; (2 zeta/pi)^(3/4) for l = 0) and S_lm the real solid
/// harmonics (Schlegel convention: S00 = 1, S10 = z, S20 = (3z^2-r^2)/2,
/// ...).  The basis-set module renormalizes every spherical contraction
/// to unit norm under exactly this convention (SameShellPrimitiveOverlap
/// in basis_set.cpp), so the stored coefficients are used as-is.
///
/// Spherical functions are stored in m order -l ... +l with m < 0 the
/// sine component and m > 0 the cosine component; Cartesian shells use
/// x^a y^b z^c monomials, a descending then b descending, with the plain
/// (2 zeta/pi)^(3/4) radial factor (the module's Cartesian path applies
/// no solid-harmonic factor).  Consumers assembling density or property
/// matrices must match this ordering to the integrals module's basis
/// ordering before contracting.
/// \ingroup qcx-grid
class AoEvaluator {
public:
    /// Creates the evaluator.
    /// \param molecule The molecule whose atoms carry the shells.
    /// \param basis The basis set; must contain every element of the
    /// molecule.
    /// \returns The evaluator, or an Error: kInvalidArgument for a missing
    /// element entry, kUnimplemented for an unsupported angular momentum
    /// (above the engine's max l).
    /// \ingroup qcx-grid
    static Result<AoEvaluator> Create(const qcx::molecule::Molecule& molecule,
                                      const qcx::basisset::BasisSet& basis);

    /// The total number of AO basis functions.
    /// \returns The AO basis-function count.
    [[nodiscard]] std::size_t AOCount() const noexcept {
        return _aoCount;
    }

    /// The AO slot ranges of every shell, in evaluator order.
    ///
    /// The ordering matches the AO blocks the evaluation fills, so index i of
    /// ShellRanges() is the shell a selection names by i.
    /// \returns The ranges.
    [[nodiscard]] std::span<const ShellRange> ShellRanges() const noexcept {
        return _shellRanges;
    }

    /// Fills \p out with the value of every AO at \p pointBohr.
    /// \param pointBohr The evaluation point, in Bohr.
    /// \param out Must have size AOCount(); overwritten.
    void Evaluate(const std::array<double, 3>& pointBohr, std::span<double> out) const;

    /// Fills \p out with the values of the shells named by \p selection.
    ///
    /// The screened-assembly entry point: a consumer that has decided which
    /// shells matter at a point pays for those shells and no others.  Every
    /// slot outside the selected ranges is LEFT UNTOUCHED, so a caller reusing
    /// one buffer across points must read only the selected ranges.
    /// \param pointBohr The evaluation point, in Bohr.
    /// \param selection Shell indices into ShellRanges(), in any order.
    /// \param out Size AOCount(); the selected ranges are overwritten.
    void EvaluateSelected(const std::array<double, 3>& pointBohr,
                          std::span<const std::size_t> selection,
                          std::span<double> out) const;

    /// Fills \p values and \p gradients with every AO and its first
    /// derivatives at \p pointBohr (Bohr).
    ///
    /// The gradient-only tier of EvaluateDerivatives: the second derivatives
    /// are neither computed nor written, which is what a density-gradient or
    /// potential assembly needs and nothing more. The values and gradients are
    /// identical, slot for slot, to the ones EvaluateDerivatives writes.
    /// \param pointBohr The evaluation point, in Bohr.
    /// \param values size AOCount(); overwritten with phi_mu.
    /// \param gradients size 3 * AOCount(); overwritten with dphi/dx,
    /// dphi/dy, dphi/dz per AO, mu-major (each 3-vector contiguous).
    void EvaluateGradients(const std::array<double, 3>& pointBohr,
                           std::span<double> values,
                           std::span<double> gradients) const;

    /// Fills \p values and \p gradients for the shells named by \p selection.
    ///
    /// The screened form of EvaluateGradients, with the same contract on the
    /// untouched slots: a caller reads only the selected ranges.  Both span
    /// sizes are still AOCount() and 3 * AOCount().
    /// \param pointBohr The evaluation point, in Bohr.
    /// \param selection Shell indices into ShellRanges(), in any order.
    /// \param values Size AOCount(); the selected ranges are overwritten.
    /// \param gradients Size 3 * AOCount(); the selected ranges are overwritten.
    void EvaluateGradientsSelected(const std::array<double, 3>& pointBohr,
                                   std::span<const std::size_t> selection,
                                   std::span<double> values,
                                   std::span<double> gradients) const;

    /// Fills \p values, \p gradients, and \p hessians with every AO and
    /// its first and second derivatives at \p pointBohr (Bohr).
    /// \param values size AOCount(); overwritten with phi_mu.
    /// \param gradients size 3 * AOCount(); overwritten with dphi/dx,
    /// dphi/dy, dphi/dz per AO, mu-major (each 3-vector contiguous).
    /// \param hessians size 6 * AOCount(); overwritten with the symmetric
    /// Hessian entries d2phi/dx2, dxdy, dxdz, dy2, dydz, dz2 per AO,
    /// mu-major.
    void EvaluateDerivatives(const std::array<double, 3>& pointBohr,
                             std::span<double> values,
                             std::span<double> gradients,
                             std::span<double> hessians) const;

    /// Fills \p values, \p gradients, and \p hessians for the shells named by
    /// \p selection.
    ///
    /// The screened form of EvaluateDerivatives, with the same contract on the
    /// untouched slots: a caller reads only the selected ranges. All three span
    /// sizes are still AOCount(), 3 * AOCount() and 6 * AOCount().
    /// \param pointBohr The evaluation point, in Bohr.
    /// \param selection Shell indices into ShellRanges(), in any order.
    /// \param values Size AOCount(); the selected ranges are overwritten.
    /// \param gradients Size 3 * AOCount(); the selected ranges are overwritten.
    /// \param hessians Size 6 * AOCount(); the selected ranges are overwritten.
    void EvaluateDerivativesSelected(const std::array<double, 3>& pointBohr,
                                     std::span<const std::size_t> selection,
                                     std::span<double> values,
                                     std::span<double> gradients,
                                     std::span<double> hessians) const;

private:
    struct Primitive {
        double exponent;
        double coefficient;
        double normalization; ///< RadialNormalization(l, exponent, spherical), precomputed once.
    };

    /// One contracted function set: the primitives of a single contraction
    /// row.  A general contraction (several rows sharing the exponent
    /// array) contributes one Contraction per row.
    struct Contraction {
        std::vector<Primitive> primitives;
    };

    struct ShellData {
        std::array<double, 3> center;
        int angularMomentum;
        bool isSpherical;
        std::vector<Contraction> contractions;
        std::size_t aoOffset;
        std::size_t functionCount; ///< Functions per contraction (2l+1 or Cartesian count).
    };

    // The ranges and the all-shells selection are built in Create rather than
    // here, so this constructor only moves and stays noexcept.
    AoEvaluator(std::vector<ShellData> shells,
                std::vector<ShellRange> shellRanges,
                std::vector<std::size_t> allShells,
                std::size_t aoCount) noexcept;

    // Real solid harmonics S_lm(r) for the shell's angular momentum,
    // written into the shell's AO block in the standard m order
    // (-l ... +l for spherical shells; Cartesian (a,b,c) order for
    // Cartesian shells).
    static void FillAngular(const ShellData& shell,
                            const std::array<double, 3>& delta,
                            std::span<double> block);

    // The shared per-point evaluation, tier-selected at COMPILE time so a
    // gradient-only consumer never pays for the second derivatives: when
    // kHessians is false the hessian slots are neither computed nor written.
    // Defined in the translation unit, instantiated there for both tiers.
    template <bool kHessians>
    void EvaluateDerivativesTier(const std::array<double, 3>& pointBohr,
                                 std::span<const std::size_t> selection,
                                 std::span<double> values,
                                 std::span<double> gradients,
                                 std::span<double> hessians) const;

    // First and second derivatives of the same angular factors (termwise
    // monomial differentiation of the solid-harmonic rows), written into
    // the shell's derivative blocks in the same function order: 3 entries
    // per function into \p gradients, 6 (xx, xy, xz, yy, yz, zz) into
    // \p hessians, which is untouched when kHessians is false.
    template <bool kHessians>
    static void FillAngularDerivativesTier(const ShellData& shell,
                                           const std::array<double, 3>& delta,
                                           std::span<double> gradients,
                                           std::span<double> hessians);

    std::vector<ShellData> _shells;
    std::vector<ShellRange> _shellRanges;
    std::vector<std::size_t> _allShells;
    std::size_t _aoCount;
};

} // namespace qcx::grid
