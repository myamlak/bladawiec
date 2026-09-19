#pragma once

// The driver-side Kohn-Sham composition.
//
// The scf module's energy seam separates the energy the SCF
// loop can form itself from the energy only the caller knows:
//
//     E_elec = Tr[D H] + 1/2 Tr[D J[D]] + energyContribution(D),
//
// with the Fock matrix and J[D] both supplied by the caller. This header is
// the caller's half of that contract for a density functional: it composes
// the integrals module's direct-family Fock halves (H + 2J(rho) from a
// buildCoulombOnly member, H - K(rho) from a buildExchangeOnly one) with
// grid's XcGridEngine into the three callbacks RunRhfScf/RunUhfScf consume.
//
// The physics it has to get right, and where each term goes:
//
//     F_KS      = H + J[D] + Vxc - c_HF K[D]/2          (closed shell)
//     F_KS,s    = H + J[D_tot] + Vxc_s - c_HF K[d_s]    (unrestricted)
//     E_xc      = Exc[D]                                (the engine's number)
//     E_x_exact = -c_HF 1/4 Tr[D K[D]]                  (closed shell)
//               = -c_HF 1/2 sum_s Tr[d_s K[d_s]]        (unrestricted)
//
// The exchange fraction c_HF comes from the functional itself
// (excgrid::XcFunctional::ExchangeFraction), and the engine's Exc already
// carries the DFT half of the exchange at weight (1 - c_HF) - the published
// hybrid recipes in excgrid's registry are composed that way, so the two
// pieces are disjoint and summing them is the whole energy.
//
// The exchange term is the reason this composition needs the K half at all.
// It is NOT the Hartree-Fock term: it is scaled by c_HF, and a pure LDA/GGA
// functional has c_HF = 0, in which case no exchange half is ever built - a
// DFT run then costs one Coulomb-only Fock build per iteration, which is
// strictly less work than the Hartree-Fock fused build it replaces.
//
// The per-density work cache. The SCF loop asks its three questions about
// ONE density in a fixed order (rhf.cpp / uhf.cpp): the Fock build on the
// iterate's density, then the Coulomb provider and the contribution on the
// density that comes out of it - which is the next iteration's Fock-build
// density. Without a cache each density would be built twice (three times
// for a hybrid); with one entry keyed on the density itself, each density's
// halves and its XC evaluation are formed once. The key is an EXACT matrix
// comparison, so the cache can never serve a different density's numbers:
// a hit means bit-identical inputs to a deterministic builder, which is a
// pure speedup with no numerical reach. The cache is per-run state in a
// std::shared_ptr shared by the three closures; the SCF loop is sequential
// (the builders' own OpenMP work happens inside one call), so no lock is
// needed. A fill is COMMIT-LAST: it computes everything into locals and
// writes the entry only after every call has succeeded, so a failed fill
// leaves the previous entry intact - key and contents alike - and the next
// ask for that density is still served by it. An entry that survived a
// failed fill carrying the failed density's numbers would be a silent wrong
// answer on the next ask; the commit-last order is what makes that
// impossible rather than merely unlikely.
//
// This header is driver-internal (src/internal/, the footprint.hpp
// precedent): the composition is unit-testable with synthetic halves - an
// exactly-known J/K and an exactly-known functional - without a molecule,
// a basis, or a grid, which is how the arithmetic above is pinned.

#include "qcx/error.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"

#include <Eigen/Core>
#include <excgrid/kernel.hpp>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::driver::internal {

/// Builds a qcx::Error from a code and message.
///
/// The driver's own two-argument helper, redeclared here rather than
/// reached for: run_driver.cpp's copy lives in that translation unit's
/// anonymous namespace, and this header is included by the composition's
/// tests without it.
inline qcx::Error Err(qcx::ErrorCode code, std::string message) {
    return qcx::Error{code, std::move(message)};
}

/// The XC functional a Kohn-Sham run consumes, resolved from the schema's
/// `method.functional` name.
///
/// The name stays a string end to end (excgrid's registry is name-keyed and
/// its names are the consumer-facing schema strings); this struct is the
/// resolved annotation the composition needs - the exact-exchange fraction
/// that leaves the DFT path, and whether the functional reads the density
/// gradient, which is the difference between the engine's LDA and GGA arms.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-driver
struct KsFunctional {
    std::string name; ///< The registry name the run asked for.
    double exchangeFraction = 0.0; ///< c_HF, the exact-exchange fraction in [0, 1].
    bool usesGradient = false; ///< True when the functional reads sigma (GGA or beyond).
};

/// The shipped functional names, comma-separated, in registry order.
///
/// One spelling of the list, so every refusal that has to name the shipped set
/// (an unknown name, an absent one) names the same set in the same order - a
/// reader comparing two diagnostics must not have to wonder whether the two
/// lists are two registries.
/// \returns The names, comma-separated.
/// \ingroup qcx-driver
inline std::string ShippedFunctionalNames() {
    std::string shipped;

    for (const std::string_view candidate : excgrid::FunctionalNames())
    {
        if (!shipped.empty())
        {
            shipped += ", ";
        }

        shipped += std::string(candidate);
    }

    return shipped;
}

/// Resolves a `method.functional` name against excgrid's shipped registry.
/// \param name The requested functional name.
/// \returns The resolved functional, or an Error (kInvalidArgument) naming
/// the shipped set - never a fallback to some other functional, because a
/// silently substituted functional is an energy the input did not ask for.
/// \ingroup qcx-driver
inline qcx::Result<KsFunctional> ResolveKsFunctional(std::string_view name) {
    const excgrid::XcFunctional* functional = excgrid::FindFunctional(name);

    if (functional == nullptr)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "method.functional = \"" + std::string(name) +
                    "\" is not a shipped functional: the run is refused rather than "
                    "substituted, because a different functional is a different energy under "
                    "the same label. Shipped: " +
                    ShippedFunctionalNames()));
    }

    return KsFunctional{
        std::string(name), functional->ExchangeFraction(), functional->UsesGradient()};
}

/// The compiled XC evaluation of one spin-density pair: the energy and both
/// spin potentials, from grid's engine (the callback seam shape, so the
/// composition is testable against an analytic functional).
/// \ingroup qcx-driver
using XcEvaluatorFn = std::function<qcx::Result<qcx::grid::XcEvaluation>(
    const Eigen::MatrixXd& densityAlpha, const Eigen::MatrixXd& densityBeta)>;

/// One half of the composition's Fock split, at a SPATIAL density.
///
/// THE CONTRACT, which a half of ANY family must meet before it reaches here:
/// the **Coulomb half returns `H + 2 J(rho)`** and the **exchange half returns
/// `H - K(rho)`** — BOTH carry a full core Hamiltonian, and neither is
/// pre-scaled. WHICH rho the composition hands each half is the composition's
/// business and differs between the two runs (see MakeRksSeam / MakeUksSeam);
/// the half itself does no scaling of its own.
///
/// THE CONTRACT IS THE DIRECT FAMILY'S NATIVE ACCOUNTING, and the other
/// families do NOT produce it natively. Three conventions are live, each
/// internally consistent, and a half handed across a mismatch moves the Fock by
/// an ENTIRE core Hamiltonian — the energy by `Tr[D H]` — with nothing in the
/// result to show it. That is the failure the driver's KS refusal names in its
/// own words: "feeding the energy seam a Coulomb matrix that is not the one the
/// Fock was built from".
///
/// - **direct, lean** — native Coulomb `H + 2 J(rho)`, native exchange
///   `H - K(rho)`. Already the contract; the adapter converts nothing
///   (`MakeDirectKsHalf`, `MakeLeanKsHalf`).
/// - **qfmm** (the composed `QfmmHfFockBuilder`) — **also already the contract**:
///   `QfmmJBuilder::BuildFock` returns `H + 2 J_QFMM(rho)` and the nested
///   exchange-only builder returns `H - K(rho)`, so `MakeQfmmKsHalf` converts
///   nothing either. VERIFIED in `1436bd93`, not assumed from
///   this table — which is the point of the sentence two paragraphs down.
/// - **ri_j_link** — native Coulomb `2 J_RI(rho)` with NO core Hamiltonian,
///   native exchange `H - K(rho)` (the nested exchange builder is the family's
///   sole H carrier). The adapter ADDS H to the Coulomb half and passes the
///   exchange half verbatim (`MakeRiJLinkKsHalf`, `RiJLinkKsHalf`).
/// - **ri_jk** — native Coulomb `J_RI(rho)` and native exchange `K_RI(rho)`:
///   NO H and no factor of two in either (`RiFullFockHalves`, whose members'
///   own docs say the fused build scales the first by two and subtracts the
///   second). The adapter adds H to BOTH halves and doubles the Coulomb one
///   (`MakeRiFullKsHalf`, `RiFullKsHalf`, run_driver.cpp) — the only conversion
///   of the four that touches both halves, and the one whose arithmetic a
///   reader has to check against this table rather than against a sibling.
///
/// So "the integrals builder convention, unchanged" is true of the direct
/// family ONLY. **A composition author adding a family states which convention
/// its builder speaks and where the conversion happens** — the conversion is
/// where a wrong number would be invisible, since every one of these shapes
/// yields a plausible Fock.
/// \ingroup qcx-driver
using HalfFockFn =
    std::function<qcx::Result<Eigen::MatrixXd>(const Eigen::MatrixXd& spatialDensity)>;

/// The three callbacks of the closed-shell Kohn-Sham run.
/// \ingroup qcx-driver
struct RksSeam {
    qcx::scf::FockBuilderFn fock; ///< F_KS[D], the spin-summed density D.
    qcx::scf::CoulombFn coulomb; ///< J[D], the same density.
    qcx::scf::EnergyContributionFn contribution; ///< Exc[D] - c_HF 1/4 Tr[D K[D]].
};

/// The three callbacks of the unrestricted Kohn-Sham run.
/// \ingroup qcx-driver
struct UksSeam {
    qcx::scf::UhfFockBuilderFn fock; ///< (F_alpha, F_beta) of the per-spin pair.
    qcx::scf::CoulombFn coulomb; ///< J[D_total], the spin-summed total.
    qcx::scf::UhfEnergyContributionFn contribution; ///< Exc - c_HF 1/2 sum_s Tr[d_s K_s].
};

namespace detail {

/// The closed-shell working cache: every per-density quantity of one D.
struct RksCache {
    Eigen::MatrixXd density; ///< The spin-summed density this entry belongs to.
    Eigen::MatrixXd coulombHalf; ///< H + J[D].
    Eigen::MatrixXd exchangeHalf; ///< H - K[D]/2 (built only when c_HF > 0).
    qcx::grid::XcEvaluation xc; ///< The closed-shell evaluation at D/2.
    bool exchangeValid = false; ///< False when c_HF = 0 and no K half was built.
    bool valid = false; ///< False until the first fill.
};

/// The unrestricted working cache pair: the Coulomb half is keyed on the
/// spin-summed TOTAL (it is the only thing it depends on), while the
/// exchange halves and the XC evaluation are keyed on the per-spin pair
/// they were formed at. Splitting the key is what keeps the unrestrained
/// run at one build per half per density: the loop's Coulomb question is
/// asked with the total alone, and the pair it belongs to is not
/// recoverable from that matrix.
struct UksCache {
    Eigen::MatrixXd total; ///< The total density the Coulomb half belongs to.
    Eigen::MatrixXd coulombHalf; ///< H + J[D_total].
    bool coulombValid = false;

    Eigen::MatrixXd densityAlpha; ///< The pair these entries belong to.
    Eigen::MatrixXd densityBeta;
    Eigen::MatrixXd exchangeHalfAlpha; ///< H - K[d_alpha] (only when c_HF > 0).
    Eigen::MatrixXd exchangeHalfBeta; ///< H - K[d_beta].
    qcx::grid::XcEvaluation xc; ///< The unrestricted evaluation at the pair.
    bool exchangeValid = false;
    bool pairValid = false;
};

/// Exact matrix identity. The caches key on it, and exactness is the point:
/// these matrices are copies of the loop's own iterate, so a hit means the
/// identical input to a deterministic build, never a near-miss.
inline bool SameMatrix(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs) {
    return lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols() && lhs == rhs;
}

} // namespace detail

/// Composes the closed-shell Kohn-Sham seam.
///
/// The halves take the SPATIAL density rho = D/2 (the RHF seam's own
/// convention): the Coulomb half returns H + 2J(rho) = H + J[D] and the
/// exchange half returns H - K(rho) = H - K[D]/2, so
///
///     F_KS[D] = coulombHalf + c_HF (exchangeHalf - H) + (V_alpha + V_beta)/2,
///
/// the halved potential being the engine's documented closed-shell chain
/// rule (EvaluateClosedShell returns dE/dD_s, so the restricted potential is
/// half the sum - using the sum as it stands double-counts the spin split).
/// The exchange term is written as a correction to the two halves rather
/// than as a fresh K build: H - K[D]/2 is exactly what the exchange half
/// already returned, so K[D] is recovered from it instead of computed twice.
///
/// \param coulombHalf The buildCoulombOnly half; must be callable.
/// \param exchangeHalf The buildExchangeOnly half; EMPTY when c_HF = 0, in
/// which case no K build is ever issued (a pure LDA/GGA functional).
/// \param coreHamiltonian H, exactly the matrix the halves were created
/// with (it is subtracted back out of each half, the direct-UHF split's own
/// arithmetic).
/// \param exchangeFraction c_HF, the functional's exact-exchange fraction.
/// \param xc The compiled XC evaluator; must be callable.
/// \returns The seam, or an Error (kInvalidArgument) for an empty callable
/// or for an exchange fraction outside [0, 1].
/// \ingroup qcx-driver
inline qcx::Result<RksSeam> MakeRksSeam(HalfFockFn coulombHalf,
                                        HalfFockFn exchangeHalf,
                                        const Eigen::MatrixXd& coreHamiltonian,
                                        double exchangeFraction,
                                        XcEvaluatorFn xc) {
    // Each empty argument is refused by NAME. The message is the only thing a
    // reader of the error holds, and one message naming both callables cannot
    // say which of them was empty.
    if (!coulombHalf)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the closed-shell Kohn-Sham composition needs a callable "
                                   "Coulomb half seam"));
    }

    if (!xc)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the closed-shell Kohn-Sham composition needs a callable XC "
                                   "evaluator seam"));
    }

    if (!(exchangeFraction >= 0.0 && exchangeFraction <= 1.0))
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the exact-exchange fraction must lie in [0, 1]"));
    }

    const bool hasExchange = exchangeHalf && exchangeFraction > 0.0;

    if (exchangeFraction > 0.0 && !exchangeHalf)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the functional carries a non-zero exact-exchange fraction, so "
                                   "the exchange half is required and must be callable"));
    }

    auto cache = std::make_shared<detail::RksCache>();

    // Fills the entry for this density, or returns the existing one. Every
    // callback below goes through here, so the build happens once per
    // density no matter which of the three asks first.
    const auto ensure = [cache, coulombHalf, exchangeHalf, hasExchange, xc](
                            const Eigen::MatrixXd& density) -> qcx::Result<void> {
        if (cache->valid && detail::SameMatrix(cache->density, density))
        {
            return {};
        }

        const Eigen::MatrixXd rho = 0.5 * density;
        auto coulomb = coulombHalf(rho);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        // A half-failed fill must never leave a partially updated entry
        // behind. Everything the fill computes is held in locals and the
        // commit below is the last thing the successful path does, so a
        // failure at any of the three calls leaves the previous entry
        // intact - key and contents alike - and the next ask for that
        // density is answered by it rather than by a rebuild. Writing the
        // exchange half where it is built would break this for the
        // evaluation, which is asked after it: the entry would then be
        // keyed on one density and hold another one's exact-exchange term,
        // which the next ask serves as a silent HIT.
        Eigen::MatrixXd builtExchange;

        if (hasExchange)
        {
            auto exchange = exchangeHalf(rho);

            if (!exchange.has_value())
            {
                return std::unexpected(exchange.error());
            }

            builtExchange = std::move(*exchange);
        }

        auto evaluation = xc(rho, rho);

        if (!evaluation.has_value())
        {
            return std::unexpected(evaluation.error());
        }

        cache->density = density;
        cache->coulombHalf = std::move(*coulomb);

        if (hasExchange)
        {
            cache->exchangeHalf = std::move(builtExchange);
        }

        cache->xc = std::move(*evaluation);
        cache->exchangeValid = hasExchange;
        cache->valid = true;
        return {};
    };

    RksSeam seam;

    seam.fock = [ensure, cache, hasExchange, exchangeFraction, coreHamiltonian](
                    const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        // The engine returns dE/dD_s; the restricted potential is half the
        // sum (the EvaluateClosedShell note).
        Eigen::MatrixXd fock =
            cache->coulombHalf + 0.5 * (cache->xc.potentialAlpha + cache->xc.potentialBeta);

        if (hasExchange)
        {
            fock += exchangeFraction * (cache->exchangeHalf - coreHamiltonian);
        }

        return fock;
    };

    seam.coulomb = [ensure, cache, coreHamiltonian](
                       const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        // The half carries one full H copy; subtracting it back leaves
        // exactly the J[D] the seam's energy formula contracts.
        return cache->coulombHalf - coreHamiltonian;
    };

    seam.contribution = [ensure, cache, hasExchange, exchangeFraction, coreHamiltonian](
                            const Eigen::MatrixXd& density) -> qcx::Result<double> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        double contribution = cache->xc.energy;

        if (hasExchange)
        {
            // K[D] = 2 (H - exchangeHalf), so the exact-exchange piece
            // -c_HF 1/4 Tr[D K[D]] is -c_HF/2 Tr[D (H - exchangeHalf)].
            const Eigen::MatrixXd kHalf = coreHamiltonian - cache->exchangeHalf;
            contribution -= 0.5 * exchangeFraction * (density.cwiseProduct(kHalf)).sum();
        }

        return contribution;
    };

    return seam;
}

/// Composes the unrestricted Kohn-Sham seam.
///
/// The density conventions differ per half and are the UHF seam's own: the
/// Coulomb half takes rho = (d_alpha + d_beta)/2 (the closed-shell-shaped
/// half-sum, whose 2J(rho) is J[D_total]), while an exchange half takes the
/// RAW per-spin density - the direct-UHF split's convention, so
///
///     F_s = coulombHalf + c_HF (exchangeHalf_s - H) + Vxc_s.
///
/// The contribution is Exc(d_alpha, d_beta) - c_HF 1/2 sum_s Tr[d_s K[d_s]],
/// the two-spin exchange piece of the seam's own note.
/// \param coulombHalf The buildCoulombOnly half; must be callable.
/// \param exchangeHalfAlpha The alpha buildExchangeOnly half; EMPTY when
/// c_HF = 0.
/// \param exchangeHalfBeta The beta exchange half; empty exactly when the
/// alpha one is.
/// \param coreHamiltonian H, exactly the matrix the halves were created with.
/// \param exchangeFraction c_HF, the functional's exact-exchange fraction.
/// \param xc The compiled XC evaluator; must be callable.
/// \returns The seam, or an Error (kInvalidArgument) as MakeRksSeam.
/// \ingroup qcx-driver
inline qcx::Result<UksSeam> MakeUksSeam(HalfFockFn coulombHalf,
                                        HalfFockFn exchangeHalfAlpha,
                                        HalfFockFn exchangeHalfBeta,
                                        const Eigen::MatrixXd& coreHamiltonian,
                                        double exchangeFraction,
                                        XcEvaluatorFn xc) {
    // As MakeRksSeam: each empty argument is refused by name, and the name
    // says which composition refused as well as which argument was empty.
    if (!coulombHalf)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the unrestricted Kohn-Sham composition needs a callable "
                                   "Coulomb half seam"));
    }

    if (!xc)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the unrestricted Kohn-Sham composition needs a callable XC "
                                   "evaluator seam"));
    }

    if (!(exchangeFraction >= 0.0 && exchangeFraction <= 1.0))
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the exact-exchange fraction must lie in [0, 1]"));
    }

    const bool hasBothExchangeHalves = exchangeHalfAlpha && exchangeHalfBeta;
    const bool hasExchange = hasBothExchangeHalves && exchangeFraction > 0.0;

    if (exchangeFraction > 0.0 && !hasBothExchangeHalves)
    {
        // Three ways to be short a half, and each is refused by name: the
        // reader holds the error and not the arguments, so "the pair is
        // incomplete" would send them to look at both halves.
        if (!exchangeHalfAlpha && !exchangeHalfBeta)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "the functional carries a non-zero exact-exchange "
                                       "fraction, so both per-spin exchange halves are required "
                                       "and must be callable"));
        }

        if (!exchangeHalfAlpha)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "the functional carries a non-zero exact-exchange "
                                       "fraction, so the alpha spin exchange half is required "
                                       "and must be callable"));
        }

        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the functional carries a non-zero exact-exchange fraction, "
                                   "so the beta spin exchange half is required and must be "
                                   "callable"));
    }

    auto cache = std::make_shared<detail::UksCache>();

    const auto ensureCoulomb = [cache,
                                coulombHalf](const Eigen::MatrixXd& total) -> qcx::Result<void> {
        if (cache->coulombValid && detail::SameMatrix(cache->total, total))
        {
            return {};
        }

        auto coulomb = coulombHalf(0.5 * total);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        cache->total = total;
        cache->coulombHalf = std::move(*coulomb);
        cache->coulombValid = true;
        return {};
    };

    const auto ensurePair = [cache, exchangeHalfAlpha, exchangeHalfBeta, hasExchange, xc](
                                const Eigen::MatrixXd& densityAlpha,
                                const Eigen::MatrixXd& densityBeta) -> qcx::Result<void> {
        if (cache->pairValid && detail::SameMatrix(cache->densityAlpha, densityAlpha) &&
            detail::SameMatrix(cache->densityBeta, densityBeta))
        {
            return {};
        }

        // The same commit-last shape as the closed-shell fill: both spin
        // halves are held in locals until the evaluation has succeeded, so a
        // failure at any of the three calls leaves the previous PAIR intact
        // and the next ask for it is answered by that pair's own numbers.
        Eigen::MatrixXd builtAlpha;
        Eigen::MatrixXd builtBeta;

        if (hasExchange)
        {
            auto alpha = exchangeHalfAlpha(densityAlpha);

            if (!alpha.has_value())
            {
                return std::unexpected(alpha.error());
            }

            auto beta = exchangeHalfBeta(densityBeta);

            if (!beta.has_value())
            {
                return std::unexpected(beta.error());
            }

            builtAlpha = std::move(*alpha);
            builtBeta = std::move(*beta);
        }

        auto evaluation = xc(densityAlpha, densityBeta);

        if (!evaluation.has_value())
        {
            return std::unexpected(evaluation.error());
        }

        cache->densityAlpha = densityAlpha;
        cache->densityBeta = densityBeta;

        if (hasExchange)
        {
            cache->exchangeHalfAlpha = std::move(builtAlpha);
            cache->exchangeHalfBeta = std::move(builtBeta);
        }

        cache->xc = std::move(*evaluation);
        cache->exchangeValid = hasExchange;
        cache->pairValid = true;
        return {};
    };

    UksSeam seam;

    seam.fock = [ensureCoulomb, ensurePair, cache, hasExchange, exchangeFraction, coreHamiltonian](
                    const Eigen::MatrixXd& densityAlpha, const Eigen::MatrixXd& densityBeta)
        -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        const auto coulombFilled = ensureCoulomb(densityAlpha + densityBeta);

        if (!coulombFilled.has_value())
        {
            return std::unexpected(coulombFilled.error());
        }

        const auto pairFilled = ensurePair(densityAlpha, densityBeta);

        if (!pairFilled.has_value())
        {
            return std::unexpected(pairFilled.error());
        }

        Eigen::MatrixXd alpha = cache->coulombHalf + cache->xc.potentialAlpha;
        Eigen::MatrixXd beta = cache->coulombHalf + cache->xc.potentialBeta;

        if (hasExchange)
        {
            alpha += exchangeFraction * (cache->exchangeHalfAlpha - coreHamiltonian);
            beta += exchangeFraction * (cache->exchangeHalfBeta - coreHamiltonian);
        }

        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{std::move(alpha), std::move(beta)};
    };

    seam.coulomb = [ensureCoulomb, cache, coreHamiltonian](
                       const Eigen::MatrixXd& total) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensureCoulomb(total);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        // The half carries one full H copy; subtracting it back leaves
        // exactly the J[D_total] the seam's energy formula contracts.
        return cache->coulombHalf - coreHamiltonian;
    };

    seam.contribution = [ensurePair, cache, hasExchange, exchangeFraction, coreHamiltonian](
                            const Eigen::MatrixXd& densityAlpha,
                            const Eigen::MatrixXd& densityBeta) -> qcx::Result<double> {
        const auto pairFilled = ensurePair(densityAlpha, densityBeta);

        if (!pairFilled.has_value())
        {
            return std::unexpected(pairFilled.error());
        }

        double contribution = cache->xc.energy;

        if (hasExchange)
        {
            const Eigen::MatrixXd alphaHalf = coreHamiltonian - cache->exchangeHalfAlpha;
            const Eigen::MatrixXd betaHalf = coreHamiltonian - cache->exchangeHalfBeta;
            contribution -= 0.5 * exchangeFraction *
                            ((densityAlpha.cwiseProduct(alphaHalf)).sum() +
                             (densityBeta.cwiseProduct(betaHalf)).sum());
        }

        return contribution;
    };

    return seam;
}

} // namespace qcx::driver::internal
