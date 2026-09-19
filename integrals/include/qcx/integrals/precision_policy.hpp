#pragma once

/// \file
/// The automatic precision ladder of the Fock builds: the budget
/// arithmetic, the per-batch band classification (the Valeev constraint),
/// the escalation schedule with its periodic fp64 referee, and the
/// method.precision cap. Pure
/// policy - no device code, no Eigen: the Fock builders feed it the
/// a-priori bounds of their screened batches and band-tag them, so the
/// hot kernels stay branch-free (precision is a dispatch dimension
/// BETWEEN class runs, never a branch inside a kernel).
///
/// The escalation reading: the remainder
/// B_eri = B(preset) − B_screen − B_ri − B_density − slack is the
/// converged delivery contract; per iteration the classification budget
/// adds the invisible-noise allowance A = min(B_density, A_cap) - early
/// iterations may commit integral noise the density error dwarfs, and as
/// B_density shrinks the budget shrinks to the remainder, so batches
/// ratchet up the ladder (fp16 bulk early → fp32-certified
/// mid → fp64 late), monotone downward in error by construction.

#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"

#include <cstddef>
#include <vector>

namespace qcx::integrals {

/// The precision bands of the ladder, ordered by increasing delivered
/// error bound: the fp16 tensor-core band (GPU only - the certified
/// lane's front-end) → the certified fp32 lane → fp32 evaluation
/// with fp64 accumulation (the [LuehrUfimtsevMartinez2011] default,
/// cheap on the half-ratio datacenter
/// class) → the fp64 reference lane.
/// \ingroup qcx-integrals
enum class PrecisionBand {
    kFp16, ///< fp16 storage/compute, fp32 accumulation (tensor cores).
    kFp32Certified, ///< The certified fp32 lane.
    kFp32EvalFp64Accumulate, ///< fp32 evaluation, fp64 accumulation (the mixed band -
                             ///< the [LuehrUfimtsevMartinez2011] default,
                             ///< cheap on the half-ratio datacenter class).
    kFp64, ///< The reference lane (also the referee's lane).
};

/// The per-run precision cap of method.precision: the knob
/// constrains the ladder, it never replaces it.
/// \ingroup qcx-integrals
enum class PrecisionCap {
    kAuto, ///< The full ladder (the policy decides).
    kFp32Certified, ///< The fp16 and mixed bands clamp up to the certified fp32 lane.
    kFp64, ///< Everything runs the fp64 lane (the pre-ladder strict path).
};

/// The certified fp32 lane's certified epsilon (the kCertifiedEpsilon of
/// md_vrr.hpp, restated for the public policy): the fp32 band's a-priori
/// bound is eps * cClass * (1 + rounding) * Q_bra * Q_ket * |D|.
inline constexpr double kCertifiedBandEpsilon = 1e-7;

/// The fp16 band's certified epsilon (an extension beyond the
/// literature): the fp16 lane stores the Boys values at half-ULP of
/// fp16 representation (2^-12 for the (0,1] value range) around the
/// certified fp32 engine's 1e-7 - the boys.hpp contract table. The same
/// chain carries it.
inline constexpr double kFp16CertifiedEpsilon = (1.0 / 4096.0) + 1e-7;

/// The fp16 band's delivered bound per unit of fp32-unit bound:
/// kFp16CertifiedEpsilon / kCertifiedBandEpsilon. The classification
/// compares delivered bounds against the budget, so the fp16 band is
/// about this many times harder to enter than the certified band.
inline constexpr double kFp16BoundScale = kFp16CertifiedEpsilon / kCertifiedBandEpsilon;

/// The per-preset J/K energy budget of the ladder: the preset's
/// J/K target in Eh - kLoose 1e-6, kNormal 1e-10, kTight 1e-12.
/// \param preset The accuracy preset.
/// \returns The preset's energy budget in Eh.
/// \ingroup qcx-integrals
inline constexpr double PresetEnergyBudget(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 1e-6;
    case AccuracyPreset::kNormal:
        return 1e-10;
    case AccuracyPreset::kTight:
        return 1e-12;
    }

    return 1e-10;
}

/// The ladder's slack: a fixed safety fraction of B(preset) held back
/// from the ERI remainder.
inline constexpr double kLadderSlackFraction = 0.1;

/// The allowance-cap multiple (the escalation layer's seed): the
/// invisible-noise allowance is capped at this multiple of the preset's
/// own budget per iteration - a per-preset early-iterations ceiling,
/// re-anchored by the re-certification step.
inline constexpr double kAllowanceCapMultiple = 1e3;

/// The certified bound's production-path budget: the preset's J/K
/// target less the screening co-term and the ladder's slack, in the same
/// bound units the routing gate and the committed coarse sum use (Eh).
/// The ladder's density-free delivery remainder
/// (EscalationSchedule::DeliveryRemainder) with B_ri = 0 - the direct
/// family composes no RI error - restated for a build that runs no
/// schedule: the quantity a build's routed certified bound sum must not
/// exceed. The density-convergence term stays out by construction (a
/// per-build bound contract, not a per-iteration allowance), so this is
/// the strictest of the ladder's budgets, not the classification one.
/// 0 means nothing fits the preset's own arithmetic (the screening
/// co-term alone consumes the target): the coarse lane routes fp64.
/// \param preset The accuracy preset.
/// \param screenedBoundSum The build's screened-out certified bound sum (Eh).
/// \returns The budget in Eh (>= 0).
/// \ingroup qcx-integrals
inline constexpr double CertifiedBoundBudget(AccuracyPreset preset,
                                             double screenedBoundSum) noexcept {
    const double presetBudget = PresetEnergyBudget(preset);
    const double remainder = presetBudget - screenedBoundSum - kLadderSlackFraction * presetBudget;
    return remainder > 0.0 ? remainder : 0.0;
}

/// The measured fp32/fp64 throughput ratio at which the certified fp32
/// lane's device default flips: at or below it fp32's throughput premium
/// does not cover
/// the lane's routing and conversion overhead. The value is the repo's
/// OWN existing fp64-premium boundary - `ProfileForRatio`'s "comfort zone
/// keeps its full width up to a 4x fp64 premium and narrows linearly
/// beyond" - reused rather than invented, so the lane's enablement and
/// the band pricing cannot drift apart.
inline constexpr double kCertifiedLaneMinRatio = 4.0;

/// The device probe's verdict on the certified fp32 lane's DEFAULT:
/// fp32 may be far more useful on GPUs and AVX machines, so the lane's
/// default is HARDWARE-AWARE rather than a
/// blanket switch. The discriminator is the measured ratio, not the
/// presence of an "AVX" or "GPU" label: a consumer AVX2 host measures
/// near 2x fp32-over-fp64, where conversion plus routing overhead
/// dominates and the lane measured 13-16% SLOWER; the local Quadro T1000
/// measures 31.2 (the consumer 1/32 rate), where the same lane should
/// pay. `ratio` is GpuComputeProfile::fp32ToFp64Ratio - 1.0 is its
/// documented "unknown" value (no device, no CUDA build, or a failed
/// measurement), i.e. the case where no probe applies, and it is below
/// the threshold by construction: the conservative interim is OFF.
/// \param ratio The measured fp32/fp64 throughput ratio (1.0 = unknown).
/// \returns True when the lane's default is on for this device.
/// \ingroup qcx-integrals
inline constexpr bool CertifiedLaneDefaultForRatio(double ratio) noexcept {
    return ratio > kCertifiedLaneMinRatio;
}

/// The referee cadence of a preset: the fp64 referee runs every Nth
/// iteration plus the final one.
/// \param preset The accuracy preset.
/// \returns The referee interval in iterations.
/// \ingroup qcx-integrals
inline constexpr int RefereeEveryForPreset(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 8;
    case AccuracyPreset::kNormal:
        return 5;
    case AccuracyPreset::kTight:
        return 2;
    }

    return 5;
}

/// The per-build budget arithmetic, the
/// formula verbatim: all terms in Eh, B(preset) the preset's J/K budget,
/// B_screen the per-iteration sum of the Schwarz bounds of the
/// screened-out quartets, B_ri the composed RI path's measured error
/// (0 on the direct path), B_density the SCF's current
/// density-convergence error, and slack a fixed safety fraction of
/// B(preset). Signed: negative while the co-terms alone exceed the
/// preset budget (early SCF iterations) - the escalation layer then
/// routes fp64.
/// \ingroup qcx-integrals
struct EriBudgetInputs {
    double bScreen = 0.0; ///< Screened-out quartets' bound sum (Eh).
    double bRi = 0.0; ///< The composed RI path's measured error (Eh).
    double bDensity = 0.0; ///< The SCF's current density-convergence error (Eh).
};

/// The ERI lane's remainder: the formula verbatim.
/// \param preset The accuracy preset.
/// \param inputs The per-build co-term consumption.
/// \returns B_eri in Eh (signed).
/// \ingroup qcx-integrals
inline constexpr double ComputeEriBudget(AccuracyPreset preset,
                                         const EriBudgetInputs& inputs) noexcept {
    return PresetEnergyBudget(preset) - inputs.bScreen - inputs.bRi - inputs.bDensity -
           kLadderSlackFraction * PresetEnergyBudget(preset);
}

/// The band-boundary profile: every band boundary and cost weight of the
/// ladder lives here - one struct, re-tuned per
/// hardware in one place. The default is the device-less / unknown
/// profile (no fp16 band, unpriced boundaries); the builders seed it
/// from the device compute profile (backend/gpu_compute_profile.hpp).
/// \ingroup qcx-integrals
struct PrecisionBandProfile {
    /// fp16 tensor cores available: the fp16 band exists only with this
    /// flag (CPU builds and device-less hosts never take it).
    bool fp16TensorCores = false;
    /// The measured fp32/fp64 throughput ratio of the device (1.0 =
    /// unknown): prices the certified comfort fraction - on the
    /// half-ratio datacenter class fp64 accumulation is cheap and the
    /// mixed band widens.
    double fp32ToFp64Ratio = 1.0;
    /// The fp16 band's budget fraction: a batch takes the fp16 band when
    /// its DELIVERED bound is at most this fraction of the per-batch
    /// budget (two decades of certified margin against the ~2.4e-4 fp16
    /// epsilon).
    double fp16Fraction = 1e-2;
    /// The certified comfort fraction: below it the certified fp32 lane
    /// carries a comfortable margin; between it and the full per-batch
    /// budget the margin is thin and the batch takes the mixed band.
    double certifiedFraction = 1e-1;
    /// The referee interval; 0 = the preset default
    /// (RefereeEveryForPreset).
    int refereeEvery = 0;
};

/// Prices the band boundaries from the measured device ratio (the
/// re-tuning point): the mixed band's domain scales with
/// how cheap fp64 is - on the half-ratio datacenter class the certified
/// comfort zone shrinks toward the mixed band's ceiling, on a 1/32
/// consumer device the certified lane keeps the wide comfort zone.
/// \param ratio The measured fp32/fp64 throughput ratio (1.0 = unknown:
/// no pricing).
/// \param fp16TensorCores The tensor-core availability flag.
/// \returns The seeded profile.
/// \ingroup qcx-integrals
inline constexpr PrecisionBandProfile ProfileForRatio(double ratio, bool fp16TensorCores) noexcept {
    PrecisionBandProfile profile;
    profile.fp16TensorCores = fp16TensorCores;
    profile.fp32ToFp64Ratio = ratio;
    // The pricing: certifiedFraction scales with min(1, 4/ratio) - the
    // comfort zone keeps its full width up to a 4x fp64 premium and
    // narrows linearly beyond (fp64 cheaper -> wider mixed band).
    const double pricing = ratio > 4.0 ? 4.0 / ratio : 1.0;
    profile.certifiedFraction = 1e-1 * pricing;
    return profile;
}

/// Classifies one batch by its certified bound against the per-batch budget
/// (the Valeev constraint - the classification unit is the batch, a
/// class run or shell-pair batch, never an individual quartet, and the
/// band is resolved between class runs, never inside a kernel).
/// bound is the batch's a-priori bound in fp32 units
/// (kCertifiedBandEpsilon * cClass * (1 + rounding) * Q_bra * Q_ket *
/// |D| summed over the batch); the fp16 test compares the band's
/// DELIVERED bound (bound scaled by kFp16BoundScale) against the budget.
/// \param bound The batch's fp32-unit certified bound.
/// \param perBatchBudget The per-batch budget share in Eh (T / nBatches;
/// <= 0 routes fp64).
/// \param profile The band profile.
/// \returns The batch's band.
/// \ingroup qcx-integrals
inline constexpr PrecisionBand ClassifyBatch(double bound,
                                             double perBatchBudget,
                                             const PrecisionBandProfile& profile) noexcept {
    if (perBatchBudget <= 0.0)
    {
        return PrecisionBand::kFp64;
    }

    if (profile.fp16TensorCores && bound * kFp16BoundScale <= profile.fp16Fraction * perBatchBudget)
    {
        return PrecisionBand::kFp16;
    }

    if (bound <= profile.certifiedFraction * perBatchBudget)
    {
        return PrecisionBand::kFp32Certified;
    }

    if (bound <= perBatchBudget)
    {
        return PrecisionBand::kFp32EvalFp64Accumulate;
    }

    return PrecisionBand::kFp64;
}

/// The band rank of the ladder (0 = fp16, 3 = fp64): the ratchet's
/// monotonicity key.
/// \param band The band.
/// \returns The rank.
/// \ingroup qcx-integrals
inline constexpr int BandRank(PrecisionBand band) noexcept {
    switch (band)
    {
    case PrecisionBand::kFp16:
        return 0;
    case PrecisionBand::kFp32Certified:
        return 1;
    case PrecisionBand::kFp32EvalFp64Accumulate:
        return 2;
    case PrecisionBand::kFp64:
        return 3;
    }

    return 3;
}

/// The more precise of two bands (the ratchet step).
/// \param a One band.
/// \param b The other band.
/// \returns The band with the higher rank.
/// \ingroup qcx-integrals
inline constexpr PrecisionBand MorePreciseBand(PrecisionBand a, PrecisionBand b) noexcept {
    return BandRank(a) >= BandRank(b) ? a : b;
}

/// Applies the per-run precision cap: the cap clamps every
/// classified band up to the capped band, never down - kAuto leaves the
/// classification alone, kFp32Certified clamps fp16 and the mixed band
/// up to the certified lane, kFp64 sends everything to the reference
/// lane (bit-compatible with the pre-ladder fp64 builds the pins
/// certify).
/// \param band The classified band.
/// \param cap The per-run cap.
/// \returns The capped band.
/// \ingroup qcx-integrals
inline constexpr PrecisionBand ApplyCap(PrecisionBand band, PrecisionCap cap) noexcept {
    switch (cap)
    {
    case PrecisionCap::kAuto:
        return band;
    case PrecisionCap::kFp32Certified:
        return band == PrecisionBand::kFp16 || band == PrecisionBand::kFp32EvalFp64Accumulate
                   ? PrecisionBand::kFp32Certified
                   : band;
    case PrecisionCap::kFp64:
        return PrecisionBand::kFp64;
    }

    return PrecisionBand::kFp64;
}

/// One SCF run's escalation state: the per-iteration
/// classification budget, the per-batch band ratchet, the committed
/// coarse-work bound sum, and the referee cadence. The band of every
/// batch only ratchets UP the ladder - monotone downward in error by
/// construction, the stronger branch of the DIIS rule: a budget
/// bounce (DIIS oscillation) holds the previous band instead of
/// switching back. One schedule per SCF run; Reset() starts a new run.
/// \ingroup qcx-integrals
class EscalationSchedule {
public:
    /// Constructs one SCF run's schedule.
    /// \param preset The run's accuracy preset (drives B(preset), the
    /// allowance cap, and the coarse-band availability gate).
    /// \param profile The band profile (tensor cores, boundary
    /// fractions, referee cadence).
    /// \param cap The per-run precision cap.
    EscalationSchedule(AccuracyPreset preset,
                       const PrecisionBandProfile& profile,
                       PrecisionCap cap);

    /// True when the preset admits the ladder's coarse bands at all
    /// (MixedPrecisionThreshold(preset) > 0): kTight is fp64-only - the
    /// strict-pins contract every band below fp64 would break.
    /// \returns False at kTight.
    bool CoarseBandsAvailable() const noexcept;

    /// The per-build classification budget of one iteration (the
    /// escalation layer): T = B_cert + A, with B_cert the density-free
    /// delivery remainder B(preset) − B_screen − B_ri − slack and A the
    /// invisible-noise allowance min(B_density, A_cap). As B_density
    /// shrinks, T shrinks to the delivery remainder and the batches
    /// ratchet up the ladder. 0 when the coarse bands are unavailable or
    /// the remainder is negative (everything routes fp64).
    /// \param inputs The per-build co-term consumption.
    /// \returns The total classification budget in Eh (>= 0).
    double ClassificationBudget(const EriBudgetInputs& inputs) const noexcept;

    /// The density-free delivery remainder B_cert = B(preset) − B_screen
    /// − B_ri − slack: the contract the FINAL build's committed
    /// coarse-work bound sum must honor (the referee's final check).
    /// \param inputs The per-build co-term consumption.
    /// \returns The delivery remainder in Eh (>= 0).
    double DeliveryRemainder(const EriBudgetInputs& inputs) const noexcept;

    /// The ratcheted band of one batch: the more precise of its previous
    /// band and its classification under perBatchBudget, capped. The
    /// batchIndex keys the batch across iterations (the builders use the
    /// assembled class-run index); a first sighting classifies from
    /// scratch.
    /// \param batchIndex The batch's position in the assembled order.
    /// \param bound The batch's fp32-unit certified bound.
    /// \param perBatchBudget The per-batch budget share (T / nBatches).
    /// \returns The ratcheted (and capped) band.
    PrecisionBand RatchetBand(std::size_t batchIndex, double bound, double perBatchBudget);

    /// Records one build's committed coarse-work bound sum (the delivered
    /// bound sum of the bands below fp64) for the referee.
    /// \param coarseBoundSum The committed sum in Eh.
    void CommitCoarseBoundSum(double coarseBoundSum) noexcept;

    /// The last committed coarse-work bound sum.
    /// \returns The committed sum in Eh.
    double LastCommittedSum() const noexcept;

    /// True when the fp64 referee must run on this iteration: every Nth
    /// iteration plus the final one.
    /// \param iteration The 1-based SCF iteration.
    /// \param isFinal True for the converged/final build.
    /// \returns True when the referee runs.
    bool RefereeDue(int iteration, bool isFinal) const noexcept;

    /// The referee's final-iteration delivery check: the final build's
    /// committed coarse-work bound sum must fit the density-free
    /// delivery remainder. A violation is a hard error, never a silent
    /// drift.
    /// \param committedCoarseSum The final build's committed sum.
    /// \param inputs The final build's co-term consumption.
    /// \returns An Error (kInternalError) when the delivery contract is
    /// violated.
    qcx::Result<void> RefereeDelivery(double committedCoarseSum,
                                      const EriBudgetInputs& inputs) const;

    /// Starts a new SCF run: clears the band ratchet and the committed
    /// sum. The preset, profile, and cap persist.
    void Reset();

private:
    AccuracyPreset _preset;
    PrecisionBandProfile _profile;
    PrecisionCap _cap;
    std::vector<PrecisionBand> _bands; // Per-batch ratchet state.
    double _committedCoarseSum = 0.0;
};

/// The referee's delivered-error check (the certified-sum contract's
/// live meaning): the ladder build's Fock deviation from the fp64
/// reference build must stay within the committed coarse-work bound sum
/// - the certified sum is an upper bound on every Fock-element error
/// (the gpu_fock_build.hpp contract). The reference build runs the same
/// screening, so a committed sum of 0 demands the bitwise fp64 build.
/// \param maxAbsDelta The max |F_ladder - F_reference| element.
/// \param committedCoarseSum The ladder build's committed sum (Eh).
/// \param tolerance Relative tolerance of the sum comparison.
/// \returns An Error (kInternalError) when the delivered error exceeds
/// the committed bound.
/// \ingroup qcx-integrals
qcx::Result<void> RefereeErrorCheck(double maxAbsDelta,
                                    double committedCoarseSum,
                                    double tolerance = 1e-12);

/// The per-call ladder inputs: the SCF's current
/// density-convergence error and the composed RI path's measured error
/// (0 on the direct path), plus the run's escalation schedule. Passed to
/// BuildFock by the SCF driver; a null pointer keeps the pre-ladder
/// two-pass build bit-identical.
/// \ingroup qcx-integrals
struct PrecisionLadderInputs {
    double densityError = 0.0; ///< The SCF gate's rms density change (Eh scale).
    double riError = 0.0; ///< The composed RI path's measured error (Eh).
    EscalationSchedule* schedule = nullptr; ///< The run's schedule (required to engage).
};

} // namespace qcx::integrals
