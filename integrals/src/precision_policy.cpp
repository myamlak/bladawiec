// The escalation-schedule state machine of the precision ladder
// (precision_policy.hpp): the per-iteration classification budget, the
// per-batch band ratchet, the committed coarse-work bound sum, and the
// referee cadence. Everything else in the header is constexpr.

#include "qcx/integrals/precision_policy.hpp"

#include <algorithm>

namespace qcx::integrals {

EscalationSchedule::EscalationSchedule(AccuracyPreset preset,
                                       const PrecisionBandProfile& profile,
                                       PrecisionCap cap) :
    _preset(preset), _profile(profile), _cap(cap) {}

bool EscalationSchedule::CoarseBandsAvailable() const noexcept {
    // The preset-level gate: kTight keeps the coarse bands
    // off (MixedPrecisionThreshold(kTight) == 0), so the ladder routes
    // everything fp64 there - the strict-pins contract.
    return MixedPrecisionThreshold(_preset) > 0.0;
}

double EscalationSchedule::ClassificationBudget(const EriBudgetInputs& inputs) const noexcept {
    if (!CoarseBandsAvailable())
    {
        return 0.0;
    }

    const double presetBudget = PresetEnergyBudget(_preset);
    const double remainder =
        presetBudget - inputs.bScreen - inputs.bRi - kLadderSlackFraction * presetBudget;

    if (remainder <= 0.0)
    {
        return 0.0;
    }

    // The invisible-noise allowance: early
    // iterations may commit integral noise the density error dwarfs,
    // capped per preset; as B_density shrinks the budget shrinks to the
    // delivery remainder.
    const double allowance = std::min(inputs.bDensity, kAllowanceCapMultiple * presetBudget);
    return remainder + allowance;
}

double EscalationSchedule::DeliveryRemainder(const EriBudgetInputs& inputs) const noexcept {
    if (!CoarseBandsAvailable())
    {
        return 0.0;
    }

    const double presetBudget = PresetEnergyBudget(_preset);
    return std::max(
        0.0, presetBudget - inputs.bScreen - inputs.bRi - kLadderSlackFraction * presetBudget);
}

// (batchIndex, bound) are the batch ordinal and the per-batch error bound -
// distinct quantities, single call site.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
PrecisionBand EscalationSchedule::RatchetBand(std::size_t batchIndex,
                                              double bound,
                                              double perBatchBudget) {
    const PrecisionBand classified = ClassifyBatch(bound, perBatchBudget, _profile);

    if (batchIndex >= _bands.size())
    {
        // kFp16 is the ladder's minimum: a first sighting classifies from
        // scratch (MorePreciseBand(kFp16, classified) == classified).
        _bands.resize(batchIndex + 1, PrecisionBand::kFp16);
    }

    _bands[batchIndex] = MorePreciseBand(_bands[batchIndex], classified);
    return ApplyCap(_bands[batchIndex], _cap);
}

void EscalationSchedule::CommitCoarseBoundSum(double coarseBoundSum) noexcept {
    _committedCoarseSum = coarseBoundSum;
}

double EscalationSchedule::LastCommittedSum() const noexcept {
    return _committedCoarseSum;
}

bool EscalationSchedule::RefereeDue(int iteration, bool isFinal) const noexcept {
    if (iteration <= 0)
    {
        return false;
    }

    if (isFinal)
    {
        return true;
    }

    const int every =
        _profile.refereeEvery > 0 ? _profile.refereeEvery : RefereeEveryForPreset(_preset);
    return iteration % every == 0;
}

qcx::Result<void> EscalationSchedule::RefereeDelivery(double committedCoarseSum,
                                                      const EriBudgetInputs& inputs) const {
    const double remainder = DeliveryRemainder(inputs);

    if (committedCoarseSum > remainder)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInternalError,
            "precision ladder: the final build's committed coarse-work bound sum exceeds "
            "the delivery remainder"});
    }

    return {};
}

void EscalationSchedule::Reset() {
    _bands.clear();
    _committedCoarseSum = 0.0;
}

qcx::Result<void> RefereeErrorCheck(double maxAbsDelta,
                                    double committedCoarseSum,
                                    double tolerance) {
    if (maxAbsDelta > committedCoarseSum * (1.0 + tolerance))
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInternalError,
            "precision ladder referee: the delivered Fock deviation exceeds the committed "
            "coarse-work bound sum"});
    }

    return {};
}

} // namespace qcx::integrals
