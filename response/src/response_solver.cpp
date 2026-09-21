#include "qcx/response/response_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace qcx::response {

std::string_view ToString(PreconditionerKind kind) {
    switch (kind)
    {
    case PreconditionerKind::kNone:
        return "none";
    case PreconditionerKind::kDiagonal:
        return "diagonal";
    case PreconditionerKind::kRegularised:
        return "regularised";
    }

    return "unknown";
}

namespace {

/// Formats a double for a refusal message in a form that names its magnitude.
/// \param value The value.
/// \returns Scientific notation with four significant digits.
std::string Scientific(double value) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(3) << value;
    return text.str();
}

} // namespace

qcx::Result<PreparedPreconditioner> PreparePreconditioner(std::span<const double> diagonal,
                                                          const ResponseSolveOptions& options) {
    if (diagonal.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the operator's preconditioner diagonal is empty"});
    }

    double largest = 0.0;

    for (const double value : diagonal)
    {
        largest = std::max(largest, std::abs(value));
    }

    if (largest == 0.0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the preconditioner diagonal is identically zero, so there is no gap to "
                       "measure and nothing to invert"});
    }

    double smallest = largest;

    for (const double value : diagonal)
    {
        smallest = std::min(smallest, std::abs(value));
    }

    PreparedPreconditioner prepared;
    prepared.inverse.assign(diagonal.size(), 1.0);
    prepared.gapRatio = smallest / largest;

    // No preconditioner inverts nothing, so the small-gap rule has nothing to
    // say about it; the gap is still measured and reported.
    if (options.preconditioner == PreconditionerKind::kNone)
    {
        prepared.kind = PreconditionerKind::kNone;
        return prepared;
    }

    // A zero entry cannot be inverted at any ratio, so it counts as small-gapped
    // whatever threshold the caller set.
    const bool smallGapped = smallest == 0.0 || prepared.gapRatio < options.smallGapRatio;

    if (!smallGapped)
    {
        if (options.preconditioner == PreconditionerKind::kRegularised)
        {
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kInvalidArgument,
                "the regularised preconditioner was asked for, but the diagonal's smallest entry "
                "is " +
                    Scientific(prepared.gapRatio) +
                    " of its largest, at or above the small-gap threshold " +
                    Scientific(options.smallGapRatio) +
                    ", so there is no small gap to regularise"});
        }

        prepared.kind = PreconditionerKind::kDiagonal;

        for (std::size_t i = 0; i < diagonal.size(); ++i)
        {
            prepared.inverse[i] = 1.0 / diagonal[i];
        }

        return prepared;
    }

    if (!options.allowSmallGapFallback)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the preconditioner diagonal is small-gapped (min|d|/max|d| = " +
                           Scientific(prepared.gapRatio) + ", below the threshold " +
                           Scientific(options.smallGapRatio) +
                           "), and the small-gap fallback is disallowed, so the " +
                           std::string(ToString(PreconditionerKind::kDiagonal)) +
                           " preconditioner is refused rather than used or replaced quietly"});
    }

    // The fallback: clamp the diagonal to a floor set by its own largest entry,
    // then invert. The clamp keeps the preconditioned direction bounded, which
    // is what lets a small-gap system converge where the plain inverse stalls.
    const double floor = options.regularisationRatio * largest;
    prepared.kind = PreconditionerKind::kRegularised;

    for (std::size_t i = 0; i < diagonal.size(); ++i)
    {
        const double magnitude = std::abs(diagonal[i]);
        const double clamped = magnitude < floor ? floor : magnitude;
        prepared.inverse[i] = 1.0 / (diagonal[i] < 0.0 ? -clamped : clamped);
    }

    return prepared;
}

} // namespace qcx::response
