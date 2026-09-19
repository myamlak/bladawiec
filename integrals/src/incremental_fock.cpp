#include "qcx/integrals/incremental_fock.hpp"

#include "internal/tensor_eigen_bridge.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <utility>

namespace qcx::integrals {

// The per-iteration state: the running Fock accumulator and the previous
// density (the mutable part that makes BuildFock non-const), plus the
// engagement gate and the consecutive-increment counter.
struct IncrementalFockBuilder::State {
    DirectJkFockBuilder _builder;
    Eigen::MatrixXd _previousDensity; ///< Zero until the first call.
    Eigen::MatrixXd _accumulatedFock; ///< The running (H + J - K) minus one H.
    std::size_t _n;
    double _deltaNormEngagementGate;
    int _maxConsecutiveIncremental;
    int _consecutiveIncrementalCount = 0;
    bool _hasPrevious = false;
    double _accumulatedBoundSum = 0.0; ///< Cumulative certified fp32-lane bound.
};

IncrementalFockBuilder::IncrementalFockBuilder(DirectJkFockBuilder builder,
                                               std::size_t n,
                                               double deltaNormEngagementGate,
                                               int maxConsecutiveIncremental) :
    _state(std::make_shared<State>(
        State{std::move(builder), {}, {}, n, deltaNormEngagementGate, maxConsecutiveIncremental})) {
    Reset();
}

void IncrementalFockBuilder::Reset() {
    _state->_previousDensity = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(_state->_n),
                                                     static_cast<Eigen::Index>(_state->_n));
    _state->_accumulatedFock = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(_state->_n),
                                                     static_cast<Eigen::Index>(_state->_n));
    _state->_consecutiveIncrementalCount = 0;
    _state->_hasPrevious = false;
    _state->_accumulatedBoundSum = 0.0;
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> IncrementalFockBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    double* certifiedBoundSumOut) {
    const std::size_t n = _state->_n;

    // Mirror the direct builder's shape check up front: the conversions
    // below assume the n x n tensor this builder was created for, so a
    // wrong-size density must be rejected here instead of silently
    // truncating the matrix.
    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "density shape mismatch"});
    }

    const Eigen::MatrixXd currentDensity = internal::TensorToEigen(density);
    const Eigen::MatrixXd deltaDensity = currentDensity - _state->_previousDensity;
    const double deltaNorm = deltaDensity.norm();

    // Rebuild-from-scratch triggers: first call ever, ΔP too large to
    // trust incremental screening (a big jump - e.g. right after a DIIS
    // extrapolation kick, or the very first iteration where "delta" would
    // otherwise be the full density anyway), or the consecutive-increment
    // cap (guards against numerical drift: incremental screening decides
    // what to skip based on how much density CHANGED, not its absolute
    // size - a quartet whose absolute density is large but changes only
    // slightly every single iteration could in principle sit right at the
    // screening threshold forever and never get a fresh, non-incremental
    // evaluation; periodically forcing a full rebuild bounds how long any
    // such drift can accumulate before it's corrected).
    const bool mustRebuild =
        !_state->_hasPrevious || deltaNorm >= _state->_deltaNormEngagementGate ||
        _state->_consecutiveIncrementalCount >= _state->_maxConsecutiveIncremental;

    const Eigen::MatrixXd contractionDensity = mustRebuild ? currentDensity : deltaDensity;

    // Wrap contractionDensity as a Tensor for the existing BuildFock API.
    auto contractionTensor = internal::EigenToTensor(contractionDensity);

    if (!contractionTensor.has_value())
    {
        return std::unexpected(contractionTensor.error());
    }

    // The wrapper's error budget: the fp32 lane's per-call certified bound
    // (0.0 when the lane is disabled). The linearity identity makes the
    // accumulated Fock a linear sum of this call's contracts, so the
    // per-call bounds add into a certified bound on the RETURNED matrix.
    double perCallBoundSum = 0.0;
    auto result = _state->_builder.BuildFock(*contractionTensor, &perCallBoundSum);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    const Eigen::MatrixXd resultMatrix = internal::TensorToEigen(*result);

    // resultMatrix is H + (J-K contribution of contractionDensity). The
    // linearity identity BuildFock(D1+D2) - H == (BuildFock(D1)-H) +
    // (BuildFock(D2)-H) requires H to appear in the accumulator exactly
    // once, so the extra H of every incremental contract must be stripped.
    if (mustRebuild)
    {
        // Full rebuild: this IS the answer, H included - the previous
        // contracts' fp32-lane errors no longer contribute to the result.
        _state->_accumulatedFock = resultMatrix;
        _state->_consecutiveIncrementalCount = 0;
        _state->_accumulatedBoundSum = perCallBoundSum;
    } else
    {
        // resultMatrix = H + delta-contribution; _accumulatedFock already
        // holds exactly one H from the last full/incremental result.
        // Adding resultMatrix directly would add a SECOND H - subtract one
        // back out via the CoreHamiltonian() accessor (the sanctioned
        // non-copy path: the builder keeps the construction-time Tensor).
        _state->_accumulatedFock +=
            resultMatrix - internal::TensorToEigen(_state->_builder.CoreHamiltonian());
        ++_state->_consecutiveIncrementalCount;
        _state->_accumulatedBoundSum += perCallBoundSum;
    }

    if (certifiedBoundSumOut != nullptr)
    {
        *certifiedBoundSumOut = _state->_accumulatedBoundSum;
    }

    _state->_previousDensity = currentDensity;
    _state->_hasPrevious = true;

    return internal::EigenToTensor(_state->_accumulatedFock);
}

int IncrementalFockBuilder::ConsecutiveIncrementalCountForTesting() const {
    return _state->_consecutiveIncrementalCount;
}

} // namespace qcx::integrals
