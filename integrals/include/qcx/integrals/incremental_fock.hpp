#pragma once

/// \file
/// Incremental Fock building: wraps an existing
/// DirectJkFockBuilder and exploits the linearity of J/K in the density
/// (BuildFock(D1+D2) - H == (BuildFock(D1)-H) + (BuildFock(D2)-H)) to
/// screen and contract only the CHANGE in density each iteration, instead
/// of rebuilding the whole Fock matrix from scratch. This class calls only
/// the existing public builder API - the one addition it needs is the
/// DirectJkFockBuilder::CoreHamiltonian() accessor (added alongside, since
/// the delta-accumulation must strip the extra H of every incremental
/// contract).
///
/// No Eigen in this header (integrals public headers keep Eigen
/// implementation-only - the stated CMake policy): the per-iteration state
/// lives in the .cpp, like DirectJkFockBuilder's own State.

#include "qcx/integrals/fock_build.hpp"

#include <cstddef>
#include <memory>

namespace qcx::integrals {

/// Wraps a DirectJkFockBuilder to accumulate Fock contributions
/// incrementally across SCF iterations. NOT thread-safe and NOT copyable -
/// it holds genuinely mutable per-iteration state (the running Fock
/// accumulator and the previous density), so BuildFock here is a non-const
/// method, unlike DirectJkFockBuilder::BuildFock. The state is
/// shared_ptr-held; copies are deleted (aliasing the accumulation across
/// two wrappers would corrupt it), moves transfer the state wholesale.
/// The core Hamiltonian is FIXED at construction (the wrapped builder's H);
/// the delta-accumulation strips it from every incremental contract, so a
/// changed H requires a new builder instance, not Reset().
/// \ingroup qcx-integrals
class IncrementalFockBuilder {
public:
    /// \param builder The underlying direct builder (copied - cheap,
    /// it's a shared_ptr<const State> under the hood).
    /// \param n Basis function count (for the zero-initial density/accumulator).
    /// \param deltaNormEngagementGate Below this Frobenius norm of ΔP,
    /// use the incremental path; at or above it, do a full rebuild and
    /// reset the running state (niedoida's uhf.cpp pattern - see the
    /// rebuild-rationale comment in incremental_fock.cpp).
    /// \param maxConsecutiveIncremental After this many consecutive
    /// incremental steps, force one full rebuild even if ΔP stays small
    /// (guards against slow numerical drift - see the rebuild-rationale
    /// comment in incremental_fock.cpp).
    explicit IncrementalFockBuilder(DirectJkFockBuilder builder,
                                    std::size_t n,
                                    double deltaNormEngagementGate = 0.1,
                                    int maxConsecutiveIncremental = 10);

    /// Computes the Fock matrix for the given density, incrementally when
    /// safe to do so. Must be called with the ACTUAL current density each
    /// SCF iteration, in sequence - this class tracks state between calls
    /// and is not meant to be called out of order or from multiple
    /// threads concurrently.
    /// \param density The same SPATIAL closed-shell density the underlying
    /// DirectJkFockBuilder expects (see DirectJkFockBuilder::BuildFock),
    /// host-canonical rank-2 tensor with shape {n, n}.
    /// \param certifiedBoundSumOut Optional out-parameter receiving the
    /// cumulative sum of the certified fp32-lane bounds of every
    /// BuildFock call made since the last full rebuild (0.0 when the lane
    /// is disabled - DirectJkFockBuilder::BuildFock reports 0.0 then):
    /// each call's certified bound bounds that call's own contract's
    /// fp32-lane error element-wise, and the Fock accumulation is a linear
    /// sum of contracts, so the per-call bounds add into a certified bound
    /// on every element of the RETURNED accumulated matrix. A full rebuild
    /// resets the sum to that call's bound.
    /// \returns The Fock matrix as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        double* certifiedBoundSumOut = nullptr);

    /// Resets to a fresh state (as if just constructed) - call this if
    /// you restart an SCF loop with a new initial guess on the same
    /// builder instance.
    void Reset();

    /// Test-only accessor: the consecutive-incremental-step counter of the
    /// per-iteration state. Pins the maxConsecutiveIncremental rebuild
    /// trigger in the tests.
    /// \returns The number of incremental steps taken since the last full
    /// rebuild (0 right after construction, Reset(), or a full rebuild).
    int ConsecutiveIncrementalCountForTesting() const;

    IncrementalFockBuilder(const IncrementalFockBuilder&) = delete;
    IncrementalFockBuilder& operator=(const IncrementalFockBuilder&) = delete;
    /// Move transfers the per-iteration state wholesale (the shared_ptr
    /// holder); the moved-from instance is empty and must not be used.
    IncrementalFockBuilder(IncrementalFockBuilder&&) = default;
    /// Move assignment, same transfer semantics as the move constructor.
    /// \returns *this.
    IncrementalFockBuilder& operator=(IncrementalFockBuilder&&) = default;

private:
    // The per-iteration state lives in the .cpp (Eigen stays
    // implementation-only in this module's public headers).
    struct State;
    std::shared_ptr<State> _state;
};

} // namespace qcx::integrals
