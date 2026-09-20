#pragma once

// Step-functions shared by RHF and UHF: symmetric orthogonalization, Fock
// diagonalization, and the J/K supermatrix construction are spin-agnostic
// operations on whatever Fock/density matrices a caller hands in - RHF and
// UHF each drive their own density/energy bookkeeping around these.
// Internal to qcx-scf (not a public header); the eventual
// FockBuilder<Derived> CRTP of architecture.md 5.4 can absorb these
// unchanged when/if that lands.

#include "qcx/memory/tensor.hpp"
#include "qcx/scf/diis.hpp"
#include "qcx/scf/uhf.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qcx::scf::internal {

// CDIIS floor: skip the extrapolation once the error is at machine
// precision (the rhf.cpp floor comment applies verbatim).
inline constexpr double kDiisFloorThreshold = 1e-14;

// The robust gate's RMS-density tightening factor: under
// UhfOptions::useRobustGate the RMS leg compares against
// densityTolerance / kDiisDensityGateFactor while the max-density leg
// holds the full densityTolerance.
inline constexpr double kDiisDensityGateFactor = 100.0;

// The robust gate's per-spin DIIS-error bound: provisional
// 1e-4, measured against the stress case. The leg guards NON-STATIONARY
// premature declaration (the direct-path 2.2e-7-off case); it cannot
// discriminate
// the O2 saddle from the pin - the saddle is a genuine stationary point of
// the SCF map too - that is the <S^2> assertion's job (already pinned).
inline constexpr double kDiisErrorGateTolerance = 1e-4;

// The period-2 limit-cycle detector's constants (provisional - measured
// at execution): the measured 2-cycle energies pin to 2e-13
// (E(100)=E(102) at parities 100-103), so the parity tolerance sits five
// orders above the pin and five below any energy movement that matters at
// the gate scale; the alternation floor discriminates a genuine 2-cycle
// from a converged fixed point (where the class gap is ~0). The pair
// (1e-8, 1e-6) is false-positive-free for converging approaches: a damped
// oscillation's parity gaps shrink monotonically, so parity-stable-at-1e-8
// forces the class gap below ~1e-8, far under the floor. A hypothetical
// "quiet" cycle alternating by < 1e-6 Eh is deliberately NOT rescued (the
// D-record; lower the floor, never the parity tolerance, if one appears).
inline constexpr double kTwoCycleParityTolerance = 1e-8;
inline constexpr double kTwoCycleAlternationFloor = 1e-6;
inline constexpr std::size_t kTwoCycleWindow = 6;

// The per-iteration CDIIS status of one RHF iteration: the trace's
// diis-status suffix. Which gate state the extrapolator was in, whether an
// extrapolation actually replaced the Fock, the commutator-error norm of
// the decision, and the extrapolation's coefficients (only when an
// extrapolation ran). The engine fills one per iteration; the C12H26
// engagement defect (the old failure was silent - DIIS "on" yet never
// steering) is diagnosed from these fields.
struct ScfDiisTraceStatus {
    bool useDiis = false; ///< options.useDiis of the run.
    bool floorSkipped = false; ///< The error sat at the machine-precision floor
                               ///< (kDiisFloorThreshold): nothing appended.
    bool appended = false; ///< This iteration's (fock, error) pair entered the history.
    bool extrapolated = false; ///< An extrapolation replaced the Fock this iteration.
    double errorNorm = 0.0; ///< The commutator-error norm ||X^T(F D S - S D F) X||_F.
    std::vector<double> coefficients; ///< The extrapolation's coefficients (extrapolated
                                      ///< only; empty otherwise).
};

// The monitor-only measures of one extrapolation's coefficients (a
// monitor-only increment; the coefficient-trajectory analysis verdict):
// negativeMass = (sum|c_i| - 1)/2 - the total magnitude of the
// negative weights while the unit-sum constraint holds - and
// parityImbalance = |sum_even c_i - sum_odd c_i| / sum|c_i| over the
// 0-based history positions (the trace's diiscoef prints oldest-first, the
// diis.cpp Append = push_back + begin()-eviction order; both measures are
// reversal-invariant anyway). That verdict is explicit that these
// values are LOGGED, never acted on: the C24H50 settling phase carries the
// run's largest negative mass (0.179 at iteration 15, 2-4x the wander
// phase), so any acting threshold calibrated on the wander phase would fire
// through the productive settling phase. No threshold, rescue, or watchdog
// wiring reads them back - the detector paths are untouched.
struct DiisCoefficientMeasures {
    double negativeMass; ///< (sum|c_i| - 1) / 2.
    double parityImbalance; ///< |sum_even c_i - sum_odd c_i| / sum|c_i|.
};

// Computes the monitor-only measures (DiisCoefficientMeasures) of a
// coefficient vector; the degenerate cases (an empty or all-zero list) have
// nothing to measure and return zeros.
inline DiisCoefficientMeasures MeasureDiisCoefficients(std::span<const double> coefficients) {
    double sumAbs = 0.0;
    double sumEven = 0.0;
    double sumOdd = 0.0;

    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
        const double value = coefficients[i];
        sumAbs += std::fabs(value);

        if (i % 2 == 0)
        {
            sumEven += value;
        } else
        {
            sumOdd += value;
        }
    }

    if (sumAbs == 0.0)
    {
        return DiisCoefficientMeasures{0.0, 0.0};
    }

    return DiisCoefficientMeasures{0.5 * (sumAbs - 1.0), std::fabs(sumEven - sumOdd) / sumAbs};
}

// The per-iteration trace writer (the memory-anomaly diagnostics): a
// pure write-only side-channel. When the caller's traceFile is non-empty,
// the SCF loop calls Write once per iteration AFTER the convergence
// decision; the line carries the iteration number (1-based), the wall
// seconds since this writer was created (steady_clock - the loop start),
// the iteration's total energy, the iteration-to-iteration energy change
// (the true |E_n - E_{n-1}|, which is exactly the UHF gate's operand and
// the RHF gate's intended one - rhf.cpp's note documents why the RHF
// gate itself reads the fresh energy against itself), the gate's RMS
// density delta, and the converged flag of that decision. DIIS-on runs
// append the diis-status suffix at the line's end (append-only: the
// consumers tokenize key=value pairs and ignore unknown keys):
// diis=off|floor|wait|ext (the gate state; off = the run's useDiis is
// false, floor = the machine-precision floor guard skipped the pair, wait
// = the pair entered the history and the two-pair minimum is not reached
// yet, ext = an extrapolation replaced the Fock), diiserr=<norm>,
// diiscoef=<c1,c2,...> when ext, and the extrapolation's monitor-only
// measures negMass and parityImb (MeasureDiisCoefficients) when ext.
// useDiis=false runs keep the six-field line shape (no suffix). An empty
// traceFile keeps the zero-cost path: the stream never opens (no
// allocation, no file, no numerical effect - the bit-parity pins are
// absolute).
class ScfTraceWriter {
public:
    explicit ScfTraceWriter(const std::string& traceFile) {
        if (!traceFile.empty())
        {
            _epoch = std::chrono::steady_clock::now();
            _stream.open(traceFile, std::ios::trunc);
        }
    }

    void Write(int iteration,
               double totalEnergy,
               double energyDelta,
               double rmsDensityChange,
               bool converged,
               const ScfDiisTraceStatus& diis = ScfDiisTraceStatus{}) {
        if (!_stream.is_open())
        {
            return;
        }

        const double wallSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - _epoch).count();
        _stream << "iter=" << iteration << " wall=" << std::fixed << std::setprecision(6)
                << wallSeconds << std::defaultfloat << " e=" << std::setprecision(17) << totalEnergy
                << " dE=" << energyDelta << " rmsd=" << rmsDensityChange
                << " conv=" << (converged ? 1 : 0);

        if (diis.useDiis)
        {
            // The gate state token: floor = the machine-precision floor
            // guard skipped the pair; wait = the pair entered the history
            // and the two-pair minimum is not reached; ext = an
            // extrapolation replaced the Fock. (diis=off is unreachable
            // inside this branch - useDiis false keeps the whole suffix.)
            _stream << " diis=";
            _stream << (diis.floorSkipped ? "floor" : diis.extrapolated ? "ext" : "wait");
            _stream << " diiserr=" << diis.errorNorm;

            if (diis.extrapolated)
            {
                _stream << " diiscoef=";

                for (std::size_t j = 0; j < diis.coefficients.size(); ++j)
                {
                    if (j > 0)
                    {
                        _stream << ',';
                    }

                    _stream << diis.coefficients[j];
                }

                // The monitor-only measures: derived from the
                // coefficients just printed, for the calibration population
                // of the wall-clock-monitor trail - logged, never acted on.
                const DiisCoefficientMeasures measures = MeasureDiisCoefficients(diis.coefficients);
                _stream << " negMass=" << measures.negativeMass
                        << " parityImb=" << measures.parityImbalance;
            }
        }

        _stream << '\n';
        _stream.flush();
    }

private:
    std::ofstream _stream;
    std::chrono::steady_clock::time_point _epoch{};
};

// The per-iteration density/Fock binary dump (the C12H26
// discriminating-experiment diagnostics): a pure write-only side-channel,
// RHF only. When the caller's densityDumpFile is non-empty, the loop
// writes ONE stream: the 8-byte magic "QXCDFDMP", a little-endian u64
// matrix dimension, then records, each being a u8 tag, a little-endian
// u64 iteration, a little-endian u64 dimension, and n*n little-endian
// doubles in row-major order. Tags: 0 = overlap (written once, iteration
// field zero), 1 = core Hamiltonian (written once), 2 = density pair
// member, 3 = the PHYSICAL Fock pair member (before any DIIS
// extrapolation). An empty densityDumpFile keeps the zero-cost path: the
// stream never opens (no allocation, no file, no numerical effect - the
// bit-parity pins are absolute). Consumed by
// tools/amf_density_invariants.py.
class ScfDensityDumpWriter {
public:
    explicit ScfDensityDumpWriter(const std::string& densityDumpFile) {
        if (!densityDumpFile.empty())
        {
            _stream.open(densityDumpFile, std::ios::binary | std::ios::trunc);
        }
    }

    ~ScfDensityDumpWriter() {
        _stream.flush(); // The destructor closes and flushes on scope exit.
    }

    void WriteOverlap(const Eigen::MatrixXd& overlap) {
        WriteMatrix(kOverlapTag, 0, overlap);
    }

    void WriteCoreHamiltonian(const Eigen::MatrixXd& core) {
        WriteMatrix(kCoreHamiltonianTag, 0, core);
    }

    void WritePair(std::uint64_t iteration,
                   const Eigen::MatrixXd& density,
                   const Eigen::MatrixXd& fock) {
        WriteMatrix(kDensityTag, iteration, density);
        WriteMatrix(kFockTag, iteration, fock);
    }

private:
    static constexpr char kMagic[8] = {'Q', 'X', 'C', 'D', 'F', 'D', 'M', 'P'};
    static constexpr std::size_t kMagicSize = 8;
    static constexpr std::uint8_t kOverlapTag = 0;
    static constexpr std::uint8_t kCoreHamiltonianTag = 1;
    static constexpr std::uint8_t kDensityTag = 2;
    static constexpr std::uint8_t kFockTag = 3;

    bool _headerWritten = false;

    void WriteMatrix(std::uint8_t tag, std::uint64_t iteration, const Eigen::MatrixXd& matrix) {
        if (!_stream.is_open())
        {
            return;
        }

        const std::uint64_t n = static_cast<std::uint64_t>(matrix.rows());

        if (n == 0 || n != static_cast<std::uint64_t>(matrix.cols()))
        {
            return; // Square matrices only; a non-square caller is a bug.
        }

        if (!_headerWritten)
        {
            _stream.write(kMagic, static_cast<std::streamsize>(kMagicSize));
            WriteLe(n);
            _headerWritten = true;
        }

        _stream.write(reinterpret_cast<const char*>(&tag), 1);
        WriteLe(iteration);
        WriteLe(n);

        for (Eigen::Index row = 0; row < matrix.rows(); ++row)
        {
            for (Eigen::Index col = 0; col < matrix.cols(); ++col)
            {
                WriteLe(matrix(row, col));
            }
        }

        _stream.flush();
    }

    void WriteLe(std::uint64_t value) {
        char bytes[8];

        for (int i = 0; i < 8; ++i)
        {
            bytes[i] = static_cast<char>((value >> (8 * i)) & 0xff);
        }

        _stream.write(bytes, 8);
    }

    void WriteLe(double value) {
        static_assert(sizeof(std::uint64_t) == sizeof(double));
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        WriteLe(bits);
    }

    std::ofstream _stream;
};

inline constexpr int kTwoCycleMinIterations = 8;

// The Coulomb and exchange supermatrices of the dense (uv|ws) repulsion
// tensor, laid out in the tensor's own row-major order: the superindex
// a*n + b addresses tensor index (a, b). The Coulomb supermatrix is
// eri(mu, nu, l, s) - the tensor's natural ordering; the exchange
// supermatrix is the same pass with the inner indices exchanged,
// eri(mu, l, s, nu). Both are density-independent and are built once per
// run. The dense n^4 pair is explicitly temporary: the RI-J path
// replaces it.
struct JkSupermatrices {
    Eigen::MatrixXd coulomb; // E[(mu*n+nu), (l*n+s)] = eri(mu, nu, l, s).
    Eigen::MatrixXd exchange; // E_x[(mu*n+nu), (l*n+s)] = eri(mu, l, s, nu).
};

// Fills both supermatrices in one n^4 pass over the dense tensor.
JkSupermatrices BuildJkSupermatrices(
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri, std::size_t n);

// The overlap eigenvalue floor:
// the RELATIVE floor below which an overlap eigenvector is treated as a
// genuinely linearly dependent direction and REMOVED from the calculation.
//
// This is a SYSTEM property - the basis set and the molecule - not a
// per-method switch. The same basis on the same molecule yields the same
// removed count for RHF, UHF, RKS and UKS; nothing here branches on the
// algorithm. A method-level switch would be the wrong place for it.
//
// The value 1e-8 is relative to the largest overlap eigenvalue, and it is
// measured, not chosen: the smallest observed ratio on a WELL-CONDITIONED
// basis in this repo is def2-TZVPD/C12H26 at 9.50e-08 and
// def2-TZVPD/C24H50 at 5.88e-08 (overlap_condition_probe, 2026-09-14), so
// the floor must sit an order BELOW those or it would drop a direction from
// a routine calculation that runs fine today. Raising it to 1e-7 was
// proposed and REFUTED by that measurement.
//
// The lower bound is the orthogonalizer's own amplification: X carries
// 1/sqrt(s), so round-off epsilon in F appears in F_orth = X^T F X at
// relative size epsilon/s - about 2.2e-8 relative at s = 1e-8, which is
// already at the SCF's own 1e-8 energy gate. Lowering the floor further
// admits noise above the quantity being converged.
inline constexpr double kOverlapEigenvalueFloorTolerance = 1e-8;

// The orthogonalizer's result: the Loewdin orthogonalizer X (n x numKept)
// plus the DISCLOSURE of what the removal dropped. A run that removes
// directions must say so - a silent truncation is a defect, not an
// optimization.
//
// The implicit conversion to Eigen::MatrixXd keeps the existing call sites
// (`const Eigen::MatrixXd x = *orthogonalized;`) unchanged now that X is no
// longer square; it is a deliberate compatibility shim for the rectangular
// transition, not a claim that the count is unimportant. Consumers that must
// act on the removal read numRemoved directly.
struct Orthogonalization {
    Eigen::MatrixXd x; ///< X = U_kept diag(1/sqrt(s_kept)), n x numKept.
    std::size_t numRemoved = 0; ///< Directions dropped (0 = the classic square X).
    /// Smallest KEPT overlap eigenvalue; equals the overall smallest when
    /// nothing was removed. Reported for the run record.
    double smallestKeptEigenvalue = 0.0;
    double largestEigenvalue = 0.0; ///< Largest overlap eigenvalue (the ratio's denominator).

    /// Reads the kept-orthonormal dimension (the SCF's working dimension).
    std::size_t KeptDimension() const noexcept {
        return static_cast<std::size_t>(x.cols());
    }

    operator Eigen::MatrixXd() const { // NOLINT(google-explicit-constructor)
        return x;
    }
};

// Loewdin symmetric orthogonalization [Lowdin1950], with the linear-dependence
// removal: X = U_kept diag(1/sqrt(s_kept)) over the eigenvectors whose overlap
// eigenvalue clears kOverlapEigenvalueFloorTolerance times the largest.
//
// The removed directions are the near-null subspace of S. They carry no
// independent information - which is exactly why 1/sqrt(s) blows up there -
// so removing them and back-transforming is exact up to the threshold rather
// than a smaller calculation. Callers run their SCF in the numKept-dimensional
// orthonormal space and map back with C_AO = X C_orth (n x numKept) and
// D_AO = X D_orth X^T (n x n, full AO dimension, rank numKept).
//
// \returns X and the removal disclosure, or an Error: kInvalidArgument when
//          every direction was removed (no orthonormal space remains) or the
//          overlap is not square or empty.
qcx::Result<Orthogonalization> OrthogonalizeOverlap(const Eigen::MatrixXd& overlap);

// Diagonalizes a Fock matrix in the orthogonal basis: C = X * V, columns
// ascending by energy. This is the transform-and-back-transform pair in one
// line: X^T F X is the numKept-dimensional orthonormal problem the solver
// sees, and the returned X * V is the AO-space coefficient matrix C_AO (n x
// numKept when directions were removed, n x n otherwise). The caller builds
// the density from those columns, so D_AO = X D_orth X^T comes out n x n in
// the full AO dimension with no further work - nothing is neglected.
Eigen::MatrixXd DiagonalizeFock(const Eigen::MatrixXd& fock, const Eigen::MatrixXd& x);

// The virtual-space level shift of Saunders and Hillier [SaundersHillier1973]:
// the Fock matrix is raised by the shift times the projector onto the virtual
// space,
//
//     F' = F + shift · (S·C_v)·(S·C_v)^T ,
//
// with S the overlap, C the coefficient matrix in the AO basis and C_v its
// trailing numKept - numOccupied columns - the virtual ones. C is S-orthonormal
// (C^T S C = I) over the kept directions, so the projector is the AO image of
// the virtual-space identity: C^T (S·C_v)(S·C_v)^T C is the identity on the
// virtual block and zero on the occupied and mixed blocks. A positive shift
// therefore lifts every virtual level by shift and leaves the occupied levels
// where they are, and the occupied-virtual gap it opens damps the density
// oscillation of the plain iterator on a near-degenerate stress case. A zero
// shift is a legal no-op. Consumers: the UHF per-spin path under
// UhfOptions::useLevelShift and the RHF cycle-escape trajectory of rhf.cpp.
// The numOccupied parameterization is the documented future-ROHF hook (the
// open-shell two-shift form: shift_1 on the single Fock over the alpha
// occupied set, then the difference form shift_2 - shift_1 over the beta set)
// - not active behavior.
inline void ApplyVirtualSpaceLevelShift(Eigen::MatrixXd& fock,
                                        const Eigen::MatrixXd& overlap,
                                        const Eigen::MatrixXd& coefficients,
                                        // (shift, numOccupied) are the shift magnitude and
                                        // the occupied count - distinct quantities, fixed
                                        // call order.
                                        //
                                        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                        double shift,
                                        int numOccupied) {
    if (shift == 0.0)
    {
        return;
    }

    const Eigen::Index n = fock.rows();
    // The projected coefficients carry the ORTHONORMAL dimension, which the
    // linear-dependence removal makes smaller than n: coefficients is
    // n x numKept, so the virtual block below is n x (numKept - numOccupied)
    // (the plain square case is numKept == n).
    const Eigen::MatrixXd projected = overlap * coefficients;
    const Eigen::Index numKept = projected.cols();

    if (projected.rows() != n || numOccupied > numKept)
    {
        return; // Shape-inconsistent caller: not a shift this routine can apply.
    }

    const Eigen::Index numVirtual = numKept - numOccupied;
    const Eigen::MatrixXd shiftedVirtual = shift * projected.rightCols(numVirtual);
    fock += shiftedVirtual * projected.rightCols(numVirtual).transpose();
}

// The UHF convergence gate (the uhf.cpp call site; shared here so the
// tests can exercise the predicate directly): the two-way Roothaan-Hall
// pair - energy change plus RMS density change (the /n divisor, the BUG-1
// lesson; never the /n^2 form) - plus, behind UhfOptions::useRobustGate,
// the max-density-change leg at the full densityTolerance and the per-spin
// DIIS-error leg below kDiisErrorGateTolerance, with the RMS leg tightened
// by kDiisDensityGateFactor. The boundary convention is strict <
// throughout, matching the two-way gate.

// The joint-system DIIS extrapolator: one extrapolation subspace shared by the
// alpha and beta channels. Pulay's extrapolation finds the weights that make
// the combination of stored errors stationary - the norm of e = sum_i c_i e_i
// is minimized under the constraint sum_i c_i = 1 [Pulay1980] - and the error
// vector itself is the Fock-density commutator of the improved form, Pulay,
// J. Comput. Chem. 3, 556 (1982). One Lagrange multiplier lambda carries the
// constraint, which gives the bordered system
//
//     [[B, -1], [-1^T, 0]] [c; lambda] = [0; -1],   B_ij = <e_i, e_j>,
//
// whose solution is c = B^-1 1 / (1^T B^-1 1); the bordered system below is
// solved directly rather than through that inverse.
//
// Both spins converge together or not at all, so the error the extrapolation
// acts on is the PAIR (e_alpha, e_beta) - the direct sum of the two spin
// errors - and one coefficient vector is shared by both spins: the i-th
// history entry contributes c_i to the alpha Fock matrix and c_i to the beta
// one. The inner product of a direct sum is the sum of the inner products of
// its parts, so the joint Gram entry is the symmetrized sum over both spins
//
//     B_ij = <e_alpha_i, e_alpha_j> + <e_beta_i, e_beta_j>,
//
// and the two channels enter it on equal footing: an iteration is weighted
// highly only when BOTH spin errors were small on it. That simultaneity is the
// whole content of the coupling - a pair of independent per-spin extrapolators
// minimizes each spin's own error in its own subspace, blind to the other
// channel, and can weigh an iteration that is bad for one spin as long as it
// is good for the other.
//
// The factor 1/2 applied to B is the average over the two channels. It is a
// uniform scaling of the whole B block, which leaves the exact minimizer c
// unchanged (only lambda scales with it); it is kept because the least-squares
// solve is scale-sensitive in floating point. B is singular whenever the
// stored errors are linearly dependent - a repeated or degenerate error pair -
// so the solve is a complete orthogonal decomposition, the least-squares
// minimum-norm one, rather than a factorization that would fail on the pivot.
// The joint form keeps the full history window: the two-phase handoff is a
// property of the per-spin histories and has no counterpart in one system
// shared by both channels.
struct JointSystemDiis {
    // historyLimit: the full history window of one shared system - the joint
    // form has no phase handoff to cap it.
    explicit JointSystemDiis(std::size_t historyLimit) : _historyLimit(historyLimit) {
        _fockHistoryAlpha.reserve(historyLimit);
        _errorHistoryAlpha.reserve(historyLimit);
        _fockHistoryBeta.reserve(historyLimit);
        _errorHistoryBeta.reserve(historyLimit);
    }

    // Appends one (fock, error) pair per spin, evicting the oldest pair
    // beyond the limit.
    void Append(const Eigen::MatrixXd& fockAlpha,
                const Eigen::MatrixXd& errorAlpha,
                const Eigen::MatrixXd& fockBeta,
                const Eigen::MatrixXd& errorBeta) {
        _fockHistoryAlpha.push_back(fockAlpha);
        _errorHistoryAlpha.push_back(errorAlpha);
        _fockHistoryBeta.push_back(fockBeta);
        _errorHistoryBeta.push_back(errorBeta);

        if (_fockHistoryAlpha.size() > _historyLimit)
        {
            _fockHistoryAlpha.erase(_fockHistoryAlpha.begin());
            _errorHistoryAlpha.erase(_errorHistoryAlpha.begin());
            _fockHistoryBeta.erase(_fockHistoryBeta.begin());
            _errorHistoryBeta.erase(_errorHistoryBeta.begin());
        }
    }

    // True once the history holds at least two pairs.
    bool Ready() const noexcept {
        return _fockHistoryAlpha.size() >= 2;
    }

    // Extrapolates BOTH spins from the shared coefficient vector: the bordered
    // joint system above is assembled and solved, and the one vector c is
    // applied to each spin's own Fock history - new_F_sigma = sum_i c_i
    // F_sigma_i. The spins therefore advance on the same weights, which is the
    // coupling the joint system exists for.
    qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> Extrapolate() const {
        if (!Ready())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "DIIS extrapolation needs at least two stored pairs"});
        }

        const Eigen::Index m = static_cast<Eigen::Index>(_fockHistoryAlpha.size());
        Eigen::MatrixXd bordered = Eigen::MatrixXd::Zero(m + 1, m + 1);

        // The joint Gram block: the symmetrized sum over both spins, averaged
        // over the two channels.
        for (Eigen::Index i = 0; i < m; ++i)
        {
            const Eigen::MatrixXd& errorAlphaI = _errorHistoryAlpha[static_cast<std::size_t>(i)];
            const Eigen::MatrixXd& errorBetaI = _errorHistoryBeta[static_cast<std::size_t>(i)];

            for (Eigen::Index j = 0; j < m; ++j)
            {
                const double alphaOverlap =
                    (errorAlphaI.array() * _errorHistoryAlpha[static_cast<std::size_t>(j)].array())
                        .sum();
                const double betaOverlap =
                    (errorBetaI.array() * _errorHistoryBeta[static_cast<std::size_t>(j)].array())
                        .sum();
                bordered(i, j) = 0.5 * (alphaOverlap + betaOverlap);
            }
        }

        // The constraint row and column: sum_i c_i = 1 is carried by the
        // multiplier in the last unknown, which leaves the corner entry zero.
        bordered.row(m).head(m).setConstant(-1.0);
        bordered.col(m).head(m).setConstant(-1.0);

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m + 1);
        rhs(m) = -1.0;

        const Eigen::VectorXd coefficients = bordered.completeOrthogonalDecomposition().solve(rhs);

        Eigen::MatrixXd extrapolatedAlpha = Eigen::MatrixXd::Zero(_fockHistoryAlpha.front().rows(),
                                                                  _fockHistoryAlpha.front().cols());
        Eigen::MatrixXd extrapolatedBeta =
            Eigen::MatrixXd::Zero(_fockHistoryBeta.front().rows(), _fockHistoryBeta.front().cols());

        for (Eigen::Index i = 0; i < m; ++i)
        {
            extrapolatedAlpha += coefficients(i) * _fockHistoryAlpha[static_cast<std::size_t>(i)];
            extrapolatedBeta += coefficients(i) * _fockHistoryBeta[static_cast<std::size_t>(i)];
        }

        return std::make_pair(std::move(extrapolatedAlpha), std::move(extrapolatedBeta));
    }

    // The histories as values (checkpointing): the alpha pair in
    // the first state, the beta pair in the second - the ScfRestartState's
    // two per-spin DiisState fields carry the joint history without new
    // serialization surface (diisAlpha = fock/error alpha, diisBeta =
    // fock/error beta; one joint history entry is a pair of entries, one
    // per state, kept in lockstep).
    std::pair<DiisState, DiisState> Snapshot() const {
        return std::make_pair(DiisState{_fockHistoryAlpha, _errorHistoryAlpha},
                              DiisState{_fockHistoryBeta, _errorHistoryBeta});
    }

    // Replaces the histories from the two per-spin states; both states must
    // carry the same number of entries (a joint history entry is a pair),
    // every matrix must be square and shape-consistent, and the alpha and
    // beta matrices must agree on the shape (a joint history entry is a
    // single coefficient vector over the combined subspace - mismatched
    // per-spin shapes cannot share one). An empty pair is a reset (no-op).
    qcx::Result<void> Restore(const DiisState& alphaState, const DiisState& betaState) {
        if (alphaState.fockHistory.size() != alphaState.errorHistory.size() ||
            betaState.fockHistory.size() != betaState.errorHistory.size() ||
            alphaState.fockHistory.size() != betaState.fockHistory.size())
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the joint DIIS histories must match per spin"});
        }

        if (alphaState.fockHistory.size() > _historyLimit)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the joint DIIS history exceeds the extrapolator's "
                                              "limit"});
        }

        for (std::size_t i = 0; i < alphaState.fockHistory.size(); ++i)
        {
            const Eigen::MatrixXd& fockAlpha = alphaState.fockHistory[i];
            const Eigen::MatrixXd& errorAlpha = alphaState.errorHistory[i];
            const Eigen::MatrixXd& fockBeta = betaState.fockHistory[i];
            const Eigen::MatrixXd& errorBeta = betaState.errorHistory[i];

            if (fockAlpha.rows() != fockAlpha.cols() || fockAlpha.rows() != errorAlpha.rows() ||
                fockAlpha.cols() != errorAlpha.cols() || fockBeta.rows() != fockBeta.cols() ||
                fockBeta.rows() != errorBeta.rows() || fockBeta.cols() != errorBeta.cols() ||
                fockAlpha.rows() != fockBeta.rows() || fockAlpha.cols() != fockBeta.cols())
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "the joint DIIS history matrices must be square and "
                               "shape-consistent"});
            }
        }

        _fockHistoryAlpha = alphaState.fockHistory;
        _errorHistoryAlpha = alphaState.errorHistory;
        _fockHistoryBeta = betaState.fockHistory;
        _errorHistoryBeta = betaState.errorHistory;
        return {};
    }

private:
    std::size_t _historyLimit;
    std::vector<Eigen::MatrixXd> _fockHistoryAlpha;
    std::vector<Eigen::MatrixXd> _errorHistoryAlpha;
    std::vector<Eigen::MatrixXd> _fockHistoryBeta;
    std::vector<Eigen::MatrixXd> _errorHistoryBeta;
};

// Runs the joint-system DIIS extrapolation when engaged: the
// (fock, error) pair of BOTH spins is appended to the joint extrapolator
// and both Focks are replaced with the extrapolated matrices once the
// history holds at least two pairs. Returns the Focks unchanged when DIIS
// is off or EITHER spin's error is at the floor - a floor-degenerate error
// vector's zero row/column in the joint B-matrix makes the constraint
// system degenerate (the per-spin floor guard's discipline applied to the
// pair: the joint history entry is a pair, so a degenerate spin drops the
// pair).
inline qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> MaybeExtrapolateJointFock(
    const UhfOptions& options,
    JointSystemDiis& diis,
    Eigen::MatrixXd fockAlpha,
    const Eigen::MatrixXd& errorAlpha,
    double diisErrorNormAlpha,
    // fockBeta/errorBeta/diisErrorNormBeta: the beta half of the joint
    // pair, consumed after the alpha half.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Eigen::MatrixXd fockBeta,
    const Eigen::MatrixXd& errorBeta,
    double diisErrorNormBeta) {
    if (!options.useDiis)
    {
        return std::make_pair(std::move(fockAlpha), std::move(fockBeta));
    }

    if (diisErrorNormAlpha <= kDiisFloorThreshold || diisErrorNormBeta <= kDiisFloorThreshold)
    {
        return std::make_pair(std::move(fockAlpha), std::move(fockBeta));
    }

    diis.Append(fockAlpha, errorAlpha, fockBeta, errorBeta);

    if (!diis.Ready())
    {
        return std::make_pair(std::move(fockAlpha), std::move(fockBeta));
    }

    const auto extrapolated = diis.Extrapolate();

    if (!extrapolated.has_value())
    {
        return std::unexpected(extrapolated.error());
    }

    return *extrapolated;
}

inline bool IsUhfConverged(int iteration,
                           double totalEnergy,
                           double previousTotalEnergy,
                           double rmsDensityChange,
                           double maxDensityChange,
                           double diisErrorAlphaNorm,
                           double diisErrorBetaNorm,
                           const UhfOptions& options) {
    const double rmsGate = options.useRobustGate ? options.densityTolerance / kDiisDensityGateFactor
                                                 : options.densityTolerance;
    const bool energyAndRmsGate =
        iteration > 0 && std::fabs(totalEnergy - previousTotalEnergy) < options.energyTolerance &&
        rmsDensityChange < rmsGate;

    if (!options.useRobustGate)
    {
        return energyAndRmsGate;
    }

    return energyAndRmsGate && maxDensityChange < options.densityTolerance &&
           std::max(diisErrorAlphaNorm, diisErrorBetaNorm) < kDiisErrorGateTolerance;
}

// The period-2 limit-cycle detector: a pure read-only observer over
// the recent total-energy history (the rhf.cpp call site keeps the
// chronological window; the IsUhfConverged test-seam precedent). It fires
// when the whole retained window is parity-stable - every |E[i] - E[i-2]|
// gap in the window below parityTolerance, i.e. three consecutive period
// checks over two full periods - AND the two parity classes genuinely
// alternate: |mean(even class) - mean(odd class)| strictly above
// alternationFloor. A converged fixed point fails the floor; a damped
// approach cannot hold parity stability while its alternation still
// exceeds the floor (the constants' comment). A history shorter than the
// full window never fires. The detector only SETS the restart flag at the
// call site - it never perturbs the trajectory. Boundary conventions are
// strict < for the parity gap and strict > for the class gap.
inline bool DetectPeriodTwoCycle(std::span<const double> recentTotalEnergies,
                                 double parityTolerance,
                                 double alternationFloor) {
    if (recentTotalEnergies.size() < kTwoCycleWindow)
    {
        return false;
    }

    for (std::size_t j = 0; j < kTwoCycleWindow / 2; ++j)
    {
        const double gap = recentTotalEnergies[kTwoCycleWindow - 1 - j] -
                           recentTotalEnergies[kTwoCycleWindow - 3 - j];

        if (!(std::fabs(gap) < parityTolerance))
        {
            return false;
        }
    }

    double sumEven = 0.0;
    double sumOdd = 0.0;

    for (std::size_t i = 0; i < kTwoCycleWindow; ++i)
    {
        if (i % 2 == 0)
        {
            sumEven += recentTotalEnergies[i];
        } else
        {
            sumOdd += recentTotalEnergies[i];
        }
    }

    const double meanEven = sumEven / static_cast<double>(kTwoCycleWindow / 2);
    const double meanOdd = sumOdd / static_cast<double>(kTwoCycleWindow / 2);
    return std::fabs(meanEven - meanOdd) > alternationFloor;
}

// The median of the lag-k gaps |E[i] - E[i - k]| over the window's
// available phase checks (the stationary-lag test's statistic).
// For a genuine period-lag sequence every gap is ~0, so the median is
// robust against one transient outlier that a strict all-below test would
// let veto the detection.
inline double MedianLagGap(std::span<const double> recentTotalEnergies, std::size_t lag) {
    std::vector<double> gaps;

    for (std::size_t i = lag; i < recentTotalEnergies.size(); ++i)
    {
        gaps.push_back(std::fabs(recentTotalEnergies[i] - recentTotalEnergies[i - lag]));
    }

    std::sort(gaps.begin(), gaps.end());
    const std::size_t mid = gaps.size() / 2;

    if (gaps.size() % 2 == 1)
    {
        return gaps[mid];
    }

    return 0.5 * (gaps[mid - 1] + gaps[mid]);
}

// The generalized period-k stationarity watchdog: the period-2
// DetectPeriodTwoCycle above is its k = 2 branch, preserved VERBATIM - any
// history that fires the parity test fires this detector identically
// (detectedPeriod = 2), and that firing behavior is unchanged. When the
// parity test does not fire, the higher lags k = 3..(window - 2) are
// scanned ascending and the FIRST stationary lag is the flag (the
// k-reduction rule: a genuine period-p cycle fires at p, never at a
// multiple). Lag k is stationary when the MEDIAN of its gaps across the
// window is strictly below parityTolerance AND every shorter lag
// j in [2, k) keeps its gap median strictly ABOVE alternationFloor - the
// floor discipline extended to every lag: a "quiet" sub-floor cycle or
// a monotone drift never fires at any k (a drift's short lags sit below
// the floor), while a genuine period-k's shorter lags mix the distinct
// cycle members and stay above it. The 6-energy window bounds the
// confirmable periods: lag 3 gets three phase checks over two full
// periods, lag 4 two checks; longer lags get a single check and are not
// confirmable (the loop bound). A history shorter than the full window
// never fires. detectedPeriod receives the fired lag; it is untouched on
// false. Boundary conventions are strict < for the stationary lag and
// strict > for the shorter-lag floor, matching the period-2 detector. The
// detector only SETS the restart flag at the call site - it never perturbs
// the trajectory.
inline bool DetectStationaryCycle(std::span<const double> recentTotalEnergies,
                                  double parityTolerance,
                                  double alternationFloor,
                                  int& detectedPeriod) {
    if (DetectPeriodTwoCycle(recentTotalEnergies, parityTolerance, alternationFloor))
    {
        detectedPeriod = 2;
        return true;
    }

    if (recentTotalEnergies.size() < kTwoCycleWindow)
    {
        return false;
    }

    for (std::size_t k = 3; k + 2 <= recentTotalEnergies.size(); ++k)
    {
        bool shorterLagsAboveFloor = true;

        for (std::size_t j = 2; j < k; ++j)
        {
            if (!(MedianLagGap(recentTotalEnergies, j) > alternationFloor))
            {
                shorterLagsAboveFloor = false;
                break;
            }
        }

        if (!shorterLagsAboveFloor || !(MedianLagGap(recentTotalEnergies, k) < parityTolerance))
        {
            continue;
        }

        detectedPeriod = static_cast<int>(k);
        return true;
    }

    return false;
}

// The returned-state consistency diagnostics, armed by
// QCX_SCF_RETURN_DIAGNOSTICS (any value): rhf.cpp/uhf.cpp print one stderr line
// per completed run carrying the returned state's two consistency residuals
// beside the total energy and the iteration count, at full precision (the
// - the converged-state contract, and the instrument that made the O2 triplet
// pin's 1.3e-6 gap measurable on the run that produced it rather than
// reconstructed from a test). The same numbers are result fields (HfResult /
// UhfResult). Nothing is computed when the variable is unset, and no run's
// numbers change either way.
inline bool ReturnDiagnosticsEnabled() {
    static const bool enabled = std::getenv("QCX_SCF_RETURN_DIAGNOSTICS") != nullptr;

    return enabled;
}

} // namespace qcx::scf::internal
