#pragma once

#include "qcx/error.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qcx::molecule {

/// Atomic mass unit expressed in electron masses.
///
/// The factor that turns the isotope masses a MolecularGeometry carries (u)
/// into the atomic units the harmonic analysis works in: the ratio of the 2018
/// CODATA atomic mass constant (1.66053906660e-27 kg) to the electron mass
/// (9.1093837015e-31 kg). Masses enter the analysis only through this ratio,
/// so a mass taken in the wrong unit scales every frequency by a constant and
/// the spectrum that comes out still looks like a spectrum.
/// \ingroup qcx-molecule
inline constexpr double kAtomicMassUnitToElectronMass = 1822.888486217313;

/// Wavenumbers (cm^-1) per Hartree.
///
/// The factor that turns a harmonic frequency in atomic units into the
/// spectroscopist's cm^-1: the 2018 CODATA Hartree energy (4.3597447222071e-18
/// J) divided by hc.
/// \ingroup qcx-molecule
inline constexpr double kWavenumbersPerHartree = 219474.6313632;

/// Relative gap below which two vibrational eigenvalues count as one
/// degenerate block.
///
/// The value decides how a degenerate set is recognised; it is deliberately
/// tight, because merging two distinct modes into one block and rotating them
/// into each other would mix real motions, which is a worse error than leaving
/// an unresolved near-degeneracy alone.
/// \ingroup qcx-molecule
inline constexpr double kDegenerateBlockTolerance = 1e-8;

/// The Cartesian structure the electronic-structure machinery is queried at.
///
/// A plain aggregate: no virtual functions, no handles to the machinery that
/// produced it. Every Cartesian vector in this header uses the same ordering -
/// grouped by atom, so index 3*i + d is direction d of atom i - and the same
/// units: Bohr for lengths, u for masses, Hartree for energies.
/// \ingroup qcx-molecule
struct MolecularGeometry {
    /// Coordinates, shape {atomCount, 3}, one row per atom, in Bohr.
    Eigen::MatrixXd coordinatesBohr;
    /// One atomic number per row of coordinatesBohr.
    std::vector<int> atomicNumbers;
    /// One isotope mass per row, in u.
    std::vector<double> masses;
    /// masses expanded to one entry per Cartesian direction (3 * atomCount),
    /// in u, atom-major: entry 3*i + d is direction d of atom i.
    std::vector<double> massesPerDirection;
    /// Whether the structure is linear, which fixes how many of the 3N
    /// translation and rotation modes exist and therefore the number of
    /// vibrational modes.
    bool isLinear = false;
    /// The smallest-to-largest principal-moment ratio the isLinear flag was
    /// decided with. Carried because the flag is a threshold decision: a
    /// nearly linear structure is a different statement from an exactly linear
    /// one, and the number of vibrational modes follows from it.
    double linearityThreshold = 0.0;

    /// Number of atoms: the row count of coordinatesBohr.
    /// \returns The atom count.
    std::size_t AtomCount() const noexcept {
        return static_cast<std::size_t>(coordinatesBohr.rows());
    }
};

/// One evaluation request: any combination of the energy, the gradient and the
/// Hessian at one geometry.
/// \ingroup qcx-molecule
enum class HarmonicRequest : std::uint8_t {
    /// Nothing asked for; an empty request is refused by an implementation
    /// rather than answered with an empty result.
    kNone = 0,
    /// The electronic energy.
    kEnergy = 1U << 0U,
    /// The gradient with respect to the nuclear coordinates.
    kGradient = 1U << 1U,
    /// The Hessian with respect to the nuclear coordinates.
    kHessian = 1U << 2U,
};

/// Combines two requests.
/// \param left One request.
/// \param right The other request.
/// \returns The request asking for everything either of them asks for.
/// \ingroup qcx-molecule
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- both operands are requests; their roles
// are symmetric by definition.
constexpr HarmonicRequest operator|(HarmonicRequest left, HarmonicRequest right) noexcept {
    // The union of two requests is not one of the enumerators, which is what a bitmask is for,
    // and the analyzer reads the enumerator list rather than the underlying type as the cast's
    // domain.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    return static_cast<HarmonicRequest>(static_cast<std::uint8_t>(left) |
                                        static_cast<std::uint8_t>(right));
}

/// Tests whether a request asks for one result.
/// \param request The request to test.
/// \param flag The result to test for.
/// \returns True when \p request includes \p flag.
/// \ingroup qcx-molecule
constexpr bool HasRequest(HarmonicRequest request, HarmonicRequest flag) noexcept {
    return (static_cast<std::uint8_t>(request) & static_cast<std::uint8_t>(flag)) != 0U;
}

/// How the columns of the normal-mode basis are scaled.
///
/// Carried as data rather than left to prose so that a consumer can refuse a
/// convention it does not implement instead of silently misreading the force
/// constants. A second normalisation would add an enumerator, never change
/// this one's meaning.
/// \ingroup qcx-molecule
enum class NormalModeNormalisation : std::uint8_t {
    /// Each column has unit norm in mass-weighted coordinates, which makes the
    /// reduced mass of every mode exactly one electron mass and makes the
    /// eigenvalue the force constant.
    kMassWeighted = 0,
};

/// How the sign of each normal-mode column is fixed.
///
/// Carried for the same reason as NormalModeNormalisation: the cubic, quartic
/// and Coriolis constants depend on the sign, so a consumer that re-derives
/// its own sign is computing a different quantity.
/// \ingroup qcx-molecule
enum class ModePhaseConvention : std::uint8_t {
    /// Each column is scaled so that its largest-magnitude element is
    /// positive, ties broken by the lowest index.
    kLargestComponentPositive = 0,
};

/// What one evaluation produced at one geometry.
///
/// Which results are present is stated twice on purpose: a field carries
/// values exactly when it was asked for, and fulfilled names what the
/// implementation actually returned - so a caller can tell "not requested"
/// from "requested and dropped".
/// \ingroup qcx-molecule
struct HarmonicPoint {
    /// Electronic energy in Hartree.
    double energy = 0.0;
    /// Gradient in Hartree/Bohr, size 3N, atom-major; empty unless the request
    /// asked for it.
    Eigen::VectorXd gradient;
    /// Hessian in Hartree/Bohr^2, shape {3N, 3N}, atom-major; empty unless the
    /// request asked for it.
    Eigen::MatrixXd hessian;
    /// Which of the requested results are present.
    HarmonicRequest fulfilled = HarmonicRequest::kNone;
    /// Whether the underlying calculation converged at this geometry. A
    /// displaced geometry whose calculation did not converge carries numbers
    /// that must not be compared with one that did.
    bool converged = false;
};

/// The seam between the anharmonic layer above and the electronic-structure
/// machinery below.
///
/// The layer above knows exactly this much: it can ask for an energy, a
/// gradient and a Hessian at a Cartesian geometry. It never learns whether
/// what answers is a self-consistent field, a density functional or a response
/// solver.
///
/// An implementation owns the equilibrium state - the orbitals and density,
/// the Fock matrix and the convergence history, the response solver's previous
/// solution vectors - and warm-starts each displaced geometry from it, because
/// a Hessian at a displaced geometry needs its own self-consistent
/// wavefunction. That is why Evaluate is not const, and why
/// EquilibriumGeometry exists: the caller builds its displaced geometries from
/// the equilibrium structure and hands them back to the same object.
/// \ingroup qcx-molecule
class HarmonicInterface {
public:
    /// Required for an interface held and destroyed through a base pointer.
    virtual ~HarmonicInterface() = default;

    /// Evaluates the requested results at a geometry.
    /// \param geometry The Cartesian geometry to evaluate at; its sizes must
    /// agree with each other.
    /// \param request Which results to produce; an empty request is refused.
    /// \returns The requested results, or an Error: kInvalidArgument for a
    /// request that asks for nothing or a geometry whose sizes disagree, and
    /// kConvergenceFailure when the calculation did not converge.
    virtual qcx::Result<HarmonicPoint> Evaluate(const MolecularGeometry& geometry,
                                                HarmonicRequest request) = 0;

    /// The equilibrium geometry this object holds its state at.
    /// \returns The equilibrium geometry, which is the origin of every
    /// displacement the caller builds.
    virtual const MolecularGeometry& EquilibriumGeometry() const = 0;
};

/// One set of degenerate vibrational modes and the rotation applied inside it.
///
/// Within a degenerate set the eigenvectors are fixed only up to a rotation of
/// the subspace, so any quantity computed in that basis - a cubic or quartic
/// constant - is not unique until a convention is chosen. The analysis fixes
/// one deterministically (see HarmonicData::degeneracyBlocks) and records the
/// rotation here; a layer above that has a better convention can apply its own
/// rotation within the block instead.
/// \ingroup qcx-molecule
struct DegeneracyBlock {
    /// Index of the block's first mode in the mode ordering (0-based, inside
    /// the leading vibrational block of HarmonicData).
    std::size_t firstMode = 0;
    /// Number of modes in the block; at least two.
    std::size_t modeCount = 0;
    /// The rotation applied inside the block: the canonical modes are the
    /// solver's modes for this block multiplied by this matrix. Recorded as
    /// the audit trail of the convention, not as data a consumer needs to
    /// rebuild the modes.
    Eigen::MatrixXd canonicalRotation;
};

/// The harmonic layer's handover to everything above it.
///
/// One plain structure: no virtual functions, no handles to the machinery that
/// produced it. Its conventions are its contract, and each of them is silently
/// wrong when it changes, so they are stated once, here:
///
/// - Cartesian vectors and matrices are atom-major: index 3*i + d is direction
///   d of atom i.
/// - Cartesian quantities are in atomic units: Hessians in Hartree/Bohr^2,
///   gradients in Hartree/Bohr, coordinates in Bohr. The masses in geometry are
///   in u; everything derived from them is in electron masses.
/// - The mass-weighted Hessian divides element (i, j) by the square roots of
///   the masses of the atoms owning i and j, in electron masses.
/// - The vibrational modes are the eigenpairs of the mass-weighted Hessian
///   projected onto the subspace orthogonal to the translations and rotations,
///   in the mass-weighted metric. The projection is what removes those modes:
///   its complement is exactly the vibrational subspace, so the mode count
///   follows the geometry instead of a hard-coded five or six.
/// - The mode basis is complete (3N columns) and orthonormal in mass-weighted
///   coordinates, so cartesianToNormal and normalToCartesian are exact
///   inverses of each other. Within the vibrational block the eigenvalues are
///   descending; the removed modes trail it.
/// - The columns are normalised and phased as normalisation and phase state.
///   One unit of normal coordinate k is the Cartesian displacement in column k
///   of normalToCartesian, in Bohr, and the reduced mass of every vibrational
///   mode is one electron mass.
/// - Frequencies are signed values and the sign is also carried as a flag and
///   as a magnitude: an imaginary mode is a negative number in frequencies and
///   a true entry in imaginary, so a layer above cannot take an absolute value
///   and treat it as a real frequency.
/// - The projection is exact only at a stationary geometry: a rotation is a
///   zero mode of the Hessian only there. Off it the projector removes real
///   components, which is why gradientNorm is carried beside the data.
/// \ingroup qcx-molecule
struct HarmonicData {
    /// The geometry the analysis was made at.
    MolecularGeometry geometry;
    /// Energy at that geometry, in Hartree.
    double energy = 0.0;
    /// Gradient at that geometry, in Hartree/Bohr, size 3N, atom-major; empty
    /// when the caller supplied none.
    Eigen::VectorXd gradient;
    /// Euclidean norm of gradient, 0 when it was not supplied. Expected to be
    /// near zero: a gradient that is not means every force constant here sits
    /// on a shifted minimum, and that the rotational projection has removed a
    /// real component.
    double gradientNorm = 0.0;
    /// The Hessian in Cartesian coordinates, {3N, 3N}, Hartree/Bohr^2.
    Eigen::MatrixXd hessianCartesian;
    /// The same Hessian mass-weighted, before projection.
    Eigen::MatrixXd hessianMassWeighted;
    /// The projected mass-weighted Hessian, P * hessianMassWeighted * P. Its
    /// eigenpairs on the vibrational subspace are the modes this structure
    /// carries; the removed modes are exactly its null space.
    Eigen::MatrixXd hessianVibrational;
    /// The projector onto the vibrational subspace in mass-weighted
    /// coordinates, I - Q Q^T with Q the removed vectors. Symmetric and
    /// idempotent, and the transformation between the two subspaces.
    Eigen::MatrixXd projector;
    /// The orthonormalised translation and rotation vectors as columns, in
    /// mass-weighted coordinates: the three translations first (x, y, z), then
    /// the rotations about x, y and z, with a rotation dropped where the
    /// geometry has no such motion (two for a linear molecule, none for a
    /// single atom).
    Eigen::MatrixXd removedVectors;
    /// One entry per mode of the basis, true where the mode was removed from
    /// the vibrational set. The mask, not a count, is the fact: the count is
    /// three translations plus two rotations for a linear molecule and three
    /// plus three otherwise, and a single atom has no rotation at all.
    std::vector<bool> removedModes;
    /// Number of vibrational modes: the modes the mask does not remove.
    std::size_t vibrationalModeCount = 0;
    /// Whether the geometry is linear, as the geometry states it.
    bool isLinear = false;
    /// The threshold that flag was decided with, copied from the geometry.
    double linearityThreshold = 0.0;
    /// Eigenvalues of the projected mass-weighted Hessian, size 3N: the
    /// vibrational modes first in descending order, the removed modes after
    /// them with an eigenvalue of exactly zero, which is what the projection
    /// makes them. Negative means imaginary.
    Eigen::VectorXd eigenvalues;
    /// Unsigned harmonic frequencies in cm^-1, in the order of eigenvalues.
    Eigen::VectorXd frequencyMagnitudes;
    /// Signed harmonic frequencies in cm^-1, in the order of eigenvalues:
    /// sign(eigenvalue) times frequencyMagnitudes. Carried beside the
    /// magnitude and the flag so that the sign cannot be lost by taking the
    /// magnitude.
    Eigen::VectorXd frequencies;
    /// One entry per mode, true where the eigenvalue is negative and the mode
    /// is therefore imaginary rather than merely small.
    std::vector<bool> imaginary;
    /// The eigenvectors of the projected mass-weighted Hessian as columns,
    /// {3N, 3N}, in the order of eigenvalues, in mass-weighted coordinates,
    /// each column in the canonical phase.
    Eigen::MatrixXd normalModes;
    /// The normal-coordinate displacement into a Cartesian one, {3N, 3N}:
    /// column k is the Cartesian displacement in Bohr produced by one unit of
    /// normal coordinate k. The vibrational modes are its leading
    /// vibrationalModeCount columns.
    Eigen::MatrixXd normalToCartesian;
    /// Its exact inverse, {3N, 3N}: the normal coordinate k per unit Cartesian
    /// displacement.
    Eigen::MatrixXd cartesianToNormal;
    /// Effective mass of each vibrational mode, in electron masses and in the
    /// order of the vibrational block. Exactly one for every mode under this
    /// header's normalisation, which is what makes the eigenvalue the force
    /// constant and the frequency its square root; a value other than one
    /// would mean the columns are normalised somewhere else.
    Eigen::VectorXd reducedMasses;
    /// The degenerate sets of the vibrational block, each with the rotation
    /// applied inside it. Blocks of one mode are not listed: their basis needs
    /// no convention.
    std::vector<DegeneracyBlock> degeneracyBlocks;
    /// The normalisation the columns carry; see NormalModeNormalisation.
    NormalModeNormalisation normalisation = NormalModeNormalisation::kMassWeighted;
    /// The phase convention the columns carry; see ModePhaseConvention.
    ModePhaseConvention phase = ModePhaseConvention::kLargestComponentPositive;
};

/// Builds the harmonic structure from the interface's answer at the
/// equilibrium geometry.
///
/// The Hessian is mass-weighted first and projected second, and the projection
/// is made in the mass-weighted metric ([Helgaker2000] for the mass weighting
/// and the normal-mode analysis). The order is not a detail: the mass-weighting
/// map does not commute with a projection built in Cartesian coordinates, so
/// projecting first leaves small couplings to the removed modes and those
/// contaminate the low-frequency end of the spectrum. The translation and
/// rotation vectors are the exact null vectors of the mass-weighted Hessian at
/// a stationary geometry; they are orthonormalised in a fixed order and the
/// projector is I - Q Q^T.
///
/// The number of modes is taken from how many of those vectors exist, not from
/// a constant, so a linear molecule gets five removed modes and a single atom
/// three. A geometry whose coordinates and isLinear flag disagree about that
/// count is refused rather than answered with the wrong number of modes.
///
/// Degenerate sets are made deterministic by the convention described on
/// DegeneracyBlock.
/// \param geometry The equilibrium geometry the analysis is made at; it
/// carries the masses and the linearity flag with its threshold.
/// \param point The interface's answer at that geometry; the Hessian must be
/// present, of shape {3N, 3N} and symmetric, and the gradient, when present,
/// of size 3N.
/// \returns The harmonic structure, or an Error (kInvalidArgument naming the
/// input that cannot be analysed).
/// \ingroup qcx-molecule
inline qcx::Result<HarmonicData> ComputeHarmonicData(const MolecularGeometry& geometry,
                                                     const HarmonicPoint& point) {
    const std::size_t atomCount = geometry.AtomCount();
    const auto directionCount = static_cast<Eigen::Index>(3 * atomCount);

    if (atomCount == 0 || geometry.coordinatesBohr.cols() != 3 ||
        geometry.atomicNumbers.size() != atomCount || geometry.masses.size() != atomCount ||
        geometry.massesPerDirection.size() != static_cast<std::size_t>(directionCount))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "harmonic data: coordinates, atomic numbers, masses and per-direction "
                       "masses must agree on the atom count"});
    }

    if (point.hessian.rows() != directionCount || point.hessian.cols() != directionCount)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "harmonic data: the Hessian must be 3N by 3N for the geometry it was "
                       "evaluated at"});
    }

    if (point.gradient.size() != 0 && point.gradient.size() != directionCount)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "harmonic data: the gradient must be empty or have 3N entries"});
    }

    // A Hessian that is not symmetric makes the eigensolver read one triangle
    // and report the other's values, so refuse it rather than pick a triangle.
    // The tolerance is relative to the matrix's own scale and loose enough for
    // a numerically assembled Hessian.
    constexpr double kSymmetryTolerance = 1e-8;

    if (!point.hessian.isApprox(point.hessian.transpose(), kSymmetryTolerance))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "harmonic data: the Hessian is not symmetric"});
    }

    // Mass weighting, in electron masses: a non-positive mass would take the
    // square root of a negative number and turn every result into a NaN.
    Eigen::VectorXd inverseSqrtMass(directionCount);
    Eigen::VectorXd sqrtMass(directionCount);

    for (std::size_t i = 0; i < static_cast<std::size_t>(directionCount); ++i)
    {
        const double mass = geometry.massesPerDirection[i] * kAtomicMassUnitToElectronMass;

        if (!(mass > 0.0))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "harmonic data: every mass must be positive, one per Cartesian "
                           "direction"});
        }

        const auto index = static_cast<Eigen::Index>(i);
        sqrtMass[index] = std::sqrt(mass);
        inverseSqrtMass[index] = 1.0 / sqrtMass[index];
    }

    const Eigen::MatrixXd hessianMassWeighted =
        inverseSqrtMass.asDiagonal() * point.hessian * inverseSqrtMass.asDiagonal();

    // The removed subspace, in mass-weighted coordinates. A translation moves
    // every atom by the same vector; a rotation moves each atom by the cross
    // product of the axis with its position about the centre of mass. Both are
    // scaled by the square root of the mass, which is what makes them the null
    // vectors of the mass-weighted Hessian rather than of the Cartesian one.
    Eigen::Vector3d centerOfMass = Eigen::Vector3d::Zero();
    double totalMass = 0.0;

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        const double mass = geometry.masses[i];
        totalMass += mass;
        centerOfMass +=
            mass * Eigen::Vector3d(geometry.coordinatesBohr(static_cast<Eigen::Index>(i), 0),
                                   geometry.coordinatesBohr(static_cast<Eigen::Index>(i), 1),
                                   geometry.coordinatesBohr(static_cast<Eigen::Index>(i), 2));
    }

    centerOfMass /= totalMass;

    const std::size_t rotationCount = atomCount == 1 ? 0 : (geometry.isLinear ? 2 : 3);
    std::vector<Eigen::VectorXd> generators;

    for (int d = 0; d < 3; ++d)
    {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(directionCount);

        for (std::size_t i = 0; i < atomCount; ++i)
        {
            translation[static_cast<Eigen::Index>(3 * i + static_cast<std::size_t>(d))] =
                sqrtMass[static_cast<Eigen::Index>(3 * i + static_cast<std::size_t>(d))];
        }

        generators.push_back(std::move(translation));
    }

    for (int d = 0; d < 3; ++d)
    {
        const Eigen::Vector3d axis = Eigen::Vector3d::Unit(d);
        Eigen::VectorXd rotation = Eigen::VectorXd::Zero(directionCount);

        for (std::size_t i = 0; i < atomCount; ++i)
        {
            const Eigen::Vector3d position(
                geometry.coordinatesBohr(static_cast<Eigen::Index>(i), 0) - centerOfMass[0],
                geometry.coordinatesBohr(static_cast<Eigen::Index>(i), 1) - centerOfMass[1],
                geometry.coordinatesBohr(static_cast<Eigen::Index>(i), 2) - centerOfMass[2]);
            const Eigen::Vector3d moved = axis.cross(position);

            for (int k = 0; k < 3; ++k)
            {
                const auto index = static_cast<Eigen::Index>(3 * i + static_cast<std::size_t>(k));
                rotation[index] += sqrtMass[index] * moved[k];
            }
        }

        generators.push_back(std::move(rotation));
    }

    // Gram-Schmidt in the fixed order above, dropping a generator that has
    // become linearly dependent - which is exactly how the rotation about a
    // linear molecule's own axis disappears.
    constexpr double kLinearDependenceTolerance = 1e-8;
    const Eigen::Index generatorCount = static_cast<Eigen::Index>(generators.size());
    Eigen::MatrixXd removedVectors(directionCount, generatorCount);
    Eigen::Index removedCount = 0;

    for (Eigen::Index g = 0; g < generatorCount; ++g)
    {
        Eigen::VectorXd vector = generators[static_cast<std::size_t>(g)];
        const double originalNorm = vector.norm();

        for (Eigen::Index k = 0; k < removedCount; ++k)
        {
            vector -= removedVectors.col(k).dot(vector) * removedVectors.col(k);
        }

        if (vector.norm() <= kLinearDependenceTolerance * originalNorm)
        {
            continue;
        }

        removedVectors.col(removedCount) = vector.normalized();
        ++removedCount;
    }

    // Coordinates and the flag are two statements of the same fact; when they
    // disagree the mode count would be wrong, so say so instead.
    const Eigen::Index expectedRemoved = 3 + static_cast<Eigen::Index>(rotationCount);

    if (removedCount != expectedRemoved)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "harmonic data: isLinear and the coordinates disagree about the number of "
                       "translation and rotation modes"});
    }

    removedVectors.conservativeResize(directionCount, removedCount);

    const Eigen::MatrixXd projector = Eigen::MatrixXd::Identity(directionCount, directionCount) -
                                      removedVectors * removedVectors.transpose();
    // Symmetrised: the projection and the mass-weighted Hessian are both
    // symmetric, so only round-off could make the product asymmetric.
    const Eigen::MatrixXd projected = projector * hessianMassWeighted * projector;
    const Eigen::MatrixXd hessianVibrational = (0.5 * (projected + projected.transpose())).eval();

    const Eigen::Index vibrationalModeCount = directionCount - removedCount;

    // The vibrational subspace is the orthogonal complement of the removed one.
    // Its basis comes from the orthogonal factor of a QR decomposition of the
    // removed vectors, which is the stable way to complete an orthonormal set:
    // scaling unit vectors down against Q would divide by a residual that can
    // be arbitrarily small.
    Eigen::MatrixXd basis(directionCount, directionCount);
    Eigen::VectorXd eigenvalues = Eigen::VectorXd::Zero(directionCount);

    if (vibrationalModeCount > 0)
    {
        Eigen::HouseholderQR<Eigen::MatrixXd> completion(removedVectors);
        const Eigen::MatrixXd removal = completion.householderQ();
        const Eigen::MatrixXd vibrationalBasis = removal.rightCols(vibrationalModeCount);

        // The subspace's own eigenproblem, not a selection of the full one's
        // eigenvalues: an imaginary mode's eigenvalue sits below the removed
        // block's exact zeros rather than above them, so "the largest ones"
        // would hand back a removed mode whenever the geometry is a saddle.
        const Eigen::MatrixXd reduced =
            vibrationalBasis.transpose() * hessianVibrational * vibrationalBasis;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> subspace(0.5 *
                                                                (reduced + reduced.transpose()));

        if (subspace.info() != Eigen::Success)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInternalError,
                           "harmonic data: the vibrational subspace did not diagonalise"});
        }

        // Eigen returns ascending eigenvalues; the vibrational block is read
        // descending.
        for (Eigen::Index k = 0; k < vibrationalModeCount; ++k)
        {
            const Eigen::Index source = vibrationalModeCount - 1 - k;
            basis.col(k) = vibrationalBasis * subspace.eigenvectors().col(source);
            eigenvalues[k] = subspace.eigenvalues()[source];
        }
    }

    for (Eigen::Index k = vibrationalModeCount; k < directionCount; ++k)
    {
        basis.col(k) = removedVectors.col(k - vibrationalModeCount);
    }

    // A degenerate set is fixed only up to a rotation of its subspace, and the
    // rotation a generalised eigensolver returns for it depends on the library,
    // the version and the thread count. Make it unique here by diagonalising a
    // fixed reference operator inside the block: the result depends on the
    // block's subspace and on that operator, never on the solver's choice of
    // basis, so it is reproducible. The reference weights the Cartesian
    // directions (x = 1, y = 2, z = 3) and, where that leaves a set unresolved,
    // the coordinate index; both are conventions rather than physical
    // preferences, and a symmetry-adapted alternative cannot be built here
    // because the symmetry module sits above this one in the module order.
    std::vector<DegeneracyBlock> degeneracyBlocks;

    if (vibrationalModeCount > 1)
    {
        Eigen::VectorXd directionWeight(directionCount);
        Eigen::VectorXd coordinateWeight(directionCount);

        for (Eigen::Index k = 0; k < directionCount; ++k)
        {
            directionWeight[k] = static_cast<double>(k % 3);
            coordinateWeight[k] = static_cast<double>(k);
        }

        std::size_t first = 0;

        while (first < static_cast<std::size_t>(vibrationalModeCount))
        {
            std::size_t last = first + 1;

            while (last < static_cast<std::size_t>(vibrationalModeCount) &&
                   std::abs(eigenvalues[static_cast<Eigen::Index>(last)] -
                            eigenvalues[static_cast<Eigen::Index>(first)]) <=
                       kDegenerateBlockTolerance *
                           std::max(std::abs(eigenvalues[static_cast<Eigen::Index>(first)]), 1e-30))
            {
                ++last;
            }

            const std::size_t blockSize = last - first;

            if (blockSize > 1)
            {
                const auto blockStart = static_cast<Eigen::Index>(first);
                const Eigen::MatrixXd canonical = basis.block(
                    0, blockStart, directionCount, static_cast<Eigen::Index>(blockSize));
                Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(
                    static_cast<Eigen::Index>(blockSize), static_cast<Eigen::Index>(blockSize));

                for (const Eigen::VectorXd* weight : {&directionWeight, &coordinateWeight})
                {
                    const Eigen::MatrixXd restricted =
                        canonical.transpose() * weight->asDiagonal() * canonical;
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> blockSolver(restricted);

                    if (blockSolver.info() != Eigen::Success)
                    {
                        return std::unexpected(
                            qcx::Error{qcx::ErrorCode::kInternalError,
                                       "harmonic data: a degenerate block did not diagonalise"});
                    }

                    const Eigen::VectorXd& blockValues = blockSolver.eigenvalues();
                    const double span =
                        blockValues[static_cast<Eigen::Index>(blockSize) - 1] - blockValues[0];
                    // A reference that is a multiple of the identity on the
                    // block resolves nothing, so the next one is tried; when
                    // none resolves it the solver's basis stands, which is the
                    // one case where a block is not canonical. The scale of
                    // the weights is the coordinate count, so the floor below
                    // is the reference's own magnitude rather than a physical
                    // quantity.
                    if (std::abs(span) >
                        kDegenerateBlockTolerance * std::max(std::abs(blockValues[0]), 1.0))
                    {
                        rotation = blockSolver.eigenvectors();
                        break;
                    }
                }

                basis.block(0, blockStart, directionCount, static_cast<Eigen::Index>(blockSize)) =
                    canonical * rotation;

                DegeneracyBlock block;
                block.firstMode = first;
                block.modeCount = blockSize;
                block.canonicalRotation = rotation;
                degeneracyBlocks.push_back(std::move(block));
            }

            first = last;
        }
    }

    // The phase: a column may be multiplied by minus one at will, and the
    // anharmonic constants depend on that sign. Canonicalise every column on
    // the largest-magnitude element, ties broken by the lowest index.
    for (Eigen::Index k = 0; k < directionCount; ++k)
    {
        Eigen::Index pivot = 0;
        basis.col(k).cwiseAbs().maxCoeff(&pivot);

        if (basis(pivot, k) < 0.0)
        {
            basis.col(k) = -basis.col(k);
        }
    }

    HarmonicData data;
    data.geometry = geometry;
    data.energy = point.energy;
    data.gradient = point.gradient;
    data.gradientNorm = point.gradient.size() == 0 ? 0.0 : point.gradient.norm();
    data.hessianCartesian = point.hessian;
    data.hessianMassWeighted = hessianMassWeighted;
    data.hessianVibrational = hessianVibrational;
    data.projector = projector;
    data.removedVectors = removedVectors;
    data.isLinear = geometry.isLinear;
    data.linearityThreshold = geometry.linearityThreshold;
    data.vibrationalModeCount = static_cast<std::size_t>(vibrationalModeCount);
    data.degeneracyBlocks = std::move(degeneracyBlocks);
    data.removedModes.resize(static_cast<std::size_t>(directionCount));
    data.imaginary.resize(static_cast<std::size_t>(directionCount));
    data.eigenvalues = eigenvalues;
    data.frequencyMagnitudes.resize(directionCount);
    data.frequencies.resize(directionCount);
    data.normalModes.resize(directionCount, directionCount);
    data.normalToCartesian.resize(directionCount, directionCount);
    data.cartesianToNormal.resize(directionCount, directionCount);
    data.reducedMasses = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(vibrationalModeCount));

    for (Eigen::Index k = 0; k < directionCount; ++k)
    {
        const double magnitude = std::sqrt(std::abs(eigenvalues[k]));
        const bool imaginary = eigenvalues[k] < 0.0;

        data.normalModes.col(k) = basis.col(k);
        data.imaginary[static_cast<std::size_t>(k)] = imaginary;
        data.removedModes[static_cast<std::size_t>(k)] = k >= vibrationalModeCount;
        data.frequencyMagnitudes[k] = magnitude * kWavenumbersPerHartree;
        data.frequencies[k] = (imaginary ? -magnitude : magnitude) * kWavenumbersPerHartree;

        // normalToCartesian: the Cartesian displacement of one unit of normal
        // coordinate, in Bohr. cartesianToNormal: the same map inverted, which
        // is the transpose of the mass-weighted mode scaled by the mass, and
        // the two are exact inverses because the basis is orthonormal.
        data.normalToCartesian.col(k) = basis.col(k).cwiseProduct(inverseSqrtMass);
        data.cartesianToNormal.row(k) = basis.col(k).cwiseProduct(sqrtMass).transpose();
    }

    return data;
}

/// Builds a geometry from a validated molecule.
///
/// The masses are the molecule's own isotope masses in u, one per atom and
/// expanded to one per Cartesian direction, and the linearity is measured from
/// the coordinates with the caller's threshold, which is recorded beside the
/// flag it decided.
/// \param molecule Validated molecule.
/// \param linearityThreshold Smallest-to-largest principal-moment ratio below
/// which the structure counts as linear; the same argument
/// ComputeMassProperties takes.
/// \returns The geometry, or an Error when the molecule's coordinates cannot
/// be read back (kInvalidArgument).
/// \ingroup qcx-molecule
inline qcx::Result<MolecularGeometry> ComputeMolecularGeometry(const Molecule& molecule,
                                                               double linearityThreshold) {
    const std::size_t atomCount = molecule.AtomCount();
    const auto& coordinates = molecule.CoordinatesBohr();
    const auto& atoms = molecule.Atoms();

    if (coordinates.Shape()[0] != atomCount || coordinates.Shape()[1] != 3)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "molecular geometry: coordinates must be {atomCount, 3}"});
    }

    MolecularGeometry geometry;
    geometry.coordinatesBohr.resize(static_cast<Eigen::Index>(atomCount), 3);
    geometry.atomicNumbers.resize(atomCount);
    geometry.masses.resize(atomCount);
    geometry.massesPerDirection.resize(3 * atomCount);

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        geometry.atomicNumbers[i] = atoms[i].atomicNumber;
        geometry.masses[i] = atoms[i].isotopicMass;

        for (int d = 0; d < 3; ++d)
        {
            geometry.coordinatesBohr(static_cast<Eigen::Index>(i), d) = coordinates(i, d);
            geometry.massesPerDirection[3 * i + static_cast<std::size_t>(d)] =
                atoms[i].isotopicMass;
        }
    }

    geometry.isLinear = ComputeMassProperties(molecule, linearityThreshold).isLinear;
    geometry.linearityThreshold = linearityThreshold;

    return geometry;
}

} // namespace qcx::molecule
