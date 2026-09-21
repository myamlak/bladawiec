// The harmonic interface's shape: the projection, the two transformations, the
// coordinate round trip, and the conventions that come out plausible rather
// than wrong when they change.
#include "qcx/molecule/elements.hpp"
#include "qcx/molecule/harmonic.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace qcx::molecule {
namespace {

// The smallest-to-largest principal-moment ratio the linearity flag is decided
// with. It is the mass-properties module's own default, and it is passed
// explicitly here because the geometry records the threshold it used.
constexpr double kLinearityThreshold = 1e-6;

// ---------------------------------------------------------------------------
// Fixture one: a two-atom molecule, deuterium then hydrogen, on the z axis,
// with a quartic potential in its bond projection,
//
//   V(s) = E0 + k s^2 / 2 + c3 s^3 / 6 + c4 s^4 / 24,
//   s    = (z0 - z1) - the same difference at equilibrium,
//
// so every derivative is known in closed form and what the interface returns
// is checked against algebra rather than against stored output. The masses are
// the element table's own hydrogen and deuterium masses: a mass unit error
// scales every frequency by sqrt(1822.888...) - a factor of 43 - and the
// spectrum still looks like a spectrum.
// ---------------------------------------------------------------------------
constexpr double kForceConstant = 0.36936; ///< Hartree/Bohr^2, a real bond's order of magnitude.
constexpr double kCubicConstant = -0.42; ///< Hartree/Bohr^3; nonzero so the sign is observable.
constexpr double kQuarticConstant = 1.31; ///< Hartree/Bohr^4.
constexpr double kEquilibriumEnergy = -1.13; ///< Hartree.
constexpr double kBondLength = 1.4011; ///< Bohr.

/// The finite-difference step in normal-coordinate units. One unit of Q is
/// 1/sqrt(reduced mass) Bohr of bond projection - about 0.029 Bohr here - so
/// this step is a 0.006 Angstrom stretch, and it is the step the round-off
/// measurement in FiniteDifferenceCubic was taken at.
constexpr double kFiniteDifferenceStep = 0.4;

/// An index into an element's isotope table.
///
/// The isotope list is not mass-ordered for every element - carbon's list
/// starts at its lightest entry, not at its most abundant one - so an index is
/// only meaningful where the fixture names the isotope it means. It is carried
/// as its own type so that the element whose table it indexes cannot be handed
/// over in its place.
struct IsotopeIndex {
    std::size_t value = 0; ///< The position in the element's isotope list.
};

/// An isotope mass (u) from the element table.
/// \param atomicNumber Element whose isotope table is read.
/// \param isotopeIndex Index of the isotope within that element.
/// \returns The isotope's mass in u.
double TableMass(int atomicNumber, IsotopeIndex isotopeIndex) {
    const ElementData* element = FindElement(atomicNumber);
    return kIsotopes[element->isotopeOffset + isotopeIndex.value].mass;
}

/// The most abundant isotope's mass (u), which is what a molecule selects when
/// its atom carries no explicit isotope.
/// \param atomicNumber Element to read.
/// \returns The isotope's mass in u.
double MostAbundantMass(int atomicNumber) {
    return FindElement(atomicNumber)->mostAbundantIsotopeMass;
}

/// A molecule's geometry through the interface, from coordinates in Bohr.
/// \param atoms Atoms in input order; the molecule renumbers them canonically.
/// \param rows One coordinate row per atom, in Bohr.
/// \returns The geometry, or the error that stopped it being built.
qcx::Result<MolecularGeometry> MakeGeometry(std::vector<Atom> atoms,
                                            const std::vector<std::vector<double>>& rows) {
    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({rows.size(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coordinates)(i, static_cast<std::size_t>(d)) = rows[i][static_cast<std::size_t>(d)];
        }
    }

    auto molecule = Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    return ComputeMolecularGeometry(*molecule, kLinearityThreshold);
}

/// Fixture one's geometry: deuterium at the origin, hydrogen at +z.
/// \returns The geometry, or the error that stopped it being built.
qcx::Result<MolecularGeometry> MakeDiatomicGeometry() {
    return MakeGeometry(
        {Atom{"H", 1, TableMass(1, IsotopeIndex{1})}, Atom{"H", 1, TableMass(1, IsotopeIndex{0})}},
        {{0.0, 0.0, 0.0}, {0.0, 0.0, kBondLength}});
}

/// Fixture one's Hessian at equilibrium: the rank-one bond-projection block.
/// \returns The Hessian, shape {6, 6}, Hartree/Bohr^2.
Eigen::MatrixXd DiatomicHessian() {
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(6, 6);
    hessian(2, 2) = kForceConstant;
    hessian(2, 5) = -kForceConstant;
    hessian(5, 2) = -kForceConstant;
    hessian(5, 5) = kForceConstant;
    return hessian;
}

/// Fixture one analysed: its geometry with its Hessian, through the interface.
/// \returns The harmonic structure, or the error that stopped the analysis.
qcx::Result<HarmonicData> AnalyzeDiatomic() {
    auto geometry = MakeDiatomicGeometry();

    if (!geometry.has_value())
    {
        return std::unexpected(geometry.error());
    }

    HarmonicPoint point;
    point.hessian = DiatomicHessian();
    point.fulfilled = HarmonicRequest::kHessian;
    point.converged = true;
    return ComputeHarmonicData(*geometry, point);
}

/// Fixture one's potential re-expressed in its normal coordinate,
/// V(Q) = E0 + lambda Q^2 / 2 + c3 Q^3 / 6 + c4 Q^4 / 24.
///
/// The mode's phase convention makes the stretch s = -Q / S with
/// S = sqrt(reduced mass in electron masses), so the cubic constant comes out
/// with the opposite sign to the potential's own - which is exactly the sign
/// the phase convention decides.
struct NormalCoordinateConstants {
    double lambda; ///< Hartree, the vibrational eigenvalue.
    double c3; ///< Hartree per Q^3.
    double c4; ///< Hartree per Q^4.
    double scale; ///< S = sqrt(reduced mass in electron masses).
};

/// Derives fixture one's normal-coordinate constants from the table's masses.
/// \returns The constants, from algebra rather than from the code under test.
NormalCoordinateConstants DiatomicConstants() {
    const double massD = TableMass(1, IsotopeIndex{1});
    const double massH = TableMass(1, IsotopeIndex{0});
    const double reducedMass = (massD * massH / (massD + massH)) * kAtomicMassUnitToElectronMass;
    const double scale = std::sqrt(reducedMass);

    return NormalCoordinateConstants{kForceConstant / reducedMass,
                                     -kCubicConstant / (scale * scale * scale),
                                     kQuarticConstant / (scale * scale * scale * scale),
                                     scale};
}

// ---------------------------------------------------------------------------
// Fixture two, and it exists to be degenerate: a linear triatomic, carbon
// between two oxygens, with the valence force field
//
//   V = ks (u1^2 + u2^2) / 2 + kint u1 u2 + kb (cx^2 + cy^2) / 2,
//   u1 = z(C) - z(O1),  u2 = z(O2) - z(C),
//   cd = d(C) - (d(O1) + d(O2)) / 2   for d = x, y,
//
// which is invariant under rigid translation and rotation by construction: the
// ends' perpendicular displacements cancel in the bend coordinate, so a rigid
// rotation gives cd = 0 exactly. Its Hessian therefore has exactly five zero
// modes - three translations and two rotations, the third vanishing on a
// linear geometry - and its two bends are one exactly degenerate pair, which
// is the case that decides whether the normal modes are reproducible.
// ---------------------------------------------------------------------------
constexpr double kBondTension = 0.9; ///< Hartree/Bohr^2 for the two stretches.
constexpr double kStretchCoupling = 0.2; ///< Hartree/Bohr^2 between them.
constexpr double kBendTension = 0.1; ///< Hartree/Bohr^2 for the bends.
constexpr double kTriatomicHalfLength = 2.2; ///< Bohr between carbon and each oxygen.

/// Fixture two's geometry: O-C-O along z, carbon at the origin.
/// \returns The geometry, or the error that stopped it being built.
qcx::Result<MolecularGeometry> MakeTriatomicGeometry() {
    return MakeGeometry(
        {Atom{"C", 6, 0.0}, Atom{"O", 8, 0.0}, Atom{"O", 8, 0.0}},
        {{0.0, 0.0, 0.0}, {0.0, 0.0, -kTriatomicHalfLength}, {0.0, 0.0, kTriatomicHalfLength}});
}

/// One of fixture two's internal coordinates as a Cartesian coefficient vector.
///
/// The canonical atom order is carbon first, then the two oxygens by z, so the
/// z coordinates are indices 2, 5 and 8 and the x coordinates 0, 3 and 6.
/// \param kind 0 = u1, 1 = u2, 2 = cx, 3 = cy.
/// \returns The coefficient vector, size 9.
Eigen::VectorXd TriatomicCoordinate(int kind) {
    Eigen::VectorXd vector = Eigen::VectorXd::Zero(9);
    const int carbonZ = 2; ///< Atom 0 is carbon: index 3*0 + 2.
    const int firstOxygenZ = 5; ///< Atom 1 is the oxygen at -z.
    const int secondOxygenZ = 8; ///< Atom 2 is the oxygen at +z.

    if (kind == 0)
    {
        vector[carbonZ] = 1.0;
        vector[firstOxygenZ] = -1.0;
    } else if (kind == 1)
    {
        vector[secondOxygenZ] = 1.0;
        vector[carbonZ] = -1.0;
    } else
    {
        const int direction = kind - 2;
        vector[direction] = 1.0;
        vector[3 + direction] = -0.5;
        vector[6 + direction] = -0.5;
    }

    return vector;
}

/// Fixture two's Hessian, Hartree/Bohr^2: the valence force field's quadratic
/// form in the four coordinates above.
/// \returns The Hessian, shape {9, 9}.
Eigen::MatrixXd TriatomicHessian() {
    const Eigen::VectorXd u1 = TriatomicCoordinate(0);
    const Eigen::VectorXd u2 = TriatomicCoordinate(1);
    const Eigen::VectorXd cx = TriatomicCoordinate(2);
    const Eigen::VectorXd cy = TriatomicCoordinate(3);

    return kBondTension * (u1 * u1.transpose() + u2 * u2.transpose()) +
           kStretchCoupling * (u1 * u2.transpose() + u2 * u1.transpose()) +
           kBendTension * (cx * cx.transpose() + cy * cy.transpose());
}

/// The bend pair's analytic eigenvalue: the force constant times the bend
/// coordinate's inverse reduced mass, which is 1/m(C) + 1/(2 m(O)) - the
/// coefficient of the central atom squared over its mass plus two quarters
/// over the oxygens' - in electron masses.
/// \returns The eigenvalue in Hartree/(Bohr^2 electron mass).
double TriatomicBendEigenvalue() {
    const double massC = MostAbundantMass(6) * kAtomicMassUnitToElectronMass;
    const double massO = MostAbundantMass(8) * kAtomicMassUnitToElectronMass;
    return kBendTension * (1.0 / massC + 1.0 / (2.0 * massO));
}

/// Fixture two analysed, optionally with a perturbation added to its Hessian.
/// \param perturbation Symmetric term added to the Hessian, or a zero matrix.
/// \returns The harmonic structure, or the error that stopped the analysis.
qcx::Result<HarmonicData> AnalyzeTriatomic(const Eigen::MatrixXd& perturbation) {
    auto geometry = MakeTriatomicGeometry();

    if (!geometry.has_value())
    {
        return std::unexpected(geometry.error());
    }

    HarmonicPoint point;
    point.hessian = TriatomicHessian() + perturbation;
    point.fulfilled = HarmonicRequest::kHessian;
    point.converged = true;
    return ComputeHarmonicData(*geometry, point);
}

// ---------------------------------------------------------------------------
// The stub the design asks for: the interface implemented over fixture one's
// analytic potential, which is what proves the interface can be implemented at
// all before anything expensive exists.
// ---------------------------------------------------------------------------
class QuarticStub final : public HarmonicInterface {
public:
    /// Creates the stub about one equilibrium geometry.
    /// \param equilibrium The geometry the potential is expanded about.
    /// \param energy The potential's value at that geometry, in Hartree.
    QuarticStub(MolecularGeometry equilibrium, double energy) :
        _equilibrium(std::move(equilibrium)), _energy(energy) {}

    /// Evaluates the potential and its derivatives at a geometry.
    /// \param geometry A two-atom geometry along z; other geometries are refused.
    /// \param request Which results to produce; an empty request is refused.
    /// \returns The requested results, or an Error (kInvalidArgument).
    qcx::Result<HarmonicPoint> Evaluate(const MolecularGeometry& geometry,
                                        HarmonicRequest request) override {
        if (request == HarmonicRequest::kNone)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "quartic stub: the request asks for nothing"});
        }

        if (geometry.AtomCount() != 2 || geometry.coordinatesBohr.cols() != 3)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "quartic stub: this model is a two-atom geometry along z"});
        }

        const double s = Displacement(geometry);
        const double slope =
            kForceConstant * s + 0.5 * kCubicConstant * s * s + kQuarticConstant * s * s * s / 6.0;
        const double curvature =
            kForceConstant + kCubicConstant * s + 0.5 * kQuarticConstant * s * s;

        HarmonicPoint point;
        point.converged = true;
        point.fulfilled = request;

        if (HasRequest(request, HarmonicRequest::kEnergy))
        {
            point.energy = _energy + 0.5 * kForceConstant * s * s +
                           kCubicConstant * s * s * s / 6.0 +
                           kQuarticConstant * s * s * s * s / 24.0;
        }

        if (HasRequest(request, HarmonicRequest::kGradient))
        {
            point.gradient = Eigen::VectorXd::Zero(6);
            point.gradient[2] = slope;
            point.gradient[5] = -slope;
        }

        if (HasRequest(request, HarmonicRequest::kHessian))
        {
            point.hessian = DiatomicHessian() * (curvature / kForceConstant);
        }

        return point;
    }

    /// The equilibrium geometry the potential is expanded about.
    /// \returns The equilibrium geometry.
    const MolecularGeometry& EquilibriumGeometry() const override {
        return _equilibrium;
    }

private:
    /// The bond-projection displacement of a geometry.
    /// \param geometry A two-atom geometry along z.
    /// \returns (z0 - z1) minus the same difference at equilibrium, in Bohr.
    double Displacement(const MolecularGeometry& geometry) const {
        return (geometry.coordinatesBohr(0, 2) - _equilibrium.coordinatesBohr(0, 2)) -
               (geometry.coordinatesBohr(1, 2) - _equilibrium.coordinatesBohr(1, 2));
    }

    MolecularGeometry _equilibrium;
    double _energy;
};

/// The stub's energy at a normal-coordinate displacement, through the whole
/// round trip: normal coordinate -> Cartesian displacement -> the stub.
/// \param data The analysed fixture, whose transformation is used.
/// \param displacement The normal-coordinate displacement Q, along mode 0.
/// \returns The energy in Hartree, or NaN when the stub refused the geometry.
double EnergyAtNormalDisplacement(const HarmonicData& data, double displacement) {
    Eigen::VectorXd normalCoordinate = Eigen::VectorXd::Zero(data.normalToCartesian.rows());
    normalCoordinate[0] = displacement;
    const Eigen::VectorXd cartesian = data.normalToCartesian * normalCoordinate;
    MolecularGeometry displaced = data.geometry;

    for (std::size_t i = 0; i < displaced.AtomCount(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            const auto index = static_cast<Eigen::Index>(3 * i + static_cast<std::size_t>(d));
            displaced.coordinatesBohr(static_cast<Eigen::Index>(i), d) += cartesian[index];
        }
    }

    QuarticStub stub(data.geometry, kEquilibriumEnergy);
    const auto point = stub.Evaluate(displaced, HarmonicRequest::kEnergy);

    if (!point.has_value())
    {
        return std::nan("");
    }

    return point->energy;
}

/// The third derivative of the potential at its minimum, from the energies at
/// +-h and +-2h, through the whole round trip.
///
/// The formula is exact for a quartic potential at any step, so there is no
/// truncation error and the only error is the round-off of the energies,
/// amplified by 1/h^3. Measured on this fixture: at h = 0.05 the recovered
/// constant is 3.4e-8 relative off the analytic one and at h = 0.4 it is
/// 1e-10 off, which is the round-off model and not a defect - so the larger
/// step is the one used, and any convention error moves this constant by a
/// factor rather than by 1e-8.
/// \param data The analysed fixture.
/// \param h The step, in normal-coordinate units.
/// \returns The recovered cubic constant, Hartree per Q^3.
double FiniteDifferenceCubic(const HarmonicData& data, double h) {
    return (EnergyAtNormalDisplacement(data, 2.0 * h) - 2.0 * EnergyAtNormalDisplacement(data, h) +
            2.0 * EnergyAtNormalDisplacement(data, -h) -
            EnergyAtNormalDisplacement(data, -2.0 * h)) /
           (2.0 * h * h * h);
}

/// The fourth derivative of the potential at its minimum, by the same route.
/// \param data The analysed fixture.
/// \param h The step, in normal-coordinate units.
/// \returns The recovered quartic constant, Hartree per Q^4. The round-off
/// here is amplified by 1/h^4 rather than 1/h^3, which is why the quartic
/// constant is compared at a looser relative tolerance than the cubic one.
double FiniteDifferenceQuartic(const HarmonicData& data, double h) {
    return (EnergyAtNormalDisplacement(data, 2.0 * h) - 4.0 * EnergyAtNormalDisplacement(data, h) +
            6.0 * EnergyAtNormalDisplacement(data, 0.0) -
            4.0 * EnergyAtNormalDisplacement(data, -h) +
            EnergyAtNormalDisplacement(data, -2.0 * h)) /
           (h * h * h * h);
}

// A known Hessian is mass-weighted, projected, diagonalised, and both
// transformations are built.
TEST(HarmonicInterfaceTest, KnownHessianIsMassWeightedProjectedAndDiagonalised) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());

    const double massD = TableMass(1, IsotopeIndex{1});
    const double massH = TableMass(1, IsotopeIndex{0});
    const double reducedMass = (massD * massH / (massD + massH)) * kAtomicMassUnitToElectronMass;

    // The atom order the rest of the fixture relies on: the heavier isotope
    // first, which is the order the molecule's canonical renumbering gives. The
    // two masses being different is what the phase test turns on, so the
    // fixture's identity is asserted rather than assumed.
    ASSERT_EQ(data->geometry.masses.size(), 2u);
    EXPECT_DOUBLE_EQ(data->geometry.masses[0], massD);
    EXPECT_DOUBLE_EQ(data->geometry.masses[1], massH);
    EXPECT_NEAR(massD, 2.014101778, 1e-9);
    EXPECT_NEAR(massH, 1.0078250321, 1e-9);
    EXPECT_GT(massD, massH);
    EXPECT_DOUBLE_EQ(data->linearityThreshold, kLinearityThreshold);

    // A two-atom molecule is linear: one vibrational mode, and the five
    // removed ones are the mask, not a constant somewhere.
    EXPECT_TRUE(data->isLinear);
    EXPECT_EQ(data->vibrationalModeCount, 1u);
    ASSERT_EQ(data->removedModes.size(), 6u);
    EXPECT_FALSE(data->removedModes[0]);
    EXPECT_TRUE(data->removedModes[5]);
    EXPECT_EQ(data->removedVectors.rows(), 6);
    EXPECT_EQ(data->removedVectors.cols(), 5);

    // Mass weighting: element (i, j) over the square roots of the masses of
    // the atoms owning i and j, in electron masses.
    const double massDElectron = massD * kAtomicMassUnitToElectronMass;
    const double massHElectron = massH * kAtomicMassUnitToElectronMass;
    EXPECT_NEAR(data->hessianMassWeighted(2, 2), kForceConstant / massDElectron, 1e-14);
    EXPECT_NEAR(data->hessianMassWeighted(5, 5), kForceConstant / massHElectron, 1e-14);
    EXPECT_NEAR(data->hessianMassWeighted(2, 5),
                -kForceConstant / std::sqrt(massDElectron * massHElectron),
                1e-14);
    EXPECT_EQ(data->hessianMassWeighted(0, 0), 0.0);

    // The projection removes the removed modes exactly: the projected Hessian
    // annihilates them and leaves the vibrational eigenvalue where the
    // unprojected one had it.
    EXPECT_NEAR(data->hessianVibrational.norm(), kForceConstant / reducedMass, 1e-10);
    EXPECT_NEAR(data->eigenvalues[0], kForceConstant / reducedMass, 1e-12 * kForceConstant);

    for (std::size_t k = 1; k < 6; ++k)
    {
        EXPECT_EQ(data->eigenvalues[static_cast<Eigen::Index>(k)], 0.0);
    }

    // The eigenvalue is the force constant in the normal coordinate, and the
    // frequency is its square root as a wavenumber.
    EXPECT_NEAR(data->frequencies[0],
                std::sqrt(kForceConstant / reducedMass) * kWavenumbersPerHartree,
                1e-6);
    EXPECT_GT(data->frequencies[0], 0.0);

    // Both transformations are built and they span the whole space now.
    EXPECT_EQ(data->normalToCartesian.rows(), 6);
    EXPECT_EQ(data->normalToCartesian.cols(), 6);
    EXPECT_EQ(data->cartesianToNormal.rows(), 6);
    EXPECT_EQ(data->cartesianToNormal.cols(), 6);

    // The mode itself: the analytic mass-weighted eigenvector, canonicalised.
    EXPECT_NEAR(data->normalModes(2, 0), -std::sqrt(massH / (massD + massH)), 1e-12);
    EXPECT_NEAR(data->normalModes(5, 0), std::sqrt(massD / (massD + massH)), 1e-12);
    EXPECT_NEAR(data->normalModes.col(0).norm(), 1.0, 1e-12);
}

// The two transformations are exact inverses, and the projection is the
// identity on the vibrational subspace and zero on the removed one.
TEST(HarmonicInterfaceTest, TransformationsAndProjectorAreConsistent) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());

    const Eigen::MatrixXd identity = data->cartesianToNormal * data->normalToCartesian;
    EXPECT_NEAR((identity - Eigen::MatrixXd::Identity(6, 6)).norm(), 0.0, 1e-12);
    EXPECT_NEAR(
        (data->normalToCartesian * data->cartesianToNormal - Eigen::MatrixXd::Identity(6, 6))
            .norm(),
        0.0,
        1e-12);

    // The projector is symmetric and idempotent in the mass-weighted metric,
    // its rank is the number of vibrational modes, and it is not the identity
    // of the full 3N space - which is what makes "on the vibrational subspace"
    // the precise statement.
    EXPECT_NEAR((data->projector - data->projector.transpose()).norm(), 0.0, 1e-12);
    EXPECT_NEAR((data->projector * data->projector - data->projector).norm(), 0.0, 1e-12);
    EXPECT_NEAR(data->projector.trace(), 1.0, 1e-12);
    // Not the identity: its distance from it is the square root of the five
    // removed directions.
    EXPECT_NEAR((data->projector - Eigen::MatrixXd::Identity(6, 6)).norm(), std::sqrt(5.0), 1e-12);

    // A rigid translation and a rigid rotation both lie in the removed
    // subspace, and the projector lives in the mass-weighted metric: it is the
    // mass-weighted translation that it annihilates, not the Cartesian one.
    // Asserting the Cartesian vector here is exactly the mistake of projecting
    // before mass-weighting, so both sides of that are stated.
    Eigen::VectorXd translation = Eigen::VectorXd::Zero(6);
    translation[2] = 1.0;
    translation[5] = 1.0;
    Eigen::VectorXd massWeightedTranslation = translation;

    for (std::size_t i = 0; i < 6; ++i)
    {
        massWeightedTranslation[static_cast<Eigen::Index>(i)] *=
            std::sqrt(data->geometry.massesPerDirection[i] * kAtomicMassUnitToElectronMass);
    }

    EXPECT_NEAR((data->projector * massWeightedTranslation).norm(), 0.0, 1e-12);
    EXPECT_GT((data->projector * translation).norm(), 1e-3);

    // The transformation, on the other hand, is Cartesian into normal, so
    // feeding it a mass-weighted vector is a different and meaningless
    // operation, and it does not vanish. The projector is the one that lives in
    // the mass-weighted metric.
    EXPECT_GT((data->cartesianToNormal * massWeightedTranslation).norm(), 1.0);

    // A Cartesian translation has no vibrational content: every vibrational
    // component of its normal coordinates is zero. The removed components are
    // not, and they are ordered as the removed vectors are - three
    // translations, then the rotations - so the z translation lands on the
    // third of them, which is normal-coordinate index 3 here.
    const Eigen::VectorXd normalCoordinates = data->cartesianToNormal * translation;
    EXPECT_NEAR(normalCoordinates[0], 0.0, 1e-12);
    EXPECT_NEAR(normalCoordinates[3],
                std::sqrt(data->geometry.masses[0] + data->geometry.masses[1]) *
                    std::sqrt(kAtomicMassUnitToElectronMass),
                1e-9);
    EXPECT_NEAR(normalCoordinates[1], 0.0, 1e-12);
    EXPECT_NEAR(normalCoordinates[2], 0.0, 1e-12);

    // And the projector fixes every vector of its own range: the mode's
    // mass-weighted image survives it unchanged.
    EXPECT_NEAR(
        (data->projector * data->normalModes.col(0) - data->normalModes.col(0)).norm(), 0.0, 1e-12);

    // The removed vectors are orthonormal and complete for the removed
    // subspace: five of them here, and the removed modes of the basis are
    // exactly those directions.
    EXPECT_NEAR(
        (data->removedVectors.transpose() * data->removedVectors - Eigen::MatrixXd::Identity(5, 5))
            .norm(),
        0.0,
        1e-12);

    for (std::size_t k = 0; k < 5; ++k)
    {
        EXPECT_NEAR(
            (data->projector * data->removedVectors.col(static_cast<Eigen::Index>(k))).norm(),
            0.0,
            1e-12);
    }
}

// At zero displacement the stub returns the equilibrium energy, a zero
// gradient, and the same Hessian.
TEST(HarmonicInterfaceTest, ZeroDisplacementReturnsTheEquilibriumPoint) {
    auto geometry = MakeDiatomicGeometry();
    ASSERT_TRUE(geometry.has_value());
    QuarticStub stub(*geometry, kEquilibriumEnergy);

    const auto point = stub.Evaluate(*geometry,
                                     HarmonicRequest::kEnergy | HarmonicRequest::kGradient |
                                         HarmonicRequest::kHessian);
    ASSERT_TRUE(point.has_value());
    EXPECT_TRUE(point->converged);

    EXPECT_DOUBLE_EQ(point->energy, kEquilibriumEnergy);

    ASSERT_EQ(point->gradient.size(), 6);
    EXPECT_EQ(point->gradient.norm(), 0.0);

    ASSERT_EQ(point->hessian.rows(), 6);
    EXPECT_TRUE(point->hessian.isApprox(DiatomicHessian(), 1e-14));

    EXPECT_TRUE(HasRequest(point->fulfilled, HarmonicRequest::kEnergy));
    EXPECT_TRUE(HasRequest(point->fulfilled, HarmonicRequest::kHessian));
    EXPECT_DOUBLE_EQ(stub.EquilibriumGeometry().coordinatesBohr(1, 2), kBondLength);

    // The analysed structure carries the stationarity measure: a zero gradient
    // is the condition under which the projection is exact.
    HarmonicPoint analysed;
    analysed.hessian = DiatomicHessian();
    analysed.gradient = point->gradient;
    analysed.fulfilled = HarmonicRequest::kGradient | HarmonicRequest::kHessian;
    const auto data = ComputeHarmonicData(*geometry, analysed);
    ASSERT_TRUE(data.has_value());
    EXPECT_DOUBLE_EQ(data->gradientNorm, 0.0);
}

// Displace along one normal coordinate, convert to Cartesian, ask the stub for
// the energy and the Hessian, convert back.
TEST(HarmonicInterfaceTest, NormalCoordinateRoundTrip) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());
    const NormalCoordinateConstants constants = DiatomicConstants();

    const double displacement = 0.05;
    Eigen::VectorXd normalCoordinate = Eigen::VectorXd::Zero(6);
    normalCoordinate[0] = displacement;
    const Eigen::VectorXd cartesian = data->normalToCartesian * normalCoordinate;

    MolecularGeometry displaced = data->geometry;

    for (std::size_t i = 0; i < displaced.AtomCount(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            const auto index = static_cast<Eigen::Index>(3 * i + static_cast<std::size_t>(d));
            displaced.coordinatesBohr(static_cast<Eigen::Index>(i), d) += cartesian[index];
        }
    }

    QuarticStub stub(data->geometry, kEquilibriumEnergy);
    const auto point =
        stub.Evaluate(displaced, HarmonicRequest::kEnergy | HarmonicRequest::kHessian);
    ASSERT_TRUE(point.has_value());

    // Back through the inverse: the normal-coordinate displacement is
    // recovered exactly, and nothing leaks into the removed directions.
    const Eigen::VectorXd recovered = data->cartesianToNormal * cartesian;
    EXPECT_NEAR(recovered[0], displacement, 1e-12);

    for (Eigen::Index k = 1; k < recovered.size(); ++k)
    {
        EXPECT_NEAR(recovered[k], 0.0, 1e-12);
    }

    // The request flags are honoured: the gradient was not asked for, so it is
    // empty and the fulfilled mask does not name it.
    EXPECT_EQ(point->gradient.size(), 0);
    EXPECT_FALSE(HasRequest(point->fulfilled, HarmonicRequest::kGradient));
    EXPECT_TRUE(HasRequest(point->fulfilled, HarmonicRequest::kHessian));

    // The energy is the analytic potential at the displacement the mode
    // produced.
    const double q = displacement;
    const double expectedEnergy = kEquilibriumEnergy + 0.5 * constants.lambda * q * q +
                                  constants.c3 * q * q * q / 6.0 +
                                  constants.c4 * q * q * q * q / 24.0;
    EXPECT_NEAR(point->energy, expectedEnergy, 1e-13);

    // The stretch this mode produces is s = -Q / S with S the square root of
    // the reduced mass in electron masses: the phase convention is what puts
    // the minus there, and the minus is what decides the sign of the cubic
    // constant below.
    EXPECT_NEAR(cartesian[2] - cartesian[5], -displacement / constants.scale, 1e-12);

    // The Hessian at the displaced geometry, taken into normal coordinates, is
    // that potential's second derivative there. The leading term is 3.0e-4 and
    // the cubic one 4.9e-7 Ha/Bohr^2, so both are resolved at this tolerance.
    const Eigen::MatrixXd normalHessian =
        data->normalToCartesian.transpose() * point->hessian * data->normalToCartesian;
    const double expectedCurvature =
        constants.lambda + constants.c3 * q + 0.5 * constants.c4 * q * q;
    EXPECT_NEAR(normalHessian(0, 0), expectedCurvature, 1e-12);
}

// The finite-difference cubic and quartic constants match the analytic
// potential's.
TEST(HarmonicInterfaceTest, FiniteDifferenceCubicAndQuarticConstants) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());
    const NormalCoordinateConstants constants = DiatomicConstants();

    const double cubic = FiniteDifferenceCubic(*data, kFiniteDifferenceStep);
    const double quartic = FiniteDifferenceQuartic(*data, kFiniteDifferenceStep);

    EXPECT_NEAR(cubic, constants.c3, 1e-8 * std::abs(constants.c3));
    EXPECT_NEAR(quartic, constants.c4, 1e-6 * std::abs(constants.c4));
}

// The mass unit: the analysis works in electron masses.
TEST(HarmonicInterfaceTest, MassUnitIsElectronMasses) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());

    // The constant is the ratio it claims to be - the 2018 CODATA atomic mass
    // constant over the electron mass - and not its reciprocal.
    EXPECT_NEAR(kAtomicMassUnitToElectronMass, 1.66053906660e-27 / 9.1093837015e-31, 1e-9);
    EXPECT_GT(kAtomicMassUnitToElectronMass, 1000.0);

    const double massD = TableMass(1, IsotopeIndex{1});
    const double massH = TableMass(1, IsotopeIndex{0});

    // The mass-weighted Hessian is the electron-mass form...
    EXPECT_NEAR(data->hessianMassWeighted(2, 2),
                kForceConstant / (massD * kAtomicMassUnitToElectronMass),
                1e-14);
    EXPECT_NEAR(data->hessianMassWeighted(5, 5),
                kForceConstant / (massH * kAtomicMassUnitToElectronMass),
                1e-14);

    // ... and not the u form, which differs by a factor of 1822.888.
    EXPECT_NE(data->hessianMassWeighted(2, 2), kForceConstant / massD);
}

// The displacement sign: a displacement along +Q is a displacement along +Q,
// on either side of the round trip.
TEST(HarmonicInterfaceTest, DisplacementSignIsNotFlipped) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());
    const NormalCoordinateConstants constants = DiatomicConstants();

    // The potential is not symmetric about its minimum when c3 differs from
    // zero, so the two sides of the displacement must differ...
    EXPECT_NE(EnergyAtNormalDisplacement(*data, kFiniteDifferenceStep),
              EnergyAtNormalDisplacement(*data, -kFiniteDifferenceStep));

    // ... and the third derivative recovered from those sides must be the
    // analytic one. A chain that adds where it should subtract walks one side
    // backwards and reports -|c3| where +|c3| is the answer: the potential's
    // own cubic constant is negative (-0.42), and the mode's phase flip is
    // exactly what turns the recovered one positive.
    const double cubic = FiniteDifferenceCubic(*data, kFiniteDifferenceStep);
    EXPECT_LT(kCubicConstant, 0.0);
    EXPECT_GT(constants.c3, 0.0);
    EXPECT_GT(cubic, 0.0);
    EXPECT_NEAR(cubic, constants.c3, 1e-8 * std::abs(constants.c3));
}

// The coordinate ordering: grouped by atom, not by direction.
TEST(HarmonicInterfaceTest, CoordinateOrderingIsGroupedByAtom) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());

    // Index 3*i + d: atom 0's z is index 2 and atom 1's z is index 5.
    EXPECT_NEAR(data->hessianCartesian(2, 2), kForceConstant, 1e-14);
    EXPECT_NEAR(data->hessianCartesian(5, 5), kForceConstant, 1e-14);
    EXPECT_NEAR(data->hessianCartesian(2, 5), -kForceConstant, 1e-14);

    // Under a direction-major ordering the two z coordinates would be indices
    // 4 and 5, so both of these would carry the force constant.
    EXPECT_EQ(data->hessianCartesian(4, 4), 0.0);
    EXPECT_EQ(data->hessianCartesian(1, 1), 0.0);

    // The gradient at a displaced geometry lives in the same order: this
    // geometry moves atom 0 in z only.
    MolecularGeometry displaced = data->geometry;
    displaced.coordinatesBohr(0, 2) += 0.03;
    QuarticStub stub(data->geometry, kEquilibriumEnergy);
    const auto point = stub.Evaluate(displaced, HarmonicRequest::kGradient);
    ASSERT_TRUE(point.has_value());
    ASSERT_EQ(point->gradient.size(), 6);
    EXPECT_NE(point->gradient[2], 0.0);
    EXPECT_NEAR(point->gradient[5], -point->gradient[2], 1e-15);
    EXPECT_EQ(point->gradient[1], 0.0);
    EXPECT_EQ(point->gradient[4], 0.0);
}

// The mode normalisation: the columns are normalised in the mass-weighted
// space.
TEST(HarmonicInterfaceTest, ModeNormalisationIsInTheMassWeightedSpace) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());

    const double massD = TableMass(1, IsotopeIndex{1});
    const double massH = TableMass(1, IsotopeIndex{0});
    const double massDElectron = massD * kAtomicMassUnitToElectronMass;
    const double massHElectron = massH * kAtomicMassUnitToElectronMass;

    const double cartesianZ0 = data->normalToCartesian(2, 0);
    const double cartesianZ1 = data->normalToCartesian(5, 0);

    // The defining statement: the Cartesian column has unit mass-weighted
    // norm, so its plain Cartesian norm is not one.
    EXPECT_NEAR(massDElectron * cartesianZ0 * cartesianZ0 +
                    massHElectron * cartesianZ1 * cartesianZ1,
                1.0,
                1e-12);

    // The Cartesian normalisation would put exactly 1 in this expression; the
    // mass-weighted one puts the value derived below, which is 0.5556 of the
    // reduced mass' reciprocal for this isotope pair.
    const double cartesianNormSquared = cartesianZ0 * cartesianZ0 + cartesianZ1 * cartesianZ1;
    EXPECT_NE(cartesianNormSquared, 1.0);
    EXPECT_NEAR(cartesianNormSquared,
                (massH * massH + massD * massD) / (massD * massH * (massD + massH)) /
                    kAtomicMassUnitToElectronMass,
                1e-15);

    // The reported reduced mass is the same statement: one electron mass per
    // vibrational mode, and the eigenvalue is the force constant because of
    // it. The convention itself is carried as data.
    ASSERT_EQ(data->reducedMasses.size(), 1);
    EXPECT_DOUBLE_EQ(data->reducedMasses[0], 1.0);
    EXPECT_EQ(data->normalisation, NormalModeNormalisation::kMassWeighted);
    EXPECT_EQ(data->phase, ModePhaseConvention::kLargestComponentPositive);
}

// The mode phase: every column may be multiplied by minus one at will, so the
// convention is stated and the cubic constants depend on it.
TEST(HarmonicInterfaceTest, ModePhaseIsCanonicalised) {
    const auto data = AnalyzeDiatomic();
    ASSERT_TRUE(data.has_value());

    const double massD = TableMass(1, IsotopeIndex{1});
    const double massH = TableMass(1, IsotopeIndex{0});

    // The heavier atom is atom 0, so the largest-magnitude component belongs
    // to atom 1 and it is the positive one. "First nonzero component positive"
    // would have put the opposite signs here, which is why the rule is
    // asserted rather than assumed.
    EXPECT_GT(std::abs(data->normalModes(5, 0)), std::abs(data->normalModes(2, 0)));
    EXPECT_GT(data->normalModes(5, 0), 0.0);
    EXPECT_LT(data->normalModes(2, 0), 0.0);
    EXPECT_NEAR(std::abs(data->normalModes(5, 0)), std::sqrt(massD / (massD + massH)), 1e-12);
    EXPECT_NEAR(std::abs(data->normalModes(2, 0)), std::sqrt(massH / (massD + massH)), 1e-12);

    // The rule, over every column - the vibrational one and the five removed
    // ones, whose sign is otherwise the Gram-Schmidt construction's.
    for (Eigen::Index k = 0; k < data->normalModes.cols(); ++k)
    {
        Eigen::Index pivot = 0;
        data->normalModes.col(k).cwiseAbs().maxCoeff(&pivot);
        EXPECT_GE(data->normalModes(pivot, k), 0.0);
    }
}

// The representation of an imaginary frequency: a signed value, a magnitude,
// and a flag, all three saying the same thing.
TEST(HarmonicInterfaceTest, ImaginaryFrequencyIsASignedValueWithAFlag) {
    auto geometry = MakeDiatomicGeometry();
    ASSERT_TRUE(geometry.has_value());

    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(6, 6);
    hessian(2, 2) = -kForceConstant;
    hessian(2, 5) = kForceConstant;
    hessian(5, 2) = kForceConstant;
    hessian(5, 5) = -kForceConstant;

    HarmonicPoint point;
    point.hessian = hessian;
    point.fulfilled = HarmonicRequest::kHessian;
    point.converged = true;

    const auto data = ComputeHarmonicData(*geometry, point);
    ASSERT_TRUE(data.has_value());

    const double reducedMass = (TableMass(1, IsotopeIndex{1}) * TableMass(1, IsotopeIndex{0}) /
                                (TableMass(1, IsotopeIndex{1}) + TableMass(1, IsotopeIndex{0}))) *
                               kAtomicMassUnitToElectronMass;
    const double magnitude = std::sqrt(kForceConstant / reducedMass) * kWavenumbersPerHartree;

    // The imaginary mode stays in the vibrational block: ranking by the
    // structure of the problem keeps it out of the near-zero removed modes.
    EXPECT_EQ(data->vibrationalModeCount, 1u);
    EXPECT_LT(data->eigenvalues[0], 0.0);

    // A magnitude alone would have put a positive number here, so the sign is
    // the assertion - and the flag and the magnitude are carried beside it so
    // that a layer above cannot lose it by taking the magnitude.
    ASSERT_EQ(data->imaginary.size(), 6u);
    ASSERT_EQ(data->frequencyMagnitudes.size(), 6);
    ASSERT_EQ(data->frequencies.size(), 6);
    EXPECT_TRUE(data->imaginary[0]);
    EXPECT_FALSE(data->imaginary[1]);
    EXPECT_LT(data->frequencies[0], 0.0);
    EXPECT_DOUBLE_EQ(data->frequencyMagnitudes[0], magnitude);
    EXPECT_NEAR(data->frequencies[0], -magnitude, 1e-6);
    EXPECT_NEAR(data->frequencies[0], -data->frequencyMagnitudes[0], 1e-12);

    // ... and the near-zero removed modes trail it.
    EXPECT_NEAR(data->eigenvalues[1], 0.0, 1e-15);
}

// The degenerate fixture: one exactly degenerate pair, and the convention that
// makes its modes unique.
TEST(HarmonicInterfaceTest, DegenerateModesAreCanonicalAndReproducible) {
    const auto data = AnalyzeTriatomic(Eigen::MatrixXd::Zero(9, 9));
    ASSERT_TRUE(data.has_value());

    // A linear triatomic has four vibrational modes and five removed ones -
    // the count comes from the geometry, not from a constant.
    EXPECT_TRUE(data->isLinear);
    EXPECT_EQ(data->vibrationalModeCount, 4u);
    ASSERT_EQ(data->removedModes.size(), 9u);
    EXPECT_FALSE(data->removedModes[3]);
    EXPECT_TRUE(data->removedModes[4]);
    EXPECT_EQ(data->removedVectors.cols(), 5);

    // The bends are exactly degenerate and are recorded as one block; the two
    // stretches are not degenerate and are not.
    ASSERT_EQ(data->degeneracyBlocks.size(), 1u);
    EXPECT_EQ(data->degeneracyBlocks[0].modeCount, 2u);
    EXPECT_EQ(data->degeneracyBlocks[0].firstMode, 2u);
    EXPECT_EQ(data->degeneracyBlocks[0].canonicalRotation.rows(), 2);
    EXPECT_NEAR(data->eigenvalues[2], data->eigenvalues[3], 1e-12 * data->eigenvalues[0]);
    EXPECT_NEAR(data->eigenvalues[2], TriatomicBendEigenvalue(), 1e-9 * TriatomicBendEigenvalue());
    EXPECT_GT(std::abs(data->eigenvalues[0] - data->eigenvalues[1]), 1e-3 * data->eigenvalues[0]);

    // The convention's outcome, asserted rather than described: the canonical
    // bend pair is direction-pure, one mode in x and one in y. A solver's
    // arbitrary rotation of the pair would mix them.
    for (std::size_t k = 2; k < 4; ++k)
    {
        const Eigen::VectorXd mode = data->normalModes.col(static_cast<Eigen::Index>(k));
        const double zContent = mode[2] * mode[2] + mode[5] * mode[5] + mode[8] * mode[8];
        const double xContent = mode[0] * mode[0] + mode[3] * mode[3] + mode[6] * mode[6];
        const double yContent = mode[1] * mode[1] + mode[4] * mode[4] + mode[7] * mode[7];

        EXPECT_NEAR(zContent, 0.0, 1e-12);
        EXPECT_TRUE(std::abs(xContent - 1.0) < 1e-9 || std::abs(yContent - 1.0) < 1e-9);
        EXPECT_NEAR(std::min(xContent, yContent), 0.0, 1e-9);
    }

    // The stretch modes stay along z.
    for (std::size_t k = 0; k < 2; ++k)
    {
        const Eigen::VectorXd mode = data->normalModes.col(static_cast<Eigen::Index>(k));
        const double zContent = mode[2] * mode[2] + mode[5] * mode[5] + mode[8] * mode[8];
        EXPECT_NEAR(zContent, 1.0, 1e-9);
    }

    // Reproducibility, and the reason the convention exists. Two perturbations
    // of opposite sign are fed in, each one confined to the degenerate pair and
    // each one splitting it by 1e-10 of its own eigenvalue - the scale a
    // different library, version or thread count perturbs a clustered pair by,
    // and still inside the degeneracy tolerance, so both are one block.
    //
    // The solver's own basis resolves the two cases in the opposite order: its
    // second bend column is the x-polarised mode for one sign and the
    // y-polarised mode for the other, so any quantity indexed by that column is
    // not reproducible. The canonical basis is the same vector in both.
    const Eigen::VectorXd cx = TriatomicCoordinate(2);
    const Eigen::VectorXd cy = TriatomicCoordinate(3);
    const Eigen::MatrixXd split =
        (1e-10 * kBendTension) * (cx * cx.transpose() - cy * cy.transpose());

    const auto perturbedUp = AnalyzeTriatomic(split);
    const auto perturbedDown = AnalyzeTriatomic(-split);
    ASSERT_TRUE(perturbedUp.has_value());
    ASSERT_TRUE(perturbedDown.has_value());
    ASSERT_EQ(perturbedUp->degeneracyBlocks.size(), 1u);
    ASSERT_EQ(perturbedDown->degeneracyBlocks.size(), 1u);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solverUp(perturbedUp->hessianVibrational);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solverDown(perturbedDown->hessianVibrational);
    ASSERT_EQ(solverUp.info(), Eigen::Success);
    ASSERT_EQ(solverDown.info(), Eigen::Success);

    const double rawOverlap =
        std::abs(solverUp.eigenvectors().col(6).dot(solverDown.eigenvectors().col(6)));
    const double canonicalOverlap =
        std::abs(perturbedUp->normalModes.col(2).dot(perturbedDown->normalModes.col(2)));

    EXPECT_LT(rawOverlap, 0.5);
    EXPECT_GT(canonicalOverlap, 1.0 - 1e-6);

    // The two numbers travel with the test's own result, so the contrast is
    // measured rather than asserted to be "not equal".
    RecordProperty("rawSolverBasisOverlap", rawOverlap);
    RecordProperty("canonicalBasisOverlap", canonicalOverlap);

    // The canonical pair is direction-pure for both signs, not merely stable:
    // what the convention hands over does not depend on the solver's choice.
    for (const HarmonicData* variant : {&*perturbedUp, &*perturbedDown})
    {
        const Eigen::VectorXd mode = variant->normalModes.col(2);
        const double xContent = mode[0] * mode[0] + mode[3] * mode[3] + mode[6] * mode[6];
        const double yContent = mode[1] * mode[1] + mode[4] * mode[4] + mode[7] * mode[7];
        EXPECT_NEAR(std::min(xContent, yContent), 0.0, 1e-6);
    }
}

// The geometry type's conventions come from the module's own data.
TEST(HarmonicInterfaceTest, GeometryFromMoleculeCarriesTheTableData) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});
    ASSERT_TRUE(coordinates.has_value());

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coordinates)(i, static_cast<std::size_t>(d)) =
                d == 2 ? kBondLength * static_cast<double>(i) : 0.0;
        }
    }

    // A mass of zero selects the element's most abundant isotope.
    auto molecule =
        Molecule::Create({Atom{"H", 1, 0.0}, Atom{"H", 1, 0.0}}, std::move(*coordinates), 0, 1);
    ASSERT_TRUE(molecule.has_value());

    const auto geometry = ComputeMolecularGeometry(*molecule, kLinearityThreshold);
    ASSERT_TRUE(geometry.has_value());

    EXPECT_EQ(geometry->AtomCount(), 2u);
    EXPECT_DOUBLE_EQ(geometry->masses[0], TableMass(1, IsotopeIndex{0}));
    EXPECT_DOUBLE_EQ(geometry->masses[1], TableMass(1, IsotopeIndex{0}));
    EXPECT_EQ(geometry->atomicNumbers[0], 1);
    EXPECT_DOUBLE_EQ(geometry->coordinatesBohr(1, 2), kBondLength);
    EXPECT_DOUBLE_EQ(geometry->linearityThreshold, kLinearityThreshold);

    // The per-direction masses are the atom-wise ones, repeated atom-major.
    ASSERT_EQ(geometry->massesPerDirection.size(), 6u);

    for (std::size_t i = 0; i < geometry->massesPerDirection.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(geometry->massesPerDirection[i], geometry->masses[i / 3]);
    }

    // Two atoms have a vanishing moment about the bond axis, so the molecule
    // is linear and carries one vibrational mode rather than none.
    EXPECT_TRUE(geometry->isLinear);
}

// An input that cannot be analysed is refused by name rather than answered
// with a plausible number.
TEST(HarmonicInterfaceTest, MalformedInputIsRefused) {
    auto geometry = MakeDiatomicGeometry();
    ASSERT_TRUE(geometry.has_value());

    HarmonicPoint hessianOnly;
    hessianOnly.hessian = DiatomicHessian();
    hessianOnly.fulfilled = HarmonicRequest::kHessian;

    // A Hessian of the wrong size.
    HarmonicPoint wrongSize = hessianOnly;
    wrongSize.hessian = Eigen::MatrixXd::Zero(5, 5);
    const auto sizeResult = ComputeHarmonicData(*geometry, wrongSize);
    ASSERT_FALSE(sizeResult.has_value());
    EXPECT_EQ(sizeResult.error().code, qcx::ErrorCode::kInvalidArgument);

    // A Hessian that is not symmetric: the eigensolver would read one triangle
    // and report the other's values.
    HarmonicPoint asymmetric = hessianOnly;
    asymmetric.hessian(2, 5) *= 2.0;
    const auto symmetryResult = ComputeHarmonicData(*geometry, asymmetric);
    ASSERT_FALSE(symmetryResult.has_value());
    EXPECT_EQ(symmetryResult.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(symmetryResult.error().message.find("symmetric"), std::string::npos);

    // A mass list that does not cover the atoms.
    MolecularGeometry truncated = *geometry;
    truncated.massesPerDirection.pop_back();
    const auto shapeResult = ComputeHarmonicData(truncated, hessianOnly);
    ASSERT_FALSE(shapeResult.has_value());
    EXPECT_EQ(shapeResult.error().code, qcx::ErrorCode::kInvalidArgument);

    // A non-positive mass would take a square root of a negative number and
    // turn every result into a NaN.
    MolecularGeometry negativeMass = *geometry;
    negativeMass.massesPerDirection[0] = -1.0;
    const auto massResult = ComputeHarmonicData(negativeMass, hessianOnly);
    ASSERT_FALSE(massResult.has_value());
    EXPECT_EQ(massResult.error().code, qcx::ErrorCode::kInvalidArgument);

    // A linearity flag that the coordinates contradict would give the wrong
    // number of vibrational modes, so it is refused rather than believed.
    MolecularGeometry mislabelled = *geometry;
    mislabelled.isLinear = false;
    const auto flagResult = ComputeHarmonicData(mislabelled, hessianOnly);
    ASSERT_FALSE(flagResult.has_value());
    EXPECT_EQ(flagResult.error().code, qcx::ErrorCode::kInvalidArgument);

    // The stub refuses a request that asks for nothing.
    QuarticStub stub(*geometry, kEquilibriumEnergy);
    const auto emptyRequest = stub.Evaluate(*geometry, HarmonicRequest::kNone);
    ASSERT_FALSE(emptyRequest.has_value());
    EXPECT_EQ(emptyRequest.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace qcx::molecule
