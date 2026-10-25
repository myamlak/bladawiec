// The total analytic gradient's acceptance: the response state it solves
// against (including the refusals that make a stale or malformed reference
// unable to reach a solve), the term decomposition, the finite-difference gate
// against displaced-geometry energies, the invariants a gradient must satisfy,
// the restricted Kohn-Sham route (the Fock reconstruction, the grid skeleton
// term and the exchange scaling, each against an independent assembly), and
// the dispersion term with its D4 refusal.
//
// The fixture is water/STO-3G at a DISTORTED, C1 geometry - every atom moved
// off the symmetric reference, so no component of any term vanishes by
// symmetry and a misplaced term cannot hide behind a zero.
//
// The derivative integrals are the integrals module's own derivative tier; the
// finite difference below is a central difference of the SCF's total energy,
// taken at two steps so its truncation error is seen rather than assumed.
//
// The Kohn-Sham rows converge a hybrid reference on the same geometry through
// the public energy seam, with the Coulomb and exchange operators
// hand-contracted over the AO tensor, so the Fock the gradient rebuilds and the
// grid term it adds each have an assembly to disagree with that shares none of
// their code.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/dispersion/dispersion.hpp"
#include "qcx/error.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/xc_gradient.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/properties/gradient.hpp"
#include "qcx/response/response_gradient.hpp"
#include "qcx/response/response_kernel.hpp"
#include "qcx/response/response_state.hpp"
#include "qcx/scf/density_energy.hpp"
#include "qcx/scf/gradient_reference.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <excgrid/grid_derivatives.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using qcx::molecule::Atom;
using qcx::molecule::Molecule;
using qcx::properties::BuildResponseState;
using qcx::properties::ComputeTotalGradient;
using qcx::properties::TotalGradientKohnShamRoute;
using qcx::properties::TotalGradientOptions;
using qcx::properties::TotalGradientXcOptions;
using qcx::response::ResponseScreening;
using qcx::response::ResponseState;
using qcx::response::ResponseStateReference;
using qcx::scf::ScfGradientReference;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;

constexpr std::size_t kNumAtoms = 3;
constexpr std::size_t kNumOccupied = 5;
constexpr std::size_t kNumBasis = 7;

// The reference geometry (Bohr), atom-major, written in the order
// Molecule::Create leaves the atoms in: ascending (Z, x, y, z), so the two H
// atoms (Z = 1) come first, the one with the smaller x first, and O (Z = 8)
// last. The ordering is not cosmetic - a gradient entry's atom index is the
// molecule's, and a finite difference that displaced the same index in any
// other order would differentiate another coordinate. The geometry is C1 (O
// displaced along z, the first H in the molecular plane, the second lifted and
// pulled in), so no Cartesian block of any term vanishes by symmetry. Every
// displacement is at most 1e-3 Bohr, far below the ~2.9 Bohr H-H separation,
// so no displaced geometry changes the order.
constexpr std::array<double, 3 * kNumAtoms> kReferenceCoordinates{
    -1.47, 1.05, -0.13, 1.430428808474167, 1.107157044080814, 0.01, 0.0, 0.0, 0.0729};

// The agreement of the derivative walks' terms with the landed surface they
// are copied from: the same call, so this only catches a transcription slip.
constexpr double kTermTolerance = 1e-12;

// The quarter-transform against a direct four-index transform: both contract
// the same AO tensor with the same coefficients, in different summation
// orders, so the difference is rounding.
constexpr double kTransformTolerance = 1e-10;

/// A geometry, as the flat atom-major array a displacement moves in.
struct Geometry {
    std::array<double, 3 * kNumAtoms> coordinates = kReferenceCoordinates;

    /// \param atom The atom index to displace.
    /// \param direction The Cartesian axis to displace along.
    /// \param delta The displacement, in Bohr.
    /// \returns The displaced geometry.
    Geometry Displaced(std::size_t atom, std::size_t direction, double delta) const {
        Geometry moved = *this;
        moved.coordinates[3 * atom + direction] += delta;
        return moved;
    }
};

/// Builds the fixture molecule at one geometry.
///
/// The atom list is written in the canonical order the coordinates arrive in
/// (see kReferenceCoordinates), so Molecule::Create's renumbering is the
/// identity and the fixture index equals the molecule's atom index. Every
/// +-1e-3 Bohr displacement leaves the two H atoms in ascending x, so that
/// stays true at every displaced geometry. The state test asserts it.
/// \param geometry The geometry.
/// \param charge The molecular charge.
/// \param multiplicity The spin multiplicity.
/// \returns The molecule, or an Error.
qcx::Result<Molecule> MakeMolecule(const Geometry& geometry, int charge = 0, int multiplicity = 1) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({kNumAtoms, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            (*coordinates)(atom, direction) = geometry.coordinates[3 * atom + direction];
        }
    }

    coordinates->MarkHostDirty();

    return Molecule::Create(std::vector<Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}, {"O", 8, 0.0}},
                            std::move(*coordinates),
                            charge,
                            multiplicity);
}

/// Runs the SCF at the molecule's geometry and extracts the gradient
/// reference.
///
/// The tolerances are the tight rung of the one convergence ladder
/// (1e-10 energy / 1e-8 density), taken because the finite-difference gate
/// below differentiates the SCF's own energy: a reference converged at the
/// operating default would put its own residue, divided by 2h, into the
/// comparison. The stationarity gate (1e-6) needs a tight reference for the
/// same reason.
/// \param molecule The molecule.
/// \param basis The basis set.
/// \returns The validated reference, or an Error.
qcx::Result<ScfGradientReference> MakeReference(const Molecule& molecule,
                                                const qcx::basisset::BasisSet& basis) {
    const auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basis);
    const auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basis);

    if (!overlap.has_value() || !kinetic.has_value() || !nuclear.has_value() || !eri.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture's integral build failed"});
    }

    qcx::scf::RhfOptions options;
    options.maxIterations = 300;
    options.energyTolerance = 1e-10;
    options.densityTolerance = 1e-8;

    const auto result = qcx::scf::RunRhfScf(
        molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    if (!result->converged)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture's SCF did not converge"});
    }

    return qcx::scf::ExtractGradientReference(*result, kNumOccupied);
}

/// The total energy at a displaced geometry, the finite difference's operand.
/// \param geometry The geometry.
/// \param basis The basis set.
/// \param atom The atom to displace.
/// \param direction The Cartesian axis.
/// \param delta The displacement, in Bohr.
/// \returns The SCF total energy (electronic plus nuclear repulsion), or an
/// Error.
qcx::Result<double> TotalEnergyAt(const Geometry& geometry,
                                  const qcx::basisset::BasisSet& basis,
                                  std::size_t atom,
                                  std::size_t direction,
                                  double delta) {
    const auto molecule = MakeMolecule(geometry.Displaced(atom, direction, delta));

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    const auto reference = MakeReference(*molecule, basis);

    if (!reference.has_value())
    {
        return std::unexpected(reference.error());
    }

    return reference->totalEnergy;
}

/// The nuclear repulsion at a displaced geometry.
/// \param geometry The geometry.
/// \param atom The atom to displace.
/// \param direction The Cartesian axis.
/// \param delta The displacement, in Bohr.
/// \returns The nuclear repulsion energy, or an Error.
qcx::Result<double> NuclearRepulsionAt(const Geometry& geometry,
                                       std::size_t atom,
                                       std::size_t direction,
                                       double delta) {
    const auto molecule = MakeMolecule(geometry.Displaced(atom, direction, delta));

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    return qcx::molecule::NuclearRepulsionEnergy(*molecule);
}

/// The molecule's coordinates, flat and atom-major.
/// \param molecule The molecule, in its own canonical order.
/// \returns The coordinates, 3 per atom.
std::vector<double> Coordinates(const Molecule& molecule) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::vector<double> flat(3 * molecule.AtomCount(), 0.0);

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            flat[3 * atom + direction] = coordinates(atom, direction);
        }
    }

    return flat;
}

/// The row-major flattening of a square Eigen matrix.
/// \param matrix The matrix.
/// \returns Its entries, row-major.
std::vector<double> RowMajor(const Eigen::MatrixXd& matrix) {
    std::vector<double> flat(static_cast<std::size_t>(matrix.size()), 0.0);

    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < matrix.cols(); ++column)
        {
            flat[static_cast<std::size_t>(row) * static_cast<std::size_t>(matrix.cols()) +
                 static_cast<std::size_t>(column)] = matrix(row, column);
        }
    }

    return flat;
}

/// The density tensor the two derivative walks take.
/// \param density The row-major density.
/// \returns The rank-2 CPU tensor, or an Error.
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> DensityTensor(
    const std::vector<double>& density) {
    auto tensor =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({kNumBasis, kNumBasis});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t row = 0; row < kNumBasis; ++row)
    {
        for (std::size_t column = 0; column < kNumBasis; ++column)
        {
            (*tensor)(row, column) = density[row * kNumBasis + column];
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

/// The MO two-electron tensor by the direct four-index transform,
/// (pq|rs) = sum C_mu,p C_nu,q C_lam,r C_sig,s (mu nu | lam sig). It is the
/// independent operand the quarter-transform is checked against.
/// \param aoTensor The AO tensor, row-major.
/// \param coefficients The MO coefficients, row-major, kNumBasis x kNumBasis.
/// \returns The MO tensor, chemists' notation, row-major.
std::vector<double> DirectMoTransform(const std::vector<double>& aoTensor,
                                      const std::vector<double>& coefficients) {
    std::vector<double> mo(kNumBasis * kNumBasis * kNumBasis * kNumBasis, 0.0);

    for (std::size_t p = 0; p < kNumBasis; ++p)
    {
        for (std::size_t q = 0; q < kNumBasis; ++q)
        {
            for (std::size_t r = 0; r < kNumBasis; ++r)
            {
                for (std::size_t s = 0; s < kNumBasis; ++s)
                {
                    double sum = 0.0;

                    for (std::size_t mu = 0; mu < kNumBasis; ++mu)
                    {
                        for (std::size_t nu = 0; nu < kNumBasis; ++nu)
                        {
                            for (std::size_t lambda = 0; lambda < kNumBasis; ++lambda)
                            {
                                for (std::size_t sigma = 0; sigma < kNumBasis; ++sigma)
                                {
                                    sum += coefficients[mu * kNumBasis + p] *
                                           coefficients[nu * kNumBasis + q] *
                                           coefficients[lambda * kNumBasis + r] *
                                           coefficients[sigma * kNumBasis + s] *
                                           aoTensor[((mu * kNumBasis + nu) * kNumBasis + lambda) *
                                                        kNumBasis +
                                                    sigma];
                                }
                            }
                        }
                    }

                    mo[((p * kNumBasis + q) * kNumBasis + r) * kNumBasis + s] = sum;
                }
            }
        }
    }

    return mo;
}

/// A raw ResponseStateReference plus its screening structure, owning every
/// array the spans point into.
///
/// Built here rather than through BuildResponseState so the state's own
/// validation can be exercised on a reference the caller controls: each
/// refusal test mutates exactly one field of what this produces.
struct RawStateInput {
    std::vector<double> coordinates;
    std::vector<double> moCoefficients;
    std::vector<double> orbitalEnergies;
    std::vector<double> density;
    std::vector<double> fock;
    std::vector<double> moTwoElectron;
    ResponseScreening screening;

    /// The state's input, its spans pointing at this object's arrays.
    /// \returns The reference to hand to ResponseState::Create or Rebase.
    ResponseStateReference Input() const {
        ResponseStateReference input;
        input.converged = true;
        input.numBasisFunctions = kNumBasis;
        input.numOccupied = kNumOccupied;
        input.numVirtual = kNumBasis - kNumOccupied;
        input.coordinatesBohr = coordinates;
        input.moCoefficients = moCoefficients;
        input.orbitalEnergies = orbitalEnergies;
        input.density = density;
        input.fock = fock;
        input.moTwoElectron = moTwoElectron;
        input.history.scfIterations = 1;
        return input;
    }
};

/// Builds the raw state input for a reference.
/// \param molecule The molecule.
/// \param basis The basis set.
/// \param reference The converged reference.
/// \param directTransform Build the MO tensor by the direct transform when
/// true; fill it with zeros when false (the refusal tests never reach the
/// operator, and the direct transform is the expensive operand).
/// \returns The raw input, or an Error.
qcx::Result<RawStateInput> MakeRawStateInput(const Molecule& molecule,
                                             const qcx::basisset::BasisSet& basis,
                                             const ScfGradientReference& reference,
                                             bool directTransform) {
    const auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basis);
    // Non-const: the direct MO transform reads the tensor through HostView,
    // the mutable handle on the host copy.
    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basis);

    if (!kinetic.has_value() || !nuclear.has_value() || !eri.has_value())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the raw reference's integral build failed"});
    }

    const Eigen::MatrixXd core = ToMatrix(*kinetic) + ToMatrix(*nuclear);
    const auto evaluated =
        qcx::scf::EvaluateRhfDensityEnergy(molecule, core, *eri, reference.density);

    if (!evaluated.has_value())
    {
        return std::unexpected(evaluated.error());
    }

    RawStateInput raw;
    raw.coordinates = Coordinates(molecule);
    raw.moCoefficients = RowMajor(reference.coefficients);
    raw.orbitalEnergies.assign(reference.orbitalEnergies.data(),
                               reference.orbitalEnergies.data() + reference.orbitalEnergies.size());
    raw.density = RowMajor(reference.density);
    raw.fock = RowMajor(evaluated->fock);

    if (directTransform)
    {
        raw.moTwoElectron = DirectMoTransform(eri->HostView(), raw.moCoefficients);
    } else
    {
        raw.moTwoElectron.assign(kNumBasis * kNumBasis * kNumBasis * kNumBasis, 0.0);
    }

    const auto identity =
        qcx::response::MakeResponseReferenceIdentity(raw.coordinates, raw.density);

    if (!identity.has_value())
    {
        return std::unexpected(identity.error());
    }

    const std::size_t pairCount = kNumBasis * (kNumBasis + 1) / 2;
    raw.screening.geometry = identity->geometry;
    raw.screening.pairValues.assign(pairCount, 1.0);
    raw.screening.pairDerivativeBounds.assign(pairCount, 0.5);
    return raw;
}

/// The total-gradient request every test starts from.
/// \returns The options, at the shipped gates.
TotalGradientOptions MakeOptions() {
    TotalGradientOptions options;
    options.eri.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    return options;
}

// --- the state ---------------------------------------------------------------

// The quarter-transform against the direct transform: the same AO tensor and
// the same coefficients, contracted in a different order. A transform whose
// axis stride or index decode is wrong disagrees here before any derivative
// reads it.
TEST(TotalGradientTest, TheStateMoTensorMatchesTheDirectTransform) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The fixture's index is the molecule's atom index: the gradient's entry
    // indices and the finite difference's displacement indices are the same.
    EXPECT_EQ(Coordinates(*molecule),
              std::vector<double>(kReferenceCoordinates.begin(), kReferenceCoordinates.end()));

    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto state = BuildResponseState(*molecule, *basis, *reference);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    EXPECT_EQ(state->NumBasisFunctions(), kNumBasis);
    EXPECT_EQ(state->NumAtoms(), kNumAtoms);
    EXPECT_EQ(state->Layout().numOccupied, kNumOccupied);
    EXPECT_EQ(state->Layout().numVirtual, kNumBasis - kNumOccupied);
    EXPECT_EQ(state->Report().solves, 0u);
    EXPECT_EQ(state->Screening().pairValues.size(), kNumBasis * (kNumBasis + 1) / 2);
    EXPECT_EQ(state->Screening().pairDerivativeBounds.size(), kNumBasis * (kNumBasis + 1) / 2);

    const auto raw = MakeRawStateInput(*molecule, *basis, *reference, true);
    ASSERT_TRUE(raw.has_value()) << raw.error().message;

    double worst = 0.0;

    for (std::size_t index = 0; index < raw->moTwoElectron.size(); ++index)
    {
        worst =
            std::max(worst, std::abs(state->MoTwoElectron()[index] - raw->moTwoElectron[index]));
    }

    std::cout << "[mo transform] worst |quarter - direct| = " << worst << "\n";
    EXPECT_LT(worst, kTransformTolerance);

    const auto identity =
        qcx::response::MakeResponseReferenceIdentity(state->CoordinatesBohr(), state->Density());
    ASSERT_TRUE(identity.has_value()) << identity.error().message;
    EXPECT_EQ(state->Screening().geometry, identity->geometry);
}

// The state's own refusals, each one field away from the input above: a
// reference that is not converged, a density that is not the occupied
// orbitals' product, a Fock that is not symmetric, and a screening structure
// stamped for another geometry.
TEST(TotalGradientTest, TheStateRefusesANonConvergedReference) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    auto raw = MakeRawStateInput(*molecule, *basis, *reference, false);
    ASSERT_TRUE(raw.has_value()) << raw.error().message;

    auto input = raw->Input();
    input.converged = false;
    const auto state = ResponseState::Create(input, raw->screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("converged"), std::string::npos) << state.error().message;
}

TEST(TotalGradientTest, TheStateRefusesADensityThatIsNotTheOrbitalProduct) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    auto raw = MakeRawStateInput(*molecule, *basis, *reference, false);
    ASSERT_TRUE(raw.has_value()) << raw.error().message;

    raw->density[0] += 0.1;
    const auto state = ResponseState::Create(raw->Input(), raw->screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("2 C_occ C_occ^T"), std::string::npos)
        << state.error().message;
}

TEST(TotalGradientTest, TheStateRefusesAnAsymmetricFock) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    auto raw = MakeRawStateInput(*molecule, *basis, *reference, false);
    ASSERT_TRUE(raw.has_value()) << raw.error().message;

    raw->fock[1] += 0.5;
    const auto state = ResponseState::Create(raw->Input(), raw->screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("not symmetric"), std::string::npos)
        << state.error().message;
}

TEST(TotalGradientTest, TheStateRefusesAForeignScreeningDigest) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    auto raw = MakeRawStateInput(*molecule, *basis, *reference, false);
    ASSERT_TRUE(raw.has_value()) << raw.error().message;

    raw->screening.geometry ^= 0x1ull;
    const auto state = ResponseState::Create(raw->Input(), raw->screening);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(state.error().message.find("geometry digest"), std::string::npos)
        << state.error().message;
}

// An odd electron count has no closed-shell response; the refusal names the
// electron count, not a zero gradient. The reference passed in belongs to the
// neutral molecule on purpose: the electron-count refusal must fire on the
// molecule, before any reference shape is read.
TEST(TotalGradientTest, AnOpenShellMoleculeIsRefusedByName) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto neutral = MakeMolecule(Geometry{});
    ASSERT_TRUE(neutral.has_value()) << neutral.error().message;
    const auto reference = MakeReference(*neutral, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto cation = MakeMolecule(Geometry{}, 1, 2);
    ASSERT_TRUE(cation.has_value()) << cation.error().message;
    ASSERT_EQ(cation->ElectronCount(), 9);

    const auto state = BuildResponseState(*cation, *basis, *reference);
    ASSERT_FALSE(state.has_value());
    EXPECT_EQ(state.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(state.error().message.find("closed shell"), std::string::npos)
        << state.error().message;
}

// --- the assembly ------------------------------------------------------------

// One cold solve per coordinate, the three named right-hand-side
// contributions, and the terms' agreement with the landed derivative
// surfaces they are copied from.
TEST(TotalGradientTest, TheAssemblySolvesEveryCoordinateOnceAndTheTermsMatchTheSurface) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto gradient = ComputeTotalGradient(*molecule, *basis, *reference, MakeOptions());
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    EXPECT_EQ(gradient->atomCount, kNumAtoms);
    EXPECT_EQ(gradient->numBasisFunctions, kNumBasis);
    EXPECT_EQ(gradient->numOccupied, kNumOccupied);
    EXPECT_EQ(gradient->numVirtual, kNumBasis - kNumOccupied);
    EXPECT_EQ(gradient->gradient.size(), static_cast<Eigen::Index>(3 * kNumAtoms));
    EXPECT_EQ(gradient->response.solves, 3 * kNumAtoms);
    EXPECT_EQ(gradient->response.warmStarted, 0u);
    EXPECT_GT(gradient->response.iterations, 0u);
    EXPECT_EQ(gradient->rightHandSideTerms,
              std::vector<std::string>({"one-electron", "coulomb", "exchange"}));
    EXPECT_LE(gradient->stationarityResidual, 1e-6);
    EXPECT_LE(gradient->energyResidual, 1e-8);
    EXPECT_GT(gradient->counts.oneElectron.orbitalPairs, 0u);
    EXPECT_GT(gradient->counts.twoElectron.quartets, 0u);

    // The same walks, called directly: the terms are their outputs, not a
    // re-derivation.
    const auto density = DensityTensor(RowMajor(reference->density));
    ASSERT_TRUE(density.has_value()) << density.error().message;

    const auto oneElectron =
        qcx::integrals::ComputeOneElectronGradient(*molecule, *basis, *density);
    ASSERT_TRUE(oneElectron.has_value()) << oneElectron.error().message;

    qcx::integrals::TwoElectronFockDerivativeOptions twoElectronOptions;
    const auto twoElectron =
        qcx::integrals::ComputeTwoElectronFockDerivative(*molecule,
                                                         *basis,
                                                         *density,
                                                         RowMajor(reference->coefficients),
                                                         kNumOccupied,
                                                         kNumBasis - kNumOccupied,
                                                         twoElectronOptions);
    ASSERT_TRUE(twoElectron.has_value()) << twoElectron.error().message;

    const Eigen::VectorXd oneElectronExpected =
        oneElectron->kineticGradient + oneElectron->nuclearGradient;
    const Eigen::VectorXd exchangeExpected = -0.5 * twoElectron->exchangeSkeleton;

    for (Eigen::Index entry = 0; entry < gradient->gradient.size(); ++entry)
    {
        EXPECT_NEAR(gradient->terms.oneElectron(entry), oneElectronExpected(entry), kTermTolerance);
        EXPECT_NEAR(
            gradient->terms.coulomb(entry), twoElectron->coulombSkeleton(entry), kTermTolerance);
        EXPECT_NEAR(gradient->terms.exchange(entry), exchangeExpected(entry), kTermTolerance);
    }
}

// The decomposition is the total: summed in the documented order, entry by
// entry, bit for bit. A total assembled independently of the terms would let
// the two disagree, which is the failure mode a component breakdown exists to
// expose.
TEST(TotalGradientTest, TheTermsSumToTheEmittedTotalExactly) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto gradient = ComputeTotalGradient(*molecule, *basis, *reference, MakeOptions());
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    ASSERT_EQ(gradient->terms.dispersion.size(), 0);

    Eigen::VectorXd sum = Eigen::VectorXd::Zero(gradient->gradient.size());
    sum += gradient->terms.oneElectron;
    sum += gradient->terms.nuclearRepulsion;
    sum += gradient->terms.coulomb;
    sum += gradient->terms.exchange;
    sum += gradient->terms.response;

    for (Eigen::Index entry = 0; entry < gradient->gradient.size(); ++entry)
    {
        EXPECT_EQ(sum(entry), gradient->gradient(entry));
    }

    EXPECT_NEAR(gradient->energies.reconstructedElectronic, gradient->energies.electronic, 1e-8);
    EXPECT_NEAR(gradient->energies.total,
                gradient->energies.electronic + gradient->energies.nuclearRepulsion,
                1e-12);
}

// The nuclear term against finite differences of the repulsion energy it
// differentiates: the one term this module forms itself.
TEST(TotalGradientTest, TheNuclearTermDifferentiatesTheRepulsionEnergy) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto gradient = ComputeTotalGradient(*molecule, *basis, *reference, MakeOptions());
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    // h = 1e-5: the central difference's truncation error is O(h^2 f'''), and
    // the repulsion's third derivative at an O-H distance is O(10); at 1e-4
    // that truncation (~1e-8) sits on the tolerance, at 1e-5 it is two orders
    // below it while the cancellation error (eps |E| / 2h) stays ~1e-11.
    constexpr double kStep = 1e-5;
    double worst = 0.0;

    for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            const auto plus = NuclearRepulsionAt(Geometry{}, atom, direction, kStep);
            const auto minus = NuclearRepulsionAt(Geometry{}, atom, direction, -kStep);
            ASSERT_TRUE(plus.has_value()) << plus.error().message;
            ASSERT_TRUE(minus.has_value()) << minus.error().message;

            const double difference = (*plus - *minus) / (2.0 * kStep);
            const std::size_t index = 3 * atom + direction;
            worst = std::max(worst,
                             std::abs(difference - gradient->terms.nuclearRepulsion(
                                                       static_cast<Eigen::Index>(index))));
        }
    }

    std::cout << "[nuclear term] worst |finite - analytic| = " << worst << "\n";
    EXPECT_LT(worst, 1e-8);
}

// Translation and torque: the sum of the gradient over the atoms is zero and
// its moment about any point is zero for an isolated molecule. Both are
// necessary conditions and neither is sufficient, but a term contracted in the
// wrong convention breaks them.
TEST(TotalGradientTest, TheGradientIsTranslationAndRotationInvariant) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto gradient = ComputeTotalGradient(*molecule, *basis, *reference, MakeOptions());
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    const std::vector<double> coordinates = Coordinates(*molecule);

    for (std::size_t direction = 0; direction < 3; ++direction)
    {
        double sum = 0.0;
        Eigen::Vector3d torque = Eigen::Vector3d::Zero();

        for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
        {
            const double force =
                gradient->gradient(static_cast<Eigen::Index>(3 * atom + direction));
            sum += force;

            const Eigen::Vector3d position(
                coordinates[3 * atom], coordinates[3 * atom + 1], coordinates[3 * atom + 2]);
            const Eigen::Vector3d atomGradient(
                gradient->gradient(static_cast<Eigen::Index>(3 * atom)),
                gradient->gradient(static_cast<Eigen::Index>(3 * atom + 1)),
                gradient->gradient(static_cast<Eigen::Index>(3 * atom + 2)));
            torque += position.cross(atomGradient);
        }

        EXPECT_LT(std::abs(sum), 1e-8) << "direction " << direction;
        EXPECT_LT(torque.cwiseAbs().maxCoeff(), 1e-8) << "direction " << direction;
    }
}

// The finite-difference gate: every Cartesian coordinate of every atom, at
// two steps, against the central difference of the SCF's total energy.
//
// The two steps are there to expose the scheme's truncation, and the output
// shows it: a two-point central difference carries (h^2/6)|E'''|, and the
// worst deviation falls from 1.6e-7 at h = 1e-3 to 1.4e-8 at h = 3e-4 - a
// factor 11, the (10/3)^2 an O(h^2) error gives. The coarse bound therefore
// admits that truncation (about twice the measured value) rather than
// pretending an instrument can gate below its own error; the fine step, where
// the truncation is under the acceptance, holds 1e-7 - the plan's number.
TEST(TotalGradientTest, TheTotalGradientMatchesTheEnergyFiniteDifference) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto gradient = ComputeTotalGradient(*molecule, *basis, *reference, MakeOptions());
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    constexpr std::array<double, 2> kSteps{1e-3, 3e-4};
    constexpr std::array<double, 2> kBounds{3e-7, 1e-7};

    for (std::size_t stepIndex = 0; stepIndex < kSteps.size(); ++stepIndex)
    {
        const double step = kSteps[stepIndex];
        double worst = 0.0;

        for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
        {
            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                const auto plus = TotalEnergyAt(Geometry{}, *basis, atom, direction, step);
                const auto minus = TotalEnergyAt(Geometry{}, *basis, atom, direction, -step);
                ASSERT_TRUE(plus.has_value()) << plus.error().message;
                ASSERT_TRUE(minus.has_value()) << minus.error().message;

                const std::size_t index = 3 * atom + direction;
                const double difference = (*plus - *minus) / (2.0 * step);
                const double analytic = gradient->gradient(static_cast<Eigen::Index>(index));
                const double deviation = std::abs(difference - analytic);
                worst = std::max(worst, deviation);

                std::cout << "[energy derivative] atom " << atom << " direction " << direction
                          << " analytic " << analytic << ", finite " << difference
                          << " at h = " << step << " (deviation " << deviation << ")\n";
            }
        }

        std::cout << "[energy derivative] h = " << step << " worst deviation = " << worst << "\n";
        EXPECT_LT(worst, kBounds[stepIndex]) << "at h = " << step;
    }
}

// A reference that is not stationary is refused before a gradient is emitted:
// the orbital gradient the gate reads is nonzero by construction (the last
// occupied and the first virtual orbitals are swapped).
TEST(TotalGradientTest, ANonStationaryReferenceIsRefusedByName) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    ScfGradientReference rotated = *reference;
    Eigen::MatrixXd coefficients = rotated.coefficients;
    coefficients.col(static_cast<Eigen::Index>(kNumOccupied - 1))
        .swap(coefficients.col(static_cast<Eigen::Index>(kNumOccupied)));
    rotated.coefficients = coefficients;

    Eigen::VectorXd energies = rotated.orbitalEnergies;
    std::swap(energies(static_cast<Eigen::Index>(kNumOccupied - 1)),
              energies(static_cast<Eigen::Index>(kNumOccupied)));
    rotated.orbitalEnergies = energies;

    const Eigen::MatrixXd occupied =
        rotated.coefficients.leftCols(static_cast<Eigen::Index>(kNumOccupied));
    rotated.density = 2.0 * occupied * occupied.transpose();

    const auto gradient = ComputeTotalGradient(*molecule, *basis, rotated, MakeOptions());
    ASSERT_FALSE(gradient.has_value());
    EXPECT_EQ(gradient.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(gradient.error().message.find("not stationary"), std::string::npos)
        << gradient.error().message;
    EXPECT_NE(gradient.error().message.find("max|2 F_ai|"), std::string::npos)
        << gradient.error().message;
}

// The gate tolerances are the request's, and a non-positive one is a malformed
// request rather than an always-passing gate.
TEST(TotalGradientTest, ANonPositiveGateToleranceIsRefused) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    TotalGradientOptions options = MakeOptions();
    options.stationarityTolerance = 0.0;

    const auto gradient = ComputeTotalGradient(*molecule, *basis, *reference, options);
    ASSERT_FALSE(gradient.has_value());
    EXPECT_EQ(gradient.error().code, qcx::ErrorCode::kInvalidArgument);
}

// --- dispersion --------------------------------------------------------------

/// A D3 request with explicit values, so the test does not depend on a
/// preset's table.
/// \returns The resolved D3 request.
qcx::dispersion::ResolvedDispersion D3Request() {
    qcx::dispersion::ResolvedDispersion request;
    request.form = qcx::dispersion::DispersionForm::kD3;
    request.source = qcx::dispersion::DispersionParameterSource::kExplicit;
    request.parameters = qcx::dispersion::D3ParameterValues{1.0, 0.7, 1.0, 14.0, 1.0, 16.0};
    return request;
}

// The D3 term: the energy and the gradient the dispersion module returns are
// what the total carries, the total includes them exactly, and the gradient
// agrees with central differences of the dispersion energy.
TEST(TotalGradientTest, TheD3DispersionTermIsAddedAndAgreesWithFiniteDifferences) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto plain = ComputeTotalGradient(*molecule, *basis, *reference, MakeOptions());
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    const qcx::dispersion::ResolvedDispersion request = D3Request();
    TotalGradientOptions options = MakeOptions();
    options.dispersion = request;
    const auto corrected = ComputeTotalGradient(*molecule, *basis, *reference, options);
    ASSERT_TRUE(corrected.has_value()) << corrected.error().message;

    const auto energy = qcx::dispersion::EvaluateDispersion(*molecule, request);
    const auto dispersion = qcx::dispersion::EvaluateDispersionGradient(*molecule, request);
    ASSERT_TRUE(energy.has_value()) << energy.error().message;
    ASSERT_TRUE(dispersion.has_value()) << dispersion.error().message;

    EXPECT_NEAR(corrected->energies.dispersion, *energy, 1e-14);
    EXPECT_NEAR(corrected->energies.total,
                corrected->energies.electronic + corrected->energies.nuclearRepulsion + *energy,
                1e-12);

    for (Eigen::Index entry = 0; entry < corrected->terms.dispersion.size(); ++entry)
    {
        EXPECT_EQ(corrected->terms.dispersion(entry), dispersion->gradient(entry));
        EXPECT_DOUBLE_EQ(corrected->gradient(entry),
                         plain->gradient(entry) + dispersion->gradient(entry));
    }

    constexpr std::array<double, 2> kSteps{1e-4, 1e-6};
    constexpr std::array<double, 2> kBounds{3e-7, 1e-7};

    for (std::size_t stepIndex = 0; stepIndex < kSteps.size(); ++stepIndex)
    {
        const double step = kSteps[stepIndex];
        double worst = 0.0;

        for (std::size_t atom = 0; atom < kNumAtoms; ++atom)
        {
            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                const auto plusMolecule = MakeMolecule(Geometry{}.Displaced(atom, direction, step));
                const auto minusMolecule =
                    MakeMolecule(Geometry{}.Displaced(atom, direction, -step));
                ASSERT_TRUE(plusMolecule.has_value()) << plusMolecule.error().message;
                ASSERT_TRUE(minusMolecule.has_value()) << minusMolecule.error().message;

                const auto plus = qcx::dispersion::EvaluateDispersion(*plusMolecule, request);
                const auto minus = qcx::dispersion::EvaluateDispersion(*minusMolecule, request);
                ASSERT_TRUE(plus.has_value()) << plus.error().message;
                ASSERT_TRUE(minus.has_value()) << minus.error().message;

                const double difference = (*plus - *minus) / (2.0 * step);
                const double analytic =
                    dispersion->gradient(static_cast<Eigen::Index>(3 * atom + direction));
                worst = std::max(worst, std::abs(difference - analytic));
            }
        }

        std::cout << "[dispersion] h = " << step << " worst |finite - analytic| = " << worst
                  << "\n";
        EXPECT_LT(worst, kBounds[stepIndex]) << "at h = " << step;
    }
}

// D4 has no coordinate-gradient tier; the refusal names the form and the
// missing tier rather than returning a zero derivative.
TEST(TotalGradientTest, AD4GradientRequestIsRefusedByName) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto reference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    qcx::dispersion::ResolvedDispersion request;
    request.form = qcx::dispersion::DispersionForm::kD4;
    request.source = qcx::dispersion::DispersionParameterSource::kExplicit;
    request.parameters = qcx::dispersion::D4ParameterValues{1.0, 0.5, 0.4, 3.0};

    TotalGradientOptions options = MakeOptions();
    options.dispersion = request;

    const auto gradient = ComputeTotalGradient(*molecule, *basis, *reference, options);
    ASSERT_FALSE(gradient.has_value());
    EXPECT_EQ(gradient.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(gradient.error().message.find("\"d4\""), std::string::npos)
        << gradient.error().message;
    EXPECT_NE(gradient.error().message.find("gradient tier"), std::string::npos)
        << gradient.error().message;
}

// --- the Kohn-Sham route -----------------------------------------------------

// The hybrid functional the Kohn-Sham rows run. Its exact-exchange fraction is
// a quarter, so the exchange the functional scales is present and the scaling
// is exercised; a pure functional would leave the whole subtraction branch
// unvisited. The driver's acceptance row runs pbe (c_HF = 0), so the two
// together cover both branches of the reconstruction.
constexpr std::string_view kKohnShamFunctional = "pbe0";
constexpr double kKohnShamExchangeFraction = 0.25;

// The screening tolerance the reference's energy path and the gradient's walks
// share: one key of the run, passed to every engine evaluation and to the
// gradient options. A potential screened at another tolerance is the potential
// of another energy.
constexpr double kKohnShamScreen = 1e-8;

// The grid the Kohn-Sham rows integrate on: the production defaults, which is
// what a run that writes no [grid] block gets.
constexpr qcx::grid::XcGridSettings kKohnShamGrid{};

/// The Kohn-Sham reference's operands: the core Hamiltonian and the AO
/// two-electron tensor in the row-major layout the hand contractions below
/// read.
struct KohnShamOperands {
    Eigen::MatrixXd core; ///< H = T + V, in the AO basis.
    std::vector<double> aoTensor; ///< (mu nu | lambda sigma), row-major.
};

/// Builds the operands at the molecule's geometry.
/// \param molecule The molecule.
/// \param basis The basis set.
/// \returns The operands, or an Error when an integral build fails.
qcx::Result<KohnShamOperands> MakeKohnShamOperands(const Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basis) {
    const auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basis);
    // Non-const: HostView is the mutable handle on the host copy.
    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basis);

    if (!kinetic.has_value() || !nuclear.has_value() || !eri.has_value())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the Kohn-Sham fixture's integral build failed"});
    }

    return KohnShamOperands{ToMatrix(*kinetic) + ToMatrix(*nuclear), eri->HostView()};
}

/// The Coulomb matrix of a spin-summed closed-shell density,
/// `J_mu nu = sum_ls D_ls (mu nu | lambda sigma)`.
///
/// A hand contraction over the AO tensor: algebraically the same operator the
/// SCF's supermatrix path contracts, in a different summation order, so a
/// rebuilt Fock compared against it pins the reconstruction's algebra rather
/// than one function's output against itself.
/// \param aoTensor The AO tensor, row-major.
/// \param density The spin-summed density.
/// \returns J[D].
Eigen::MatrixXd CoulombMatrix(const std::vector<double>& aoTensor, const Eigen::MatrixXd& density) {
    const Eigen::Index n = density.rows();
    Eigen::MatrixXd coulomb = Eigen::MatrixXd::Zero(n, n);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            double sum = 0.0;

            for (Eigen::Index lambda = 0; lambda < n; ++lambda)
            {
                for (Eigen::Index sigma = 0; sigma < n; ++sigma)
                {
                    sum += density(lambda, sigma) * aoTensor[static_cast<std::size_t>(
                                                        ((mu * n + nu) * n + lambda) * n + sigma)];
                }
            }

            coulomb(mu, nu) = sum;
        }
    }

    return coulomb;
}

/// The exchange matrix of a spin-summed closed-shell density,
/// `K_mu nu = sum_ls D_ls (mu lambda | sigma nu)`, the supermatrix convention
/// the SCF's own build contracts.
/// \param aoTensor The AO tensor, row-major.
/// \param density The spin-summed density.
/// \returns K[D].
Eigen::MatrixXd ExchangeMatrix(const std::vector<double>& aoTensor,
                               const Eigen::MatrixXd& density) {
    const Eigen::Index n = density.rows();
    Eigen::MatrixXd exchange = Eigen::MatrixXd::Zero(n, n);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            double sum = 0.0;

            for (Eigen::Index lambda = 0; lambda < n; ++lambda)
            {
                for (Eigen::Index sigma = 0; sigma < n; ++sigma)
                {
                    sum += density(lambda, sigma) * aoTensor[static_cast<std::size_t>(
                                                        ((mu * n + lambda) * n + sigma) * n + nu)];
                }
            }

            exchange(mu, nu) = sum;
        }
    }

    return exchange;
}

/// The restricted potential `V = (V_alpha + V_beta) / 2` at a density.
///
/// The closed-shell split and the engine's chain rule, called exactly as the
/// energy path calls it: `(D/2, D/2)` through EvaluateScreened at the run's
/// screening tolerance.
/// \param engine The grid engine the reference was converged with.
/// \param density The spin-summed density.
/// \returns The potential, or the engine's Error.
qcx::Result<Eigen::MatrixXd> KohnShamPotential(const qcx::grid::XcGridEngine& engine,
                                               const Eigen::MatrixXd& density) {
    const Eigen::MatrixXd halfDensity = 0.5 * density;
    const auto evaluation = engine.EvaluateScreened(halfDensity, halfDensity, kKohnShamScreen);

    if (!evaluation.has_value())
    {
        return std::unexpected(evaluation.error());
    }

    return Eigen::MatrixXd(0.5 * (evaluation->potentialAlpha + evaluation->potentialBeta));
}

/// Runs the restricted Kohn-Sham SCF the reference belongs to.
///
/// The energy seam is the one the driver's composition fills: a Fock builder, a
/// Coulomb provider and the energy the trace formula cannot recover. Here the
/// Fock is `H + J + V_xc - (c_HF/2) K` and the contribution is
/// `E_xc - (c_HF/4) Tr[D K]`, both from the operands' own contractions, so the
/// reference is a converged stationary point of exactly the energy and operator
/// the gradient rebuilds. The tolerance rung is the tight one: the stationarity
/// gate (1e-6) and the energy reconstruction (1e-8, against an energy formed in
/// a different summation order) both read this reference.
/// \param molecule The molecule.
/// \param basis The basis set.
/// \param engine The grid engine, created from the settings the gradient will
/// be handed.
/// \param operands The core Hamiltonian and the AO tensor.
/// \returns The validated reference, or an Error.
qcx::Result<ScfGradientReference> MakeKohnShamReference(const Molecule& molecule,
                                                        const qcx::basisset::BasisSet& basis,
                                                        const qcx::grid::XcGridEngine& engine,
                                                        const KohnShamOperands& operands) {
    const auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    // Read off the functional itself, as the driver's composition reads it: a
    // caller-set fraction could disagree with the functional the energy was
    // integrated with.
    const double exchangeFraction = engine.ExchangeFraction();

    const qcx::scf::FockBuilderFn fockBuilder =
        [&operands, &engine, exchangeFraction](
            const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const Eigen::MatrixXd halfDensity = 0.5 * density;
        const auto evaluation = engine.EvaluateScreened(halfDensity, halfDensity, kKohnShamScreen);

        if (!evaluation.has_value())
        {
            return std::unexpected(evaluation.error());
        }

        Eigen::MatrixXd fock =
            operands.core + CoulombMatrix(operands.aoTensor, density) +
            Eigen::MatrixXd(0.5 * (evaluation->potentialAlpha + evaluation->potentialBeta));

        if (exchangeFraction != 0.0)
        {
            fock -= 0.5 * exchangeFraction * ExchangeMatrix(operands.aoTensor, density);
        }

        return fock;
    };

    const qcx::scf::CoulombFn coulomb =
        [&operands](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        return CoulombMatrix(operands.aoTensor, density);
    };

    const qcx::scf::EnergyContributionFn contribution =
        [&operands, &engine, exchangeFraction](
            const Eigen::MatrixXd& density) -> qcx::Result<double> {
        const Eigen::MatrixXd halfDensity = 0.5 * density;
        const auto evaluation = engine.EvaluateScreened(halfDensity, halfDensity, kKohnShamScreen);

        if (!evaluation.has_value())
        {
            return std::unexpected(evaluation.error());
        }

        double energy = evaluation->energy;

        if (exchangeFraction != 0.0)
        {
            // The closed-shell exchange 1/2 sum D K: the gradient's gate
            // subtracts (1/2) c_HF of it, so the contribution carries the same
            // half-trace at the same scale and the two energy expressions are
            // one.
            const double exchangeHalfTrace =
                0.5 * density.cwiseProduct(ExchangeMatrix(operands.aoTensor, density)).sum();
            energy -= 0.5 * exchangeFraction * exchangeHalfTrace;
        }

        return energy;
    };

    qcx::scf::RhfOptions options;
    options.maxIterations = 300;
    // The kTight rung (energy/density 1e-10/1e-8), so the fixture's
    // convergence is declared by its rung rather than a private gate.
    options.energyTolerance = 1e-10;
    options.densityTolerance = 1e-8;

    const auto result = qcx::scf::RunRhfScf(
        molecule, ToMatrix(*overlap), operands.core, options, fockBuilder, coulomb, contribution);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    if (!result->converged)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the Kohn-Sham fixture's SCF did not converge"});
    }

    return qcx::scf::ExtractGradientReference(*result, kNumOccupied);
}

// The Kohn-Sham Fock reconstruction: the state must own the matrix the
// reference actually converged against, not the Roothaan Fock that omits the
// potential and leaves the exchange unscaled. The expectation is assembled here
// from the hand contractions and the run's own potential - an independent route
// to the same matrix - and the default (Hartree-Fock) route still reads the
// evaluator's Fock bit for bit.
TEST(TotalGradientTest, TheKohnShamRouteRebuildsTheReferenceFock) {
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto operands = MakeKohnShamOperands(*molecule, *basis);
    ASSERT_TRUE(operands.has_value()) << operands.error().message;

    auto engine =
        qcx::grid::XcGridEngine::Create(*molecule, *basis, kKohnShamFunctional, kKohnShamGrid);
    ASSERT_TRUE(engine.has_value()) << engine.error().message;
    EXPECT_DOUBLE_EQ(engine->ExchangeFraction(), kKohnShamExchangeFraction);

    const auto reference = MakeKohnShamReference(*molecule, *basis, *engine, *operands);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    const auto potential = KohnShamPotential(*engine, reference->density);
    ASSERT_TRUE(potential.has_value()) << potential.error().message;

    // The kernel is never applied while the state is built: the callback counts
    // its calls, and the reconstruction reads only the potential and the
    // fraction.
    auto kernelCalls = std::make_shared<int>(0);
    TotalGradientKohnShamRoute route;
    route.exchangeFraction = engine->ExchangeFraction();
    route.potential = *potential;
    route.kernel = qcx::response::KohnShamKernelTerm{
        std::string(kKohnShamFunctional),
        [kernelCalls](std::span<const double>, std::span<double>) -> qcx::Result<void> {
            ++*kernelCalls;
            return {};
        }};

    auto state = BuildResponseState(*molecule, *basis, *reference, {}, {}, route);
    ASSERT_TRUE(state.has_value()) << state.error().message;

    EXPECT_TRUE(state->HasKernel());
    EXPECT_EQ(state->KernelName(), kKohnShamFunctional);
    EXPECT_DOUBLE_EQ(state->ExchangeFraction(), kKohnShamExchangeFraction);
    EXPECT_EQ(*kernelCalls, 0);

    const Eigen::MatrixXd expected =
        operands->core + CoulombMatrix(operands->aoTensor, reference->density) + *potential -
        0.5 * kKohnShamExchangeFraction * ExchangeMatrix(operands->aoTensor, reference->density);

    const std::vector<double> fock(state->Fock().begin(), state->Fock().end());
    ASSERT_EQ(fock.size(), kNumBasis * kNumBasis);

    for (std::size_t row = 0; row < kNumBasis; ++row)
    {
        for (std::size_t column = 0; column < kNumBasis; ++column)
        {
            EXPECT_NEAR(fock[row * kNumBasis + column],
                        expected(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)),
                        kTransformTolerance)
                << "row " << row << " column " << column;
        }
    }

    // The default route is untouched: the state's Fock is the evaluator's own
    // matrix, bit for bit, and the operator is the Hartree-Fock one.
    const auto rhfReference = MakeReference(*molecule, *basis);
    ASSERT_TRUE(rhfReference.has_value()) << rhfReference.error().message;

    auto rhfState = BuildResponseState(*molecule, *basis, *rhfReference);
    ASSERT_TRUE(rhfState.has_value()) << rhfState.error().message;

    EXPECT_FALSE(rhfState->HasKernel());
    EXPECT_TRUE(rhfState->KernelName().empty());
    EXPECT_DOUBLE_EQ(rhfState->ExchangeFraction(), 1.0);

    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis);
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    const auto evaluated = qcx::scf::EvaluateRhfDensityEnergy(
        *molecule, ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, rhfReference->density);
    ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;

    const std::vector<double> rhfFock(rhfState->Fock().begin(), rhfState->Fock().end());

    for (std::size_t row = 0; row < kNumBasis; ++row)
    {
        for (std::size_t column = 0; column < kNumBasis; ++column)
        {
            EXPECT_DOUBLE_EQ(
                rhfFock[row * kNumBasis + column],
                evaluated->fock(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)));
        }
    }
}

// The exchange-correlation terms are the run's own grid walks, not second
// integrations: the total's xc term is compared against a direct
// EvaluateXcGradient on the same engine, density and thresholds, the exchange
// term against the derivative walk's skeleton at the functional's fraction, and
// the reconstruction gate against the reference's electronic energy.
TEST(TotalGradientTest, TheKohnShamXcTermIsTheGridWalksOwnGradientAndEnergy) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "grid finite differences over an XC walk: Release-only";
    }

    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeMolecule(Geometry{});
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto operands = MakeKohnShamOperands(*molecule, *basis);
    ASSERT_TRUE(operands.has_value()) << operands.error().message;

    auto engine =
        qcx::grid::XcGridEngine::Create(*molecule, *basis, kKohnShamFunctional, kKohnShamGrid);
    ASSERT_TRUE(engine.has_value()) << engine.error().message;
    EXPECT_DOUBLE_EQ(engine->ExchangeFraction(), kKohnShamExchangeFraction);

    const auto reference = MakeKohnShamReference(*molecule, *basis, *engine, *operands);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    TotalGradientOptions options = MakeOptions();
    options.xc = TotalGradientXcOptions{
        std::string(kKohnShamFunctional), kKohnShamGrid, kKohnShamScreen, {}};

    const auto gradient = ComputeTotalGradient(*molecule, *basis, *reference, options);
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    EXPECT_EQ(gradient->terms.xc.size(), static_cast<Eigen::Index>(3 * kNumAtoms));
    EXPECT_EQ(gradient->terms.dispersion.size(), 0);
    EXPECT_EQ(gradient->rightHandSideTerms,
              std::vector<std::string>({"one-electron", "coulomb", "exchange", "xc"}));
    EXPECT_LE(gradient->stationarityResidual, 1e-6);
    EXPECT_LE(gradient->energyResidual, 1e-8);
    EXPECT_NEAR(gradient->energies.reconstructedElectronic, reference->electronicEnergy, 1e-8);
    EXPECT_NEAR(gradient->energies.total,
                gradient->energies.electronic + gradient->energies.nuclearRepulsion,
                1e-12);
    EXPECT_GT(gradient->terms.xc.cwiseAbs().maxCoeff(), 1e-3);

    // The terms sum to the emitted total, xc included, in the documented order.
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(gradient->gradient.size());
    sum += gradient->terms.oneElectron;
    sum += gradient->terms.nuclearRepulsion;
    sum += gradient->terms.coulomb;
    sum += gradient->terms.exchange;
    sum += gradient->terms.xc;
    sum += gradient->terms.response;

    for (Eigen::Index entry = 0; entry < gradient->gradient.size(); ++entry)
    {
        EXPECT_EQ(sum(entry), gradient->gradient(entry));
    }

    // The same walk, called directly on the run's engine, density and
    // thresholds.
    const Eigen::MatrixXd halfDensity = 0.5 * reference->density;
    const auto geometry = qcx::grid::ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;

    const auto provider =
        excgrid::CreateGridDerivativeProvider(*geometry, qcx::grid::ToExcgridParams(kKohnShamGrid));
    ASSERT_TRUE(provider.has_value())
        << qcx::grid::TranslateExcgridError(provider.error(), "the fixture's provider").message;

    const auto thresholds =
        qcx::grid::XcIntegrationThresholds::FromEnergyPath(kKohnShamGrid, kKohnShamScreen);
    const auto walk = qcx::grid::EvaluateXcGradient(
        *engine, halfDensity, halfDensity, **provider, thresholds, {});
    ASSERT_TRUE(walk.has_value()) << walk.error().message;

    for (Eigen::Index entry = 0; entry < gradient->terms.xc.size(); ++entry)
    {
        EXPECT_NEAR(gradient->terms.xc(entry), walk->gradient(entry), kTermTolerance)
            << "entry " << entry;
    }

    // The walk's energy is the energy path's: the same points under the same
    // thresholds, so the gradient is the derivative of the energy the reference
    // was converged against.
    const auto energyPath = engine->EvaluateScreened(halfDensity, halfDensity, kKohnShamScreen);
    ASSERT_TRUE(energyPath.has_value()) << energyPath.error().message;
    EXPECT_DOUBLE_EQ(walk->energy, energyPath->energy);

    // The exchange term is the derivative walk's skeleton at the functional's
    // exact-exchange fraction.
    const auto density = DensityTensor(RowMajor(reference->density));
    ASSERT_TRUE(density.has_value()) << density.error().message;

    qcx::integrals::TwoElectronFockDerivativeOptions twoElectronOptions;
    twoElectronOptions.accuracy = options.eri.accuracy;
    const auto twoElectron =
        qcx::integrals::ComputeTwoElectronFockDerivative(*molecule,
                                                         *basis,
                                                         *density,
                                                         RowMajor(reference->coefficients),
                                                         kNumOccupied,
                                                         kNumBasis - kNumOccupied,
                                                         twoElectronOptions);
    ASSERT_TRUE(twoElectron.has_value()) << twoElectron.error().message;

    for (Eigen::Index entry = 0; entry < gradient->terms.exchange.size(); ++entry)
    {
        EXPECT_NEAR(gradient->terms.exchange(entry),
                    -0.5 * kKohnShamExchangeFraction * twoElectron->exchangeSkeleton(entry),
                    kTermTolerance)
            << "entry " << entry;
    }
}

} // namespace
