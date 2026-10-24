// The Kohn-Sham orbital-Hessian operator, against the numerical Hessian of the
// energy it claims to differentiate.
//
// The reference is a converged restricted Kohn-Sham solution built by the same
// seam the SCF probe uses: the density functional's energy and potential, the
// direct builder's Coulomb and exact-exchange halves, and the converged
// orbitals the SCF returns. The energy the finite difference walks is written
// out here - core, Coulomb, scaled exchange, XC - so agreement is not a
// comparison of the operator with itself.
//
// The acceptance runs on the components this excgrid pin publishes a
// second-derivative tier for: slater (the LDA case), pbe exchange with an
// explicit c_HF = 0.25 (the hybrid case; the composite pbe0 publishes no tier
// at this pin) and tau_x (the kinetic-energy-density case). The water case is
// the larger basis and the one the control rides on.

#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/grid/xc_kernel_action.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/response/response_kernel.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::response {
namespace {

using qcx::testing::ToMatrix;

constexpr std::size_t kRadialPoints = 20;
constexpr std::size_t kAngularPoints = 50;

// Keep every shell on the kernel side: the SCF's energy path is the dense one,
// and the kernel must contract the same field it integrated.
constexpr double kScreeningTolerance = -1.0;

constexpr double kFiniteDifferenceStep = 1e-3;
constexpr double kFiniteDifferenceTolerance = 1e-6;

std::string Scientific(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6e", value);

    return buffer;
}

/// The molecule, basis and one-electron ingredients of a reference.
struct MoleculeInputs {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd core;
};

std::optional<MoleculeInputs> BuildMoleculeInputs(bool water) {
    auto molecule = water ? qcx::testing::MakeH2oSto3g() : qcx::testing::MakeH2Sto3g();

    if (!molecule.has_value())
    {
        return std::nullopt;
    }

    auto basis = water ? qcx::testing::MakeH2oSto3gBasis() : qcx::testing::MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::nullopt;
    }

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlap.has_value())
    {
        return std::nullopt;
    }

    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    if (!kinetic.has_value())
    {
        return std::nullopt;
    }

    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!nuclear.has_value())
    {
        return std::nullopt;
    }

    return MoleculeInputs{std::move(*molecule),
                          std::move(*basis),
                          ToMatrix(*overlap),
                          ToMatrix(*kinetic) + ToMatrix(*nuclear)};
}

/// A converged Kohn-Sham reference and the MO-basis data the operator reads.
struct KsReference {
    MoleculeInputs inputs;
    std::shared_ptr<qcx::grid::XcGridEngine> engine;
    qcx::scf::HfResult scf;
    std::vector<double> orbitalEnergies;
    std::vector<double> moTwoElectron;
    Eigen::MatrixXd coreMo;
    std::size_t numOccupied = 0;
    std::size_t numVirtual = 0;
    double exchangeFraction = 0.0;
};

/// The Kohn-Sham seam: the restricted Fock, the Coulomb matrix and the energy
/// contribution, composed as the driver composes them. The exact-exchange
/// fraction is a parameter because a pure functional with a caller-chosen
/// fraction is how the hybrid acceptance runs at this pin.
qcx::Result<KsReference> BuildKsReference(bool water,
                                          std::string_view functionalName,
                                          double exchangeFraction) {
    auto inputs = BuildMoleculeInputs(water);

    if (!inputs.has_value())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture could not be built"});
    }

    const qcx::grid::XcGridSettings settings{kRadialPoints, kAngularPoints};
    auto engine =
        qcx::grid::XcGridEngine::Create(inputs->molecule, inputs->basis, functionalName, settings);

    if (!engine.has_value())
    {
        return std::unexpected(engine.error());
    }

    auto shared = std::make_shared<qcx::grid::XcGridEngine>(std::move(*engine));

    const bool hasExchange = exchangeFraction > 0.0;

    auto coreTensor = qcx::testing::ToTensor(inputs->core);

    if (!coreTensor.has_value())
    {
        return std::unexpected(coreTensor.error());
    }

    auto coreShared = std::make_shared<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>(
        std::move(*coreTensor));

    qcx::integrals::FockBuildOptions coulombOptions;
    coulombOptions.buildCoulombOnly = true;
    qcx::integrals::FockBuildOptions exchangeOptions;
    exchangeOptions.buildExchangeOnly = true;

    auto coulombBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        inputs->molecule, inputs->basis, *coreShared, coulombOptions);

    if (!coulombBuilder.has_value())
    {
        return std::unexpected(coulombBuilder.error());
    }

    std::shared_ptr<qcx::integrals::DirectJkFockBuilder> exchangeBuilder;

    if (hasExchange)
    {
        auto built = qcx::integrals::DirectJkFockBuilder::Create(
            inputs->molecule, inputs->basis, *coreShared, exchangeOptions);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        exchangeBuilder = std::make_shared<qcx::integrals::DirectJkFockBuilder>(std::move(*built));
    }

    auto coulombShared =
        std::make_shared<qcx::integrals::DirectJkFockBuilder>(std::move(*coulombBuilder));

    struct Cache {
        Eigen::MatrixXd density;
        Eigen::MatrixXd coulombHalf;
        Eigen::MatrixXd exchangeHalf;
        qcx::grid::XcEvaluation xc;
        bool valid = false;
    };

    auto cache = std::make_shared<Cache>();

    const auto ensure = [cache, coulombShared, exchangeBuilder, hasExchange, shared](
                            const Eigen::MatrixXd& density) -> qcx::Result<void> {
        if (cache->valid && cache->density.rows() == density.rows() &&
            cache->density.cols() == density.cols() && cache->density == density)
        {
            return {};
        }

        const auto rhoTensor = qcx::testing::ToTensor(0.5 * density);

        if (!rhoTensor.has_value())
        {
            return std::unexpected(rhoTensor.error());
        }

        const auto coulombTensor = coulombShared->BuildFock(*rhoTensor);

        if (!coulombTensor.has_value())
        {
            return std::unexpected(coulombTensor.error());
        }

        if (hasExchange)
        {
            const auto exchangeTensor = exchangeBuilder->BuildFock(*rhoTensor);

            if (!exchangeTensor.has_value())
            {
                return std::unexpected(exchangeTensor.error());
            }

            cache->exchangeHalf = ToMatrix(*exchangeTensor);
        }

        const auto evaluation = shared->EvaluateClosedShell(density);

        if (!evaluation.has_value())
        {
            return std::unexpected(evaluation.error());
        }

        cache->density = density;
        cache->coulombHalf = ToMatrix(*coulombTensor);
        cache->xc = *evaluation;
        cache->valid = true;
        return {};
    };

    qcx::scf::FockBuilderFn fock =
        [ensure, cache, hasExchange, exchangeFraction, core = inputs->core](
            const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        Eigen::MatrixXd assembled =
            cache->coulombHalf + 0.5 * (cache->xc.potentialAlpha + cache->xc.potentialBeta);

        if (hasExchange)
        {
            assembled += exchangeFraction * (cache->exchangeHalf - core);
        }

        return assembled;
    };

    qcx::scf::CoulombFn coulomb =
        [ensure, cache, core = inputs->core](
            const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        return cache->coulombHalf - core;
    };

    qcx::scf::EnergyContributionFn contribution =
        [ensure, cache, hasExchange, exchangeFraction, core = inputs->core](
            const Eigen::MatrixXd& density) -> qcx::Result<double> {
        const auto filled = ensure(density);

        if (!filled.has_value())
        {
            return std::unexpected(filled.error());
        }

        double value = cache->xc.energy;

        if (hasExchange)
        {
            value -=
                0.5 * exchangeFraction * (density.cwiseProduct(core - cache->exchangeHalf)).sum();
        }

        return value;
    };

    auto result = qcx::scf::RunRhfScf(inputs->molecule,
                                      inputs->overlap,
                                      inputs->core,
                                      qcx::scf::RhfOptions{},
                                      fock,
                                      coulomb,
                                      contribution,
                                      &inputs->basis);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    if (!result->converged)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kConvergenceFailure, "the reference SCF did not converge"});
    }

    const std::size_t numOrbitals = static_cast<std::size_t>(result->coefficients.rows());
    const std::size_t numOccupied = static_cast<std::size_t>(inputs->molecule.ElectronCount()) / 2;
    const std::size_t numVirtual = numOrbitals - numOccupied;
    const Eigen::MatrixXd& coefficients = result->coefficients;

    const Eigen::MatrixXd coreMo = coefficients.transpose() * inputs->core * coefficients;

    const auto aoEri = qcx::integrals::BuildEriTensorGeneral(inputs->molecule, inputs->basis);

    if (!aoEri.has_value())
    {
        return std::unexpected(aoEri.error());
    }

    std::vector<double> moEri(numOrbitals * numOrbitals * numOrbitals * numOrbitals, 0.0);

    for (std::size_t p = 0; p < numOrbitals; ++p)
    {
        for (std::size_t q = 0; q < numOrbitals; ++q)
        {
            for (std::size_t r = 0; r < numOrbitals; ++r)
            {
                for (std::size_t s = 0; s < numOrbitals; ++s)
                {
                    double value = 0.0;

                    for (std::size_t mu = 0; mu < numOrbitals; ++mu)
                    {
                        const double first = coefficients(static_cast<Eigen::Index>(mu),
                                                          static_cast<Eigen::Index>(p));

                        if (first == 0.0)
                        {
                            continue;
                        }

                        for (std::size_t nu = 0; nu < numOrbitals; ++nu)
                        {
                            const double second =
                                first * coefficients(static_cast<Eigen::Index>(nu),
                                                     static_cast<Eigen::Index>(q));

                            for (std::size_t lambda = 0; lambda < numOrbitals; ++lambda)
                            {
                                const double third =
                                    second * coefficients(static_cast<Eigen::Index>(lambda),
                                                          static_cast<Eigen::Index>(r));

                                for (std::size_t sigma = 0; sigma < numOrbitals; ++sigma)
                                {
                                    value += third *
                                             coefficients(static_cast<Eigen::Index>(sigma),
                                                          static_cast<Eigen::Index>(s)) *
                                             (*aoEri)(mu, nu, lambda, sigma);
                                }
                            }
                        }
                    }

                    moEri[((p * numOrbitals + q) * numOrbitals + r) * numOrbitals + s] = value;
                }
            }
        }
    }

    // The operator's energy denominators are the diagonal of the dense Fock the
    // reference implies, not the SCF's returned eigenvalues: the SCF
    // diagonalizes the Fock built at its last density estimate, while the
    // finite difference is taken at the returned coefficients. Only the
    // diagonal enters the identity, so reading it at the returned reference
    // makes the gate independent of the SCF's state-fixed-point residual.
    const Eigen::MatrixXd occupied = coefficients.leftCols(static_cast<Eigen::Index>(numOccupied));
    const Eigen::MatrixXd referenceDensity = 2.0 * occupied * occupied.transpose();
    const auto referenceEvaluation = shared->EvaluateClosedShell(referenceDensity);

    if (!referenceEvaluation.has_value())
    {
        return std::unexpected(referenceEvaluation.error());
    }

    const Eigen::MatrixXd moPotential =
        coefficients.transpose() *
        (0.5 * (referenceEvaluation->potentialAlpha + referenceEvaluation->potentialBeta)) *
        coefficients;
    std::vector<double> orbitalEnergies(numOrbitals, 0.0);

    for (std::size_t p = 0; p < numOrbitals; ++p)
    {
        double value = coreMo(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(p)) +
                       moPotential(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(p));

        for (std::size_t i = 0; i < numOccupied; ++i)
        {
            // F_pp = core + Vxc + sum_i 2 [(pp|ii) - c/2 (pi|pi)]: the
            // closed-shell occupation doubles the Coulomb term, and the
            // exchange integral carries half the fraction because the density
            // is spin-summed.
            const double coulombIntegral =
                moEri[((p * numOrbitals + p) * numOrbitals + i) * numOrbitals + i];
            const double exchangeIntegral =
                moEri[((p * numOrbitals + i) * numOrbitals + p) * numOrbitals + i];
            value += 2.0 * (coulombIntegral - 0.5 * exchangeFraction * exchangeIntegral);
        }

        orbitalEnergies[p] = value;
    }

    return KsReference{std::move(*inputs),
                       std::move(shared),
                       std::move(*result),
                       std::move(orbitalEnergies),
                       std::move(moEri),
                       coreMo,
                       numOccupied,
                       numVirtual,
                       exchangeFraction};
}

/// The energy the finite difference walks: the electronic energy of the
/// closed-shell density on the renormalised occupied-orbital path, with the XC
/// part integrated by the engine on the same grid the operator's kernel reads.
double Energy(const KsReference& reference, std::span<const double> amplitudes) {
    const std::size_t numOrbitals = reference.numOccupied + reference.numVirtual;
    const std::size_t numOccupied = reference.numOccupied;

    Eigen::MatrixXd mixing = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(numOrbitals),
                                                   static_cast<Eigen::Index>(numOccupied));

    for (std::size_t i = 0; i < numOccupied; ++i)
    {
        mixing(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 1.0;
    }

    for (std::size_t a = 0; a < reference.numVirtual; ++a)
    {
        for (std::size_t i = 0; i < numOccupied; ++i)
        {
            mixing(static_cast<Eigen::Index>(numOccupied + a), static_cast<Eigen::Index>(i)) =
                amplitudes[i * reference.numVirtual + a];
        }
    }

    const Eigen::MatrixXd overlap = mixing.transpose() * mixing;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> root(overlap);
    const Eigen::MatrixXd inverseRoot = root.eigenvectors() *
                                        root.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() *
                                        root.eigenvectors().transpose();
    const Eigen::MatrixXd orbitals = mixing * inverseRoot;
    const Eigen::MatrixXd density = 2.0 * orbitals * orbitals.transpose();

    double energy = (reference.coreMo.cwiseProduct(density)).sum();
    double twoElectron = 0.0;
    // Neumaier compensation: the second difference divides by 4h^2 = 4e-8, so
    // an n^4 sum whose terms are order one leaves a visible floor without it.
    // The water case (n = 7) is where that floor crosses the gate.
    double compensation = 0.0;

    for (std::size_t p = 0; p < numOrbitals; ++p)
    {
        for (std::size_t q = 0; q < numOrbitals; ++q)
        {
            for (std::size_t r = 0; r < numOrbitals; ++r)
            {
                for (std::size_t s = 0; s < numOrbitals; ++s)
                {
                    const double densityElement =
                        density(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q));
                    const double partner =
                        density(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(s));
                    const std::size_t index =
                        ((p * numOrbitals + q) * numOrbitals + r) * numOrbitals + s;
                    const std::size_t exchangeIndex =
                        ((p * numOrbitals + r) * numOrbitals + q) * numOrbitals + s;
                    const double term =
                        densityElement * partner *
                        (reference.moTwoElectron[index] -
                         0.5 * reference.exchangeFraction * reference.moTwoElectron[exchangeIndex]);
                    const double sum = twoElectron + term;
                    compensation += std::abs(twoElectron) >= std::abs(term)
                                        ? (twoElectron - sum) + term
                                        : (term - sum) + twoElectron;
                    twoElectron = sum;
                }
            }
        }
    }

    energy += 0.5 * (twoElectron + compensation);

    const Eigen::MatrixXd& coefficients = reference.scf.coefficients;
    const Eigen::MatrixXd aoDensity = coefficients * density * coefficients.transpose();
    const auto evaluation = reference.engine->EvaluateClosedShell(aoDensity);

    if (!evaluation.has_value())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return energy + evaluation->energy;
}

/// One matrix element of the numerical Hessian, by the four-point central
/// difference at a Richardson-extrapolated step.
double NumericalEntry(const KsReference& reference,
                      std::size_t first,
                      std::size_t second,
                      double step) {
    const std::size_t dimension = reference.numOccupied * reference.numVirtual;
    std::vector<double> amplitudes(dimension, 0.0);

    // Both corners accumulate, so a diagonal entry is a displacement of the one
    // amplitude by two steps rather than two writes to the same slot: the
    // stencil stays a valid second difference at 2h, and dividing by 4h^2 is
    // exactly that second difference's denominator.
    amplitudes[first] += step;
    amplitudes[second] += step;
    const double plusPlus = Energy(reference, amplitudes);

    amplitudes[second] -= 2.0 * step;
    const double plusMinus = Energy(reference, amplitudes);

    amplitudes[first] -= 2.0 * step;
    const double minusMinus = Energy(reference, amplitudes);

    amplitudes[second] += 2.0 * step;
    const double minusPlus = Energy(reference, amplitudes);

    return (plusPlus - plusMinus - minusPlus + minusMinus) / (4.0 * step * step);
}

double NumericalHessianEntry(const KsReference& reference, std::size_t first, std::size_t second) {
    const double coarse = NumericalEntry(reference, first, second, kFiniteDifferenceStep);
    const double fine = NumericalEntry(reference, first, second, 0.5 * kFiniteDifferenceStep);

    return (4.0 * fine - coarse) / 3.0;
}

/// The action as a dense matrix, one unit vector at a time.
Eigen::MatrixXd DenseAction(const OrbitalHessianOperator& op) {
    const std::size_t dimension = op.Dimension();
    const auto size = static_cast<Eigen::Index>(dimension);
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(size, size);
    std::vector<double> unit(dimension, 0.0);
    std::vector<double> product(dimension, 0.0);

    for (std::size_t column = 0; column < dimension; ++column)
    {
        std::fill(unit.begin(), unit.end(), 0.0);
        unit[column] = 1.0;
        const auto applied = op.Apply(unit, product);

        if (!applied.has_value())
        {
            return Eigen::MatrixXd::Constant(size, size, std::numeric_limits<double>::quiet_NaN());
        }

        for (std::size_t row = 0; row < dimension; ++row)
        {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
                product[row];
        }
    }

    return matrix;
}

/// The operator's four-times-action matrix against the numerical Hessian.
double CompareWithNumerical(const KsReference& reference, const OrbitalHessianOperator& op) {
    const Eigen::MatrixXd action = DenseAction(op);
    const std::size_t dimension = op.Dimension();
    double maxDeviation = 0.0;

    for (std::size_t row = 0; row < dimension; ++row)
    {
        for (std::size_t column = 0; column < dimension; ++column)
        {
            const double numerical = NumericalHessianEntry(reference, row, column);
            const double analytic =
                4.0 * action(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
            maxDeviation = std::max(maxDeviation, std::abs(numerical - analytic));
        }
    }

    return maxDeviation;
}

/// Everything one acceptance holds: the reference, the kernel action, the
/// kernel operator and the bare operator.
struct OperatorFixture {
    KsReference reference;
    std::shared_ptr<qcx::grid::XcKernelAction> action;
    std::optional<KohnShamOrbitalHessian> kernelOperator;
    std::optional<OrbitalHessianOperator> bareOperator;
};

std::optional<OperatorFixture> MakeOperatorFixture(bool water,
                                                   std::string_view functionalName,
                                                   double exchangeFraction) {
    auto reference = BuildKsReference(water, functionalName, exchangeFraction);

    if (!reference.has_value())
    {
        return std::nullopt;
    }

    auto action =
        qcx::grid::XcKernelAction::Create(reference->engine->Grid().Blocks(),
                                          reference->engine->Evaluator(),
                                          reference->engine->Functional(),
                                          reference->engine->ShellEnvelopes(),
                                          reference->scf.coefficients,
                                          reference->scf.density,
                                          reference->numOccupied,
                                          reference->numVirtual,
                                          qcx::grid::XcKernelSettings{kScreeningTolerance});

    if (!action.has_value())
    {
        return std::nullopt;
    }

    auto sharedAction = std::make_shared<qcx::grid::XcKernelAction>(std::move(*action));

    const ResponseLayout layout{reference->numOccupied, reference->numVirtual};
    const OrbitalEnergies energies{reference->orbitalEnergies};
    const MoTwoElectronTensor moTwoElectron{reference->moTwoElectron};

    auto bare = OrbitalHessianOperator::Create(layout, energies, moTwoElectron, exchangeFraction);

    if (!bare.has_value())
    {
        return std::nullopt;
    }

    KohnShamKernelTerm term{
        std::string(functionalName),
        [sharedAction](std::span<const double> trial, std::span<double> result) {
            return sharedAction->Apply(trial, result);
        }};

    auto built = KohnShamOrbitalHessian::Create(
        layout, energies, moTwoElectron, exchangeFraction, std::move(term));

    if (!built.has_value())
    {
        return std::nullopt;
    }

    OperatorFixture fixture{
        std::move(*reference), std::move(sharedAction), std::move(*built), std::move(*bare)};
    return fixture;
}

void CheckAcceptance(bool water, std::string_view functionalName, double exchangeFraction) {
    auto fixture = MakeOperatorFixture(water, functionalName, exchangeFraction);
    ASSERT_TRUE(fixture.has_value()) << functionalName;

    const double deviation =
        CompareWithNumerical(fixture->reference, fixture->kernelOperator->BaseOperator());

    ::testing::Test::RecordProperty(std::string(functionalName) + " agreement (max deviation)",
                                    Scientific(deviation));
    EXPECT_LT(deviation, kFiniteDifferenceTolerance)
        << functionalName << ": the operator and the numerical Hessian disagree";
}

TEST(ResponseKernelTest, TheKernelOperatorMatchesTheNumericalHessianForSlaterOnH2) {
    CheckAcceptance(false, "slater", 0.0);
}

TEST(ResponseKernelTest, TheKernelOperatorMatchesTheNumericalHessianForPbeWithExactExchangeOnH2) {
    CheckAcceptance(false, "pbe", 0.25);
}

TEST(ResponseKernelTest, TheKernelOperatorMatchesTheNumericalHessianForTauXOnH2) {
    CheckAcceptance(false, "tau_x", 0.0);
}

TEST(ResponseKernelTest, TheKernelOperatorMatchesTheNumericalHessianForSlaterOnH2o) {
    CheckAcceptance(true, "slater", 0.0);
}

// The kernel term is load-bearing: with it the numerical Hessian matches, and
// without it the same numerical Hessian is off by far more than the gate. A
// bare operator that happened to match would make every acceptance above empty.
TEST(ResponseKernelTest, TheKernelTermIsWhatMakesTheHessianMatch) {
    auto fixture = MakeOperatorFixture(true, "slater", 0.0);
    ASSERT_TRUE(fixture.has_value());

    const std::size_t dimension = fixture->reference.numOccupied * fixture->reference.numVirtual;
    const Eigen::MatrixXd bareAction = DenseAction(*fixture->bareOperator);
    const Eigen::MatrixXd kernelAction = DenseAction(fixture->kernelOperator->BaseOperator());

    double maxBareDeviation = 0.0;
    double maxKernelDeviation = 0.0;

    for (std::size_t row = 0; row < dimension; ++row)
    {
        for (std::size_t column = 0; column < dimension; ++column)
        {
            const double numerical = NumericalHessianEntry(fixture->reference, row, column);
            const double bareEntry =
                4.0 * bareAction(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
            const double kernelEntry = 4.0 * kernelAction(static_cast<Eigen::Index>(row),
                                                          static_cast<Eigen::Index>(column));
            maxBareDeviation = std::max(maxBareDeviation, std::abs(numerical - bareEntry));
            maxKernelDeviation = std::max(maxKernelDeviation, std::abs(numerical - kernelEntry));
        }
    }

    RecordProperty("bare-operator disagreement (max)", Scientific(maxBareDeviation));
    RecordProperty("kernel-operator agreement (max)", Scientific(maxKernelDeviation));

    EXPECT_LT(maxKernelDeviation, kFiniteDifferenceTolerance);
    EXPECT_GT(maxBareDeviation, 1e-4) << "the kernel term does not change the Hessian measurably";
}

TEST(ResponseKernelTest, ExchangeFractionOneKeepsTheHartreeFockLoopBitIdentical) {
    constexpr std::size_t kOccupied = 3;
    constexpr std::size_t kVirtual = 3;
    const std::size_t numOrbitals = kOccupied + kVirtual;
    std::vector<double> energies(numOrbitals);
    std::vector<double> integrals(numOrbitals * numOrbitals * numOrbitals * numOrbitals);
    std::vector<double> trial(kOccupied * kVirtual);

    for (std::size_t i = 0; i < numOrbitals; ++i)
    {
        energies[i] = 0.3 * static_cast<double>(i) + 0.05;
    }

    for (std::size_t i = 0; i < integrals.size(); ++i)
    {
        integrals[i] = std::sin(0.11 * static_cast<double>(i) + 0.7);
    }

    for (std::size_t i = 0; i < trial.size(); ++i)
    {
        trial[i] = std::cos(0.23 * static_cast<double>(i));
    }

    const ResponseLayout layout{kOccupied, kVirtual};
    const OrbitalEnergies orbitalEnergies{energies};
    const MoTwoElectronTensor moTwoElectron{integrals};

    const auto plain = OrbitalHessianOperator::Create(layout, orbitalEnergies, moTwoElectron);
    const auto fraction =
        OrbitalHessianOperator::Create(layout, orbitalEnergies, moTwoElectron, 1.0);
    ASSERT_TRUE(plain.has_value() && fraction.has_value());

    std::vector<double> plainResult(trial.size(), 0.0);
    std::vector<double> fractionResult(trial.size(), 0.0);
    ASSERT_TRUE(plain->Apply(trial, plainResult).has_value());
    ASSERT_TRUE(fraction->Apply(trial, fractionResult).has_value());

    for (std::size_t i = 0; i < trial.size(); ++i)
    {
        // Exact, not near: the fraction-1.0 branch is the loop the Hartree-Fock
        // anchors were computed on, and a rewritten copy would drift.
        EXPECT_EQ(plainResult[i], fractionResult[i]) << i;
    }
}

TEST(ResponseKernelTest, AnExchangeFractionOutsideTheUnitIntervalIsRefused) {
    constexpr std::size_t kOccupied = 2;
    constexpr std::size_t kVirtual = 2;
    const std::size_t numOrbitals = kOccupied + kVirtual;
    const std::vector<double> energies(numOrbitals, 0.5);
    const std::vector<double> integrals(numOrbitals * numOrbitals * numOrbitals * numOrbitals, 0.1);

    const ResponseLayout layout{kOccupied, kVirtual};

    for (const double fraction : {1.5, -0.1, std::nan("")})
    {
        const auto built = OrbitalHessianOperator::Create(
            layout, OrbitalEnergies{energies}, MoTwoElectronTensor{integrals}, fraction);
        ASSERT_FALSE(built.has_value());
        EXPECT_EQ(built.error().code, qcx::ErrorCode::kInvalidArgument);
    }
}

TEST(ResponseKernelTest, AKernelTermWithoutANameOrAnActionIsRefused) {
    constexpr std::size_t kOccupied = 2;
    constexpr std::size_t kVirtual = 2;
    const std::size_t numOrbitals = kOccupied + kVirtual;
    const std::vector<double> energies(numOrbitals, 0.5);
    const std::vector<double> integrals(numOrbitals * numOrbitals * numOrbitals * numOrbitals, 0.1);
    const ResponseLayout layout{kOccupied, kVirtual};

    KohnShamKernelTerm emptyName;
    emptyName.apply = [](std::span<const double>, std::span<double>) -> qcx::Result<void> {
        return {};
    };

    const auto named = KohnShamOrbitalHessian::Create(layout,
                                                      OrbitalEnergies{energies},
                                                      MoTwoElectronTensor{integrals},
                                                      0.0,
                                                      std::move(emptyName));
    ASSERT_FALSE(named.has_value());
    EXPECT_EQ(named.error().code, qcx::ErrorCode::kInvalidArgument);

    KohnShamKernelTerm noAction{"pbe", HessianExtraTermFn{}};
    const auto acted = KohnShamOrbitalHessian::Create(layout,
                                                      OrbitalEnergies{energies},
                                                      MoTwoElectronTensor{integrals},
                                                      0.0,
                                                      std::move(noAction));
    ASSERT_FALSE(acted.has_value());
    EXPECT_EQ(acted.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace qcx::response
