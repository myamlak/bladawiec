// Synthetic tests of the restricted open-shell step functions: the Roothaan
// effective Fock's block table (the method's whole content - the three
// stationarity conditions live in three of its blocks), its two limits (the
// closed-shell spin average and the high-spin alpha Fock), the two-shift
// level shift, the weighted DIIS error factors, and the loop's closed-form
// end-to-end behaviour on a constant Fock pair, where every number can be
// written down in advance. No integrals are built here: these tests pin the
// algebra the direct-rohf pins then run through real builders.

#include "qcx/scf/diis.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/rohf.hpp"
#include "rohf_fock.hpp"
#include "scf_common.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

// A fixed dense orthogonal matrix: the identity composed with one Givens
// rotation per (i, j) pair, at a fixed angle. Deterministic and exactly
// orthogonal to rounding - a hardware eigen-solve would bring platform
// sign and ordering choices into the fixtures for nothing.
Eigen::MatrixXd MakeRotationChain(Eigen::Index n) {
    Eigen::MatrixXd q = Eigen::MatrixXd::Identity(n, n);

    for (Eigen::Index i = 0; i < n; ++i)
    {
        for (Eigen::Index j = i + 1; j < n; ++j)
        {
            const double angle = 0.17 * static_cast<double>((i + 1) * (j + 1));
            Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(n, n);
            rotation(i, i) = std::cos(angle);
            rotation(i, j) = -std::sin(angle);
            rotation(j, i) = std::sin(angle);
            rotation(j, j) = std::cos(angle);
            q = q * rotation;
        }
    }

    return q;
}

// A fixed symmetric matrix with no accidental structure (a Cauchy-like
// off-diagonal decay plus a scaled diagonal).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd MakeSymmetricMatrix(Eigen::Index n, double scale, double diagonal) {
    Eigen::MatrixXd matrix(n, n);

    for (Eigen::Index i = 0; i < n; ++i)
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            matrix(i, j) = scale / (1.0 + static_cast<double>(i + j)) +
                           (i == j ? diagonal + 0.37 * static_cast<double>(i) : 0.0);
        }
    }

    return matrix;
}

// The (r, s) block of m in the q basis: q_r^T m q_s.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd BlockOf(const Eigen::MatrixXd& q,
                        const Eigen::MatrixXd& m,
                        Eigen::Index first,
                        Eigen::Index count,
                        Eigen::Index otherFirst,
                        Eigen::Index otherCount) {
    return q.middleCols(first, count).transpose() * m * q.middleCols(otherFirst, otherCount);
}

double MaxAbs(const Eigen::MatrixXd& matrix) {
    return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff();
}

// The three projectors of an orthonormal column set split [closed | open |
// virtual].
struct BlockProjectors {
    Eigen::MatrixXd closed;
    Eigen::MatrixXd open;
    Eigen::MatrixXd virtualSpace;
};

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
BlockProjectors MakeBlockProjectors(const Eigen::MatrixXd& q,
                                    Eigen::Index numClosed,
                                    Eigen::Index numOpen) {
    const Eigen::Index n = q.rows();
    BlockProjectors projectors;
    const Eigen::MatrixXd closedBlock = q.leftCols(numClosed);
    const Eigen::MatrixXd openBlock = q.middleCols(numClosed, numOpen);
    projectors.closed = closedBlock * closedBlock.transpose();
    projectors.open = openBlock * openBlock.transpose();
    projectors.virtualSpace = Eigen::MatrixXd::Identity(n, n) - projectors.closed - projectors.open;
    return projectors;
}

// H2 at R = 1.4 Bohr with a chosen multiplicity: the loop reads only the
// electron count, the multiplicity and the nuclear repulsion, so these
// synthetic runs need no integrals at all.
qcx::Result<qcx::molecule::Molecule> MakeH2WithMultiplicity(int multiplicity) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.4;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        multiplicity);
}

TEST(RohfAssemblyTest, EffectiveFockBlocksAreTheRoothaanOnes) {
    constexpr Eigen::Index n = 6;
    constexpr Eigen::Index numClosed = 2;
    constexpr Eigen::Index numOpen = 1;
    constexpr Eigen::Index numVirtual = n - numClosed - numOpen;

    const Eigen::MatrixXd fockAlpha = MakeSymmetricMatrix(n, 1.7, -3.0);
    const Eigen::MatrixXd fockBeta = MakeSymmetricMatrix(n, -0.9, -2.0);
    const Eigen::MatrixXd q = MakeRotationChain(n);
    const BlockProjectors p = MakeBlockProjectors(q, numClosed, numOpen);
    const Eigen::MatrixXd average = 0.5 * (fockAlpha + fockBeta);

    const Eigen::MatrixXd effective =
        qcx::scf::internal::AssembleRoothaanFock(fockAlpha, fockBeta, p.closed, p.open);

    // The loop's matrix is symmetric because each of its five terms is.
    EXPECT_LT(MaxAbs(effective - effective.transpose()), 1e-13);

    // The block table, cell by cell. The three blocks whose value is not the
    // spin average are the ones that carry the method: the open-open and
    // open-virtual blocks are the alpha Fock (whose vanishing off-diagonal
    // part is the open stationarity condition), and the closed-open block is
    // the beta Fock (whose vanishing is the closed stationarity condition).
    EXPECT_LT(MaxAbs(BlockOf(q, effective, 0, numClosed, 0, numClosed) -
                     BlockOf(q, average, 0, numClosed, 0, numClosed)),
              1e-13);
    EXPECT_LT(MaxAbs(BlockOf(q, effective, numClosed, numOpen, numClosed, numOpen) -
                     BlockOf(q, fockAlpha, numClosed, numOpen, numClosed, numOpen)),
              1e-13);
    EXPECT_LT(MaxAbs(BlockOf(q, effective, 0, numClosed, numClosed, numOpen) -
                     BlockOf(q, fockBeta, 0, numClosed, numClosed, numOpen)),
              1e-13);
    EXPECT_LT(
        MaxAbs(
            BlockOf(
                q, effective, numClosed + numOpen, numVirtual, numClosed + numOpen, numVirtual) -
            BlockOf(q, average, numClosed + numOpen, numVirtual, numClosed + numOpen, numVirtual)),
        1e-13);
    EXPECT_LT(MaxAbs(BlockOf(q, effective, numClosed, numOpen, numClosed + numOpen, numVirtual) -
                     BlockOf(q, fockAlpha, numClosed, numOpen, numClosed + numOpen, numVirtual)),
              1e-13);
    EXPECT_LT(MaxAbs(BlockOf(q, effective, 0, numClosed, numClosed + numOpen, numVirtual) -
                     BlockOf(q, average, 0, numClosed, numClosed + numOpen, numVirtual)),
              1e-13);

    // The same table written as a whole-matrix reconstruction: a reader who
    // doubts one cell can check the sum.
    const Eigen::MatrixXd reconstructed =
        p.closed * average * p.closed + p.open * fockAlpha * p.open + p.closed * fockBeta * p.open +
        p.open * fockBeta * p.closed + p.closed * average * p.virtualSpace +
        p.virtualSpace * average * p.closed + p.open * fockAlpha * p.virtualSpace +
        p.virtualSpace * fockAlpha * p.open + p.virtualSpace * average * p.virtualSpace;
    EXPECT_LT(MaxAbs(effective - reconstructed), 1e-13);

    // The stationarity reading: at a pair whose beta Fock vanishes on the
    // closed-open space, that block of the effective Fock vanishes too. The
    // removal is the two-sided projector subtraction that defines the block
    // (P_c X P_o + P_o X P_c), not a one-sided rank-one edit - the closed
    // and open spaces are orthogonal complements here, so the block it
    // removes is exactly the one the assertion reads back.
    Eigen::MatrixXd solvedBeta = fockBeta;
    const Eigen::MatrixXd closedOpenBlock = p.closed * solvedBeta * p.open;
    const Eigen::MatrixXd openClosedBlock = p.open * solvedBeta * p.closed;
    solvedBeta -= closedOpenBlock + openClosedBlock;
    const Eigen::MatrixXd solvedEffective =
        qcx::scf::internal::AssembleRoothaanFock(fockAlpha, solvedBeta, p.closed, p.open);
    EXPECT_LT(MaxAbs(BlockOf(q, solvedEffective, 0, numClosed, numClosed, numOpen)), 1e-13);
}

TEST(RohfAssemblyTest, ClosedShellLimitIsTheSpinAverage) {
    constexpr Eigen::Index n = 5;
    constexpr Eigen::Index numClosed = 3;

    const Eigen::MatrixXd fockAlpha = MakeSymmetricMatrix(n, 1.1, -4.0);
    const Eigen::MatrixXd fockBeta = MakeSymmetricMatrix(n, 0.4, -5.0);
    const Eigen::MatrixXd q = MakeRotationChain(n);
    const BlockProjectors p = MakeBlockProjectors(q, numClosed, 0);

    const Eigen::MatrixXd effective =
        qcx::scf::internal::AssembleRoothaanFock(fockAlpha, fockBeta, p.closed, p.open);

    // nO = 0: every Delta term carries a P_o factor, so the assembly is the
    // spin average exactly and the loop's effective Fock IS the RHF Fock -
    // which is why singlet ROHF reproduces RHF.
    EXPECT_LT(MaxAbs(effective - 0.5 * (fockAlpha + fockBeta)), 1e-14);
}

TEST(RohfAssemblyTest, HighSpinLimitIsTheAlphaFockOnTheOccupiedSpace) {
    constexpr Eigen::Index n = 5;
    constexpr Eigen::Index numOpen = 3;
    constexpr Eigen::Index numVirtual = n - numOpen;

    const Eigen::MatrixXd fockAlpha = MakeSymmetricMatrix(n, 1.3, -2.5);
    const Eigen::MatrixXd fockBeta = MakeSymmetricMatrix(n, -0.6, -1.5);
    const Eigen::MatrixXd q = MakeRotationChain(n);
    const BlockProjectors p = MakeBlockProjectors(q, 0, numOpen);

    const Eigen::MatrixXd effective =
        qcx::scf::internal::AssembleRoothaanFock(fockAlpha, fockBeta, p.closed, p.open);

    // nC = 0: the closed block is empty and P_c = 0, so the occupied block of
    // the effective Fock is the alpha Fock and the open-virtual block too -
    // the high-spin limit, where the loop's fixed point is the UHF fixed
    // point of the same determinant.
    EXPECT_LT(MaxAbs(BlockOf(q, effective, 0, numOpen, 0, numOpen) -
                     BlockOf(q, fockAlpha, 0, numOpen, 0, numOpen)),
              1e-13);
    EXPECT_LT(MaxAbs(BlockOf(q, effective, 0, numOpen, numOpen, numVirtual) -
                     BlockOf(q, fockAlpha, 0, numOpen, numOpen, numVirtual)),
              1e-13);

    // And it is the UHF alpha equation: diagonalizing it is diagonalizing
    // the alpha Fock on the occupied space, with the alpha stationarity
    // blocks (occ-occ and occ-virtual) the only ones it can enforce.
    const Eigen::MatrixXd alphaOccupied = BlockOf(q, fockAlpha, 0, numOpen, 0, numOpen);
    EXPECT_LT(MaxAbs(alphaOccupied - alphaOccupied.transpose()), 1e-13);
}

TEST(RohfStepTest, TheTwoShiftLevelShiftLiftsTheOpenAndVirtualBlocksOnly) {
    constexpr Eigen::Index n = 6;
    constexpr Eigen::Index numClosed = 2;
    constexpr Eigen::Index numOpen = 1;
    constexpr Eigen::Index numVirtual = n - numClosed - numOpen;
    constexpr double openShift = 0.25;
    constexpr double virtualShift = 0.75;

    const Eigen::MatrixXd fock = MakeSymmetricMatrix(n, 0.7, -6.0);
    const Eigen::MatrixXd q = MakeRotationChain(n);
    const BlockProjectors p = MakeBlockProjectors(q, numClosed, numOpen);

    Eigen::MatrixXd shifted = fock;
    qcx::scf::internal::ApplyOpenShellLevelShift(
        shifted, p.closed, p.open, openShift, virtualShift);

    // The closed block is the zero of the closed-relative form; the open and
    // virtual blocks move by their own shifts, and every cross-block entry
    // stays where it was.
    Eigen::MatrixXd expected = fock;
    expected += openShift * p.open;
    expected += virtualShift * p.virtualSpace;
    EXPECT_LT(MaxAbs(shifted - expected), 1e-13);

    EXPECT_LT(MaxAbs(BlockOf(q, shifted, 0, numClosed, 0, numClosed) -
                     BlockOf(q, fock, 0, numClosed, 0, numClosed)),
              1e-14);
    EXPECT_LT(MaxAbs(BlockOf(q, shifted, numClosed, numOpen, numClosed, numOpen) -
                     BlockOf(q, fock, numClosed, numOpen, numClosed, numOpen) -
                     openShift * Eigen::MatrixXd::Identity(numOpen, numOpen)),
              1e-14);
    EXPECT_LT(
        MaxAbs(
            BlockOf(q, shifted, numClosed + numOpen, numVirtual, numClosed + numOpen, numVirtual) -
            BlockOf(q, fock, numClosed + numOpen, numVirtual, numClosed + numOpen, numVirtual) -
            virtualShift * Eigen::MatrixXd::Identity(numVirtual, numVirtual)),
        1e-14);

    // A zero pair is a legal no-op (the default the loop carries).
    Eigen::MatrixXd unchanged = fock;
    qcx::scf::internal::ApplyOpenShellLevelShift(unchanged, p.closed, p.open, 0.0, 0.0);
    EXPECT_LE(MaxAbs(unchanged - fock), 0.0);
}

TEST(RohfStepTest, TheWeightedErrorFactorsReproduceTheCommutatorExactly) {
    constexpr Eigen::Index n = 6;
    constexpr Eigen::Index numClosed = 2;
    constexpr Eigen::Index numOpen = 1;
    constexpr Eigen::Index numOccupied = numClosed + numOpen;

    const Eigen::MatrixXd fock = MakeSymmetricMatrix(n, 0.9, -7.0);
    const Eigen::MatrixXd q = MakeRotationChain(n);

    // The restricted density's own factor: C_s = [sqrt(2) C_c, C_o], so that
    // D = C_s C_s^T carries the 2/1 occupations of the closed/open blocks.
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(numOccupied);
    weights.head(numClosed).setConstant(std::sqrt(2.0));
    const Eigen::MatrixXd scaled = q.leftCols(numOccupied) * weights.asDiagonal();

    const Eigen::MatrixXd a = fock * scaled;

    const auto factors = qcx::scf::internal::ProjectDiisErrorFactors(a, scaled, weights);
    ASSERT_TRUE(factors.has_value()) << factors.error().message;

    // The commutator e = A B^T - B A^T, which is the DIIS error of the loop
    // in its solver space, must be reconstructed by the factors exactly: the
    // B^T B = diag(2 I_c, I_o) projection is exact, and the identity
    // e = B s B^T + R B^T - B R^T is an algebraic regroup of the same
    // products.
    const Eigen::MatrixXd commutator = a * scaled.transpose() - scaled * a.transpose();
    EXPECT_LT(MaxAbs(factors->Expand() - commutator), 1e-14);

    // And the split is the weighted one: the uniform RHF form (M = B^T A / 2)
    // would leave a different R, so pin the projected piece's orthogonality.
    EXPECT_LT(MaxAbs(scaled.transpose() * factors->r), 1e-12);

    // A weight vector that does not match the columns, and a non-positive
    // weight, are values rather than silent mis-splits.
    const Eigen::VectorXd tooShort = Eigen::VectorXd::Ones(numOccupied - 1);
    const auto shapeError = qcx::scf::internal::ProjectDiisErrorFactors(a, scaled, tooShort);
    ASSERT_FALSE(shapeError.has_value());
    EXPECT_EQ(shapeError.error().code, qcx::ErrorCode::kInvalidArgument);

    Eigen::VectorXd negative = weights;
    negative(0) = -1.0;
    const auto weightError = qcx::scf::internal::ProjectDiisErrorFactors(a, scaled, negative);
    ASSERT_FALSE(weightError.has_value());
    EXPECT_EQ(weightError.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(RohfLoopTest, RejectsAnElectronCountAndMultiplicityThatDoNotFillTheBlocks) {
    auto molecule = MakeH2WithMultiplicity(1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // Two electrons cannot make a doublet with an odd open count: nO = 1
    // leaves one electron over an even split.
    auto doublet = MakeH2WithMultiplicity(2);
    ASSERT_TRUE(doublet.has_value()) << doublet.error().message;

    const Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd core = Eigen::MatrixXd::Zero(2, 2);
    const qcx::scf::UhfFockBuilderFn builder =
        [](const Eigen::MatrixXd&,
           const Eigen::MatrixXd&) -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{Eigen::MatrixXd::Zero(2, 2),
                                                           Eigen::MatrixXd::Zero(2, 2)};
    };

    const qcx::scf::RohfOptions options;
    const auto run = qcx::scf::RunRohfScf(*doublet, overlap, core, options, builder);
    ASSERT_FALSE(run.has_value());
    EXPECT_EQ(run.error().code, qcx::ErrorCode::kInvalidArgument);

    // An empty callback and a non-positive option pair are values too.
    const auto empty =
        qcx::scf::RunRohfScf(*molecule, overlap, core, options, qcx::scf::UhfFockBuilderFn{});
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, qcx::ErrorCode::kInvalidArgument);

    qcx::scf::RohfOptions badOptions;
    badOptions.maxIterations = 0;
    const auto bad = qcx::scf::RunRohfScf(*molecule, overlap, core, badOptions, builder);
    ASSERT_FALSE(bad.has_value());
    EXPECT_EQ(bad.error().code, qcx::ErrorCode::kInvalidArgument);

    // The level-shift pair is the closed-relative two-shift form, so the open
    // shift can never exceed the virtual one (that would shift the closed
    // block downwards - the "open above a virtual" failure the method names).
    qcx::scf::RohfOptions openAboveVirtual;
    openAboveVirtual.useLevelShift = true;
    openAboveVirtual.levelShiftOpen = 0.8;
    openAboveVirtual.levelShiftVirtual = 0.3;
    const auto badShift = qcx::scf::RunRohfScf(*molecule, overlap, core, openAboveVirtual, builder);
    ASSERT_FALSE(badShift.has_value());
    EXPECT_EQ(badShift.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(badShift.error().message.find("levelShiftVirtual"), std::string::npos);
}

TEST(RohfLoopTest, TheLevelShiftedRunConvergesToTheUnshiftedFixedPoint) {
    auto molecule = MakeH2WithMultiplicity(3);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd core = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd fock(2, 2);
    fock << -1.0, 0.3, 0.3, 0.5;
    const Eigen::MatrixXd constantFock = fock;
    const qcx::scf::UhfFockBuilderFn builder =
        [constantFock](
            const Eigen::MatrixXd&,
            const Eigen::MatrixXd&) -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{constantFock, constantFock};
    };

    const auto plain =
        qcx::scf::RunRohfScf(*molecule, overlap, core, qcx::scf::RohfOptions{}, builder);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    qcx::scf::RohfOptions shifted;
    shifted.useLevelShift = true;
    shifted.levelShiftOpen = 0.25;
    shifted.levelShiftVirtual = 0.75;
    const auto run = qcx::scf::RunRohfScf(*molecule, overlap, core, shifted, builder);
    ASSERT_TRUE(run.has_value()) << run.error().message;
    ASSERT_TRUE(run->converged);

    // The shift is a diagonalization aid, not a change of equations: the
    // converged state is the same unshifted fixed point, and the reported
    // spectrum is the UNshifted effective Fock's in both cases.
    EXPECT_NEAR(run->totalEnergy, plain->totalEnergy, 1e-13);
    EXPECT_NEAR(run->spinSquared, plain->spinSquared, 1e-13);
    EXPECT_LT(MaxAbs(run->densityAlpha - plain->densityAlpha), 1e-13);
    EXPECT_LT(MaxAbs(run->densityBeta - plain->densityBeta), 1e-13);
    EXPECT_LT(MaxAbs(run->orbitalEnergiesAlpha - plain->orbitalEnergiesAlpha), 1e-13);
}

TEST(RohfLoopTest, AConstantFockPairRunsTheFixedOccupationRule) {
    auto molecule = MakeH2WithMultiplicity(3);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The identity overlap makes the orthogonalizer the identity, so the
    // loop's solver space IS the AO space and every number below is the
    // textbook one. A constant Fock pair makes the fixed point a single
    // diagonalization: the first iterate's projectors are zero (no seed), the
    // second iterate's are the first one's orbitals, so the run stops at the
    // second iterate.
    const Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd core = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd fock(2, 2);
    fock << -1.0, 0.3, 0.3, 0.5;
    const Eigen::MatrixXd constantFock = fock;

    const qcx::scf::UhfFockBuilderFn builder =
        [constantFock](
            const Eigen::MatrixXd&,
            const Eigen::MatrixXd&) -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{constantFock, constantFock};
    };

    const qcx::scf::RohfOptions options;
    const auto run = qcx::scf::RunRohfScf(*molecule, overlap, core, options, builder);
    ASSERT_TRUE(run.has_value()) << run.error().message;
    EXPECT_TRUE(run->converged);
    EXPECT_LE(run->iterations, 3);

    // Triplet H2: nC = 0 and nO = 2, so BOTH orbitals are open and carry one
    // alpha electron each - the alpha density is the identity and beta is
    // zero. The energy is 1/2 Tr[D_alpha F] + the nuclear repulsion of H2 at
    // 1.4 Bohr.
    EXPECT_LT(MaxAbs(run->densityAlpha - Eigen::MatrixXd::Identity(2, 2)), 1e-13);
    EXPECT_LT(MaxAbs(run->densityBeta), 1e-15);
    EXPECT_NEAR(run->electronicEnergy, 0.5 * (-1.0 + 0.5), 1e-13);
    EXPECT_NEAR(run->totalEnergy, run->electronicEnergy + 1.0 / 1.4, 1e-13);
    EXPECT_NEAR(run->spinSquared, 2.0, 1e-12);

    // The reported spectrum is the effective Fock's, ascending, in both
    // channels (this loop diagonalizes ONE matrix).
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(constantFock);
    ASSERT_EQ(run->orbitalEnergiesAlpha.size(), 2);
    EXPECT_NEAR(run->orbitalEnergiesAlpha(0), solver.eigenvalues()(0), 1e-13);
    EXPECT_NEAR(run->orbitalEnergiesAlpha(1), solver.eigenvalues()(1), 1e-13);
    EXPECT_LE(MaxAbs(run->orbitalEnergiesAlpha - run->orbitalEnergiesBeta), 0.0);

    // The restart record describes the returned state.
    EXPECT_LE(MaxAbs(run->restart.densityAlpha - run->densityAlpha), 0.0);
    EXPECT_LE(MaxAbs(run->restart.densityBeta - run->densityBeta), 0.0);
    EXPECT_NEAR(run->restart.previousTotalEnergy, run->totalEnergy, 1e-13);
}

TEST(RohfLoopTest, TheClosedShellSyntheticRunEqualsTheRhfLoop) {
    auto molecule = MakeH2WithMultiplicity(1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd core = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd fock(2, 2);
    fock << -1.0, 0.3, 0.3, 0.5;
    const Eigen::MatrixXd constantFock = fock;

    const qcx::scf::FockBuilderFn rhfBuilder =
        [constantFock](const Eigen::MatrixXd&) -> qcx::Result<Eigen::MatrixXd> {
        return constantFock;
    };
    const qcx::scf::UhfFockBuilderFn rohfBuilder =
        [constantFock](
            const Eigen::MatrixXd&,
            const Eigen::MatrixXd&) -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{constantFock, constantFock};
    };

    const qcx::scf::RhfOptions rhfOptions;
    const qcx::scf::RohfOptions rohfOptions;
    const auto rhf = qcx::scf::RunRhfScf(*molecule, overlap, core, rhfOptions, rhfBuilder);
    const auto rohf = qcx::scf::RunRohfScf(*molecule, overlap, core, rohfOptions, rohfBuilder);
    ASSERT_TRUE(rhf.has_value()) << rhf.error().message;
    ASSERT_TRUE(rohf.has_value()) << rohf.error().message;
    ASSERT_TRUE(rhf->converged);
    EXPECT_TRUE(rohf->converged);

    // The nO = 0 limit at the loop level: the same constant Fock gives the
    // same closed-shell determinant and the same energy through both loops.
    EXPECT_NEAR(rohf->totalEnergy, rhf->totalEnergy, 1e-13);
    EXPECT_NEAR(rohf->electronicEnergy, rhf->electronicEnergy, 1e-13);
    EXPECT_LT(MaxAbs(rohf->densityAlpha + rohf->densityBeta - rhf->density), 1e-13);
    EXPECT_NEAR(rohf->spinSquared, 0.0, 1e-13);
}

} // namespace
