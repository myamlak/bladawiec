#include "qcx/error.hpp"
#include "qcx/response/response_band.hpp"
#include "qcx/response/response_matrix.hpp"
#include "qcx/response/response_operator.hpp"
#include "qcx/response/response_solver.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

// The band selector's suite: the ceilings at their boundaries, the assembly
// the explicit bands consume, and one synthetic small-gap problem solved
// through all three bands and compared against a dense LDLT reference. Every
// fixture here is made up, so a failure names linear algebra before any
// chemistry is involved.

namespace {

using qcx::response::AssembleDenseResponseMatrix;
using qcx::response::AssembleSparseResponseMatrix;
using qcx::response::DenseResponseOperator;
using qcx::response::ResponseBandLimits;
using qcx::response::ResponseBandProblem;
using qcx::response::ResponseBandSolveReport;
using qcx::response::ResponseLayout;
using qcx::response::ResponseMatrixOptions;
using qcx::response::ResponseSolveOptions;
using qcx::response::ResponseSolver;
using qcx::response::SelectResponseBand;
using qcx::response::SolverBand;
using qcx::response::ToString;

/// A fixture's dimension: the count of rows or columns it is drawn at.
struct Dimension {
    std::size_t value = 0; ///< The count.
};

/// The seed a fixture's generator is started from.
struct Seed {
    std::uint32_t value = 0; ///< The generator seed.
};

/// Row-major flattening of a dense matrix.
std::vector<double> Flatten(const Eigen::MatrixXd& matrix) {
    const int rows = static_cast<int>(matrix.rows());
    const int columns = static_cast<int>(matrix.cols());
    std::vector<double> flat(static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns));

    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            const std::size_t offset =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(columns) +
                static_cast<std::size_t>(column);
            flat[offset] = matrix(row, column);
        }
    }

    return flat;
}

/// A strictly increasing diagonal spanning [1, largest] geometrically.
std::vector<double> MakeSpreadDiagonal(Dimension dimension, double largest) {
    std::vector<double> diagonal(dimension.value);

    for (std::size_t i = 0; i < dimension.value; ++i)
    {
        const double exponent = static_cast<double>(i) / static_cast<double>(dimension.value - 1);
        diagonal[i] = std::pow(largest, exponent);
    }

    return diagonal;
}

/// The 1D chain Laplacian with Dirichlet ends.
Eigen::MatrixXd MakeChainLaplacian(std::size_t dimension) {
    const int size = static_cast<int>(dimension);
    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(size, size);

    for (int i = 0; i < size; ++i)
    {
        laplacian(i, i) = 2.0;

        if (i > 0)
        {
            laplacian(i, i - 1) = -1.0;
            laplacian(i - 1, i) = -1.0;
        }
    }

    return laplacian;
}

/// Right-hand sides, contiguous and right-hand-side-major.
std::vector<double> MakeRightHandSides(Dimension dimension, std::size_t count, Seed seed) {
    std::mt19937 generator(seed.value);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<double> sides(count * dimension.value);

    for (double& value : sides)
    {
        value = distribution(generator);
    }

    return sides;
}

/// Builds the dense operator from an Eigen matrix and an explicit
/// preconditioner.
qcx::Result<DenseResponseOperator> MakeDenseOperator(std::size_t numOccupied,
                                                     std::size_t numVirtual,
                                                     const Eigen::MatrixXd& matrix,
                                                     std::span<const double> preconditioner) {
    const std::vector<double> flat = Flatten(matrix);

    return DenseResponseOperator::Create(
        ResponseLayout{numOccupied, numVirtual}, flat, preconditioner);
}

/// The small-gap physics every band is solved on: a well-conditioned operator
/// whose diagonal preconditioner spans ten decades. The operator is
/// I + 0.2 * (chain Laplacian) + diag(spread), so its smallest eigenvalue is
/// about one and its largest about two, whatever the spread does.
struct SmallGapFixture {
    std::size_t dimension = 0;
    Eigen::MatrixXd matrix;
    std::vector<double> preconditioner;
    ResponseLayout layout;
};

SmallGapFixture MakeSmallGapFixture(std::size_t dimension) {
    SmallGapFixture fixture;
    fixture.dimension = dimension;
    fixture.preconditioner = MakeSpreadDiagonal(Dimension{dimension}, 1e-10);
    fixture.matrix =
        Eigen::MatrixXd::Identity(static_cast<int>(dimension), static_cast<int>(dimension)) +
        0.2 * MakeChainLaplacian(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        fixture.matrix(static_cast<int>(i), static_cast<int>(i)) += fixture.preconditioner[i];
    }

    // Only the product matters; any occupied/virtual split of the dimension
    // is a legal layout for the dense operator.
    fixture.layout = ResponseLayout{1, dimension};

    return fixture;
}

/// The count of nonzero entries a dense matrix would assemble as CSR.
std::size_t CountNonzeros(const Eigen::MatrixXd& matrix) {
    std::size_t count = 0;

    for (int row = 0; row < matrix.rows(); ++row)
    {
        for (int column = 0; column < matrix.cols(); ++column)
        {
            if (matrix(row, column) != 0.0)
            {
                ++count;
            }
        }
    }

    return count;
}

/// One band's answer: its report and the concatenated solution.
struct BandRun {
    ResponseBandSolveReport report;
    std::vector<double> solution;
};

qcx::Result<BandRun> RunBand(const DenseResponseOperator& op,
                             const ResponseBandProblem& problem,
                             std::size_t rhsCount,
                             const std::vector<double>& rhs,
                             ResponseSolver<DenseResponseOperator>& matrixFreeSolver,
                             const ResponseBandLimits& limits) {
    std::vector<double> solution(rhsCount * op.Dimension());
    auto report = qcx::response::SolveResponseByBand(
        op, problem, rhsCount, rhs, solution, matrixFreeSolver, limits);

    if (!report.has_value())
    {
        return std::unexpected(report.error());
    }

    return BandRun{*report, std::move(solution)};
}

/// The worst absolute difference between a solved band and a dense LDLT
/// reference over every right-hand side.
double WorstDifferenceFromLdlt(const SmallGapFixture& fixture,
                               const std::vector<double>& solution,
                               std::size_t rhsCount,
                               const std::vector<double>& rhs) {
    const int size = static_cast<int>(fixture.dimension);
    Eigen::LDLT<Eigen::MatrixXd> reference(fixture.matrix);
    EXPECT_EQ(reference.info(), Eigen::Success);
    double worst = 0.0;

    for (std::size_t k = 0; k < rhsCount; ++k)
    {
        Eigen::VectorXd side(size);

        for (int i = 0; i < size; ++i)
        {
            side(i) = rhs[k * fixture.dimension + static_cast<std::size_t>(i)];
        }

        const Eigen::VectorXd exact = reference.solve(side);

        for (int i = 0; i < size; ++i)
        {
            worst =
                std::max(worst,
                         std::fabs(solution[k * fixture.dimension + static_cast<std::size_t>(i)] -
                                   exact(i)));
        }
    }

    return worst;
}

} // namespace

// --- the selector's ceilings ------------------------------------------------

TEST(ResponseBandTest, SelectorUsesTheDenseBandUpToItsCeiling) {
    const ResponseBandLimits limits;

    const auto atCeiling = SelectResponseBand(ResponseBandProblem{128, 0}, limits);
    ASSERT_TRUE(atCeiling.has_value()) << atCeiling.error().message;
    EXPECT_EQ(*atCeiling, SolverBand::kDense);

    const auto aboveCeiling = SelectResponseBand(ResponseBandProblem{129, 0}, limits);
    ASSERT_TRUE(aboveCeiling.has_value()) << aboveCeiling.error().message;
    EXPECT_EQ(*aboveCeiling, SolverBand::kExplicitCsr);

    const auto empty = SelectResponseBand(ResponseBandProblem{0, 0}, limits);
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(ResponseBandTest, SelectorUsesTheCsrBandUpToItsDimensionCeiling) {
    const ResponseBandLimits limits;

    const auto atCeiling = SelectResponseBand(ResponseBandProblem{5000, 0}, limits);
    ASSERT_TRUE(atCeiling.has_value()) << atCeiling.error().message;
    EXPECT_EQ(*atCeiling, SolverBand::kExplicitCsr);

    const auto aboveCeiling = SelectResponseBand(ResponseBandProblem{5001, 0}, limits);
    ASSERT_TRUE(aboveCeiling.has_value()) << aboveCeiling.error().message;
    EXPECT_EQ(*aboveCeiling, SolverBand::kMatrixFree);
}

TEST(ResponseBandTest, SelectorLeavesTheCsrBandBeyondTheNonzeroCap) {
    const ResponseBandLimits limits;

    const auto atCap = SelectResponseBand(ResponseBandProblem{5000, 25'000'000}, limits);
    ASSERT_TRUE(atCap.has_value()) << atCap.error().message;
    EXPECT_EQ(*atCap, SolverBand::kExplicitCsr);

    const auto aboveCap = SelectResponseBand(ResponseBandProblem{5000, 25'000'001}, limits);
    ASSERT_TRUE(aboveCap.has_value()) << aboveCap.error().message;
    EXPECT_EQ(*aboveCap, SolverBand::kMatrixFree);

    // The gate is a policy the caller can tighten: the same 400-unknown
    // problem falls to the matrix-free band when its nonzero estimate exceeds
    // the cap in force.
    ResponseBandLimits tightened;
    tightened.csrMaxNonzeros = 159'999;
    const auto forced = SelectResponseBand(ResponseBandProblem{400, 160'000}, tightened);
    ASSERT_TRUE(forced.has_value()) << forced.error().message;
    EXPECT_EQ(*forced, SolverBand::kMatrixFree);

    const auto inside = SelectResponseBand(ResponseBandProblem{400, 160'000}, limits);
    ASSERT_TRUE(inside.has_value()) << inside.error().message;
    EXPECT_EQ(*inside, SolverBand::kExplicitCsr);
}

// --- the assembly -----------------------------------------------------------

TEST(ResponseBandTest, AssemblyMatchesTheActionWithoutAssumingSymmetry) {
    constexpr std::size_t kOccupied = 3;
    constexpr std::size_t kVirtual = 4;
    const std::size_t dimension = kOccupied * kVirtual;
    const int size = static_cast<int>(dimension);

    // Deliberately not symmetric: the assembler must take each entry from the
    // action, not copy one triangle.
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(size, size);

    for (int row = 0; row < size; ++row)
    {
        for (int column = 0; column < size; ++column)
        {
            matrix(row, column) =
                0.3 * std::sin(0.7 * static_cast<double>(row) + 1.1 * static_cast<double>(column)) +
                (row == column ? 5.0 : 0.0);
        }
    }

    ASSERT_NE(matrix(0, 1), matrix(1, 0));
    const std::vector<double> diagonal = MakeSpreadDiagonal(Dimension{dimension}, 4.0);
    const auto op = MakeDenseOperator(kOccupied, kVirtual, matrix, diagonal);
    ASSERT_TRUE(op.has_value()) << op.error().message;

    const auto dense = AssembleDenseResponseMatrix(*op);
    ASSERT_TRUE(dense.has_value()) << dense.error().message;
    EXPECT_EQ(dense->dimension, dimension);
    ASSERT_EQ(dense->values.size(), dimension * dimension);

    for (int row = 0; row < size; ++row)
    {
        for (int column = 0; column < size; ++column)
        {
            EXPECT_DOUBLE_EQ(dense->values[static_cast<std::size_t>(row) * dimension +
                                           static_cast<std::size_t>(column)],
                             matrix(row, column));
        }
    }

    const auto sparse = AssembleSparseResponseMatrix(*op);
    ASSERT_TRUE(sparse.has_value()) << sparse.error().message;
    ASSERT_EQ(sparse->rowOffsets.size(), dimension + 1);
    EXPECT_EQ(sparse->rowOffsets.front(), 0u);
    EXPECT_EQ(sparse->rowOffsets.back(), sparse->values.size());
    ASSERT_EQ(sparse->columnIndices.size(), sparse->values.size());

    // Rebuild the dense matrix from the triple and compare it entry for
    // entry, including both sides of the diagonal.
    Eigen::MatrixXd rebuilt = Eigen::MatrixXd::Zero(size, size);

    for (std::size_t row = 0; row < dimension; ++row)
    {
        for (std::size_t entry = sparse->rowOffsets[row]; entry < sparse->rowOffsets[row + 1];
             ++entry)
        {
            rebuilt(static_cast<int>(row), static_cast<int>(sparse->columnIndices[entry])) =
                sparse->values[entry];
        }
    }

    EXPECT_EQ(sparse->values.size(), CountNonzeros(matrix));

    for (int row = 0; row < size; ++row)
    {
        for (int column = 0; column < size; ++column)
        {
            EXPECT_DOUBLE_EQ(rebuilt(row, column), matrix(row, column));
        }
    }
}

TEST(ResponseBandTest, SparseAssemblyDropsOnlyWhatTheToleranceSays) {
    constexpr std::size_t kOccupied = 2;
    constexpr std::size_t kVirtual = 4;
    const std::size_t dimension = kOccupied * kVirtual;
    const int size = static_cast<int>(dimension);

    // A mostly zero matrix with one entry far below a positive tolerance.
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(size, size);

    for (int i = 0; i < size; ++i)
    {
        matrix(i, i) = 1.0 + static_cast<double>(i);
    }

    matrix(0, 1) = 1e-12;
    const std::vector<double> diagonal = MakeSpreadDiagonal(Dimension{dimension}, 4.0);
    const auto op = MakeDenseOperator(kOccupied, kVirtual, matrix, diagonal);
    ASSERT_TRUE(op.has_value()) << op.error().message;

    // At the default tolerance only exact zeros are dropped: the 1e-12 entry
    // is a value the operator produced and it stays.
    const auto exact = AssembleSparseResponseMatrix(*op);
    ASSERT_TRUE(exact.has_value()) << exact.error().message;
    EXPECT_EQ(exact->values.size(), CountNonzeros(matrix));
    EXPECT_EQ(exact->values.size(), dimension + 1);

    // A positive tolerance drops it along with the exact zeros.
    ResponseMatrixOptions dropped;
    dropped.dropTolerance = 1e-9;
    const auto belowTolerance = AssembleSparseResponseMatrix(*op, dropped);
    ASSERT_TRUE(belowTolerance.has_value()) << belowTolerance.error().message;
    EXPECT_EQ(belowTolerance->values.size(), dimension);

    // A negative tolerance is a refusal, named rather than clamped to zero.
    ResponseMatrixOptions refused;
    refused.dropTolerance = -1.0;
    const auto negative = AssembleSparseResponseMatrix(*op, refused);
    ASSERT_FALSE(negative.has_value());
    EXPECT_EQ(negative.error().code, qcx::ErrorCode::kInvalidArgument);
}

// --- one physics, three bands ----------------------------------------------

TEST(ResponseBandTest, TheSameSmallGapPhysicsIsSolvedThroughAllThreeBands) {
    struct Case {
        std::size_t dimension;
        ResponseBandLimits limits;
        SolverBand expected;
        ResponseBandProblem problem;
        Seed seed;
    };

    ResponseBandLimits forcedMatrixFree;
    forcedMatrixFree.csrMaxNonzeros = 159'999;

    const std::vector<Case> cases = {
        {64,
         ResponseBandLimits{},
         SolverBand::kDense,
         ResponseBandProblem{64, std::size_t{64} * 64},
         Seed{1u}},
        {400,
         ResponseBandLimits{},
         SolverBand::kExplicitCsr,
         ResponseBandProblem{400, std::size_t{400} * 400},
         Seed{2u}},
        // The same 400-unknown problem, pushed off the explicit band by the
        // nonzero gate.
        {400,
         forcedMatrixFree,
         SolverBand::kMatrixFree,
         ResponseBandProblem{400, std::size_t{400} * 400},
         Seed{2u}},
    };

    for (const Case& testCase : cases)
    {
        const SmallGapFixture fixture = MakeSmallGapFixture(testCase.dimension);
        const auto op = MakeDenseOperator(fixture.layout.numOccupied,
                                          fixture.layout.numVirtual,
                                          fixture.matrix,
                                          fixture.preconditioner);
        ASSERT_TRUE(op.has_value()) << op.error().message;

        constexpr std::size_t kSides = 2;
        const std::vector<double> rhs =
            MakeRightHandSides(Dimension{testCase.dimension}, kSides, testCase.seed);

        ResponseSolveOptions options;
        options.maxIterations = 200;
        // The operator's own diagonal is about one, so a floor at half the
        // largest entry lands near it and recovers a usable preconditioner;
        // this is the fallback the small-gap fixture exists to exercise.
        options.regularisationRatio = 0.5;
        ResponseSolver<DenseResponseOperator> solver(options);

        const auto run = RunBand(*op, testCase.problem, kSides, rhs, solver, testCase.limits);
        ASSERT_TRUE(run.has_value())
            << run.error().message << " (band " << ToString(testCase.expected) << ")";

        EXPECT_EQ(run->report.band, testCase.expected);

        if (testCase.expected == SolverBand::kDense)
        {
            EXPECT_EQ(run->report.nonzeros, testCase.dimension * testCase.dimension);
            EXPECT_EQ(run->report.iterations, 0u);
        } else if (testCase.expected == SolverBand::kExplicitCsr)
        {
            EXPECT_EQ(run->report.nonzeros, CountNonzeros(fixture.matrix));
            EXPECT_GT(run->report.iterations, 0u);
        } else
        {
            EXPECT_EQ(run->report.nonzeros, 0u);
            EXPECT_GT(run->report.iterations, 0u);
            EXPECT_EQ(run->report.regularised, kSides);
        }

        // The small-gap signature is reported whichever band ran.
        EXPECT_NEAR(run->report.gapRatio, 1e-10, 1e-14);

        // The answer is an answer: the true residual recomputed from the
        // operator after the solve, and the solution against dense LDLT.
        EXPECT_LE(run->report.worstResidual, 1e-8)
            << "band " << ToString(testCase.expected) << " worst residual "
            << run->report.worstResidual;

        const double worst = WorstDifferenceFromLdlt(fixture, run->solution, kSides, rhs);
        EXPECT_LT(worst, 1e-6) << "band " << ToString(testCase.expected) << " differs from LDLT by "
                               << worst;

        std::cout << "[band] " << ToString(testCase.expected) << " dimension " << testCase.dimension
                  << ", " << run->report.nonzeros << " nonzeros, " << run->report.iterations
                  << " iterations, residual " << run->report.worstResidual << ", LDLT difference "
                  << worst << "\n";
    }
}

// --- warm starts stay on the matrix-free band ------------------------------

TEST(ResponseBandTest, TheExplicitBandsAreColdAndTheMatrixFreeBandReuses) {
    constexpr std::size_t kDimension = 400;
    constexpr std::size_t kSides = 2;
    const SmallGapFixture fixture = MakeSmallGapFixture(kDimension);
    const auto op = MakeDenseOperator(fixture.layout.numOccupied,
                                      fixture.layout.numVirtual,
                                      fixture.matrix,
                                      fixture.preconditioner);
    ASSERT_TRUE(op.has_value()) << op.error().message;

    const std::vector<double> rhs =
        MakeRightHandSides(Dimension{kDimension}, kSides, Seed{20261019u});
    const ResponseBandProblem problem{kDimension, kDimension * kDimension};
    ResponseBandLimits forcedMatrixFree;
    forcedMatrixFree.csrMaxNonzeros = 159'999;

    ResponseSolveOptions options;
    options.maxIterations = 200;
    options.regularisationRatio = 0.5;
    ResponseSolver<DenseResponseOperator> solver(options);

    const auto cold = RunBand(*op, problem, kSides, rhs, solver, forcedMatrixFree);
    ASSERT_TRUE(cold.has_value()) << cold.error().message;
    EXPECT_EQ(cold->report.band, SolverBand::kMatrixFree);
    EXPECT_EQ(cold->report.warmStarted, 0u);

    const auto warm = RunBand(*op, problem, kSides, rhs, solver, forcedMatrixFree);
    ASSERT_TRUE(warm.has_value()) << warm.error().message;
    EXPECT_EQ(warm->report.band, SolverBand::kMatrixFree);
    EXPECT_EQ(warm->report.warmStarted, kSides);

    // The same solver now holds a previous solution for both ordinals, and
    // the explicit band still reports a cold solve: that is the design, not
    // an accident of empty state.
    const auto explicitRun = RunBand(*op, problem, kSides, rhs, solver, ResponseBandLimits{});
    ASSERT_TRUE(explicitRun.has_value()) << explicitRun.error().message;
    EXPECT_EQ(explicitRun->report.band, SolverBand::kExplicitCsr);
    EXPECT_EQ(explicitRun->report.warmStarted, 0u);
}

// --- refusals ---------------------------------------------------------------

TEST(ResponseBandTest, TheDenseBandRefusesASingularOperator) {
    constexpr std::size_t kOccupied = 2;
    constexpr std::size_t kVirtual = 2;
    const std::size_t dimension = kOccupied * kVirtual;
    const int size = static_cast<int>(dimension);

    Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(size, size);
    // A zero row is exactly singular: the LU factorisation sees a zero pivot.
    matrix.row(0).setZero();
    const std::vector<double> diagonal(dimension, 1.0);
    const auto op = MakeDenseOperator(kOccupied, kVirtual, matrix, diagonal);
    ASSERT_TRUE(op.has_value()) << op.error().message;

    const std::vector<double> rhs = MakeRightHandSides(Dimension{dimension}, 1, Seed{3u});
    ResponseSolver<DenseResponseOperator> solver;
    const auto run = RunBand(*op,
                             ResponseBandProblem{dimension, dimension * dimension},
                             1,
                             rhs,
                             solver,
                             ResponseBandLimits{});
    ASSERT_FALSE(run.has_value());
    EXPECT_EQ(run.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(run.error().message.find("singular"), std::string::npos) << run.error().message;
}

TEST(ResponseBandTest, TheBandSolveRefusesMismatchedShapes) {
    constexpr std::size_t kDimension = 16;
    const SmallGapFixture fixture = MakeSmallGapFixture(kDimension);
    const auto op = MakeDenseOperator(fixture.layout.numOccupied,
                                      fixture.layout.numVirtual,
                                      fixture.matrix,
                                      fixture.preconditioner);
    ASSERT_TRUE(op.has_value()) << op.error().message;

    const std::vector<double> rhs = MakeRightHandSides(Dimension{kDimension}, 2, Seed{4u});
    ResponseSolver<DenseResponseOperator> solver;
    std::vector<double> solution(2 * kDimension);

    const auto noSides = qcx::response::SolveResponseByBand(
        *op, ResponseBandProblem{kDimension, 0}, 0, rhs, solution, solver, ResponseBandLimits{});
    ASSERT_FALSE(noSides.has_value());
    EXPECT_EQ(noSides.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto wrongProblem =
        qcx::response::SolveResponseByBand(*op,
                                           ResponseBandProblem{kDimension + 1, 0},
                                           2,
                                           rhs,
                                           solution,
                                           solver,
                                           ResponseBandLimits{});
    ASSERT_FALSE(wrongProblem.has_value());
    EXPECT_EQ(wrongProblem.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto shortRhs = qcx::response::SolveResponseByBand(
        *op, ResponseBandProblem{kDimension, 0}, 3, rhs, solution, solver, ResponseBandLimits{});
    ASSERT_FALSE(shortRhs.has_value());
    EXPECT_EQ(shortRhs.error().code, qcx::ErrorCode::kInvalidArgument);

    std::vector<double> shortSolution(kDimension);
    const auto shortOutput = qcx::response::SolveResponseByBand(*op,
                                                                ResponseBandProblem{kDimension, 0},
                                                                2,
                                                                rhs,
                                                                shortSolution,
                                                                solver,
                                                                ResponseBandLimits{});
    ASSERT_FALSE(shortOutput.has_value());
    EXPECT_EQ(shortOutput.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(ResponseBandTest, BandNamesAreStable) {
    EXPECT_EQ(ToString(SolverBand::kDense), "dense");
    EXPECT_EQ(ToString(SolverBand::kExplicitCsr), "explicit-csr");
    EXPECT_EQ(ToString(SolverBand::kMatrixFree), "matrix-free");
}
