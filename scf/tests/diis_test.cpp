// CDIIS tests: the extrapolator's two-pair closed form, its deterministic
// degenerate-history behavior, the history cap, and the full H2/STO-3G
// pipeline converging to the same pinned energy with the default
// (DIIS-on) options as with plain Roothaan (rhf_test.cpp). The joint-system
// form gets the same treatment through the internal seam
// (scf_common.hpp): its coupled two-pair closed form, degenerate history,
// eviction, checkpoint round-trip, restore validation, and the
// floor-degenerate-pair veto of MaybeExtrapolateJointFock. The exposure
// contract of ExtrapolateWithCoefficients is verified against
// hand-computed solves, and the monitor-only coefficient measures of the
// trace increment (MeasureDiisCoefficients: negMass/parityImb) get
// the truth-table treatment plus their trace-line wiring.
#include "h2_sto3g.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/scf/diis.hpp"
#include "qcx/scf/rhf.hpp"
#include "scf_common.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>

namespace {

using qcx::scf::internal::JointSystemDiis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

TEST(DiisExtrapolatorTest, ExtrapolateBeforeReadyIsRejected) {
    // One stored pair cannot define an extrapolation: the call must report
    // the misuse instead of reading an empty history.
    qcx::scf::DiisExtrapolator diis(8);
    diis.Append(Eigen::MatrixXd::Identity(2, 2), Eigen::MatrixXd::Ones(2, 2));
    const auto result = diis.Extrapolate();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DiisExtrapolatorTest, TwoPairClosedForm) {
    // Hand-computed closed form for two pairs: with errors e0 = I and
    // e1 = 2I + J (J the (0,1)/(1,0) swap), the constrained minimum of
    // ||c0 e0 + c1 e1|| with c0 + c1 = 1 sits at
    // c0 = <e1, e1 - e0> / ||e0 - e1||^2 = 6 / 4 = 1.5, c1 = -0.5.
    // With f0 = I and f1 = 3I the extrapolated matrix is 1.5 I - 0.5*3I = 0.
    qcx::scf::DiisExtrapolator diis(8);
    const Eigen::MatrixXd f0 = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd f1 = 3.0 * Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd e0 = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd e1 = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    e1(0, 1) = 1.0;
    e1(1, 0) = 1.0;

    diis.Append(f0, e0);
    EXPECT_FALSE(diis.Ready()); // one pair does not extrapolate
    diis.Append(f1, e1);
    EXPECT_TRUE(diis.Ready());

    const auto result = diis.Extrapolate();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NEAR(result->norm(), 0.0, 1e-12);
}

TEST(DiisExtrapolatorTest, DegenerateHistoryResolvesToEqualWeights) {
    // Two identical error vectors leave the B-system singular: the
    // minimum-norm least-squares solution is the equal-weight combination,
    // so a degenerate history extrapolates deterministically instead of
    // arbitrarily.
    qcx::scf::DiisExtrapolator diis(8);
    const Eigen::MatrixXd f0 = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd f1 = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd error = Eigen::MatrixXd::Ones(2, 2);

    diis.Append(f0, error);
    diis.Append(f1, error);
    const auto result = diis.Extrapolate();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NEAR((*result - 0.5 * (f0 + f1)).norm(), 0.0, 1e-12);
}

TEST(DiisExtrapolatorTest, HistoryIsCappedAtTheLimit) {
    // A capped extrapolator must behave exactly like one fed only the last
    // historyLimit pairs - eviction, not retention with truncation.
    const auto f = [](int i) { return Eigen::MatrixXd::Constant(2, 2, static_cast<double>(i)); };
    // Four fixed basis patterns: three generic combinations stay linearly
    // independent in the 4-dimensional space of 2x2 matrices.
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(2, 2);
    a(0, 1) = 1.0;
    b(1, 0) = 1.0;
    c(0, 0) = 1.0;
    d(1, 1) = 1.0;
    const auto e = [&](int i) {
        return static_cast<double>(i + 1) * a + static_cast<double>(2 * i + 3) * b +
               static_cast<double>(i * i + 1) * c + static_cast<double>(i * i * i + 1) * d;
    };

    qcx::scf::DiisExtrapolator capped(3);
    qcx::scf::DiisExtrapolator reference(3);

    for (int i = 0; i < 6; ++i)
    {
        capped.Append(f(i), e(i));

        if (i >= 3)
        {
            reference.Append(f(i), e(i));
        }
    }

    const auto cappedResult = capped.Extrapolate();
    const auto referenceResult = reference.Extrapolate();
    ASSERT_TRUE(cappedResult.has_value()) << cappedResult.error().message;
    ASSERT_TRUE(referenceResult.has_value()) << referenceResult.error().message;
    EXPECT_NEAR((*cappedResult - *referenceResult).norm(), 0.0, 1e-12);

    capped.Reset();
    EXPECT_FALSE(capped.Ready());
}

TEST(DiisExtrapolatorTest, ExtrapolateWithCoefficientsMatchesTheTwoPairClosedForm) {
    // The coefficient-exposure contract: the returned weights are
    // the COD solve's actual Fock combination coefficients - hand-computed
    // for the TwoPairClosedForm case, c = (1.5, -0.5) with a zero
    // extrapolated matrix - and the plain Extrapolate() overload keeps
    // working, returning the same matrix (the additive rule).
    qcx::scf::DiisExtrapolator diis(8);
    const Eigen::MatrixXd f0 = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd f1 = 3.0 * Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd e0 = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd e1 = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    e1(0, 1) = 1.0;
    e1(1, 0) = 1.0;

    diis.Append(f0, e0);
    diis.Append(f1, e1);

    const auto result = diis.ExtrapolateWithCoefficients();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_EQ(result->second.size(), Eigen::Index{2});
    EXPECT_NEAR(result->second(0), 1.5, 1e-12);
    EXPECT_NEAR(result->second(1), -0.5, 1e-12);
    EXPECT_NEAR(result->first.norm(), 0.0, 1e-12);

    const auto plain = diis.Extrapolate();
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_NEAR((*plain - result->first).norm(), 0.0, 1e-12);
}

TEST(DiisExtrapolatorTest, ExtrapolateWithCoefficientsMatchesTheThreePairOrthogonalSolve) {
    // Hand-computed three-pair solve: the errors e0 = diag(1, 0),
    // e1 = diag(0, 2) and e2 = J (the (0,1)/(1,0) swap) are mutually
    // orthogonal, so the B-system is diag(<e_i, e_i>) = diag(1, 4, 2) under
    // the unit-sum constraint and c_i = lambda / d_i with
    // lambda = 1 / sum(1 / d_i) = 4/7: c = (4/7, 1/7, 2/7). With
    // f_i = (i + 1) I the extrapolated matrix is (4/7 + 2/7 + 6/7) I =
    // (12/7) I.
    qcx::scf::DiisExtrapolator diis(8);
    Eigen::MatrixXd e0 = Eigen::MatrixXd::Zero(2, 2);
    e0(0, 0) = 1.0;
    Eigen::MatrixXd e1 = Eigen::MatrixXd::Zero(2, 2);
    e1(1, 1) = 2.0;
    Eigen::MatrixXd e2 = Eigen::MatrixXd::Zero(2, 2);
    e2(0, 1) = 1.0;
    e2(1, 0) = 1.0;
    const Eigen::MatrixXd f0 = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd f1 = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd f2 = 3.0 * Eigen::MatrixXd::Identity(2, 2);

    diis.Append(f0, e0);
    diis.Append(f1, e1);
    diis.Append(f2, e2);

    const auto result = diis.ExtrapolateWithCoefficients();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_EQ(result->second.size(), Eigen::Index{3});
    EXPECT_NEAR(result->second(0), 4.0 / 7.0, 1e-12);
    EXPECT_NEAR(result->second(1), 1.0 / 7.0, 1e-12);
    EXPECT_NEAR(result->second(2), 2.0 / 7.0, 1e-12);
    EXPECT_NEAR(result->second.sum(), 1.0, 1e-12);
    EXPECT_NEAR(
        (result->first - (12.0 / 7.0) * Eigen::MatrixXd::Identity(2, 2)).norm(), 0.0, 1e-12);
}

TEST(RhfTest, H2Sto3gDiisMatchesPlainRoothaan) {
    const auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;

    // Default options: CDIIS on - the production path.
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Same pinned reference as the plain-Roothaan test (rhf_test.cpp): DIIS
    // accelerates convergence, it must not change the converged answer.
    EXPECT_NEAR(result->totalEnergy, -1.1167143252, 1e-8);
    EXPECT_TRUE(result->converged);
    EXPECT_LT(result->iterations, 10);
}

TEST(DiisTest, SnapshotRestoreRoundTrips) {
    // The checkpointing contract: a snapshot is value semantics -
    // a fresh extrapolator restored from it extrapolates bit-identically to
    // the original.
    qcx::scf::DiisExtrapolator original(8);
    qcx::scf::DiisExtrapolator restored(8);

    for (int i = 0; i < 3; ++i)
    {
        original.Append(Eigen::MatrixXd::Constant(2, 2, static_cast<double>(i + 1)),
                        Eigen::MatrixXd::Ones(2, 2) * static_cast<double>(2 * i + 1));
    }

    const qcx::scf::DiisState snapshot = original.Snapshot();
    ASSERT_EQ(snapshot.fockHistory.size(), std::size_t{3});
    ASSERT_EQ(snapshot.errorHistory.size(), std::size_t{3});

    const auto restoredResult = restored.Restore(snapshot);
    ASSERT_TRUE(restoredResult.has_value()) << restoredResult.error().message;
    EXPECT_TRUE(restored.Ready());

    const auto originalExtrapolation = original.Extrapolate();
    const auto restoredExtrapolation = restored.Extrapolate();
    ASSERT_TRUE(originalExtrapolation.has_value()) << originalExtrapolation.error().message;
    ASSERT_TRUE(restoredExtrapolation.has_value()) << restoredExtrapolation.error().message;
    EXPECT_EQ(std::memcmp(restoredExtrapolation->data(),
                          originalExtrapolation->data(),
                          originalExtrapolation->size() * sizeof(double)),
              0);

    // An empty state is a reset (no-op): restoring into a fresh
    // extrapolator leaves it not ready, default-constructed options behave
    // exactly as before.
    qcx::scf::DiisExtrapolator cleared(8);
    const auto clearedResult = cleared.Restore(qcx::scf::DiisState{});
    ASSERT_TRUE(clearedResult.has_value()) << clearedResult.error().message;
    EXPECT_FALSE(cleared.Ready());
}

TEST(DiisTest, PhaseHandoffPreservesExtrapolation) {
    // The two-phase handoff: a short-window extrapolator's history
    // is snapshotted and restored into a full-window one mid-run. The
    // restored history must extrapolate exactly like the same history fed
    // to the full-window extrapolator directly - the handoff is gap-free
    // (no entry lost or duplicated).
    const auto f = [](int i) { return Eigen::MatrixXd::Constant(2, 2, static_cast<double>(i)); };
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(2, 2);
    a(0, 1) = 1.0;
    b(1, 0) = 1.0;
    c(0, 0) = 1.0;
    d(1, 1) = 1.0;
    const auto e = [&](int i) {
        return static_cast<double>(i + 1) * a + static_cast<double>(2 * i + 3) * b +
               static_cast<double>(i * i + 1) * c + static_cast<double>(i * i * i + 1) * d;
    };

    // (a) The handoff path: 3 pairs in the 2-limit phase-1 window (the
    // first is evicted, the history holds pairs 1..2), snapshot, restore
    // into an 8-limit extrapolator, then continue with pairs 3..7.
    qcx::scf::DiisExtrapolator phaseOne(2);

    for (int i = 0; i < 3; ++i)
    {
        phaseOne.Append(f(i), e(i));
    }

    qcx::scf::DiisExtrapolator restored(8);
    const auto restoreResult = restored.Restore(phaseOne.Snapshot());
    ASSERT_TRUE(restoreResult.has_value()) << restoreResult.error().message;

    for (int i = 3; i < 8; ++i)
    {
        restored.Append(f(i), e(i));
    }

    // (b) The reference: the same tail fed straight to the 8-limit
    // extrapolator - the history contents are identical to (a)'s.
    qcx::scf::DiisExtrapolator reference(8);

    for (int i = 1; i < 8; ++i)
    {
        reference.Append(f(i), e(i));
    }

    const auto restoredResult = restored.Extrapolate();
    const auto referenceResult = reference.Extrapolate();
    ASSERT_TRUE(restoredResult.has_value()) << restoredResult.error().message;
    ASSERT_TRUE(referenceResult.has_value()) << referenceResult.error().message;
    EXPECT_NEAR((*restoredResult - *referenceResult).norm(), 0.0, 1e-12);
}

TEST(DiisTest, RestoreValidatesShapeAndLimit) {
    // A restore that would exceed the history limit is refused, as are
    // non-square or shape-mismatched pairs - the extrapolator must never
    // extrapolate from a corrupted history.
    qcx::scf::DiisExtrapolator capped(2);
    qcx::scf::DiisState tooLarge;
    tooLarge.fockHistory = {
        Eigen::MatrixXd::Identity(2, 2), Eigen::MatrixXd::Ones(2, 2), Eigen::MatrixXd::Zero(2, 2)};
    tooLarge.errorHistory = tooLarge.fockHistory;
    const auto limitResult = capped.Restore(tooLarge);
    ASSERT_FALSE(limitResult.has_value());
    EXPECT_EQ(limitResult.error().code, qcx::ErrorCode::kInvalidArgument);

    qcx::scf::DiisExtrapolator diis(8);
    qcx::scf::DiisState nonSquare;
    nonSquare.fockHistory = {Eigen::MatrixXd::Identity(2, 3)};
    nonSquare.errorHistory = {Eigen::MatrixXd::Identity(2, 3)};
    const auto shapeResult = diis.Restore(nonSquare);
    ASSERT_FALSE(shapeResult.has_value());
    EXPECT_EQ(shapeResult.error().code, qcx::ErrorCode::kInvalidArgument);

    qcx::scf::DiisState mismatched;
    mismatched.fockHistory = {Eigen::MatrixXd::Identity(2, 2)};
    mismatched.errorHistory = {Eigen::MatrixXd::Identity(3, 3)};
    const auto mismatchResult = diis.Restore(mismatched);
    ASSERT_FALSE(mismatchResult.has_value());
    EXPECT_EQ(mismatchResult.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DiisTest, JointSystemTwoPairClosedForm) {
    // Hand-computed closed form for the joint B-system: the
    // B-matrix is 0.5·(<eA_i,eA_j>+<eB_i,eB_j>) with one unit-sum
    // constraint, so the two-pair coefficient follows the per-spin formula
    // with the joint inner products. With eA0 = I, eA1 = 2I + J (J the
    // (0,1)/(1,0) swap), eB0 = 2I, eB1 = 4I the joint entries are
    // b00 = 0.5(2+8) = 5, b01 = 0.5(4+16) = 10, b11 = 0.5(10+32) = 21,
    // c0 = 1 - c1 with c1 = (b00 - b01)/(b00 - 2 b01 + b11) = -5/6, so
    // c0 = 11/6. With fA0 = fB0 = I and fA1 = fB1 = 3I both spins
    // extrapolate to (11/6)I - (5/6)(3I) = -(2/3)I - a coupling the
    // per-spin solves would not produce (alpha alone: 0, beta alone: -I).
    JointSystemDiis diis(8);
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd& eA0 = identity;
    Eigen::MatrixXd eA1 = 2.0 * identity;
    eA1(0, 1) = 1.0;
    eA1(1, 0) = 1.0;
    const Eigen::MatrixXd eB0 = 2.0 * identity;
    const Eigen::MatrixXd eB1 = 4.0 * identity;

    diis.Append(identity, eA0, identity, eB0);
    EXPECT_FALSE(diis.Ready()); // one pair does not extrapolate
    diis.Append(3.0 * identity, eA1, 3.0 * identity, eB1);
    EXPECT_TRUE(diis.Ready());

    const auto result = diis.Extrapolate();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NEAR((result->first + (2.0 / 3.0) * identity).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result->second + (2.0 / 3.0) * identity).norm(), 0.0, 1e-12);
}

TEST(DiisTest, JointSystemDegenerateHistoryResolvesToEqualWeights) {
    // All four stored errors identical: the joint B-system is singular and
    // the minimum-norm least-squares solution is the equal-weight
    // combination - a degenerate joint history extrapolates
    // deterministically, exactly like the per-spin one.
    JointSystemDiis diis(8);
    const Eigen::MatrixXd f0 = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd f1 = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd error = Eigen::MatrixXd::Ones(2, 2);

    diis.Append(f0, error, f0, error);
    diis.Append(f1, error, f1, error);
    const auto result = diis.Extrapolate();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NEAR((result->first - 0.5 * (f0 + f1)).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result->second - 0.5 * (f0 + f1)).norm(), 0.0, 1e-12);
}

TEST(DiisTest, JointSystemHistoryIsCappedAtTheLimit) {
    // Eviction, not retention with truncation: with limit 3 and four
    // appended pairs of IDENTICAL errors, the equal-weight minimum-norm
    // solution must weight only the surviving pairs 1..3 (extrapolated
    // (1+2+3)/3 I = 2I) - the retained pair-0 fock (0I) would pull the
    // four-way equal-weight combination down to (0+1+2+3)/4 I = 1.5I.
    JointSystemDiis diis(3);

    for (int i = 0; i < 4; ++i)
    {
        const Eigen::MatrixXd fock = static_cast<double>(i) * Eigen::MatrixXd::Identity(2, 2);
        diis.Append(fock, Eigen::MatrixXd::Ones(2, 2), fock, Eigen::MatrixXd::Ones(2, 2));
    }

    const auto result = diis.Extrapolate();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NEAR((result->first - 2.0 * Eigen::MatrixXd::Identity(2, 2)).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result->second - 2.0 * Eigen::MatrixXd::Identity(2, 2)).norm(), 0.0, 1e-12);
}

TEST(DiisTest, JointSystemSnapshotRestoreRoundTrips) {
    // The joint checkpointing contract (two-field shape): the
    // snapshot is the per-spin DiisState pair, and a fresh joint
    // extrapolator restored from it extrapolates bit-identically to the
    // original. An empty pair is a reset (no-op).
    JointSystemDiis original(8);
    JointSystemDiis restored(8);

    for (int i = 0; i < 3; ++i)
    {
        original.Append(Eigen::MatrixXd::Constant(2, 2, static_cast<double>(i + 1)),
                        Eigen::MatrixXd::Ones(2, 2) * static_cast<double>(2 * i + 1),
                        Eigen::MatrixXd::Constant(2, 2, static_cast<double>(3 * i + 2)),
                        Eigen::MatrixXd::Ones(2, 2) * static_cast<double>(i + 2));
    }

    const auto snapshot = original.Snapshot();
    ASSERT_EQ(snapshot.first.fockHistory.size(), std::size_t{3});
    ASSERT_EQ(snapshot.second.fockHistory.size(), std::size_t{3});

    const auto restoreResult = restored.Restore(snapshot.first, snapshot.second);
    ASSERT_TRUE(restoreResult.has_value()) << restoreResult.error().message;
    EXPECT_TRUE(restored.Ready());

    const auto originalExtrapolation = original.Extrapolate();
    const auto restoredExtrapolation = restored.Extrapolate();
    ASSERT_TRUE(originalExtrapolation.has_value()) << originalExtrapolation.error().message;
    ASSERT_TRUE(restoredExtrapolation.has_value()) << restoredExtrapolation.error().message;
    EXPECT_EQ(std::memcmp(restoredExtrapolation->first.data(),
                          originalExtrapolation->first.data(),
                          originalExtrapolation->first.size() * sizeof(double)),
              0);
    EXPECT_EQ(std::memcmp(restoredExtrapolation->second.data(),
                          originalExtrapolation->second.data(),
                          originalExtrapolation->second.size() * sizeof(double)),
              0);

    JointSystemDiis cleared(8);
    const auto clearedResult = cleared.Restore(qcx::scf::DiisState{}, qcx::scf::DiisState{});
    ASSERT_TRUE(clearedResult.has_value()) << clearedResult.error().message;
    EXPECT_FALSE(cleared.Ready());
}

TEST(DiisTest, JointSystemRestoreValidatesShapeAndLimit) {
    // A joint restore that would exceed the history limit is refused, as
    // are non-square, per-spin-mismatched, count-mismatched, and
    // cross-spin-shape-mismatched pairs - one coefficient vector over the
    // combined subspace cannot extrapolate a corrupted or asymmetric
    // history.
    JointSystemDiis capped(2);
    qcx::scf::DiisState tooLarge;
    tooLarge.fockHistory = {
        Eigen::MatrixXd::Identity(2, 2), Eigen::MatrixXd::Ones(2, 2), Eigen::MatrixXd::Zero(2, 2)};
    tooLarge.errorHistory = tooLarge.fockHistory;
    const auto limitResult = capped.Restore(tooLarge, tooLarge);
    ASSERT_FALSE(limitResult.has_value());
    EXPECT_EQ(limitResult.error().code, qcx::ErrorCode::kInvalidArgument);

    JointSystemDiis diis(8);
    qcx::scf::DiisState nonSquare;
    nonSquare.fockHistory = {Eigen::MatrixXd::Identity(2, 3)};
    nonSquare.errorHistory = {Eigen::MatrixXd::Identity(2, 3)};
    const auto shapeResult = diis.Restore(nonSquare, nonSquare);
    ASSERT_FALSE(shapeResult.has_value());
    EXPECT_EQ(shapeResult.error().code, qcx::ErrorCode::kInvalidArgument);

    qcx::scf::DiisState mismatched;
    mismatched.fockHistory = {Eigen::MatrixXd::Identity(2, 2)};
    mismatched.errorHistory = {Eigen::MatrixXd::Identity(3, 3)};
    const auto mismatchResult = diis.Restore(mismatched, mismatched);
    ASSERT_FALSE(mismatchResult.has_value());
    EXPECT_EQ(mismatchResult.error().code, qcx::ErrorCode::kInvalidArgument);

    // Count mismatch across the spins: a joint history entry is a pair.
    qcx::scf::DiisState twoEntries;
    twoEntries.fockHistory = {Eigen::MatrixXd::Identity(2, 2), Eigen::MatrixXd::Ones(2, 2)};
    twoEntries.errorHistory = twoEntries.fockHistory;
    const auto countResult = diis.Restore(twoEntries, mismatched);
    ASSERT_FALSE(countResult.has_value());
    EXPECT_EQ(countResult.error().code, qcx::ErrorCode::kInvalidArgument);

    // Cross-spin shape mismatch (each per-spin pair consistent): the two
    // spins share one coefficient vector, so their matrices must agree on
    // the shape.
    qcx::scf::DiisState alphaState;
    alphaState.fockHistory = {Eigen::MatrixXd::Identity(2, 2)};
    alphaState.errorHistory = {Eigen::MatrixXd::Ones(2, 2)};
    qcx::scf::DiisState betaState;
    betaState.fockHistory = {Eigen::MatrixXd::Identity(3, 3)};
    betaState.errorHistory = {Eigen::MatrixXd::Ones(3, 3)};
    const auto crossSpinResult = diis.Restore(alphaState, betaState);
    ASSERT_FALSE(crossSpinResult.has_value());
    EXPECT_EQ(crossSpinResult.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DiisTest, JointMaybeExtrapolateVetoesFloorDegeneratePairs) {
    // The joint floor guard: EITHER spin's floor-degenerate error
    // drops the whole pair - a degenerate spin's zero row/column in the
    // joint B-matrix would make the constraint system degenerate, and the
    // pair is the joint history entry. A vetoed call returns the Focks
    // unchanged and appends nothing; with useDiis off the call is a pure
    // pass-through.
    qcx::scf::UhfOptions options;
    options.useDiis = true;
    JointSystemDiis diis(8);
    const Eigen::MatrixXd fA0 = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd fB0 = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    const Eigen::MatrixXd eA0 = Eigen::MatrixXd::Ones(2, 2);
    const Eigen::MatrixXd eB0 = 2.0 * Eigen::MatrixXd::Ones(2, 2);

    const auto vetoedAlpha =
        qcx::scf::internal::MaybeExtrapolateJointFock(options, diis, fA0, eA0, 0.0, fB0, eB0, 5.0);
    ASSERT_TRUE(vetoedAlpha.has_value()) << vetoedAlpha.error().message;
    EXPECT_NEAR((vetoedAlpha->first - fA0).norm(), 0.0, 1e-15);
    EXPECT_NEAR((vetoedAlpha->second - fB0).norm(), 0.0, 1e-15);

    const auto vetoedBeta =
        qcx::scf::internal::MaybeExtrapolateJointFock(options, diis, fA0, eA0, 5.0, fB0, eB0, 0.0);
    ASSERT_TRUE(vetoedBeta.has_value()) << vetoedBeta.error().message;
    EXPECT_NEAR((vetoedBeta->first - fA0).norm(), 0.0, 1e-15);
    EXPECT_NEAR((vetoedBeta->second - fB0).norm(), 0.0, 1e-15);

    // Two valid appends follow the two vetoes: if a vetoed pair had been
    // stored, the first valid call would already be ready.
    const auto first =
        qcx::scf::internal::MaybeExtrapolateJointFock(options, diis, fA0, eA0, 5.0, fB0, eB0, 5.0);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    EXPECT_FALSE(diis.Ready());
    const auto second =
        qcx::scf::internal::MaybeExtrapolateJointFock(options, diis, fA0, eA0, 5.0, fB0, eB0, 5.0);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    EXPECT_TRUE(diis.Ready());

    // useDiis off: pure pass-through, nothing stored.
    qcx::scf::UhfOptions plainOptions;
    JointSystemDiis plainDiis(8);
    const auto passthrough = qcx::scf::internal::MaybeExtrapolateJointFock(
        plainOptions, plainDiis, fA0, eA0, 5.0, fB0, eB0, 5.0);
    ASSERT_TRUE(passthrough.has_value()) << passthrough.error().message;
    EXPECT_NEAR((passthrough->first - fA0).norm(), 0.0, 1e-15);
    EXPECT_NEAR((passthrough->second - fB0).norm(), 0.0, 1e-15);
    EXPECT_FALSE(plainDiis.Ready());
}

TEST(DiisTest, CoefficientMeasuresTruthTable) {
    // The monitor-only measures as pure helpers: hand-computed rows
    // over healthy synthetic coefficient vectors. Under the unit-sum
    // constraint (sum c_i = 1) negativeMass = (sum|c_i| - 1)/2 equals the
    // total magnitude of the negative weights - the signed-weight identity - and
    // parityImbalance splits the even/odd 0-based history positions.
    using qcx::scf::internal::MeasureDiisCoefficients;
    const auto expect = [](std::initializer_list<double> coefficients,
                           double negativeMass,
                           double parityImbalance) {
        const auto measures = MeasureDiisCoefficients(std::vector<double>(coefficients));
        EXPECT_NEAR(measures.negativeMass, negativeMass, 1e-12);
        EXPECT_NEAR(measures.parityImbalance, parityImbalance, 1e-12);
    };

    expect({0.5, 0.5}, 0.0, 0.0); // equal split: no negatives, balanced parity
    expect({0.25, 0.75}, 0.0, 0.5); // all positive, the weight on the odd slot
    expect({1.0}, 0.0, 1.0); // one slot: the whole mass on the even position
    expect({1.5, -0.5}, 0.5, 1.0); // the two-pair closed form: one negative weight
    expect({-0.25, 0.5, 0.75}, 0.25, 0.0); // exact unit sum: negMass is the negative total
    expect({4.0 / 7.0, 1.0 / 7.0, 2.0 / 7.0}, 0.0, 5.0 / 7.0); // the three-pair solve
    expect({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
           0.0,
           1.0); // the self-flush shape: all mass on the newest slot
    expect({}, 0.0, 0.0); // empty: nothing to measure
    expect({0.0, 0.0}, 0.0, 0.0); // all-zero: the degenerate guard
}

TEST(DiisTest, CoefficientMeasuresAreReversalInvariant) {
    // The ordering note: the trace's diiscoef prints oldest-first
    // (diis.cpp Append = push_back + begin()-eviction), and the analysis
    // leaned on the measures not depending on it - a reversed history must
    // give the same measures, so a future reorder cannot silently change
    // the logged population.
    using qcx::scf::internal::MeasureDiisCoefficients;
    const std::vector<double> forward{1.5, -0.5, 0.25, -0.25, 1.0};
    const std::vector<double> reversed(forward.rbegin(), forward.rend());
    const auto forwardMeasures = MeasureDiisCoefficients(forward);
    const auto reversedMeasures = MeasureDiisCoefficients(reversed);
    EXPECT_NEAR(forwardMeasures.negativeMass, reversedMeasures.negativeMass, 1e-15);
    EXPECT_NEAR(forwardMeasures.parityImbalance, reversedMeasures.parityImbalance, 1e-15);
}

TEST(DiisTest, TraceLineLogsTheMonitorMeasuresWhenExtrapolated) {
    // The trace wiring: an extrapolated (diis=ext) line carries
    // negMass/parityImb derived from the printed coefficients - for the
    // two-pair solve (1.5, -0.5): negMass = (2 - 1)/2 = 0.5 and
    // parityImb = |1.5 + 0.5| / 2 = 1 - while a non-extrapolated status
    // keeps the diiscoef-less shape (nothing was measured, nothing is
    // printed). Direct ScfTraceWriter exercise: the real H2/STO-3G pipeline
    // never extrapolates (its commutator error sits at the floor from the
    // first iteration).
    const std::filesystem::path tracePath =
        std::filesystem::temp_directory_path() / "qcx_scf_diis_measures_trace.log";
    std::filesystem::remove(tracePath);

    qcx::scf::internal::ScfTraceWriter writer(tracePath.string());
    qcx::scf::internal::ScfDiisTraceStatus extrapolated;
    extrapolated.useDiis = true;
    extrapolated.extrapolated = true;
    extrapolated.errorNorm = 0.5;
    extrapolated.coefficients = {1.5, -0.5};
    writer.Write(3, -1.0, 1e-3, 1e-4, false, extrapolated);

    qcx::scf::internal::ScfDiisTraceStatus waiting;
    waiting.useDiis = true;
    waiting.errorNorm = 0.7;
    writer.Write(4, -0.99, 1e-3, 1e-4, false, waiting);

    std::ifstream trace(tracePath);
    std::string extLine;
    std::string waitLine;
    std::getline(trace, extLine);
    std::getline(trace, waitLine);

    EXPECT_NE(extLine.find("diis=ext"), std::string::npos);
    EXPECT_NE(extLine.find(" diiscoef=1.5,-0.5"), std::string::npos);
    EXPECT_NE(extLine.find(" negMass=0.5"), std::string::npos);
    EXPECT_NE(extLine.find(" parityImb=1"), std::string::npos);
    EXPECT_NE(waitLine.find("diis=wait"), std::string::npos);
    EXPECT_EQ(waitLine.find(" negMass="), std::string::npos);
    EXPECT_EQ(waitLine.find(" parityImb="), std::string::npos);
}

} // namespace
