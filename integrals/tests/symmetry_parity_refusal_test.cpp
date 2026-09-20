// The symmetry-parity refusal and the RI-K counter sink (2026-09-15). The
// state audit found two governed paths with no
// point-group reduction: ri_k (whose builder RECEIVED RiEngineOptions and
// silently ignored the reduction fields) and qfmm (whose option set could not
// carry one). The RI-K builder now REFUSES a requested reduction by name (the
// refusal contract: honoured, refused with its condition named, or demoted with
// the disclosure - never silently substituted) and carries the counters a
// later increment is sized from. The refusal fires before any
// tensor work, so the controls - the same call with the fields cleared - are
// the only expensive legs here.
//
// The qfmm leg is NOT here: `qfmm_fock_build.*` was still changing under a
// concurrent edit (2026-09-15 00:51), so its refusal - the
// same shape, refused with the measured counts - is parked as
// a patch instead of being committed into another working set.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/ri_full_fock.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qfmm_fixture.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <string_view>

namespace {

using qcx::integrals::RiFullFockBuilder;
using qcx::integrals::RiTermCounters;
using qcx::integrals::SymmetryReduction;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;

// The aux fit the RI-K legs use: the fixture basis directory the other RI
// tests parse, filtered to water's two elements.
qcx::Result<qcx::basisset::BasisSet> ParseFixtureBasis(std::string_view directory) {
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 2> elements{1, 8};
    return qcx::basisset::ParseNwchemDirectoryFiltered((root / directory).string(), elements);
}

// A reduction is never VALIDATED by these tests - both refusals fire on the
// POINTER (a request), before any group is consulted - so a
// default-constructed reduction is the right fixture: it is also the shape a
// caller that never resolved a group would hand over.
SymmetryReduction TrivialReduction() {
    return SymmetryReduction{};
}

TEST(SymmetryParityRefusalTest, RiFullFockRefusesARequestedReduction) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jkfit g shells";
    }

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = ParseFixtureBasis("def2-universal-jkfit");
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = qcx::integrals::test::BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const SymmetryReduction reduction = TrivialReduction();

    // Both reduction fields, one at a time: either one alone is a request this
    // path cannot honour, and the pre-fix builder accepted both and ignored
    // them (the silent substitution the audit found).
    qcx::integrals::RiEngineOptions orbitalOnly;
    orbitalOnly.symmetryReduction = &reduction;
    const auto refusedOrbital =
        RiFullFockBuilder::Create(*molecule, *basis, *aux, *core, orbitalOnly);
    ASSERT_FALSE(refusedOrbital.has_value());
    EXPECT_EQ(refusedOrbital.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(refusedOrbital.error().message.find("point-group reduction"), std::string::npos)
        << refusedOrbital.error().message;

    qcx::integrals::RiEngineOptions auxOnly;
    auxOnly.auxSymmetryReduction = &reduction;
    const auto refusedAux = RiFullFockBuilder::Create(*molecule, *basis, *aux, *core, auxOnly);
    ASSERT_FALSE(refusedAux.has_value());
    EXPECT_EQ(refusedAux.error().code, qcx::ErrorCode::kUnimplemented);
    // The refusal must NAME the thing refused, not only the cause: this
    // builder receives TWO reduction fields and either one alone refuses, so a
    // message that named only "the reduction" would leave the caller guessing
    // which of its two pointers to clear ("refused with its condition
    // named").
    EXPECT_NE(refusedAux.error().message.find("auxSymmetryReduction"), std::string::npos)
        << refusedAux.error().message;

    // The control: the same call with both fields cleared succeeds and carries
    // its Create-time counters. Without this leg the refusals above could be
    // firing on the fixture rather than on the option - and the counters are
    // the number a later increment is aimed by.
    const auto plain = RiFullFockBuilder::Create(*molecule, *basis, *aux, *core);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    const RiTermCounters& counters = plain->TermCounters();
    EXPECT_GT(counters.x, 0u) << "the metric and tensor passes engaged no aux entry";
    EXPECT_GT(counters.p3, 0u) << "the screened tensor pass engaged no orbital pair";
    EXPECT_GT(counters.g3, 0u) << "the screened tensor pass evaluated no kernel weight";
    EXPECT_EQ(counters.qx, 0u) << "this builder calls no quartet kernel";
    EXPECT_EQ(counters.gx, 0u) << "this builder calls no quartet kernel";
}

// The sink is the SHARED rule (internal/md_vrr_3c.hpp): the chunked build's
// totals over the full aux shell range must equal the monolithic build's on
// the same inputs, which is what makes the RI-K counters a measurement rather
// than a second opinion. The monolithic leg is the RI-J engine's own counter
// (ri_engine.hpp BuildRiTensor).
TEST(SymmetryParityRefusalTest, TheChunkSinkReproducesTheMonolithicCounters) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jkfit g shells";
    }

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = ParseFixtureBasis("def2-universal-jkfit");
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = qcx::integrals::test::BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const auto plain = RiFullFockBuilder::Create(*molecule, *basis, *aux, *core);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    qcx::integrals::RiEngineOptions options;
    RiTermCounters monolithic;
    const auto tensor =
        qcx::integrals::BuildRiTensor(*molecule, *basis, *aux, options, &monolithic);
    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;

    const RiTermCounters& chunked = plain->TermCounters();
    // The measured number the next increment is sized by:
    // printed, not only asserted, so the count is quotable.
    std::printf("RI-K counters, H2O/STO-3G + def2-universal-jkfit: x=%zu p3=%zu g3=%zu "
                "(monolithic BuildRiTensor: x=%zu p3=%zu g3=%zu)\n",
                chunked.x,
                chunked.p3,
                chunked.g3,
                monolithic.x,
                monolithic.p3,
                monolithic.g3);
    // The metric occurrence is charged by both legs (the RI-K Create runs
    // BuildAuxMetric as well), so `x` and `g3` differ by exactly that
    // occurrence; `p3` - the orbital pair side, which the metric never
    // engages - must match to the unit.
    EXPECT_EQ(chunked.p3, monolithic.p3);
    EXPECT_GE(chunked.x, monolithic.x);
    EXPECT_GE(chunked.g3, monolithic.g3);
}

} // namespace
