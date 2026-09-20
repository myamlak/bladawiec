// NWChem basis parser: happy paths, every rejection path, and a walk over
// the vendored BSE corpus.
#include "qcx/basisset/basis_set.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace qcx::basisset {
namespace {

constexpr double kValueTolerance = 1e-10;
constexpr std::size_t kMinCorpusElementsPerFamily = 10;

// Compares two parsed element entries bit-exactly: shell count, angular
// momenta, spherical flags, and the exponent/coefficient vectors.
void ExpectElementBasisBitIdentical(const ElementBasis& actual, const ElementBasis& expected) {
    EXPECT_EQ(actual.atomicNumber, expected.atomicNumber);
    EXPECT_EQ(actual.symbol, expected.symbol);
    ASSERT_EQ(actual.shells.size(), expected.shells.size());

    for (std::size_t i = 0; i < actual.shells.size(); ++i)
    {
        EXPECT_EQ(actual.shells[i].angularMomentum, expected.shells[i].angularMomentum);
        EXPECT_EQ(actual.shells[i].isSpherical, expected.shells[i].isSpherical);
        EXPECT_EQ(actual.shells[i].exponents, expected.shells[i].exponents);
        ASSERT_EQ(actual.shells[i].coefficients.size(), expected.shells[i].coefficients.size());

        for (std::size_t c = 0; c < actual.shells[i].coefficients.size(); ++c)
        {
            EXPECT_EQ(actual.shells[i].coefficients[c], expected.shells[i].coefficients[c]);
        }
    }
}

// STO-3G H, published values (Hehre, Stewart & Pople 1969, Table I).
constexpr char kSto3gHydrogen[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
END
)";

// A multi-primitive l = 1 shell: exercises the 2^l factor and the
// cross-exponent terms of the normalization closed form (an l = 0
// single-primitive test cannot - S(a, a) = 1 for every l, and s has no
// (ab)^(l/2) factor).
constexpr char kPFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
B    P
      2.0000000000E+00       3.0000000000E-01
      5.0000000000E-01       7.0000000000E-01
END
)";

// One element with an orbital block and an ECP for the same element
// (def2-style shape).
constexpr char kEcpFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
Ag    S
      1.0000000000E+00       1.0000000000E+00
ECP
Ag nelec 28
Ag ul
2      14.2200000            -33.68992012
END
)";

TEST(BasisSetTest, ParsesSto3gHydrogen) {
    const auto basis = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    ASSERT_EQ(basis->Elements().size(), 1u);
    const ElementBasis& hydrogen = basis->Elements()[0];
    EXPECT_EQ(hydrogen.atomicNumber, 1);
    EXPECT_EQ(hydrogen.symbol, "H");
    ASSERT_EQ(hydrogen.shells.size(), 1u);
    const Shell& shell = hydrogen.shells[0];
    EXPECT_EQ(shell.angularMomentum, 0);
    EXPECT_TRUE(shell.isSpherical);
    ASSERT_EQ(shell.exponents.size(), 3u);
    EXPECT_NEAR(shell.exponents[0], 3.4252509140, kValueTolerance);
    EXPECT_NEAR(shell.exponents[1], 0.6239137298, kValueTolerance);
    EXPECT_NEAR(shell.exponents[2], 0.1688554040, kValueTolerance);
    EXPECT_NEAR(shell.coefficients[0][0], 0.1543289673, kValueTolerance);
    EXPECT_NEAR(shell.coefficients[0][1], 0.5353281423, kValueTolerance);
    EXPECT_NEAR(shell.coefficients[0][2], 0.4446345422, kValueTolerance);
    EXPECT_NE(basis->Find("H"), nullptr);
    EXPECT_EQ(basis->Find(2), nullptr);
}

TEST(BasisSetTest, SpShellFlattensToTwoSharingExponents) {
    constexpr char kSpFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
Li    S
      1.0000000000E+00       1.0000000000E+00
Li    SP
      2.0000000000E+00       0.5000000000E+00       0.6000000000E+00
END
)";
    const auto basis = ParseNwchemText(kSpFixture);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const ElementBasis& lithium = *basis->Find("Li");
    ASSERT_EQ(lithium.shells.size(), 3u);
    EXPECT_EQ(lithium.shells[0].angularMomentum, 0);
    EXPECT_EQ(lithium.shells[1].angularMomentum, 0);
    EXPECT_EQ(lithium.shells[2].angularMomentum, 1);
    // The flattened s and p shells share the exponent array.
    EXPECT_EQ(lithium.shells[1].exponents, lithium.shells[2].exponents);
    // Unit-norm normalization at parse (NormalizeContractions): a single
    // already-normalized primitive renormalizes to coefficient exactly 1.
    EXPECT_NEAR(lithium.shells[1].coefficients[0][0], 1.0, kValueTolerance);
    EXPECT_NEAR(lithium.shells[2].coefficients[0][0], 1.0, kValueTolerance);
}

TEST(BasisSetTest, CartesianHeaderFlipsIsSpherical) {
    constexpr char kCartesianFixture[] = R"(BASIS "ao basis" CARTESIAN PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
END
)";
    const auto basis = ParseNwchemText(kCartesianFixture);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    EXPECT_FALSE(basis->Find("H")->shells[0].isSpherical);
}

TEST(BasisSetTest, EcpStubIsRecorded) {
    const auto basis = ParseNwchemText(kEcpFixture);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    ASSERT_EQ(basis->Ecps().size(), 1u);
    EXPECT_EQ(basis->Ecps()[0].atomicNumber, 47);
    EXPECT_EQ(basis->Ecps()[0].coreElectronCount, 28);
    // The radial terms were skipped without disturbing the orbital basis.
    EXPECT_NE(basis->Find("Ag"), nullptr);
}

TEST(BasisSetTest, GeneralContractionCarriesMultipleCoefficientSets) {
    // aug-cc-pVDZ-style general contraction: one exponent, four coefficients.
    constexpr char kGeneralFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
Al    S
      6.415000E+04           2.902500E-04          -7.580480E-05           1.750780E-05           0.000000E+00
      9.617000E+03           2.250640E-03          -5.817910E-04           1.342080E-04           0.000000E+00
END
)";
    const auto basis = ParseNwchemText(kGeneralFixture);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const Shell& shell = basis->Find("Al")->shells[0];
    ASSERT_EQ(shell.exponents.size(), 2u);
    ASSERT_EQ(shell.coefficients.size(), 4u);
    // Re-measured 2026-08-22: the unit-norm normalization at parse scales
    // each contraction row by 1/sqrt(norm^2) - the raw 2.902500E-04 row
    // renormalizes to 0.11978241107869977 (see NormalizesContractionsToUnitNorm
    // for the invariant itself).
    EXPECT_NEAR(shell.coefficients[0][0], 0.11978241107869977, kValueTolerance);
    EXPECT_NEAR(shell.coefficients[3][0], 0.0, kValueTolerance);
    EXPECT_EQ(shell.coefficients[2].size(), 2u);
}

TEST(BasisSetTest, NormalizesContractionsToUnitNorm) {
    // The unit-norm regression guard: the parser renormalizes every contraction
    // to unit norm under the engine convention (basis_set.cpp
    // NormalizeContractions) - the closed-form primitive-pair overlap
    // S(a, b) = 2^l (4ab)^(3/4) (ab)^(l/2) / (a + b)^(l + 3/2). The raw
    // STO-3G H contraction self-overlap is 1 + 7e-11 (the 2026-08-22
    // finding: without the parse-time fix the RI-J metric condition number
    // explodes on the jfit auxiliary norm spread).
    const auto basis = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const Shell& shell = basis->Find("H")->shells[0];
    ASSERT_EQ(shell.exponents.size(), 3u);

    double normSquared = 0.0;

    for (std::size_t k = 0; k < shell.exponents.size(); ++k)
    {
        for (std::size_t m = 0; m < shell.exponents.size(); ++m)
        {
            const double a = shell.exponents[k];
            const double b = shell.exponents[m];
            const double ab = a * b;
            // The l = 0 case of the closed form: 2^0 (4ab)^(3/4) (ab)^0 /
            // (a + b)^(3/2).
            const double overlap = std::pow(4.0 * ab, 0.75) / std::pow(a + b, 1.5);
            normSquared += shell.coefficients[0][k] * shell.coefficients[0][m] * overlap;
        }
    }

    EXPECT_NEAR(normSquared, 1.0, 1e-12);

    // The l > 0 branch: the 2^l factor and the (ab)^(l/2) cross terms are
    // exercised only by a multi-primitive shell with angular momentum - an
    // l = 0 single-primitive shell would hide a dropped 2^l (S(a, a) = 1 for
    // every l) and all s cross terms scale with 2^0.
    const auto pBasis = ParseNwchemText(kPFixture);
    ASSERT_TRUE(pBasis.has_value()) << pBasis.error().message;
    const Shell& pShell = pBasis->Find("B")->shells[0];
    ASSERT_EQ(pShell.angularMomentum, 1);
    ASSERT_EQ(pShell.exponents.size(), 2u);

    double pNormSquared = 0.0;

    for (std::size_t k = 0; k < pShell.exponents.size(); ++k)
    {
        for (std::size_t m = 0; m < pShell.exponents.size(); ++m)
        {
            const double a = pShell.exponents[k];
            const double b = pShell.exponents[m];
            const double ab = a * b;
            // The l = 1 case of the closed form: 2^1 (4ab)^(3/4) (ab)^(1/2)
            // / (a + b)^(5/2).
            const double overlap =
                2.0 * std::pow(4.0 * ab, 0.75) * std::pow(ab, 0.5) / std::pow(a + b, 2.5);
            pNormSquared += pShell.coefficients[0][k] * pShell.coefficients[0][m] * overlap;
        }
    }

    EXPECT_NEAR(pNormSquared, 1.0, 1e-12);
}

TEST(BasisSetTest, AppendRenormalizationIsIdempotent) {
    // The append path renormalizes the incoming block with the same
    // NormalizeContractions as parse, so appending to an already-normalized
    // set must leave every existing coefficient bit-unchanged (scale 1) and
    // land the new element normalized too.
    auto basis = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const double hydrogenCoefficient = basis->Find("H")->shells[0].coefficients[0][0];

    const auto result = basis->AppendNwchemText(kPFixture);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(basis->Find("H")->shells[0].coefficients[0][0], hydrogenCoefficient);

    const Shell& boron = basis->Find("B")->shells[0];
    ASSERT_EQ(boron.exponents.size(), 2u);
    double normSquared = 0.0;

    for (std::size_t k = 0; k < boron.exponents.size(); ++k)
    {
        for (std::size_t m = 0; m < boron.exponents.size(); ++m)
        {
            const double a = boron.exponents[k];
            const double b = boron.exponents[m];
            const double ab = a * b;
            const double overlap =
                2.0 * std::pow(4.0 * ab, 0.75) * std::pow(ab, 0.5) / std::pow(a + b, 2.5);
            normSquared += boron.coefficients[0][k] * boron.coefficients[0][m] * overlap;
        }
    }

    EXPECT_NEAR(normSquared, 1.0, 1e-12);
}

TEST(BasisSetTest, EveryRejectionPathReportsTheRightError) {
    EXPECT_EQ(ParseNwchemText("").error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(ParseNwchemFile("no/such/file.nwchem").error().code, qcx::ErrorCode::kIOError);

    constexpr char kBadHeader[] = R"(BASIS "other basis" SPHERICAL PRINT
END
)";
    EXPECT_EQ(ParseNwchemText(kBadHeader).error().code, qcx::ErrorCode::kInvalidArgument);

    constexpr char kMissingEnd[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
)";
    EXPECT_EQ(ParseNwchemText(kMissingEnd).error().code, qcx::ErrorCode::kInvalidArgument);

    constexpr char kBadNumber[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0ab             1.0000000000E+00
END
)";
    const auto badNumber = ParseNwchemText(kBadNumber);
    EXPECT_EQ(badNumber.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(badNumber.error().message.find("line 3"), std::string::npos);

    constexpr char kUnknownShell[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    Q
      1.0000000000E+00       1.0000000000E+00
END
)";
    EXPECT_EQ(ParseNwchemText(kUnknownShell).error().code, qcx::ErrorCode::kInvalidArgument);

    constexpr char kUnknownElement[] = R"(BASIS "ao basis" SPHERICAL PRINT
Qq    S
      1.0000000000E+00       1.0000000000E+00
END
)";
    EXPECT_EQ(ParseNwchemText(kUnknownElement).error().code, qcx::ErrorCode::kInvalidArgument);

    constexpr char kSpTwoColumns[] = R"(BASIS "ao basis" SPHERICAL PRINT
Li    SP
      1.0000000000E+00       1.0000000000E+00
END
)";
    EXPECT_EQ(ParseNwchemText(kSpTwoColumns).error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BasisSetTest, VendoredCorpusParsesEndToEnd) {
    const std::filesystem::path root(QcxBasisDataDir);
    ASSERT_TRUE(std::filesystem::is_directory(root));

    std::vector<std::string> families;

    for (const auto& entry : std::filesystem::directory_iterator(root))
    {
        if (entry.is_directory())
        {
            families.push_back(entry.path().filename().string());
        }
    }

    ASSERT_GE(families.size(), 12u);

    for (const auto& family : families)
    {
        SCOPED_TRACE(family);
        std::size_t fileCount = 0;

        for (const auto& entry : std::filesystem::directory_iterator(root / family))
        {
            if (entry.path().extension() != ".nwchem")
            {
                continue;
            }

            ++fileCount;
            const std::string expectedSymbol = entry.path().stem().string();
            const auto basis = ParseNwchemFile(entry.path().string());
            ASSERT_TRUE(basis.has_value()) << basis.error().message;
            const ElementBasis* element = basis->Find(expectedSymbol);
            ASSERT_NE(element, nullptr) << "filename symbol must be in the file";

            for (const auto& shell : element->shells)
            {
                ASSERT_FALSE(shell.exponents.empty());

                for (const double exponent : shell.exponents)
                {
                    EXPECT_TRUE(std::isfinite(exponent) && exponent > 0.0);
                }

                ASSERT_GE(shell.coefficients.size(), 1u);

                for (const auto& contraction : shell.coefficients)
                {
                    EXPECT_EQ(contraction.size(), shell.exponents.size());
                }
            }
        }

        EXPECT_GE(fileCount, kMinCorpusElementsPerFamily);
    }
}

TEST(BasisSetTest, DirectoryParseMergesAndSorts) {
    const std::filesystem::path root(QcxBasisDataDir);
    const auto basis = ParseNwchemDirectory((root / "sto-3g").string());
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    EXPECT_GE(basis->Elements().size(), kMinCorpusElementsPerFamily);

    for (std::size_t i = 1; i < basis->Elements().size(); ++i)
    {
        EXPECT_LT(basis->Elements()[i - 1].atomicNumber, basis->Elements()[i].atomicNumber);
    }
    // Spot check the real STO-3G H file against the published values.
    const ElementBasis* hydrogen = basis->Find("H");
    ASSERT_NE(hydrogen, nullptr);
    EXPECT_NEAR(hydrogen->shells[0].exponents[0], 3.4252509140, kValueTolerance);
}

TEST(BasisSetTest, FilteredDirectoryParseMatchesUnfilteredForItsElements) {
    // The auxiliary family the crossover grid uses: the filtered parse of
    // {H, C, N, O} must be element-for-element bit-identical to the full
    // 86-element parse - the filter is value-neutral by construction.
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 4> wanted{1, 6, 7, 8};
    const auto unfiltered = ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(unfiltered.has_value()) << unfiltered.error().message;
    const auto filtered =
        ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(), wanted);
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;

    ASSERT_EQ(filtered->Elements().size(), wanted.size());

    for (std::size_t i = 0; i < wanted.size(); ++i)
    {
        const ElementBasis* expected = unfiltered->Find(wanted[i]);
        ASSERT_NE(expected, nullptr);
        ExpectElementBasisBitIdentical(filtered->Elements()[i], *expected);
    }

    // None of the light elements carries an ECP stub.
    EXPECT_TRUE(filtered->Ecps().empty());
}

TEST(BasisSetTest, FilteredDirectoryParseExcludesForeignElements) {
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 2> wanted{1, 8};
    const auto filtered =
        ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(), wanted);
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;
    ASSERT_EQ(filtered->Elements().size(), 2u);
    EXPECT_EQ(filtered->Elements()[0].atomicNumber, 1);
    EXPECT_EQ(filtered->Elements()[1].atomicNumber, 8);
    // Foreign elements of the full set are absent: by symbol and by Z.
    EXPECT_EQ(filtered->Find("C"), nullptr);
    EXPECT_EQ(filtered->Find(47), nullptr); // Ag: present in the 86-file set.
}

TEST(BasisSetTest, FilteredDirectoryParseToleratesDuplicateAtomicNumbers) {
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 3> wanted{1, 8, 1};
    const auto filtered =
        ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(), wanted);
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;
    ASSERT_EQ(filtered->Elements().size(), 2u);
    EXPECT_EQ(filtered->Elements()[0].atomicNumber, 1);
    EXPECT_EQ(filtered->Elements()[1].atomicNumber, 8);
}

TEST(BasisSetTest, FilteredDirectoryParseCarriesEcpStubs) {
    // The ECP files of a filtered family survive the filter exactly as the
    // unfiltered parse records them (def2-svp's Ag stub, 28 core electrons).
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 2> wanted{1, 47};
    const auto filtered = ParseNwchemDirectoryFiltered((root / "def2-svp").string(), wanted);
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;
    ASSERT_EQ(filtered->Ecps().size(), 1u);
    EXPECT_EQ(filtered->Ecps()[0].atomicNumber, 47);
    EXPECT_EQ(filtered->Ecps()[0].coreElectronCount, 28);
}

TEST(BasisSetTest, FilteredDirectoryParseRejectsMissingElementFile) {
    // Og (Z=118) has no vendored file in any family: kInvalidArgument naming
    // the element and the directory - the failure BuildShellPairs would
    // raise later, surfaced here with the clearer message.
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 2> wanted{1, 118};
    const auto result =
        ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(), wanted);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("Og"), std::string::npos);
    EXPECT_NE(result.error().message.find("def2-universal-jfit"), std::string::npos);
}

TEST(BasisSetTest, FilteredDirectoryParseRejectsOutOfRangeAtomicNumber) {
    const std::filesystem::path root(QcxBasisDataDir);
    const std::array<int, 2> wanted{1, 119};
    const auto result =
        ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(), wanted);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BasisSetTest, FilteredDirectoryParseErrorsAreKIOError) {
    EXPECT_EQ(ParseNwchemDirectoryFiltered("no/such/dir", std::array<int, 1>{1}).error().code,
              qcx::ErrorCode::kIOError);
    const std::filesystem::path root(QcxBasisDataDir);
    EXPECT_EQ(
        ParseNwchemDirectoryFiltered((root / "sto-3g" / "H.nwchem").string(), std::array<int, 1>{1})
            .error()
            .code,
        qcx::ErrorCode::kIOError);
}

TEST(BasisSetTest, Def2EcpFilesCarryTheStub) {
    const std::filesystem::path root(QcxBasisDataDir);
    const auto basis = ParseNwchemFile((root / "def2-svp" / "Ag.nwchem").string());
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    ASSERT_EQ(basis->Ecps().size(), 1u);
    EXPECT_EQ(basis->Ecps()[0].atomicNumber, 47);
    EXPECT_EQ(basis->Ecps()[0].coreElectronCount, 28);
}

TEST(BasisSetTest, RejectsEndDirectlyAfterShellHeader) {
    constexpr char kFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
END
)";
    const auto result = ParseNwchemText(kFixture);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("END inside a shell block"), std::string::npos);
}

TEST(BasisSetTest, RejectsContentAfterOrbitalEndThatIsNotEcp) {
    constexpr char kFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
END
O    S
)";
    EXPECT_EQ(ParseNwchemText(kFixture).error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BasisSetTest, RejectsEcpBlockWithoutClosingEnd) {
    constexpr char kFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
END
ECP
Ag nelec 28
)";
    const auto result = ParseNwchemText(kFixture);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("missing END"), std::string::npos);
}

TEST(BasisSetTest, RejectsInconsistentColumnCountWithinShell) {
    constexpr char kFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
      5.0000000000E-01       2.0000000000E-01       3.0000000000E-01
END
)";
    EXPECT_EQ(ParseNwchemText(kFixture).error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BasisSetTest, RejectsPrimitiveBeforeAnyElement) {
    constexpr char kFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
      1.0000000000E+00       1.0000000000E+00
END
)";
    EXPECT_EQ(ParseNwchemText(kFixture).error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BasisSetTest, MergeRejectsDuplicateElement) {
    const auto first = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    const auto second = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    BasisSet merged = *first;
    const auto result = merged.Merge(*second);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("duplicate element 'H'"), std::string::npos);
}

// Merge used to push other's elements before the ECP
// duplicate check, so a failed merge stranded part of other (unsorted) in
// the set - and a retry then failed with a phantom duplicate. The set must
// be bit-identical to its pre-merge state on any rejection.
TEST(BasisSetTest, MergeRollsBackOnDuplicateEcp) {
    constexpr char kOtherFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
Li    S
      1.0000000000E+00       1.0000000000E+00
END
ECP
Ag nelec 28
END
)";
    const auto merged = ParseNwchemText(kEcpFixture); // Ag orbital + Ag ECP.
    ASSERT_TRUE(merged.has_value()) << merged.error().message;
    const auto other = ParseNwchemText(kOtherFixture); // Li + duplicate Ag ECP.
    ASSERT_TRUE(other.has_value()) << other.error().message;
    const std::size_t elementCount = merged->Elements().size();
    const std::size_t ecpCount = merged->Ecps().size();
    BasisSet candidate = *merged;
    const auto result = candidate.Merge(*other);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("in ECPs"), std::string::npos);
    // Li must not leak in from the failed merge; the set keeps exactly the
    // pre-merge element and ECP lists.
    EXPECT_EQ(candidate.Elements().size(), elementCount);
    EXPECT_EQ(candidate.Ecps().size(), ecpCount);
    EXPECT_EQ(candidate.Find("Li"), nullptr);

    for (std::size_t k = 0; k < candidate.Ecps().size(); ++k)
    {
        EXPECT_EQ(candidate.Ecps()[k].atomicNumber, merged->Ecps()[k].atomicNumber);
        EXPECT_EQ(candidate.Ecps()[k].coreElectronCount, merged->Ecps()[k].coreElectronCount);
    }

    for (const auto& element : candidate.Elements())
    {
        EXPECT_NE(merged->Find(element.symbol), nullptr);
    }
}

TEST(BasisSetTest, MergeRollsBackWhenLaterElementDuplicates) {
    constexpr char kOtherFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
Li    S
      1.0000000000E+00       1.0000000000E+00
H     S
      1.0000000000E+00       1.0000000000E+00
END
)";
    const auto merged = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(merged.has_value()) << merged.error().message;
    const auto other = ParseNwchemText(kOtherFixture); // Li first, H duplicates.
    ASSERT_TRUE(other.has_value()) << other.error().message;
    BasisSet candidate = *merged;
    const auto result = candidate.Merge(*other);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    // The earlier, non-duplicate element must not have been committed, and
    // the original H keeps its content.
    EXPECT_EQ(candidate.Find("Li"), nullptr);
    EXPECT_EQ(candidate.Elements().size(), 1u);
    ASSERT_NE(candidate.Find("H"), nullptr);
    EXPECT_EQ(candidate.Find("H")->shells.size(), merged->Find("H")->shells.size());
    EXPECT_EQ(candidate.Find("H")->shells[0].exponents, merged->Find("H")->shells[0].exponents);
    // The retry fails identically - no phantom state accumulated.
    const auto retry = candidate.Merge(*other);
    EXPECT_EQ(retry.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BasisSetTest, RejectsDuplicateEnd) {
    constexpr char kFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
END
END
)";
    const auto result = ParseNwchemText(kFixture);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("duplicate END"), std::string::npos);
}

TEST(BasisSetTest, RejectsContentAfterEcpClosingEnd) {
    constexpr char kFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
END
ECP
Ag nelec 28
END
garbage
)";
    const auto result = ParseNwchemText(kFixture);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("content after the closing END"), std::string::npos);
}

TEST(BasisSetTest, RejectsNonPositiveExponents) {
    constexpr char kNegativeExponent[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
     -1.0000000000E+00       1.0000000000E+00
END
)";
    const auto negative = ParseNwchemText(kNegativeExponent);
    EXPECT_EQ(negative.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(negative.error().message.find("exponent must be positive"), std::string::npos);

    constexpr char kZeroExponent[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      0.0000000000E+00       1.0000000000E+00
END
)";
    const auto zero = ParseNwchemText(kZeroExponent);
    EXPECT_EQ(zero.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(zero.error().message.find("exponent must be positive"), std::string::npos);
}

TEST(BasisSetTest, RejectsInvalidEcpCoreCounts) {
    constexpr char kNegativeCount[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
END
ECP
Ag nelec -5
END
)";
    const auto negative = ParseNwchemText(kNegativeCount);
    EXPECT_EQ(negative.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(negative.error().message.find("at least 1"), std::string::npos);

    constexpr char kFractionalCount[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
END
ECP
Ag nelec 28.5
END
)";
    const auto fractional = ParseNwchemText(kFractionalCount);
    EXPECT_EQ(fractional.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(fractional.error().message.find("invalid ECP core count"), std::string::npos);

    // The stub path (no ECP keyword line) validates identically.
    constexpr char kNegativeStub[] = R"(BASIS "ao basis" SPHERICAL PRINT
Ag nelec -5
END
)";
    EXPECT_EQ(ParseNwchemText(kNegativeStub).error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BasisSetTest, AppendSortsElementsAndEcpsByZ) {
    constexpr char kOutOfOrderFixture[] = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      1.0000000000E+00       1.0000000000E+00
H    S
      1.0000000000E+00       1.0000000000E+00
END
ECP
Xe nelec 46
Ag nelec 28
END
)";
    const auto basis = ParseNwchemText(kOutOfOrderFixture);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    ASSERT_EQ(basis->Elements().size(), 2u);
    EXPECT_EQ(basis->Elements()[0].symbol, "H");
    EXPECT_EQ(basis->Elements()[1].symbol, "O");
    ASSERT_EQ(basis->Ecps().size(), 2u);
    EXPECT_EQ(basis->Ecps()[0].atomicNumber, 47);
    EXPECT_EQ(basis->Ecps()[1].atomicNumber, 54);
}

TEST(BasisSetTest, AppendRejectsDuplicateElement) {
    auto basis = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto result = basis->AppendNwchemText(kSto3gHydrogen);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("duplicate element 'H'"), std::string::npos);
}

TEST(BasisSetTest, AppendRejectsDuplicateEcp) {
    constexpr char kEcpOnly[] = R"(BASIS "ao basis" SPHERICAL PRINT
ECP
Ag nelec 28
END
)";
    // Across calls: the second block repeats the ECP.
    auto basis = ParseNwchemText(kEcpOnly);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto result = basis->AppendNwchemText(kEcpOnly);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("in ECPs"), std::string::npos);

    // Within one block as well (Merge rejects the same case).
    constexpr char kRepeatedEcp[] = R"(BASIS "ao basis" SPHERICAL PRINT
ECP
Ag nelec 28
Ag nelec 28
END
)";
    const auto repeated = ParseNwchemText(kRepeatedEcp);
    EXPECT_EQ(repeated.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(repeated.error().message.find("in ECPs"), std::string::npos);
}

// AppendNwchemText used to append elements and shells
// live, so an error mid-block stranded the block's earlier shells in the
// set (unnormalized, unsorted) and the next append built on them. The set
// must be bit-identical to its pre-append state on any rejection.
TEST(BasisSetTest, AppendRollsBackOnElementDuplicate) {
    constexpr char kFailingBlock[] = R"(BASIS "ao basis" SPHERICAL PRINT
Li    S
      1.0000000000E+00       1.0000000000E+00
O     S
      1.0000000000E+00       1.0000000000E+00
H     S
      1.0000000000E+00       1.0000000000E+00
END
)";
    auto basis = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto result = basis->AppendNwchemText(kFailingBlock);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("duplicate element 'H'"), std::string::npos);
    // The block's Li and O shells must not leak into the set; the original
    // H stays untouched.
    EXPECT_EQ(basis->Elements().size(), 1u);
    EXPECT_NE(basis->Find("H"), nullptr);
    EXPECT_EQ(basis->Find("Li"), nullptr);
    EXPECT_EQ(basis->Find("O"), nullptr);
}

TEST(BasisSetTest, AppendRollsBackOnEcpDuplicate) {
    constexpr char kFailingBlock[] = R"(BASIS "ao basis" SPHERICAL PRINT
Li    S
      1.0000000000E+00       1.0000000000E+00
END
ECP
Ag nelec 28
END
)";
    auto basis = ParseNwchemText(kEcpFixture); // Ag orbital + Ag ECP.
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const std::size_t elementCount = basis->Elements().size();
    const auto result = basis->AppendNwchemText(kFailingBlock);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("in ECPs"), std::string::npos);
    // Li (the block's only new element) must not leak in.
    EXPECT_EQ(basis->Find("Li"), nullptr);
    EXPECT_EQ(basis->Elements().size(), elementCount);
    EXPECT_EQ(basis->Ecps().size(), 1u);
}

TEST(BasisSetTest, AppendRollsBackOnGrammarError) {
    constexpr char kFailingBlock[] = R"(BASIS "ao basis" SPHERICAL PRINT
Li    S
      1.0000000000E+00       1.0000000000E+00
O     S
      1.0000000000E+00       1.0000000000E+00
garbage
END
)";
    auto basis = ParseNwchemText(kSto3gHydrogen);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto result = basis->AppendNwchemText(kFailingBlock);
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(basis->Elements().size(), 1u);
    EXPECT_EQ(basis->Find("Li"), nullptr);
    EXPECT_EQ(basis->Find("O"), nullptr);
    // The set stays fully usable after the failed append.
    EXPECT_NE(basis->Find("H"), nullptr);
}

TEST(BasisSetTest, DirectoryParseErrorsAreKIOError) {
    EXPECT_EQ(ParseNwchemDirectory("no/such/dir").error().code, qcx::ErrorCode::kIOError);
    const std::filesystem::path root(QcxBasisDataDir);
    EXPECT_EQ(ParseNwchemDirectory((root / "sto-3g" / "H.nwchem").string()).error().code,
              qcx::ErrorCode::kIOError);
}

TEST(BasisSetTest, MergeAcceptsElementAndEcpForSameZ) {
    const auto basis = ParseNwchemText(kEcpFixture);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    BasisSet merged;
    const auto result = merged.Merge(*basis);
    EXPECT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(merged.Find("Ag"), nullptr);
    ASSERT_EQ(merged.Ecps().size(), 1u);
    EXPECT_EQ(merged.Ecps()[0].atomicNumber, 47);
}

TEST(BasisSetTest, DirectoryParseMergesEcpFiles) {
    const std::filesystem::path root(QcxBasisDataDir);
    const auto basis = ParseNwchemDirectory((root / "def2-svp").string());
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    EXPECT_NE(basis->Find("Ag"), nullptr);

    for (std::size_t i = 1; i < basis->Ecps().size(); ++i)
    {
        EXPECT_LT(basis->Ecps()[i - 1].atomicNumber, basis->Ecps()[i].atomicNumber);
    }

    const auto agEcp =
        std::find_if(basis->Ecps().begin(), basis->Ecps().end(), [](const EcpDefinition& ecp) {
            return ecp.atomicNumber == 47;
        });
    ASSERT_NE(agEcp, basis->Ecps().end());
    EXPECT_EQ(agEcp->coreElectronCount, 28);
}

} // namespace
} // namespace qcx::basisset
