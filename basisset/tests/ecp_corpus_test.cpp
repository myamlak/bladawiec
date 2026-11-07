// The ECP corpus inventory and the basis-dependent electron count.
//
// The vendored def2 families carry the only ECP data in data/basis. Their
// `SYMBOL nelec N` metadata names N, the number of CORE electrons the ECP
// replaces - not the valence count. Xe (Z = 54, nelec 28) therefore has 26
// valence electrons and W (Z = 74, nelec 60) has 14. The corpus references
// are [Weigend2005] for the def2 pairing, [Peterson2003] for Xe and
// [Andrae1990] for W.
//
// Every ECP block also carries its radial terms: a local `ul` channel, S/P/D
// projectors, and F/G projectors on the lanthanides. The parser keeps them in
// file order (counts and coefficients unchanged), and
// qcx::basisset::ValenceElectronCount turns the core counts into the
// basis-dependent electron count, sum(Z - core) - charge. The count is
// basis-dependent because the same element is all-electron in one basis and
// core-replaced in another (Xe: 54 electrons under STO-3G, 26 under def2-SVP).
//
// The fixtures are the cost-ranked cells of the ECP acceptance matrix: an
// atom (Xe, W), an off-centre ECP (XeH+), a p-projector molecule (XeF2), and
// a d-projector, high-symmetry molecule (WF6).

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using qcx::basisset::BasisSet;
using qcx::basisset::EcpChannel;
using qcx::basisset::EcpDefinition;
using qcx::basisset::EcpTerm;
using qcx::molecule::Atom;
using qcx::molecule::Molecule;

qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> MakeCoordinates(
    const std::vector<std::vector<double>>& rows) {
    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({rows.size(), 3});
    EXPECT_TRUE(coordinates.has_value());

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coordinates)(i, static_cast<std::size_t>(d)) = rows[i][static_cast<std::size_t>(d)];
        }
    }

    return std::move(*coordinates);
}

qcx::Result<BasisSet> LoadFamily(const std::string& directory) {
    const std::filesystem::path root(QcxBasisDataDir);
    return qcx::basisset::ParseNwchemDirectory((root / directory).string());
}

const EcpDefinition* FindEcp(const BasisSet& basis, int atomicNumber) {
    for (const EcpDefinition& ecp : basis.Ecps())
    {
        if (ecp.atomicNumber == atomicNumber)
        {
            return &ecp;
        }
    }

    return nullptr;
}

std::optional<int> CoreCount(const BasisSet& basis, int atomicNumber) {
    const EcpDefinition* ecp = FindEcp(basis, atomicNumber);

    if (ecp != nullptr)
    {
        return ecp->coreElectronCount;
    }

    return std::nullopt;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- Z and core count.
void ExpectCore(const BasisSet& basis, int atomicNumber, int expected) {
    const std::optional<int> core = CoreCount(basis, atomicNumber);
    ASSERT_TRUE(core.has_value()) << "Z=" << atomicNumber;
    EXPECT_EQ(*core, expected) << "Z=" << atomicNumber;
}

void ExpectNoCore(const BasisSet& basis, int atomicNumber) {
    EXPECT_FALSE(CoreCount(basis, atomicNumber).has_value()) << "Z=" << atomicNumber;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- charge and multiplicity.
qcx::Result<Molecule> MakeMolecule(std::vector<Atom> atoms,
                                   const std::vector<std::vector<double>>& coordinates,
                                   int charge,
                                   int multiplicity) {
    return Molecule::Create(std::move(atoms), MakeCoordinates(coordinates), charge, multiplicity);
}

void ExpectValenceCount(const Molecule& molecule, const BasisSet& basis, int expected) {
    const qcx::Result<int> count = qcx::basisset::ValenceElectronCount(molecule, basis);
    ASSERT_TRUE(count.has_value()) << count.error().message;
    EXPECT_EQ(*count, expected);
}

TEST(EcpCorpusTest, Def2FamiliesCarryTheExpectedEcpInventory) {
    // The five def2 families are the only ECP carriers in the corpus. Their
    // coverage is Rb-Xe (core 28), Cs-La (core 46), Ce-Lu (core 28, with the
    // F and G projectors) and Hf-Rn (core 60); def2-TZVPD and def2-TZVPPD do
    // not vendor the lanthanides, so they carry 36 elements against 50.
    struct Family {
        const char* directory;
        std::size_t ecpElements;
    };

    const std::array<Family, 5> families{{
        {"def2-svp", 50},
        {"def2-tzvp", 50},
        {"def2-qzvp", 50},
        {"def2-tzvpd", 36},
        {"def2-tzvppd", 36},
    }};

    for (const Family& family : families)
    {
        SCOPED_TRACE(family.directory);
        const auto basis = LoadFamily(family.directory);
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        EXPECT_EQ(basis->Ecps().size(), family.ecpElements);

        // The core count is the count REPLACED, not the valence count.
        ExpectCore(*basis, 37, 28); // Rb
        ExpectCore(*basis, 54, 28); // Xe: 54 - 28 = 26 valence
        ExpectCore(*basis, 55, 46); // Cs
        ExpectCore(*basis, 74, 60); // W: 74 - 60 = 14 valence
        ExpectCore(*basis, 86, 60); // Rn

        // The all-electron side of the boundary: Kr is below the Rb cutoff.
        ExpectNoCore(*basis, 36);

        for (const EcpDefinition& ecp : basis->Ecps())
        {
            EXPECT_NE(basis->Find(ecp.atomicNumber), nullptr);
            EXPECT_TRUE(ecp.coreElectronCount >= 28 && ecp.coreElectronCount <= 60);
            EXPECT_LT(ecp.coreElectronCount, ecp.atomicNumber);
        }
    }
}

TEST(EcpCorpusTest, EveryCorpusEcpCarriesItsFullChannelTable) {
    // The corpus is uniform: every ECP has a local ul channel with at least
    // one Gaussian term and S/P/D projectors; Ce-Lu (Z = 58..71) add F and G
    // (l = 3, 4). Every corpus term carries n = 2 - a corpus fact, not a
    // parser assumption (a parser test round-trips other powers).
    const std::array<const char*, 5> families{
        "def2-svp", "def2-tzvp", "def2-qzvp", "def2-tzvpd", "def2-tzvppd"};

    for (const char* directory : families)
    {
        SCOPED_TRACE(directory);
        const auto basis = LoadFamily(directory);
        ASSERT_TRUE(basis.has_value()) << basis.error().message;

        for (const EcpDefinition& ecp : basis->Ecps())
        {
            SCOPED_TRACE(ecp.atomicNumber);
            const bool lanthanide = ecp.atomicNumber >= 58 && ecp.atomicNumber <= 71;
            ASSERT_EQ(ecp.channels.size(), lanthanide ? 6u : 4u);

            // Channel order is the file order: ul, then S, P, D, F, G.
            EXPECT_EQ(ecp.channels[0].angularMomentum, qcx::basisset::kEcpLocalChannel);
            EXPECT_EQ(ecp.channels[1].angularMomentum, 0);
            EXPECT_EQ(ecp.channels[2].angularMomentum, 1);
            EXPECT_EQ(ecp.channels[3].angularMomentum, 2);

            if (lanthanide)
            {
                EXPECT_EQ(ecp.channels[4].angularMomentum, 3);
                EXPECT_EQ(ecp.channels[5].angularMomentum, 4);
            }

            const EcpChannel* local = ecp.FindChannel(qcx::basisset::kEcpLocalChannel);
            ASSERT_NE(local, nullptr);
            EXPECT_FALSE(local->terms.empty());
            EXPECT_NE(ecp.FindChannel(0), nullptr);
            EXPECT_NE(ecp.FindChannel(1), nullptr);
            EXPECT_NE(ecp.FindChannel(2), nullptr);
            EXPECT_EQ(ecp.FindChannel(3) != nullptr, lanthanide);
            EXPECT_EQ(ecp.FindChannel(4) != nullptr, lanthanide);

            for (const EcpChannel& channel : ecp.channels)
            {
                EXPECT_FALSE(channel.terms.empty());

                for (const EcpTerm& term : channel.terms)
                {
                    EXPECT_EQ(term.power, 2);
                    EXPECT_GT(term.exponent, 0.0);
                }
            }
        }
    }
}

TEST(EcpCorpusTest, LanthanideElementsCarryTheCore28TableAndFGProjectors) {
    // Ce-Lu are the only core-28 group beyond Rb-Xe, and their projectors
    // reach G (l = 4); La (57), Y and Ba keep the four-channel table.
    const auto basis = LoadFamily("def2-svp");
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    for (int z = 58; z <= 71; ++z)
    {
        SCOPED_TRACE(z);
        ExpectCore(*basis, z, 28);
        const EcpDefinition* ecp = FindEcp(*basis, z);
        ASSERT_NE(ecp, nullptr);
        EXPECT_NE(ecp->FindChannel(3), nullptr);
        EXPECT_NE(ecp->FindChannel(4), nullptr);
    }

    for (const int z : {39, 56, 57})
    {
        SCOPED_TRACE(z);
        const EcpDefinition* ecp = FindEcp(*basis, z);
        ASSERT_NE(ecp, nullptr);
        EXPECT_EQ(ecp->FindChannel(3), nullptr);
        EXPECT_EQ(ecp->FindChannel(4), nullptr);
    }

    const auto diffuse = LoadFamily("def2-tzvpd");
    ASSERT_TRUE(diffuse.has_value()) << diffuse.error().message;
    ExpectNoCore(*diffuse, 58); // no lanthanides vendored in this family.
}

TEST(EcpCorpusTest, XenonDef2SvpTermsRoundTrip) {
    // Xe, def2-SVP: the ul and S/P/D channels of data/basis/def2-svp/Xe.nwchem
    // with their exact term counts, and a coefficient spot-checked at both
    // ends of the local channel's file order.
    const auto basis = LoadFamily("def2-svp");
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const EcpDefinition* ecp = FindEcp(*basis, 54);
    ASSERT_NE(ecp, nullptr);
    EXPECT_EQ(ecp->coreElectronCount, 28);
    ASSERT_EQ(ecp->channels.size(), 4u);

    const EcpChannel* local = ecp->FindChannel(qcx::basisset::kEcpLocalChannel);
    const EcpChannel* s = ecp->FindChannel(0);
    const EcpChannel* p = ecp->FindChannel(1);
    const EcpChannel* d = ecp->FindChannel(2);
    ASSERT_NE(local, nullptr);
    ASSERT_NE(s, nullptr);
    ASSERT_NE(p, nullptr);
    ASSERT_NE(d, nullptr);
    ASSERT_EQ(local->terms.size(), 4u);
    ASSERT_EQ(s->terms.size(), 7u);
    ASSERT_EQ(p->terms.size(), 8u);
    ASSERT_EQ(d->terms.size(), 10u);

    // The local channel: first and last terms in file order.
    EXPECT_EQ(local->terms.front().power, 2);
    EXPECT_DOUBLE_EQ(local->terms.front().exponent, 20.88155700);
    EXPECT_DOUBLE_EQ(local->terms.front().coefficient, -23.08929500);
    EXPECT_EQ(local->terms.back().power, 2);
    EXPECT_DOUBLE_EQ(local->terms.back().exponent, 5.36118800);
    EXPECT_DOUBLE_EQ(local->terms.back().coefficient, -0.38692400);

    // One spot-check in each projector channel.
    EXPECT_DOUBLE_EQ(s->terms.front().exponent, 40.00518400);
    EXPECT_DOUBLE_EQ(s->terms.front().coefficient, 49.99796200);
    EXPECT_DOUBLE_EQ(p->terms.front().exponent, 15.70177200);
    EXPECT_DOUBLE_EQ(p->terms.front().coefficient, 67.43914200);
    EXPECT_DOUBLE_EQ(d->terms.front().exponent, 15.18560000);
    EXPECT_DOUBLE_EQ(d->terms.front().coefficient, 35.43690800);
}

TEST(EcpCorpusTest, TungstenDef2TzvpTermsRoundTrip) {
    // W, def2-TZVP: ul, S, P, D with one/three/three/three terms. The local
    // coefficient is positive for W against Xe's negative one.
    const auto basis = LoadFamily("def2-tzvp");
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const EcpDefinition* ecp = FindEcp(*basis, 74);
    ASSERT_NE(ecp, nullptr);
    EXPECT_EQ(ecp->coreElectronCount, 60);
    ASSERT_EQ(ecp->channels.size(), 4u);

    const EcpChannel* local = ecp->FindChannel(qcx::basisset::kEcpLocalChannel);
    const EcpChannel* s = ecp->FindChannel(0);
    const EcpChannel* p = ecp->FindChannel(1);
    const EcpChannel* d = ecp->FindChannel(2);
    ASSERT_NE(local, nullptr);
    ASSERT_NE(s, nullptr);
    ASSERT_NE(p, nullptr);
    ASSERT_NE(d, nullptr);
    ASSERT_EQ(local->terms.size(), 1u);
    ASSERT_EQ(s->terms.size(), 3u);
    ASSERT_EQ(p->terms.size(), 3u);
    ASSERT_EQ(d->terms.size(), 3u);

    EXPECT_EQ(local->terms[0].power, 2);
    EXPECT_DOUBLE_EQ(local->terms[0].exponent, 2.25888846);
    EXPECT_DOUBLE_EQ(local->terms[0].coefficient, 14.15257947);

    EXPECT_DOUBLE_EQ(s->terms.front().exponent, 14.32285640);
    EXPECT_DOUBLE_EQ(s->terms.front().coefficient, 1192.39588226);
    EXPECT_DOUBLE_EQ(p->terms.back().exponent, 2.25888846);
    EXPECT_DOUBLE_EQ(p->terms.back().coefficient, -14.15257947);
    EXPECT_DOUBLE_EQ(d->terms.front().exponent, 6.59799743);
    EXPECT_DOUBLE_EQ(d->terms.front().coefficient, 108.30134897);
}

TEST(EcpCorpusTest, AllElectronFamiliesCarryNoEcpStubs) {
    // The ECP rule is per basis, not per element: these families describe
    // Xe and W with every electron present, and a run on them is all-electron.
    for (const char* directory : {"sto-3g", "cc-pvdz"})
    {
        SCOPED_TRACE(directory);
        const auto basis = LoadFamily(directory);
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        EXPECT_TRUE(basis->Ecps().empty());
        EXPECT_FALSE(basis->Elements().empty());
    }

    // STO-3G covers Xe all-electron, so the same element that carries a
    // 28-core ECP in def2-SVP is a 54-electron atom here.
    const auto sto3g = LoadFamily("sto-3g");
    ASSERT_TRUE(sto3g.has_value()) << sto3g.error().message;
    EXPECT_NE(sto3g->Find(54), nullptr);
}

TEST(EcpElectronCountTest, CountsAreSumOfValenceMinusCharge) {
    // qcx::basisset::ValenceElectronCount is basis-dependent: the same
    // element is all-electron in one basis and core-replaced in another.
    const auto svp = LoadFamily("def2-svp");
    const auto tzvp = LoadFamily("def2-tzvp");
    const auto sto3g = LoadFamily("sto-3g");
    ASSERT_TRUE(svp.has_value()) << svp.error().message;
    ASSERT_TRUE(tzvp.has_value()) << tzvp.error().message;
    ASSERT_TRUE(sto3g.has_value()) << sto3g.error().message;

    // Xe atom: 54 - 28 = 26 valence electrons; STO-3G has no ECP for Xe, so
    // the same atom keeps all 54 electrons there.
    const auto xenon = MakeMolecule({{"Xe", 54, 0.0}}, {{0.0, 0.0, 0.0}}, 0, 1);
    ASSERT_TRUE(xenon.has_value()) << xenon.error().message;
    ExpectValenceCount(*xenon, *svp, 26);
    ExpectValenceCount(*xenon, *sto3g, 54);

    // The molecule alone cannot know the core (it has no basis); this method
    // sums Z and must not be used to size an ECP run's occupancy.
    EXPECT_EQ(xenon->ElectronCount(), 54);

    // W atom: 74 - 60 = 14 valence electrons.
    const auto tungsten = MakeMolecule({{"W", 74, 0.0}}, {{0.0, 0.0, 0.0}}, 0, 1);
    ASSERT_TRUE(tungsten.has_value()) << tungsten.error().message;
    ExpectValenceCount(*tungsten, *tzvp, 14);

    // XeH+: 54 - 28 + 1 - 1 = 26. The off-centre ECP is the Xe atom's.
    const auto xenonHydride =
        MakeMolecule({{"Xe", 54, 0.0}, {"H", 1, 0.0}}, {{0.0, 0.0, 0.0}, {0.0, 0.0, 3.0}}, 1, 1);
    ASSERT_TRUE(xenonHydride.has_value()) << xenonHydride.error().message;
    ExpectValenceCount(*xenonHydride, *svp, 26);

    // XeF2: 54 - 28 + 2 * 9 = 44. Fluorine is all-electron in def2-SVP.
    const auto xenonDifluoride = MakeMolecule({{"Xe", 54, 0.0}, {"F", 9, 0.0}, {"F", 9, 0.0}},
                                              {{0.0, 0.0, 0.0}, {0.0, 0.0, 3.5}, {0.0, 0.0, -3.5}},
                                              0,
                                              1);
    ASSERT_TRUE(xenonDifluoride.has_value()) << xenonDifluoride.error().message;
    ExpectValenceCount(*xenonDifluoride, *svp, 44);

    // WF6: 74 - 60 + 6 * 9 = 68. The W d projector is the ECP channel here.
    const auto tungstenHexafluoride = MakeMolecule({{"W", 74, 0.0},
                                                    {"F", 9, 0.0},
                                                    {"F", 9, 0.0},
                                                    {"F", 9, 0.0},
                                                    {"F", 9, 0.0},
                                                    {"F", 9, 0.0},
                                                    {"F", 9, 0.0}},
                                                   {{0.0, 0.0, 0.0},
                                                    {3.5, 0.0, 0.0},
                                                    {-3.5, 0.0, 0.0},
                                                    {0.0, 3.5, 0.0},
                                                    {0.0, -3.5, 0.0},
                                                    {0.0, 0.0, 3.5},
                                                    {0.0, 0.0, -3.5}},
                                                   0,
                                                   1);
    ASSERT_TRUE(tungstenHexafluoride.has_value()) << tungstenHexafluoride.error().message;
    ExpectValenceCount(*tungstenHexafluoride, *tzvp, 68);

    // H2: no ECP on either atom in either basis, 2 electrons.
    const auto hydrogen =
        MakeMolecule({{"H", 1, 0.0}, {"H", 1, 0.0}}, {{0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}}, 0, 1);
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;
    ExpectValenceCount(*hydrogen, *svp, 2);
    ExpectValenceCount(*hydrogen, *sto3g, 2);
}

TEST(EcpElectronCountTest, RefusesAnAtomTheBasisDoesNotDescribe) {
    // An all-electron basis that stops at Kr cannot describe Xe at all: there
    // is no orbital entry and no ECP, so no Z - core value exists. The count
    // is refused by ELEMENT, not by atomic number or a generic basis error.
    const auto allElectron = LoadFamily("cc-pvdz");
    ASSERT_TRUE(allElectron.has_value()) << allElectron.error().message;
    ASSERT_EQ(allElectron->Find(54), nullptr);

    const auto xenon = MakeMolecule({{"Xe", 54, 0.0}}, {{0.0, 0.0, 0.0}}, 0, 1);
    ASSERT_TRUE(xenon.has_value()) << xenon.error().message;

    const auto count = qcx::basisset::ValenceElectronCount(*xenon, *allElectron);
    ASSERT_FALSE(count.has_value());
    EXPECT_EQ(count.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(count.error().message.find("'Xe'"), std::string::npos);
}

TEST(EcpElectronCountTest, RefusesACoreThatLeavesNoValenceElectrons) {
    // A core count that is not smaller than Z would produce a non-positive
    // "valence" count; the refusal names the element instead.
    constexpr char kBadEcp[] = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
END
ECP
H nelec 1
END
)";
    const auto basis = qcx::basisset::ParseNwchemText(kBadEcp);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const auto hydrogen = MakeMolecule({{"H", 1, 0.0}}, {{0.0, 0.0, 0.0}}, 0, 1);
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;

    const auto count = qcx::basisset::ValenceElectronCount(*hydrogen, *basis);
    ASSERT_FALSE(count.has_value());
    EXPECT_EQ(count.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(count.error().message.find("'H'"), std::string::npos);
    EXPECT_NE(count.error().message.find("smaller than Z"), std::string::npos);
}

} // namespace
