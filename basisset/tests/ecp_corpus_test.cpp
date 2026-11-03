// The ECP corpus inventory and the basis-dependent electron count.
//
// The vendored def2 families carry the only ECP data in data/basis. Their
// `SYMBOL nelec N` metadata names N, the number of CORE electrons the ECP
// replaces - not the valence count. Xe (Z = 54, nelec 28) therefore has 26
// valence electrons and W (Z = 74, nelec 60) has 14. The corpus references
// are [Weigend2005] for the def2 pairing, [Peterson2003] for Xe and
// [Andrae1990] for W.
//
// The parser keeps only the (Z, core count) stub; the radial terms are a
// later step, so nothing here reads them. The electron-count cells recompute
// sum(Z - core) - charge from that stub. They are written against the
// intended public entry `qcx::basisset::ValenceElectronCount(molecule,
// basis)`, which does not exist yet: the local helper below is the temporary
// oracle and must be replaced by the real entry when it lands. The count is
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
using qcx::basisset::EcpDefinition;
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

std::optional<int> CoreCount(const BasisSet& basis, int atomicNumber) {
    for (const EcpDefinition& ecp : basis.Ecps())
    {
        if (ecp.atomicNumber == atomicNumber)
        {
            return ecp.coreElectronCount;
        }
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

bool HasAtom(const Molecule& molecule, int atomicNumber) {
    for (const Atom& atom : molecule.Atoms())
    {
        if (atom.atomicNumber == atomicNumber)
        {
            return true;
        }
    }

    return false;
}

/// The intended electron count, sum(Z - core) - charge, computed from the
/// parsed ECP stubs. TODO(ECP): `qcx::basisset::ValenceElectronCount(const
/// qcx::molecule::Molecule&, const qcx::basisset::BasisSet&)` is the public
/// entry this helper stands in for; when it lands, every cell below must call
/// it instead and the helper must be deleted.
int TemporaryValenceElectronCount(const Molecule& molecule, const BasisSet& basis) {
    int count = -molecule.Charge();

    for (const Atom& atom : molecule.Atoms())
    {
        count += atom.atomicNumber;
    }

    for (const EcpDefinition& ecp : basis.Ecps())
    {
        if (HasAtom(molecule, ecp.atomicNumber))
        {
            count -= ecp.coreElectronCount;
        }
    }

    return count;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- charge and multiplicity.
qcx::Result<Molecule> MakeMolecule(std::vector<Atom> atoms,
                                   std::vector<std::vector<double>> coordinates,
                                   int charge,
                                   int multiplicity) {
    return Molecule::Create(std::move(atoms), MakeCoordinates(coordinates), charge, multiplicity);
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

TEST(EcpCorpusTest, LanthanideElementsCarryTheCore28Stub) {
    // Ce-Lu are the only core-28 group beyond Rb-Xe, and their projectors
    // reach G (l = 4). The parser records only the core count, so this cell
    // pins the corpus coverage rather than the channel table; the channel
    // structure is read by the integral step when the radial terms are
    // retained.
    const auto basis = LoadFamily("def2-svp");
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    for (int z = 58; z <= 71; ++z)
    {
        SCOPED_TRACE(z);
        ExpectCore(*basis, z, 28);
    }

    const auto diffuse = LoadFamily("def2-tzvpd");
    ASSERT_TRUE(diffuse.has_value()) << diffuse.error().message;
    ExpectNoCore(*diffuse, 58); // no lanthanides vendored in this family.
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
    // The intended public entry is basis-dependent:
    // qcx::basisset::ValenceElectronCount(molecule, basis). The temporary
    // oracle above reads the same stub it will read.
    const auto svp = LoadFamily("def2-svp");
    const auto tzvp = LoadFamily("def2-tzvp");
    const auto sto3g = LoadFamily("sto-3g");
    ASSERT_TRUE(svp.has_value()) << svp.error().message;
    ASSERT_TRUE(tzvp.has_value()) << tzvp.error().message;
    ASSERT_TRUE(sto3g.has_value()) << sto3g.error().message;

    // Xe atom: 54 - 28 = 26 valence electrons.
    const auto xenon = MakeMolecule({{"Xe", 54, 0.0}}, {{0.0, 0.0, 0.0}}, 0, 1);
    ASSERT_TRUE(xenon.has_value()) << xenon.error().message;
    EXPECT_EQ(TemporaryValenceElectronCount(*xenon, *svp), 26);
    EXPECT_EQ(TemporaryValenceElectronCount(*xenon, *sto3g), 54);

    // The molecule alone cannot know the core (it has no basis); this method
    // sums Z and must not be used to size an ECP run's occupancy.
    EXPECT_EQ(xenon->ElectronCount(), 54);

    // W atom: 74 - 60 = 14 valence electrons.
    const auto tungsten = MakeMolecule({{"W", 74, 0.0}}, {{0.0, 0.0, 0.0}}, 0, 1);
    ASSERT_TRUE(tungsten.has_value()) << tungsten.error().message;
    EXPECT_EQ(TemporaryValenceElectronCount(*tungsten, *tzvp), 14);

    // XeH+: 54 - 28 + 1 - 1 = 26. The off-centre ECP is the Xe atom's.
    const auto xenonHydride =
        MakeMolecule({{"Xe", 54, 0.0}, {"H", 1, 0.0}}, {{0.0, 0.0, 0.0}, {0.0, 0.0, 3.0}}, 1, 1);
    ASSERT_TRUE(xenonHydride.has_value()) << xenonHydride.error().message;
    EXPECT_EQ(TemporaryValenceElectronCount(*xenonHydride, *svp), 26);

    // XeF2: 54 - 28 + 2 * 9 = 44. Fluorine is all-electron in def2-SVP.
    const auto xenonDifluoride = MakeMolecule({{"Xe", 54, 0.0}, {"F", 9, 0.0}, {"F", 9, 0.0}},
                                              {{0.0, 0.0, 0.0}, {0.0, 0.0, 3.5}, {0.0, 0.0, -3.5}},
                                              0,
                                              1);
    ASSERT_TRUE(xenonDifluoride.has_value()) << xenonDifluoride.error().message;
    EXPECT_EQ(TemporaryValenceElectronCount(*xenonDifluoride, *svp), 44);

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
    EXPECT_EQ(TemporaryValenceElectronCount(*tungstenHexafluoride, *tzvp), 68);

    // H2 in def2-SVP: no ECP stub on either atom, 2 electrons.
    const auto hydrogen =
        MakeMolecule({{"H", 1, 0.0}, {"H", 1, 0.0}}, {{0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}}, 0, 1);
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;
    EXPECT_EQ(TemporaryValenceElectronCount(*hydrogen, *svp), 2);
}

} // namespace
