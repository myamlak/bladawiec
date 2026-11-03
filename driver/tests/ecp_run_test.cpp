// The ECP acceptance matrix at the driver boundary.
//
// The ECP data is the def2 family: [Weigend2005] pairs the orbital basis with
// the ECP, [Peterson2003] fits the Xe 28-core potential and [Andrae1990] the
// W 60-core one. Until the operator lands, every run on an ECP-carrying basis
// is refused by ValidateInput; the first cell below pins that refusal on the
// exact fixtures, and the acceptance cells skip with the reason that names
// what will satisfy them.
//
// Fixtures, by role and cost (nbf = contracted basis functions, Release-only
// is the Debug fast-smoke gate):
//
//   Xe / def2-SVP     50 nbf, 26 e, 13 occupied  - the electron count and
//                     the +(Z - N_core)/r tail; cheapest atomic cell.
//   XeH+ / def2-SVP   55 nbf, 26 e, 13 occupied  - the ECP at an off-centre
//                     atom (the potential must sit on Xe, not the origin).
//   XeF2 / def2-SVP   78 nbf, 44 e, 22 occupied  - the p projector channels.
//   W / def2-TZVP     40 nbf, 14 e,  7 occupied  - the 60-core count at a
//                     transition metal; the d projector.
//   WF6 / def2-TZVP  226 nbf, 68 e, 34 occupied  - the d projector under Oh
//                     symmetry; Release-only (about (226/50)^4 ~ 400 times
//                     the Xe cell's operation count).
//
// The independent-code reference is OWED: PySCF is not installed in this
// environment, so the cross-check below records the exact recipe instead of
// a number. No published total atomic ECP energy is used as an acceptance
// target: the source papers tabulate fit targets and basis accuracy, not
// absolute SCF energies, and a number invented for a test would be worse
// than an honest skip.
//
// A scalar-relativistic ECP's total energy differs from the all-electron
// value by physics, not by an error: the fitted potential absorbs the
// relativistic contraction of the core and the Darwin/mass-velocity terms the
// all-electron non-relativistic Hamiltonian cannot see. A cell that compares
// an ECP energy with an all-electron reference is therefore rejected by
// construction; the valid reference is the same ECP block in an independent
// code.

#include "fast_test_mode.hpp"
#include "qcx/driver/run_driver.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/run_input.hpp"

#include <array>
#include <expected>
#include <gtest/gtest.h>
#include <string>

namespace {

using qcx::driver::RunDriver;
using qcx::io::ParseRunInput;

const char* kXenonToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["Xe", 0.0, 0.0, 0.0],
]

[basis]
orbital = "def2-svp"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[scf]
use_diis = true
)";

const char* kXenonHydrideToml = R"(
[molecule]
charge = 1
multiplicity = 1
atoms = [
    ["Xe", 0.0, 0.0, 0.0],
    ["H", 0.0, 0.0, 1.6],
]

[basis]
orbital = "def2-svp"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[scf]
use_diis = true
)";

const char* kXenonDifluorideToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["Xe", 0.0, 0.0, 0.0],
    ["F", 0.0, 0.0, 2.0],
    ["F", 0.0, 0.0, -2.0],
]

[basis]
orbital = "def2-svp"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[scf]
use_diis = true
)";

const char* kTungstenToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["W", 0.0, 0.0, 0.0],
]

[basis]
orbital = "def2-tzvp"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[scf]
use_diis = true
)";

const char* kTungstenHexafluorideToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["W", 0.0, 0.0, 0.0],
    ["F", 2.0, 0.0, 0.0],
    ["F", -2.0, 0.0, 0.0],
    ["F", 0.0, 2.0, 0.0],
    ["F", 0.0, -2.0, 0.0],
    ["F", 0.0, 0.0, 2.0],
    ["F", 0.0, 0.0, -2.0],
]

[basis]
orbital = "def2-tzvp"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[scf]
use_diis = true
)";

qcx::Result<std::string> RunToml(const char* toml) {
    const auto input = ParseRunInput(toml);

    if (!input.has_value())
    {
        return std::unexpected(input.error());
    }

    return RunDriver(*input);
}

struct Fixture {
    const char* name; ///< The row name for a failure message.
    const char* toml; ///< The deck.
    const char* element; ///< The element whose ECP refuses it today.
};

const std::array<Fixture, 5> kFixtures{{
    {"Xe atom", kXenonToml, "Xe"},
    {"XeH+", kXenonHydrideToml, "Xe"},
    {"XeF2", kXenonDifluorideToml, "Xe"},
    {"W atom", kTungstenToml, "W"},
    {"WF6", kTungstenHexafluorideToml, "W"},
}};

TEST(EcpRunTest, EveryEcpFixtureIsRefusedUntilTheOperatorLands) {
    // The refusal is the current, correct behaviour: the one-electron
    // Hamiltonian has no ECP term, so an ECP deck would run all-electron
    // with a valence-only basis. Each fixture is refused by name, with the
    // element's core count in the message. When the operator lands this cell
    // must be inverted deliberately: the same decks must run, and the
    // acceptance cells below must stop skipping.
    for (const Fixture& fixture : kFixtures)
    {
        SCOPED_TRACE(fixture.name);
        const auto result = RunToml(fixture.toml);
        ASSERT_FALSE(result.has_value()) << fixture.name;
        EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(result.error().message.find("effective-core potential"), std::string::npos);
        EXPECT_NE(result.error().message.find(fixture.element), std::string::npos);
        EXPECT_NE(result.error().message.find("electron count would be wrong"), std::string::npos);
    }
}

TEST(EcpRunTest, XenonAtomReportsTwentySixValenceElectrons) {
    // Satisfied when: the refusal is replaced by the wired operator, the run
    // converges (or reports its non-convergence), and the emitted document's
    // populations block sums to 26 electrons:
    //   sum(document["populations"]["mulliken"]["total"]) == 26 within 1e-8.
    // The density trace is the run's reported electron count; a wrong count
    // either fails the closed-shell occupancy (54 electrons into 25 spatial
    // orbitals) or leaves a trace of the wrong integer. The same cell also
    // pins the long-range tail: without the +(Z - N_core)/r term the energy
    // is that of a 54-electron all-electron problem and will not match the
    // independent-code cross-check below.
    GTEST_SKIP() << "ECP operator not implemented; ValidateInput refuses the deck";
}

TEST(EcpRunTest, XenonHydrideCationAppliesTheEcpAtTheOffCentreAtom) {
    // Satisfied when: XeH+ (26 electrons, closed shell) runs with the ECP
    // potential at Xe, not at the origin. The density trace must sum to 26;
    // a potential placed at the wrong centre breaks the molecular symmetry
    // and the SCF walk, which this cell reports as a failed convergence or a
    // count mismatch.
    GTEST_SKIP() << "ECP operator not implemented; ValidateInput refuses the deck";
}

TEST(EcpRunTest, XenonDifluorideExercisesThePProjectorChannels) {
    // Satisfied when: XeF2 (44 electrons) runs with the S, P and D projector
    // channels active. The trace must sum to 44. A projector term applied at
    // the wrong l, or double-counted with the local channel, moves the
    // orbital energies and usually the convergence; the unit-level selection
    // rule and quadrature cells in integrals/tests/ecp_test.cpp pin the
    // channel algebra itself.
    GTEST_SKIP() << "ECP operator not implemented; ValidateInput refuses the deck";
}

TEST(EcpRunTest, TungstenAtomReportsFourteenValenceElectrons) {
    // Satisfied when: W / def2-TZVP (74 - 60 = 14 electrons) runs and the
    // density trace sums to 14. The W 60-core potential is the second
    // core-count convention in the corpus; the Xe cell pins 28.
    GTEST_SKIP() << "ECP operator not implemented; ValidateInput refuses the deck";
}

TEST(EcpRunTest, TungstenHexafluorideExercisesTheDProjectorUnderSymmetry) {
    // Release-only: 226 nbf and 68 electrons; about (226/50)^4 ~ 400 times
    // the Xe atom's direct-SCF operation count. Satisfied when: WF6 runs and
    // the density trace sums to 68 under the octahedral symmetry the driver
    // resolves. The d projector of W is the only non-local channel in this
    // fixture; the f shells are orbital polarization functions, all-electron
    // on F.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "Release-only: WF6 is the heaviest acceptance cell";
    }

    GTEST_SKIP() << "ECP operator not implemented; ValidateInput refuses the deck";
}

TEST(EcpRunTest, IndependentCodeCrossCheckIsOwed) {
    // OWED: no independent code is installed here (python -c "import pyscf"
    // raises ModuleNotFoundError), so no number is quoted. The recipe must
    // use the SAME vendored NWChem block, not PySCF's built-in def2-SVP:
    //
    //   1. Read data/basis/def2-svp/Xe.nwchem. The orbital shells become
    //      PySCF's basis dict {symbol: [[l, [exp, coef, ...]], ...]} and the
    //      ECP block becomes the ecp dict {symbol: [[l, [exp, coef], ...],
    //      ...]} with l = -1 for the `ul` local channel and l = 0, 1, 2 for
    //      S, P, D. The channel blocks already carry the -V_loc terms the
    //      NWChem convention stores, so no further conversion is applied.
    //   2. from pyscf import gto, scf
    //      mol = gto.M(atom=[('Xe', (0.0, 0.0, 0.0))], unit='Bohr',
    //                  basis=basis_dict, ecp=ecp_dict, verbose=0)
    //      mf = scf.RHF(mol).run()
    //      assert mol.nelectron == 26
    //      print(mf.e_tot)
    //   3. Compare mf.e_tot with this run's total_energy_hartree at 1e-8.
    //
    // `pyscf.gto.basis.parse_nwchem` may parse the file directly; its exact
    // entry point and return shape are UNVERIFIED here because the package is
    // absent. The dict conversion above is the fallback if it does not.
    GTEST_SKIP() << "OWED: PySCF is not installed; the cross-check recipe is in the comment";
}

} // namespace
