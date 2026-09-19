// Molden-format writer tests: the
// golden structural lines, the two encoded conventions (stored [GTO]
// coefficients as-is, N_l(zeta) applied at evaluation only; the [MO]
// rows permuted from the qcx m order to the molden 5D/7F slots) and the
// rejection classes (cartesian shell, shape mismatch, missing element,
// unwritable path). The AO-value reference reimplements the evaluator
// convention - the radial part (ao_evaluator.hpp RadialNormalization) and
// the Schlegel solid harmonics in closed form - the grid-module
// reference-reimplementation pattern (grid/tests/ao_evaluator_test.cpp).
// The spec's 5D/7F tables and pyscf's order_ao_index (molden_format.html,
// pyscf molden.py) are the external-convention referees, cited not
// executed.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/io/molden_export.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::io {
namespace {

constexpr double kPi = 3.14159265358979323846;

// The evaluator's per-primitive normalization (ao_evaluator.cpp
// RadialNormalization, reimplemented for the reference):
// N_l(zeta) = 2^(l+5/4) zeta^(l/2+3/4) / sqrt(2 pi^(3/2) (2l-1)!!).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double RadialReference(int l,
                       double r2,
                       const std::vector<double>& exponents,
                       const std::vector<double>& coefficients) {
    double doubleFactorial = 1.0;

    for (int k = 3; k <= 2 * l - 1; k += 2)
    {
        doubleFactorial *= static_cast<double>(k);
    }

    double value = 0.0;

    for (std::size_t i = 0; i < exponents.size(); ++i)
    {
        const double norm = std::pow(2.0, l + 1.25) * std::pow(exponents[i], 0.5 * l + 0.75) /
                            std::sqrt(2.0 * std::pow(kPi, 1.5) * doubleFactorial);
        value += coefficients[i] * norm * std::exp(-exponents[i] * r2);
    }

    return value;
}

// The Schlegel solid harmonics (solid_harmonics.hpp convention), closed
// form for l <= 3 - verified against the generated tables for l = 2, 3.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double SolidHarmonic(int l, int m, double x, double y, double z) {
    const double r2 = x * x + y * y + z * z;

    if (l == 0)
    {
        return 1.0;
    }

    if (l == 1)
    {
        if (m == -1)
        {
            return std::sqrt(3.0) * y;
        }

        if (m == 0)
        {
            return z;
        }

        return std::sqrt(3.0) * x;
    }

    if (l == 2)
    {
        if (m == -2)
        {
            return std::sqrt(3.0) * x * y;
        }

        if (m == -1)
        {
            return std::sqrt(3.0) * y * z;
        }

        if (m == 0)
        {
            return 0.5 * (3.0 * z * z - r2);
        }

        if (m == 1)
        {
            return std::sqrt(3.0) * x * z;
        }

        return 0.5 * std::sqrt(3.0) * (x * x - y * y);
    }

    if (m == -3)
    {
        return 0.25 * std::sqrt(10.0) * (3.0 * x * x * y - y * y * y);
    }

    if (m == -2)
    {
        return std::sqrt(15.0) * x * y * z;
    }

    if (m == -1)
    {
        return 0.25 * std::sqrt(6.0) * (5.0 * z * z - r2) * y;
    }

    if (m == 0)
    {
        return 0.5 * (5.0 * z * z * z - 3.0 * z * r2);
    }

    if (m == 1)
    {
        return 0.25 * std::sqrt(6.0) * (5.0 * z * z - r2) * x;
    }

    if (m == 2)
    {
        return 0.5 * std::sqrt(15.0) * (x * x - y * y) * z;
    }

    return 0.25 * std::sqrt(10.0) * (x * x * x - 3.0 * x * y * y);
}

// One hydrogen at the origin (fixtures only offer H2/He; the
// grid/tests/ao_evaluator_test.cpp pattern).
qcx::Result<qcx::molecule::Molecule> MakeSingleHydrogen() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}}, std::move(*coordinates), 0, 1);
}

// Two hydrogens: the first at the origin, the second at the given Bohr
// coordinates (canonical order keeps the origin atom first).
// (x, y, z) are the second hydrogen's Bohr coordinates - one position
// triple, always passed together.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
qcx::Result<qcx::molecule::Molecule> MakeHydrogenPair(double x, double y, double z) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = x;
    (*coordinates)(1, 1) = y;
    (*coordinates)(1, 2) = z;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

std::filesystem::path TempPath(std::string_view name) {
    return std::filesystem::temp_directory_path() / name;
}

std::string ReadTextFile(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

std::vector<std::string> SplitLines(std::string_view text) {
    std::vector<std::string> lines;
    std::size_t start = 0;

    while (start < text.size())
    {
        const std::size_t end = text.find('\n', start);

        if (end == std::string_view::npos)
        {
            lines.emplace_back(text.substr(start));
            break;
        }

        lines.emplace_back(text.substr(start, end - start));
        start = end + 1;
    }

    return lines;
}

// The written [GTO] "exponent coefficient" rows: every line between the
// [GTO] and [MO] headers that is neither an element line (uppercase
// symbol) nor a shell line (lowercase label) nor blank.
std::vector<double> WrittenGtoCoefficients(const std::vector<std::string>& lines) {
    std::vector<double> coefficients;
    bool inGto = false;

    for (const auto& line : lines)
    {
        if (line == "[MO]")
        {
            break;
        }

        if (line == "[GTO]")
        {
            inGto = true;
            continue;
        }

        if (!inGto || line.empty())
        {
            continue;
        }

        const unsigned char first = static_cast<unsigned char>(line[0]);

        if (std::isupper(first) || std::islower(first))
        {
            continue;
        }

        std::istringstream row(line);
        double exponent = 0.0;
        double coefficient = 0.0;
        row >> exponent >> coefficient;
        coefficients.push_back(coefficient);
    }

    return coefficients;
}

// One written [MO] orbital block: the record values and the 1-based
// coefficient rows.
struct MoBlock {
    std::string spin;
    double energy = 0.0;
    double occupation = 0.0;
    std::vector<std::pair<int, double>> rows;
};

std::vector<MoBlock> ParseMoBlocks(const std::vector<std::string>& lines) {
    std::vector<MoBlock> blocks;
    bool inMo = false;
    MoBlock current;
    bool inRows = false;

    for (const auto& line : lines)
    {
        if (line == "[MO]")
        {
            inMo = true;
            continue;
        }

        if (!inMo)
        {
            continue;
        }

        if (line.empty())
        {
            if (inRows)
            {
                blocks.push_back(std::move(current));
                current = MoBlock{};
                inRows = false;
            }

            continue;
        }

        if (line.rfind("Sym= ", 0) == 0)
        {
            continue;
        }

        if (line.rfind("Ene= ", 0) == 0)
        {
            std::istringstream(line.substr(5)) >> current.energy;
            continue;
        }

        if (line.rfind("Spin= ", 0) == 0)
        {
            current.spin = line.substr(6);
            continue;
        }

        if (line.rfind("Occup= ", 0) == 0)
        {
            std::istringstream(line.substr(7)) >> current.occupation;
            continue;
        }

        inRows = true;
        std::istringstream row(line);
        int index = 0;
        double coefficient = 0.0;
        row >> index >> coefficient;
        current.rows.emplace_back(index, coefficient);
    }

    return blocks;
}

constexpr std::string_view kSto3gH = R"(
BASIS "ao basis" SPHERICAL PRINT
H S
3.42525091 0.15432897
0.62391373 0.53532814
0.16885540 0.44463454
END
)";

TEST(MoldenExportTest, GoldenStructuralPins) {
    const auto basis = qcx::basisset::ParseNwchemText(kSto3gH);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeHydrogenPair(1.4, 0.5, 2.0);
    ASSERT_TRUE(molecule.has_value());

    const Eigen::MatrixXd coefficients = (Eigen::MatrixXd(2, 2) << 1.0, 0.25, 0.5, 0.75).finished();
    const Eigen::VectorXd energies = (Eigen::VectorXd(2) << 0.5, 1.5).finished();
    const Eigen::VectorXd occupations = (Eigen::VectorXd(2) << 2.0, 0.0).finished();
    const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
    const std::array<MoldenMolecularOrbitals, 1> blocks{block};
    const auto path = TempPath("qcx_molden_golden_structural.molden");

    const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto lines = SplitLines(ReadTextFile(path));
    std::filesystem::remove(path);
    ASSERT_EQ(lines.size(), 34u);

    std::size_t i = 0;
    ASSERT_EQ(lines[i++], "[Molden Format]");
    ASSERT_EQ(lines[i++], "[Atoms] (AU)");
    ASSERT_EQ(lines[i++], "H 1 1 0.000000000000 0.000000000000 0.000000000000");
    ASSERT_EQ(lines[i++], "H 2 1 1.400000000000 0.500000000000 2.000000000000");
    ASSERT_EQ(lines[i++], "[5D]");
    ASSERT_EQ(lines[i++], "[7F]");
    ASSERT_EQ(lines[i++], "[GTO]");

    // Atom 1 block: element line, shell line, three STO-3G primitive rows
    // (exponent prefix - free-format doubles, so 0.16885540 prints as
    // 0.1688554 at 14 significant digits; the coefficients carry the
    // parse-time unit-norm scale, pinned at value level by the
    // closed-form test).
    ASSERT_EQ(lines[i++], "H 0");
    ASSERT_EQ(lines[i++], "s 3 1.00");
    ASSERT_EQ(lines[i++].find("3.42525091 "), 0u);
    ASSERT_EQ(lines[i++].find("0.62391373 "), 0u);
    ASSERT_EQ(lines[i++].find("0.1688554 "), 0u);
    ASSERT_EQ(lines[i++], "");

    // Atom 2 block.
    ASSERT_EQ(lines[i++], "H 0");
    ASSERT_EQ(lines[i++], "s 3 1.00");
    ASSERT_EQ(lines[i++].find("3.42525091 "), 0u);
    ASSERT_EQ(lines[i++].find("0.62391373 "), 0u);
    ASSERT_EQ(lines[i++].find("0.1688554 "), 0u);
    ASSERT_EQ(lines[i++], "");

    // Orbital 0: the records (space after =), then all n rows with 1-based
    // indices, then the block separator.
    ASSERT_EQ(lines[i++], "[MO]");
    ASSERT_EQ(lines[i++], "Sym= A");
    ASSERT_EQ(lines[i++], "Ene= 0.5");
    ASSERT_EQ(lines[i++], "Spin= Alpha");
    ASSERT_EQ(lines[i++], "Occup= 2");
    ASSERT_EQ(lines[i++], "1 1");
    ASSERT_EQ(lines[i++], "2 0.5");
    ASSERT_EQ(lines[i++], "");

    // Orbital 1.
    ASSERT_EQ(lines[i++], "Sym= A");
    ASSERT_EQ(lines[i++], "Ene= 1.5");
    ASSERT_EQ(lines[i++], "Spin= Alpha");
    ASSERT_EQ(lines[i++], "Occup= 0");
    ASSERT_EQ(lines[i++], "1 0.25");
    ASSERT_EQ(lines[i++], "2 0.75");
    ASSERT_EQ(lines[i], "");
}

TEST(MoldenExportTest, WrittenCoefficientsMatchEvaluatorConvention) {
    // Unit-norm contractions (single primitive, coefficient 1.0: the
    // parse rescale s = 1 exactly), so the written coefficients equal the
    // raw BSE values and the file alone - stored coefficients times the
    // reimplemented N_l times the Schlegel harmonic - reproduces the
    // evaluator's AO values at 1e-12. Any normalization tampering in the
    // writer (a divide-by-N_l, say) fails this pin loudly.
    constexpr std::string_view kBasisText = R"(
BASIS "ao basis" SPHERICAL PRINT
H S
1.0 1.0
H P
0.5 1.0
END
)";
    const auto basis = qcx::basisset::ParseNwchemText(kBasisText);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    ASSERT_EQ(element->shells.size(), 2u);
    const auto& sExponents = element->shells[0].exponents;
    const auto& sStored = element->shells[0].coefficients[0];
    const auto& pExponents = element->shells[1].exponents;
    const auto& pStored = element->shells[1].coefficients[0];

    const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(4, 4);
    const Eigen::VectorXd energies = Eigen::VectorXd::Zero(4);
    const Eigen::VectorXd occupations = Eigen::VectorXd::Ones(4);
    const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
    const std::array<MoldenMolecularOrbitals, 1> blocks{block};
    const auto path = TempPath("qcx_molden_closed_form.molden");

    const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto lines = SplitLines(ReadTextFile(path));
    std::filesystem::remove(path);
    const auto written = WrittenGtoCoefficients(lines);
    ASSERT_EQ(written.size(), 2u);

    // The s = 1-class exactness: stored == raw BSE coefficients.
    EXPECT_NEAR(written[0], 1.0, 1e-12);
    EXPECT_NEAR(written[1], 1.0, 1e-12);
    const std::vector<double> writtenS{written[0]};
    const std::vector<double> writtenP{written[1]};

    // One off-origin point where the four functions take distinct values.
    constexpr double kX = 0.3;
    constexpr double kY = -0.2;
    constexpr double kZ = 0.4;
    const double r2 = kX * kX + kY * kY + kZ * kZ;

    // s (m = 0).
    EXPECT_NEAR(RadialReference(0, r2, sExponents, writtenS) * SolidHarmonic(0, 0, kX, kY, kZ),
                RadialReference(0, r2, sExponents, sStored) * SolidHarmonic(0, 0, kX, kY, kZ),
                1e-12);

    // The p shell in the 5D/7F slots (0, +1, -1): written row 1 = m = 0
    // (pz), row 2 = m = +1 (px), row 3 = m = -1 (py).
    EXPECT_NEAR(RadialReference(1, r2, pExponents, writtenP) * SolidHarmonic(1, 0, kX, kY, kZ),
                RadialReference(1, r2, pExponents, pStored) * SolidHarmonic(1, 0, kX, kY, kZ),
                1e-12);
    EXPECT_NEAR(RadialReference(1, r2, pExponents, writtenP) * SolidHarmonic(1, 1, kX, kY, kZ),
                RadialReference(1, r2, pExponents, pStored) * SolidHarmonic(1, 1, kX, kY, kZ),
                1e-12);
    EXPECT_NEAR(RadialReference(1, r2, pExponents, writtenP) * SolidHarmonic(1, -1, kX, kY, kZ),
                RadialReference(1, r2, pExponents, pStored) * SolidHarmonic(1, -1, kX, kY, kZ),
                1e-12);
}

TEST(MoldenExportTest, SphericalOrderingMatchesMoldenTables) {
    // One d shell and one f shell, unit-norm single primitives.
    constexpr std::string_view kBasisText = R"(
BASIS "ao basis" SPHERICAL PRINT
H D
1.0 1.0
H F
0.7 1.0
END
)";
    // The spec's 5D/7F tables (molden_format.html; pyscf's
    // order_ao_index), literal: the molden slot (1-based) of the
    // spherical function with m = -l..+l.
    constexpr std::array<int, 5> kDFunctionSlots{5, 3, 1, 2, 4};
    constexpr std::array<int, 7> kFFunctionSlots{7, 5, 3, 1, 2, 4, 6};

    // The inverse of the literal tables: the m of one slot.
    // Capture the tables explicitly: Clang requires capture of local
    // constexpr variables (strict odr-use); GCC/MSVC accept them bare.
    const auto mOfSlot = [kDFunctionSlots, kFFunctionSlots](int slot, int l) {
        if (l == 2)
        {
            for (int i = 0; i < 5; ++i)
            {
                if (kDFunctionSlots[i] == slot)
                {
                    return i - 2;
                }
            }
        } else
        {
            for (int i = 0; i < 7; ++i)
            {
                if (kFFunctionSlots[i] == slot)
                {
                    return i - 3;
                }
            }
        }

        return 0;
    };

    const auto basis = qcx::basisset::ParseNwchemText(kBasisText);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const auto* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    ASSERT_EQ(element->shells.size(), 2u);
    const auto& dExponents = element->shells[0].exponents;
    const auto& dStored = element->shells[0].coefficients[0];
    const auto& fExponents = element->shells[1].exponents;
    const auto& fStored = element->shells[1].coefficients[0];

    // Identity MO rows: orbital c is basis function c, so the written
    // position of its 1.0 pins the slot of every function.
    const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(12, 12);
    const Eigen::VectorXd energies = Eigen::VectorXd::Zero(12);
    const Eigen::VectorXd occupations = Eigen::VectorXd::Ones(12);
    const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
    const std::array<MoldenMolecularOrbitals, 1> blocks{block};
    const auto path = TempPath("qcx_molden_ordering.molden");

    const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto lines = SplitLines(ReadTextFile(path));
    std::filesystem::remove(path);
    const auto writtenGto = WrittenGtoCoefficients(lines);
    ASSERT_EQ(writtenGto.size(), 2u);
    EXPECT_NEAR(writtenGto[0], 1.0, 1e-12);
    EXPECT_NEAR(writtenGto[1], 1.0, 1e-12);

    const auto parsed = ParseMoBlocks(lines);
    ASSERT_EQ(parsed.size(), 12u);
    ASSERT_EQ(parsed[0].spin, "Alpha");

    // The row-position pin: the primary exact check of the permutation.
    // The rows are global over the written file - the d block occupies
    // rows 1..5 in 5D order, the f block rows 6..12 in 7F order.
    for (int orbital = 0; orbital < 12; ++orbital)
    {
        ASSERT_EQ(parsed[orbital].rows.size(), 12u);
        int writtenRow = 0;

        for (const auto& [index, coefficient] : parsed[orbital].rows)
        {
            if (std::abs(coefficient - 1.0) < 1e-12)
            {
                writtenRow = index;
            } else
            {
                EXPECT_NEAR(coefficient, 0.0, 1e-12);
            }
        }

        const int expectedSlot =
            (orbital < 5) ? kDFunctionSlots[orbital] : 5 + kFFunctionSlots[orbital - 5];
        const int m = (orbital < 5) ? orbital - 2 : orbital - 8;
        EXPECT_EQ(writtenRow, expectedSlot) << "orbital " << orbital << " (qcx m " << m << ")";
    }

    // The parse-back value pin: recompute the twelve functions at a point
    // from the written rows (each slot's m per the spec tables) and match
    // the evaluator-convention values of the qcx functions at 1e-12 - a
    // wrong permutation or sign error flips the match.
    constexpr double kX = 0.25;
    constexpr double kY = -0.4;
    constexpr double kZ = 0.35;
    const double r2 = kX * kX + kY * kY + kZ * kZ;

    for (int slot = 1; slot <= 12; ++slot)
    {
        // The orbital whose 1.0 sits at this written row.
        int orbital = -1;

        for (int candidate = 0; candidate < 12; ++candidate)
        {
            for (const auto& [index, coefficient] : parsed[candidate].rows)
            {
                if (index == slot && std::abs(coefficient - 1.0) < 1e-12)
                {
                    orbital = candidate;
                }
            }
        }

        ASSERT_NE(orbital, -1);

        const int l = (slot <= 5) ? 2 : 3;
        // The tables are per-shell, the rows global: the f block's slots
        // start again at 1 once the 5 d rows are consumed.
        const int m = mOfSlot((slot <= 5) ? slot : slot - 5, l);
        const auto& exponents = (slot <= 5) ? dExponents : fExponents;
        const auto& stored = (slot <= 5) ? dStored : fStored;
        const std::vector<double> written{(slot <= 5) ? writtenGto[0] : writtenGto[1]};

        // Written side: the slot's m (from the tables) with the written
        // [GTO] coefficients.
        const double writtenValue =
            RadialReference(l, r2, exponents, written) * SolidHarmonic(l, m, kX, kY, kZ);
        // In-memory side: the qcx function at that slot with its own m.
        const int orbitalM = (orbital < 5) ? orbital - 2 : orbital - 8;
        const double storedValue =
            RadialReference(l, r2, exponents, stored) * SolidHarmonic(l, orbitalM, kX, kY, kZ);
        EXPECT_NEAR(writtenValue, storedValue, 1e-12) << "slot " << slot;
    }
}

TEST(MoldenExportTest, CartesianShellIsRejected) {
    constexpr std::string_view kCartesianText = R"(
BASIS "ao basis" CARTESIAN PRINT
H D
1.0 1.0
END
)";
    const auto basis = qcx::basisset::ParseNwchemText(kCartesianText);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(6, 6);
    const Eigen::VectorXd energies = Eigen::VectorXd::Zero(6);
    const Eigen::VectorXd occupations = Eigen::VectorXd::Ones(6);
    const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
    const std::array<MoldenMolecularOrbitals, 1> blocks{block};
    const auto path = TempPath("qcx_molden_cartesian_rejected.molden");

    const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(MoldenExportTest, ShapeMismatchIsRejected) {
    constexpr std::string_view kBasisText = R"(
BASIS "ao basis" SPHERICAL PRINT
H D
1.0 1.0
H F
0.7 1.0
END
)";
    const auto basis = qcx::basisset::ParseNwchemText(kBasisText);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    // Coefficients not n x n (n = 12 here).
    {
        const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(4, 4);
        const Eigen::VectorXd energies = Eigen::VectorXd::Zero(4);
        const Eigen::VectorXd occupations = Eigen::VectorXd::Ones(4);
        const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
        const std::array<MoldenMolecularOrbitals, 1> blocks{block};
        const auto path = TempPath("qcx_molden_shape_coefficients.molden");

        const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_FALSE(std::filesystem::exists(path));
    }

    // Energies not n.
    {
        const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(12, 12);
        const Eigen::VectorXd energies = Eigen::VectorXd::Zero(4);
        const Eigen::VectorXd occupations = Eigen::VectorXd::Ones(12);
        const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
        const std::array<MoldenMolecularOrbitals, 1> blocks{block};
        const auto path = TempPath("qcx_molden_shape_energies.molden");

        const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_FALSE(std::filesystem::exists(path));
    }

    // Occupations not n.
    {
        const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(12, 12);
        const Eigen::VectorXd energies = Eigen::VectorXd::Zero(12);
        const Eigen::VectorXd occupations = Eigen::VectorXd::Ones(4);
        const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
        const std::array<MoldenMolecularOrbitals, 1> blocks{block};
        const auto path = TempPath("qcx_molden_shape_occupations.molden");

        const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_FALSE(std::filesystem::exists(path));
    }
}

TEST(MoldenExportTest, MissingElementIsRejected) {
    constexpr std::string_view kBasisText = R"(
BASIS "ao basis" SPHERICAL PRINT
H S
1.0 1.0
END
)";
    const auto basis = qcx::basisset::ParseNwchemText(kBasisText);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    // One H with the H basis plus one He the basis does not cover: the
    // validation reports the missing element before the file opens.
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});
    ASSERT_TRUE(coordinates.has_value());
    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    const auto molecule = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"He", 2, 0.0}},
        std::move(*coordinates),
        0,
        1);
    ASSERT_TRUE(molecule.has_value());

    const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::VectorXd energies = Eigen::VectorXd::Zero(2);
    const Eigen::VectorXd occupations = Eigen::VectorXd::Ones(2);
    const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
    const std::array<MoldenMolecularOrbitals, 1> blocks{block};
    const auto path = TempPath("qcx_molden_missing_element.molden");

    const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(MoldenExportTest, UnwritablePathFails) {
    const auto basis = qcx::basisset::ParseNwchemText(kSto3gH);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    const Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(1, 1);
    const Eigen::VectorXd energies = Eigen::VectorXd::Zero(1);
    const Eigen::VectorXd occupations = Eigen::VectorXd::Ones(1);
    const MoldenMolecularOrbitals block{"Alpha", coefficients, energies, occupations};
    const std::array<MoldenMolecularOrbitals, 1> blocks{block};
    // A path inside a directory that does not exist.
    const auto path = TempPath("qcx_molden_no_such_dir") / "out.molden";

    const auto result = WriteMoldenFile(*molecule, *basis, blocks, path.string());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kIOError);
}

} // namespace
} // namespace qcx::io
