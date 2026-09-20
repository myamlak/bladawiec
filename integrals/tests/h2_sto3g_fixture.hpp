#pragma once

// Committed mpmath reference-grid loader (tools/gen_integrals_reference.py).
// The H2/STO-3G construction fixtures that used to live here now live in
// tests/fixtures/h2_sto3g.hpp (shared with the scf tests); this header keeps
// only the integrals-specific CSV part.

#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

namespace qcx::integrals::test {

// One row of the committed reference grid.
struct ReferenceValue {
    std::string kind; // "S", "T", "V", or "ERI".
    int i;
    int j;
    int k; // -1 for one-electron rows.
    int l; // -1 for one-electron rows.
    double value;
};

// Loads QcxIntegralsDataDir/h2_sto3g_reference.csv (28 rows).
inline std::vector<ReferenceValue> LoadReferenceValues() {
    const std::string path = std::string(QcxIntegralsDataDir) + "/h2_sto3g_reference.csv";
    std::ifstream file(path);

    if (!file)
    {
        ADD_FAILURE() << "missing reference data: " << path;
        return {};
    }

    std::vector<ReferenceValue> rows;
    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        ReferenceValue row{};
        std::getline(ss, cell, ',');
        row.kind = cell;
        std::getline(ss, cell, ',');
        row.i = std::stoi(cell);
        std::getline(ss, cell, ',');
        row.j = std::stoi(cell);
        std::getline(ss, cell, ',');
        row.k = cell.empty() ? -1 : std::stoi(cell);
        std::getline(ss, cell, ',');
        row.l = cell.empty() ? -1 : std::stoi(cell);
        std::getline(ss, cell, ',');
        row.value = std::strtod(cell.c_str(), nullptr);
        rows.push_back(row);
    }

    return rows;
}

} // namespace qcx::integrals::test
