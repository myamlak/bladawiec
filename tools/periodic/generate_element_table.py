#!/usr/bin/env python3
"""Generate molecule/include/qcx/molecule/elements.hpp from elements_data.json.

Deterministic and offline: the committed JSON is the source of truth, so the
output only changes when the JSON does. The generated header is a public
header - it carries full /// Doxygen comments and respects the 100-column
clang-format limit (the pre-commit hook verifies the committed file).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
HEADER = REPO / "molecule" / "include" / "qcx" / "molecule" / "elements.hpp"

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def fmt(value, digits=12):
    if value is None:
        return "0.0"
    return f"{value:.{digits}g}"


def build() -> str:
    data = json.loads((HERE / "elements_data.json").read_text(encoding="utf-8"))
    els = data["elements"]
    assert len(els) == 118 and [e["z"] for e in els] == list(range(1, 119))

    lines = []
    a = lines.append
    a("// GENERATED FILE - do not edit by hand.")
    a("// Source of truth: tools/periodic/elements_data.json; regenerate with")
    a("//   python tools/periodic/generate_element_table.py")
    a("#pragma once")
    a("")
    a('#include <array>')
    a('#include <cstddef>')
    a('#include <string_view>')
    a("")
    a("namespace qcx::molecule {")
    a("")
    a("/// \\defgroup qcx-molecule Molecule module")
    a("/// Geometry, atoms, isotopes, connectivity, and the full periodic table.")
    a("/// \\{")
    a("")
    # Member docs are preceding lines, not trailing ///< comments: clang-format
    # re-aligns trailing comments, which would break the byte-identical
    # regeneration check (and full-line docs stay untouched by it).
    a("/// One isotope: a nuclear mass and its natural abundance.")
    a("struct Isotope {")
    a("    /// Nuclear mass in atomic mass units (u, AME).")
    a("    double mass;")
    a("    /// Natural abundance in percent; 0 for trace/no data.")
    a("    double abundance;")
    a("};")
    a("")
    a("/// One entry of the periodic table.")
    a("struct ElementData {")
    a("    /// Atomic number Z, 1..118.")
    a("    int atomicNumber;")
    a("    /// IUPAC symbol, exact case.")
    a("    std::string_view symbol;")
    a("    /// IUPAC element name.")
    a("    std::string_view name;")
    a("    /// Conventional standard atomic weight (u).")
    a("    double standardAtomicWeight;")
    a("    /// Mass of the most abundant isotope (u).")
    a("    double mostAbundantIsotopeMass;")
    a("    /// Single-bond covalent radius; 0 = none published (never bonds).")
    a("    double covalentRadiusBohr;")
    a("    /// Van der Waals radius; 0 = none published.")
    a("    double vanDerWaalsRadiusBohr;")
    a("    /// Start index into kIsotopes.")
    a("    std::size_t isotopeOffset;")
    a("    /// Number of isotopes for this element.")
    a("    std::size_t isotopeCount;")
    a("};")
    a("")
    a("/// Number of elements in kElements.")
    a("inline constexpr std::size_t kElementCount = 118;")

    isotope_lines = []
    offset = 0
    offsets = []
    for e in els:
        offsets.append((offset, len(e["isotopes"])))
        offset += len(e["isotopes"])
        for iso in e["isotopes"]:
            isotope_lines.append(f"    {{{fmt(iso['mass'])}, {fmt(iso['abundance'], 12)}}},")

    a("")
    a("/// All isotopes of all elements, flattened; indexed via ElementData offsets.")
    # const, not constexpr: MSVC's constexpr evaluator rejects ~3000-element
    # aggregate initializers (C2131). FindElement remains constexpr; the
    # tables themselves are runtime constants. The double brace is required:
    # MSVC rejects the single-outer-brace form for struct-element arrays.
    a("inline const std::array<Isotope, " + str(offset) + "> kIsotopes = {{")
    a("\n".join(isotope_lines))
    a("}};")
    a("")
    a("/// The periodic table, Z 1..118 (Cordero covalent [Cordero2008] / Alvarez vdW [Alvarez2013] radii,")
    a("/// masses from AME2020 [Wang2021]).")
    a("inline const std::array<ElementData, kElementCount> kElements = {{")
    for e, (off, count) in zip(els, offsets):
        sym = e["symbol"]
        name = e["name"]
        line = (
            f'    {{{e["z"]}, "{sym}", "{name}", {fmt(e["standardAtomicWeight"])}, '
            f"{fmt(e['mostAbundantIsotopeMass'])}, "
            f"{fmt(e['covalentRadiusAngstrom'] and e['covalentRadiusAngstrom'] * ANGSTROM_TO_BOHR)}, "
            f"{fmt(e['vdwRadiusAngstrom'] and e['vdwRadiusAngstrom'] * ANGSTROM_TO_BOHR)}, "
            f"{off}, {count}}},"
        )
        a(line)
    a("}};")
    a("")
    # Control-statement braces go on their own line (repo .clang-format
    # BraceWrapping.AfterControlStatement, 2026-08-16) - the emitted text
    # must stay byte-identical to what clang-format would produce.
    a("/// Returns the table entry for Z, or nullptr when Z is outside 1..118.")
    a("/// \\param atomicNumber Z of the element to find.")
    a("/// \\returns The table entry, or nullptr when Z is outside 1..118.")
    a("constexpr const ElementData* FindElement(int atomicNumber) noexcept {")
    a("    if (atomicNumber < 1 || atomicNumber > 118)")
    a("    {")
    a("        return nullptr;")
    a("    }")
    a("")
    a("    return &kElements[static_cast<std::size_t>(atomicNumber - 1)];")
    a("}")
    a("")
    a("/// Returns the table entry for an exact-case IUPAC symbol, or nullptr.")
    a("/// \\param symbol IUPAC symbol (exact case).")
    a("/// \\returns The table entry, or nullptr when the symbol is unknown.")
    a("constexpr const ElementData* FindElement(std::string_view symbol) noexcept {")
    a("    for (const auto& e : kElements)")
    a("    {")
    a("        if (e.symbol == symbol)")
    a("        {")
    a("            return &e;")
    a("        }")
    a("    }")
    a("")
    a("    return nullptr;")
    a("}")
    a("")
    a("/// \\}")
    a("} // namespace qcx::molecule")
    a("")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        target = Path(sys.argv[2])
        content = build()
        if target.read_text(encoding="utf-8") == content:
            print(f"{target} matches the generated output")
            return 0
        print(f"{target} differs from the generated output - "
              "run tools/periodic/generate_element_table.py")
        return 1
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    HEADER.write_text(build(), encoding="utf-8", newline="\n")
    print(f"wrote {HEADER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
