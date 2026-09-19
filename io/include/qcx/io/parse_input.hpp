// TOML parsing for the qcx run schema. The parser is
// strict about the vocabulary it knows (unknown method/builder/accuracy
// names and malformed atoms are kInvalidArgument) and lenient about
// extra keys - forward-compatible additions must not break old files.

#pragma once

#include "qcx/error.hpp"
#include "qcx/io/run_input.hpp"

#include <filesystem>
#include <string_view>

namespace qcx::io {

/// Parses a run description from TOML text.
/// \param tomlText The file contents.
/// \returns The run description, or an Error (kInvalidArgument for a TOML
/// syntax error, a missing molecule/atoms block, an empty atom list, a
/// malformed atom row, an unknown method/builder/accuracy/guess/unit name, a
/// wrong-typed scalar - a present key must hold its schema type, e.g.
/// charge = "x" or use_diis = 1 never silently default - a multiplicity
/// below 1, or a non-positive SCF tolerance). The charge is stored
/// unvalidated - negative charges are valid chemistry (the driver's Fukui
/// anion runs charge -1).
///
/// The atom rows' unit is resolved HERE, from `[molecule] units` (absent =
/// Angstrom): the returned RunAtom coordinates are always Bohr, and
/// RunMoleculeInput::coordinateUnit names the unit they were read under. This
/// is the schema's one conversion site - no consumer applies another.
/// \ingroup qcx-io
qcx::Result<RunInput> ParseRunInput(std::string_view tomlText);

/// Parses a run description from a TOML file.
/// \param path The .toml path.
/// \returns The run description, or an Error as ParseRunInput, plus
/// kInvalidArgument when the file cannot be read.
/// \ingroup qcx-io
qcx::Result<RunInput> ParseRunInputFile(const std::filesystem::path& path);

} // namespace qcx::io
