// Semantic validation of the parsed run input: the schema-level
// checks that need the element table or the bundled basis data - things the
// TOML parser cannot know. Every check runs in one pass and reports every
// problem found; the run aborts before any molecule, basis-set, or Fock-
// builder construction when the report is not clean.

#pragma once

#include "qcx/io/run_input.hpp"

#include <string>
#include <vector>

namespace qcx::io {

/// The result of one semantic-validation pass: zero issues means the input
/// is ready for construction. Validation cannot fail internally (issues are
/// the output, not an error), so ValidateInput returns the report by value
/// instead of the usual Result<>.
/// \ingroup qcx-io
struct ValidationReport {
    /// One message per problem, in discovery order; each is ready for the
    /// "qcx: " CLI prefix (no trailing newline, no trailing period).
    std::vector<std::string> issues;

    /// True when every check passed.
    /// \returns Whether the input is semantically valid.
    /// \ingroup qcx-io
    bool IsValid() const noexcept {
        return issues.empty();
    }
};

/// Validates a parsed run description semantically, before any molecule,
/// basis-set, or Fock-builder construction.
///
/// Checks the invariants the TOML parser cannot know:
/// \li every atom symbol is a real element (qcx::molecule::FindElement);
/// \li the electron count (sum of Z minus charge) is at least 1 and its
///     parity matches the spin multiplicity;
/// \li a closed-shell method (rhf, rks) is requested only with multiplicity
///     1 (their restricted density carries no spin polarization, so any
///     other multiplicity would silently run the singlet state); the rule is
///     decided by one exhaustive switch over the method enum, whose
///     unhandled-enumerator diagnostic is an error, so a future closed-shell
///     method cannot inherit "no restriction" silently;
/// \li basis.orbital resolves to a bundled basis directory, and basis.aux
///     (explicit, or auto-selected on the RHF ri_j_link path) resolves too,
///     through the same ParseNwchemDirectory call the driver uses.
///
/// Deliberately not checked here: TOML syntax/type/vocabulary errors (the
/// parser's job, fail-fast single-error) and the method/builder/guess
/// combination rules (implementation knowledge, kUnimplemented, kept in the
/// driver's ValidateCombination).
/// \param input The parsed run description.
/// \returns The report; check IsValid().
/// \ingroup qcx-io
ValidationReport ValidateInput(const RunInput& input);

} // namespace qcx::io
