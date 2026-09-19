// The public symmetry-reduction seam (symmetry_reduction.hpp):
// detect the point group and delegate the signed-permutation extraction to
// the internal group realization (symmetry_blocks.cpp).

#include "qcx/scf/symmetry_reduction.hpp"

#include "internal/symmetry_blocks.hpp"
#include "qcx/symmetry/detection.hpp"

namespace qcx::scf {

qcx::Result<qcx::integrals::SymmetryReduction> BuildSymmetryReduction(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
    const qcx::symmetry::SymmetryAnalysis analysis = qcx::symmetry::DetectPointGroup(molecule);
    return internal::BuildSymmetryReduction(molecule, basisSet, analysis);
}

} // namespace qcx::scf
