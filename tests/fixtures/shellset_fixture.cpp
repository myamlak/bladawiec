#include "shellset_fixture.hpp"

#include "qcx/memory/tensor.hpp"

#include <utility>
#include <vector>

namespace qcx::testing {

qcx::Result<qcx::molecule::Molecule> MakeShellsetMolecule() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.4;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"He", 2, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

} // namespace qcx::testing
