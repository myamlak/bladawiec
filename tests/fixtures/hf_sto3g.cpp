#include "hf_sto3g.hpp"

#include "h2_sto3g.hpp"
#include "qcx/memory/tensor.hpp"

#include <utility>
#include <vector>

namespace qcx::testing {

qcx::Result<qcx::basisset::BasisSet> MakeHfSto3gBasis() {
    auto fluorine = qcx::basisset::ParseNwchemText(kSto3gFluorine);

    if (!fluorine.has_value())
    {
        return std::unexpected(fluorine.error());
    }

    auto hydrogen = qcx::basisset::ParseNwchemText(kSto3gHydrogen);

    if (!hydrogen.has_value())
    {
        return std::unexpected(hydrogen.error());
    }

    auto merged = fluorine->Merge(*hydrogen);

    if (!merged.has_value())
    {
        return std::unexpected(merged.error());
    }

    return std::move(*fluorine);
}

qcx::Result<qcx::molecule::Molecule> MakeHfSto3g() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.732500911056906; // 0.9168 A, the experimental bond
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"F", 9, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

} // namespace qcx::testing
