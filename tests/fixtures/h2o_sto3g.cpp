#include "h2o_sto3g.hpp"

#include "h2_sto3g.hpp"
#include "qcx/memory/tensor.hpp"

#include <utility>
#include <vector>

namespace qcx::testing {

qcx::Result<qcx::basisset::BasisSet> MakeH2oSto3gBasis() {
    auto oxygen = qcx::basisset::ParseNwchemText(kSto3gOxygen);

    if (!oxygen.has_value())
    {
        return std::unexpected(oxygen.error());
    }

    auto hydrogen = qcx::basisset::ParseNwchemText(kSto3gHydrogen);

    if (!hydrogen.has_value())
    {
        return std::unexpected(hydrogen.error());
    }

    auto merged = oxygen->Merge(*hydrogen);

    if (!merged.has_value())
    {
        return std::unexpected(merged.error());
    }

    return std::move(*oxygen);
}

qcx::Result<qcx::molecule::Molecule> MakeH2oSto3g() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    // d = r sin(angle/2), h = r cos(angle/2), r = 0.9572 A, angle = 104.52 deg.
    (*coordinates)(1, 0) = 1.430428808474167;
    (*coordinates)(1, 1) = 1.107157044080814;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = -1.430428808474167;
    (*coordinates)(2, 1) = 1.107157044080814;
    (*coordinates)(2, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

} // namespace qcx::testing
