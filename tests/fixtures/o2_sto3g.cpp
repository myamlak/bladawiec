#include "o2_sto3g.hpp"

#include "h2o_sto3g.hpp"
#include "qcx/memory/tensor.hpp"

#include <utility>
#include <vector>

namespace qcx::testing {

qcx::Result<qcx::basisset::BasisSet> MakeO2Sto3gBasis() {
    return qcx::basisset::ParseNwchemText(kSto3gOxygen);
}

qcx::Result<qcx::molecule::Molecule> MakeO2Sto3gTriplet() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 2.2818443;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"O", 8, 0.0}},
        std::move(*coordinates),
        0,
        3);
}

} // namespace qcx::testing
