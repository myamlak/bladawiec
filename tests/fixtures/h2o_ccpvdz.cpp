#include "h2o_ccpvdz.hpp"

#include "h2o_sto3g.hpp"

#include <filesystem>
#include <utility>

namespace qcx::testing {

qcx::Result<qcx::basisset::BasisSet> MakeH2oCcpvdzBasis() {
    const std::filesystem::path root(QcxBasisDataDir);
    return qcx::basisset::ParseNwchemDirectory((root / "cc-pvdz").string());
}

qcx::Result<qcx::basisset::BasisSet> MakeH2oCcpvdzRifitBasis() {
    const std::filesystem::path root(QcxBasisDataDir);
    return qcx::basisset::ParseNwchemDirectory((root / "cc-pvdz-rifit").string());
}

qcx::Result<qcx::molecule::Molecule> MakeH2oCcpvdz() {
    // The molecule is basis-independent: the same experimental H2O the
    // STO-3G fixture builds.
    return MakeH2oSto3g();
}

} // namespace qcx::testing
