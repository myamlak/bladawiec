// The QFMM composed HF builder (qfmm_hf_build.hpp): F = H + 2J_QFMM(rho) -
// K(rho), the RIJCOSX-style combination point applied to QFMM. Per BuildFock
// the Coulomb half is the QfmmJBuilder's H + 2J (near field through the
// restricted direct Coulomb builder plus the multipole far field) and the
// exchange half is the exchange-only DirectJkFockBuilder's H - K; each
// split result carries one full H copy, so one is subtracted back per call
// (the double-H trap - the MakeDirectUhfFockBuilder pattern of the driver,
// convention). BuildFock is written as those two halves in that order, and both are
// public on their own (BuildCoulombOnly / BuildExchangeOnly) for the
// Kohn-Sham energy seam, which contracts J[D] separately from the Fock: the
// half it is handed is then literally the Fock's own, not a second builder's
// agreeing contraction. The UHF channels run the same two states per BuildUhfFock:
// one Coulomb call on 0.5 (P_alpha + P_beta), one exchange call per spin
// on the raw spin densities, one H subtracted back per channel - the
// driver assembly of MakeDirectUhfFockBuilder with the QFMM builder in the
// coulomb slot (the J loop's baked 2.0 turns 2J(0.5 P_tot) into J(P_tot),
// and the exchange-only -K is linear in its input, so the raw P_sigma
// contracts the channel's own K(P_sigma)). Create() owns both builder
// states. With a
// workspaceBudget the two nested Creates share it - the QFMM half's
// Create-time estimate (outer store plus its near-field builder) reserves
// first, then this exchange half's - nesting order = reservation order, the
// direct-UHF composition of the driver.

#include "qcx/integrals/qfmm_hf_build.hpp"

#include "internal/tensor_eigen_bridge.hpp"

#include <Eigen/Core>
#include <memory>
#include <utility>

namespace qcx::integrals {

struct QfmmHfFockBuilder::State {
    QfmmJBuilder _jBuilder;
    DirectJkFockBuilder _kBuilder;
    // The core Hamiltonian this builder was constructed with (H in
    // F = H + 2J - K): every nested split result carries its own full H
    // copy, so one is subtracted back per BuildFock call (the double-H
    // trap). The Eigen copy is held here (one per Create, not one per
    // call) - the tensor would need a per-call conversion otherwise.
    Eigen::MatrixXd _coreHamiltonian;
};

QfmmHfFockBuilder::QfmmHfFockBuilder(std::shared_ptr<const State> state) :
    _state(std::move(state)) {}

std::size_t QfmmHfFockBuilder::FarFieldPairCount() const noexcept {
    return _state->_jBuilder.FarFieldPairCount();
}

const std::optional<FockModeInfo>& QfmmHfFockBuilder::ModeInfo() const noexcept {
    return _state->_jBuilder.ModeInfo();
}

const std::optional<FockModeInfo>& QfmmHfFockBuilder::ExchangeModeInfo() const noexcept {
    return _state->_kBuilder.ModeInfo();
}

QfmmModelRecord QfmmHfFockBuilder::ModelRecord() const noexcept {
    return _state->_jBuilder.ModelRecord();
}

qcx::Result<QfmmHfFockBuilder> QfmmHfFockBuilder::Create(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
    const QfmmOptions& options) {
    // The Coulomb half: the QFMM builder (validates the QfmmOptions itself -
    // theta/lMult/maxLeafSize). The exchange half: a DirectJkFockBuilder in
    // its exchange-only mode at the SAME accuracy preset (the K path of the
    // composed Fock must screen like the fused direct build it replaces -
    // the equal-preset comparison protocol), with the
    // same pass-throughs the QFMM near-field builder gets
    // (useDensityScreening / useCertifiedMixedPrecision / maxParallelChunks
    // - the serial pin reaches both halves).
    auto jBuilder = QfmmJBuilder::Create(molecule, basisSet, coreHamiltonian, options);

    if (!jBuilder.has_value())
    {
        return std::unexpected(jBuilder.error());
    }

    FockBuildOptions exchangeOptions;
    exchangeOptions.accuracy = options.accuracy;
    exchangeOptions.buildExchangeOnly = true;
    exchangeOptions.useDensityScreening = options.useDensityScreening;
    exchangeOptions.useCertifiedMixedPrecision = options.useCertifiedMixedPrecision;
    exchangeOptions.maxParallelChunks = options.maxParallelChunks;
    // The shared adaptive-memory budget: the QFMM Create above reserved its
    // estimate (outer store + near-field builder) against the counter
    // first, and this exchange half's Create reserves its own next -
    // nesting order = reservation order (fock_build.hpp FockModeInfo).
    exchangeOptions.workspaceBudget = options.workspaceBudget;
    auto kBuilder =
        DirectJkFockBuilder::Create(molecule, basisSet, coreHamiltonian, exchangeOptions);

    if (!kBuilder.has_value())
    {
        return std::unexpected(kBuilder.error());
    }

    // State is an aggregate, and its members are not default-constructible
    // (the builders are Create-returned) - member-wise aggregate init from
    // the moved results.
    auto state = std::make_shared<State>(State{
        std::move(*jBuilder), std::move(*kBuilder), internal::TensorToEigen(coreHamiltonian)});

    return QfmmHfFockBuilder(std::move(state));
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>
QfmmHfFockBuilder::BuildCoulombOnly(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const {
    // The nested QFMM builder's own result: H + 2J_QFMM(rho) (near + far),
    // the direct family's buildCoulombOnly convention. No H bookkeeping
    // here - this call IS the half the composition's Coulomb convention
    // asks for (unlike the RI-J link's, whose contraction omits H).
    return _state->_jBuilder.BuildFock(density);
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>
QfmmHfFockBuilder::BuildExchangeOnly(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const {
    // The nested exchange-only DirectJkFockBuilder's own result: H - K(rho),
    // the buildExchangeOnly convention.
    return _state->_kBuilder.BuildFock(density);
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> QfmmHfFockBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const {
    // The Coulomb half: H + 2J(rho) (near + far) from the QFMM builder -
    // its own shape validation runs first.
    auto coulomb = BuildCoulombOnly(density);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    // The exchange half: H - K(rho) from the exchange-only direct builder.
    auto exchange = BuildExchangeOnly(density);

    if (!exchange.has_value())
    {
        return std::unexpected(exchange.error());
    }

    // The double-H trap: both split results carry one full H copy, so one
    // is subtracted back - the composed result is H + 2J_QFMM(rho) - K(rho).
    Eigen::MatrixXd fock = internal::TensorToEigen(*coulomb);
    fock += internal::TensorToEigen(*exchange);
    fock -= _state->_coreHamiltonian;

    return internal::EigenToTensor(fock);
}

qcx::Result<std::pair<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>,
                      qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>>
QfmmHfFockBuilder::BuildUhfFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& alphaDensity,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& betaDensity) const {
    // The per-spin densities must match in shape and be square (the
    // half-sum below needs the Eigen pairing BEFORE any engine validation
    // runs - an Eigen binary op on mismatched shapes would be a debug
    // assert, never a Result). Symmetry itself stays the nested builders'
    // documented precondition, like BuildFock.
    const Eigen::MatrixXd alpha = internal::TensorToEigen(alphaDensity);
    const Eigen::MatrixXd beta = internal::TensorToEigen(betaDensity);

    if (alpha.rows() != alpha.cols() || beta.rows() != beta.cols() || alpha.rows() != beta.rows())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the alpha and beta densities of the composed UHF "
                                          "Fock must be square and match in shape"});
    }

    // The Coulomb half on the HALF-SUMMED density: the J loop bakes in the
    // 2.0 factor, so 2J(0.5 (P_a + P_b)) is the physical J(P_a + P_b), and
    // the QFMM far field sees the spin-summed charge (the
    // MakeDirectUhfFockBuilder convention). The alpha and beta channels
    // share this one call.
    auto totalTensor = internal::EigenToTensor(0.5 * (alpha + beta));

    if (!totalTensor.has_value())
    {
        return std::unexpected(totalTensor.error());
    }

    auto coulomb = _state->_jBuilder.BuildFock(*totalTensor);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    // The exchange half per spin on the RAW spin density: the
    // exchange-only builder's -K is linear in its input, so P_sigma needs
    // no halving - the UHF channel contracts its own K(P_sigma) (the
    // closed-shell BuildFock's halving happens at the caller, where rho =
    // D/2 is formed).
    auto exchangeAlpha = _state->_kBuilder.BuildFock(alphaDensity);

    if (!exchangeAlpha.has_value())
    {
        return std::unexpected(exchangeAlpha.error());
    }

    auto exchangeBeta = _state->_kBuilder.BuildFock(betaDensity);

    if (!exchangeBeta.has_value())
    {
        return std::unexpected(exchangeBeta.error());
    }

    // The double-H trap per channel: every split result carries one full H
    // copy, so one is subtracted back - F_sigma = H + J(P_tot) - K(P_sigma).
    Eigen::MatrixXd fockAlpha = internal::TensorToEigen(*coulomb);
    fockAlpha += internal::TensorToEigen(*exchangeAlpha);
    fockAlpha -= _state->_coreHamiltonian;

    Eigen::MatrixXd fockBeta = internal::TensorToEigen(*coulomb);
    fockBeta += internal::TensorToEigen(*exchangeBeta);
    fockBeta -= _state->_coreHamiltonian;

    auto fockAlphaTensor = internal::EigenToTensor(fockAlpha);

    if (!fockAlphaTensor.has_value())
    {
        return std::unexpected(fockAlphaTensor.error());
    }

    auto fockBetaTensor = internal::EigenToTensor(fockBeta);

    if (!fockBetaTensor.has_value())
    {
        return std::unexpected(fockBetaTensor.error());
    }

    return std::pair<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>,
                     qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>{
        std::move(*fockAlphaTensor), std::move(*fockBetaTensor)};
}

} // namespace qcx::integrals
