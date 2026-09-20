// The arbitrary-density RHF/UHF Fock/energy evaluator (density_energy.hpp,
// the ETS-NOCV embedding seam). The contraction and energy
// bookkeeping mirror rhf.cpp's BuildFock/ComputeElectronicEnergy and
// uhf.cpp's BuildUhfFock/ComputeUhfElectronicEnergy verbatim, so the
// evaluator reproduces the loops' values at any density the loops would
// have produced; the future FockBuilder<Derived> CRTP absorbs both copies.
#include "qcx/scf/density_energy.hpp"

#include "internal/scf_common.hpp"
#include "qcx/linalg/dense_ops.hpp"
#include "qcx/molecule/mass_properties.hpp"

#include <cstddef>
#include <utility>

namespace qcx::scf {

namespace {

// The J and K contractions of one density against the supermatrices,
// through the linalg seam (Eigen for small blocks, vendor BLAS above the
// threshold). dVec unpacks the density in the tensor's row-major
// order so its entries line up with the supermatrix columns (the rhf.cpp /
// uhf.cpp unpack verbatim).
qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> ContractJk(
    const internal::JkSupermatrices& super, const Eigen::MatrixXd& density) {
    const Eigen::Index n = density.rows();
    Eigen::MatrixXd dVec(n * n, 1);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            dVec(mu * n + nu, 0) = density(mu, nu);
        }
    }

    auto j = qcx::linalg::DenseMultiply(super.coulomb, dVec);

    if (!j.has_value())
    {
        return std::unexpected(j.error());
    }

    auto k = qcx::linalg::DenseMultiply(super.exchange, dVec);

    if (!k.has_value())
    {
        return std::unexpected(k.error());
    }

    return std::make_pair(std::move(*j), std::move(*k));
}

// The 1/2 Tr[D (H + F)] bookkeeping of rhf.cpp.
double ComputeElectronicEnergy(const Eigen::MatrixXd& coreHamiltonian,
                               const Eigen::MatrixXd& fock,
                               const Eigen::MatrixXd& density) {
    return 0.5 * (density.cwiseProduct(coreHamiltonian + fock)).sum();
}

// The UHF bookkeeping of uhf.cpp: 1/2 Tr[D_a (H + F_a) + D_b (H + F_b)].
double ComputeUhfElectronicEnergy(const Eigen::MatrixXd& coreHamiltonian,
                                  const Eigen::MatrixXd& fockAlpha,
                                  const Eigen::MatrixXd& fockBeta,
                                  const Eigen::MatrixXd& densityAlpha,
                                  const Eigen::MatrixXd& densityBeta) {
    return 0.5 * ((densityAlpha.cwiseProduct(coreHamiltonian + fockAlpha)).sum() +
                  (densityBeta.cwiseProduct(coreHamiltonian + fockBeta)).sum());
}

} // namespace

qcx::Result<RhfDensityEnergy> EvaluateRhfDensityEnergy(
    const qcx::molecule::Molecule& molecule,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& density,
    bool includeExchange) {
    const std::size_t n = eri.Shape()[0];

    if (eri.Shape()[1] != n || eri.Shape()[2] != n || eri.Shape()[3] != n ||
        static_cast<std::size_t>(coreHamiltonian.rows()) != n ||
        static_cast<std::size_t>(coreHamiltonian.cols()) != n ||
        static_cast<std::size_t>(density.rows()) != n ||
        static_cast<std::size_t>(density.cols()) != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "core Hamiltonian and density must match the ERI function count"});
    }

    auto super = internal::BuildJkSupermatrices(eri, n);

    auto jk = ContractJk(super, density);

    if (!jk.has_value())
    {
        return std::unexpected(jk.error());
    }

    Eigen::MatrixXd fock = coreHamiltonian;

    for (std::size_t mu = 0; mu < n; ++mu)
    {
        for (std::size_t nu = 0; nu < n; ++nu)
        {
            // F = H + J - K/2: the 1/2 is the closed-shell exchange
            // prefactor of the Roothaan Fock matrix (rhf.cpp verbatim).
            // includeExchange = false drops the exchange for the ETS
            // electrostatic term (the classical Coulomb-only functional).
            const Eigen::Index flat = static_cast<Eigen::Index>(mu * n + nu);
            fock(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) +=
                (*jk).first(flat, 0) - (includeExchange ? 0.5 * (*jk).second(flat, 0) : 0.0);
        }
    }

    RhfDensityEnergy result;
    result.fock = std::move(fock);
    result.electronicEnergy = ComputeElectronicEnergy(coreHamiltonian, result.fock, density);
    result.totalEnergy = result.electronicEnergy + qcx::molecule::NuclearRepulsionEnergy(molecule);
    return result;
}

qcx::Result<UhfDensityEnergy> EvaluateUhfDensityEnergy(
    const qcx::molecule::Molecule& molecule,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    bool includeExchange) {
    const std::size_t n = eri.Shape()[0];

    if (eri.Shape()[1] != n || eri.Shape()[2] != n || eri.Shape()[3] != n ||
        static_cast<std::size_t>(coreHamiltonian.rows()) != n ||
        static_cast<std::size_t>(coreHamiltonian.cols()) != n ||
        static_cast<std::size_t>(densityAlpha.rows()) != n ||
        static_cast<std::size_t>(densityAlpha.cols()) != n ||
        static_cast<std::size_t>(densityBeta.rows()) != n ||
        static_cast<std::size_t>(densityBeta.cols()) != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "core Hamiltonian and both densities must match the ERI function count"});
    }

    auto super = internal::BuildJkSupermatrices(eri, n);

    // The Coulomb field of the combined density, once (uhf.cpp verbatim).
    auto jkTotal = ContractJk(super, densityAlpha + densityBeta);

    if (!jkTotal.has_value())
    {
        return std::unexpected(jkTotal.error());
    }

    auto jkAlpha = ContractJk(super, densityAlpha);

    if (!jkAlpha.has_value())
    {
        return std::unexpected(jkAlpha.error());
    }

    auto jkBeta = ContractJk(super, densityBeta);

    if (!jkBeta.has_value())
    {
        return std::unexpected(jkBeta.error());
    }

    Eigen::MatrixXd fockAlpha = coreHamiltonian;
    Eigen::MatrixXd fockBeta = coreHamiltonian;

    for (std::size_t mu = 0; mu < n; ++mu)
    {
        for (std::size_t nu = 0; nu < n; ++nu)
        {
            // F_sigma = H + J(D_a + D_b) - K(D_sigma) (uhf.cpp verbatim);
            // includeExchange = false drops the exchange (the ETS
            // Coulomb-only functional).
            const Eigen::Index flat = static_cast<Eigen::Index>(mu * n + nu);
            const Eigen::Index i = static_cast<Eigen::Index>(mu);
            const Eigen::Index j = static_cast<Eigen::Index>(nu);
            fockAlpha(i, j) +=
                (*jkTotal).first(flat, 0) - (includeExchange ? (*jkAlpha).second(flat, 0) : 0.0);
            fockBeta(i, j) +=
                (*jkTotal).first(flat, 0) - (includeExchange ? (*jkBeta).second(flat, 0) : 0.0);
        }
    }

    UhfDensityEnergy result;
    result.fockAlpha = std::move(fockAlpha);
    result.fockBeta = std::move(fockBeta);
    result.electronicEnergy = ComputeUhfElectronicEnergy(
        coreHamiltonian, result.fockAlpha, result.fockBeta, densityAlpha, densityBeta);
    result.totalEnergy = result.electronicEnergy + qcx::molecule::NuclearRepulsionEnergy(molecule);
    return result;
}

} // namespace qcx::scf
