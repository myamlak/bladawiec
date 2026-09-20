// Electric multipole moments: the dipole and the traceless
// quadrupole Theta = 3M - Tr(M) I, built from the per-spin densities and
// the integrals builders' dipole/quadrupole matrices. The quadrupole is
// explicitly symmetrized below, so a caller reading the lower half gets the
// real value rather than a stale zero (documented in the header).
#include "qcx/properties/multipoles.hpp"

#include "internal/tensor_to_eigen.hpp"
#include "qcx/integrals/one_electron.hpp"

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace qcx::properties {
namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

// The nuclear part of the moment tensor relative to the origin: sum_a
// Z_a (R_a - R0) for the dipole and sum_a Z_a (R_a - R0) (R_a - R0)^T for
// the quadrupole.
void NuclearMoments(const qcx::molecule::Molecule& molecule,
                    const std::array<double, 3>& origin,
                    Eigen::Vector3d& dipole,
                    Eigen::Matrix3d& quadrupole) {
    const auto& coordinates = molecule.CoordinatesBohr();
    dipole.setZero();
    quadrupole.setZero();

    for (std::size_t atomIndex = 0; atomIndex < molecule.AtomCount(); ++atomIndex)
    {
        const double charge = static_cast<double>(molecule.Atoms()[atomIndex].atomicNumber);
        Eigen::Vector3d position;
        position << coordinates(atomIndex, 0) - origin[0], coordinates(atomIndex, 1) - origin[1],
            coordinates(atomIndex, 2) - origin[2];
        dipole += charge * position;
        quadrupole += charge * (position * position.transpose());
    }
}

// The electronic part: -sum_ij P(i,j) M(j,i); both P and the moment
// matrices are symmetric, so the (j,i) contraction equals the (i,j) one
// and reduces to a plain elementwise sum over the total density. The
// dipole is a VECTOR - it must not share storage with the quadrupole's
// diagonal: the original single-matrix return wrote the dipole into
// electronic(k, k), the quadrupole loop then OVERWROTE those diagonals,
// and the caller's electronic.diagonal() read the quadrupole moments back
// as the dipole (the O2 driver test caught it: a -30.6 z-dipole for the
// symmetric triplet, exactly nuclear - Tr(P Q_zz)).
struct ElectronicMomentsResult {
    Eigen::Vector3d dipole;
    Eigen::Matrix3d quadrupole;
};

ElectronicMomentsResult ElectronicMoments(const Eigen::MatrixXd& density,
                                          const std::array<CpuTensor2, 3>& dipole,
                                          const std::array<CpuTensor2, 6>& quadrupole) {
    ElectronicMomentsResult result;
    result.quadrupole = Eigen::Matrix3d::Zero();

    for (std::size_t k = 0; k < 3; ++k)
    {
        const Eigen::MatrixXd dipoleK = internal::TensorToEigen(dipole[k]);
        result.dipole(static_cast<Eigen::Index>(k)) = -(density.array() * dipoleK.array()).sum();
    }

    // The six components in the xx, xy, xz, yy, yz, zz order.
    constexpr std::array<std::pair<int, int>, 6> kComponents = {
        {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};

    for (std::size_t component = 0; component < kComponents.size(); ++component)
    {
        const auto [k, l] = kComponents[component];
        const Eigen::MatrixXd quadrupoleKl = internal::TensorToEigen(quadrupole[component]);
        const double value = -(density.array() * quadrupoleKl.array()).sum();
        result.quadrupole(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(l)) = value;
        result.quadrupole(static_cast<Eigen::Index>(l), static_cast<Eigen::Index>(k)) = value;
    }

    return result;
}

} // namespace

qcx::Result<MultipoleMoments> AnalyzeMultipoles(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const Eigen::MatrixXd& densityAlpha,
                                                const Eigen::MatrixXd& densityBeta,
                                                const std::array<double, 3>& origin) {
    auto dipoleTensors = qcx::integrals::BuildDipoleMatrix(molecule, basisSet, origin);

    if (!dipoleTensors.has_value())
    {
        return std::unexpected(dipoleTensors.error());
    }

    auto quadrupoleTensors = qcx::integrals::BuildQuadrupoleMatrix(molecule, basisSet, origin);

    if (!quadrupoleTensors.has_value())
    {
        return std::unexpected(quadrupoleTensors.error());
    }

    const Eigen::Index n = static_cast<Eigen::Index>((*dipoleTensors)[0].Shape()[0]);

    if (densityAlpha.rows() != n || densityAlpha.cols() != n || densityBeta.rows() != n ||
        densityBeta.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "densities must be square n x n with n the basis function count"});
    }

    const Eigen::MatrixXd density = densityAlpha + densityBeta;
    MultipoleMoments result;
    NuclearMoments(molecule, origin, result.dipole, result.quadrupole);
    // ElectronicMoments already carries the physical minus sign
    // (-sum_ij P(i,j) M(j,i)), so the electronic block ADDS onto the
    // nuclear one; subtracting would double-negate and flip the moments.
    const ElectronicMomentsResult electronic =
        ElectronicMoments(density, *dipoleTensors, *quadrupoleTensors);
    result.dipole += electronic.dipole;
    result.quadrupole += electronic.quadrupole;

    // The traceless quadrupole Theta = 3M - Tr(M) I, then the explicit
    // symmetrization (a caller reading the lower half must get the
    // real value, not a stale zero).
    result.quadrupole =
        3.0 * result.quadrupole - result.quadrupole.trace() * Eigen::Matrix3d::Identity();
    result.quadrupole = 0.5 * (result.quadrupole + result.quadrupole.transpose());
    return result;
}

} // namespace qcx::properties
