// Mass properties via parallel blocked reductions over the atoms.
#include "qcx/molecule/mass_properties.hpp"

#include "qcx/backend/cpu_backend.hpp"

#include <Eigen/Dense>

namespace qcx::molecule {

MassProperties ComputeMassProperties(const Molecule& molecule, double linearityThreshold) {
    const auto& coords = molecule.CoordinatesBohr();
    const auto& atoms = molecule.Atoms();
    const std::size_t n = atoms.size();

    // Total mass and center of mass (parallel reductions over atoms). The
    // lambdas declare explicit return types: Eigen's operator+ returns an
    // expression template referencing its operands, and returning one from a
    // lambda by value would dangle as soon as the temporaries die.
    const Eigen::Vector3d massWeighted = qcx::backend::ParallelReduce<Eigen::Vector3d>(
        n,
        Eigen::Vector3d::Zero(),
        [&](const Eigen::Vector3d& acc, std::size_t i) -> Eigen::Vector3d {
            return acc + atoms[i].isotopicMass *
                             Eigen::Vector3d(coords(i, 0), coords(i, 1), coords(i, 2));
        },
        [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) -> Eigen::Vector3d {
            return a + b;
        });
    const double totalMass = qcx::backend::ParallelReduce<double>(
        n,
        0.0,
        [&](const double& acc, std::size_t i) { return acc + atoms[i].isotopicMass; },
        [](const double& a, const double& b) { return a + b; });
    const Eigen::Vector3d centerOfMass = massWeighted / totalMass;

    // Inertia tensor about the center of mass (parallel reduction; negative
    // off-diagonals per the standard mass-weighted convention).
    const Eigen::Matrix3d inertia = qcx::backend::ParallelReduce<Eigen::Matrix3d>(
        n,
        Eigen::Matrix3d::Zero(),
        [&](const Eigen::Matrix3d& acc, std::size_t i) -> Eigen::Matrix3d {
            const double m = atoms[i].isotopicMass;
            const double x = coords(i, 0) - centerOfMass[0];
            const double y = coords(i, 1) - centerOfMass[1];
            const double z = coords(i, 2) - centerOfMass[2];
            Eigen::Matrix3d contribution;
            contribution << m * (y * y + z * z), -m * x * y, -m * x * z, -m * x * y,
                m * (x * x + z * z), -m * y * z, -m * x * z, -m * y * z, m * (x * x + y * y);
            return acc + contribution;
        },
        [](const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) -> Eigen::Matrix3d {
            return a + b;
        });

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(inertia);
    // Eigen returns ascending eigenvalues; expose descending moments.
    MassProperties properties;
    properties.totalMass = totalMass;
    properties.centerOfMass = centerOfMass;
    properties.inertiaTensor = inertia;
    properties.principalMoments = solver.eigenvalues().reverse();
    properties.principalAxes = solver.eigenvectors().rowwise().reverse();
    // A single atom has no axis to be linear about: every moment is zero,
    // which would trivially satisfy the threshold.
    properties.isLinear = n > 1 && properties.principalMoments[2] <=
                                       linearityThreshold * properties.principalMoments[0];
    return properties;
}

double NuclearRepulsionEnergy(const Molecule& molecule) {
    const auto& coords = molecule.CoordinatesBohr();
    const auto& atoms = molecule.Atoms();
    const std::size_t n = atoms.size();
    // The triangular row loop: iteration i does n - i - 1 pair evaluations,
    // so a static schedule leaves the tail threads idle (mol-1; the same
    // shape connectivity.cpp schedules dynamically). Called every SCF
    // iteration.
    return qcx::backend::ParallelReduce<double>(
        n,
        0.0,
        [&](const double& acc, std::size_t i) {
            double partial = acc;

            for (std::size_t j = i + 1; j < n; ++j)
            {
                const double dx = coords(i, 0) - coords(j, 0);
                const double dy = coords(i, 1) - coords(j, 1);
                const double dz = coords(i, 2) - coords(j, 2);
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                partial += static_cast<double>(atoms[i].atomicNumber * atoms[j].atomicNumber) / r;
            }

            return partial;
        },
        [](const double& a, const double& b) { return a + b; },
        /*dynamicSchedule=*/true);
}

} // namespace qcx::molecule
