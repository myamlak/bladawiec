#pragma once

// The O2/STO-3G triplet needs the SAD-seeded UHF start: the RunUhfScf
// default (P = 0) locks the higher-lying saddle solution at
// -147.37855918, while the unpolarized SAD start
// (alpha = beta = the SAD average - what pyscf's default minao guess
// effectively seeds) converges to the pinned ground state
// (-147.63394678545018, <S^2> = 2.003410857681). Test-only helpers;
// the pattern is the scf module's own uhf_test.cpp assembly.
#include "o2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace qcx::properties::testing {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeO2Sto3gBasis;
using qcx::testing::ToMatrix;

// The dense per-element atomic integrals of the SAD guess: one O atom at
// the origin, charge 0, triplet multiplicity (the fragment run; the
// geometry never affects its own integrals).
inline qcx::Result<qcx::scf::AtomicUhfInputs> BuildAtomicOxygenInputs() {
    auto basis = MakeO2Sto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto coordinates = CpuTensor2::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();
    auto atom = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}}, std::move(*coordinates), 0, 3);

    if (!atom.has_value())
    {
        return std::unexpected(atom.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*atom, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*atom, *basis);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*atom, *basis);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(*atom, *basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    return qcx::scf::AtomicUhfInputs{
        ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), std::move(*eri)};
}

// The O2/STO-3G SAD guess: the fragment runs + embedding.
inline qcx::Result<qcx::scf::SadGuess> BuildSadGuessForO2(const qcx::molecule::Molecule& molecule,
                                                          const qcx::basisset::BasisSet& basisSet) {
    auto atomicInputs = BuildAtomicOxygenInputs();

    if (!atomicInputs.has_value())
    {
        return std::unexpected(atomicInputs.error());
    }

    std::map<int, qcx::scf::AtomicUhfInputs> atomicMap;
    atomicMap.emplace(8, std::move(*atomicInputs));
    return qcx::scf::BuildSadGuess(molecule, basisSet, atomicMap);
}

// The dense-ERI UHF run seeded with the unpolarized SAD start; converges
// to the pinned O2 ground state (the default P = 0 start would lock the
// higher-lying saddle instead).
inline qcx::Result<qcx::scf::UhfResult> RunDenseUhfO2(const qcx::molecule::Molecule& molecule,
                                                      const qcx::basisset::BasisSet& basisSet) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basisSet);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    auto sad = BuildSadGuessForO2(molecule, basisSet);

    if (!sad.has_value())
    {
        return std::unexpected(sad.error());
    }

    const Eigen::MatrixXd unpolarizedStart = 0.5 * (sad->densityAlpha + sad->densityBeta);
    qcx::scf::UhfOptions options;
    options.initialDensityAlpha = unpolarizedStart;
    options.initialDensityBeta = unpolarizedStart;
    return qcx::scf::RunUhfScf(
        molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri, options);
}

} // namespace qcx::properties::testing
