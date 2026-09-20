#pragma once

// The shared harness of the QFMM acceptance tests (the monopole gate and
// the general-L_mult gate): the one-electron core Hamiltonian, the
// physical density, the QFMM build (octree near field - the direct builder
// restricted to the near-field pair-pair set, buildCoulombOnly - plus the
// far-field multipole accumulation at any L_mult, 2J convention) and the
// plain direct-sum ground truth. Both builds run fp64-only
// (useCertifiedMixedPrecision = false), serial (maxParallelChunks = 1): the
// theta -> 0 gate must be BIT-exact, and the serial pin makes the
// near-field parts of the two builds bitwise equal, so the measured
// difference is exactly the multipole truncation. Everything is inline -
// the header is included from both test translation units, and the helper
// bodies are parameterized by the multipole order.

#include "internal/md_batch.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_multipole.hpp"
#include "internal/qfmm_tree.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace qcx::integrals::test {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

// The tensor-conversion helpers (tests/fixtures/tensor_conversions.hpp)
// used unqualified throughout the harness below. They are brought into
// this namespace so every consumer translation unit compiles without its
// own using-block - the lookup happens at the inline-body include time.
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V from the one-electron engines (the FockBuildTest pattern).
inline qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basisSet) {
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

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();
    return std::move(*kinetic);
}

// A symmetric density with |D| <= 0.75 and a diagonal near 0.5 - the
// magnitudes the screening gates expect (the FockBuildTest pattern).
inline Eigen::MatrixXd PhysicalDensity(std::size_t n) {
    std::mt19937_64 rng(20260817);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
}

// The full QFMM build: the near field is the direct builder restricted to
// the octree's near-field pair-pair set (buildCoulombOnly - J only, no K),
// the far field is the multipole accumulation at \p lMult (moments, leaf
// aggregation, M2M, M2L + L2L, and the 2J far-field accumulation). Both
// use fp64 only (useCertifiedMixedPrecision = false) so the acceptance
// measurements isolate the multipole truncation from the certified-lane error.
inline qcx::Result<Eigen::MatrixXd> BuildQfmmFock(const qcx::molecule::Molecule& molecule,
                                                  const qcx::basisset::BasisSet& basisSet,
                                                  const CpuTensor2& coreHamiltonian,
                                                  const Eigen::MatrixXd& density,
                                                  double theta,
                                                  int lMult) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto pairStore = qcx::integrals::internal::BuildPairData(molecule, basisSet, *pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(pairStore.error());
    }

    // The fixture's geometry pass uses the kTight rung of the per-preset
    // extent ladder (QfmmExtentForPreset = the committed
    // kQfmmExtentThreshold): the fixture gates are the tight-rung
    // measurement config (the harness adopts the same call as the builder,
    // so gates stay comparable; the builder itself drives the extent from
    // the preset).
    const auto geometries = qcx::integrals::internal::ComputePairGeometries(
        *pairStore, qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kTight));
    const auto tree = qcx::integrals::internal::BuildQfmmTree(geometries);

    if (!tree.has_value())
    {
        return std::unexpected(tree.error());
    }

    std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
    std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
    qcx::integrals::internal::BuildInteractionLists(
        *tree, theta, farFieldPairs, nearFieldLeafPairs);

    // The near-field restriction: every pair-pair (p, q) inside a
    // near-field leaf pair, in BOTH orientations (the screening loop's
    // (bra, ket) pair-pair is canonical, ket <= bra) - the leaf-pair bitset
    // form (PairPairRestriction; the pair-pair key set it replaced was a
    // quadratic memory blowup).
    qcx::integrals::PairPairRestriction nearFieldKeys;
    nearFieldKeys.nLeaves = tree->nodes.size();
    nearFieldKeys.leafOfPair = tree->leafOfPair;
    nearFieldKeys.words.assign((tree->nodes.size() * tree->nodes.size() + 63) / 64, 0);

    for (const auto& [leafA, leafB] : nearFieldLeafPairs)
    {
        nearFieldKeys.InsertBoth(leafA, leafB);
    }

    qcx::integrals::FockBuildOptions options;
    options.useCertifiedMixedPrecision = false;
    options.buildCoulombOnly = true;
    // The serial pin: the chunked screening's combine order is thread-
    // schedule dependent (documented in BuildFock), which would leave a
    // last-bit difference between two builds - and the theta=0 gate must
    // be BIT-exact. Serial is deterministic, so the near-field parts of
    // the QFMM and the direct build are bitwise equal and the measured
    // difference is exactly the multipole truncation.
    options.maxParallelChunks = 1;
    options.restrictToPairPairs = std::move(nearFieldKeys);
    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, coreHamiltonian, options);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    auto densityTensor = ToTensor(density);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        return std::unexpected(fock.error());
    }

    Eigen::MatrixXd result = ToMatrix(*fock);

    // The far field: the moment table at the uniform order (every pair and
    // node at \p lMult - the fixed-order shape in the order-vector form:
    // uniform per-pair/per-node orders reproduce the fixed-order loops
    // bit-identically, only the per-pair stride is (lMult + 1)^2 blocks
    // instead of the always-full stride), the density-weighted leaf
    // moments, the M2M pass, the M2L + L2L potentials, and the J
    // accumulation in the direct builder's 2J convention.
    const std::vector<int> pairOrders(pairStore->size(), lMult);
    const auto table =
        qcx::integrals::internal::BuildMomentTable(*pairList, *pairStore, geometries, pairOrders);

    if (!table.has_value())
    {
        return std::unexpected(table.error());
    }

    const std::vector<int> nodeOrders(tree->nodes.size(), lMult);
    const auto leafMoments = qcx::integrals::internal::AggregateLeafMoments(
        *pairList, *tree, *table, geometries, density, nodeOrders);
    const auto nodeMoments =
        qcx::integrals::internal::AggregateNodeMoments(*tree, leafMoments, nodeOrders);
    const auto potentials = qcx::integrals::internal::BuildFarFieldPotentials(
        *tree, farFieldPairs, nodeMoments, pairOrders, nodeOrders);
    qcx::integrals::internal::AccumulateFarFieldJ(
        result, *pairList, *tree, *table, potentials, geometries, nodeOrders);

    return result;
}

// The plain direct-sum ground truth: the unrestricted direct builder,
// same fp64-only configuration.
inline qcx::Result<Eigen::MatrixXd> BuildDirectFock(const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basisSet,
                                                    const CpuTensor2& coreHamiltonian,
                                                    const Eigen::MatrixXd& density) {
    qcx::integrals::FockBuildOptions options;
    options.useCertifiedMixedPrecision = false;
    options.buildCoulombOnly = true;
    // The same serial pin as the QFMM side: both builds must be bitwise
    // deterministic for the near-field parts to cancel exactly.
    options.maxParallelChunks = 1;
    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, coreHamiltonian, options);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    auto densityTensor = ToTensor(density);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        return std::unexpected(fock.error());
    }

    return ToMatrix(*fock);
}

} // namespace qcx::integrals::test
