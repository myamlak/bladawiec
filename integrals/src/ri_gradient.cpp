// The RI Coulomb derivative tier, the derivative Fock blocks and the
// two-electron derivative Fock-block entry (ri_gradient.hpp,
// one_electron.hpp, two_electron.hpp).
//
// RI Coulomb: the fitting coefficients are a function of the geometry, so
// the derivative of (mu nu|P) * c_P is not the derivative of (mu nu|P)
// alone. Written as E_J = 1/2 d^T M^-1 d with d_P = sum_uv D_uv (uv|P) and M
// the auxiliary metric, the derivatives follow from d^(U) and M^(U): for a
// tuple T,
//
//   d^(U)_P = sum_uv D_uv d(uv|P)^(U),
//   w^(U) = M^-1 (d^(U) - sum_{nonempty V subset U} M^(V) w^(U \ V)),
//   E^(T) = 1/2 sum_{U subset T} d^(U) . w^(T \ U),
//
// with w^(empty) = w = M^-1 d. Order 1 uses the closed form
// d^(a) . w - 1/2 w^T M^(a) w, which is the same expression; order 2 walks
// the recursion.
//
// Both d and M are evaluated through the landed two-electron derivative
// tier on the quartet representation the energy path already uses: the
// auxiliary shell is paired with a phantom s function of exponent 0, so
// (uv|P) is the quartet (uv|P|s) and (P|Q) is (s|P|s|Q). The phantom's own
// operators are the zero-exponent ones, so the constant contributes no
// derivative of its own. The auxiliary index is contracted one task at a
// time: no nBasis^2 x nAux tensor is formed.
//
// One retained set serves the energy and every derivative - the union of
// the energy path's density-free predicate and the derivative-aware one -
// and the metric is differentiated whole, so the reported derivatives belong
// to the reported energy. The two energies and the skeletons of the
// two-electron entry are accumulated from the same value and derivative
// blocks with the energy path's own weights.
//
// [Helgaker2000] [McMurchie1978]

#include "qcx/integrals/ri_gradient.hpp"

#include "internal/fock_screen.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_derivative.hpp"
#include "internal/md_eri_derivative.hpp"
#include "internal/ri_derivative.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/two_electron.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qcx::integrals {

namespace {

using qcx::integrals::internal::MdEriDerivativeQuartet;
using qcx::integrals::internal::MdEriDerivativeScratch;
using qcx::integrals::internal::MdPairData;
using qcx::integrals::internal::MdShellInput;
using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// A row-major MO coefficient view over the caller's span.
using MoMatrix =
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/// The basis set's shells, checked against the engine's angular-momentum cap.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \returns The shell pair list, or an Error.
qcx::Result<ShellPairList> CheckedShellPairs(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    for (const ShellInfo& shell : pairList->shells)
    {
        if (!SupportsL(shell.angularMomentum))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "shell angular momentum exceeds kMaxEngineL of this build"});
        }
    }

    return pairList;
}

/// The dot product of two equal-length blocks.
/// \param left The first block.
/// \param right The second block.
/// \returns sum_i left[i] * right[i].
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): left, right.
double DotProduct(std::span<const double> left, std::span<const double> right) noexcept {
    double sum = 0.0;

    for (std::size_t index = 0; index < left.size(); ++index)
    {
        sum += left[index] * right[index];
    }

    return sum;
}

/// The population count of a sub-tuple mask.
/// \param mask The mask.
/// \returns The number of set bits.
int PopCount(unsigned mask) noexcept {
    int count = 0;

    for (; mask != 0; mask &= mask - 1)
    {
        ++count;
    }

    return count;
}

/// Gathers a target tuple's coordinates at the positions one sub-tuple mask
/// names, in ascending position order.
/// \param coordinates The flat tuple-major coordinate list.
/// \param offset The target tuple's first coordinate.
/// \param mask The sub-tuple mask.
/// \param subTuple Out: the sub-tuple's coordinates.
/// \returns The sub-tuple's size (the mask's population count).
std::size_t GatherSubTuple(std::span<const std::size_t> coordinates,
                           std::size_t offset,
                           unsigned mask,
                           std::span<std::size_t> subTuple) noexcept {
    std::size_t count = 0;
    std::size_t position = 0;

    for (unsigned bits = mask; bits != 0; bits >>= 1, ++position)
    {
        if ((bits & 1U) != 0U)
        {
            subTuple[count] = coordinates[offset + position];
            ++count;
        }
    }

    return count;
}

/// The energy path's density-free task predicate: Q_uv * Q_P against the
/// threshold. A pair with no bound keeps the task.
/// \param bra The orbital pair's bound.
/// \param ket The auxiliary shell's diagonal pair bound.
/// \param threshold The preset threshold with the slack.
/// \returns True when the task passes.
bool EnergyPredicate(const TwoElectronPairBound& bra,
                     const TwoElectronPairBound& ket,
                     double threshold) noexcept {
    if (!bra.hasBound() || !ket.hasBound())
    {
        return true;
    }

    return bra.value * ket.value >= threshold;
}

/// The next pair-data chunk of the one-electron derivative walk: the pairs
/// from `start` up to the first pair whose payload would push the chunk's
/// materialized pair data past `capBytes`. A pair heavier than the cap is its
/// own chunk, so zero walks one pair at a time.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): start, capBytes.
std::size_t NextPairChunk(const qcx::molecule::Molecule& molecule,
                          const qcx::basisset::BasisSet& basisSet,
                          const ShellPairList& pairList,
                          std::size_t start,
                          std::size_t capBytes,
                          std::vector<std::size_t>& chunkPairs) {
    chunkPairs.clear();
    std::size_t chunkPayloadBytes = 0;
    std::size_t end = start;

    while (end < pairList.pairs.size())
    {
        const std::size_t payload =
            internal::ChunkPairPayloadBytes(molecule, basisSet, pairList, end);

        if (end > start && chunkPayloadBytes + payload > capBytes)
        {
            break;
        }

        chunkPayloadBytes += payload;
        chunkPairs.push_back(end);
        ++end;
    }

    return end;
}

/// The nuclear charges the attraction builder sums, one per atom.
std::vector<double> NuclearCharges(const qcx::molecule::Molecule& molecule) {
    std::vector<double> charges;
    charges.reserve(molecule.AtomCount());

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        charges.push_back(static_cast<double>(atom.atomicNumber));
    }

    return charges;
}

/// Adds one pair's derivative blocks into the 3N AO derivative buffer: one
/// (nFuncsA x nFuncsB) block per requested coordinate. A diagonal shell pair
/// covers the full square; an off-diagonal pair covers one rectangle of a
/// symmetric matrix, so its transpose is added too.
/// \param pair The pair's contracted data.
/// \param offsetA Shell i's first function.
/// \param offsetB Shell j's first function.
/// \param coordinates The requested coordinates, one tuple each.
/// \param blocks The packed blocks, tuple-major.
/// \param target The 3N AO derivative matrices.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): offsetA, offsetB.
void ScatterPairBlocks(const MdPairData& pair,
                       std::size_t offsetA,
                       std::size_t offsetB,
                       std::span<const std::size_t> coordinates,
                       const std::vector<double>& blocks,
                       std::vector<Eigen::MatrixXd>& target) {
    const bool diagonal = offsetA == offsetB;

    for (std::size_t tuple = 0; tuple < coordinates.size(); ++tuple)
    {
        Eigen::MatrixXd& matrix = target[coordinates[tuple]];
        const double* block = blocks.data() + tuple * pair.nFuncs;

        for (std::size_t fa = 0; fa < pair.nFuncsA; ++fa)
        {
            for (std::size_t fb = 0; fb < pair.nFuncsB; ++fb)
            {
                const double value = block[fa * pair.nFuncsB + fb];
                matrix(static_cast<Eigen::Index>(offsetA + fa),
                       static_cast<Eigen::Index>(offsetB + fb)) += value;

                if (!diagonal)
                {
                    matrix(static_cast<Eigen::Index>(offsetB + fb),
                           static_cast<Eigen::Index>(offsetA + fa)) += value;
                }
            }
        }
    }
}

/// The four-target exchange contraction's energy value: sum over the
/// kernel's targets of D_target * (k / kMultiplicity), which is the walk's
/// accumulation of 1/2 Tr(D K).
/// \param block The packed quartet block.
/// \param shape The four shells' function counts.
/// \param density The density matrix.
/// \param offsets The four shells' first functions.
/// \param kMultiplicity The quartet's fixing-involution product.
/// \returns The exchange energy the block contributes.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): shape, offsets.
double ExchangeEnergyBlock(std::span<const double> block,
                           const std::array<std::size_t, 4>& shape,
                           const Eigen::MatrixXd& density,
                           const std::array<std::size_t, 4>& offsets,
                           double kMultiplicity) {
    const std::size_t nA = shape[0];
    const std::size_t nB = shape[1];
    const std::size_t nC = shape[2];
    const std::size_t nD = shape[3];
    const std::size_t oA = offsets[0];
    const std::size_t oB = offsets[1];
    const std::size_t oC = offsets[2];
    const std::size_t oD = offsets[3];
    double total = 0.0;

    // K1: target (a,c), density block (b,d).
    for (std::size_t fa = 0; fa < nA; ++fa)
    {
        for (std::size_t fc = 0; fc < nC; ++fc)
        {
            double k = 0.0;

            for (std::size_t fb = 0; fb < nB; ++fb)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    k += block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)] *
                         density(static_cast<Eigen::Index>(oB + fb),
                                 static_cast<Eigen::Index>(oD + fd));
                }
            }

            total +=
                density(static_cast<Eigen::Index>(oA + fa), static_cast<Eigen::Index>(oC + fc)) *
                (k / kMultiplicity);
        }
    }

    // K2: target (a,d), density block (b,c).
    for (std::size_t fa = 0; fa < nA; ++fa)
    {
        for (std::size_t fd = 0; fd < nD; ++fd)
        {
            double k = 0.0;

            for (std::size_t fb = 0; fb < nB; ++fb)
            {
                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    k += block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)] *
                         density(static_cast<Eigen::Index>(oB + fb),
                                 static_cast<Eigen::Index>(oC + fc));
                }
            }

            total +=
                density(static_cast<Eigen::Index>(oA + fa), static_cast<Eigen::Index>(oD + fd)) *
                (k / kMultiplicity);
        }
    }

    // K3: target (b,c), density block (a,d).
    for (std::size_t fb = 0; fb < nB; ++fb)
    {
        for (std::size_t fc = 0; fc < nC; ++fc)
        {
            double k = 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    k += block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)] *
                         density(static_cast<Eigen::Index>(oA + fa),
                                 static_cast<Eigen::Index>(oD + fd));
                }
            }

            total +=
                density(static_cast<Eigen::Index>(oB + fb), static_cast<Eigen::Index>(oC + fc)) *
                (k / kMultiplicity);
        }
    }

    // K4: target (b,d), density block (a,c).
    for (std::size_t fb = 0; fb < nB; ++fb)
    {
        for (std::size_t fd = 0; fd < nD; ++fd)
        {
            double k = 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    k += block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)] *
                         density(static_cast<Eigen::Index>(oA + fa),
                                 static_cast<Eigen::Index>(oC + fc));
                }
            }

            total +=
                density(static_cast<Eigen::Index>(oB + fb), static_cast<Eigen::Index>(oD + fd)) *
                (k / kMultiplicity);
        }
    }

    return total;
}

/// The four-target exchange contraction's derivative: the kernel's K'
/// accumulation into one coordinate's AO matrix, and the skeleton value
/// 1/2 Tr(D K') the same targets produce.
/// \param block The packed derivative block.
/// \param shape The four shells' function counts.
/// \param density The density matrix.
/// \param offsets The four shells' first functions.
/// \param kMultiplicity The quartet's fixing-involution product.
/// \param target In-out: the coordinate's AO K' matrix.
/// \returns The skeleton entry the block contributes.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): shape, offsets.
double AccumulateExchangeDerivative(std::span<const double> block,
                                    const std::array<std::size_t, 4>& shape,
                                    const Eigen::MatrixXd& density,
                                    const std::array<std::size_t, 4>& offsets,
                                    double kMultiplicity,
                                    Eigen::MatrixXd& target) {
    const std::size_t nA = shape[0];
    const std::size_t nB = shape[1];
    const std::size_t nC = shape[2];
    const std::size_t nD = shape[3];
    const std::size_t oA = offsets[0];
    const std::size_t oB = offsets[1];
    const std::size_t oC = offsets[2];
    const std::size_t oD = offsets[3];
    double skeleton = 0.0;

    // K1: target (a,c), density block (b,d). The transpose write is the
    // kernel's own: the pair of writes stands for the two ordered density
    // orientations, and it is emitted even when the two elements coincide.
    for (std::size_t fa = 0; fa < nA; ++fa)
    {
        for (std::size_t fc = 0; fc < nC; ++fc)
        {
            double k = 0.0;

            for (std::size_t fb = 0; fb < nB; ++fb)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    k += block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)] *
                         density(static_cast<Eigen::Index>(oB + fb),
                                 static_cast<Eigen::Index>(oD + fd));
                }
            }

            const double kScaled = k / kMultiplicity;
            target(static_cast<Eigen::Index>(oA + fa), static_cast<Eigen::Index>(oC + fc)) +=
                kScaled;
            target(static_cast<Eigen::Index>(oC + fc), static_cast<Eigen::Index>(oA + fa)) +=
                kScaled;
            skeleton +=
                density(static_cast<Eigen::Index>(oA + fa), static_cast<Eigen::Index>(oC + fc)) *
                kScaled;
        }
    }

    // K2: target (a,d), density block (b,c).
    for (std::size_t fa = 0; fa < nA; ++fa)
    {
        for (std::size_t fd = 0; fd < nD; ++fd)
        {
            double k = 0.0;

            for (std::size_t fb = 0; fb < nB; ++fb)
            {
                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    k += block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)] *
                         density(static_cast<Eigen::Index>(oB + fb),
                                 static_cast<Eigen::Index>(oC + fc));
                }
            }

            const double kScaled = k / kMultiplicity;
            target(static_cast<Eigen::Index>(oA + fa), static_cast<Eigen::Index>(oD + fd)) +=
                kScaled;
            target(static_cast<Eigen::Index>(oD + fd), static_cast<Eigen::Index>(oA + fa)) +=
                kScaled;
            skeleton +=
                density(static_cast<Eigen::Index>(oA + fa), static_cast<Eigen::Index>(oD + fd)) *
                kScaled;
        }
    }

    // K3: target (b,c), density block (a,d).
    for (std::size_t fb = 0; fb < nB; ++fb)
    {
        for (std::size_t fc = 0; fc < nC; ++fc)
        {
            double k = 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    k += block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)] *
                         density(static_cast<Eigen::Index>(oA + fa),
                                 static_cast<Eigen::Index>(oD + fd));
                }
            }

            const double kScaled = k / kMultiplicity;
            target(static_cast<Eigen::Index>(oB + fb), static_cast<Eigen::Index>(oC + fc)) +=
                kScaled;
            target(static_cast<Eigen::Index>(oC + fc), static_cast<Eigen::Index>(oB + fb)) +=
                kScaled;
            skeleton +=
                density(static_cast<Eigen::Index>(oB + fb), static_cast<Eigen::Index>(oC + fc)) *
                kScaled;
        }
    }

    // K4: target (b,d), density block (a,c).
    for (std::size_t fb = 0; fb < nB; ++fb)
    {
        for (std::size_t fd = 0; fd < nD; ++fd)
        {
            double k = 0.0;

            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                for (std::size_t fc = 0; fc < nC; ++fc)
                {
                    k += block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)] *
                         density(static_cast<Eigen::Index>(oA + fa),
                                 static_cast<Eigen::Index>(oC + fc));
                }
            }

            const double kScaled = k / kMultiplicity;
            target(static_cast<Eigen::Index>(oB + fb), static_cast<Eigen::Index>(oD + fd)) +=
                kScaled;
            target(static_cast<Eigen::Index>(oD + fd), static_cast<Eigen::Index>(oB + fb)) +=
                kScaled;
            skeleton +=
                density(static_cast<Eigen::Index>(oB + fb), static_cast<Eigen::Index>(oD + fd)) *
                kScaled;
        }
    }

    return skeleton;
}

/// The Coulomb derivative Fock block's accumulation into one coordinate's AO
/// matrix: J'_uv = sum_ls D_ls (uv|ls)'. The canonical pair pair stands for
/// the ordered target orientations - target (a, b) with density (c, d), and
/// target (c, d) with density (a, b) - through the pair multiplicities; a
/// same-pair quartet carries only the first, because the second is its own
/// transpose.
/// \param block The packed derivative block.
/// \param shape The four shells' function counts.
/// \param density The density matrix.
/// \param offsets The four shells' first functions.
/// \param braMultiplicity The bra pair's ordered multiplicity.
/// \param ketMultiplicity The ket pair's ordered multiplicity.
/// \param samePair True when the bra and ket canonical pairs coincide.
/// \param target In-out: the coordinate's AO J' matrix.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): shape, offsets, the multiplicities.
void AccumulateCoulombDerivativeBlock(std::span<const double> block,
                                      const std::array<std::size_t, 4>& shape,
                                      const Eigen::MatrixXd& density,
                                      const std::array<std::size_t, 4>& offsets,
                                      double braMultiplicity,
                                      double ketMultiplicity,
                                      bool samePair,
                                      Eigen::MatrixXd& target) {
    const std::size_t nA = shape[0];
    const std::size_t nB = shape[1];
    const std::size_t nC = shape[2];
    const std::size_t nD = shape[3];
    const std::size_t oA = offsets[0];
    const std::size_t oB = offsets[1];
    const std::size_t oC = offsets[2];
    const std::size_t oD = offsets[3];
    const bool braDiagonal = oA == oB;
    const bool ketDiagonal = oC == oD;

    for (std::size_t fa = 0; fa < nA; ++fa)
    {
        for (std::size_t fb = 0; fb < nB; ++fb)
        {
            const double densityA =
                density(static_cast<Eigen::Index>(oA + fa), static_cast<Eigen::Index>(oB + fb));

            for (std::size_t fc = 0; fc < nC; ++fc)
            {
                for (std::size_t fd = 0; fd < nD; ++fd)
                {
                    const double value = block[EriBlockIndex(fa, fb, fc, fd, nA, nB, nC, nD)];
                    const double densityB = density(static_cast<Eigen::Index>(oC + fc),
                                                    static_cast<Eigen::Index>(oD + fd));
                    const double weightA = ketMultiplicity * densityB;
                    target(static_cast<Eigen::Index>(oA + fa),
                           static_cast<Eigen::Index>(oB + fb)) += weightA * value;

                    if (!braDiagonal)
                    {
                        target(static_cast<Eigen::Index>(oB + fb),
                               static_cast<Eigen::Index>(oA + fa)) += weightA * value;
                    }

                    if (!samePair)
                    {
                        const double weightB = braMultiplicity * densityA;
                        target(static_cast<Eigen::Index>(oC + fc),
                               static_cast<Eigen::Index>(oD + fd)) += weightB * value;

                        if (!ketDiagonal)
                        {
                            target(static_cast<Eigen::Index>(oD + fd),
                                   static_cast<Eigen::Index>(oC + fc)) += weightB * value;
                        }
                    }
                }
            }
        }
    }
}

/// The mutable state of one RI derivative assembly's walks: the shells, the
/// density, the retained set, the factored inverse and the coefficients.
struct RiDerivativeWalk {
    const ShellPairList* pairList = nullptr; ///< The orbital shell pairs.
    const std::vector<MdShellInput>* shells = nullptr; ///< The flat orbital shells.
    const ShellPairList* auxPairList = nullptr; ///< The auxiliary shell pairs.
    const std::vector<MdShellInput>* auxShells = nullptr; ///< The flat auxiliary shells.
    const std::vector<MdShellInput>* phantoms = nullptr; ///< One per auxiliary shell.
    const Eigen::MatrixXd* densityMatrix = nullptr; ///< The density.
    const internal::RiTaskScreen* screen = nullptr; ///< The retained task set.
    const internal::FlooredMetricInverse* inverse = nullptr; ///< The metric's inverse.
    const Eigen::VectorXd* auxiliaryDensity = nullptr; ///< d over the retained set.
    const Eigen::VectorXd* coefficients = nullptr; ///< w = M^-1 d.
    std::size_t nAtoms = 0; ///< The geometry's atom count.
    MdEriDerivativeScratch* scratch = nullptr; ///< The tier's scratch.
    std::vector<Eigen::VectorXd> singletonDensity; ///< d^(a) per coordinate, cached.

    /// One target tuple's derivative, by the sub-tuple recursion.
    /// \param target The target's `order` coordinates, in request order.
    /// \returns The target's derivative value.
    qcx::Result<double> AssembleTuple(std::span<const std::size_t> target) {
        const std::size_t order = target.size();
        const unsigned full = (1U << static_cast<unsigned>(order)) - 1U;
        const std::size_t vectorSize = static_cast<std::size_t>(auxiliaryDensity->size());
        std::vector<Eigen::VectorXd> densitySub(full + 1U);
        std::vector<Eigen::VectorXd> coefficientSub(full + 1U);
        std::array<std::size_t, 2> subTuple{};
        densitySub[0] = *auxiliaryDensity;
        coefficientSub[0] = *coefficients;

        for (unsigned mask = 1; mask <= full; ++mask)
        {
            if (PopCount(mask) == 1)
            {
                std::size_t position = 0;

                while ((mask & (1U << position)) == 0U)
                {
                    ++position;
                }

                const std::size_t coordinate = target[position];

                if (singletonDensity[coordinate].size() == 0)
                {
                    Eigen::VectorXd accumulated =
                        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(vectorSize));
                    auto walked = internal::AccumulateAuxiliaryDensityDerivative(
                        *shells,
                        *pairList,
                        *auxShells,
                        *auxPairList,
                        *phantoms,
                        *densityMatrix,
                        *screen,
                        std::span<const std::size_t>(&coordinate, 1),
                        accumulated,
                        *scratch,
                        nAtoms);

                    if (!walked.has_value())
                    {
                        return std::unexpected(walked.error());
                    }

                    singletonDensity[coordinate] = std::move(accumulated);
                }

                densitySub[mask] = singletonDensity[coordinate];
                continue;
            }

            const std::size_t size = GatherSubTuple(target, 0, mask, subTuple);
            Eigen::VectorXd accumulated =
                Eigen::VectorXd::Zero(static_cast<Eigen::Index>(vectorSize));
            auto walked = internal::AccumulateAuxiliaryDensityDerivative(
                *shells,
                *pairList,
                *auxShells,
                *auxPairList,
                *phantoms,
                *densityMatrix,
                *screen,
                std::span<const std::size_t>(subTuple.data(), size),
                accumulated,
                *scratch,
                nAtoms);

            if (!walked.has_value())
            {
                return std::unexpected(walked.error());
            }

            densitySub[mask] = std::move(accumulated);
        }

        for (int size = 1; size <= static_cast<int>(order); ++size)
        {
            for (unsigned mask = 1; mask <= full; ++mask)
            {
                if (PopCount(mask) != size)
                {
                    continue;
                }

                Eigen::VectorXd rightHandSide = densitySub[mask];

                for (unsigned sub = mask; sub != 0U; sub = (sub - 1U) & mask)
                {
                    const std::size_t subSize = GatherSubTuple(target, 0, sub, subTuple);
                    Eigen::VectorXd applied =
                        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(vectorSize));
                    auto appliedMetric = internal::ApplyMetricDerivative(
                        *auxShells,
                        *auxPairList,
                        *phantoms,
                        std::span<const std::size_t>(subTuple.data(), subSize),
                        coefficientSub[mask ^ sub],
                        applied,
                        *scratch,
                        nAtoms);

                    if (!appliedMetric.has_value())
                    {
                        return std::unexpected(appliedMetric.error());
                    }

                    rightHandSide -= applied;
                }

                coefficientSub[mask] = inverse->Apply(rightHandSide);
            }
        }

        double value = 0.0;

        for (unsigned sub = 0; sub <= full; ++sub)
        {
            value += densitySub[sub].dot(coefficientSub[full ^ sub]);
        }

        return 0.5 * value;
    }
};

} // namespace

qcx::Result<RiCoulombDerivative> ComputeRiCoulombDerivative(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const CpuTensor2& density,
    std::span<const std::size_t> coordinates,
    const RiCoulombDerivativeOptions& options) {
    if (options.order < 1 || options.order > internal::kMaxMdDerivativeOrder)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the RI Coulomb derivative order " +
                                              std::to_string(options.order) + " is outside 1.." +
                                              std::to_string(internal::kMaxMdDerivativeOrder)});
    }

    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the RI Coulomb derivative batch byte cap must be positive"});
    }

    const std::size_t order = static_cast<std::size_t>(options.order);

    if (coordinates.size() % order != 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the " + std::to_string(coordinates.size()) +
                           " RI Coulomb derivative coordinates do not divide into tuples of "
                           "order " +
                           std::to_string(options.order)});
    }

    auto pairList = CheckedShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto auxPairList = CheckedShellPairs(molecule, auxBasisSet);

    if (!auxPairList.has_value())
    {
        return std::unexpected(auxPairList.error());
    }

    const std::size_t n = pairList->functionCount;
    const std::size_t nAux = auxPairList->functionCount;

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the density matrix is not nBasis x nBasis for this basis set"});
    }

    const std::size_t nAtoms = molecule.AtomCount();

    for (const std::size_t coordinate : coordinates)
    {
        if (coordinate >= 3 * nAtoms)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the RI Coulomb derivative coordinate " + std::to_string(coordinate) +
                               " names atom " + std::to_string(coordinate / 3) + ", outside the " +
                               std::to_string(nAtoms) + " of the geometry"});
        }
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    auto auxShells = internal::FlattenShells(molecule, auxBasisSet, *auxPairList);

    if (!auxShells.has_value())
    {
        return std::unexpected(auxShells.error());
    }

    const std::size_t tuples = coordinates.size() / order;
    RiCoulombDerivative result;
    result.values = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(tuples));
    result.counts.orbitalPairs = pairList->pairs.size();
    result.counts.auxShells = auxShells->size();
    result.counts.tensorTasksOffered = pairList->pairs.size() * auxShells->size();
    result.counts.metricQuartets = auxPairList->pairs.size();
    result.counts.order = options.order;
    result.counts.tuples = tuples;

    if (n == 0 || nAux == 0)
    {
        return result;
    }

    // The energy path's metric, through the blocked build that writes the
    // symmetric copy directly into the returned matrix.
    RiEngineOptions engineOptions;
    engineOptions.accuracy = options.accuracy;
    engineOptions.maxBatchBytes = options.maxBatchBytes;
    engineOptions.metricFloorEpsilon = options.metricFloorEpsilon;
    auto metric = BuildBlockedAuxMetric(molecule, auxBasisSet, engineOptions);

    if (!metric.has_value())
    {
        return std::unexpected(metric.error());
    }

    internal::FlooredMetricInverse inverse;
    auto factored = inverse.Init(*metric, options.metricFloorEpsilon);

    if (!factored.has_value())
    {
        return std::unexpected(factored.error());
    }

    Eigen::MatrixXd densityMatrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            densityMatrix(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v)) =
                density(u, v);
        }
    }

    // The phantom each auxiliary shell is paired with, one per auxiliary
    // shell, at that shell's own centre.
    std::vector<MdShellInput> phantoms;
    phantoms.reserve(auxShells->size());

    for (const MdShellInput& shell : *auxShells)
    {
        phantoms.push_back(internal::PhantomShell({shell.cx, shell.cy, shell.cz}));
    }

    internal::RiTaskScreen screen;
    Eigen::VectorXd coefficients;

    if (options.useDensityScreening)
    {
        auto pairBounds = ComputeTwoElectronPairBounds(molecule, basisSet);

        if (!pairBounds.has_value())
        {
            return std::unexpected(pairBounds.error());
        }

        auto auxPairBounds = ComputeTwoElectronPairBounds(molecule, auxBasisSet);

        if (!auxPairBounds.has_value())
        {
            return std::unexpected(auxPairBounds.error());
        }

        screen.active = true;
        screen.threshold =
            SchwarzThreshold(options.accuracy) / (1.0 + internal::kNeighborListSlack);
        screen.pairBounds = std::move(*pairBounds);
        screen.pairDensity = internal::BuildShellPairMaxDensity(densityMatrix, *pairList);
        screen.auxDiagonalBounds.resize(auxShells->size());

        for (std::size_t auxShell = 0; auxShell < auxShells->size(); ++auxShell)
        {
            screen.auxDiagonalBounds[auxShell] =
                (*auxPairBounds)[PairIndexOf(auxShell, auxShell, *auxPairList)];
        }

        // The energy path's own retained set first: it is density-free, so
        // its d is what the coefficients the derivative predicate reads are
        // solved from.
        Eigen::VectorXd energyDensity = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nAux));
        MdEriDerivativeScratch scratch;
        std::vector<double> value;

        for (std::size_t pairIndex = 0; pairIndex < pairList->pairs.size(); ++pairIndex)
        {
            const ShellPairIndex& pair = pairList->pairs[pairIndex];
            const ShellInfo& braInfo = pairList->shells[pair.i];
            const ShellInfo& ketInfo = pairList->shells[pair.j];
            const std::size_t nI = ShellFunctionCount(braInfo);
            const std::size_t nJ = ShellFunctionCount(ketInfo);
            const double pairFactor = static_cast<double>(internal::PairMultiplicity(pair));

            for (std::size_t auxShell = 0; auxShell < auxShells->size(); ++auxShell)
            {
                const ShellInfo& auxInfo = auxPairList->shells[auxShell];

                if (!EnergyPredicate(screen.pairBounds[pairIndex],
                                     screen.auxDiagonalBounds[auxShell],
                                     screen.threshold))
                {
                    continue;
                }

                MdEriDerivativeQuartet quartet =
                    internal::TaskQuartet(*shells, pair, *auxShells, phantoms, auxShell);
                quartet.atoms = {
                    braInfo.atomIndex, ketInfo.atomIndex, auxInfo.atomIndex, auxInfo.atomIndex};
                const std::size_t nK = ShellFunctionCount(auxInfo);
                value.assign(nI * nJ * nK, 0.0);
                internal::BuildEriQuartetValue(quartet, 1, value, scratch);
                internal::AccumulateTaskDensity(
                    value,
                    nI,
                    nJ,
                    nK,
                    pairFactor,
                    densityMatrix,
                    braInfo.functionOffset,
                    ketInfo.functionOffset,
                    auxInfo.functionOffset,
                    std::span<double>(energyDensity.data(),
                                      static_cast<std::size_t>(energyDensity.size())));
            }
        }

        const Eigen::VectorXd initialCoefficients = inverse.Apply(energyDensity);
        screen.auxWeightMax.resize(auxShells->size());

        for (std::size_t auxShell = 0; auxShell < auxShells->size(); ++auxShell)
        {
            const ShellInfo& auxInfo = auxPairList->shells[auxShell];
            screen.auxWeightMax[auxShell] = internal::AuxiliaryWeightMax(
                initialCoefficients, auxInfo.functionOffset, ShellFunctionCount(auxInfo));
        }
    }

    // The retained set's own d, the one the reported energy and coefficients
    // are defined by.
    Eigen::VectorXd auxiliaryDensity = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nAux));
    MdEriDerivativeScratch scratch;
    std::vector<double> value;

    for (std::size_t pairIndex = 0; pairIndex < pairList->pairs.size(); ++pairIndex)
    {
        const ShellPairIndex& pair = pairList->pairs[pairIndex];
        const ShellInfo& braInfo = pairList->shells[pair.i];
        const ShellInfo& ketInfo = pairList->shells[pair.j];
        const std::size_t nI = ShellFunctionCount(braInfo);
        const std::size_t nJ = ShellFunctionCount(ketInfo);
        const double pairFactor = static_cast<double>(internal::PairMultiplicity(pair));

        for (std::size_t auxShell = 0; auxShell < auxShells->size(); ++auxShell)
        {
            if (!screen.Retains(pairIndex, auxShell, pairFactor))
            {
                continue;
            }

            const ShellInfo& auxInfo = auxPairList->shells[auxShell];
            MdEriDerivativeQuartet quartet =
                internal::TaskQuartet(*shells, pair, *auxShells, phantoms, auxShell);
            quartet.atoms = {
                braInfo.atomIndex, ketInfo.atomIndex, auxInfo.atomIndex, auxInfo.atomIndex};
            const std::size_t nK = ShellFunctionCount(auxInfo);
            value.assign(nI * nJ * nK, 0.0);
            internal::BuildEriQuartetValue(quartet, 1, value, scratch);
            internal::AccumulateTaskDensity(
                value,
                nI,
                nJ,
                nK,
                pairFactor,
                densityMatrix,
                braInfo.functionOffset,
                ketInfo.functionOffset,
                auxInfo.functionOffset,
                std::span<double>(auxiliaryDensity.data(),
                                  static_cast<std::size_t>(auxiliaryDensity.size())));
            ++result.counts.tensorQuartets;
        }
    }

    coefficients = inverse.Apply(auxiliaryDensity);
    result.energy = 0.5 * auxiliaryDensity.dot(coefficients);

    if (tuples == 0)
    {
        return result;
    }

    if (order == 1)
    {
        // The closed form: E^(a) = d^(a) . w - 1/2 w^T M^(a) w.
        std::vector<std::size_t> targetCoordinates(coordinates.begin(), coordinates.end());
        std::sort(targetCoordinates.begin(), targetCoordinates.end());
        targetCoordinates.erase(std::unique(targetCoordinates.begin(), targetCoordinates.end()),
                                targetCoordinates.end());
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(3 * nAtoms));
        std::vector<std::size_t> active;
        std::vector<double> weights;
        std::vector<double> derivatives;

        for (std::size_t pairIndex = 0; pairIndex < pairList->pairs.size(); ++pairIndex)
        {
            const ShellPairIndex& pair = pairList->pairs[pairIndex];
            const ShellInfo& braInfo = pairList->shells[pair.i];
            const ShellInfo& ketInfo = pairList->shells[pair.j];
            const std::size_t nI = ShellFunctionCount(braInfo);
            const std::size_t nJ = ShellFunctionCount(ketInfo);
            const double pairFactor = static_cast<double>(internal::PairMultiplicity(pair));

            for (std::size_t auxShell = 0; auxShell < auxShells->size(); ++auxShell)
            {
                if (!screen.Retains(pairIndex, auxShell, pairFactor))
                {
                    continue;
                }

                const ShellInfo& auxInfo = auxPairList->shells[auxShell];
                active.clear();

                for (const std::size_t coordinate : targetCoordinates)
                {
                    const std::size_t atom = coordinate / 3;

                    if (atom == braInfo.atomIndex || atom == ketInfo.atomIndex ||
                        atom == auxInfo.atomIndex)
                    {
                        active.push_back(coordinate);
                    }
                }

                if (active.empty())
                {
                    continue;
                }

                MdEriDerivativeQuartet quartet =
                    internal::TaskQuartet(*shells, pair, *auxShells, phantoms, auxShell);
                quartet.atoms = {
                    braInfo.atomIndex, ketInfo.atomIndex, auxInfo.atomIndex, auxInfo.atomIndex};
                const std::size_t nK = ShellFunctionCount(auxInfo);
                const std::size_t elements = nI * nJ * nK;
                weights.assign(elements, 0.0);

                for (std::size_t fa = 0; fa < nI; ++fa)
                {
                    for (std::size_t fb = 0; fb < nJ; ++fb)
                    {
                        const double densityElement =
                            densityMatrix(static_cast<Eigen::Index>(braInfo.functionOffset + fa),
                                          static_cast<Eigen::Index>(ketInfo.functionOffset + fb));

                        for (std::size_t fp = 0; fp < nK; ++fp)
                        {
                            const double wP = coefficients(
                                static_cast<Eigen::Index>(auxInfo.functionOffset + fp));
                            weights[EriBlockIndex(fa, fb, fp, 0, nI, nJ, nK, 1)] =
                                pairFactor * densityElement * wP;
                        }
                    }
                }

                std::size_t first = 0;

                while (first < active.size())
                {
                    const std::size_t perBatch = std::max<std::size_t>(
                        1, options.maxBatchBytes / (elements * sizeof(double)));
                    const std::size_t count = std::min(perBatch, active.size() - first);
                    const std::span<const std::size_t> chunk(active.data() + first, count);
                    derivatives.assign(count * elements, 0.0);
                    auto differentiated = internal::BuildEriQuartetDerivative(
                        quartet, 1, chunk, derivatives, scratch, nAtoms);

                    if (!differentiated.has_value())
                    {
                        return std::unexpected(differentiated.error());
                    }

                    for (std::size_t tuple = 0; tuple < count; ++tuple)
                    {
                        const std::span<const double> block(derivatives.data() + tuple * elements,
                                                            elements);
                        gradient(static_cast<Eigen::Index>(chunk[tuple])) +=
                            DotProduct(weights, block);
                    }

                    first += count;
                }
            }
        }

        // The metric term: -1/2 sum_PQ w_P w_Q d(M_PQ)/dR, differentiated
        // whole and integrated with both orientations of each off-diagonal
        // auxiliary pair.
        std::vector<std::size_t> ownCoordinates;

        for (const ShellPairIndex& pair : auxPairList->pairs)
        {
            const ShellInfo& braInfo = auxPairList->shells[pair.i];
            const ShellInfo& ketInfo = auxPairList->shells[pair.j];
            const std::size_t nI = ShellFunctionCount(braInfo);
            const std::size_t nJ = ShellFunctionCount(ketInfo);
            const double pairFactor = static_cast<double>(internal::PairMultiplicity(pair));
            internal::QuartetCoordinates(
                {braInfo.atomIndex, braInfo.atomIndex, ketInfo.atomIndex, ketInfo.atomIndex},
                ownCoordinates);
            MdEriDerivativeQuartet quartet;
            quartet.shells = {&phantoms[pair.i],
                              auxShells->data() + pair.i,
                              &phantoms[pair.j],
                              auxShells->data() + pair.j};
            quartet.atoms = {
                braInfo.atomIndex, braInfo.atomIndex, ketInfo.atomIndex, ketInfo.atomIndex};
            const std::size_t elements = nI * nJ;
            weights.assign(elements, 0.0);

            for (std::size_t fp = 0; fp < nI; ++fp)
            {
                const double wP =
                    coefficients(static_cast<Eigen::Index>(braInfo.functionOffset + fp));

                for (std::size_t fq = 0; fq < nJ; ++fq)
                {
                    const double wQ =
                        coefficients(static_cast<Eigen::Index>(ketInfo.functionOffset + fq));
                    weights[EriBlockIndex(0, fp, 0, fq, 1, nI, 1, nJ)] =
                        -0.5 * pairFactor * wP * wQ;
                }
            }

            derivatives.assign(ownCoordinates.size() * elements, 0.0);
            auto differentiated = internal::BuildEriQuartetDerivative(
                quartet, 1, ownCoordinates, derivatives, scratch, nAtoms);

            if (!differentiated.has_value())
            {
                return std::unexpected(differentiated.error());
            }

            for (std::size_t tuple = 0; tuple < ownCoordinates.size(); ++tuple)
            {
                const std::span<const double> block(derivatives.data() + tuple * elements,
                                                    elements);
                gradient(static_cast<Eigen::Index>(ownCoordinates[tuple])) +=
                    DotProduct(weights, block);
            }
        }

        for (std::size_t tuple = 0; tuple < tuples; ++tuple)
        {
            result.values(static_cast<Eigen::Index>(tuple)) =
                gradient(static_cast<Eigen::Index>(coordinates[tuple]));
        }

        return result;
    }

    // Order 2: the sub-tuple recursion, one target at a time.
    RiDerivativeWalk walk;
    walk.pairList = &*pairList;
    walk.shells = &*shells;
    walk.auxPairList = &*auxPairList;
    walk.auxShells = &*auxShells;
    walk.phantoms = &phantoms;
    walk.densityMatrix = &densityMatrix;
    walk.screen = &screen;
    walk.inverse = &inverse;
    walk.auxiliaryDensity = &auxiliaryDensity;
    walk.coefficients = &coefficients;
    walk.nAtoms = nAtoms;
    walk.scratch = &scratch;
    walk.singletonDensity.resize(3 * nAtoms);

    for (std::size_t tuple = 0; tuple < tuples; ++tuple)
    {
        const std::span<const std::size_t> target(coordinates.data() + tuple * order, order);
        auto assembled = walk.AssembleTuple(target);

        if (!assembled.has_value())
        {
            return std::unexpected(assembled.error());
        }

        result.values(static_cast<Eigen::Index>(tuple)) = *assembled;
    }

    return result;
}

qcx::Result<RiCoulombDerivative> ComputeRiCoulombDerivative(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const CpuTensor2& density,
    const RiCoulombDerivativeOptions& options) {
    std::vector<std::size_t> coordinates;

    if (options.order == 1)
    {
        coordinates.resize(3 * molecule.AtomCount());

        for (std::size_t flat = 0; flat < coordinates.size(); ++flat)
        {
            coordinates[flat] = flat;
        }
    } else if (options.order == 2)
    {
        const std::size_t count = 3 * molecule.AtomCount();
        coordinates.assign(2 * count * count, 0);

        for (std::size_t first = 0; first < count; ++first)
        {
            for (std::size_t second = 0; second < count; ++second)
            {
                coordinates[2 * (first * count + second)] = first;
                coordinates[2 * (first * count + second) + 1] = second;
            }
        }
    }

    return ComputeRiCoulombDerivative(molecule,
                                      basisSet,
                                      auxBasisSet,
                                      density,
                                      std::span<const std::size_t>(coordinates),
                                      options);
}

qcx::Result<RiCoulombGradient> ComputeRiCoulombGradient(const qcx::molecule::Molecule& molecule,
                                                        const qcx::basisset::BasisSet& basisSet,
                                                        const qcx::basisset::BasisSet& auxBasisSet,
                                                        const CpuTensor2& density,
                                                        const RiEngineOptions& options) {
    RiCoulombDerivativeOptions derivativeOptions;
    derivativeOptions.order = 1;
    derivativeOptions.useDensityScreening = true;
    derivativeOptions.accuracy = options.accuracy;
    derivativeOptions.metricFloorEpsilon = options.metricFloorEpsilon;
    derivativeOptions.maxBatchBytes = options.maxBatchBytes;
    auto derivative =
        ComputeRiCoulombDerivative(molecule, basisSet, auxBasisSet, density, derivativeOptions);

    if (!derivative.has_value())
    {
        return std::unexpected(derivative.error());
    }

    RiCoulombGradient result;
    result.gradient = std::move(derivative->values);
    result.energy = derivative->energy;
    result.counts.orbitalPairs = derivative->counts.orbitalPairs;
    result.counts.auxShells = derivative->counts.auxShells;
    result.counts.tensorQuartets = derivative->counts.tensorQuartets;
    result.counts.metricQuartets = derivative->counts.metricQuartets;
    return result;
}

qcx::Result<OneElectronFockDerivative> ComputeOneElectronFockDerivative(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    std::span<const double> moCoefficients,
    std::size_t numOccupied,
    std::size_t numVirtual,
    const OneElectronFockDerivativeOptions& options) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    const std::size_t n = pairList->functionCount;
    const std::size_t nMo = numOccupied + numVirtual;

    if (nMo > n)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the " + std::to_string(numOccupied) + " occupied and " + std::to_string(numVirtual) +
                " virtual orbitals exceed the " + std::to_string(n) + " basis functions"});
    }

    if (moCoefficients.size() != n * nMo)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the MO coefficient span holds " +
                                              std::to_string(moCoefficients.size()) +
                                              " values, not the " + std::to_string(n * nMo) +
                                              " of nBasis x (numOccupied + numVirtual)"});
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::size_t nAtoms = molecule.AtomCount();
    const std::size_t coordinateCount = 3 * nAtoms;
    OneElectronFockDerivative result;
    result.moOverlapDerivative.assign(coordinateCount * nMo * nMo, 0.0);
    result.occupiedVirtual =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(coordinateCount * numVirtual),
                              static_cast<Eigen::Index>(numOccupied));

    if (n == 0 || pairList->pairs.empty() || nMo == 0)
    {
        return result;
    }

    const MoMatrix coefficients(
        moCoefficients.data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(nMo));
    const std::vector<double> charges = NuclearCharges(molecule);
    const std::vector<Eigen::Vector3d> centers = internal::AtomPositions(molecule);
    std::vector<std::size_t> allCoordinates(coordinateCount);

    for (std::size_t flat = 0; flat < allCoordinates.size(); ++flat)
    {
        allCoordinates[flat] = flat;
    }

    std::vector<Eigen::MatrixXd> overlapAO;
    std::vector<Eigen::MatrixXd> hamiltonianAO;
    overlapAO.reserve(coordinateCount);
    hamiltonianAO.reserve(coordinateCount);

    for (std::size_t coordinate = 0; coordinate < coordinateCount; ++coordinate)
    {
        overlapAO.push_back(
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));
        hamiltonianAO.push_back(
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));
    }

    std::vector<MdPairData> store(pairList->pairs.size());
    internal::FillPairGeometry(store, *shells, *pairList);
    internal::MdDerivativeScratch scratch;
    std::vector<std::size_t> chunkPairs;
    std::vector<std::size_t> ownCoordinates;
    std::vector<double> blocks;

    for (std::size_t start = 0; start < pairList->pairs.size();)
    {
        const std::size_t end = NextPairChunk(
            molecule, basisSet, *pairList, start, options.maxPairChunkBytes, chunkPairs);
        internal::BuildChunkPairData(store, *shells, *pairList, chunkPairs);
        ++result.counts.pairChunks;

        for (const std::size_t pairIndex : chunkPairs)
        {
            const ShellPairIndex& index = pairList->pairs[pairIndex];
            const MdPairData& pair = store[pairIndex];
            const std::size_t offsetA = pairList->shells[index.i].functionOffset;
            const std::size_t offsetB = pairList->shells[index.j].functionOffset;
            const std::size_t braAtom = pairList->shells[index.i].atomIndex;
            const std::size_t ketAtom = pairList->shells[index.j].atomIndex;
            ownCoordinates.clear();

            for (int axis = 0; axis < 3; ++axis)
            {
                ownCoordinates.push_back(3 * braAtom + static_cast<std::size_t>(axis));
            }

            if (ketAtom != braAtom)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    ownCoordinates.push_back(3 * ketAtom + static_cast<std::size_t>(axis));
                }
            }

            blocks.assign(ownCoordinates.size() * pair.nFuncs, 0.0);
            auto overlap = internal::BuildOverlapPairDerivative(
                pair, braAtom, ketAtom, 1, ownCoordinates, blocks, scratch);

            if (!overlap.has_value())
            {
                return std::unexpected(overlap.error());
            }

            ScatterPairBlocks(pair, offsetA, offsetB, ownCoordinates, blocks, overlapAO);

            blocks.assign(ownCoordinates.size() * pair.nFuncs, 0.0);
            auto kinetic = internal::BuildKineticPairDerivative(
                pair, braAtom, ketAtom, 1, ownCoordinates, blocks, scratch);

            if (!kinetic.has_value())
            {
                return std::unexpected(kinetic.error());
            }

            ScatterPairBlocks(pair, offsetA, offsetB, ownCoordinates, blocks, hamiltonianAO);

            blocks.assign(allCoordinates.size() * pair.nFuncs, 0.0);
            auto nuclear = internal::BuildNuclearPairDerivative(
                pair, braAtom, ketAtom, charges, centers, 1, allCoordinates, blocks, scratch);

            if (!nuclear.has_value())
            {
                return std::unexpected(nuclear.error());
            }

            ScatterPairBlocks(pair, offsetA, offsetB, allCoordinates, blocks, hamiltonianAO);

            ++result.counts.orbitalPairs;
            result.counts.overlapTuples += ownCoordinates.size();
            result.counts.hamiltonianTuples += ownCoordinates.size() + allCoordinates.size();
        }

        internal::ReleaseChunkPairData(store, chunkPairs);
        start = end;
    }

    for (std::size_t coordinate = 0; coordinate < coordinateCount; ++coordinate)
    {
        const Eigen::MatrixXd overlapBlock =
            coefficients.transpose() * overlapAO[coordinate] * coefficients;
        double* overlapOut = result.moOverlapDerivative.data() + coordinate * nMo * nMo;

        for (std::size_t p = 0; p < nMo; ++p)
        {
            for (std::size_t q = 0; q < nMo; ++q)
            {
                overlapOut[p * nMo + q] =
                    overlapBlock(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q));
            }
        }

        const Eigen::MatrixXd hamiltonianBlock =
            coefficients.transpose() * hamiltonianAO[coordinate] * coefficients;

        for (std::size_t virtualOrbital = 0; virtualOrbital < numVirtual; ++virtualOrbital)
        {
            for (std::size_t occupied = 0; occupied < numOccupied; ++occupied)
            {
                result.occupiedVirtual(
                    static_cast<Eigen::Index>(coordinate * numVirtual + virtualOrbital),
                    static_cast<Eigen::Index>(occupied)) =
                    hamiltonianBlock(static_cast<Eigen::Index>(numOccupied + virtualOrbital),
                                     static_cast<Eigen::Index>(occupied));
            }
        }
    }

    return result;
}

qcx::Result<TwoElectronFockDerivative> ComputeTwoElectronFockDerivative(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const CpuTensor2& density,
    std::span<const double> moCoefficients,
    std::size_t numOccupied,
    std::size_t numVirtual,
    const TwoElectronFockDerivativeOptions& options) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    for (const ShellInfo& shell : pairList->shells)
    {
        if (!SupportsL(shell.angularMomentum))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "shell angular momentum exceeds kMaxEngineL of this build"});
        }
    }

    const std::size_t n = pairList->functionCount;

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the density matrix is not nBasis x nBasis for this basis set"});
    }

    const std::size_t nMo = numOccupied + numVirtual;

    if (nMo > n)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the " + std::to_string(numOccupied) + " occupied and " + std::to_string(numVirtual) +
                " virtual orbitals exceed the " + std::to_string(n) + " basis functions"});
    }

    if (moCoefficients.size() != n * nMo)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the MO coefficient span holds " +
                                              std::to_string(moCoefficients.size()) +
                                              " values, not the " + std::to_string(n * nMo) +
                                              " of nBasis x (numOccupied + numVirtual)"});
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::size_t nAtoms = molecule.AtomCount();
    const std::size_t coordinateCount = 3 * nAtoms;
    TwoElectronFockDerivative result;
    result.coulombOccupiedVirtual =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(coordinateCount * numVirtual),
                              static_cast<Eigen::Index>(numOccupied));
    result.exchangeOccupiedVirtual =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(coordinateCount * numVirtual),
                              static_cast<Eigen::Index>(numOccupied));
    result.coulombSkeleton = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(coordinateCount));
    result.exchangeSkeleton = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(coordinateCount));

    if (n == 0 || pairList->pairs.empty() || nMo == 0)
    {
        return result;
    }

    const MoMatrix coefficients(
        moCoefficients.data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(nMo));
    Eigen::MatrixXd densityMatrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            densityMatrix(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v)) =
                density(u, v);
        }
    }

    std::vector<TwoElectronPairBound> bounds;
    std::vector<double> pairDensity;
    std::vector<double> pairCMax(pairList->pairs.size(), 0.0);
    double threshold = SchwarzThreshold(options.accuracy) / (1.0 + internal::kNeighborListSlack);

    if (options.useDensityScreening)
    {
        if (!options.warmStart.pairValues.empty() || !options.warmStart.pairDerivatives.empty())
        {
            auto lifted = LiftFockDerivativeWarmStart(options.warmStart, pairList->pairs.size());

            if (!lifted.has_value())
            {
                return std::unexpected(lifted.error());
            }

            bounds = std::move(*lifted);

            if (options.warmStart.threshold > 0.0)
            {
                threshold = options.warmStart.threshold;
            }
        } else
        {
            auto computed = ComputeTwoElectronPairBounds(molecule, basisSet);

            if (!computed.has_value())
            {
                return std::unexpected(computed.error());
            }

            bounds = std::move(*computed);
        }

        pairDensity = internal::BuildShellPairMaxDensity(densityMatrix, *pairList);
        std::vector<double> functionCMax(n, 0.0);

        for (std::size_t p = 0; p < n; ++p)
        {
            for (std::size_t column = 0; column < nMo; ++column)
            {
                functionCMax[p] =
                    std::max(functionCMax[p],
                             std::abs(coefficients(static_cast<Eigen::Index>(p),
                                                   static_cast<Eigen::Index>(column))));
            }
        }

        for (std::size_t pairIndex = 0; pairIndex < pairList->pairs.size(); ++pairIndex)
        {
            const ShellPairIndex& pair = pairList->pairs[pairIndex];
            const ShellInfo& braInfo = pairList->shells[pair.i];
            const ShellInfo& ketInfo = pairList->shells[pair.j];

            for (std::size_t f = 0; f < ShellFunctionCount(braInfo); ++f)
            {
                pairCMax[pairIndex] =
                    std::max(pairCMax[pairIndex], functionCMax[braInfo.functionOffset + f]);
            }

            for (std::size_t f = 0; f < ShellFunctionCount(ketInfo); ++f)
            {
                pairCMax[pairIndex] =
                    std::max(pairCMax[pairIndex], functionCMax[ketInfo.functionOffset + f]);
            }
        }
    }

    std::vector<Eigen::MatrixXd> coulombAO;
    std::vector<Eigen::MatrixXd> exchangeAO;
    coulombAO.reserve(coordinateCount);
    exchangeAO.reserve(coordinateCount);

    for (std::size_t coordinate = 0; coordinate < coordinateCount; ++coordinate)
    {
        coulombAO.push_back(
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));
        exchangeAO.push_back(
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));
    }

    MdEriDerivativeScratch scratch;
    std::vector<std::size_t> coordinates;
    std::vector<double> value;
    std::vector<double> derivatives;
    std::vector<double> weights;
    const std::size_t pairCount = pairList->pairs.size();

    for (std::size_t braIndex = 0; braIndex < pairCount; ++braIndex)
    {
        const ShellPairIndex& braPair = pairList->pairs[braIndex];
        const ShellInfo& braI = pairList->shells[braPair.i];
        const ShellInfo& braJ = pairList->shells[braPair.j];
        const std::size_t nI = ShellFunctionCount(braI);
        const std::size_t nJ = ShellFunctionCount(braJ);
        const std::size_t braMultiplicity = internal::PairMultiplicity(braPair);
        ++result.counts.braPairs;

        for (std::size_t ketIndex = 0; ketIndex <= braIndex; ++ketIndex)
        {
            ++result.counts.ketPairs;
            const ShellPairIndex& ketPair = pairList->pairs[ketIndex];
            const ShellInfo& ketK = pairList->shells[ketPair.i];
            const ShellInfo& ketL = pairList->shells[ketPair.j];
            const std::size_t nK = ShellFunctionCount(ketK);
            const std::size_t nL = ShellFunctionCount(ketL);
            const std::size_t ketMultiplicity = internal::PairMultiplicity(ketPair);
            const std::size_t pairPairFactor = braIndex == ketIndex ? 1 : 2;
            const double permutationWeight =
                0.5 * static_cast<double>(pairPairFactor * braMultiplicity * ketMultiplicity);
            bool keepCoulomb = true;
            bool keepCoulombBlock = true;
            bool keepExchange = true;

            if (options.useDensityScreening)
            {
                const QuartetDerivativeBound bound =
                    QuartetDerivativeProduct(bounds[braIndex], bounds[ketIndex]);
                const double largest = std::max(bound.value, bound.derivative);
                const double coulombScale =
                    permutationWeight * pairDensity[braIndex] * pairDensity[ketIndex];
                keepCoulomb = coulombScale * largest >= threshold;
                const double coulombDerivativeScale =
                    4.0 * (pairCMax[braIndex] * pairCMax[braIndex] * pairDensity[ketIndex] +
                           pairCMax[ketIndex] * pairCMax[ketIndex] * pairDensity[braIndex]);
                keepCoulombBlock = coulombDerivativeScale * largest >= threshold;
                const double crossDensity = std::max(
                    {pairDensity[internal::CanonicalPairIndexOf(braPair.i, ketPair.i, *pairList)],
                     pairDensity[internal::CanonicalPairIndexOf(braPair.i, ketPair.j, *pairList)],
                     pairDensity[internal::CanonicalPairIndexOf(braPair.j, ketPair.i, *pairList)],
                     pairDensity[internal::CanonicalPairIndexOf(braPair.j, ketPair.j, *pairList)]});
                const double exchangeScale =
                    8.0 * pairCMax[braIndex] * pairCMax[ketIndex] * crossDensity;
                keepExchange = exchangeScale * largest >= threshold;
            }

            if (!keepCoulomb && !keepCoulombBlock && !keepExchange)
            {
                ++result.counts.screenedQuartets;
                continue;
            }

            MdEriDerivativeQuartet quartet;
            quartet.shells = {&(*shells)[braPair.i],
                              &(*shells)[braPair.j],
                              &(*shells)[ketPair.i],
                              &(*shells)[ketPair.j]};
            quartet.atoms = {braI.atomIndex, braJ.atomIndex, ketK.atomIndex, ketL.atomIndex};
            internal::QuartetCoordinates(quartet.atoms, coordinates);
            const std::size_t elements = internal::EriQuartetBlockElements(quartet);
            derivatives.assign(coordinates.size() * elements, 0.0);
            auto differentiated = internal::BuildEriQuartetDerivative(
                quartet, 1, coordinates, derivatives, scratch, nAtoms);

            if (!differentiated.has_value())
            {
                return std::unexpected(differentiated.error());
            }

            value.assign(elements, 0.0);
            internal::BuildEriQuartetValue(quartet, 1, value, scratch);
            const std::array<std::size_t, 4> shape = {nI, nJ, nK, nL};
            const std::array<std::size_t, 4> offsets = {
                braI.functionOffset, braJ.functionOffset, ketK.functionOffset, ketL.functionOffset};
            // The exchange contraction sums the quartet's four cross
            // pairings; the ordered shell sum the canonical walk stands for
            // repeats that sum by a count the divisor must give back: 2 for
            // every diagonal pair (one orientation instead of two), and 2
            // when the same pair supplies both sides (the pair swap re-lists
            // two of the four targets). The contraction then carries exactly
            // 1/2 Tr(D K) for the quartets the canonical walk represents.
            const bool braDiagonal = braPair.i == braPair.j;
            const bool ketDiagonal = ketPair.i == ketPair.j;
            const double kMultiplicity =
                braIndex == ketIndex ? (braDiagonal ? 8.0 : 2.0)
                                     : (braDiagonal ? 2.0 : 1.0) * (ketDiagonal ? 2.0 : 1.0);

            if (keepCoulomb)
            {
                weights.assign(elements, 0.0);

                for (std::size_t fa = 0; fa < nI; ++fa)
                {
                    for (std::size_t fb = 0; fb < nJ; ++fb)
                    {
                        const double densityBra =
                            densityMatrix(static_cast<Eigen::Index>(offsets[0] + fa),
                                          static_cast<Eigen::Index>(offsets[1] + fb));

                        for (std::size_t fc = 0; fc < nK; ++fc)
                        {
                            for (std::size_t fd = 0; fd < nL; ++fd)
                            {
                                const double densityKet =
                                    densityMatrix(static_cast<Eigen::Index>(offsets[2] + fc),
                                                  static_cast<Eigen::Index>(offsets[3] + fd));
                                weights[EriBlockIndex(fa, fb, fc, fd, nI, nJ, nK, nL)] =
                                    permutationWeight * densityBra * densityKet;
                            }
                        }
                    }
                }

                result.coulombEnergy += DotProduct(weights, value);

                for (std::size_t coordinate = 0; coordinate < coordinates.size(); ++coordinate)
                {
                    const std::span<const double> block(derivatives.data() + coordinate * elements,
                                                        elements);
                    result.coulombSkeleton(static_cast<Eigen::Index>(coordinates[coordinate])) +=
                        DotProduct(weights, block);
                }
            }

            if (keepExchange)
            {
                result.exchangeEnergy +=
                    ExchangeEnergyBlock(value, shape, densityMatrix, offsets, kMultiplicity);

                for (std::size_t coordinate = 0; coordinate < coordinates.size(); ++coordinate)
                {
                    const std::span<const double> block(derivatives.data() + coordinate * elements,
                                                        elements);
                    result.exchangeSkeleton(static_cast<Eigen::Index>(coordinates[coordinate])) +=
                        AccumulateExchangeDerivative(block,
                                                     shape,
                                                     densityMatrix,
                                                     offsets,
                                                     kMultiplicity,
                                                     exchangeAO[coordinates[coordinate]]);
                }
            }

            if (keepCoulombBlock)
            {
                for (std::size_t coordinate = 0; coordinate < coordinates.size(); ++coordinate)
                {
                    const std::span<const double> block(derivatives.data() + coordinate * elements,
                                                        elements);
                    AccumulateCoulombDerivativeBlock(block,
                                                     shape,
                                                     densityMatrix,
                                                     offsets,
                                                     static_cast<double>(braMultiplicity),
                                                     static_cast<double>(ketMultiplicity),
                                                     braIndex == ketIndex,
                                                     coulombAO[coordinates[coordinate]]);
                }
            }

            ++result.counts.quartets;
        }
    }

    for (std::size_t coordinate = 0; coordinate < coordinateCount; ++coordinate)
    {
        const Eigen::MatrixXd coulombBlock =
            coefficients.transpose() * coulombAO[coordinate] * coefficients;
        const Eigen::MatrixXd exchangeBlock =
            coefficients.transpose() * exchangeAO[coordinate] * coefficients;

        for (std::size_t virtualOrbital = 0; virtualOrbital < numVirtual; ++virtualOrbital)
        {
            for (std::size_t occupied = 0; occupied < numOccupied; ++occupied)
            {
                const Eigen::Index row =
                    static_cast<Eigen::Index>(coordinate * numVirtual + virtualOrbital);
                const Eigen::Index column = static_cast<Eigen::Index>(occupied);
                const Eigen::Index moRow = static_cast<Eigen::Index>(numOccupied + virtualOrbital);
                result.coulombOccupiedVirtual(row, column) = coulombBlock(moRow, column);
                result.exchangeOccupiedVirtual(row, column) = exchangeBlock(moRow, column);
            }
        }
    }

    return result;
}

} // namespace qcx::integrals
