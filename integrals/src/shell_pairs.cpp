// The shell/pair flattening (shell_pairs.hpp): shells in molecule-canonical
// atom order with function offsets, the canonical pair list (upper
// triangle, pair-index order), and the shared quartet canonicalization of
// the batch engines (the storage decorator's ordering contract).

#include "qcx/integrals/shell_pairs.hpp"

#include "internal/md_defs.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace qcx::integrals {

qcx::Result<ShellPairList> BuildShellPairs(const qcx::molecule::Molecule& molecule,
                                           const qcx::basisset::BasisSet& basisSet) {
    ShellPairList list;

    for (std::size_t atomIndex = 0; atomIndex < molecule.AtomCount(); ++atomIndex)
    {
        const qcx::molecule::Atom& atom = molecule.Atoms()[atomIndex];
        const qcx::basisset::ElementBasis* element = basisSet.Find(atom.atomicNumber);

        if (element == nullptr)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "basis set has no entry for " + atom.symbol});
        }

        for (std::size_t shellIndex = 0; shellIndex < element->shells.size(); ++shellIndex)
        {
            const qcx::basisset::Shell& shell = element->shells[shellIndex];

            if (shell.angularMomentum > internal::kMaxShellL)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kUnimplemented,
                               "shell angular momentum beyond the parser cap (6)"});
            }

            ShellInfo info{};
            info.angularMomentum = shell.angularMomentum;
            info.isSpherical = shell.isSpherical;
            info.contractionCount = shell.coefficients.size();
            info.functionOffset = list.functionCount;
            info.atomIndex = atomIndex;
            info.elementShellIndex = shellIndex;
            list.functionCount += internal::FunctionCount(
                shell.angularMomentum, shell.isSpherical, shell.coefficients.size());
            list.shells.push_back(info);
        }
    }

    const std::size_t nShells = list.shells.size();

    for (std::size_t i = 0; i < nShells; ++i)
    {
        for (std::size_t j = i; j < nShells; ++j)
        {
            list.pairs.push_back(ShellPairIndex{i, j});
        }
    }

    return list;
}

qcx::Result<std::vector<CanonicalQuartetInfo>> CanonicalizeQuartetOrder(
    const ShellPairList& pairList, const std::vector<ShellQuartet>& quartets) {
    if (quartets.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the quartet list is empty"});
    }

    const std::size_t nShells = pairList.shells.size();
    std::vector<CanonicalQuartetInfo> infos;
    infos.reserve(quartets.size());

    for (const ShellQuartet& quartet : quartets)
    {
        if (quartet.i >= nShells || quartet.j >= nShells || quartet.k >= nShells ||
            quartet.l >= nShells)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "quartet shell index out of range"});
        }

        // The 8-fold canonical form: i <= j, k <= l, pair (i,j) >= (k,l);
        // then the class canonical form L_bra <= L_ket (bra/ket swap is a
        // value-identical 8-fold partner).
        ShellQuartet canonical = quartet;

        if (canonical.i > canonical.j)
        {
            std::swap(canonical.i, canonical.j);
        }

        if (canonical.k > canonical.l)
        {
            std::swap(canonical.k, canonical.l);
        }

        if (PairIndexOf(canonical.i, canonical.j, pairList) <
            PairIndexOf(canonical.k, canonical.l, pairList))
        {
            std::swap(canonical.i, canonical.k);
            std::swap(canonical.j, canonical.l);
        }

        std::size_t braPair = PairIndexOf(canonical.i, canonical.j, pairList);
        std::size_t ketPair = PairIndexOf(canonical.k, canonical.l, pairList);
        int lBra = pairList.shells[canonical.i].angularMomentum +
                   pairList.shells[canonical.j].angularMomentum;
        int lKet = pairList.shells[canonical.k].angularMomentum +
                   pairList.shells[canonical.l].angularMomentum;

        if (lBra > lKet)
        {
            std::swap(canonical.i, canonical.k);
            std::swap(canonical.j, canonical.l);
            std::swap(braPair, ketPair);
            std::swap(lBra, lKet);
        }

        if (lKet > 2 * kMaxEngineL)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "quartet total angular momentum exceeds kMaxEngineL of this build"});
        }

        const ShellInfo& braI = pairList.shells[canonical.i];
        const ShellInfo& braJ = pairList.shells[canonical.j];
        // The bra-pair contraction row pairs (the kernel's GEMM group key);
        // equals MdPairData.rowPairs by construction.
        infos.push_back(CanonicalQuartetInfo{canonical,
                                             lBra,
                                             lKet,
                                             braPair,
                                             ketPair,
                                             braI.contractionCount * braJ.contractionCount});
    }

    // Class-major, then the ket-group order the kernel's pass 1 expects:
    // tasks with the same ket pair and bra row-pair count contiguous (the
    // group shares the GEMM shapes). Distinct canonical quartets never tie
    // on all five keys, so the sort is deterministic.
    std::sort(infos.begin(),
              infos.end(),
              [](const CanonicalQuartetInfo& a, const CanonicalQuartetInfo& b) {
                  if (a.lKet != b.lKet)
                  {
                      return a.lKet < b.lKet;
                  }

                  if (a.lBra != b.lBra)
                  {
                      return a.lBra < b.lBra;
                  }

                  if (a.ketPair != b.ketPair)
                  {
                      return a.ketPair < b.ketPair;
                  }

                  if (a.braRowPairs != b.braRowPairs)
                  {
                      return a.braRowPairs < b.braRowPairs;
                  }

                  return a.braPair < b.braPair;
              });

    return infos;
}

} // namespace qcx::integrals
