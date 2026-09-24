// The Boys-argument size-scaling probe: whether the (order, argument)
// distribution of one Fock build holds its shape as the system grows, and
// what the Schwarz screen removes.
//
// The call-count harness is benchmarks/boys_argument_probe.cpp; this probe
// reuses its stated argument formula (x = p q / (p + q) |P - Q|^2 over one
// (bra primitive pair, ket primitive pair) quadruple), its canonical cell
// rule and its screen (Q_bra x Q_ket against SchwarzThreshold(kNormal) x
// kNeighborListSlack), and its region boundaries - read from the boys
// submodule's own generated table, never restated - but walks the screened
// SETS rather than one build's emission order: region fractions, table-band
// fractions, the fraction above both reference arguments and the argument
// histogram are counts of the same quadruples, and the screened-OUT side is
// walked the same way, so the two populations are directly comparable. That
// comparison is what the size trend needs and what the parent probe, which
// reports the surviving set alone, cannot carry.
//
// Everything here is a re-derivation, not instrumentation: the argument is
// recomputed from the engine's formula and the cell set from the engine's
// screen, never read off a live caller. The engine's own counters are the
// validation: one real LeanDirectFockBuilder build per size (--no-build
// skips it) is compared against the walk's quartet and call counts before
// any fraction is read off.
//
// The screened-out quadruple count is exact (a cell's quadruple count is the
// product of its two primitive-pair counts); the screened-out ARGUMENT
// distribution is a systematic cell sample, at 1 / 2^--sample-shift of the
// screened-out cells, because the full screened-out quadruple set is the
// unscreened n^4 walk. The sampled counts and the sampling fraction are both
// printed; the fraction above x is a ratio estimator over the sample.
//
// Usage: qcx-bench-boys-argument-scaling [--basis <family>] [--carbons <n>]
//                                        [--no-build] [--sample-shift <k>]
#include "alkane_sto3g.hpp"
#include "boys/boys_coefficients.hpp"
#include "internal/fock_screen.hpp"
#include "internal/md_batch.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_tree.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The highest pair class the engine can dispatch to: two shells of
/// kMaxShellL = 6.
constexpr int kMaxPairClass = 12;

/// The highest quartet class.
constexpr int kMaxClass = 24;

/// The region-A table band edge, read from the library's own piece table:
/// every order's fits sit on the same two intervals, so order 0's pieces are
/// the whole of region A's banding.
constexpr double kBandA0 = boys::detail::kPieces[boys::detail::kPieceStart[0]].b;
constexpr double kBandX0 = boys::detail::kX0;
constexpr double kBandX1 = boys::detail::kX1;

/// The argument histogram's edges: half-decade bins from 1e-4 to 1e6, with
/// bin 0 reserved for x == 0 exactly (the parent probe's grid and its index
/// convention: bins 1..21 are the printed half-decade intervals, the last one
/// open-ended).
constexpr int kHistBins = 21;
constexpr double kHistT0 = -4.0;
constexpr double kHistT1 = 6.0;

/// The mantissa table of the bit-level log2 the histogram's binning uses:
/// log2(1 + i / 128) for the mantissa's top seven bits.
std::array<double, 129> gLog2Frac = [] {
    std::array<double, 129> out{};

    for (int i = 0; i <= 128; ++i)
    {
        out[static_cast<std::size_t>(i)] = std::log2(1.0 + static_cast<double>(i) / 128.0);
    }

    return out;
}();

/// The argument-region code of \p x, in the library's own boundaries.
/// \param x The Boys argument, >= 0.
/// \returns 0 for x == 0, else 1 (region A), 2 (region B) or 3 (region C).
inline int RegionOf(double x) noexcept {
    if (x <= 0.0)
    {
        return 0;
    }

    if (x < kBandX0)
    {
        return 1;
    }

    if (x < kBandX1)
    {
        return 2;
    }

    return 3;
}

/// The table-band code of \p x: the finest interval the library's own
/// dispatch separates.
/// \param x The Boys argument, >= 0.
/// \returns 0 zero, 1 A-low, 2 A-high, 3 B, 4 C.
inline int BandOf(double x) noexcept {
    if (x <= 0.0)
    {
        return 0;
    }

    if (x < kBandA0)
    {
        return 1;
    }

    if (x < kBandX0)
    {
        return 2;
    }

    if (x < kBandX1)
    {
        return 3;
    }

    return 4;
}

/// The histogram bin of \p x, computed from the double's own exponent and
/// mantissa instead of a logarithm call: two decades of log10 per bin width,
/// so the table's error (below 1e-5 decades) cannot move a bin but at an
/// exact edge.
/// \param x The Boys argument, >= 0.
/// \returns 0 for x == 0, else 1..kHistBins.
inline int HistBin(double x) noexcept {
    if (x <= 0.0)
    {
        return 0;
    }

    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    const int exponent = static_cast<int>((bits >> 52) & 0x7FFULL) - 1023;
    const int index = static_cast<int>((bits >> 45) & 0x7FULL);
    const double decades =
        (static_cast<double>(exponent) + gLog2Frac[static_cast<std::size_t>(index)]) *
        0.30102999566398120;
    const double scaled =
        std::clamp((decades - kHistT0) * 2.0, 0.0, static_cast<double>(kHistBins - 1));

    return 1 + static_cast<int>(scaled);
}

/// One population's argument census: every field is a count of quadruples
/// (Boys calls) except the two argument extrema.
struct ArgCensus {
    std::size_t cells = 0; ///< Pair-pair cells contributing.
    std::size_t calls = 0; ///< Primitive quadruples (Boys calls).
    std::array<std::size_t, 4> region{}; ///< Per argument region.
    std::array<std::size_t, 5> band{}; ///< Per table band.
    std::vector<std::size_t> hist = std::vector<std::size_t>(kHistBins + 1, 0);
    std::size_t above100 = 0;
    std::size_t above1000 = 0;
    std::size_t zeroCalls = 0;
    double maxArgument = 0.0;
    double minNonZeroArgument = std::numeric_limits<double>::max();

    /// Accumulates one call's argument: every per-call field of this census.
    /// \param x The Boys argument.
    void Accumulate(double x) noexcept {
        ++calls;
        ++region[static_cast<std::size_t>(RegionOf(x))];
        ++band[static_cast<std::size_t>(BandOf(x))];
        ++hist[static_cast<std::size_t>(HistBin(x))];
        maxArgument = x > maxArgument ? x : maxArgument;

        if (x > 0.0)
        {
            minNonZeroArgument = x < minNonZeroArgument ? x : minNonZeroArgument;
        } else
        {
            ++zeroCalls;
        }

        if (x > 100.0)
        {
            ++above100;
        }

        if (x > 1000.0)
        {
            ++above1000;
        }
    }

    /// Accumulates one cell's whole (bra primitive pair, ket primitive pair)
    /// rectangle: the engine's per-quadruple work, verbatim. \p also, when
    /// given, receives the same arguments in the same pass - the near-field
    /// split costs one branch per call, never a second walk.
    /// \param bra The bra pair's primitive products, (p, Px, Py, Pz) rows.
    /// \param nBra The bra pair's primitive-pair count.
    /// \param ket The ket pair's primitive products, (p, Px, Py, Pz) rows.
    /// \param nKet The ket pair's primitive-pair count.
    /// \param also An optional second population for the same calls.
    void AddCell(const double* bra,
                 std::size_t nBra,
                 const double* ket,
                 std::size_t nKet,
                 ArgCensus* also = nullptr) noexcept {
        ++cells;

        if (also != nullptr)
        {
            ++also->cells;
        }

        for (std::size_t g = 0; g < nKet; ++g)
        {
            const double* const k = ket + 4 * g;
            const double q = k[0];
            const double kx = k[1];
            const double ky = k[2];
            const double kz = k[3];

            for (std::size_t b = 0; b < nBra; ++b)
            {
                const double* const p4 = bra + 4 * b;
                const double p = p4[0];
                const double dx = p4[1] - kx;
                const double dy = p4[2] - ky;
                const double dz = p4[3] - kz;
                const double x = (p * q / (p + q)) * (dx * dx + dy * dy + dz * dz);

                Accumulate(x);

                if (also != nullptr)
                {
                    also->Accumulate(x);
                }
            }
        }
    }
};

/// The QFMM octree's near-field leaf-pair set as a bit table: the cells the
/// linear-scaling builder still computes exactly.
struct NearFieldDomain {
    std::size_t nLeaves = 0; ///< Octree leaf count.
    std::size_t leafPairs = 0; ///< Leaf pairs on the near-field list.
    std::size_t farPairs = 0; ///< Node pairs on the far-field (multipole) list.
    std::vector<std::size_t> leafOfPair; ///< Leaf of every canonical pair.
    std::vector<std::uint64_t> bits; ///< nLeaves x nLeaves, canonical index.

    /// Whether the cell (bra pair, ket pair) is near field.
    /// \param a The bra pair's index.
    /// \param b The ket pair's index.
    /// \returns True when both sides' leaves are on the near-field list.
    bool Member(std::size_t a, std::size_t b) const noexcept {
        const std::size_t row = a < b ? b : a;
        const std::size_t column = a < b ? a : b;
        const std::size_t index = row * nLeaves + column;

        return (bits[index >> 6] >> (index & 63U)) & 1U;
    }
};

/// The screen-in and screen-out sides of one walk, with the screened-in side
/// additionally split by the QFMM octree's own near/far classification when
/// that domain is engaged.
struct Census {
    ArgCensus inside; ///< Screened in: exact.
    ArgCensus insideNear; ///< Screened in AND near field (QFMM); exact.
    ArgCensus insideFar; ///< Screened in AND far field (QFMM): the multipole side; exact.
    std::size_t insideNearCells = 0;
    std::size_t insideFarCells = 0;
    ArgCensus outsideSample; ///< Screened out: a systematic cell sample.
    std::size_t insideCells = 0; ///< Exact screened-in cell count (== inside.cells).
    std::size_t outsideCells = 0; ///< Exact screened-out cell count.
    std::size_t outsideCalls = 0; ///< Exact screened-out quadruple count.
    std::size_t outsideSampledCells = 0; ///< Cells whose quadruples were walked.
    std::size_t totalCells = 0; ///< Every canonical cell.
    std::vector<std::size_t> callsByClass; ///< Screened-in Boys calls per class L.
    std::vector<std::size_t> insideCellsByClass;
    std::vector<std::size_t> outsideCellsByClass;
    std::vector<std::size_t> outsideCallsByClass;
};

/// One shell pair's flattened primitive products and its walk role.
struct PairData {
    std::size_t offset = 0; ///< First row in the flat (p, Px, Py, Pz) array.
    std::size_t count = 0; ///< Primitive pairs.
    int l = 0; ///< la + lb.
    std::size_t rowPairs = 0; ///< Contraction-count product.
};

qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
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

qcx::Result<CpuTensor2> MakeDensity(std::size_t n) {
    auto density = CpuTensor2::Create({n, n});

    if (!density.has_value())
    {
        return std::unexpected(density.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        (*density)(i, i) = 0.5;
    }

    density->MarkHostDirty();

    return std::move(*density);
}

void PrintUsage(const char* program) {
    std::printf("usage: %s [--basis <family>] [--carbons <n>] [--no-build] "
                "[--sample-shift <k>] [--near-field]\n",
                program);
}

/// Builds the QFMM near-field domain at the linear-scaling builder's own
/// operating point: its pair store, its default product-ball extents and
/// preset extent cutoff, its leaf cap, and its own interaction lists under
/// the preset's own angle. The pair-pair set that builder's near-field direct
/// pass enumerates is the product of the two leaves of every near-field leaf
/// pair, so the cell test the walk makes below is a decision of the same set.
/// \param molecule The molecule.
/// \param basisSet The basis.
/// \param pairList The canonical shell pairs (the pair store's own order).
/// \param accuracy The preset (its angle and its extent cutoff are read).
/// \returns The domain; an empty bit table means it was too large to hold.
qcx::Result<NearFieldDomain> BuildNearFieldDomain(const qcx::molecule::Molecule& molecule,
                                                  const qcx::basisset::BasisSet& basisSet,
                                                  const qcx::integrals::ShellPairList& pairList,
                                                  qcx::integrals::AccuracyPreset accuracy) {
    auto pairStore = qcx::integrals::internal::BuildPairData(molecule, basisSet, pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(pairStore.error());
    }

    const std::vector<qcx::integrals::internal::QfmmPairGeometry> geometries =
        qcx::integrals::internal::ComputePairGeometries(
            *pairStore,
            qcx::integrals::QfmmExtentForPreset(accuracy),
            qcx::integrals::internal::QfmmExtentModel::kProductBall);
    auto tree = qcx::integrals::internal::BuildQfmmTree(geometries,
                                                        qcx::integrals::internal::kQfmmMaxLeafSize);

    if (!tree.has_value())
    {
        return std::unexpected(tree.error());
    }

    qcx::integrals::internal::QfmmSeparation separation;
    separation.test = qcx::integrals::internal::QfmmSeparationTest::kWidthTheta;
    separation.theta = qcx::integrals::ThetaForPreset(accuracy);
    std::vector<std::pair<std::size_t, std::size_t>> farPairs;
    std::vector<std::pair<std::size_t, std::size_t>> nearPairs;
    qcx::integrals::internal::BuildInteractionLists(*tree, separation, farPairs, nearPairs);

    NearFieldDomain domain;
    domain.nLeaves = tree->nodes.size();
    domain.leafOfPair = tree->leafOfPair;
    domain.leafPairs = nearPairs.size();
    domain.farPairs = farPairs.size();

    // One bit per unordered leaf pair, so the walk's test is a shift and an
    // and. The cap is the table's own size, not the classification's: past it
    // the run reports the domain as unavailable instead of guessing.
    constexpr std::size_t kMaxLeaves = 131072;

    if (domain.nLeaves == 0 || domain.nLeaves > kMaxLeaves)
    {
        return domain;
    }

    domain.bits.assign((domain.nLeaves * domain.nLeaves + 63) / 64, 0);

    for (const std::pair<std::size_t, std::size_t>& leafPair : nearPairs)
    {
        const std::size_t row = leafPair.first < leafPair.second ? leafPair.second : leafPair.first;
        const std::size_t column =
            leafPair.first < leafPair.second ? leafPair.first : leafPair.second;
        const std::size_t index = row * domain.nLeaves + column;
        domain.bits[index >> 6] |= std::uint64_t{1} << (index & 63U);
    }

    return domain;
}

double Percent(std::size_t n, std::size_t total) {
    return total == 0 ? 0.0 : 100.0 * static_cast<double>(n) / static_cast<double>(total);
}

/// Prints one population's argument census.
/// \param label The population's name.
/// \param c The census.
/// \param exact True when every call was walked, false for a cell sample.
void PrintCensus(const char* label, const ArgCensus& c, bool exact) {
    std::printf(
        "%s: cells %zu  Boys calls %zu%s\n", label, c.cells, c.calls, exact ? "" : " (sampled)");
    std::printf(
        "  regions: zero %.4f%%  A(0,%.6g) %.4f%%  B[%.6g,%.6g) %.4f%%  C[%.6g,inf) %.4f%%\n",
        Percent(c.region[0], c.calls),
        kBandX0,
        Percent(c.region[1], c.calls),
        kBandX0,
        kBandX1,
        Percent(c.region[2], c.calls),
        kBandX1,
        Percent(c.region[3], c.calls));
    std::printf("  bands: zero %.4f%%  A-low %.4f%%  A-high %.4f%%  B %.4f%%  C %.4f%%\n",
                Percent(c.band[0], c.calls),
                Percent(c.band[1], c.calls),
                Percent(c.band[2], c.calls),
                Percent(c.band[3], c.calls),
                Percent(c.band[4], c.calls));
    std::printf("  above 100: %.4f%%   above 1000: %.4f%%   max %.6g   min non-zero %.6g   "
                "x == 0 %.4f%%\n",
                Percent(c.above100, c.calls),
                Percent(c.above1000, c.calls),
                c.maxArgument,
                c.minNonZeroArgument,
                Percent(c.zeroCalls, c.calls));
}

/// Prints one population's argument histogram.
/// \param label The population's name.
/// \param c The census.
void PrintHistogram(const char* label, const ArgCensus& c) {
    std::printf("%s histogram (bin 0 is x == 0, then half-decade edges):\n", label);

    for (int i = 0; i <= kHistBins; ++i)
    {
        const std::size_t count = c.hist[static_cast<std::size_t>(i)];

        if (count == 0)
        {
            continue;
        }

        if (i == 0)
        {
            std::printf("    %14s  %14zu  %9.4f%%\n", "x == 0", count, Percent(count, c.calls));

            continue;
        }

        const double lo = kHistT0 + (kHistT1 - kHistT0) * static_cast<double>(i - 1) /
                                        static_cast<double>(kHistBins - 1);
        const double hi = kHistT0 + (kHistT1 - kHistT0) * static_cast<double>(i) /
                                        static_cast<double>(kHistBins - 1);
        std::printf(
            "    [1e%-6.3g,1e%-6.3g)  %14zu  %9.4f%%\n", lo, hi, count, Percent(count, c.calls));
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string family = "def2-svp";
    std::size_t carbons = 24;
    bool runBuild = true;
    bool nearField = false;
    int sampleShift = 10;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--basis" && i + 1 < argc)
        {
            family = argv[++i];
        } else if (arg == "--carbons" && i + 1 < argc)
        {
            carbons = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (arg == "--no-build")
        {
            runBuild = false;
        } else if (arg == "--near-field")
        {
            nearField = true;
        } else if (arg == "--sample-shift" && i + 1 < argc)
        {
            sampleShift = std::atoi(argv[++i]);
        } else
        {
            PrintUsage(argv[0]);

            return 1;
        }
    }

    if (carbons < 1 || sampleShift < 0 || sampleShift > 30)
    {
        PrintUsage(argv[0]);

        return 1;
    }

    const auto accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto molecule = qcx::testing::MakeAlkaneSto3g(carbons);

    if (!molecule.has_value())
    {
        std::printf("molecule: %s\n", molecule.error().message.c_str());

        return 1;
    }

    const std::string basisDirectory = std::string(QcxBasisDataDir) + "/" + family;
    auto basis = qcx::basisset::ParseNwchemDirectory(basisDirectory);

    if (!basis.has_value())
    {
        std::printf("basis %s: %s\n", basisDirectory.c_str(), basis.error().message.c_str());

        return 1;
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        std::printf("pair list: %s\n", pairList.error().message.c_str());

        return 1;
    }

    const std::size_t nPairs = pairList->pairs.size();
    std::printf("fixture: c%zuH%zu / %s, %zu basis functions, %zu shells, %zu canonical pairs\n",
                carbons,
                2 * carbons + 2,
                family.c_str(),
                pairList->functionCount,
                pairList->shells.size(),
                nPairs);
    std::printf("boundaries from the library's own table: kX0 %.17g  kX1 %.17g  region-A edge "
                "%.17g\n",
                boys::detail::kX0,
                boys::detail::kX1,
                kBandA0);

    // The primitive products of every canonical pair, flattened as
    // (p, Px, Py, Pz) rows - the parent probe's own pair data.
    std::vector<PairData> pairData(nPairs);
    std::vector<double> flat;
    std::vector<std::vector<std::size_t>> byL(static_cast<std::size_t>(kMaxPairClass) + 1);

    for (std::size_t k = 0; k < nPairs; ++k)
    {
        const qcx::integrals::ShellPairIndex& pair = pairList->pairs[k];
        const qcx::integrals::ShellInfo& first = pairList->shells[pair.i];
        const qcx::integrals::ShellInfo& second = pairList->shells[pair.j];
        const qcx::basisset::ElementBasis* firstElement =
            basis->Find(molecule->Atoms()[first.atomIndex].atomicNumber);
        const qcx::basisset::ElementBasis* secondElement =
            basis->Find(molecule->Atoms()[second.atomIndex].atomicNumber);

        if (firstElement == nullptr || secondElement == nullptr)
        {
            std::printf("pair %zu: the basis has no entry for one of its shells\n", k);

            return 1;
        }

        const auto& firstExponents = firstElement->shells[first.elementShellIndex].exponents;
        const auto& secondExponents = secondElement->shells[second.elementShellIndex].exponents;
        const auto& coordinates = molecule->CoordinatesBohr();
        const std::array<double, 3> firstCenter{coordinates(first.atomIndex, 0),
                                                coordinates(first.atomIndex, 1),
                                                coordinates(first.atomIndex, 2)};
        const std::array<double, 3> secondCenter{coordinates(second.atomIndex, 0),
                                                 coordinates(second.atomIndex, 1),
                                                 coordinates(second.atomIndex, 2)};

        pairData[k].offset = flat.size() / 4;
        pairData[k].l = first.angularMomentum + second.angularMomentum;
        pairData[k].rowPairs = first.contractionCount * second.contractionCount;

        for (const double a : firstExponents)
        {
            for (const double b : secondExponents)
            {
                const double p = a + b;
                flat.push_back(p);
                flat.push_back((a * firstCenter[0] + b * secondCenter[0]) / p);
                flat.push_back((a * firstCenter[1] + b * secondCenter[1]) / p);
                flat.push_back((a * firstCenter[2] + b * secondCenter[2]) / p);
            }
        }

        pairData[k].count = flat.size() / 4 - pairData[k].offset;

        if (pairData[k].l <= kMaxPairClass)
        {
            byL[static_cast<std::size_t>(pairData[k].l)].push_back(k);
        }
    }

    auto qSchwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);

    if (!qSchwarz.has_value())
    {
        std::printf("schwarz: %s\n", qSchwarz.error().message.c_str());

        return 1;
    }

    const double cutoff =
        qcx::integrals::SchwarzThreshold(accuracy) * qcx::integrals::internal::kNeighborListSlack;
    std::printf("screen: Q_bra x Q_ket against %.17g (SchwarzThreshold x kNeighborListSlack)\n",
                cutoff);

    qcx::integrals::FockBuildStats stats;

    if (runBuild)
    {
        auto core = BuildCoreHamiltonian(*molecule, *basis);

        if (!core.has_value())
        {
            std::printf("core: %s\n", core.error().message.c_str());

            return 1;
        }

        auto density = MakeDensity(pairList->functionCount);

        if (!density.has_value())
        {
            std::printf("density: %s\n", density.error().message.c_str());

            return 1;
        }

        qcx::integrals::LeanFockBuildOptions options;
        options.accuracy = accuracy;
        options.maxParallelChunks = 1;
        auto builder =
            qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, options);

        if (!builder.has_value())
        {
            std::printf("builder: %s\n", builder.error().message.c_str());

            return 1;
        }

        auto fock = builder->BuildFock(*density, &stats);

        if (!fock.has_value())
        {
            std::printf("build: %s\n", fock.error().message.c_str());

            return 1;
        }

        const auto millis = [](std::chrono::nanoseconds span) {
            return std::chrono::duration<double, std::milli>(span).count();
        };
        std::printf("engine: %zu fp64 quartets, %zu VRR calls, %.1f s total wall\n",
                    stats.fp64QuartetCount,
                    stats.kernelVrrQuadruples,
                    millis(stats.totalWallTime) / 1000.0);
    }

    // ---- The walk ----------------------------------------------------------
    Census census;
    census.inside.cells = 0;
    census.callsByClass.assign(kMaxClass + 1, 0);
    census.insideCellsByClass.assign(kMaxClass + 1, 0);
    census.outsideCellsByClass.assign(kMaxClass + 1, 0);
    census.outsideCallsByClass.assign(kMaxClass + 1, 0);

    // The QFMM near-field split, when asked for: the linear-scaling path's own
    // near-field pair set, so the census says which of the calls a build that
    // took that path would still make.
    NearFieldDomain domain;

    if (nearField)
    {
        auto built = BuildNearFieldDomain(*molecule, *basis, *pairList, accuracy);

        if (!built.has_value())
        {
            std::printf("near field: %s\n", built.error().message.c_str());

            return 1;
        }

        domain = std::move(*built);
        std::printf("qfmm near field: %zu leaves, %zu far node pairs, %zu near-field leaf pairs, "
                    "angle %.6g (ThetaForPreset), extent tau %.6g\n",
                    domain.nLeaves,
                    domain.farPairs,
                    domain.leafPairs,
                    qcx::integrals::ThetaForPreset(accuracy),
                    qcx::integrals::QfmmExtentForPreset(accuracy));

        if (domain.bits.empty())
        {
            std::printf("qfmm near field: %zu leaves is past this probe's bit-table cap; the split "
                        "is skipped, not guessed\n",
                        domain.nLeaves);
            domain.leafOfPair.clear();
        }
    }

    const std::size_t sampleMask =
        sampleShift >= 30 ? 0 : ((static_cast<std::size_t>(1) << sampleShift) - 1);
    std::size_t outsideSeen = 0;

    const auto walkStart = std::chrono::steady_clock::now();

    for (int lBra = 0; lBra <= kMaxPairClass; ++lBra)
    {
        for (int lKet = lBra; lKet <= kMaxPairClass; ++lKet)
        {
            const auto& braList = byL[static_cast<std::size_t>(lBra)];
            const auto& ketList = byL[static_cast<std::size_t>(lKet)];

            for (const std::size_t kp : ketList)
            {
                const double* const ketPrim = flat.data() + 4 * pairData[kp].offset;
                const std::size_t nKet = pairData[kp].count;
                const double qk = (*qSchwarz)[kp];
                const int lClass = lBra + lKet;

                for (const std::size_t bp : braList)
                {
                    // The canonical cell is (row, ket) with ket <= row; when
                    // both sides carry the same class each unordered pair is
                    // visited twice and the index order breaks the tie.
                    if (lBra == lKet && bp < kp)
                    {
                        continue;
                    }

                    ++census.totalCells;

                    const double* const braPrim = flat.data() + 4 * pairData[bp].offset;
                    const std::size_t nBra = pairData[bp].count;
                    const std::size_t quadruples = nBra * nKet;

                    if ((*qSchwarz)[bp] * qk >= cutoff)
                    {
                        const bool split = !domain.leafOfPair.empty();
                        const bool near =
                            split && domain.Member(domain.leafOfPair[bp], domain.leafOfPair[kp]);

                        census.inside.AddCell(
                            braPrim,
                            nBra,
                            ketPrim,
                            nKet,
                            split ? (near ? &census.insideNear : &census.insideFar) : nullptr);
                        census.insideCellsByClass[static_cast<std::size_t>(lClass)] += 1;
                        census.callsByClass[static_cast<std::size_t>(lClass)] += quadruples;

                        if (split)
                        {
                            (near ? census.insideNearCells : census.insideFarCells) += 1;
                        }

                        continue;
                    }

                    census.outsideCells += 1;
                    census.outsideCalls += quadruples;
                    census.outsideCellsByClass[static_cast<std::size_t>(lClass)] += 1;
                    census.outsideCallsByClass[static_cast<std::size_t>(lClass)] += quadruples;

                    // The screened-out argument distribution: a systematic
                    // cell sample, the cells and their quadruples both
                    // counted so the ratio carries its own denominator.
                    if (sampleMask != 0 && (outsideSeen++ & sampleMask) == 0)
                    {
                        census.outsideSampledCells += 1;
                        census.outsideSample.AddCell(braPrim, nBra, ketPrim, nKet);
                    }
                }
            }
        }
    }

    const double walkSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - walkStart).count();
    census.insideCells = census.inside.cells;

    std::printf("walked %.1f s: %zu canonical cells (%zu screened in %.4f%%, %zu screened out)\n",
                walkSeconds,
                census.totalCells,
                census.insideCells,
                Percent(census.insideCells, census.totalCells),
                census.outsideCells);
    std::printf(
        "screened-in quartets %zu, Boys calls %zu\n", census.inside.cells, census.inside.calls);
    std::printf(
        "screened-out quartets %zu, quadruples %zu\n", census.outsideCells, census.outsideCalls);

    if (runBuild)
    {
        std::printf("engine cross-check: quartets %s, calls %s\n",
                    census.inside.cells == stats.fp64QuartetCount ? "MATCH" : "MISMATCH",
                    census.inside.calls == stats.kernelVrrQuadruples ? "MATCH" : "MISMATCH");
    }

    // ---- Report ------------------------------------------------------------
    std::printf("\n--- screened in (every call counted) ---\n");
    PrintCensus("screened in", census.inside, true);
    PrintHistogram("screened in", census.inside);

    if (nearField && !domain.leafOfPair.empty())
    {
        std::printf("\n--- screened in, split by the QFMM octree's own near/far lists (the "
                    "linear-scaling path's near field) ---\n");
        std::printf("near %zu cells, %zu calls; far %zu cells, %zu calls; sum %s\n",
                    census.insideNearCells,
                    census.insideNear.calls,
                    census.insideFarCells,
                    census.insideFar.calls,
                    census.insideNearCells + census.insideFarCells == census.insideCells &&
                            census.insideNear.calls + census.insideFar.calls == census.inside.calls
                        ? "MATCH"
                        : "MISMATCH");
        PrintCensus("near field", census.insideNear, true);
        PrintHistogram("near field", census.insideNear);
        PrintCensus("far field", census.insideFar, true);

        // The far-field cells are the ones the multipole path carries: the
        // near-field direct pass never makes their Boys calls.
        std::printf("near-field share of the screened-in calls %.4f%%, of the cells %.4f%%\n",
                    Percent(census.insideNear.calls, census.inside.calls),
                    Percent(census.insideNearCells, census.insideCells));
    }

    std::printf("\n--- screened out (argument distribution: cell sample) ---\n");
    std::printf("sampled %zu of %zu cells (1 / %zu), %zu of %zu quadruples\n",
                census.outsideSampledCells,
                census.outsideCells,
                sampleMask + 1,
                census.outsideSample.calls,
                census.outsideCalls);
    PrintCensus("screened out", census.outsideSample, false);
    PrintHistogram("screened out", census.outsideSample);

    // The mechanism, stated as the two populations' own numbers: with the
    // sample's ratio carried onto the exact screened-out total, the fraction
    // of the UNSCHEDULED argument set that the screen removes.
    const double outsideAboveRate = census.outsideSample.calls == 0
                                        ? 0.0
                                        : static_cast<double>(census.outsideSample.above100) /
                                              static_cast<double>(census.outsideSample.calls);
    const double outsideAboveEst = outsideAboveRate * static_cast<double>(census.outsideCalls);
    const double allAbove = static_cast<double>(census.inside.above100) + outsideAboveEst;
    const double allCalls = static_cast<double>(census.inside.calls + census.outsideCalls);

    std::printf("\nmechanism: the unscreened walk is %zu quadruples; of every 100 of them %.4f "
                "exceed x = 100\n",
                census.inside.calls + census.outsideCalls,
                allCalls == 0.0 ? 0.0 : 100.0 * allAbove / allCalls);
    std::printf("  of the quadruples above x = 100, screening removes %.4f%% and keeps %.4f%%\n",
                allAbove == 0.0 ? 0.0 : 100.0 * outsideAboveEst / allAbove,
                allAbove == 0.0 ? 0.0
                                : 100.0 * static_cast<double>(census.inside.above100) / allAbove);
    std::printf("  enrichment: %.4f%% of the screened-out quadruples exceed 100 against %.4f%% of "
                "the screened-in ones (%.3fx)\n",
                Percent(census.outsideSample.above100, census.outsideSample.calls),
                Percent(census.inside.above100, census.inside.calls),
                census.outsideSample.above100 == 0
                    ? 0.0
                    : (static_cast<double>(census.outsideSample.above100) /
                       static_cast<double>(census.outsideSample.calls)) /
                          (static_cast<double>(census.inside.above100) /
                           static_cast<double>(census.inside.calls)));

    double classSumCalls = 0.0;
    std::size_t atOrBelow3 = 0;

    for (int l = 0; l <= kMaxClass; ++l)
    {
        classSumCalls += static_cast<double>(l * census.callsByClass[static_cast<std::size_t>(l)]);

        if (l <= 3)
        {
            atOrBelow3 += census.callsByClass[static_cast<std::size_t>(l)];
        }
    }

    std::printf("  mean class L over the screened-in calls %.3f; calls at L <= 3 %.4f%%\n",
                census.inside.calls == 0 ? 0.0
                                         : classSumCalls / static_cast<double>(census.inside.calls),
                Percent(atOrBelow3, census.inside.calls));

    // One machine-greppable line per size, so a ladder of runs is one table.
    std::printf("\nSUMMARY,%zu,%zu,%zu,%zu,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6g,%.6f,%.6f,"
                "%.1f\n",
                carbons,
                pairList->functionCount,
                census.inside.cells,
                census.inside.calls,
                census.outsideCells,
                census.outsideCalls,
                Percent(census.inside.region[0], census.inside.calls),
                Percent(census.inside.region[1], census.inside.calls),
                Percent(census.inside.region[2], census.inside.calls),
                Percent(census.inside.region[3], census.inside.calls),
                Percent(census.inside.above100, census.inside.calls),
                Percent(census.inside.above1000, census.inside.calls),
                census.inside.maxArgument,
                Percent(census.outsideSample.above100, census.outsideSample.calls),
                allCalls == 0.0 ? 0.0 : 100.0 * allAbove / allCalls,
                walkSeconds);

    // The split's own one-line record, on the run that asked for it.
    if (nearField && !domain.leafOfPair.empty())
    {
        std::printf("NEAR,%zu,%zu,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6g,%zu,%.6f,%.6g\n",
                    carbons,
                    pairList->functionCount,
                    census.insideNear.calls,
                    census.insideNearCells,
                    Percent(census.insideNear.calls, census.inside.calls),
                    Percent(census.insideNear.above100, census.insideNear.calls),
                    Percent(census.insideNear.region[1], census.insideNear.calls),
                    Percent(census.insideNear.region[2], census.insideNear.calls),
                    Percent(census.insideNear.region[3], census.insideNear.calls),
                    census.insideNear.maxArgument,
                    census.insideFar.calls,
                    Percent(census.insideFar.above100, census.insideFar.calls),
                    census.insideFar.maxArgument);
    }

    return 0;
}
