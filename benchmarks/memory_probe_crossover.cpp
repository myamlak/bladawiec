// The recompute-against-store crossover for basis values and derivatives.
//
// The comparison is per grid point:
//
//   RECOMPUTE  costs one AO-tier evaluation at that point (values, or values +
//              first derivatives, or the full second-derivative tier), and the
//              result lives in a register-scale scratch;
//   STORE      costs one write of the same bytes when the point is first
//              visited plus one read when a consumer revisits it.
//
// Both sides are measured on this machine, against the real evaluator on real
// grid points. The store side is swept over footprints from cache-resident to
// DRAM-resident, because that is the dimension the crossover actually has: the
// per-AO cost of a recompute and the per-AO cost of a store-and-reload are
// both linear in AOCount, so a single crossover in AO count does not exist -
// what moves is what the stored array's FOOTPRINT does to the reload.

#include "memory_probe.hpp"
#include "memory_probe_census.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace memprobe {
namespace {

using Clock = std::chrono::steady_clock;

/// Nanoseconds between two clock readings.
double NanoSeconds(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::nano>(b - a).count();
}

/// The median of a sample set (the median, not the mean: the machine is shared
/// and a single descheduled sample would drag a mean).
double Median(std::vector<double> samples) {
    if (samples.empty())
    {
        return 0.0;
    }

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

/// Water, in Bohr (the fixture the storage and allocation probes use).
qcx::Result<qcx::molecule::Molecule> MakeWater() {
    constexpr double kOH = 1.8110629;
    constexpr double kHalfAngle = 0.9112530;

    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = kOH * std::sin(kHalfAngle);
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = kOH * std::cos(kHalfAngle);
    (*coordinates)(2, 0) = -kOH * std::sin(kHalfAngle);
    (*coordinates)(2, 1) = 0.0;
    (*coordinates)(2, 2) = kOH * std::cos(kHalfAngle);
    coordinates->MarkHostDirty();

    std::vector<qcx::molecule::Atom> atoms;
    atoms.push_back(qcx::molecule::Atom{"O", 8, 0.0});
    atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
    atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

/// Benzene, in Bohr (the fixture the storage and allocation probes use).
qcx::Result<qcx::molecule::Molecule> MakeBenzene() {
    constexpr double kRC = 2.636168;
    constexpr double kRH = 4.686530;

    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({12, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms;

    for (std::size_t k = 0; k < 6; ++k)
    {
        const double angle = static_cast<double>(k) * 3.14159265358979323846 / 3.0;
        (*coordinates)(k, 0) = kRC * std::cos(angle);
        (*coordinates)(k, 1) = kRC * std::sin(angle);
        (*coordinates)(k, 2) = 0.0;
        atoms.push_back(qcx::molecule::Atom{"C", 6, 0.0});
    }

    for (std::size_t k = 0; k < 6; ++k)
    {
        const double angle = static_cast<double>(k) * 3.14159265358979323846 / 3.0;
        (*coordinates)(6 + k, 0) = kRH * std::cos(angle);
        (*coordinates)(6 + k, 1) = kRH * std::sin(angle);
        (*coordinates)(6 + k, 2) = 0.0;
        atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

/// Grid points on a sphere shell around the fixture's centroid, at a spread of
/// radii: points at a physical range of distances from the nuclei, which is
/// what the AO tier's cost depends on.
std::vector<std::array<double, 3>> MakePoints(std::size_t count,
                                              const qcx::molecule::Molecule& molecule) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::array<double, 3> centroid{0.0, 0.0, 0.0};
    const std::size_t atomCount = molecule.Atoms().size();

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            centroid[axis] += coordinates(i, axis) / static_cast<double>(atomCount);
        }
    }

    std::vector<std::array<double, 3>> points;
    points.reserve(count);

    // A deterministic spread: the k-th point sits on a spiral over the sphere
    // and its radius cycles through a physical range.
    for (std::size_t k = 0; k < count; ++k)
    {
        const double t = (static_cast<double>(k) + 0.5) / static_cast<double>(count);
        const double phi = std::acos(1.0 - 2.0 * t);
        const double theta = 2.399963229728653 * static_cast<double>(k);
        const double radius = 0.6 + 4.0 * std::fabs(std::sin(0.7 * static_cast<double>(k)));

        points.push_back({centroid[0] + radius * std::sin(phi) * std::cos(theta),
                          centroid[1] + radius * std::sin(phi) * std::sin(theta),
                          centroid[2] + radius * std::cos(phi)});
    }

    return points;
}

/// The AO-tier rates at one fixture: ns per point for each tier.
struct TierRates {
    double valuesNsPerPoint = 0.0;
    double gradientsNsPerPoint = 0.0;
    double hessiansNsPerPoint = 0.0;
    std::size_t aoCount = 0;
    double checksum = 0.0;
};

/// Times the three AO tiers over \p points, \p repeats times each.
///
/// \p order names the tier sequence, so the same tier can be timed first and
/// last: a per-tier difference that follows the ORDER rather than the tier is
/// a measurement artefact and has to show up as one.
TierRates MeasureTiers(const qcx::grid::AoEvaluator& evaluator,
                       const std::vector<std::array<double, 3>>& points,
                       std::size_t repeats,
                       const std::string& order) {
    TierRates rates;
    rates.aoCount = evaluator.AOCount();

    const std::size_t ao = rates.aoCount;
    std::vector<double> values(ao);
    std::vector<double> gradients(3 * ao);
    std::vector<double> hessians(6 * ao);

    auto run = [&](char tier) {
        std::vector<double> samples;

        for (std::size_t repeat = 0; repeat < repeats; ++repeat)
        {
            const auto start = Clock::now();

            for (const auto& point : points)
            {
                if (tier == 'v')
                {
                    evaluator.Evaluate(point, values);
                } else if (tier == 'g')
                {
                    evaluator.EvaluateGradients(point, values, gradients);
                } else
                {
                    evaluator.EvaluateDerivatives(point, values, gradients, hessians);
                }

                rates.checksum += values[0];
            }

            samples.push_back(NanoSeconds(start, Clock::now()) /
                              static_cast<double>(points.size()));
        }

        return Median(samples);
    };

    for (const char tier : order)
    {
        const double nsPerPoint = run(tier);

        if (tier == 'v')
        {
            rates.valuesNsPerPoint = nsPerPoint;
        } else if (tier == 'g')
        {
            rates.gradientsNsPerPoint = nsPerPoint;
        } else
        {
            rates.hessiansNsPerPoint = nsPerPoint;
        }
    }

    return rates;
}

/// The store-and-reload rate at one footprint: ns per point to write the points'
/// worth of bytes and read them back once.
struct StoreRate {
    double nsPerPoint = 0.0;
    double bytesPerPoint = 0.0;
    double gigabytesPerSecond = 0.0;
    std::size_t points = 0;
};

/// Times one write pass plus one read pass over \p footprintBytes.
StoreRate MeasureStore(std::size_t footprintBytes, std::size_t bytesPerPoint, std::size_t repeats) {
    StoreRate rate;
    rate.bytesPerPoint = static_cast<double>(bytesPerPoint);
    rate.points = footprintBytes / (bytesPerPoint == 0 ? 1 : bytesPerPoint);
    const std::size_t doubles = rate.points * bytesPerPoint / sizeof(double);

    if (rate.points == 0 || doubles == 0)
    {
        return rate;
    }

    std::vector<double> buffer(doubles, 1.0);
    // The evaluator's scratch, standing in for the per-point data that a store
    // would copy FROM; it stays cache-resident, so the write pass moves the
    // large array's bytes and not the source's.
    const std::size_t scratchDoubles = bytesPerPoint / sizeof(double);
    std::vector<double> scratch(scratchDoubles == 0 ? 1 : scratchDoubles, 0.25);
    std::vector<double> samples;
    double sink = 0.0;

    for (std::size_t repeat = 0; repeat < repeats; ++repeat)
    {
        const auto start = Clock::now();

        // Write: every point's block of bytes, in address order. memcpy is the
        // traffic model - a scalar element loop measures the loop's own
        // dependency chain, not the memory system, and reads ~4 GB/s at every
        // footprint including the cache-resident ones.
        double* destination = buffer.data();

        for (std::size_t point = 0; point < rate.points; ++point)
        {
            std::memcpy(destination, scratch.data(), bytesPerPoint);
            destination += scratchDoubles;
        }

        // Read: the consumer's pass over the same bytes.
        double* source = buffer.data();

        for (std::size_t point = 0; point < rate.points; ++point)
        {
            std::memcpy(scratch.data(), source, bytesPerPoint);
            sink += scratch[0];
            source += scratchDoubles;
        }

        samples.push_back(NanoSeconds(start, Clock::now()) / static_cast<double>(rate.points));
    }

    rate.nsPerPoint = Median(samples);
    const double seconds = rate.nsPerPoint * static_cast<double>(rate.points) * 1.0e-9;
    rate.gigabytesPerSecond =
        seconds > 0.0 ? 2.0 * static_cast<double>(doubles * sizeof(double)) / seconds / 1.0e9 : 0.0;
    sink *= 1.0e-30;

    if (sink == 12345.6789)
    {
        std::printf("m5_unreachable_sink = %.3f\n", sink);
    }

    return rate;
}

/// Runs one fixture's measurement and prints it.
void MeasureFixture(const std::string& tag,
                    const qcx::molecule::Molecule& molecule,
                    const qcx::basisset::BasisSet& basis,
                    const std::vector<std::size_t>& footprints,
                    std::size_t points,
                    std::size_t repeats,
                    const std::string& tierOrder) {
    auto evaluator = qcx::grid::AoEvaluator::Create(molecule, basis);

    if (!evaluator.has_value())
    {
        std::printf("m5_%s = ERROR %s\n", tag.c_str(), evaluator.error().message.c_str());
        return;
    }

    const std::vector<std::array<double, 3>> gridPoints = MakePoints(points, molecule);
    const TierRates rates = MeasureTiers(*evaluator, gridPoints, repeats, tierOrder);
    const std::size_t ao = rates.aoCount;

    std::printf(
        "m5_%s order=%s = ao=%zu points=%zu values_ns_per_point=%.4f gradients_ns_per_point=%.4f "
        "hessians_ns_per_point=%.4f values_ns_per_ao=%.6f gradients_ns_per_ao=%.6f "
        "hessians_ns_per_ao=%.6f checksum=%.6e\n",
        tag.c_str(),
        tierOrder.c_str(),
        ao,
        gridPoints.size(),
        rates.valuesNsPerPoint,
        rates.gradientsNsPerPoint,
        rates.hessiansNsPerPoint,
        rates.valuesNsPerPoint / static_cast<double>(ao),
        rates.gradientsNsPerPoint / static_cast<double>(ao),
        rates.hessiansNsPerPoint / static_cast<double>(ao),
        rates.checksum);

    // The store side, at the same bytes-per-point the tiers produce.
    const std::size_t gradientBytesPerPoint = 4 * ao * sizeof(double);
    const std::size_t hessianBytesPerPoint = 10 * ao * sizeof(double);

    for (const std::size_t footprint : footprints)
    {
        const StoreRate gradientStore = MeasureStore(footprint, gradientBytesPerPoint, repeats);
        const StoreRate hessianStore = MeasureStore(footprint, hessianBytesPerPoint, repeats);

        std::printf("m5_%s_store footprint_bytes=%zu points=%zu gradient_bytes_per_point=%zu "
                    "store_reload_ns_per_point=%.4f store_reload_GB_per_s=%.2f "
                    "store_to_recompute_ratio=%.3f\n",
                    tag.c_str(),
                    footprint,
                    gradientStore.points,
                    gradientBytesPerPoint,
                    gradientStore.nsPerPoint,
                    gradientStore.gigabytesPerSecond,
                    rates.gradientsNsPerPoint > 0.0
                        ? gradientStore.nsPerPoint / rates.gradientsNsPerPoint
                        : 0.0);

        std::printf("m5_%s_store_hessian footprint_bytes=%zu hessian_bytes_per_point=%zu "
                    "store_reload_ns_per_point=%.4f store_to_recompute_ratio=%.3f\n",
                    tag.c_str(),
                    footprint,
                    hessianBytesPerPoint,
                    hessianStore.nsPerPoint,
                    rates.hessiansNsPerPoint > 0.0
                        ? hessianStore.nsPerPoint / rates.hessiansNsPerPoint
                        : 0.0);
    }
}

} // namespace

int RunM5(const std::vector<std::string>& args) {
    std::size_t points = 4096;
    std::size_t repeats = 5;
    std::string basisName = "def2-svp";
    std::string tierOrder = "vgh";

    for (std::size_t i = 0; i < args.size(); ++i)
    {
        if (args[i] == "--points" && i + 1 < args.size())
        {
            points = static_cast<std::size_t>(std::stoul(args[i + 1]));
        } else if (args[i] == "--repeats" && i + 1 < args.size())
        {
            repeats = static_cast<std::size_t>(std::stoul(args[i + 1]));
        } else if (args[i] == "--basis" && i + 1 < args.size())
        {
            basisName = args[i + 1];
        } else if (args[i] == "--order" && i + 1 < args.size())
        { tierOrder = args[i + 1]; }
    }

    // Cache-resident to DRAM-resident: the footprint is the crossover's real
    // axis, so it is swept across it. 256 MiB is the largest here - well under
    // the 16 GiB cap, and already four times this machine's last-level cache.
    const std::vector<std::size_t> footprints = {256u * 1024u,
                                                 4u * 1024u * 1024u,
                                                 32u * 1024u * 1024u,
                                                 128u * 1024u * 1024u,
                                                 256u * 1024u * 1024u};

    std::printf("m5_fixture = H2O/def2-svp and C6H6/%s, AO tier of grid/ao_evaluator.cpp on a "
                "spiral point set, %zu points, median of %zu passes\n",
                basisName.c_str(),
                points,
                repeats);

    auto water = MakeWater();

    if (!water.has_value())
    {
        std::printf("m5 = ERROR building water: %s\n", water.error().message.c_str());
        return 1;
    }

    const std::array<int, 2> waterElements{1, 8};
    auto waterBasis =
        qcx::basisset::ParseNwchemDirectoryFiltered(BasisDataDir() + "/def2-svp", waterElements);

    if (!waterBasis.has_value())
    {
        std::printf("m5 = ERROR loading basis: %s\n", waterBasis.error().message.c_str());
        return 1;
    }

    MeasureFixture("water_def2svp", *water, *waterBasis, footprints, points, repeats, tierOrder);

    auto benzene = MakeBenzene();

    if (!benzene.has_value())
    {
        std::printf("m5 = ERROR building benzene: %s\n", benzene.error().message.c_str());
        return 1;
    }

    const std::array<int, 2> benzeneElements{1, 6};

    for (const std::string& family : {basisName, std::string("def2-tzvp")})
    {
        auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(BasisDataDir() + "/" + family,
                                                                 benzeneElements);

        if (!basis.has_value())
        {
            std::printf(
                "m5_benzene_%s = ERROR %s\n", family.c_str(), basis.error().message.c_str());
            continue;
        }

        MeasureFixture(
            "benzene_" + family, *benzene, *basis, footprints, points, repeats, tierOrder);
    }

    std::printf("m5_note = the store side is one write plus one read-back of the same bytes; a "
                "consumer that re-reads the array k times pays the read k times\n");

    return 0;
}

} // namespace memprobe
