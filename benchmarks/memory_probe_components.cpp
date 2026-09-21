// The real active component count in production.
//
// MEASURED, not read off a signature: for every shipped functional name the
// harness perturbs each input the production seam accepts, one at a time, and
// asks whether the kernel's OUTPUT moves. A component a functional does not
// consume cannot move the output at any probe point, whatever the call takes.
// The headline number is the largest consumption any shipped functional
// shows; the per-functional table is the evidence.
//
// The seam measured is excgrid v0.1.2's (external/excgrid @ 3c551d8,
// include/excgrid/kernel.hpp): XcFunctional::Evaluate takes the five doubles
// (rhoA, rhoB, sigmaAa, sigmaAb, sigmaBb) and returns a XcKernelValue with
// eight output slots. The component table (components.hpp, seven active
// identifiers) is a wider surface than this seam can express; that
// gap is itself part of the measurement and is reported as such.

#include "excgrid/kernel.hpp"
#include "memory_probe.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace memprobe {
namespace {

/// The inputs the v0.1.2 production seam accepts.
constexpr std::size_t kSeamInputs = 5;

constexpr std::array<const char*, kSeamInputs> kInputNames = {
    "rhoA", "rhoB", "sigmaAa", "sigmaAb", "sigmaBb"};

/// The identifiers the component table marks active.
constexpr std::size_t kSchemaActive = 7;

/// The output slots of a kernel value, in declaration order.
constexpr std::size_t kOutputSlots = 8;

constexpr std::array<const char*, kOutputSlots> kOutputNames = {
    "exc", "vrhoA", "vrhoB", "vsigmaAa", "vsigmaAb", "vsigmaBb", "vtauA", "vtauB"};

/// One probe point: the five seam inputs.
struct Probe {
    const char* label;
    std::array<double, kSeamInputs> inputs;
};

/// The probe points the sweep runs at.
///
/// Three regimes, because a kernel may branch: a spin-polarized valence-like
/// point (the common case on a molecular grid), a spin-unpolarized one (the
/// closed-shell branch many kernels take when rhoA == rhoB), and a
/// low-density tail point, where the gradient enhancement factors are large
/// and the sigma path is at its most sensitive. A component counts as
/// consumed when it moves the output at ANY probe point.
constexpr std::array<Probe, 3> kProbes = {{
    {"polarized", {0.13, 0.11, 0.021, 0.008, 0.017}},
    {"unpolarized", {0.12, 0.12, 0.019, 0.019, 0.019}},
    {"low-density-tail", {1.0e-4, 8.0e-5, 1.2e-8, 4.0e-9, 9.0e-9}},
}};

/// The eight output slots of a kernel value, as an array.
std::array<double, kOutputSlots> Slots(const excgrid::XcKernelValue& v) {
    return {v.exc, v.vrhoA, v.vrhoB, v.vsigmaAa, v.vsigmaAb, v.vsigmaBb, v.vtauA, v.vtauB};
}

/// Whether two kernel values differ in any output slot.
bool Differs(const excgrid::XcKernelValue& a, const excgrid::XcKernelValue& b) {
    return Slots(a) != Slots(b);
}

/// The output mask: bit j set when slot j is nonzero.
std::uint32_t OutputMask(const excgrid::XcKernelValue& v) {
    std::uint32_t mask = 0;
    const std::array<double, kOutputSlots> slots = Slots(v);

    for (std::size_t j = 0; j < kOutputSlots; ++j)
    {
        if (slots[j] != 0.0)
        {
            mask |= 1U << j;
        }
    }

    return mask;
}

std::size_t BitCount(std::uint32_t mask) {
    std::size_t count = 0;

    while (mask != 0)
    {
        count += mask & 1U;
        mask >>= 1U;
    }

    return count;
}

/// Comma-joined names of the bits set in \p mask.
std::string NamesOf(std::uint32_t mask, std::span<const char* const> names) {
    std::string out;

    for (std::size_t i = 0; i < names.size(); ++i)
    {
        if ((mask & (1U << i)) != 0)
        {
            if (!out.empty())
            {
                out += ",";
            }

            out += names[i];
        }
    }

    return out.empty() ? std::string("-") : out;
}

} // namespace

int RunM1(const std::vector<std::string>&) {
    const std::span<const std::string_view> names = excgrid::FunctionalNames();

    std::printf("m1_fixture = excgrid @ 3c551d8 (v0.1.2) registry; %zu shipped names; "
                "%zu seam inputs; %zu schema-active identifiers\n",
                names.size(),
                kSeamInputs,
                kSchemaActive);

    std::size_t maxConsumed = 0;
    std::size_t namesAtMax = 0;
    std::uint32_t unionMask = 0;
    std::uint32_t unionOutputs = 0;
    std::size_t consumedTwo = 0;
    std::size_t consumedFour = 0;
    std::size_t consumedFive = 0;
    std::size_t tauOutputsEverNonzero = 0;

    for (const std::string_view name : names)
    {
        const excgrid::XcFunctional* functional = excgrid::FindFunctional(name);

        if (functional == nullptr)
        {
            std::printf("functional.%s = MISSING\n", std::string(name).c_str());
            continue;
        }

        std::uint32_t inputMask = 0;
        std::uint32_t outputMask = 0;

        for (const Probe& probe : kProbes)
        {
            const excgrid::XcKernelValue base = functional->Evaluate(probe.inputs[0],
                                                                     probe.inputs[1],
                                                                     probe.inputs[2],
                                                                     probe.inputs[3],
                                                                     probe.inputs[4]);
            outputMask |= OutputMask(base);

            for (std::size_t i = 0; i < kSeamInputs; ++i)
            {
                std::array<double, kSeamInputs> perturbed = probe.inputs;
                // A relative step on a nonzero input, an absolute one on a
                // zero input; far above double rounding at these magnitudes
                // and far below any saturation.
                perturbed[i] += perturbed[i] != 0.0 ? perturbed[i] * 1.0e-3 : 1.0e-6;

                const excgrid::XcKernelValue moved = functional->Evaluate(
                    perturbed[0], perturbed[1], perturbed[2], perturbed[3], perturbed[4]);

                if (Differs(base, moved))
                {
                    inputMask |= 1U << i;
                }
            }
        }

        const std::size_t consumed = BitCount(inputMask);
        const std::size_t outputCount = BitCount(outputMask);

        if ((outputMask & (1U << 6)) != 0 || (outputMask & (1U << 7)) != 0)
        {
            ++tauOutputsEverNonzero;
        }

        std::printf("functional.%s = consumed %zu of %zu seam inputs (uses=%s) "
                    "nonzero_outputs=%zu (slots=%s) usesGradient=%d\n",
                    std::string(name).c_str(),
                    consumed,
                    kSeamInputs,
                    NamesOf(inputMask, kInputNames).c_str(),
                    outputCount,
                    NamesOf(outputMask, kOutputNames).c_str(),
                    functional->UsesGradient() ? 1 : 0);

        unionMask |= inputMask;
        unionOutputs |= outputMask;

        if (consumed == 2)
        {
            ++consumedTwo;
        }

        if (consumed == 4)
        {
            ++consumedFour;
        }

        if (consumed == 5)
        {
            ++consumedFive;
        }

        if (consumed > maxConsumed)
        {
            maxConsumed = consumed;
            namesAtMax = 1;
        } else if (consumed == maxConsumed)
        { ++namesAtMax; }
    }

    std::printf("m1_functionals_measured = %zu\n", names.size());
    std::printf("m1_max_consumed_inputs = %zu (by %zu names)\n", maxConsumed, namesAtMax);
    std::printf("m1_union_of_consumed_inputs = %zu of %zu (mask=%s)\n",
                BitCount(unionMask),
                kSeamInputs,
                NamesOf(unionMask, kInputNames).c_str());
    std::printf("m1_union_of_nonzero_outputs = %zu of %zu (mask=%s)\n",
                BitCount(unionOutputs),
                kOutputSlots,
                NamesOf(unionOutputs, kOutputNames).c_str());
    std::printf("m1_names_consuming_2_inputs = %zu\n", consumedTwo);
    std::printf("m1_names_consuming_4_inputs = %zu\n", consumedFour);
    std::printf("m1_names_consuming_5_inputs = %zu\n", consumedFive);
    std::printf("m1_names_with_nonzero_tau_slots = %zu (of %zu shipped)\n",
                tauOutputsEverNonzero,
                names.size());
    std::printf("m1_schema_active_identifiers = %zu (0..6 of kComponentCapacity=32)\n",
                kSchemaActive);

    return 0;
}

} // namespace memprobe
