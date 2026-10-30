#include "qcx/grid/ao_evaluator.hpp"

#include "qcx/grid/internal/solid_harmonics.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace qcx::grid {

namespace {

constexpr double kPi = 3.14159265358979323846;

// The integrals module's per-primitive normalization (basis_set.cpp
// SameShellPrimitiveOverlap: S(a, b) = 2^l (4ab)^(3/4) (ab)^(l/2) /
// (a + b)^(l + 3/2), unit self-overlap for every l).  With the Schlegel
// solid harmonics (int S_lm^2 dOmega = 4 pi / (2l + 1)) the equivalent
// radial factor is
//     N_l(a) = 2^(l + 5/4) a^(l/2 + 3/4) / sqrt(2 pi^(3/2) (2l - 1)!!)
// which reduces to (2a/pi)^(3/4) for l = 0.  The basis parser
// renormalizes spherical contractions against exactly this convention,
// so the evaluator must apply it too - a mismatch shows up as a
// per-l norm error on the grid (e.g. 1.75 instead of 1.0 for p).
// Cartesian shells keep the plain (2a/pi)^(3/4): the engine's Cartesian
// path applies no solid-harmonic factor and the parser leaves Cartesian
// coefficients raw.
double RadialNormalization(int angularMomentum, double exponent, bool solidHarmonic) {
    if (!solidHarmonic)
    {
        return std::pow(2.0 * exponent / kPi, 0.75);
    }

    double doubleFactorial = 1.0; // (-1)!! = (2l-1)!! for l = 0.

    for (int k = 3; k <= 2 * angularMomentum - 1; k += 2)
    {
        doubleFactorial *= static_cast<double>(k);
    }

    return std::pow(2.0, angularMomentum + 1.25) *
           std::pow(exponent, 0.5 * angularMomentum + 0.75) /
           std::sqrt(2.0 * std::pow(kPi, 1.5) * doubleFactorial);
}

// Number of functions in one shell of the given angular momentum.
std::size_t FunctionCount(int angularMomentum, bool isSpherical) {
    if (isSpherical)
    {
        return 2 * static_cast<std::size_t>(angularMomentum) + 1;
    }

    // Cartesian: (l+1)(l+2)/2 monomials.
    const std::size_t l = static_cast<std::size_t>(angularMomentum);
    return (l + 1) * (l + 2) / 2;
}

// The termwise derivatives of one monomial c * x^ix y^iy z^iz: d/dx is
// c * ix * x^(ix-1) y^iy z^iz, zero when the exponent is zero, and the
// second and third derivatives repeat the rule (c * ix * (ix-1) *
// x^(ix-2) ...).  The accumulators cover the symmetric Hessian entries in
// the (xx, xy, xz, yy, yz, zz) order and the fully symmetric third
// derivatives in the (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz)
// order the derivative evaluator writes; the third-order slots stay zero
// unless DifferentiateMonomialToThird fills them.
struct MonomialDerivatives {
    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;
    double hxx = 0.0;
    double hxy = 0.0;
    double hxz = 0.0;
    double hyy = 0.0;
    double hyz = 0.0;
    double hzz = 0.0;
    double xxx = 0.0;
    double xxy = 0.0;
    double xxz = 0.0;
    double xyy = 0.0;
    double xyz = 0.0;
    double xzz = 0.0;
    double yyy = 0.0;
    double yyz = 0.0;
    double yzz = 0.0;
    double zzz = 0.0;

    MonomialDerivatives& operator+=(const MonomialDerivatives& other) {
        gx += other.gx;
        gy += other.gy;
        gz += other.gz;
        hxx += other.hxx;
        hxy += other.hxy;
        hxz += other.hxz;
        hyy += other.hyy;
        hyz += other.hyz;
        hzz += other.hzz;
        return *this;
    }

    // The same addition for one tier: the third-order slots are touched only
    // where that tier computes them, so the gradient and Hessian tiers
    // accumulate exactly what they accumulated before the third tier existed.
    template <bool kThirds> MonomialDerivatives& Accumulate(const MonomialDerivatives& other) {
        *this += other;

        if constexpr (kThirds)
        {
            xxx += other.xxx;
            xxy += other.xxy;
            xxz += other.xxz;
            xyy += other.xyy;
            xyz += other.xyz;
            xzz += other.xzz;
            yyy += other.yyy;
            yyz += other.yyz;
            yzz += other.yzz;
            zzz += other.zzz;
        }

        return *this;
    }
};

MonomialDerivatives DifferentiateMonomial(
    double coefficient,
    std::size_t ix,
    std::size_t iy,
    std::size_t iz,
    const std::array<double, internal::kMaxSolidHarmonicL + 1>& powX,
    const std::array<double, internal::kMaxSolidHarmonicL + 1>& powY,
    const std::array<double, internal::kMaxSolidHarmonicL + 1>& powZ) {
    MonomialDerivatives result;

    if (ix > 0)
    {
        result.gx = static_cast<double>(ix) * coefficient * powX[ix - 1] * powY[iy] * powZ[iz];

        if (ix > 1)
        {
            result.hxx = static_cast<double>(ix * (ix - 1)) * coefficient * powX[ix - 2] *
                         powY[iy] * powZ[iz];
        }

        if (iy > 0)
        {
            result.hxy =
                static_cast<double>(ix * iy) * coefficient * powX[ix - 1] * powY[iy - 1] * powZ[iz];
        }

        if (iz > 0)
        {
            result.hxz =
                static_cast<double>(ix * iz) * coefficient * powX[ix - 1] * powY[iy] * powZ[iz - 1];
        }
    }

    if (iy > 0)
    {
        result.gy = static_cast<double>(iy) * coefficient * powX[ix] * powY[iy - 1] * powZ[iz];

        if (iy > 1)
        {
            result.hyy = static_cast<double>(iy * (iy - 1)) * coefficient * powX[ix] *
                         powY[iy - 2] * powZ[iz];
        }

        if (iz > 0)
        {
            result.hyz =
                static_cast<double>(iy * iz) * coefficient * powX[ix] * powY[iy - 1] * powZ[iz - 1];
        }
    }

    if (iz > 0)
    {
        result.gz = static_cast<double>(iz) * coefficient * powX[ix] * powY[iy] * powZ[iz - 1];

        if (iz > 1)
        {
            result.hzz = static_cast<double>(iz * (iz - 1)) * coefficient * powX[ix] * powY[iy] *
                         powZ[iz - 2];
        }
    }

    return result;
}

// The third-order sibling of DifferentiateMonomial: the same termwise rule
// one order further, into the ten distinct third derivatives.  It duplicates
// no lower-order term - DifferentiateMonomial supplies those, so the gradient
// and Hessian entries are the ones the cheaper tiers compute.
MonomialDerivatives DifferentiateMonomialToThird(
    double coefficient,
    std::size_t ix,
    std::size_t iy,
    std::size_t iz,
    const std::array<double, internal::kMaxSolidHarmonicL + 1>& powX,
    const std::array<double, internal::kMaxSolidHarmonicL + 1>& powY,
    const std::array<double, internal::kMaxSolidHarmonicL + 1>& powZ) {
    MonomialDerivatives result = DifferentiateMonomial(coefficient, ix, iy, iz, powX, powY, powZ);

    if (ix > 0)
    {
        if (ix > 1)
        {
            if (ix > 2)
            {
                result.xxx = static_cast<double>(ix * (ix - 1) * (ix - 2)) * coefficient *
                             powX[ix - 3] * powY[iy] * powZ[iz];
            }

            if (iy > 0)
            {
                result.xxy = static_cast<double>(ix * (ix - 1) * iy) * coefficient * powX[ix - 2] *
                             powY[iy - 1] * powZ[iz];
            }

            if (iz > 0)
            {
                result.xxz = static_cast<double>(ix * (ix - 1) * iz) * coefficient * powX[ix - 2] *
                             powY[iy] * powZ[iz - 1];
            }
        }

        if (iy > 0)
        {
            if (iy > 1)
            {
                result.xyy = static_cast<double>(ix * iy * (iy - 1)) * coefficient * powX[ix - 1] *
                             powY[iy - 2] * powZ[iz];
            }

            if (iz > 0)
            {
                result.xyz = static_cast<double>(ix * iy * iz) * coefficient * powX[ix - 1] *
                             powY[iy - 1] * powZ[iz - 1];
            }
        }

        if (iz > 0)
        {
            if (iz > 1)
            {
                result.xzz = static_cast<double>(ix * iz * (iz - 1)) * coefficient * powX[ix - 1] *
                             powY[iy] * powZ[iz - 2];
            }
        }
    }

    if (iy > 0)
    {
        if (iy > 1)
        {
            if (iy > 2)
            {
                result.yyy = static_cast<double>(iy * (iy - 1) * (iy - 2)) * coefficient *
                             powX[ix] * powY[iy - 3] * powZ[iz];
            }

            if (iz > 0)
            {
                result.yyz = static_cast<double>(iy * (iy - 1) * iz) * coefficient * powX[ix] *
                             powY[iy - 2] * powZ[iz - 1];
            }
        }

        if (iz > 0)
        {
            if (iz > 1)
            {
                result.yzz = static_cast<double>(iy * iz * (iz - 1)) * coefficient * powX[ix] *
                             powY[iy - 1] * powZ[iz - 2];
            }
        }
    }

    if (iz > 0)
    {
        if (iz > 1)
        {
            if (iz > 2)
            {
                result.zzz = static_cast<double>(iz * (iz - 1) * (iz - 2)) * coefficient *
                             powX[ix] * powY[iy] * powZ[iz - 3];
            }
        }
    }

    return result;
}

} // namespace

Result<AoEvaluator> AoEvaluator::Create(const qcx::molecule::Molecule& molecule,
                                        const qcx::basisset::BasisSet& basis) {
    std::vector<ShellData> shells;
    std::size_t aoCount = 0;

    const auto& atoms = molecule.Atoms();
    const auto& coordinates = molecule.CoordinatesBohr();

    for (std::size_t atomIndex = 0; atomIndex < atoms.size(); ++atomIndex)
    {
        const qcx::basisset::ElementBasis* element = basis.Find(atoms[atomIndex].atomicNumber);

        if (element == nullptr)
        {
            return std::unexpected(Error{ErrorCode::kInvalidArgument,
                                         "AoEvaluator: no basis entry for element Z=" +
                                             std::to_string(atoms[atomIndex].atomicNumber)});
        }

        for (const qcx::basisset::Shell& shell : element->shells)
        {
            if (shell.angularMomentum > internal::kMaxSolidHarmonicL)
            {
                return std::unexpected(
                    Error{ErrorCode::kUnimplemented,
                          "AoEvaluator: angular momentum " + std::to_string(shell.angularMomentum) +
                              " is not supported (max " +
                              std::to_string(internal::kMaxSolidHarmonicL) + ")"});
            }

            ShellData data;
            data.center = {
                coordinates(atomIndex, 0), coordinates(atomIndex, 1), coordinates(atomIndex, 2)};
            data.angularMomentum = shell.angularMomentum;
            data.isSpherical = shell.isSpherical;
            data.functionCount = FunctionCount(shell.angularMomentum, shell.isSpherical);
            data.aoOffset = aoCount;

            // One contraction per coefficient row; a general contraction
            // (several rows sharing the exponent array) becomes several
            // function sets.
            for (const auto& coefficients : shell.coefficients)
            {
                Contraction contraction;
                contraction.primitives.reserve(shell.exponents.size());

                for (std::size_t i = 0; i < shell.exponents.size(); ++i)
                {
                    // The radial normalization is invariant per
                    // (l, exponent, spherical), so it is hoisted out of
                    // the per-grid-point evaluation.
                    contraction.primitives.push_back({shell.exponents[i],
                                                      coefficients[i],
                                                      RadialNormalization(shell.angularMomentum,
                                                                          shell.exponents[i],
                                                                          shell.isSpherical)});
                }

                data.contractions.push_back(std::move(contraction));
            }

            aoCount += data.contractions.size() * data.functionCount;
            shells.push_back(std::move(data));
        }
    }

    // The shell ranges and the all-shells selection are built here rather than
    // in the constructor, which then only moves.
    std::vector<ShellRange> shellRanges;
    shellRanges.reserve(shells.size());

    for (const ShellData& shell : shells)
    {
        // The block spans every contraction row: the per-point loops write
        // rows at aoOffset + c * functionCount, so a range reporting a single
        // row would lose the rest of the shell's slots.
        shellRanges.push_back(
            ShellRange{shell.aoOffset, shell.contractions.size() * shell.functionCount});
    }

    std::vector<std::size_t> allShells(shells.size());

    for (std::size_t index = 0; index < allShells.size(); ++index)
    {
        allShells[index] = index;
    }

    return AoEvaluator(std::move(shells), std::move(shellRanges), std::move(allShells), aoCount);
}

AoEvaluator::AoEvaluator(std::vector<ShellData> shells,
                         std::vector<ShellRange> shellRanges,
                         std::vector<std::size_t> allShells,
                         std::size_t aoCount) noexcept :
    _shells(std::move(shells)), _shellRanges(std::move(shellRanges)),
    _allShells(std::move(allShells)), _aoCount(aoCount) {}

void AoEvaluator::Evaluate(const std::array<double, 3>& pointBohr, std::span<double> out) const {
    EvaluateSelected(pointBohr, _allShells, out);
}

void AoEvaluator::EvaluateSelected(const std::array<double, 3>& pointBohr,
                                   std::span<const std::size_t> selection,
                                   std::span<double> out) const {
    // Scratch for one shell's angular factors (max 28: l=6 Cartesian monomials).
    std::array<double, 28> angular{};

    for (const std::size_t shellIndex : selection)
    {
        const ShellData& shell = _shells[shellIndex];
        const std::array<double, 3> delta = {pointBohr[0] - shell.center[0],
                                             pointBohr[1] - shell.center[1],
                                             pointBohr[2] - shell.center[2]};
        const double r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

        FillAngular(shell, delta, angular);

        // The angular factor is shared by every contraction of the shell;
        // each contraction contributes functionCount AOs with its own
        // contracted radial part.
        std::span<double> block = out.subspan(shell.aoOffset);

        for (std::size_t c = 0; c < shell.contractions.size(); ++c)
        {
            double radial = 0.0;

            for (const Primitive& primitive : shell.contractions[c].primitives)
            {
                radial += primitive.coefficient * primitive.normalization *
                          std::exp(-primitive.exponent * r2);
            }

            for (std::size_t i = 0; i < shell.functionCount; ++i)
            {
                block[c * shell.functionCount + i] = angular[i] * radial;
            }
        }
    }
}

void AoEvaluator::FillAngular(const ShellData& shell,
                              const std::array<double, 3>& delta,
                              std::span<double> block) {
    const double x = delta[0];
    const double y = delta[1];
    const double z = delta[2];

    if (!shell.isSpherical)
    {
        // Cartesian monomials x^a y^b z^c, a descending then b descending.
        const int l = shell.angularMomentum;
        std::size_t index = 0;

        for (int a = l; a >= 0; --a)
        {
            for (int b = l - a; b >= 0; --b)
            {
                const int c = l - a - b;
                block[index++] = std::pow(x, a) * std::pow(y, b) * std::pow(z, c);
            }
        }

        return;
    }

    // Real solid harmonics, Schlegel convention, m order -l ... +l with
    // m < 0 the sine component and m > 0 the cosine component.  The basis
    // module renormalizes spherical contractions to unit norm under
    // exactly this convention (the (2a/pi)^(3/4) radial factor times
    // these solid harmonics), so the stored coefficients are exact.  The
    // rows are generated (tools/grid/gen_solid_harmonics.py) from the
    // integrals module's Racah rows scaled by sqrt(4 pi / (2l + 1)) and
    // certified against int S_lm^2 dOmega = 4 pi / (2l + 1) at 30 digits;
    // the grid module cannot include the integrals tables (module DAG).
    const int l = shell.angularMomentum;

    // Monomial powers up to the shell's angular momentum.
    std::array<double, internal::kMaxSolidHarmonicL + 1> powX{};
    std::array<double, internal::kMaxSolidHarmonicL + 1> powY{};
    std::array<double, internal::kMaxSolidHarmonicL + 1> powZ{};
    powX[0] = 1.0;
    powY[0] = 1.0;
    powZ[0] = 1.0;

    for (int n = 1; n <= l; ++n)
    {
        powX[n] = powX[n - 1] * x;
        powY[n] = powY[n - 1] * y;
        powZ[n] = powZ[n - 1] * z;
    }

    const std::size_t count = 2 * static_cast<std::size_t>(l) + 1;

    for (std::size_t m = 0; m < count; ++m)
    {
        // Slot within the l-block is m_real + l; with the loop index m
        // (m_real = m - l) that reduces to m.
        const std::size_t rowIndex =
            static_cast<std::size_t>(l) * internal::kSolidHarmonicRowSlotsPerL + m;
        const internal::SolidHarmonicRow& row = internal::kSolidHarmonicRows[rowIndex];
        double value = 0.0;

        for (std::size_t i = 0; i < row.count; ++i)
        {
            const internal::SolidHarmonicTerm& term = internal::kSolidHarmonicTerms[row.offset + i];
            value += term.coefficient * powX[term.ix] * powY[term.iy] * powZ[term.iz];
        }

        block[m] = value;
    }
}

template <auto kTier>
void AoEvaluator::FillAngularDerivativesTier(
    const ShellData& shell,
    const std::array<double, 3>& delta,
    // (gradients, hessians, thirds) are the per-function derivative output
    // spans - 3, 6 and 10 slots per function.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    std::span<double> gradients,
    std::span<double> hessians,
    std::span<double> thirds) {
    constexpr bool kHessians = kTier != DerivativeTier::kGradients;
    constexpr bool kThirds = kTier == DerivativeTier::kThirds;
    const double x = delta[0];
    const double y = delta[1];
    const double z = delta[2];
    const int l = shell.angularMomentum;

    // Monomial powers up to the shell's angular momentum; the derivatives
    // lower the degree, so the same arrays cover every term.
    std::array<double, internal::kMaxSolidHarmonicL + 1> powX{};
    std::array<double, internal::kMaxSolidHarmonicL + 1> powY{};
    std::array<double, internal::kMaxSolidHarmonicL + 1> powZ{};
    powX[0] = 1.0;
    powY[0] = 1.0;
    powZ[0] = 1.0;

    for (int n = 1; n <= l; ++n)
    {
        powX[n] = powX[n - 1] * x;
        powY[n] = powY[n - 1] * y;
        powZ[n] = powZ[n - 1] * z;
    }

    const auto write = [&](std::size_t function, const MonomialDerivatives& derivatives) {
        gradients[3 * function + 0] = derivatives.gx;
        gradients[3 * function + 1] = derivatives.gy;
        gradients[3 * function + 2] = derivatives.gz;

        if constexpr (kHessians)
        {
            hessians[6 * function + 0] = derivatives.hxx;
            hessians[6 * function + 1] = derivatives.hxy;
            hessians[6 * function + 2] = derivatives.hxz;
            hessians[6 * function + 3] = derivatives.hyy;
            hessians[6 * function + 4] = derivatives.hyz;
            hessians[6 * function + 5] = derivatives.hzz;
        }

        if constexpr (kThirds)
        {
            thirds[10 * function + 0] = derivatives.xxx;
            thirds[10 * function + 1] = derivatives.xxy;
            thirds[10 * function + 2] = derivatives.xxz;
            thirds[10 * function + 3] = derivatives.xyy;
            thirds[10 * function + 4] = derivatives.xyz;
            thirds[10 * function + 5] = derivatives.xzz;
            thirds[10 * function + 6] = derivatives.yyy;
            thirds[10 * function + 7] = derivatives.yyz;
            thirds[10 * function + 8] = derivatives.yzz;
            thirds[10 * function + 9] = derivatives.zzz;
        }
    };

    // The third tier differentiates each monomial one order further; the
    // cheaper tiers call the function they called before.
    const auto differentiate =
        [&](double coefficient, std::size_t ix, std::size_t iy, std::size_t iz) {
            if constexpr (kThirds)
            {
                return DifferentiateMonomialToThird(coefficient, ix, iy, iz, powX, powY, powZ);
            } else
            {
                return DifferentiateMonomial(coefficient, ix, iy, iz, powX, powY, powZ);
            }
        };

    if (!shell.isSpherical)
    {
        // Cartesian monomials x^a y^b z^c, a descending then b descending
        // (the FillAngular order); the coefficient is 1.
        std::size_t function = 0;

        for (int a = l; a >= 0; --a)
        {
            for (int b = l - a; b >= 0; --b)
            {
                const int c = l - a - b;
                write(function,
                      differentiate(1.0,
                                    static_cast<std::size_t>(a),
                                    static_cast<std::size_t>(b),
                                    static_cast<std::size_t>(c)));
                ++function;
            }
        }

        return;
    }

    // Real solid-harmonic rows, same m order as FillAngular: each row's
    // monomial terms differentiated termwise.
    const std::size_t count = 2 * static_cast<std::size_t>(l) + 1;

    for (std::size_t m = 0; m < count; ++m)
    {
        const std::size_t rowIndex =
            static_cast<std::size_t>(l) * internal::kSolidHarmonicRowSlotsPerL + m;
        const internal::SolidHarmonicRow& row = internal::kSolidHarmonicRows[rowIndex];
        MonomialDerivatives derivatives;

        for (std::size_t i = 0; i < row.count; ++i)
        {
            const internal::SolidHarmonicTerm& term = internal::kSolidHarmonicTerms[row.offset + i];
            derivatives.Accumulate<kThirds>(
                differentiate(term.coefficient, term.ix, term.iy, term.iz));
        }

        write(m, derivatives);
    }
}

template <auto kTier>
void AoEvaluator::EvaluateDerivativesTier(const std::array<double, 3>& pointBohr,
                                          std::span<const std::size_t> selection,
                                          std::span<double> values,
                                          std::span<double> gradients,
                                          std::span<double> hessians,
                                          std::span<double> thirds) const {
    constexpr bool kHessians = kTier != DerivativeTier::kGradients;
    constexpr bool kThirds = kTier == DerivativeTier::kThirds;

    // Scratch for one shell's angular factors and their derivatives
    // (max 28 functions per shell, 3 + 6 + 10 derivative slots per function).
    // The third block is one element wide in the cheaper tiers, which neither
    // fill nor read it, so they do not zero a buffer they never use.
    constexpr std::size_t kMaxGradientSlots = std::size_t{3} * 28;
    constexpr std::size_t kMaxHessianSlots = std::size_t{6} * 28;
    constexpr std::size_t kThirdScratch = kThirds ? std::size_t{10} * 28 : 1;
    std::array<double, 28> angular{};
    std::array<double, kMaxGradientSlots> angularGradients{};
    std::array<double, kMaxHessianSlots> angularHessians{};
    std::array<double, kThirdScratch> angularThirds{};

    for (const std::size_t shellIndex : selection)
    {
        const ShellData& shell = _shells[shellIndex];
        const std::array<double, 3> delta = {pointBohr[0] - shell.center[0],
                                             pointBohr[1] - shell.center[1],
                                             pointBohr[2] - shell.center[2]};
        const double r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

        FillAngular(shell, delta, angular);
        FillAngularDerivativesTier<kTier>(
            shell, delta, angularGradients, angularHessians, angularThirds);

        std::span<double> valueBlock = values.subspan(shell.aoOffset);
        std::span<double> gradientBlock = gradients.subspan(shell.aoOffset * 3);
        // Left empty by the tiers above the one asked for, which pass no
        // such span at all: a subspan of an empty span would be out of range,
        // and the writes below are compiled out for those tiers anyway.
        [[maybe_unused]] std::span<double> hessianBlock;
        [[maybe_unused]] std::span<double> thirdBlock;

        if constexpr (kHessians)
        {
            hessianBlock = hessians.subspan(shell.aoOffset * 6);
        }

        if constexpr (kThirds)
        {
            thirdBlock = thirds.subspan(shell.aoOffset * 10);
        }

        // The angular factors are shared by every contraction of the
        // shell; each contraction contributes functionCount AOs with its
        // own contracted radial part.  With R the contracted radial
        // factor, its derivatives are analytic per primitive
        // (e^{-zeta r2} is computed once, then the 1 + 3 + 6 + 10 factors):
        //     grad R = -2 sum_i d_i N_l zeta_i e^{-zeta_i r2} Delta
        //     (H_R)_jk = sum_i d_i N_l e^{-zeta_i r2}
        //                (4 zeta_i^2 Delta_j Delta_k - 2 zeta_i delta_jk)
        // so, with phi = S R,
        //     grad phi = R grad S + S grad R
        //     H_phi = R H_S + gradS gradR^T + gradR gradS^T + S H_R
        // and the third derivative by one more differentiation of that line.
        for (std::size_t c = 0; c < shell.contractions.size(); ++c)
        {
            double radial = 0.0;
            std::array<double, 3> gradR{};
            [[maybe_unused]] std::array<double, 6> hessR{};
            [[maybe_unused]] std::array<double, 10> thirdR{};

            for (const Primitive& primitive : shell.contractions[c].primitives)
            {
                const double factor = primitive.coefficient * primitive.normalization *
                                      std::exp(-primitive.exponent * r2);
                const double twoZ = 2.0 * primitive.exponent;
                const double fourZ2 = 4.0 * primitive.exponent * primitive.exponent;
                radial += factor;
                gradR[0] += -twoZ * delta[0] * factor;
                gradR[1] += -twoZ * delta[1] * factor;
                gradR[2] += -twoZ * delta[2] * factor;

                if constexpr (kHessians)
                {
                    hessR[0] += factor * (fourZ2 * delta[0] * delta[0] - twoZ);
                    hessR[1] += factor * fourZ2 * delta[0] * delta[1];
                    hessR[2] += factor * fourZ2 * delta[0] * delta[2];
                    hessR[3] += factor * (fourZ2 * delta[1] * delta[1] - twoZ);
                    hessR[4] += factor * fourZ2 * delta[1] * delta[2];
                    hessR[5] += factor * (fourZ2 * delta[2] * delta[2] - twoZ);
                }

                // One differentiation of the Hessian line, for
                // f = N_l(zeta) exp(-zeta r2) and Delta = x - R_A:
                //     d3_abc f = f [4 zeta^2 (delta_ab Delta_c
                //                              + delta_ac Delta_b
                //                              + delta_bc Delta_a)
                //                   - 8 zeta^3 Delta_a Delta_b Delta_c],
                // written out in the packed third-derivative order.
                if constexpr (kThirds)
                {
                    const double x = delta[0];
                    const double y = delta[1];
                    const double z = delta[2];
                    const double eightZ3 = twoZ * fourZ2;
                    thirdR[0] += factor * (3.0 * fourZ2 * x - eightZ3 * x * x * x);
                    thirdR[1] += factor * (fourZ2 * y - eightZ3 * x * x * y);
                    thirdR[2] += factor * (fourZ2 * z - eightZ3 * x * x * z);
                    thirdR[3] += factor * (fourZ2 * x - eightZ3 * x * y * y);
                    thirdR[4] += factor * (-eightZ3 * x * y * z);
                    thirdR[5] += factor * (fourZ2 * x - eightZ3 * x * z * z);
                    thirdR[6] += factor * (3.0 * fourZ2 * y - eightZ3 * y * y * y);
                    thirdR[7] += factor * (fourZ2 * z - eightZ3 * y * y * z);
                    thirdR[8] += factor * (fourZ2 * y - eightZ3 * y * z * z);
                    thirdR[9] += factor * (3.0 * fourZ2 * z - eightZ3 * z * z * z);
                }
            }

            const std::size_t offset = c * shell.functionCount;

            for (std::size_t i = 0; i < shell.functionCount; ++i)
            {
                const double s = angular[i];
                const double gs0 = angularGradients[3 * i + 0];
                const double gs1 = angularGradients[3 * i + 1];
                const double gs2 = angularGradients[3 * i + 2];

                valueBlock[offset + i] = s * radial;
                const std::size_t g = (offset + i) * 3;
                gradientBlock[g + 0] = radial * gs0 + s * gradR[0];
                gradientBlock[g + 1] = radial * gs1 + s * gradR[1];
                gradientBlock[g + 2] = radial * gs2 + s * gradR[2];

                if constexpr (kHessians)
                {
                    const double hs0 = angularHessians[6 * i + 0];
                    const double hs1 = angularHessians[6 * i + 1];
                    const double hs2 = angularHessians[6 * i + 2];
                    const double hs3 = angularHessians[6 * i + 3];
                    const double hs4 = angularHessians[6 * i + 4];
                    const double hs5 = angularHessians[6 * i + 5];
                    const std::size_t h = (offset + i) * 6;
                    hessianBlock[h + 0] = radial * hs0 + 2.0 * gs0 * gradR[0] + s * hessR[0];
                    hessianBlock[h + 1] =
                        radial * hs1 + gs0 * gradR[1] + gs1 * gradR[0] + s * hessR[1];
                    hessianBlock[h + 2] =
                        radial * hs2 + gs0 * gradR[2] + gs2 * gradR[0] + s * hessR[2];
                    hessianBlock[h + 3] = radial * hs3 + 2.0 * gs1 * gradR[1] + s * hessR[3];
                    hessianBlock[h + 4] =
                        radial * hs4 + gs1 * gradR[2] + gs2 * gradR[1] + s * hessR[4];
                    hessianBlock[h + 5] = radial * hs5 + 2.0 * gs2 * gradR[2] + s * hessR[5];

                    // The product rule one order further: R and S each carry a
                    // third derivative, and every pairing of their lower
                    // orders contributes one cross term.
                    if constexpr (kThirds)
                    {
                        const double ts0 = angularThirds[10 * i + 0];
                        const double ts1 = angularThirds[10 * i + 1];
                        const double ts2 = angularThirds[10 * i + 2];
                        const double ts3 = angularThirds[10 * i + 3];
                        const double ts4 = angularThirds[10 * i + 4];
                        const double ts5 = angularThirds[10 * i + 5];
                        const double ts6 = angularThirds[10 * i + 6];
                        const double ts7 = angularThirds[10 * i + 7];
                        const double ts8 = angularThirds[10 * i + 8];
                        const double ts9 = angularThirds[10 * i + 9];
                        const std::size_t t = (offset + i) * 10;
                        thirdBlock[t + 0] = radial * ts0 + 3.0 * gradR[0] * hs0 +
                                            3.0 * gs0 * hessR[0] + s * thirdR[0];
                        thirdBlock[t + 1] = radial * ts1 + gradR[1] * hs0 + 2.0 * gradR[0] * hs1 +
                                            2.0 * gs0 * hessR[1] + gs1 * hessR[0] + s * thirdR[1];
                        thirdBlock[t + 2] = radial * ts2 + gradR[2] * hs0 + 2.0 * gradR[0] * hs2 +
                                            2.0 * gs0 * hessR[2] + gs2 * hessR[0] + s * thirdR[2];
                        thirdBlock[t + 3] = radial * ts3 + 2.0 * gradR[1] * hs1 + gradR[0] * hs3 +
                                            gs0 * hessR[3] + 2.0 * gs1 * hessR[1] + s * thirdR[3];
                        thirdBlock[t + 4] = radial * ts4 + gradR[2] * hs1 + gradR[1] * hs2 +
                                            gradR[0] * hs4 + gs0 * hessR[4] + gs1 * hessR[2] +
                                            gs2 * hessR[1] + s * thirdR[4];
                        thirdBlock[t + 5] = radial * ts5 + 2.0 * gradR[2] * hs2 + gradR[0] * hs5 +
                                            gs0 * hessR[5] + 2.0 * gs2 * hessR[2] + s * thirdR[5];
                        thirdBlock[t + 6] = radial * ts6 + 3.0 * gradR[1] * hs3 +
                                            3.0 * gs1 * hessR[3] + s * thirdR[6];
                        thirdBlock[t + 7] = radial * ts7 + gradR[2] * hs3 + 2.0 * gradR[1] * hs4 +
                                            2.0 * gs1 * hessR[4] + gs2 * hessR[3] + s * thirdR[7];
                        thirdBlock[t + 8] = radial * ts8 + 2.0 * gradR[2] * hs4 + gradR[1] * hs5 +
                                            gs1 * hessR[5] + 2.0 * gs2 * hessR[4] + s * thirdR[8];
                        thirdBlock[t + 9] = radial * ts9 + 3.0 * gradR[2] * hs5 +
                                            3.0 * gs2 * hessR[5] + s * thirdR[9];
                    }
                }
            }
        }
    }
}

void AoEvaluator::EvaluateGradients(const std::array<double, 3>& pointBohr,
                                    std::span<double> values,
                                    std::span<double> gradients) const {
    EvaluateGradientsSelected(pointBohr, _allShells, values, gradients);
}

void AoEvaluator::EvaluateGradientsSelected(const std::array<double, 3>& pointBohr,
                                            std::span<const std::size_t> selection,
                                            std::span<double> values,
                                            std::span<double> gradients) const {
    EvaluateDerivativesTier<DerivativeTier::kGradients>(
        pointBohr, selection, values, gradients, {}, {});
}

void AoEvaluator::EvaluateDerivatives(const std::array<double, 3>& pointBohr,
                                      std::span<double> values,
                                      std::span<double> gradients,
                                      std::span<double> hessians) const {
    EvaluateDerivativesTier<DerivativeTier::kHessians>(
        pointBohr, _allShells, values, gradients, hessians, {});
}

void AoEvaluator::EvaluateDerivativesSelected(const std::array<double, 3>& pointBohr,
                                              std::span<const std::size_t> selection,
                                              std::span<double> values,
                                              std::span<double> gradients,
                                              std::span<double> hessians) const {
    EvaluateDerivativesTier<DerivativeTier::kHessians>(
        pointBohr, selection, values, gradients, hessians, {});
}

void AoEvaluator::EvaluateDerivativesToThird(const std::array<double, 3>& pointBohr,
                                             std::span<double> values,
                                             std::span<double> gradients,
                                             std::span<double> hessians,
                                             std::span<double> thirds) const {
    EvaluateDerivativesToThirdSelected(pointBohr, _allShells, values, gradients, hessians, thirds);
}

void AoEvaluator::EvaluateDerivativesToThirdSelected(const std::array<double, 3>& pointBohr,
                                                     std::span<const std::size_t> selection,
                                                     std::span<double> values,
                                                     std::span<double> gradients,
                                                     std::span<double> hessians,
                                                     std::span<double> thirds) const {
    EvaluateDerivativesTier<DerivativeTier::kThirds>(
        pointBohr, selection, values, gradients, hessians, thirds);
}

// The tier definitions live in this translation unit, so every instantiation
// is emitted here.  The names in an explicit instantiation are not subject to
// access checking, which is what lets the private ShellData and the private
// DerivativeTier appear in them.
template void AoEvaluator::EvaluateDerivativesTier<AoEvaluator::DerivativeTier::kGradients>(
    const std::array<double, 3>&,
    std::span<const std::size_t>,
    std::span<double>,
    std::span<double>,
    std::span<double>,
    std::span<double>) const;
template void AoEvaluator::EvaluateDerivativesTier<AoEvaluator::DerivativeTier::kHessians>(
    const std::array<double, 3>&,
    std::span<const std::size_t>,
    std::span<double>,
    std::span<double>,
    std::span<double>,
    std::span<double>) const;
template void AoEvaluator::EvaluateDerivativesTier<AoEvaluator::DerivativeTier::kThirds>(
    const std::array<double, 3>&,
    std::span<const std::size_t>,
    std::span<double>,
    std::span<double>,
    std::span<double>,
    std::span<double>) const;
template void AoEvaluator::FillAngularDerivativesTier<AoEvaluator::DerivativeTier::kGradients>(
    const AoEvaluator::ShellData&,
    const std::array<double, 3>&,
    std::span<double>,
    std::span<double>,
    std::span<double>);
template void AoEvaluator::FillAngularDerivativesTier<AoEvaluator::DerivativeTier::kHessians>(
    const AoEvaluator::ShellData&,
    const std::array<double, 3>&,
    std::span<double>,
    std::span<double>,
    std::span<double>);
template void AoEvaluator::FillAngularDerivativesTier<AoEvaluator::DerivativeTier::kThirds>(
    const AoEvaluator::ShellData&,
    const std::array<double, 3>&,
    std::span<double>,
    std::span<double>,
    std::span<double>);

} // namespace qcx::grid
