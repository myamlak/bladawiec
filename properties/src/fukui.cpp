#include "qcx/properties/fukui.hpp"

namespace qcx::properties {

qcx::Result<FukuiIndices> AnalyzeFukui(
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& neutralAlpha,
    // (neutralBeta, anionAlpha) are the beta of the neutral and the alpha
    // of the anion - distinct species channels, fixed call order.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& neutralBeta,
    const Eigen::MatrixXd& anionAlpha,
    // (anionBeta, cationAlpha) are distinct species channels, fixed call
    // order.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& anionBeta,
    const Eigen::MatrixXd& cationAlpha,
    const Eigen::MatrixXd& cationBeta,
    const std::vector<AoRange>& aoRanges) {
    auto neutral = AnalyzeMulliken(overlap, neutralAlpha, neutralBeta, aoRanges);

    if (!neutral.has_value())
    {
        return std::unexpected(neutral.error());
    }

    auto anion = AnalyzeMulliken(overlap, anionAlpha, anionBeta, aoRanges);

    if (!anion.has_value())
    {
        return std::unexpected(anion.error());
    }

    auto cation = AnalyzeMulliken(overlap, cationAlpha, cationBeta, aoRanges);

    if (!cation.has_value())
    {
        return std::unexpected(cation.error());
    }

    // f_A^+ = q_A(N + 1) - q_A(N), f_A^- = q_A(N) - q_A(N - 1),
    // f_A^0 = (f^+ + f^-) / 2 [Parr1984], [YangMortier1986].
    FukuiIndices indices;
    indices.nucleophilic = anion->total - neutral->total;
    indices.electrophilic = neutral->total - cation->total;
    indices.radical = 0.5 * (indices.nucleophilic + indices.electrophilic);

    return indices;
}

} // namespace qcx::properties
