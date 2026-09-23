// Measures how region C's accuracy moves when its lower boundary moves.
//
// The branch evaluates one asymptotic term, (1/2)sqrt(pi/x), and then the
// asymptotic recursion, so the degree truncation that relaxes regions A and B
// has nothing to truncate here. The boundary is the only magnitude the branch
// carries. Every error below is measured against the committed high-precision
// grid, not assumed from the formula beside it.

#include "boys/boys.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr double kHalfSqrtPi = 0.886226925452758014; // mirrors the kernel
constexpr long double kPi = 3.14159265358979323846264338327950288L;

struct Row {
    int n;
    double x;
    long double value;
};

/// The kernel's region C body with the boundary supplied rather than fixed.
double RegionC(int n, double x) {
    double f = kHalfSqrtPi / std::sqrt(x);

    for (int l = 0; l < n; ++l)
    {
        f = (l + 0.5) * f / x;
    }

    return f;
}

/// The erfc tail the branch drops at its own boundary: F_0's error where the
/// region starts, and the largest error anywhere in it.
long double PredictedBound(double x1) {
    return std::expl(-static_cast<long double>(x1)) / std::sqrt(kPi * static_cast<long double>(x1));
}

std::vector<Row> ReadGrid(const char* path) {
    std::FILE* file = std::fopen(path, "r");

    if (file == nullptr)
    {
        std::printf("cannot open %s\n", path);
        std::exit(2);
    }

    std::vector<Row> rows;
    char line[512];

    if (std::fgets(line, sizeof(line), file) == nullptr)
    {
        std::fclose(file);
        return rows;
    }

    while (std::fgets(line, sizeof(line), file) != nullptr)
    {
        char* nField = std::strtok(line, ",");
        char* xField = std::strtok(nullptr, ",");
        char* vField = std::strtok(nullptr, ",\r\n");

        if (nField == nullptr || xField == nullptr || vField == nullptr)
        {
            continue;
        }

        rows.push_back(
            {std::atoi(nField), std::strtod(xField, nullptr), std::strtold(vField, nullptr)});
    }

    std::fclose(file);
    return rows;
}

/// The one-sided check that the reproduction above is the shipped branch:
/// where the boundary is the shipped one, the two must agree bit for bit.
int CheckReproductionIsFaithful(const std::vector<Row>& rows) {
    double out[boys::kMaxBoysOrder + 1];
    int bad = 0;
    int checked = 0;

    for (const Row& row : rows)
    {
        if (row.x < 28.9893377388207400)
        {
            continue;
        }

        boys::BoysBatch<boys::kBoysFullAccuracyMultiplier>(row.n, row.x, out);

        const double mine = RegionC(row.n, row.x);
        ++checked;

        if (std::memcmp(&mine, &out[row.n], sizeof(double)) != 0)
        {
            ++bad;
        }
    }

    std::printf("  reproduction agrees with the shipped branch on %d of %d region-C values\n",
                checked - bad,
                checked);

    return bad;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::printf("usage: region_c_ladder <boys_reference.csv>\n");
        return 2;
    }

    const std::vector<Row> rows = ReadGrid(argv[1]);
    std::printf("grid: %zu values, x up to %.6g\n", rows.size(), rows.back().x);
    CheckReproductionIsFaithful(rows);

    // Candidate lower boundaries for region C, shipped value first.
    const double boundaries[] = {
        28.9893377388207400, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0};

    std::printf("\n  boundary   measured max error  predicted   ratio   at (n, x)\n");

    for (const double x1 : boundaries)
    {
        long double worst = 0.0L;
        int worstN = 0;
        double worstX = 0.0;
        long long tested = 0;

        for (const Row& row : rows)
        {
            if (row.x < x1)
            {
                continue;
            }

            ++tested;

            const long double error =
                std::fabs(static_cast<long double>(RegionC(row.n, row.x)) - row.value);

            if (error > worst)
            {
                worst = error;
                worstN = row.n;
                worstX = row.x;
            }
        }

        const long double predicted = PredictedBound(x1);

        std::printf("  %8.4f   %.4Le          %.4Le   %.2f   (%d, %.4f)   %lld values\n",
                    x1,
                    worst,
                    predicted,
                    static_cast<double>(worst / predicted),
                    worstN,
                    worstX,
                    tested);
    }

    return 0;
}
