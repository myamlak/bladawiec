// The accuracy ladder of the Boys kernel, measured on the committed
// 45-digit reference grid. One construction: the same Chebyshev seed fits the
// kernel ships, evaluated at the compile-time effective degree d'(m) the
// multiplier selects. Only the magnitude moves between rungs; the family does
// not. Prints one CSV row per rung so a study can pick its tiers from measured
// numbers rather than from the envelope alone.
//
// Header-only on purpose: the primary templates live in the kernel's internal
// header (the submodule keeps src/ off the public include path), which is the
// same header the kernel's own contract tests include directly.

#include "boys/boys_impl.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Row {
    int n = 0;
    double x = 0.0;
    double value = 0.0;
};

std::vector<Row> ReadReference(const char* path) {
    std::vector<Row> rows;
    std::ifstream file(path);

    if (!file)
    {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == 'n')
        {
            continue;
        }

        Row row;
        const int fields = std::sscanf(line.c_str(), "%d,%lf,%lf", &row.n, &row.x, &row.value);

        if (fields == 3)
        {
            rows.push_back(row);
        }
    }

    return rows;
}

struct Worst {
    double error = 0.0;
    int n = 0;
    double x = 0.0;

    void Update(double e, int order, double arg) {
        if (e > error)
        {
            error = e;
            n = order;
            x = arg;
        }
    }
};

/// One rung: the three lanes over every reference row, worst error per region.
struct Rung {
    Worst singleA;
    Worst singleB;
    Worst singleC;
    Worst batch;
    Worst f32;
};

template <double kM> Rung Measure(const std::vector<Row>& rows) {
    Rung rung;
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (const Row& row : rows)
    {
        const double value = boys::BoysSingle<kM>(row.n, row.x);
        const double error = std::abs(value - row.value);

        if (row.x < boys::detail::kX0)
        {
            rung.singleA.Update(error, row.n, row.x);
        } else if (row.x < boys::detail::kX1)
        {
            rung.singleB.Update(error, row.n, row.x);
        } else
        {
            rung.singleC.Update(error, row.n, row.x);
        }

        boys::BoysBatch<kM>(row.n, row.x, batch.data());
        const double batchError = std::abs(batch[static_cast<std::size_t>(row.n)] - row.value);
        rung.batch.Update(batchError, row.n, row.x);

        const float f32 = boys::BoysSingleF32<kM>(row.n, static_cast<float>(row.x));
        const double f32Error = std::abs(static_cast<double>(f32) - row.value);

        if (std::isfinite(f32Error))
        {
            rung.f32.Update(f32Error, row.n, row.x);
        }
    }

    return rung;
}

template <double kM> void PrintRung(const Rung& rung) {
    // The degree the multiplier actually selected, so a rung reports the
    // construction it ran at and not only the error it achieved.
    constexpr auto kDegreesA =
        boys::detail::RegionADegrees<kM, boys::detail::BoysRole::kDoubleSingle>();
    constexpr auto kDegreesB =
        boys::detail::RegionBDegrees<kM, boys::detail::BoysRole::kDoubleSingle>();
    int minA = kDegreesA[0];
    int minB = kDegreesB[0];

    for (const int d : kDegreesA)
    {
        if (d < minA)
        {
            minA = d;
        }
    }

    for (const int d : kDegreesB)
    {
        if (d < minB)
        {
            minB = d;
        }
    }

    std::printf("%.10g,%.3e,%.3e,%.3e,%.3e,%.3e,%d,%d\n",
                kM,
                rung.singleA.error,
                rung.singleB.error,
                rung.singleC.error,
                rung.batch.error,
                rung.f32.error,
                minA,
                minB);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: m_ladder <boys_reference.csv>\n");
        return 2;
    }

    const std::vector<Row> rows = ReadReference(argv[1]);
    std::fprintf(stderr, "reference rows: %zu\n", rows.size());

    std::printf("m,err_single_A,err_single_B,err_single_C,err_batch,err_f32,minDegA,minDegB\n");

    // The ladder: a geometric sequence of multipliers. m = 1 is the shipped
    // full-accuracy kernel and doubles as this sweep's reference rung.
    PrintRung<1.0>(Measure<1.0>(rows));
    PrintRung<2.0>(Measure<2.0>(rows));
    PrintRung<4.0>(Measure<4.0>(rows));
    PrintRung<8.0>(Measure<8.0>(rows));
    PrintRung<16.0>(Measure<16.0>(rows));
    PrintRung<32.0>(Measure<32.0>(rows));
    PrintRung<64.0>(Measure<64.0>(rows));
    PrintRung<128.0>(Measure<128.0>(rows));
    PrintRung<256.0>(Measure<256.0>(rows));
    PrintRung<1024.0>(Measure<1024.0>(rows));
    PrintRung<4096.0>(Measure<4096.0>(rows));
    PrintRung<16384.0>(Measure<16384.0>(rows));
    PrintRung<65536.0>(Measure<65536.0>(rows));

    return 0;
}
