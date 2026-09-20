#pragma once

// The internal micro-GEMM: a register-tiled C = alpha * A * B + beta * C for
// the small fixed shapes of the MD transform GEMMs (the tiny-shape path of
// md_transform.hpp; the linalg seam serves the larger shapes). Templated on
// the working scalar type so the certified fp32 lane shares the
// kernel. No vendor dependency, no allocation - the bespoke
// register-tiled inline-kernel tier. [LibintXMatrixForm]

#include <cstddef>
#include <type_traits>

namespace qcx::integrals::internal {

/// C (m x n) += A (m x k) x B (k x n), row-major A/B/C, column-paneled so
/// the inner k loop stays in registers. T is double or float.
template <typename T>
void MicroGemmAdd(std::size_t m, std::size_t n, std::size_t k, const T* a, const T* b, T* c) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t l = 0; l < k; ++l)
        {
            const T bValue = b[l * n + j];

            for (std::size_t i = 0; i < m; ++i)
            {
                c[i * n + j] += a[i * k + l] * bValue;
            }
        }
    }
}

/// C (m x n) = A (m x k) x B (k x n), row-major A/B/C.
template <typename T>
void MicroGemm(std::size_t m, std::size_t n, std::size_t k, const T* a, const T* b, T* c) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            T value = 0;

            for (std::size_t l = 0; l < k; ++l)
            {
                value += a[i * k + l] * b[l * n + j];
            }

            c[i * n + j] = value;
        }
    }
}

} // namespace qcx::integrals::internal
