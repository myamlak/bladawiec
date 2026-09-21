// The DEVICE high-water mark for one grid block, with the full
// second-derivative matrix against with the contraction.
//
// This measures the ALLOCATION LAYOUT, on the real device, with the real
// allocator: every buffer the two candidate designs need at one block is
// allocated with cudaMalloc, and the device's own free-memory counter
// (cudaMemGetInfo) is read before and after. It does not measure a working
// kernel - no device-side XC kernel exists in this repository yet - so it is
// a footprint measurement of the two layouts and nothing more. That limit is
// stated with the number.
//
// The two layouts, for P points, A basis functions and C active components:
//
//   MATRIX        coordinates, weights, basis values, 3 gradients, 6 hessians,
//                 the C x C second-derivative matrix PER POINT, and the C-wide
//                 right-hand side;
//   CONTRACTION   the same, with the per-point C x C matrix replaced by the
//                 C-wide contraction result.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

namespace {

/// One named buffer of a layout: what it is and how many bytes it takes.
struct BufferSpec {
    const char* name;
    std::size_t bytes;
};

/// The MATRIX layout's buffers.
std::vector<BufferSpec> MatrixLayout(std::size_t points,
                                     std::size_t basisFunctions,
                                     std::size_t activeComponents) {
    const std::size_t doubles = sizeof(double);

    return {{"coordinates", points * 3 * doubles},
            {"weights", points * doubles},
            {"basis_values", points * basisFunctions * doubles},
            {"basis_gradients", points * basisFunctions * 3 * doubles},
            {"basis_hessians", points * basisFunctions * 6 * doubles},
            {"second_derivative_matrix", points * activeComponents * activeComponents * doubles},
            {"rhs_vector", points * activeComponents * doubles}};
}

/// The CONTRACTION layout's buffers: the matrix replaced by its result.
std::vector<BufferSpec> ContractionLayout(std::size_t points,
                                          std::size_t basisFunctions,
                                          std::size_t activeComponents) {
    const std::size_t doubles = sizeof(double);

    return {{"coordinates", points * 3 * doubles},
            {"weights", points * doubles},
            {"basis_values", points * basisFunctions * doubles},
            {"basis_gradients", points * basisFunctions * 3 * doubles},
            {"basis_hessians", points * basisFunctions * 6 * doubles},
            {"contraction_result", points * activeComponents * doubles},
            {"rhs_vector", points * activeComponents * doubles}};
}

/// Allocates every buffer of \p layout, reporting the device free-memory
/// counter's drop and its own sum of requested bytes.
/// \returns True when every allocation succeeded.
bool AllocateLayout(const char* label,
                    const std::vector<BufferSpec>& layout,
                    std::size_t& deviceDropBytes,
                    std::size_t& requestedBytes) {
    std::size_t freeBefore = 0;
    std::size_t total = 0;
    cudaMemGetInfo(&freeBefore, &total);

    std::vector<void*> blocks;
    blocks.reserve(layout.size());
    requestedBytes = 0;

    for (const BufferSpec& buffer : layout)
    {
        void* device = nullptr;
        const cudaError_t status = cudaMalloc(&device, buffer.bytes);
        requestedBytes += buffer.bytes;

        if (status != cudaSuccess)
        {
            std::printf("m4_%s_allocation_failed = %s (%zu bytes, %s)\n",
                        label,
                        cudaGetErrorString(status),
                        buffer.bytes,
                        buffer.name);
            break;
        }

        blocks.push_back(device);
    }

    std::size_t freeAfter = 0;
    cudaMemGetInfo(&freeAfter, &total);

    if (freeAfter < freeBefore)
    {
        deviceDropBytes = freeBefore - freeAfter;
    } else
    {
        deviceDropBytes = 0;
    }

    std::printf("m4_%s = buffers=%zu requested_bytes=%zu device_free_drop_bytes=%zu "
                "device_total_bytes=%zu device_free_after_bytes=%zu\n",
                label,
                layout.size(),
                requestedBytes,
                deviceDropBytes,
                total,
                freeAfter);

    for (void* block : blocks)
    {
        cudaFree(block);
    }

    return true;
}

} // namespace

int main() {
    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);

    if (countStatus != cudaSuccess || deviceCount == 0)
    {
        std::printf("m4 = NO DEVICE AVAILABLE (%s, count=%d)\n",
                    cudaGetErrorString(countStatus),
                    deviceCount);
        return 0;
    }

    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);

    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);

    std::printf("m4_device = %s, compute %d.%d, total_bytes=%zu free_bytes=%zu\n",
                properties.name,
                properties.major,
                properties.minor,
                totalBytes,
                freeBytes);

    constexpr std::size_t kActiveComponents = 16; // a second-derivative capacity

    for (const std::size_t points : {1024u, 65536u, 1048576u})
    {
        for (const std::size_t basisFunctions : {500u, 1000u})
        {
            std::size_t matrixDrop = 0;
            std::size_t matrixRequested = 0;
            std::size_t contractionDrop = 0;
            std::size_t contractionRequested = 0;

            std::printf("m4_fixture = points=%zu ao=%zu active=%zu\n",
                        points,
                        basisFunctions,
                        kActiveComponents);

            AllocateLayout("matrix",
                           MatrixLayout(points, basisFunctions, kActiveComponents),
                           matrixDrop,
                           matrixRequested);
            AllocateLayout("contraction",
                           ContractionLayout(points, basisFunctions, kActiveComponents),
                           contractionDrop,
                           contractionRequested);

            const std::size_t saving =
                matrixRequested > contractionRequested ? matrixRequested - contractionRequested : 0;
            std::printf("m4_saving_bytes = %zu (%.2f MiB), ratio=%.4f\n",
                        saving,
                        static_cast<double>(saving) / (1024.0 * 1024.0),
                        contractionRequested != 0 ? static_cast<double>(matrixRequested) /
                                                        static_cast<double>(contractionRequested)
                                                  : 0.0);
            std::printf("m4_points_that_fit_this_device_matrix = %.0f "
                        "contraction = %.0f\n",
                        static_cast<double>(totalBytes) /
                            static_cast<double>(matrixRequested / points),
                        static_cast<double>(totalBytes) /
                            static_cast<double>(contractionRequested / points));
        }
    }

    return 0;
}
