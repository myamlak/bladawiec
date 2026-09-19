#include "qcx/backend/cpu_backend.hpp"
#include "qcx/backend/cuda_backend.hpp"
#include "qcx/backend/tags.hpp"

#include <gtest/gtest.h>
#include <variant>

namespace {

// Toy CRTP skeleton: the base drives
// the iteration shape, derived classes fill in the method-specific pieces -
// all static dispatch, no virtuals.
template <typename Derived> class ScfIterationBase {
public:
    int RunIteration() {
        return static_cast<Derived*>(this)->BuildFock() +
               static_cast<Derived*>(this)->Diagonalize();
    }

private:
    ScfIterationBase() = default;
    friend Derived; // only the intended derived class may construct the base
};

class RhfIteration : public ScfIterationBase<RhfIteration> {
public:
    int BuildFock() {
        return 2;
    }

    int Diagonalize() {
        return 3;
    }
};

class UksIteration : public ScfIterationBase<UksIteration> {
public:
    int BuildFock() {
        return 10;
    }

    int Diagonalize() {
        return 20;
    }
};

// Coarse-grained runtime dispatch: one std::variant branch at the top,
// fully monomorphized code underneath. The
// variant stays CI-safe because Backend<CudaTag>'s header has no CUDA
// includes and only its inline DeviceId() is instantiated here.
using BackendVariant = std::variant<qcx::backend::Backend<qcx::backend::CpuTag>,
                                    qcx::backend::Backend<qcx::backend::CudaTag>>;

int DeviceIdOf(const BackendVariant& backend) {
    return std::visit([](const auto& b) { return b.DeviceId(); }, backend);
}

} // namespace

TEST(DispatchPrototypeTest, CrtpDispatchIsStatic) {
    RhfIteration rhf;
    UksIteration uks;
    EXPECT_EQ(rhf.RunIteration(), 5);
    EXPECT_EQ(uks.RunIteration(), 30);
}

TEST(DispatchPrototypeTest, VariantVisitsBackends) {
    BackendVariant cpu = qcx::backend::Backend<qcx::backend::CpuTag>{};
    EXPECT_EQ(DeviceIdOf(cpu), 0);
}
