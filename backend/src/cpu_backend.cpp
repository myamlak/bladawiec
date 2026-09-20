// The CPU backend is header-only; this translation unit exists so qcx-backend
// is a real (non-INTERFACE) library - and it carries the one
// piece that cannot be header-only: the OpenMP runtime's own thread maximum
// (BoundOmpRuntimeThreads / OmpRuntimeMaxThreads), which needs <omp.h> from a
// TU compiled with the module's OpenMP flag. The engine's team can live in a
// header because it is plain arithmetic over the topology probe; the runtime's
// maximum is a call into the runtime.
#include "qcx/backend/cpu_backend.hpp"

#include <omp.h>

namespace qcx::backend {

void BoundOmpRuntimeThreads(int team) noexcept {
    if (team > 0)
    {
        omp_set_num_threads(team);
    }
}

int OmpRuntimeMaxThreads() noexcept {
    return omp_get_max_threads();
}

} // namespace qcx::backend
