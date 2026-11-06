// Two-rank (and any N-rank) MPI smoke test for the build's MPI runtime.
//
// This file uses the bare MPI C interface only - no qcx header, no wrapper - so
// a failure here is the MPI installation's, not the execution context's. Every
// rank's local value is rank + 1, so the all-reduce sum over n ranks is
// n(n+1)/2. The value and threading-level checks are combined with MPI_MIN, so
// a wrong result or a refused threading level on any rank fails the launcher's
// exit code.

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char** argv) {
    int provided = 0;

    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
    {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 2;
    }

    int size = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int local = rank + 1;
    const int expected = size * (size + 1) / 2;

    int reducedInt = 0;
    MPI_Allreduce(&local, &reducedInt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    const double localDouble = static_cast<double>(local);
    double reducedDouble = 0.0;
    MPI_Allreduce(&localDouble, &reducedDouble, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int versionLength = 0;
    MPI_Get_library_version(version, &versionLength);

    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int hostLength = 0;
    MPI_Get_processor_name(host, &hostLength);

    std::printf("rank %d of %d: local value = %d, host = %s\n", rank, size, local, host);
    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::printf("rank count = %d\n", size);
        std::printf("int allreduce sum = %d (expected %d)\n", reducedInt, expected);
        std::printf("double allreduce sum = %.1f (expected %.1f)\n",
                    reducedDouble,
                    static_cast<double>(expected));
        std::printf("mpi library = %s", version);

        if (versionLength > 0 && version[versionLength - 1] != '\n')
        {
            std::printf("\n");
        }
    }

    const int valueOk =
        (reducedInt == expected && reducedDouble == static_cast<double>(expected)) ? 1 : 0;
    const int threadOk = (provided >= MPI_THREAD_FUNNELED) ? 1 : 0;

    int valuesOk = 0;
    MPI_Allreduce(&valueOk, &valuesOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    int threadsOk = 0;
    MPI_Allreduce(&threadOk, &threadsOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0 && threadsOk == 0)
    {
        std::printf("threading level: provided %d, requested %d (MPI_THREAD_FUNNELED)\n",
                    provided,
                    static_cast<int>(MPI_THREAD_FUNNELED));
    }

    std::printf("rank %d: value check = %s\n", rank, valueOk ? "pass" : "FAIL");
    std::fflush(stdout);

    MPI_Finalize();

    if (valuesOk == 0 || threadsOk == 0)
    {
        if (rank == 0)
        {
            std::printf("SMOKE FAIL (process exit code 1)\n");
        }

        return 1;
    }

    if (rank == 0)
    {
        std::printf("SMOKE PASS (process exit code 0)\n");
    }

    return 0;
}
