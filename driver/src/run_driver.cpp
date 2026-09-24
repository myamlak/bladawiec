// The qcx run driver implementation . Composition order:
// validate the method/builder/guess combinations the schema documents,
// build the molecule (the parser already resolved the file's unit and
// converted to Bohr once - this layer copies, never converts), load the
// bundled orbital basis
// (and the auto-selected aux when the method needs one),
// build the one-electron integrals, wire the Fock builder into the scf
// seam with the density-convention adapters the seam tests establish
// (rhf.hpp/uhf.hpp document the contracts; direct_rhf_test.cpp /
// direct_uhf_test.cpp are the reference assemblies), run the SCF under a
// steady-clock measurement, and serialize the JSON result.

#include "qcx/driver/run_driver.hpp"

#include "qcx/backend/cpu_backend.hpp"
#include "qcx/backend/gpu_compute_profile.hpp"
#include "qcx/backend/host_compute_profile.hpp"
#include "qcx/backend/topology_profile.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/driver/basis_counts.hpp"
#include "qcx/driver/process_caps.hpp"
#include "qcx/driver/selection_heuristic.hpp"
#include "qcx/driver/selection_resolution.hpp"
#include "qcx/grid/molecular_grid.hpp"
#include "qcx/integrals/aux_basis.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/gpu_fock_build.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/precision_policy.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/integrals/qfmm_hf_build.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/ri_full_fock.hpp"
#include "qcx/io/molden_export.hpp"
#include "qcx/io/result_json.hpp"
#include "qcx/io/validate_input.hpp"
#include "qcx/memory/allocation_instrument.hpp"
#include "qcx/memory/workspace_budget.hpp"
#include "qcx/molecule/elements.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/properties/charges.hpp"
#include "qcx/properties/density_at_nuclei.hpp"
#include "qcx/properties/eddb.hpp"
#include "qcx/properties/esp.hpp"
#include "qcx/properties/fukui.hpp"
#include "qcx/properties/multipoles.hpp"
#include "qcx/properties/nalewajski.hpp"
#include "qcx/properties/nocv.hpp"
#include "qcx/properties/populations.hpp"
#include "qcx/properties/qtaim.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/symmetry_reduction.hpp"
#include "qcx/scf/uhf.hpp"
// storage is an OPTIONAL module (QCX_ENABLE_IO; the root CMakeLists gates the
// subdirectory and driver/CMakeLists guards both the link and this
// definition). QcxHasStorage marks what only a build with the module can
// compile: the ERI store decorator, the disk rung, and the checkpoint read and
// write. Each guarded site below REFUSES or DISCLOSES by name rather than
// dropping the request.
#if defined(QcxHasStorage)
#include "qcx/storage/cached_eri_batch_engine.hpp"
#include "qcx/storage/disk_ri_fock_build.hpp"
#include "qcx/storage/scf_checkpoint.hpp"
#endif

// The Kohn-Sham composition (driver-internal, the
// internal/footprint.hpp precedent in integrals/src).
#include "internal/ks_composition.hpp"
// The Kohn-Sham grid: the engine, its derivative provider's geometry and
// parameters, and the thresholds both walks read.
#include "internal/ks_grid.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::driver {

// The refusal-clause composers and the NOCV dense-tensor seam the driver
// composes onto the engine's own Create-time refusal texts. Their
// definitions sit at the end of this file (they were the predictive memory
// model's neighbours in memory_model.cpp until it was deleted on
// 2026-09-17; the clauses and the NOCV charge are not predictions and stay).
std::string RiJReinstatementClause();
std::string DirectReinstatementClause();
std::string NocvDenseTensorReinstatementClause();
qcx::Result<void> CheckNocvDenseTensorFit(std::size_t nBasis,
                                          std::size_t nPairs,
                                          double memoryCapGiB);

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;

// The run's disk-tier ERI store handle (`[method] eri_cache_store`; the
// engine-decorator seam's driver side).
//
// The store is built INSIDE the builder's Create-time factory call, because
// that is the ONE point at which the raw engine pair it decorates exists
// (FockBuildOptions::engineDecorator, called once at Create). The driver
// knows the store's PATH and owns what the factory built, which is what lets
// it read the decorator's stats after the SCF - the counters the run record
// publishes.
struct EriStoreHandle {
#if defined(QcxHasStorage)
    /// The decorator the factory built. Null before the factory ran and after
    /// a refused open: "the request never reached a builder" and "the store
    /// refused" are told apart by `refusal` and by the wiring's own record,
    /// never by this pointer alone.
    std::unique_ptr<qcx::storage::CachedEriBatchEngine> store;
#endif
    /// The store's own error text when its open was REFUSED - verbatim, the
    /// one string `resources_resolved.eri_store.demoted_reason` carries (the disclosure rule:
    /// a demotion names its cause, and the cause is the store's own words,
    /// never a paraphrase). Empty while no attempt has been made and after a
    /// successful open. In a build without the storage module the factory
    /// writes StorageModuleAbsentRefusalText() here instead: there is no store
    /// to open, and the demotion's cause is the build's.
    std::string refusal;
};

// What the wiring did with `[method] eri_cache_store`, carried from the
// options assembly to the record. Empty (its default) means the request never
// reached a builder. It is a run-owned out-parameter of the wiring function,
// the `statsSink`/`boundAccum` pattern beside it: the record is assembled one
// scope away from the Create that decides the outcome, and this is the value
// that crosses that boundary - the same reason those two collectors do.
struct EriStoreWiring {
    /// The handle, set exactly when a decorator was handed to a builder. The
    /// factory closures hold a copy, so the store is alive for every call that
    /// routes through it and for the record read afterwards.
    std::shared_ptr<EriStoreHandle> handle;
    /// True when the run's direct builder carried a POSITIVE in-memory cache
    /// budget - the tier that serves an unengaged request's batches. Read at
    /// the install site from the same options struct the builder gets, so the
    /// record's `engaged` word cannot describe a tier the run did not have.
    bool ramTierInForce = false;
};

// The engine-decorator factory for the disk-tier ERI store. Declared here
// because it has TWO install sites that must share it - the restricted
// wiring's machinery member and the unrestricted runner's two halves - and
// the first of those is defined above the factory's own definition. Its
// contract is documented once, at the definition.
qcx::integrals::EngineDecoratorFactory MakeEriStoreFactory(std::shared_ptr<EriStoreHandle> handle,
                                                           const qcx::molecule::Molecule& molecule,
                                                           const qcx::basisset::BasisSet& basis,
                                                           std::string orbitalBasisName,
                                                           std::string auxBasisName,
                                                           std::filesystem::path storePath);

// The wired builder plus the artifacts the cap left in force for it. The
// trailing members carry the ri_j budget path (the budget path): the workspace
// budget the engine's Create-time estimate reserved from, kept alive past
// the Create call (the engine nulls its pointer on return) for the
// resources_resolved record, and the engine's mode record.
struct WiredFockBuilder {
    qcx::scf::FockBuilderFn fn;
    /// True when the RI-J disk route was FORCED rather than ladder-selected
    /// (the input's [diagnostics] force_disk_ri, which replaced the retired
    /// method.ri_tensor_mode = "forced_disk" word): the driver skipped the
    /// composed in-memory ladder and went straight to the storage-module
    /// disk builder. It rides into the run
    /// record's mode_record.forced_disk - the one fact that
    /// tells a forced measurement cell from a run the ladder itself sent to
    /// disk. False on the ladder's own disk fallback and everywhere else.
    bool forcedDiskRung = false;
    /// True when the RI-J 3c orbit expansion actually ran on this builder -
    /// the ENGINE's own answer
    /// (RiJkFockBuilder::OrbitExpansionEngaged), read at the wiring point and
    /// never recomputed here. It rides into
    /// resources_resolved.ri_orbit_expansion, whose `requested` member names
    /// the key that asked - the two together are what separate a run that
    /// asked and got it, a run that asked and could not use it (a trivial
    /// group), and a run that never asked. False everywhere else, including
    /// every family whose options carry no reduction.
    bool orbitExpansionEngaged = false;
    /// True when THIS route built the two point-group reductions and handed
    /// them to the RI-J builder . Distinct from
    /// orbitExpansionEngaged, and the record needs both: a route that never
    /// wired an orbit-aware builder (the disk rung's storage-module builder,
    /// every non-RI family) must report `not_applicable`, NOT
    /// `inert_trivial_group`, which is a statement about the molecule's point
    /// group and would be a fabrication here.
    bool orbitExpansionWired = false;
    std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;
    std::optional<qcx::integrals::FockModeInfo> modeInfo;
    /// The composed full-RI builder's own Create-time rung decision (the
    /// ri_jk family, RiFullFockModeInfo): the RI-K counterpart of modeInfo
    /// above, and a SEPARATE carrier rather than a conversion of it - the two
    /// families' rung vocabularies and term decompositions differ (kFast/
    /// kBlocked against kFastPath/kLightPath, a retained transform against the
    /// RI-J engine's pair stores and chunk arenas), so folding one into the
    /// other's block would make `mode_record` mean two things. Absent on every
    /// other family, and absent on a ri_jk run created without a budget
    /// (engaged == false: no decision was made, and a record claiming one
    /// would state a read that did not happen).
    std::optional<qcx::integrals::RiFullFockModeInfo> riJkModeInfo;
    /// The composed-QFMM model the Coulomb half ran, set at the
    /// kQfmm arm of WireRhfFockBuilder from the builder's own
    /// ModelRecord() - the engine's resolution, never a driver-side copy of
    /// it. Empty on every other family, which is what leaves the record's
    /// `qfmm_model` block absent there.
    std::optional<qcx::integrals::QfmmModelRecord> qfmmModel;
    // The ri_j builder copy: the run flow reads the engine's
    // accumulated term counters through it after the SCF loop. The copy
    // shares the State with the seam lambda's captured builder (the
    // MakeRhfFockBuilder capture is a copy of the same builder), so the
    // post-run read sees every per-call accumulation. Absent on the other
    // builder kinds.
    std::optional<qcx::integrals::RiJkFockBuilder> riJkBuilder;
    // The disk route's scratch-store path (the disk-rung knob): set only
    // when the disk rung engaged. The store's HDF5 file
    // stays open for the builder's whole lifetime (one handle), so the
    // removal must wait for every builder copy to die - the seam lambda's
    // captured DiskRiFockBuilder lives inside fn, which is why the
    // destructor below clears the seam first and only then removes the
    // file (the destructor body runs before any member destruction). The
    // removal is deliberately non-fatal: a leftover scratch file in the OS
    // temp dir is cleared by the next run's fresh-path contract.
    std::optional<std::filesystem::path> diskStorePath;

    // The Kohn-Sham seam's companions the Kohn-Sham composition: the energy seam
    // (10.1) needs J[D] and the functional's own contribution alongside the
    // Fock, so a Kohn-Sham run hands the loop three callbacks where a
    // Hartree-Fock run hands it one. Set exactly when the composition was
    // engaged; both stay empty on every Hartree-Fock run, which is what makes
    // the run site's branch read the seam rather than a flag. Not part of the
    // brace-initialized head - the value constructor below leaves them empty
    // and the KS arms fill them - so the {seam} sites stay valid.
    std::optional<qcx::scf::CoulombFn> coulomb;
    std::optional<qcx::scf::EnergyContributionFn> contribution;

    // The SECOND half's Create-time mode record, for the Kohn-Sham machinery
    // run only. A Kohn-Sham Fock is built from two split builders admitted
    // against the same workspace budget, and each decides its own rung - so a
    // record carrying only the Coulomb half's would hide a disengaged
    // exchange half (the WiredUhfBuilders reasoning, which carries both for
    // the same reason). Absent on every other run: a Hartree-Fock machinery
    // run holds one fused builder and one record.
    std::optional<qcx::integrals::FockModeInfo> exchangeModeInfo;

    // The removal ordering above is why the cleanup lives in a destructor
    // instead of a member: the destructor body runs before ANY member
    // destruction, so clearing fn here releases the captured
    // DiskRiFockBuilder State (its HDF5 handle) while the file is still
    // removable. A user-declared destructor suppresses the implicit move
    // operations (and the unique_ptr member deletes the copy), so the
    // moves are defaulted back - WireRhfFockBuilder returns this struct by
    // value. The struct stays an aggregate (a destructor is not a
    // constructor), so the brace-initialization sites remain valid.
    ~WiredFockBuilder() {
        if (diskStorePath.has_value())
        {
            fn = {};

            std::error_code ignored;
            std::filesystem::remove(*diskStorePath, ignored);
        }
    }

    WiredFockBuilder(WiredFockBuilder&&) = default;
    WiredFockBuilder& operator=(WiredFockBuilder&&) = default;
    WiredFockBuilder(const WiredFockBuilder&) = delete;
    WiredFockBuilder& operator=(const WiredFockBuilder&) = delete;

    // The value constructor keeps the {seam} brace sites valid: the
    // defaulted moves are user-declared constructors, which disqualify the
    // struct from aggregate initialization, so the one-element lists need a
    // real constructor.
    WiredFockBuilder(qcx::scf::FockBuilderFn seam) : fn(std::move(seam)) {}
};

// The certified-bound collector: the run's reduction over
// the per-call certifiedBoundSumOut values the direct family's machinery
// members report. The driver owns ONE of these next to the SCF branch's
// stats stream and hands its address to the seam (the same stable-address
// contract the stream has: the seam lambdas capture it, and it outlives
// the SCF call). It is a plain accumulator, not a stream - nothing is
// written until the run record is built, and a run that observes no call
// at all (every non-machinery builder kind) leaves it empty, which is
// what makes the record's certified_bound block absent rather than a
// fabricated zero. Observe() is the per-call fold: it counts the call,
// keeps the latest bound as the final one, and raises the maximum.
class CertifiedBoundAccumulator {
public:
    /// Fold one observed call's bound sum (hartree) and its budget
    /// enforcement outcome into the run totals.
    /// \param callBoundHa The call's certifiedBoundSumOut value.
    /// \param budget The call's CertifiedBudgetOutcome; an unenforced call
    /// leaves the enforcement members at their zero defaults and its
    /// `enforced` flag false, which is what marks them not-measured.
    void Observe(double callBoundHa,
                 const qcx::integrals::CertifiedBudgetOutcome& budget) noexcept {
        ++_calls;
        _lastCallHa = callBoundHa;
        _maxCallHa = _calls == 1 ? callBoundHa : std::max(_maxCallHa, callBoundHa);
        _enforced = budget.enforced;
        _budgetHa = budget.budgetHa;
        _routedHa = budget.routedHa;
        _routedQuartets = static_cast<long long>(budget.routedQuartets);
        _fellBackToFp64 = budget.fellBackToFp64;
    }

    /// The number of calls folded in - 0 when no machinery call was made,
    /// the state the record reads as "absent".
    /// \returns The observed call count.
    long long Calls() const noexcept {
        return _calls;
    }

    /// The final observed call's bound sum (hartree); 0.0 when none was
    /// observed (read only alongside Calls()).
    /// \returns The last call's bound sum.
    double LastCallHa() const noexcept {
        return _lastCallHa;
    }

    /// The largest observed call's bound sum (hartree).
    /// \returns The maximum call bound sum.
    double MaxCallHa() const noexcept {
        return _maxCallHa;
    }

    /// True when the final observed call ran the global certified-bound budget
    /// enforcement ([method] enforce_certified_bound) - the flag that makes
    /// the five members below measurements rather than zero defaults.
    /// \returns The enforcement state of the final observed call.
    bool Enforced() const noexcept {
        return _enforced;
    }

    /// The final observed call's derived budget (hartree).
    /// \returns The budget in Eh.
    double BudgetHa() const noexcept {
        return _budgetHa;
    }

    /// The final observed call's routed bound sum (hartree) - the quantity
    /// the budget was compared against.
    /// \returns The routed bound sum in Eh.
    double RoutedHa() const noexcept {
        return _routedHa;
    }

    /// The final observed call's routed quartet count.
    /// \returns The routed quartet count.
    long long RoutedQuartets() const noexcept {
        return _routedQuartets;
    }

    /// The comparison's verdict on the final observed call.
    /// \returns True when the build fell back to the fp64 lane.
    bool FellBackToFp64() const noexcept {
        return _fellBackToFp64;
    }

private:
    long long _calls = 0;
    double _lastCallHa = 0.0;
    double _maxCallHa = 0.0;
    bool _enforced = false;
    double _budgetHa = 0.0;
    double _routedHa = 0.0;
    long long _routedQuartets = 0;
    bool _fellBackToFp64 = false;
};

// The driver's grid for the quadrature-based charge analyses: the pinned
// 80 x 194 grid of the charges tests (charges_test.cpp).
constexpr std::size_t kGridRadialPoints = 80;
constexpr std::size_t kGridAngularPoints = 194;

// The GiB conversion (2^30 bytes), for the audit block's cache budget.
constexpr double kGiB = 1073741824.0;

// The n x n matrices the driver holds live across the setup ramp: the core
// Hamiltonian is T + V (both live through the sum that folds them) and the
// overlap matrix is built while the core Hamiltonian is still held, so two
// is the maximum concurrent count and the ramp's charged matrix term must
// not fall below it (integrals::SetupRampPeakBytes).
constexpr std::size_t kRampLiveMatrices = 2;

// The workspace budget's grant under a memory cap: the cap MINUS the run's
// reserve, so the pre-gate arm's floor and the engine's rung condition are
// drawn from ONE quantity instead of two that can drift apart.
//
// The reserve is `integrals::SetupAdmissionReserveBytes` - the engine's own
// ramp-and-sweeps bound times its own slack, which is also what the pre-gate
// arm refuses on - and it is computed ONCE per run and threaded to every budget
// site, never re-derived at one. This reservation is the half the deleted
// predictive model used to supply (every site was
// `WorkspaceBudget::Create(cap - modeledBase)`); its ADMISSION half was
// re-derived as the pre-gate arm on 2026-09-18 and the reservation half was
// not, so every budgeted rung was handed the WHOLE cap and the cheaper rungs
// (RiFullFockRung::kBlocked, the RI-J disk rung) were unreachable through the
// driver at every cap. The two fixtures that pinned them: `ri_jk_propane` needs
// the grant below its fast class (51,829,880 B) while the arm admits only
// cap >= 238,481,196 B, and `ri_j_disk` needs the grant under 9,895,696 B (the
// engine's own clamp-refusal edge) while the process needs ~48 MB of cap.
//
// A reserve that does not fit under the cap is NOT subtracted: that is a cap
// below the run's own need, which the pre-gate arm (or the absent-cap path)
// already owns, and granting zero there would silently route a working run onto
// the null-budget legacy path.
//
// The two arguments are one quantity and its reserve: all eight budget sites
// read them off the same input, in the same order (the cap from the run's own
// resources, the reserve computed once per run above), so the pair is ordered
// by its source rather than by the signature.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline std::size_t WorkspaceGrantBytes(double memoryCapGiB, std::size_t reserveBytes) noexcept {
    const auto capBytes = static_cast<std::size_t>(memoryCapGiB * kGiB);
    return reserveBytes < capBytes ? capBytes - reserveBytes : capBytes;
}

// The lean ceiling this file used to carry as the no-builder
// default's split point is GONE with the 2026-09-13 ladder: the boundary is
// the ladder's own (qcx::driver::kLeanTierMaxBasisFunctions, declared with
// the policy in selection_resolution.hpp) and the member choice is the
// resolution's, so nothing here re-decides either.

// The pre-gate setup admission's clause, appended to the admission text it
// consults (CheckPreGateSetupAdmission): the reader must be able to tell a
// decision taken BEFORE the shared setup ramp from the same decision taken
// after it - the placement is the fix, so the refusal names it. Space-
// prefixed for direct appending, the reinstatement-clause convention.
constexpr const char* kPreGateSetupClause =
    " The refusal lands before the shared pre-gate setup ramp (the core-Hamiltonian and "
    "overlap builds walk the whole canonical pair list, materializing one pair-data chunk "
    "at a time): the admission is consulted before the allocation it sizes, so this cap "
    "refuses here instead of killing the process inside the ramp.";

// Builds a qcx::Error from a code and message.
qcx::Error Err(qcx::ErrorCode code, std::string message) {
    return qcx::Error{code, std::move(message)};
}

// The unique atomic numbers of a molecule, sorted: the keep-set for the
// filtered basis parses (ParseNwchemDirectoryFiltered). With the filtered
// parse, the parsed set carries exactly the molecule's elements, so the
// model's per-atom basis-function sum and the integrals engine's
// molecule-scoped count are equal by construction (the filtered-parse
// contract); a directory-wide parse would over-count elements the
// molecule does not contain.
std::vector<int> UniqueAtomicNumbers(const qcx::molecule::Molecule& molecule) {
    std::vector<int> zs;
    zs.reserve(molecule.Atoms().size());

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        zs.push_back(atom.atomicNumber);
    }

    std::sort(zs.begin(), zs.end());
    zs.erase(std::unique(zs.begin(), zs.end()), zs.end());
    return zs;
}

// Tensor <-> Eigen conversions, element-by-element (row-major to
// column-major pinning). This is the production copy of the
// tests/fixtures/tensor_conversions.hpp helpers: the driver is a library
// and must not link the test-fixture include path, so the ~30 lines are
// kept here under the anonymous namespace.
Eigen::MatrixXd ToMatrix(const CpuTensor2& tensor) {
    const std::size_t n = tensor.Shape()[0];
    Eigen::MatrixXd matrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = tensor(i, j);
        }
    }

    return matrix;
}

qcx::Result<CpuTensor2> ToTensor(const Eigen::MatrixXd& matrix) {
    const std::size_t n = static_cast<std::size_t>(matrix.rows());
    auto tensor = CpuTensor2::Create({n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*tensor)(i, j) = matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

// Eigen vector -> plain std::vector (contiguous, no conversion).
std::vector<double> ToStdVector(const Eigen::VectorXd& vector) {
    return std::vector<double>(vector.data(), vector.data() + vector.size());
}

// Eigen matrix -> plain nested std::vector (row-major copy).
std::vector<std::vector<double>> ToStdMatrix(const Eigen::MatrixXd& matrix) {
    const std::size_t rows = static_cast<std::size_t>(matrix.rows());
    const std::size_t cols = static_cast<std::size_t>(matrix.cols());
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            result[i][j] = matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    return result;
}

// Fills the io result's SCF fields shared by the RHF and UHF branches
// (converged/iterations/energies/loop time); the UHF branch adds the
// spin-squared diagnostic itself.
void FillScfResult(qcx::io::RunResult& result,
                   bool converged,
                   // The scalars are one record's fields (iteration count,
                   // the gate's two achieved residuals, total/electronic
                   // energy, loop time) written under the same names in a
                   // fixed call order.
                   //
                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                   int iterations,
                   double achievedEnergyDelta,
                   double achievedRmsDensityChange,
                   double totalEnergy,
                   double electronicEnergy,
                   double scfLoopMs,
                   // The linear-dependence removal's count (a disclosure):
                   // 0 is a measurement - a well-conditioned system - not the
                   // "unknown" a null means in the record.
                   std::size_t numRemovedOverlapDirections) {
    result.converged = converged;
    result.iterations = iterations;
    result.energyDeltaHartree = achievedEnergyDelta;
    result.rmsDensityDelta = achievedRmsDensityChange;
    result.totalEnergyHartree = totalEnergy;
    result.electronicEnergyHartree = electronicEnergy;
    result.timingsMs.scfLoopMs = scfLoopMs;
    result.numRemovedOverlapDirections = numRemovedOverlapDirections;
}

// Forward declarations: the per-analyzer builders serve FillProperties and
// the SAD fragment machinery serves RunDriver/RunDirectUhfScf, but all stay
// grouped with the other builders at the bottom of the file; declared here
// so the analyzer block reads top-down.
qcx::Result<std::map<int, qcx::scf::AtomicUhfInputs>> BuildAtomicInputs(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basis);
qcx::Result<std::vector<std::vector<std::size_t>>> MapFragmentsToCanonical(
    const std::vector<std::vector<std::size_t>>& fragments,
    const std::vector<std::size_t>& canonicalOrder);
qcx::Result<qcx::molecule::Molecule> MakeChargedMolecule(const qcx::molecule::Molecule& molecule,
                                                         int charge,
                                                         int multiplicity);

struct UhfRun {
    qcx::scf::UhfResult scf;
    double scfLoopMs = 0.0; ///< The RunUhfScf wall time (main-run timing).
    // The adaptive seam's artifacts (wired like the RHF path): the budget
    // (its committed bytes land in resources_resolved) and the TWO Fock
    // builders' Create-time mode records - the coulomb half (modeInfo)
    // and the exchange half (exchangeModeInfo): the UHF Fock builds from
    // both, and each is admission-gated against the same budget, so the
    // class-table gate's observation surface needs both (a disengaged
    // exchange half is invisible in the coulomb half's record). The
    // composed-QFMM UHF runner fills the same three members: its
    // shared-budget Create exposes the QFMM half's record here and the
    // exchange-only half's there.
    std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;
    std::optional<qcx::integrals::FockModeInfo> modeInfo;
    std::optional<qcx::integrals::FockModeInfo> exchangeModeInfo;
    /// The composed-QFMM model the Coulomb half ran: the
    /// engine's own record (QfmmHfFockBuilder::ModelRecord), carried here so
    /// the caller writes what ran rather than what was asked for. The
    /// composed-QFMM runner fills it; every other UHF runner leaves it empty
    /// (no QFMM model ran), which is what makes the record's `qfmm_model`
    /// block absent rather than defaulted there.
    std::optional<qcx::integrals::QfmmModelRecord> qfmmModel;
    /// The ri_j_link arm's own carriers, the WiredFockBuilder pair's UHF
    /// twins: the builder whose per-run calibration term counters the
    /// record publishes on an instrumented run (its Create-time 3c pass
    /// counted too - the same read the restricted leg makes), and the orbit
    /// expansion's request/outcome pair (whether this run wired
    /// the key, and the engine's own answer to whether the mechanism
    /// engaged). Left empty/false by every other UHF runner - nothing else
    /// has a ri_j_link builder or a 3c task grid to reduce.
    std::optional<qcx::integrals::RiJkFockBuilder> riJkBuilder;
    bool orbitExpansionEngaged = false;
    bool orbitExpansionWired = false;
    /// The composed full-RI family's Create-time rung decision,
    /// carried exactly as WiredUhfBuilders carries it and filled by the
    /// shared tail from there: the unrestricted ri_jk runner alone sets one,
    /// and every other UHF arm's record keeps the block absent rather than
    /// stating a decision no builder made.
    std::optional<qcx::integrals::RiFullFockModeInfo> riJkModeInfo;
};

qcx::Result<UhfRun> RunDirectUhfScf(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const qcx::backend::GpuComputeProfile& deviceComputeProfile,
    std::optional<bool> certifiedLaneRequest,
    // The resolved member choice, from ONE place: the builder resolution's
    // ResolvedBuilderSelection::leanMember (the 2026-09-13 ladder). Before
    // the ladder each consumer re-derived it from (input, nBasis), which a
    // demoted run would have got wrong.
    bool leanMember,
    // The run's disk-tier ERI store wiring (`method.eri_cache_store`), the
    // `EriStoreWiring` out-parameter the machinery arm installs the
    // engine-decorator factory on and the record reads back afterwards. Null
    // means no caller is collecting it (the Fukui charged-species runs), in
    // which case no decorator is installed and nothing is left hanging.
    EriStoreWiring* eriStore = nullptr,
    bool enableTrace = false);

// One composed-QFMM UHF run: the definition follows RunDirectUhfScf.
qcx::Result<UhfRun> RunQfmmUhfScf(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes);

// One unrestricted RI-J-link run (the per-spin adapter of the ri_j_link
// family): the definition follows RunQfmmUhfScf.
qcx::Result<UhfRun> RunRiJLinkUhfScf(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const std::filesystem::path& root,
    bool enableTrace);

// The opt-in analyzers: each returns nullopt when its
// analysis was not requested; a requested analysis that cannot run fails
// the whole run - it must never be silently dropped from the JSON
// (FillProperties assembles the io block from the engaged results).
qcx::Result<std::optional<qcx::io::RunCharges>> BuildChargesBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& spinSummed,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input);
qcx::Result<std::optional<qcx::io::RunEspFit>> BuildEspBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& spinSummed,
    const qcx::io::RunInput& input);
qcx::Result<std::optional<qcx::io::RunEddb>> BuildEddbBlock(const qcx::molecule::Molecule& molecule,
                                                            const qcx::basisset::BasisSet& basis,
                                                            const Eigen::MatrixXd& alphaDensity,
                                                            const Eigen::MatrixXd& betaDensity,
                                                            const qcx::io::RunInput& input);
qcx::Result<std::optional<qcx::io::RunNalewajski>> BuildNalewajskiBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& alphaDensity,
    const Eigen::MatrixXd& betaDensity,
    const qcx::io::RunInput& input);
qcx::Result<std::optional<qcx::io::RunFukui>> BuildFukuiBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const Eigen::MatrixXd& alphaDensity,
    const Eigen::MatrixXd& betaDensity,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const qcx::backend::GpuComputeProfile& deviceComputeProfile,
    std::optional<bool> certifiedLaneRequest,
    // The charged species inherit the run's own member choice (same basis,
    // same method, so the same ladder answer - the resolution is read once
    // per run and travels here).
    bool leanMember);
qcx::Result<std::optional<qcx::io::RunNocv>> BuildNocvBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& core,
    const Eigen::MatrixXd& spinSummed,
    const std::vector<std::size_t>& canonicalOrder,
    const qcx::io::RunInput& input);
qcx::Result<std::optional<qcx::io::RunDensityAtNuclei>> BuildDensityAtNucleiBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& spinSummed,
    const qcx::io::RunInput& input);
qcx::Result<std::optional<qcx::io::RunQtaim>> BuildQtaimBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& spinSummed,
    const qcx::io::RunInput& input);
qcx::Result<std::optional<qcx::io::RunXcGradient>> BuildXcGradientBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& alphaDensity,
    const Eigen::MatrixXd& betaDensity,
    const qcx::io::RunInput& input);

// The Molden export: writes the [Molden Format]
// [Atoms] (AU) [5D] [7F] [GTO] [MO] file of the converged or
// last-iterate SCF result when the input requested it; a file that
// cannot be written fails the run - a broken request, never a silent
// skip. The occupations are computed here (the aufbau responsibility);
// the moBlocks carry the spin channel(s) the SCF branch assembled.
qcx::Result<void> WriteMoldenExport(qcx::io::RunResult& result,
                                    const qcx::molecule::Molecule& molecule,
                                    const qcx::basisset::BasisSet& basis,
                                    std::span<const qcx::io::MoldenMolecularOrbitals> moBlocks,
                                    const qcx::io::RunInput& input);
// The aufbau occupations of one spin channel: occupiedValue per occupied
// orbital (RHF 2.0, UHF 1.0), 0.0 beyond.
Eigen::VectorXd AufbauOccupations(std::size_t nOccupied,
                                  double occupiedValue,
                                  std::size_t nOrbitals);

// Wires the RHF Fock builder per the input's builder selection (the
// QFMM/GPU/RI-J-link/direct chain of RunDriver), with the modeled memory
// terms the cap was checked against.
qcx::Result<WiredFockBuilder> WireRhfFockBuilder(
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    qcx::io::BuilderKind kind,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const std::filesystem::path& root,
    const qcx::backend::GpuComputeProfile& deviceComputeProfile,
    std::optional<bool> certifiedLaneRequest);

// The properties block plus the opt-in analyses
//  of the converged (or last-iterate) density, converted to io's
// plain data. Per-spin densities follow the properties-module convention:
// P_sigma = C_sigma,occ C_sigma,occ^T with no factor of 2 (an RHF caller
// passes HfResult::density / 2; a UHF caller passes
// densityAlpha/densityBeta unchanged). The spin-summed density D = P_alpha
// + P_beta is what the spin-summed analyses (charges, ESP, NOCV) take.
// The always-on populations/moments block runs here; each opt-in analysis
// is delegated to its per-analyzer builder (Build*Block), which returns
// nullopt when the analysis was not requested. A requested analysis that
// cannot run fails the whole run on the builder's error - it must never be
// silently dropped from the JSON.
qcx::Result<void> FillProperties(
    qcx::io::RunResult& result,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& alphaDensity,
    const Eigen::MatrixXd& betaDensity,
    const Eigen::MatrixXd& overlap,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& core,
    const std::vector<std::size_t>& canonicalOrder,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const qcx::backend::GpuComputeProfile& deviceComputeProfile,
    std::optional<bool> certifiedLaneRequest,
    // The run's resolved member choice, for the Fukui charged species: they
    // run through the direct-UHF runner and must wire the member the parent
    // run wired (same basis, same method - the resolution answers once).
    bool leanMember) {
    auto analyses = qcx::properties::AnalyzePopulations(molecule, basis, alphaDensity, betaDensity);

    if (!analyses.has_value())
    {
        return std::unexpected(analyses.error());
    }

    auto moments = qcx::properties::AnalyzeMultipoles(molecule, basis, alphaDensity, betaDensity);

    if (!moments.has_value())
    {
        return std::unexpected(moments.error());
    }

    qcx::io::RunProperties properties;
    properties.populations.mullikenAlpha = ToStdVector(analyses->mulliken.alpha);
    properties.populations.mullikenBeta = ToStdVector(analyses->mulliken.beta);
    properties.populations.mullikenTotal = ToStdVector(analyses->mulliken.total);
    properties.populations.mullikenSpin = ToStdVector(analyses->mulliken.spin);
    properties.populations.lowdinAlpha = ToStdVector(analyses->lowdin.alpha);
    properties.populations.lowdinBeta = ToStdVector(analyses->lowdin.beta);
    properties.populations.lowdinTotal = ToStdVector(analyses->lowdin.total);
    properties.populations.lowdinSpin = ToStdVector(analyses->lowdin.spin);
    properties.populations.mayerBondOrders = ToStdMatrix(analyses->mayer.bondOrders);
    properties.populations.mayerFreeValences = ToStdVector(analyses->mayer.freeValences);
    properties.populations.mayerTotalValences = ToStdVector(analyses->mayer.totalValences);
    properties.populations.gopinathanJugBondOrders =
        ToStdMatrix(analyses->gopinathanJug.bondOrders);
    properties.moments.dipole = {moments->dipole(0), moments->dipole(1), moments->dipole(2)};

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            properties.moments.quadrupole[i][j] =
                moments->quadrupole(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    // --- The opt-in analyses: one builder per block,
    // each returning nullopt when its analysis was not requested. --------
    const Eigen::MatrixXd spinSummed = alphaDensity + betaDensity;

    auto charges = BuildChargesBlock(molecule, basis, spinSummed, atomicInputs, input);

    if (!charges.has_value())
    {
        return std::unexpected(charges.error());
    }

    properties.charges = std::move(*charges);

    auto esp = BuildEspBlock(molecule, basis, spinSummed, input);

    if (!esp.has_value())
    {
        return std::unexpected(esp.error());
    }

    properties.esp = std::move(*esp);

    auto eddb = BuildEddbBlock(molecule, basis, alphaDensity, betaDensity, input);

    if (!eddb.has_value())
    {
        return std::unexpected(eddb.error());
    }

    properties.eddb = std::move(*eddb);

    auto nalewajski = BuildNalewajskiBlock(molecule, basis, alphaDensity, betaDensity, input);

    if (!nalewajski.has_value())
    {
        return std::unexpected(nalewajski.error());
    }

    properties.nalewajski = std::move(*nalewajski);

    auto fukui = BuildFukuiBlock(molecule,
                                 basis,
                                 coreTensor,
                                 overlap,
                                 core,
                                 alphaDensity,
                                 betaDensity,
                                 atomicInputs,
                                 input,
                                 workspaceReserveBytes,
                                 deviceComputeProfile,
                                 certifiedLaneRequest,
                                 leanMember);

    if (!fukui.has_value())
    {
        return std::unexpected(fukui.error());
    }

    properties.fukui = std::move(*fukui);

    auto nocv = BuildNocvBlock(molecule, basis, core, spinSummed, canonicalOrder, input);

    if (!nocv.has_value())
    {
        return std::unexpected(nocv.error());
    }

    properties.nocv = std::move(*nocv);

    auto densityAtNuclei = BuildDensityAtNucleiBlock(molecule, basis, spinSummed, input);

    if (!densityAtNuclei.has_value())
    {
        return std::unexpected(densityAtNuclei.error());
    }

    properties.densityAtNuclei = std::move(*densityAtNuclei);

    auto qtaim = BuildQtaimBlock(molecule, basis, spinSummed, input);

    if (!qtaim.has_value())
    {
        return std::unexpected(qtaim.error());
    }

    properties.qtaim = std::move(*qtaim);

    auto xcGradient = BuildXcGradientBlock(molecule, basis, alphaDensity, betaDensity, input);

    if (!xcGradient.has_value())
    {
        return std::unexpected(xcGradient.error());
    }

    properties.xcGradient = std::move(*xcGradient);

    result.properties = std::move(properties);
    return {};
}

// The aux basis name one resolved builder puts in effect: the explicit
// [basis].aux when given, else the universal-J SelectAuxBasis result -
// the same rule the ri_j-link wiring applies (the only aux consumer in
// v1; every other builder has no aux). The restart checkpoint
// fingerprint binds this name (scf_checkpoint.hpp), so the save site
// and the guess-restart load site must compute it identically - a
// checkpoint only restores into a run that replicates the original
// run's aux configuration.
//
// The FockBuilderKind the aux rule is asked for on a resolved io builder kind:
// the ONE mapping between the io builder vocabulary and the integrals aux rule,
// shared by the two functions below so that the checkpoint fingerprint's aux
// name and the run record's weak-region notice cannot come apart about which
// fit a kind asks for. A second copy of this mapping is exactly how they would
// stop agreeing - the reason IsDiffuseOrbitalBasis is shared rather than
// repeated in aux_basis.hpp, quoted at its definition.
qcx::integrals::FockBuilderKind AuxRuleKind(qcx::io::BuilderKind kind) {
    return kind == qcx::io::BuilderKind::kRiJk ? qcx::integrals::FockBuilderKind::kRiJk
                                               : qcx::integrals::FockBuilderKind::kDefault;
}

qcx::Result<std::string> AuxNameInEffect(const qcx::io::RunInput& input,
                                         qcx::io::BuilderKind kind) {
    // The kind decides whether an aux is in effect, and the answer comes from
    // the enum's own vocabulary (qcx::io::BuilderConsumesAux, guarded there):
    // the local equality this replaced answered "none" for any kind nobody
    // had written a rule for, which is the wrong answer to inherit for a
    // checkpoint binding - a builder that consumed an aux would store a
    // fingerprint without it, and a restart into a different aux would be
    // accepted silently.
    if (!qcx::io::BuilderConsumesAux(kind))
    {
        return std::string{};
    }

    if (input.basis.aux.has_value())
    {
        return *input.basis.aux;
    }

    // The auto-selected family follows the KIND, not a fixed kDefault: the two
    // aux consumers resolve to different fits for the same orbital basis
    // (aux_basis.hpp SelectAuxBasis - def2-* maps to -jfit for RI-J's link and
    // to the JK fit for ri_jk). A fixed kDefault here would bind the RI-J fit
    // into an ri_jk run's fingerprint, so a checkpoint written by a run whose
    // aux in effect was the JK fit would restore into a run resolving the J-fit
    // - the silent-configuration-drift the fingerprint exists to catch.
    auto selected = qcx::integrals::SelectAuxBasis(input.basis.orbital, AuxRuleKind(kind));

    if (!selected.has_value())
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "no aux basis for \"" + input.basis.orbital +
                                       "\"; specify [basis].aux explicitly"));
    }

    return *selected;
}

// The weak-region notice the run owes a reader about the auxiliary basis its own
// rule chose, or nothing when it owes none (the aux-auto rule; the notice's text is
// qcx::integrals::AuxSelectionNotice and its record home is
// RunApproximation::auxNotice).
//
// READ AT THE SAME RESOLUTION POINT as the name above, and from the same two
// inputs, so it cannot describe a selection the wiring did not make. The disclosure rule
// outcome it discloses is this driver's demotion of an aux REQUEST to a
// DEFAULT, so the disclosure belongs to the resolution that made it; written at
// the record site instead it would be a second opinion about a decision already
// taken here, and the two could disagree.
std::optional<std::string> AuxNoticeInEffect(const qcx::io::RunInput& input,
                                             qcx::io::BuilderKind kind) {
    // A builder with no aux has no aux SELECTION and so no weak region to be in.
    // This is the predicate AuxNameInEffect returns early on, so the two cannot
    // come apart about which kinds reach the rule at all.
    if (!qcx::io::BuilderConsumesAux(kind))
    {
        return std::nullopt;
    }

    // The notice is the disclosure half of a DEMOTION, and this driver
    // demotes only what it chose. An explicit [basis].aux is a request the run
    // USED, and the notice's own text is a statement about the auto-selection:
    // it names the fit SelectAuxBasis returns, which is NOT the fit this run
    // used whenever the input named a different one. Emitting it here would put
    // two different aux names inside one disclosure block - a false record, the
    // defect class this arc exists to close rather than to add.
    if (input.basis.aux.has_value())
    {
        return std::nullopt;
    }

    return qcx::integrals::AuxSelectionNotice(input.basis.orbital, AuxRuleKind(kind));
}

// Whether the shared machinery tail of WireRhfFockBuilder may build for this
// kind. The tail IS the direct family's machinery (the lean member included:
// its arm above only sets options and falls through), and every other wired
// kind returns from its own branch before reaching it - so the only true
// answer is kDirect, and a kind that reaches the tail without being it must be
// refused rather than silently built by the wrong family.
//
// The switch is exhaustive BY CONSTRUCTION - no `default:` arm, and the region
// below makes an unhandled enumerator a BUILD FAILURE - so a kind added to
// BuilderKind without a branch of its own in the dispatch stops the build
// HERE, instead of falling into the machinery tail and producing a run whose
// record names it while the direct family did the work. The equality this
// replaced answered "no" for every new kind and the tail ran anyway.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error : 4061)
#pragma warning(error : 4062)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

bool MachineryFamily(qcx::io::BuilderKind kind) {
    switch (kind)
    {
    case qcx::io::BuilderKind::kDirect:
        return true;

    // The kinds whose own branches return above. They are listed rather than
    // left to a fall-through so the compiler can hold this list against the
    // enum: each one is a value that must never reach the machinery tail.
    case qcx::io::BuilderKind::kRiJLink:
    case qcx::io::BuilderKind::kRiJk:
    case qcx::io::BuilderKind::kQfmm:
    case qcx::io::BuilderKind::kGpu:
    case qcx::io::BuilderKind::kGpuSplit:
        return false;
    }

    // Reached only by a value no enumerator names (a programmatic caller's
    // out-of-range cast): not the machinery's, so the tail refuses it by value
    // below instead of building the direct family under a label nothing
    // substantiates.
    return false;
}

// Whether this kind is one of the GPU family: the kinds whose kernels run on
// the DEVICE, and whose certified fp32 lane default is therefore resolved
// against the device probe rather than the host one . Every
// other kind - the direct family and its lean member, ri_j_link, ri_jk, qfmm
// - runs on the CPU, including the retargeted device-less "gpu"
// fallback, which resolves to a CPU kind here before this predicate is read.
//
// Exhaustive by construction, the MachineryFamily rule above: a kind added to
// BuilderKind must be classified rather than inheriting a default, or the
// build stops. The failure it prevents is SILENT and it is the one the
// compute-profile probe named: a device number reaching a CPU builder
// turns the HOST lane ON, on a box whose device probe reads 31.2 while its
// own host probe reads 2 - the exact inversion of the measured host case.
bool GpuFamily(qcx::io::BuilderKind kind) {
    switch (kind)
    {
    case qcx::io::BuilderKind::kGpu:
    case qcx::io::BuilderKind::kGpuSplit:
        return true;

    case qcx::io::BuilderKind::kDirect:
    case qcx::io::BuilderKind::kRiJLink:
    case qcx::io::BuilderKind::kRiJk:
    case qcx::io::BuilderKind::kQfmm:
        return false;
    }

    // Reached only by a value no enumerator names (the MachineryFamily tail):
    // refuse it by value rather than classify an unknown as a host run.
    return false;
}

// The compute target's profile: the HOST probe's measured numbers
// carried in the one profile struct every builder reads, with the
// device-only capability flags left at their no-device defaults. The tensor
// flags are genuinely false here - a CPU target has no tensor cores - so the
// value is a coherent description of the target, not a stub.
//
// The field it seeds is still NAMED for the device (FockBuildOptions::
// deviceComputeProfile), because the certified lane predates the host arm.
// Renaming it to `computeProfile` is an open item, and a NAMED OPEN OWNER
// DECISION rather than this change's to take; this function is the one
// place where the two names meet, and the
// field's own documentation carries the same reading.
qcx::backend::GpuComputeProfile ComputeTargetProfileOf(
    const qcx::backend::HostComputeProfile& host) noexcept {
    qcx::backend::GpuComputeProfile profile;
    profile.fp32ToFp64Ratio = host.fp32ToFp64Ratio;
    profile.fp32Gflops = host.fp32Gflops;
    profile.fp64Gflops = host.fp64Gflops;
    return profile;
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

// The SCF run paths a [method] word can select, as resolved by ResolveScfPath
// below. The driver's dispatch reads THIS - never the raw enumerator - so
// which physics a method word runs is decided in one place.
//
// The four values are two facts crossed: which SCF LOOP runs (the restricted
// or the per-spin one - the paths that also decide the property runs, the
// guess tier, the checkpoint rules) and which FOCK the loop is handed (the
// Hartree-Fock builder or the Kohn-Sham composition). The loop question is
// what the wiring branches read, so the cross is spelled out here rather than
// hidden behind a separate flag: a word that reached a loop without its
// matching seam would be the silent-substitution defect in its other half.
// Not a bulk-storage type; shrinking the base type is a deferred micro-optimization.
// NOLINTNEXTLINE(performance-enum-size)
enum class ScfPath {
    kRestricted, // Closed-shell Hartree-Fock: RunRhfScf.
    kUnrestricted, // Per-spin Hartree-Fock: RunUhfScf.
    kRestrictedKs, // Closed-shell Kohn-Sham (RKS): RunRhfScf, KS seam.
    kUnrestrictedKs, // Per-spin Kohn-Sham (UKS): RunUhfScf, KS seam.
};

// Whether a run path is the per-spin one. Both unrestricted worlds share it:
// the loop, the property runs and the guess tier read this, and the answer is
// the same whether the Fock came from the HF builder or the KS composition.
/// \param path A resolved run path.
/// \returns True for kUnrestricted and kUnrestrictedKs.
constexpr bool IsUnrestrictedPath(ScfPath path) noexcept {
    return path == ScfPath::kUnrestricted || path == ScfPath::kUnrestrictedKs;
}

// Whether a run path composes the Kohn-Sham seam. This is the ONE test the
// wiring branches read to decide which seam to build - the same discipline
// ResolveScfPath itself follows for the loop, so the two halves of the cross
// cannot disagree about which method word they are serving.
/// \param path A resolved run path.
/// \returns True for the two Kohn-Sham paths.
constexpr bool IsKohnShamPath(ScfPath path) noexcept {
    return path == ScfPath::kRestrictedKs || path == ScfPath::kUnrestrictedKs;
}

// The exhaustiveness guard around ResolveScfPath's switch. The diagnostics
// have to be promoted here to be worth anything: measured on this project's
// lanes, MSVC emits C4062/C4061 for an unhandled enumerator at neither /W3
// nor /W4 (both are off-by-default), GCC emits nothing without -Wall, and
// Clang's -Wswitch is a warning nobody reads. C4061 and -Wswitch-enum are the
// variants that also catch a `default:` arm added to silence the check. Only
// the two switches over ScfPath - the classifier and its word-namer below,
// which is what the record's `method` member is written from - are in
// the region; every other switch in the translation unit keeps the project's
// default diagnostic settings.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error : 4061)
#pragma warning(error : 4062)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

// Resolve a [method] word to the run path that will execute it, or refuse it.
//
// This is the ONE place where a method word becomes physics: the up-front
// combination check and RunDriver's dispatch both read it, so no caller has
// to remember which words are wired. The switch is exhaustive BY
// CONSTRUCTION - no `default:` arm, and the region above makes an unhandled
// enumerator a BUILD FAILURE - so a slice that adds a MethodType enumerator
// cannot leave it unclassified; the compiler stops it. The equality test this
// replaced could not: `method == kUhf` answers false for every word nobody
// had thought about yet, and the dispatch took the restricted branch on false,
// so an unwired method ran Hartree-Fock and the result went out under that
// method's own label with nothing downstream able to tell the two apart.
//
// The tail after the switch is the runtime half, for a value no enumerator
// names (a programmatic caller's out-of-range cast): it is refused by value,
// never dispatched.
qcx::Result<ScfPath> ResolveScfPath(qcx::io::MethodType method) {
    switch (method)
    {
    case qcx::io::MethodType::kRhf:
        return ScfPath::kRestricted;

    case qcx::io::MethodType::kUhf:
        return ScfPath::kUnrestricted;

    // The Kohn-Sham families the Kohn-Sham composition. The composition that folds the
    // XC potential into the Fock build lives in
    // driver/src/internal/ks_composition.hpp and is wired through all four
    // builder arms; classifying the two words here is what lets the dispatch
    // below reach it. RKS is the closed-shell loop with the Kohn-Sham seam,
    // UKS the per-spin one - and the seam is selected by the wiring branches
    // reading IsKohnShamPath, never by a second copy of this classification,
    // so the loop and the Fock cannot disagree about which path is running.
    //
    // These two labels were the LAST thing to change, not the first, and the
    // order is the point. While the composition was being assembled this
    // refusal was the only thing standing between an `rks`/`uks` request and
    // a run of plain Hartree-Fock under a DFT label - the silent-substitution
    // defect class this whole switch exists to end - so it came off only once
    // the composition was complete AND measured: the 12 KsCompositionTest
    // run-layer tests green at the seam, and the end-to-end driver runs
    // (DriverPinTest's RksSlaterH2... / UksSlaterH2... pairs) green through
    // both builder families, at one geometry and one set of [scf] tolerances,
    // one [method] word away from the Hartree-Fock run whose energy they must
    // NOT reproduce.
    case qcx::io::MethodType::kRks:
        return ScfPath::kRestrictedKs;

    case qcx::io::MethodType::kUks:
        return ScfPath::kUnrestrictedKs;
    }

    // Reached only by a value the enum does not name: refused by value, so
    // nothing runs under a label the run record cannot substantiate.
    return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                               "method.type carries the value " +
                                   std::to_string(static_cast<int>(method)) +
                                   ", which no run path is wired for: this build runs "
                                   "\"rhf\", \"uhf\", \"rks\" and \"uks\", so the request is "
                                   "refused rather than run as Hartree-Fock under a label "
                                   "nothing downstream checks"));
}

// The [method] word that names the run path the classifier resolved (the method record,
// ) - the value the record's `method` member is written from, which
// the serializer spells through io's ToString(MethodType).
//
// It reads the classifier's OWN output, never the input enumerator re-read, so
// the word in the artifact and the loop that produced the numbers cannot
// disagree (the same discipline IsUnrestrictedPath and IsKohnShamPath follow).
// The vocabulary is io's, the inverse of ParseMethod, so the record's four
// words are the input's four words: an author writes "rks" and reads "rks"
// back.
//
// The switch sits inside the exhaustiveness region above for the reason that
// region exists: a fifth path added to ScfPath without a word decided here
// would otherwise be named by whichever word the tail returned - a record
// substantiating a run that did not happen, which is the silent-substitution
// class this file's classification exists to end. The compiler stops that
// instead.
/// \param path A run path resolved by ResolveScfPath.
/// \returns The input-vocabulary method word for it.
qcx::io::MethodType ScfPathMethodWord(ScfPath path) noexcept {
    switch (path)
    {
    case ScfPath::kRestricted:
        return qcx::io::MethodType::kRhf;
    case ScfPath::kUnrestricted:
        return qcx::io::MethodType::kUhf;
    case ScfPath::kRestrictedKs:
        return qcx::io::MethodType::kRks;
    case ScfPath::kUnrestrictedKs:
        return qcx::io::MethodType::kUks;
    }

    // Unreachable while the region above is in force; the tail exists because
    // a switch that falls off its end is a warning, and a warning silenced by
    // a `default:` arm is not checked at all.
    return qcx::io::MethodType::kRhf;
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

// The family a per-family KEY refusal must name, without claiming more than
// the input states (an absent key is a request for the system's
// judgement, not a word to be reported as if the author had written it).
// Three shapes, because the input has three: an explicit family word names
// itself; the explicit "lean" spelling is the direct family's within-family
// member, so it names the family it resolves into and the word that put it
// there; and an absent builder names the default family and says that no word
// was given. The Kohn-Sham whitelist below names its family the same way.
//
// It names the REQUEST, and the rules that read it run before
// ResolveBuilderSelection, exactly as that whitelist does. The one request
// the resolution can retarget - the device-less GPU fallback, which lands
// on the default family - consumes no QFMM knob either, so for the keys below
// the refusal is the same sentence whichever of the two families an author
// reads it as.
/// \param input The parsed run input.
/// \returns The family text a refusal quotes.
std::string FamilyNameForRefusal(const qcx::io::RunInput& input) {
    if (input.method.leanDirect)
    {
        return "the direct family (fock_builder = \"lean\", the direct family's within-family "
               "member)";
    }

    if (input.method.builder.has_value())
    {
        return "fock_builder = \"" + std::string(qcx::io::ToString(*input.method.builder)) + "\"";
    }

    return "the no-fock_builder size ladder's own tier for this run (the owner's ruling "
           "2026-09-13: the direct family's lean member at nBasis <= 1000, ri_j_link to 2000, "
           "qfmm above - a tier this run cannot wire is demoted and disclosed)";
}

// The QFMM schema knobs the file actually wrote, in schema order. The refusal
// names them rather than the whole set: an author acts on the key, and a
// message naming four knobs to a file that wrote one sends them looking for
// three keys that are not there.
/// \param input The parsed run input.
/// \returns The dotted names of the QFMM knobs present in the input (possibly
/// empty).
std::vector<std::string> PresentQfmmKnobNames(const qcx::io::RunInput& input) {
    std::vector<std::string> names;

    if (input.method.theta.has_value())
    {
        names.emplace_back("method.theta");
    }

    if (input.method.lMult.has_value())
    {
        names.emplace_back("method.l_mult");
    }

    if (input.method.maxLeafSize.has_value())
    {
        names.emplace_back("method.max_leaf_size");
    }

    if (input.method.crossoverBasisFunctionCount.has_value())
    {
        names.emplace_back("method.crossover_basis_function_count");
    }

    return names;
}

// Join key names the way the refusals quote them: "a", "a and b", "a, b and
// c". Written out rather than folded, because the two-key case is the one a
// reader is most likely to hit and a bare list separator would read as a
// truncated list.
/// \param names The dotted key names, in schema order (possibly empty).
/// \returns The joined phrase, or an empty string for an empty list.
std::string JoinKeyNames(const std::vector<std::string>& names) {
    std::string joined;

    for (std::size_t i = 0; i < names.size(); ++i)
    {
        if (i > 0)
        {
            joined += (i + 1 == names.size()) ? " and " : ", ";
        }

        joined += names[i];
    }

    return joined;
}

// The unrestricted ri_j_link leg's disk-rung refusal, in ONE home: the
// combination check states it up front (the explicit family word) and the
// runner states it again (so a caller that reached the runner another way
// cannot have a rung request dropped) - both read this text, so the two
// senders cannot drift apart.
std::string RiJLinkDiskRungRefusalText() {
    return "the RI disk rung (method.ri_tensor_mode = \"disk\" / [diagnostics] force_disk_ri) is "
           "wired on the restricted ri_j_link leg only: the storage-module disk-backed store has "
           "no per-spin adapter, so an unrestricted run cannot reach it - drop the key, or use "
           "fock_builder = \"direct\" / \"qfmm\"";
}

// The same rung refusal for the Kohn-Sham lanes, in ONE home for the same
// reason: the combination check states it up front and the wiring states it
// again, so a caller that reached the wiring another way cannot have a rung
// request silently dropped. The reason differs from the unrestricted one and
// is the Kohn-Sham contract's own: the disk builder exposes no half, so the
// energy seam would have nothing to contract - not a missing per-spin
// adapter.
std::string RiJLinkKsDiskRungRefusalText() {
    return "the RI disk rung (method.ri_tensor_mode = \"disk\" / [diagnostics] force_disk_ri) is "
           "not wired for a Kohn-Sham run: the storage-module disk-backed builder composes one "
           "fused Hartree-Fock Fock and exposes neither half, so the run would have no J[D] for "
           "its energy seam - drop the key, or use fock_builder = \"direct\" (that family's "
           "split builders carry the halves on every rung)";
}

// The unrestricted legs' disk-tier ERI store refusal, in ONE home: the
// combination check states it beside the requested path and the family the
// run resolved to, and no other site restates it.
//
// THE LOAD-BEARING SENTENCE IS ONE STRING LITERAL ON ONE SOURCE LINE, and
// deliberately so. The sentence this text replaces was split across two
// adjacent literals at a point INSIDE the sentence ("...the restricted
// paths' direct " / "family machinery member..."), so a single-line search
// for the words a record or a document quotes found nothing - a refusal an
// author cannot find is a refusal an author cannot act on, which is the
// same rule the dispatch's own refusals follow.
//
// THE LITERAL IS ALSO SHORT ENOUGH TO SURVIVE THE FORMATTER, which is a
// second constraint and not a stylistic one: `.clang-format` inherits
// BreakStringLiterals from LLVM, so a literal longer than the 100-column
// limit is SPLIT by clang-format - the single-line property is lost by the
// act of committing it cleanly. Measured on this function: a 100-character
// sentence came back out as two literals. The sentence here is 72
// characters, which leaves room for its indentation and its quotes.
// \param storePath The path the input named, verbatim.
// \param resolvedKind The builder family the unrestricted run resolved to.
// \param isKs True on the per-spin Kohn-Sham path (UKS), false on UHF.
// \returns The refusal text.
std::string UnrestrictedEriStoreRefusalText(const std::string& storePath,
                                            qcx::io::BuilderKind resolvedKind,
                                            bool isKs) {
    // The sentence a reader searches for, in ONE home, so the assertion the
    // test makes and the source a reader greps cannot drift apart.
    const std::string seam =
        "the disk-tier ERI store is wired on the direct family's per-spin halves";

    return "method.eri_cache_store = \"" + storePath + "\" resolves to fock_builder = \"" +
           std::string(qcx::io::ToString(resolvedKind)) + "\" on " +
           (isKs ? std::string{"a UKS run"} : std::string{"a UHF run"}) + ", and " + seam +
           ", which this builder is not";
}

// The storage module's own absence, in ONE home for the same reason the two
// texts above share theirs: the fact belongs to the BUILD, not to the site that
// runs into it, so every storage-backed route (the disk rung, the checkpoint
// read, the checkpoint write) refuses with this same sentence and the ERI store
// decorator discloses it as its demotion's cause. It names the build's
// configuration - never the caller's input, which is well-formed and simply
// cannot be served here - because that is the fact a reader has to act on: the
// request is not wrong, this configuration is storage-less.
std::string StorageModuleAbsentRefusalText() {
    return "this build was configured without the storage module (QCX_ENABLE_IO=OFF), so the "
           "storage-backed route cannot be honoured: the module's sources are not compiled in and "
           "no storage target exists - reconfigure with QCX_ENABLE_IO=ON to run it";
}

// The Kohn-Sham composition refusal's REASON, in ONE home: the combination
// check states it for the family the input named.
//
// It was one of TWO texts while the composed full-RI family was a second
// Kohn-Sham refusal, because the two families were held back by two different
// facts and a reader sent after the wrong mechanism looks in the wrong place -
// the rule the ri_j_link disk-rung texts state for the same reason. The device
// family fuses J and K inside one batched kernel launch (one screening pass,
// one partition, one launch, atomicAdd into one buffer), so a composition would
// need two calls of the whole cost. The composed full-RI family does NOT fuse
// any more: it exposes both halves through the pair entry point, which is
// exactly what the unrestricted leg consumes (MakeRiFullUhfFockBuilder), so its
// cell was never structural and has been OPENED - the pair's bare accounting is
// converted at the seam (MakeRiFullKsHalf, both Kohn-Sham lanes) instead of
// being left as a reason to refuse. With that family admitted, the device
// families are the ones this text is stated for.
std::string FusedKsRefusalReasonText() {
    return "fuses its Coulomb and exchange halves inside one BuildFock, so it would need its own "
           "Kohn-Sham composition";
}

// The guess request's RESOLUTION POINT: the ONE place `guess.type` and
// `guess.restart_path` are read as the single request they are. The value's
// own consumers are the solver wiring sites (the GWH/core start, the SAD
// fragment start, the restart read); this answers for the PAIR, which no site
// did - so a restart_path written beside a guess word that does not read it
// was stored by `io` and ignored by the driver, with nothing in the run's
// record able to say so.
//
// The rule is one line: restart_path belongs to restart. The other three words
// take no checkpoint file at all, so a path written beside one of them names a
// file no code path will open.
// \param input The parsed input.
// \returns Nothing, or the refusal naming both keys.
qcx::Result<void> ResolveGuess(const qcx::io::RunInput& input) {
    if (!input.guessRestartPath.empty() && input.guess != qcx::io::GuessKind::kRestart)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "guess.restart_path = \"" + input.guessRestartPath +
                                       "\" was written beside a guess.type that is not "
                                       "\"restart\", and no other start reads a checkpoint file - "
                                       "so the key would be stored and silently ignored. Write "
                                       "guess.type = \"restart\" to load it, or remove "
                                       "guess.restart_path"));
    }

    return {};
}

// The not-yet-wired combinations the schema documents; the driver rejects
// them up front so a silent substitution can never run the wrong physics.
qcx::Result<void> ValidateCombination(const qcx::io::RunInput& input) {
    // The builder is optional: absent means the size ladder (the owner's
    // ruling 2026-09-13 - the direct family's lean member at nBasis <= 1000,
    // ri_j_link to 2000, qfmm above; the resolver never picks an unwired or
    // UHF-unsafe kind, and it demotes a tier this run cannot wire rather
    // than picking it), and the within-family "lean" spelling leaves the
    // slot absent too (RunMethodInput::leanDirect - its own rejection
    // below). The explicit-only rejections must not fire for an absent
    // builder - but the METHOD whitelists below are the same restriction the
    // ladder consults (LadderRunnabilityFor), so an absent key cannot route
    // onto a family the same file would refuse by name for an explicit
    // one.
    // fock_builder = "ri_jk" used to be refused HERE, with a message whose
    // first clause was "no full-RI exchange path exists". The composed full-RI builder answered
    // that clause by existence (integrals RiFullFockBuilder, the composed
    // F = H + 2 J_RI - K_RI with no nested direct-exchange call), so the
    // refusal is gone and the request resolves to a run. The message's SECOND
    // clause - that the approximated exchange is a different result class from
    // the direct family's exact kernels, so existing pins and reference
    // comparisons do not transfer - was answered by DISCLOSURE, not by
    // deletion: the run record carries the approximation statement (see the
    // ri_jk arm of WireRhfFockBuilder and RunSelection's approximation
    // members). Nothing is silently substituted: an ri_jk request either runs
    // as ri_jk and says so, is refused by name, or never resolves at all.
    //
    // What still refuses it, and was verified rather than assumed: on the
    // unrestricted leg, nothing of this family does any more (the per-spin
    // adapter landed with the pair entry point); on the Kohn-Sham lanes, the
    // whitelist named the direct family alone when the full-RI builder
    // landed, took the RI-J
    // link beside it on 2026-09-16, then the composed QFMM builder the same
    // day, and took THIS family the same day too, once the pair's bare
    // accounting was converted at the seam instead of being read as a reason
    // to refuse (MakeRiFullKsHalf). What those two whitelists still refuse is
    // the device families and the ri_j_link disk rungs. Both refusals used to
    // be unreachable behind this one, so it is the one that first
    // exercises them - driver tests pin both, and each later admission moved
    // the cell the pin sits on with its own reason.

    // gpu_split has no run path at all: the kind exists so the selection
    // vocabulary can name the intra-build batch-partition candidate and
    // report it, while execution stays on the wired kinds, and no [method]
    // word spells it (parse_input.cpp ParseBuilder) - so a programmatic caller
    // is the only way to ask for it. Refused here, by name, because the
    // alternative is not a degraded run but a WRONG one: the wiring's if-chain
    // names no branch for it, so without this refusal the request falls
    // through to the shared machinery tail and the direct family executes
    // while resources_resolved goes out reporting "gpu_split" (measured: the
    // identical energy and iteration count as an explicit direct run, and a
    // record whose primary builder fields name a builder that ran nothing).
    if (input.method.builder.has_value() &&
        *input.method.builder == qcx::io::BuilderKind::kGpuSplit)
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                   "fock_builder = \"gpu_split\" has no run path in v1 (the kind "
                                   "names the candidate the selection vocabulary reports, and "
                                   "execution stays on the wired kinds); use \"direct\", \"qfmm\", "
                                   "\"gpu\" or \"ri_j_link\", or leave fock_builder absent"));
    }

    // The method word resolves through the run path's own classifier: this
    // check and the dispatch below therefore cannot disagree about which words
    // are wired, and an unclassified word is refused HERE, up front, with the
    // rest of the combinations rather than at the SCF.
    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value())
    {
        return std::unexpected(scfPath.error());
    }

    // The per-spin question, asked of the RESOLVED path rather than of a
    // method word: both unrestricted worlds (UHF and UKS) answer true, and
    // every rule below reads the same answer the dispatch will act on.
    const bool isUnrestricted = IsUnrestrictedPath(*scfPath);

    // A family word AND the lean flag together state two requests at once:
    // the parse site writes exactly one of the two (the lean word sets
    // leanDirect and leaves the builder slot absent), so only a PROGRAMMATIC
    // caller can produce the pair. Left alone it would wire the lean builder
    // (the wiring reads the flag) while the selection record reported the
    // explicit family word - the selector-record disagreement the
    // memory_model.lean flag exists to prevent. Refuse it here rather than
    // silently honoring one half (the no-silent-substitution rule, and the
    // same posture as the UHF + lean refusal below).
    if (input.method.builder.has_value() && input.method.leanDirect)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "fock_builder = \"" +
                                       std::string(qcx::io::ToString(*input.method.builder)) +
                                       "\" and fock_builder = \"lean\" cannot both be requested: "
                                       "lean is the direct family's within-family member, not a "
                                       "family word (the parse site keeps it out of the builder "
                                       "slot), so the two spellings state different requests - "
                                       "give the lean word alone, or the family word alone"));
    }

    // The lean spelling on a UHF run needs no rule of its own (the unrestricted
    // seam): RunDirectUhfScf carries the lean arm, and the per-spin
    // adapter - MakeLeanUhfFockBuilder - is the half-mode pair the direct
    // seam already assembles, so the within-family request resolves
    // exactly as it does for RHF (an explicit "lean" is admitted at any
    // size; the absent key takes it at nBasis <= the lean ceiling).

    // The Kohn-Sham rules come FIRST, and the order is load-bearing rather
    // than cosmetic: a UKS run is BOTH unrestricted and Kohn-Sham
    // (IsUnrestrictedPath names kUnrestrictedKs), so whichever whitelist is
    // read first decides which reason a UKS request gets. The Kohn-Sham one
    // is the accurate one there - a UKS run asking for the disk rung is
    // refused because the disk builder exposes no half, not because of the
    // unrestricted leg's missing per-spin adapter (a mechanism the Kohn-Sham
    // run was never going to use) - while a UKS run asking for the device
    // family is refused by BOTH, and the composition rule is the reason a
    // reader can act on. Reading them the other way round gives a UKS author
    // a refusal about a different run.
    //
    // The family that used to make this ordering load-bearing on the
    // ri_jk cell has since been admitted by both whitelists, so what the two
    // rules still disagree about is narrower than the ordering's own note
    // claims: the sets are disjoint where it matters, and every family the
    // Kohn-Sham whitelist refuses is one the unrestricted whitelist refuses
    // too - so the reorder moves the TEXT only, and only for the runs both
    // already refuse.

    // The builder-combination whitelist for the Kohn-Sham lanes. A
    // Kohn-Sham Fock is not a Hartree-Fock Fock with an extra term
    // bolted on: the energy formula contracts J[D] on its own (the seam line
    // at the head of internal/ks_composition.hpp), so the run needs a
    // Coulomb half it can hand over SEPARATELY from the Fock it forms - and
    // the half it hands over has to be the one the Fock was built from.
    //
    // The direct family's split builders expose exactly those halves
    // (buildCoulombOnly / buildExchangeOnly), and the composition is written
    // against them. The RI-J link does too, in its own accounting: it was
    // named in this rule's first version as one of the families that "fuse J
    // and K inside their own BuildFock", and that was WRONG about it - the
    // fused BuildFock is a composition of two separate subsystems
    // (RiJkFockBuilder::BuildCoulombOnly over the RI contraction and
    // BuildExchangeOnly over the nested direct-exchange builder,
    // ri_engine.cpp), and both were already public and already wired for the
    // unrestricted leg. The composition's Coulomb half is that
    // BuildCoulombOnly with the family's missing H added back
    // (MakeRiJLinkKsHalf), so what the seam contracts is the builder's own
    // 2 J_RI at the density the Fock was built from.
    //
    // What still refuses here, and WHY, cell by cell:
    //   - ri_jk: OPENED 2026-09-16, on this rule's own terms. The composed
    //     full-RI builder EXPOSES its two halves (the pair entry point
    //     RiFullFockBuilder::BuildFockHalves, landed for the unrestricted
    //     leg), so this cell was never structural - what held it closed was
    //     the pair's bare ACCOUNTING (J_RI(rho) with no factor of two,
    //     K_RI(rho) with no core Hamiltonian), and that is a conversion at
    //     the seam rather than a reason to refuse. MakeRiFullKsHalf states
    //     and performs it - H added to both halves, the Coulomb one doubled -
    //     and both Kohn-Sham lanes (this leg's arm and the unrestricted
    //     runner's) build the composition over the ONE builder the fused
    //     Hartree-Fock Fock would have been built from.
    //   - gpu / gpu_split: the device builder genuinely fuses - one screening
    //     pass, one batch partition, one kernel launch, with J and K
    //     atomicAdd-ing into the same buffer - so a composition would need two
    //     calls and would double the builder's whole cost. (It could be
    //     opened for a PURE functional, where the K half is dropped and one
    //     Coulomb-only pass is cheaper than the fused HF call, but not for a
    //     hybrid - and the CUDA arm is not testable on this machine, so the
    //     cell stays closed rather than opened blind.)
    //   - qfmm: OPENED 2026-09-16. It was named here as "the composed builder
    //     is a J builder plus an exchange-only direct half added together
    //     inside BuildFock; the J half exists on its own (QfmmJBuilder) and
    //     this is the NEXT family to open". The half does exist on its own,
    //     and it was the composed builder that had to say so: its two nested
    //     results are already the composition's own two conventions
    //     (QfmmHfFockBuilder::BuildCoulombOnly returns H + 2 J_QFMM(rho),
    //     BuildExchangeOnly returns H - K(rho)), so the driver's arm cuts
    //     both from the ONE builder the Fock would have been built from and
    //     adds no core Hamiltonian to either (MakeQfmmKsHalf's enum). The
    //     family is also the only non-direct one with a native per-spin
    //     path, which is why this one admission moves the RKS and UKS cells
    //     together.
    // The within-family member needs no rule of its own (the lean builder
    // carries the same two half-mode flags), and the ri_j_link DISK rungs
    // are refused by name below: the storage-module disk builder composes a
    // fused Hartree-Fock Fock and exposes neither half.
    const bool isKs = IsKohnShamPath(*scfPath);

    if (isKs && input.method.builder.has_value() &&
        *input.method.builder != qcx::io::BuilderKind::kDirect &&
        *input.method.builder != qcx::io::BuilderKind::kQfmm &&
        *input.method.builder != qcx::io::BuilderKind::kRiJLink &&
        *input.method.builder != qcx::io::BuilderKind::kRiJk)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kUnimplemented,
                "a Kohn-Sham run (method.type = \"rks\"/\"uks\") is wired with "
                "fock_builder = \"direct\", \"qfmm\", \"ri_j_link\" and \"ri_jk\" in v1 (those "
                "are the families whose builders expose the J[D] the Kohn-Sham energy seam "
                "contracts separately); "
                "fock_builder = \"" +
                    std::string(qcx::io::ToString(*input.method.builder)) + "\" " +
                    FusedKsRefusalReasonText() +
                    " - drop the key, or use \"direct\" (or \"lean\"), \"qfmm\", "
                    "\"ri_j_link\" or \"ri_jk\""));
    }

    // The ri_j_link family's disk rungs on the Kohn-Sham lanes: the
    // storage-module disk-backed builder composes ONE fused Hartree-Fock
    // Fock and exposes neither half, so a Kohn-Sham run has no J[D] to hand
    // its energy seam there. Refused by name rather than let the wiring
    // reach the disk branch and return that fused seam - the shape the
    // unrestricted leg's disk rule uses, and for the same reason (an
    // explicit rung request is never demoted).
    if (isKs && input.method.builder.has_value() &&
        *input.method.builder == qcx::io::BuilderKind::kRiJLink &&
        (input.method.riTensorMode == qcx::io::RiTensorMode::kDisk ||
         input.diagnostics.forceDiskRi))
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented, RiJLinkKsDiskRungRefusalText()));
    }

    // The unrestricted leg's builder whitelist. It named direct and qfmm
    // only, with the reason in its own message - "the per-spin adapters of
    // the other builders are untested" - and that reason was MECHANICAL, not
    // structural: a family whose Fock is assembled from a Coulomb half and a
    // per-spin exchange half needs an adapter, not a redesign. The
    // ri_j_link family now has one (MakeRiJLinkUhfFockBuilder, over
    // RiJkFockBuilder::BuildCoulombOnly / BuildExchangeOnly), so its cell
    // runs and the message names what is left.
    //
    // What still refuses, and why, cell by cell:
    //   - ri_jk: NO LONGER HERE. The composed full-RI builder exposes its two
    //     halves (RiFullFockBuilder::BuildFockHalves) and now has a per-spin
    //     adapter (MakeRiFullUhfFockBuilder, over ONE call per spin), so this
    //     leg runs it - and the builder still refuses a point-group reduction
    //     by name, a separate gap on its own track, which the runner carries
    //     the request into exactly as the restricted arm does.
    //   - gpu / gpu_split: no per-spin adapter exists for the device builder
    //     either, and gpu_split additionally has no run path at all
    //     (refused above by name on every method).
    //   - the ri_j_link DISK rungs (method.ri_tensor_mode = "disk",
    //     [diagnostics] force_disk_ri): the storage-module disk-backed store
    //     has no per-spin adapter, so the rule at the end of this block
    //     refuses them by name on this leg rather than letting a run drop a
    //     rung request it cannot serve (no-silent-substitution rule).
    // The Kohn-Sham whitelist, which is read BEFORE this one (the order's
    // own note above), is a DIFFERENT refusal, and its cell shrank away
    // entirely: ri_j_link's composition landed, the composed full-RI family
    // moved from "exposes no half" through "its halves carry the wrong
    // accounting for the seam" to OPENED (2026-09-16, MakeRiFullKsHalf), and
    // the device families are what it now states. A UKS ri_jk request is
    // therefore admitted by both whitelists and runs the composition on this
    // leg's own runner.
    if (isUnrestricted && input.method.builder.has_value() &&
        *input.method.builder != qcx::io::BuilderKind::kDirect &&
        *input.method.builder != qcx::io::BuilderKind::kQfmm &&
        *input.method.builder != qcx::io::BuilderKind::kRiJLink &&
        *input.method.builder != qcx::io::BuilderKind::kRiJk)
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                   "UHF is wired with fock_builder = \"direct\", \"qfmm\", "
                                   "\"ri_j_link\" and \"ri_jk\" in v1 (gpu and gpu_split have no "
                                   "per-spin adapter; the ri_j_link disk rungs are refused "
                                   "separately, by name)"));
    }

    // The ri_j_link family's disk rungs on the unrestricted leg: the
    // storage-module disk-backed store's per-spin adapter does not exist, so
    // the request is refused by name - an explicit rung request is never
    // demoted, and the ladder keeps a NO-KEY run off this tier when
    // either key is named (LadderRunnabilityFor, the same fact as a demotion
    // reason). Said here for the explicit spelling, because the ladder only
    // governs the absent one.
    if (isUnrestricted && input.method.builder.has_value() &&
        *input.method.builder == qcx::io::BuilderKind::kRiJLink &&
        (input.method.riTensorMode == qcx::io::RiTensorMode::kDisk ||
         input.diagnostics.forceDiskRi))
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented, RiJLinkDiskRungRefusalText()));
    }

    // The condensed-Fukui response runs the N +- 1 species through the
    // DIRECT-UHF runner (BuildFukuiBlock). A Kohn-Sham run asking for it
    // would therefore compute every Fukui index from Hartree-Fock species
    // densities while the parent density is a Kohn-Sham one - one analysis
    // mixing two physics under one label, which is the silent-substitution
    // defect this whole slice exists to prevent. Refused by name until the
    // charged species can be run on the run's own seam.
    if (isKs && input.properties.fukui)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kUnimplemented,
                "properties.fukui is not wired for a Kohn-Sham run in v1: the N +- 1 species "
                "are run by the direct-UHF runner, so their densities would be Hartree-Fock "
                "ones while the parent's is a Kohn-Sham one - the indices would mix two "
                "physics under one label"));
    }

    // The per-family MECHANISM keys, refused by name on a family that has no
    // route for them (the key split, owner-ruled 2026-09-13: a key naming
    // a mechanism the author is relying on is REFUSED, while a key that is
    // only a size hint is dropped and the run proceeds). The distinction is
    // the #24 defect class - an unwired request running something else under
    // the label it carried - and the side a key falls on is judged by what
    // the key NAMES, not by how strongly it is worded.
    //
    // This rule covers the four QFMM knobs: method.theta, method.l_mult,
    // method.max_leaf_size and method.crossover_basis_function_count name the
    // composed-QFMM builder's own tuning. theta is an accuracy control, not a
    // performance knob - it decides which pairs the near/far split computes
    // exactly (integrals/src/qfmm_fock_build.cpp) - so a run that dropped it
    // would compute at a screening the document does not ask for.
    //
    // The far-field model parameters (the extent model, the separation mode and
    // the surface-ball buffer, QfmmOptions::extentModel / separationMode /
    // separationK) are NOT knobs of this rule, and the reason is structural
    // rather than a judgement: no input key spells them, at any spelling -
    // they are engine defaults, and a default is the system's judgement
    // (the default arm) rather than a request to refuse or to drop. What they
    // get instead is the record: the run's `qfmm_model` block
    // states the model the build ran, which is the disclosure half of the
    // same contract and the only half applicable to a value no document can
    // set. The day a key exposes one of them, this rule's side is already
    // decided by what the key would NAME: all three name a MECHANISM - the
    // objects the multipole expansion is of, and which pairs are near field -
    // exactly as theta names the screening, so all three fall on the REFUSE
    // side and not the size-hint side. The near-miss worth recording is
    // separationK: it is IGNORED by the width test (the mode that is not
    // running reads nothing of it), which is the shape that puts
    // method.ri_chunk_bytes on the DROP side - but that key drops because no
    // other FAMILY can honour it, while this rule judges the requested
    // family, and separationK is read inside the family that owns it.
    //
    // The split's other side is method.ri_chunk_bytes alone, and it is NOT
    // refused: a chunk size names no mechanism, and a family with no disk rung
    // has no chunking to size, so the key is dropped and the run proceeds. Its
    // disclosure in the record is OWED, not done - the note at the key's one
    // read site (engageDiskRung, the ri_j_link branch) carries that finding.
    //
    // This rule does NOT extend to method.ri_tensor_mode's rung word, and that
    // is a DEFERRED question rather than a judgement made here. Whether the
    // ladder word ("disk") should refuse on a family with no disk rung, where
    // the sibling force key demotes, is recorded as Finding B of
    //  for the owner, not applied
    // here.
    //
    // The FORCE is a separate key and demotes: [diagnostics] force_disk_ri
    // (the owner's ruling 2026-09-13, which moved the force out of the rung
    // selector) keeps the posture the retired "forced_disk" word carried -
    // DEMOTE with the record stating both sides (resources_resolved.
    // ri_tensor_mode carries requested/resolved/outcome/reason)
    // Demote-with-disclosure - "if it cannot work, demote to your judged best choice"). That
    // disclosure replaced a by-name refusal landed earlier the same day, once
    // the record could do its work, and it stands: the runnable choice (the
    // resolved family's own in-memory path) exists and the record can state
    // it, so a refusal here would reverse a landed, tested posture. The
    // key's refusals that would survive the disclosure rule - no runnable choice, or a
    // demotion the record could not state - do not arise for this mechanism.
    //
    // The rule judges the REQUESTED family word, as the Kohn-Sham whitelist
    // above does, and it fires only for an EXPLICIT key: an absent key is the
    // system's judgement (the default arm), and an absent knob keeps the engine's
    // preset default.
    const bool isQfmmFamily =
        input.method.builder.has_value() && *input.method.builder == qcx::io::BuilderKind::kQfmm;
    const std::vector<std::string> qfmmKnobs = PresentQfmmKnobNames(input);

    if (!isQfmmFamily && !qfmmKnobs.empty())
    {
        const std::string verb = qfmmKnobs.size() == 1 ? " is set on a run" : " are set on a run";
        const std::string keyWord = qfmmKnobs.size() == 1 ? "the key" : "the keys";

        return std::unexpected(
            Err(qcx::ErrorCode::kUnimplemented,
                JoinKeyNames(qfmmKnobs) + verb +
                    " whose fock_builder is not \"qfmm\": this run "
                    "is on " +
                    FamilyNameForRefusal(input) +
                    ". The four QFMM knobs (theta, l_mult, max_leaf_size, "
                    "crossover_basis_function_count) are consumed by fock_builder = \"qfmm\" "
                    "alone, and no other family in this build reads them, so the request is "
                    "refused rather than run as if the key had not been written - theta is an "
                    "accuracy control (it decides which pairs the near/far split computes "
                    "exactly), and a run that dropped it silently would compute at a screening "
                    "the document does not ask for. Use fock_builder = \"qfmm\", or remove " +
                    keyWord));
    }

    // The guess RESOLUTION POINT, and it is the pair's ONE home.
    // `guess.type` and `guess.restart_path` are one request between them, and
    // the value itself is consumed at the solver wiring sites below (the
    // GWH/core start, the SAD fragment start, the restart read) - which is
    // why the parsed enum was never dead code. What was missing is the place
    // that answers for the PAIR: a restart_path written beside any other guess
    // word was stored (`io` keeps what the file says, by its own contract) and
    // then read by nothing, so the input named a checkpoint file, no path
    // opened it, and the run record could not say so. That is the knob-
    // conformance defect exactly - a key a user can write that the code
    // neither honours, refuses by name, nor demotes with a disclosure - and a
    // refusal by name is its closure, because the request is well-formed and a
    // demotion has nothing to disclose: the run works, the key does not.
    if (auto guess = ResolveGuess(input); !guess.has_value())
    {
        return std::unexpected(guess.error());
    }

    if (!isUnrestricted && input.guess == qcx::io::GuessKind::kSad)
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                   "the SAD guess needs the per-element atomic fragment "
                                   "inputs only the UHF path builds in v1; RHF supports "
                                   "guess core, gwh and restart (gwh is also the RHF loop's "
                                   "default start)"));
    }

    // The restart read (guess restart) and write (checkpoint_file)
    // paths are RHF-only in v1: the RHF solver is the one whose options
    // carry the ScfRestartState seed, and nothing on the UHF side could
    // consume a checkpoint the driver would otherwise write. Rejected up
    // front so no file is produced that no run can read back.
    if (isUnrestricted && input.guess == qcx::io::GuessKind::kRestart)
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                   "guess restart seeds the RHF solver in v1 (the only "
                                   "solver with a restart-state option); UHF "
                                   "supports guess core, gwh and sad"));
    }

    if (isUnrestricted && !input.scf.checkpointFile.empty())
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                   "scf.checkpoint_file saves the RHF last-iterate state in "
                                   "v1 (the restart read path is RHF-only); UHF "
                                   "runs cannot write a checkpoint nothing could consume"));
    }

    // The ETS-NOCV decomposition's per-spin resolution assumes the
    // closed-shell D_sigma = D/2 (nocv.hpp); a UHF molecular density would
    // silently feed it the wrong spin channels, so the combination is
    // rejected up front like the others.
    if (isUnrestricted && !input.properties.nocvFragments.empty())
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                   "the ETS-NOCV analysis is closed-shell in v1; "
                                   "nocv_fragments cannot be requested on a UHF run"));
    }

    // The NOCV fragment groups index the atom rows as written in the file;
    // an out-of-range index is pure input validation (only the atom count
    // is needed) and must fail before the SCF. MapFragmentsToCanonical
    // re-checks the same bounds when the groups are mapped through the
    // canonicalization after the run.
    for (const auto& group : input.properties.nocvFragments)
    {
        for (const std::size_t index : group)
        {
            if (index >= input.molecule.atoms.size())
            {
                return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                           "nocv_fragments atom index " + std::to_string(index) +
                                               " is out of range (the file lists " +
                                               std::to_string(input.molecule.atoms.size()) +
                                               " atoms)"));
            }
        }
    }

    return {};
}

// Atom table: symbol -> Atom for Molecule::Create (isotopic mass 0 = the
// most-abundant isotope). FindElement validates the symbol itself.
qcx::Result<std::vector<qcx::molecule::Atom>> BuildAtoms(
    const std::vector<qcx::io::RunAtom>& atoms) {
    std::vector<qcx::molecule::Atom> result;
    result.reserve(atoms.size());

    for (const auto& atom : atoms)
    {
        const auto* element = qcx::molecule::FindElement(atom.symbol);

        if (element == nullptr)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "unknown element symbol \"" + atom.symbol + "\""));
        }

        result.push_back(qcx::molecule::Atom{atom.symbol, element->atomicNumber, 0.0});
    }

    return result;
}

qcx::Result<CpuTensor2> MakeCoordinates(const std::vector<qcx::io::RunAtom>& atoms) {
    auto coordinates = CpuTensor2::Create({atoms.size(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        // Straight copy: the parser resolved the file's unit and applied it
        // ONCE (io/parse_input.cpp ParseAtom), so RunAtom coordinates are
        // already the Bohr Molecule::Create wants. A factor here would be a
        // second conversion - the mistake the [molecule] units key exists to
        // make impossible (the 2026-08-26 and 2026-09-15 hunts).
        (*coordinates)(i, 0) = atoms[i].x;
        (*coordinates)(i, 1) = atoms[i].y;
        (*coordinates)(i, 2) = atoms[i].z;
    }

    coordinates->MarkHostDirty();
    return coordinates;
}

// H = T + V, host-canonical.
qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basis) {
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basis);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basis);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();
    return kinetic;
}

// The driver-side per-call Fock-build stats stream (the trace
// side-channel): when [scf] trace_file is set, the main SCF path hands
// every BuildFock call a fresh FockBuildStats and appends one line per
// call to <traceFile>.stats:
//   call=<n> wall=<s> total_wall_ms=<s> nq64=<fp64QuartetCount>
//         nq32=<fp32QuartetCount>
//         np=<significantPairCount> g=<primitiveProductSum>
//         xvol=(nq64+nq32)*exchangeOccupied
//         merge=<mergeWallTime seconds> fpb=<eri intensity>
// total_wall_ms is the CALL's own wall, measured by the SEAM around the
// builder call (one steady_clock span on the calling thread, the density and
// matrix conversions outside it) - NOT FockBuildStats::totalWallTime, and the
// difference is the point: this row family serves the direct machinery, the
// RI-J link and the disk route, and on the RI-J link
// RiJkFockBuilder::BuildFock forwards its stats out-parameter to the NESTED
// exchange-only direct half (ri_engine.cpp), so that field there is the half's
// span under the whole call's name. Reading the field would therefore publish,
// per builder, a number that means something different - the certified-bound-
// zero defect in column form. A reader who is tempted to simplify this to
// stats.totalWallTime should note that the value would then be right for the
// direct machinery and wrong for the RI-J comparison rows, silently. A
// one-call run reads its call wall straight off this column, which is what the
// timing limbs do; the wall= column stays the cumulative stream wall.
// The calibration terms ride the same lines:
// np = the call's distinct significant shell pairs (term P), g = the
// call's primitive-product sum over significant quartets (term G), and
// xvol = the call's exchange-contraction volume Q*O_occ - the call's
// screened quartet count times the occupied count of its K channel (0 for
// a call that contracts no exchange: the UHF J-only call; the RHF fused
// and the RI-J exchange-half calls contract K against the closed shell's
// ElectronCount()/2 spatial orbitals; the UHF exchange calls against
// their spin's nAlpha/nBeta). The wall column is steady_clock seconds
// since this stream opened (the scf trace lines use their own loop-start
// epoch; the join tool estimates the launch offset). The increment-1
// measurement bundle rides the line tail: merge = the call's
// merge-chain wall seconds (FockBuildStats::mergeWallTime - the
// canonical-order per-batch shared-Fock delta merges; zero when the
// call's batches folded per index or on the class-aware path, whose time
// fields stay zero by contract), and fpb = the ERI phase's
// compute-vs-memory intensity, g over the ERI-values lane traffic at the
// per-quartet unit-element convention (8 B per fp64-lane quartet, 4 B
// values + the exact 8 B bound per fp32-lane quartet): a Roofline-style
// trend instrument, deliberately uncalibrated for the per-quartet block
// sizes.
//
// The LEAN row family (the stats-out, the nBasis <= 1000 auto
// branch): call=<n> wall=<s> nq64=<screened-in quartets> nq32=0
//       xvol=<nq64 * exchangeOccupied> total_wall_ms=<s> eri_wall_ms=<s>
//       eri_prep_wall_ms=<s> contract_wall_ms=<s> batch_count=<n> -
// exactly the columns the lean builder's per-call stats can back (nq32 at
// its true zero: no certified lane on the fp64-only recompute path).
// total_wall_ms is the CALL's own wall - one steady_clock span around the
// whole BuildFock call on the calling thread
// (FockBuildStats::totalWallTime) - so a single-Fock-call run yields it
// directly instead of by differencing the cumulative wall column, whose
// gap to it on the row is the stream's pre-call setup; it stays
// wall-comparable at every k. The
// split-probe tail (the split-probe instrument) is the lean path's
// phase accounting: eri_wall_ms covers each flush's block-compute span
// (pair data, batch assembly, the RunBatches kernel dispatch) and
// contract_wall_ms its per-quartet contraction loop, batch_count the
// assembled class batches run. eri_prep_wall_ms is the ERI span's
// sub-boundary - its pre-kernel part (pair data, assembly, tile setup) -
// so eri_wall_ms - eri_prep_wall_ms is the kernel time proper; it is a
// sub-span of eri_wall_ms, never added to it. The prep probe splits
// that sub-boundary into its three named parts -
// eri_prep_pair_ms (the flush's chunk pair data: the not-built-marker walk
// over the flush's listed pairs plus one contracted transform per pair it
// builds - read against pair_builds for the per-build cost),
// eri_prep_assemble_ms (the class-batch assembly) and eri_prep_tile_ms
// (the tile sizing, the value arena, the batch copies) - which sum to at
// most eri_prep_wall_ms. pair_builds=<n> is the call's pair-data BUILD
// count - the pairs the call's chunk pair-data passes actually built, one
// per pair per window while the flush retention holds, so the column reads
// the built volume and not the flush re-cutting's listed one (the
// pair-proportional denominator of the pair part above). contract_block_calls=<n>
// is the ACCUMULATION-side denominator (the mechanism-value audit,
// 2026-09-13): one per ContractBlock call - per assembled class task, plus
// one per surviving member cell when the orbit expansion is engaged - so
// read against nq64 it gives the evaluation-to-accumulation ratio in the
// run instead of by reconstruction. On c8h18/def2-SVP the expansion takes
// nq64 from 8,811,481 to 2,379,972 while this column is invariant at
// 8,811,481. The window probe adds the per-window spread columns when the
// seam handed a
// per-window record: window_count=<n>, window_wall_max_ms /
// window_wall_min_ms / window_wall_mean_ms over the call's row windows -
// the per-window reading. The windows
// oversubscribe the team (an auto count plans 8 x the OpenMP team, clamped
// to the row count) and run on the dynamic schedule at a chunk of one
// window, so the spread is the row-cost variance the schedule averages out
// over the region, not a thread's own share; each entry is a true wall
// span of its window (they may overlap at k > 1), and together with
// window_count they say how the work was cut, never that a slow window IS
// the region's wall. The kernel-span probe (the instrument the split-probe
// lever ranking is decided on) subdivides that kernel time in the SAME row:
// kernel_vrr_wall_ms (the VRR/Boys recurrence, one pass-1 group's one ket
// primitive pair), kernel_ket_wall_ms (the group's batched ket transform)
// and kernel_bra_wall_ms (the whole pass-2 task loop) are disjoint
// sub-spans, so their sum is at most eri_wall_ms - eri_prep_wall_ms and the
// remainder belongs to no phase; kernel_vrr_quads, kernel_prim_passes,
// kernel_groups and kernel_gate_seam_calls denominate them (the
// primitive-pair incidence, the ket transform call count, the pass-1 group
// census, and the transform calls that missed the micro-gate shape test -
// the ket calls are kernel_prim_passes and the bra calls nq64, so the
// eligible share is derivable). They are timed inside the class kernels at
// a granularity of one primitive pair per group, never per quartet, and
// they follow the family's k = 1 / k > 1 convention like every other span
// here. np/g (the calibration
// terms), merge and fpb stay ABSENT
// from the lean rows rather than reading as false zeros: the lean path
// counts no pair/primitive calibration volumes and has no merge chain,
// and the harness's stats parser is presence-driven (an absent column is
// recorded as absent). The probe tail is presence-driven the same way: a
// builder that does not fill the split (or a seam that hands no
// per-window record) simply omits its columns.
//
// The QFMM row family (the composed QFMM Coulomb + exchange-only direct
// halves): call=<n> wall=<s> total_wall_ms=<s> far_pairs=<n> - the columns
// the composed builder can honestly back, and no others. It exposes NO
// per-call FockBuildStats at all: its Coulomb half is the QFMM near/far
// split, whose far-field multipole passes (the density-weighted leaf
// moments, M2M, M2L + L2L, the 2J accumulation) have no quartet
// representation, so nq64/nq32/np/g/xvol/merge/fpb and the lean-backed
// probe columns stay ABSENT rather than reading as false zeros (the row
// family's presence-driven convention above). total_wall_ms is the seam's
// own steady_clock span around the whole composed BuildFock call on the
// calling thread - the same whole-call quantity the lean row's
// total_wall_ms is - so a single-call run (the tournament's measurement
// shape) reads its call wall straight off the row. far_pairs is the
// composed builder's far-field pair-pair count
// (QfmmHfFockBuilder::FarFieldPairCount, the liveness flag): a
// run-level constant on every row, zero exactly when the multipole far
// field degenerated to the near-field direct build (the vacuous-ladder
// liveness trap).
//
// A stream that never opens - trace_file empty - keeps the seams'
// zero-cost path untouched (the bit-parity pins are absolute). A
// stream that opens against a builder kind that backs no per-call stats
// AT ALL (the GPU and gpu_split families) stays empty by that kind's own
// contract: zero rows there is the requested trace's honest content, not a
// wiring defect. The reporting set is pinned by
// DriverRunTest.TraceEnabledRunsEmitTheRowFamilyOfTheirBuilder.
class ScfCallStatsStream {
public:
    void Open(const std::string& traceFile) {
        if (!traceFile.empty())
        {
            _epoch = std::chrono::steady_clock::now();
            _stream.open(traceFile + ".stats", std::ios::trunc);
        }
    }

    /// The number of per-call rows this stream has written - 0 on a stream
    /// that never opened, and 0 on a stream opened against a builder kind
    /// that backs no per-call stats at all. A caller reads it after the SCF
    /// to tell a traced run from a silently empty trace (WarnOnEmptyTrace).
    /// \returns The rows written so far.
    long long CallCount() const noexcept {
        return _callCount;
    }

    void Record(const qcx::integrals::FockBuildStats& stats,
                std::size_t exchangeOccupied,
                std::chrono::nanoseconds totalWall) {
        if (!_stream.is_open())
        {
            return;
        }

        ++_callCount;
        const double wallSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - _epoch).count();
        const std::size_t significantQuartets = stats.fp64QuartetCount + stats.fp32QuartetCount;
        const double mergeSeconds = std::chrono::duration<double>(stats.mergeWallTime).count();
        // The fpb column (see the class comment): g over the lane traffic
        // at the per-quartet unit-element convention - 8 B per fp64-lane
        // quartet (one fp64 element) and the certified lane's 4 B fp32
        // element plus its exact 8 B per-quartet bound per fp32-lane
        // quartet. Zero when the call screened no quartets.
        const std::size_t laneTrafficBytes =
            8 * stats.fp64QuartetCount + 12 * stats.fp32QuartetCount;
        const double eriIntensity = laneTrafficBytes != 0
                                        ? static_cast<double>(stats.primitiveProductSum) /
                                              static_cast<double>(laneTrafficBytes)
                                        : 0.0;
        const double totalMillis = std::chrono::duration<double, std::milli>(totalWall).count();
        _stream << "call=" << _callCount << " wall=" << std::fixed << std::setprecision(6)
                << wallSeconds << std::defaultfloat << " total_wall_ms=" << std::fixed
                << std::setprecision(3) << totalMillis << std::defaultfloat
                << " nq64=" << stats.fp64QuartetCount << " nq32=" << stats.fp32QuartetCount
                << " np=" << stats.significantPairCount << " g=" << stats.primitiveProductSum
                << " xvol=" << significantQuartets * exchangeOccupied << " merge=" << std::fixed
                << std::setprecision(6) << mergeSeconds << std::defaultfloat
                << " fpb=" << eriIntensity << '\n';
        _stream.flush();
    }

    // The lean row family (see the class comment): the columns the lean
    // builder's per-call stats can back only - nq64 (the call's
    // screened-in shell-quartet count) and the true nq32 zero (no
    // certified lane on the recompute path) plus xvol, and the split-probe
    // split-probe tail: eri_wall_ms / eri_prep_wall_ms (the
    // call's block-compute span and its pre-kernel part) / contract_wall_ms
    // (the contraction span) and batch_count (the assembled class batches
    // run). The prep probe splits the pre-kernel part into
    // eri_prep_pair_ms / eri_prep_assemble_ms / eri_prep_tile_ms and adds
    // pair_builds (the call's pair-data BUILD count - the pairs its chunk
    // pair-data passes built, the flush retention making it one per pair
    // per window rather than the far larger listed volume) and
    // total_wall_ms (the
    // call's own whole-call span - the single-call number the re-scoped
    // gate reads off a one-call run, where the cumulative wall column has
    // no delta to give). The window
    // probe adds the per-window spread (window_count plus the
    // window_wall_max/min/mean_ms) when the caller handed a per-window
    // record. np/g/merge/fpb are not formed (the
    // lean path counts no calibration volumes and has no merge chain);
    // their absence keeps the row honest instead of reading as false
    // zeros, the same convention as the probe tail's own presence.
    void RecordLean(const qcx::integrals::FockBuildStats& stats, std::size_t exchangeOccupied) {
        if (!_stream.is_open())
        {
            return;
        }

        ++_callCount;
        const double wallSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - _epoch).count();
        const std::size_t screenedQuartets = stats.fp64QuartetCount;
        // The call's own wall (the single-Fock-call timing seam): one span
        // around the whole call, so it needs no differencing of the
        // cumulative wall column - the first row carries it too.
        const double totalMillis =
            std::chrono::duration<double, std::milli>(stats.totalWallTime).count();
        // The split-probe tail (the split-probe instrument): the two lean
        // phase spans in milliseconds - at k = 1 true wall spans whose sum
        // with the row's wall delta leaves the screen/reduce residual, at
        // k > 1 window-accumulated totals (the FockBuildStats convention).
        const double eriMillis =
            std::chrono::duration<double, std::milli>(stats.eriWallTime).count();
        const double eriPrepMillis =
            std::chrono::duration<double, std::milli>(stats.eriPrepWallTime).count();
        const double prepPairMillis =
            std::chrono::duration<double, std::milli>(stats.eriPrepPairDataWallTime).count();
        const double prepAssembleMillis =
            std::chrono::duration<double, std::milli>(stats.eriPrepAssembleWallTime).count();
        const double prepTileMillis =
            std::chrono::duration<double, std::milli>(stats.eriPrepTileSetupWallTime).count();
        const double contractMillis =
            std::chrono::duration<double, std::milli>(stats.contractWallTime).count();
        // The kernel-span probe (the kernel-span instrument): the three disjoint
        // sub-spans of the kernel time (eriWallTime - eriPrepWallTime) and
        // their denominators. The phases are timed inside the class kernels
        // at one primitive pair per group and once around the pass-2 bra
        // loop, so their sum leaves a residual against the kernel span (the
        // per-batch layout walk, scratch.assign, the group splitting and
        // the loop overheads) rather than closing it exactly - the same
        // partial-coverage contract the prep split already has. Their k > 1
        // convention is the file's: window-accumulated totals, not wall
        // spans.
        const double kernelVrrMillis =
            std::chrono::duration<double, std::milli>(stats.kernelVrrWallTime).count();
        const double kernelKetMillis =
            std::chrono::duration<double, std::milli>(stats.kernelKetWallTime).count();
        const double kernelBraMillis =
            std::chrono::duration<double, std::milli>(stats.kernelBraWallTime).count();
        // The per-window family (the window probe): the caller's
        // per-window record reduced to the spread columns - the windows'
        // own wall spread read straight off the row. Present only when the
        // caller handed a record (null omits all four, the row family's
        // presence-driven convention); the record is never empty when it
        // was handed over (one entry per window).
        const std::vector<std::chrono::nanoseconds>* windowTimes = stats.windowWallTimesOut;
        const bool haveWindows = windowTimes != nullptr && !windowTimes->empty();
        double windowMaxMillis = 0.0;
        double windowMinMillis = 0.0;
        double windowMeanMillis = 0.0;

        if (haveWindows)
        {
            long long totalNanos = 0;
            long long maxNanos = 0;
            long long minNanos = windowTimes->front().count();

            for (const std::chrono::nanoseconds entry : *windowTimes)
            {
                const long long nanos = entry.count();
                totalNanos += nanos;
                maxNanos = std::max(maxNanos, nanos);
                minNanos = std::min(minNanos, nanos);
            }

            windowMaxMillis = static_cast<double>(maxNanos) / 1.0e6;
            windowMinMillis = static_cast<double>(minNanos) / 1.0e6;
            windowMeanMillis =
                static_cast<double>(totalNanos) / static_cast<double>(windowTimes->size()) / 1.0e6;
        }

        _stream << "call=" << _callCount << " wall=" << std::fixed << std::setprecision(6)
                << wallSeconds << std::defaultfloat << " nq64=" << screenedQuartets
                << " nq32=0 xvol=" << screenedQuartets * exchangeOccupied
                << " total_wall_ms=" << std::fixed << std::setprecision(3) << totalMillis
                << " eri_wall_ms=" << eriMillis << " eri_prep_wall_ms=" << eriPrepMillis
                << " eri_prep_pair_ms=" << prepPairMillis
                << " eri_prep_assemble_ms=" << prepAssembleMillis
                << " eri_prep_tile_ms=" << prepTileMillis << " contract_wall_ms=" << contractMillis
                << " kernel_vrr_wall_ms=" << kernelVrrMillis
                << " kernel_ket_wall_ms=" << kernelKetMillis
                << " kernel_bra_wall_ms=" << kernelBraMillis << std::defaultfloat
                << " batch_count=" << stats.batchCount << " pair_builds=" << stats.pairBuildCount
                << " contract_block_calls=" << stats.contractBlockCalls
                << " kernel_vrr_quads=" << stats.kernelVrrQuadruples
                << " kernel_gate_seam_calls=" << stats.kernelGateSeamCalls
                << " kernel_groups=" << stats.kernelGroupCount
                << " kernel_prim_passes=" << stats.kernelPrimPasses;

        if (haveWindows)
        {
            _stream << std::fixed << std::setprecision(3) << " window_count=" << windowTimes->size()
                    << " window_wall_max_ms=" << windowMaxMillis
                    << " window_wall_min_ms=" << windowMinMillis
                    << " window_wall_mean_ms=" << windowMeanMillis << std::defaultfloat;
        }

        _stream << '\n';
        _stream.flush();
    }

    // The QFMM row family (see the class comment): the columns the composed
    // QFMM builder can honestly back and nothing more - call / wall (the
    // stream's own clock, like every family's row), total_wall_ms (the
    // seam's whole-call span) and far_pairs (the composed builder's
    // far-field pair-pair count, the liveness flag; a run-level
    // constant). totalWall is the span the seam measured around the
    // composed BuildFock call, so it is the call's own wall on the calling
    // thread at every k. No quartet counter is formed: the composed
    // builder exposes none, and the far field's multipole passes have no
    // quartet representation - omitting the columns keeps the row honest
    // instead of reading as false zeros, the family convention above.
    void RecordQfmm(std::chrono::nanoseconds totalWall, std::size_t farFieldPairs) {
        if (!_stream.is_open())
        {
            return;
        }

        ++_callCount;
        const double wallSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - _epoch).count();
        const double totalMillis = std::chrono::duration<double, std::milli>(totalWall).count();
        _stream << "call=" << _callCount << " wall=" << std::fixed << std::setprecision(6)
                << wallSeconds << std::defaultfloat << " total_wall_ms=" << std::fixed
                << std::setprecision(3) << totalMillis << std::defaultfloat
                << " far_pairs=" << farFieldPairs << '\n';
        _stream.flush();
    }

private:
    std::ofstream _stream;
    std::chrono::steady_clock::time_point _epoch{};
    long long _callCount = 0;
};

// The silent-empty-trace guard: a run asked for a per-call trace
// ([scf] trace_file) that recorded ZERO rows says so loudly, right where
// the trace's scope closes. An empty <trace_file>.stats is legitimate ONLY
// for a builder kind that backs no per-call stats at all (the GPU and
// gpu_split families); every other kind's seam records one row per
// BuildFock call, so zero rows there is a wiring defect - and without this
// guard it is indistinguishable from success, because the run exits 0
// either way (the qfmm cells of the defaults tournament,
// window-20260911-063141Z: exit 0, 0 parseable rows, no diagnostic, and a
// timing limb that silently could not be measured). The warning uses the
// qcx: warning: form the selection divergence uses, so a harness can grep
// it; it never fails the run - the trace is a side-channel and the run's
// physics is unaffected.
void WarnOnEmptyTrace(const qcx::io::RunInput& input,
                      const ScfCallStatsStream& statsStream,
                      qcx::io::BuilderKind kind) {
    if (input.scf.traceFile.empty() || statsStream.CallCount() != 0)
    {
        return;
    }

    const std::string_view family = qcx::io::ToString(kind);
    std::fprintf(stderr,
                 "qcx: warning: [scf] trace_file was set but the run recorded 0 per-call "
                 "rows: the wired builder family \"%.*s\" backs no per-call stats sink, so no "
                 "timing limb can be measured from this run\n",
                 static_cast<int>(family.size()),
                 family.data());
}

// The RHF seam hands over the spin-summed density D; the integrals builders
// contract the spatial rho = D/2 (rhf.hpp documents the adapter scaling).
// Templated over the builder type: the direct/RI/QFMM/GPU builders all
// expose the same BuildFock(Tensor) const -> Result<Tensor> contract.
// The optional stats sink (null = off) collects the per-call quartet
// counts of the builders that expose a stats-out parameter (the direct
// family's BuildFock(density, boundSumOut, statsOut, ...) and the RI-J
// builder's BuildFock(density, statsOut)); the others fall through to the
// plain call with no collection. exchangeOccupied is the call's K-channel
// occupancy (the RHF run's ElectronCount()/2 spatial orbitals - the
// direct and RI-J exchange halves contract K against it); it sizes the
// The Q*O_occ calibration volume on the recorded lines. Zero when no
// stats sink is wired (the GPU seam), where it is never read. The QFMM
// family does NOT ride this template: its composed builder exposes no
// FockBuildStats at all, so it takes MakeQfmmRhfFockBuilder's own row
// family rather than falling through to the plain call's silent
// no-collection path.
//
// The optional certified-bound accumulator (null = off) is the
// observability seam: the builder's certifiedBoundSumOut out-parameter is
// the fp32 lane's density-weighted kernel-bound sum for one call - the
// quantity the driver discarded before . Only the direct
// family's machinery members expose the three-argument form, so the
// `requires` branch that passes the pointer is the same branch that must
// OBSERVE the result: the fold into the accumulator lives INSIDE it, and
// the lean member, the RI-J/QFMM/GPU families and the disk route - every
// builder taking a shorter BuildFock - never reach it. That gating is the
// schema's contract, not a tidiness choice: those builders never write
// the out-parameter, so observing outside the branch would fold
// `callBound`'s 0.0 initializer as if it were a measurement and publish a
// fabricated certified zero on exactly the routes that cannot certify
// anything (the RI-J link computes a real bound in its nested exchange
// half and drops it, which makes the fabrication a denial of a computed
// quantity). The builder computes the sum regardless of the pointer (the
// accumulator is its own local), so collecting it costs one write per
// call and cannot move a single Fock element.
template <typename Builder>
qcx::scf::FockBuilderFn MakeRhfFockBuilder(const Builder& builder,
                                           ScfCallStatsStream* statsSink = nullptr,
                                           std::size_t exchangeOccupied = 0,
                                           CertifiedBoundAccumulator* boundAccum = nullptr) {
    return [builder, statsSink, exchangeOccupied, boundAccum](
               const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto rho = ToTensor(0.5 * density);

        if (!rho.has_value())
        {
            return std::unexpected(rho.error());
        }

        qcx::integrals::FockBuildStats callStats;
        qcx::integrals::FockBuildStats* statsOut = statsSink != nullptr ? &callStats : nullptr;
        double callBound = 0.0;
        qcx::integrals::CertifiedBudgetOutcome callBudget;
        // The call's own wall span, taken by the SEAM and recorded as the row's
        // total_wall_ms - NOT read from FockBuildStats::totalWallTime, because
        // that field means different things per builder on this row family: on
        // the RI-J link RiJkFockBuilder::BuildFock forwards statsOut to its
        // NESTED exchange-only direct half (ri_engine.cpp), so the field there
        // is the exchange half's span and not the call's. A column whose
        // meaning changes with the builder is the certified-bound-zero defect
        // again (a number published under a name that does not mean what the
        // name says), so the span below is measured HERE, around the dispatch,
        // for every builder that reaches this seam - the same window
        // RecordQfmm's own span uses, with the density conversion above and
        // the matrix conversion below both outside it. The null sink reads no
        // clock at all, keeping the documented zero-cost path exact.
        const auto callStarted = statsSink != nullptr ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        const auto fockResult = [&]() -> qcx::Result<CpuTensor2> {
            if constexpr (requires { builder.BuildFock(*rho, nullptr, statsOut); })
            {
                auto fock = builder.BuildFock(*rho,
                                              boundAccum != nullptr ? &callBound : nullptr,
                                              statsOut,
                                              nullptr,
                                              boundAccum != nullptr ? &callBudget : nullptr);

                // The fold is gated by the same `requires` test that chose
                // this branch: only a builder that was actually handed the
                // out-parameter can report a bound, so no shorter-form
                // builder's never-written 0.0 ever reaches the accumulator
                // as an observed zero (see this function's header comment).
                // The budget outcome rides the same fold: a builder that did
                // not run the enforcement leaves it at its zero defaults,
                // whose `enforced` flag is what keeps them out of the
                // record as measurements.
                if (fock.has_value() && boundAccum != nullptr)
                {
                    boundAccum->Observe(callBound, callBudget);
                }

                return fock;
            } else if constexpr (requires { builder.BuildFock(*rho, statsOut); })
            {
                return builder.BuildFock(*rho, statsOut);
            } else
            {
                return builder.BuildFock(*rho);
            }
        }();
        const auto callWall = statsSink != nullptr ? std::chrono::steady_clock::now() - callStarted
                                                   : std::chrono::nanoseconds{};

        if (!fockResult.has_value())
        {
            return std::unexpected(fockResult.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->Record(callStats, exchangeOccupied, callWall);
        }

        return ToMatrix(*fockResult);
    };
}

// The composed full-RI exchange seam (the composed full-RI builder). The
// builder's BuildFock takes the OCCUPIED BLOCK beside the density, and the SCF
// seam it has to fit (scf/rhf.hpp FockBuilderFn) is DENSITY-ONLY - so this seam
// derives the block from the density it is handed.
//
// Why deriving it is exact rather than a compromise. K_RI is contracted as
// sum_P (B^P C_occ)(B^P C_occ)^T, which equals sum_P B^P (C_occ C_occ^T) B^P
// exactly for ANY block C_occ - the identity verified at 4.4e-16, and the
// one RiFullFockBuilder's own header states. The exchange half
// therefore depends on the block only THROUGH rho = C_occ C_occ^T, so any
// factor of rho rebuilds the same K: the block is a factorization of the
// density, not an independent input.
//
// What that buys, and why it is the smallest reading of an unruled choice: a
// RESUMED run never has the loop's own orbitals to hand over - ScfRestartState
// stores densities and DIIS histories and no orbital block - so a seam that took
// the block from its caller would have to widen that state and its checkpoint
// schema with it. Deriving the block from the density makes the resumed and the
// fresh run take the SAME path and leaves scf/ untouched. The alternative route
// (widening FockBuilderFn, and the restart state behind it) is the larger one
// and is recorded as owed to the owner rather than taken here.
//
// The class this lands in, stated rather than implied: for a density that IS the
// occupied projector - rho = C_occ C_occ^T, which is what the RHF loop's own
// iterate density is - the recovery reproduces the occupied space exactly, and
// the exchange contracts it at the builder's own accuracy. A DIIS-extrapolated
// density is not a projector (it is not C C^T for any C of the loop's own), so
// the recovered block is the closest rank-nOccupied factor of it and the
// contraction reads that factor. That is the one place this seam can differ from
// handing the loop's orbitals over, and the run record carries the
// approximation statement so a reader of the energy knows which it is.
// \param builder The created full-RI builder. Held by value: every copy shares
// the one transformed tensor (shared state), so no per-call rebuild.
// \param nOccupied The occupied count - the rank of the density this seam is
// handed, ElectronCount()/2 for the closed-shell case.
// \param statsSink The optional call-stats stream, recorded per call on the same
// terms MakeRhfFockBuilder records it (the seam measures the span, never the
// builder's own totalWallTime).
// \returns The seam: density in, F = H + 2 J_RI - K_RI out.
qcx::scf::FockBuilderFn MakeRiFullRhfFockBuilder(const qcx::integrals::RiFullFockBuilder& builder,
                                                 std::size_t nOccupied,
                                                 ScfCallStatsStream* statsSink = nullptr) {
    return [builder, nOccupied, statsSink](
               const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        // The convention MakeRhfFockBuilder applies, and the one the builder's
        // own contract states: the spatial closed-shell density rho = D/2.
        const Eigen::MatrixXd rho = 0.5 * density;
        const Eigen::Index n = rho.rows();

        if (nOccupied == 0 || static_cast<Eigen::Index>(nOccupied) > n)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "the full-RI exchange seam needs an occupied count in "
                                       "[1, nBasis]; got " +
                                           std::to_string(nOccupied) + " for a density with " +
                                           std::to_string(n) + " rows"));
        }

        // rho is symmetric and positive semi-definite by construction (a density
        // over real orbitals), so the self-adjoint solver is the right one and
        // its eigenvalues come back ASCENDING: the occupied space is the top
        // nOccupied of them.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(rho);

        if (solver.info() != Eigen::Success)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                       "the full-RI exchange seam could not diagonalize the "
                                       "density it was handed"));
        }

        // The clamp is numerical, not physical: a PSD matrix can come back with
        // eigenvalues a few ulps below zero and the square root of a negative
        // one is NaN. Those eigenvalues belong to the virtual space, so nothing
        // physical rides on the clamp.
        const Eigen::Index nOcc = static_cast<Eigen::Index>(nOccupied);
        const Eigen::VectorXd occupiedValues = solver.eigenvalues().tail(nOcc).cwiseMax(0.0);
        const Eigen::MatrixXd occupiedBlock =
            solver.eigenvectors().rightCols(nOcc) * occupiedValues.cwiseSqrt().asDiagonal();

        auto rhoTensor = ToTensor(rho);

        if (!rhoTensor.has_value())
        {
            return std::unexpected(rhoTensor.error());
        }

        qcx::integrals::FockBuildStats callStats;
        qcx::integrals::FockBuildStats* statsOut = statsSink != nullptr ? &callStats : nullptr;
        const auto callStarted = statsSink != nullptr ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        auto fock = builder.BuildFock(*rhoTensor, occupiedBlock, statsOut);
        const auto callWall = statsSink != nullptr ? std::chrono::steady_clock::now() - callStarted
                                                   : std::chrono::nanoseconds{};

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->Record(callStats, nOccupied, callWall);
        }

        return ToMatrix(*fock);
    };
}

// The lean seam (the stats-out): the same per-call collection shape
// as MakeRhfFockBuilder, but for the lean builder's stats-carrying
// BuildFock (the fp64-only recompute path fills fp64QuartetCount,
// batchCount, the pair-build count, the eri/contract split - with the
// prep's three-part split - the per-window record and totalWallTime of
// the carrier)
// and with the lean ROW family on the stream (RecordLean - the columns
// the lean stats can back; the machinery-path columns np/g/merge/fpb stay
// absent, never false zeros). The seam hands the builder its per-window
// record (the window probe's storage - the builder owns none).
// Null sink keeps the plain seam (no collection, no row). exchangeOccupied
// sizes the lean rows' xvol (the Q*O_occ exchange volume), the same
// K-channel occupancy as the counted machinery seams.
qcx::scf::FockBuilderFn MakeLeanRhfFockBuilder(const qcx::integrals::LeanDirectFockBuilder& builder,
                                               ScfCallStatsStream* statsSink,
                                               std::size_t exchangeOccupied) {
    return [builder, statsSink, exchangeOccupied](
               const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto rho = ToTensor(0.5 * density);

        if (!rho.has_value())
        {
            return std::unexpected(rho.error());
        }

        qcx::integrals::FockBuildStats callStats;
        // The window probe: the seam's own per-window record, refilled by
        // the stats-carrying call (one entry per row window) and read by
        // RecordLean for the row's spread columns. Only lives when a stats
        // sink is wired - the null sink keeps the plain call untouched.
        std::vector<std::chrono::nanoseconds> windowTimes;
        callStats.windowWallTimesOut = statsSink != nullptr ? &windowTimes : nullptr;
        auto fock = builder.BuildFock(*rho, statsSink != nullptr ? &callStats : nullptr);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->RecordLean(callStats, exchangeOccupied);
        }

        return ToMatrix(*fock);
    };
}

// The QFMM seam (the composed QFMM Coulomb + exchange-only direct halves of
// QfmmHfFockBuilder): the same per-call row contract as the lean seam, over
// the one family whose builder exposes no FockBuildStats at all. The
// composed builder takes a density and returns the Fock matrix - its
// Coulomb half runs the QFMM near/far split (the far field's multipole
// passes carry no quartet count) - so the seam measures what only the
// caller can: one steady_clock span around the whole composed BuildFock
// call, recorded as the QFMM row family (RecordQfmm) together with the
// builder's far-field pair-pair liveness (QfmmHfFockBuilder::
// FarFieldPairCount - the liveness flag; zero means the far field degenerated to
// the near-field direct build). The row carries no quartet counter rather
// than a fabricated zero. Null sink keeps the plain seam (no timing, no
// row); the timing span is taken on the calling thread either way, so the
// recorded wall is the call's own at every k.
qcx::scf::FockBuilderFn MakeQfmmRhfFockBuilder(const qcx::integrals::QfmmHfFockBuilder& builder,
                                               ScfCallStatsStream* statsSink) {
    return [builder, statsSink](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto rho = ToTensor(0.5 * density);

        if (!rho.has_value())
        {
            return std::unexpected(rho.error());
        }

        // The QFMM row's whole-call span: the builder's own call only, the
        // lean seam's totalWallTime window (the density conversion above
        // stays outside it).
        const auto callStarted = std::chrono::steady_clock::now();
        auto fock = builder.BuildFock(*rho);
        const auto callStopped = std::chrono::steady_clock::now();

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->RecordQfmm(callStopped - callStarted, builder.FarFieldPairCount());
        }

        return ToMatrix(*fock);
    };
}

// The shared per-spin assembly of the TWO UHF seams - the direct family's
// machinery member and its lean member (the unrestricted seam). The seam hands over
// the actual per-spin densities; the Coulomb half runs J-only on the
// half-summed density rho = 0.5 (P_a + P_b) (the J loop bakes in the 2.0
// factor) and the exchange half runs K-only per spin; each split result
// carries one full H copy, so one is subtracted back per channel (the
// double-H trap; direct_uhf_test.cpp is the reference assembly).
//
// The two seams differ ONLY in which builder runs a half and how a half's
// per-call stats are collected - the machinery member's BuildFock takes
// the certified-bound and stats out-parameters and records the machinery
// row family, the lean member's takes the stats pointer alone and records
// the lean row family - so the half-build is injected as one callable and
// the arithmetic below is shared by construction rather than copied (the
// one-formula discipline the lean envelope's single refusal site follows:
// two copies of this assembly could drift, one cannot).
//
// \param coulombBuilder The J-only half's builder (buildCoulombOnly).
// \param exchangeBuilder The K-only half's builder (buildExchangeOnly).
// \param coreHamiltonian H, subtracted back once per channel.
// \param exchangeAlphaCount The alpha exchange call's K-channel occupancy
// (its xvol); the beta call's count travels per call below.
// \param exchangeBetaCount The beta exchange call's K-channel occupancy.
// \param buildHalf The per-half call: builds one half's Fock matrix from a
// host-canonical density, records its stats under the exchange occupancy
// it is handed (0 for the J-only half), and returns it.
template <typename Builder, typename BuildHalf>
qcx::scf::UhfFockBuilderFn MakeUhfSplitFockBuilder(const Builder& coulombBuilder,
                                                   const Builder& exchangeBuilder,
                                                   const Eigen::MatrixXd& coreHamiltonian,
                                                   std::size_t exchangeAlphaCount,
                                                   std::size_t exchangeBetaCount,
                                                   BuildHalf buildHalf) {
    return [coulombBuilder,
            exchangeBuilder,
            coreHamiltonian,
            exchangeAlphaCount,
            exchangeBetaCount,
            buildHalf](const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)
               -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        auto dTotalHalf = ToTensor(0.5 * (dAlpha + dBeta));

        if (!dTotalHalf.has_value())
        {
            return std::unexpected(dTotalHalf.error());
        }

        auto coulomb = buildHalf(coulombBuilder, *dTotalHalf, 0);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        auto dAlphaTensor = ToTensor(dAlpha);

        if (!dAlphaTensor.has_value())
        {
            return std::unexpected(dAlphaTensor.error());
        }

        auto exchangeAlpha = buildHalf(exchangeBuilder, *dAlphaTensor, exchangeAlphaCount);

        if (!exchangeAlpha.has_value())
        {
            return std::unexpected(exchangeAlpha.error());
        }

        auto dBetaTensor = ToTensor(dBeta);

        if (!dBetaTensor.has_value())
        {
            return std::unexpected(dBetaTensor.error());
        }

        auto exchangeBeta = buildHalf(exchangeBuilder, *dBetaTensor, exchangeBetaCount);

        if (!exchangeBeta.has_value())
        {
            return std::unexpected(exchangeBeta.error());
        }

        const Eigen::MatrixXd fAlpha =
            ToMatrix(*coulomb) + ToMatrix(*exchangeAlpha) - coreHamiltonian;
        const Eigen::MatrixXd fBeta =
            ToMatrix(*coulomb) + ToMatrix(*exchangeBeta) - coreHamiltonian;
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{fAlpha, fBeta};
    };
}

// The direct machinery's UHF seam: the stats stream records the
// three BuildFock calls with their K-channel occupancies (the calibration
// terms): the J-only call contracts no exchange (xvol 0), the
// alpha exchange call contracts against exchangeAlphaCount occupied alpha
// orbitals, the beta call against exchangeBetaCount.
qcx::scf::UhfFockBuilderFn MakeDirectUhfFockBuilder(
    const qcx::integrals::DirectJkFockBuilder& coulombBuilder,
    const qcx::integrals::DirectJkFockBuilder& exchangeBuilder,
    const Eigen::MatrixXd& coreHamiltonian,
    ScfCallStatsStream* statsSink,
    std::size_t exchangeAlphaCount,
    std::size_t exchangeBetaCount) {
    // The per-call stats wrapper: one fresh FockBuildStats per BuildFock
    // call, recorded to the trace stream on success. Null sink keeps the
    // exact zero-cost path of the plain calls.
    const auto buildHalf =
        [statsSink](const qcx::integrals::DirectJkFockBuilder& builder,
                    const CpuTensor2& density,
                    const std::size_t exchangeOccupied) -> qcx::Result<CpuTensor2> {
        qcx::integrals::FockBuildStats callStats;
        // The half-call's own wall span, taken by this seam for the same
        // reason the RHF machinery seam takes its own (see the comment
        // there): the builder-backed totalWallTime is not the call's span on
        // every builder this seam can wrap. Null sink reads no clock.
        const auto callStarted = statsSink != nullptr ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        auto fock =
            builder.BuildFock(density, nullptr, statsSink != nullptr ? &callStats : nullptr);
        const auto callWall = statsSink != nullptr ? std::chrono::steady_clock::now() - callStarted
                                                   : std::chrono::nanoseconds{};

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->Record(callStats, exchangeOccupied, callWall);
        }

        return fock;
    };

    return MakeUhfSplitFockBuilder(coulombBuilder,
                                   exchangeBuilder,
                                   coreHamiltonian,
                                   exchangeAlphaCount,
                                   exchangeBetaCount,
                                   buildHalf);
}

// The lean member's UHF seam (the unrestricted seam): the same assembly over two
// lean half-mode builders (buildCoulombOnly / buildExchangeOnly, the
// options LeanFockBuildOptions carries for exactly this adapter). What
// differs from the machinery seam is only what those builders ARE - no
// modeled ladder, no cache, no workspace budget, and the point-group
// reduction carried as their own pair-orbit action rather than as a
// retained class table (LeanFockBuildOptions::symmetryReduction AND
// symmetryOrbitExpansion, set together on both halves from ONE reduction -
// the decision the RHF arm's wiring documents in full, and the UHF arm
// below restates it); the admission is the lean member's
// own Create-time last-resort ceiling, and which trace row family the
// per-call stats reach: the lean rows
// (RecordLean - the columns the lean stats can back, never
// machinery-style false zeros), with the same per-window record the RHF
// lean seam hands over (the window probe). Null sink keeps the plain
// seam: no collection, no row.
qcx::scf::UhfFockBuilderFn MakeLeanUhfFockBuilder(
    const qcx::integrals::LeanDirectFockBuilder& coulombBuilder,
    const qcx::integrals::LeanDirectFockBuilder& exchangeBuilder,
    const Eigen::MatrixXd& coreHamiltonian,
    ScfCallStatsStream* statsSink,
    std::size_t exchangeAlphaCount,
    std::size_t exchangeBetaCount) {
    const auto buildHalf =
        [statsSink](const qcx::integrals::LeanDirectFockBuilder& builder,
                    const CpuTensor2& density,
                    const std::size_t exchangeOccupied) -> qcx::Result<CpuTensor2> {
        qcx::integrals::FockBuildStats callStats;
        // The window probe: the seam's own per-window record, refilled by
        // the stats-carrying call (one entry per row window) and read by
        // RecordLean for the row's spread columns. Only lives when a stats
        // sink is wired.
        std::vector<std::chrono::nanoseconds> windowTimes;
        callStats.windowWallTimesOut = statsSink != nullptr ? &windowTimes : nullptr;
        auto fock = builder.BuildFock(density, statsSink != nullptr ? &callStats : nullptr);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->RecordLean(callStats, exchangeOccupied);
        }

        return fock;
    };

    return MakeUhfSplitFockBuilder(coulombBuilder,
                                   exchangeBuilder,
                                   coreHamiltonian,
                                   exchangeAlphaCount,
                                   exchangeBetaCount,
                                   buildHalf);
}

// ---------------------------------------------------------------------------
// The Kohn-Sham composition the Kohn-Sham composition.
//
// The physics and the per-density cache live in
// driver/src/internal/ks_composition.hpp, which is unit-tested against
// synthetic halves. What stands here is the other half of the wiring: turning
// the run's input into the composition's inputs - the resolved functional and
// its grid engine, and the direct family's two half-mode builders behind the
// composition's HalfFockFn.
// ---------------------------------------------------------------------------

// The run's resolved Kohn-Sham context: the functional the input asked for, the
// XC evaluator that compiles it over the molecule's grid, and the grid itself -
// the engine, the geometry and parameters its derivative provider is built
// from, and the thresholds both walks read (KsGrid).
//
// The engine owns the grid, so it is built ONCE per run and held in a
// shared_ptr the evaluator lambda captures - the evaluator is copied into the
// seam closures, which the SCF loop holds for the run's duration, and the grid
// must outlive every copy of them. The grid rides along beside the evaluator so
// that a gradient walk reads the same engine and the same thresholds rather
// than resolving a second grid of its own.
struct KsContext {
    qcx::driver::internal::KsFunctional functional;
    qcx::driver::internal::XcEvaluatorFn evaluator;
    qcx::driver::internal::KsGrid grid;
};

// The XC grid settings for this run: the parser's resolved `[grid]`
// block carried into the engine's own type, field for field and in the same
// order.
//
// ONE function, called from both places the six numbers are needed - the
// engine's Create and the record's `xc_grid` block - so the grid a run built
// and the grid its JSON discloses cannot be two different grids. It is a
// shape change and not a resolution: the defaults, and the refusals that
// bound them, are the parser's (parse_input.cpp's [grid] block) and this
// function adds nothing to either - which is why the record may name these as
// what the engine was created with rather than as what the file asked for.
// \param grid The input's `[grid]` block as the parser resolved it; an absent
// block is the caller's `value_or(RunGridInput{})`, whose member defaults ARE
// the engine's own.
// \returns The settings for XcGridEngine::Create.
qcx::grid::XcGridSettings ResolveXcGridSettings(const qcx::io::RunGridInput& grid) noexcept {
    return qcx::grid::XcGridSettings{grid.radialPoints,
                                     grid.angularPoints,
                                     grid.alpha,
                                     grid.radialExponent,
                                     grid.trimWeight,
                                     grid.blockTarget};
}

// The record's xc_grid block from the settings the engine was created with -
// the same six numbers, in the record's own shape.
// \param settings The settings XcGridEngine::Create was called with.
// \returns The block for RunResult::xcGrid.
qcx::io::RunXcGrid MakeRunXcGrid(const qcx::grid::XcGridSettings& settings) noexcept {
    return qcx::io::RunXcGrid{settings.radialPoints,
                              settings.angularPoints,
                              settings.alpha,
                              settings.radialExponent,
                              settings.trimWeight,
                              settings.blockTarget};
}

// Resolves the input's `method.functional` and builds the grid engine for it.
//
// The functional name is REQUIRED here even though the io layer only makes it
// permitted: there is no defensible default functional, and a run that
// substituted one would publish an energy under a label the input did not ask
// for - the same posture as the unknown-name refusal, applied to the absent
// case.
// \param input The run input (its method.functional, method.screening_tolerance
// and grid).
// \param molecule The molecule the grid is centered on.
// \param basis The orbital basis the AO tier evaluates.
// \returns The context, or an Error: kInvalidArgument for an absent or
// unknown functional, or whatever XcGridEngine::Create refuses.
qcx::Result<KsContext> ResolveKsContext(const qcx::io::RunInput& input,
                                        const qcx::molecule::Molecule& molecule,
                                        const qcx::basisset::BasisSet& basis) {
    if (!input.method.functional.has_value())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "a Kohn-Sham run needs method.functional: the run is refused rather than "
                "defaulted, because a functional the input did not name is an energy under a "
                "label nobody asked for. Shipped: " +
                    qcx::driver::internal::ShippedFunctionalNames()));
    }

    auto functional = qcx::driver::internal::ResolveKsFunctional(*input.method.functional);

    if (!functional.has_value())
    {
        return std::unexpected(functional.error());
    }

    // The run's grid: the input's `[grid]` block, resolved by the parser,
    // reaches the grid build here and nowhere else - this is the ONE Create
    // for the whole run, whichever builder arm called it. The settings object
    // and the tolerance below go to that build TOGETHER, and the thresholds
    // they produce are what the energy path is handed here and what a gradient
    // walk is handed later: one object, read once, so neither walk can screen
    // by a rule the other did not use.
    const qcx::grid::XcGridSettings settings =
        ResolveXcGridSettings(input.grid.value_or(qcx::io::RunGridInput{}));

    // The tolerance the schema documents: an absent key is
    // kDefaultScreeningTolerance (the engine's measured cheap route), a
    // present one is the author's - 0.0 selecting the dense path, which is the
    // cross-check spelling the key exists for. Neither is this composition's
    // choice to make, so neither is decided here.
    const double tolerance =
        input.method.screeningTolerance.value_or(qcx::io::kDefaultScreeningTolerance);

    auto grid =
        qcx::driver::internal::CreateKsGrid(molecule, basis, functional->name, settings, tolerance);

    if (!grid.has_value())
    {
        return std::unexpected(grid.error());
    }

    KsContext context{std::move(*functional), {}, std::move(*grid)};

    // EvaluateScreened is the entry point used for BOTH lanes. The closed
    // shell passes the spin pair (rho, rho) at rho = D/2, which is exactly
    // the split EvaluateClosedShellScreened(D) performs internally - both
    // spins share one density-weight pass there because the two matrices are
    // the same, so calling the pair form directly is the same integration.
    // The returned potentials are dE/dD_s in both lanes; the restricted
    // consumer halves their sum (the engine's own chain-rule note).
    //
    // The screen the evaluator passes is read off the context's own thresholds
    // rather than off a copy of the double: the energy path and the gradient
    // walk then screen by one value, and a divergence between the two is
    // unrepresentable rather than merely unlikely.
    context.evaluator =
        [engine = context.grid.engine, thresholds = context.grid.thresholds](
            const Eigen::MatrixXd& densityAlpha,
            const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
        return engine->EvaluateScreened(densityAlpha, densityBeta, thresholds.screeningTolerance);
    };

    return context;
}

// One Kohn-Sham half behind the composition's HalfFockFn, over the direct
// family's split builders.
//
// The half is built from a SPATIAL density (the composition owns that
// convention): a buildCoulombOnly member returns H + 2J(rho), a
// buildExchangeOnly member H - K(rho), and neither scales anything itself.
// The per-call stats collection is the split seam's own (the RHF seam
// shape): one row per call, the null sink keeping the plain call, and the
// certified-bound out-parameter read ONLY when a collector is wired - a
// builder that was never handed the out-parameter cannot report a bound, so
// its never-written zero must not reach the accumulator as an observed one
// (the MakeRhfFockBuilder note).
// \param builder The half's direct-family builder.
// \param statsSink The run's per-call stats stream, or null.
// \param exchangeOccupied The K-channel occupancy this half's call contracts
// (the composition's exchange half: the run's occupied count; the Coulomb
// half contracts no exchange, so it passes 0).
// \param boundAccum The run's certified-bound collector, or null.
// \returns The half as a HalfFockFn.
qcx::driver::internal::HalfFockFn MakeDirectKsHalf(
    const qcx::integrals::DirectJkFockBuilder& builder,
    ScfCallStatsStream* statsSink,
    std::size_t exchangeOccupied,
    CertifiedBoundAccumulator* boundAccum) {
    return [builder, statsSink, exchangeOccupied, boundAccum](
               const Eigen::MatrixXd& spatialDensity) -> qcx::Result<Eigen::MatrixXd> {
        auto rho = ToTensor(spatialDensity);

        if (!rho.has_value())
        {
            return std::unexpected(rho.error());
        }

        qcx::integrals::FockBuildStats callStats;
        double callBound = 0.0;
        qcx::integrals::CertifiedBudgetOutcome callBudget;
        // The call's own span, taken HERE rather than read off the stats the
        // builder fills: FockBuildStats::totalWallTime is not the call's span
        // on every builder this seam family can wrap (the MakeRhfFockBuilder
        // note), and a half's row must mean the same thing as a whole call's.
        // The null sink reads no clock at all.
        const auto callStarted = statsSink != nullptr ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        auto fock = builder.BuildFock(*rho,
                                      boundAccum != nullptr ? &callBound : nullptr,
                                      statsSink != nullptr ? &callStats : nullptr,
                                      nullptr,
                                      boundAccum != nullptr ? &callBudget : nullptr);
        const auto callWall = statsSink != nullptr ? std::chrono::steady_clock::now() - callStarted
                                                   : std::chrono::nanoseconds{};

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (boundAccum != nullptr)
        {
            boundAccum->Observe(callBound, callBudget);
        }

        if (statsSink != nullptr)
        {
            statsSink->Record(callStats, exchangeOccupied, callWall);
        }

        return ToMatrix(*fock);
    };
}

// The lean member's Kohn-Sham half: the same shape over the lean builder,
// whose stats-carrying BuildFock takes no certified-bound out-parameter at
// all (LeanFockBuildOptions has no enforcement field), and whose row family
// is RecordLean - the columns the lean stats can back, never machinery-style
// false zeros. The window record is handed over the same way the RHF
// lean seam hands it.
// \param builder The half's lean builder.
// \param statsSink The run's per-call stats stream, or null.
// \param exchangeOccupied The K-channel occupancy (0 for the Coulomb half).
// \returns The half as a HalfFockFn.
qcx::driver::internal::HalfFockFn MakeLeanKsHalf(
    const qcx::integrals::LeanDirectFockBuilder& builder,
    ScfCallStatsStream* statsSink,
    std::size_t exchangeOccupied) {
    return [builder, statsSink, exchangeOccupied](
               const Eigen::MatrixXd& spatialDensity) -> qcx::Result<Eigen::MatrixXd> {
        auto rho = ToTensor(spatialDensity);

        if (!rho.has_value())
        {
            return std::unexpected(rho.error());
        }

        qcx::integrals::FockBuildStats callStats;
        std::vector<std::chrono::nanoseconds> windowTimes;
        callStats.windowWallTimesOut = statsSink != nullptr ? &windowTimes : nullptr;
        auto fock = builder.BuildFock(*rho, statsSink != nullptr ? &callStats : nullptr);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->RecordLean(callStats, exchangeOccupied);
        }

        return ToMatrix(*fock);
    };
}

// The composed-QFMM UHF seam adapter: the scf loop's per-spin Fock
// calls with the whole J/K assembly inside QfmmHfFockBuilder::BuildUhfFock
// (the Coulomb half on the half-summed density through the QFMM near/far
// split, the exchange half per spin on the raw densities, one H copy
// subtracted per channel - the MakeDirectUhfFockBuilder math of the direct-builder UHF assembly).
// The lambda is only the Eigen <-> tensor boundary: the scf seam speaks Eigen matrices, the
// integrals builder host-canonical tensors. Unlike MakeDirectUhfFockBuilder there is no per-call
// stats wrapper here, so the per-spin exchange occupancies are not needed and a traced
// composed-QFMM UHF run lands zero rows in scf_trace.stats - the RHF arm's QFMM row family
// (MakeQfmmRhfFockBuilder / RecordQfmm) has no UHF counterpart yet, because RunQfmmUhfScf owns no
// stats stream of its own (its traceFile feeds the scf loop's own trace lines only). Known and
// named, never a silent zero: see DriverRunTest.TraceEnabledRunsEmitTheRowFamilyOfTheirBuilder.
qcx::scf::UhfFockBuilderFn MakeQfmmUhfFockBuilder(
    const qcx::integrals::QfmmHfFockBuilder& uhfBuilder) {
    return [uhfBuilder](const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)
               -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        auto dAlphaTensor = ToTensor(dAlpha);

        if (!dAlphaTensor.has_value())
        {
            return std::unexpected(dAlphaTensor.error());
        }

        auto dBetaTensor = ToTensor(dBeta);

        if (!dBetaTensor.has_value())
        {
            return std::unexpected(dBetaTensor.error());
        }

        auto focks = uhfBuilder.BuildUhfFock(*dAlphaTensor, *dBetaTensor);

        if (!focks.has_value())
        {
            return std::unexpected(focks.error());
        }

        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{ToMatrix(focks->first),
                                                           ToMatrix(focks->second)};
    };
}

// The RI-J link's per-spin adapter (the unrestricted leg's ri_j_link seam):
// MakeDirectUhfFockBuilder's assembly over the RI family's two halves
// - ONE Coulomb-only call on the half-summed density 0.5 (P_alpha + P_beta),
// whose 2 J_RI is the J(P_alpha + P_beta) of the unrestricted Fock, and one
// exchange-only call per spin on that spin's RAW density (the exchange
// half's -K is linear in its input, so P_sigma needs no halving) - with the
// halves summed per spin channel.
//
// What differs from that direct-family model is ONE line of accounting, and
// it is the family's own (ri_engine.cpp's BuildFock comment): on the RI path
// the contractions carry no core Hamiltonian and the exchange-only call is
// the sole H carrier, so nothing is subtracted per channel. On the direct
// family BOTH split halves carry a full H and the assembly subtracts one
// copy each (the double-H trap). The two compositions are deliberately not
// unified - the accounting differs - and this seam is the driver-side twin
// of that note.
//
// The three calls are recorded the way the direct arm's three are: the
// J-only call under occupancy 0, each exchange half under its own spin's -
// so a traced unrestricted ri_j_link run lands the row family its direct
// twin lands, and the restricted ri_j_link run lands the same through
// MakeRhfFockBuilder. The per-call wall span is measured HERE for the
// reason the sibling seams measure their own: on this family
// FockBuildStats::totalWallTime is the NESTED exchange half's span
// (ri_engine.cpp forwards the sink to it), so it is not the quantity a
// row's total_wall_ms promises. Null sink reads no clock.
qcx::scf::UhfFockBuilderFn MakeRiJLinkUhfFockBuilder(const qcx::integrals::RiJkFockBuilder& builder,
                                                     ScfCallStatsStream* statsSink,
                                                     std::size_t exchangeAlphaCount,
                                                     std::size_t exchangeBetaCount) {
    const auto buildHalf = [statsSink](std::size_t exchangeOccupied,
                                       auto&& call,
                                       const CpuTensor2& density) -> qcx::Result<CpuTensor2> {
        qcx::integrals::FockBuildStats callStats;
        const auto callStarted = statsSink != nullptr ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        auto fock = call(density, statsSink != nullptr ? &callStats : nullptr);
        const auto callWall = statsSink != nullptr ? std::chrono::steady_clock::now() - callStarted
                                                   : std::chrono::nanoseconds{};

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->Record(callStats, exchangeOccupied, callWall);
        }

        return fock;
    };

    // The two half calls, as the sibling adapters hold their builders: BY
    // VALUE (the RI builder is a shared-state handle, so a copy is cheap and
    // the seam cannot outlive what it points at).
    const auto coulombCall = [builder](const CpuTensor2& density,
                                       qcx::integrals::FockBuildStats* stats) {
        return builder.BuildCoulombOnly(density, stats);
    };
    const auto exchangeCall = [builder](const CpuTensor2& density,
                                        qcx::integrals::FockBuildStats* stats) {
        return builder.BuildExchangeOnly(density, stats);
    };

    return [buildHalf, coulombCall, exchangeCall, exchangeAlphaCount, exchangeBetaCount](
               const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)
               -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        auto dTotalHalf = ToTensor(0.5 * (dAlpha + dBeta));

        if (!dTotalHalf.has_value())
        {
            return std::unexpected(dTotalHalf.error());
        }

        auto coulomb = buildHalf(0, coulombCall, *dTotalHalf);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        auto dAlphaTensor = ToTensor(dAlpha);

        if (!dAlphaTensor.has_value())
        {
            return std::unexpected(dAlphaTensor.error());
        }

        auto exchangeAlpha = buildHalf(exchangeAlphaCount, exchangeCall, *dAlphaTensor);

        if (!exchangeAlpha.has_value())
        {
            return std::unexpected(exchangeAlpha.error());
        }

        auto dBetaTensor = ToTensor(dBeta);

        if (!dBetaTensor.has_value())
        {
            return std::unexpected(dBetaTensor.error());
        }

        auto exchangeBeta = buildHalf(exchangeBetaCount, exchangeCall, *dBetaTensor);

        if (!exchangeBeta.has_value())
        {
            return std::unexpected(exchangeBeta.error());
        }

        // No core-Hamiltonian subtraction: the Coulomb half carries none and
        // the exchange half is the only carrier (this seam's comment above).
        const Eigen::MatrixXd fAlpha = ToMatrix(*coulomb) + ToMatrix(*exchangeAlpha);
        const Eigen::MatrixXd fBeta = ToMatrix(*coulomb) + ToMatrix(*exchangeBeta);
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{fAlpha, fBeta};
    };
}

// The composed full-RI builder's per-spin adapter (the ri_jk family's
// unrestricted seam): MakeDirectUhfFockBuilder's assembly,
//   F_sigma = H + C(P_alpha) + C(P_beta) - K(P_sigma),
// over RiFullFockBuilder::BuildFockHalves - the pair entry point, ONE call
// per spin, whose two halves the fused BuildFock composes as H + 2 J_RI - K_RI.
//
// THE ACCOUNTING, which is the third convention this file carries and the one
// an assembly gets WRONG INVISIBLY (all three shapes produce a plausible
// Fock): the pair is BARE. Its Coulomb half is J_RI(rho), with no factor of
// two, and its exchange half is K_RI(rho), with no core Hamiltonian - where
// the ri_j_link family's contraction carries the factor 2 and puts H in the
// exchange half alone, and the direct/qfmm families put a full H in BOTH
// halves (their assemblies subtract one copy per channel). Here H is added
// ONCE PER CHANNEL by this seam and nothing is subtracted, which is what lets
// the two Coulomb halves be summed across spins: each is linear in the
// density it was handed, so C(P_alpha) + C(P_beta) is C of the total.
//
// THE DENSITY CONVENTION, derived from the two facts above rather than
// asserted: the argument is the RAW per-spin density the scf seam hands over.
// At P_alpha = P_beta = D/2 (a closed-shell deck) the sum is
// H + 2 J_RI(D/2) - K_RI(D/2), which is the restricted leg's fused build
// element for element - the identity the driver's closed-shell pin measures.
// The sibling adapters pass a half-summed density to their Coulomb call
// because THEIR builder's contraction already carries the factor of two; on
// this family the sum across spins is where the total comes from, and halving
// here would halve J.
//
// THE OCCUPIED BLOCK, which this seam has to derive because it - unlike its
// siblings - is handed DENSITIES ONLY (scf/uhf.hpp's seam contract): the
// self-adjoint eigendecomposition of the spin's density, its top nOccupied
// eigenvectors clamped at zero and scaled by the square root of their
// eigenvalues, exactly MakeRiFullRhfFockBuilder's recipe, so the block's own
// P = C C^T is the density the exchange half reads. The block's COLUMN COUNT
// is the spin's occupied count floored at one: the builder refuses an empty
// block by name, and a spin with no electrons (a doublet's beta channel) has
// a zero density whose top eigenvalue is zero, so sqrt-clamping drives the
// whole block to zero - the exchange half is then exactly K(0) = 0 and the
// Coulomb half never reads the block. The floor states a shape the builder
// accepts, not a claim about the spin.
//
// THE PER-CALL STATS ROW: ONE row per spin, recorded under that SPIN's own
// exchange occupancy. The call contracts a Coulomb half and an exchange half
// together, so its single row cannot carry two occupancies, and the exchange
// is the half that HAS one - the Coulomb half is occupied-space independent,
// and this family's Coulomb-only call is recorded under 0 for exactly that
// reason (MakeRiJLinkUhfFockBuilder). The choice moves no number on this
// family: xvol is significantQuartets * occupancy and these calls evaluate no
// quartet kernel, so the column is the measured zero the builder writes - but
// a row is a claim, and "this call contracted no occupied space" would be
// false where "this call's exchange contracted this spin's space" is true.
//
// The two calls also carry the pair's SECOND difference from the fused RHF
// path: the fused route would be BuildFock(rho_half, C_sigma) - a
// spin-SUMMED density beside ONE spin's occupied block - so the pair route is
// not merely cheaper, it is the only one of the two that pairs a block with
// the density it belongs to.
// \param builder The created composed full-RI builder. Held by value: every
// copy shares the one transformed tensor (shared state), so no per-call
// rebuild.
// \param coreHamiltonian H, added once per spin channel (see the accounting
// note above). Must be the same matrix the scf loop runs with.
// \param statsSink The optional call-stats stream, one row per spin (see the
// row note above). Null sink returns nil.
// \param occupiedAlphaCount The alpha occupied count - the rank of the block
// the alpha call's exchange half contracts.
// \param occupiedBetaCount The beta count.
// \returns The seam: the per-spin density pair in, (F_alpha, F_beta) out.
qcx::scf::UhfFockBuilderFn MakeRiFullUhfFockBuilder(
    const qcx::integrals::RiFullFockBuilder& builder,
    const Eigen::MatrixXd& coreHamiltonian,
    ScfCallStatsStream* statsSink,
    std::size_t occupiedAlphaCount,
    std::size_t occupiedBetaCount) {
    // One spin's whole contribution: the factorized occupied block, then the
    // pair entry point's one traversal. The call's own wall span is measured
    // HERE for the sibling seams' reason - the builder-backed totalWallTime
    // is a nested half's span on this family, not the call's.
    const auto buildSpin =
        [builder,
         statsSink](const Eigen::MatrixXd& density,
                    std::size_t occupiedCount) -> qcx::Result<qcx::integrals::RiFullFockHalves> {
        const Eigen::Index n = density.rows();
        const Eigen::Index columns =
            static_cast<Eigen::Index>(occupiedCount > 0 ? occupiedCount : 1);

        if (columns > n)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "the full-RI per-spin seam needs an occupied count in "
                                       "[1, nBasis]; got " +
                                           std::to_string(occupiedCount) + " for a density with " +
                                           std::to_string(n) + " rows"));
        }

        // The density of a real-orbital density pair is symmetric and positive
        // semi-definite, so the self-adjoint solver is the right one and its
        // eigenvalues come back ASCENDING: the occupied space is the top
        // nOccupied of them. The clamp is numerical, not physical - a PSD
        // matrix can come back with eigenvalues a few ulps below zero and the
        // square root of a negative one is NaN - and those eigenvalues belong
        // to the virtual space, so nothing physical rides on it.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(density);

        if (solver.info() != Eigen::Success)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                       "the full-RI per-spin seam could not diagonalize the "
                                       "density it was handed"));
        }

        const Eigen::VectorXd occupiedValues = solver.eigenvalues().tail(columns).cwiseMax(0.0);
        const Eigen::MatrixXd occupiedBlock =
            solver.eigenvectors().rightCols(columns) * occupiedValues.cwiseSqrt().asDiagonal();

        auto densityTensor = ToTensor(density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        qcx::integrals::FockBuildStats callStats;
        qcx::integrals::FockBuildStats* statsOut = statsSink != nullptr ? &callStats : nullptr;
        const auto callStarted = statsSink != nullptr ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        auto halves = builder.BuildFockHalves(*densityTensor, occupiedBlock, statsOut);
        const auto callWall = statsSink != nullptr ? std::chrono::steady_clock::now() - callStarted
                                                   : std::chrono::nanoseconds{};

        if (!halves.has_value())
        {
            return std::unexpected(halves.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->Record(callStats, occupiedCount, callWall);
        }

        return std::move(*halves);
    };

    return [buildSpin, coreHamiltonian, occupiedAlphaCount, occupiedBetaCount](
               const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)
               -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        auto alphaHalves = buildSpin(dAlpha, occupiedAlphaCount);

        if (!alphaHalves.has_value())
        {
            return std::unexpected(alphaHalves.error());
        }

        auto betaHalves = buildSpin(dBeta, occupiedBetaCount);

        if (!betaHalves.has_value())
        {
            return std::unexpected(betaHalves.error());
        }

        // The spin-summed Coulomb half: C(P_alpha) + C(P_beta) = C(D_total),
        // and at a closed shell exactly the fused build's 2 J_RI(D/2). H is
        // added ONCE per channel and nothing is subtracted (the accounting
        // note above) - the line the whole file's conventions turn on.
        const Eigen::MatrixXd coulomb =
            ToMatrix(alphaHalves->coulomb) + ToMatrix(betaHalves->coulomb);
        const Eigen::MatrixXd fAlpha = coreHamiltonian + coulomb - ToMatrix(alphaHalves->exchange);
        const Eigen::MatrixXd fBeta = coreHamiltonian + coulomb - ToMatrix(betaHalves->exchange);
        return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{fAlpha, fBeta};
    };
}

// Which half of the RI-J link a Kohn-Sham HalfFockFn is over.
//
// The two halves differ by exactly ONE matrix, and the difference is the
// family's own accounting rather than a choice made here: the RI-J
// contraction carries no core Hamiltonian at all (BuildCoulombOnly returns
// 2 J_RI(rho) and nothing else) while the exchange-only call is the sole H
// carrier (it returns H - K(rho)). The composition's HalfFockFn contract is
// the DIRECT family's, whose both halves carry a full H each - so the
// Coulomb half below must add H to reach the contracted shape, and the
// exchange half is the builder's own result verbatim. Adding it to both, or
// to neither, moves the Fock by an entire core Hamiltonian and the energy by
// Tr[D H]; the two cases are not interchangeable and this enum is what keeps
// them apart at the call site.
// Not a bulk-storage type; shrinking the base type is a deferred micro-optimization.
// NOLINTNEXTLINE(performance-enum-size)
enum class RiJLinkKsHalf {
    kCoulomb, ///< Returns H + 2 J_RI(rho): the composition's Coulomb-half convention.
    kExchange, ///< Returns H - K(rho): the nested exchange builder's own result.
};

// One Kohn-Sham half behind the composition's HalfFockFn, over the RI-J
// link's split builders - the RI family's MakeDirectKsHalf, with the
// accounting difference the enum above states.
//
// The half takes a SPATIAL density (the composition owns that convention)
// and captures the builder BY VALUE, as the sibling adapters do: the RI
// builder is a shared-state handle, so a copy is cheap and the seam cannot
// outlive what it points at.
//
// The per-call stats row is the sibling adapters' own shape - FockBuildStats
// plus the call's own wall span measured HERE, because on this family
// FockBuildStats::totalWallTime is the NESTED exchange half's span
// (ri_engine.cpp forwards the sink to it), so it is not the quantity a row's
// total_wall_ms promises. The Coulomb call writes the measured zero that file
// documents (the RI contractions evaluate no quartets) and is recorded under
// occupancy 0; the exchange call is recorded under the occupancy its run
// contracts. Neither call takes a certified-bound out-parameter (the RI
// family's options carry no enforcement field), so this seam reports no bound
// at all rather than folding a never-written zero - the MakeLeanKsHalf rule.
// \param builder The RI-J link builder both halves come from.
// \param coreHamiltonian H, added back into the Coulomb half (see the enum).
// It must be the EXACT matrix the seam is created with: the composition
// subtracts it back out of this half, and an H that is a different object
// leaves a residue in the J[D] the energy seam contracts.
// \param half Which half this callable is.
// \param statsSink The run's per-call stats stream, or null.
// \param exchangeOccupied The K-channel occupancy this half's call contracts
// (the composition's exchange half: the run's occupied count; the Coulomb
// half contracts no exchange, so it passes 0).
// \returns The half as a HalfFockFn.
qcx::driver::internal::HalfFockFn MakeRiJLinkKsHalf(const qcx::integrals::RiJkFockBuilder& builder,
                                                    const Eigen::MatrixXd& coreHamiltonian,
                                                    RiJLinkKsHalf half,
                                                    ScfCallStatsStream* statsSink,
                                                    std::size_t exchangeOccupied) {
    return [builder, coreHamiltonian, half, statsSink, exchangeOccupied](
               const Eigen::MatrixXd& spatialDensity) -> qcx::Result<Eigen::MatrixXd> {
        auto rho = ToTensor(spatialDensity);

        if (!rho.has_value())
        {
            return std::unexpected(rho.error());
        }

        qcx::integrals::FockBuildStats callStats;
        qcx::integrals::FockBuildStats* statsOut = statsSink != nullptr ? &callStats : nullptr;
        const auto callStarted = statsSink != nullptr ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        // The Coulomb half is the Coulomb-ONLY entry point and nothing else:
        // this is the one place where "the half is the fused BuildFock" could
        // be written by mistake, and it would be invisible in the result
        // because BuildFock's extra term is the exchange it already scales.
        auto fock = half == RiJLinkKsHalf::kCoulomb ? builder.BuildCoulombOnly(*rho, statsOut)
                                                    : builder.BuildExchangeOnly(*rho, statsOut);
        const auto callWall = statsSink != nullptr ? std::chrono::steady_clock::now() - callStarted
                                                   : std::chrono::nanoseconds{};

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->Record(callStats, exchangeOccupied, callWall);
        }

        Eigen::MatrixXd matrix = ToMatrix(*fock);

        if (half == RiJLinkKsHalf::kCoulomb)
        {
            matrix += coreHamiltonian;
        }

        return matrix;
    };
}

// Which half of the composed full-RI pair a Kohn-Sham HalfFockFn is over.
//
// THE THIRD ACCOUNTING on this path, and the only one whose conversion touches
// BOTH halves. The pair entry point (RiFullFockBuilder::BuildFockHalves) hands
// back the fused build's two terms BEFORE the fused build composes them: its
// Coulomb half is J_RI(rho), bare - no factor of two - and its exchange half is
// K_RI(rho), bare as well - no core Hamiltonian, in EITHER half (the pair
// carries no H at all; RiFullFockHalves states that accounting). Neither of the
// two conventions the sibling enums describe is therefore reached natively: the
// RI-J link's Coulomb half carries the factor of two and no H, and the direct
// and composed-QFMM families put a full H copy into each half. The
// composition's contract - Coulomb half H + 2 J(rho), exchange half H - K(rho),
// the direct family's accounting - is reached from this pair by adding H to
// BOTH halves and DOUBLING the Coulomb one.
//
// The failure this enum prevents is the one that leaves nothing in the result
// to show it: adding H to one half only moves the Fock by an entire core
// Hamiltonian (the energy by Tr[D H]), leaving the factor of two out of the
// Coulomb half moves it by an entire J, and every one of the shapes that can
// come out of those mistakes is a plausible Fock matrix.
// Not a bulk-storage type; shrinking the base type is a deferred micro-optimization.
// NOLINTNEXTLINE(performance-enum-size)
enum class RiFullKsHalf {
    /// Returns H + 2 J_RI(rho): the pair's bare Coulomb half, doubled, with
    /// the core Hamiltonian the pair omits added in.
    kCoulomb,
    /// Returns H - K_RI(rho): the pair's bare exchange half, negated, with the
    /// same core Hamiltonian added in.
    kExchange,
};

// The composed full-RI pair's ONE-ENTRY MEMO, shared by the Kohn-Sham halves
// of one run.
//
// THE BUILD THIS SAVES. One BuildFockHalves call is both halves, so a caller
// that asks for them one at a time pays the whole pair build once per ask -
// and the composition asks exactly that way, at ONE density (MakeRksSeam's
// ensure: rho for the Coulomb half, then rho again for the exchange half the
// Fock's exact-exchange term and the energy seam's J[D] both read). MEASURED
// on H2O/STO-3G + def2-universal-jkfit, rks + b3lyp + ri_jk, 6 iterations
// (build/windows-msvc/ks-rijk-probe/trace_rks_b3lyp.csv.stats): rows 14
// against 7 for the same deck on slater, whose c_HF = 0 asks the Coulomb half
// alone - one row per density on both decks is the honest count, and the
// second row per density was the same pair built twice.
//
// THE KEY IS EXACT, AND THAT IS THE WHOLE SAFETY ARGUMENT. The memo may serve
// a half only when the density is bit-identical to the one the entry was built
// at, because this seam's contract is that the energy reads the SAME density
// the Fock was built from: a tolerance-keyed or near-matching cache would
// answer the energy seam with a different density's J[D], which is a plausible
// number from an input nobody stated . The key carries the occupied
// count as well, because BuildFockHalves is handed that count and derives the
// block from it - two asks at one density with different counts are two
// different builds (the UKS arm hands its Coulomb half the alpha count while
// its beta exchange half carries the beta one). Given the density and the
// count, the call is fully determined, so a hit is the identical call to a
// deterministic builder: a pure saving, with no numerical reach.
//
// Both Kohn-Sham lanes of this family hold ONE cache per run - the RKS arm
// its two halves, the UKS arm its three - and where two asks sit at the SAME
// density the pair is built once, so the Fock and the energy seam read one
// build rather than two that agree. That is every density on the RKS arm
// (both halves take rho = D/2), and on the UKS arm exactly those densities
// whose spins coincide, where the half-sum and each raw spin density are one
// matrix: MEASURED on the H2O/STO-3G b3lyp unrestricted deck, rows 21 before
// this memo against 11 after, the first five fills serving three asks from
// one build each and the last two - the spins no longer bit-equal - building
// all three. The differences measured on the other decks: rks/b3lyp 14 -> 7,
// rks/slater 7 -> 7, uks/slater 7 -> 7, and every deck's energy unmoved.
struct RiFullKsPairCache {
    Eigen::MatrixXd density; ///< The exact spatial density the entry was built at.
    std::size_t occupiedCount = 0; ///< The occupied-block rank it was built with.
    Eigen::MatrixXd coulombHalf; ///< H + 2 J_RI(density).
    Eigen::MatrixXd exchangeHalf; ///< H - K_RI(density).
    bool valid = false; ///< False until the first successful build.
};

// One Kohn-Sham half behind the composition's HalfFockFn, over the composed
// full-RI pair entry point - the arm that opens the ri_jk family's Kohn-Sham
// cell on both Kohn-Sham lanes.
//
// The half takes a SPATIAL density (the composition owns that convention, and
// on this family it is also the builder's own: BuildFockHalves reads the
// spatial rho = D/2, MakeRiFullRhfFockBuilder's line). What the pair entry
// point also needs is the OCCUPIED BLOCK of that density, and unlike its
// sibling adapters this seam is handed a density only - the composition's
// HalfFockFn carries no block - so the factorization runs HERE, by the
// recipe the family's other two seams already use rather than a third one: the
// density is symmetric and positive semi-definite over real orbitals, so the
// self-adjoint solver is the right one and its eigenvalues come back
// ASCENDING - the occupied space is the top `occupiedCount` of them, clamped
// at zero (a PSD matrix can come back a few ulps below zero and the square
// root of a negative one is NaN, and those eigenvalues belong to the virtual
// space) and scaled by the square root of their eigenvalues, so the block's own
// P = C C^T is the density the exchange half reads.
//
// The COLUMN COUNT is floored at one: the builder refuses an empty block by
// name, and a spin with no electrons has a zero density whose top eigenvalue is
// zero, so the clamp drives the whole block to zero and the exchange half is
// exactly K(0) = 0. The floor states a shape the builder accepts, not a claim
// about the spin - and the stats row below records the count as the run stated
// it, floor or no floor.
//
// ONE CALL IS BOTH HALVES, AND ONE RUN BUILDS IT ONCE. BuildFockHalves
// contracts the Coulomb and the exchange half together, whichever of them the
// caller keeps, so the per-call stats row carries the EXCHANGE occupancy - the
// sibling per-spin adapter's rule at the same entry point. That move costs no
// number on this family (the RI contractions evaluate no quartet kernel, so
// xvol is the measured zero the builder writes), but a row is a claim, and
// "this call contracted no occupied space" would be false of a call that
// contracted exactly this one.
//
// The two calls of a RUN therefore share one memo (RiFullKsPairCache), because
// the composition asks its two halves at the SAME density: without it a hybrid
// pays the pair build TWICE per density - once for the Fock's Coulomb half and
// once more for the exchange half the Fock's own exchange term and the energy
// seam's J[D] are then read from. See that struct for the key and why it is
// exact.
// \param builder The created composed full-RI builder both halves come from.
// \param coreHamiltonian H, added to BOTH halves (see the enum above). It must
// be the EXACT matrix the seam is created with: the composition recovers the
// J[D] its energy formula contracts by subtracting that matrix back out of the
// Coulomb half, so a different one would leave a residue in the energy.
// \param half Which half this callable is.
// \param occupiedCount The rank of the occupied block this half's call
// factors the density down to (the run's occupied count for that density).
// \param pairCache The run's one-entry memo of the pair, shared by the halves
// the composition asks at one density. Must be non-null; the two halves of a
// run must share ONE cache, since two caches silently restore the double build
// this parameter exists to remove.
// \param statsSink The run's per-call stats stream, or null.
// \returns The half as a HalfFockFn.
qcx::driver::internal::HalfFockFn MakeRiFullKsHalf(
    const qcx::integrals::RiFullFockBuilder& builder,
    const Eigen::MatrixXd& coreHamiltonian,
    RiFullKsHalf half,
    std::size_t occupiedCount,
    const std::shared_ptr<RiFullKsPairCache>& pairCache,
    ScfCallStatsStream* statsSink) {
    return [builder, coreHamiltonian, half, occupiedCount, pairCache, statsSink](
               const Eigen::MatrixXd& spatialDensity) -> qcx::Result<Eigen::MatrixXd> {
        // A HIT serves a half this run ALREADY built, and it is taken only on
        // the exact pair of inputs BuildFockHalves was handed: the spatial
        // density, bit for bit (the composition's own predicate, one spelling
        // of the key for both caches), and the occupied count, from which the
        // block is derived - so a hit means the identical call to a
        // deterministic builder. Nothing fuzzier would be safe here: this
        // seam's contract is that the energy reads the SAME density the Fock
        // was built from, and a near-match would hand the energy seam a
        // different one (the defect class - a plausible number from an
        // input nobody stated). A hit records NO stats row: a row is a claim
        // that a build happened, and this call's build is the row the miss
        // already wrote.
        if (pairCache->valid && pairCache->occupiedCount == occupiedCount &&
            qcx::driver::internal::detail::SameMatrix(pairCache->density, spatialDensity))
        {
            return half == RiFullKsHalf::kCoulomb ? pairCache->coulombHalf
                                                  : pairCache->exchangeHalf;
        }

        const Eigen::Index n = spatialDensity.rows();
        const Eigen::Index columns =
            static_cast<Eigen::Index>(occupiedCount > 0 ? occupiedCount : 1);

        if (columns > n)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "the full-RI Kohn-Sham half needs an occupied count in "
                                       "[1, nBasis]; got " +
                                           std::to_string(occupiedCount) + " for a density with " +
                                           std::to_string(n) + " rows"));
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(spatialDensity);

        if (solver.info() != Eigen::Success)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                       "the full-RI Kohn-Sham half could not diagonalize the "
                                       "density it was handed"));
        }

        const Eigen::VectorXd occupiedValues = solver.eigenvalues().tail(columns).cwiseMax(0.0);
        const Eigen::MatrixXd occupiedBlock =
            solver.eigenvectors().rightCols(columns) * occupiedValues.cwiseSqrt().asDiagonal();

        auto densityTensor = ToTensor(spatialDensity);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        qcx::integrals::FockBuildStats callStats;
        qcx::integrals::FockBuildStats* statsOut = statsSink != nullptr ? &callStats : nullptr;
        const auto callStarted = statsSink != nullptr ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
        auto halves = builder.BuildFockHalves(*densityTensor, occupiedBlock, statsOut);
        const auto callWall = statsSink != nullptr ? std::chrono::steady_clock::now() - callStarted
                                                   : std::chrono::nanoseconds{};

        if (!halves.has_value())
        {
            return std::unexpected(halves.error());
        }

        if (statsSink != nullptr)
        {
            statsSink->Record(callStats, occupiedCount, callWall);
        }

        // THE CONVERSION, as the two lines the enum above commits to: the
        // Coulomb half gains the factor of two the pair leaves to its fused
        // caller, and BOTH halves gain the core Hamiltonian the pair omits.
        const Eigen::MatrixXd coulombHalf = coreHamiltonian + 2.0 * ToMatrix(halves->coulomb);
        const Eigen::MatrixXd exchangeHalf = coreHamiltonian - ToMatrix(halves->exchange);

        // COMMIT-LAST, the composition's own order (its ensure/pair fill): the
        // entry is written only after every call above has succeeded, so a
        // failed build leaves the previous entry intact - key and contents
        // alike - and the next ask for that density is still served by it. An
        // entry carrying a FAILED call's half would be served as a hit.
        pairCache->density = spatialDensity;
        pairCache->occupiedCount = occupiedCount;
        pairCache->coulombHalf = coulombHalf;
        pairCache->exchangeHalf = exchangeHalf;
        pairCache->valid = true;

        return half == RiFullKsHalf::kCoulomb ? coulombHalf : exchangeHalf;
    };
}

// Which half of the composed QFMM builder a Kohn-Sham HalfFockFn is over.
//
// The accounting here is NOT the RI-J link's, and the difference is the
// whole reason this enum exists beside that one. The composed QFMM builder
// is assembled from a QfmmJBuilder and an exchange-only DirectJkFockBuilder
// (qfmm_hf_build.cpp), and BOTH nested results are already in the
// composition's own convention: the Coulomb half returns H + 2J_QFMM(rho)
// and the exchange half returns H - K(rho) - each carries one full H copy,
// which is what the fused Fock subtracts back once. So neither half gains
// or loses an H at the call site (the RI-J link's Coulomb half must add H,
// because its contraction carries no core Hamiltonian at all). Adding an H
// to either call below, or subtracting one, would move the Fock by an
// entire core Hamiltonian and the energy by Tr[D H]; the enum keeps the two
// calls apart so a reader can check the accounting against this paragraph.
// Not a bulk-storage type; shrinking the base type is a deferred micro-optimization.
// NOLINTNEXTLINE(performance-enum-size)
enum class QfmmKsHalf {
    kCoulomb, ///< Returns H + 2 J_QFMM(rho): the composition's Coulomb-half convention.
    kExchange, ///< Returns H - K(rho): the nested exchange-only builder's own result.
};

// One Kohn-Sham half behind the composition's HalfFockFn, over the composed
// QFMM builder's two exposed halves (QfmmHfFockBuilder::
// BuildCoulombOnly / BuildExchangeOnly).
//
// The half takes a SPATIAL density (the composition owns that convention)
// and captures the builder BY VALUE, as the sibling adapters do: the
// composed builder is a shared-state handle, so a copy is cheap and the
// seam cannot outlive what it points at. Both halves are the SAME builder
// instance the run's fused Fock would have been built from, so the J the
// energy seam contracts is that builder's own at the loop's density.
//
// The per-call stats row: this family backs no FockBuildStats at either
// half - the QFMM Coulomb half's far-field multipole passes carry no
// quartet count, and the seam family was never wired to the machinery row -
// so the halves record nothing rather than a fabricated zero (the
// MakeQfmmRhfFockBuilder rule, whose whole-composed-call row this seam
// deliberately does not split: a half call is not a Fock build, and a row
// claiming to be one would misstate what the columns mean). A traced run on
// this cell therefore lands zero per-call rows, which the empty-trace
// warning states rather than leaving silent - the composed-QFMM UHF arm's
// own documented behaviour.
// \param builder The composed builder both halves come from.
// \param half Which half this callable is.
// \returns The half as a HalfFockFn.
qcx::driver::internal::HalfFockFn MakeQfmmKsHalf(const qcx::integrals::QfmmHfFockBuilder& builder,
                                                 QfmmKsHalf half) {
    return [builder, half](const Eigen::MatrixXd& spatialDensity) -> qcx::Result<Eigen::MatrixXd> {
        auto rho = ToTensor(spatialDensity);

        if (!rho.has_value())
        {
            return std::unexpected(rho.error());
        }

        // The Coulomb half is BuildCoulombOnly and nothing else: this is the
        // one place where "the half is the fused BuildFock" could be written
        // by mistake, and it would be invisible in the result because the
        // fused call's extra term is the exchange this run also carries.
        auto fock = half == QfmmKsHalf::kCoulomb ? builder.BuildCoulombOnly(*rho)
                                                 : builder.BuildExchangeOnly(*rho);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        return ToMatrix(*fock);
    };
}

// The SAD fragment runs (BuildSadGuess) need per-element atomic integrals.
// The fragment is one neutral atom at the origin in its ground-state
// multiplicity, run with the MOLECULAR basis (a single-element atom of a
// same-basis system - the direct_uhf_test.cpp reference pattern).
int AtomicGroundStateMultiplicity(int atomicNumber) {
    switch (atomicNumber)
    {
    case 1: // H 2S
        return 2;
    case 2: // He 1S
        return 1;
    case 3: // Li 2S
        return 2;
    case 4: // Be 1S
        return 1;
    case 5: // B 2P
        return 2;
    case 6: // C 3P
        return 3;
    case 7: // N 4S
        return 4;
    case 8: // O 3P
        return 3;
    case 9: // F 2P
        return 2;
    case 10: // Ne 1S
        return 1;
    default:
        return 0; // not in the v1 table
    }
}

qcx::Result<std::map<int, qcx::scf::AtomicUhfInputs>> BuildAtomicInputs(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basis) {
    std::map<int, qcx::scf::AtomicUhfInputs> inputs;

    for (const auto& atom : molecule.Atoms())
    {
        if (inputs.contains(atom.atomicNumber))
        {
            continue;
        }

        const int multiplicity = AtomicGroundStateMultiplicity(atom.atomicNumber);

        if (multiplicity == 0)
        {
            return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                       "the SAD atomic-multiplicity table (v1, Z <= 10) does "
                                       "not cover element Z = " +
                                           std::to_string(atom.atomicNumber)));
        }

        auto coordinates = CpuTensor2::Create({1, 3});

        if (!coordinates.has_value())
        {
            return std::unexpected(coordinates.error());
        }

        (*coordinates)(0, 0) = 0.0;
        (*coordinates)(0, 1) = 0.0;
        (*coordinates)(0, 2) = 0.0;
        coordinates->MarkHostDirty();

        auto fragment = qcx::molecule::Molecule::Create(
            std::vector<qcx::molecule::Atom>{{atom.symbol, atom.atomicNumber, 0.0}},
            std::move(*coordinates),
            0,
            multiplicity);

        if (!fragment.has_value())
        {
            return std::unexpected(fragment.error());
        }

        auto overlap = qcx::integrals::BuildOverlapMatrix(*fragment, basis);

        if (!overlap.has_value())
        {
            return std::unexpected(overlap.error());
        }

        auto kinetic = qcx::integrals::BuildKineticMatrix(*fragment, basis);

        if (!kinetic.has_value())
        {
            return std::unexpected(kinetic.error());
        }

        auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*fragment, basis);

        if (!nuclear.has_value())
        {
            return std::unexpected(nuclear.error());
        }

        auto eri = qcx::integrals::BuildEriTensorGeneral(*fragment, basis);

        if (!eri.has_value())
        {
            return std::unexpected(eri.error());
        }

        inputs.emplace(atom.atomicNumber,
                       qcx::scf::AtomicUhfInputs{ToMatrix(*overlap),
                                                 ToMatrix(*kinetic) + ToMatrix(*nuclear),
                                                 std::move(*eri)});
    }

    return inputs;
}

// Molecule::Create canonically renumbers the atoms by (Z, x, y, z) and
// permutes the coordinates to match (molecule.cpp). The run input's
// nocv_fragments indices refer to the atom rows AS WRITTEN IN THE FILE, so
// the driver recomputes the same permutation (identical keys, same
// comparator - the canonicalization is deterministic) and the fragment
// groups reach the analysis in canonical atom order.
std::vector<std::size_t> CanonicalAtomOrder(const std::vector<qcx::molecule::Atom>& inputAtoms,
                                            const CpuTensor2& coordinatesBohr) {
    const std::size_t n = inputAtoms.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (inputAtoms[a].atomicNumber != inputAtoms[b].atomicNumber)
        {
            return inputAtoms[a].atomicNumber < inputAtoms[b].atomicNumber;
        }

        for (std::size_t d = 0; d < 3; ++d)
        {
            if (coordinatesBohr(a, d) != coordinatesBohr(b, d))
            {
                return coordinatesBohr(a, d) < coordinatesBohr(b, d);
            }
        }

        return false;
    });
    return order;
}

// Maps the file-order fragment groups through the canonicalization's
// inverse permutation; the disjoint/cover validation of the resulting
// canonical partition is the analysis's own (AnalyzeNocvEts).
qcx::Result<std::vector<std::vector<std::size_t>>> MapFragmentsToCanonical(
    const std::vector<std::vector<std::size_t>>& fragments,
    const std::vector<std::size_t>& canonicalOrder) {
    std::vector<std::size_t> inverse(canonicalOrder.size());

    for (std::size_t newIndex = 0; newIndex < canonicalOrder.size(); ++newIndex)
    {
        inverse[canonicalOrder[newIndex]] = newIndex;
    }

    std::vector<std::vector<std::size_t>> mapped;
    mapped.reserve(fragments.size());

    for (const auto& group : fragments)
    {
        std::vector<std::size_t> mappedGroup;
        mappedGroup.reserve(group.size());

        for (const std::size_t index : group)
        {
            if (index >= inverse.size())
            {
                return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                           "nocv_fragments atom index " + std::to_string(index) +
                                               " is out of "
                                               "range (the file lists " +
                                               std::to_string(inverse.size()) + " atoms)"));
            }

            mappedGroup.push_back(inverse[index]);
        }

        mapped.push_back(std::move(mappedGroup));
    }

    return mapped;
}

// The Fukui charged species share geometry and basis with the neutral; the
// charge and multiplicity change, the atoms and coordinates do not.
qcx::Result<qcx::molecule::Molecule> MakeChargedMolecule(const qcx::molecule::Molecule& molecule,
                                                         int charge,
                                                         int multiplicity) {
    auto coordinates = molecule.CoordinatesBohr().Clone();

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    return qcx::molecule::Molecule::Create(
        molecule.Atoms(), std::move(*coordinates), charge, multiplicity);
}

// The ERI cache budget the cap leaves for the direct family, in bytes: the
// cap minus a 1 GiB reserve, divided by 3 because the cache costs 3x its
// budget in live memory (the two payload lanes plus the entry map,
// integrals footprint.hpp), and held at 8 GiB - beyond that a cache stops
// helping the batch schedule and only delays a hard cap hit. The grant is
// a RESERVE, not a prediction: the engine's own Create-time cache clamp
// re-verifies it against the workspace budget's remaining bytes ('s
// sure-fit gate), and the job-object cap is the backstop. Zero means "no
// cache" - the uncached behavior, bit-identical pins. The direct family is
// the only family with the knob (FockBuildOptions::maxCacheBytes); the
// ri_j/qfmm/gpu builders have no cache.
std::size_t DirectFamilyCacheBytes(double memoryCapGiB) {
    // The 1 GiB reserve the grant keeps below the cap, so a run at the
    // granted cache never brushes the cap through cache growth alone.
    constexpr double kCacheReserveGiB = 1.0;
    const double budgetGiB = memoryCapGiB - kCacheReserveGiB;

    if (budgetGiB <= 0.0)
    {
        return 0;
    }

    return static_cast<std::size_t>(std::min(budgetGiB / 3.0, 8.0) * kGiB);
}

// The wired per-spin UHF seam plus the adaptive seam's artifacts: what a
// builder-selection arm of a UHF runner hands the shared SCF tail - the
// UHF analogue of WiredFockBuilder on the RHF side. The machinery arm
// fills all four members (its two admission-gated builders' Create-time
// mode records and the workspace budget they share); the lean arm fills
// only the seam - the lean member has no budget and no Create-time mode
// record to report, so its three artifacts stay empty and the run record's
// workspace_budget and mode blocks stay absent (the null honesty policy;
// the RHF lean arm records the same way).
struct WiredUhfBuilders {
    qcx::scf::UhfFockBuilderFn fn;
    std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;
    std::optional<qcx::integrals::FockModeInfo> modeInfo;
    std::optional<qcx::integrals::FockModeInfo> exchangeModeInfo;
    // The Kohn-Sham seam's companions, exactly as on the RHF side: set only
    // when the composition was engaged, so a UKS run hands the loop three
    // callbacks and a UHF run one. The Coulomb callback takes the SPIN-SUMMED
    // total (the seam's own contract), which is why the composition caches its
    // half under a key of its own.
    std::optional<qcx::scf::CoulombFn> coulomb;
    std::optional<qcx::scf::UhfEnergyContributionFn> contribution;
    /// The composed-QFMM model the Coulomb half ran, the
    /// WiredFockBuilder twin of this carrier: set by the composed-QFMM
    /// runner's two arms (the composed Hartree-Fock arm and the Kohn-Sham one),
    /// read by the shared tail below so the record's `qfmm_model` block
    /// states the model that ran rather than one the input asked for.
    /// Empty on every other UHF route.
    std::optional<qcx::integrals::QfmmModelRecord> qfmmModel;
    // The ri_j_link arm's carriers (the WiredFockBuilder pair's UHF twins),
    // filled by the unrestricted RI-J-link runner alone: the builder whose
    // per-run calibration counters the record publishes on an instrumented
    // run, and the orbit expansion's request/outcome pair. Every other arm
    // leaves them empty/false - none of them holds a ri_j_link builder.
    std::optional<qcx::integrals::RiJkFockBuilder> riJkBuilder;
    bool orbitExpansionEngaged = false;
    bool orbitExpansionWired = false;
    /// The composed full-RI family's own Create-time rung decision (schema
    /// 30, RiFullFockModeInfo): the WiredFockBuilder member's UHF twin,
    /// filled by the unrestricted ri_jk runner alone and read by the shared
    /// tail's record so resources_resolved.ri_jk_mode states the rung the
    /// engine decided rather than one the input asked for. It needs its own
    /// member rather than the modeInfo slot above: that slot is typed
    /// FockModeInfo, which is the RI-J engine's record, and this family's
    /// rung vocabulary and term decomposition are its own - folding one into
    /// the other's block would make one key mean two things (the RHF
    /// carrier's own note).
    ///
    /// What deliberately does NOT ride beside it is the term_counters block:
    /// RiFullFockBuilder does expose TermCounters(), and this leg therefore
    /// COULD publish one, but that block's key vocabulary is the ri_j_link
    /// family's - qx/gx are defined as the nested DIRECT exchange's counts
    /// (ri_engine.hpp RiTermCounters, and .8 says in as many words
    /// that the block is an instrumented ri_j_link run's) - while this
    /// builder's exchange is occ-RI-K and evaluates no quartet kernel, so a
    /// published pair would read as that family's measured zero. The
    /// restricted ri_jk leg publishes none either, so the two legs of the one
    /// family agree; a block in this family's own vocabulary is the shape
    /// that would close the gap, and it is named as an open item rather than
    /// approximated here.
    std::optional<qcx::integrals::RiFullFockModeInfo> riJkModeInfo;
};

// The shared tail of the UHF builder selections (the direct family's
// machinery member and its lean member): the SCF options, the guess tier,
// the loop, the empty-trace warning and the UhfRun assembly. Everything
// that separates the two arms - which builders ran, which admission sized
// them, which stats row family their calls recorded - has already happened
// by the time this is called; nothing downstream differs.
// \param nAlpha The alpha electron count (the guess tier and the loop's
// per-spin bookkeeping).
// \param nBeta The beta electron count.
// \param statsStream The run's per-call stats stream (already opened or
// not per enableTrace); read only for the empty-trace warning.
// \param wired The selected arm's seam and artifacts, consumed here.
// \param kind The builder kind that ran, for the empty-trace warning's own
// message (the warning tells the reader WHICH family recorded nothing).
// Defaulted because the direct runner's three arms all are that family; the
// unrestricted RI-J-link runner passes its own.
qcx::Result<UhfRun> RunUhfScfWithBuilder(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    bool enableTrace,
    int nAlpha,
    int nBeta,
    ScfCallStatsStream& statsStream,
    WiredUhfBuilders wired,
    qcx::io::BuilderKind kind = qcx::io::BuilderKind::kDirect) {
    qcx::scf::UhfOptions options;
    options.maxIterations = input.scf.maxIterations;
    options.energyTolerance = input.scf.energyTolerance;
    options.densityTolerance = input.scf.densityTolerance;
    options.useDiis = input.scf.useDiis;
    options.traceFile = enableTrace ? input.scf.traceFile : "";
    options.fullGroupLabeling = input.symmetry.fullGroup;

    // kCore resolves HERE, to the same GWH seed kGwh asks for by name (both
    // unrestricted seed sites - the machinery arm and the lean arm below -
    // carry this identically). The unfixed form left the unrestricted start
    // UNSET, and `core` means "the method's default start", which the
    // RHF path already resolves to GWH: the asymmetry was the defect.
    //
    // Measured on O2/STO-3G at 1.2075 A (2.2818443 Bohr), the ruled
    // defaults, after the case was cleared of a coordinator unit error: the
    // unset start landed -147.37855917618 - the saddle, 0.2536 Ha ABOVE the
    // ROHF bound -147.63216699080 - in 37 iterations, while GWH and SAD
    // both reach the pyscf UHF value -147.63394678556 (2.7e-8) in 23 and 34
    // iterations. OH/STO-3G is unaffected by either (2e-8 from pyscf).
    // Owner ruling 2026-09-15: use our good defaults rather than an
    // inferior start chosen to match a reference.
    if (input.guess == qcx::io::GuessKind::kGwh || input.guess == qcx::io::GuessKind::kCore)
    {
        auto gwh = qcx::scf::BuildGwhGuess(overlap, core, nAlpha, nBeta);

        if (!gwh.has_value())
        {
            return std::unexpected(gwh.error());
        }

        options.initialDensityAlpha = gwh->first;
        options.initialDensityBeta = gwh->second;
    } else if (input.guess == qcx::io::GuessKind::kSad)
    {
        // The caller's fragment inputs (RunDriver builds them once when
        // the SAD guess is requested); the engaged-optional invariant is
        // the contract this branch relies on.
        if (!atomicInputs.has_value())
        {
            return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                       "the SAD guess has no caller-built fragment inputs"));
        }

        auto sad = qcx::scf::BuildSadGuess(molecule, basis, *atomicInputs);

        if (!sad.has_value())
        {
            return std::unexpected(sad.error());
        }

        // The unpolarized SAD start (alpha = beta = the SAD average): the
        // Direct_uhf_test reference. The raw polarized densities
        // converge to a different near-degenerate O2 UHF fixed point
        // (Mayer 1.46 vs 2.0, a garbage dipole) that looks right in
        // energy but is not the pinned ground state.
        const Eigen::MatrixXd unpolarizedStart = 0.5 * (sad->densityAlpha + sad->densityBeta);
        options.initialDensityAlpha = unpolarizedStart;
        options.initialDensityBeta = unpolarizedStart;
    }

    const auto scfStarted = std::chrono::steady_clock::now();
    // The seam the wiring resolved - the RHF site's branch, per spin: a
    // Kohn-Sham run hands the loop three callbacks (the Coulomb one receives
    // the SPIN-SUMMED total, the contribution the per-spin pair), because the
    // Hartree-Fock trace identity does not survive a Fock whose exchange is
    // scaled by the functional's exact-exchange fraction. The presence of the
    // companions IS the distinction; nothing else could disagree with the
    // seam that actually runs.
    const bool isKsSeam = wired.coulomb.has_value() && wired.contribution.has_value();
    auto scf = isKsSeam ? qcx::scf::RunUhfScf(molecule,
                                              overlap,
                                              core,
                                              options,
                                              wired.fn,
                                              *wired.coulomb,
                                              *wired.contribution,
                                              &basis)
                        : qcx::scf::RunUhfScf(molecule, overlap, core, options, wired.fn, &basis);
    const auto scfStopped = std::chrono::steady_clock::now();

    if (!scf.has_value())
    {
        return std::unexpected(scf.error());
    }

    // A traced run that recorded nothing says so (see WarnOnEmptyTrace).
    // Gated on enableTrace: the Fukui charged-species runs pass false while
    // the input still carries the main run's trace_file, and they must not
    // warn about a trace they were never asked to write.
    if (enableTrace)
    {
        WarnOnEmptyTrace(input, statsStream, kind);
    }

    return UhfRun{std::move(*scf),
                  std::chrono::duration<double, std::milli>(scfStopped - scfStarted).count(),
                  std::move(wired.workspaceBudget),
                  wired.modeInfo,
                  wired.exchangeModeInfo,
                  // The selected arm's own QFMM model, when one ran (the
                  // composed-QFMM runner's two arms set it; every other UHF
                  // arm leaves it empty) - so the record's qfmm_model block
                  // stays absent on a route where no QFMM half ran rather
                  // than defaulting to a model nothing built.
                  wired.qfmmModel,
                  std::move(wired.riJkBuilder),
                  wired.orbitExpansionEngaged,
                  wired.orbitExpansionWired,
                  // The composed full-RI arm's own record, moved from the
                  // wired carrier for the reason above: only the ri_jk runner
                  // fills it, and the record reads the run rather than the
                  // selection - a run whose selection says ri_jk but whose arm
                  // never carried a builder would otherwise publish a rung
                  // decision that does not exist.
                  wired.riJkModeInfo};
}

// One direct-UHF run: the per-spin J/K split builders (the
// direct_uhf_test.cpp reference assembly), the guess, and the SCF loop.
// The main UHF branch and the Fukui analysis's charged-species runs share
// this path - the charged molecules only differ in charge/multiplicity
// (same geometry, basis, and core Hamiltonian). The SAD guess consumes
// the caller-built fragment inputs (atomicInputs; RunDriver builds them
// once - the optional is engaged exactly when the SAD guess is requested,
// and the charged Fukui species inherit the run's guess). UhfRun is
// declared with the forward declarations above (std::expected needs the
// complete type at the call site).
//
// Two arms, one per member of the direct family (the unrestricted seam): the
// lean arm when the resolution's leanMember holds - the SAME value the RHF
// wiring branch reads and the pre-gate setup admission consults, so the
// three can never disagree about which runs reach the lean member, an
// agreement that matters because the admission is sized by one member's
// number - and the machinery arm otherwise. The arm choice is the whole
// difference: both arms build a J-only and a K-only half, hand them to the
// matching seam adapter, and return through the shared tail.
qcx::Result<UhfRun> RunDirectUhfScf(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const qcx::backend::GpuComputeProfile& deviceComputeProfile,
    std::optional<bool> certifiedLaneRequest,
    bool leanMember,
    EriStoreWiring* eriStore,
    bool enableTrace) {
    // The per-call stats stream of the trace side-channel: the main
    // UHF branch only (enableTrace) - the Fukui charged-species runs are
    // property analyses and stay off the trace, so their loops cannot
    // overwrite the main run's trace file. Opened before the arm below,
    // because either arm's seam carries it.
    ScfCallStatsStream statsStream;

    if (enableTrace)
    {
        statsStream.Open(input.scf.traceFile);
    }

    const int electrons = molecule.ElectronCount();
    const int nAlpha = (electrons + molecule.Multiplicity() - 1) / 2;
    const int nBeta = electrons - nAlpha;

    // The run path this runner is serving, through the ONE classifier (the
    // same value the up-front check read and the dispatch acted on).
    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value())
    {
        return std::unexpected(scfPath.error());
    }

    const bool isKs = IsKohnShamPath(*scfPath);

    // The Kohn-Sham context, resolved BEFORE either arm creates a builder: a
    // pure functional (c_HF = 0) needs no exchange half at all, and in the
    // lean arm each half's Create is a last-resort admission decision - one
    // taken for a builder nothing would ever call is a refusal this run must
    // not incur. The context also carries the grid engine, which is built
    // once here and lives in the seam's shared_ptr for the run's duration.
    std::optional<KsContext> ksContext;

    if (isKs)
    {
        auto resolved = ResolveKsContext(input, molecule, basis);

        if (!resolved.has_value())
        {
            return std::unexpected(resolved.error());
        }

        ksContext = std::move(*resolved);
    }

    // Whether a K-only builder is needed at all: always on the Hartree-Fock
    // path (both halves ARE the Fock), and on the Kohn-Sham path only when
    // the functional carries exact exchange.
    const bool needsExchangeHalf = !isKs || ksContext->functional.exchangeFraction > 0.0;

    // The lean arm (the unrestricted seam): the SAME member choice the RHF
    // wiring branch reads and the pre-gate setup admission consults - the
    // resolution's ResolvedBuilderSelection::leanMember - which under the
    // 2026-09-13 ladder is the absent key at nBasis <= 1000, an explicit
    // fock_builder = "lean", or a run the ladder DEMOTED onto the member
    // (no-key UHF above the lean boundary: neither upper tier has a UHF
    // arm). The lean member's own Create-time last-resort ceiling is this
    // arm's admission (LeanFockBuildOptions::memoryCapGiB below), exactly as
    // it is on the RHF side: no modeled ladder, no workspace budget, no ERI
    // cache - and the two halves' ceilings are
    // the same number, because the split flags do not enter the envelope's
    // arithmetic (EstimatePeakBytes reads the preset and the window plan
    // alone). An explicit "direct" never reaches here: the family word
    // leaves the builder slot engaged and the ladder out of the resolution,
    // so leanMember is false by construction (Ruling B - the explicit
    // machinery spellings stay byte-stable).
    if (leanMember)
    {
        qcx::integrals::LeanFockBuildOptions leanCoulombOptions;
        leanCoulombOptions.accuracy = input.method.accuracy;
        leanCoulombOptions.buildCoulombOnly = true;
        leanCoulombOptions.memoryCapGiB = input.resources.memoryCapGiB;

        qcx::integrals::LeanFockBuildOptions leanExchangeOptions;
        leanExchangeOptions.accuracy = input.method.accuracy;
        leanExchangeOptions.buildExchangeOnly = true;
        leanExchangeOptions.memoryCapGiB = input.resources.memoryCapGiB;

        // Both halves are handed the SAME point-group classification and the
        // same orbit engagement, from ONE reduction built once for the pair:
        // the RHF arm's block documents the decision and its measurements in
        // full (c8h18/def2-SVP, C2h). The halves are separate builders over a
        // single group, so a half engaged without the other would put the two
        // Fock contributions on different summation orders for no reason.

        auto leanReduction = qcx::scf::BuildSymmetryReduction(molecule, basis);

        if (!leanReduction.has_value() &&
            leanReduction.error().code != qcx::ErrorCode::kUnimplemented)
        {
            return std::unexpected(leanReduction.error());
        }

        if (leanReduction.has_value() && leanReduction->groupOrder > 1)
        {
            leanCoulombOptions.symmetryReduction = &*leanReduction;
            leanCoulombOptions.symmetryOrbitExpansion = true;
            leanExchangeOptions.symmetryReduction = &*leanReduction;
            leanExchangeOptions.symmetryOrbitExpansion = true;
        }

        auto leanCoulomb = qcx::integrals::LeanDirectFockBuilder::Create(
            molecule, basis, coreTensor, leanCoulombOptions);

        if (!leanCoulomb.has_value())
        {
            return std::unexpected(leanCoulomb.error());
        }

        // The unrestricted Kohn-Sham arm: one Coulomb half on the half-summed
        // density and, when the functional carries exact exchange, one K-only
        // half serving BOTH spin channels (the builder is density-agnostic -
        // the same instance builds H - K[d] for either spin, exactly as the
        // Hartree-Fock split's single exchange builder does), each recorded
        // under its own channel's occupancy.
        if (isKs)
        {
            qcx::driver::internal::HalfFockFn coulombHalf =
                MakeLeanKsHalf(*leanCoulomb, &statsStream, 0);
            qcx::driver::internal::HalfFockFn exchangeHalfAlpha;
            qcx::driver::internal::HalfFockFn exchangeHalfBeta;

            if (needsExchangeHalf)
            {
                auto leanExchange = qcx::integrals::LeanDirectFockBuilder::Create(
                    molecule, basis, coreTensor, leanExchangeOptions);

                if (!leanExchange.has_value())
                {
                    return std::unexpected(leanExchange.error());
                }

                exchangeHalfAlpha =
                    MakeLeanKsHalf(*leanExchange, &statsStream, static_cast<std::size_t>(nAlpha));
                exchangeHalfBeta =
                    MakeLeanKsHalf(*leanExchange, &statsStream, static_cast<std::size_t>(nBeta));
            }

            auto seam = qcx::driver::internal::MakeUksSeam(coulombHalf,
                                                           exchangeHalfAlpha,
                                                           exchangeHalfBeta,
                                                           core,
                                                           ksContext->functional.exchangeFraction,
                                                           ksContext->evaluator);

            if (!seam.has_value())
            {
                return std::unexpected(seam.error());
            }

            WiredUhfBuilders wiredKs{std::move(seam->fock), nullptr, std::nullopt, std::nullopt};
            wiredKs.coulomb = std::move(seam->coulomb);
            wiredKs.contribution = std::move(seam->contribution);
            return RunUhfScfWithBuilder(molecule,
                                        basis,
                                        overlap,
                                        core,
                                        atomicInputs,
                                        input,
                                        enableTrace,
                                        nAlpha,
                                        nBeta,
                                        statsStream,
                                        std::move(wiredKs));
        }

        auto leanExchange = qcx::integrals::LeanDirectFockBuilder::Create(
            molecule, basis, coreTensor, leanExchangeOptions);

        if (!leanExchange.has_value())
        {
            return std::unexpected(leanExchange.error());
        }

        // The per-spin K occupancies ride the seam: the stats stream sizes
        // each recorded call's Q*O_occ volume with its K channel's
        // occupancy (the alpha exchange call contracts nAlpha occupied
        // orbitals, the beta call nBeta; the J-only call none).
        const auto leanBuilder = MakeLeanUhfFockBuilder(*leanCoulomb,
                                                        *leanExchange,
                                                        core,
                                                        &statsStream,
                                                        static_cast<std::size_t>(nAlpha),
                                                        static_cast<std::size_t>(nBeta));
        WiredUhfBuilders wiredLean{leanBuilder, nullptr, std::nullopt, std::nullopt};
        return RunUhfScfWithBuilder(molecule,
                                    basis,
                                    overlap,
                                    core,
                                    atomicInputs,
                                    input,
                                    enableTrace,
                                    nAlpha,
                                    nBeta,
                                    statsStream,
                                    std::move(wiredLean));
    }

    // The machinery arm: the direct family's budgeted member - the modeled
    // ladder against the cap, the ERI caches, the adaptive workspace
    // budget, the class-aware symmetry reduction. Reached by an explicit
    // fock_builder = "direct" at every size and by the absent key above the
    // lean ceiling.
    qcx::integrals::FockBuildOptions coulombOptions;
    coulombOptions.accuracy = input.method.accuracy;
    coulombOptions.buildCoulombOnly = true;
    // The run's device compute profile: both halves carry it, so
    // the certified fp32 lane's default is the same verdict on the coulomb
    // and the exchange build of one run - the profile is the caller's, read
    // once at the run boundary, never probed per builder instance.
    coulombOptions.deviceComputeProfile = deviceComputeProfile;
    // The certified fp32 lane's REQUEST (the owner's ruling 2026-09-13, the
    // [method] force_certified_lane key): both halves carry it for the same
    // reason the profile does - one verdict per run, never one per builder -
    // and because a forced lane that reached only one half would put the
    // coulomb and exchange builds of one Fock on different precision lanes.
    // Absent (nullopt) leaves the field unset, which is the probe's verdict;
    // the lean arm below builds LeanFockBuildOptions, which has no fp32 lane
    // at all, so a forced request never reaches here on that arm (the
    // resolution point refuses it by name before any builder is wired).
    coulombOptions.useCertifiedMixedPrecision = certifiedLaneRequest;

    qcx::integrals::FockBuildOptions exchangeOptions;
    exchangeOptions.accuracy = input.method.accuracy;
    exchangeOptions.buildExchangeOnly = true;
    exchangeOptions.deviceComputeProfile = deviceComputeProfile;
    exchangeOptions.useCertifiedMixedPrecision = certifiedLaneRequest;

    // The cap's leftover sizes the coulomb/exchange ERI caches below the
    // cap.
    const std::size_t cacheBytes = DirectFamilyCacheBytes(input.resources.memoryCapGiB);
    coulombOptions.maxCacheBytes = cacheBytes;
    exchangeOptions.maxCacheBytes = cacheBytes;

    // The disk-tier ERI store (`[method] eri_cache_store`) on the
    // unrestricted legs, and this is its one install site: the machinery
    // arm's two option structs, on the SAME grant path the cap-derived
    // cache budget above rides. The `RunDirectUhfScf` seam the older
    // refusal announced is this one - the per-spin coulomb and exchange
    // halves assemble their own FockBuildOptions here, so the restricted
    // site's install could not reach them.
    //
    // ONE factory, TWO halves, and the pairing is deliberate: the factory
    // is a copyable closure over ONE shared handle, so the first half's
    // Create opens the store and the second reuses it rather than opening a
    // second append-only writer over the same file (the `MakeEriStoreFactory`
    // contract, the same one the Kohn-Sham composition relies on).
    //
    // The install is BESIDE the cap-derived grant, never derived from it:
    // an explicit store request is not a cap-derived grant, so it is not
    // re-derived from `cacheBytes` and it does not move it. The two facts
    // the record needs are read from the options the builders actually
    // get: whether a factory was installed at all, and whether a positive
    // in-memory tier stands behind it.
    //
    // A store that cannot open costs the run a cache, never the run (the
    // owner's 2026-09-12 ruling): the factory returns the raw engine pair
    // unchanged and leaves its cause on the handle for the record to
    // disclose. The LEAN arm above never reaches here - it builds
    // LeanFockBuildOptions, which carries no engine pair to decorate - so a
    // lean run's request demotes through the record's own lean sentence,
    // exactly as the restricted member's does.
    const bool eriStoreRequested = !input.method.eriCacheStore.empty();

    if (eriStore != nullptr && eriStoreRequested && !leanMember)
    {
        auto auxName = AuxNameInEffect(input, qcx::io::BuilderKind::kDirect);

        if (!auxName.has_value())
        {
            return std::unexpected(auxName.error());
        }

        eriStore->handle = std::make_shared<EriStoreHandle>();
        // The path the input named is used as written, the restricted
        // site's rule and the same reason: the EriStore layer creates a
        // missing file and verifies the fingerprint of an existing one, so
        // a stale or foreign store at the path is a demotion the record
        // discloses rather than a run this site silently redirects.
        const qcx::integrals::EngineDecoratorFactory decorator =
            MakeEriStoreFactory(eriStore->handle,
                                molecule,
                                basis,
                                input.basis.orbital,
                                *auxName,
                                input.method.eriCacheStore);
        coulombOptions.engineDecorator = decorator;
        exchangeOptions.engineDecorator = decorator;
        // The tier that would serve an unengaged request, read from the
        // same options struct the builders get - the coulomb half's, since
        // the cap-derived grant above writes the same number into both.
        eriStore->ramTierInForce = coulombOptions.maxCacheBytes > 0;
    }

    // The adaptive seam, wired like the RHF direct branch:
    // the cap is the workspace budget shared by the coulomb and exchange
    // builders - their Create-time estimates decide the rung and reserve
    // against it (the cap acts as a ceiling, never a target; the
    // class-table admission gate rides the same budget). The cap-derived
    // cache grant stays - the engine's Create-time cache clamp re-verifies
    // it against the budget's remaining bytes ('s sure-fit gate). A
    // zero cap means "no input cap" and keeps the legacy null-budget path
    // byte-for-byte.
    std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;

    if (input.resources.memoryCapGiB > 0.0)
    {
        auto budget = qcx::memory::WorkspaceBudget::Create(
            WorkspaceGrantBytes(input.resources.memoryCapGiB, workspaceReserveBytes));

        if (!budget.has_value())
        {
            return std::unexpected(budget.error());
        }

        workspaceBudget = std::make_unique<qcx::memory::WorkspaceBudget>(std::move(*budget));
    }

    // The class-aware seam (the symmetry-blocking): the same reduction the RHF branch
    // wires - a realizable non-trivial group engages the class path, whose
    // Create-time table is admission-gated against the budget above (a
    // table that cannot fit disengages the class path; the plain screened
    // path runs - never the 0xC0000409 death).
    auto reduction = qcx::scf::BuildSymmetryReduction(molecule, basis);

    if (!reduction.has_value() && reduction.error().code != qcx::ErrorCode::kUnimplemented)
    {
        return std::unexpected(reduction.error());
    }

    coulombOptions.workspaceBudget = workspaceBudget.get();
    exchangeOptions.workspaceBudget = workspaceBudget.get();

    if (reduction.has_value() && reduction->groupOrder > 1)
    {
        coulombOptions.symmetryReduction = &*reduction;
        exchangeOptions.symmetryReduction = &*reduction;
    }

    auto coulomb =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basis, coreTensor, coulombOptions);

    if (!coulomb.has_value())
    {
        // On the budget path the engine's Create-time refusal is the
        // admission decision (its text is AM-owned, never edited): the
        // driver composes the standing reinstatement clause onto it so
        // the diagnostic carries the full ladder (the RHF branch's shape).
        if (workspaceBudget != nullptr)
        {
            return std::unexpected(
                Err(coulomb.error().code,
                    coulomb.error().message + qcx::driver::DirectReinstatementClause()));
        }

        return std::unexpected(coulomb.error());
    }

    // The K-only half: built on the Hartree-Fock path always (the two halves
    // ARE that Fock), and on the Kohn-Sham path only when the functional
    // carries exact exchange - a pure LDA/GGA run has c_HF = 0, and its
    // K-only builder must not be created at all, because its Create is an
    // admission decision against the shared budget that a run with no use for
    // the builder must not be made to pass.
    std::optional<qcx::integrals::DirectJkFockBuilder> exchange;

    if (needsExchangeHalf)
    {
        auto created = qcx::integrals::DirectJkFockBuilder::Create(
            molecule, basis, coreTensor, exchangeOptions);

        if (!created.has_value())
        {
            if (workspaceBudget != nullptr)
            {
                return std::unexpected(
                    Err(created.error().code,
                        created.error().message + qcx::driver::DirectReinstatementClause()));
            }

            return std::unexpected(created.error());
        }

        exchange = std::move(*created);
    }

    // The unrestricted Kohn-Sham arm: the same two split builders the
    // Hartree-Fock split below uses, composed instead of fused - the Coulomb
    // half's 2J on the half-summed density and the K-only half's per-spin
    // contraction, with the grid engine's per-spin potential and its Exc.
    if (isKs)
    {
        qcx::driver::internal::HalfFockFn coulombHalf =
            MakeDirectKsHalf(*coulomb, &statsStream, 0, nullptr);
        qcx::driver::internal::HalfFockFn exchangeHalfAlpha;
        qcx::driver::internal::HalfFockFn exchangeHalfBeta;

        if (needsExchangeHalf)
        {
            exchangeHalfAlpha = MakeDirectKsHalf(
                *exchange, &statsStream, static_cast<std::size_t>(nAlpha), nullptr);
            exchangeHalfBeta =
                MakeDirectKsHalf(*exchange, &statsStream, static_cast<std::size_t>(nBeta), nullptr);
        }

        auto seam = qcx::driver::internal::MakeUksSeam(coulombHalf,
                                                       exchangeHalfAlpha,
                                                       exchangeHalfBeta,
                                                       core,
                                                       ksContext->functional.exchangeFraction,
                                                       ksContext->evaluator);

        if (!seam.has_value())
        {
            return std::unexpected(seam.error());
        }

        const std::optional<qcx::integrals::FockModeInfo> exchangeMode =
            needsExchangeHalf ? std::optional<qcx::integrals::FockModeInfo>(exchange->ModeInfo())
                              : std::optional<qcx::integrals::FockModeInfo>{};
        WiredUhfBuilders wiredKs{
            std::move(seam->fock), std::move(workspaceBudget), coulomb->ModeInfo(), exchangeMode};
        wiredKs.coulomb = std::move(seam->coulomb);
        wiredKs.contribution = std::move(seam->contribution);
        return RunUhfScfWithBuilder(molecule,
                                    basis,
                                    overlap,
                                    core,
                                    atomicInputs,
                                    input,
                                    enableTrace,
                                    nAlpha,
                                    nBeta,
                                    statsStream,
                                    std::move(wiredKs));
    }

    // The per-spin K occupancies ride the seam: the stats stream sizes
    // each recorded call's Q*O_occ volume with its K channel's occupancy
    // (the alpha exchange call contracts nAlpha occupied orbitals, the
    // beta call nBeta; the J-only call none).
    const auto uhfBuilder = MakeDirectUhfFockBuilder(*coulomb,
                                                     *exchange,
                                                     core,
                                                     &statsStream,
                                                     static_cast<std::size_t>(nAlpha),
                                                     static_cast<std::size_t>(nBeta));
    WiredUhfBuilders wiredDirect{
        uhfBuilder, std::move(workspaceBudget), coulomb->ModeInfo(), exchange->ModeInfo()};
    return RunUhfScfWithBuilder(molecule,
                                basis,
                                overlap,
                                core,
                                atomicInputs,
                                input,
                                enableTrace,
                                nAlpha,
                                nBeta,
                                statsStream,
                                std::move(wiredDirect));
}

// One composed-QFMM UHF run: the per-spin J/K assembly with the
// QFMM builder in the coulomb slot - QfmmHfFockBuilder::BuildUhfFock runs
// the MakeDirectUhfFockBuilder math internally (one Coulomb call on the
// half-summed density 0.5 (P_a + P_b) through the QFMM near/far split,
// one exchange-only call per spin on the raw densities, one H copy
// subtracted back per channel - the composed builder owns both nested
// states, so no seam-side split wiring exists here). The guess handling
// is the direct run's verbatim: the pinned O2 ground state is reached
// from the unpolarized SAD start (the direct_uhf_test reference).
// The adaptive seam is wired like the direct runner's: the cap is the
// workspace budget shared with the composed builder's two nested Creates
// (each half's Create-time estimate decides its own rung against the
// shared counter and reserves - the QFMM half first, nesting order =
// reservation order), and the returned run carries the budget plus both
// halves' mode records. The job-object cap is the admission.
qcx::Result<UhfRun> RunQfmmUhfScf(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes) {
    // The run path, resolved ONCE and read by the Kohn-Sham arm below (the
    // one run path this runner serves two of). This runner used to refuse a
    // Kohn-Sham run outright, with the reason that the composed builder
    // builds its Coulomb half inside the near/far split and could not expose
    // the J[D] the energy seam contracts separately. That clause is answered
    // by construction now (QfmmHfFockBuilder::BuildCoulombOnly - the same
    // nested call the fused Fock opens with), so the refusal is gone and the
    // arm below runs. The shape the refusal protected against - a Kohn-Sham
    // request falling through to the fused Hartree-Fock seam - is held by
    // two things instead: the arm below RETURNS for a Kohn-Sham path, before
    // the fused seam is constructed, and a caller that reached this runner
    // another way still passes ValidateCombination's own whitelist.
    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value())
    {
        return std::unexpected(scfPath.error());
    }

    // The QfmmOptions of the input: the accuracy preset plus the four
    // schema knobs, exactly the composition of the RHF kQfmm branch of
    // WireRhfFockBuilder - an absent key keeps the engine default (the io
    // parser already validated the schema key's contract by name).
    qcx::integrals::QfmmOptions qfmmOptions;
    qfmmOptions.accuracy = input.method.accuracy;

    if (input.method.theta.has_value())
    {
        qfmmOptions.theta = *input.method.theta;
    }

    if (input.method.lMult.has_value())
    {
        qfmmOptions.lMult = *input.method.lMult;
    }

    if (input.method.maxLeafSize.has_value())
    {
        qfmmOptions.maxLeafSize = *input.method.maxLeafSize;
    }

    if (input.method.crossoverBasisFunctionCount.has_value())
    {
        qfmmOptions.crossoverBasisFunctionCount = *input.method.crossoverBasisFunctionCount;
    }

    // The adaptive seam (the budget path), wired like the direct runner above: the
    // cap is the workspace budget shared with the composed builder - each
    // of its nested Creates decides its own rung against the shared
    // counter and reserves what it needs (the QFMM half's estimate first,
    // the exchange half's second - nesting order = reservation order), and
    // the cap acts as a ceiling, never a target. A zero cap means "no
    // input cap" and keeps the legacy null-budget path byte-for-byte.
    std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;

    if (input.resources.memoryCapGiB > 0.0)
    {
        auto budget = qcx::memory::WorkspaceBudget::Create(
            WorkspaceGrantBytes(input.resources.memoryCapGiB, workspaceReserveBytes));

        if (!budget.has_value())
        {
            return std::unexpected(budget.error());
        }

        workspaceBudget = std::make_unique<qcx::memory::WorkspaceBudget>(std::move(*budget));
    }

    qfmmOptions.workspaceBudget = workspaceBudget.get();

    auto builder =
        qcx::integrals::QfmmHfFockBuilder::Create(molecule, basis, coreTensor, qfmmOptions);

    if (!builder.has_value())
    {
        // On the budget path the engine's Create-time refusal is the
        // admission decision (its text is AM-owned, never edited): the
        // driver composes the standing reinstatement clause onto it so
        // the diagnostic carries the full ladder (the direct runner's
        // shape).
        if (workspaceBudget != nullptr)
        {
            return std::unexpected(
                Err(builder.error().code,
                    builder.error().message + qcx::driver::DirectReinstatementClause()));
        }

        return std::unexpected(builder.error());
    }

    // The unrestricted Kohn-Sham arm: the composition
    // over the same two halves the fused build adds, with the per-spin
    // conventions the unrestricted seam's own (MakeUksSeam's note) - ONE
    // Coulomb call on the half-summed density, whose H + 2 J_QFMM(0.5 (d_a +
    // d_b)) is H + J_QFMM[D_total], and one exchange call per spin on that
    // spin's RAW density, whose H - K(d_s) is the spin channel's -K half.
    // Both halves are CUT from the builder created above, so the J the energy
    // seam contracts is that builder's own at the loop's density.
    //
    // No core-Hamiltonian bookkeeping rides this arm, and that is the
    // family's own fact (QfmmKsHalf's enum): the composed builder's two
    // nested results are already the composition's two conventions, each
    // carrying one full H copy - which is exactly the pair MakeUksSeam's
    // arithmetic expects. The RI-J link's unrestricted arm beside this one
    // adds H into its Coulomb half; here that would move both channels by an
    // entire core Hamiltonian.
    //
    // The per-call stats stream: no half of this family backs a per-call
    // row, so the stream below is created and never opened - a traced UKS
    // run on this cell records nothing, and the empty-trace warning says so
    // rather than leaving a zero count to be read as a wiring defect (the
    // composed-QFMM UHF arm's own documented surface).
    const int ksElectrons = molecule.ElectronCount();
    const int ksNAlpha = (ksElectrons + molecule.Multiplicity() - 1) / 2;
    const int ksNBeta = ksElectrons - ksNAlpha;

    if (IsKohnShamPath(*scfPath))
    {
        auto ksContext = ResolveKsContext(input, molecule, basis);

        if (!ksContext.has_value())
        {
            return std::unexpected(ksContext.error());
        }

        qcx::driver::internal::HalfFockFn coulombHalf =
            MakeQfmmKsHalf(*builder, QfmmKsHalf::kCoulomb);
        qcx::driver::internal::HalfFockFn exchangeHalfAlpha;
        qcx::driver::internal::HalfFockFn exchangeHalfBeta;

        if (ksContext->functional.exchangeFraction > 0.0)
        {
            exchangeHalfAlpha = MakeQfmmKsHalf(*builder, QfmmKsHalf::kExchange);
            exchangeHalfBeta = MakeQfmmKsHalf(*builder, QfmmKsHalf::kExchange);
        }

        auto seam = qcx::driver::internal::MakeUksSeam(coulombHalf,
                                                       exchangeHalfAlpha,
                                                       exchangeHalfBeta,
                                                       core,
                                                       ksContext->functional.exchangeFraction,
                                                       ksContext->evaluator);

        if (!seam.has_value())
        {
            return std::unexpected(seam.error());
        }

        ScfCallStatsStream statsStream;
        WiredUhfBuilders wiredKs{std::move(seam->fock), nullptr, std::nullopt, std::nullopt};
        wiredKs.coulomb = std::move(seam->coulomb);
        wiredKs.contribution = std::move(seam->contribution);
        // The same carriers the Hartree-Fock arm below returns, plus the
        // model record: this run's Coulomb half IS that QFMM half, so the
        // record's `qfmm_model` block states the same engine resolution an
        // RHF run on this family states .
        wiredKs.workspaceBudget = std::move(workspaceBudget);
        wiredKs.modeInfo = builder->ModeInfo();
        wiredKs.exchangeModeInfo = builder->ExchangeModeInfo();
        wiredKs.qfmmModel = builder->ModelRecord();
        return RunUhfScfWithBuilder(molecule,
                                    basis,
                                    overlap,
                                    core,
                                    atomicInputs,
                                    input,
                                    // This runner is the main UHF run's route
                                    // only (the Fukui charged-species runs go
                                    // to RunDirectUhfScf), so tracing is on:
                                    // the literal the two sibling call sites at
                                    // the dispatch pass for the same reason.
                                    true,
                                    ksNAlpha,
                                    ksNBeta,
                                    statsStream,
                                    std::move(wiredKs),
                                    qcx::io::BuilderKind::kQfmm);
    }

    const auto uhfBuilder = MakeQfmmUhfFockBuilder(*builder);

    qcx::scf::UhfOptions options;
    options.maxIterations = input.scf.maxIterations;
    options.energyTolerance = input.scf.energyTolerance;
    options.densityTolerance = input.scf.densityTolerance;
    options.useDiis = input.scf.useDiis;
    options.traceFile = input.scf.traceFile;
    options.fullGroupLabeling = input.symmetry.fullGroup;

    const int electrons = molecule.ElectronCount();
    const int nAlpha = (electrons + molecule.Multiplicity() - 1) / 2;
    const int nBeta = electrons - nAlpha;

    // kCore resolves HERE, to the same GWH seed kGwh asks for by name (both
    // unrestricted seed sites - the machinery arm and the lean arm below -
    // carry this identically). The unfixed form left the unrestricted start
    // UNSET, and `core` means "the method's default start", which the
    // RHF path already resolves to GWH: the asymmetry was the defect.
    //
    // Measured on O2/STO-3G at 1.2075 A (2.2818443 Bohr), the ruled
    // defaults, after the case was cleared of a coordinator unit error: the
    // unset start landed -147.37855917618 - the saddle, 0.2536 Ha ABOVE the
    // ROHF bound -147.63216699080 - in 37 iterations, while GWH and SAD
    // both reach the pyscf UHF value -147.63394678556 (2.7e-8) in 23 and 34
    // iterations. OH/STO-3G is unaffected by either (2e-8 from pyscf).
    // Owner ruling 2026-09-15: use our good defaults rather than an
    // inferior start chosen to match a reference.
    if (input.guess == qcx::io::GuessKind::kGwh || input.guess == qcx::io::GuessKind::kCore)
    {
        auto gwh = qcx::scf::BuildGwhGuess(overlap, core, nAlpha, nBeta);

        if (!gwh.has_value())
        {
            return std::unexpected(gwh.error());
        }

        options.initialDensityAlpha = gwh->first;
        options.initialDensityBeta = gwh->second;
    } else if (input.guess == qcx::io::GuessKind::kSad)
    {
        // The caller's fragment inputs (RunDriver builds them once when
        // the SAD guess is requested); the engaged-optional invariant is
        // the contract this branch relies on.
        if (!atomicInputs.has_value())
        {
            return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                       "the SAD guess has no caller-built fragment inputs"));
        }

        auto sad = qcx::scf::BuildSadGuess(molecule, basis, *atomicInputs);

        if (!sad.has_value())
        {
            return std::unexpected(sad.error());
        }

        // The unpolarized SAD start (alpha = beta = the SAD average): the
        // Direct_uhf_test reference, shared verbatim with the direct
        // run - the raw polarized densities converge to a different
        // near-degenerate O2 UHF fixed point that is not the pinned
        // ground state.
        const Eigen::MatrixXd unpolarizedStart = 0.5 * (sad->densityAlpha + sad->densityBeta);
        options.initialDensityAlpha = unpolarizedStart;
        options.initialDensityBeta = unpolarizedStart;
    }

    const auto scfStarted = std::chrono::steady_clock::now();
    auto scf = qcx::scf::RunUhfScf(molecule, overlap, core, options, uhfBuilder, &basis);
    const auto scfStopped = std::chrono::steady_clock::now();

    if (!scf.has_value())
    {
        return std::unexpected(scf.error());
    }

    return UhfRun{std::move(*scf),
                  std::chrono::duration<double, std::milli>(scfStopped - scfStarted).count(),
                  std::move(workspaceBudget),
                  builder->ModeInfo(),
                  builder->ExchangeModeInfo(),
                  // The composed builder's own Coulomb-half record (schema
                  // 29, the engine's resolution - the theta gate and the
                  // preset's theta both resolve inside the engine, so this
                  // is a read and not a recomputation).
                  builder->ModelRecord()};
}

// One unrestricted RI-J-link run (the per-spin adapter of the ri_j_link
// family): F_sigma = H + J_RI(P_alpha + P_beta) - K(P_sigma), the
// MakeDirectUhfFockBuilder assembly run over the RI builder's two
// exposed halves - one Coulomb-only call on the half-summed density and one
// exchange-only call per spin, summed per channel with no core-Hamiltonian
// subtraction (MakeRiJLinkUhfFockBuilder above states that accounting and
// why it is not the direct family's). The guess handling, the loop, the
// empty-trace warning and the UhfRun assembly are the shared UHF tail's
// (RunUhfScfWithBuilder) - this runner's own work is the RI family's
// admission: the aux basis in effect, the cap-minus-base workspace budget
// the engine's Create-time estimate decides against, the builder, and the
// seam.
//
// What this runner does NOT wire is the family's disk rungs
// (method.ri_tensor_mode = "disk", [diagnostics] force_disk_ri): the
// storage-module disk-backed store has no per-spin adapter, so the
// unrestricted leg has no disk route at all, and the request is refused BY
// NAME here (and up front in ValidateCombination) rather than dropped -
// the no-silent-substitution rule, and the reason the ladder keeps a
// no-key unrestricted run off this tier when either key is named
// (LadderRunnabilityFor states the same fact as a demotion reason). The
// Kohn-Sham lanes' own disk refusal is the same rule with a different reason
// (no half for the energy seam, RiJLinkKsDiskRungRefusalText), stated here on
// the same keys.
qcx::Result<UhfRun> RunRiJLinkUhfScf(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const std::filesystem::path& root,
    bool enableTrace) {
    // A Kohn-Sham run IS admitted here: this runner owns the
    // unrestricted RI-J link's Kohn-Sham arm below, over the same two halves
    // the Hartree-Fock assembly uses. The refusal this comment used to carry
    // ("RunDirectUhfScf is the only Kohn-Sham wiring") went with the RI-J
    // composition: what it protected against - a fused Hartree-Fock Fock
    // reaching a Kohn-Sham seam - is now excluded by the arm's own shape
    // rather than by refusing the run.
    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value())
    {
        return std::unexpected(scfPath.error());
    }

    const bool isKs = IsKohnShamPath(*scfPath);

    // The disk rungs of this family, refused by name on this leg: the
    // request names a route the unrestricted wiring does not have, and a
    // run that dropped it would be the silent substitution the disclosure rule forbids
    // (the rung selector and the diagnostic force are two different keys
    // onto the same missing adapter).
    //
    // TWO texts, because there are two reasons and this leg can be either
    // run: the UHF leg has no per-spin adapter for the store, and the UKS leg
    // has no half for its energy seam. A refusal that named the wrong one
    // would send its reader after a mechanism the run never reached.
    if (input.method.riTensorMode == qcx::io::RiTensorMode::kDisk || input.diagnostics.forceDiskRi)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kUnimplemented,
                isKs ? RiJLinkKsDiskRungRefusalText() : RiJLinkDiskRungRefusalText()));
    }

    // The auxiliary basis IN EFFECT (the RHF ri_j_link arm's rule, verbatim):
    // [basis].aux when the input gives one, else the auto-selected default
    // for this orbital basis - the same name the aux-in-effect disclosure
    // and the checkpoint fingerprint read.
    std::optional<std::string> auxName;

    if (input.basis.aux.has_value())
    {
        auxName = input.basis.aux;
    } else
    {
        auto selected = qcx::integrals::SelectAuxBasis(input.basis.orbital,
                                                       qcx::integrals::FockBuilderKind::kDefault);

        if (!selected.has_value())
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "no aux basis for \"" + input.basis.orbital +
                                           "\"; specify [basis].aux explicitly"));
        }

        auxName = *selected;
    }

    // The filtered parse: keep only the molecule's
    // elements, so the parsed aux set and the engine's molecule-scoped aux
    // count are equal by construction - the memory model's nAux is then the
    // engine's nAux, no re-scan.
    auto auxBasis = qcx::basisset::ParseNwchemDirectoryFiltered((root / *auxName).string(),
                                                                UniqueAtomicNumbers(molecule));

    if (!auxBasis.has_value())
    {
        return std::unexpected(auxBasis.error());
    }

    // The ri_j admission, delegated to the engine's Create-time estimate
    // (the budget path) exactly as on the RHF leg: the driver grants the cap as the
    // workspace budget and the engine picks the rung its estimate fits.
    // memory_cap_gib = 0 (the escape hatch) takes the legacy null-budget
    // path byte-for-byte.
    std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;

    if (input.resources.memoryCapGiB > 0.0)
    {
        auto budget = qcx::memory::WorkspaceBudget::Create(
            WorkspaceGrantBytes(input.resources.memoryCapGiB, workspaceReserveBytes));

        if (!budget.has_value())
        {
            return std::unexpected(budget.error());
        }

        workspaceBudget = std::make_unique<qcx::memory::WorkspaceBudget>(std::move(*budget));
    }

    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = input.method.accuracy;
    // The Coulomb half is spin-independent (the J of the run is one
    // contraction on the spin-summed density), so the orbit expansion
    // reduces the same task grid an RHF run reduces - the RHF arm's block,
    // verbatim, with its own outcome read back from the engine below.
    bool wiredOrbitExpansion = false;

    std::optional<qcx::integrals::SymmetryReduction> orbitOrbitalReduction;
    std::optional<qcx::integrals::SymmetryReduction> orbitAuxReduction;

    if (input.method.riOrbitExpansion.value_or(false))
    {
        auto orbitalReduction = qcx::scf::BuildSymmetryReduction(molecule, basis);

        if (!orbitalReduction.has_value())
        {
            return std::unexpected(orbitalReduction.error());
        }

        auto auxReduction = qcx::scf::BuildSymmetryReduction(molecule, *auxBasis);

        if (!auxReduction.has_value())
        {
            return std::unexpected(auxReduction.error());
        }

        orbitOrbitalReduction = std::move(*orbitalReduction);
        orbitAuxReduction = std::move(*auxReduction);
        riOptions.symmetryReduction = &*orbitOrbitalReduction;
        riOptions.auxSymmetryReduction = &*orbitAuxReduction;
        // Set explicitly rather than left to the default: the request is
        // what this key is (the RHF arm's reasoning).
        riOptions.symmetryOrbitExpansion = true;
        wiredOrbitExpansion = true;
    }

    riOptions.workspaceBudget = workspaceBudget.get();

    auto builder =
        qcx::integrals::RiJkFockBuilder::Create(molecule, basis, *auxBasis, coreTensor, riOptions);

    if (!builder.has_value())
    {
        // On the budget path the engine's Create-time refusal is the
        // admission decision (its text is AM-owned, never edited): the
        // driver composes the standing reinstatement clause onto it so the
        // diagnostic carries the full ladder (the RHF arm's shape).
        if (workspaceBudget != nullptr)
        {
            return std::unexpected(
                Err(builder.error().code, builder.error().message + RiJReinstatementClause()));
        }

        return std::unexpected(builder.error());
    }

    // The per-call stats stream of the trace side-channel, opened the
    // direct runner's way: the seam records one row per half-call (three per
    // SCF iteration: the J half under occupancy 0, each exchange half under
    // its spin's), so an instrumented unrestricted ri_j_link run lands the
    // row family its direct twin lands rather than an empty file.
    ScfCallStatsStream statsStream;

    if (enableTrace)
    {
        statsStream.Open(input.scf.traceFile);
    }

    const int electrons = molecule.ElectronCount();
    const int nAlpha = (electrons + molecule.Multiplicity() - 1) / 2;
    const int nBeta = electrons - nAlpha;

    // The unrestricted Kohn-Sham arm: the composition over the same
    // two halves, with the per-spin conventions the unrestricted seam's own
    // (MakeUksSeam's note) - ONE Coulomb call on the half-summed density,
    // whose 2 J_RI(0.5 (d_a + d_b)) is J_RI[D_total], and one exchange call
    // per spin on that spin's RAW density, whose H - K(d_s) is the spin
    // channel's -K half.
    //
    // The Coulomb half carries the H the family's contraction omits
    // (MakeRiJLinkKsHalf's enum states the accounting), and it carries the
    // SAME H object the seam is created with - the composition recovers the
    // seam's J by subtracting that object back out, so a different matrix
    // there would leave a residue in the number the energy formula
    // contracts.
    if (isKs)
    {
        auto ksContext = ResolveKsContext(input, molecule, basis);

        if (!ksContext.has_value())
        {
            return std::unexpected(ksContext.error());
        }

        qcx::driver::internal::HalfFockFn coulombHalf =
            MakeRiJLinkKsHalf(*builder, core, RiJLinkKsHalf::kCoulomb, &statsStream, 0);
        qcx::driver::internal::HalfFockFn exchangeHalfAlpha;
        qcx::driver::internal::HalfFockFn exchangeHalfBeta;

        if (ksContext->functional.exchangeFraction > 0.0)
        {
            exchangeHalfAlpha = MakeRiJLinkKsHalf(*builder,
                                                  core,
                                                  RiJLinkKsHalf::kExchange,
                                                  &statsStream,
                                                  static_cast<std::size_t>(nAlpha));
            exchangeHalfBeta = MakeRiJLinkKsHalf(*builder,
                                                 core,
                                                 RiJLinkKsHalf::kExchange,
                                                 &statsStream,
                                                 static_cast<std::size_t>(nBeta));
        }

        auto seam = qcx::driver::internal::MakeUksSeam(coulombHalf,
                                                       exchangeHalfAlpha,
                                                       exchangeHalfBeta,
                                                       core,
                                                       ksContext->functional.exchangeFraction,
                                                       ksContext->evaluator);

        if (!seam.has_value())
        {
            return std::unexpected(seam.error());
        }

        WiredUhfBuilders wiredKs{std::move(seam->fock), nullptr, std::nullopt, std::nullopt};
        wiredKs.coulomb = std::move(seam->coulomb);
        wiredKs.contribution = std::move(seam->contribution);
        // The ri_j_link carriers, exactly as the Hartree-Fock arm below sets
        // them: the same family on the same builder owes the record the same
        // budget, engine decision, counter source and orbit expansion pair.
        wiredKs.workspaceBudget = std::move(workspaceBudget);
        wiredKs.modeInfo = builder->ModeInfo();
        wiredKs.riJkBuilder = *builder;
        wiredKs.orbitExpansionEngaged = builder->OrbitExpansionEngaged();
        wiredKs.orbitExpansionWired = wiredOrbitExpansion;
        return RunUhfScfWithBuilder(molecule,
                                    basis,
                                    overlap,
                                    core,
                                    atomicInputs,
                                    input,
                                    enableTrace,
                                    nAlpha,
                                    nBeta,
                                    statsStream,
                                    std::move(wiredKs),
                                    qcx::io::BuilderKind::kRiJLink);
    }

    // The per-spin K occupancies ride the seam: the stats stream sizes each
    // recorded exchange call's Q*O_occ volume with its own channel's
    // occupancy, exactly as the direct arm's adapter does.
    const auto uhfBuilder = MakeRiJLinkUhfFockBuilder(
        *builder, &statsStream, static_cast<std::size_t>(nAlpha), static_cast<std::size_t>(nBeta));
    WiredUhfBuilders wired{
        uhfBuilder, std::move(workspaceBudget), builder->ModeInfo(), std::nullopt};
    // The calibration counter carrier: the run flow reads the run's accumulated term
    // counters through this copy after the SCF loop (it shares the seam
    // lambda's builder state - same-run counters), the RHF leg's own read.
    wired.riJkBuilder = *builder;
    // The engine's own disclosure, read here and nowhere recomputed (the RHF
    // arm's rule): the record's ri_orbit_expansion block reports this and the
    // request the input named, and the two cannot drift apart because the
    // outcome is this call.
    wired.orbitExpansionEngaged = builder->OrbitExpansionEngaged();
    wired.orbitExpansionWired = wiredOrbitExpansion;
    return RunUhfScfWithBuilder(molecule,
                                basis,
                                overlap,
                                core,
                                atomicInputs,
                                input,
                                enableTrace,
                                nAlpha,
                                nBeta,
                                statsStream,
                                std::move(wired),
                                qcx::io::BuilderKind::kRiJLink);
}

// One unrestricted composed-full-RI run (the ri_jk family's per-spin
// adapter): F_sigma = H + J_RI(P_tot) - K(P_sigma), the
// MakeRiFullUhfFockBuilder assembly over RiFullFockBuilder::BuildFockHalves -
// ONE call per spin, each contracting a Coulomb half and an exchange half -
// with the guess handling, the loop, the empty-trace warning and the UhfRun
// assembly left to the shared UHF tail. What this runner owns is the family's
// admission, which is the RESTRICTED arm's, verbatim: the JK-fit aux in
// effect and its quality refusal, the cap-minus-base workspace budget the
// engine's Create-time estimate decides its rung against, the builder, and
// the seam.
//
// TWO things it inherits by construction rather than re-deciding, both
// verified rather than assumed:
//   - the point-group reduction refusal. A named method.ri_orbit_expansion
//     is CARRIED into the options (the restricted arm's rule), so the
//     builder's own Create-time refusal - it names the two fields it cannot
//     honour - is reachable from this leg too, rather than the request being
//     dropped into a plain walk under a request that asked for a mechanism
//     .
//   - the RI-K accuracy disclosure. It is emitted once, in RunDriver, gated on
//     the RESOLVED builder kind and never on the SCF leg, so this runner owes
//     it nothing and cannot state a different one - and both characterised
//     cells of that disclosure are closed-shell water, which a reader of an
//     unrestricted run's record has to know: see the open item recorded
//     beside the unrestricted fixtures.
//
// The Kohn-Sham lanes were refused here BY NAME as a deliberate SECOND sender
// for one cell, so that lifting the combination rule alone would be a loud
// failure rather than a Hartree-Fock Fock under a Kohn-Sham label (a caller
// reaching this runner with no Coulomb/contribution companions falls through to
// the shared tail, and that tail runs the plain unrestricted Hartree-Fock loop).
// The rule has since been lifted for real, and the guard is now the KS arm
// below rather than a refusal: a UKS run resolves, builds the composition over
// this runner's own builder, and hands the loop the three callbacks - the loud
// failure the guard existed for is replaced by the thing it was guarding for.
qcx::Result<UhfRun> RunRiJkUhfScf(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& core,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const std::filesystem::path& root,
    bool enableTrace) {
    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value())
    {
        return std::unexpected(scfPath.error());
    }

    // The Kohn-Sham leg is served by this runner too, and by the SAME
    // machinery down to the builder: the family's admission (the JK-fit aux
    // and the standing quality ruling), the budget and its rung, the point-group carrier and the
    // created builder are identical, and what differs is the seam built at the
    // end - the composition over the pair entry point instead of the per-spin
    // assembly (the KS arm below).
    const bool isKs = IsKohnShamPath(*scfPath);

    // The aux IN EFFECT, resolved the way the restricted arm resolves it for
    // this family - the JK KIND, not the RI-J link's default fit. The KIND is
    // the whole point: def2-* maps to def2-universal-jkfit here and to
    // def2-universal-jfit for ri_j_link, so a runner that asked for the
    // default fit would run the wrong auxiliary basis under an ri_jk record.
    const auto auxName = AuxNameInEffect(input, qcx::io::BuilderKind::kRiJk);

    if (!auxName.has_value())
    {
        return std::unexpected(auxName.error());
    }

    // The standing quality ruling at the site that RESOLVES the name (the restricted arm's
    // check, read from the predicate's own message so the two cannot drift):
    // every cc-*/aug-cc-* base auto-selects a -rifit, which is a J-only fit
    // and cannot serve this family's exact-exchange half.
    if (!qcx::integrals::IsJkOptimizedAux(*auxName))
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, qcx::integrals::JkOptimizedAuxRefusal(*auxName)));
    }

    auto auxBasis = qcx::basisset::ParseNwchemDirectoryFiltered((root / *auxName).string(),
                                                                UniqueAtomicNumbers(molecule));

    if (!auxBasis.has_value())
    {
        return std::unexpected(auxBasis.error());
    }

    // The budget path (the restricted arm's, lane for lane): the driver
    // grants the cap and the ENGINE's Create-time estimate picks the rung -
    // fast when the raw tensor and its transform fit, blocked otherwise.
    // memory_cap_gib = 0 (the escape hatch) takes the legacy null-budget
    // path byte-for-byte.
    std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;

    if (input.resources.memoryCapGiB > 0.0)
    {
        auto budget = qcx::memory::WorkspaceBudget::Create(
            WorkspaceGrantBytes(input.resources.memoryCapGiB, workspaceReserveBytes));

        if (!budget.has_value())
        {
            return std::unexpected(budget.error());
        }

        workspaceBudget = std::make_unique<qcx::memory::WorkspaceBudget>(std::move(*budget));
    }

    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = input.method.accuracy;
    riOptions.workspaceBudget = workspaceBudget.get();

    // The point-group reduction CARRIER, carried rather than dropped for the
    // restricted arm's reason (see this runner's note above): the builder
    // refuses the pair BY NAME, and supplying it is what keeps that refusal
    // reachable from a run.
    std::optional<qcx::integrals::SymmetryReduction> orbitOrbitalReduction;
    std::optional<qcx::integrals::SymmetryReduction> orbitAuxReduction;

    if (input.method.riOrbitExpansion.value_or(false))
    {
        auto orbitalReduction = qcx::scf::BuildSymmetryReduction(molecule, basis);

        if (!orbitalReduction.has_value())
        {
            return std::unexpected(orbitalReduction.error());
        }

        auto auxReduction = qcx::scf::BuildSymmetryReduction(molecule, *auxBasis);

        if (!auxReduction.has_value())
        {
            return std::unexpected(auxReduction.error());
        }

        orbitOrbitalReduction = std::move(*orbitalReduction);
        orbitAuxReduction = std::move(*auxReduction);
        riOptions.symmetryReduction = &*orbitOrbitalReduction;
        riOptions.auxSymmetryReduction = &*orbitAuxReduction;
        riOptions.symmetryOrbitExpansion = true;
    }

    auto builder = qcx::integrals::RiFullFockBuilder::Create(
        molecule, basis, *auxBasis, coreTensor, riOptions);

    if (!builder.has_value())
    {
        // The builder's own refusal rides out verbatim, as on the restricted
        // arm: it is AM-owned, and its text already names the field it cannot
        // honour (the reduction) or the rung it cannot take. It needs no
        // reinstatement composition the way the ri_j_link arm's does.
        return std::unexpected(builder.error());
    }

    // The per-call stats stream, opened the sibling runners' way: the seam
    // records ONE row per spin (see MakeRiFullUhfFockBuilder), so an
    // instrumented unrestricted ri_jk run lands rows rather than an empty
    // file. That is the whole of what an instrumented run of this family
    // carries - its calibration counters are NOT published through the ri_j_link
    // block's vocabulary, and the WiredUhfBuilders member states why.
    ScfCallStatsStream statsStream;

    if (enableTrace)
    {
        statsStream.Open(input.scf.traceFile);
    }

    const int electrons = molecule.ElectronCount();
    const int nAlpha = (electrons + molecule.Multiplicity() - 1) / 2;
    const int nBeta = electrons - nAlpha;

    // The unrestricted Kohn-Sham arm: the composition over the same pair
    // entry point, with the per-spin conventions the unrestricted seam's own
    // (MakeUksSeam's note) - ONE Coulomb call on the half-summed density,
    // whose 2 J_RI(0.5 (d_a + d_b)) is J_RI[D_total], and one exchange call
    // per spin on that spin's RAW density, whose H - K(d_s) is the spin
    // channel's -K half. Both are the builder's own density conventions on
    // this family (BuildFockHalves reads the spatial rho = D/2, and the
    // per-spin pair adapter passes each spin's raw density to the same entry
    // point), so the arguments need no rescaling here either.
    //
    // The conversion is MakeRiFullKsHalf's and the same on both lanes: the
    // pair's bare halves reach the composition with H added to each and the
    // Coulomb one doubled. The H object is shared with the seam below, because
    // the composition recovers the seam's J by subtracting that exact matrix
    // back out of the Coulomb half.
    if (isKs)
    {
        auto ksContext = ResolveKsContext(input, molecule, basis);

        if (!ksContext.has_value())
        {
            return std::unexpected(ksContext.error());
        }

        // ONE cache for the whole arm's halves, consulted by the Coulomb half
        // and both per-spin exchange halves through the same exact key
        // (RiFullKsPairCache). Where the spins coincide the three asks
        // coincide with them and the pair is built once - MEASURED on the
        // H2O/STO-3G b3lyp deck: 21 rows before this memo against 11 after,
        // the first five fills serving all three asks from one build each and
        // the last two, the spins no longer bit-equal, building all three.
        // Where they differ the cache misses and costs only the entry's copy.
        // It is shared rather than scoped per half because the KEY is what
        // makes a hit safe (same density AND same occupied count), so a
        // narrower scope would only hide that this arm's degenerate pair is
        // already served by it.
        const auto pairCache = std::make_shared<RiFullKsPairCache>();
        qcx::driver::internal::HalfFockFn coulombHalf =
            MakeRiFullKsHalf(*builder,
                             core,
                             RiFullKsHalf::kCoulomb,
                             static_cast<std::size_t>(nAlpha),
                             pairCache,
                             &statsStream);
        qcx::driver::internal::HalfFockFn exchangeHalfAlpha;
        qcx::driver::internal::HalfFockFn exchangeHalfBeta;

        if (ksContext->functional.exchangeFraction > 0.0)
        {
            exchangeHalfAlpha = MakeRiFullKsHalf(*builder,
                                                 core,
                                                 RiFullKsHalf::kExchange,
                                                 static_cast<std::size_t>(nAlpha),
                                                 pairCache,
                                                 &statsStream);
            exchangeHalfBeta = MakeRiFullKsHalf(*builder,
                                                core,
                                                RiFullKsHalf::kExchange,
                                                static_cast<std::size_t>(nBeta),
                                                pairCache,
                                                &statsStream);
        }

        auto seam = qcx::driver::internal::MakeUksSeam(coulombHalf,
                                                       exchangeHalfAlpha,
                                                       exchangeHalfBeta,
                                                       core,
                                                       ksContext->functional.exchangeFraction,
                                                       ksContext->evaluator);

        if (!seam.has_value())
        {
            return std::unexpected(seam.error());
        }

        WiredUhfBuilders wiredKs{
            std::move(seam->fock), std::move(workspaceBudget), std::nullopt, std::nullopt};
        wiredKs.coulomb = std::move(seam->coulomb);
        wiredKs.contribution = std::move(seam->contribution);
        // The carriers the Hartree-Fock arm below sets, set here as well, for
        // the reason the RKS arm states: the same family on the same builder
        // owes the record the same budget and the same engine rung decision.
        //
        // The orbit-expansion pair stays where this runner leaves it on the
        // Hartree-Fock arm - unset - and that is the honest reading rather than
        // a dropped carrier: a named method.ri_orbit_expansion is CARRIED into
        // the options above, the builder refuses the mechanism by name at
        // Create, so no run that ever reaches this seam asked for it. There is
        // no engaged outcome to read either way (RiFullFockBuilder has no
        // OrbitExpansionEngaged counterpart), which is the note the restricted
        // arm's carrier records.
        wiredKs.riJkModeInfo = builder->ModeInfo();
        return RunUhfScfWithBuilder(molecule,
                                    basis,
                                    overlap,
                                    core,
                                    atomicInputs,
                                    input,
                                    enableTrace,
                                    nAlpha,
                                    nBeta,
                                    statsStream,
                                    std::move(wiredKs),
                                    qcx::io::BuilderKind::kRiJk);
    }

    const auto uhfBuilder = MakeRiFullUhfFockBuilder(*builder,
                                                     core,
                                                     &statsStream,
                                                     static_cast<std::size_t>(nAlpha),
                                                     static_cast<std::size_t>(nBeta));
    WiredUhfBuilders wired{uhfBuilder, std::move(workspaceBudget), std::nullopt, std::nullopt};
    // The engine's own Create-time rung decision, carried out rather than
    // recomputed (the RHF arm's rule): the record's ri_jk_mode block is a read
    // of what ran, and a driver-side second opinion could disagree with it.
    wired.riJkModeInfo = builder->ModeInfo();
    return RunUhfScfWithBuilder(molecule,
                                basis,
                                overlap,
                                core,
                                atomicInputs,
                                input,
                                enableTrace,
                                nAlpha,
                                nBeta,
                                statsStream,
                                std::move(wired),
                                qcx::io::BuilderKind::kRiJk);
}

// The opt-in charges block (Hirshfeld/Voronoi): both analyses share the
// charges tests' pinned grid (charges_test.cpp, 80 x 194), built here on
// demand. The Hirshfeld promolecular fragment densities are exactly the
// SAD guess's (charges.hpp), assembled from the caller-built fragment
// inputs - RunDriver builds them once whenever any consumer needs them,
// even when the run itself started from another guess.
qcx::Result<std::optional<qcx::io::RunCharges>> BuildChargesBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& spinSummed,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input) {
    if (!input.properties.hirshfeld && !input.properties.voronoi)
    {
        return std::nullopt;
    }

    auto grid = qcx::grid::MolecularGrid::Create(molecule, kGridRadialPoints, kGridAngularPoints);

    if (!grid.has_value())
    {
        return std::unexpected(grid.error());
    }

    qcx::io::RunCharges charges;

    if (input.properties.hirshfeld)
    {
        // The promolecular densities need the SAD fragment inputs; the
        // caller's invariant is that the optional is engaged whenever
        // hirshfeld is requested.
        if (!atomicInputs.has_value())
        {
            return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                       "the hirshfeld analysis has no caller-built "
                                       "fragment inputs"));
        }

        auto sad = qcx::scf::BuildSadGuess(molecule, basis, *atomicInputs);

        if (!sad.has_value())
        {
            return std::unexpected(sad.error());
        }

        auto aoRanges = qcx::properties::AoIndexRangesByAtom(molecule, basis);

        if (!aoRanges.has_value())
        {
            return std::unexpected(aoRanges.error());
        }

        auto hirshfeld = qcx::properties::AnalyzeHirshfeld(
            molecule, basis, *grid, spinSummed, sad->densityAlpha, sad->densityBeta, *aoRanges);

        if (!hirshfeld.has_value())
        {
            return std::unexpected(hirshfeld.error());
        }

        charges.hirshfeld = ToStdVector(*hirshfeld);
    }

    if (input.properties.voronoi)
    {
        auto voronoi = qcx::properties::AnalyzeVoronoi(molecule, basis, *grid, spinSummed);

        if (!voronoi.has_value())
        {
            return std::unexpected(voronoi.error());
        }

        charges.voronoi = ToStdVector(*voronoi);
    }

    return charges;
}

// The opt-in ESP-fit block: the io vocabulary maps onto the properties
// module's schemes; the exhaustive switch fails loudly (kUnimplemented) on
// a third scheme instead of silently collapsing into Merz-Kollman.
qcx::Result<std::optional<qcx::io::RunEspFit>> BuildEspBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& spinSummed,
    const qcx::io::RunInput& input) {
    if (!input.properties.esp.has_value())
    {
        return std::nullopt;
    }

    // The exhaustive switch below assigns every reachable scheme; the
    // default returns kUnimplemented before the use, so no uninitialized
    // path exists.
    //
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    qcx::properties::EspFitScheme scheme;

    switch (*input.properties.esp)
    {
    case qcx::io::EspFitScheme::kChelpg:
        scheme = qcx::properties::EspFitScheme::kChelpg;
        break;
    case qcx::io::EspFitScheme::kMerzKollman:
        scheme = qcx::properties::EspFitScheme::kMerzKollman;
        break;
    default:
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented, "unknown ESP fit scheme"));
    }

    auto fit = qcx::properties::AnalyzeEspCharges(molecule, basis, spinSummed, scheme);

    if (!fit.has_value())
    {
        return std::unexpected(fit.error());
    }

    qcx::io::RunEspFit block;
    block.charges = ToStdVector(fit->charges);
    block.rmsError = fit->rmsError;
    block.pointCount = fit->pointCount;
    return block;
}

// The opt-in EDDB block.
qcx::Result<std::optional<qcx::io::RunEddb>> BuildEddbBlock(const qcx::molecule::Molecule& molecule,
                                                            const qcx::basisset::BasisSet& basis,
                                                            const Eigen::MatrixXd& alphaDensity,
                                                            const Eigen::MatrixXd& betaDensity,
                                                            const qcx::io::RunInput& input) {
    if (!input.properties.eddb)
    {
        return std::nullopt;
    }

    auto eddb = qcx::properties::AnalyzeEddb(molecule, basis, alphaDensity, betaDensity);

    if (!eddb.has_value())
    {
        return std::unexpected(eddb.error());
    }

    qcx::io::RunEddb block;
    block.totalPopulation = eddb->totalPopulation;
    block.atomicPopulations = ToStdVector(eddb->atomicPopulations);
    block.nobdOccupations = ToStdVector(eddb->nobdOccupations);
    block.centralAtomCount = eddb->centralAtomCount;
    block.twoCenterOrbitalCount = eddb->twoCenterOrbitalCount;
    return block;
}

// The opt-in Nalewajski-Mrozek bond-order block.
qcx::Result<std::optional<qcx::io::RunNalewajski>> BuildNalewajskiBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& alphaDensity,
    const Eigen::MatrixXd& betaDensity,
    const qcx::io::RunInput& input) {
    if (!input.properties.nalewajski)
    {
        return std::nullopt;
    }

    auto nalewajski =
        qcx::properties::AnalyzeNalewajskiBondOrders(molecule, basis, alphaDensity, betaDensity);

    if (!nalewajski.has_value())
    {
        return std::unexpected(nalewajski.error());
    }

    qcx::io::RunNalewajski block;
    block.bondOrders = ToStdMatrix(nalewajski->bondOrders);
    block.diatomicCovalent = ToStdMatrix(nalewajski->diatomicCovalent);
    block.atomicIonicValence = ToStdVector(nalewajski->atomicIonicValence);
    block.atomicCovalentValence = ToStdVector(nalewajski->atomicCovalentValence);
    block.totalValence = ToStdVector(nalewajski->totalValence);
    return block;
}

// The opt-in condensed-Fukui block: the response needs the N + 1 and N - 1
// species at the same geometry and basis (fukui.hpp), run here with the
// direct-UHF path at the S +- 1/2 multiplicities: the cation drops a half
// spin (a singlet's cation is a doublet), the anion gains one. The charged
// species inherit the run's guess, so the SAD fragment inputs are the
// caller-built ones when the guess asks for SAD.
qcx::Result<std::optional<qcx::io::RunFukui>> BuildFukuiBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& overlap,
    // The four matrices (overlap, core, alpha/beta density) are distinct
    // SCF objects named at the call site - deliberately not swappable.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& core,
    const Eigen::MatrixXd& alphaDensity,
    const Eigen::MatrixXd& betaDensity,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const qcx::backend::GpuComputeProfile& deviceComputeProfile,
    std::optional<bool> certifiedLaneRequest,
    bool leanMember) {
    if (!input.properties.fukui)
    {
        return std::nullopt;
    }

    const int cationMultiplicity = (molecule.Multiplicity() > 1) ? molecule.Multiplicity() - 1 : 2;
    const int anionMultiplicity = molecule.Multiplicity() + 1;

    auto cation = MakeChargedMolecule(molecule, molecule.Charge() + 1, cationMultiplicity);

    if (!cation.has_value())
    {
        return std::unexpected(cation.error());
    }

    auto anion = MakeChargedMolecule(molecule, molecule.Charge() - 1, anionMultiplicity);

    if (!anion.has_value())
    {
        return std::unexpected(anion.error());
    }

    auto cationScf = RunDirectUhfScf(*cation,
                                     basis,
                                     coreTensor,
                                     overlap,
                                     core,
                                     atomicInputs,
                                     input,
                                     workspaceReserveBytes,
                                     deviceComputeProfile,
                                     certifiedLaneRequest,
                                     leanMember);

    if (!cationScf.has_value())
    {
        return std::unexpected(cationScf.error());
    }

    // Convergence gates: the
    // cation/anion densities feed AnalyzeFukui, and a non-converged run
    // returns the LAST iterate - a missed convergence would silently skew
    // every Fukui index. The charged species are harder to converge than
    // the neutral parent, so the run surfaces the failure instead of
    // emitting garbage indices.
    if (!cationScf->scf.converged)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the cation species SCF did not converge within its iteration budget; "
                       "the Fukui response needs converged species states"});
    }

    auto anionScf = RunDirectUhfScf(*anion,
                                    basis,
                                    coreTensor,
                                    overlap,
                                    core,
                                    atomicInputs,
                                    input,
                                    workspaceReserveBytes,
                                    deviceComputeProfile,
                                    certifiedLaneRequest,
                                    leanMember);

    if (!anionScf.has_value())
    {
        return std::unexpected(anionScf.error());
    }

    if (!anionScf->scf.converged)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the anion species SCF did not converge within its iteration budget; "
                       "the Fukui response needs converged species states"});
    }

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(molecule, basis);

    if (!aoRanges.has_value())
    {
        return std::unexpected(aoRanges.error());
    }

    auto fukui = qcx::properties::AnalyzeFukui(overlap,
                                               alphaDensity,
                                               betaDensity,
                                               anionScf->scf.densityAlpha,
                                               anionScf->scf.densityBeta,
                                               cationScf->scf.densityAlpha,
                                               cationScf->scf.densityBeta,
                                               *aoRanges);

    if (!fukui.has_value())
    {
        return std::unexpected(fukui.error());
    }

    qcx::io::RunFukui block;
    block.nucleophilic = ToStdVector(fukui->nucleophilic);
    block.electrophilic = ToStdVector(fukui->electrophilic);
    block.radical = ToStdVector(fukui->radical);
    return block;
}

// The opt-in ETS-NOCV block . The UHF rejection lives in
// ValidateCombination, with the other not-wired combinations; only the
// analysis itself runs here.
qcx::Result<std::optional<qcx::io::RunNocv>> BuildNocvBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& core,
    const Eigen::MatrixXd& spinSummed,
    const std::vector<std::size_t>& canonicalOrder,
    const qcx::io::RunInput& input) {
    if (input.properties.nocvFragments.empty())
    {
        return std::nullopt;
    }

    auto fragments = MapFragmentsToCanonical(input.properties.nocvFragments, canonicalOrder);

    if (!fragments.has_value())
    {
        return std::unexpected(fragments.error());
    }

    // The NOCV dense-tensor charge (a measured note): the
    // analysis-phase peak - the a-priori worst-case bytes of the dense
    // ERI tensor build (the canonical quartet list + the batch payload +
    // the 8 n^4 tensor, the same estimate the engine's admission gate
    // applies) - is charged against the resource cap BEFORE the builder
    // is called. The integrals-side maxTensorBytes gate (4bcaa64) already
    // refuses unaffordable builds gracefully; this charge surfaces the
    // refusal through the driver's own reinstatement ladder (the adaptive-seam
    // seam pattern) in the user-facing memory_cap_gib terms, so no NOCV
    // run ever reaches the builder when the modeled analysis phase cannot
    // fit the cap - never a raw allocation failure under the hard cap.
    auto denseFit = qcx::driver::CheckNocvDenseTensorFit(CountBasisFunctions(molecule, basis),
                                                         CountShellPairs(molecule, basis),
                                                         input.resources.memoryCapGiB);

    if (!denseFit.has_value())
    {
        return std::unexpected(denseFit.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basis);

    if (!eri.has_value())
    {
        // The engine's admission-gate refusal carries the driver's
        // standing NOCV clause composed onto it (the adaptive-seam pattern,
        // like the ri_j and direct branches): the engine text alone names
        // the maxTensorBytes knob the driver does not expose, and the
        // composed ladder adds the user-facing options. Only the gate's
        // kInvalidArgument refusals compose - the shell-cap and
        // out-of-memory codes propagate verbatim.
        if (eri.error().code == qcx::ErrorCode::kInvalidArgument)
        {
            return std::unexpected(
                Err(eri.error().code,
                    eri.error().message + qcx::driver::NocvDenseTensorReinstatementClause()));
        }

        return std::unexpected(eri.error());
    }

    // The fragment SCFs follow the run's [scf] convergence tolerances
    // (without this they would silently keep whatever the scf
    // header defaults are - the recorded partition pins are tight-gate
    // values, and the input's tolerances are the user's contract for the
    // whole run).
    qcx::scf::UhfOptions fragmentOptions;
    fragmentOptions.maxIterations = input.scf.maxIterations;
    fragmentOptions.energyTolerance = input.scf.energyTolerance;
    fragmentOptions.densityTolerance = input.scf.densityTolerance;
    fragmentOptions.useDiis = input.scf.useDiis;

    auto nocv = qcx::properties::AnalyzeNocvEts(
        molecule, basis, core, *eri, spinSummed, *fragments, fragmentOptions);

    if (!nocv.has_value())
    {
        return std::unexpected(nocv.error());
    }

    qcx::io::RunNocv block;
    block.electrostatic = nocv->electrostatic;
    block.pauli = nocv->pauli;
    block.orbital = nocv->orbital;
    block.orbitalUnrestricted = nocv->orbitalUnrestricted;
    block.bindingEnergy = nocv->bindingEnergy;
    block.fragmentEnergies = ToStdVector(nocv->fragmentEnergies);
    block.orbitalComponents = ToStdVector(nocv->orbitalComponents);
    block.nocvEigenvalues = ToStdVector(nocv->nocvEigenvalues);
    block.orbitalComponentsAlpha = ToStdVector(nocv->orbitalComponentsAlpha);
    block.orbitalComponentsBeta = ToStdVector(nocv->orbitalComponentsBeta);
    block.nocvEigenvaluesAlpha = ToStdVector(nocv->nocvEigenvaluesAlpha);
    block.nocvEigenvaluesBeta = ToStdVector(nocv->nocvEigenvaluesBeta);
    return block;
}

// The opt-in density-at-nuclei block (first increment): the
// spin-summed density evaluated at every nucleus - a point evaluation
// with no quadrature, the Bader QTAIM prerequisite.
qcx::Result<std::optional<qcx::io::RunDensityAtNuclei>> BuildDensityAtNucleiBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& spinSummed,
    const qcx::io::RunInput& input) {
    if (!input.properties.densityAtNuclei)
    {
        return std::nullopt;
    }

    auto values = qcx::properties::AnalyzeDensityAtNuclei(molecule, basis, spinSummed);

    if (!values.has_value())
    {
        return std::unexpected(values.error());
    }

    qcx::io::RunDensityAtNuclei block;
    block.values = ToStdVector(*values);
    return block;
}

// The opt-in fixed-density exchange-correlation gradient block: the derivative
// of the exchange-correlation energy this run integrated, taken at the density
// it converged to and on the grid it integrated over.
//
// The functional name, the `[grid]` block and the screening tolerance are
// resolved HERE exactly as the energy path resolved them (ResolveKsContext and
// the same two helpers), so the walk differentiates the energy the run
// computed. That is the whole risk in this block: a walk built from re-read or
// re-derived keys screens by another rule or truncates another point set, and
// the gradient it reports is then the derivative of a different energy while
// every number in it still looks reasonable. The tolerance and the settings
// object go to CreateKsGrid together for the same reason the energy path hands
// them over together. The grid engine itself is built a second time rather than
// carried from the SCF, because the energy path's engine does not outlive its
// branch; the KEYS are what must not diverge, and they are read once here.
//
// The densities arrive in the same convention the energy path was given: on the
// restricted lane both spins carry half the converged density matrix, which is
// the split the engine's own closed-shell entry point performs.
//
// Driver-internal helper of FillProperties (the forward declarations above).
qcx::Result<std::optional<qcx::io::RunXcGradient>> BuildXcGradientBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& alphaDensity,
    const Eigen::MatrixXd& betaDensity,
    const qcx::io::RunInput& input) {
    if (!input.properties.xcGradient)
    {
        return std::nullopt;
    }

    // The validator refuses this key on a method that names no functional
    // (validate_input.cpp's key policy), so the absent case here is a
    // programmatic caller that never passed through validation.
    if (!input.method.functional.has_value())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "properties.xc_gradient: no method.functional to differentiate"));
    }

    auto functional = qcx::driver::internal::ResolveKsFunctional(*input.method.functional);

    if (!functional.has_value())
    {
        return std::unexpected(functional.error());
    }

    const qcx::grid::XcGridSettings settings =
        ResolveXcGridSettings(input.grid.value_or(qcx::io::RunGridInput{}));
    const double tolerance =
        input.method.screeningTolerance.value_or(qcx::io::kDefaultScreeningTolerance);

    auto grid =
        qcx::driver::internal::CreateKsGrid(molecule, basis, functional->name, settings, tolerance);

    if (!grid.has_value())
    {
        return std::unexpected(grid.error());
    }

    // The walk adds into a total, so the block's gradient is the walk's own
    // vector carried in the run's total-gradient shape: 3N, atom-major, in the
    // molecule's own atom order (the ordering the grid's geometry was built in).
    Eigen::VectorXd total =
        Eigen::VectorXd::Zero(3 * static_cast<Eigen::Index>(molecule.AtomCount()));
    auto walk =
        qcx::driver::internal::AddXcGradientContribution(*grid, alphaDensity, betaDensity, total);

    if (!walk.has_value())
    {
        return std::unexpected(walk.error());
    }

    qcx::io::RunXcGradient block;
    block.gradient = ToStdVector(total);
    block.energyHartree = walk->energy;
    return block;
}

// The opt-in Bader QTAIM block (second increment): the (3,-1)
// bond critical points of the spin-summed density with their bond paths,
// plus the reported non-bond and unconverged critical points (the failed-
// search policy - reported, never a failure).
qcx::Result<std::optional<qcx::io::RunQtaim>> BuildQtaimBlock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& spinSummed,
    const qcx::io::RunInput& input) {
    if (!input.properties.qtaim)
    {
        return std::nullopt;
    }

    auto result = qcx::properties::AnalyzeQtaim(molecule, basis, spinSummed);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    qcx::io::RunQtaim block;
    block.bondCriticalPoints.reserve(result->bondCriticalPoints.size());

    for (const auto& bcp : result->bondCriticalPoints)
    {
        qcx::io::RunBondCriticalPoint out;
        out.atomA = bcp.atomA;
        out.atomB = bcp.atomB;
        out.positionBohr = std::vector<double>(bcp.positionBohr.begin(), bcp.positionBohr.end());
        out.density = bcp.density;
        out.laplacian = bcp.laplacian;
        out.ellipticity = bcp.ellipticity;
        out.eigenvalues = std::vector<double>(bcp.eigenvalues.begin(), bcp.eigenvalues.end());
        out.bondPath.reserve(bcp.bondPath.size());

        for (const auto& point : bcp.bondPath)
        {
            out.bondPath.emplace_back(point.begin(), point.end());
        }

        block.bondCriticalPoints.push_back(std::move(out));
    }

    block.otherCriticalPoints.reserve(result->otherCriticalPoints.size());

    for (const auto& ocp : result->otherCriticalPoints)
    {
        qcx::io::RunOtherCriticalPoint out;
        out.positionBohr = std::vector<double>(ocp.positionBohr.begin(), ocp.positionBohr.end());
        out.rank = ocp.rank;
        out.signatureSum = ocp.signatureSum;
        block.otherCriticalPoints.push_back(std::move(out));
    }

    block.unconvergedSeeds.reserve(result->unconvergedSeeds.size());

    for (const auto& seed : result->unconvergedSeeds)
    {
        block.unconvergedSeeds.emplace_back(seed.begin(), seed.end());
    }

    return block;
}

// The aufbau occupations of one spin channel: occupiedValue per occupied
// orbital (RHF 2.0, UHF 1.0), 0.0 beyond. The (count, value, length)
// triple is one record, passed together in a fixed call order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::VectorXd AufbauOccupations(std::size_t nOccupied,
                                  double occupiedValue,
                                  std::size_t nOrbitals) {
    Eigen::VectorXd occupations = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nOrbitals));

    for (std::size_t i = 0; i < nOccupied; ++i)
    {
        occupations(static_cast<Eigen::Index>(i)) = occupiedValue;
    }

    return occupations;
}

// The Molden export: writes the [Molden Format]
// [Atoms] (AU) [5D] [7F] [GTO] [MO] file of the converged or
// last-iterate SCF result when the input requested it; a file that
// cannot be written fails the run - a broken request, never a silent
// skip. The occupations arrive computed (the aufbau responsibility);
// the moBlocks carry the spin channel(s) the SCF branch assembled.
qcx::Result<void> WriteMoldenExport(qcx::io::RunResult& result,
                                    const qcx::molecule::Molecule& molecule,
                                    const qcx::basisset::BasisSet& basis,
                                    std::span<const qcx::io::MoldenMolecularOrbitals> moBlocks,
                                    const qcx::io::RunInput& input) {
    if (input.properties.molden.empty())
    {
        return {};
    }

    auto written = qcx::io::WriteMoldenFile(molecule, basis, moBlocks, input.properties.molden);

    if (!written.has_value())
    {
        return std::unexpected(written.error());
    }

    result.properties->molden = qcx::io::RunMolden{input.properties.molden};
    return {};
}

// The full-group labeling block of the run JSON: converts the scf
// stage's SymmetryLabels into the io schema record. Empty when the stage
// did not run - a C1 molecule, the [symmetry] full_group = false switch,
// or an unrealizable group.
std::optional<qcx::io::RunSymmetry> MakeRunSymmetry(
    const std::optional<qcx::scf::SymmetryLabels>& labels) {
    if (!labels.has_value())
    {
        return std::nullopt;
    }

    qcx::io::RunSymmetry runSymmetry;
    runSymmetry.fullGroup = std::string(qcx::symmetry::ToString(labels->fullGroup));
    runSymmetry.abelianReduction = std::string(qcx::symmetry::ToString(labels->abelianReduction));
    runSymmetry.labels = labels->labels;
    runSymmetry.irrepIndices = labels->irrepIndices;

    for (const auto& record : labels->canonicalized)
    {
        runSymmetry.canonicalized.push_back(
            qcx::io::RunDegenerateSubspace{record.irrepLabel, record.moIndices});
    }

    for (const auto& record : labels->straddled)
    {
        runSymmetry.straddled.push_back(
            qcx::io::RunDegenerateSubspace{record.irrepLabel, record.moIndices});
    }

    runSymmetry.symmetrizationSubset = labels->symmetrizationSubset;
    runSymmetry.averagedElementCount = labels->averagedElementCount;
    return runSymmetry;
}

// The symmetry-blocking guard's action word, and with it the PRESENCE rule of
// the record's `symmetry_blocking` key: kNotRequested is the absence - the run
// was never in a position to block, so there is nothing to disclose - and
// every other enumerator names a decision the run must report. The two ride
// together so a state cannot be present in the record without a word, nor
// worded without being present. Total over the guard's enum by construction
// (no default arm; the unreachable line below is what a sixth enumerator would
// have to be reasoned about at), because a silently unmapped state is exactly
// the run record that differs from what ran .
std::optional<std::string> SymmetryBlockingActionWord(qcx::scf::SymmetryBlockingAction action) {
    switch (action)
    {
    case qcx::scf::SymmetryBlockingAction::kNotRequested:
        return std::nullopt;
    case qcx::scf::SymmetryBlockingAction::kUsed:
        return std::string("kUsed");
    case qcx::scf::SymmetryBlockingAction::kDemoted:
        return std::string("kDemoted");
    case qcx::scf::SymmetryBlockingAction::kRefused:
        return std::string("kRefused");
    case qcx::scf::SymmetryBlockingAction::kUnavailable:
        return std::string("kUnavailable");
    }

    // Unreachable: the switch above handles every enumerator of
    // SymmetryBlockingAction. If a sixth state is ever added, this is the line
    // that has to be reasoned about, and the compiler's own missing-return
    // diagnostic points here.
    std::unreachable();
}

// The symmetry-blocking disclosure of one spin channel: the guard's
// report, translated member for member. The action is COPIED from the report
// the deciding code produced - the guard is the one site that knows which path
// a diagonalization took (scf/src/uhf.cpp's diagonalizeFock lambda) - never
// re-derived here from the two solve counts, which a second derivation would
// be free to disagree with. Empty when the run never asked, which is what
// keeps the key absent on a run whose blocking was never in play.
std::optional<qcx::io::RunSymmetryBlocking> MakeRunSymmetryBlocking(
    const qcx::scf::SymmetryBlockingReport& report) {
    const auto action = SymmetryBlockingActionWord(report.action);

    if (!action.has_value())
    {
        return std::nullopt;
    }

    qcx::io::RunSymmetryBlocking blocking;
    blocking.action = *action;
    blocking.blockedSolveCount = report.blockedSolveCount;
    blocking.plainSolveCount = report.plainSolveCount;
    blocking.generatorCommutatorNorms = report.generatorCommutatorNorms;
    blocking.maxGeneratorCommutatorNorm = report.maxGeneratorCommutatorNorm;
    blocking.tolerance = report.tolerance;
    return blocking;
}

// The workspace_budget record (the budget path): the cap-minus-base budget the
// ri_j branch granted the engine, and the engine's cumulative Create-time
// commit. Absent on the legacy null-budget path.
std::optional<qcx::io::RunWorkspaceBudget> MakeRunWorkspaceBudget(
    const std::unique_ptr<qcx::memory::WorkspaceBudget>& budget) {
    if (budget == nullptr)
    {
        return std::nullopt;
    }

    qcx::io::RunWorkspaceBudget record;
    record.capacityBytes = budget->CapacityBytes();
    record.committedBytes = budget->CommittedBytes();
    return record;
}

// The engine's Create-time mode record (the budget path): mirrors the integrals
// FockModeInfo into the io schema. Absent on the legacy null-budget path
// (no decision was made there). `forcedDisk` is the driver's own fact, not
// an engine one: true only on the forced disk route (the [diagnostics]
// force_disk_ri override, which replaced the retired
// method.ri_tensor_mode = "forced_disk" word), so a reader
// can tell a forced measurement cell from a ladder-selected disk run. Every
// engine record and the ladder's own disk fallback pass false (the default
// - the UHF exchange half is an engine decision and can never be forced).
// The rung label of the mode record: the ONE place a
// FockBuildMode becomes the word the record carries, exhaustive BY
// CONSTRUCTION - no `default:` arm - so an enumerator added to
// integrals::FockBuildMode cannot reach the record unlabelled.
//
// Why the guard, and why the mapping lives here rather than beside the enum.
// The chain this replaced was `if (kLightPath) ... else if (kDisk) ... else
// "kFastPath"`, whose tail asserts a SPECIFIC rung for any value nobody
// enumerated: a fourth FockBuildMode would be RECORDED as kFastPath while a
// different rung ran, so the record would name an algorithm that did not run.
// That is the misreport class the disclosure rule forbids, in its sharpest form - the wrong
// answer is plausible rather than obviously wrong, so nothing downstream can
// tell the two apart. The guarded switch turns it into a build failure, the
// shape accuracy.hpp's mapper guard already uses: MSVC's C4061/C4062 and
// GCC/Clang's -Wswitch-enum promoted, C4061 being the variant that also
// catches a `default:` arm added to silence the check. The region spans the
// one function below; every other switch in this translation unit keeps the
// project's default diagnostic settings.
//
// The vocabulary is the driver's own: kDisk is the disk route's record (no
// engine sets it - integrals cannot link storage), so the mapping is written
// where the record is written rather than moved into the engine's public
// header. The tail is `"unknown"` - honest rather than plausible, the
// parse_input.cpp ToString posture - and is unreachable for every value the
// enum names.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error : 4061)
#pragma warning(error : 4062)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
std::string_view ModeRungLabel(qcx::integrals::FockBuildMode mode) noexcept {
    switch (mode)
    {
    case qcx::integrals::FockBuildMode::kFastPath:
        return "kFastPath";
    case qcx::integrals::FockBuildMode::kLightPath:
        return "kLightPath";
    case qcx::integrals::FockBuildMode::kDisk:
        return "kDisk";
    }

    return "unknown";
}
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

std::optional<qcx::io::RunModeRecord> MakeRunModeRecord(
    const std::optional<qcx::integrals::FockModeInfo>& info, bool forcedDisk = false) {
    if (!info.has_value())
    {
        return std::nullopt;
    }

    qcx::io::RunModeRecord record;
    // The rung names keep the engine's enumerator vocabulary (the
    // `mode: kFastPath` pin) - the k-prefixed PascalCase of the
    // presets, not a new lowercase vocabulary. kDisk is the driver's disk
    // route's own record (no engine sets it - integrals cannot link
    // storage); the engine's records stay kFastPath/kLightPath. The mapping
    // and its exhaustiveness guard are ModeRungLabel's above.
    record.mode = std::string{ModeRungLabel(info->mode)};

    record.predictedBytes = info->predictedBytes;
    record.reservedBytes = info->reservedBytes;
    record.budgetBytes = info->budgetBytes;
    record.remainingAtDecision = info->remainingAtDecision;
    record.maxBatchBytes = info->maxBatchBytes;
    record.pairStoreBytes = info->pairStoreBytes;
    record.patternBytes = info->patternBytes;
    record.scratchBytes = info->scratchBytes;
    record.structuralBytes = info->structuralBytes;
    record.cacheBytes = info->cacheBytes;
    record.lightStoreBytes = info->lightStoreBytes;
    record.chunkArenaBytes = info->chunkArenaBytes;
    record.chunkPatternBytes = info->chunkPatternBytes;
    record.chunkIndexBytes = info->chunkIndexBytes;
    record.lightShellsBytes = info->lightShellsBytes;
    record.chunkPairs = info->chunkPairs;
    record.diskBytes = info->diskBytes;
    record.tensorBytes = info->tensorBytes;
    record.riMatrixBytes = info->riMatrixBytes;
    record.taskListBytes = info->taskListBytes;
    record.metricBytes = info->metricBytes;
    record.orbitalAuxBytes = info->orbitalAuxBytes;
    record.outerStoreBytes = info->outerStoreBytes;
    record.exchangeBytes = info->exchangeBytes;
    record.exchangeScratchBytes = info->exchangeScratchBytes;
    record.patternExcluded = info->patternExcluded;
    record.tensorExcluded = info->tensorExcluded;
    record.classTableBytes = info->classTableBytes;
    record.classPathDisengaged = info->classPathDisengaged;
    record.concurrentSlots = info->concurrentSlots;
    record.forcedDisk = forcedDisk;
    record.defaultTeamSize = info->defaultTeamSize;
    return record;
}

// The composed full-RI builder's Create-time rung record: the
// ri_jk family's own decision, mirrored from RiFullFockModeInfo, the way
// MakeRunModeRecord mirrors FockModeInfo. It is a separate block rather than
// a member of mode_record - the two families' rung words and term
// decompositions differ (the WiredFockBuilder member states why), and a
// reader of a number must not have to know which family wrote it to know what
// it means. The rung word keeps the engine's enumerator vocabulary
// (kFastPath/kLightPath's precedent) so a reader can grep the word the record
// carries in the source that chose it. Only a decided builder reaches the
// block: `engaged == false` is the no-budget path, which consults no budget
// and takes the fast rung, and a record of a decision that did not happen
// would be a fabricated read.
std::optional<qcx::io::RunRiJkMode> MakeRiJkModeRecord(
    const std::optional<qcx::integrals::RiFullFockModeInfo>& info) {
    if (!info.has_value() || !info->engaged)
    {
        return std::nullopt;
    }

    qcx::io::RunRiJkMode record;
    record.rung = info->rung == qcx::integrals::RiFullFockRung::kBlocked ? "kBlocked" : "kFast";
    record.predictedBytes = info->predictedBytes;
    record.reservedBytes = info->reservedBytes;
    record.budgetBytes = info->budgetBytes;
    record.remainingAtDecision = info->remainingAtDecision;
    record.maxBatchBytes = info->maxBatchBytes;
    record.structuralBytes = info->structuralBytes;
    record.rootBytes = info->rootBytes;
    record.tensorBytes = info->tensorBytes;
    record.occTransformBytes = info->occTransformBytes;
    record.fockBytes = info->fockBytes;
    record.arenaBytes = info->arenaBytes;
    record.sliceFunctions = info->sliceFunctions;
    record.sliceCount = info->sliceCount;
    return record;
}

// The composed-QFMM builder's model record: mirrors the
// integrals QfmmModelRecord - the engine's own answer to "which model
// produced this Fock matrix" - into the io schema. It is a READ, not a
// resolution: the theta the octree was built with (the explicit value that
// selected the width test, or the gate's 0) is decided inside
// QfmmJBuilder::Create, so a driver-side recomputation of it could disagree
// with the build it describes - the rule the ri_orbit_expansion record states
// for the same reason. The engine also records the parameter of the test that
// RAN and zeroes the other one, and this function copies both members
// verbatim rather than "improving" them: the zero is the engine's contract,
// and the schema document carries the read-yours-only rule it implies.
//
// The TWO words keep the engine's enumerator vocabulary, the mode_record
// precedent (kFastPath/kLightPath rather than a new lowercase vocabulary),
// so a reader can grep the word the record carries in the source that chose
// it. They are the whole disclosure: the flip of 2026-09-15 made the pair
// "kProductBall" + "kSurfaceBall" the default, and a record that named only
// one of the two would leave the other invisible exactly where it changed.
//
// Absent (nullopt) when the run wired no composed-QFMM builder: the io
// member is empty there and the document carries no block, never a
// fabricated default.
std::optional<qcx::io::RunQfmmModel> MakeQfmmModelRecord(
    const std::optional<qcx::integrals::QfmmModelRecord>& model) {
    if (!model.has_value())
    {
        return std::nullopt;
    }

    qcx::io::RunQfmmModel record;
    record.extentModel = model->extentModel == qcx::integrals::QfmmExtentMode::kProductBall
                             ? "kProductBall"
                             : "kMidpointBound";
    record.separationMode =
        model->separationMode == qcx::integrals::QfmmSeparationMode::kSurfaceBall ? "kSurfaceBall"
                                                                                  : "kWidthTheta";
    record.separationK = model->separationK;
    record.theta = model->theta;
    // The far field's own accuracy record, carried verbatim for
    // the reason above: every one of these is a value the engine resolved
    // while it built the far field - the preset budget it held the field to,
    // the share it split over the pairs, the count it split them over, how
    // many the budget moved out of the field, and the a-priori bound at the
    // order cap for the worst interaction and its fall-through count. The
    // driver resolves none of them and must not: a recomputed budget or a
    // re-inferred pair count could disagree with the build the energy came
    // from, which is the mismatch this block exists to expose. These are also
    // the values that make a budget MISS visible - a run that used to report
    // an energy with no sign that its far field was outside the accuracy it
    // was held to now states the count and the bound that say so.
    record.errorAwareAdmission = model->errorAwareAdmission;
    record.farFieldBudget = model->farFieldBudget;
    record.perInteractionBudget = model->perInteractionBudget;
    record.geometricFarPairCount = model->geometricFarPairCount;
    record.pairsMovedToNearField = model->pairsMovedToNearField;
    record.fellThroughToCap = model->fellThroughToCap;
    record.worstTruncationBound = model->worstTruncationBound;
    return record;
}

// The requested-vs-ran pairing of the RI tensor mode (the
// enforcement contract's disclosure surface): what the input NAMED, what
// actually ran, how the two relate, and why when they do not agree. It is
// filled on every run that named EITHER request key - method.ri_tensor_mode
// or [diagnostics] force_disk_ri - on the paths with no mode
// record (the lean arm, the legacy null-budget path) and every family with
// no RI disk rung (a `disk` request on a direct/qfmm/gpu/lean run) included,
// which is exactly where a request used to leave no trace at all.
//
// The four outcome words, and the judgement behind the one that is easiest
// to get wrong: "ladder_fit" is NOT a demotion. `disk` PERMITS the disk rung
// as the ladder's last rung, so a fitting memory rung riding is what the
// request itself prescribes - the knob's documented inert-by-design
// contract. Reporting that normal run as a fallback would be the record
// lying in the other direction from the one this block exists to fix.
//
// The FORCE is judged, not merely recorded, and its posture is inherited
// rather than re-decided: [diagnostics] force_disk_ri is the separate key
// that replaced method.ri_tensor_mode = "forced_disk" (the owner's ruling
// 2026-09-13), so it carries exactly the demote-with-disclosure posture the displaced word
// carried (cbee51cf: the request DEMOTES, with the record stating both sides,
// because the runnable choice - the resolved family's own in-memory path -
// exists and the record can state it). Nothing here refuses: a demotion the
// record can state is the contract's remedy, and refusing a request this
// block can disclose would reverse a landed, tested posture.
//
// When BOTH keys are named the force decides the route (it is the stronger
// request for the same store) and this block states the force, so `forced`
// and `requested` are never a blend of the two.
std::optional<qcx::io::RunRiTensorMode> MakeRiTensorModeRecord(
    const qcx::io::RunInput& input,
    qcx::io::BuilderKind resolvedKind,
    const std::optional<qcx::integrals::FockModeInfo>& info) {
    const bool forced = input.diagnostics.forceDiskRi;

    if (!input.method.riTensorMode.has_value() && !forced)
    {
        // An omitted key is the DEFAULT request, and the disclosure rule makes a default
        // the system's judgement rather than a user request - absent, never
        // a fabricated "auto".
        return std::nullopt;
    }

    const bool diskEngaged = info.has_value() && info->mode == qcx::integrals::FockBuildMode::kDisk;

    qcx::io::RunRiTensorMode record;
    record.forced = forced;

    if (forced)
    {
        // [diagnostics] force_disk_ri: the request NAMES the key it came
        // from (the word "forced_disk" survives here because it names the
        // REQUEST, and a reader who sees it must be able to find the key
        // that now carries it). Honoured means the storage-module disk
        // builder ran; anything else is the disclosed demotion below.
        record.requested = "forced_disk";
        record.resolved = diskEngaged ? "disk" : "in_memory";

        if (diskEngaged)
        {
            record.outcome = "honoured";
            return record;
        }

        // The force could not be honoured: the request made a claim about
        // what ran, the run did not satisfy it, so the best runnable
        // arrangement ran instead and the record states BOTH sides
        // (the expected outcome - a disclosed demotion, never a silent
        // substitution). Two texts, because they are two different facts: a
        // family with no ri_j_link disk route at all (the common case -
        // direct / qfmm / gpu / lean), and the ri_j_link route that exists
        // but did not engage (an inconsistency this text describes rather
        // than blames on the family).
        record.outcome = "demoted";

        if (resolvedKind == qcx::io::BuilderKind::kRiJLink)
        {
            record.reason =
                "the forced disk route (the ri_j_link family's storage-module disk-backed "
                "RI-J store) did not engage on this run, so the in-memory rung ran instead";
        } else
        {
            record.reason =
                "the forced disk route is the ri_j_link family's (the storage-module disk-backed "
                "RI-J store); fock_builder = \"" +
                std::string(qcx::io::ToString(resolvedKind)) +
                "\" has no disk rung, so the in-memory rung ran instead";
        }

        return record;
    }

    const qcx::io::RiTensorMode requested = *input.method.riTensorMode;
    record.requested = std::string{qcx::io::ToString(requested)};
    record.resolved = diskEngaged ? "disk" : "in_memory";

    if (diskEngaged || requested == qcx::io::RiTensorMode::kAuto)
    {
        // `auto` names no disk request at all, so the in-memory ladder IS
        // its answer; an engaged disk route is what the other two words
        // asked for.
        record.outcome = "honoured";
        return record;
    }

    if (requested == qcx::io::RiTensorMode::kDisk)
    {
        if (resolvedKind == qcx::io::BuilderKind::kRiJLink)
        {
            record.outcome = "ladder_fit";
            record.reason = "a fitting in-memory rung rode: the disk rung is the ladder's LAST "
                            "rung by design, so the knob permits it without demanding it";
        } else
        {
            record.outcome = "not_applicable";
            record.reason = "the RI disk rung is the ri_j_link family's route; fock_builder = \"" +
                            std::string(qcx::io::ToString(resolvedKind)) +
                            "\" has no disk rung, so the request had nothing to select";
        }

        return record;
    }

    // Unreachable with the two-word rung selector: `auto` and `disk` are the
    // only requests left on this key, and both returned above (an engaged
    // disk route, and `auto`, are "honoured"; `disk` is "ladder_fit" or
    // "not_applicable"). The demotion branch that used to close this
    // function belongs to the FORCE, which is now [diagnostics]
    // force_disk_ri and states its own outcome above - a rung selector has
    // no demotion to disclose, because a `disk` request the ladder answered
    // with a fitting memory rung got what it prescribed. If a third rung
    // word is ever added, this is the line that has to be reasoned about.
    return record;
}

// The ri_chunk_bytes request record: the disclosure arm of the key split
// key split (owner-ruled 2026-09-13). The key is a pure SIZE HINT and is read
// at exactly one site - the ri_j_link family's disk-rung lambda
// (engageDiskRung below), where it lands in DiskRiFockOptions::chunkBytes - so
// on every other family, and on ri_j_link when a fitting in-memory rung rides,
// the run proceeds and the hint is never read. Dropping it is the ruled
// behaviour (refusing a size hint on a family with no chunking to size is
// pedantic); saying NOTHING about the drop was the defect, and this block is
// what closes it.
//
// The honour predicate is deliberately the SAME fact the ri_tensor_mode pairing
// reads rather than a second derivation of the wiring: `info->mode == kDisk`.
// The two records therefore cannot disagree about whether the disk route ran.
// That is sound because a disk mode info has exactly one producer -
// engageDiskRung itself, the sole reader of this key - so "the record says
// kDisk" means "the hint was handed to a store", not a correlation that could
// drift from the wiring.
//
// Nothing here refuses: a disclosure the record can state is the contract's
// remedy, and the key stays non-fatal by the ruling.
std::optional<qcx::io::RunRiChunkBytes> MakeRiChunkBytesRecord(
    const qcx::io::RunInput& input,
    qcx::io::BuilderKind resolvedKind,
    const std::optional<qcx::integrals::FockModeInfo>& info) {
    if (!input.method.riChunkBytes.has_value())
    {
        // An omitted key is not a request: absent, never a fabricated drop
        // (the null-honesty rule the sibling block follows: an absent key is
        // reported as unreported, never as a drop).
        return std::nullopt;
    }

    qcx::io::RunRiChunkBytes record;
    record.requestedBytes = *input.method.riChunkBytes;

    if (info.has_value() && info->mode == qcx::integrals::FockBuildMode::kDisk)
    {
        // The disk rung engaged, so the hint was read and the store took it
        // as its chunk size. No reason: an absent reason is never a
        // fabricated "fine".
        record.outcome = "honoured";
        return record;
    }

    record.outcome = "dropped";

    // Two texts, because they are two different facts: a family with no
    // ri_j_link disk rung at all (the common case - direct / qfmm / gpu /
    // lean), and the ri_j_link route that exists but did not engage (a fitting
    // memory rung rode, which is the ladder behaving by design rather than a
    // defect this text should blame on the family).
    if (resolvedKind == qcx::io::BuilderKind::kRiJLink)
    {
        record.reason =
            "the disk rung did not engage on this run (a fitting in-memory rung rode), so the "
            "chunk-size hint was never read";
    } else
    {
        record.reason =
            "the chunk-size hint sizes the ri_j_link family's storage-module disk-backed "
            "store; fock_builder = \"" +
            std::string(qcx::io::ToString(resolvedKind)) +
            "\" has no disk rung, so the hint was dropped";
    }

    return record;
}

// The engine-decorator factory for the disk-tier ERI store (the
// `[method] eri_cache_store` key; the seam is `EngineDecoratorFactory`,
// ): the closure the builder calls ONCE at Create with the two raw
// engines it would have used itself, returning the pair every later request
// runs through.
//
// Why the store is built HERE rather than beside the options it decorates:
// the engines a store wraps are the BUILDER's own - they close over its pair
// store and its batch sizing - and they exist for the first time inside
// Create. The factory is therefore the only place that can hand the store the
// engines it recomputes misses with.
//
// ONE STORE PER RUN, and the demotion is decided here. The Kohn-Sham
// composition builds a Coulomb and an exchange half out of the SAME options
// struct, so this factory is called once per half: the second call reuses the
// store the first opened rather than opening a second decorator over the same
// file, because two append-only writers over one chunk store is not a tier,
// it is a race. A store that REFUSES - an unwritable path, an HDF5 failure, a
// fingerprint mismatch - is not retried on a later call either: the run
// proceeds on the raw engine pair the builder handed over, and the refusal
// text stays on the handle for the record to disclose (the owner's 2026-09-12
// ruling: honour, else demote to the best runnable choice with the demotion
// disclosed). Returning the engines unchanged rather than an Error is that
// ruling's whole point - a store that cannot open costs the run a cache, it
// does not cost it the run.
// \param handle The run's handle; the factory writes the store or the refusal
// onto it, and the driver reads it after the SCF for the record.
// \param molecule The system (the store's fingerprint identity).
// \param basis The orbital basis (function counts; the fingerprint).
// \param orbitalBasisName The orbital basis-set name (the fingerprint).
// \param auxBasisName The auxiliary basis-set name in effect, or empty.
// \param storePath The store file the input named (created when missing).
// \returns The factory.
qcx::integrals::EngineDecoratorFactory MakeEriStoreFactory(std::shared_ptr<EriStoreHandle> handle,
                                                           const qcx::molecule::Molecule& molecule,
                                                           const qcx::basisset::BasisSet& basis,
                                                           std::string orbitalBasisName,
                                                           std::string auxBasisName,
                                                           std::filesystem::path storePath) {
    return [handle = std::move(handle),
            &molecule,
            &basis,
            orbitalBasisName = std::move(orbitalBasisName),
            auxBasisName = std::move(auxBasisName),
            storePath = std::move(storePath)](const qcx::integrals::EriBatchEngineFn& fp64,
                                              const qcx::integrals::CertifiedEriBatchEngineFn& fp32)
               -> qcx::Result<qcx::integrals::DecoratedEngines> {
#if defined(QcxHasStorage)
        if (handle->store == nullptr && handle->refusal.empty())
        {
            auto opened = qcx::storage::CachedEriBatchEngine::Create(
                storePath, molecule, basis, orbitalBasisName, auxBasisName, fp64, fp32);

            if (!opened.has_value())
            {
                handle->refusal = opened.error().message;
            } else
            {
                handle->store =
                    std::make_unique<qcx::storage::CachedEriBatchEngine>(std::move(*opened));
            }
        }

        if (handle->store == nullptr)
        {
            // The demotion: the raw pair unchanged is what the builder routes
            // through, so an unopenable store leaves the run's arithmetic
            // exactly where it would have been without the key.
            return qcx::integrals::DecoratedEngines{fp64, fp32};
        }

        qcx::storage::CachedEriBatchEngine* store = handle->store.get();
        qcx::integrals::DecoratedEngines decorated;

        if (fp64)
        {
            decorated.fp64 = [store](const std::vector<qcx::integrals::ShellQuartet>& quartets) {
                return store->ComputeEriBatch(quartets);
            };
        }

        // An empty lane stays empty: the builder reads an empty std::function
        // as "this lane is unimplemented" (kUnimplemented on its call), and a
        // decorator answering for a lane nobody injected would change that
        // contract.
        if (fp32)
        {
            decorated.fp32 = [store](const std::vector<qcx::integrals::ShellQuartet>& quartets) {
                return store->ComputeEriBatchCertified(quartets);
            };
        }

        return decorated;
#else
        // No storage module in this build (QCX_ENABLE_IO=OFF): the request
        // cannot be attempted at all, so the factory does exactly what a
        // refused open does - hands the raw engine pair back unchanged and
        // leaves the cause on the handle, where MakeEriStoreRecord reads it
        // for the record's demotion: the run proceeds on the engines it would
        // have used, and the record says why the cache is not there.
        // The captures the store's Create would have read are unused in this
        // arm; the void casts state that rather than leave a warning behind.
        (void)molecule;
        (void)basis;
        (void)orbitalBasisName;
        (void)auxBasisName;
        (void)storePath;
        handle->refusal = StorageModuleAbsentRefusalText();
        return qcx::integrals::DecoratedEngines{fp64, fp32};
#endif
    };
}

// The eri_store request record: the outcome of
// `method.eri_cache_store`, the key whose consumer sits OUTSIDE the builder.
// `integals` cannot name `storage`'s decorator, so the driver
// constructs it, owns it and reads its stats - which makes the run's own JSON
// the only place a reader can learn whether a named disk store was what ran.
// Without this block a run that asked for a store and a run that never asked
// serialize identically, and a request the run quietly dropped is the
// substitution the disclosure rule forbids.
//
// WHAT RAN is read from the STORE that exists and from its OWN traffic, never
// from the fact that a decorator was handed over - and the difference is
// MEASURED, not theoretical. The builder disengages the entire engine tier -
// decorator included - on the class-aware path (a nontrivial symmetry
// reduction, the class-aware) and on the LightPath (whose miss-assembly reads the full
// pair store), and both decisions are taken INSIDE the builder's Create,
// after the driver has handed its factory over. On those runs the factory is
// never called, so no store is ever built: a record that read "a factory was
// installed" would call them "disk" while not one quartet went through the
// decorator, which is exactly the claim this block exists to keep
// unfabricatable. The traffic test below is the same rule applied one step
// further in - it guards a disengagement the driver cannot see at all, and by
// today's two paths it is unreachable (a store that exists is a store the
// engine engaged, so requests reach it); it stays because the claim it
// protects is the block's whole point, and a guard that has to be reasoned
// about is cheaper than a claim that has to be trusted.
//
// The request is written whenever the input NAMED a path, on the honoured and
// the demoted arm alike, because the path is what makes the demotion a
// disclosure rather than a substitution.
// \param input The parsed input (the request's own text).
// \param resolvedKind The builder family the run wired.
// \param leanMember The resolved within-family member choice.
// \param wiring What the wiring did with the request.
// \returns The record, or nothing when no path was named.
std::optional<qcx::io::RunEriStore> MakeEriStoreRecord(const qcx::io::RunInput& input,
                                                       qcx::io::BuilderKind resolvedKind,
                                                       bool leanMember,
                                                       const EriStoreWiring& wiring) {
    if (input.method.eriCacheStore.empty())
    {
        // An omitted (or empty) key is not a request: absent, never a
        // fabricated outcome (the null-honesty rule the sibling blocks
        // follow).
        return std::nullopt;
    }

    qcx::io::RunEriStore record;
    record.path = input.method.eriCacheStore;

#if defined(QcxHasStorage)
    const qcx::storage::CachedEriBatchEngine* store =
        wiring.handle != nullptr ? wiring.handle->store.get() : nullptr;
    const qcx::storage::CachedEriStats* stats = store != nullptr ? &store->Stats() : nullptr;

    if (stats != nullptr && (stats->hitQuartets > 0 || stats->missQuartets > 0))
    {
        // Honoured: the decorator served the run's batches. No reason - an
        // absent reason is never a fabricated "fine".
        record.engaged = "disk";
        record.demoted = false;
        record.hitQuartets = stats->hitQuartets;
        record.missQuartets = stats->missQuartets;
        record.readMs = stats->readMs;
        record.recomputeMs = stats->recomputeMs;

        for (const auto& [cls, hits] : stats->hitQuartetsByClass)
        {
            record.hitQuartetsByClass.push_back(
                qcx::io::RunEriStoreClassHits{cls.first, cls.second, hits});
        }

        return record;
    }
#endif

    // The demotion arms. On a build without the storage module
    // (QCX_ENABLE_IO=OFF) they are the whole of this function's behaviour: no
    // store can exist there, so the deviation above cannot have run and the
    // cause the reader is handed is the build's own, carried on the handle.
    record.demoted = true;
    // The tier that served the run instead: the builder's own in-memory cache
    // where one was in force, the raw engine pair otherwise. Read from the
    // options the builder actually got, never re-derived from the cap.
    record.engaged = wiring.ramTierInForce ? "in_memory_cache" : "none";

#if defined(QcxHasStorage)
    if (stats != nullptr)
    {
        // Opened, and never called: the engine engaged the tier and no batch
        // reached the decorator. No path in this tree produces this (the two
        // disengagements are caught above, where no store exists), so the
        // sentence states the observation and claims no cause.
        record.demotedReason =
            "the store was opened but the builder routed no ERI batch through it, so no quartet "
            "was cached";
        return record;
    }
#endif

    if (wiring.handle != nullptr)
    {
        // A store that reached the factory and refused, or a factory the
        // engine never reached at all. The first carries the store's own
        // error verbatim; the second names the mechanism, which is known and
        // is not a defect of the request - both disengagements are the
        // engine's own documented arrangement, taken inside the builder's
        // Create after this driver's work was done.
        record.demotedReason =
            wiring.handle->refusal.empty()
                ? std::string{"the decorator was installed on the builder's options but the "
                              "engine never called its factory: the whole engine tier - the "
                              "decorator with it - is disengaged by construction where the "
                              "builder takes the class-aware path (a nontrivial symmetry "
                              "reduction is in effect for this molecule) or the LightPath "
                              "(whose miss-assembly reads the full pair store), so no store was "
                              "opened"}
                : wiring.handle->refusal;
        return record;
    }

    // The request never reached a builder at all: this family has no batch
    // engine pair to decorate. Two sentences, because they are two facts -
    // the within-family lean member, and every other family.
    if (leanMember)
    {
        record.demotedReason =
            "the direct family's LEAN member evaluates its quartets through the Schwarz-only "
            "engine and carries no batch engine pair to decorate, so the store was not opened";
        return record;
    }

    record.demotedReason =
        "method.eri_cache_store drives the engine-decorator seam, which only the direct family's "
        "machinery member installs (it is the one member whose FockBuildOptions carry the batch "
        "engine pair a store decorates); fock_builder = \"" +
        std::string(qcx::io::ToString(resolvedKind)) +
        "\" has no such seam, so the store was not opened";

    return record;
}

// The orbit-expansion request record: the outcome of
// method.ri_orbit_expansion, the increment's only driver-side key. The
// outcome comes from the ENGINE's own disclosure
// (WiredFockBuilder::orbitExpansionEngaged, itself read from
// RiJkFockBuilder::OrbitExpansionEngaged) rather than from anything recomputed
// here, so the record cannot disagree with the engine about whether the
// reduction ran - and the request is written beside it, because "asked and did
// not get it" and "never asked" are different facts.
//
// Four outcome words, and the last three are why the block exists: a request
// that lands on the ri_j_link family where the point group is trivial builds
// both reductions and still engages nothing, a request that lands on the disk
// route never meets an orbit-aware builder, and a request on any other family
// has no 3c task grid to reduce. Without this record all four serialize as the
// same run. Nothing here refuses: a disclosure the record can state is the
// contract's remedy, and the key's own refusal - a reduction that cannot
// be built - is raised by name at the wiring point.
std::optional<qcx::io::RunRiOrbitExpansion> MakeRiOrbitExpansionRecord(
    const qcx::io::RunInput& input,
    qcx::io::BuilderKind resolvedKind,
    bool expansionWired,
    bool expansionEngaged) {
    if (!input.method.riOrbitExpansion.has_value())
    {
        // An omitted key is not a request: absent, never a fabricated outcome
        // (the null-honesty rule the sibling blocks follow: an absent key is
        // reported as unreported, never as a drop).
        return std::nullopt;
    }

    qcx::io::RunRiOrbitExpansion record;
    record.requested = *input.method.riOrbitExpansion;

    if (!record.requested)
    {
        // Named and false: the plain walk ran because that is what the input
        // asked for, not because anything refused.
        record.outcome = "not_requested";
        return record;
    }

    if (!expansionWired)
    {
        // Two ways to be unwired, and they are different sentences: a family
        // with no such grid at all, and the ri_j_link family's own disk route,
        // whose storage-module builder is not an orbit-aware one.
        //
        // The second sentence says what the family does with the KEY, not what
        // the family does about symmetry: the direct family's class path
        // engages on its own (from the molecule's point group, no input key)
        // and reports through its own statement, so claiming the family "is
        // wired without a point-group reduction" would be false there. What is
        // true on every family this arm can still be reached on is that the
        // key is not consumed, so the request did not run.
        record.outcome = "not_applicable";
        record.reason =
            resolvedKind == qcx::io::BuilderKind::kRiJLink
                ? std::string("this ri_j_link run built the storage-module disk-backed store, "
                              "which evaluates the 3c tensor chunk-at-a-time and takes no "
                              "point-group reduction, so the expansion did not run")
                : std::string("method.ri_orbit_expansion drives the ri_j_link family's walk over "
                              "its 3c (bra pair x aux shell) task grid; fock_builder = \"") +
                      std::string(qcx::io::ToString(resolvedKind)) +
                      "\" does not consume the key, so the expansion it asks for did not run";
        return record;
    }

    if (expansionEngaged)
    {
        // Asked and got it. No reason: an absent reason is never a fabricated
        // "fine".
        record.outcome = "engaged";
        return record;
    }

    // Asked, both reductions built, and the engine still did not engage the
    // mechanism: the only way that happens is a group of order 1 on one of the
    // two bases (the identity action - every cell is its own orbit), which is
    // a property of the molecule, not a defect of the request.
    record.outcome = "inert_trivial_group";
    record.reason =
        "method.ri_orbit_expansion was requested, but the point group's order is 1 on the orbital "
        "or the auxiliary basis, so every task-grid cell is its own orbit and the expansion is the "
        "identity";
    return record;
}

// One unique scratch-store path for the disk route's chunked tensor (the
// disk-rung knob): the OS temp dir plus a per-process
// counter, remove-first - a stale file from a crashed run must not meet
// the append-only refusal (a store path is single-builder scratch, and
// Create refuses a file that already carries chunks). The Windows
// one-open-handle removal caveat is handled at the WiredFockBuilder level,
// not here.
qcx::Result<std::filesystem::path> MakeRiDiskStorePath() {
    static std::atomic<int> counter{0};
    std::error_code directoryError;
    const std::filesystem::path directory = std::filesystem::temp_directory_path(directoryError);

    if (directoryError)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "the disk rung's scratch store needs a writable temp "
                                   "directory: " +
                                       directoryError.message()));
    }

    std::filesystem::path path =
        directory / ("qcx_ri_j_disk_" + std::to_string(counter.fetch_add(1)) + ".h5");

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return path;
}

// The lean selector this file used to carry (IsLeanDirectSelection: the
// Lean rule, "an ABSENT fock_builder at nBasis <= 1000, or the explicit
// lean word") is GONE with the 2026-09-13 ladder, and it is not replaced
// here. Under the ladder the member choice is the RESOLUTION's
// (ResolvedBuilderSelection::leanMember) and every consumer reads it from
// there: the pre-gate setup admission, the RHF wiring branch, the UHF
// runner's lean arm, the certified-lane route judgement and the record.
// The old shape re-derived the choice from (input, nBasis) in each of
// those places, and under the ladder that derivation is WRONG for a
// demoted run - a run whose size names ri_j_link or qfmm but which the
// ladder sends to the lean member would have been sized and wired as the
// machinery while the record named lean.

// The no-key ladder's tier runnability for one run (the owner's ruling
// 2026-09-13): the two facts the resolution's demotion reads, stated once
// here because they are the DRIVER's wiring knowledge - the resolution
// module must not re-derive them, and a second spelling here would let the
// demotion text and the wiring's own refusal describe one restriction
// differently.
//
// ri_j_link is unavailable in two cases:
//   - no v1 arm for the method or for one of the family's rungs: on the
//     unrestricted leg the disk rungs have no per-spin adapter, and on the
//     Kohn-Sham lanes they have no half to hand the energy seam. Both refuse
//     the key by name in ValidateCombination too, and the ladder must not
//     route around a refusal it would make for the same request. DATED: this
//     reason USED to name the Kohn-Sham lanes themselves, when the whole
//     family was unwired there - measured live (a no-key UHF run at 1210
//     basis functions demotes to the lean member, and a non-hybrid Kohn-Sham
//     run at 200 demoted off the tier the two-tier order had promoted).
//   - no auxiliary basis the wiring can resolve. DATED: an early dry run
//     measured this as live under the pre-ruling rule, which refused every
//     orbital name outside def2-* and cc-* (the constraint that left
//     ri_j_link unestablished). The ruling of
//     2026-09-13 replaced that rule with a quality tier that always
//     produces a default, so an unmatched orbital NAME no longer reaches
//     this arm - it stays for the degenerate names, and so that a rule
//     which refuses again is demoted rather than run.
// qfmm is runnable on the Kohn-Sham lanes since 2026-09-16 (the composition
// over QfmmHfFockBuilder's two exposed halves), so no reason is stated for
// it anywhere. The reason that stood here - "its composed builder adds the
// two halves inside one BuildFock and exposes neither" - was the STRUCTURAL
// one, and it was answered by exposing them rather than by argument; the
// ladder's knowledge must track the wiring, because a reason left standing
// would demote a no-key Kohn-Sham run above the 2000-basis-function boundary
// to ri_j_link while the wiring would have run qfmm.
//
// The aux answer is the NAME-level one (does a rule or an explicit key
// provide an aux name), deliberately: the aux DIRECTORY parse happens in
// the wiring, behind the setup ramp, and a failed parse of an explicitly
// provided aux is a user error with its own error, not a demotion - and an
// explicit request is never demoted at all .
struct LadderTierRunnability {
    std::optional<std::string> riJ; ///< The reason ri_j_link cannot run (empty = it can).
    std::optional<std::string> qfmm; ///< The reason qfmm cannot run (empty = it can).
};

LadderTierRunnability LadderRunnabilityFor(const qcx::io::RunInput& input) {
    LadderTierRunnability availability;

    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value())
    {
        // Unreachable in the driver's own order: ValidateCombination
        // resolves the same path (and refuses an unclassified word) before
        // the resolution runs. Stated as a reason rather than an empty one
        // so a caller reaching this out of order gets a demotion with a
        // why, never a silent stay on a tier nothing established.
        const std::string reason = "the run path is unresolved: " + scfPath.error().message;
        availability.riJ = reason;
        availability.qfmm = reason;
        return availability;
    }

    // The v1 family whitelists (ValidateCombination, byte for byte): the
    // Kohn-Sham lanes are wired on the direct family, the composed QFMM
    // builder and the RI-J link, and UHF with direct, qfmm and ri_j_link -
    // the per-spin adapters (and, for Kohn-Sham, the half the energy seam
    // contracts) of the other builders do not exist.
    //
    // The Kohn-Sham branch does NOT return early any more. It did while
    // BOTH upper tiers were unwirable there, because then no cell of the
    // rest of this function could apply to a Kohn-Sham run. The RI-J link's
    // composition landed, so ri_j_link is now runnable on this path and the
    // remaining rules are its too - above all the aux rule at the bottom,
    // which a Kohn-Sham ri_j run needs exactly as its Hartree-Fock twin
    // does. The `else` matters as well as the fall-through: a UKS run is
    // BOTH unrestricted and Kohn-Sham (IsUnrestrictedPath names
    // kUnrestrictedKs), so without it the UKS branch below would apply the
    // UHF disk rule under the Kohn-Sham one's nose.
    //
    // The qfmm tier left this branch on 2026-09-16 with its composition, so
    // what remains is the rungs' rule alone - and the branch now states NO
    // tier reason at all, which is the honest shape: every tier the size
    // ladder names above the lean member is wirable on a Kohn-Sham run.
    if (IsKohnShamPath(*scfPath))
    {
        // The rungs this path still cannot wire. Same reason as the
        // combination check's (one rung, one text): the storage-module disk
        // builder composes a fused Hartree-Fock Fock and exposes no half.
        if (input.method.riTensorMode == qcx::io::RiTensorMode::kDisk ||
            input.diagnostics.forceDiskRi)
        {
            availability.riJ = RiJLinkKsDiskRungRefusalText();
        }
    } else if (IsUnrestrictedPath(*scfPath))
    {
        // The unrestricted leg's own remaining cell: the per-spin adapter
        // landed with the RI-J link family (RunRiJLinkUhfScf), so what is
        // left unwired here is the family's disk rungs - refused by name for
        // an explicit family word (ValidateCombination), and a DEMOTION
        // REASON here for the absent one, which is the ladder's own rule
        // rather than a second refusal: a no-key unrestricted run that names
        // a disk knob is sent to the tier it can wire (the lean member)
        // instead of onto a tier that would refuse it. The Kohn-Sham lanes
        // own the same cell with their own reason, in the branch above - and
        // they must, because a UKS run reaches this branch too and the
        // reason a UKS run gets has to be about its own run.
        if (input.method.riTensorMode == qcx::io::RiTensorMode::kDisk ||
            input.diagnostics.forceDiskRi)
        {
            availability.riJ = RiJLinkDiskRungRefusalText();
        }
    }

    if (!availability.riJ.has_value())
    {
        auto auxName = AuxNameInEffect(input, qcx::io::BuilderKind::kRiJLink);

        if (!auxName.has_value())
        {
            availability.riJ = auxName.error().message;
        }
    }

    return availability;
}

// Whether the resolved route carries the certified fp32 lane's REQUEST into a
// live fp32 lane (the owner's ruling 2026-09-13, the [method]
// force_certified_lane key). The request travels in exactly ONE field -
// FockBuildOptions::useCertifiedMixedPrecision, which ResolveCertifiedLane is
// the one reader of - so the routes that can honour it are the ones that
// assemble a FockBuildOptions for the direct family's MACHINERY members: the
// RHF wiring's directOptions, and the direct-UHF runner's coulomb/exchange
// halves (which the UKS composition copies and the Fukui charged species
// inherit through the same runner).
//
// Every other route DROPS the request rather than honouring it, and each is
// refused BY NAME at the resolution point instead of computing as if the key
// were absent:
//   - the direct family's LEAN member (the auto-lean default at nBasis <=
//     1000, or an explicit fock_builder = "lean", on both the RHF and the UHF
//     legs): LeanFockBuildOptions has no such field AND the lean builder runs
//     no fp32 lane at all, so the request is dropped BY CONSTRUCTION - not by
//     oversight - and lean is the DEFAULT small-molecule route;
//   - ri_j_link / qfmm / gpu: RiEngineOptions has no such field, and the
//     families' own lanes (the RI-J exchange half's, derived from the preset
//     at ri_engine.cpp:1788; QfmmOptions::useCertifiedMixedPrecision, a field
//     the driver does not seed - the gap recorded as an open item and deliberately NOT
//     widened here) do not read the request.
//
// The same judgement, in the same shape, that the sibling [method]
// enforce_certified_bound key's resolution point makes: the two keys name the
// same lane and neither may vanish. It is NOT the same predicate - the
// enforcement is additionally unreachable on both UHF legs, whose wiring never
// assigns its field, while the lane's request travels to the UHF machinery arm
// with the compute profile - and the routes here are judged on the RESOLVED
// kind plus the RESOLVED member choice, so this cannot disagree with the
// builder that actually runs.
bool RouteCarriesCertifiedLaneRequest(qcx::io::BuilderKind kind, bool leanMember) {
    return kind == qcx::io::BuilderKind::kDirect && !leanMember;
}

// The lean member's name in the run record: the same word the
// [method].fock_builder key accepts for the explicit within-family request,
// so the record's spelling and the input's cannot drift apart.
constexpr const char* kLeanMemberName = "lean";

// The ri_jk family's member word (the ruling stated below). It is
// NOT the family word: "ri_jk" names the family, and what a consumer of the
// energy must be able to read off the record is the CONTRACTION FORM - occ-RI-K,
// the exchange contracted in the occupied-orbital basis rather than evaluated
// as quartets. A reader who saw "ri_jk" beside "ri_jk" would learn nothing about
// the half of the path the approximate result actually comes from.
constexpr const char* kOccRiKMemberName = "occ_ri_k";

// The exchange-contraction word of a run whose exchange half was NOT
// approximated (the notice-only block, RunApproximation::exchange).
// It states the reading the ABSENT block already carries - an exact kernel -
// on a block that is present only because the aux selection warns, so a reader
// who sees a warning beside an unfitted kernel does not have to reconstruct
// which half the warning is about. The word is the one
// RunApproximation::exchange documents for this arm and the one the io-side
// contract pins (`approximation_notice_test.cpp`).
constexpr const char* kExactKernelExchangeWord = "exact";

// The within-family member name of a resolved (kind, lean) pair: the lean
// member's own name when the selector above wired it, the family's own
// builder word otherwise. The family word itself keeps its slot in the
// record's `builder` (and in memory_model.builder) - the member name is the
// additive sibling that names what the family word cannot, so a consumer
// reads the member instead of inferring it from an absent explicit_builder
// (the record must not need inference - a consumer reads the answer the
// producer wrote, it does not reconstruct it).
std::string BuilderMemberName(qcx::io::BuilderKind kind, bool lean) {
    if (lean)
    {
        return std::string(kLeanMemberName);
    }

    // The one family whose member word is not its own family word (the ruling).
    // Stated HERE, in the single function that produces the record's
    // builder_member, so the member the record names and the member the wiring
    // executed cannot disagree - which is the whole point of the key.
    if (kind == qcx::io::BuilderKind::kRiJk)
    {
        return std::string(kOccRiKMemberName);
    }

    return std::string(qcx::io::ToString(kind));
}

// The ladder's ONE exception as a fact for the resolution (the owner's
// ruling 2026-09-13, refining the same day's ladder): a Kohn-Sham run whose
// functional carries NO exact exchange. The key is the functional's
// CHARACTER, never the method - a hybrid Kohn-Sham run keeps the shared
// order because it needs K - so this resolves the NAME against the registry
// (the same lookup ResolveKsContext runs later) and reads the method only to
// decide whether a functional applies at all. Keying on "is it Kohn-Sham"
// would silently mis-order every hybrid DFT run.
//
// An absent or unknown name states FALSE - the shared order - with the run's
// own refusal naming what it could not resolve (ResolveKsContext refuses it
// by name; this judgement is never the one that reports it).
bool IsNonHybridKohnSham(const qcx::io::RunInput& input) {
    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value() || !IsKohnShamPath(*scfPath) || !input.method.functional.has_value())
    {
        return false;
    }

    const auto functional = qcx::driver::internal::ResolveKsFunctional(*input.method.functional);
    return functional.has_value() && functional->exchangeFraction <= 0.0;
}

// The engine's own last-resort convention, adopted rather than invented:
// a Create-time envelope refuses when it exceeds its ceiling by more than
// this factor (integrals/src/lean_fock_build.cpp kRefusalSlack, "the
// generous slack"), which is how the engine covers the process baseline
// its own terms do not charge - the image, the parsed basis and the
// runtime sit on top of every term it sums. The ramp admission uses the
// same rule for the same reason: SetupRampPeakBytes is never under the
// RAMP's allocation, and the baseline rides above it, which is why the
// pin this arm exists for sits just above the raw bound - the measured
// 582.1 MiB of commit at C50H102/def2-SVP against a ~464 MB bound, the
// ~118 MB difference being exactly that baseline.
//
// The slack itself is NOT stated here: it is integrals::kSetupAdmissionSlack,
// beside the reserve that both this arm and the workspace grant read, so the
// two floors cannot disagree about it.

// The live-matrix count this arm passes is kRampLiveMatrices, declared with
// the workspace-grant helpers at the top of this file: the reservation the
// grant makes reads the same constant, and the earliest budget site precedes
// this arm.

// The setup ramp's refusal text: the bytes the ramp needs, the cap they do
// not fit, and what actually moves the number. The ramp is
// family-independent (the core-Hamiltonian and overlap builds every path
// pays), so this text names no builder and carries no builder rung - the
// ladder's rungs trade SCF-time memory, and the ramp is not SCF-time
// memory. The bytes are printed exactly rather than rounded to GiB: a
// positive value rendered as "0 GiB" is the defect OneDecimal's note
// records, and an exact byte count cannot fall into it at any size.
std::string SetupRampRefusalMessage(double setupBytes, double memoryCapGiB) {
    return "the run's setup ramp needs " + std::to_string(static_cast<std::size_t>(setupBytes)) +
           " bytes at this input (the shared setup ramp - the core-Hamiltonian and overlap "
           "builds walk the canonical pair list - plus the Schwarz screening sweeps every "
           "budgeted builder runs in front of its own budget decision) but the resource cap is " +
           std::to_string(memoryCapGiB) +
           " GiB: the setup would allocate past the cap before any builder's own Create-time "
           "estimate is consulted. Reinstatement options: (1) raise memory_cap_gib in "
           "[resources]; (2) a smaller orbital basis or auxiliary basis (the pair stores grow "
           "with the canonical pair counts and the sweeps' arenas with their chunk sizes).";
}

// The pre-gate setup admission (the above-ceiling crash class, the adaptive-seam/
// The spawn-site family's newest consumer): the decision the run's own gates make
// must be CONSULTABLE BEFORE the setup ramp those gates sit behind.
//
// RunDriver's shared setup ramp - BuildCoreHamiltonian (the kinetic and
// nuclear one-electron matrices) and BuildOverlapMatrix - is what every
// path pays before its first gate, and at the time of the defect it walked
// the WHOLE contracted pair store once per matrix (BuildPairMatrix through
// BuildPairData): three sequential materializations of ~2.4 KB per
// canonical shell pair that no estimate charged and no gate consulted,
// because every gate lived behind them (the lean member's Create-time
// ceiling, and the memory model the driver then carried). Below the ramp's own commit
// footprint a cap therefore killed the process 0xC0000409 INSIDE the ramp -
// pre-iteration, empty trace and stats, no admission decision of any kind -
// and identically on the explicit-lean, explicit-direct and absent-key legs,
// because the ramp is what every path pays (measured at C50H102/def2-SVP,
// 1210 basis functions: the crash band reached 0.90 GiB on the lean leg and
// 0.54 GiB on the machinery legs; the lean member's own envelope already
// CHARGES the ramp - EstimatePeakBytes' runStoreBytes term,
// lean_fock_build.cpp - it simply could not run before it).
//
// Since 93d5dfc5 the ramp builds its pair data CHUNK-WISE: the geometry-only
// skeleton stays resident, one kPairChunkBytes chunk of contracted data is
// built into it, and the chunk is released COMPLETELY before the next
// (ReleaseChunkPairData, md_batch.hpp), so the residency is the skeleton
// plus one chunk and NOT the store - 9.9495 -> 0.7586 GiB per call at
// C42H86/def2-QZVP (MODELLED), which is also the shape the lean envelope
// charges. Where a store fits one chunk the chunking changes nothing, and
// the fixture the tight-cap pin below runs on is such a size: C50H102/
// def2-SVP's store is 441,172,808 B over 183,921 canonical pairs
// (PairStoreBytes, footprint.hpp; 606 shells, 1210 functions), under the
// 512 MiB cap, so the ramp there still materializes the whole store - 582.1
// MiB of commit measured on the machinery leg, 942 MiB on the lean leg -
// and the 0.5 GiB cap still sits under it.
//
// This pre-gate check runs the SAME admission the wiring would apply, one step
// earlier: the lean member's own envelope (EstimateLeanEnvelope - the
// builder's number and text, byte for byte), so no run is admitted or
// refused here that the wiring would have decided differently - only the
// ordering changes, and the ramp is never entered under a cap its own
// admission already refuses.
//
// The check has TWO arms, and the second is the crash class's own answer.
//
// The lean member's arm runs the SAME admission the wiring would apply, one
// step earlier: its own envelope (EstimateLeanEnvelope - the builder's
// number and text, byte for byte), so no run is admitted or refused here
// that the wiring would have decided differently. That envelope already
// charges the ramp (its runStoreBytes term).
//
// The second arm is the ramp ITSELF, for every family whose own admission
// sits BEHIND it. The model's deletion (2026-09-17) removed the arms that
// covered the direct machinery, the composed QFMM and the aux-parsed ri_j
// budget path, and those families reach their builder Creates only AFTER
// the ramp has run - so a cap below the ramp's footprint killed the process
// inside the ramp before any of their exact estimates was consulted. That
// is not a theory: it is MEASURED, on this pin's own fixture
// (C50H102/def2-SVP, 1210 basis functions, cap 0.5 GiB, the absent-key leg:
// exit 0xC0000409 inside the ramp, where the leg refused before the
// deletion). The bound is the ENGINE's own (integrals::SetupRampPeakBytes
// - the same composition the lean envelope charges, so the two can never
// disagree about what the ramp costs), never a second copy of the
// arithmetic and never a hand-maintained table.
//
// The comparison is against the cap ITSELF, with no slack: the ramp's bytes
// are the allocation the run is about to perform, so a ramp that does not
// fit the cap has no rung to fall to (the ladder trades SCF-time memory,
// not the ramp's) and the refusal is the only honest verdict. The lean arm
// above keeps its own generous-slack rule, which is the builder's.
//
// Deliberately NOT covered: the GPU family (the CUDA-less build must keep
// failing with its own kUnimplemented, not a memory verdict). It keeps its
// wiring-side gates unchanged.
//
// \param input The run input (the method type, the cap's source).
// \param kind The resolved builder kind.
// \param molecule The molecule the ramp will run on.
// \param basisSet The parsed orbital basis.
// \param memoryCapGiB The cap in effect (0 = none).
// \param leanMember The resolved member choice (the builder resolution's
// ResolvedBuilderSelection::leanMember): the check sizes the admission of
// the member that will actually run, so under the 2026-09-13 ladder a run
// DEMOTED to the lean member is admitted against the lean envelope rather
// than the machinery's admission.
// \returns The run's workspace RESERVE in bytes - the same number this arm
// refused on, handed back so the workspace the wiring then builds is granted
// the cap MINUS it (WorkspaceGrantBytes). Zero when no setup arm ran: a lean
// member's admission IS its own whole-run envelope (no ramp reservation is
// missing for it), and the GPU family is skipped below. The refusal is the
// error the wiring would have raised (kInvalidArgument), taken before the ramp
// allocates anything.
qcx::Result<double> CheckPreGateSetupAdmission(const qcx::io::RunInput& input,
                                               qcx::io::BuilderKind kind,
                                               const qcx::molecule::Molecule& molecule,
                                               const qcx::basisset::BasisSet& basisSet,
                                               const qcx::basisset::BasisSet* auxBasisSet,
                                               double memoryCapGiB,
                                               bool leanMember) {
    // 0 = no cap: ApplyProcessCaps applies nothing, so nothing can die at
    // the cap, no workspace budget is constructed (a zero cap is the
    // documented escape hatch onto the engine's legacy null-budget path),
    // and whatever gate remains is the wiring's own - the pre-existing
    // behavior, unchanged.
    if (memoryCapGiB <= 0.0)
    {
        return 0.0;
    }

    // The lean member of the direct family: its Create-time ceiling is a
    // function of (molecule, basis set, options) alone - the core
    // Hamiltonian Create consumes is not one of its inputs - so the
    // envelope is computable here. Both method types reach it (the unrestricted
    // seam gave the UHF wiring its lean arm), and the UHF leg consults the
    // same number the RHF leg does - the envelope reads the preset and the
    // window plan, neither of which the half-mode split flags touch, so
    // the two lean builders a UHF run creates carry an identical envelope
    // and each Create applies the check below verbatim. The decision is
    // therefore the wiring's own, one ramp earlier, on both legs.
    if (kind == qcx::io::BuilderKind::kDirect && leanMember)
    {
        // The options the WIRING will build, field for field - the point-group
        // classification and the orbit engagement included, because
        // EstimatePeakBytes charges the retained pair-orbit tables when they
        // are engaged and that charge belongs in this admission's number.
        // Checking a disengaged envelope would admit a run the builder's
        // own Create-time check then refuses, which is the drift this function
        // exists to prevent.
        qcx::integrals::LeanFockBuildOptions leanOptions;
        leanOptions.accuracy = input.method.accuracy;
        leanOptions.memoryCapGiB = memoryCapGiB;

        auto leanReduction = qcx::scf::BuildSymmetryReduction(molecule, basisSet);

        if (!leanReduction.has_value() &&
            leanReduction.error().code != qcx::ErrorCode::kUnimplemented)
        {
            return std::unexpected(leanReduction.error());
        }

        if (leanReduction.has_value() && leanReduction->groupOrder > 1)
        {
            leanOptions.symmetryReduction = &*leanReduction;
            leanOptions.symmetryOrbitExpansion = true;
        }

        auto envelope = qcx::integrals::EstimateLeanEnvelope(molecule, basisSet, leanOptions);

        if (!envelope.has_value())
        {
            return std::unexpected(envelope.error());
        }

        if (!envelope->refusal.empty())
        {
            return std::unexpected(
                Err(qcx::ErrorCode::kInvalidArgument, envelope->refusal + kPreGateSetupClause));
        }

        // No ramp reservation is missing for the lean member: its admission
        // above IS its own whole-run envelope, which already charges the
        // process baseline this arm's reserve exists to cover. Returning zero
        // here is what keeps a demoted run's grant exactly where it was.
        return 0.0;
    }

    // The setup arm: every family that reaches its own admission only after
    // the setup has run (the block comment above states why, and what it
    // cost when this arm did not exist). Skipped for the GPU family, whose
    // CUDA-less build must keep failing with its own kUnimplemented.
    //
    // The bound is the SETUP's, not the ramp's alone. It was the ramp's
    // alone until 2026-09-18, and the gap that left was MEASURED on the
    // C50H102/def2-SVP fixture, leg `absent_key`, fixture
    // `lean_above_ceiling`: the ramp-only bound read 483833976 B, so at a
    // 1.0 GiB cap this arm correctly did not fire - and the process then
    // died 0xC0000409 writing NO diagnostic, because the Schwarz screening
    // sweeps the builder's Create runs in front of its own budget decision
    // had already realized the MD kernels' per-thread batch arenas
    // (md_attribution.hpp ThreadScratchVector) at the caller's
    // maxBatchBytes, never at the budget. The instrumented trace of that
    // route measures 683.6 MiB of live arenas and a 1690.7 MiB process peak
    // BEFORE the RI-J estimate is ever evaluated. So the death band was
    // 0.676 - ~1.7 GiB: below it the ramp alone refused, above it the
    // engine's own Create-time estimate does, and inside it NOTHING did.
    //
    // The sweep term is the engine's own (SchwarzSweepBytes, composed into
    // SetupRampPeakBytes' bound as internal::SetupPeakBytes), never a second
    // copy of the arithmetic and never a hand-maintained table.
    //
    // Falsified here, and NOT relied on: an earlier reading held that the
    // same leg "does not die at 1.5 GiB, it runs past 300 s". That was a
    // LEG-ATTRIBUTION error - in the multi-leg child, at >= 1.30 GiB the
    // FIRST leg stops refusing and starts a real SCF, so legs 2 and 3 are
    // never reached and the timeout was leg 1 running. Isolated, leg 3 at
    // 1.5 GiB dies at 1457.2 MiB in 11.8 s. Do not rebuild that invariant.
    //
    // Standing answer to the structural question, recorded so it is not
    // re-derived: the ROOT fix is to move the builder's budget decision in
    // front of its own sweeps, so the arenas are sized by the budget rather
    // than by maxBatchBytes. That is circular as it stands - the decision's
    // estimate consumes the sweeps' OWN outputs (the counted pattern, the
    // counted task grid, the pair lists), so the decision cannot precede the
    // work whose size it needs. That circularity is exactly why this
    // pre-gate arm exists, and it is the reason to keep it rather than to
    // treat it as a stopgap.
    if (kind == qcx::io::BuilderKind::kGpu)
    {
        return 0.0;
    }

    // THE ONE SOURCE: the reserve the workspace grant also reads, so this
    // refusal and that grant are the same number by construction
    // (integrals::SetupAdmissionReserveBytes - the ramp-and-sweeps bound times
    // the one slack constant, composed in the engine).
    const auto reserveBytes = qcx::integrals::SetupAdmissionReserveBytes(
        molecule, basisSet, auxBasisSet, kRampLiveMatrices);

    if (!reserveBytes.has_value())
    {
        return std::unexpected(reserveBytes.error());
    }

    if (*reserveBytes > memoryCapGiB * kGiB)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                SetupRampRefusalMessage(*reserveBytes / qcx::integrals::kSetupAdmissionSlack,
                                        memoryCapGiB) +
                    kPreGateSetupClause));
    }

    return *reserveBytes;
}

// Wires the RHF Fock builder per the input's builder selection: the
// QFMM, GPU (CUDA build only - the plain build fails loudly instead of
// falling through to the direct builder, the silent-substitution failure
// mode the schema rejects by design), RI-J-link, and direct paths. The RI
// path auto-selects the aux basis when [basis].aux is absent.
//
// Every branch except ri_j runs the memory seam first - the modeled peak
// at (builder, n, nAux) against the cap - and refuses with the
// reinstatement ladder before a builder Create can allocate the dense
// tensor (the C60 trap: Create itself is where the over-cap death used to
// happen). The ri_j branch delegates the admission to the engine's
// Create-time estimate (the budget path): it sizes the cap-minus-base workspace
// budget after the aux parse, which is the earliest point the aux count
// is known, and refuses only when the cap does not clear the base term; a
// zero cap escapes to the legacy null-budget path.
qcx::Result<WiredFockBuilder> WireRhfFockBuilder(
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    qcx::io::BuilderKind kind,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const CpuTensor2& coreTensor,
    const std::filesystem::path& root,
    const qcx::backend::GpuComputeProfile& deviceComputeProfile,
    std::optional<bool> certifiedLaneRequest,
    // The resolved member choice (ResolvedBuilderSelection::leanMember):
    // which member of the direct family this run wires. It is the
    // resolution's own answer, never re-derived here - under the
    // 2026-09-13 ladder a run above the 1000 boundary can be sent to the
    // lean member (a demotion), and the old derivation would have wired the
    // machinery under the record's "lean" name.
    bool leanMember,
    ScfCallStatsStream* statsSink = nullptr,
    CertifiedBoundAccumulator* boundAccum = nullptr,
    // The disk-tier ERI store's run-owned wiring (method.eri_cache_store):
    // filled here where the direct family's options are assembled, read by
    // the caller once the SCF has run. The `statsSink`/`boundAccum` pattern
    // beside it - a collector the run owns and the wiring writes.
    EriStoreWiring* eriStore = nullptr) {
    qcx::integrals::FockBuildOptions directOptions;
    directOptions.accuracy = input.method.accuracy;
    // The run's device compute profile: without it the machinery
    // member's certified fp32 lane resolves on the probe's documented
    // "unknown" profile, i.e. the conservative interim, and the run's own
    // measured device verdict never reaches the builder that routes the
    // quartets. The profile is probed once at the run boundary (RunDriver,
    // beside the topology probe) and travels here as the caller's value:
    // this function constructs builders per instance, and the probe's
    // contract - and its CUDA-runtime init under a memory cap - is once per
    // run.
    directOptions.deviceComputeProfile = deviceComputeProfile;
    // The certified fp32 lane's REQUEST (the owner's ruling 2026-09-13, the
    // [method] force_certified_lane key): the ONE resolution point in
    // RunDriver has already decided that this route carries the request, so
    // what arrives here is either an explicit `true` (the lane is forced on
    // for this build, whatever the profile above measures) or nullopt (the
    // field stays unset and ResolveCertifiedLane reads the profile's
    // verdict - the only other branch). It is NOT a second judgement: the
    // route test (the direct family's machinery member) and the within-family
    // member choice (the resolution's leanMember) are the resolution point's
    // own, and the lean member below - which builds LeanFockBuildOptions and
    // has no fp32 lane - is refused there by name rather than reached with a
    // request it would have to drop.
    directOptions.useCertifiedMixedPrecision = certifiedLaneRequest;
    // The global certified-bound budget enforcement (the verdict's second half):
    // the [method] switch travels to the machinery member's options. The
    // ROUTE question - which runs may ask for this at all - is settled at
    // the resolution point in RunDriver, before any builder is wired: only
    // this machinery member reaches here (the lean member below builds
    // LeanFockBuildOptions, which has no enforcement field, and every other
    // family and both UHF legs drop the option), so a request that could not
    // be enforced was already refused by name. What is left here is the
    // WITHIN-route half: this member's FastPath honours the flag, and its
    // other rungs (the LightPath chunk loop, an engaged precision ladder)
    // fail inside BuildFock rather than computing as if the key were absent.
    directOptions.enforceCertifiedBoundBudget = input.method.enforceCertifiedBound;

    // The disk-tier ERI store (`[method] eri_cache_store`), and this
    // is its one install site. The key's own doc states the arrangement: the
    // driver constructs `storage`'s decorator, owns it for the run and reads
    // its stats afterwards, because `integrals` cannot name `storage` .
    //
    // Only this family reaches here with the seam available: the request is
    // installed on the machinery member's options, and the two things that
    // cannot carry it are decided where they are known - the LEAN member
    // builds LeanFockBuildOptions (no engine pair at all), and every other
    // family builds its own options struct. Both are demotions with the
    // reason recorded, never refusals: a store that cannot be used costs the
    // run a cache, not the run. A build without the storage module
    // (QCX_ENABLE_IO=OFF) is the third such case and the only one that is not
    // about the request: the factory is installed all the same and writes the
    // build's own cause onto the handle, because that is where the record
    // reads the demotion's reason from.
    //
    // The installer follows the ENGINE's own engagement rule rather than a
    // second copy of it. It does not matter that the class-aware path
    // and the LightPath disengage the whole tier inside Create: the decorator
    // is simply never called there, and the record reads the store's own
    // traffic, so an installed-but-unused store is disclosed as the demotion
    // it is (MakeEriStoreRecord). Reproducing those two conditions here would
    // be a second opinion about a decision the engine owns.
    const bool eriStoreRequested = !input.method.eriCacheStore.empty();

    if (eriStore != nullptr && eriStoreRequested && kind == qcx::io::BuilderKind::kDirect &&
        !leanMember)
    {
        auto auxName = AuxNameInEffect(input, kind);

        if (!auxName.has_value())
        {
            return std::unexpected(auxName.error());
        }

        eriStore->handle = std::make_shared<EriStoreHandle>();
        // The path the input named is used as written: the EriStore layer
        // creates a missing file and verifies the fingerprint of an existing
        // one, so a stale or foreign store at the path is a demotion the
        // record discloses rather than a run this site silently redirects.
        directOptions.engineDecorator = MakeEriStoreFactory(eriStore->handle,
                                                            molecule,
                                                            basis,
                                                            input.basis.orbital,
                                                            *auxName,
                                                            input.method.eriCacheStore);
        // The tier that would serve an unengaged request, read from the same
        // options struct the builder gets.
        eriStore->ramTierInForce = directOptions.maxCacheBytes > 0;
    }

    // The RHF exchange-contraction occupancy:
    // the fused direct and the RI-J exchange-half calls contract K against
    // the closed shell's occupied spatial orbitals - ElectronCount()/2
    // (the RHF branch is closed-shell only; the open-shell runs go through
    // RunDirectUhfScf, which passes its per-spin counts instead).
    const std::size_t rhfOccupied = static_cast<std::size_t>(molecule.ElectronCount() / 2);

    // The run path this wiring is serving, resolved through the ONE
    // classifier (the same value the up-front combination check read and the
    // dispatch below acts on) - never a second reading of the raw method word.
    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value())
    {
        return std::unexpected(scfPath.error());
    }

    const bool isKs = IsKohnShamPath(*scfPath);

    // The Kohn-Sham builder whitelist, enforced at the wiring as well as up
    // front (ValidateCombination): the composition is written against
    // builders that expose their two halves SEPARATELY, and the families that
    // add them together inside one BuildFock have no half to hand the energy
    // seam - so without this guard a Kohn-Sham run on one of them would fall
    // through to the shared machinery tail below, be built as a fused
    // Hartree-Fock Fock, and leave the seam's companion callbacks empty,
    // which is the one state the run site must never see. The refusal is
    // stated here so the wiring's own admission cannot be bypassed by a
    // caller that reached it another way.
    //
    // The admitted set is validated in the same commit as the refusals, and
    // it is the combination check's set (one whitelist, two reads): the
    // direct family, the composed QFMM builder, the RI-J link and the composed
    // full-RI pair. Each of the four owns its own Kohn-Sham arm below
    // (MakeDirectKsHalf / MakeLeanKsHalf, MakeQfmmKsHalf, MakeRiJLinkKsHalf,
    // MakeRiFullKsHalf); gpu and gpu_split reach neither this guard's admitted
    // set nor a branch of their own on this path.
    if (isKs && kind != qcx::io::BuilderKind::kDirect && kind != qcx::io::BuilderKind::kQfmm &&
        kind != qcx::io::BuilderKind::kRiJLink && kind != qcx::io::BuilderKind::kRiJk)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kUnimplemented,
                "a Kohn-Sham run is wired on the direct family, the composed QFMM builder, the "
                "RI-J link and the composed full-RI pair (ri_jk), and fock_builder = \"" +
                    std::string(qcx::io::ToString(kind)) +
                    "\" was resolved for it: the composition needs the J[D] half the Kohn-Sham "
                    "energy seam contracts separately, and this builder adds its Coulomb and "
                    "exchange halves together inside one BuildFock"));
    }

    // The lean wiring, TWO spellings of one within-family member
    // (the direct family's small-molecule lean Schwarz-only builder): the
    // auto default - an ABSENT [method].fock_builder at nBasis <= 1000 -
    // and the explicit fock_builder = "lean" (RunMethodInput::leanDirect),
    // which is the >1000 opt-in: explicit, it is admitted at ANY size,
    // above the ceiling included. An explicit "direct" reaches the
    // machinery at every size (Ruling B: explicit direct = the existing
    // family, byte-stable pins) and an absent key above the ceiling keeps
    // resolving to the machinery - only an explicit "lean" crosses the
    // ceiling into this branch. No modeled ladder and no workspace budget
    // here: the admission is the builder's own Create-time last-resort
    // check (memory_cap_gib travels as LeanFockBuildOptions.memoryCapGiB,
    // Ruling A). A point-group classification IS handed to the lean
    // builder here (LeanFockBuildOptions::symmetryReduction and
    // symmetryOrbitExpansion are set TOGETHER, from the same detector the
    // machinery arm below uses - the block further down states the decision
    // and its measurements in full). This line's earlier "the SAME reduction
    // the machinery arm builds reaches the lean arm as a CLASSIFICATION" and
    // its "both arms now use the fullest symmetry each can" were withdrawn
    // (2026-09-12), when the lean arm held no classification at all; the
    // first is TRUE again by construction (one detector, two arms) and the
    // second is true in the only sense that matters - each arm exploits the
    // group to the depth its own path allows, the machinery's class path
    // retaining the orbit MEMBERS while the lean path derives the per-pair
    // action instead. What a lean-vs-machinery comparison still carries is
    // that depth difference and the pair-path asymmetry, never a shared
    // reduction object: the lean arm is handed the same GROUP, not the same
    // table.
    //
    // The per-call stats stream rides the seam in the lean row family
    // (the stats-out: MakeLeanRhfFockBuilder + RecordLean - the
    // lean builder now carries the per-call screened-quartet counter, so
    // the trace rows carry nq64/nq32/xvol, never machinery-style false
    // zeros for the columns the lean path cannot back). The memory audit
    // block carries the builder's own Create-time envelope
    // (EstimatedPeakBytes) in the base term - the direct-family
    // precedent: no dense tensor, so the base IS the modeled peak. The
    // selector itself is the resolution's own member choice (leanMember),
    // the same value the pre-gate setup admission CheckPreGateSetupAdmission
    // reads - one answer, five consumers, no second derivation.
    if (kind == qcx::io::BuilderKind::kDirect && leanMember)
    {
        qcx::integrals::LeanFockBuildOptions leanOptions;
        leanOptions.accuracy = input.method.accuracy;
        leanOptions.memoryCapGiB = input.resources.memoryCapGiB;
        // A Kohn-Sham run takes this builder as its COULOMB half, so the
        // split flag is set: the half must return H + 2J(rho) and nothing
        // else, because the composition forms the Fock and the energy from
        // the halves itself (the Hartree-Fock pass takes the fused form, the
        // flag's default).
        leanOptions.buildCoulombOnly = isKs;

        // The point-group reduction IS handed to the lean builder here, with
        // BOTH of its fields set together: symmetryReduction (the
        // classification) and symmetryOrbitExpansion (the orbit half), which
        // Create refuses apart - the reduction defines the orbits the walk
        // enumerates (kInvalidArgument, lean_fock_build.cpp).
        //
        // This deliberately replaces the previous wiring, which left the
        // reduction null on a measurement that was real but measured the wrong
        // mechanism: the engaged and the disengaged builds BOTH report
        // nq64 = 8811481 on c8h18/def2-SVP, from which it concluded the
        // classification "drops zero cells and buys nothing at this size". What
        // that number measures the reach of is the MASK alone, and the mask's
        // reach IS zero at this size. That part stands, and the reason is
        // structural: the mask drops a cell only when one group element fixes
        // all four of its shells function-for-function, which on a Cartesian
        // s/p/d basis the eight Abelian groups achieve only for a shell centred
        // ON the element - every other element permutes atoms, relating
        // symmetry-distinct NON-zero integrals instead of forcing zeros - and
        // this fixture's C2h centre is a C-C bond midpoint with no shell on it.
        //
        // The ORBIT half does not need a shell on the centre, because it drops
        // no cell: it REPLACES a member's own evaluation with its orbit
        // representative's, which the ERI's invariance under the group action
        // legitimises. Measured on the same fixture on 2026-09-13: 8,811,481
        // screened cells collapse to 2,379,972 ERI blocks - 0.2701 by count, 0.2873
        // mass-weighted - and the lean per-pair orbit map reproduces the
        // machinery's quartet orbit count to the unit (2,379,972 against
        // 2,379,972). Budget ~3.5x on the ERI span, NOT the 3.70x count ratio.
        //
        // What it costs, stated because it is unwelcome: the expansion adds a
        // per-member copy the plain walk has no equivalent of, so the
        // CONTRACTION span RISES (~1.29x on that lane). The net is
        // (E + C) / (0.2873 E + 1.29 C) in the ERI:contract split, and that
        // split is reported per call in FockBuildStats, so a run reads its own
        // net off rather than assuming one. The Fock matrix also moves in its
        // last bits against the plain walk, because the engine's summation
        // order for a member's axis order is not its representative's: that is
        // the documented price of the mechanism, exact in exact arithmetic, and
        // the reduction's own machinery pins its class path the same way at
        // 1e-12 rather than byte-for-byte.
        //
        // The group comes from the SAME detector the machinery arm below uses,
        // so the arms agree on the group and differ only in how deep they
        // exploit it: the machinery's class path retains the orbit MEMBERS
        // (~0.71 GB on this fixture) while the lean path derives the per-pair
        // action instead (~90 KB, charged in the Create-time envelope by
        // EstimatePeakBytes), which is the asymmetry a lean-vs-machinery
        // comparison carries and must not paper over. A C1 molecule, a
        // non-realizable group and a group wider than the classification's
        // eight elements each come back with no usable reduction, and the lean
        // path then runs exactly as it did before - the plain screened walk.

        auto leanReduction = qcx::scf::BuildSymmetryReduction(molecule, basis);

        if (!leanReduction.has_value() &&
            leanReduction.error().code != qcx::ErrorCode::kUnimplemented)
        {
            return std::unexpected(leanReduction.error());
        }

        if (leanReduction.has_value() && leanReduction->groupOrder > 1)
        {
            leanOptions.symmetryReduction = &*leanReduction;
            leanOptions.symmetryOrbitExpansion = true;
        }

        auto lean =
            qcx::integrals::LeanDirectFockBuilder::Create(molecule, basis, coreTensor, leanOptions);

        if (!lean.has_value())
        {
            return std::unexpected(lean.error());
        }

        if (isKs)
        {
            auto ksContext = ResolveKsContext(input, molecule, basis);

            if (!ksContext.has_value())
            {
                return std::unexpected(ksContext.error());
            }

            // The exchange half exists only when the functional carries exact
            // exchange: a pure LDA/GGA run has c_HF = 0, and its K-only
            // builder must not be created at all - a builder that never runs
            // would still be admitted against the lean ceiling (and still
            // hold its own memory) under a functional that has no use for it.
            qcx::driver::internal::HalfFockFn coulombHalf = MakeLeanKsHalf(*lean, statsSink, 0);
            qcx::driver::internal::HalfFockFn exchangeHalf;

            if (ksContext->functional.exchangeFraction > 0.0)
            {
                qcx::integrals::LeanFockBuildOptions leanExchangeOptions;
                leanExchangeOptions.accuracy = input.method.accuracy;
                leanExchangeOptions.memoryCapGiB = input.resources.memoryCapGiB;
                leanExchangeOptions.buildExchangeOnly = true;

                // The K-only half rides the SAME classification and the same
                // orbit engagement as the Coulomb half above (one reduction,
                // both halves): the two are separate builders over one group,
                // and a half engaged without the other would put the run's two
                // Fock contributions on different summation orders for no
                // reason. In scope from the block above.
                if (leanReduction.has_value() && leanReduction->groupOrder > 1)
                {
                    leanExchangeOptions.symmetryReduction = &*leanReduction;
                    leanExchangeOptions.symmetryOrbitExpansion = true;
                }

                auto leanExchange = qcx::integrals::LeanDirectFockBuilder::Create(
                    molecule, basis, coreTensor, leanExchangeOptions);

                if (!leanExchange.has_value())
                {
                    return std::unexpected(leanExchange.error());
                }

                exchangeHalf = MakeLeanKsHalf(*leanExchange, statsSink, rhfOccupied);
            }

            auto seam = qcx::driver::internal::MakeRksSeam(coulombHalf,
                                                           exchangeHalf,
                                                           ToMatrix(coreTensor),
                                                           ksContext->functional.exchangeFraction,
                                                           ksContext->evaluator);

            if (!seam.has_value())
            {
                return std::unexpected(seam.error());
            }

            WiredFockBuilder ksWired{std::move(seam->fock)};
            ksWired.coulomb = std::move(seam->coulomb);
            ksWired.contribution = std::move(seam->contribution);
            return ksWired;
        }

        WiredFockBuilder wired{MakeLeanRhfFockBuilder(*lean, statsSink, rhfOccupied)};
        return wired;
    }

    if (kind == qcx::io::BuilderKind::kQfmm)
    {
        qcx::integrals::QfmmOptions qfmmOptions;
        qfmmOptions.accuracy = input.method.accuracy;

        // The QFMM schema knobs reach QfmmOptions here: an absent key
        // keeps the engine default (the preset
        // theta/order, leaf size 8, the crossover constant); the io
        // parser already rejected anything outside the engine contract, by
        // the schema key's name.
        if (input.method.theta.has_value())
        {
            qfmmOptions.theta = *input.method.theta;
        }

        if (input.method.lMult.has_value())
        {
            qfmmOptions.lMult = *input.method.lMult;
        }

        if (input.method.maxLeafSize.has_value())
        {
            qfmmOptions.maxLeafSize = *input.method.maxLeafSize;
        }

        if (input.method.crossoverBasisFunctionCount.has_value())
        {
            qfmmOptions.crossoverBasisFunctionCount = *input.method.crossoverBasisFunctionCount;
        }

        // The adaptive seam (the budget path), wired like the direct branch below:
        // the cap is the workspace budget shared with the composed builder -
        // each of its nested Creates decides its own rung against the shared
        // counter and reserves what it needs (the QFMM half's estimate
        // first, the exchange half's second - nesting order = reservation
        // order), and the cap acts as a ceiling, never a target.
        // memory_cap_gib = 0 keeps the legacy null-budget path byte-for-byte.
        std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;

        if (input.resources.memoryCapGiB > 0.0)
        {
            auto budget = qcx::memory::WorkspaceBudget::Create(
                WorkspaceGrantBytes(input.resources.memoryCapGiB, workspaceReserveBytes));

            if (!budget.has_value())
            {
                return std::unexpected(budget.error());
            }

            workspaceBudget = std::make_unique<qcx::memory::WorkspaceBudget>(std::move(*budget));
        }

        qfmmOptions.workspaceBudget = workspaceBudget.get();

        // The composed RHF builder (a RIJCOSX-style combination point):
        // F = H + 2J_QFMM(rho) - K(rho) - the
        // Coulomb half through the QFMM near/far split, the exchange half through the exchange-only
        // direct builder. Before the composed builder this branch wired the exchange-less
        // QfmmJBuilder (a J-only SCF).
        auto builder =
            qcx::integrals::QfmmHfFockBuilder::Create(molecule, basis, coreTensor, qfmmOptions);

        if (!builder.has_value())
        {
            // On the budget path the engine's Create-time refusal is the
            // admission decision (its text is AM-owned, never edited): the
            // driver composes the standing reinstatement clause onto it so
            // the diagnostic carries the full ladder (the direct branch's
            // shape).
            if (workspaceBudget != nullptr)
            {
                return std::unexpected(
                    Err(builder.error().code,
                        builder.error().message + qcx::driver::DirectReinstatementClause()));
            }

            return std::unexpected(builder.error());
        }

        // The Kohn-Sham arm of this family the Kohn-Sham composition. It replaces the
        // fused Hartree-Fock seam with the composition, over TWO halves cut
        // from the ONE builder created above - so the J the energy seam
        // contracts is this builder's own 2 J_QFMM at the loop's density,
        // not a second contraction that happens to agree.
        //
        // The halves need no core-Hamiltonian bookkeeping at all, and that is
        // the family's own fact rather than a convenience (QfmmKsHalf's
        // enum): the composed builder is a QFMM J half and an exchange-only
        // direct half, and BOTH already return the composition's own
        // conventions - H + 2J_QFMM(rho) and H - K(rho). The RI-J link's arm
        // beside this one adds H back into its Coulomb half; here that would
        // be a defect of exactly Tr[D H], which is why the case is stated
        // rather than implied by the sibling's shape.
        //
        // A hybrid functional's K half is created inside Create either way
        // (the composed builder owns both nested states), but it is never
        // CALLED unless the functional asks for exact exchange: a pure
        // LDA/GGA run passes no exchange half at all, and MakeRksSeam refuses
        // a non-zero fraction without one - so the composition and the
        // wiring cannot disagree about whether K was requested.
        if (isKs)
        {
            auto ksContext = ResolveKsContext(input, molecule, basis);

            if (!ksContext.has_value())
            {
                return std::unexpected(ksContext.error());
            }

            qcx::driver::internal::HalfFockFn coulombHalf =
                MakeQfmmKsHalf(*builder, QfmmKsHalf::kCoulomb);
            qcx::driver::internal::HalfFockFn exchangeHalf;

            if (ksContext->functional.exchangeFraction > 0.0)
            {
                exchangeHalf = MakeQfmmKsHalf(*builder, QfmmKsHalf::kExchange);
            }

            auto seam = qcx::driver::internal::MakeRksSeam(coulombHalf,
                                                           exchangeHalf,
                                                           ToMatrix(coreTensor),
                                                           ksContext->functional.exchangeFraction,
                                                           ksContext->evaluator);

            if (!seam.has_value())
            {
                return std::unexpected(seam.error());
            }

            WiredFockBuilder ksWired{std::move(seam->fock)};
            ksWired.coulomb = std::move(seam->coulomb);
            ksWired.contribution = std::move(seam->contribution);
            // Every carrier the Hartree-Fock arm below sets, set here as well
            // (the family and the builder are the same, so the record must
            // state the same budget, the same engine decision and the same
            // model the Coulomb half ran), plus the exchange half's own
            // Create-time record - this run's Fock is built from both halves,
            // so the exchange half's decision has the same claim on the
            // record as the Coulomb half's (the machinery arm's own two
            // carriers, one branch down).
            ksWired.workspaceBudget = std::move(workspaceBudget);
            ksWired.modeInfo = builder->ModeInfo();
            ksWired.exchangeModeInfo = builder->ExchangeModeInfo();
            ksWired.qfmmModel = builder->ModelRecord();
            return ksWired;
        }

        // The record surface is the single-mode_record RHF shape: the
        // composed Coulomb (QFMM) half's Create-time decision; the
        // exchange half rides the same budget (its commit is inside
        // workspace_budget's committed_bytes), and on UHF runs the
        // RunQfmmUhfScf surface exposes both halves separately.
        //
        // The per-call stats sink rides the QFMM seam (the stats-out
        // for a family that exposes no FockBuildStats): with a trace file
        // requested the seam records one QFMM row per composed BuildFock
        // call - call / wall / total_wall_ms / far_pairs - so the family
        // can produce a timing limb. Before this the branch took the
        // generic no-stats seam, whose null sink left the trace file
        // holding ZERO rows while the run still exited 0 - indistinguishable
        // from a wiring defect at the harness (the qfmm cell error of
        // window-20260911-063141Z).
        WiredFockBuilder wired{MakeQfmmRhfFockBuilder(*builder, statsSink)};
        wired.workspaceBudget = std::move(workspaceBudget);
        wired.modeInfo = builder->ModeInfo();
        // The composed Coulomb half's own model record: the
        // engine's resolution of the geometry model, the separation test and
        // the theta the octree was actually built with. Read from the
        // builder and never recomputed here - the kTight-preset gate and the
        // preset's theta both resolve inside QfmmJBuilder::Create.
        wired.qfmmModel = builder->ModelRecord();
        return wired;
    }
#if defined(QcxHasCuda)
    else if (kind == qcx::io::BuilderKind::kGpu)
    {
        auto builder =
            qcx::integrals::GpuJkFockBuilder::Create(molecule, basis, coreTensor, directOptions);

        if (!builder.has_value())
        {
            return std::unexpected(builder.error());
        }

        return WiredFockBuilder{MakeRhfFockBuilder(*builder)};
    }
#else
    // The GPU builder's implementation lives in the CUDA TUs, which the
    // default build never compiles; kGpu fails loudly instead of falling
    // through to the direct builder (the silent-substitution failure
    // mode the schema rejects by design).
    else if (kind == qcx::io::BuilderKind::kGpu)
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                   "fock_builder = \"gpu\" requires the CUDA build "
                                   "(windows-msvc-cuda preset)"));
    }
#endif
    else if (kind == qcx::io::BuilderKind::kRiJk)
    {
        // The auxiliary basis IN EFFECT, by the same rule the ri_j_link arm
        // applies one branch down: the explicit [basis].aux when given, else
        // the auto-selection for THIS kind. The kind matters - SelectAuxBasis
        // maps def2-* to the JK fit for kRiJk and to the J-fit for kDefault -
        // so resolving this branch with kDefault would wire a J-only fit into
        // a full-RI exchange path, which is the defect the standing quality ruling exists to
        // refuse.
        std::optional<std::string> auxName;

        if (input.basis.aux.has_value())
        {
            auxName = input.basis.aux;
        } else
        {
            auto selected = qcx::integrals::SelectAuxBasis(input.basis.orbital,
                                                           qcx::integrals::FockBuilderKind::kRiJk);

            if (!selected.has_value())
            {
                return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                           "no aux basis for \"" + input.basis.orbital +
                                               "\"; specify [basis].aux explicitly"));
            }

            auxName = *selected;
        }

        // The standing quality ruling, at the site that RESOLVES the name - the builder cannot
        // apply it, because a parsed BasisSet carries no name. This is the
        // refusal the validate pass and the run have to agree about, and it is
        // spelled from the predicate's own message
        // (integrals JkOptimizedAuxRefusal) so the two cannot drift apart.
        // It fires for the two doors into the same defect: the auto-selection
        // above (every cc-*/aug-cc-* base maps to a -rifit, which is a J-only
        // fit) and an explicit [basis].aux naming one.
        if (!qcx::integrals::IsJkOptimizedAux(*auxName))
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       qcx::integrals::JkOptimizedAuxRefusal(*auxName)));
        }

        // The same filtered parse the ri_j_link arm uses: only the
        // molecule's elements, so the parsed aux set and the engine's
        // molecule-scoped aux count are equal by construction.
        auto auxBasis = qcx::basisset::ParseNwchemDirectoryFiltered((root / *auxName).string(),
                                                                    UniqueAtomicNumbers(molecule));

        if (!auxBasis.has_value())
        {
            return std::unexpected(auxBasis.error());
        }

        // The budget path of the RI-K family (the ri_j_link arm's shape): the
        // driver grants the cap as
        // a workspace budget and the ENGINE's Create-time estimate decides the
        // rung - fast when the raw tensor and its transform fit, blocked
        // (RiFullFockRung::kBlocked, the ladder's batched rung, no extra
        // flops) otherwise. memory_cap_gib = 0 keeps the legacy null-budget
        // path byte-for-byte.
        std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;

        if (input.resources.memoryCapGiB > 0.0)
        {
            auto budget = qcx::memory::WorkspaceBudget::Create(
                WorkspaceGrantBytes(input.resources.memoryCapGiB, workspaceReserveBytes));

            if (!budget.has_value())
            {
                return std::unexpected(budget.error());
            }

            workspaceBudget = std::make_unique<qcx::memory::WorkspaceBudget>(std::move(*budget));
        }

        qcx::integrals::RiEngineOptions riOptions;
        riOptions.accuracy = input.method.accuracy;
        riOptions.workspaceBudget = workspaceBudget.get();

        // The point-group reduction CARRIER (the mechanism key):
        // method.ri_orbit_expansion asked
        // for a reduction that changes what the run evaluates, so the request
        // is carried to the builder rather than dropped here. RiFullFockBuilder
        // refuses the pair BY NAME - its 3-center tensor is counted but never
        // reduced - and that refusal, which names the two fields and the
        // option set, is the honest answer: the key names a mechanism, so a
        // family that cannot honour it refuses rather than running the plain
        // walk under the request. Carrying it is also what keeps the refusal
        // reachable: before this the driver supplied none, so the builder's
        // own refusal could never fire from a run.
        bool wiredOrbitExpansion = false;
        std::optional<qcx::integrals::SymmetryReduction> orbitOrbitalReduction;
        std::optional<qcx::integrals::SymmetryReduction> orbitAuxReduction;

        if (input.method.riOrbitExpansion.value_or(false))
        {
            auto orbitalReduction = qcx::scf::BuildSymmetryReduction(molecule, basis);

            if (!orbitalReduction.has_value())
            {
                return std::unexpected(orbitalReduction.error());
            }

            auto auxReduction = qcx::scf::BuildSymmetryReduction(molecule, *auxBasis);

            if (!auxReduction.has_value())
            {
                return std::unexpected(auxReduction.error());
            }

            orbitOrbitalReduction = std::move(*orbitalReduction);
            orbitAuxReduction = std::move(*auxReduction);
            riOptions.symmetryReduction = &*orbitOrbitalReduction;
            riOptions.auxSymmetryReduction = &*orbitAuxReduction;
            riOptions.symmetryOrbitExpansion = true;
            wiredOrbitExpansion = true;
        }

        auto builder = qcx::integrals::RiFullFockBuilder::Create(
            molecule, basis, *auxBasis, coreTensor, riOptions);

        if (!builder.has_value())
        {
            // The builder's own refusal rides out verbatim (AM-owned, never
            // edited). It needs no reinstatement composition the way the
            // ri_j_link arm's does: the ladder's refusal already names the
            // order, the blocked rung's minimum, the recompute rung's absence
            // here and the disk store's family, and the reduction's
            // refusal names the field it cannot honour.
            return std::unexpected(builder.error());
        }

        // The Kohn-Sham arm of this family. It replaces the fused
        // Hartree-Fock seam (H + 2 J_RI - K_RI, one call) with the
        // composition, over the SAME builder created above - so the J[D] the
        // energy seam contracts is this builder's own, at the density the
        // Fock was built from, and not a second contraction that happens to
        // agree.
        //
        // The conversion the pair's bare accounting needs is MakeRiFullKsHalf's
        // and stated there: H into BOTH halves, the factor of two onto the
        // Coulomb one. What is stated HERE is the density convention, because
        // it is this leg's rather than the adapter's: the composition hands
        // the halves the SPATIAL density rho = D/2 of the loop's iterate -
        // which is exactly the density convention RiFullFockBuilder's own
        // contract names, so no halving or doubling of the ARGUMENT happens
        // on this path (the argument is already the builder's own).
        //
        // A hybrid functional's K half is created only when the functional
        // asks for exact exchange (c_HF > 0): a pure LDA/GGA run hands the
        // composition no exchange half at all, so no exchange term reaches the
        // Fock (the pair entry point still computes one INSIDE the Coulomb
        // half's call and this seam discards it - the one-call note in
        // MakeRiFullKsHalf; it is wasted arithmetic on this path, never a
        // contribution to the Fock, which is the direction that would be
        // invisible). The guard is the composition's own too - MakeRksSeam
        // refuses a non-zero fraction without a callable exchange half - so the
        // two cannot disagree about whether K was requested.
        if (isKs)
        {
            auto ksContext = ResolveKsContext(input, molecule, basis);

            if (!ksContext.has_value())
            {
                return std::unexpected(ksContext.error());
            }

            // ONE H object, handed to the halves AND to the composition: the
            // composition recovers the seam's J by subtracting it back out of
            // the Coulomb half, so an H that is a different matrix would leave
            // a residue in the number the energy formula contracts.
            const Eigen::MatrixXd coreHamiltonian = ToMatrix(coreTensor);
            // ONE cache for both halves, and it is what makes the two calls
            // the composition makes at ONE density cost the builder one pair
            // build: on this arm both are asked at the SAME rho = D/2, so the
            // exchange half is a memo hit on the Coulomb half's own build
            // (RiFullKsPairCache, whose note carries the measured row counts).
            const auto pairCache = std::make_shared<RiFullKsPairCache>();
            qcx::driver::internal::HalfFockFn coulombHalf = MakeRiFullKsHalf(*builder,
                                                                             coreHamiltonian,
                                                                             RiFullKsHalf::kCoulomb,
                                                                             rhfOccupied,
                                                                             pairCache,
                                                                             statsSink);
            qcx::driver::internal::HalfFockFn exchangeHalf;

            if (ksContext->functional.exchangeFraction > 0.0)
            {
                exchangeHalf = MakeRiFullKsHalf(*builder,
                                                coreHamiltonian,
                                                RiFullKsHalf::kExchange,
                                                rhfOccupied,
                                                pairCache,
                                                statsSink);
            }

            auto seam = qcx::driver::internal::MakeRksSeam(coulombHalf,
                                                           exchangeHalf,
                                                           coreHamiltonian,
                                                           ksContext->functional.exchangeFraction,
                                                           ksContext->evaluator);

            if (!seam.has_value())
            {
                return std::unexpected(seam.error());
            }

            WiredFockBuilder ksWired{std::move(seam->fock)};
            ksWired.coulomb = std::move(seam->coulomb);
            ksWired.contribution = std::move(seam->contribution);
            // Every carrier the Hartree-Fock arm below sets, set here as well:
            // a Kohn-Sham ri_jk run is the same family on the same builder, so
            // its record must state the same budget, the same engine rung
            // decision and the same orbit-expansion request. A carrier left
            // unset here would be a record that silently reports a run other
            // than the one that happened .
            ksWired.workspaceBudget = std::move(workspaceBudget);
            ksWired.riJkModeInfo = builder->ModeInfo();
            // The request was CARRIED and the reading is the Hartree-Fock
            // arm's: RiFullFockBuilder has no OrbitExpansionEngaged
            // counterpart, so there is no outcome to read and the engaged flag
            // stays false rather than claiming a mechanism that never ran.
            ksWired.orbitExpansionWired = wiredOrbitExpansion;
            return ksWired;
        }

        // The occupied count the seam's density-factorization needs: this is
        // the RHF path, so the closed-shell count is exact.
        const std::size_t nOccupied = static_cast<std::size_t>(molecule.ElectronCount() / 2);
        WiredFockBuilder wired{MakeRiFullRhfFockBuilder(*builder, nOccupied, statsSink)};
        wired.workspaceBudget = std::move(workspaceBudget);
        // The engine's own Create-time rung decision (RiFullFockModeInfo): the
        // record's ri_jk_mode block reports it, so a run that rode the blocked
        // rung cannot serialize as one that rode the fast rung.
        wired.riJkModeInfo = builder->ModeInfo();
        // The request above was CARRIED, and the reading says so: what this
        // arm cannot do is report an outcome the builder never disclosed -
        // RiFullFockBuilder has no OrbitExpansionEngaged counterpart to
        // RiJkFockBuilder's, so there is nothing to read. The carrier's own
        // contract is therefore recorded rather than assumed: a later
        // increment that honours the reduction here must expose that
        // disclosure BEFORE this flag can lead the record to an outcome word.
        wired.orbitExpansionWired = wiredOrbitExpansion;
        return wired;
    } else if (kind == qcx::io::BuilderKind::kRiJLink)
    {
        std::optional<std::string> auxName;

        if (input.basis.aux.has_value())
        {
            auxName = input.basis.aux;
        } else
        {
            auto selected = qcx::integrals::SelectAuxBasis(
                input.basis.orbital, qcx::integrals::FockBuilderKind::kDefault);

            if (!selected.has_value())
            {
                return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                           "no aux basis for \"" + input.basis.orbital +
                                               "\"; specify [basis].aux explicitly"));
            }

            auxName = *selected;
        }

        // The filtered parse: keep only the
        // molecule's elements, so the parsed aux set and the engine's
        // molecule-scoped aux count are equal by construction.
        auto auxBasis = qcx::basisset::ParseNwchemDirectoryFiltered((root / *auxName).string(),
                                                                    UniqueAtomicNumbers(molecule));

        if (!auxBasis.has_value())
        {
            return std::unexpected(auxBasis.error());
        }

        const std::size_t nBasis = CountBasisFunctions(molecule, basis);
        const std::size_t nAux = CountBasisFunctions(molecule, *auxBasis);

        // The ri_j admission is delegated to the engine's Create-time
        // estimate (the budget path): the driver grants the cap as the workspace
        // budget and the engine picks the rung the estimate fits - fast when
        // the tensor path fits, light otherwise (C60 and nt84 flip from
        // "refused" to "the engine decides"). memory_cap_gib = 0 (the escape
        // hatch) keeps the legacy null-budget path byte-for-byte.
        std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;

        if (input.resources.memoryCapGiB > 0.0)
        {
            auto budget = qcx::memory::WorkspaceBudget::Create(
                WorkspaceGrantBytes(input.resources.memoryCapGiB, workspaceReserveBytes));

            if (!budget.has_value())
            {
                return std::unexpected(budget.error());
            }

            workspaceBudget = std::make_unique<qcx::memory::WorkspaceBudget>(std::move(*budget));
        }

        // The disk-rung knob engages only after the
        // in-memory ladder's estimate-time refusal, and that ladder refuses
        // only when a budget exists: on the legacy null-budget path (the
        // memory_cap_gib = 0 escape hatch) the engine Create always
        // succeeds, so the knob would be a silent no-op - refuse loudly
        // instead, naming the knob. (The disk rung needs a budget to have a
        // refusal to fall back from, and its own working set needs RAM too -
        // the knob never bypasses the cap.) The one route this refusal does
        // NOT cover is the forced diagnostic mode, which has no ladder to
        // refuse from (see below): there the disk builder runs with no
        // budget granted.
        if (input.method.riTensorMode == qcx::io::RiTensorMode::kDisk && workspaceBudget == nullptr)
        {
            return std::unexpected(
                Err(qcx::ErrorCode::kInvalidArgument,
                    "method.ri_tensor_mode = \"disk\" needs the budget path: memory_cap_gib = 0 "
                    "keeps the legacy null-budget path, whose engine ladder never refuses - the "
                    "disk rung (the ladder's last rung) can only engage on that refusal. Set "
                    "resources.memory_cap_gib above zero."));
        }

        // The Kohn-Sham lanes' own admission on this branch, stated HERE as
        // well as in the combination check for the same reason the guard at
        // the top of this function is: without it a Kohn-Sham run on the
        // disk rung would reach the two returns below, each of which wires a
        // FUSED Hartree-Fock seam (MakeRhfFockBuilder) and leaves the
        // composition's companion callbacks empty. That is the one state the
        // run site must never see, and it is silent - the run would compute
        // a Kohn-Sham energy from a Hartree-Fock Fock. Both spellings are
        // covered (the ladder's rung word and the diagnostic force), so the
        // refusal cannot be routed around by naming the other key.
        if (isKs && (input.method.riTensorMode == qcx::io::RiTensorMode::kDisk ||
                     input.diagnostics.forceDiskRi))
        {
            return std::unexpected(
                Err(qcx::ErrorCode::kUnimplemented, RiJLinkKsDiskRungRefusalText()));
        }

        // The disk route's one construction site (the disk-rung knob), shared
        // by the two admissions below so the wiring cannot
        // drift between them: the ladder's own fallback and the forced
        // diagnostic mode. `refusalWhy` carries the engine's estimate-time
        // refusal that sent the fallback here (composed onto a Create
        // failure so the reader still sees it); the forced call passes an
        // empty string - no ladder ran, so there is no refusal to report
        // and the text names the FORCED admission instead. `forced` travels
        // into the run record (mode_record.forced_disk), which
        // is what keeps a forced measurement cell distinguishable from a
        // run the ladder sent to disk.
        const auto engageDiskRung =
            [&](bool forced, const std::string& refusalWhy) -> qcx::Result<WiredFockBuilder> {
#if !defined(QcxHasStorage)
            // No storage module in this build (QCX_ENABLE_IO=OFF): the disk rung
            // IS the storage module's disk-backed builder, so there is no rung
            // to engage and nothing that could stand in for one - falling back
            // to an in-memory rung would compute the run under a disk label.
            // Refused by name, with the build's own fact as the cause; the
            // ladder's refusal text and the force flag are the arms below's
            // business and would name a mechanism this build does not have.
            (void)forced;
            (void)refusalWhy;
            return std::unexpected(
                Err(qcx::ErrorCode::kUnimplemented, StorageModuleAbsentRefusalText()));
#else
            qcx::storage::DiskRiFockOptions diskOptions;
            diskOptions.accuracy = input.method.accuracy;

            // The chunk-size hint is read HERE and nowhere else. Outside this
            // lambda the key is parsed, validated and never consumed - and
            // that is the ruled behaviour, not an omission: under the key
            // split (owner-ruled 2026-09-13) a pure SIZE HINT is DROPPED on a
            // family that has no use for it and the run proceeds, because a
            // family with no disk rung has no chunking to size and refusing
            // the key would be pedantic. The same drop covers the ri_j_link
            // ladder's own fitting-memory-rung case, where the disk rung never
            // engages and this line never runs.
            //
            // The DISCLOSURE half of that ruling LANDS with this line as its
            // subject: the drop is recorded in resources_resolved.ri_chunk_bytes
            // , filled by MakeRiChunkBytesRecord and named after the
            // key it reports on. The record's honour predicate is this lambda's
            // own trace - a kDisk mode info, which nothing else in the driver
            // produces - so the record cannot claim the hint was taken on a run
            // that never reached the assignment below.
            if (input.method.riChunkBytes.has_value())
            {
                diskOptions.chunkBytes = *input.method.riChunkBytes;
            }

            // The scratch store: a fresh temp file (MakeRiDiskStorePath
            // removes anything stale first - the append-only contract
            // makes a store path single-builder scratch).
            auto storePath = MakeRiDiskStorePath();

            if (!storePath.has_value())
            {
                return std::unexpected(storePath.error());
            }

            auto moleculeCopy =
                MakeChargedMolecule(molecule, molecule.Charge(), molecule.Multiplicity());

            if (!moleculeCopy.has_value())
            {
                return std::unexpected(moleculeCopy.error());
            }

            auto diskBuilder = qcx::storage::DiskRiFockBuilder::Create(*storePath,
                                                                       std::move(*moleculeCopy),
                                                                       basis,
                                                                       *auxBasis,
                                                                       input.basis.orbital,
                                                                       *auxName,
                                                                       coreTensor,
                                                                       diskOptions);

            if (!diskBuilder.has_value())
            {
                // A failed Create may have left a partial store file behind
                // (no handle keeps it open - the error preceded any
                // builder). The engine's refusal text rides along so the
                // reader sees why the rung was engaged.
                std::error_code ignored;
                std::filesystem::remove(*storePath, ignored);

                std::string admission =
                    "forced by [diagnostics] force_disk_ri (the in-memory ladder was "
                    "bypassed: the diagnostic mode, not the production path)";

                if (!refusalWhy.empty())
                {
                    admission = "engaged as the disk rung after the in-memory ladder's refusal: " +
                                refusalWhy;
                }

                return std::unexpected(Err(diskBuilder.error().code,
                                           diskBuilder.error().message + " (" + admission + ")"));
            }

            // The disk rung's mode record is synthesized driver-side - no
            // engine record exists (the ladder refused before deciding, and
            // the forced mode never ran it): mode kDisk plus the modeled
            // dense on-disk payload (chunk stores are
            // dense, so 8 n^2 nAux is also the file's payload - the payload
            // reconciliation target) and the budget context the refusal
            // left untouched. All other terms stay zero, and with no budget
            // (reachable on the forced route alone) the budget terms stay
            // zero too - never a fabricated grant.
            qcx::integrals::FockModeInfo diskInfo;
            diskInfo.mode = qcx::integrals::FockBuildMode::kDisk;
            diskInfo.diskBytes = 8 * nBasis * nBasis * nAux;

            if (workspaceBudget != nullptr)
            {
                diskInfo.budgetBytes = workspaceBudget->CapacityBytes();
                diskInfo.remainingAtDecision = workspaceBudget->Remaining();
            }

            WiredFockBuilder wired{
                MakeRhfFockBuilder(*diskBuilder, statsSink, rhfOccupied, boundAccum)};
            wired.workspaceBudget = std::move(workspaceBudget);
            wired.modeInfo = diskInfo;
            wired.forcedDiskRung = forced;
            // The removal duty travels with the wired builder: its
            // destructor clears the seam (releasing the captured
            // DiskRiFockBuilder State - the store's one open handle) and
            // then removes the scratch file.
            wired.diskStorePath = *storePath;
            return wired;
#endif
        };

        // The forced diagnostic mode ([diagnostics] force_disk_ri, the
        // separate key that replaced method.ri_tensor_mode = "forced_disk" -
        // a force is not a rung selection, and a rung word re-used to mean
        // "force" is one key naming two mechanisms): the measurement override
        // that bypasses the composed in-memory ladder entirely - no
        // RiJkFockBuilder::Create above, so no estimate-time refusal to fall
        // back from and no dependence on a cap tight enough to produce one.
        // It is NOT a rung selector and NOT the production path: "disk"
        // keeps the ladder's LAST rung (the disk-algorithms principle:
        // direct screened first, batched/blocked, recompute, disk LAST). A
        // benchmark cell uses it to measure the disk-backed builder itself
        // at sizes where every memory rung fits - exactly the situation in
        // which the ladder never reaches disk and a "disk" cell would
        // silently measure the in-memory builder under a disk label. Unlike
        // kDisk it needs no budget path: the refusal above exists because
        // the ladder refuses only under a budget, and this route has no
        // ladder - with memory_cap_gib = 0 the disk builder simply runs with
        // no budget granted (the record's budget terms stay zero, the
        // legacy-path absence). The record states the override, so the
        // cell's label is honest.
        //
        // This branch is the ONLY read of the force on the route decision,
        // and it sits inside the ri_j_link family's wiring: on every other
        // family the key reaches no builder here, and
        // MakeRiTensorModeRecord discloses that as a demotion instead of the
        // request vanishing (the demotion, not the silent
        // substitution).
        if (input.diagnostics.forceDiskRi)
        {
            return engageDiskRung(true, std::string{});
        }

        qcx::integrals::RiEngineOptions riOptions;
        riOptions.accuracy = input.method.accuracy;
        riOptions.workspaceBudget = workspaceBudget.get();
        bool wiredOrbitExpansion = false;

        // The orbit expansion, on the explicit key ONLY .
        // The engine's own permission defaults ON, so the driver's not
        // supplying a reduction is what keeps every run on the plain walk:
        // this branch is the only producer of the two reductions, and it
        // runs only when method.ri_orbit_expansion asked for them. Nothing is
        // inferred from the group or the size - an expansion that changes
        // what a run evaluates is asked for, never assumed - and the outcome
        // is read back from the ENGINE (OrbitExpansionEngaged) rather than
        // recomputed here, so the record and the engine cannot disagree.
        std::optional<qcx::integrals::SymmetryReduction> orbitOrbitalReduction;
        std::optional<qcx::integrals::SymmetryReduction> orbitAuxReduction;

        if (input.method.riOrbitExpansion.value_or(false))
        {
            auto orbitalReduction = qcx::scf::BuildSymmetryReduction(molecule, basis);

            if (!orbitalReduction.has_value())
            {
                return std::unexpected(orbitalReduction.error());
            }

            auto auxReduction = qcx::scf::BuildSymmetryReduction(molecule, *auxBasis);

            if (!auxReduction.has_value())
            {
                return std::unexpected(auxReduction.error());
            }

            orbitOrbitalReduction = std::move(*orbitalReduction);
            orbitAuxReduction = std::move(*auxReduction);
            riOptions.symmetryReduction = &*orbitOrbitalReduction;
            riOptions.auxSymmetryReduction = &*orbitAuxReduction;
            // Set explicitly rather than left to the default: the request is
            // what this key is, and a reader of this branch should not have to
            // know the engine's default to see that.
            riOptions.symmetryOrbitExpansion = true;
            wiredOrbitExpansion = true;
        }

        auto builder = qcx::integrals::RiJkFockBuilder::Create(
            molecule, basis, *auxBasis, coreTensor, riOptions);

        if (!builder.has_value())
        {
            // The disk rung's driver route (the disk-rung knob): the
            // explicit opt-in engages the storage-module
            // disk-backed builder as the ladder's LAST rung - only here,
            // on the in-memory ladder's estimate-time refusal under the
            // budget. The composed in-memory ladder always ran first (the
            // Create above), so the knob never makes disk a first choice;
            // when a memory rung fits, the run rides it and the knob is
            // inert by design (automatic selection stays deferred until
            // the measured crossover lands).
            if (input.method.riTensorMode == qcx::io::RiTensorMode::kDisk)
            {
                return engageDiskRung(false, builder.error().message);
            }

            // On the budget path the engine's Create-time refusal is the
            // admission decision (its text is AM-owned, never edited): the
            // driver composes the standing reinstatement clause onto it so
            // the diagnostic carries the full ladder.
            if (workspaceBudget != nullptr)
            {
                return std::unexpected(
                    Err(builder.error().code, builder.error().message + RiJReinstatementClause()));
            }

            return std::unexpected(builder.error());
        }

        // The Kohn-Sham arm of this family . It replaces the fused
        // Hartree-Fock seam with the composition, over TWO halves cut from
        // the ONE builder created above - so the J the energy seam contracts
        // is this builder's own 2 J_RI at the loop's density, not a second
        // contraction that happens to agree.
        //
        // The Coulomb half is BuildCoulombOnly plus the family's missing H
        // (MakeRiJLinkKsHalf's enum states the accounting); the exchange half
        // is the builder's own H - K(rho). Both are the SAME builder instance
        // the fused BuildFock runs, so the K half of a Kohn-Sham ri_j_link run
        // is the K half of the Hartree-Fock one - one accuracy preset, one
        // batch cap, one certified-lane default, one shared budget.
        //
        // A hybrid functional's K half is created only when the functional
        // asks for exact exchange (c_HF > 0): a pure LDA/GGA run never issues
        // the exchange call at all, which is the tier the resolver promoted
        // this family for in the first place ("a pure functional needs J and
        // no K"). The guard is the composition's own too - MakeRksSeam
        // refuses a non-zero fraction without a callable exchange half - so
        // the two cannot disagree about whether K was requested.
        if (isKs)
        {
            auto ksContext = ResolveKsContext(input, molecule, basis);

            if (!ksContext.has_value())
            {
                return std::unexpected(ksContext.error());
            }

            // ONE H object, handed to the halves AND to the composition: the
            // composition recovers the seam's J by subtracting it back out of
            // the Coulomb half, so an H that is a different matrix would leave
            // a residue in the number the energy formula contracts.
            const Eigen::MatrixXd coreHamiltonian = ToMatrix(coreTensor);
            qcx::driver::internal::HalfFockFn coulombHalf =
                MakeRiJLinkKsHalf(*builder, coreHamiltonian, RiJLinkKsHalf::kCoulomb, statsSink, 0);
            qcx::driver::internal::HalfFockFn exchangeHalf;

            if (ksContext->functional.exchangeFraction > 0.0)
            {
                exchangeHalf = MakeRiJLinkKsHalf(
                    *builder, coreHamiltonian, RiJLinkKsHalf::kExchange, statsSink, rhfOccupied);
            }

            auto seam = qcx::driver::internal::MakeRksSeam(coulombHalf,
                                                           exchangeHalf,
                                                           coreHamiltonian,
                                                           ksContext->functional.exchangeFraction,
                                                           ksContext->evaluator);

            if (!seam.has_value())
            {
                return std::unexpected(seam.error());
            }

            WiredFockBuilder ksWired{std::move(seam->fock)};
            ksWired.coulomb = std::move(seam->coulomb);
            ksWired.contribution = std::move(seam->contribution);
            // Every carrier the Hartree-Fock arm below sets, set here as well:
            // a Kohn-Sham ri_j_link run is the same family on the same builder,
            // so its record must state the same budget, the same engine
            // decision, the same calibration counter source and the same orbit
            // expansion request/outcome pair. A carrier left unset here would
            // be a record that silently reports a run other than the one that
            // happened .
            ksWired.workspaceBudget = std::move(workspaceBudget);
            ksWired.modeInfo = builder->ModeInfo();
            ksWired.riJkBuilder = *builder;
            ksWired.orbitExpansionEngaged = builder->OrbitExpansionEngaged();
            ksWired.orbitExpansionWired = wiredOrbitExpansion;
            return ksWired;
        }

        // The per-call stats sink rides the seam on the direct family and
        // the RI-J link (both expose per-call quartet counts); the QFMM
        // family takes its own seam and row family (the kQfmm branch), and
        // the GPU builder has no per-call stats and keeps the plain seam.
        WiredFockBuilder wired{MakeRhfFockBuilder(*builder, statsSink, rhfOccupied, boundAccum)};
        wired.workspaceBudget = std::move(workspaceBudget);
        // The engine's Create-time mode record: set on the budget path,
        // nullopt on the legacy path (the serializer leaves it absent).
        wired.modeInfo = builder->ModeInfo();
        // The calibration counter carrier: the run flow reads the run's
        // accumulated term counters through this copy after the SCF loop
        // (it shares the seam lambda's builder State - same-run counters).
        wired.riJkBuilder = *builder;
        // The engine's own disclosure, read here and nowhere recomputed: the
        // record's ri_orbit_expansion block reports this and the request the
        // input named, and the two cannot drift apart because the outcome is
        // this call.
        wired.orbitExpansionEngaged = builder->OrbitExpansionEngaged();
        wired.orbitExpansionWired = wiredOrbitExpansion;
        return wired;
    }

    // The machinery tail's admission. Only the direct family may build here;
    // any other kind that arrived without its own branch is refused BY NAME
    // (the value included, for a cast no enumerator names) rather than built
    // by the wrong family under its own label. The list itself is the guarded
    // switch above, so a kind nobody placed stops the BUILD instead of
    // falling through to here.
    if (!MachineryFamily(kind))
    {
        return std::unexpected(Err(qcx::ErrorCode::kUnimplemented,
                                   "fock_builder = \"" + std::string(qcx::io::ToString(kind)) +
                                       "\" (value " + std::to_string(static_cast<int>(kind)) +
                                       ") has no run path in v1: this build's wiring names "
                                       "\"direct\", \"qfmm\", \"gpu\" and \"ri_j_link\", so a "
                                       "request named here would otherwise be built by the "
                                       "direct family's machinery under its own label"));
    }

    // The adaptive seam (the budget path), wired like the ri_j branch: the cap is the
    // workspace budget, and the engine's Create-time estimate decides the
    // rung and reserves only what it needs - the cap acts as a ceiling,
    // never a target. The ERI cache gets no cap-derived grant: maxCacheBytes
    // stays at its default 0 (cache off) under the user's O(N^2)-default
    // selection bound, with the engine's Create-time cache clamp as the
    // sure-fit gate for any explicit grant . memory_cap_gib = 0 keeps
    // the legacy null-budget path byte-for-byte.
    std::unique_ptr<qcx::memory::WorkspaceBudget> workspaceBudget;

    if (input.resources.memoryCapGiB > 0.0)
    {
        auto budget = qcx::memory::WorkspaceBudget::Create(
            WorkspaceGrantBytes(input.resources.memoryCapGiB, workspaceReserveBytes));

        if (!budget.has_value())
        {
            return std::unexpected(budget.error());
        }

        workspaceBudget = std::make_unique<qcx::memory::WorkspaceBudget>(std::move(*budget));
    }

    directOptions.workspaceBudget = workspaceBudget.get();

    // The class-aware seam (the symmetry-blocking): the reduction extracted once - the
    // same point-group detection the blocked diagonalization uses. A
    // kUnimplemented extraction (a point group no non-identity element
    // expresses as a signed coordinate permutation) or a trivial (C1)
    // group keeps the plain path; a realizable non-trivial group engages
    // the class path, which also keeps the ERI cache disengaged .
    auto reduction = qcx::scf::BuildSymmetryReduction(molecule, basis);

    if (!reduction.has_value() && reduction.error().code != qcx::ErrorCode::kUnimplemented)
    {
        return std::unexpected(reduction.error());
    }

    if (reduction.has_value() && reduction->groupOrder > 1)
    {
        directOptions.symmetryReduction = &*reduction;
    }

    // The Kohn-Sham machinery run: TWO split builders out of the same
    // options struct rather than the one fused builder below - the Coulomb
    // half (buildCoulombOnly) and, when the functional carries exact
    // exchange, the K-only half (buildExchangeOnly). Everything upstream is
    // shared with the Hartree-Fock arm by construction: the same
    // admission-gated workspace budget, the same class-aware reduction, the
    // same enforcement flag (which is why an RKS run on this member is the
    // one Kohn-Sham lane that can honour `enforce_certified_bound`), and the
    // same reinstatement clause on a Create-time refusal.
    if (isKs)
    {
        auto ksContext = ResolveKsContext(input, molecule, basis);

        if (!ksContext.has_value())
        {
            return std::unexpected(ksContext.error());
        }

        directOptions.buildCoulombOnly = true;
        auto coulombBuilder =
            qcx::integrals::DirectJkFockBuilder::Create(molecule, basis, coreTensor, directOptions);

        if (!coulombBuilder.has_value())
        {
            if (workspaceBudget != nullptr)
            {
                return std::unexpected(
                    Err(coulombBuilder.error().code,
                        coulombBuilder.error().message + qcx::driver::DirectReinstatementClause()));
            }

            return std::unexpected(coulombBuilder.error());
        }

        qcx::driver::internal::HalfFockFn coulombHalf =
            MakeDirectKsHalf(*coulombBuilder, statsSink, 0, boundAccum);
        qcx::driver::internal::HalfFockFn exchangeHalf;
        std::optional<qcx::integrals::FockModeInfo> exchangeModeInfo;

        if (ksContext->functional.exchangeFraction > 0.0)
        {
            qcx::integrals::FockBuildOptions exchangeOptions = directOptions;
            exchangeOptions.buildCoulombOnly = false;
            exchangeOptions.buildExchangeOnly = true;
            auto exchangeBuilder = qcx::integrals::DirectJkFockBuilder::Create(
                molecule, basis, coreTensor, exchangeOptions);

            if (!exchangeBuilder.has_value())
            {
                if (workspaceBudget != nullptr)
                {
                    return std::unexpected(Err(exchangeBuilder.error().code,
                                               exchangeBuilder.error().message +
                                                   qcx::driver::DirectReinstatementClause()));
                }

                return std::unexpected(exchangeBuilder.error());
            }

            exchangeHalf = MakeDirectKsHalf(*exchangeBuilder, statsSink, rhfOccupied, boundAccum);
            exchangeModeInfo = exchangeBuilder->ModeInfo();
        }

        auto seam = qcx::driver::internal::MakeRksSeam(coulombHalf,
                                                       exchangeHalf,
                                                       ToMatrix(coreTensor),
                                                       ksContext->functional.exchangeFraction,
                                                       ksContext->evaluator);

        if (!seam.has_value())
        {
            return std::unexpected(seam.error());
        }

        WiredFockBuilder ksWired{std::move(seam->fock)};
        ksWired.workspaceBudget = std::move(workspaceBudget);
        ksWired.modeInfo = coulombBuilder->ModeInfo();
        ksWired.exchangeModeInfo = exchangeModeInfo;
        ksWired.coulomb = std::move(seam->coulomb);
        ksWired.contribution = std::move(seam->contribution);
        return ksWired;
    }

    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basis, coreTensor, directOptions);

    if (!builder.has_value())
    {
        // On the budget path the engine's Create-time refusal is the
        // admission decision (its text is AM-owned, never edited): the
        // driver composes the standing reinstatement clause onto it so
        // the diagnostic carries the full ladder.
        if (workspaceBudget != nullptr)
        {
            return std::unexpected(
                Err(builder.error().code,
                    builder.error().message + qcx::driver::DirectReinstatementClause()));
        }

        return std::unexpected(builder.error());
    }

    // The direct path carries the per-call stats sink (see the ri_j_link
    // branch's comment for the QFMM/GPU routing). The engine's
    // Create-time mode record: set on the budget path, nullopt on the
    // legacy path (the serializer leaves it absent).
    WiredFockBuilder wired{MakeRhfFockBuilder(*builder, statsSink, rhfOccupied, boundAccum)};
    wired.workspaceBudget = std::move(workspaceBudget);
    wired.modeInfo = builder->ModeInfo();
    // The analyzer's leak verdict on this scope is a false positive: the
    // QFMM branch's builder state is the shared_ptr pimpl (QfmmHfFockBuilder
    // State, itself holding the nested QfmmJBuilder), and the std::function
    // returned to the caller destroys its captured target at type-erasure -
    // invisible to the analyzer.
    return wired;
} // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

// The shared UHF tail of RunDriver: FillScfResult through the Molden
// export - the per-spin outcome of a converged or last-iterate UhfRun.
// Both UHF branches (the direct runner and the composed-QFMM runner) end
// here; only the runner that produced the scf differs above. The mode and
// budget records are NOT part of this tail - each branch records its own
// runner's artifacts (the direct branch's two builder records, the
// composed branch's absent ones).
qcx::Result<void> FillUhfRunOutcome(
    qcx::io::RunResult& result,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    const Eigen::MatrixXd& overlap,
    const CpuTensor2& coreTensor,
    const Eigen::MatrixXd& core,
    const qcx::scf::UhfResult& scf,
    double scfLoopMs,
    const std::vector<std::size_t>& canonicalOrder,
    const std::optional<std::map<int, qcx::scf::AtomicUhfInputs>>& atomicInputs,
    const qcx::io::RunInput& input,
    std::size_t workspaceReserveBytes,
    const qcx::backend::GpuComputeProfile& deviceComputeProfile,
    std::optional<bool> certifiedLaneRequest,
    // The run's resolved member choice, forwarded to the properties block
    // (the Fukui charged species run through the direct-UHF runner).
    bool leanMember) {
    FillScfResult(result,
                  scf.converged,
                  scf.iterations,
                  scf.achievedEnergyDelta,
                  scf.achievedRmsDensityChange,
                  scf.totalEnergy,
                  scf.electronicEnergy,
                  scfLoopMs,
                  scf.numRemovedOverlapDirections);
    result.spinSquared = scf.spinSquared;

    // The full-group labeling stage: the per-MO labels of both
    // spin channels; absent when the stage did not run (a C1 molecule,
    // the [symmetry] full_group = false switch, or an unrealizable
    // group) - never a fabricated C1 record.
    result.symmetry = MakeRunSymmetry(scf.symmetryLabelsAlpha);
    result.symmetryBeta = MakeRunSymmetry(scf.symmetryLabelsBeta);

    // The symmetry-blocking disclosure: what this run did with
    // the blocking it was in a position to use. The guard grades the decision
    // per spin and per diagonalization, so both channels travel - one spin's
    // Fock may be symmetry-adapted while the other has broken the group - and
    // a run that never asked fills neither (the key stays out of the record).
    result.symmetryBlockingAlpha = MakeRunSymmetryBlocking(scf.symmetryBlockingAlpha);
    result.symmetryBlockingBeta = MakeRunSymmetryBlocking(scf.symmetryBlockingBeta);

    auto properties = FillProperties(result,
                                     molecule,
                                     basis,
                                     scf.densityAlpha,
                                     scf.densityBeta,
                                     overlap,
                                     coreTensor,
                                     core,
                                     canonicalOrder,
                                     atomicInputs,
                                     input,
                                     workspaceReserveBytes,
                                     deviceComputeProfile,
                                     certifiedLaneRequest,
                                     leanMember);

    if (!properties.has_value())
    {
        return std::unexpected(properties.error());
    }

    // The Molden export of the converged or
    // last-iterate result; the aufbau occupations of the two spin
    // channels, alpha first. The occupation vectors are materialized
    // as named locals before the blocks array: the blocks hold
    // references into them.
    const int electrons = molecule.ElectronCount();
    const int nAlpha = (electrons + molecule.Multiplicity() - 1) / 2;
    const int nBeta = electrons - nAlpha;
    const auto alphaOccupations =
        AufbauOccupations(static_cast<std::size_t>(nAlpha),
                          1.0,
                          static_cast<std::size_t>(scf.coefficientsAlpha.cols()));
    const auto betaOccupations =
        AufbauOccupations(static_cast<std::size_t>(nBeta),
                          1.0,
                          static_cast<std::size_t>(scf.coefficientsBeta.cols()));
    const std::array<qcx::io::MoldenMolecularOrbitals, 2> moBlocks{
        qcx::io::MoldenMolecularOrbitals{
            "Alpha", scf.coefficientsAlpha, scf.orbitalEnergiesAlpha, alphaOccupations},
        qcx::io::MoldenMolecularOrbitals{
            "Beta", scf.coefficientsBeta, scf.orbitalEnergiesBeta, betaOccupations}};

    auto molden = WriteMoldenExport(result, molecule, basis, moBlocks, input);

    if (!molden.has_value())
    {
        return std::unexpected(molden.error());
    }

    return {};
}

// The RAII owner of the run's attribution window ([memory_instrument],
// Attribution/attribution): the instrument THIS RunDriver call enabled is disabled on
// every remaining exit path — success and error returns alike. Closing
// the window matters twice over: the watchdog's final trace row only
// lands on disable, and a joinable watchdog thread in the memory module's
// static storage would std::terminate the process at teardown if the run
// exited without disabling. Ownership is adopted (not unconditional) so
// an embedding process's own instrument — enabled outside RunDriver —
// survives a plain run; only the window this call opened is closed.
class AttributionWindowGuard {
public:
    AttributionWindowGuard() = default;

    AttributionWindowGuard(const AttributionWindowGuard&) = delete;
    AttributionWindowGuard& operator=(const AttributionWindowGuard&) = delete;
    AttributionWindowGuard(AttributionWindowGuard&&) = delete;
    AttributionWindowGuard& operator=(AttributionWindowGuard&&) = delete;

    ~AttributionWindowGuard() {
        if (_ownsWindow)
        {
            // Adopt() runs only after a successful enable, so the close
            // cannot fail here; a destructor cannot propagate the Result
            // (the (void) answers the nodiscard diagnostic).
            // NOLINTNEXTLINE(bugprone-unused-return-value)
            static_cast<void>(qcx::memory::AllocationInstrumentDisable());
        }
    }

    /// Claims the window this run opened: call right after the enable
    /// below succeeds; the destructor then closes it on every exit path.
    void Adopt() noexcept {
        _ownsWindow = true;
    }

private:
    bool _ownsWindow = false;
};

} // namespace

qcx::Result<RunOutcome> RunDriverOutcome(const qcx::io::RunInput& input) {
    const auto started = std::chrono::steady_clock::now();

    // The hard memory cap goes on FIRST, before any computation - the OS
    // kills the process at the cap instead of the machine dying. Fail-
    // closed: when the cap is requested but cannot be applied, the run
    // refuses here - an uncapped run must never start. The record lands
    // in the resources_resolved JSON block.
    auto caps = ApplyProcessCaps(input.resources);

    if (!caps.has_value())
    {
        return std::unexpected(caps.error());
    }

    // The thread ceiling goes on before any SCF work - every CPU parallel
    // primitive reads DefaultOmpTeamSize, so clamping it here clamps the
    // whole run (the Fock-build chunk split included: a ceiling of 1
    // degrades the run to the single-chunk serial path, whose bit-identity
    // is pinned). The clamp applies after the team cache, so this works no
    // matter what warmed the cache first. thread_cap is recorded in
    // resources_resolved; a ceiling of 1 also becomes visible in the JSON
    // as the resolved thread_cap.
    qcx::backend::SetOmpThreadCeiling(input.resources.threadCap);

    // The attribution opt-in ([memory_instrument]): the instrumented
    // window opens at run entry — after the caps are on, before any
    // computation — so every allocation the run makes rides inside it.
    // The guard above closes the window on every exit path. Fail-closed:
    // an instrumented run whose enable fails refuses here — the run must
    // never proceed outside the window its input asked for (an already-
    // enabled instrument means the caller's window is still open, and
    // process-global stats cannot host two).
    //
    // The trace metadata carries what the driver knows at this point: the
    // input cap, in the same bytes ApplyProcessCaps enforced (one source
    // of the cap fact, the [resources] block; 0 = uncapped), and the
    // resolved team after the ceiling clamp. The run-identity fields stay
    // empty — the attribution harness names its runs and pairs its own record to
    // the trace (the delta probe's enabler precedent).
    AttributionWindowGuard attributionWindow;

    if (input.memoryInstrument.enabled)
    {
        qcx::memory::AttributionTraceMetadata metadata;
        metadata.memoryCapBytes =
            input.resources.memoryCapGiB > 0.0
                ? static_cast<std::uint64_t>(input.resources.memoryCapGiB * (1ULL << 30))
                : 0;
        metadata.threadCount = static_cast<std::uint32_t>(qcx::backend::DefaultOmpTeamSize());

        qcx::memory::AllocationInstrumentOptions options;
        options.snapshotIntervalMs =
            std::chrono::milliseconds(input.memoryInstrument.snapshotIntervalMs);
        options.traceFilePath = input.memoryInstrument.traceFile;
        options.metadata = std::move(metadata);

        auto instrument = qcx::memory::AllocationInstrumentEnable(options);

        if (!instrument.has_value())
        {
            return std::unexpected(instrument.error());
        }

        attributionWindow.Adopt();
    }

    // The refusal record . The
    // instrument opens HERE, at run entry, but every admission gate lives
    // behind it: a refused run therefore reaches no tagged allocation site at
    // all, and its trace ends as a run's worth of watchdog snapshot rows with
    // every per-tag column zero and every `predicted_high_water_*` meta value
    // at the placeholder. That record is indistinguishable from a run that
    // allocated nothing tagged, which is why the call site is not the
    // instrument but this funnel: every refusal this function raises is
    // recorded in the trace before the window closes, so a refused run's
    // trace names the refusal and the reason.
    //
    // It DECIDES nothing - the refusal itself is each site's own, unchanged,
    // and this lambda hands the very same Error back. It adds one
    // `#refused <reason>` line, and only when a trace file is open (a
    // stats-only instrument, or one an embedding process owns, records
    // nothing and is never an error here).
    const auto refuse = [](qcx::Error error) -> qcx::Result<RunOutcome> {
        qcx::memory::AllocationInstrumentRecordRefusal(error.message);
        return std::unexpected(std::move(error));
    };

    // The topology probe: once per run, before the builder resolution -
    // the node/CPU/host/device bundle the device-presence condition reads (an
    // explicit "gpu" with no device present falls back to the direct
    // family). It sits after the thread-ceiling clamp so the effective
    // team it feeds is the run's team.
    //
    // The device half is cap-gated: the sweep initializes the CUDA
    // runtime, and a CUDA-linked process already holds its host footprint
    // (gpuProbeCommitGiB - the runtime DLLs are load-time) against the
    // job-object cap; under a cap below that footprint the probe's init
    // is a hard access violation, never a clean CUDA error. The gate is
    // the GPU tier's minimum admission floor (GpuProbeFloorGiB, the same
    // formula the retired estimator's floor used): below it the tier
    // cannot fit the cap anyway, so the gate never changes the fallback -
    // the resolution simply sees "no CUDA device", exactly like a
    // no-CUDA lane.
    const bool probeDevices = input.resources.memoryCapGiB >=
                              qcx::driver::GpuProbeFloorGiB(qcx::driver::SelectionCostConstants{});
    const qcx::backend::TopologyProfile topology =
        qcx::backend::DetectTopologyProfile(probeDevices);

    // The device compute profile: the certified fp32 lane's default
    // , probed at the same run boundary and behind the SAME cap
    // gate - one probe, one gate, one run. Its read cannot live in the
    // certified lane's construction point (integrals): that point is per
    // builder instance, the probe's own contract is once per run, and its
    // CUDA-runtime init under a sub-footprint cap is the access violation
    // the gate above exists for. The value travels to every wiring site
    // that assembles FockBuildOptions, so the lane's default is one
    // verdict per run rather than one per builder. A closed gate leaves
    // the documented "unknown" profile, whose ratio is below the lane's
    // threshold by construction: the conservative interim, and exactly
    // the value a no-CUDA lane reports.
    const qcx::backend::GpuComputeProfile deviceProbeProfile =
        probeDevices ? qcx::backend::DetectGpuComputeProfile(0) : qcx::backend::GpuComputeProfile{};

    // The HOST compute profile: the CPU arm's producer, read
    // at this same run boundary, once per run, and deliberately NOT behind the
    // cap gate above - the reverse of the device case, stated because copying
    // the gate is the intuitive move and it would be wrong. The host probe
    // initializes nothing, allocates nothing, starts no OpenMP team and
    // touches no memory beyond the stack: a register-bound FMA chain cannot be
    // distorted by a cap, so gating it would only lose the number on exactly
    // the capped runs whose verdict it decides.
    const qcx::backend::HostComputeProfile hostComputeProfile =
        qcx::backend::DetectHostComputeProfile();

    const qcx::io::RunResourcesResolved resourcesResolved = {input.resources.memoryCapGiB,
                                                             input.resources.threadCap,
                                                             caps->inProcessCapApplied,
                                                             std::move(caps->note)};

    // Semantic validation first: fail before any molecule,
    // basis-set, or Fock-builder construction. The CLI prints the full
    // report; library callers receive the first issue as the Error.
    const auto validation = qcx::io::ValidateInput(input);

    if (!validation.IsValid())
    {
        return refuse(Err(qcx::ErrorCode::kInvalidArgument, validation.issues.front()));
    }

    auto combination = ValidateCombination(input);

    if (!combination.has_value())
    {
        return refuse(combination.error());
    }

    auto atoms = BuildAtoms(input.molecule.atoms);

    if (!atoms.has_value())
    {
        return refuse(atoms.error());
    }

    auto coordinates = MakeCoordinates(input.molecule.atoms);

    if (!coordinates.has_value())
    {
        return refuse(coordinates.error());
    }

    // Computed before the move into Molecule::Create below: the input-order
    // -> canonical-order permutation the NOCV fragment groups map through.
    const auto canonicalOrder = CanonicalAtomOrder(*atoms, *coordinates);

    auto molecule = qcx::molecule::Molecule::Create(std::move(*atoms),
                                                    std::move(*coordinates),
                                                    input.molecule.charge,
                                                    input.molecule.multiplicity);

    if (!molecule.has_value())
    {
        return refuse(molecule.error());
    }

    const std::filesystem::path root(QcxBasisDataDir);
    // The filtered parse: keep only the molecule's
    // elements, so the parsed set and the engine's molecule-scoped counts
    // are equal by construction - the memory model's basis count is then
    // the engine's n, no re-scan.
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered((root / input.basis.orbital).string(),
                                                             UniqueAtomicNumbers(*molecule));

    if (!basis.has_value())
    {
        return refuse(basis.error());
    }

    // The builder resolution (the owner's ruling 2026-09-13): the no-builder
    // default is a size LADDER - the direct family's lean member at
    // nBasis <= 1000, ri_j_link to 2000, qfmm above, the SAME for RHF, UHF,
    // RKS and UKS - and direct (the budgeted machinery), both disk routes
    // and gpu are explicit opt-ins. Its ONE exception is keyed on the
    // functional's character, not the method (IsNonHybridKohnSham: a pure
    // Kohn-Sham functional promotes ri_j_link above the lean member). The
    // tiers this run cannot WIRE are stated here (LadderRunnabilityFor: the
    // aux the ri_j engine needs, and the v1 method whitelists) so the ladder
    // demotes and discloses instead of running a tier nothing established;
    // the device-less GPU fallback resolves here too. The record is
    // reported in resources_resolved and a divergence reaches stderr loudly
    // (never a silent substitution). The estimator's pricing inputs stay
    // retired - only the tier RUNNABILITY travels in, and an explicit
    // ri_j_link run keeps its authoritative aux parse in the wiring.
    const std::size_t nBasis = CountBasisFunctions(*molecule, *basis);
    const std::size_t nPairs = CountShellPairs(*molecule, *basis);
    const LadderTierRunnability ladderRunnability = LadderRunnabilityFor(input);
    const bool nonHybridKohnSham = IsNonHybridKohnSham(input);

    const qcx::driver::SelectionResolutionInput resolutionInput{topology,
                                                                input,
                                                                nBasis,
                                                                0,
                                                                ladderRunnability.riJ,
                                                                ladderRunnability.qfmm,
                                                                nonHybridKohnSham,
                                                                nPairs,
                                                                qcx::backend::DefaultOmpTeamSize()};
    auto selection = qcx::driver::ResolveBuilderSelection(resolutionInput);

    if (!selection.has_value())
    {
        return refuse(selection.error());
    }

    if (!selection->warning.empty())
    {
        std::fprintf(stderr, "qcx: warning: %s\n", selection->warning.c_str());
    }

    // The DEVICE requirement (`[builder] device`) against the builder the run
    // actually resolved, and this is the resolution point for the whole
    // requirement. The parser already refuses the two `[builder]` keys disagreeing
    // with one another; what it cannot see is the RESOLVED kinds - the deprecated
    // `[method] fock_builder` word reaches the device path without passing through
    // the axes block, a programmatic caller fills the field directly, and a
    // `gpu` request in a build with no CUDA device is demoted to the ladder (the
    // selection's own warning above) before this point is reached.
    //
    // So the requirement is answered against `selection->kind`, the ONE value the
    // wiring acts on: a `cuda:<index>` requirement needs the device kind, a
    // `host` requirement needs any other, and either mismatch is REFUSED BY NAME
    // rather than resolved or demoted. Demotion is deliberately not the posture
    // here although it is for the `gpu` request alone: a request that names a
    // mechanism may be demoted with its demotion disclosed, but a REQUIREMENT
    // states where the run must execute, and a run that executed elsewhere while
    // the record said "cuda:0" is the substitution this key exists to make
    // impossible.
    if (input.builder.device.has_value())
    {
        const std::string selector = qcx::io::DeviceSelectorText(*input.builder.device);
        const bool deviceRequired = input.builder.device->target == qcx::io::DeviceTarget::kCuda;
        const bool routedToDevice = selection->kind == qcx::io::BuilderKind::kGpu;

        if (deviceRequired && !routedToDevice)
        {
            return refuse(Err(
                qcx::ErrorCode::kUnimplemented,
                "builder.device = \"" + selector +
                    "\" requires the device path, and this run resolved to fock_builder = \"" +
                    std::string(qcx::io::ToString(selection->kind)) +
                    "\" on the host: the requirement cannot be honoured, so the run is refused "
                    "rather than executed elsewhere. State builder.device = \"host\", or wire the "
                    "device backend (builder.execution_backend = \"gpu\") in a build with a CUDA "
                    "device present"));
        }

        if (!deviceRequired && routedToDevice)
        {
            return refuse(Err(
                qcx::ErrorCode::kUnimplemented,
                "builder.device = \"" + selector +
                    "\" requires the host path, and this run resolved to the device builder: one "
                    "run cannot require both, so the pair is refused. Drop the key, or state a "
                    "cuda:<index> device"));
        }
    }

    // The COMPUTE TARGET's profile . Both probes were taken at
    // the run boundary above; WHICH one counts is decided here, by the family
    // the run actually wired, and it is decided once. A device number reaching
    // a CPU builder would turn the HOST lane ON on a box whose device probe
    // reads 31.2 while its host probe reads 2 - the device-vs-host inversion
    // the probe measured - and a host number reaching a device builder
    // would turn the device lane OFF where it is the one that pays.
    //
    // The value is const and read-only past this point, so the input every
    // builder seeds cannot drift from the record made below. The field it
    // travels in is still named deviceComputeProfile (FockBuildOptions): the
    // rename to `computeProfile` is a named open item, a named open owner
    // decision .
    const bool gpuTarget = GpuFamily(selection->kind);
    const qcx::backend::GpuComputeProfile computeProfile =
        gpuTarget ? deviceProbeProfile : ComputeTargetProfileOf(hostComputeProfile);

    // The pre-gate setup admission, BEFORE the ramp it sizes
    // (CheckPreGateSetupAdmission): the setup below walks the whole
    // canonical pair list once per one-electron matrix (the kinetic,
    // nuclear and overlap builds) - one chunk of contracted pair data at a
    // time since 93d5dfc5, the whole store per walk before it - so a cap
    // under that footprint killed the process here, inside the ramp, with no
    // admission decision reached: the wiring's own gates all sit behind it.
    // Every input this check needs (the molecule, the parsed basis, the
    // resolved kind, the cap) is final at this point; the refusal is the
    // wiring's own text, taken one step earlier.
    //
    // The auxiliary basis joins those inputs when the family consumes one:
    // the check's bound charges one Schwarz sweep per basis in effect, and
    // the aux sweep is the larger of the two on the fixture that measured
    // this gap. It is resolved and parsed the SAME way the wiring does
    // (AuxNameInEffect + ParseNwchemDirectoryFiltered over the same root and
    // elements), so the bound cannot charge a sweep the route never runs. A
    // name that does not resolve here is NOT refused here: the wiring's own
    // aux error is the accurate one, and the check must not pre-empt it -
    // the bound is then the orbital sweep alone, which is the narrower (and
    // still never-under) charge.
    std::optional<qcx::basisset::BasisSet> preGateAuxBasis;

    if (qcx::io::BuilderConsumesAux(selection->kind))
    {
        auto preGateAuxName = AuxNameInEffect(input, selection->kind);

        if (preGateAuxName.has_value() && !preGateAuxName->empty())
        {
            auto parsedPreGateAux = qcx::basisset::ParseNwchemDirectoryFiltered(
                (root / *preGateAuxName).string(), UniqueAtomicNumbers(*molecule));

            if (parsedPreGateAux.has_value())
            {
                preGateAuxBasis = std::move(*parsedPreGateAux);
            }
        }
    }

    auto preGateSetup =
        CheckPreGateSetupAdmission(input,
                                   selection->kind,
                                   *molecule,
                                   *basis,
                                   preGateAuxBasis.has_value() ? &*preGateAuxBasis : nullptr,
                                   input.resources.memoryCapGiB,
                                   selection->leanMember);

    if (!preGateSetup.has_value())
    {
        return refuse(preGateSetup.error());
    }

    // THE ONE RESERVE, from the one source the arm above also refused on: the
    // grant every budget site below is built from is the cap MINUS this number
    // (WorkspaceGrantBytes), so the arm's floor and the engine's rung condition
    // cannot drift apart again. Its two consumers are this thread-through and
    // nothing else - a site that computed its own would be the defect this
    // replaced, which tools/check_workspace_grant.py fails on.
    const std::size_t workspaceReserveBytes = static_cast<std::size_t>(*preGateSetup);

    auto coreTensor = BuildCoreHamiltonian(*molecule, *basis);

    if (!coreTensor.has_value())
    {
        return refuse(coreTensor.error());
    }

    auto overlapTensor = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlapTensor.has_value())
    {
        return refuse(overlapTensor.error());
    }

    const Eigen::MatrixXd core = ToMatrix(*coreTensor);
    const Eigen::MatrixXd overlap = ToMatrix(*overlapTensor);
    // The dispatch resolves the method through the same classifier the
    // up-front combination check reads - the two cannot disagree about which
    // words are wired - and re-reads it here so the branches below cannot be
    // entered on an unclassified word even if that check is ever reshaped.
    const auto scfPath = ResolveScfPath(input.method.method);

    if (!scfPath.has_value())
    {
        return refuse(scfPath.error());
    }

    const bool isUnrestricted = IsUnrestrictedPath(*scfPath);
    const bool isKs = IsKohnShamPath(*scfPath);

    // `enforce_certified_bound` is a CLAIM, and this is its resolution point.
    //
    // The enforcement is folded into the Fock build by ONE route: the RHF
    // direct family's machinery member, whose FastPath compares the routed
    // bound sum against the preset-derived budget and whose other rungs (the
    // LightPath chunk loop, an engaged precision ladder) refuse BY NAME with
    // kUnimplemented from inside BuildFock (fock_build.cpp) - those two are
    // honoured-or-refused and need nothing here.
    //
    // Every other route DROPS the request, so each is refused here instead,
    // before any builder is wired:
    //   - the direct family's LEAN member (the auto-lean default at
    //     nBasis <= 1000, or an explicit fock_builder = "lean"): the lean arm
    //     builds LeanFockBuildOptions, which has no enforcement field at all
    //     (maxBatchBytes, memoryCapGiB, maxParallelChunks, forceScalarContract,
    //     symmetryReduction, buildCoulombOnly, buildExchangeOnly), so the field
    //     is dropped BY CONSTRUCTION - not by oversight - and lean is the
    //     default small-molecule route;
    //   - ri_j_link / qfmm / gpu: RiEngineOptions has no such field either, and
    //     the option appears in no RI, QFMM or CUDA translation unit;
    //   - either UNRESTRICTED leg: the direct-UHF wiring never assigns
    //     FockBuildOptions::enforceCertifiedBoundBudget, and the UKS
    //     composition that rides it does not either - the option travels to
    //     the direct family's options struct in the RHF wiring alone.
    //
    // RKS is the one Kohn-Sham lane that CAN honour it, for the same reason
    // the machinery member can: the composition builds its halves out of the
    // RHF direct options (directOptions), the enforcement rides that struct,
    // and the run's certified-bound collector is folded by the halves - so an
    // RKS run on the machinery member reports the bound it enforced.
    //
    // The judgement reads the RESOLVED kind and the RESOLVED member choice
    // the wiring reads (the resolution's leanMember), so it cannot disagree
    // with the builder that actually runs - which is also what admits the
    // retargeted device-less gpu fallback: it lands on this machinery
    // member (the ladder's tier above the lean boundary), so it CAN honour
    // the request and is not refused.
    const bool reachesCertifiedBoundRoute = !isUnrestricted &&
                                            selection->kind == qcx::io::BuilderKind::kDirect &&
                                            !selection->leanMember;

    if (input.method.enforceCertifiedBound && !reachesCertifiedBoundRoute)
    {
        // The route names the METHOD, not its shape: "an unrestricted run"
        // is true of two different words now that UKS exists, and a refusal
        // that stops naming what it refuses is a refusal an author cannot
        // act on (the same rule the dispatch's own refusals follow).
        const std::string route =
            isUnrestricted ? (isKs ? std::string{"a UKS run (the per-spin direct wiring carries no "
                                                 "enforcement)"}
                                   : std::string{"a UHF run (the per-spin direct wiring carries no "
                                                 "enforcement)"})
                           : (selection->kind == qcx::io::BuilderKind::kDirect
                                  ? std::string{"the direct family's LEAN member"}
                                  : "fock_builder = \"" +
                                        std::string(qcx::io::ToString(selection->kind)) + "\"");

        return refuse(
            Err(qcx::ErrorCode::kUnimplemented,
                "method.enforce_certified_bound = true resolves to " + route +
                    ", which drops the request instead of enforcing it: the certified-bound "
                    "budget enforcement is implemented by the RHF direct family's machinery "
                    "member only, so no other route can report the certified bound it asserts. "
                    "The run is refused rather than computing as if the key were absent. Drop "
                    "the key, or run it on that member - the direct family above the lean "
                    "ceiling, or an explicit fock_builder = \"direct\" at any size"));
    }

    // The disk-tier ERI store (`method.eri_cache_store`) on the
    // unrestricted legs: the key's resolution point, and the shape the
    // refusal above and ResolveScfPath both follow.
    //
    // WIRED, and this is the narrowing that says how far. The key's consumer
    // is a batch engine pair on a builder's options, and on the unrestricted
    // legs the DIRECT family is the one that owns such a pair: its runner
    // installs the decorator on the per-spin coulomb and exchange halves, on
    // the same grant path their cap-derived cache budget rides (this is the
    // seam the earlier text announced as "not yet wired on the UHF path",
    // and that sentence is what this change discharges). The other
    // unrestricted families - ri_j_link, ri_jk, qfmm, gpu, gpu_split - wire
    // their own builders and carry no such pair, so the key still cannot be
    // honoured there.
    //
    // Those stay REFUSED rather than silently dropped: an explicit request
    // the run drops is a disclosure-rule violation whether or not a
    // follow-on is planned. A refusal is the honest answer exactly here and
    // not on the restricted legs, where the same families DEMOTE through the
    // record: a refusal is a run that never reaches serialization, so it is
    // the one outcome the `eri_store` block cannot disclose, and it is
    // therefore reserved for the case the block cannot state - an
    // unrestricted family whose requested arrangement has no other name.
    if (!input.method.eriCacheStore.empty() && isUnrestricted &&
        selection->kind != qcx::io::BuilderKind::kDirect)
    {
        return refuse(Err(
            qcx::ErrorCode::kUnimplemented,
            UnrestrictedEriStoreRefusalText(input.method.eriCacheStore, selection->kind, isKs)));
    }

    // The certified fp32 lane's REQUEST (the owner's ruling 2026-09-13): the
    // [method] force_certified_lane key, and this is its resolution point -
    // the ONE place the request is read, in the shape ResolveScfPath and the
    // sibling refusal above both follow.
    //
    // Why the key exists at all. Both arms of the certified lane's default had
    // only ever been exercised by injecting DetectHostComputeProfile's or the
    // device probe's return value in SOURCE (the host-ratio probe forced 31.2,
    // the device arm did the same under the device compute profile). On a machine whose probe
    // resolves OFF - this laptop's host probe measures ~2 against a 4.0
    // threshold - the ON path was therefore unreachable from any input, and
    // the pins could only ever assert the OFF behaviour. This key makes the
    // ON path a supported, recorded request.
    //
    // The request is the value the assembly sites seed into
    // FockBuildOptions::useCertifiedMixedPrecision: `true` is the explicit
    // request ResolveCertifiedLane returns as written (the disclosure rule - the probe's
    // verdict is then not consulted for this run), and nullopt leaves the
    // field UNSET, which is the probe's verdict. The probe still decides the
    // DEFAULT; this key is the visible override, and the record says which of
    // the two the lane's state came from.
    //
    // Two judgements, both on the RESOLVED kind/route rather than on any raw
    // method word, and both refusing BY NAME rather than running as if the key
    // were absent:
    //   1. the ROUTE - only the direct family's machinery members carry the
    //      request into a live fp32 lane (RouteCarriesCertifiedLaneRequest);
    //   2. the PRESET - at kTight the lane's gate admits no quartet at all
    //      (MixedPrecisionThreshold(kTight) is 0.0 by construction, the pinned default), so
    //      the request cannot be honoured there on ANY route. This one is
    //      deliberately NOT the sibling key's posture: the budget enforcement
    //      is ACCEPTED and runs vacuously where the lane is not engaged (the
    //      the ruling), but a request that names the LANE ITSELF cannot
    //      be honoured by a preset that disables it, and accepting it would
    //      be the silent no-op the disclosure rule forbids.
    const std::optional<bool> certifiedLaneRequest =
        input.method.forceCertifiedLane ? std::optional<bool>{true} : std::nullopt;

    if (certifiedLaneRequest.has_value() &&
        !RouteCarriesCertifiedLaneRequest(selection->kind, selection->leanMember))
    {
        const std::string route =
            selection->kind == qcx::io::BuilderKind::kDirect
                ? std::string{"the direct family's LEAN member"}
                : "fock_builder = \"" + std::string(qcx::io::ToString(selection->kind)) + "\"";

        return refuse(
            Err(qcx::ErrorCode::kUnimplemented,
                "method.force_certified_lane = true resolves to " + route +
                    ", which runs no fp32 lane the request can reach: the certified lane's "
                    "request travels in FockBuildOptions::useCertifiedMixedPrecision, and this "
                    "route assembles a different options struct (the direct family's lean "
                    "member runs the lean builder, which has no fp32 lane at all) or one whose "
                    "lane is derived elsewhere (ri_j_link's exchange half reads the accuracy "
                    "preset directly, and the qfmm and gpu families do not seed the field). The "
                    "run is refused rather than computing as if the key were absent. Drop the "
                    "key, or run it on the direct family's machinery member - the direct family "
                    "above the lean ceiling, or an explicit fock_builder = \"direct\" at any "
                    "size"));
    }

    if (certifiedLaneRequest.has_value() &&
        input.method.accuracy == qcx::integrals::AccuracyPreset::kTight)
    {
        return refuse(
            Err(qcx::ErrorCode::kUnimplemented,
                "method.force_certified_lane = true resolves to accuracy = \"kTight\", whose "
                "mixed-precision gate admits no quartet at all (MixedPrecisionThreshold(kTight) "
                "is 0.0 by construction, so the fp32 lane is off at this preset whatever the "
                "request says - the strict-pins contract). The run is refused rather than "
                "computing with the lane off as if the key were absent. Drop the key, or run at "
                "accuracy = \"kNormal\" or \"kLoose\""));
    }

    // The per-element SAD fragment inputs (BuildAtomicInputs) are shared
    // by the SAD guess, the Hirshfeld promolecular densities, and the
    // Fukui charged-species runs - built once here whenever any consumer
    // needs them, never per call. The optional is engaged exactly when the
    // SAD guess or the Hirshfeld analysis is requested; the charged Fukui
    // species inherit the run's guess, so a SAD guess covers them too.
    std::optional<std::map<int, qcx::scf::AtomicUhfInputs>> atomicInputs;

    if (input.properties.hirshfeld || (isUnrestricted && input.guess == qcx::io::GuessKind::kSad))
    {
        auto built = BuildAtomicInputs(*molecule, *basis);

        if (!built.has_value())
        {
            return refuse(built.error());
        }

        atomicInputs = std::move(*built);
    }

    qcx::io::RunResult result;
    result.resourcesResolved = resourcesResolved;
    // The unit the geometry was READ under: the parser's resolved
    // [molecule] units value ("angstrom" unless the input asked for bohr), so
    // the record answers the question without the input file. It is the
    // parser's own resolved member, never re-derived here - a second
    // resolution would be a second home for the answer (the checkpoint-
    // fingerprint rule applied to a unit).
    result.moleculeUnits = input.molecule.coordinateUnit;

    // The run's own physics . `method` is the word the
    // dispatch acted on - ScfPathMethodWord reads the classifier's own output
    // (scfPath, resolved above), never the input enumerator re-read - so the
    // record names the path that ran and not the request it came from. Before
    // this key the record named the Fock builder and no physics at all: an
    // RKS run and its RHF twin were indistinguishable in the artifact a user
    // keeps, differing only in their numbers.
    result.method = ScfPathMethodWord(*scfPath);

    // The two Kohn-Sham-only disclosures, written under the SAME condition
    // that made them exist: a run whose path integrates a density functional
    // named one ([method] functional, which an rhf/uhf run's input refuses)
    // and built one XC grid (the engine ResolveKsContext creates, the only
    // grid build in the driver). A Hartree-Fock run named no functional and
    // built no XC grid, so both keys are ABSENT there - never null, and never
    // an xc_grid of defaults claiming a grid whose quadrature never touched
    // those numbers.
    if (isKs && input.method.functional.has_value())
    {
        // The registry's own canonical spelling, not the file's string: the
        // same lookup ResolveKsContext ran before anything was wired, read
        // again here for the record (the pre-gate admission's own second
        // lookup - the registry is a pure function of the name, and the two
        // callers cannot disagree about what a name resolves to).
        auto functional = qcx::driver::internal::ResolveKsFunctional(*input.method.functional);

        if (!functional.has_value())
        {
            return refuse(functional.error());
        }

        result.functional = functional->name;
        // The grid the engine was CREATED WITH: the same resolved
        // settings ResolveKsContext handed to XcGridEngine::Create, through
        // the same one function, so the record cannot name a grid the run did
        // not build.
        result.xcGrid =
            MakeRunXcGrid(ResolveXcGridSettings(input.grid.value_or(qcx::io::RunGridInput{})));
    }

    // The selection record: the wired builder (the
    // no-key ladder's tier for this run - the direct family's lean member
    // at nBasis <= 1000, ri_j_link to 2000, qfmm above, or the tier a
    // demotion landed on), the pick the record reports, the input's
    // explicit builder when one was given, and the divergence warning -
    // serialized as the last member of resources_resolved (kSchemaVersion
    // 7). The absent members stay absent (the null honesty policy).
    qcx::io::RunSelection runSelection;
    runSelection.builder = selection->kind;
    runSelection.picked = selection->picked;
    runSelection.reasoning = selection->reasoning;
    // The within-family member name: the family word above stays
    // the family word - the lean member reports "direct" like the machinery -
    // and this names the member that ran, so a consumer reads the choice
    // rather than inferring it from explicit_builder's absence. The member
    // is the RESOLUTION's own (leanMember - the RHF wiring branch, the UHF
    // runner's lean arm, the pre-gate setup admission and the certified-lane
    // route judgement all read that same value), so the name and the builder
    // that actually ran cannot disagree; the memory_model lean flag is filled
    // from the same value at the same sites, and the driver tests pin the
    // three against each other.
    runSelection.builderMember = BuilderMemberName(
        selection->kind, selection->kind == qcx::io::BuilderKind::kDirect && selection->leanMember);

    // The RESOLVED selection in the orthogonal axis vocabulary (the
    // owner's ruling 2026-09-17), the record's top-level `builder_axes` block.
    // Every word is read off the RESOLUTION rather than off the input, so the
    // block cannot name an axis the wiring did not take: the family and the
    // backend come from the resolved kind, and the tier from the same
    // within-family predicate the builder_member name above is built from -
    // which is why the two are filled side by side from one value. The legacy
    // half is the input's own deprecation record (the word the deprecated
    // [method] fock_builder key spelled, or absent), and `requested_by` names
    // which of the three spellings produced the selection.
    const qcx::io::BuilderAxes resolvedAxes = qcx::io::AxesOfBuilder(
        selection->kind, selection->kind == qcx::io::BuilderKind::kDirect && selection->leanMember);

    // The three spellings, as the record states them. The deprecated key's word
    // is the input's own, so a run whose selection was DEMOTED still names the
    // word the file wrote while the axes above name what ran - which is the
    // point of the block.
    const char* const requestedBy =
        input.builder.legacyFockBuilderWord.has_value() ? "fock_builder"
        : (input.builder.integralFamily.has_value() || input.builder.storageTier.has_value() ||
           input.builder.executionBackend.has_value())
            ? "axes"
            : "ladder";

    result.builderAxes = qcx::io::RunBuilderAxes{
        std::string(qcx::io::ToString(resolvedAxes.integralFamily)),
        std::string(qcx::io::ToString(resolvedAxes.storageTier)),
        std::string(qcx::io::ToString(resolvedAxes.executionBackend)),
        input.builder.legacyFockBuilderWord,
        requestedBy,
        // The device the run REQUIRED, in the selector vocabulary the file
        // wrote (schema 39). Absent when the key was absent - an omitted key
        // is not a requirement - and present only on the runs that reached
        // serialization with the requirement already answered against the
        // resolved kind above, so a document carrying it is a document whose
        // kernels executed where the input said they must.
        input.builder.device.has_value()
            ? std::optional<std::string>(qcx::io::DeviceSelectorText(*input.builder.device))
            : std::nullopt};

    // The auxiliary-basis weak-region notice: the disclosure
    // half of the demotion the aux rule performs. The rule ALWAYS resolves a
    // default, so a run whose selection lands in a region the RI fit serves less
    // reliably is demoted-to-a-default WITH disclosure rather than refused - and
    // a demotion the record cannot show is not a demotion, it is the silent
    // substitution the contract exists to forbid. Read at the resolution point
    // (AuxNoticeInEffect, beside the name rule the checkpoint fingerprint binds)
    // rather than recomputed here, so the notice cannot describe a selection the
    // wiring did not make.
    const std::optional<std::string> auxNotice = AuxNoticeInEffect(input, selection->kind);

    // The exchange-approximation disclosure . An ri_jk run's exchange
    // half is contracted through the auxiliary fit, so its energy is a DIFFERENT
    // result class from the direct family's exact kernels - the second clause of
    // the refusal that used to stand here, answered by disclosure rather than by
    // deletion. The aux beside it is read through the driver's
    // own single-place rule (AuxNameInEffect - the same function the checkpoint
    // fingerprint binds), so the disclosure cannot name an aux the wiring did not
    // resolve.
    //
    // A resolution failure is fatal HERE rather than left to omit the key: the
    // alternative is a record reporting an approximated energy without saying
    // what approximated it, which is the disclosure quietly dropped - and the
    // wiring would refuse the same request moments later anyway, so nothing that
    // could have run is lost by refusing at the record.
    if (selection->kind == qcx::io::BuilderKind::kRiJk)
    {
        auto auxInEffect = AuxNameInEffect(input, selection->kind);

        if (!auxInEffect.has_value())
        {
            return refuse(auxInEffect.error());
        }

        runSelection.approximation =
            qcx::io::RunApproximation{std::string(kOccRiKMemberName), *auxInEffect};

        // The rule here is that this feature closes with its error
        // DISCLOSED - and a disclosure the record cannot emit is not one, so the
        // achieved per-atom error is written here beside the bar it is read
        // against (RunExchangeError). This is the ONLY site that
        // builds an approximation block in the driver, and it is the kRiJk arm,
        // so the measured-error member is present exactly on the runs whose
        // exchange half is approximated and absent on every other arm including
        // the exact-kernel notice blocks (the RunExchangeError presence rule).
        //
        // The two numbers come from their own homes and are not recomputed here:
        // the achieved value is the ri_jk accuracy cell's measurement, carried
        // by the integrals mapping (RiExchangeWorstMeasuredPerAtomError - the
        // path's worst characterized cell, with the fixture named so the record
        // shows whose error it is), and the bar is this run's own preset's
        // (RiExchangeErrorBudgetPerAtom). The preset word goes beside the bar
        // because the record echoes [method] accuracy nowhere else.
        const qcx::integrals::RiExchangeErrorMeasurement measuredExchangeError =
            qcx::integrals::RiExchangeWorstMeasuredPerAtomError();

        runSelection.approximation->exchangeError = qcx::io::RunExchangeError{
            measuredExchangeError.perAtomEh,
            qcx::integrals::RiExchangeErrorBudgetPerAtom(input.method.accuracy),
            std::string(qcx::io::ToString(input.method.accuracy)),
            std::string(measuredExchangeError.fixture),
            std::string(measuredExchangeError.auxBasis)};
    } else if (auxNotice.has_value())
    {
        // The notice-only arm the ruling creates, and the arm
        // that keeps the RI-J weak regions visible. `ri_j_link` fits its COULOMB
        // half through an aux while its exchange half is the exact direct
        // kernel, so it builds no approximation on its own - but the weak
        // regions the measurement actually covers (minimal, unpolarized,
        // diffuse, ECP) are reached on THIS path, and an ri_jk-only emitter
        // would leave all of them unstated in the record.
        //
        // A notice must never hang off an ABSENT block: a reader who sees a
        // warning must not have to infer that the block was omitted because the
        // kernel was exact. So the block is built here, and `exchange` reads
        // "exact" - the meaning the absent case already carries, STATED rather
        // than left to be reconstructed from the absence of a fitted-kernel
        // member name. No measured-error member hangs here: the fitted
        // exchange's error is not an error of this run's.
        auto auxInEffect = AuxNameInEffect(input, selection->kind);

        if (!auxInEffect.has_value())
        {
            return refuse(auxInEffect.error());
        }

        runSelection.approximation =
            qcx::io::RunApproximation{std::string(kExactKernelExchangeWord), *auxInEffect};
    }

    // The notice rides the block whenever one exists, on EITHER arm: it is a
    // field ON the disclosure, never a replacement for it, and the two arms
    // differ in which half of the run is approximated rather than in whether
    // the reader is told about the aux.
    if (runSelection.approximation.has_value() && auxNotice.has_value())
    {
        runSelection.approximation->auxNotice = *auxNotice;
    }

    if (selection->explicitBuilder)
    {
        runSelection.explicitBuilder = input.method.builder;
    }

    if (!selection->warning.empty())
    {
        runSelection.warning = selection->warning;
    }

    result.resourcesResolved.selection = std::move(runSelection);

    // The certified fp32 lane's DECIDED record: the
    // numbers its default was read off, the verdict those numbers resolved to,
    // and the threshold they were compared against - all of them, on every run
    // that resolved a profile, because a passing gate must still emit its
    // readings. What is recorded is the lane's DEFAULT,
    // never a caller's explicit request: an explicit request resolves at one
    // point and is not a property of this machine. RunCertifiedBound is
    // the same lane's DELIVERED record, and the two together let a reader
    // check the verdict against the machine that produced it.
    //
    // TWO ARMS, ONE OF THEM UNCOVERED, stated here because the fill below is
    // one block and reads as one thing. Every value below is chosen by a
    // `gpuTarget ?` ternary - four of them - so the block has a HOST arm (the
    // direct/lean, ri_j_link, ri_jk and qfmm families) and a DEVICE arm (the
    // kGpu/kGpuSplit families, run_driver.cpp GpuFamily). No test in this tree
    // exercises the device arm, and none can: GpuFamily is true only for
    // kGpu/kGpuSplit, a non-CUDA build refuses kGpu outright (the branch above
    // the machinery tail), and the device-less fallback resolves an
    // explicit "gpu" request to a CPU kind BEFORE either the predicate or this
    // block is read - so on every build this tree can produce, gpuTarget is
    // false and these four ternaries take their host side. The driver pin that
    // covers this block therefore covers its HOST half only, and says so. A
    // CUDA lane is what would cover the other half.
    {
        qcx::io::RunComputeProfile computeRecord;
        computeRecord.source = gpuTarget ? "device" : "host";
        computeRecord.fp32Gflops = computeProfile.fp32Gflops;
        computeRecord.fp64Gflops = computeProfile.fp64Gflops;
        computeRecord.ratio = computeProfile.fp32ToFp64Ratio;
        // The device probe measures once and reports no pair count and no
        // spread - so those two stay at their "not reported" values rather
        // than borrowing the host arm's. Its own success criterion is what
        // `measured` reads (gpu_compute_profile.cpp:42): a positive
        // throughput means the measurement ran.
        computeRecord.measured =
            gpuTarget ? computeProfile.fp32Gflops > 0.0 : hostComputeProfile.measured;
        computeRecord.simdLane = !gpuTarget && hostComputeProfile.simdLane;
        computeRecord.pairs = gpuTarget ? 0 : hostComputeProfile.pairs;
        computeRecord.ratioSpread = gpuTarget ? 0.0 : hostComputeProfile.ratioSpread;
        computeRecord.certifiedLaneMinRatio = qcx::integrals::kCertifiedLaneMinRatio;
        computeRecord.certifiedLaneDefault =
            qcx::integrals::CertifiedLaneDefaultForRatio(computeRecord.ratio);
        // The one field in this block that is NOT a property of the machine
        // (the owner's ruling 2026-09-13): true when the input
        // FORCED the lane through [method] force_certified_lane instead of
        // letting the verdict above decide it. It is recorded BESIDE the
        // verdict, never instead of it - the block keeps publishing what the
        // probe measured and what its verdict would have been - so a reader
        // can never read a forced run's lane state as this machine's
        // measurement. The value is the resolution point's own request (the
        // only producer), not a second reading of the input key.
        computeRecord.certifiedLaneForced = certifiedLaneRequest.has_value();
        // The block is validated where it is MADE, not where it is described
        // (RunComputeProfile::Create): a present block names an arm that
        // resolved, carries the one policy threshold, and leaves the device
        // arm's not-reported readings at their zeros. This producer cannot
        // break any of the three - it is the only one - which is exactly why
        // the check lives on the type: a second producer inherits it, where it
        // would inherit a comment only by reading it. UNREACHABLE in a correct
        // build, so a refusal here is a defect: the run stops rather than
        // emitting a record it cannot stand behind.
        auto validated = qcx::io::RunComputeProfile::Create(std::move(computeRecord));

        if (!validated.has_value())
        {
            return refuse(validated.error());
        }

        result.resourcesResolved.computeProfile = std::move(*validated);
    }

    if (isUnrestricted)
    {
        // The run's disk-tier ERI store wiring (`method.eri_cache_store`) on
        // this arm, the out-parameter the direct runner installs the
        // engine-decorator factory on and this scope reads back for the
        // record. Declared for every unrestricted family even though only the
        // direct one can install a decorator: the record is the DISCLOSURE
        // surface, so a request that resolved to a family with no engine pair
        // must still serialize what it asked for (the null-honesty rule this
        // block's siblings follow).
        EriStoreWiring eriStore;

        if (selection->kind == qcx::io::BuilderKind::kRiJLink)
        {
            // The unrestricted RI-J-link path (the per-spin adapter): the
            // same F_sigma = H + J(P_tot) - K(P_sigma) assembly the direct
            // arm runs, with the RI-J engine's Coulomb half in place of the
            // direct one - and, unlike the direct family, no per-channel H
            // subtraction (the RI contractions carry no H; the exchange-only
            // half is the sole carrier). The memory model is the family's own
            // (nBasis, nAux) pair, so the audit block records the terms the
            // cap was weighed against - the RHF leg's numbers.
            const auto auxName = AuxNameInEffect(input, qcx::io::BuilderKind::kRiJLink);

            if (!auxName.has_value())
            {
                return refuse(auxName.error());
            }

            auto auxBasis = qcx::basisset::ParseNwchemDirectoryFiltered(
                (root / *auxName).string(), UniqueAtomicNumbers(*molecule));

            if (!auxBasis.has_value())
            {
                return refuse(auxBasis.error());
            }

            auto run = RunRiJLinkUhfScf(*molecule,
                                        *basis,
                                        *coreTensor,
                                        overlap,
                                        core,
                                        atomicInputs,
                                        input,
                                        workspaceReserveBytes,
                                        root,
                                        true);

            if (!run.has_value())
            {
                return refuse(run.error());
            }

            result.resourcesResolved.workspaceBudget = MakeRunWorkspaceBudget(run->workspaceBudget);
            result.resourcesResolved.modeRecord = MakeRunModeRecord(run->modeInfo);
            // The exchange half's own record: absent on this family, exactly
            // as on its RHF twin (RiJkFockBuilder exposes one Create-time
            // record for the composed build; the nested exchange half's
            // decision rides inside it).
            result.resourcesResolved.exchangeModeRecord = MakeRunModeRecord(run->exchangeModeInfo);
            // The requested-vs-ran pairing and its two sibling
            // disclosures, filled from the same mode info the RHF leg uses.
            // The `disk` and forced-disk requests cannot reach this arm (the
            // runner and the combination check refuse them by name), so what
            // these records state here is the in-memory ladder's own answer.
            result.resourcesResolved.riTensorMode =
                MakeRiTensorModeRecord(input, selection->kind, run->modeInfo);
            result.resourcesResolved.riChunkBytes =
                MakeRiChunkBytesRecord(input, selection->kind, run->modeInfo);
            // The orbit-expansion request record: the engine's own
            // answer, read off this run's builder - the Coulomb half is
            // spin-independent, so an unrestricted run's 3c grid reduces
            // exactly as the restricted leg's does.
            result.resourcesResolved.riOrbitExpansion = MakeRiOrbitExpansionRecord(
                input, selection->kind, run->orbitExpansionWired, run->orbitExpansionEngaged);

            // The calibration term_counters block, published on the same
            // terms the RHF leg publishes it: an instrumented run (a trace
            // file set) whose calls carried the per-call stats sink the
            // exchange counts ride. A UHF run that records no rows publishes
            // no block rather than a fabricated zero.
            if (!input.scf.traceFile.empty() && run->riJkBuilder.has_value())
            {
                const qcx::integrals::RiTermCounters& counters = run->riJkBuilder->TermCounters();
                result.termCounters = qcx::io::RunTermCounters{
                    counters.x, counters.p3, counters.g3, counters.qx, counters.gx};
            }

            auto outcome = FillUhfRunOutcome(result,
                                             *molecule,
                                             *basis,
                                             overlap,
                                             *coreTensor,
                                             core,
                                             run->scf,
                                             run->scfLoopMs,
                                             canonicalOrder,
                                             atomicInputs,
                                             input,
                                             workspaceReserveBytes,
                                             computeProfile,
                                             certifiedLaneRequest,
                                             selection->leanMember);

            if (!outcome.has_value())
            {
                return refuse(outcome.error());
            }
        } else if (selection->kind == qcx::io::BuilderKind::kRiJk)
        {
            // The unrestricted composed-full-RI path (the per-spin adapter):
            // the same F_sigma = H + J(P_tot) - K(P_sigma) assembly the direct
            // arm runs, with ONE BuildFockHalves call per spin in place of the
            // direct family's two split calls - and, unlike that family, no
            // per-channel H subtraction (the pair carries no H at all; the
            // seam adds one copy per channel, MakeRiFullUhfFockBuilder's
            // accounting note). The memory model is the family's own
            // (nBasis, nAux) pair, so the audit block records the terms the
            // cap was weighed against - the RHF ri_jk leg's numbers.
            const auto auxName = AuxNameInEffect(input, qcx::io::BuilderKind::kRiJk);

            if (!auxName.has_value())
            {
                return refuse(auxName.error());
            }

            auto auxBasis = qcx::basisset::ParseNwchemDirectoryFiltered(
                (root / *auxName).string(), UniqueAtomicNumbers(*molecule));

            if (!auxBasis.has_value())
            {
                return refuse(auxBasis.error());
            }

            auto run = RunRiJkUhfScf(*molecule,
                                     *basis,
                                     *coreTensor,
                                     overlap,
                                     core,
                                     atomicInputs,
                                     input,
                                     workspaceReserveBytes,
                                     root,
                                     true);

            if (!run.has_value())
            {
                return refuse(run.error());
            }

            result.resourcesResolved.workspaceBudget = MakeRunWorkspaceBudget(run->workspaceBudget);
            // The composed full-RI builder's own Create-time rung decision -
            // the record the restricted leg publishes from its
            // own carrier: read off the run so the block cannot disagree with
            // the build it describes, and absent on a run whose builder was
            // created without a budget (no decision was made).
            result.resourcesResolved.riJkMode = MakeRiJkModeRecord(run->riJkModeInfo);
            // The requested-vs-ran pairing and its two sibling
            // disclosures, filled from the mode info this arm has - which is
            // EMPTY, and that is the honest reading rather than a dropped
            // one: those three records are the RI-J engine's vocabulary
            // (FockModeInfo), and this family publishes its own rung through
            // ri_jk_mode above instead. A named `disk` rung or chunk hint is
            // therefore reported as not engaged, which is the truth - this
            // builder has no disk route and no chunked-arena rung - and never
            // as a demotion this arm performed.
            result.resourcesResolved.riTensorMode =
                MakeRiTensorModeRecord(input, selection->kind, run->modeInfo);
            result.resourcesResolved.riChunkBytes =
                MakeRiChunkBytesRecord(input, selection->kind, run->modeInfo);
            // The orbit-expansion request record: this arm carries
            // a named request into the builder, which refuses the mechanism by
            // name, so a run that reaches here with the key set has already
            // been refused. Nothing engaged and nothing was wired through -
            // stated, not left unmentioned.
            result.resourcesResolved.riOrbitExpansion = MakeRiOrbitExpansionRecord(
                input, selection->kind, run->orbitExpansionWired, run->orbitExpansionEngaged);

            // No term_counters block here, and it is a decision rather than an
            // omission: see the WiredUhfBuilders member that states why this
            // family's counters are not published through the ri_j_link
            // block's vocabulary (its qx/gx name a nested direct exchange this
            // builder does not have), and why the restricted leg of the same
            // family publishes none either.

            auto outcome = FillUhfRunOutcome(result,
                                             *molecule,
                                             *basis,
                                             overlap,
                                             *coreTensor,
                                             core,
                                             run->scf,
                                             run->scfLoopMs,
                                             canonicalOrder,
                                             atomicInputs,
                                             input,
                                             workspaceReserveBytes,
                                             computeProfile,
                                             certifiedLaneRequest,
                                             selection->leanMember);

            if (!outcome.has_value())
            {
                return refuse(outcome.error());
            }
        } else if (selection->kind == qcx::io::BuilderKind::kQfmm)
        {
            // The composed-QFMM UHF path: F_sigma = H + J(P_tot) -
            // K(P_sigma) - the Coulomb half through the QFMM near/far
            // split and the per-spin exchange through the exchange-only
            // direct half, the whole MakeDirectUhfFockBuilder assembly run
            // inside QfmmHfFockBuilder::BuildUhfFock (the O2 UHF pin gate
            // rides this branch). The base term against the cap is the
            // composed kind's (the direct family - the RHF kQfmm branch
            // models the same way), and the budget/mode records assigned
            // below come from the runner: RunQfmmUhfScf grants the shared
            // cap-minus-base budget and exposes both nested halves'
            // Create-time mode records (the QFMM half's here, the
            // exchange half's under exchange_mode_record - the direct UHF
            // branch's surface).
            auto run = RunQfmmUhfScf(*molecule,
                                     *basis,
                                     *coreTensor,
                                     overlap,
                                     core,
                                     atomicInputs,
                                     input,
                                     workspaceReserveBytes);

            if (!run.has_value())
            {
                return refuse(run.error());
            }

            result.resourcesResolved.workspaceBudget = MakeRunWorkspaceBudget(run->workspaceBudget);
            result.resourcesResolved.modeRecord = MakeRunModeRecord(run->modeInfo);
            result.resourcesResolved.exchangeModeRecord = MakeRunModeRecord(run->exchangeModeInfo);
            // The composed-QFMM model this run's Coulomb half ran (schema
            // 29): filled from the engine's own record that the runner
            // carried out (MakeQfmmModelRecord), so a UHF run answers the
            // same question an RHF one does. Absent on the direct-UHF arm,
            // whose runner leaves the carrier empty.
            result.qfmmModel = MakeQfmmModelRecord(run->qfmmModel);
            // The requested-vs-ran pairing: a UHF run has no
            // ri_j_link route in v1, so a `disk` request here is
            // not_applicable and a forced one demotes - both stated rather
            // than vanishing with the missing mode record.
            result.resourcesResolved.riTensorMode =
                MakeRiTensorModeRecord(input, selection->kind, run->modeInfo);
            // The chunk-size hint's own disclosure, filled from
            // the same mode info: a UHF run has no disk rung, so a named
            // method.ri_chunk_bytes is dropped here and the record says so
            // instead of leaving the key unmentioned.
            result.resourcesResolved.riChunkBytes =
                MakeRiChunkBytesRecord(input, selection->kind, run->modeInfo);
            // The orbit-expansion request record: a UHF run has
            // no ri_j_link route in v1, so a named key is not_applicable
            // here - stated, not left unmentioned.
            result.resourcesResolved.riOrbitExpansion =
                MakeRiOrbitExpansionRecord(input, selection->kind, false, false);

            auto outcome = FillUhfRunOutcome(result,
                                             *molecule,
                                             *basis,
                                             overlap,
                                             *coreTensor,
                                             core,
                                             run->scf,
                                             run->scfLoopMs,
                                             canonicalOrder,
                                             atomicInputs,
                                             input,
                                             workspaceReserveBytes,
                                             computeProfile,
                                             certifiedLaneRequest,
                                             selection->leanMember);

            if (!outcome.has_value())
            {
                return refuse(outcome.error());
            }
        } else
        {
            auto run = RunDirectUhfScf(*molecule,
                                       *basis,
                                       *coreTensor,
                                       overlap,
                                       core,
                                       atomicInputs,
                                       input,
                                       workspaceReserveBytes,
                                       computeProfile,
                                       certifiedLaneRequest,
                                       selection->leanMember,
                                       &eriStore,
                                       true);

            if (!run.has_value())
            {
                return refuse(run.error());
            }

            // The adaptive seam's records, wired like the RHF path:
            // the budget's committed bytes and the TWO direct builders'
            // Create-time mode records (the class-table admission gate's
            // observation surface - the term charged or the path
            // disengaged). The UHF Fock builds from both the coulomb and
            // the exchange halves, each admission-gated against the same
            // budget, so the exchange half's record gets its own block
            // (the coulomb half's alone would show only half the
            // evidence).
            result.resourcesResolved.workspaceBudget = MakeRunWorkspaceBudget(run->workspaceBudget);
            result.resourcesResolved.modeRecord = MakeRunModeRecord(run->modeInfo);
            result.resourcesResolved.exchangeModeRecord = MakeRunModeRecord(run->exchangeModeInfo);
            // The requested-vs-ran pairing: a UHF run has no
            // ri_j_link route in v1, so a `disk` request here is
            // not_applicable and a forced one demotes - both stated rather
            // than vanishing with the missing mode record.
            result.resourcesResolved.riTensorMode =
                MakeRiTensorModeRecord(input, selection->kind, run->modeInfo);
            // The chunk-size hint's own disclosure, filled from
            // the same mode info: a UHF run has no disk rung, so a named
            // method.ri_chunk_bytes is dropped here and the record says so
            // instead of leaving the key unmentioned.
            result.resourcesResolved.riChunkBytes =
                MakeRiChunkBytesRecord(input, selection->kind, run->modeInfo);
            // The orbit-expansion request record: a UHF run has
            // no ri_j_link route in v1, so a named key is not_applicable
            // here - stated, not left unmentioned.
            result.resourcesResolved.riOrbitExpansion =
                MakeRiOrbitExpansionRecord(input, selection->kind, false, false);

            // The disk-tier ERI store's own disclosure on the unrestricted
            // legs, the RHF arm's block one scope over and the same rule: the
            // store's consumer sits outside the builder, so the run's own JSON
            // is the only place a reader can learn whether a named disk store
            // was what ran. `engaged` is read from the store's own traffic and
            // never from the install, so the class-aware disengagement and the
            // lean member's missing engine pair each report the demotion they
            // are.
            result.resourcesResolved.eriStore =
                MakeEriStoreRecord(input, selection->kind, selection->leanMember, eriStore);

            auto outcome = FillUhfRunOutcome(result,
                                             *molecule,
                                             *basis,
                                             overlap,
                                             *coreTensor,
                                             core,
                                             run->scf,
                                             run->scfLoopMs,
                                             canonicalOrder,
                                             atomicInputs,
                                             input,
                                             workspaceReserveBytes,
                                             computeProfile,
                                             certifiedLaneRequest,
                                             selection->leanMember);

            if (!outcome.has_value())
            {
                return refuse(outcome.error());
            }
        }
    } else
    {
        // The restart read (guess.type = "restart"): the
        // checkpoint store is loaded before any builder work so a missing
        // file or a foreign store fails the run fast, and the state seeds
        // RhfOptions.initialScfState at the options assembly below - the
        // density replaces the default GWH start and the DIIS
        // history with the previous energies seed the accelerator and the
        // convergence gate, so the run resumes from the saved last
        // iterate instead of re-wandering it. The fingerprint's aux name
        // is the one the run's own wiring puts in effect (AuxNameInEffect
        // - same rule as the save site), which binds the checkpoint to
        // the configuration that produced it: a restart run must
        // replicate the original run's basis/aux choices. The storage
        // load returns per-stored-method channels, so a UHF store (only
        // producible by storage-module clients in v1 - the driver refuses
        // to write one) is refused here by name rather than seeding the
        // RHF loop with an empty density.
        std::optional<qcx::scf::ScfRestartState> restartState;

        if (input.guess == qcx::io::GuessKind::kRestart)
        {
            auto auxName = AuxNameInEffect(input, selection->kind);

            if (!auxName.has_value())
            {
                return refuse(auxName.error());
            }

#if defined(QcxHasStorage)
            auto state = qcx::storage::LoadScfCheckpoint(
                input.guessRestartPath, *molecule, input.basis.orbital, *auxName);

            if (!state.has_value())
            {
                // The storage error's own code travels (kIOError for an
                // unreadable/corrupt file, kInvalidArgument for a foreign
                // store); the prefix names the schema key at fault.
                return refuse(
                    Err(state.error().code, "guess.restart_path: " + state.error().message));
            }

            if (state->density.size() == 0)
            {
                return refuse(Err(qcx::ErrorCode::kInvalidArgument,
                                  "the checkpoint at guess.restart_path stores a UHF "
                                  "state; guess restart seeds the RHF solver in v1"));
            }

            restartState = std::move(*state);
#else
            // No storage module in this build (QCX_ENABLE_IO=OFF): the reader
            // is the module's, and the run cannot fall back to a default start
            // here - "restart" that silently begins from GWH is exactly the
            // substitution this refusal prevents. The file is left untouched.
            return refuse(Err(qcx::ErrorCode::kUnimplemented, StorageModuleAbsentRefusalText()));
#endif
        }

        // The per-call stats stream of the trace side-channel: opened
        // when [scf] trace_file is set; the seam lambdas record one line
        // per main-SCF BuildFock call. The stream outlives the SCF call
        // (the seam captures its address).
        ScfCallStatsStream statsStream;
        statsStream.Open(input.scf.traceFile);

        // The certified-bound collector: the same
        // stable-address contract as the stream, but UNCONDITIONAL - the
        // fp32 lane's accumulated error bound is a run-record quantity,
        // not a trace side-channel, so it is collected whether or not a
        // trace file was requested. The machinery members fold every
        // main-SCF call into it; every other builder kind leaves it
        // empty, and an empty collector serializes as an absent block.
        CertifiedBoundAccumulator boundAccum;

        // The disk-tier ERI store's run-owned wiring (method.eri_cache_store,
        // ): the direct family's machinery arm fills it, every other
        // arm leaves it empty, and the record below reads whichever it is -
        // which is what makes a request the run could not honour a DISCLOSED
        // demotion rather than a substitution.
        EriStoreWiring eriStore;

        auto wired = WireRhfFockBuilder(input,
                                        workspaceReserveBytes,
                                        selection->kind,
                                        *molecule,
                                        *basis,
                                        *coreTensor,
                                        root,
                                        computeProfile,
                                        certifiedLaneRequest,
                                        selection->leanMember,
                                        &statsStream,
                                        &boundAccum,
                                        &eriStore);

        if (!wired.has_value())
        {
            return refuse(wired.error());
        }

        qcx::scf::RhfOptions options;
        options.maxIterations = input.scf.maxIterations;
        options.energyTolerance = input.scf.energyTolerance;
        options.densityTolerance = input.scf.densityTolerance;
        options.useDiis = input.scf.useDiis;
        options.traceFile = input.scf.traceFile;
        options.densityDumpFile = input.scf.densityDumpFile;
        options.fullGroupLabeling = input.symmetry.fullGroup;

        // The RHF guess tier: gwh seeds the loop through the restart
        // seam, and guess restart hands the loaded checkpoint state to the
        // same seed (the load itself happened at the top of this branch,
        // before the builder work). The RHF loop's own default start IS
        // the GWH guess (rhf.cpp), so guess core and guess gwh take the
        // same start; sad stays UHF-only (ValidateCombination rejects it
        // up front).
        if (input.guess == qcx::io::GuessKind::kGwh)
        {
            const int nOcc = molecule->ElectronCount() / 2;
            auto gwh = qcx::scf::BuildGwhGuess(overlap, core, nOcc, nOcc);

            if (!gwh.has_value())
            {
                return refuse(gwh.error());
            }

            // Closed shell: D = dAlpha + dBeta.
            options.initialScfState.density = gwh->first + gwh->second;
        } else if (input.guess == qcx::io::GuessKind::kRestart)
        {
            // The stored density and DIIS history ride the seeded state.
            options.initialScfState = *restartState;
        }

        const auto scfStarted = std::chrono::steady_clock::now();
        // The seam the wiring resolved: a Kohn-Sham run hands the loop three
        // callbacks, because the Hartree-Fock trace identity Tr[D (H + F)]/2
        // does not survive a Fock whose exchange is scaled by the functional's
        // exact-exchange fraction - the loop must be given J[D] separately
        // the Kohn-Sham arm. The presence of the companion callbacks IS the
        // distinction: a Hartree-Fock run's wiring leaves them empty, so this
        // branch cannot be taken by a run that was built as Hartree-Fock, and
        // a Kohn-Sham run cannot reach the one-callback overload. There is no
        // flag that could disagree with the seam that actually runs.
        const bool isKsSeam = wired->coulomb.has_value() && wired->contribution.has_value();
        auto scf = isKsSeam
                       ? qcx::scf::RunRhfScf(*molecule,
                                             overlap,
                                             core,
                                             options,
                                             wired->fn,
                                             *wired->coulomb,
                                             *wired->contribution,
                                             &*basis)
                       : qcx::scf::RunRhfScf(*molecule, overlap, core, options, wired->fn, &*basis);
        const auto scfStopped = std::chrono::steady_clock::now();

        if (!scf.has_value())
        {
            return refuse(scf.error());
        }

        // A traced run that recorded nothing says so (see WarnOnEmptyTrace):
        // the empty .stats file must never be the only evidence that no
        // timing limb came out of the run.
        WarnOnEmptyTrace(input, statsStream, selection->kind);

        FillScfResult(result,
                      scf->converged,
                      scf->iterations,
                      scf->achievedEnergyDelta,
                      scf->achievedRmsDensityChange,
                      scf->totalEnergy,
                      scf->electronicEnergy,
                      std::chrono::duration<double, std::milli>(scfStopped - scfStarted).count(),
                      scf->numRemovedOverlapDirections);

        // The full-group labeling stage: the per-MO labels; absent
        // when the stage did not run (a C1 molecule, the [symmetry]
        // full_group = false switch, or an unrealizable group) - never a
        // fabricated C1 record.
        result.symmetry = MakeRunSymmetry(scf->symmetryLabels);

        // The last-iterate checkpoint write (scf.checkpoint_file,
        // RHF only): the restart state - the final-iterate density, the
        // DIIS history, the previous energies (all filled identically on
        // the converged and the budget-exit paths, rhf.cpp) - is saved
        // whenever the run exits without converging (the budget-exit
        // case, where a later run seeds from the last iterate instead of
        // re-wandering the guess) and, with checkpoint_converged, also on
        // the converged exit. A stale file at the path is removed first -
        // the store is append-only by contract, and a rerun on the same
        // path means a previous attempt's store - and a write failure
        // fails the run: the checkpoint is the deliverable of the
        // non-converged exit. The fingerprint binds the molecule and the
        // orbital/aux basis names in effect (AuxNameInEffect, the same
        // rule the kRestart load applies), so the resume run must
        // replicate this run's configuration.
        if (!input.scf.checkpointFile.empty() && (!scf->converged || input.scf.checkpointConverged))
        {
            auto auxName = AuxNameInEffect(input, selection->kind);

            if (!auxName.has_value())
            {
                return refuse(auxName.error());
            }

#if defined(QcxHasStorage)
            std::error_code removeError;
            std::filesystem::remove(input.scf.checkpointFile, removeError);

            if (removeError)
            {
                return refuse(Err(qcx::ErrorCode::kIOError,
                                  "scf.checkpoint_file: cannot remove the stale file "
                                  "at the path: " +
                                      removeError.message()));
            }

            auto saved = qcx::storage::SaveScfCheckpoint(
                input.scf.checkpointFile, *molecule, input.basis.orbital, *auxName, scf->restart);

            if (!saved.has_value())
            {
                return refuse(saved.error());
            }
#else
            // No storage module in this build (QCX_ENABLE_IO=OFF): the writer is
            // the module's, and the refusal stands AHEAD of the stale-file
            // removal above on purpose - a run that cannot write the checkpoint
            // must not delete the one already at the path on its way to failing.
            return refuse(Err(qcx::ErrorCode::kUnimplemented, StorageModuleAbsentRefusalText()));
#endif
        }

        // The budget-path audit: the cap-minus-base
        // workspace budget granted to the engine, and the engine's
        // Create-time mode record with its firing estimate terms - the
        // evidence (mode = light with the exclusion terms on the big runs).
        // Absent on the legacy null-budget path (cap 0 or a cap at/below
        // the modeled base), where no decision was made.
        result.resourcesResolved.workspaceBudget = MakeRunWorkspaceBudget(wired->workspaceBudget);
        result.resourcesResolved.modeRecord =
            MakeRunModeRecord(wired->modeInfo, wired->forcedDiskRung);

        // The ri_jk builder's own rung decision: the RI-K
        // counterpart of the line above, read from the Create-time record the
        // wiring carried out - never recomputed here, so the block cannot
        // disagree with the build it describes. Absent on every other family
        // and on a ri_jk run created without a budget.
        result.resourcesResolved.riJkMode = MakeRiJkModeRecord(wired->riJkModeInfo);

        // The composed-QFMM model this run's Coulomb half ran:
        // empty on every family that wired no such builder, which is what
        // leaves the block absent there rather than defaulted.
        result.qfmmModel = MakeQfmmModelRecord(wired->qfmmModel);

        // The Kohn-Sham machinery run's second half (see the WiredFockBuilder
        // member): absent everywhere else, exactly as the UHF path leaves it
        // absent for its own non-machinery arms.
        result.resourcesResolved.exchangeModeRecord = MakeRunModeRecord(wired->exchangeModeInfo);

        // The requested-vs-ran pairing: the one record that
        // states what the input asked of the RI tensor mode and what ran,
        // on this budget-path run and on the paths with no mode record
        // alike (the lean arm, the null-budget path) - the request leaves a
        // trace wherever it was named.
        result.resourcesResolved.riTensorMode =
            MakeRiTensorModeRecord(input, selection->kind, wired->modeInfo);

        // The chunk-size hint's own disclosure: the same one
        // record for method.ri_chunk_bytes, read off the same mode info. The
        // hint has exactly one consumer (the disk-rung lambda that built this
        // run's wired builder, when it was the disk route), so on the paths
        // that never reach it - every family with no disk rung, and the
        // ri_j_link ladder when an in-memory rung fitted - a named key is
        // dropped, and this is where the run says so.
        result.resourcesResolved.riChunkBytes =
            MakeRiChunkBytesRecord(input, selection->kind, wired->modeInfo);

        // The orbit-expansion request record: the same shape for
        // method.ri_orbit_expansion, whose outcome is the engine's own answer
        // read off this run's builder - the request is written whenever the
        // input named the key, whatever the family, so a key consumed one
        // branch deep leaves a trace on the paths that never reach it.
        result.resourcesResolved.riOrbitExpansion = MakeRiOrbitExpansionRecord(
            input, selection->kind, wired->orbitExpansionWired, wired->orbitExpansionEngaged);

        // The disk-tier ERI store's own disclosure: the same
        // shape for method.eri_cache_store, whose consumer sits outside the
        // builder entirely (the driver constructs the decorator, owns it and
        // reads its stats - the module boundary does not let `integrals` name `storage`). The
        // request is written whenever the input named a path, whatever the
        // family and whatever the store did, because a request that is
        // accepted and then dropped is the one run record that can differ
        // from what ran.
        result.resourcesResolved.eriStore =
            MakeEriStoreRecord(input, selection->kind, selection->leanMember, eriStore);

        // The calibration term_counters block (the RI-J emission path,
        // ri_engine.hpp RiTermCounters): the engine's per-run totals,
        // published exactly when the run was instrumented - the ri_j link
        // with a trace file set (every main-SCF BuildFock call then carried
        // the per-call stats sink the exchange counts ride, so the snapshot
        // is complete: all five ids, zero = the term never occurred).
        if (selection->kind == qcx::io::BuilderKind::kRiJLink && !input.scf.traceFile.empty() &&
            wired->riJkBuilder.has_value())
        {
            const qcx::integrals::RiTermCounters& counters = wired->riJkBuilder->TermCounters();
            result.termCounters = qcx::io::RunTermCounters{
                counters.x, counters.p3, counters.g3, counters.qx, counters.gx};
        }

        // The certified mixed-precision bound block: the
        // fp32 lane's accumulated density-weighted kernel-bound sum, the
        // quantity the seam's accumulator collected from every main-SCF
        // BuildFock call. Published exactly when at least one call
        // reported it - the direct family's machinery members, whose
        // BuildFock exposes the certifiedBoundSumOut out-parameter, and
        // the seam folds into the accumulator from INSIDE the `requires`
        // branch that passes that pointer. Every other builder kind (the
        // lean member's Schwarz-only builder, the RI-J/QFMM/GPU families,
        // the UHF branch) takes a shorter BuildFock and never reaches the
        // fold, leaving the block ABSENT: those paths have no such
        // quantity, and a fabricated 0.0 would read as "certified with
        // zero error" - on the RI-J link it would additionally deny the
        // bound its nested fp32 exchange half computes and drops. A
        // present block that reads 0.0 is the lane's true zero - the
        // kTight gate (MixedPrecisionThreshold = 0.0) routes no quartet,
        // so the lane contributed exactly nothing.
        //
        //  adds the global budget enforcement's members to the
        // same block (`enforced`, `budget_ha`, `routed_ha`,
        // `routed_quartets`, `fell_back_to_fp64`), folded from the same
        // calls: when the run asked for the enforcement
        // ([method] enforce_certified_bound) they say what the routed
        // bound sum was, what budget the preset derived for it, and
        // whether the comparison sent the build to the fp64 lane. A
        // fall-back is the one reading a consumer must not mistake for a
        // quiet zero: `routed_ha` exceeds `budget_ha` and the lane
        // delivered nothing BECAUSE of it.
        if (boundAccum.Calls() > 0)
        {
            result.certifiedBound = qcx::io::RunCertifiedBound{boundAccum.Calls(),
                                                               boundAccum.LastCallHa(),
                                                               boundAccum.MaxCallHa(),
                                                               boundAccum.Enforced(),
                                                               boundAccum.BudgetHa(),
                                                               boundAccum.RoutedHa(),
                                                               boundAccum.RoutedQuartets(),
                                                               boundAccum.FellBackToFp64()};
        }

        // The density handed over: HfResult::density is the spin-summed
        // D = 2 rho;
        // the properties module takes per-spin densities, equal for the
        // closed shell.
        auto properties = FillProperties(result,
                                         *molecule,
                                         *basis,
                                         scf->density / 2.0,
                                         scf->density / 2.0,
                                         overlap,
                                         *coreTensor,
                                         core,
                                         canonicalOrder,
                                         atomicInputs,
                                         input,
                                         workspaceReserveBytes,
                                         computeProfile,
                                         certifiedLaneRequest,
                                         selection->leanMember);

        if (!properties.has_value())
        {
            return refuse(properties.error());
        }

        // The Molden export of the converged or
        // last-iterate result; one closed-shell channel with aufbau
        // occupations of 2.0.
        const int nOcc = molecule->ElectronCount() / 2;
        const auto occupations =
            AufbauOccupations(static_cast<std::size_t>(nOcc),
                              2.0,
                              static_cast<std::size_t>(scf->coefficients.cols()));
        const std::array<qcx::io::MoldenMolecularOrbitals, 1> moBlocks{
            qcx::io::MoldenMolecularOrbitals{
                "Alpha", scf->coefficients, scf->orbitalEnergies, occupations}};

        auto molden = WriteMoldenExport(result, *molecule, *basis, moBlocks, input);

        if (!molden.has_value())
        {
            return refuse(molden.error());
        }
    }

    const auto stopped = std::chrono::steady_clock::now();
    result.timingsMs.totalMs = std::chrono::duration<double, std::milli>(stopped - started).count();

    // The outcome the caller reads: the record, its JSON document, and the
    // two facts the record does not hold - the molecule-scoped basis-function
    // count the size ladder resolved, and the atom permutation the
    // canonicalization produced (the per-atom result vectors are indexed in
    // the molecule's canonical order, so a report that paired the input's
    // rows with them directly would print one atom's charge beside another
    // atom's symbol).
    RunOutcome runOutcome;
    runOutcome.json = qcx::io::SerializeRunResultJson(result);
    runOutcome.result = std::move(result);
    runOutcome.reportFacts.basisFunctionCount = nBasis;
    runOutcome.reportFacts.canonicalAtomOrder = canonicalOrder;

    return runOutcome;
}

qcx::Result<std::string> RunDriver(const qcx::io::RunInput& input) {
    auto outcome = RunDriverOutcome(input);

    if (!outcome.has_value())
    {
        return std::unexpected(std::move(outcome.error()));
    }

    return std::move(outcome->json);
}

namespace {

// Counts the shells of one element as seen by the engine (the parsed set
// carries only the elements the parse directory provides; the engine
// counts molecule-scoped shells the same way, so a missing element
// contributes zero on both sides).
std::size_t ShellCountPerElement(const qcx::basisset::BasisSet& basis, int atomicNumber) noexcept {
    const qcx::basisset::ElementBasis* element = basis.Find(atomicNumber);

    if (element == nullptr)
    {
        return 0;
    }

    return element->shells.size();
}

double CountPerAtom(const qcx::basisset::BasisSet& basis, int atomicNumber) noexcept {
    const qcx::basisset::ElementBasis* element = basis.Find(atomicNumber);

    if (element == nullptr)
    {
        // The parsed set carries only the elements the parse directory
        // provides; the engine counts molecule-scoped functions the same
        // way, so a missing element contributes nothing on both sides.
        return 0.0;
    }

    double count = 0.0;

    for (const qcx::basisset::Shell& shell : element->shells)
    {
        const int l = shell.angularMomentum;
        count += shell.isSpherical ? static_cast<double>(2 * l + 1)
                                   : static_cast<double>((l + 1) * (l + 2)) / 2.0;
    }

    return count;
}

} // namespace

std::size_t CountBasisFunctions(const qcx::molecule::Molecule& molecule,
                                const qcx::basisset::BasisSet& basis) noexcept {
    double count = 0.0;

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        count += CountPerAtom(basis, atom.atomicNumber);
    }

    return static_cast<std::size_t>(count);
}

std::size_t CountShellPairs(const qcx::molecule::Molecule& molecule,
                            const qcx::basisset::BasisSet& basis) noexcept {
    std::size_t nShells = 0;

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        nShells += ShellCountPerElement(basis, atom.atomicNumber);
    }

    return nShells * (nShells + 1) / 2;
}

// The standing reinstatement clause of the ri_j budget path (the budget path): the
// four options the driver's engine-composed refusals name, the light rung
// (landed) in the recompute slot and the disk-backed store always
// last. Composed onto the engine's own Create-time refusal texts
// (AM-owned, never edited) verbatim.
std::string RiJReinstatementClause() {
    return " Reinstatement options: (1) raise memory_cap_gib in [resources]; "
           "(2) the per-iteration recompute path - the RI-J light rung, "
           "landed; the engine's Create-time estimate runs it when the fast "
           "path's tensor does not fit the workspace budget the cap grants); "
           "(3) a tighter accuracy preset (the certified fp32 lane); "
           "(4) the disk-backed tensor store, engaged with "
           "method.ri_tensor_mode = \"disk\" (that knob, landed; the disk "
           "rung runs as the ladder's LAST rung "
           "after this refusal - disk never a first choice).";
}

// The standing reinstatement clause of the direct-family budget path:
// the cap raise, the tighter accuracy preset, and the disk-backed
// store LAST - the direct ladder, without the ri_j tensor rungs. Composed
// onto the engine's own Create-time refusal texts verbatim (AM-owned,
// never edited).
std::string DirectReinstatementClause() {
    return " Reinstatement options: (1) raise memory_cap_gib in [resources]; "
           "(2) a tighter accuracy preset (the certified fp32 lane); "
           "(3) the disk-backed store (assessed, not "
           "implemented).";
}

// The NOCV dense-tensor charge of the analysis phase, in GiB: the
// a-priori worst-case bytes of the dense ERI tensor build (the
// integrals-side DenseEriTensorEstimateBytes - the canonical quartet list
// + the batch payload + the 8 n^4 tensor, the same formula the engine's
// admission gate applies). The charge is the analysis-phase peak: the
// NOCV block builds the tensor after the SCF completes, so the SCF-phase
// working set does not overlap the build, and the fragment SCFs that
// follow are subset-sized. This is an ENGINE estimate, not the predictive
// model the driver carried until 2026-09-17: it survives the model's
// deletion because the analysis it gates needs the dense tensor and the
// estimate is the engine's own.
double NocvDenseTensorChargeGiB(std::size_t nBasis, std::size_t nPairs) noexcept {
    return static_cast<double>(qcx::integrals::DenseEriTensorEstimateBytes(nBasis, nPairs)) / kGiB;
}

// The standing reinstatement clause of the NOCV dense-tensor charge: the
// ETS-NOCV analysis is the dense tensor's only consumer, so the ladder
// names the cap raise, the opt-in block's removal, and a smaller basis
// (the estimate grows as n^4) - no light rung or disk-backed store exists
// for the analysis itself. Composed onto the engine's admission-gate
// refusal texts verbatim (AM-owned, never edited).
std::string NocvDenseTensorReinstatementClause() {
    return " Reinstatement options: (1) raise memory_cap_gib in [resources]; "
           "(2) remove the nocv_fragments block from [properties] (the "
           "ETS-NOCV analysis is opt-in - the rest of the run needs no dense "
           "tensor); (3) a smaller orbital basis (the estimate grows as "
           "n^4).";
}

// The refusal message of the NOCV dense-tensor charge: the analysis phase
// (the a-priori engine estimate) does not fit the resource cap. Carries
// the standing NOCV reinstatement clause (the tokens memory_cap_gib /
// nocv_fragments present).
std::string NocvDenseTensorRefusalMessage(std::size_t nBasis,
                                          std::size_t nPairs,
                                          double memoryCapGiB) {
    return "the NOCV fragment analysis needs the dense ERI tensor, which estimates " +
           std::to_string(qcx::integrals::DenseEriTensorEstimateBytes(nBasis, nPairs)) +
           " bytes (the canonical quartet list + the batch payload + the 8 n^4 tensor at "
           "n = " +
           std::to_string(nBasis) + ", nPairs = " + std::to_string(nPairs) +
           ") but the resource cap is " + std::to_string(memoryCapGiB) +
           " GiB - the analysis-phase charge exceeds the cap." +
           NocvDenseTensorReinstatementClause();
}

// The NOCV dense-tensor seam check: refuses the run when the analysis-phase
// charge does not fit the cap (the a-priori estimate exceeds it), before
// the driver calls the builder - never a raw allocation failure under the
// hard cap. A zero cap (memory_cap_gib = 0, the escape hatch) never
// refuses here: the integrals-side admission cap (maxTensorBytes, 2 GiB
// default) stays the backstop, exactly like the budget path's zero-cap
// semantics.
qcx::Result<void> CheckNocvDenseTensorFit(std::size_t nBasis,
                                          std::size_t nPairs,
                                          double memoryCapGiB) {
    if (memoryCapGiB <= 0.0)
    {
        return {};
    }

    const std::size_t estimateBytes = qcx::integrals::DenseEriTensorEstimateBytes(nBasis, nPairs);

    if (estimateBytes <= static_cast<std::size_t>(memoryCapGiB * kGiB))
    {
        return {};
    }

    return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                      NocvDenseTensorRefusalMessage(nBasis, nPairs, memoryCapGiB)});
}
} // namespace qcx::driver
