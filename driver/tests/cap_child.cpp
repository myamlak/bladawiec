// The small-cap test child (the team_size_test pattern: a
// dedicated binary, because a memory cap applies to the whole process — a
// test that capped its own host test binary would kill the suite).
//
// Usage: qcx-driver-cap-child <memory_cap_gib> <marker_file> [fixture]
//   fixture: "direct" (default), "ri_j", "ri_j_disk", "water_cluster",
//            "nocv_dodecane", "lean_above_ceiling" or "cap_verdict"
//
// Runs the fixture through the real path (ParseRunInput + RunDriver) with
// the given [resources] memory_cap_gib and writes "completed" to the
// marker file only when the run finished (the mode_record's admission-gate
// key-value pairs follow on the budget path - the class table charged or
// the path disengaged - scanned across the whole serialized document, and
// the RI tensor-mode request pairing follows on any fixture that named a
// request key, scanned inside its own block).
// The "lean_above_ceiling" fixture is the one multi-leg fixture (below):
// it writes one "<leg>: completed" or "<leg>: refused: <message>" line per
// leg, and its exit code is 1 when any leg refused.
// A cap smaller than the run's own footprint terminates the child at the
// cap — the marker file is then never written. Exit codes:
// 0 = the run completed (the JSON went to stdout), 1 = the run refused
// (the marker carries "refused: <message>"), 2 = bad arguments.

#include "qcx/driver/process_caps.hpp"
#include "qcx/driver/run_driver.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The C_nH_{2n+2} alkane geometry of the driver test fixture
// (tests/fixtures/alkane_sto3g.cpp, MakeAlkaneSto3g), in Bohr: the planar
// trans zigzag backbone at the tetrahedral C-C-C angle, the exact
// tetrahedral 3-fan on the terminal carbons and the exact tetrahedral pair
// (out of the backbone plane) on the internal ones. Reproduced here
// because this child links only qcx-driver and cannot call the fixture
// helper; the constants, the loop order and the arithmetic are the
// fixture's, so the TOML built from this geometry is the test suite's own
// text. Atom order is carbon-major, as the fixture builds it. The chain
// branch only (carbonCount >= 2) - the shape every caller here uses.
std::vector<std::array<double, 3>> AlkaneCoordinates(std::size_t carbonCount) {
    constexpr double kBohrPerAngstrom = 1.889726125457828;
    const double tetrahedral = std::acos(-1.0 / 3.0);
    const double outOfPlane = std::sqrt(2.0 / 3.0);
    const double cc = 2.9066; // C-C 1.538 A, Bohr.
    const double ch = 1.09 * kBohrPerAngstrom; // C-H 1.09 A, Bohr.
    const double stepX = cc * outOfPlane;
    const double stepY = cc / std::sqrt(3.0);

    std::vector<std::array<double, 3>> coordinates(carbonCount);

    for (std::size_t i = 0; i < carbonCount; ++i)
    {
        coordinates[i] = {static_cast<double>(i) * stepX, (i % 2 == 1) ? stepY : 0.0, 0.0};
    }

    // One hydrogen row per placement, in the fixture's order: the H sits
    // at its carbon's (x, y) plus the offset, and the offset's z is the
    // row's z (the backbone is planar, z = 0).
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const auto placeHydrogen = [&](std::size_t carbon, double x, double y, double z) {
        coordinates.push_back({coordinates[carbon][0] + x, coordinates[carbon][1] + y, z});
    };

    // The three-H fan at the tetrahedral angle from the bond direction
    // (ux, uy), spread at 120 degrees around it in the plane perpendicular
    // to the bond (w1 in the xy plane, w2 = z).
    const auto placeTerminalFan = [&](std::size_t carbon, double ux, double uy) {
        const double w1x = -uy;
        const double w1y = ux;

        for (std::size_t k = 0; k < 3; ++k)
        {
            const double phi = 2.0 * std::numbers::pi * static_cast<double>(k) / 3.0;
            placeHydrogen(
                carbon,
                ch * (std::cos(tetrahedral) * ux + std::sin(tetrahedral) * std::cos(phi) * w1x),
                ch * (std::cos(tetrahedral) * uy + std::sin(tetrahedral) * std::cos(phi) * w1y),
                ch * std::sin(tetrahedral) * std::sin(phi));
        }
    };

    placeTerminalFan(0, stepX / cc, stepY / cc);

    // Internal carbons: the exact tetrahedral pair, h = -(a + c)/2 +/-
    // sqrt(2/3) * zhat for the unit bond directions a and c from the
    // carbon (the backbone angle IS the tetrahedral angle, so
    // |a + c|^2 = 4/3 and h.a = h.c = cos(tau)). The two H's mirror across
    // the backbone plane.
    for (std::size_t i = 1; i + 1 < carbonCount; ++i)
    {
        const double ax = (coordinates[i - 1][0] - coordinates[i][0]) / cc;
        const double ay = (coordinates[i - 1][1] - coordinates[i][1]) / cc;
        const double cx = (coordinates[i + 1][0] - coordinates[i][0]) / cc;
        const double cy = (coordinates[i + 1][1] - coordinates[i][1]) / cc;

        for (const double sign : {1.0, -1.0})
        {
            placeHydrogen(
                i, ch * (-0.5 * (ax + cx)), ch * (-0.5 * (ay + cy)), ch * sign * outOfPlane);
        }
    }

    // The terminal carbon's fan, around the last bond.
    const std::size_t last = carbonCount - 1;
    placeTerminalFan(last,
                     (coordinates[last - 1][0] - coordinates[last][0]) / cc,
                     (coordinates[last - 1][1] - coordinates[last][1]) / cc);

    return coordinates;
}

// The above-ceiling TOML: C50H102 (the large C50H102 molecule)
// with the def2-SVP orbital basis, 24n + 10 = 1210 contracted functions -
// the smallest above-ceiling molecule this suite can build. The [method]
// fock_builder line carries \p builderWord ("lean" or "direct") or is
// absent when it is null (the auto key); the SCF block carries the
// OPERATING defaults (1e-8 / 1e-6, the 2026-09-04 threshold rule - no
// fixture runs at 1e-10).
std::string T2ChainToml(double capGib, const char* builderWord) {
    const std::size_t kCarbonCount = 50;
    const std::vector<std::array<double, 3>> bohr = AlkaneCoordinates(kCarbonCount);
    std::ostringstream text;
    text << "[molecule]\ncharge = 0\nmultiplicity = 1\natoms = [\n";

    for (std::size_t atom = 0; atom < bohr.size(); ++atom)
    {
        text << "    [\"" << (atom < kCarbonCount ? "C" : "H") << "\", " << std::setprecision(17)
             << bohr[atom][0] / qcx::molecule::kAngstromToBohr << ", "
             << bohr[atom][1] / qcx::molecule::kAngstromToBohr << ", "
             << bohr[atom][2] / qcx::molecule::kAngstromToBohr << "],\n";
    }

    text << "]\n\n[basis]\norbital = \"def2-svp\"\n\n[method]\ntype = \"rhf\"\n";

    if (builderWord != nullptr)
    {
        text << "fock_builder = \"" << builderWord << "\"\n";
    }

    text << "accuracy = \"kNormal\"\n\n[scf]\nmax_iterations = 100\n"
            "energy_tolerance = 1e-8\ndensity_tolerance = 1e-6\n\n[resources]\n"
            "memory_cap_gib = "
         << capGib << "\n";
    return text.str();
}

// The above-ceiling fixture (1210 basis functions): above the
// 1000-function LEAN boundary the ABSENT key resolves to the ri_j_link tier
// under the 2026-09-13 ladder (def2-SVP carries a bundled universal-J aux,
// so the tier is runnable and no demotion fires), and only the explicit
// fock_builder = "lean" word reaches the direct family's within-family lean
// member - each refusal NAMES the family its spelling reached, so the pair
// IS the above-1000 selector pin.
// THREE legs run in ONE child under ONE cap: same binary, same baseline,
// same job object, so the readings are comparable. The explicit "direct"
// leg is the third spelling of this size (Ruling B: explicit direct = the
// machinery at every size), and it is what made the pre-gate setup crash a
// ramp property rather than a selector one - it died 0xC0000409 at the
// same caps the lean leg did. The cap is the admission probe: far below
// both members' own Create-time requirements, which each refusal prints
// (the lean builder's required_peak_bytes, the machinery's modeled base),
// so each run refuses at the wiring - and since the pre-gate setup
// admission landed it also refuses BELOW the ramp's own footprint, where
// the process used to die before any decision. A looser cap admits the
// lean builder into a real SCF.
//
// The absent-key leg is also the leg that MEASURED the ri_j family ramp gap
// the ladder opened: until the family joined the pre-gate setup check,
// this leg - and only this one, the two explicit legs refused cleanly - died
// 0xC0000409 at the 0.5 GiB cap, inside the ramp, with no admission reached.
int RunLeanAboveCeiling(const char* markerPath, double capGib) {
    struct Leg {
        const char* name;
        const char* builderWord; ///< null = the key is absent (the auto rule).
    };

    const Leg legs[] = {
        {"explicit_lean", "lean"}, {"explicit_direct", "direct"}, {"absent_key", nullptr}};
    std::ofstream marker(markerPath);
    int exitCode = 0;

    // One leg's line, flushed as it is written. A later leg can be KILLED at
    // the cap: the job object terminates the process without unwinding, so
    // the stream's destructor never runs, and the legs already decided are
    // exactly the evidence that death destroys. Reachable since the
    // 2026-09-17 owner ruling, which sends an over-cap leg PAST the pre-gate
    // admission - the leg that now proceeds is the one the cap can kill, and
    // an unflushed marker then reports an empty file for a child whose first
    // legs refused cleanly.
    const auto recordLeg = [&marker](const std::string& line) {
        marker << line << '\n';
        marker.flush();
    };

    for (const Leg& leg : legs)
    {
        auto input = qcx::io::ParseRunInput(T2ChainToml(capGib, leg.builderWord));

        if (!input.has_value())
        {
            std::fprintf(stderr, "%s: parse failed: %s\n", leg.name, input.error().message.c_str());
            recordLeg(std::string(leg.name) + ": parse failed: " + input.error().message);
            exitCode = 1;
            continue;
        }

        auto result = qcx::driver::RunDriver(*input);

        if (!result.has_value())
        {
            std::fprintf(stderr, "%s: run refused: %s\n", leg.name, result.error().message.c_str());
            recordLeg(std::string(leg.name) + ": refused: " + result.error().message);
            exitCode = 1;
            continue;
        }

        std::puts(result->c_str());
        recordLeg(std::string(leg.name) + ": completed");
    }

    return exitCode;
}

// The process-cap verdict probe (the 2026-09-17 owner ruling's capped-child
// pin): applies the cap the caller gave it and records what the memory
// model's seam decides UNDER it - no molecule, no SCF, because the verdict
// is about the cap in force and not about a run finishing (and because the
// cap this probe applies is one it must survive).
//
// A child fixture, not a host test: ApplyProcessCaps is process-wide and its
// job handles are process-lifetime, so a test that applied a cap in the test
// binary would cap every later test in that binary.
int RunCapVerdictProbe(const char* markerPath, double capGib) {
    std::ofstream marker(markerPath);

    qcx::io::RunResourcesInput resources{};
    resources.memoryCapGiB = capGib;

    const auto applied = qcx::driver::ApplyProcessCaps(resources);

    if (!applied.has_value())
    {
        marker << "applied: error\nappliedMessage: " << applied.error().message << "\n";
        return 1;
    }

    if (!applied->inProcessCapApplied)
    {
        marker << "applied: noop\nappliedNote: " << applied->note.value_or("") << "\n";
        return 1;
    }

    marker << "applied: bound\n";
    marker << "enforcedAtCap: " << (qcx::driver::ProcessMemoryCapEnforced(capGib) ? 1 : 0) << "\n";
    // The conservative direction, pinned rather than assumed: a cap TIGHTER
    // than the one in force reads false, so the seam keeps its refusal
    // instead of retiring it on a limit the query cannot see the whole of.
    marker << "enforcedAtTighter: "
           << (qcx::driver::ProcessMemoryCapEnforced(capGib / 10.0) ? 1 : 0) << "\n";

    // The advisory verdict the memory model used to be asked for here was
    // deleted with the model on 2026-09-17: the cap IS the admission now,
    // and nothing else has a verdict to record. What this probe still
    // measures is the cap machinery itself, above.

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4)
    {
        std::fprintf(stderr,
                     "usage: qcx-driver-cap-child <memory_cap_gib> <marker_file> [fixture]\n");
        return 2;
    }

    const double capGib = std::atof(argv[1]);
    const std::string fixture = argc == 4 ? argv[3] : "direct";

    // The multi-leg fixture runs its own texts and writes its own marker.
    if (fixture == "lean_above_ceiling")
    {
        return RunLeanAboveCeiling(argv[2], capGib);
    }

    // The cap-verdict probe: applies the cap and records the cap-reading
    // seam's verdict under it, with no run behind it (the memory model
    // that shared this probe until 2026-09-17 is deleted).
    if (fixture == "cap_verdict")
    {
        return RunCapVerdictProbe(argv[2], capGib);
    }

    std::string toml;

    if (fixture == "ri_j")
    {
        // The ri_j fixture: H2O def2-svp (n = 24, the measured base point),
        // auto-selected universal-J aux. The driver's budget-path refusal
        // (cap not clearing the base) lands before the engine sees the f
        // shells, so no SupportsL guard is needed.
        toml = "[molecule]\n"
               "charge = 0\n"
               "multiplicity = 1\n"
               "atoms = [\n"
               "    [\"O\", 0.0, 0.0, 0.0],\n"
               "    [\"H\", 0.7569503270127429, 0.58588227657553, 0.0],\n"
               "    [\"H\", -0.7569503270127429, 0.58588227657553, 0.0],\n"
               "]\n"
               "[basis]\n"
               "orbital = \"def2-svp\"\n"
               "[method]\n"
               "type = \"rhf\"\n"
               "fock_builder = \"ri_j_link\"\n"
               "accuracy = \"kLoose\"\n"
               "[resources]\n"
               "memory_cap_gib = " +
               std::to_string(capGib) + "\n";
    } else if (fixture == "ri_jk")
    {
        // The composed full-RI (ri_jk) fixture: H2O def2-svp with the
        // auto-selected JK fit, the same input the in-process pins run. It is
        // NOT the rung-selection fixture: the band here sits below this
        // child's real commit, so no cap inside it completed (measured:
        // 0.0575-0.0600 GiB, 150 s each, while a cap above the band finished
        // in 2 s) - the rung cell uses ri_jk_propane below, whose band clears
        // the footprint.
        //
        // The 0.045 GiB cap the pins below use is UNDER the modeled base
        // (0.056 GiB at n = 24), so the pre-2026-09-17 driver refused here
        // before any builder work. Since the owner ruling of that date the
        // driver no longer refuses on that prediction: the leg proceeds on the
        // legacy null-budget path with the job-object cap as the only limit.
        // MEASURED at that cap: the leg was still running after 16 minutes at
        // 43.4 MB private against a 46.0 MB cap - it fits, and it is slow -
        // so the refusal a pin may still assert here is the pre-ruling one.
        toml = "[molecule]\n"
               "charge = 0\n"
               "multiplicity = 1\n"
               "atoms = [\n"
               "    [\"O\", 0.0, 0.0, 0.0],\n"
               "    [\"H\", 0.7569503270127429, 0.58588227657553, 0.0],\n"
               "    [\"H\", -0.7569503270127429, 0.58588227657553, 0.0],\n"
               "]\n"
               "[basis]\n"
               "orbital = \"def2-svp\"\n"
               "[method]\n"
               "type = \"rhf\"\n"
               "fock_builder = \"ri_jk\"\n"
               "accuracy = \"kLoose\"\n"
               "[resources]\n"
               "memory_cap_gib = " +
               std::to_string(capGib) + "\n";
    } else if (fixture == "ri_jk_uhf")
    {
        // The ri_jk fixture's UNRESTRICTED mirror (this wiring's runner): the
        // same molecule, basis, cap arithmetic and refusal, one method word
        // apart. The budget path is a COPY in each leg (the runner computes
        // its own grant and composes its own refusal), so the restricted
        // child fixture above cannot stand for this one: a UHF arm that
        // dropped the refusal would run under a cap the model says it does
        // not fit, and nothing else in the suite reaches that branch.
        toml = "[molecule]\n"
               "charge = 0\n"
               "multiplicity = 1\n"
               "atoms = [\n"
               "    [\"O\", 0.0, 0.0, 0.0],\n"
               "    [\"H\", 0.7569503270127429, 0.58588227657553, 0.0],\n"
               "    [\"H\", -0.7569503270127429, 0.58588227657553, 0.0],\n"
               "]\n"
               "[basis]\n"
               "orbital = \"def2-svp\"\n"
               "[method]\n"
               "type = \"uhf\"\n"
               "fock_builder = \"ri_jk\"\n"
               "accuracy = \"kLoose\"\n"
               "[resources]\n"
               "memory_cap_gib = " +
               std::to_string(capGib) + "\n";
    } else if (fixture == "ri_jk_propane")
    {
        // The RI-K rung-selection fixture: propane (C3H8) at def2-svp with
        // the auto-selected JK fit. Propane rather than water because the
        // RI-K ladder's band - the cap range where the grant (cap minus the
        // modeled base) is too small for the FAST rung's class and large
        // enough for the BLOCKED one - has to sit ABOVE this child's real
        // footprint to be observable at all: at n = 24 the model's base term
        // (0.056 GiB) is smaller than the run's real commit, so the band
        // lands under it and every band cap thrashes the job object. At
        // n = 82 the modeled base carries the direct family's screened-
        // quartet working set - which this path never allocates - and the
        // band clears the footprint with room to spare. The geometry rows
        // are MakeAlkaneSto3g(3) (tests/fixtures/alkane_sto3g.cpp, Angstrom),
        // reproduced as literals because this binary links qcx-driver only.
        toml = "[molecule]\n"
               "charge = 0\n"
               "multiplicity = 1\n"
               "atoms = [\n"
               "    [\"C\", 0, 0, 0],\n"
               "    [\"C\", 1.2558586824583173, 0.88802619057827925, 0],\n"
               "    [\"C\", 2.5117173649166347, 0, 0],\n"
               "    [\"H\", -0.88998127321122145, 0.62931179341669208, 0],\n"
               "    [\"H\", -2.4202861936828414e-16, -0.62931179341669174, "
               "0.88998127321122145],\n"
               "    [\"H\", 1.815214645262131e-16, -0.62931179341669252, "
               "-0.88998127321122122],\n"
               "    [\"H\", 1.2558586824583173, 1.5173379839949712, 0.88998127321122145],\n"
               "    [\"H\", 1.2558586824583173, 1.5173379839949712, -0.88998127321122145],\n"
               "    [\"H\", 2.2150569405128944, -1.0488529890278202, 0],\n"
               "    [\"H\", 3.1050382137241157, 0.20977059780556379, 0.88998127321122145],\n"
               "    [\"H\", 3.1050382137241161, 0.20977059780556442, -0.88998127321122122],\n"
               "]\n"
               "[basis]\n"
               "orbital = \"def2-svp\"\n"
               "[method]\n"
               "type = \"rhf\"\n"
               "fock_builder = \"ri_jk\"\n"
               "accuracy = \"kLoose\"\n"
               "[resources]\n"
               "memory_cap_gib = " +
               std::to_string(capGib) + "\n";
    } else if (fixture == "ri_j_disk")
    {
        // The disk-rung fixture: H2O/STO-3G with the
        // explicit universal-J aux (STO-3G has no auto-selection rule), the
        // ri_tensor_mode = "disk" knob, and the tight SCF block of the
        // storage pin runs. The cap is argv-sized: below base + the light
        // rung's estimate the in-memory ladder refuses at the engine's
        // Create and the disk rung must engage and complete the SCF at the
        // RI-J pin -74.96369193 (disk_ri_fock_build_test.cpp mirror).
        toml = "[molecule]\n"
               "charge = 0\n"
               "multiplicity = 1\n"
               "atoms = [\n"
               "    [\"O\", 0.0, 0.0, 0.0],\n"
               "    [\"H\", 0.7569503270127429, 0.58588227657553, 0.0],\n"
               "    [\"H\", -0.7569503270127429, 0.58588227657553, 0.0],\n"
               "]\n"
               "[basis]\n"
               "orbital = \"sto-3g\"\n"
               "aux = \"def2-universal-jfit\"\n"
               "[method]\n"
               "type = \"rhf\"\n"
               "fock_builder = \"ri_j_link\"\n"
               "accuracy = \"kTight\"\n"
               "ri_tensor_mode = \"disk\"\n"
               // Measured 2026-09-16 and VESTIGIAL: the operating default was run
               // against the committed 1e-10/1e-10 pair on this deck shape (the
               // ri_j_link + disk + aux fixture, here under the argv-sized cap) and
               // every fact the parent cell asserts is unchanged: kDisk,
               // disk_bytes 27832, converged true, and energy
               // -74.96369192536702 at BOTH gates - 4.6e-9 from the pin, against
               // the 1e-5 band it is compared at (run_driver_test.cpp). The tight
               // block was the storage pin's copy, not the child's requirement.
               "[scf]\n"
               "max_iterations = 100\n"
               "energy_tolerance = 1e-8\n"
               "density_tolerance = 1e-6\n"
               "use_diis = true\n"
               "[resources]\n"
               "memory_cap_gib = " +
               std::to_string(capGib) + "\n";
    } else if (fixture == "water_cluster")
    {
        // The pair-class admission-gate fixture (the pair-class root
        // fix, 2026-08-31): 24 H2O monomers, 20 Bohr apart along y and
        // mirrored about y = 0 with the mirrored orientation, so the
        // cluster carries the C2v/D2h structure and the class path would
        // request engagement. STO-3G: 120 shells, 7,260 canonical pairs.
        // The class table's never-under estimate (~6.3 GiB) cannot fit
        // the 5.2 GiB cap's cap-minus-base budget, so the admission gate
        // disengages the class path and the plain screened path runs -
        // the completion (or the ladder refusal) proves the gate, where
        // the pre-fix build died 0xC0000409. The full-precision
        // (setprecision 17) literals keep the mirror pairs exact - the
        // point-group detection needs them.
        const double hx = 0.7569503270127429;
        const double hy = 0.58588227657553;
        std::ostringstream atoms;

        for (std::size_t k = 1; k <= 12; ++k)
        {
            for (const int side : {1, -1})
            {
                const double y = 20.0 * static_cast<double>(k) * static_cast<double>(side);
                const double h = y + hy * static_cast<double>(side);
                atoms << "    [\"O\", 0.0, " << std::setprecision(17) << y << ", 0.0],\n";
                atoms << "    [\"H\", " << std::setprecision(17) << hx << ", "
                      << std::setprecision(17) << h << ", 0.0],\n";
                atoms << "    [\"H\", " << std::setprecision(17) << -hx << ", "
                      << std::setprecision(17) << h << ", 0.0],\n";
            }
        }

        toml = "[molecule]\n"
               "charge = 0\n"
               "multiplicity = 1\n"
               "atoms = [\n" +
               atoms.str() +
               "]\n"
               "[basis]\n"
               "orbital = \"sto-3g\"\n"
               "[method]\n"
               "type = \"rhf\"\n"
               "fock_builder = \"direct\"\n"
               "accuracy = \"kLoose\"\n"
               "[scf]\n"
               "max_iterations = 30\n"
               "[resources]\n"
               "memory_cap_gib = " +
               std::to_string(capGib) + "\n";
    } else if (fixture == "nocv_dodecane")
    {
        // The liveness NOCV analysis-refusal fixture (the spawn-site
        // relocation, 2026-09-05): C12H26 STO-3G (n = 86, nPairs = 1953,
        // direct base 0.208 GiB) with the NOCV analysis requested under a
        // cap that admits the SCF but cannot fit the dense-tensor analysis
        // charge (936,271,648 B = 0.87 GiB at n = 86, the same estimate the
        // engine's admission gate applies): the driver must refuse with the
        // ladder BEFORE the builder - a graceful refusal, never a raw
        // allocation failure under the job-object cap. The refusal is
        // model-time (the run refuses before any SCF work), so the child
        // exits 1 with the "refused:" marker promptly. The geometry rows
        // came from the driver test fixture (tools/bench/cases/
        // c12h26_sto3g_rhf.toml, MakeAlkaneSto3g(12), Angstrom).
        toml = R"([molecule]
charge = 0
multiplicity = 1
atoms = [
    ["C", 0.0, 0.0, 0.0],
    ["C", 1.2558586830112795, 0.8880261909692825, 0.0],
    ["C", 2.511717366022559, 0.0, 0.0],
    ["C", 3.7675760490338384, 0.8880261909692825, 0.0],
    ["C", 5.023434732045118, 0.0, 0.0],
    ["C", 6.279293415056397, 0.8880261909692825, 0.0],
    ["C", 7.535152098067677, 0.0, 0.0],
    ["C", 8.791010781078956, 0.8880261909692825, 0.0],
    ["C", 10.046869464090236, 0.0, 0.0],
    ["C", 11.302728147101515, 0.8880261909692825, 0.0],
    ["C", 12.558586830112795, 0.0, 0.0],
    ["C", 13.814445513124074, 0.8880261909692825, 0.0],
    ["H", -0.8899812736030855, 0.6293117936937818, 0.0],
    ["H", -2.4202861947485077e-16, -0.6293117936937815, 0.8899812736030855],
    ["H", 1.815214646061381e-16, -0.6293117936937822, -0.8899812736030853],
    ["H", 1.2558586830112795, 1.5173379846630641, 0.8899812736030855],
    ["H", 1.2558586830112795, 1.5173379846630641, -0.8899812736030855],
    ["H", 2.511717366022559, -0.6293117936937818, 0.8899812736030855],
    ["H", 2.511717366022559, -0.6293117936937818, -0.8899812736030855],
    ["H", 3.7675760490338384, 1.5173379846630641, 0.8899812736030855],
    ["H", 3.7675760490338384, 1.5173379846630641, -0.8899812736030855],
    ["H", 5.023434732045118, -0.6293117936937818, 0.8899812736030855],
    ["H", 5.023434732045118, -0.6293117936937818, -0.8899812736030855],
    ["H", 6.279293415056397, 1.5173379846630641, 0.8899812736030855],
    ["H", 6.279293415056397, 1.5173379846630641, -0.8899812736030855],
    ["H", 7.535152098067677, -0.6293117936937818, 0.8899812736030855],
    ["H", 7.535152098067677, -0.6293117936937818, -0.8899812736030855],
    ["H", 8.791010781078956, 1.5173379846630641, 0.8899812736030855],
    ["H", 8.791010781078956, 1.5173379846630641, -0.8899812736030855],
    ["H", 10.046869464090236, -0.6293117936937818, 0.8899812736030855],
    ["H", 10.046869464090236, -0.6293117936937818, -0.8899812736030855],
    ["H", 11.302728147101515, 1.5173379846630641, 0.8899812736030855],
    ["H", 11.302728147101515, 1.5173379846630641, -0.8899812736030855],
    ["H", 12.558586830112795, -0.6293117936937818, 0.8899812736030855],
    ["H", 12.558586830112795, -0.6293117936937818, -0.8899812736030855],
    ["H", 14.70442678672716, 0.2587143972755007, 0.0],
    ["H", 13.814445513124074, 1.5173379846630641, 0.8899812736030855],
    ["H", 13.814445513124074, 1.5173379846630646, -0.8899812736030853],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"

[properties]
nocv_fragments = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]]

[resources]
memory_cap_gib = )" +
               std::to_string(capGib) + "\n";
    } else
    {
        // The H2 STO-3G fixture with the [resources] block appended; the
        // cap is the argv value so the parent drives the exact scenario.
        toml = "[molecule]\n"
               "charge = 0\n"
               "multiplicity = 1\n"
               "atoms = [[\"H\", 0.0, 0.0, 0.0], "
               "[\"H\", 0.74084809526419992, 0.0, 0.0]]\n"
               "[basis]\n"
               "orbital = \"sto-3g\"\n"
               "[method]\n"
               "type = \"rhf\"\n"
               "fock_builder = \"direct\"\n"
               "accuracy = \"kTight\"\n"
               "[resources]\n"
               "memory_cap_gib = " +
               std::to_string(capGib) + "\n";
    }

    auto input = qcx::io::ParseRunInput(toml);

    if (!input.has_value())
    {
        std::fprintf(stderr, "parse failed: %s\n", input.error().message.c_str());
        return 1;
    }

    auto result = qcx::driver::RunDriver(*input);

    if (!result.has_value())
    {
        std::fprintf(stderr, "run refused: %s\n", result.error().message.c_str());
        std::ofstream marker(argv[2]);
        marker << "refused: " << result.error().message << "\n";
        return 1;
    }

    std::puts(result->c_str());

    std::ofstream marker(argv[2]);
    marker << "completed\n";
    // The mode_record's admission-gate tokens on the budget path (the
    // pair-class fix's observation surface): the parent cannot capture the
    // JSON on stdout, so the marker carries the two key-value pairs as
    // they appear in the serialized document. The scan runs over the WHOLE
    // JSON - not a fixed-offset slice from "mode_record": an UHF run now
    // emits a second mode_record block (the exchange half's), and the keys
    // serialize at the block's end, so a slice is brittle to key
    // reordering and growth.
    const std::string& json = *result;
    // A key's line as it appears in the serialized document. A non-null
    // blockKey bounds the scan to one block and is REQUIRED for a key whose
    // name is not unique to one of them: the request records ("requested" in
    // two blocks, "outcome" in three) serialize as siblings, so a
    // whole-document scan would report whichever block the serializer
    // happened to write first and the marker would state a different fact
    // under the same token. A null blockKey scans the whole document, which
    // is what a key in exactly one block wants.
    const auto writeToken = [&marker, &json](const char* key, const char* blockKey = nullptr) {
        const std::string prefix = std::string("\"") + key + "\": ";
        std::size_t pos = json.find(prefix);

        if (blockKey != nullptr)
        {
            const std::string blockOpen = std::string("\"") + blockKey + "\": {";
            const std::size_t blockStart = json.find(blockOpen);

            if (blockStart == std::string::npos)
            {
                return;
            }

            // The block's members are all scalars, so the brace that closes
            // it is the next one after it opens. A nested member added later
            // would end the window early and leave the token ABSENT, which
            // fails the parent's assertion on the missing line instead of
            // handing it a neighbouring block's value.
            const std::size_t afterOpen = blockStart + blockOpen.size();
            const std::size_t blockEnd = json.find('}', afterOpen);
            const std::size_t scoped = json.find(prefix, afterOpen);

            if (blockEnd == std::string::npos || scoped == std::string::npos || scoped > blockEnd)
            {
                return;
            }

            pos = scoped;
        }

        if (pos == std::string::npos)
        {
            return;
        }

        // The value ends at the next comma, brace, or newline.
        const std::size_t end = json.find_first_of(",}\n", pos + prefix.size());
        marker << json.substr(pos, end == std::string::npos ? std::string::npos : end - pos + 1)
               << "\n";
    };
    writeToken("class_path_disengaged");
    writeToken("class_table_bytes");
    // The RI-K rung's observation surface (the ri_jk fixtures
    // above): the rung word the ri_jk_mode block carries, so the parent can
    // assert WHICH rung the driver's grant bought rather than that the run
    // merely completed, and the tensor term beside it - the 8 n^2 nAux array
    // named arithmetically, which ties the block to the fixture rather than to
    // its own prose. Both tokens are absent on every other fixture (no block,
    // no tokens), and on this family no other block writes either key (the
    // ri_jk arm sets no mode_record).
    writeToken("rung");
    writeToken("tensor_bytes");
    // The disk rung's observation surface (the disk-rung knob):
    // the mode_record's mode word and modeled disk payload, and the run's
    // total energy - the parent asserts the disk route rode the refusal
    // (mode kDisk), recorded the dense 8 n^2 nAux model, and reproduced
    // the storage-suite RI-J pin. The tokens are absent when the run's
    // record carries none of them (the other fixtures are unaffected).
    writeToken("mode");
    writeToken("disk_bytes");
    writeToken("total_energy_hartree");
    // The RI tensor-mode request pairing (the ri_tensor_mode block, the disk
    // rung's own disclosure of what was ASKED): the request word, what it
    // resolved to and the outcome word. Nothing else in the marker can say
    // which rung the input named, so without these three a run that asked
    // for the disk route and rode the in-memory ladder instead - or the
    // reverse - is indistinguishable in the marker from a run that asked for
    // nothing, which is the state the record of this cell exists to make
    // unreportable. The three are scanned INSIDE the block: "requested" and
    // "outcome" are also members of the sibling request records (the chunk
    // hint's, the orbit expansion's), and a run naming two of those keys
    // would otherwise hand the parent a neighbour's word.
    writeToken("requested", "ri_tensor_mode");
    writeToken("resolved", "ri_tensor_mode");
    writeToken("outcome", "ri_tensor_mode");

    return 0;
}
