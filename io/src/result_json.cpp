// Result serialization to the JSON schema (the properties and analyzer
// blocks). See
// result_json.hpp for the null-vs-fabricated policy.

#include "qcx/io/result_json.hpp"

#include <cmath>
#include <nlohmann/json.hpp>

namespace qcx::io {
namespace {

// The nullable member policy: an optional unset member emits null, never a
// placeholder number that would read as real output.
nlohmann::json OptionalVector(const std::optional<std::vector<double>>& values) {
    return values.has_value() ? nlohmann::json(*values) : nlohmann::json(nullptr);
}

// The non-finite policy: JSON has no NaN/Infinity literal
// and dump() would silently write null for them - indistinguishable from
// an unset member. Every non-finite number becomes an explicit string
// marker ("nan", "inf", "-inf") so a consumer can tell "not computed"
// (null) from "computed and non-finite".
void MarkNonFiniteNumbers(nlohmann::json& value) {
    if (value.is_number())
    {
        const double number = value.get<double>();

        if (!std::isfinite(number))
        {
            if (std::isnan(number))
            {
                value = "nan";
            } else if (number > 0.0)
            {
                value = "inf";
            } else
            {
                value = "-inf";
            }
        }

        return;
    }

    if (value.is_array())
    {
        for (auto& element : value)
        {
            MarkNonFiniteNumbers(element);
        }

        return;
    }

    if (value.is_object())
    {
        for (auto it = value.begin(); it != value.end(); ++it)
        {
            MarkNonFiniteNumbers(it.value());
        }
    }
}

// The full-group labeling block: the per-MO labels, the detected
// full group and the Abelian reduction, the canonicalization/straddle
// records, and the finite symmetrization subset. The MO coefficients and
// the symmetrized density stay out of the JSON (the Molden export
// carries the coefficients); the consumer needs the labels, not the
// matrices.
nlohmann::json SerializeSymmetry(const RunSymmetry& symmetry) {
    std::vector<nlohmann::json> canonicalized;
    canonicalized.reserve(symmetry.canonicalized.size());

    for (const auto& record : symmetry.canonicalized)
    {
        canonicalized.push_back(
            nlohmann::json{{"irrep_label", record.irrepLabel}, {"mo_indices", record.moIndices}});
    }

    std::vector<nlohmann::json> straddled;
    straddled.reserve(symmetry.straddled.size());

    for (const auto& record : symmetry.straddled)
    {
        straddled.push_back(
            nlohmann::json{{"irrep_label", record.irrepLabel}, {"mo_indices", record.moIndices}});
    }

    return nlohmann::json{{"full_group", symmetry.fullGroup},
                          {"abelian_reduction", symmetry.abelianReduction},
                          {"labels", symmetry.labels},
                          {"irrep_indices", symmetry.irrepIndices},
                          {"canonicalized", canonicalized},
                          {"straddled", straddled},
                          {"symmetrization_subset", symmetry.symmetrizationSubset},
                          {"averaged_element_count", symmetry.averagedElementCount}};
}

// The symmetry-blocking disclosure of one spin channel (schema 32): the
// guard's action word and the measurement it was taken on. The action is
// carried through as the word the guard's own enum says - never re-derived
// from the two solve counts beside it, which are evidence for the reader, not
// the source of the verdict.
nlohmann::json SerializeSymmetryBlocking(const RunSymmetryBlocking& blocking) {
    return nlohmann::json{{"action", blocking.action},
                          {"blocked_solve_count", blocking.blockedSolveCount},
                          {"plain_solve_count", blocking.plainSolveCount},
                          {"generator_commutator_norms", blocking.generatorCommutatorNorms},
                          {"max_generator_commutator_norm", blocking.maxGeneratorCommutatorNorm},
                          {"tolerance", blocking.tolerance}};
}

} // namespace

std::string SerializeRunResultJson(const RunResult& result) {
    nlohmann::json document;
    document["schema_version"] = RunResult::kSchemaVersion;
    // The coordinate unit the run's geometry was READ under (schema 28): the
    // resolved interpretation - "angstrom" for an absent [molecule] units key,
    // which is what every input file meant before the key existed. Always
    // present, so a record is self-describing: a reader holding the JSON can
    // tell whether the molecule was read as Angstrom or Bohr without the input
    // file, which is the check that was missing.
    document["molecule_units"] = ToString(result.moleculeUnits);
    // The run's own physics (schema 34). `method` is always present -
    // every run is one of the four words, and before this key the record named
    // its Fock builder and no physics at all. `functional` and `xc_grid` are
    // written exactly when the run had them: a Kohn-Sham run named a
    // functional and built an XC grid, a Hartree-Fock run did neither, and an
    // absent key is that statement (the null-honesty rule - never a null and
    // never a block of defaults standing in for a grid that never existed).
    document["method"] = ToString(result.method);

    if (result.functional.has_value())
    {
        document["functional"] = *result.functional;
    }

    if (result.xcGrid.has_value())
    {
        const auto& grid = *result.xcGrid;
        document["xc_grid"] = nlohmann::json{{"radial_points", grid.radialPoints},
                                             {"angular_points", grid.angularPoints},
                                             {"alpha", grid.alpha},
                                             {"radial_exponent", grid.radialExponent},
                                             {"trim_weight", grid.trimWeight},
                                             {"block_target", grid.blockTarget}};
    }

    document["converged"] = result.converged;
    document["iterations"] = result.iterations;
    // The achieved residuals that qualify `converged` (schema 17): always
    // present, null when no SCF loop filled them - the spin_squared pattern,
    // never a fabricated zero.
    document["energy_delta_hartree"] = result.energyDeltaHartree.has_value()
                                           ? nlohmann::json(*result.energyDeltaHartree)
                                           : nullptr;
    document["rms_density_delta"] =
        result.rmsDensityDelta.has_value() ? nlohmann::json(*result.rmsDensityDelta) : nullptr;
    // The linear-dependence removal's disclosure (schema 27): the same null
    // convention as the two residuals above - a run that filled no SCF loop
    // reports null, never a fabricated zero.
    document["num_removed_overlap_directions"] =
        result.numRemovedOverlapDirections.has_value()
            ? nlohmann::json(*result.numRemovedOverlapDirections)
            : nullptr;
    document["total_energy_hartree"] = result.totalEnergyHartree;
    document["electronic_energy_hartree"] = result.electronicEnergyHartree;
    document["spin_squared"] =
        result.spinSquared.has_value() ? nlohmann::json(*result.spinSquared) : nullptr;

    if (result.properties.has_value())
    {
        const auto& populations = result.properties->populations;
        document["populations"] =
            nlohmann::json{{"mulliken",
                            nlohmann::json{{"alpha", populations.mullikenAlpha},
                                           {"beta", populations.mullikenBeta},
                                           {"total", populations.mullikenTotal},
                                           {"spin", populations.mullikenSpin}}},
                           {"lowdin",
                            nlohmann::json{{"alpha", populations.lowdinAlpha},
                                           {"beta", populations.lowdinBeta},
                                           {"total", populations.lowdinTotal},
                                           {"spin", populations.lowdinSpin}}},
                           {"mayer",
                            nlohmann::json{{"bond_orders", populations.mayerBondOrders},
                                           {"free_valences", populations.mayerFreeValences},
                                           {"total_valences", populations.mayerTotalValences}}},
                           {"gopinathan_jug",
                            nlohmann::json{{"bond_orders", populations.gopinathanJugBondOrders}}}};
        document["moments"] = nlohmann::json{{"dipole", result.properties->moments.dipole},
                                             {"quadrupole", result.properties->moments.quadrupole}};

        // The opt-in analyzer blocks: each emitted block lists
        // its members, with null for an analysis the run did not request.
        if (result.properties->charges.has_value())
        {
            document["charges"] =
                nlohmann::json{{"hirshfeld", OptionalVector(result.properties->charges->hirshfeld)},
                               {"voronoi", OptionalVector(result.properties->charges->voronoi)}};
        }

        if (result.properties->esp.has_value())
        {
            const auto& esp = *result.properties->esp;
            document["esp"] = nlohmann::json{{"charges", esp.charges},
                                             {"rms_error", esp.rmsError},
                                             {"point_count", esp.pointCount}};
        }

        if (result.properties->eddb.has_value())
        {
            const auto& eddb = *result.properties->eddb;
            document["eddb"] =
                nlohmann::json{{"total_population", eddb.totalPopulation},
                               {"atomic_populations", eddb.atomicPopulations},
                               {"nobd_occupations", eddb.nobdOccupations},
                               {"central_atom_count", eddb.centralAtomCount},
                               {"two_center_orbital_count", eddb.twoCenterOrbitalCount}};
        }

        if (result.properties->fukui.has_value())
        {
            const auto& fukui = *result.properties->fukui;
            document["fukui"] = nlohmann::json{{"nucleophilic", fukui.nucleophilic},
                                               {"electrophilic", fukui.electrophilic},
                                               {"radical", fukui.radical}};
        }

        if (result.properties->nalewajski.has_value())
        {
            const auto& nalewajski = *result.properties->nalewajski;
            document["nalewajski"] =
                nlohmann::json{{"bond_orders", nalewajski.bondOrders},
                               {"diatomic_covalent", nalewajski.diatomicCovalent},
                               {"atomic_ionic_valence", nalewajski.atomicIonicValence},
                               {"atomic_covalent_valence", nalewajski.atomicCovalentValence},
                               {"total_valence", nalewajski.totalValence}};
        }

        if (result.properties->nocv.has_value())
        {
            const auto& nocv = *result.properties->nocv;
            document["nocv"] =
                nlohmann::json{{"electrostatic", nocv.electrostatic},
                               {"pauli", nocv.pauli},
                               {"orbital", nocv.orbital},
                               {"orbital_unrestricted", nocv.orbitalUnrestricted},
                               {"binding_energy", nocv.bindingEnergy},
                               {"fragment_energies", nocv.fragmentEnergies},
                               {"orbital_components", nocv.orbitalComponents},
                               {"nocv_eigenvalues", nocv.nocvEigenvalues},
                               {"orbital_components_alpha", nocv.orbitalComponentsAlpha},
                               {"orbital_components_beta", nocv.orbitalComponentsBeta},
                               {"nocv_eigenvalues_alpha", nocv.nocvEigenvaluesAlpha},
                               {"nocv_eigenvalues_beta", nocv.nocvEigenvaluesBeta}};
        }

        if (result.properties->densityAtNuclei.has_value())
        {
            document["density_at_nuclei"] =
                nlohmann::json{{"values", result.properties->densityAtNuclei->values}};
        }

        if (result.properties->qtaim.has_value())
        {
            const auto& qtaim = *result.properties->qtaim;
            std::vector<nlohmann::json> bcpRows;
            bcpRows.reserve(qtaim.bondCriticalPoints.size());

            for (const auto& bcp : qtaim.bondCriticalPoints)
            {
                bcpRows.push_back(nlohmann::json{{"atom_a", bcp.atomA},
                                                 {"atom_b", bcp.atomB},
                                                 {"position_bohr", bcp.positionBohr},
                                                 {"density", bcp.density},
                                                 {"laplacian", bcp.laplacian},
                                                 {"ellipticity", bcp.ellipticity},
                                                 {"eigenvalues", bcp.eigenvalues},
                                                 {"bond_path", bcp.bondPath}});
            }

            std::vector<nlohmann::json> ocpRows;
            ocpRows.reserve(qtaim.otherCriticalPoints.size());

            for (const auto& ocp : qtaim.otherCriticalPoints)
            {
                ocpRows.push_back(nlohmann::json{{"position_bohr", ocp.positionBohr},
                                                 {"rank", ocp.rank},
                                                 {"signature_sum", ocp.signatureSum}});
            }

            document["qtaim"] = nlohmann::json{{"bond_critical_points", bcpRows},
                                               {"other_critical_points", ocpRows},
                                               {"unconverged_seeds", qtaim.unconvergedSeeds}};
        }

        if (result.properties->molden.has_value())
        {
            document["molden"] = nlohmann::json{{"file", result.properties->molden->file}};
        }
    } else
    {
        // Same honesty policy as spin_squared: null means "not computed",
        // not "zero".
        document["populations"] = nullptr;
        document["moments"] = nullptr;
    }

    // The full-group labeling blocks: the single RHF channel, the
    // UHF beta channel. Absent when the stage did not run (a C1 molecule,
    // the [symmetry] full_group = false switch, or an unrealizable group)
    // — never a fabricated C1 record.
    if (result.symmetry.has_value())
    {
        document["symmetry"] = SerializeSymmetry(*result.symmetry);
    }

    if (result.symmetryBeta.has_value())
    {
        document["symmetry_beta"] = SerializeSymmetry(*result.symmetryBeta);
    }

    // The symmetry-blocking disclosure (schema 32): the guard's decision
    // per spin channel, beside the labeling blocks it qualifies. The key is
    // written when EITHER channel disclosed anything, and a channel that did
    // not disclose is null inside it - never a fabricated kUsed, and never a
    // fabricated kNotRequested either: a run that never asked carries no
    // block at all, which is the statement.
    if (result.symmetryBlockingAlpha.has_value() || result.symmetryBlockingBeta.has_value())
    {
        document["symmetry_blocking"] =
            nlohmann::json{{"alpha",
                            result.symmetryBlockingAlpha.has_value()
                                ? SerializeSymmetryBlocking(*result.symmetryBlockingAlpha)
                                : nlohmann::json(nullptr)},
                           {"beta",
                            result.symmetryBlockingBeta.has_value()
                                ? SerializeSymmetryBlocking(*result.symmetryBlockingBeta)
                                : nlohmann::json(nullptr)}};
    }

    // The term_counters block (run_driver.cpp publishes it on
    // an instrumented ri_j run - the schema doc's exactly-five-ids
    // contract); absent on every other run, never a fabricated block.
    if (result.termCounters.has_value())
    {
        const auto& counters = *result.termCounters;
        document["term_counters"] = nlohmann::json{{"x", counters.x},
                                                   {"p3", counters.p3},
                                                   {"g3", counters.g3},
                                                   {"qx", counters.qx},
                                                   {"gx", counters.gx}};
    }

    // The certified mixed-precision bound block (run_driver.cpp fills
    // it from the direct builder's certifiedBoundSumOut out-parameter on
    // the machinery members); absent wherever no call reported the
    // quantity, never a fabricated block and never a false zero. Schema
    // 16 adds the global budget enforcement's own members: the routed
    // bound sum, the preset-derived budget, and the comparison's verdict.
    // They are one block, not two, because they answer one question (what
    // the fp32 lane delivered and whether the preset's budget allowed it),
    // and a reader must never see the enforcement's numbers without the
    // `enforced` flag that says they were measured.
    if (result.certifiedBound.has_value())
    {
        const auto& bound = *result.certifiedBound;
        document["certified_bound"] = nlohmann::json{{"calls", bound.calls},
                                                     {"last_call_ha", bound.lastCallHa},
                                                     {"max_call_ha", bound.maxCallHa},
                                                     {"enforced", bound.enforced},
                                                     {"budget_ha", bound.budgetHa},
                                                     {"routed_ha", bound.routedHa},
                                                     {"routed_quartets", bound.routedQuartets},
                                                     {"fell_back_to_fp64", bound.fellBackToFp64}};
    }

    // The QFMM model block (schema 29): what the composed-QFMM builder's
    // Coulomb half actually ran, mirrored from the engine's own record
    // (QfmmHfFockBuilder::ModelRecord) rather than recomputed from the input,
    // so the document cannot disagree with the build about which model
    // produced the reported energy. Absent on every other family, never a
    // fabricated block.
    if (result.qfmmModel.has_value())
    {
        const auto& model = *result.qfmmModel;
        document["qfmm_model"] = nlohmann::json{{"extent_model", model.extentModel},
                                                {"separation_mode", model.separationMode},
                                                {"separation_k", model.separationK},
                                                {"theta", model.theta}};
    }

    // The resources_resolved block: the caps the run applied and the
    // enforcement record; cap_note appears only when the in-process cap
    // was not applied.
    const auto& resources = result.resourcesResolved;
    nlohmann::json resourcesBlock = {{"memory_cap_gib", resources.memoryCapGiB},
                                     {"thread_cap", resources.threadCap},
                                     {"in_process_cap_applied", resources.inProcessCapApplied}};

    if (resources.capNote.has_value())
    {
        resourcesBlock["cap_note"] = *resources.capNote;
    }

    // The ri_j budget path record: the cap-derived
    // workspace budget the engine's Create-time estimate reserved from,
    // and the mode decision with its firing estimate terms. Present only
    // on the budget path (a positive cap); the legacy null-budget path
    // leaves both absent.
    if (resources.workspaceBudget.has_value())
    {
        resourcesBlock["workspace_budget"] =
            nlohmann::json{{"capacity_bytes", resources.workspaceBudget->capacityBytes},
                           {"committed_bytes", resources.workspaceBudget->committedBytes}};
    }

    // The mode-record block of one builder's Create-time decision: shared
    // by the coulomb half (mode_record) and, on UHF runs, the exchange
    // half (exchange_mode_record - the UHF Fock builds from two direct
    // builders, and both admission-gate observations must be visible).
    const auto writeModeRecord = [&resourcesBlock](const char* key, const RunModeRecord& mode) {
        resourcesBlock[key] = nlohmann::json{{"mode", mode.mode},
                                             {"forced_disk", mode.forcedDisk},
                                             {"predicted_bytes", mode.predictedBytes},
                                             {"reserved_bytes", mode.reservedBytes},
                                             {"budget_bytes", mode.budgetBytes},
                                             {"remaining_at_decision", mode.remainingAtDecision},
                                             {"max_batch_bytes", mode.maxBatchBytes},
                                             {"pair_store_bytes", mode.pairStoreBytes},
                                             {"pattern_bytes", mode.patternBytes},
                                             {"scratch_bytes", mode.scratchBytes},
                                             {"structural_bytes", mode.structuralBytes},
                                             {"cache_bytes", mode.cacheBytes},
                                             {"light_store_bytes", mode.lightStoreBytes},
                                             {"chunk_arena_bytes", mode.chunkArenaBytes},
                                             {"chunk_pattern_bytes", mode.chunkPatternBytes},
                                             {"chunk_index_bytes", mode.chunkIndexBytes},
                                             {"light_shells_bytes", mode.lightShellsBytes},
                                             {"chunk_pairs", mode.chunkPairs},
                                             {"disk_bytes", mode.diskBytes},
                                             {"tensor_bytes", mode.tensorBytes},
                                             {"ri_matrix_bytes", mode.riMatrixBytes},
                                             {"task_list_bytes", mode.taskListBytes},
                                             {"metric_bytes", mode.metricBytes},
                                             {"orbital_aux_bytes", mode.orbitalAuxBytes},
                                             {"outer_store_bytes", mode.outerStoreBytes},
                                             {"exchange_bytes", mode.exchangeBytes},
                                             {"exchange_scratch_bytes", mode.exchangeScratchBytes},
                                             {"pattern_excluded", mode.patternExcluded},
                                             {"tensor_excluded", mode.tensorExcluded},
                                             {"class_table_bytes", mode.classTableBytes},
                                             {"class_path_disengaged", mode.classPathDisengaged},
                                             {"concurrent_slots", mode.concurrentSlots},
                                             {"default_team_size", mode.defaultTeamSize}};
    };

    if (resources.modeRecord.has_value())
    {
        writeModeRecord("mode_record", *resources.modeRecord);
    }

    // The RI tensor-mode request record (schema 19, the requested-vs-ran
    // pairing; schema 22 emits it for both request keys): written whenever
    // the input NAMED method.ri_tensor_mode or [diagnostics] force_disk_ri,
    // whatever ran. `reason` is omitted when the outcome is "honoured" (the
    // null honesty policy: the selection record's absent-member rule).
    // `forced` names the KEY that carried the request and is always written.
    if (resources.riTensorMode.has_value())
    {
        const RunRiTensorMode& mode = *resources.riTensorMode;
        nlohmann::json modeBlock = {{"requested", mode.requested},
                                    {"resolved", mode.resolved},
                                    {"outcome", mode.outcome},
                                    {"forced", mode.forced}};

        if (mode.reason.has_value())
        {
            modeBlock["reason"] = *mode.reason;
        }

        resourcesBlock["ri_tensor_mode"] = std::move(modeBlock);
    }

    if (resources.exchangeModeRecord.has_value())
    {
        writeModeRecord("exchange_mode_record", *resources.exchangeModeRecord);
    }

    // The composed full-RI builder's rung record (schema 30): the ri_jk
    // family's own Create-time decision, in ITS vocabulary. Deliberately not
    // written through writeModeRecord above, in either direction: the two
    // records' rung words (kFast/kBlocked against kFastPath/kLightPath) and
    // term decompositions name different rungs and different allocations, so
    // sharing a block would leave every number's meaning to be derived from
    // selection.builder. Absent when no budgeted RI-K builder was created -
    // the no-budget path takes the fast rung without a decision to record.
    if (resources.riJkMode.has_value())
    {
        const RunRiJkMode& rung = *resources.riJkMode;
        resourcesBlock["ri_jk_mode"] =
            nlohmann::json{{"rung", rung.rung},
                           {"predicted_bytes", rung.predictedBytes},
                           {"reserved_bytes", rung.reservedBytes},
                           {"budget_bytes", rung.budgetBytes},
                           {"remaining_at_decision", rung.remainingAtDecision},
                           {"max_batch_bytes", rung.maxBatchBytes},
                           {"structural_bytes", rung.structuralBytes},
                           {"root_bytes", rung.rootBytes},
                           {"tensor_bytes", rung.tensorBytes},
                           {"occ_transform_bytes", rung.occTransformBytes},
                           {"fock_bytes", rung.fockBytes},
                           {"arena_bytes", rung.arenaBytes},
                           {"slice_functions", rung.sliceFunctions},
                           {"slice_count", rung.sliceCount}};
    }

    // The ri_chunk_bytes request record (schema 23): written whenever the
    // input NAMED method.ri_chunk_bytes, whatever ran. The hint's consumption
    // site sits inside the ri_j_link family's disk-rung lambda, so without
    // this block a run that wrote the key and never read it said nothing about
    // it anywhere in the document. `reason` is omitted when the outcome is
    // "honoured" (the null honesty policy, the ri_tensor_mode rule above).
    if (resources.riChunkBytes.has_value())
    {
        const RunRiChunkBytes& chunk = *resources.riChunkBytes;
        nlohmann::json chunkBlock = {{"requested_bytes", chunk.requestedBytes},
                                     {"outcome", chunk.outcome}};

        if (chunk.reason.has_value())
        {
            chunkBlock["reason"] = *chunk.reason;
        }

        resourcesBlock["ri_chunk_bytes"] = std::move(chunkBlock);
    }

    // The ri_orbit_expansion request record (schema 26): written whenever the
    // input NAMED method.ri_orbit_expansion, whatever the family and whatever
    // ran. The key's consumption is one branch deep (the ri_j_link arm) and
    // the engine it drives is inert until that branch hands it a point-group
    // reduction, so a requested-but-inert run and a requested-and-engaged run
    // would otherwise serialize identically. `requested` is always written
    // (a false request is a fact about the input); `reason` is omitted when
    // the outcome is "engaged" (the null honesty policy, the ri_tensor_mode
    // rule above).
    if (resources.riOrbitExpansion.has_value())
    {
        const RunRiOrbitExpansion& orbit = *resources.riOrbitExpansion;
        nlohmann::json orbitBlock = {{"requested", orbit.requested}, {"outcome", orbit.outcome}};

        if (orbit.reason.has_value())
        {
            orbitBlock["reason"] = *orbit.reason;
        }

        resourcesBlock["ri_orbit_expansion"] = std::move(orbitBlock);
    }

    // The eri_store request record (schema 33): written whenever the input
    // NAMED method.eri_cache_store, on BOTH the honoured and the demoted arm,
    // because the block is the demotion's disclosure surface and not only the
    // success path's evidence. The requested path is written in both cases -
    // a disclosure that did not say what was asked for could not be read as a
    // disclosure of anything. `demoted_reason` is omitted when the request was
    // honoured (the null honesty policy, the ri_tensor_mode rule above), and
    // the per-class array is omitted when it is empty rather than written as
    // an empty one (the same rule applied one level down).
    if (resources.eriStore.has_value())
    {
        const RunEriStore& store = *resources.eriStore;
        nlohmann::json storeBlock = {{"path", store.path},
                                     {"engaged", store.engaged},
                                     {"demoted", store.demoted},
                                     {"hit_quartets", store.hitQuartets},
                                     {"miss_quartets", store.missQuartets},
                                     {"read_ms", store.readMs},
                                     {"recompute_ms", store.recomputeMs}};

        if (store.demotedReason.has_value())
        {
            storeBlock["demoted_reason"] = *store.demotedReason;
        }

        if (!store.hitQuartetsByClass.empty())
        {
            nlohmann::json classes = nlohmann::json::array();

            for (const RunEriStoreClassHits& row : store.hitQuartetsByClass)
            {
                classes.push_back(
                    {{"l_bra", row.lBra}, {"l_ket", row.lKet}, {"hit_quartets", row.hitQuartets}});
            }

            storeBlock["hit_quartets_by_class"] = std::move(classes);
        }

        resourcesBlock["eri_store"] = std::move(storeBlock);
    }

    // The compute-profile block (schema 20): the hardware input the
    // certified fp32 lane's default was resolved against, with the verdict
    // and the threshold it was compared against, so the decision is auditable
    // without the reader re-deriving the rule. Every number is written, not
    // only the ones that fired: a passing gate still emits its readings.
    if (resources.computeProfile.has_value())
    {
        const RunComputeProfile& profile = *resources.computeProfile;
        resourcesBlock["compute_profile"] =
            nlohmann::json{{"source", profile.source},
                           {"fp32_gflops", profile.fp32Gflops},
                           {"fp64_gflops", profile.fp64Gflops},
                           {"ratio", profile.ratio},
                           {"measured", profile.measured},
                           {"simd_lane", profile.simdLane},
                           {"pairs", profile.pairs},
                           {"ratio_spread", profile.ratioSpread},
                           {"certified_lane_default", profile.certifiedLaneDefault},
                           {"certified_lane_forced", profile.certifiedLaneForced},
                           {"certified_lane_min_ratio", profile.certifiedLaneMinRatio}};
    }

    // The selection record serializes LAST in the block:
    // consumers that read only the early keys are unaffected by the schema
    // addition (the schema_version bump says what changed).
    if (resources.selection.has_value())
    {
        const auto& selection = *resources.selection;
        nlohmann::json selectionBlock = {{"builder", std::string(ToString(selection.builder))},
                                         {"reasoning", selection.reasoning}};

        // The within-family member name (schema 15), the additive sibling
        // of `builder`: the family word stays the family word (a lean run
        // reports "direct"), and this key says which member of it ran.
        // Absent when the record never resolved (the null honesty policy).
        if (selection.builderMember.has_value())
        {
            selectionBlock["builder_member"] = *selection.builderMember;
        }

        if (selection.picked.has_value())
        {
            selectionBlock["picked"] = std::string(ToString(*selection.picked));
        }

        if (selection.explicitBuilder.has_value())
        {
            selectionBlock["explicit_builder"] = std::string(ToString(*selection.explicitBuilder));
        }

        if (selection.warning.has_value())
        {
            selectionBlock["warning"] = *selection.warning;
        }

        // The exchange-approximation disclosure (schema 24). Written when an
        // approximate builder was wired OR the aux selection carries a notice
        // (the trigger is wider than that - see RunApproximation's own
        // note). Absence means "exact AND no notice" rather than an unknown, so
        // a consumer never has to guess whether the writer forgot it; the block
        // itself costs the record nothing, because the common case has neither
        // an approximation nor a notice and emits no block at all.
        if (selection.approximation.has_value())
        {
            selectionBlock["approximation"] = {{"exchange", selection.approximation->exchange},
                                               {"aux_basis", selection.approximation->auxBasis}};

            // The aux-selection notice. Omitted when the selection is in
            // no weak region: an absent key is the "no warning" of this field,
            // the same null honesty the block's own absence uses one level up.
            if (selection.approximation->auxNotice.has_value())
            {
                selectionBlock["approximation"]["aux_notice"] = *selection.approximation->auxNotice;
            }

            // The measured-error disclosure (schema 31). Written exactly when the
            // block's `exchange` names an APPROXIMATED contraction, which is a
            // NARROWER rule than the block's own presence (schema 25 keeps the
            // block on an exact-kernel run carrying a notice): on that arm the
            // fitted exchange's error is no error of this run's, so writing it
            // would be a lie about the run rather than a disclosure of it. An
            // absent key is the "not approximated" of this field, the same null
            // honesty the notice beside it uses.
            if (selection.approximation->exchangeError.has_value())
            {
                const auto& exchangeError = *selection.approximation->exchangeError;
                selectionBlock["approximation"]["exchange_error"] = {
                    {"measured_per_atom_hartree", exchangeError.measuredPerAtomHartree},
                    {"bar_per_atom_hartree", exchangeError.barPerAtomHartree},
                    {"bar_preset", exchangeError.barPreset},
                    {"measured_on", exchangeError.measuredOn},
                    {"measured_aux_basis", exchangeError.measuredAuxBasis}};
            }
        }

        // The coefficient-seam provenance block (schema 11). RETIRED
        // PRODUCER: the ranking that filled it died with
        // the lean flip, so no driver path sets `coefficients` and
        // this branch is unreachable in a real run - the serializer keeps
        // the shape for the record's versioned contract and the io tests
        // are its only caller. Absence IS the record's "none".
        if (selection.coefficients.has_value())
        {
            const auto& coefficients = *selection.coefficients;
            nlohmann::json coefficientsBlock = {{"file_version", coefficients.fileVersion},
                                                {"machine_class_key", coefficients.machineClassKey},
                                                {"basis_family_id", coefficients.basisFamilyId},
                                                {"preset_id", coefficients.presetId},
                                                {"borrowed", coefficients.borrowed}};
            nlohmann::json scores = nlohmann::json::array();

            for (const RunSelectionScore& score : coefficients.scores)
            {
                scores.push_back({{"candidate", score.candidate},
                                  {"score_seconds", score.scoreSeconds},
                                  {"source", score.source},
                                  {"selectable", score.selectable}});
            }

            coefficientsBlock["scores"] = std::move(scores);

            if (coefficients.cellKey.has_value())
            {
                coefficientsBlock["cell_key"] = *coefficients.cellKey;
            }

            if (coefficients.inflation.has_value())
            {
                coefficientsBlock["inflation"] = *coefficients.inflation;
            }

            selectionBlock["coefficients"] = std::move(coefficientsBlock);
        }

        resourcesBlock["selection"] = std::move(selectionBlock);
    }

    document["resources_resolved"] = std::move(resourcesBlock);

    document["timings_ms"] = {{"total", result.timingsMs.totalMs},
                              {"scf_loop", result.timingsMs.scfLoopMs}};

    // The RESOLVED builder selection in the orthogonal axis vocabulary (schema
    // 35). Always present - every run resolves a
    // builder, so the triple is always determinable, and an absent block could
    // only mean the driver forgot to fill it. `legacy_spelling` is the absent
    // half and carries the deprecation: it and `deprecated_key` are written
    // exactly when the input named its selection through the deprecated
    // `[method] fock_builder` key, so a reader can see which vocabulary the file
    // was written in. `deprecated_key` names the KEY rather than repeating the
    // deprecation in prose, because a record member that carries an explanation
    // is a second thing to keep current; `requested_by` is what keeps the
    // absence unambiguous - it distinguishes the `[builder]` axes from the size
    // ladder, which `legacy_spelling`'s absence alone cannot.
    {
        const auto& axes = result.builderAxes;
        nlohmann::json axesBlock{{"integral_family", axes.integralFamily},
                                 {"storage_tier", axes.storageTier},
                                 {"execution_backend", axes.executionBackend},
                                 {"requested_by", axes.requestedBy}};

        if (axes.legacySpelling.has_value())
        {
            axesBlock["legacy_spelling"] = *axes.legacySpelling;
            axesBlock["deprecated_key"] = "method.fock_builder";
        }

        document["builder_axes"] = std::move(axesBlock);
    }

    MarkNonFiniteNumbers(document);
    return document.dump(2);
}

} // namespace qcx::io
