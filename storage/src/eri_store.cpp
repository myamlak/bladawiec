// The HDF5 integral store (eri_store.hpp): the schema
// owner. Layout: one flat packed values dataset per precision plus a
// manifest (the index), the quartets table (the row space), per-chunk
// FNV-1a-64 checksums verified on every serve, and a /metadata/store_checksum
// over (quartets table bytes + manifest bytes + the /metadata scalar block)
// recomputed on every append and verified at open.
//
// The store_checksum canonical scalar-block order (documented here):
// screening_preset string bytes, schwarz_threshold 8 bytes,
// density_threshold 8 bytes, mixed_precision_threshold 8 bytes,
// engine_version 4 bytes, schema_version 4 bytes, provenance string bytes.
//
// Append-only: chunks are never rewritten; a failed append can leave
// orphan value bytes - unreachable through the manifest, harmless; the
// next store_checksum recompute ignores them. Windows locks the file for
// as long as the HighFive::File lives - the store holds exactly one
// handle, single-writer, no concurrency.
//
// Manifest rows (the one-dtype-all-uint64 rule). fp64 (10 columns):
// [chunk_id, first_quartet, quartet_count, value_offset_bytes,
//  value_bytes, checksum, l_bra, l_ket, engine_version, schema_version]
// fp32 (13 columns): the same 10 + [bounds_offset_bytes, bounds_bytes,
//  bounds_checksum] - the bounds dataset carries its own per-chunk FNV-1a-64
// checksum, verified on serve exactly like the values chunk.
// Per-quartet block offsets are DERIVED, not stored: block size =
// nI*nJ*nK*nL from ShellFunctionCount over the canonical quartet's shells
// (the engine's base-accumulation rule).

#include "qcx/storage/eri_store.hpp"

#include "hdf5_util.hpp"
#include "qcx/storage/fingerprint.hpp"
#include "store_molecule.hpp"
#include "test_hooks.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5PropertyList.hpp>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace qcx::storage {

// The canonical-quartet key of the store's lookup maps.
using QuartetKey = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>;

// One stored block's location: the chunk id (manifest row index) and the
// position inside the chunk's quartet run.
struct BlockLocation {
    std::size_t chunkId = 0;
    std::size_t position = 0;
};

// The store's implementation state (eri_store.hpp's PImpl): the file
// handle, the pair list (block-size derivation), the /metadata scalars,
// the in-memory mirrors of the append-only tables (the checksum input),
// and the manifest-derived lookups.
struct EriStore::State {
    HighFive::File file;
    qcx::integrals::ShellPairList pairList;
    std::string fingerprint;

    // /metadata scalars (the checksum's scalar block).
    std::string screeningPreset;
    double schwarzThreshold = 0.0;
    double densityThreshold = 0.0;
    double mixedPrecisionThreshold = 0.0;
    std::string provenance;

    // In-memory mirrors of the append-only tables (the checksum input).
    std::vector<std::uint64_t> quartets; // 4 per stored quartet.
    std::vector<std::uint64_t> fp64Manifest; // kManifestFp64Columns per chunk.
    std::vector<std::uint64_t> fp32Manifest; // kManifestFp32Columns per chunk.

    // Lookups: canonical quartet -> (chunkId, position in run), per
    // precision; built at Create/Open by scanning the tables.
    std::map<QuartetKey, BlockLocation> fp64Lookup;
    std::map<QuartetKey, BlockLocation> fp32Lookup;
};

namespace {

inline constexpr std::size_t kQuartetsColumns = 4;
inline constexpr std::size_t kManifestFp64Columns = 10;
inline constexpr std::size_t kManifestFp32Columns = 13;

// Manifest column indexes (both precisions share the first 10).
inline constexpr std::size_t kChunkIdColumn = 0;
inline constexpr std::size_t kFirstQuartetColumn = 1;
inline constexpr std::size_t kQuartetCountColumn = 2;
inline constexpr std::size_t kValueOffsetColumn = 3;
inline constexpr std::size_t kValueBytesColumn = 4;
inline constexpr std::size_t kChecksumColumn = 5;
inline constexpr std::size_t kLBraColumn = 6;
inline constexpr std::size_t kLKetColumn = 7;
inline constexpr std::size_t kEngineVersionColumn = 8;
inline constexpr std::size_t kSchemaVersionColumn = 9;
inline constexpr std::size_t kBoundsOffsetColumn = 10;
inline constexpr std::size_t kBoundsBytesColumn = 11;
inline constexpr std::size_t kBoundsChecksumColumn = 12;

// The block element count of a canonical quartet - the same rule the
// engine's base accumulation uses (nI*nJ*nK*nL; eri_batch.hpp's
// EriBlockIndex spans this).
std::size_t BlockElementCount(const qcx::integrals::ShellPairList& pairList,
                              const qcx::integrals::ShellQuartet& quartet) {
    return qcx::integrals::ShellFunctionCount(pairList.shells[quartet.i]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.j]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.k]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.l]);
}

// The class of a canonical quartet (the chunk's run key).
std::pair<int, int> ClassOf(const qcx::integrals::ShellPairList& pairList,
                            const qcx::integrals::ShellQuartet& quartet) {
    return {pairList.shells[quartet.i].angularMomentum + pairList.shells[quartet.j].angularMomentum,
            pairList.shells[quartet.k].angularMomentum +
                pairList.shells[quartet.l].angularMomentum};
}

// The /metadata scalar block in its canonical order (see the file comment).
// (schwarzThreshold .. schemaVersion) are the five scalars of the block in
// the schema order - distinct quantities, appended in fixed order.
void AppendScalarBlock(std::vector<std::byte>& bytes,
                       const std::string& screeningPreset,
                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                       double schwarzThreshold,
                       double densityThreshold,
                       double mixedPrecisionThreshold,
                       std::uint32_t engineVersion,
                       std::uint32_t schemaVersion,
                       const std::string& provenance) {
    const auto append = [&bytes](const void* data, std::size_t size) {
        const auto* raw = static_cast<const std::byte*>(data);
        bytes.insert(bytes.end(), raw, raw + size);
    };
    append(screeningPreset.data(), screeningPreset.size());
    append(&schwarzThreshold, sizeof(schwarzThreshold));
    append(&densityThreshold, sizeof(densityThreshold));
    append(&mixedPrecisionThreshold, sizeof(mixedPrecisionThreshold));
    append(&engineVersion, sizeof(engineVersion));
    append(&schemaVersion, sizeof(schemaVersion));
    append(provenance.data(), provenance.size());
}

void AppendU64Table(std::vector<std::byte>& bytes, const std::vector<std::uint64_t>& table) {
    bytes.reserve(bytes.size() + table.size() * sizeof(std::uint64_t));
    const auto* raw = reinterpret_cast<const std::byte*>(table.data());
    bytes.insert(bytes.end(), raw, raw + table.size() * sizeof(std::uint64_t));
}

// The store checksum over the append-only tables + the scalar block.
std::uint64_t ComputeStoreChecksum(const std::vector<std::uint64_t>& quartets,
                                   const std::vector<std::uint64_t>& fp64Manifest,
                                   const std::vector<std::uint64_t>& fp32Manifest,
                                   const std::string& screeningPreset,
                                   double schwarzThreshold,
                                   double densityThreshold,
                                   double mixedPrecisionThreshold,
                                   std::uint32_t engineVersion,
                                   std::uint32_t schemaVersion,
                                   const std::string& provenance) {
    std::vector<std::byte> bytes;
    bytes.reserve(quartets.size() * 8 + fp64Manifest.size() * 8 + fp32Manifest.size() * 8 + 128);
    AppendU64Table(bytes, quartets);
    AppendU64Table(bytes, fp64Manifest);
    AppendU64Table(bytes, fp32Manifest);
    AppendScalarBlock(bytes,
                      screeningPreset,
                      schwarzThreshold,
                      densityThreshold,
                      mixedPrecisionThreshold,
                      engineVersion,
                      schemaVersion,
                      provenance);
    return internal::Fnv1a64(bytes.data(), bytes.size());
}

// The accuracy-preset name recorded in /metadata (the "kLoose"/"kNormal"/
// "kTight" spellings of the schema tree).
//
// Exhaustive BY CONSTRUCTION, and this was the one labeler of
// the four the audit enumerated that was not honest: kNormal shared its
// `break` with the tail, so ANY unplaced AccuracyPreset wrote "kNormal" into
// a persisted file - a plausible wrong answer, which no reader of the file
// can tell from the true one. The three io-side ToString siblings already
// return `"unknown"` for that case; this now matches them. The guard below is
// accuracy.hpp's mapper guard restated here, because that region spans its
// own file's mappers and cannot reach this one: without it a new preset would
// be forced to a placement in the mappers but could still land on this tail
// silently.
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
std::string PresetName(qcx::integrals::AccuracyPreset preset) {
    switch (preset)
    {
    case qcx::integrals::AccuracyPreset::kLoose:
        return "kLoose";
    case qcx::integrals::AccuracyPreset::kNormal:
        return "kNormal";
    case qcx::integrals::AccuracyPreset::kTight:
        return "kTight";
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

// The HDF5 chunking of each table (the schema's constants).
HighFive::DataSetCreateProps ExtendibleChunking(const std::vector<hsize_t>& chunkDims) {
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking(chunkDims));
    return props;
}

// Creates the empty extendible tables of /integrals/ao/eri.
qcx::Result<void> CreateEriTables(HighFive::File& file) {
    return WrapH5([&]() {
        HighFive::Group eri = file.createGroup("integrals/ao/eri");
        eri.createDataSet("quartets",
                          HighFive::DataSpace({0, kQuartetsColumns},
                                              {HighFive::DataSpace::UNLIMITED, kQuartetsColumns}),
                          HighFive::AtomicType<std::uint64_t>(),
                          ExtendibleChunking({1024, kQuartetsColumns}));
        eri.createDataSet("fp64/values",
                          HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
                          HighFive::AtomicType<double>(),
                          ExtendibleChunking({1048576}));
        eri.createDataSet(
            "fp64/manifest",
            HighFive::DataSpace({0, kManifestFp64Columns},
                                {HighFive::DataSpace::UNLIMITED, kManifestFp64Columns}),
            HighFive::AtomicType<std::uint64_t>(),
            ExtendibleChunking({1024, kManifestFp64Columns}));
        eri.createDataSet("fp32/values",
                          HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
                          HighFive::AtomicType<float>(),
                          ExtendibleChunking({2097152}));
        eri.createDataSet("fp32/bounds",
                          HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}),
                          HighFive::AtomicType<double>(),
                          ExtendibleChunking({1024}));
        eri.createDataSet(
            "fp32/manifest",
            HighFive::DataSpace({0, kManifestFp32Columns},
                                {HighFive::DataSpace::UNLIMITED, kManifestFp32Columns}),
            HighFive::AtomicType<std::uint64_t>(),
            ExtendibleChunking({1024, kManifestFp32Columns}));
    });
}

// Creates the empty /derived staging slots (the derived-dataset contract; each carries
// the source_fingerprint/engine_version/schema_version attributes).
qcx::Result<void> CreateDerivedSlots(HighFive::File& file) {
    auto derived = WrapH5Value([&]() { return file.createGroup("derived"); });

    if (!derived.has_value())
    {
        return std::unexpected(derived.error());
    }

    // One staging slot: the empty extendible dataset plus its attributes
    // (source_fingerprint fixed-width - hdf5_util.hpp - and the version
    // scalars). Every step reports its own error.
    const auto makeSlot = [&](const std::string& path,
                              const std::vector<std::size_t>& shape) -> qcx::Result<void> {
        if (auto created = WrapH5([&]() {
                // A local constant, not the static member directly:
                // UNLIMITED is an in-class-initialized non-constexpr static,
                // so binding it to the vector ctor's const& is an ODR-use
                // needing an out-of-line definition - clang/GCC then fail at
                // link while MSVC materializes the member. The local folds
                // the constant.
                const std::size_t unlimited = HighFive::DataSpace::UNLIMITED;
                const std::vector<std::size_t> maxDims(shape.size(), unlimited);
                derived->createDataSet(path,
                                       HighFive::DataSpace(shape, maxDims),
                                       HighFive::AtomicType<double>(),
                                       ExtendibleChunking(std::vector<hsize_t>(shape.size(), 8)));
            });
            !created.has_value())
        {
            return std::unexpected(created.error());
        }

        auto slot = WrapH5Value([&]() { return derived->getDataSet(path); });

        if (!slot.has_value())
        {
            return std::unexpected(slot.error());
        }

        if (auto written = WriteScalarStringAttribute(*slot, "source_fingerprint", "");
            !written.has_value())
        {
            return std::unexpected(written.error());
        }

        return WrapH5([&]() {
            auto engineAttr = slot->createAttribute<std::uint32_t>(
                "engine_version",
                HighFive::DataSpace::From(qcx::integrals::kIntegralEngineVersion));
            engineAttr.write(qcx::integrals::kIntegralEngineVersion);
            auto schemaAttr = slot->createAttribute<std::uint32_t>(
                "schema_version", HighFive::DataSpace::From(kStoreSchemaVersion));
            schemaAttr.write(kStoreSchemaVersion);
        });
    };

    for (const auto& slotSpec : {
             std::pair<const char*, std::vector<std::size_t>>{"mo_integrals/iajb", {0, 0, 0, 0}},
             std::pair<const char*, std::vector<std::size_t>>{"mo_integrals/abcd", {0, 0, 0, 0}},
             std::pair<const char*, std::vector<std::size_t>>{"mo_integrals/kpqrs", {0, 0, 0, 0}},
             std::pair<const char*, std::vector<std::size_t>>{"grid_properties/ao_values", {0, 2}},
         })
    {
        if (auto slot = makeSlot(slotSpec.first, slotSpec.second); !slot.has_value())
        {
            return std::unexpected(slot.error());
        }
    }

    return {};
}

// Writes the /metadata group (versions, thresholds, provenance, checksum).
qcx::Result<void> WriteMetadata(HighFive::File& file,
                                const StoreOptions& options,
                                const std::vector<std::uint64_t>& quartets,
                                const std::vector<std::uint64_t>& fp64Manifest,
                                const std::vector<std::uint64_t>& fp32Manifest) {
    const std::string presetName = PresetName(options.accuracy);
    const double schwarz = qcx::integrals::SchwarzThreshold(options.accuracy);
    const double density = qcx::integrals::DensityThreshold(options.accuracy);
    const double mixed = qcx::integrals::MixedPrecisionThreshold(options.accuracy);
    const std::uint64_t checksum = ComputeStoreChecksum(quartets,
                                                        fp64Manifest,
                                                        fp32Manifest,
                                                        presetName,
                                                        schwarz,
                                                        density,
                                                        mixed,
                                                        qcx::integrals::kIntegralEngineVersion,
                                                        kStoreSchemaVersion,
                                                        options.provenance);

    auto metadata = WrapH5Value([&]() { return file.createGroup("metadata"); });

    if (!metadata.has_value())
    {
        return std::unexpected(metadata.error());
    }

    // The strings are fixed-width char datasets (hdf5_util.hpp).
    if (auto written = WriteScalarString(*metadata, "screening_preset", presetName);
        !written.has_value())
    {
        return std::unexpected(written.error());
    }

    if (auto written = WriteScalarString(*metadata, "provenance", options.provenance);
        !written.has_value())
    {
        return std::unexpected(written.error());
    }

    return WrapH5([&]() {
        metadata->createDataSet("schwarz_threshold", schwarz);
        metadata->createDataSet("density_threshold", density);
        metadata->createDataSet("mixed_precision_threshold", mixed);
        metadata->createDataSet("engine_version", qcx::integrals::kIntegralEngineVersion);
        metadata->createDataSet("schema_version", kStoreSchemaVersion);
        metadata->createDataSet("store_checksum", checksum);
    });
}

// Reads /metadata into the state, verifying the store checksum.
qcx::Result<void> ReadAndVerifyMetadata(HighFive::File& file, EriStore::State& state) {
    std::string presetName;
    double schwarz = 0.0;
    double density = 0.0;
    double mixed = 0.0;
    std::uint32_t engineVersion = 0;
    std::uint32_t schemaVersion = 0;
    std::uint64_t storedChecksum = 0;

    auto metadata = WrapH5Value([&]() { return file.getGroup("metadata"); });

    if (!metadata.has_value())
    {
        return std::unexpected(metadata.error());
    }

    if (auto value = ReadScalarString(*metadata, "screening_preset"); value.has_value())
    {
        presetName = *value;
    } else
    {
        return std::unexpected(value.error());
    }

    if (auto value = ReadScalarString(*metadata, "provenance"); value.has_value())
    {
        state.provenance = *value;
    } else
    {
        return std::unexpected(value.error());
    }

    auto result = WrapH5([&]() {
        metadata->getDataSet("schwarz_threshold").read(schwarz);
        metadata->getDataSet("density_threshold").read(density);
        metadata->getDataSet("mixed_precision_threshold").read(mixed);
        metadata->getDataSet("engine_version").read(engineVersion);
        metadata->getDataSet("schema_version").read(schemaVersion);
        metadata->getDataSet("store_checksum").read(storedChecksum);
    });

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    state.screeningPreset = presetName;
    state.schwarzThreshold = schwarz;
    state.densityThreshold = density;
    state.mixedPrecisionThreshold = mixed;

    const std::uint64_t expected = ComputeStoreChecksum(state.quartets,
                                                        state.fp64Manifest,
                                                        state.fp32Manifest,
                                                        presetName,
                                                        schwarz,
                                                        density,
                                                        mixed,
                                                        engineVersion,
                                                        schemaVersion,
                                                        state.provenance);

    if (expected != storedChecksum)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, "store checksum mismatch (corrupt store)"});
    }

    return {};
}

// Builds one precision's lookup from the tables: for each chunk row, the
// quartets [firstQuartet, firstQuartet + quartetCount) map to
// (chunkId, position).
template <std::size_t kColumns>
std::map<QuartetKey, BlockLocation> BuildLookup(const std::vector<std::uint64_t>& quartets,
                                                const std::vector<std::uint64_t>& manifest) {
    std::map<QuartetKey, BlockLocation> lookup;
    const std::size_t chunkCount = manifest.size() / kColumns;

    for (std::size_t chunk = 0; chunk < chunkCount; ++chunk)
    {
        const std::uint64_t* row = manifest.data() + chunk * kColumns;
        const std::uint64_t firstQuartet = row[kFirstQuartetColumn];
        const std::uint64_t quartetCount = row[kQuartetCountColumn];

        for (std::size_t position = 0; position < quartetCount; ++position)
        {
            const std::uint64_t* quartet = quartets.data() + (firstQuartet + position) * 4;
            lookup.emplace(QuartetKey{quartet[0], quartet[1], quartet[2], quartet[3]},
                           BlockLocation{chunk, position});
        }
    }

    return lookup;
}

// Reads a whole uint64 table into a flat vector (the mirror).
qcx::Result<std::vector<std::uint64_t>> ReadU64Table(HighFive::File& file,
                                                     const std::string& path) {
    std::vector<std::uint64_t> table;

    auto read = WrapH5([&]() {
        HighFive::DataSet dataSet = file.getDataSet(path);
        const std::vector<std::size_t> dims = dataSet.getSpace().getDimensions();
        table.resize(std::accumulate(
            dims.begin(), dims.end(), std::size_t{1}, std::multiplies<std::size_t>()));
        // read() rejects a flat buffer against a rank-2 dataspace (its
        // dimension check only squeezes singleton dims); the raw overload
        // writes the contiguous flat layout.
        dataSet.read_raw(table.data(), dataSet.getDataType());
    });

    if (!read.has_value())
    {
        return std::unexpected(read.error());
    }

    return table;
}

// Reads one stored one-electron matrix (the tensor is row-major, HDF5 is
// row-major - the host view is the on-disk layout).
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> ReadOneElectron(
    const EriStore::State& state, const std::string& name) {
    std::vector<std::size_t> dims;
    std::vector<double> flat;

    auto read = WrapH5([&]() {
        HighFive::DataSet dataSet = state.file.getDataSet("one_electron/" + name);
        dims = dataSet.getSpace().getDimensions();
        flat.resize(dims.at(0) * dims.at(1));
        dataSet.read_raw(flat.data(), dataSet.getDataType());
    });

    if (!read.has_value())
    {
        return std::unexpected(read.error());
    }

    if (flat.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the one-electron matrices are not stored"});
    }

    auto tensor =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({dims.at(0), dims.at(1)});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    tensor->HostView() = std::move(flat);
    return tensor;
}

// The total of one manifest column over the mirror's chunks: the mirror
// manifest is the source of truth for the append placement (see the
// AppendImpl comment), so the offsets are sums over it, never reads of the
// on-disk tables.
std::size_t ManifestColumnSum(const std::vector<std::uint64_t>& manifest,
                              std::size_t columns,
                              std::size_t columnIndex) {
    std::size_t sum = 0;

    for (std::size_t chunk = 0; chunk < manifest.size() / columns; ++chunk)
    {
        sum += manifest[chunk * columns + columnIndex];
    }

    return sum;
}

// The shared append for both precisions. kCertified selects the fp32 lane:
// values stored as float verbatim (never promoted through double) plus the
// per-quartet bounds in quartets-table order.
template <bool kCertified>
qcx::Result<void> AppendImpl(
    EriStore::State& state,
    const std::vector<qcx::integrals::ShellQuartet>& quartets,
    const std::vector<std::conditional_t<kCertified, float, double>>& values,
    const std::vector<double>& errorBounds) {
    using ValueElement = std::conditional_t<kCertified, float, double>;
    constexpr std::size_t kColumns = kCertified ? kManifestFp32Columns : kManifestFp64Columns;

    if (quartets.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the quartet list is empty"});
    }

    // Canonical form + class validation (the decorator always passes
    // CanonicalizeQuartetOrder output; this guards misuse).
    const auto& shells = state.pairList.shells;
    const std::pair<int, int> klass = ClassOf(state.pairList, quartets.front());
    std::size_t blockBytes = 0;

    for (const qcx::integrals::ShellQuartet& quartet : quartets)
    {
        if (quartet.i >= shells.size() || quartet.j >= shells.size() ||
            quartet.k >= shells.size() || quartet.l >= shells.size())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "quartet shell index out of range"});
        }

        if (quartet.i > quartet.j || quartet.k > quartet.l)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "quartets must be in canonical form"});
        }

        // The class canonical form: L_bra <= L_ket (CanonicalizeQuartetOrder
        // enforces it with the class swap LAST, after the 8-fold pair swap,
        // so the engine's computed output can legitimately carry a bra pair
        // that sorts after the ket pair - the pair order is not an
        // invariant of the stored runs, only the class canonical form is).
        // A run must arrive as the canonical class, never its mirror.
        if (ClassOf(state.pairList, quartet).first > ClassOf(state.pairList, quartet).second)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "quartets must be in canonical class order"});
        }

        if (ClassOf(state.pairList, quartet) != klass)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "one class run per append"});
        }

        blockBytes += BlockElementCount(state.pairList, quartet) * sizeof(ValueElement);
    }

    if (blockBytes != values.size() * sizeof(ValueElement))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "block sizes do not match the values"});
    }

    if constexpr (kCertified)
    {
        if (errorBounds.size() != quartets.size())
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the error bounds must match the quartets"});
        }
    }

    // The mirrors are the source of truth for placement: a failure between
    // the quartets write and the manifest write leaves orphan rows on the
    // on-disk tables (documented below), and the store checksum covers the
    // mirrors - so the manifest row must point at the mirror state, and the
    // resizes below then truncate the orphans (HDF5 shrink is legal),
    // keeping the tables bit-identical to the mirrors. Deriving the extents
    // from the on-disk tables instead would hand the next append a manifest
    // row whose firstQuartet points past the mirrors - every later load of
    // that chunk would compute block offsets from the wrong quartets.
    const std::size_t chunkId = kCertified ? state.fp32Manifest.size() / kManifestFp32Columns
                                           : state.fp64Manifest.size() / kManifestFp64Columns;
    const std::size_t firstQuartet = state.quartets.size() / 4;
    const std::size_t valueOffsetBytes = ManifestColumnSum(
        kCertified ? state.fp32Manifest : state.fp64Manifest, kColumns, kValueBytesColumn);
    const std::size_t valuesExtent = valueOffsetBytes / sizeof(ValueElement);
    std::size_t boundsExtent = 0;

    if constexpr (kCertified)
    {
        boundsExtent =
            ManifestColumnSum(state.fp32Manifest, kManifestFp32Columns, kBoundsBytesColumn) /
            sizeof(double);
    }

    const std::size_t valueBytes = values.size() * sizeof(ValueElement);
    const std::uint64_t checksum =
        internal::Fnv1a64(reinterpret_cast<const std::byte*>(values.data()), valueBytes);
    std::array<std::uint64_t, kManifestFp32Columns> row{};
    row[kChunkIdColumn] = chunkId;
    row[kFirstQuartetColumn] = firstQuartet;
    row[kQuartetCountColumn] = quartets.size();
    row[kValueOffsetColumn] = valueOffsetBytes;
    row[kValueBytesColumn] = valueBytes;
    row[kChecksumColumn] = checksum;
    row[kLBraColumn] = static_cast<std::uint64_t>(klass.first);
    row[kLKetColumn] = static_cast<std::uint64_t>(klass.second);
    row[kEngineVersionColumn] = qcx::integrals::kIntegralEngineVersion;
    row[kSchemaVersionColumn] = kStoreSchemaVersion;

    if constexpr (kCertified)
    {
        row[kBoundsOffsetColumn] = boundsExtent * sizeof(double);
        row[kBoundsBytesColumn] = errorBounds.size() * sizeof(double);
        row[kBoundsChecksumColumn] =
            internal::Fnv1a64(reinterpret_cast<const std::byte*>(errorBounds.data()),
                              errorBounds.size() * sizeof(double));
    }

    // The new mirrors (committed only after every HDF5 write succeeds).
    std::vector<std::uint64_t> newQuartets = state.quartets;
    newQuartets.reserve(newQuartets.size() + quartets.size() * 4);

    for (const qcx::integrals::ShellQuartet& quartet : quartets)
    {
        newQuartets.push_back(quartet.i);
        newQuartets.push_back(quartet.j);
        newQuartets.push_back(quartet.k);
        newQuartets.push_back(quartet.l);
    }

    std::vector<std::uint64_t> newManifest = kCertified ? state.fp32Manifest : state.fp64Manifest;
    newManifest.insert(
        newManifest.end(), row.begin(), row.begin() + static_cast<long long>(kColumns));

    const std::uint64_t newChecksum =
        ComputeStoreChecksum(newQuartets,
                             kCertified ? state.fp64Manifest : newManifest,
                             kCertified ? newManifest : state.fp32Manifest,
                             state.screeningPreset,
                             state.schwarzThreshold,
                             state.densityThreshold,
                             state.mixedPrecisionThreshold,
                             qcx::integrals::kIntegralEngineVersion,
                             kStoreSchemaVersion,
                             state.provenance);

    // The append sequence: quartets, values, (bounds), manifest, checksum,
    // flush. A failure before the manifest write leaves orphan rows on the
    // on-disk tables; the next append's mirror-derived placement truncates
    // them (the resizes shrink the tables), so the tables stay bit-identical
    // to the mirrors and the checksum keeps covering exactly the stored
    // state. A failure after the manifest write leaves a store whose
    // checksum the next Open refuses - a user-visible error.
    std::vector<std::uint64_t> appendedRows;
    appendedRows.reserve(quartets.size() * 4);

    for (const qcx::integrals::ShellQuartet& quartet : quartets)
    {
        appendedRows.push_back(quartet.i);
        appendedRows.push_back(quartet.j);
        appendedRows.push_back(quartet.k);
        appendedRows.push_back(quartet.l);
    }

    auto written = WrapH5([&]() {
        HighFive::DataSet quartetsTable = state.file.getDataSet("/integrals/ao/eri/quartets");
        quartetsTable.resize({firstQuartet + quartets.size(), 4});
        quartetsTable.select({firstQuartet, 0}, {quartets.size(), 4})
            .write_raw(appendedRows.data(), quartetsTable.getDataType());
    });

    if (!written.has_value())
    {
        return std::unexpected(written.error());
    }

    // Test-only failure injection (test_hooks.hpp): simulate the failure
    // between the quartets and values writes - the orphan-rows case that
    // the mirror-derived placement (above) recovers from. Only the C1
    // regression test arms this.
    if (internal::failAfterQuartetsWrite)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, "injected failure after the quartets write"});
    }

    written = WrapH5([&]() {
        HighFive::DataSet valuesTable = state.file.getDataSet(
            kCertified ? "/integrals/ao/eri/fp32/values" : "/integrals/ao/eri/fp64/values");
        valuesTable.resize({valuesExtent + values.size()});
        valuesTable.select({valuesExtent}, {values.size()}).write(values);
    });

    if (!written.has_value())
    {
        return std::unexpected(written.error());
    }

    if constexpr (kCertified)
    {
        written = WrapH5([&]() {
            HighFive::DataSet boundsTable = state.file.getDataSet("/integrals/ao/eri/fp32/bounds");
            boundsTable.resize({boundsExtent + errorBounds.size()});
            boundsTable.select({boundsExtent}, {errorBounds.size()}).write(errorBounds);
        });

        if (!written.has_value())
        {
            return std::unexpected(written.error());
        }
    }

    written = WrapH5([&]() {
        HighFive::DataSet manifestTable = state.file.getDataSet(
            kCertified ? "/integrals/ao/eri/fp32/manifest" : "/integrals/ao/eri/fp64/manifest");
        manifestTable.resize({chunkId + 1, kColumns});
        // The row's tail (the fp32-only columns) must not be written into
        // the fp64 lane - write exactly the kColumns-sized slice.
        const std::vector<std::uint64_t> rowSlice(row.begin(),
                                                  row.begin() + static_cast<long long>(kColumns));
        manifestTable.select({chunkId, 0}, {1, kColumns}).write(rowSlice);
    });

    if (!written.has_value())
    {
        return std::unexpected(written.error());
    }

    written = WrapH5([&]() {
        HighFive::DataSet checksumTable = state.file.getDataSet("/metadata/store_checksum");
        checksumTable.write(newChecksum);
    });

    if (!written.has_value())
    {
        return std::unexpected(written.error());
    }

    written = WrapH5([&]() { state.file.flush(); });

    if (!written.has_value())
    {
        return std::unexpected(written.error());
    }

    // Commit the mirrors + lookups.
    state.quartets = std::move(newQuartets);

    if constexpr (kCertified)
    {
        state.fp32Manifest = std::move(newManifest);
    } else
    {
        state.fp64Manifest = std::move(newManifest);
    }

    auto& lookup = kCertified ? state.fp32Lookup : state.fp64Lookup;

    for (std::size_t position = 0; position < quartets.size(); ++position)
    {
        const qcx::integrals::ShellQuartet& quartet = quartets[position];
        lookup.emplace(QuartetKey{quartet.i, quartet.j, quartet.k, quartet.l},
                       BlockLocation{chunkId, position});
    }

    return {};
}

// The shared load for both precisions: locates every requested quartet in
// the manifest-derived lookup, reads each touched chunk ONCE, verifies the
// chunk checksum, and extracts the blocks in request order.
template <bool kCertified>
qcx::Result<
    std::conditional_t<kCertified, qcx::integrals::CertifiedBatch, qcx::integrals::EriBatch>>
LoadImpl(const EriStore::State& state, const std::vector<CanonicalQuartetInfo>& request) {
    using ValueElement = std::conditional_t<kCertified, float, double>;
    using BatchType =
        std::conditional_t<kCertified, qcx::integrals::CertifiedBatch, qcx::integrals::EriBatch>;
    constexpr std::size_t kColumns = kCertified ? kManifestFp32Columns : kManifestFp64Columns;

    if (request.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the quartet list is empty"});
    }

    const auto& manifest = kCertified ? state.fp32Manifest : state.fp64Manifest;
    const auto& lookup = kCertified ? state.fp32Lookup : state.fp64Lookup;

    // Pass 1: locate + validate every quartet.
    std::vector<BlockLocation> located;
    located.reserve(request.size());
    std::vector<std::size_t> blockSizes;
    blockSizes.reserve(request.size());
    std::size_t totalElements = 0;

    for (const CanonicalQuartetInfo& info : request)
    {
        const auto it =
            lookup.find(QuartetKey{info.quartet.i, info.quartet.j, info.quartet.k, info.quartet.l});

        if (it == lookup.end())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "quartet not stored in this store"});
        }

        // The chunk stamp gate: a chunk written under a different engine
        // version is never served (the fp32 a-priori bound is only valid
        // for the pipeline that produced it; fp64 same rule).
        const std::uint64_t* row = manifest.data() + it->second.chunkId * kColumns;

        if (row[kEngineVersionColumn] != qcx::integrals::kIntegralEngineVersion ||
            row[kSchemaVersionColumn] != kStoreSchemaVersion)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                              "chunk written under a different engine version"});
        }

        const std::size_t blockSize = BlockElementCount(state.pairList, info.quartet);
        located.push_back(it->second);
        blockSizes.push_back(blockSize);
        totalElements += blockSize;
    }

    // Request-order output offsets (the values are packed per computed
    // quartet in the request's canonical order).
    std::vector<std::size_t> requestOffsets(request.size(), 0);

    for (std::size_t i = 1; i < request.size(); ++i)
    {
        requestOffsets[i] = requestOffsets[i - 1] + blockSizes[i - 1];
    }

    // Group by chunk in first-appearance order.
    std::vector<std::size_t> chunkOrder;
    std::map<std::size_t, std::vector<std::size_t>> chunkGroups;

    for (std::size_t i = 0; i < request.size(); ++i)
    {
        const std::size_t chunkId = located[i].chunkId;

        if (chunkGroups.count(chunkId) == 0)
        {
            chunkOrder.push_back(chunkId);
        }

        chunkGroups[chunkId].push_back(i);
    }

    BatchType result;
    result.computed.reserve(request.size());

    for (const CanonicalQuartetInfo& info : request)
    {
        result.computed.push_back(info.quartet);
    }

    result.values.resize(totalElements);

    if constexpr (kCertified)
    {
        result.errorBounds.resize(request.size());
    }

    const std::string valuesPath =
        kCertified ? "/integrals/ao/eri/fp32/values" : "/integrals/ao/eri/fp64/values";

    for (const std::size_t chunkId : chunkOrder)
    {
        const std::uint64_t* row = manifest.data() + chunkId * kColumns;
        const std::size_t chunkOffsetElements = row[kValueOffsetColumn] / sizeof(ValueElement);
        const std::size_t chunkElements = row[kValueBytesColumn] / sizeof(ValueElement);
        std::vector<ValueElement> chunkValues(chunkElements);

        auto read = WrapH5([&]() {
            state.file.getDataSet(valuesPath)
                .select({chunkOffsetElements}, {chunkElements})
                .read(chunkValues);
        });

        if (!read.has_value())
        {
            return std::unexpected(read.error());
        }

        const std::uint64_t actualChecksum =
            internal::Fnv1a64(reinterpret_cast<const std::byte*>(chunkValues.data()),
                              chunkValues.size() * sizeof(ValueElement));

        if (actualChecksum != row[kChecksumColumn])
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kIOError, "chunk checksum mismatch (corrupt store)"});
        }

        // The block offset of each stored quartet of the chunk (the run's
        // packed prefix sums).
        const std::size_t firstQuartet = row[kFirstQuartetColumn];
        const std::size_t quartetCount = row[kQuartetCountColumn];
        std::vector<std::size_t> blockOffsets(quartetCount, 0);

        for (std::size_t position = 1; position < quartetCount; ++position)
        {
            const std::uint64_t* storedQuartet =
                state.quartets.data() + (firstQuartet + position - 1) * 4;
            blockOffsets[position] =
                blockOffsets[position - 1] +
                BlockElementCount(
                    state.pairList,
                    {storedQuartet[0], storedQuartet[1], storedQuartet[2], storedQuartet[3]});
        }

        for (const std::size_t i : chunkGroups[chunkId])
        {
            const std::size_t position = located[i].position;
            std::memcpy(&result.values[requestOffsets[i]],
                        &chunkValues[blockOffsets[position]],
                        blockSizes[i] * sizeof(ValueElement));
        }

        if constexpr (kCertified)
        {
            const std::size_t boundsOffset = row[kBoundsOffsetColumn] / sizeof(double);
            const std::size_t boundsElements = row[kBoundsBytesColumn] / sizeof(double);
            std::vector<double> chunkBounds(boundsElements);

            auto readBounds = WrapH5([&]() {
                state.file.getDataSet("/integrals/ao/eri/fp32/bounds")
                    .select({boundsOffset}, {boundsElements})
                    .read(chunkBounds);
            });

            if (!readBounds.has_value())
            {
                return std::unexpected(readBounds.error());
            }

            // The bounds dataset is outside the store checksum (like the
            // values chunks): its own per-chunk checksum covers it on serve
            // An unverified bound would silently under-report the
            // certified error - the wrong kind of corruption for a
            // certified-precision pipeline.
            const std::uint64_t actualBoundsChecksum =
                internal::Fnv1a64(reinterpret_cast<const std::byte*>(chunkBounds.data()),
                                  chunkBounds.size() * sizeof(double));

            if (actualBoundsChecksum != row[kBoundsChecksumColumn])
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                                  "certified bounds checksum mismatch (corrupt "
                                                  "store)"});
            }

            for (const std::size_t i : chunkGroups[chunkId])
            {
                result.errorBounds[i] = chunkBounds[located[i].position];
            }
        }
    }

    return result;
}

} // namespace

EriStore::EriStore(std::shared_ptr<State> state) : _state(std::move(state)) {}

qcx::Result<EriStore> EriStore::Create(const std::filesystem::path& path,
                                       const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       std::string_view orbitalBasisName,
                                       std::string_view auxBasisName,
                                       const StoreOptions& options) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    // The store is append-only and Create owns fresh files only: refuse to
    // clobber an existing file (Open is the read-write entry point). The
    // existence check runs before the open, so the Truncate flag is never
    // set - a plain ReadWrite|Create open would silently reuse a stale
    // file, and truncation would destroy it.
    if (std::filesystem::exists(path))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, "the store file already exists"});
    }

    std::shared_ptr<State> state;
    auto openResult = WrapH5([&]() {
        state = std::make_shared<State>(
            State{HighFive::File(path.string(), HighFive::File::ReadWrite | HighFive::File::Create),
                  std::move(*pairList),
                  std::string{}});
    });

    if (!openResult.has_value())
    {
        return std::unexpected(openResult.error());
    }

    auto moleculeResult =
        internal::WriteMoleculeGroup(state->file, molecule, orbitalBasisName, auxBasisName);

    if (!moleculeResult.has_value())
    {
        return std::unexpected(moleculeResult.error());
    }

    state->fingerprint = *moleculeResult;

    if (auto tables = CreateEriTables(state->file); !tables.has_value())
    {
        return std::unexpected(tables.error());
    }

    if (auto slots = CreateDerivedSlots(state->file); !slots.has_value())
    {
        return std::unexpected(slots.error());
    }

    // /one_electron exists as a group; its datasets appear at
    // WriteOneElectron (the schema's lazy-create rule).
    if (auto oneElectron = WrapH5([&]() { state->file.createGroup("one_electron"); });
        !oneElectron.has_value())
    {
        return std::unexpected(oneElectron.error());
    }

    if (auto metadata = WriteMetadata(state->file, options, {}, {}, {}); !metadata.has_value())
    {
        return std::unexpected(metadata.error());
    }

    state->screeningPreset = PresetName(options.accuracy);
    state->schwarzThreshold = qcx::integrals::SchwarzThreshold(options.accuracy);
    state->densityThreshold = qcx::integrals::DensityThreshold(options.accuracy);
    state->mixedPrecisionThreshold = qcx::integrals::MixedPrecisionThreshold(options.accuracy);
    state->provenance = options.provenance;

    if (auto flush = WrapH5([&]() { state->file.flush(); }); !flush.has_value())
    {
        return std::unexpected(flush.error());
    }

    return EriStore(std::move(state));
}

qcx::Result<EriStore> EriStore::Open(const std::filesystem::path& path,
                                     const qcx::molecule::Molecule& molecule,
                                     const qcx::basisset::BasisSet& basisSet,
                                     std::string_view orbitalBasisName,
                                     std::string_view auxBasisName,
                                     const StoreOptions& options) {
    (void)options;

    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    std::shared_ptr<State> state;
    auto openResult = WrapH5([&]() {
        state =
            std::make_shared<State>(State{HighFive::File(path.string(), HighFive::File::ReadWrite),
                                          std::move(*pairList),
                                          std::string{}});
    });

    if (!openResult.has_value())
    {
        // A truncated file fails here (superblock/EOF damage) - kIOError.
        return std::unexpected(openResult.error());
    }

    // Fingerprint + every readable /molecule field, element-wise.
    auto moleculeResult =
        internal::VerifyMoleculeGroup(state->file, molecule, orbitalBasisName, auxBasisName);

    if (!moleculeResult.has_value())
    {
        return std::unexpected(moleculeResult.error());
    }

    state->fingerprint = *moleculeResult;

    // The append-only tables -> mirrors.
    auto quartets = ReadU64Table(state->file, "/integrals/ao/eri/quartets");

    if (!quartets.has_value())
    {
        return std::unexpected(quartets.error());
    }

    auto fp64Manifest = ReadU64Table(state->file, "/integrals/ao/eri/fp64/manifest");

    if (!fp64Manifest.has_value())
    {
        return std::unexpected(fp64Manifest.error());
    }

    auto fp32Manifest = ReadU64Table(state->file, "/integrals/ao/eri/fp32/manifest");

    if (!fp32Manifest.has_value())
    {
        return std::unexpected(fp32Manifest.error());
    }

    state->quartets = std::move(*quartets);
    state->fp64Manifest = std::move(*fp64Manifest);
    state->fp32Manifest = std::move(*fp32Manifest);

    // /metadata + the store checksum.
    if (auto metadata = ReadAndVerifyMetadata(state->file, *state); !metadata.has_value())
    {
        return std::unexpected(metadata.error());
    }

    state->fp64Lookup = BuildLookup<kManifestFp64Columns>(state->quartets, state->fp64Manifest);
    state->fp32Lookup = BuildLookup<kManifestFp32Columns>(state->quartets, state->fp32Manifest);

    return EriStore(std::move(state));
}

qcx::Result<void> EriStore::WriteOneElectron(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& overlap,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& kinetic,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& nuclear) {
    State& state = *_state;
    const std::size_t n = state.pairList.functionCount;
    const auto isSquareShape = [n](const auto& tensor) {
        return tensor.Shape() == std::array<std::size_t, 2>{n, n};
    };

    if (!isSquareShape(overlap) || !isSquareShape(kinetic) || !isSquareShape(nuclear))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the one-electron matrices must be {n, n} with n the "
                                          "basis function count"});
    }

    // Append-only: refuse when any matrix is already present.
    bool alreadyStored = false;
    auto existsCheck = WrapH5([&]() {
        HighFive::Group oneElectron = state.file.getGroup("one_electron");
        alreadyStored = oneElectron.exist("overlap") || oneElectron.exist("kinetic") ||
                        oneElectron.exist("nuclear_attraction");
    });

    if (!existsCheck.has_value())
    {
        return std::unexpected(existsCheck.error());
    }

    if (alreadyStored)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the one-electron matrices are already stored"});
    }

    // The tensor is row-major, HDF5 is row-major - the host view is the
    // on-disk layout (no transposition at this boundary). The tensor comes
    // in as const (HostView is mutable-only), so flatten element-wise.
    const auto writeMatrix = [&](const std::string& name, const auto& tensor) {
        std::vector<double> flat;
        flat.reserve(n * n);

        for (std::size_t row = 0; row < n; ++row)
        {
            for (std::size_t col = 0; col < n; ++col)
            {
                flat.push_back(tensor(row, col));
            }
        }

        // The matrices are fixed-size ({n, n}): a chunk layout is illegal
        // here (HDF5 requires chunk <= maximum dims for fixed dimensions),
        // and contiguous storage is the right layout for a small matrix.
        // write() would reject the flat buffer against the rank-2 dataspace
        // (its dimension check only squeezes singleton dims).
        return WrapH5([&]() {
            HighFive::DataSet dataSet =
                state.file.createDataSet("one_electron/" + name,
                                         HighFive::DataSpace({n, n}, {n, n}),
                                         HighFive::AtomicType<double>());
            dataSet.write_raw(flat.data(), dataSet.getDataType());
        });
    };

    if (auto written = writeMatrix("overlap", overlap); !written.has_value())
    {
        return std::unexpected(written.error());
    }

    if (auto written = writeMatrix("kinetic", kinetic); !written.has_value())
    {
        return std::unexpected(written.error());
    }

    if (auto written = writeMatrix("nuclear_attraction", nuclear); !written.has_value())
    {
        return std::unexpected(written.error());
    }

    return WrapH5([&]() { state.file.flush(); });
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> EriStore::ReadOverlapMatrix()
    const {
    return ReadOneElectron(*_state, "overlap");
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> EriStore::ReadKineticMatrix()
    const {
    return ReadOneElectron(*_state, "kinetic");
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>
EriStore::ReadNuclearAttractionMatrix() const {
    return ReadOneElectron(*_state, "nuclear_attraction");
}

qcx::Result<void> EriStore::AppendBatch(const std::vector<ShellQuartet>& quartets,
                                        const std::vector<double>& values) {
    return AppendImpl<false>(*_state, quartets, values, {});
}

qcx::Result<EriBatch> EriStore::LoadBatch(const std::vector<CanonicalQuartetInfo>& request) const {
    return LoadImpl<false>(*_state, request);
}

qcx::Result<void> EriStore::AppendCertifiedBatch(const std::vector<ShellQuartet>& quartets,
                                                 const std::vector<float>& values,
                                                 const std::vector<double>& errorBounds) {
    return AppendImpl<true>(*_state, quartets, values, errorBounds);
}

qcx::Result<CertifiedBatch> EriStore::LoadCertifiedBatch(
    const std::vector<CanonicalQuartetInfo>& request) const {
    return LoadImpl<true>(*_state, request);
}

bool EriStore::ContainsFp64(const ShellQuartet& quartet) const noexcept {
    return _state->fp64Lookup.contains(QuartetKey{quartet.i, quartet.j, quartet.k, quartet.l});
}

bool EriStore::ContainsFp32(const ShellQuartet& quartet) const noexcept {
    return _state->fp32Lookup.contains(QuartetKey{quartet.i, quartet.j, quartet.k, quartet.l});
}

std::size_t EriStore::StoredChunkCount() const noexcept {
    return _state->fp64Manifest.size() / kManifestFp64Columns;
}

std::size_t EriStore::StoredQuartetCount() const noexcept {
    return _state->quartets.size() / 4;
}

} // namespace qcx::storage
