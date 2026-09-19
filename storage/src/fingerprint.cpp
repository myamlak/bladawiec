// The deterministic system fingerprint (fingerprint.hpp): serializes
// the canonical byte layout of the system into a vector of bytes with
// explicit per-field memcpys (no struct reinterpret_cast - padding would
// make the hash layout-dependent), then FNV-1a-64's and hex-encodes it.
// Also hosts the /molecule group writer and verifier (store_molecule.hpp)
// - the readable double-check lives with the hash it mirrors.

#include "qcx/storage/fingerprint.hpp"

#include "hdf5_util.hpp"
#include "store_molecule.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace qcx::storage {

namespace {

// The 8-byte magic that opens every hashed layout (fingerprint.hpp
// documents the offsets; keep in sync).
inline constexpr std::string_view kStoreMagic = "QCXSTOR1";

// Lowercase hex encoding, 16 chars for the 8-byte hash.
inline constexpr std::array<char, 16> kHexDigits = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

void AppendBytes(std::vector<std::byte>& out, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::byte*>(data);
    out.insert(out.end(), bytes, bytes + size);
}

void AppendString(std::vector<std::byte>& out, std::string_view value) {
    const std::uint32_t length = static_cast<std::uint32_t>(value.size());
    AppendBytes(out, &length, sizeof(length));
    AppendBytes(out, value.data(), value.size());
}

std::string HexEncode(std::uint64_t hash) {
    std::string hex(16, '0');

    for (std::size_t i = 0; i < 16; ++i)
    {
        hex[15 - i] = kHexDigits[static_cast<unsigned int>(hash >> (4 * i)) & 0xF];
    }

    return hex;
}

// The element-count formula of the readable /molecule fields, in
// first-occurrence element order with per-element counts ("O1 H2" for the
// H2O fixture - element symbol then count, space-joined).
std::string ComputeFormula(const qcx::molecule::Molecule& molecule) {
    std::string formula;
    std::vector<int> countedZ;

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        if (std::find(countedZ.begin(), countedZ.end(), atom.atomicNumber) != countedZ.end())
        {
            continue;
        }

        if (!formula.empty())
        {
            formula += ' ';
        }

        const std::size_t count = std::count_if(molecule.Atoms().begin(),
                                                molecule.Atoms().end(),
                                                [&](const qcx::molecule::Atom& other) {
                                                    return other.atomicNumber == atom.atomicNumber;
                                                });
        formula += atom.symbol;
        formula += std::to_string(count);
        countedZ.push_back(atom.atomicNumber);
    }

    return formula;
}

} // namespace

std::string ComputeFingerprint(const qcx::molecule::Molecule& molecule,
                               std::string_view orbitalBasisName,
                               std::string_view auxBasisName) {
    std::vector<std::byte> bytes;
    bytes.reserve(64 + molecule.AtomCount() * 28 + orbitalBasisName.size() + auxBasisName.size());

    AppendBytes(bytes, kStoreMagic.data(), kStoreMagic.size());
    const std::uint32_t schemaVersion = kStoreSchemaVersion;
    AppendBytes(bytes, &schemaVersion, sizeof(schemaVersion));
    const std::uint32_t engineVersion = qcx::integrals::kIntegralEngineVersion;
    AppendBytes(bytes, &engineVersion, sizeof(engineVersion));
    const std::int32_t charge = molecule.Charge();
    AppendBytes(bytes, &charge, sizeof(charge));
    const std::int32_t multiplicity = molecule.Multiplicity();
    AppendBytes(bytes, &multiplicity, sizeof(multiplicity));
    const std::uint32_t atomCount = static_cast<std::uint32_t>(molecule.AtomCount());
    AppendBytes(bytes, &atomCount, sizeof(atomCount));

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        const std::int32_t z = atom.atomicNumber;
        AppendBytes(bytes, &z, sizeof(z));
    }

    // Coordinates as raw bits, row-major over atoms then {x, y, z}.
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coordinates =
        molecule.CoordinatesBohr();

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const double value = coordinates(atom, axis);
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            AppendBytes(bytes, &bits, sizeof(bits));
        }
    }

    AppendString(bytes, orbitalBasisName);
    AppendString(bytes, auxBasisName);

    return HexEncode(internal::Fnv1a64(bytes.data(), bytes.size()));
}

namespace internal {

qcx::Result<std::string> WriteMoleculeGroup(HighFive::File& file,
                                            const qcx::molecule::Molecule& molecule,
                                            std::string_view orbitalBasisName,
                                            std::string_view auxBasisName) {
    const std::string fingerprint = ComputeFingerprint(molecule, orbitalBasisName, auxBasisName);
    auto group = WrapH5Value([&]() { return file.createGroup("molecule"); });

    if (!group.has_value())
    {
        return std::unexpected(group.error());
    }

    // The strings are fixed-width char datasets (hdf5_util.hpp); each
    // helper reports its own error.
    if (auto written = WriteScalarString(*group, "fingerprint", fingerprint); !written.has_value())
    {
        return std::unexpected(written.error());
    }

    if (auto written = WriteScalarString(*group, "formula", ComputeFormula(molecule));
        !written.has_value())
    {
        return std::unexpected(written.error());
    }

    if (auto written = WriteScalarString(*group, "orbital_basis_name", orbitalBasisName);
        !written.has_value())
    {
        return std::unexpected(written.error());
    }

    if (auto written = WriteScalarString(*group, "aux_basis_name", auxBasisName);
        !written.has_value())
    {
        return std::unexpected(written.error());
    }

    return WrapH5([&]() {
               group->createDataSet("charge", molecule.Charge());
               group->createDataSet("multiplicity", molecule.Multiplicity());

               std::vector<int> atomZ;

               for (const qcx::molecule::Atom& atom : molecule.Atoms())
               {
                   atomZ.push_back(atom.atomicNumber);
               }

               group->createDataSet("atom_z", atomZ);

               // Row-major over atoms then {x, y, z} - the same layout the hash
               // consumes (memcpy per element, never arithmetic).
               const auto& coordinates = molecule.CoordinatesBohr();
               std::vector<double> flat;
               flat.reserve(molecule.AtomCount() * 3);

               for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
               {
                   for (std::size_t axis = 0; axis < 3; ++axis)
                   {
                       flat.push_back(coordinates(atom, axis));
                   }
               }

               group->createDataSet("coordinates_bohr", flat);
               group->createDataSet("engine_version", qcx::integrals::kIntegralEngineVersion);
               group->createDataSet("schema_version", kStoreSchemaVersion);
           })
        .and_then([&]() -> qcx::Result<std::string> { return fingerprint; });
}

qcx::Result<std::string> VerifyMoleculeGroup(HighFive::File& file,
                                             const qcx::molecule::Molecule& molecule,
                                             std::string_view orbitalBasisName,
                                             std::string_view auxBasisName) {
    const std::string expectedFingerprint =
        ComputeFingerprint(molecule, orbitalBasisName, auxBasisName);
    std::string storedFingerprint;
    std::string storedFormula;
    std::string storedOrbitalName;
    std::string storedAuxName;
    int storedCharge = 0;
    int storedMultiplicity = 0;
    std::uint32_t storedEngineVersion = 0;
    std::uint32_t storedSchemaVersion = 0;
    std::vector<int> storedAtomZ;
    std::vector<double> storedCoordinates;

    auto group = WrapH5Value([&]() { return file.getGroup("molecule"); });

    if (!group.has_value())
    {
        return std::unexpected(group.error());
    }

    // The fixed-width string datasets read through the safe char-copy path.
    if (auto value = ReadScalarString(*group, "fingerprint"); value.has_value())
    {
        storedFingerprint = *value;
    } else
    {
        return std::unexpected(value.error());
    }

    if (auto value = ReadScalarString(*group, "formula"); value.has_value())
    {
        storedFormula = *value;
    } else
    {
        return std::unexpected(value.error());
    }

    if (auto value = ReadScalarString(*group, "orbital_basis_name"); value.has_value())
    {
        storedOrbitalName = *value;
    } else
    {
        return std::unexpected(value.error());
    }

    if (auto value = ReadScalarString(*group, "aux_basis_name"); value.has_value())
    {
        storedAuxName = *value;
    } else
    {
        return std::unexpected(value.error());
    }

    auto readResult = WrapH5([&]() {
        group->getDataSet("charge").read(storedCharge);
        group->getDataSet("multiplicity").read(storedMultiplicity);
        group->getDataSet("atom_z").read(storedAtomZ);
        group->getDataSet("coordinates_bohr").read(storedCoordinates);
        group->getDataSet("engine_version").read(storedEngineVersion);
        group->getDataSet("schema_version").read(storedSchemaVersion);
    });

    if (!readResult.has_value())
    {
        return std::unexpected(readResult.error());
    }

    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coordinates =
        molecule.CoordinatesBohr();
    std::vector<double> expectedFlat;
    expectedFlat.reserve(molecule.AtomCount() * 3);

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            expectedFlat.push_back(coordinates(atom, axis));
        }
    }

    const bool coordinatesMatch = storedCoordinates.size() == expectedFlat.size() &&
                                  std::memcmp(storedCoordinates.data(),
                                              expectedFlat.data(),
                                              storedCoordinates.size() * sizeof(double)) == 0;

    std::vector<int> expectedZ;

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        expectedZ.push_back(atom.atomicNumber);
    }

    const bool fieldsMatch =
        storedFingerprint == expectedFingerprint && storedFormula == ComputeFormula(molecule) &&
        storedCharge == molecule.Charge() && storedMultiplicity == molecule.Multiplicity() &&
        storedAtomZ == expectedZ && coordinatesMatch &&
        storedOrbitalName == std::string(orbitalBasisName) &&
        storedAuxName == std::string(auxBasisName) &&
        storedEngineVersion == qcx::integrals::kIntegralEngineVersion &&
        storedSchemaVersion == kStoreSchemaVersion;

    if (!fieldsMatch)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "store belongs to a different system"});
    }

    return storedFingerprint;
}

} // namespace internal
} // namespace qcx::storage
