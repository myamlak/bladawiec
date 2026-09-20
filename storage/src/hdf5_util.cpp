// The shared scalar read/write helpers (hdf5_util.hpp): one place where
// the module's metadata scalars cross the HighFive boundary. Every helper
// translates HighFive exceptions to qcx::Error through WrapH5. The string
// helpers read/write fixed-length char datasets (kMaxSchemaStringLength) -
// the vlen machinery is unusable on this build (see hdf5_util.hpp).

#include "hdf5_util.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

namespace qcx::storage {

namespace {

// The fixed-width read result: everything up to the first NUL.
std::string TrimAtNull(const char* buffer) {
    const char* end = std::find(buffer, buffer + kMaxSchemaStringLength, '\0');
    return std::string(buffer, end);
}

// The zero-padded fixed-width write buffer.
std::array<char, kMaxSchemaStringLength> PadToFixedWidth(std::string_view value) {
    std::array<char, kMaxSchemaStringLength> buffer{};
    std::copy(value.begin(), value.end(), buffer.begin());
    return buffer;
}

} // namespace

qcx::Result<std::string> ReadScalarString(HighFive::Group& group, const std::string& name) {
    std::array<char, kMaxSchemaStringLength> buffer{};

    auto read = WrapH5([&]() {
        HighFive::DataSet dataSet = group.getDataSet(name);
        dataSet.read_raw(buffer.data(), dataSet.getDataType());
    });

    if (!read.has_value())
    {
        return std::unexpected(read.error());
    }

    return TrimAtNull(buffer.data());
}

qcx::Result<void> WriteScalarString(HighFive::Group& group,
                                    const std::string& name,
                                    std::string_view value) {
    if (value.size() >= kMaxSchemaStringLength)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "schema string \"" + name + "\" exceeds the fixed 256-byte width"});
    }

    return WrapH5([&]() {
        std::array<char, kMaxSchemaStringLength> buffer = PadToFixedWidth(value);
        HighFive::DataSet dataSet = group.createDataSet(
            name, HighFive::DataSpace({1}), HighFive::AtomicType<char[kMaxSchemaStringLength]>());
        dataSet.write_raw(buffer.data(), dataSet.getDataType());
    });
}

qcx::Result<void> WriteScalarStringAttribute(HighFive::DataSet& dataSet,
                                             const std::string& name,
                                             std::string_view value) {
    if (value.size() >= kMaxSchemaStringLength)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "schema attribute \"" + name + "\" exceeds the fixed 256-byte width"});
    }

    return WrapH5([&]() {
        std::array<char, kMaxSchemaStringLength> buffer = PadToFixedWidth(value);
        HighFive::Attribute attribute = dataSet.createAttribute(
            name, HighFive::DataSpace({1}), HighFive::AtomicType<char[kMaxSchemaStringLength]>());
        attribute.write_raw(buffer.data(), attribute.getDataType());
    });
}

} // namespace qcx::storage
