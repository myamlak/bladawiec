// HighFive throws std::exception-derived errors on every failure; qcx is
// exception-free (core/include/qcx/error.hpp). Translate at the module
// boundary. HDF5 failures are kIOError; non-HDF5 exceptions (bad_alloc)
// are kInternalError.

#pragma once

#include "qcx/error.hpp"

#include <cstddef>
#include <cstdint>
#include <highfive/H5Group.hpp>
#include <string>
#include <string_view>

namespace qcx::storage {

// The fixed byte width of every schema string dataset/attribute:
// HDF5 variable-length strings (the HighFive AtomicType<std::string>
// default) crash this MSVC build's vlen read path against HDF5 2.2.0, so
// all schema strings are fixed-width char arrays instead. 256 bytes
// covers the longest field (provenance tags, basis names).
inline constexpr std::size_t kMaxSchemaStringLength = 256;

template <typename Fn> qcx::Result<void> WrapH5(Fn&& fn) {
    try
    {
        fn();
        return {};
    } catch (const HighFive::Exception& e)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, std::string("HDF5: ") + e.what()});
    } catch (const std::exception& e)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, std::string("HDF5: ") + e.what()});
    }
}

// The value-returning twin: the lambda produces the object (a group or
// dataset opened by the HighFive call) and its result is handed back on
// success. Writers open into the return value instead of the deprecated
// declaration-then-assign default construction (HighFive's default
// constructors create unsafe uninitialized objects).
template <typename Fn> auto WrapH5Value(Fn&& fn) -> qcx::Result<decltype(fn())> {
    try
    { return fn(); } catch (const HighFive::Exception& e)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, std::string("HDF5: ") + e.what()});
    } catch (const std::exception& e)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, std::string("HDF5: ") + e.what()});
    }
}

// Scalar read/write helpers shared across the module's storage writers.
// Each wraps the HighFive call in WrapH5 so callers never see an
// exception; failures surface as qcx::Error. Every string helper reads
// and writes a fixed-length (kMaxSchemaStringLength) char dataset - the
// vlen string machinery is unusable on this build (see the constant
// above) - so the value must fit the fixed width or the write fails.

qcx::Result<std::string> ReadScalarString(HighFive::Group& group, const std::string& name);

qcx::Result<void> WriteScalarString(HighFive::Group& group,
                                    const std::string& name,
                                    std::string_view value);

// The attribute twin (the /derived staging-slot source_fingerprint
// attribute is fixed-length for the same reason).

qcx::Result<void> WriteScalarStringAttribute(HighFive::DataSet& dataSet,
                                             const std::string& name,
                                             std::string_view value);

} // namespace qcx::storage
