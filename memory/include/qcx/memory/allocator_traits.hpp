#pragma once

#include "qcx/backend/tags.hpp"
#include "qcx/error.hpp"

#include <cstddef>
#include <cstring>
#include <new>

namespace qcx::memory {

/// The assumed cache-line size in bytes of the first-touch pass: the stride
/// of one write-back per line. The pass therefore stores at least once into
/// every 64-B line and every (>= 4 KiB) page of the window, which is what
/// first-touch page placement keys on.
/// \ingroup qcx-memory
inline constexpr std::size_t kCacheLineBytes = 64;

/// Per-backend allocation, placement, and copy primitives.
///
/// DeviceBuffer allocates exclusively through this indirection, keeping the
/// exclusive-allocation-site promise honest as more backends land (see
/// cuda_allocator_traits.hpp).
/// \ingroup qcx-memory
/// \tparam Backend Execution backend; specialize this template per backend.
template <typename Backend> struct AllocatorTraits;

/// Allocation traits for the CPU backend: operator new/delete + memcpy.
/// \ingroup qcx-memory
template <> struct AllocatorTraits<qcx::backend::CpuTag> {
    /// Allocates \p bytes of raw host memory.
    /// \param bytes Number of bytes to allocate; never zero.
    /// \returns The allocated block, or an Error (kOutOfMemory) when the
    /// host allocator cannot satisfy the request.
    static qcx::Result<void*> Allocate(std::size_t bytes) {
        // nothrow so allocation failure follows the Result contract instead
        // of escaping as an exception.
        void* p = ::operator new(bytes, std::nothrow);

        if (p == nullptr)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kOutOfMemory, "host allocation failed"});
        }

        return p;
    }

    /// Releases memory previously returned by Allocate.
    /// \param p Host allocation to release (nullptr is a no-op).
    static void Deallocate(void* p) noexcept {
        ::operator delete(p);
    }

    /// First-touch pass over the window [\p p, \p p + \p bytes): writes
    /// each cache line's first byte back to itself (volatile, so the
    /// store is never elided).
    ///
    /// The write-back is the only non-destructive store pattern that still
    /// counts as a page access: an anonymous page is placed on the NUMA node
    /// of the thread that first accesses it and never migrates, so running
    /// this pass on the threads that will consume the block - BEFORE the
    /// block is filled - distributes its pages over the consuming team's
    /// nodes. Over freshly allocated (still uninitialized) memory the reads
    /// and writes are raw bytes, which is well-defined; over filled memory
    /// the pass restores every byte it reads, so it changes nothing.
    /// Callers stripe the block themselves (see TouchPagesAcrossTeam) or
    /// run this from one thread for a single-threaded consumer.
    /// \param p Window start; nullptr is a no-op.
    /// \param bytes Window size in bytes; 0 is a no-op.
    static void TouchPages(void* p, std::size_t bytes) noexcept {
        if (p == nullptr || bytes == 0)
        {
            return;
        }

        auto* line = static_cast<volatile std::byte*>(p);
        const std::size_t lineCount = (bytes + kCacheLineBytes - 1) / kCacheLineBytes;

        for (std::size_t i = 0; i < lineCount; ++i)
        {
            line[i * kCacheLineBytes] = line[i * kCacheLineBytes];
        }
    }

    /// Copies \p bytes into host memory \p dst from \p src (plain memcpy on the CPU).
    /// \param dst Destination host buffer.
    /// \param src Source buffer.
    /// \param bytes Number of bytes to copy.
    /// \returns Success (memcpy cannot fail on the CPU).
    static qcx::Result<void> CopyToHost(void* dst, const void* src, std::size_t bytes) {
        std::memcpy(dst, src, bytes);
        return {};
    }

    /// Copies \p bytes from host memory \p src into \p dst (plain memcpy on the CPU).
    /// \param dst Destination buffer.
    /// \param src Source host buffer.
    /// \param bytes Number of bytes to copy.
    /// \returns Success (memcpy cannot fail on the CPU).
    static qcx::Result<void> CopyFromHost(void* dst, const void* src, std::size_t bytes) {
        std::memcpy(dst, src, bytes);
        return {};
    }
};

} // namespace qcx::memory
