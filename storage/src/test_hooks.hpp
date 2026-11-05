// Test-only failure injection for the store's append path (the C1
// regression test in tests/eri_store_test.cpp). Not part of the schema or
// the public API: arming this makes AppendImpl return kIOError between the
// quartets and values writes, simulating the orphan-rows failure that the
// mirror-derived append placement recovers from (see the AppendImpl
// comment in eri_store.cpp). The flag lives in internal so only the module
// and its tests can reach it.

#pragma once

namespace qcx::storage::internal {

// The append failure-injection flag: when true, the next AppendBatch fails
// after its quartets write (kIOError). Consumed by AppendImpl; set and
// reset by the C1 regression test around the failed append only.
//
// The storage lives in the library and the two sides reach it through these
// entries: a header-inline variable would be one copy per image in the
// composed shared build, so the test would arm its own copy while the
// library's AppendImpl read the untouched one.

/// True while the next AppendBatch must fail after its quartets write.
/// \returns The injection flag.
bool FailAfterQuartetsWrite() noexcept;

/// Arms or clears the append failure injection.
/// \param armed true to fail the next AppendBatch after its quartets write.
void SetFailAfterQuartetsWrite(bool armed) noexcept;

} // namespace qcx::storage::internal
