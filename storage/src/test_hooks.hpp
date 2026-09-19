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
inline bool failAfterQuartetsWrite = false;

} // namespace qcx::storage::internal
