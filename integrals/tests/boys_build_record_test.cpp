// The Boys kernel's build record, and the gate that stands behind it.
//
// Two things are pinned here. The record's first: BoysBuildRecord prints what
// the library reports, so the strings in it are the library's own and a
// revision that moves a default moves the text - which is the whole reason the
// record exists, an inherited default being otherwise visible only to a reader
// diffing two coefficient tables. The gate's second: this build asks for a
// multiply-add route and the arithmetic is delivered at one, and the two are
// compared rather than assumed. A build that states a route it does not run is
// a different arithmetic, not a different spelling.

#include "boys/backend.hpp"
#include "qcx/integrals/boys.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using qcx::integrals::BackendInfo;
using qcx::integrals::BoysBackends;
using qcx::integrals::BoysBuildRecord;
using qcx::integrals::kBoysMulAddRouteAskedFor;
using qcx::integrals::MulAddRoute;
using qcx::integrals::MulAddRouteName;
using qcx::integrals::VersionString;

// The route one arithmetic was delivered at, from the route this build asked
// for and that backend's own contraction measurement. The library's contract is
// that a contracting build compiles both routes to the fused step, so a
// separate request is delivered as fused there and the fused route is the one
// reported, because that is the arithmetic the values are computed at. Nothing
// else admits a disagreement with the request.
MulAddRoute DeliveredRoute(MulAddRoute asked, bool contracts) {
    if (asked == MulAddRoute::kSeparate && contracts)
    {
        return MulAddRoute::kFused;
    }

    return asked;
}

// The library's row for one arithmetic, found by the name the library gives
// that arithmetic rather than by a name written here a second time.
const BackendInfo* FindBackend(std::string_view name) {
    for (const BackendInfo& backend : BoysBackends())
    {
        if (std::string_view(backend.name) == name)
        {
            return &backend;
        }
    }

    return nullptr;
}

// The record as printed, carried through RecordProperty so a run that passes
// still shows the combination it passed under rather than only a run that
// fails.
TEST(BoysBuildRecordTest, TheRecordCarriesTheLibrarysOwnAnswers) {
    const std::string record = BoysBuildRecord();

    RecordProperty("boys_build_record", record);
    std::cout << record << "\n";

    // Each of these is what the library reports at the moment of the call, so a
    // record transcribed from a revision's table would stop matching the moment
    // one of them moved.
    EXPECT_NE(record.find(VersionString()), std::string::npos);
    EXPECT_NE(record.find(MulAddRouteName(kBoysMulAddRouteAskedFor)), std::string::npos);

    for (const BackendInfo& backend : BoysBackends())
    {
        EXPECT_NE(record.find(backend.name), std::string::npos) << backend.name;
        EXPECT_NE(record.find(MulAddRouteName(backend.route)), std::string::npos) << backend.name;
    }
}

// The route the engine's arithmetic is delivered at is the route this build
// asked for.
TEST(BoysBuildRecordTest, TheEnginesArithmeticRunsTheRouteThisBuildAskedFor) {
    const MulAddRoute asked = kBoysMulAddRouteAskedFor;

    // The scalar pair is the arithmetic the engine's Boys calls run: the entries
    // it calls are defined in the library's own unit, and these rows are
    // measured there.
    const BackendInfo* const fp64 = FindBackend(boys::backend::ScalarFp64::kName);
    const BackendInfo* const fp32 = FindBackend(boys::backend::ScalarFp32::kName);

    ASSERT_NE(fp64, nullptr);
    ASSERT_NE(fp32, nullptr);

    EXPECT_EQ(fp64->route, DeliveredRoute(asked, fp64->contracts))
        << "asked for " << MulAddRouteName(asked) << ", fp64 delivered "
        << MulAddRouteName(fp64->route);

    EXPECT_EQ(fp32->route, DeliveredRoute(asked, fp32->contracts))
        << "asked for " << MulAddRouteName(asked) << ", fp32 delivered "
        << MulAddRouteName(fp32->route);

    // A contracting arithmetic cannot report the separate route: the library
    // states the two differ exactly where a contracting build was asked for the
    // separate route, and there the fused route is the one reported.
    for (const BackendInfo& backend : BoysBackends())
    {
        if (backend.contracts)
        {
            EXPECT_EQ(backend.route, MulAddRoute::kFused) << backend.name;
        }
    }
}

} // namespace
