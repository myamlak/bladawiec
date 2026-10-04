// The Boys kernel's build record (qcx/integrals/boys.hpp BoysBuildRecord).
//
// Every fact it prints is read from the library's own accessors at the moment
// of the call: the version, the multiply-add route the values were computed at,
// the arithmetic backends, the defaults the default entries instantiate, and
// the fit routes this build carries. Nothing here restates a table, so a
// library revision that moves a default moves this text, which is what makes an
// inherited default visible rather than silent.
//
// The three facts that are this tree's and not the library's - the submodule
// revision, the route this build asked for, and the labels the library gives no
// name accessor for - are marked as such where they are printed.

#include "boys/boys_device_tables.hpp"
#include "qcx/integrals/boys.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace qcx::integrals {
namespace {

// One fact: the label, the value, and the accessor the value came from. The
// third column is what makes the record checkable - a reader can find each
// number's source and confirm the record reports rather than restates.
std::string Fact(std::string_view label, std::string_view value, std::string_view source) {
    std::ostringstream line;
    line << "  " << std::left << std::setw(26) << label << std::setw(26) << value << " (" << source
         << ")";
    return line.str();
}

// The name the library's own fit-route report gives a route value. The rows
// carry it, so the record prints the library's spelling instead of one written
// here a second time.
std::string_view FitRouteLabel(boys::FitRoute route) {
    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
    {
        if (row.route == route)
        {
            return row.name;
        }
    }

    return "unknown";
}

// The two enumerations the library publishes no name accessor for. Their values
// are read from the library; only the enumerator's spelling is written here,
// and the record says so on the note line it closes with.
std::string_view BudgetLabel(boys::BoysBudget budget) {
    return budget == boys::BoysBudget::kFp16 ? "kFp16" : "kFloat";
}

std::string_view RegionBExpLabel(boys::RegionBExp exp) {
    return exp == boys::RegionBExp::kFast ? "kFast" : "kAccurate";
}

// A fit route's certified slot of the argument line, as the report's own row
// states it.
std::string FormatInterval(double lo, double hi, int stored) {
    std::ostringstream text;
    text << "[" << std::setprecision(4) << lo << ", " << std::setprecision(4) << hi << "), "
         << stored << " stored";
    return text.str();
}

// A reported error figure, in the form the accuracy reports use.
std::string FormatError(double error) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(3) << error;
    return text.str();
}

} // namespace

std::string BoysBuildRecord() {
    std::ostringstream record;
    record << "qcx-integrals Boys kernel: what this build is\n\n";

    record << "library\n";
    record << Fact("version", VersionString(), "boys::VersionString") << "\n";
    record << Fact("submodule revision", QcxBoysPin, "this build, captured at configure time")
           << "\n";
    record << Fact("multiply-add asked for",
                   MulAddRouteName(kBoysMulAddRouteAskedFor),
                   "QCX_INTEGRALS_BOYS_MULADD_SEPARATE")
           << "\n";
    record
        << "  The asked-for route is what this tree's build states, and the guards in\n"
           "  qcx/integrals/boys.hpp refuse a translation unit whose definitions disagree with "
           "it.\n"
           "  What the arithmetic is delivered by is a different question - a compiler flag is a\n"
           "  licence rather than an instruction - and the table below is the library's own\n"
           "  answer, measured where the kernels are. boys_build_record_test fails when the two\n"
           "  disagree.\n\n";

    record
        << "arithmetic backends (boys::backend::BoysBackends)\n"
           "  one row per arithmetic: the route its values were computed at, and whether a bare\n"
           "  product-plus-add in it is one rounding in this build\n";

    for (const BackendInfo& backend : BoysBackends())
    {
        record << Fact(backend.name,
                       MulAddRouteName(backend.route),
                       backend.contracts ? "contraction measured" : "no contraction")
               << "\n";
    }

    record << "\nthe five axes this tree does not name (the entries take the library's own)\n";
    record << Fact("fit route",
                   FitRouteLabel(boys::DefaultPolicyFp64::kRoute),
                   "boys::DefaultPolicyFp64::kRoute")
           << "\n";
    record << Fact("evaluation scheme",
                   boys::EvalSchemeName(boys::DefaultPolicyFp64::kScheme),
                   "boys::DefaultPolicyFp64::kScheme")
           << "\n";
    record << Fact("interval granularity",
                   boys::GranularityName(boys::DefaultPolicyFp64::kGranularity),
                   "boys::DefaultPolicyFp64::kGranularity")
           << "\n";
    record << Fact("packing axis",
                   boys::PackAxisName(boys::DefaultPolicyFp64::kPack),
                   "boys::DefaultPolicyFp64::kPack")
           << "\n";
    record << Fact("engine budget",
                   BudgetLabel(boys::DefaultPolicyFp64::kBudget),
                   "boys::DefaultPolicyFp64::kBudget")
           << "\n";
    record << Fact("half-lane budget",
                   BudgetLabel(boys::DefaultPolicyFp16::kBudget),
                   "boys::DefaultPolicyFp16::kBudget")
           << "\n";
    record
        << "  These are the policy the default entries instantiate, so a revision that moves one\n"
           "  of them moves this text. This tree names none of them on purpose: the library's own\n"
           "  measured defaults are the ones its bounds are certified for, and a seam this tree\n"
           "  maintained would be a second place for them to be wrong.\n\n";

    record << "certified fit routes this build carries (boys::BoysFitRoutes)\n";

    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
    {
        record << Fact(row.name,
                       FormatInterval(row.lo, row.hi, row.stored),
                       "delivered " + FormatError(row.delivered) + " <= bound " +
                           FormatError(row.bound))
               << "\n";
    }

    record
        << "\nregion-B exponential: the device lane's option, and not one of the five axes above\n";
    record << Fact("library default",
                   RegionBExpLabel(boys::kDefaultRegionBExp),
                   "boys::kDefaultRegionBExp")
           << "\n";
    record
        << "  The library's per-precision default table has no row for this axis, and no\n"
           "  default entry of this surface carries it. This tree names no value for it, so the\n"
           "  constant above is the library's and a revision that moves it moves what this line\n"
           "  prints - the same exposure the five axes above have, stated here because the axis\n"
           "  is not one of them.\n\n";

    record << "Two labels above are spelled by this file rather than read, because the library\n"
              "publishes no name accessor for those enumerations: the engine budget and the\n"
              "region-B exponential. Every value beside them is the library's.\n";

    return record.str();
}

} // namespace qcx::integrals
