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

// The two labels the library publishes no name accessor for. Their values are
// read from the library; only the enumerator's spelling is written here, and the
// record says so on the note line it closes with.
std::string_view BudgetLabel(boys::BoysBudget budget) {
    return budget == boys::BoysBudget::kFp16 ? "kFp16" : "kFloat";
}

std::string_view RowBasisLabel(boys::RowBasis basis) {
    return basis == boys::RowBasis::kMeasured ? "measured" : "chosen";
}

// The seven axes one class's unnamed call compiles, read from the library's own
// default-policy table through DefaultPolicyFor. This is the answer an entry's
// policy parameter defaults to - not SeamPolicy<>, which composes the seam's
// names and is printed below as the fallback it is.
template <boys::Precision kPrecision, boys::Shape kShape, boys::Device kDevice>
std::string ClassRowFacts(std::string_view row) {
    using PolicyRow = boys::DefaultPolicyFor<kPrecision, kShape, kDevice>;
    using Policy = typename PolicyRow::Type;
    const std::string source(row);

    std::ostringstream text;
    text << Fact("fit route", FitRouteLabel(Policy::kRoute), source + "::Type::kRoute") << "\n";
    text << Fact("evaluation scheme",
                 boys::EvalSchemeName(Policy::kScheme),
                 source + "::Type::kScheme")
         << "\n";
    text << Fact("interval granularity",
                 boys::GranularityName(Policy::kGranularity),
                 source + "::Type::kGranularity")
         << "\n";
    text << Fact("packing axis", boys::PackAxisName(Policy::kPack), source + "::Type::kPack")
         << "\n";
    text << Fact("engine budget", BudgetLabel(Policy::kBudget), source + "::Type::kBudget") << "\n";
    text << Fact("division form",
                 boys::DivisionFormName(Policy::kDivision),
                 source + "::Type::kDivision")
         << "\n";
    text << Fact("region-B exponential",
                 boys::RegionBExpName(Policy::kRegionBExp),
                 source + "::Type::kRegionBExp")
         << "\n";
    text << Fact("row basis", RowBasisLabel(PolicyRow::kBasis), source + "::kBasis") << "\n";
    text << Fact("row record",
                 PolicyRow::kRecord[0] == '\0' ? "(none: a stated choice)" : PolicyRow::kRecord,
                 source + "::kRecord")
         << "\n";
    return text.str();
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

    // An entry's policy parameter defaults to its class's row (boys/kernels.hpp,
    // DefaultPolicy<Precision, Shape>), so these are what the engine runs. The
    // seam's seven names below are not: they are the fallback a build with no
    // row list composes every class from, and the row replaces them.
    record << "\nthe class rows an unnamed call runs (boys/boys_build_defaults.hpp)\n";
    record << "  One row per class a call site in this tree reaches, read through\n"
              "  boys::DefaultPolicyFor. A row is what that class's unnamed call compiles;\n"
              "  the seam's seven names below are a different combination.\n\n";
    record << "host, double, all orders - BoysAllOrders: the VRR seed, the derivative seed,\n"
              "the one-electron seed and the gradient gateway\n";
    record << ClassRowFacts<boys::Precision::kFp64, boys::Shape::kAllOrders, boys::Device::kHost>(
        "boys::DefaultPolicyFor<kFp64, kAllOrders>");
    record << "\nhost, float, all orders - BoysAllOrdersF32: the certified fp32 lane\n";
    record << ClassRowFacts<boys::Precision::kFp32, boys::Shape::kAllOrders, boys::Device::kHost>(
        "boys::DefaultPolicyFor<kFp32, kAllOrders>");
    record << "\nhost, double, single order - BoysSingle: the dense s-only path\n";
    record << ClassRowFacts<boys::Precision::kFp64, boys::Shape::kSingle, boys::Device::kHost>(
        "boys::DefaultPolicyFor<kFp64, kSingle>");
    record << "\ndevice, double and float, all orders - BoysDeviceAllNF64 / BoysDeviceAllNF32\n"
              "  The raw device entries take the build's two device names directly and resolve\n"
              "  no class row; these are the two cells its template defaults read.\n";
    record << Fact("division form",
                   boys::DivisionFormName(boys::kDefaultDeviceDivisionForm),
                   "boys::kDefaultDeviceDivisionForm")
           << "\n";
    record << Fact("region-B exponential",
                   boys::RegionBExpName(boys::kDefaultRegionBExp),
                   "boys::kDefaultRegionBExp")
           << "\n";
    record << "  A device call that names a policy (boys/boys_cuda_policy.hpp, WithPolicy<>)\n"
              "  resolves through DefaultPolicyFor's device rows instead; that layer is not what\n"
              "  this tree's two device calls compile.\n\n";

    record << "the seam's seven names (the fallback a class row replaces)\n";
    record << "  boys::DefaultPolicyFp64 is boys::SeamPolicy<>: the seven names below composed\n"
              "  with their own defaults. boys_build_defaults.hpp states them as the fallback a\n"
              "  build that carries no row list compiles every class from. Every class this\n"
              "  tree's calls reach carries a row, so no entry this build compiles runs this\n"
              "  combination; it is printed because it is the seam's own identity, and because\n"
              "  these are the axes a reader would otherwise take for the entries' answer.\n\n";
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
    record << Fact("division form",
                   boys::DivisionFormName(boys::DefaultPolicyFp64::kDivision),
                   "boys::DefaultPolicyFp64::kDivision")
           << "\n";
    record << Fact("region-B exponential",
                   boys::RegionBExpName(boys::DefaultPolicyFp64::kRegionBExp),
                   "boys::DefaultPolicyFp64::kRegionBExp")
           << "\n";
    record << Fact("half-lane budget",
                   BudgetLabel(boys::DefaultPolicyFp16::kBudget),
                   "boys::DefaultPolicyFp16::kBudget")
           << "\n";
    record << "  The half-lane budget above is the one cell SeamPolicy<> carries at the half\n"
              "  lane; the fp16 and bf16 class rows are the table's own and are not this name.\n\n";

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
        << "\nthe region-B exponential is one of the seven cells every class row above carries;\n"
           "the device entries' own default is the two-name block above. This tree names no\n"
           "value for it on either surface, and a revision that moves either name moves what\n"
           "the row or the block prints.\n\n";

    record << "Two labels above are spelled by this file rather than read, because the library\n"
              "publishes no name accessor for them: the engine budget and the row basis. Every\n"
              "value beside them is the library's.\n";

    return record.str();
}

} // namespace qcx::integrals
