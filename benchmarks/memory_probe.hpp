#pragma once

// The shared plumbing of the memory-measurement probe: what the grid engine
// stores per grid point, how many allocations each path makes, what a device
// block's two candidate layouts cost, what storing against recomputing costs,
// and how the Becke weight derivative scales with the atom count.
//
// Each measurement is one subcommand, prints one number per line as
// `key = value`, and names its own fixture in the same output.

#include <string>
#include <string_view>
#include <vector>

namespace memprobe {

/// One measurement subcommand.
/// \param argv The command's own arguments (argv[0] is the subcommand).
/// \returns The process exit code.
int RunM1(const std::vector<std::string>& argv);
int RunM2M3(const std::vector<std::string>& argv);
int RunM5(const std::vector<std::string>& argv);
int RunM6(const std::vector<std::string>& argv);

/// The basis-set data directory this harness was built against.
/// \returns The directory path.
std::string BasisDataDir();

} // namespace memprobe
