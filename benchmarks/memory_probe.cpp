#include "memory_probe.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace memprobe {

std::string BasisDataDir() {
    return std::string(QcxBasisDataDir);
}

} // namespace memprobe

int main(int argc, char** argv) {
    // Unbuffered: a long measurement must show where it is when it is watched.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<std::string> args;

    for (int i = 1; i < argc; ++i)
    {
        args.emplace_back(argv[i]);
    }

    if (args.empty())
    {
        std::cerr << "usage: qcx-bench-memory-probe <m1|m2m3|m5|m6> [...]\n";
        return 2;
    }

    const std::string command = args.front();
    args.erase(args.begin());

    if (command == "m1")
    {
        return memprobe::RunM1(args);
    }

    if (command == "m2m3")
    {
        return memprobe::RunM2M3(args);
    }

    if (command == "m5")
    {
        return memprobe::RunM5(args);
    }

    if (command == "m6")
    {
        return memprobe::RunM6(args);
    }

    std::cerr << "unknown subcommand: " << command << "\n";
    return 2;
}
