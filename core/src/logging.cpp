#include "qcx/logging.hpp"

namespace qcx::log {

void Init() {
    spdlog::set_pattern("[%H:%M:%S] [%^%l%$] %v");
}

} // namespace qcx::log
