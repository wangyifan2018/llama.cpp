#pragma once
#include <cstdlib>

namespace crispasr_env {
inline const char * get(const char * key) { return std::getenv(key); }
} // namespace crispasr_env
