#pragma once

#include <cstdlib>

// store grad for intermediate node just for debugging
static bool config_keep_grad_for_nonleaf() {
  char *str = std::getenv("keep_grad_for_nonleaf");
  return str && str[0] == '1' && str[1] == '\0';
}
