#pragma once

#include <random>
#include <iostream>

class Generator {
 public:
  float uniform(float start, float end) {
    uint32_t val = get_engine()();
    constexpr int ndig = std::numeric_limits<float>::digits;
    constexpr int mask = (1 << ndig) - 1;
    constexpr float div = 1.0f / (1 << ndig);
    float val01 = (val & mask) * div;
    return val01 * (end - start) + start;
  }

  int32_t uniformInt(int low, int high) {
    int range = high - low;
    return get_engine()() % range + low;
  }

  static void set_seed(int seed) {
    get_engine() = std::mt19937(seed);
  }

  static std::mt19937& get_engine() {
    static std::mt19937 gen(19337);
    return gen;
  }
 private:
};

static inline void set_seed(int seed) {
    Generator::set_seed(seed); 
}
