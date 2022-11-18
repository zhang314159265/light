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

  // TODO avoid code duplication with uniform
  double uniform64(double start, double end) {
    uint64_t val = random64();
    constexpr int ndig = std::numeric_limits<double>::digits;
    constexpr uint64_t mask = (1ULL << ndig) - 1;
    constexpr double div = 1.0 / (1ULL << ndig);
    double val01 = (val & mask) * div;
    return val01 * (end - start) + start;
  }

  int32_t uniformInt(int low, int high) {
    int range = high - low;
    return random() % range + low;
  }

  uint64_t random64() {
    return (((uint64_t) random() << 32) | random());
  }

  uint32_t random() {
    return get_engine()();
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
