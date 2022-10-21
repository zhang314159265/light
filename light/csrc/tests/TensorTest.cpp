#include <iostream>
#include <gtest/gtest.h>
#include "Tensor.h"

TEST(TensorTest, TestAdd) {
  Tensor a({2, 3}, ScalarType::Float);
  Tensor b({2, 3}, ScalarType::Float);
  Tensor c({2, 3}, ScalarType::Float);
  a.init(2.0f);
  b.init(3.0f);
  c.init(5.0f);
  ASSERT_TRUE(Tensor::add(a, b).equal(c));
}
