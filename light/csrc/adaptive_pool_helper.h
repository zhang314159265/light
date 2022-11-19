#pragma once

static int calc_start_idx(int out_idx, int out_size, int in_size) {
  return out_idx * in_size / out_size;
}

static int calc_end_idx(int out_idx, int out_size, int in_size) {
  return ((out_idx + 1) * in_size + out_size - 1) / out_size;
}
