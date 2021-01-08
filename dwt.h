#pragma once

inline int ceil_to(int val, int to) {
  return ((val + to - 1) / to) * to;
}

inline int div_ceiled(int dividend, int divisor) {
  return (dividend + divisor - 1) / divisor;
}

int dwt_53_cpu(const char *input, const char *output, bool inverse = false);
int dwt_53_gpu(const char *input, const char *output, bool inverse = false);
int dwt_97_cpu(const char *input, const char *output, bool inverse = false);
int dwt_97_gpu(const char *input, const char *output, bool inverse = false);
