#pragma once

#include <cstdint>
#include <cstddef>

inline int shift_right_and_round(int a, int b) {
   return (a + (1 << (b - 1))) >> b;
}

void fwt_53_2d(int16_t *image, size_t width, size_t height, size_t stride, int16_t *temp = nullptr);
void iwt_53_2d(int16_t *image, size_t width, size_t height, size_t stride, int16_t *temp = nullptr);

void fwt_97_2d(float *image, size_t width, size_t height, size_t stride, float *temp = nullptr);
void iwt_97_2d(float *image, size_t width, size_t height, size_t stride, float *temp = nullptr);
