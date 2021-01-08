
#include "dwt_cpu.h"
#include <vector>

void fwt_53_2d(int16_t *image, size_t width, size_t height, size_t stride, int16_t *temp) {
  int half_width  = (width  + 1) >> 1;
  int half_height = (height + 1) >> 1;

  std::vector<int16_t> tmp {};
  if (!temp) {
    tmp.resize(width * height);
    temp = tmp.data();
  }

  for (size_t y = 1; y - 1 < height; y += 2) {
    for (size_t x = 1; x - 1 < width; x += 2) {
      bool x_in = x < width;
      bool y_in = y < height;

      bool x1_in = x + 1 < width;
      bool y1_in = y + 1 < height;

      int center_center = !x_in  || !y_in  ? 0 : image[y * stride + x];
      int center_right  = !x1_in || !y_in  ? 0 : image[y * stride + (x + 1)];
      int bottom_center = !x_in  || !y1_in ? 0 : image[(y + 1) * stride + x];
      int bottom_right  = !x1_in || !y1_in ? 0 : image[(y + 1) * stride + (x + 1)];
      int top_center    = !x_in  ? 0 : image[(y - 1) * stride + x];
      int center_left   = !y_in  ? 0 : image[y * stride + (x - 1)];
      int top_right     = !x1_in ? 0 : image[(y - 1) * stride + (x + 1)];
      int bottom_left   = !y1_in ? 0 : image[(y + 1) * stride + (x - 1)];
      int top_left      = image[(y - 1) * stride + (x - 1)];

      top_center    -= shift_right_and_round(top_left + top_right, 1);
      bottom_center -= shift_right_and_round(bottom_left + bottom_right, 1);
      center_center -= shift_right_and_round(center_left + center_right, 1) + shift_right_and_round(top_center + bottom_center, 1);
      center_left   -= shift_right_and_round(top_left + bottom_left, 1);

      if (x_in && y_in) {
        temp[y * width + x] = center_center;
      }

      if (x_in) {
        temp[(y - 1) * width + x] = top_center;
      }

      if (y_in) {
        temp[y * width + (x - 1)] = center_left;
      }

      temp[(y - 1) * width + (x - 1)] = top_left;
    }
  }

  for (size_t y = 0; y < height; y += 2) {
    for (size_t x = 0; x < width; x += 2) {
      bool x_in = x + 1 < width;
      bool y_in = y + 1 < height;

      size_t gx = x >> 1;
      size_t gy = y >> 1;

      int center_center = temp[y * width + x];
      int top_left      = !x    || !y    ? 0 : temp[(y - 1) * width + (x - 1)];
      int bottom_left   = !x    || !y_in ? 0 : temp[(y + 1) * width + (x - 1)];
      int top_right     = !x_in || !y    ? 0 : temp[(y - 1) * width + (x + 1)];
      int bottom_right  = !x_in || !y_in ? 0 : temp[(y + 1) * width + (x + 1)];
      int center_left   = !x    ? 0 : temp[y * width + (x - 1)];
      int top_center    = !y    ? 0 : temp[(y - 1) * width + x];
      int center_right  = !x_in ? 0 : temp[y * width + (x + 1)];
      int bottom_center = !y_in ? 0 : temp[(y + 1) * width + x];

      center_left   += shift_right_and_round(top_left + bottom_left, 2);
      center_right  += shift_right_and_round(top_right + bottom_right, 2);
      center_center += shift_right_and_round(center_left + center_right, 2) + shift_right_and_round(top_center + bottom_center, 2);
      bottom_center += shift_right_and_round(bottom_left + bottom_right, 2);

      image[gy * stride + gx] = center_center;

      if (x_in) {
        image[gy * stride + (gx + half_width)] = center_right;
      }

      if (y_in) {
        image[(gy + half_height) * stride + gx] = bottom_center;
      }

      if (x_in && y_in) {
        image[(gy + half_height) * stride + (gx + half_width)] = bottom_right;
      }
    }
  }
}

void iwt_53_2d(int16_t *image, size_t width, size_t height, size_t stride, int16_t *temp) {
  int half_width  = (width  + 1) >> 1;
  int half_height = (height + 1) >> 1;

  std::vector<int16_t> tmp {};
  if (!temp) {
    tmp.resize(width * height);
    temp = tmp.data();
  }

  for (size_t y = 0; y < height; y += 2) {
    for (size_t x = 0; x < width; x += 2) {
      bool x_in = x + 1 < width;
      bool y_in = y + 1 < height;

      size_t gx = x >> 1;
      size_t gy = y >> 1;

      int center_center = image[gy * stride + gx];

      int top_left      = !gx   || !gy   ? 0 : image[(gy + half_height - 1) * stride + (gx + half_width - 1)];
      int bottom_left   = !gx   || !y_in ? 0 : image[(gy + half_height)     * stride + (gx + half_width - 1)];
      int top_right     = !x_in || !gy   ? 0 : image[(gy + half_height - 1) * stride + (gx + half_width)];
      int bottom_right  = !x_in || !y_in ? 0 : image[(gy + half_height)     * stride + (gx + half_width)];

      int center_left   = !gx   ? 0 : image[gy * stride + (gx + half_width - 1)];
      int top_center    = !gy   ? 0 : image[(gy + half_height - 1) * stride + gx];
      int center_right  = !x_in ? 0 : image[gy * stride + (gx + half_width)];
      int bottom_center = !y_in ? 0 : image[(gy + half_height) * stride + gx];

      top_center    -= shift_right_and_round(top_left + top_right, 2);
      bottom_center -= shift_right_and_round(bottom_left + bottom_right, 2);
      center_center -= shift_right_and_round(center_left + center_right, 2) + shift_right_and_round(top_center + bottom_center, 2);
      center_right  -= shift_right_and_round(top_right + bottom_right, 2);

      temp[y * width + x] = center_center;

      if (y_in) {
        temp[(y + 1) * width + x] = bottom_center;
      }

      if (x_in) {
        temp[y * width + (x + 1)] = center_right;
      }

      if (x_in && y_in) {
        temp[(y + 1) * width + (x + 1)] = bottom_right;
      }
    }
  }

  for (size_t y = 1; y - 1 < height; y += 2) {
    for (size_t x = 1; x - 1 < width; x += 2) {
      bool x_in = x < width;
      bool y_in = y < height;

      bool x1_in = x + 1 < width;
      bool y1_in = y + 1 < height;

      int center_center = !x_in  || !y_in  ? 0 : temp[y * width + x];
      int center_right  = !x1_in || !y_in  ? 0 : temp[y * width + (x + 1)];
      int bottom_center = !x_in  || !y1_in ? 0 : temp[(y + 1) * width + x];
      int bottom_right  = !x1_in || !y1_in ? 0 : temp[(y + 1) * width + (x + 1)];
      int top_center    = !x_in  ? 0 : temp[(y - 1) * width + x];
      int center_left   = !y_in  ? 0 : temp[y * width + (x - 1)];
      int top_right     = !x1_in ? 0 : temp[(y - 1) * width + (x + 1)];
      int bottom_left   = !y1_in ? 0 : temp[(y + 1) * width + (x - 1)];
      int top_left      = temp[(y - 1) * width + (x - 1)];

      center_left   += shift_right_and_round(top_left + bottom_left, 1);
      center_right  += shift_right_and_round(top_right + bottom_right, 1);
      center_center += shift_right_and_round(center_left + center_right, 1) + shift_right_and_round(top_center + bottom_center, 1);
      top_center    += shift_right_and_round(top_left + top_right, 1);

      if (x_in && y_in) {
        image[y * stride + x] = center_center;
      }

      if (x_in) {
        image[(y - 1) * stride + x] = top_center;
      }

      if (y_in) {
        image[y * stride + (x - 1)] = center_left;
      }

      image[(y - 1) * stride + (x - 1)] = top_left;
    }
  }
}

void fwt_97_2d(float *image, size_t width, size_t height, size_t stride, float *temp) {
  int half_width  = (width  + 1) >> 1;
  int half_height = (height + 1) >> 1;

  std::vector<float> tmp {};
  if (!temp) {
    tmp.resize(width * height);
    temp = tmp.data();
  }

  for (size_t y = 1; y - 1 < height; y += 2) {
    for (size_t x = 1; x - 1 < width; x += 2) {
      bool x_in = x < width;
      bool y_in = y < height;

      bool x1_in = x + 1 < width;
      bool y1_in = y + 1 < height;

      float center_center = !x_in  || !y_in  ? 0 : image[y * stride + x];
      float center_right  = !x1_in || !y_in  ? 0 : image[y * stride + (x + 1)];
      float bottom_center = !x_in  || !y1_in ? 0 : image[(y + 1) * stride + x];
      float bottom_right  = !x1_in || !y1_in ? 0 : image[(y + 1) * stride + (x + 1)];
      float top_center    = !x_in  ? 0 : image[(y - 1) * stride + x];
      float center_left   = !y_in  ? 0 : image[y * stride + (x - 1)];
      float top_right     = !x1_in ? 0 : image[(y - 1) * stride + (x + 1)];
      float bottom_left   = !y1_in ? 0 : image[(y + 1) * stride + (x - 1)];
      float top_left      = image[(y - 1) * stride + (x - 1)];

      top_center    -= 1.586134342059924 * (top_left + top_right);
      bottom_center -= 1.586134342059924 * (bottom_left + bottom_right);
      center_center -= 1.586134342059924 * (center_left + center_right + top_center + bottom_center);
      center_left   -= 1.586134342059924 * (top_left + bottom_left);

      if (x_in && y_in) {
        temp[y * width + x] = center_center;
      }

      if (x_in) {
        temp[(y - 1) * width + x] = top_center;
      }

      if (y_in) {
        temp[y * width + (x - 1)] = center_left;
      }

      temp[(y - 1) * width + (x - 1)] = top_left;
    }
  }

  for (size_t y = 0; y < height; y += 2) {
    for (size_t x = 0; x < width; x += 2) {
      bool x_in = x + 1 < width;
      bool y_in = y + 1 < height;

      float center_center = temp[y * width + x];
      float top_left      = !x    || !y    ? 0 : temp[(y - 1) * width + (x - 1)];
      float bottom_left   = !x    || !y_in ? 0 : temp[(y + 1) * width + (x - 1)];
      float top_right     = !x_in || !y    ? 0 : temp[(y - 1) * width + (x + 1)];
      float bottom_right  = !x_in || !y_in ? 0 : temp[(y + 1) * width + (x + 1)];
      float center_left   = !x    ? 0 : temp[y * width + (x - 1)];
      float top_center    = !y    ? 0 : temp[(y - 1) * width + x];
      float center_right  = !x_in ? 0 : temp[y * width + (x + 1)];
      float bottom_center = !y_in ? 0 : temp[(y + 1) * width + x];

      center_left   -= 0.052980118572961 * (top_left + bottom_left);
      center_right  -= 0.052980118572961 * (top_right + bottom_right);
      center_center -= 0.052980118572961 * (center_left + center_right + top_center + bottom_center);
      bottom_center -= 0.052980118572961 * (bottom_left + bottom_right);

      image[y * stride + x] = center_center;

      if (x_in) {
        image[y * stride + (x + 1)] = center_right;
      }

      if (y_in) {
        image[(y + 1) * stride + x] = bottom_center;
      }

      if (x_in && y_in) {
        image[(y + 1) * stride + (x + 1)] = bottom_right;
      }
    }
  }

  for (size_t y = 1; y - 1 < height; y += 2) {
    for (size_t x = 1; x - 1 < width; x += 2) {
      bool x_in = x < width;
      bool y_in = y < height;

      bool x1_in = x + 1 < width;
      bool y1_in = y + 1 < height;

      float center_center = !x_in  || !y_in  ? 0 : image[y * stride + x];
      float center_right  = !x1_in || !y_in  ? 0 : image[y * stride + (x + 1)];
      float bottom_center = !x_in  || !y1_in ? 0 : image[(y + 1) * stride + x];
      float bottom_right  = !x1_in || !y1_in ? 0 : image[(y + 1) * stride + (x + 1)];
      float top_center    = !x_in  ? 0 : image[(y - 1) * stride + x];
      float center_left   = !y_in  ? 0 : image[y * stride + (x - 1)];
      float top_right     = !x1_in ? 0 : image[(y - 1) * stride + (x + 1)];
      float bottom_left   = !y1_in ? 0 : image[(y + 1) * stride + (x - 1)];
      float top_left      = image[(y - 1) * stride + (x - 1)];

      top_center    += 0.882911075530934 * (top_left + top_right);
      bottom_center += 0.882911075530934 * (bottom_left + bottom_right);
      center_center += 0.882911075530934 * (center_left + center_right + top_center + bottom_center);
      center_left   += 0.882911075530934 * (top_left + bottom_left);

      if (x_in && y_in) {
        temp[y * width + x] = center_center;
      }

      if (x_in) {
        temp[(y - 1) * width + x] = top_center;
      }

      if (y_in) {
        temp[y * width + (x - 1)] = center_left;
      }

      temp[(y - 1) * width + (x - 1)] = top_left;
    }
  }

  for (size_t y = 0; y < height; y += 2) {
    for (size_t x = 0; x < width; x += 2) {
      bool x_in = x + 1 < width;
      bool y_in = y + 1 < height;

      size_t gx = x >> 1;
      size_t gy = y >> 1;

      float center_center = temp[y * width + x];
      float top_left      = !x    || !y    ? 0 : temp[(y - 1) * width + (x - 1)];
      float bottom_left   = !x    || !y_in ? 0 : temp[(y + 1) * width + (x - 1)];
      float top_right     = !x_in || !y    ? 0 : temp[(y - 1) * width + (x + 1)];
      float bottom_right  = !x_in || !y_in ? 0 : temp[(y + 1) * width + (x + 1)];
      float center_left   = !x    ? 0 : temp[y * width + (x - 1)];
      float top_center    = !y    ? 0 : temp[(y - 1) * width + x];
      float center_right  = !x_in ? 0 : temp[y * width + (x + 1)];
      float bottom_center = !y_in ? 0 : temp[(y + 1) * width + x];

      center_left   += 0.443506852043971 * (top_left + bottom_left);
      center_right  += 0.443506852043971 * (top_right + bottom_right);
      center_center += 0.443506852043971 * (center_left + center_right + top_center + bottom_center);
      bottom_center += 0.443506852043971 * (bottom_left + bottom_right);

      image[gy * stride + gx] = center_center;

      if (x_in) {
        image[gy * stride + (gx + half_width)] = center_right;
      }

      if (y_in) {
        image[(gy + half_height) * stride + gx] = bottom_center;
      }

      if (x_in && y_in) {
        image[(gy + half_height) * stride + (gx + half_width)] = bottom_right;
      }
    }
  }
}

void iwt_97_2d(float *image, size_t width, size_t height, size_t stride, float *temp) {
  int half_width  = (width  + 1) >> 1;
  int half_height = (height + 1) >> 1;

  std::vector<float> tmp {};
  if (!temp) {
    tmp.resize(width * height);
    temp = tmp.data();
  }

  for (size_t y = 0; y < height; y += 2) {
    for (size_t x = 0; x < width; x += 2) {
      bool x_in = x + 1 < width;
      bool y_in = y + 1 < height;

      size_t gx = x >> 1;
      size_t gy = y >> 1;

      float center_center = image[gy * stride + gx];

      float top_left      = !gx   || !gy   ? 0 : image[(gy + half_height - 1) * stride + (gx + half_width - 1)];
      float bottom_left   = !gx   || !y_in ? 0 : image[(gy + half_height)     * stride + (gx + half_width - 1)];
      float top_right     = !x_in || !gy   ? 0 : image[(gy + half_height - 1) * stride + (gx + half_width)];
      float bottom_right  = !x_in || !y_in ? 0 : image[(gy + half_height)     * stride + (gx + half_width)];

      float center_left   = !gx   ? 0 : image[gy * stride + (gx + half_width - 1)];
      float top_center    = !gy   ? 0 : image[(gy + half_height - 1) * stride + gx];
      float center_right  = !x_in ? 0 : image[gy * stride + (gx + half_width)];
      float bottom_center = !y_in ? 0 : image[(gy + half_height) * stride + gx];

      top_center    -= 0.443506852043971 * (top_left + top_right);
      bottom_center -= 0.443506852043971 * (bottom_left + bottom_right);
      center_center -= 0.443506852043971 * (center_left + center_right + top_center + bottom_center);
      center_right  -= 0.443506852043971 * (top_right + bottom_right);

      temp[y * width + x] = center_center;

      if (y_in) {
        temp[(y + 1) * width + x] = bottom_center;
      }

      if (x_in) {
        temp[y * width + (x + 1)] = center_right;
      }

      if (x_in && y_in) {
        temp[(y + 1) * width + (x + 1)] = bottom_right;
      }
    }
  }

  for (size_t y = 1; y - 1 < height; y += 2) {
    for (size_t x = 1; x - 1 < width; x += 2) {
      bool x_in = x < width;
      bool y_in = y < height;

      bool x1_in = x + 1 < width;
      bool y1_in = y + 1 < height;

      float center_center = !x_in  || !y_in  ? 0 : temp[y * width + x];
      float center_right  = !x1_in || !y_in  ? 0 : temp[y * width + (x + 1)];
      float bottom_center = !x_in  || !y1_in ? 0 : temp[(y + 1) * width + x];
      float bottom_right  = !x1_in || !y1_in ? 0 : temp[(y + 1) * width + (x + 1)];
      float top_center    = !x_in  ? 0 : temp[(y - 1) * width + x];
      float center_left   = !y_in  ? 0 : temp[y * width + (x - 1)];
      float top_right     = !x1_in ? 0 : temp[(y - 1) * width + (x + 1)];
      float bottom_left   = !y1_in ? 0 : temp[(y + 1) * width + (x - 1)];
      float top_left      = temp[(y - 1) * width + (x - 1)];

      center_left   -= 0.882911075530934 * (top_left + bottom_left);
      center_right  -= 0.882911075530934 * (top_right + bottom_right);
      center_center -= 0.882911075530934 * (center_left + center_right + top_center + bottom_center);
      top_center    -= 0.882911075530934 * (top_left + top_right);

      if (x_in && y_in) {
        image[y * stride + x] = center_center;
      }

      if (x_in) {
        image[(y - 1) * stride + x] = top_center;
      }

      if (y_in) {
        image[y * stride + (x - 1)] = center_left;
      }

      image[(y - 1) * stride + (x - 1)] = top_left;
    }
  }

  for (size_t y = 0; y < height; y += 2) {
    for (size_t x = 0; x < width; x += 2) {
      bool x_in = x + 1 < width;
      bool y_in = y + 1 < height;

      float center_center = image[y * stride + x];
      float top_left      = !x    || !y    ? 0 : image[(y - 1) * stride + (x - 1)];
      float bottom_left   = !x    || !y_in ? 0 : image[(y + 1) * stride + (x - 1)];
      float top_right     = !x_in || !y    ? 0 : image[(y - 1) * stride + (x + 1)];
      float bottom_right  = !x_in || !y_in ? 0 : image[(y + 1) * stride + (x + 1)];
      float center_left   = !x    ? 0 : image[y * stride + (x - 1)];
      float top_center    = !y    ? 0 : image[(y - 1) * stride + x];
      float center_right  = !x_in ? 0 : image[y * stride + (x + 1)];
      float bottom_center = !y_in ? 0 : image[(y + 1) * stride + x];

      top_center    += 0.052980118572961 * (top_left + top_right);
      bottom_center += 0.052980118572961 * (bottom_left + bottom_right);
      center_center += 0.052980118572961 * (center_left + center_right + top_center + bottom_center);
      center_right  += 0.052980118572961 * (top_right + bottom_right);

      temp[y * width + x] = center_center;

      if (y_in) {
        temp[(y + 1) * width + x] = bottom_center;
      }

      if (x_in) {
        temp[y * width + (x + 1)] = center_right;
      }

      if (x_in && y_in) {
        temp[(y + 1) * width + (x + 1)] = bottom_right;
      }
    }
  }

  for (size_t y = 1; y - 1 < height; y += 2) {
    for (size_t x = 1; x - 1 < width; x += 2) {
      bool x_in = x < width;
      bool y_in = y < height;

      bool x1_in = x + 1 < width;
      bool y1_in = y + 1 < height;

      float center_center = !x_in  || !y_in  ? 0 : temp[y * width + x];
      float center_right  = !x1_in || !y_in  ? 0 : temp[y * width + (x + 1)];
      float bottom_center = !x_in  || !y1_in ? 0 : temp[(y + 1) * width + x];
      float bottom_right  = !x1_in || !y1_in ? 0 : temp[(y + 1) * width + (x + 1)];
      float top_center    = !x_in  ? 0 : temp[(y - 1) * width + x];
      float center_left   = !y_in  ? 0 : temp[y * width + (x - 1)];
      float top_right     = !x1_in ? 0 : temp[(y - 1) * width + (x + 1)];
      float bottom_left   = !y1_in ? 0 : temp[(y + 1) * width + (x - 1)];
      float top_left      = temp[(y - 1) * width + (x - 1)];

      center_left   += 1.586134342059924 * (top_left + bottom_left);
      center_right  += 1.586134342059924 * (top_right + bottom_right);
      center_center += 1.586134342059924 * (center_left + center_right + top_center + bottom_center);
      top_center    += 1.586134342059924 * (top_left + top_right);

      if (x_in && y_in) {
        image[y * stride + x] = center_center;
      }

      if (x_in) {
        image[(y - 1) * stride + x] = top_center;
      }

      if (y_in) {
        image[y * stride + (x - 1)] = center_left;
      }

      image[(y - 1) * stride + (x - 1)] = top_left;
    }
  }
}
