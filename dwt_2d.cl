__kernel void fwt_53_2d_hh(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2 + 1;
  int y = gy * 2 + 1;

  float coefs[3][3];

  if (x < width && y < height) {
    coefs[0][0] = input[(y - 1) * stride + (x - 1)];
    coefs[0][1] = input[(y - 1) * stride + x];
    coefs[1][0] = input[y * stride + (x - 1)];

    coefs[0][2] = x + 1 >= width ? 0.f : input[(y - 1) * stride + (x + 1)];
    coefs[1][2] = x + 1 >= width ? 0.f : input[y * stride + (x + 1)];

    coefs[2][0] = y + 1 >= height ? 0.f : input[(y + 1) * stride + (x - 1)];
    coefs[2][1] = y + 1 >= height ? 0.f : input[(y + 1) * stride + x];

    coefs[2][2] = x + 1 >= width || y + 1 >= height ? 0.f : input[(y + 1) * stride + (x + 1)];

    coefs[0][1] -= floor((coefs[0][0] + coefs[0][2] + 1.f) / 2.f);
    coefs[2][1] -= floor((coefs[2][0] + coefs[2][2] + 1.f) / 2.f);

    input[y * stride + x] -= floor((coefs[1][0] + coefs[1][2] + 1.f) / 2.f) + floor((coefs[0][1] + coefs[2][1] + 1.f) / 2.f);
  }
}

__kernel void fwt_53_2d_lh(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2;
  int y = gy * 2;

  if ((x + 1 < width && y < height) || (x < width && y + 1 < height)) {
    float left_top = input[y * stride + x];
    float right_bottom = x + 1 >= width || y + 1 >= height ? 0.f : input[(y + 1) * stride + (x + 1)];

    if (x + 1 < width) {
      float right = x + 2 >= width ? 0.f : input[y * stride + (x + 2)];
      float top = y - 1 < 0 ? 0.f : input[(y - 1) * stride + (x + 1)];

      input[y * stride + x + 1] += floor((top + right_bottom + 2.f) / 4.f) - floor((left_top + right + 1.f) / 2.f);
    }

    if (y + 1 < height) {
      float bottom = y + 2 >= height ? 0.f : input[(y + 2) * stride + x];
      float left = x - 1 < 0 ? 0.f : input[(y + 1) * stride + (x - 1)];

      input[(y + 1) * stride + x] += floor((left + right_bottom + 2.f) / 4.f) - floor((left_top + bottom + 1.f) / 2.f);
    }
  }
}

__kernel void fwt_53_2d_ll(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2;
  int y = gy * 2;

  float coefs[3][3];

  if (x < width && y < height) {
    coefs[0][0] = x - 1 < 0      || y - 1 < 0       ? 0.f : input[(y - 1) * stride + (x - 1)];
    coefs[0][2] = x + 1 >= width || y - 1 < 0       ? 0.f : input[(y - 1) * stride + (x + 1)];
    coefs[2][0] = x - 1 < 0      || y + 1 >= height ? 0.f : input[(y + 1) * stride + (x - 1)];
    coefs[2][2] = x + 1 >= width || y + 1 >= height ? 0.f : input[(y + 1) * stride + (x + 1)];

    coefs[0][1] = y - 1 < 0       ? 0.f : input[(y - 1) * stride + x];
    coefs[1][0] = x - 1 < 0       ? 0.f : input[y * stride + (x - 1)];
    coefs[1][2] = x + 1 >= width  ? 0.f : input[y * stride + (x + 1)];
    coefs[2][1] = y + 1 >= height ? 0.f : input[(y + 1) * stride + x];

    coefs[1][1] = input[y * stride + x];

    coefs[0][1] -= floor((coefs[0][0] + coefs[0][2] + 2.f) / 4.f);
    coefs[2][1] -= floor((coefs[2][0] + coefs[2][2] + 2.f) / 4.f);
    coefs[1][1] += floor((coefs[1][0] + coefs[1][2] + 2.f) / 4.f) + floor((coefs[0][1] + coefs[2][1] + 2.f) / 4.f);

    input[y * stride + x] = coefs[1][1];
  }
}

__kernel void iwt_53_2d_ll(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2;
  int y = gy * 2;

  float coefs[3][3];

  if (x < width && y < height) {
    coefs[0][0] = x - 1 < 0      || y - 1 < 0       ? 0.f : input[(y - 1) * stride + (x - 1)];
    coefs[0][2] = x + 1 >= width || y - 1 < 0       ? 0.f : input[(y - 1) * stride + (x + 1)];
    coefs[2][0] = x - 1 < 0      || y + 1 >= height ? 0.f : input[(y + 1) * stride + (x - 1)];
    coefs[2][2] = x + 1 >= width || y + 1 >= height ? 0.f : input[(y + 1) * stride + (x + 1)];

    coefs[0][1] = y - 1 < 0       ? 0.f : input[(y - 1) * stride + x];
    coefs[1][0] = x - 1 < 0       ? 0.f : input[y * stride + (x - 1)];
    coefs[1][2] = x + 1 >= width  ? 0.f : input[y * stride + (x + 1)];
    coefs[2][1] = y + 1 >= height ? 0.f : input[(y + 1) * stride + x];

    coefs[1][1] = input[y * stride + x];

    coefs[0][1] -= floor((coefs[0][0] + coefs[0][2] + 2.f) / 4.f);
    coefs[2][1] -= floor((coefs[2][0] + coefs[2][2] + 2.f) / 4.f);
    coefs[1][1] -= floor((coefs[1][0] + coefs[1][2] + 2.f) / 4.f) + floor((coefs[0][1] + coefs[2][1] + 2.f) / 4.f);

    input[y * stride + x] = coefs[1][1];
  }
}

__kernel void iwt_53_2d_lh(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2;
  int y = gy * 2;

  if ((x + 1 < width && y < height) || (x < width && y + 1 < height)) {
    float left_top = input[y * stride + x];
    float right_bottom = x + 1 >= width || y + 1 >= height ? 0.f : input[(y + 1) * stride + (x + 1)];

    if (x + 1 < width) {
      float right = x + 2 >= width ? 0.f : input[y * stride + (x + 2)];
      float top = y - 1 < 0 ? 0.f : input[(y - 1) * stride + (x + 1)];

      input[y * stride + (x + 1)] -= floor((top + right_bottom + 2.f) / 4.f) - floor((left_top + right + 1.f) / 2.f);
    }

    if (y + 1 < height) {
      float bottom = y + 2 >= height ? 0.f : input[(y + 2) * stride + x];
      float left = x - 1 < 0 ? 0.f : input[(y + 1) * stride + (x - 1)];

      input[(y + 1) * stride + x] -= floor((left + right_bottom + 2.f) / 4.f) - floor((left_top + bottom + 1.f) / 2.f);
    }
  }
}

__kernel void iwt_53_2d_hh(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2 + 1;
  int y = gy * 2 + 1;

  float coefs[3][3];

  if (x < width && y < height) {
    coefs[0][0] = input[(y - 1) * stride + (x - 1)];
    coefs[0][1] = input[(y - 1) * stride + x];
    coefs[1][0] = input[y * stride + (x - 1)];

    coefs[0][2] = x + 1 >= width ? 0.f : input[(y - 1) * stride + (x + 1)];
    coefs[1][2] = x + 1 >= width ? 0.f : input[y * stride + (x + 1)];

    coefs[2][0] = y + 1 >= height ? 0.f : input[(y + 1) * stride + (x - 1)];
    coefs[2][1] = y + 1 >= height ? 0.f : input[(y + 1) * stride + x];

    coefs[2][2] = x + 1 >= width || y + 1 >= height ? 0.f : input[(y + 1) * stride + (x + 1)];

    coefs[0][1] -= floor((coefs[0][0] + coefs[0][2] + 1.f) / 2.f);
    coefs[2][1] -= floor((coefs[2][0] + coefs[2][2] + 1.f) / 2.f);

    input[y * stride + x] += floor((coefs[1][0] + coefs[1][2] + 1.f) / 2.f) + floor((coefs[0][1] + coefs[2][1] + 1.f) / 2.f);
  }
}

__kernel void split_2d(__global const float *input, __global float *output, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2;
  int y = gy * 2;

  int half_width  = (width  + 1) >> 1;
  int half_height = (height + 1) >> 1;

  if (x < width && y < height) {
    output[gy * width + gx] = input[y * stride + x];

    if (x + 1 < width) {
      output[gy * width + (gx + half_width)] = input[y * stride + (x + 1)];
    }

    if (y + 1 < height) {
      output[(gy + half_height) * width + gx] = input[(y + 1) * stride + x];
    }

    if (x + 1 < width && y + 1 < height) {
      output[(gy + half_height) * width + (gx + half_width)] = input[(y + 1) * stride + (x + 1)];
    }
  }
}

__kernel void unsplit_2d(__global const float *input, __global float *output, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2;
  int y = gy * 2;

  int half_width  = (width  + 1) >> 1;
  int half_height = (height + 1) >> 1;

  if (x < width && y < height) {
    output[y * stride + x] = input[gy * width + gx];

    if (x + 1 < width) {
      output[y * stride + (x + 1)] = input[gy * width + (gx + half_width)];
    }

    if (y + 1 < height) {
      output[(y + 1) * stride + x] = input[(gy + half_height) * width + gx];
    }

    if (x + 1 < width && y + 1 < height) {
      output[(y + 1) * stride + (x + 1)] = input[(gy + half_height) * width + (gx + half_width)];
    }
  }
}
