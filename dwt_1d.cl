__kernel void fwt_53_high(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2 + 1;

  if (x < width && gy < height) {
    float left  = input[gy * stride + x - 1];
    float right = x >= width - 1 ? 0.f : input[gy * stride + x + 1];

    input[gy * stride + x] -= floor((left + right + 1) / 2.f);
  }
}

__kernel void fwt_53_low(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2;

  if (x < width && gy < height) {
    float left  = x <= 0 ? 0.f : input[gy * stride + x - 1];
    float right = x >= width - 1 ? 0.f : input[gy * stride + x + 1];

    input[gy * stride + x] += floor((left + right + 2) / 4.f);
  }
}

__kernel void iwt_53_high(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2;

  if (x < width && gy < height) {
    float left  = x <= 0 ? 0.f : input[gy * stride + x - 1];
    float right = x >= width - 1 ? 0.f : input[gy * stride + x + 1];

    input[gy * stride + x] -= floor((left + right + 2) / 4.f);
  }
}

__kernel void iwt_53_low(__global float *input, int width, int height, int stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx * 2 + 1;

  if (x < width && gy < height) {
    float left  = input[gy * stride + x - 1];
    float right = x >= width - 1 ? 0.f : input[gy * stride + x + 1];

    input[gy * stride + x] += floor((left + right + 1) / 2.f);
  }
}

__kernel void split(__global const float *input, __global float *output, int width, int height, int input_stride, int output_stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  if (gy < height) {
    {
      int in_l = gx * 2;
      if (in_l < width) {
        int out_l = gx;
        output[out_l * output_stride + gy] = input[gy * input_stride + in_l];
      }
    }
    {
      int in_h = gx * 2 + 1;
      if (in_h < width) {
        int out_h = (width + 1) / 2 + gx;
        output[out_h * output_stride + gy] = input[gy * input_stride + in_h];
      }
    }
  }
}

__kernel void unsplit(__global const float *input, __global float *output, int width, int height, int input_stride, int output_stride) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  if (gy < height) {
    {
      int out_l = gx * 2;
      if (out_l < width) {
        int in_l = gx;
        output[gy * output_stride + out_l] = input[in_l * input_stride + gy];
      }
    }

    {
      int out_h = gx * 2 + 1;
      if (out_h < width) {
        int in_h = (width + 1) / 2 + gx;
        output[gy * output_stride + out_h] = input[in_h * input_stride + gy];
      }
    }
  }
}
