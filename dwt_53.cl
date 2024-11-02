__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline int shift_right_and_round(int a, int b) {
   return (a + (1 << (b - 1))) >> b;
}

__kernel void fwt_53_2d_hh(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = (gx << 1) + 1;
  int y = (gy << 1) + 1;

  if (x - 1 < width && y - 1 < height) {
    bool x_in = x < width;
    bool y_in = y < height;

    bool x1_in = x + 1 < width;
    bool y1_in = y + 1 < height;

    int center_center = !x_in  || !y_in  ? 0 : read_imagei(input, sampler, (int2)(x,     y)).x;
    int center_right  = !x1_in || !y_in  ? 0 : read_imagei(input, sampler, (int2)(x + 1, y)).x;
    int bottom_center = !x_in  || !y1_in ? 0 : read_imagei(input, sampler, (int2)(x,     y + 1)).x;
    int bottom_right  = !x1_in || !y1_in ? 0 : read_imagei(input, sampler, (int2)(x + 1, y + 1)).x;
    int top_center    = !x_in  ? 0 : read_imagei(input, sampler, (int2)(x,     y - 1)).x;
    int center_left   = !y_in  ? 0 : read_imagei(input, sampler, (int2)(x - 1, y)).x;
    int top_right     = !x1_in ? 0 : read_imagei(input, sampler, (int2)(x + 1, y - 1)).x;
    int bottom_left   = !y1_in ? 0 : read_imagei(input, sampler, (int2)(x - 1, y + 1)).x;
    int top_left      = read_imagei(input, sampler, (int2)(x - 1, y - 1)).x;

    top_center    -= shift_right_and_round(top_left + top_right, 1);
    bottom_center -= shift_right_and_round(bottom_left + bottom_right, 1);
    center_center -= shift_right_and_round(center_left + center_right, 1) + shift_right_and_round(top_center + bottom_center, 1);
    center_left   -= shift_right_and_round(top_left + bottom_left, 1);


    if (x_in && y_in) {
      write_imagei(output, (int2)(x, y), (int4)(center_center));
    }

    if (x_in) {
      write_imagei(output, (int2)(x, y - 1), (int4)(top_center));
    }

    if (y_in) {
      write_imagei(output, (int2)(x - 1, y), (int4)(center_left));
    }

    write_imagei(output, (int2)(x - 1, y - 1), (int4)(top_left));

  }
}

__kernel void fwt_53_2d_ll(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx << 1;
  int y = gy << 1;

  if (x < width && y < height) {

    bool x_in = x + 1 < width;
    bool y_in = y + 1 < height;

    int center_center = read_imagei(input, sampler, (int2)(x,     y)).x;
    int top_left      = read_imagei(input, sampler, (int2)(x - 1, y - 1)).x;
    int top_center    = read_imagei(input, sampler, (int2)(x,     y - 1)).x;
    int center_left   = read_imagei(input, sampler, (int2)(x - 1, y)).x;
    int top_right     = !x_in ? 0 : read_imagei(input, sampler, (int2)(x + 1, y - 1)).x;
    int center_right  = !x_in ? 0 : read_imagei(input, sampler, (int2)(x + 1, y)).x;
    int bottom_left   = !y_in ? 0 : read_imagei(input, sampler, (int2)(x - 1, y + 1)).x;
    int bottom_center = !y_in ? 0 : read_imagei(input, sampler, (int2)(x,     y + 1)).x;
    int bottom_right  = !x_in || !y_in ? 0 : read_imagei(input, sampler, (int2)(x + 1, y + 1)).x;

    center_left   += shift_right_and_round(top_left + bottom_left, 2);
    center_right  += shift_right_and_round(top_right + bottom_right, 2);
    center_center += shift_right_and_round(center_left + center_right, 2) + shift_right_and_round(top_center + bottom_center, 2);
    bottom_center += shift_right_and_round(bottom_left + bottom_right, 2);

    int half_width  = (width  + 1) >> 1;
    int half_height = (height + 1) >> 1;

    write_imagei(output, (int2)(gx, gy), (int4)(center_center));

    if (x_in) {
      write_imagei(output, (int2)(gx + half_width, gy), (int4)(center_right));
    }

    if (y_in) {
      write_imagei(output, (int2)(gx, gy + half_height), (int4)(bottom_center));
    }

    if (x_in && y_in) {
      write_imagei(output, (int2)(gx + half_width, gy + half_height), (int4)(bottom_right));
    }
  }
}


__kernel void iwt_53_2d_ll(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx << 1;
  int y = gy << 1;

  if (x < width && y < height) {

    bool x_in = x + 1 < width;
    bool y_in = y + 1 < height;

    int half_width  = (width  + 1) >> 1;
    int half_height = (height + 1) >> 1;

    int center_center = read_imagei(input, sampler, (int2)(gx, gy)).x;

    int top_left      = !gx   || !gy   ? 0 : read_imagei(input, sampler, (int2)(gx + half_width - 1, gy + half_height - 1)).x;
    int bottom_left   = !gx   || !y_in ? 0 : read_imagei(input, sampler, (int2)(gx + half_width - 1, gy + half_height)).x;
    int top_right     = !x_in || !gy   ? 0 : read_imagei(input, sampler, (int2)(gx + half_width,     gy + half_height - 1)).x;
    int bottom_right  = !x_in || !y_in ? 0 : read_imagei(input, sampler, (int2)(gx + half_width,     gy + half_height)).x;

    int center_left   = !gx   ? 0 : read_imagei(input, sampler, (int2)(gx + half_width - 1, gy)).x;
    int top_center    = !gy   ? 0 : read_imagei(input, sampler, (int2)(gx,                  gy + half_height - 1)).x;
    int center_right  = !x_in ? 0 : read_imagei(input, sampler, (int2)(gx + half_width,     gy)).x;
    int bottom_center = !y_in ? 0 : read_imagei(input, sampler, (int2)(gx,                  gy + half_height)).x;

    top_center    -= shift_right_and_round(top_left + top_right, 2);
    bottom_center -= shift_right_and_round(bottom_left + bottom_right, 2);
    center_center -= shift_right_and_round(center_left + center_right, 2) + shift_right_and_round(top_center + bottom_center, 2);
    center_right  -= shift_right_and_round(top_right + bottom_right, 2);

    write_imagei(output, (int2)(x, y), (int4)(center_center));

    if (y_in) {
      write_imagei(output, (int2)(x, y + 1), (int4)(bottom_center));
    }

    if (x_in) {
      write_imagei(output, (int2)(x + 1, y), (int4)(center_right));
    }

    if (x_in && y_in) {
      write_imagei(output, (int2)(x + 1, y + 1), (int4)(bottom_right));
    }
  }
}

__kernel void iwt_53_2d_hh(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = (gx << 1) + 1;
  int y = (gy << 1) + 1;

  if (x - 1 < width && y - 1 < height) {
    bool x_in = x < width;
    bool y_in = y < height;

    bool x1_in = x + 1 < width;
    bool y1_in = y + 1 < height;

    int center_center = !x_in  || !y_in  ? 0 : read_imagei(input, sampler, (int2)(x,     y)).x;
    int center_right  = !x1_in || !y_in  ? 0 : read_imagei(input, sampler, (int2)(x + 1, y)).x;
    int bottom_center = !x_in  || !y1_in ? 0 : read_imagei(input, sampler, (int2)(x,     y + 1)).x;
    int bottom_right  = !x1_in || !y1_in ? 0 : read_imagei(input, sampler, (int2)(x + 1, y + 1)).x;
    int top_center    = !x_in  ? 0 : read_imagei(input, sampler, (int2)(x,     y - 1)).x;
    int center_left   = !y_in  ? 0 : read_imagei(input, sampler, (int2)(x - 1, y)).x;
    int top_right     = !x1_in ? 0 : read_imagei(input, sampler, (int2)(x + 1, y - 1)).x;
    int bottom_left   = !y1_in ? 0 : read_imagei(input, sampler, (int2)(x - 1, y + 1)).x;
    int top_left      = read_imagei(input, sampler, (int2)(x - 1, y - 1)).x;

    center_left   += shift_right_and_round(top_left + bottom_left, 1);
    center_right  += shift_right_and_round(top_right + bottom_right, 1);
    center_center += shift_right_and_round(center_left + center_right, 1) + shift_right_and_round(top_center + bottom_center, 1);
    top_center    += shift_right_and_round(top_left + top_right, 1);

    if (x_in && y_in) {
      write_imagei(output, (int2)(x, y), (int4)(center_center));
    }

    if (x_in) {
      write_imagei(output, (int2)(x, y - 1), (int4)(top_center));
    }

    if (y_in) {
      write_imagei(output, (int2)(x - 1, y), (int4)(center_left));
    }

    write_imagei(output, (int2)(x - 1, y - 1), (int4)(top_left));
  }
}
