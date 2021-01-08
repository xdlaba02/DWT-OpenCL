__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void fwt_97_2d_hh_1(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = (gx << 1) + 1;
  int y = (gy << 1) + 1;

  if (x - 1 < width && y - 1 < height) {
    bool x_in = x < width;
    bool y_in = y < height;

    bool x1_in = x + 1 < width;
    bool y1_in = y + 1 < height;

    float center_center = !x_in  || !y_in  ? 0.f : read_imagef(input, sampler, (int2)(x,     y)).x;
    float center_right  = !x1_in || !y_in  ? 0.f : read_imagef(input, sampler, (int2)(x + 1, y)).x;
    float bottom_center = !x_in  || !y1_in ? 0.f : read_imagef(input, sampler, (int2)(x,     y + 1)).x;
    float bottom_right  = !x1_in || !y1_in ? 0.f : read_imagef(input, sampler, (int2)(x + 1, y + 1)).x;
    float top_center    = !x_in  ? 0.f : read_imagef(input, sampler, (int2)(x,     y - 1)).x;
    float center_left   = !y_in  ? 0.f : read_imagef(input, sampler, (int2)(x - 1, y)).x;
    float top_right     = !x1_in ? 0.f : read_imagef(input, sampler, (int2)(x + 1, y - 1)).x;
    float bottom_left   = !y1_in ? 0.f : read_imagef(input, sampler, (int2)(x - 1, y + 1)).x;
    float top_left      = read_imagef(input, sampler, (int2)(x - 1, y - 1)).x;

    top_center    -= 1.586134342059924 * (top_left + top_right);
    bottom_center -= 1.586134342059924 * (bottom_left + bottom_right);
    center_center -= 1.586134342059924 * (center_left + center_right + top_center + bottom_center);
    center_left   -= 1.586134342059924 * (top_left + bottom_left);


    if (x_in && y_in) {
      write_imagef(output, (int2)(x, y), (float4)(center_center));
    }

    if (x_in) {
      write_imagef(output, (int2)(x, y - 1), (float4)(top_center));
    }

    if (y_in) {
      write_imagef(output, (int2)(x - 1, y), (float4)(center_left));
    }

    write_imagef(output, (int2)(x - 1, y - 1), (float4)(top_left));

  }
}

__kernel void fwt_97_2d_ll_1(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx << 1;
  int y = gy << 1;

  if (x < width && y < height) {

    bool x_in = x + 1 < width;
    bool y_in = y + 1 < height;

    float center_center = read_imagef(input, sampler, (int2)(x,     y)).x;
    float top_left      = read_imagef(input, sampler, (int2)(x - 1, y - 1)).x;
    float top_center    = read_imagef(input, sampler, (int2)(x,     y - 1)).x;
    float center_left   = read_imagef(input, sampler, (int2)(x - 1, y)).x;
    float top_right     = !x_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y - 1)).x;
    float center_right  = !x_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y)).x;
    float bottom_left   = !y_in ? 0 : read_imagef(input, sampler, (int2)(x - 1, y + 1)).x;
    float bottom_center = !y_in ? 0 : read_imagef(input, sampler, (int2)(x,     y + 1)).x;
    float bottom_right  = !x_in || !y_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y + 1)).x;

    center_left   -= 0.052980118572961 * (top_left + bottom_left);
    center_right  -= 0.052980118572961 * (top_right + bottom_right);
    center_center -= 0.052980118572961 * (center_left + center_right + top_center + bottom_center);
    bottom_center -= 0.052980118572961 * (bottom_left + bottom_right);

    write_imagef(output, (int2)(x, y), (float4)(center_center));

    if (x_in) {
      write_imagef(output, (int2)(x + 1, y), (float4)(center_right));
    }

    if (y_in) {
      write_imagef(output, (int2)(x, y + 1), (float4)(bottom_center));
    }

    if (x_in && y_in) {
      write_imagef(output, (int2)(x + 1, y + 1), (float4)(bottom_right));
    }
  }
}

__kernel void fwt_97_2d_hh_2(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = (gx << 1) + 1;
  int y = (gy << 1) + 1;

  if (x - 1 < width && y - 1 < height) {
    bool x_in = x < width;
    bool y_in = y < height;

    bool x1_in = x + 1 < width;
    bool y1_in = y + 1 < height;

    float center_center = !x_in  || !y_in  ? 0.f : read_imagef(input, sampler, (int2)(x,     y)).x;
    float center_right  = !x1_in || !y_in  ? 0.f : read_imagef(input, sampler, (int2)(x + 1, y)).x;
    float bottom_center = !x_in  || !y1_in ? 0.f : read_imagef(input, sampler, (int2)(x,     y + 1)).x;
    float bottom_right  = !x1_in || !y1_in ? 0.f : read_imagef(input, sampler, (int2)(x + 1, y + 1)).x;
    float top_center    = !x_in  ? 0.f : read_imagef(input, sampler, (int2)(x,     y - 1)).x;
    float center_left   = !y_in  ? 0.f : read_imagef(input, sampler, (int2)(x - 1, y)).x;
    float top_right     = !x1_in ? 0.f : read_imagef(input, sampler, (int2)(x + 1, y - 1)).x;
    float bottom_left   = !y1_in ? 0.f : read_imagef(input, sampler, (int2)(x - 1, y + 1)).x;
    float top_left      = read_imagef(input, sampler, (int2)(x - 1, y - 1)).x;

    top_center    += 0.882911075530934 * (top_left + top_right);
    bottom_center += 0.882911075530934 * (bottom_left + bottom_right);
    center_center += 0.882911075530934 * (center_left + center_right + top_center + bottom_center);
    center_left   += 0.882911075530934 * (top_left + bottom_left);


    if (x_in && y_in) {
      write_imagef(output, (int2)(x, y), (float4)(center_center));
    }

    if (x_in) {
      write_imagef(output, (int2)(x, y - 1), (float4)(top_center));
    }

    if (y_in) {
      write_imagef(output, (int2)(x - 1, y), (float4)(center_left));
    }

    write_imagef(output, (int2)(x - 1, y - 1), (float4)(top_left));

  }
}

__kernel void fwt_97_2d_ll_2(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx << 1;
  int y = gy << 1;

  if (x < width && y < height) {

    bool x_in = x + 1 < width;
    bool y_in = y + 1 < height;

    float center_center = read_imagef(input, sampler, (int2)(x,     y)).x;
    float top_left      = read_imagef(input, sampler, (int2)(x - 1, y - 1)).x;
    float top_center    = read_imagef(input, sampler, (int2)(x,     y - 1)).x;
    float center_left   = read_imagef(input, sampler, (int2)(x - 1, y)).x;
    float top_right     = !x_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y - 1)).x;
    float center_right  = !x_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y)).x;
    float bottom_left   = !y_in ? 0 : read_imagef(input, sampler, (int2)(x - 1, y + 1)).x;
    float bottom_center = !y_in ? 0 : read_imagef(input, sampler, (int2)(x,     y + 1)).x;
    float bottom_right  = !x_in || !y_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y + 1)).x;

    center_left   += 0.443506852043971 * (top_left + bottom_left);
    center_right  += 0.443506852043971 * (top_right + bottom_right);
    center_center += 0.443506852043971 * (center_left + center_right + top_center + bottom_center);
    bottom_center += 0.443506852043971 * (bottom_left + bottom_right);

    int half_width  = (width  + 1) >> 1;
    int half_height = (height + 1) >> 1;

    write_imagef(output, (int2)(gx, gy), (float4)(center_center));

    if (x_in) {
      write_imagef(output, (int2)(gx + half_width, gy), (float4)(center_right));
    }

    if (y_in) {
      write_imagef(output, (int2)(gx, gy + half_height), (float4)(bottom_center));
    }

    if (x_in && y_in) {
      write_imagef(output, (int2)(gx + half_width, gy + half_height), (float4)(bottom_right));
    }
  }
}

__kernel void iwt_97_2d_ll_1(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx << 1;
  int y = gy << 1;

  if (x < width && y < height) {

    bool x_in = x + 1 < width;
    bool y_in = y + 1 < height;

    int half_width  = (width  + 1) >> 1;
    int half_height = (height + 1) >> 1;

    float center_center = read_imagef(input, sampler, (int2)(gx, gy)).x;

    float top_left      = !gx   || !gy   ? 0 : read_imagef(input, sampler, (int2)(gx + half_width - 1, gy + half_height - 1)).x;
    float bottom_left   = !gx   || !y_in ? 0 : read_imagef(input, sampler, (int2)(gx + half_width - 1, gy + half_height)).x;
    float top_right     = !x_in || !gy   ? 0 : read_imagef(input, sampler, (int2)(gx + half_width,     gy + half_height - 1)).x;
    float bottom_right  = !x_in || !y_in ? 0 : read_imagef(input, sampler, (int2)(gx + half_width,     gy + half_height)).x;

    float center_left   = !gx   ? 0 : read_imagef(input, sampler, (int2)(gx + half_width - 1, gy)).x;
    float top_center    = !gy   ? 0 : read_imagef(input, sampler, (int2)(gx,                  gy + half_height - 1)).x;
    float center_right  = !x_in ? 0 : read_imagef(input, sampler, (int2)(gx + half_width,     gy)).x;
    float bottom_center = !y_in ? 0 : read_imagef(input, sampler, (int2)(gx,                  gy + half_height)).x;

    top_center    -= 0.443506852043971 * (top_left + top_right);
    bottom_center -= 0.443506852043971 * (bottom_left + bottom_right);
    center_center -= 0.443506852043971 * (center_left + center_right + top_center + bottom_center);
    center_right  -= 0.443506852043971 * (top_right + bottom_right);

    write_imagef(output, (int2)(x, y), (float4)(center_center));

    if (y_in) {
      write_imagef(output, (int2)(x, y + 1), (float4)(bottom_center));
    }

    if (x_in) {
      write_imagef(output, (int2)(x + 1, y), (float4)(center_right));
    }

    if (x_in && y_in) {
      write_imagef(output, (int2)(x + 1, y + 1), (float4)(bottom_right));
    }
  }
}

__kernel void iwt_97_2d_hh_1(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = (gx << 1) + 1;
  int y = (gy << 1) + 1;

  if (x - 1 < width && y - 1 < height) {
    bool x_in = x < width;
    bool y_in = y < height;

    bool x1_in = x + 1 < width;
    bool y1_in = y + 1 < height;

    float center_center = !x_in  || !y_in  ? 0 : read_imagef(input, sampler, (int2)(x,     y)).x;
    float center_right  = !x1_in || !y_in  ? 0 : read_imagef(input, sampler, (int2)(x + 1, y)).x;
    float bottom_center = !x_in  || !y1_in ? 0 : read_imagef(input, sampler, (int2)(x,     y + 1)).x;
    float bottom_right  = !x1_in || !y1_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y + 1)).x;
    float top_center    = !x_in  ? 0 : read_imagef(input, sampler, (int2)(x,     y - 1)).x;
    float center_left   = !y_in  ? 0 : read_imagef(input, sampler, (int2)(x - 1, y)).x;
    float top_right     = !x1_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y - 1)).x;
    float bottom_left   = !y1_in ? 0 : read_imagef(input, sampler, (int2)(x - 1, y + 1)).x;
    float top_left      = read_imagef(input, sampler, (int2)(x - 1, y - 1)).x;

    center_left   -= 0.882911075530934 * (top_left + bottom_left);
    center_right  -= 0.882911075530934 * (top_right + bottom_right);
    center_center -= 0.882911075530934 * (center_left + center_right + top_center + bottom_center);
    top_center    -= 0.882911075530934 * (top_left + top_right);

    if (x_in && y_in) {
      write_imagef(output, (int2)(x, y), (float4)(center_center));
    }

    if (x_in) {
      write_imagef(output, (int2)(x, y - 1), (float4)(top_center));
    }

    if (y_in) {
      write_imagef(output, (int2)(x - 1, y), (float4)(center_left));
    }

    write_imagef(output, (int2)(x - 1, y - 1), (float4)(top_left));
  }
}

__kernel void iwt_97_2d_ll_2(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = gx << 1;
  int y = gy << 1;

  if (x < width && y < height) {

    bool x_in = x + 1 < width;
    bool y_in = y + 1 < height;

    float center_center = read_imagef(input, sampler, (int2)(x,     y)).x;
    float top_left      = read_imagef(input, sampler, (int2)(x - 1, y - 1)).x;
    float top_center    = read_imagef(input, sampler, (int2)(x,     y - 1)).x;
    float center_left   = read_imagef(input, sampler, (int2)(x - 1, y)).x;
    float top_right     = !x_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y - 1)).x;
    float center_right  = !x_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y)).x;
    float bottom_left   = !y_in ? 0 : read_imagef(input, sampler, (int2)(x - 1, y + 1)).x;
    float bottom_center = !y_in ? 0 : read_imagef(input, sampler, (int2)(x,     y + 1)).x;
    float bottom_right  = !x_in || !y_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y + 1)).x;

    top_center    += 0.052980118572961 * (top_left + top_right);
    bottom_center += 0.052980118572961 * (bottom_left + bottom_right);
    center_center += 0.052980118572961 * (center_left + center_right + top_center + bottom_center);
    center_right  += 0.052980118572961 * (top_right + bottom_right);

    write_imagef(output, (int2)(x, y), (float4)(center_center));

    if (y_in) {
      write_imagef(output, (int2)(x, y + 1), (float4)(bottom_center));
    }

    if (x_in) {
      write_imagef(output, (int2)(x + 1, y), (float4)(center_right));
    }

    if (x_in && y_in) {
      write_imagef(output, (int2)(x + 1, y + 1), (float4)(bottom_right));
    }
  }
}

__kernel void iwt_97_2d_hh_2(read_only image2d_t input, write_only image2d_t output, int width, int height) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int x = (gx << 1) + 1;
  int y = (gy << 1) + 1;

  if (x - 1 < width && y - 1 < height) {
    bool x_in = x < width;
    bool y_in = y < height;

    bool x1_in = x + 1 < width;
    bool y1_in = y + 1 < height;

    float center_center = !x_in  || !y_in  ? 0 : read_imagef(input, sampler, (int2)(x,     y)).x;
    float center_right  = !x1_in || !y_in  ? 0 : read_imagef(input, sampler, (int2)(x + 1, y)).x;
    float bottom_center = !x_in  || !y1_in ? 0 : read_imagef(input, sampler, (int2)(x,     y + 1)).x;
    float bottom_right  = !x1_in || !y1_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y + 1)).x;
    float top_center    = !x_in  ? 0 : read_imagef(input, sampler, (int2)(x,     y - 1)).x;
    float center_left   = !y_in  ? 0 : read_imagef(input, sampler, (int2)(x - 1, y)).x;
    float top_right     = !x1_in ? 0 : read_imagef(input, sampler, (int2)(x + 1, y - 1)).x;
    float bottom_left   = !y1_in ? 0 : read_imagef(input, sampler, (int2)(x - 1, y + 1)).x;
    float top_left      = read_imagef(input, sampler, (int2)(x - 1, y - 1)).x;

    center_left   += 1.586134342059924 * (top_left + bottom_left);
    center_right  += 1.586134342059924 * (top_right + bottom_right);
    center_center += 1.586134342059924 * (center_left + center_right + top_center + bottom_center);
    top_center    += 1.586134342059924 * (top_left + top_right);

    if (x_in && y_in) {
      write_imagef(output, (int2)(x, y), (float4)(center_center));
    }

    if (x_in) {
      write_imagef(output, (int2)(x, y - 1), (float4)(top_center));
    }

    if (y_in) {
      write_imagef(output, (int2)(x - 1, y), (float4)(center_left));
    }

    write_imagef(output, (int2)(x - 1, y - 1), (float4)(top_left));
  }
}
