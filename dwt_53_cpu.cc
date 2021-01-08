#include "dwt.h"
#include "ppm.h"
#include "dwt_cpu.h"

#include <iostream>
#include <chrono>
#include <vector>

int dwt_53_cpu(const char *input, const char *output, bool inverse) {
  PPM input_image {};
  PPM output_image {};

  if (input_image.mmapPPM(input)) {
    std::cerr << "reading " << input << " failed" << "\n";
    return EXIT_FAILURE;
  }

  size_t width = input_image.width();
  size_t height = input_image.height();

  double time {};

  auto eventTime = [](const std::chrono::duration<double, std::milli> &event) {
    return event.count();
  };

  std::vector<int16_t> image_host(width * height);
  std::vector<int16_t> image_tmp(width * height);

  if (!inverse) {
    if (output_image.createPPM(output, input_image.width(), input_image.height(), (input_image.max_value() << 2) + 3)) {
      std::cerr << "creating " << output << " failed" << "\n";
      return EXIT_FAILURE;
    }

    for (size_t ch = 0; ch < 3; ch++) {
      for (size_t i = 0; i < image_host.size(); i++) {
        image_host[i] = input_image.get(i * 3 + ch) - (input_image.max_value() >> 1);
      }

      for (size_t depth = 1; depth < std::max(width, height); depth <<= 1) {
        int w = div_ceiled(width, depth);
        int h = div_ceiled(height, depth);

        auto start = std::chrono::steady_clock::now();
        fwt_53_2d(image_host.data(), w, h, width, image_tmp.data());
        auto end = std::chrono::steady_clock::now();

        time += eventTime(end - start);
      }

      for (size_t i = 0; i < image_host.size(); i++) {
        output_image.put(i * 3 + ch, image_host[i] + (output_image.max_value() >> 1));
      }
    }
  }
  else {
    if (output_image.createPPM(output, input_image.width(), input_image.height(), input_image.max_value() >> 2)) {
      std::cerr << "creating " << output << " failed" << "\n";
      return EXIT_FAILURE;
    }

    size_t max_depth = 1;
    while (max_depth < std::max(width, height)) {
      max_depth <<= 1;
    }
    max_depth >>= 1;

    for (size_t ch = 0; ch < 3; ch++) {
      for (size_t i = 0; i < image_host.size(); i++) {
        image_host[i] = input_image.get(i * 3 + ch) - (input_image.max_value() >> 1);
      }

      for (int depth = max_depth; depth > 0; depth >>= 1) {
        int w = div_ceiled(width, depth);
        int h = div_ceiled(height, depth);

        auto start = std::chrono::steady_clock::now();
        iwt_53_2d(image_host.data(), w, h, width, image_tmp.data());
        auto end = std::chrono::steady_clock::now();

        time += eventTime(end - start);
      }

      for (size_t i = 0; i < image_host.size(); i++) {
        output_image.put(i * 3 + ch, image_host[i] + (output_image.max_value() >> 1));        
      }
    }
  }

  std::cout << time << " ms\n";

  return EXIT_SUCCESS;
}
