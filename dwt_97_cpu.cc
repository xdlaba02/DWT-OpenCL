#include "dwt.h"
#include "ppm.h"
#include "dwt_cpu.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

int dwt_97_cpu(const char *input, const char *output, bool inverse) {
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

  std::vector<float> image_host(width * height);
  std::vector<float> image_tmp(width * height);

  if (!inverse) {
    if (output_image.createPPM(output, input_image.width(), input_image.height(), 8191)) {
      std::cerr << "creating " << output << " failed" << "\n";
      return EXIT_FAILURE;
    }

    for (size_t ch = 0; ch < 3; ch++) {
      for (size_t i = 0; i < image_host.size(); i++) {
        image_host[i] = input_image.get(i * 3 + ch);
      }

      for (size_t depth = 1; depth < std::max(width, height); depth <<= 1) {
        int w = div_ceiled(width, depth);
        int h = div_ceiled(height, depth);

        auto start = std::chrono::steady_clock::now();
        fwt_97_2d(image_host.data(), w, h, width, image_tmp.data());
        auto end = std::chrono::steady_clock::now();

        time += eventTime(end - start);
      }

      for (size_t i = 0; i < image_host.size(); i++) {
        output_image.put(i * 3 + ch, std::round((image_host[i] + (input_image.max_value() << 2) + 3) * 4));
      }
    }
  }
  else {
    if (output_image.createPPM(output, input_image.width(), input_image.height(), 255)) {
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
        image_host[i] = (input_image.get(i * 3 + ch) / 4.f) - 1023.f;
      }

      for (int depth = max_depth; depth > 0; depth >>= 1) {
        int w = div_ceiled(width, depth);
        int h = div_ceiled(height, depth);

        auto start = std::chrono::steady_clock::now();
        iwt_97_2d(image_host.data(), w, h, width, image_tmp.data());
        auto end = std::chrono::steady_clock::now();

        time += eventTime(end - start);
      }

      for (size_t i = 0; i < image_host.size(); i++) {
        output_image.put(i * 3 + ch, std::round(std::abs(image_host[i])));
      }
    }
  }

  std::cout << time << " ms\n";

  return EXIT_SUCCESS;
}
