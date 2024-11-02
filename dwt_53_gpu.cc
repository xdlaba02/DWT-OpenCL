#include "dwt.h"
#include "ppm.h"
#include "dwt_gpu.h"

#include <iostream>
#include <vector>
#include <cmath>

#define CL_HPP_TARGET_OPENCL_VERSION 220
#include <CL/opencl.hpp>

int dwt_53_gpu(const char *input, const char *output, bool inverse) {
  cl_int err {};

  PPM input_image {};
  PPM output_image {};

  if (input_image.mmapPPM(input)) {
    std::cerr << "reading " << input << " failed" << "\n";
    return EXIT_FAILURE;
  }

  cl_int width = input_image.width();
  cl_int height = input_image.height();

  double time {};

  auto eventTime = [](cl::Event &event) {
    return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000.0;
  };

  cl::Device selected_device;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;

  if (init_cl(selected_device, context, queue) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  if (compile_program(selected_device, context, "dwt_53.cl", program) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  std::vector<int16_t> image_host(width * height);

  cl::Image2D image1_dev(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_SIGNED_INT16), width, height, 0, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateImage2D: image1_dev" << "\n";
    return EXIT_FAILURE;
  }

  cl::Image2D image2_dev(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_SIGNED_INT16), width, height, 0, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateImage2D: image2_dev" << "\n";
    return EXIT_FAILURE;
  }

  cl::Event write_event;
  std::vector<cl::Event> dwt_hh_events;
  std::vector<cl::Event> dwt_ll_events;
  cl::Event read_event;

  cl::NDRange local(16, 16);

  cl::array<size_t, 3> region {};
  region[0] = width;
  region[1] = height;
  region[2] = 1;

  if (!inverse) {
    if (output_image.createPPM(output, input_image.width(), input_image.height(), (input_image.max_value() << 2) + 3)) {
      std::cerr << "creating " << output << " failed" << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel fwt_53_2d_hh(program, "fwt_53_2d_hh", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: fwt_53_2d_hh " << err << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel fwt_53_2d_ll(program, "fwt_53_2d_ll", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: fwt_53_2d_ll " << err << "\n";
      return EXIT_FAILURE;
    }

    for (size_t ch = 0; ch < 3; ch++) {
      for (size_t i = 0; i < image_host.size(); i++) {
        image_host[i] = input_image.get(i * 3 + ch) - (input_image.max_value() >> 1);
      }

      err = queue.enqueueWriteImage(image1_dev, CL_FALSE, {}, region, 0, 0, image_host.data(), nullptr, &write_event);
      if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueWriteImage: image1_dev" << "\n";
        return EXIT_FAILURE;
      }

      for (int depth = 1; depth < std::max(width, height); depth <<= 1) {
        int w = div_ceiled(width, depth);
        int h = div_ceiled(height, depth);

        cl::NDRange global(ceil_to(div_ceiled(w, 2), local[0]), ceil_to(div_ceiled(h, 2), local[1]));

        fwt_53_2d_hh.setArg(0, image1_dev);
        fwt_53_2d_hh.setArg(1, image2_dev);
        fwt_53_2d_hh.setArg(2, w);
        fwt_53_2d_hh.setArg(3, h);

        fwt_53_2d_ll.setArg(0, image2_dev);
        fwt_53_2d_ll.setArg(1, image1_dev);
        fwt_53_2d_ll.setArg(2, w);
        fwt_53_2d_ll.setArg(3, h);

        dwt_hh_events.emplace_back();
        dwt_ll_events.emplace_back();

        err = queue.enqueueNDRangeKernel(fwt_53_2d_hh, cl::NullRange, global, local, nullptr, &dwt_hh_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: fwt_53_2d_hh" << "\n";
          return EXIT_FAILURE;
        }

        err = queue.enqueueNDRangeKernel(fwt_53_2d_ll, cl::NullRange, global, local, nullptr, &dwt_ll_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: fwt_53_2d_ll" << "\n";
          return EXIT_FAILURE;
        }
      }

      err = queue.enqueueReadImage(image1_dev, CL_FALSE, {}, region, 0, 0, image_host.data(), nullptr, &read_event);
      if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueReadImage: image1_dev" << "\n";
        return EXIT_FAILURE;
      }

      err = queue.finish();
      if (err != CL_SUCCESS) {
        std::cerr << "clFinish " << err << "\n";
        return EXIT_FAILURE;
      }

      for (size_t i = 0; i < image_host.size(); i++) {
        output_image.put(i * 3 + ch, image_host[i] + (output_image.max_value() >> 1));
      }

      time += eventTime(write_event) + eventTime(read_event);
    }
  }
  else {
    if (output_image.createPPM(output, input_image.width(), input_image.height(), input_image.max_value() >> 2)) {
      std::cerr << "creating " << output << " failed" << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel iwt_53_2d_hh(program, "iwt_53_2d_hh", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: iwt_53_2d_hh " << err << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel iwt_53_2d_ll(program, "iwt_53_2d_ll", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: iwt_53_2d_ll " << err << "\n";
      return EXIT_FAILURE;
    }

    int max_depth = 1;
    while (max_depth < std::max(width, height)) {
      max_depth <<= 1;
    }
    max_depth >>= 1;

    for (size_t ch = 0; ch < 3; ch++) {
      for (size_t i = 0; i < image_host.size(); i++) {
        image_host[i] = input_image.get(i * 3 + ch) - (input_image.max_value() >> 1);
      }

      err = queue.enqueueWriteImage(image1_dev, CL_FALSE, {}, region, 0, 0, image_host.data(), nullptr, &write_event);
      if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueWriteImage: image1_dev" << "\n";
        return EXIT_FAILURE;
      }

      for (int depth = max_depth; depth > 0; depth >>= 1) {
        int w = div_ceiled(width, depth);
        int h = div_ceiled(height, depth);

        cl::NDRange global(ceil_to(div_ceiled(w, 2), local[0]), ceil_to(div_ceiled(h, 2), local[1]));

        iwt_53_2d_ll.setArg(0, image1_dev);
        iwt_53_2d_ll.setArg(1, image2_dev);
        iwt_53_2d_ll.setArg(2, w);
        iwt_53_2d_ll.setArg(3, h);

        iwt_53_2d_hh.setArg(0, image2_dev);
        iwt_53_2d_hh.setArg(1, image1_dev);
        iwt_53_2d_hh.setArg(2, w);
        iwt_53_2d_hh.setArg(3, h);

        dwt_hh_events.emplace_back();
        dwt_ll_events.emplace_back();

        err = queue.enqueueNDRangeKernel(iwt_53_2d_ll, cl::NullRange, global, local, nullptr, &dwt_ll_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: iwt_53_2d_ll" << "\n";
          return EXIT_FAILURE;
        }

        err = queue.enqueueNDRangeKernel(iwt_53_2d_hh, cl::NullRange, global, local, nullptr, &dwt_hh_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: iwt_53_2d_hh" << "\n";
          return EXIT_FAILURE;
        }
      }

      err = queue.enqueueReadImage(image1_dev, CL_FALSE, {}, region, 0, 0, image_host.data(), nullptr, &read_event);
      if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueReadImage: image1_dev" << "\n";
        return EXIT_FAILURE;
      }

      err = queue.finish();
      if (err != CL_SUCCESS) {
        std::cerr << "clFinish" << "\n";
        return EXIT_FAILURE;
      }

      for (size_t i = 0; i < image_host.size(); i++) {
        output_image.put(i * 3 + ch, image_host[i] + (output_image.max_value() >> 1));
      }

      time += eventTime(write_event) + eventTime(read_event);
    }
  }

  for (size_t i = 0; i < dwt_hh_events.size(); i++) {
    time += eventTime(dwt_hh_events[i]) + eventTime(dwt_ll_events[i]);
  }

  std::cout << time << " ms\n";

  return EXIT_SUCCESS;
}
