#include "dwt.h"
#include "ppm.h"
#include "dwt_gpu.h"

#include <iostream>
#include <vector>
#include <cmath>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.hpp>

int dwt_97_gpu(const char *input, const char *output, bool inverse) {
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

  if (compile_program(selected_device, context, "dwt_97.cl", program) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  std::vector<float> image_host(width * height);

  cl::Image2D image1_dev(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), width, height, 0, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateImage2D: image1_dev" << "\n";
    return EXIT_FAILURE;
  }

  cl::Image2D image2_dev(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), width, height, 0, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateImage2D: image2_dev" << "\n";
    return EXIT_FAILURE;
  }

  cl::Event write_event;
  std::vector<cl::Event> dwt_hh_1_events;
  std::vector<cl::Event> dwt_ll_1_events;
  std::vector<cl::Event> dwt_hh_2_events;
  std::vector<cl::Event> dwt_ll_2_events;
  cl::Event read_event;

  cl::NDRange local(16, 16);

  cl::size_t<3> region {};
  region[0] = width;
  region[1] = height;
  region[2] = 1;

  if (!inverse) {
    if (output_image.createPPM(output, input_image.width(), input_image.height(), 8191)) {
      std::cerr << "creating " << output << " failed" << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel fwt_97_2d_hh_1(program, "fwt_97_2d_hh_1", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: fwt_97_2d_hh_1 " << err << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel fwt_97_2d_ll_1(program, "fwt_97_2d_ll_1", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: fwt_97_2d_ll_1 " << err << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel fwt_97_2d_hh_2(program, "fwt_97_2d_hh_2", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: fwt_97_2d_hh_2 " << err << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel fwt_97_2d_ll_2(program, "fwt_97_2d_ll_2", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: fwt_97_2d_ll_2 " << err << "\n";
      return EXIT_FAILURE;
    }

    for (size_t ch = 0; ch < 3; ch++) {
      for (size_t i = 0; i < image_host.size(); i++) {
        image_host[i] = input_image.get(i * 3 + ch);
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

        fwt_97_2d_hh_1.setArg(0, image1_dev);
        fwt_97_2d_hh_1.setArg(1, image2_dev);
        fwt_97_2d_hh_1.setArg(2, w);
        fwt_97_2d_hh_1.setArg(3, h);

        fwt_97_2d_ll_1.setArg(0, image2_dev);
        fwt_97_2d_ll_1.setArg(1, image1_dev);
        fwt_97_2d_ll_1.setArg(2, w);
        fwt_97_2d_ll_1.setArg(3, h);

        fwt_97_2d_hh_2.setArg(0, image1_dev);
        fwt_97_2d_hh_2.setArg(1, image2_dev);
        fwt_97_2d_hh_2.setArg(2, w);
        fwt_97_2d_hh_2.setArg(3, h);

        fwt_97_2d_ll_2.setArg(0, image2_dev);
        fwt_97_2d_ll_2.setArg(1, image1_dev);
        fwt_97_2d_ll_2.setArg(2, w);
        fwt_97_2d_ll_2.setArg(3, h);

        dwt_hh_1_events.emplace_back();
        dwt_ll_1_events.emplace_back();
        dwt_hh_2_events.emplace_back();
        dwt_ll_2_events.emplace_back();

        err = queue.enqueueNDRangeKernel(fwt_97_2d_hh_1, cl::NullRange, global, local, nullptr, &dwt_hh_1_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: fwt_97_2d_hh_1" << "\n";
          return EXIT_FAILURE;
        }

        err = queue.enqueueNDRangeKernel(fwt_97_2d_ll_1, cl::NullRange, global, local, nullptr, &dwt_ll_1_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: fwt_97_2d_ll_1" << "\n";
          return EXIT_FAILURE;
        }

        err = queue.enqueueNDRangeKernel(fwt_97_2d_hh_2, cl::NullRange, global, local, nullptr, &dwt_hh_2_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: fwt_97_2d_hh_2" << "\n";
          return EXIT_FAILURE;
        }

        err = queue.enqueueNDRangeKernel(fwt_97_2d_ll_2, cl::NullRange, global, local, nullptr, &dwt_ll_2_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: fwt_97_2d_ll_2" << "\n";
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
        output_image.put(i * 3 + ch, std::round((image_host[i] + (input_image.max_value() << 2) + 3) * 4));
      }

      time += eventTime(write_event) + eventTime(read_event);
    }
  }
  else {
    if (output_image.createPPM(output, input_image.width(), input_image.height(), 255)) {
      std::cerr << "creating " << output << " failed" << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel iwt_97_2d_hh_1(program, "iwt_97_2d_hh_1", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: iwt_97_2d_hh_1 " << err << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel iwt_97_2d_ll_1(program, "iwt_97_2d_ll_1", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: iwt_97_2d_ll_1 " << err << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel iwt_97_2d_hh_2(program, "iwt_97_2d_hh_2", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: iwt_97_2d_hh_2 " << err << "\n";
      return EXIT_FAILURE;
    }

    cl::Kernel iwt_97_2d_ll_2(program, "iwt_97_2d_ll_2", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "clCreateKernel: iwt_97_2d_ll_2 " << err << "\n";
      return EXIT_FAILURE;
    }

    int max_depth = 1;
    while (max_depth < std::max(width, height)) {
      max_depth <<= 1;
    }
    max_depth >>= 1;

    for (size_t ch = 0; ch < 3; ch++) {
      for (size_t i = 0; i < image_host.size(); i++) {
        image_host[i] = (input_image.get(i * 3 + ch) / 4.f) - 1023.f;
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

        iwt_97_2d_ll_1.setArg(0, image1_dev);
        iwt_97_2d_ll_1.setArg(1, image2_dev);
        iwt_97_2d_ll_1.setArg(2, w);
        iwt_97_2d_ll_1.setArg(3, h);

        iwt_97_2d_hh_1.setArg(0, image2_dev);
        iwt_97_2d_hh_1.setArg(1, image1_dev);
        iwt_97_2d_hh_1.setArg(2, w);
        iwt_97_2d_hh_1.setArg(3, h);

        iwt_97_2d_ll_2.setArg(0, image1_dev);
        iwt_97_2d_ll_2.setArg(1, image2_dev);
        iwt_97_2d_ll_2.setArg(2, w);
        iwt_97_2d_ll_2.setArg(3, h);

        iwt_97_2d_hh_2.setArg(0, image2_dev);
        iwt_97_2d_hh_2.setArg(1, image1_dev);
        iwt_97_2d_hh_2.setArg(2, w);
        iwt_97_2d_hh_2.setArg(3, h);

        dwt_hh_1_events.emplace_back();
        dwt_ll_1_events.emplace_back();
        dwt_hh_2_events.emplace_back();
        dwt_ll_2_events.emplace_back();

        err = queue.enqueueNDRangeKernel(iwt_97_2d_ll_1, cl::NullRange, global, local, nullptr, &dwt_ll_1_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: iwt_97_2d_ll_1" << "\n";
          return EXIT_FAILURE;
        }

        err = queue.enqueueNDRangeKernel(iwt_97_2d_hh_1, cl::NullRange, global, local, nullptr, &dwt_hh_1_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: iwt_97_2d_hh_1" << "\n";
          return EXIT_FAILURE;
        }


        err = queue.enqueueNDRangeKernel(iwt_97_2d_ll_2, cl::NullRange, global, local, nullptr, &dwt_ll_2_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: iwt_97_2d_ll_2" << "\n";
          return EXIT_FAILURE;
        }

        err = queue.enqueueNDRangeKernel(iwt_97_2d_hh_2, cl::NullRange, global, local, nullptr, &dwt_hh_2_events.back());
        if (err != CL_SUCCESS) {
          std::cerr << "clEnqueueNDRangeKernel: iwt_97_2d_hh_2" << "\n";
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
        output_image.put(i * 3 + ch, std::round(std::abs(image_host[i])));
      }

      time += eventTime(write_event) + eventTime(read_event);
    }
  }

  for (size_t i = 0; i < dwt_hh_1_events.size(); i++) {
    time += eventTime(dwt_hh_1_events[i]) + eventTime(dwt_ll_1_events[i]) + eventTime(dwt_hh_2_events[i]) + eventTime(dwt_ll_2_events[i]);
  }

  std::cout << time << " ms\n";

  return EXIT_SUCCESS;
}
