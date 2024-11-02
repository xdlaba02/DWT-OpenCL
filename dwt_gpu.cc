#include "dwt_gpu.h"

#include <iostream>
#include <fstream>

#define CL_HPP_TARGET_OPENCL_VERSION 220
#include <CL/opencl.hpp>

int init_cl(cl::Device &selected_device, cl::Context &context, cl::CommandQueue &queue) {
  cl_int err {};

  {
    std::vector<cl::Platform> platforms {};

    err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS) {
      std::cerr << "cl::Platform::get failed" << "\n";
      return EXIT_FAILURE;
    }

    auto find_device = [&]() {
      for (size_t i = 0; i < platforms.size(); i++) {
        std::vector<cl::Device> platform_devices;

        err = platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &platform_devices);
        if (err != CL_SUCCESS) {
          std::cerr << "getDevices failed" << "\n";
          return false;
        }

        if (platform_devices.size() != 0) {
          auto platform = platforms[i].getInfo<CL_PLATFORM_NAME>(&err);
          if (err != CL_SUCCESS) {
            std::cerr << "getInfo failed" << "\n";
            return false;
          }

          auto device = platform_devices[0].getInfo<CL_DEVICE_NAME>(&err);
          if (err != CL_SUCCESS) {
            std::cerr << "getInfo failed" << "\n";
            return false;
          }

          std::cerr << platform << ": " << device << "\n";
          selected_device = platform_devices[0];
          return true;
        }
      }
      return false;
    };

    if (!find_device()) {
      std::cerr << "Device not found" << "\n";
      return EXIT_FAILURE;
    }
  }

  context = cl::Context(selected_device, nullptr, nullptr, nullptr, &err);

  if (err != CL_SUCCESS) {
    std::cerr << "Context creation failed" << "\n";
    return EXIT_FAILURE;
  }

  queue = cl::CommandQueue(context, selected_device, CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "CommandQueue creation failed" << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int compile_program(cl::Device &selected_device, cl::Context &context, const char *source_name, cl::Program &program) {
  std::ifstream source_file(source_name);
  if (!source_file) {
    std::cerr << source_name << " opening failed" << "\n";
    return EXIT_FAILURE;
  }

  cl::Program::Sources sources {{std::istreambuf_iterator<char>(source_file), std::istreambuf_iterator<char>()}};

  cl_int err {};
  program = cl::Program(context, sources, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Program creation failed" << "\n";
    return EXIT_FAILURE;
  }

  if (program.build(std::vector<cl::Device>(1, selected_device)) != CL_SUCCESS) {
    std::cerr << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device, &err).c_str();
    if (err != CL_SUCCESS) {
      std::cerr << "getBuildInfo failed" << "\n";
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
