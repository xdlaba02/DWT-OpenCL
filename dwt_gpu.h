#pragma once

namespace cl {
  class Device;
  class Context;
  class CommandQueue;
  class Program;
}

int init_cl(cl::Device &selected_device, cl::Context &context, cl::CommandQueue &queue);
int compile_program(cl::Device &selected_device, cl::Context &context, const char *source_name, cl::Program &program);
