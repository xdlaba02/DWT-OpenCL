#include <iostream>

#include "dwt.h"

void print_help(const char *argv0) {
  std::cerr << "usage: " << "\n";
  std::cerr << argv0 << " {forward|inverse} {cpu|gpu} {53|97} <input.ppm> <output.ppm> \n";
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  bool inverse {};

  if (std::string(argv[1]) == "forward") {
    inverse = false;
  }
  else if (std::string(argv[1]) == "inverse") {
    inverse = true;
  }
  else {
    std::cerr << "unknown argument: " << argv[1] << '\n';
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  if (std::string(argv[2]) == "cpu") {
    if (std::string(argv[3]) == "53") {
      return dwt_53_cpu(argv[4], argv[5], inverse);
    }
    else if (std::string(argv[3]) == "97") {
      return dwt_97_cpu(argv[4], argv[5], inverse);
    }
    else {
      std::cerr << "unknown argument: " << argv[3] << '\n';
      print_help(argv[0]);
      return EXIT_FAILURE;
    }
  }
  else if (std::string(argv[2]) == "gpu") {
    if (std::string(argv[3]) == "53") {
      return dwt_53_gpu(argv[4], argv[5], inverse);
    }
    else if (std::string(argv[3]) == "97") {
      return dwt_97_gpu(argv[4], argv[5], inverse);
    }
    else {
      std::cerr << "unknown argument: " << argv[3] << '\n';
      print_help(argv[0]);
      return EXIT_FAILURE;
    }
  }
  else {
    std::cerr << "unknown argument: " << argv[2] << '\n';
    print_help(argv[0]);
    return EXIT_FAILURE;
  }
}
