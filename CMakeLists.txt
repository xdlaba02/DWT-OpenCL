cmake_minimum_required(VERSION 3.14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(dwt)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -pedantic -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -s")

# ugly but easy, ok for smol project
file(GLOB SRC_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME} ${SRC_FILES})

find_package(OpenCL REQUIRED)

target_link_libraries(${PROJECT_NAME} OpenCL)

add_custom_target(shaders ALL DEPENDS dwt_53.cl dwt_97.cl)

function(copy_shader SHADER)
  add_custom_command(
    OUTPUT ${SHADER}
    COMMAND cp ${PROJECT_SOURCE_DIR}/${SHADER} ${SHADER}
    MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/${SHADER}
  )
endfunction()

copy_shader(dwt_53.cl)
copy_shader(dwt_97.cl)
