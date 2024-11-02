cmake_minimum_required(VERSION 3.28)

project(dwt)

find_package(OpenCL REQUIRED)

add_executable(dwt)

file(GLOB SOURCES CONFIGURE_DEPENDS *.cc *.h *.cl)

target_sources(dwt PRIVATE ${SOURCES})

target_link_libraries(dwt PRIVATE OpenCL::OpenCL)

add_custom_command(TARGET dwt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/dwt_53.cl $<TARGET_FILE_DIR:dwt>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/dwt_97.cl $<TARGET_FILE_DIR:dwt>
)
