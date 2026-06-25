# Tùy chọn dùng CMake thay cho Makefile (cần: brew install cmake).
#   cmake -B build && cmake --build build && ./build/run_tests
cmake_minimum_required(VERSION 3.16)
project(mlcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

file(GLOB_RECURSE LIB_SRC lib/src/*.cpp)
add_library(mlcpp ${LIB_SRC})
target_include_directories(mlcpp PUBLIC lib/include)

# Accelerate framework trên macOS cho phép nhân ma trận nhanh.
if(APPLE)
  target_compile_definitions(mlcpp PUBLIC MLCPP_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
  target_link_libraries(mlcpp PUBLIC "-framework Accelerate")
endif()

file(GLOB APP_SRC apps/*.cpp)
foreach(app ${APP_SRC})
  get_filename_component(name ${app} NAME_WE)
  add_executable(${name} ${app})
  target_link_libraries(${name} PRIVATE mlcpp)
endforeach()

file(GLOB TEST_SRC tests/*.cpp)
add_executable(run_tests ${TEST_SRC})
target_include_directories(run_tests PRIVATE tests)
target_link_libraries(run_tests PRIVATE mlcpp)

enable_testing()
add_test(NAME unit COMMAND run_tests)
