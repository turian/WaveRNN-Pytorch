cmake_minimum_required (VERSION 3.0)

project(WaveRNN)

set(CMAKE_CXX_STANDARD 11)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable (vocoder vocoder.cpp)
add_library(wavernn net_impl.cpp)
target_link_libraries(vocoder wavernn Eigen3::Eigen)


