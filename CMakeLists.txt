cmake_minimum_required(VERSION 3.10)

project(3rdeye-track)

add_subdirectory(${CMAKE_SOURCE_DIR}/test/ncnn_predict)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/eigen_test)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/kalman_filter_test)
