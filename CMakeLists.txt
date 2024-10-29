cmake_minimum_required(VERSION 3.18)

project(3rdeye-track)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# prepare resources files
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/models)
    message(STATUS "Downloading models...")
    file(DOWNLOAD  http://cdn.fyrabh.top/repo_resurces/3rdeye_track/model/models.zip
    ${CMAKE_CURRENT_BINARY_DIR}/models.zip STATUS status)
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/models.zip DESTINATION
    ${CMAKE_SOURCE_DIR})
endif()

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/media)
    message(STATUS "Downloading media...")
    file(DOWNLOAD http://cdn.fyrabh.top/repo_resurces/3rdeye_track/media.zip
        ${CMAKE_CURRENT_BINARY_DIR}/media.zip STATUS status)
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/media.zip DESTINATION
    ${CMAKE_SOURCE_DIR})
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/demo/ncnn_predict)
add_subdirectory(${CMAKE_SOURCE_DIR}/demo/yolov8_SORT)
add_subdirectory(${CMAKE_SOURCE_DIR}/demo/yunet_SORT)

# you can't add this demo here, because it needs arm-rockchip830-linux-uclibcgnueabihf-gcc
# you should goto it's directory and build it manually↓↓↓↓
# add_subdirectory(${CMAKE_SOURCE_DIR}/demo/rv1103_rknn_inference)
