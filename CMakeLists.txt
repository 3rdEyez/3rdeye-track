cmake_minimum_required(VERSION 3.10)

project(3rdeye-track)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# prepare test files
set(TEST_IMAGE_FILE "${CMAKE_SOURCE_DIR}/media/bus.jpg")
set(TEST_VIDEO_FILE "${CMAKE_SOURCE_DIR}/media/testvideo.mp4")
if (NOT EXISTS ${TEST_IMAGE_FILE})
    message(STATUS "Download test image")
    file(DOWNLOAD
        "https://cdn.fyrabh.top/repo_resurces/3rdeye_track/image/bus.jpg" ${TEST_IMAGE_FILE})
endif()
if  (NOT EXISTS ${TEST_VIDEO_FILE})
    message(STATUS "Download test video")
    file(DOWNLOAD
        "https://cdn.fyrabh.top/repo_resurces/3rdeye_track/video/testvideo.mp4" ${TEST_VIDEO_FILE})
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/SORTTracker)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/ncnn_predict)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/eigen_test)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/kalman_filter_test)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/yolov8_SORT)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/yunet_test)
add_subdirectory(${CMAKE_SOURCE_DIR}/test/basetest)
