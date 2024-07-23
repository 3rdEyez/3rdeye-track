cmake_minimum_required(VERSION 3.10)
project(yolov8_sort)

find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(FATAL_ERROR "ncnn not found")
else()
    message(STATUS "ncnn found at ${ncnn_DIR}")
endif()

find_package(OpenCV REQUIRED)

add_executable(yolov8_sort
${CMAKE_SOURCE_DIR}/src/ncnn_detector.cpp
${CMAKE_SOURCE_DIR}/src/detect_utils.cpp
${CMAKE_SOURCE_DIR}/src/kalman_filter.cpp
${CMAKE_SOURCE_DIR}/3rd_party/hungarian-algorithm-cpp/Hungarian.cpp
main.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/3rd_party
    ${CMAKE_SOURCE_DIR}/3rd_party/hungarian-algorithm-cpp
)

target_link_libraries(
    yolov8_sort PRIVATE 
    ncnn
    ${OpenCV_LIBS}
)

# some definitions
add_definitions(-DNCNN_USE_GPU)

# check YOLOV8_NCNN_MODEL is set
set(YOLOV8_NCNN_MODEL_SET ${YOLOV8_NCNN_MODEL})
if(NOT YOLOV8_NCNN_MODEL_SET)
    message(FATAL_ERROR "YOLOV8_NCNN_MODEL not set. "
        "add -DYOLOV8_NCNN_MODEL=path/to/model to CMakeLists.txt or with cmake -DYOLOV8_NCNN_MODEL=path/to/model in command line")
else()
    # set YOLOV8_NCNN_MODEL defnition
    add_definitions(-DYOLOV8_NCNN_MODEL="${YOLOV8_NCNN_MODEL}")
    message(STATUS "YOLOV8_NCNN_MODEL=${YOLOV8_NCNN_MODEL}")
endif()

# check TEST_VIDEO_FILE　is set
set(TEST_VIDEO_FILE ${TEST_VIDEO_FILE})
if(NOT TEST_VIDEO_FILE)
    message(FATAL_ERROR "TEST_VIDEO_FILE not set. "
        "add -DTEST_VIDEO_FILE=path/to/image to CMakeLists.txt or with cmake -DTEST_VIDEO_FILE=path/to/image in command line")
else()
    # set TEST_VIDEO_FILE defnition
    add_definitions(-DTEST_VIDEO_FILE="${TEST_VIDEO_FILE}")
    message(STATUS "TEST_VIDEO_FILE=${TEST_VIDEO_FILE}")
endif()
