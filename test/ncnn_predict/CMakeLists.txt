cmake_minimum_required(VERSION 3.10)
project(ncnn_predict)

find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(FATAL_ERROR "ncnn not found")
else()
    message(STATUS "ncnn found at ${ncnn_DIR}")
endif()

find_package(OpenCV REQUIRED)

add_executable(ncnn_predict
${CMAKE_SOURCE_DIR}/src/ncnn_detector.cpp
${CMAKE_SOURCE_DIR}/src/detect_utils.cpp
main.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/inc
)

target_link_libraries(
    ncnn_predict PRIVATE 
    ncnn
    ${OpenCV_LIBS}
)

# some definitions
add_definitions(-DNCNN_USE_GPU=0)

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

# check TEST_IMAGE_FILE　is set
set(TEST_IMAGE_FILE_SET ${TEST_IMAGE_FILE})
if(NOT TEST_IMAGE_FILE_SET)
    message(FATAL_ERROR "TEST_IMAGE_FILE not set. "
        "add -DTEST_IMAGE_FILE=path/to/image to CMakeLists.txt or with cmake -DTEST_IMAGE_FILE=path/to/image in command line")
else()
    # set TEST_IMAGE_FILE defnition
    add_definitions(-DTEST_IMAGE_FILE="${TEST_IMAGE_FILE}")
    message(STATUS "TEST_IMAGE_FILE=${TEST_IMAGE_FILE}")
endif()
