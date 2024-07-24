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
${CMAKE_SOURCE_DIR}/src/sort_tracker.cpp
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
add_definitions(
    -DSAVE_VIDEO
    -DNCNN_USE_GPU=0
    -DYOLOV8_NCNN_MODEL="${CMAKE_SOURCE_DIR}/model/ncnn/yolov8n-face_ncnn_model"
    )

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
