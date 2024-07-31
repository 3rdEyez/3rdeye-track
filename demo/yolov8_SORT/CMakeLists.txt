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
ncnn_detector.cpp
main.cpp
)

target_link_libraries(
    yolov8_sort PRIVATE 
    ncnn
    ${OpenCV_LIBS}
    SORTTracker
)

# some definitions
add_definitions(-DSAVE_VIDEO
                -DNCNN_USE_GPU=0
                -DYOLOV8_NCNN_MODEL="${CMAKE_SOURCE_DIR}/models/ncnn/yolov8n_ncnn_model"
                -DTEST_VIDEO_FILE="${TEST_VIDEO_FILE}")