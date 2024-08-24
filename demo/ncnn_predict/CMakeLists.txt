cmake_minimum_required(VERSION 3.18)
project(ncnn_predict)

find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(FATAL_ERROR "ncnn not found")
else()
    message(STATUS "ncnn found at ${ncnn_DIR}")
endif()

find_package(OpenCV REQUIRED)

add_executable(ncnn_predict
ncnn_detector.cpp
main.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}
)

if(NOT TARGET SORTTracker)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../SORTTracker SORTTracker.lib)
endif()

target_link_libraries(
    ncnn_predict PRIVATE 
    ncnn
    ${OpenCV_LIBS}
    SORTTracker
)

# some definitions
add_definitions(-DNCNN_USE_GPU=0
                -DYOLOV8_NCNN_MODEL="${CMAKE_SOURCE_DIR}/models/ncnn/yolov8n_ncnn_model"
                -DTEST_IMAGE_FILE="${TEST_IMAGE_FILE}")

