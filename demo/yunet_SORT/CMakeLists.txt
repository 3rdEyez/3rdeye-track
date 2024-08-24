cmake_minimum_required(VERSION 3.18)
project(yunet_SORT)

find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(FATAL_ERROR "ncnn not found")
else()
    message(STATUS "ncnn found at ${ncnn_DIR}")
endif()

find_package(OpenCV REQUIRED)
set(OpenCV_SHARED OFF)

add_executable(yunet_SORT
    yunet_ncnn_detector.cpp
    main.cpp
)

target_link_libraries(
    yunet_SORT PRIVATE 
    ncnn
    ${OpenCV_LIBS}
    SORTTracker
)

# some definitions
add_definitions(-DNCNN_USE_GPU=0
                -DSAVE_VIDEO=1
                -DIMSHOW=1
                -DYOLOV8_NCNN_MODEL="${CMAKE_SOURCE_DIR}/models/ncnn/yolov8n_ncnn_model"
                -DTEST_IMAGE_FILE="${TEST_IMAGE_FILE}")

install(TARGETS yunet_SORT RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/ncnn/yunet_n_320_320_ncnn_model DESTINATION models)