cmake_minimum_required(VERSION 3.10)

project(ncnn_predict)

find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(FATAL_ERROR "ncnn not found")
else()
    message(STATUS "ncnn found at ${ncnn_DIR}")
endif()

find_package(OpenCV REQUIRED)

add_definitions(-DNCNN_USE_GPU)

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