cmake_minimum_required(VERSION 3.18)
project(rv1103_rknn_inference_demo)
set(CMAKE_CXX_STANDARD 11)

##############################
### setup rknpu2 for rv1103
##############################
set(rknn2lib_armhf_uclibc_url "https://cdn.fyrabh.top/repo_resurces/3rdeye_track/libs/rknpu2.zip")
set(rknn2lib_armhf_uclibc_archive "${CMAKE_CURRENT_BINARY_DIR}/rknpu2.zip")
set(rknn2lib_armhf_uclibc_extract_path "${CMAKE_CURRENT_BINARY_DIR}")
set(RKNN_LIB "${CMAKE_CURRENT_BINARY_DIR}/rknpu2/Linux/armhf-uclibc/librknnmrt.a")

if (NOT EXISTS ${rknn2lib_armhf_uclibc_archive})
    file(DOWNLOAD ${rknn2lib_armhf_uclibc_url} ${rknn2lib_armhf_uclibc_archive} SHOW_PROGRESS)
    message(STATUS "Downloaded rknpu2.zip")
endif()

file (ARCHIVE_EXTRACT INPUT ${rknn2lib_armhf_uclibc_archive} DESTINATION ${rknn2lib_armhf_uclibc_extract_path})

##############################
### setup librga for rv1103
##############################
set(librga_armhf_uclibc_url "https://cdn.fyrabh.top/repo_resurces/3rdeye_track/libs/librga.zip")
set(librga_armhf_uclibc_archive "${CMAKE_CURRENT_BINARY_DIR}/librga.zip")
set(librga_armhf_uclibc_extract_path "${CMAKE_CURRENT_BINARY_DIR}/librga")
set(LIBRGA_LIB "${CMAKE_CURRENT_BINARY_DIR}/librga/Linux/armhf_uclibc/librga.a")

if (NOT EXISTS ${librga_armhf_uclibc_archive})
    file(DOWNLOAD ${librga_armhf_uclibc_url} ${librga_armhf_uclibc_archive} SHOW_PROGRESS)
    message(STATUS "Downloaded librga.zip")
endif()

file (ARCHIVE_EXTRACT INPUT ${librga_armhf_uclibc_archive} DESTINATION ${librga_armhf_uclibc_extract_path})


include_directories(
    ${rknn2lib_armhf_uclibc_extract_path}/rknpu2/include
    ${librga_armhf_uclibc_extract_path}/librga/include)

add_executable(rv1103_rknn_inference_demo
    main.cpp
    yunet_rknn_detector.cpp)

add_definitions(
    -DRV1106_1103)

target_link_libraries(rv1103_rknn_inference_demo
    ${RKNN_LIB}
    ${LIBRGA_LIB}
    ${OpenCV_LIB}
    SORTTracker)