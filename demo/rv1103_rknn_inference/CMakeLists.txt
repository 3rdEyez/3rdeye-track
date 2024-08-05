cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER /opt/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc)
set(CMAKE_CXX_COMPILER /opt/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++)

project(rv1103_rknn_inference_demo)


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

################################################
### setup opencv-mobile for rv1103
### see https://github.com/nihui/opencv-mobile
################################################
if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/opencv-mobile)
    file(DOWNLOAD https://cdn.fyrabh.top/repo_resurces/3rdeye_track/libs/opencv-mobile-4.10.0-luckfox-pico.zip
        ${CMAKE_CURRENT_BINARY_DIR}/opencv-mobile.zip SHOW_PROGRESS)
    message(STATUS "Downloaded opencv-mobile.zip")
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/opencv-mobile.zip DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/opencv-mobile-4.10.0-luckfox-pico ${CMAKE_CURRENT_BINARY_DIR}/opencv-mobile)
endif()
set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv-mobile/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)


##################################
### add allocator sources
##################################
file(GLOB_RECURSE allocator_srcs ${CMAKE_CURRENT_SOURCE_DIR}/allocator/*.cpp)
add_library(allocator STATIC ${allocator_srcs})
target_include_directories(allocator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/allocator/dma)
target_include_directories(allocator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/allocator/drm)


# include SORTTracker
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../SORTTracker SORTTracker.lib)

include_directories(
    ${rknn2lib_armhf_uclibc_extract_path}/rknpu2/include
    ${librga_armhf_uclibc_extract_path}/include)

message((${librga_armhf_uclibc_extract_path}/librga/include))

add_executable(rv1103_rknn_inference_demo
    main.cpp
    yunet_rknn_detector.cpp)

add_definitions(
    -DRV1106_1103)

target_link_libraries(rv1103_rknn_inference_demo
    ${RKNN_LIB}
    ${LIBRGA_LIB}
    ${OpenCV_LIB}
    allocator
    SORTTracker)

install(TARGETS rv1103_rknn_inference_demo RUNTIME DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../models/rknn/yunet_n_320_320.rknn DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../media/bus.jpg DESTINATION .)
