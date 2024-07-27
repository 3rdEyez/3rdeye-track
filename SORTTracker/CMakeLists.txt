cmake_minimum_required(VERSION 3.0)

project(SORTTracker)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/hungarian-algorithm-cpp
)

add_library(SORTTracker STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/hungarian-algorithm-cpp/Hungarian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/utils.cpp
    src/kalman_filter.cpp
    src/sort_tracker.cpp
    share/tracker.cpp
)

target_link_libraries(SORTTracker ${OpenCV_LIBS})

install(TARGETS SORTTracker DESTINATION lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/share/tracker.h DESTINATION inc)