cmake_minimum_required(VERSION 3.0)

project(SORTTracker)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

add_library(SORTTracker STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/hungarian-algorithm-cpp/Hungarian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cv_utils.cpp
    src/kalman_filter.cpp
    src/sort_tracker.cpp
    src/tracker_wrapper.cpp
)

target_include_directories(SORTTracker PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/share
)

target_include_directories(SORTTracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/hungarian-algorithm-cpp
)

target_link_libraries(SORTTracker ${OpenCV_LIBS})

install(TARGETS SORTTracker DESTINATION lib)
file(GLOB SORTTracker_headers ${CMAKE_CURRENT_SOURCE_DIR}/share/*.h)
install(FILES ${SORTTracker_headers} DESTINATION inc)
