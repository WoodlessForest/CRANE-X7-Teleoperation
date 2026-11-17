# Build the Python extension and its Linux robot communication dependencies.
cmake_minimum_required(VERSION 3.16)
project(py_x7 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_WITH_OPENMP "Enable OpenMP" ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if (BUILD_WITH_OPENMP)
    find_package(OpenMP)
endif()

# Use the source trees prepared by the dependency installer.
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/../third_party)

set(DXL_ROOT ${THIRD_PARTY_DIR}/DynamixelSDK)

if(NOT EXISTS "${DXL_ROOT}/c++/include/dynamixel_sdk/dynamixel_sdk.h")
    message(FATAL_ERROR "DynamixelSDK is missing. Run scripts/setup_linux.sh first.")
endif()

set(DXL_SRCS
    ${DXL_ROOT}/c++/src/dynamixel_sdk/group_bulk_read.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/group_bulk_write.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/group_sync_read.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/group_sync_write.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/group_handler.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/packet_handler.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/protocol1_packet_handler.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/protocol2_packet_handler.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/port_handler.cpp
    ${DXL_ROOT}/c++/src/dynamixel_sdk/port_handler_linux.cpp
)

add_library(dxl_sdk_obj OBJECT ${DXL_SRCS})

target_include_directories(dxl_sdk_obj PUBLIC
    ${DXL_ROOT}/c++/include
    ${DXL_ROOT}/c++/include/dynamixel_sdk
    ${DXL_ROOT}/c++
)

target_link_libraries(dxl_sdk_obj PUBLIC Threads::Threads)

set(RTMANIP_DIR
    ${THIRD_PARTY_DIR}/rt_manipulators_lib/rt_manipulators_lib
)
if(NOT EXISTS "${RTMANIP_DIR}/include/recording_feedback.hpp")
    message(FATAL_ERROR "Patched rt_manipulators_lib is missing. Run scripts/setup_linux.sh first.")
endif()
file(GLOB RTMANIP_SRCS ${RTMANIP_DIR}/src/*.cpp)

add_library(rt_manipulators_obj OBJECT ${RTMANIP_SRCS})

target_include_directories(rt_manipulators_obj PUBLIC
    ${RTMANIP_DIR}/include
    ${DXL_ROOT}/c++/include
    ${DXL_ROOT}/c++/include/dynamixel_sdk
)

target_link_libraries(rt_manipulators_obj PUBLIC
    yaml-cpp
    Threads::Threads
    Eigen3::Eigen
)

set(PY_X7_SRCS
    ${CMAKE_SOURCE_DIR}/bindings/pybind_x7.cpp
    ${CMAKE_SOURCE_DIR}/src/x7_controller.cpp
    ${CMAKE_SOURCE_DIR}/src/rt_manipulators_ik.cpp
)

# Link the controller and dependency objects into the importable module.
pybind11_add_module(py_x7 MODULE
    ${PY_X7_SRCS}
    $<TARGET_OBJECTS:dxl_sdk_obj>
    $<TARGET_OBJECTS:rt_manipulators_obj>
)

target_include_directories(py_x7 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${RTMANIP_DIR}/include
    ${DXL_ROOT}/c++/include
)

target_link_libraries(py_x7 PRIVATE
    yaml-cpp
    Eigen3::Eigen
    Threads::Threads
)

if (BUILD_WITH_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(py_x7 PRIVATE OpenMP::OpenMP_CXX)
endif()

# Place the extension beside the Python runtime modules.
set_target_properties(py_x7 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../python
)

message(STATUS "=============================================")
message(STATUS "  Configured target: py_x7")
message(STATUS "  Output -> ${CMAKE_SOURCE_DIR}/../python/")
message(STATUS "=============================================")
