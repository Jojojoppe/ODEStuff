cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(ode VERSION 1.0.0 DESCRIPTION "ODE stuff")

# Set C++ standard (GiNaC requires at least C++11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GiNaC REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/odelib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)