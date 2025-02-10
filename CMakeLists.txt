cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(ode VERSION 1.0.0 DESCRIPTION "ODE stuff")

add_subdirectory(${PROJECT_SOURCE_DIR}/odelib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)