cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(ode VERSION 1.0.0 DESCRIPTION "ODE stuff")

add_subdirectory(${PROJECT_SOURCE_DIR}/odelib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)

include(${PROJECT_SOURCE_DIR}/cmake/GitExternal.cmake)
Git_External(
  ${PROJECT_SOURCE_DIR}/external/cmake_common 
  https://github.com/Eyescale/CMake.git
  master
)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/cmake_common)
include(SubProject)

Git_External(
  ${PROJECT_SOURCE_DIR}/external/ginac 
  https://github.com/AlexeiSheplyakov/GiNaC.git
  master
)
add_subproject(external/ginac)