add_library(odelib STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/odesolver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/solver_euler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/solver_rk.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/solver_radau.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/solver_bdf.cpp
)
target_include_directories(odelib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
)
target_include_directories(odelib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)