add_executable(test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
target_link_libraries(test PUBLIC odelib)