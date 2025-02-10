add_executable(test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
target_link_libraries(test PRIVATE 
  odelib
)

add_executable(ginactest
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ginactest.cpp
)
target_link_libraries(ginactest PRIVATE
  ${GINAC_LIBRARIES}
)