set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.10)
project(ght_face_eyes CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(ght_face_eyes src/ght_face_eyes.cpp)
target_link_libraries(ght_face_eyes PRIVATE ${OpenCV_LIBS})

# Output to vision/bin
set_target_properties(ght_face_eyes PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
