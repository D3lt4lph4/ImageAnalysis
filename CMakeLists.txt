cmake_minimum_required(VERSION 2.6)

project(ImageAnalysisAssignment)

find_package( OpenCV REQUIRED )

file(GLOB_RECURSE source_file1 "src/modelMaker/*cpp")
file(GLOB_RECURSE source_file2 "src/solver/*cpp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH build/bin)
include_directories(include)

add_executable(imageDecision ${source_file2})
target_link_libraries(imageDecision "-lutil" ${OpenCV_LIBS})

add_executable(modelsCreator ${source_file1})
target_link_libraries(modelsCreator "-lutil" ${OpenCV_LIBS})
