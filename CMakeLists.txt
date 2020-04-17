# - Basic CMake setup
# Check version meets our requirements
# Declare project, which will configure compiler for us
cmake_minimum_required(VERSION 3.14)
project(MLClustering)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Falaise REQUIRED)

# Need ROOT library separately
find_package(ROOT REQUIRED)

#Need also Python Packages
find_package(Python 3 REQUIRED Interpreter Development NumPy)

# find the headers
include_directories(.)
# temporary workaround to get needed libraries
#include_directories(${PYTHON_INCLUDE_DIRS})
#include_directories(/Users/matteoceschia/snemo-sdk/lib/python3.7/site-packages/numpy/core/include/)
#LINK_DIRECTORIES(/Users/matteoceschia/snemo-sdk/opt/python/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin)

# Build a dynamic library from our sources
add_library(MLClustering SHARED mlclustering_library.h
			      mlclustering_module.h
			      mlclustering_library.cpp
			      mlclustering_module.cpp)

target_link_libraries(MLClustering PUBLIC Falaise::FalaiseModule PRIVATE Python::Python Python::NumPy)

# Build a dynamic library from our sources for algorithm only
# required for testing executables.
include_directories(${ROOT_INCLUDE_DIRS})
add_library(MLClusAlgorithm SHARED mlclustering_library.h
			  mlclustering_library.cpp)
target_link_libraries(MLClusAlgorithm PUBLIC ${ROOT_LIBRARIES})

#set(CMAKE_INSTALL_PLUGINDIR "/Users/matteoceschia/snemo-sdk/Cellar/falaise/4.0.0/lib/Falaise/modules/")
#install(TARGETS MLClustering DESTINATION ${CMAKE_INSTALL_PLUGINDIR})

# Configure example pipeline script for use from the build dir
configure_file("mlcModuleExample.conf.in" "mlcModuleExample.conf" @ONLY)

# Build the testing code, tell CTest about it
#enable_testing()
#set(CMAKE_CXX_STANDARD 11)

# main catch library compile only once
#add_library(Catch testing/catch-main.cpp)
#target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/testing)

# add test executable
#include_directories(. testing)
#add_executable(trial testing/trial.cpp)
#target_link_libraries(trial PUBLIC Catch MLClusAlgorithm)

# add test executable
#add_executable(graphtrial testing/graphtrial.cpp)
#target_link_libraries(graphtrial PUBLIC Catch MLClusAlgorithm)

# add test executable
#add_executable(cleanuptrial testing/cleanuptrial.cpp)
#target_link_libraries(cleanuptrial PUBLIC Catch MLClusAlgorithm)

# CTest comand
#add_test(NAME trial
#  COMMAND trial -s
#)

#add_test(NAME graphtrial
#  COMMAND graphtrial -s
#)

#add_test(NAME cleanuptrial
#  COMMAND cleanuptrial -s
#)
