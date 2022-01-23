cmake_minimum_required(VERSION 2.8)
project(ndt)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native -fopenmp -std=c++1y ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -fopenmp -std=c++1y")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c++1y ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++1y")

# Check C++11 or C++0x support
# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#   add_definitions(-DCOMPILEDWITHC11)
#   message(STATUS "Using flag -std=c++11.")
# elseif(COMPILER_SUPPORTS_CXX0X)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#   add_definitions(-DCOMPILEDWITHC0X)
#   message(STATUS "Using flag -std=c++0x.")
# else()
#   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()


if(NOT DEFINED python_version)
  set(python_version 3)
endif()

if(python_version STREQUAL 3)
  find_package(PythonLibs 3 REQUIRED)
  find_package(Boost COMPONENTS python38 numpy38 REQUIRED)
elseif(python_version STREQUAL 2)
  find_package(PythonLibs 2.7 REQUIRED)
  find_package(Boost COMPONENTS python numpy REQUIRED)
else()
  message("-Dpython_version=2 or -Dpython_version=3")
  EXIT()
endif()


set(Boost_LIBRARIES_2 ${Boost_LIBRARIES})

message(${Boost_LIBRARIES_2})


# Eigen3
find_package(Eigen3 REQUIRED)
if((NOT TARGET Eigen3::Eigen) AND (DEFINED EIGEN3_INCLUDE_DIR))
  add_library(AliasEigen3 INTERFACE)
  target_include_directories(AliasEigen3 INTERFACE ${EIGEN3_INCLUDE_DIR})
  add_library(Eigen3::Eigen ALIAS AliasEigen3)
endif()

# PCL
find_package(PCL REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}
  include
  ${PYTHON_INCLUDE_DIRS}
  Eigen3::Eigen
  ${PCL_INCLUDE_DIRS}
  )

add_library(${PROJECT_NAME} SHARED
  src/ndt_wrapper.cpp
  )

message(${Boost_LIBRARIES_2})

target_link_libraries(${PROJECT_NAME}
  python3.8
  ${Boost_LIBRARIES_2}
  ${PYTHON_LIBRARIES}
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)
