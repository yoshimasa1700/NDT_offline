cmake_minimum_required(VERSION 2.8)
project(ndt)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -fopenmp -std=c++1y ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fopenmp -std=c++1y")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g3 -std=c++1y ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g3 -std=c++1y")


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

# Eigen3
find_package(Eigen3 REQUIRED)
if((NOT TARGET Eigen3::Eigen) AND (DEFINED EIGEN3_INCLUDE_DIR))
  add_library(AliasEigen3 INTERFACE)
  target_include_directories(AliasEigen3 INTERFACE ${EIGEN3_INCLUDE_DIR})
  add_library(Eigen3::Eigen ALIAS AliasEigen3)
endif()

include_directories(
  include
  ${PYTHON_INCLUDE_DIRS}
  Eigen3::Eigen
  )

add_library(${PROJECT_NAME} SHARED
  src/ndt_wrapper.cpp
  )


target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  Eigen3::Eigen
)

find_package(GTest REQUIRED)

include_directories(
  include
  ${GTEST_INCLUDE_DIRS}
  )

add_executable(test_sample
  test/test.cpp
  )

target_link_libraries(test_sample
  GTest::GTest GTest::Main
  ${Boost_LIBRARIES}
  Eigen3::Eigen
  )
