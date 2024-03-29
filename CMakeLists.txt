PROJECT(GENIE4L2 LANGUAGES CXX CUDA)

CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
set (CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-sign-compare")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_CUDA_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-sign-compare")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(
  "genie-dev/src/"
  "/usr/loca/cuda/include/"
)

find_package(CUDA 9.0 REQUIRED)
find_package(Boost COMPONENTS system program_options serialization iostreams REQUIRED)
find_package(MPI REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(fmt REQUIRED)

# AUX_SOURCE_DIRECTORY(. DIR_SRCS)
ADD_EXECUTABLE(genie_nn 
  "main.cpp" 
  "genie4l2.cu"
  "genie4l2_dist.cu"
)
add_library(genie4l2 STATIC "genie4l2.cu" "genie4l2_dist.cu")
TARGET_LINK_LIBRARIES( genie_nn LINK_PUBLIC "${CMAKE_CURRENT_LIST_DIR}/genie-dev/build/lib/libgenie.a" ${Boost_LIBRARIES} fmt::fmt)