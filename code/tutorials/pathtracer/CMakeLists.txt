## ======================================================================== ##
## Copyright 2009-2015 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

INCLUDE(tutorial)
ADD_TUTORIAL(pathtracer)


SET(BUILD_TUTORIALS_OPTIX ON CACHE BOOL "Enable to build Embree OPTIX tutorials")

IF (__XEON__ AND BUILD_TUTORIALS_OPTIX)

  SET(OptiX_INSTALL_DIR "${PROJECT_SOURCE_DIR}/../optix" CACHE PATH "Patch to OptiX installation.")
  FIND_PATH(OPTIX_CMAKE_PATH FindOptiX.cmake ${OptiX_INSTALL_DIR}/SDK/CMake)
  MARK_AS_ADVANCED(OPTIX_CMAKE_PATH)
  IF (NOT OPTIX_CMAKE_PATH)
    MESSAGE(FATAL_ERROR "OptiX installation not found. Please set OptiX_INSTALL_DIR to point to the OptiX installation.")
  ENDIF ()
  SET(CMAKE_MODULE_PATH ${OPTIX_CMAKE_PATH} ${CMAKE_MODULE_PATH})

  FIND_PACKAGE(OptiX REQUIRED)
  FIND_PACKAGE(CUDA 2.3 REQUIRED)

  SET(CUDA_64_BIT_DEVICE_CODE ON)
  SET(CUDA_NVCC_FLAGS --gpu-architecture sm_20 --use_fast_math)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS} ${OptiX_INCLUDE})

  FUNCTION(ADD_OPTIX_EXECUTABLE target_name)
   CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})
   CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )
   ADD_EXECUTABLE(${target_name}
      ${source_files}
      ${generated_files}
      ${cmake_options}
      )
  ENDFUNCTION()

  ADD_OPTIX_EXECUTABLE(pathtracer_optix
    pathtracer.cpp
    pathtracer_optix.cpp
    pathtracer_optix.cu
   )
  TARGET_LINK_LIBRARIES(pathtracer_optix tutorial image optix ${optix_rpath})

ENDIF()
