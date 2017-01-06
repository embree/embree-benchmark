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

#OPTION(TUTORIAL_USE_PTEX "Enables ptex texture support for tutorials.") # FIXME: remove Ptex support

IF (TUTORIAL_USE_PTEX)
  FIND_PACKAGE(Ptex REQUIRED)
  ADD_DEFINITIONS(-DUSE_PTEX)
  INCLUDE_DIRECTORIES(${PTEX_INCLUDE_DIRS})
ENDIF()

ADD_SUBDIRECTORY(common)

SET (__XEON__ 1)
ADD_SUBDIRECTORY(pathtracer)

IF (ENABLE_XEON_PHI_SUPPORT)
  SET (__XEON__ 0)
  SET (__HOST__ 1)
  ADD_SUBDIRECTORY(pathtracer pathtracer_host)
  SET (__HOST__ 0)
  ADD_SUBDIRECTORY(pathtracer pathtracer_device)
ENDIF()
