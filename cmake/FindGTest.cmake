# This file is used to find googletest library in CMake script, based on code
# from
#
#   https://github.com/BVLC/caffe/blob/master/cmake/Modules/FindGlog.cmake
#
# which is licensed under the 3-Clause BSD License.
#
# - Try to find googletest
#
# The following variables are optionally searched for defaults
#  GTEST_ROOT_DIR:            Base directory where all GTEST components are found
#
# The following are set after configuration is done:
#  GTEST_FOUND
#  GTEST_INCLUDE_DIRS
#  GTEST_LIBRARIES
#  GTEST_MAIN_LIBRARIES
#  GTEST_LIBRARY_DIRS
#  GTEST_MAIN_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(GTEST_ROOT_DIR "" CACHE PATH "Folder contains GoogleTest")

find_path(GTEST_INCLUDE_DIR gtest/gtest.h
    PATHS ${GTEST_ROOT_DIR})

if(MSVC)
    find_library(GTEST_LIBRARY_RELEASE libgtest_static
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES Release)
    find_library(GTEST_MAIN_LIBRARY_RELEASE libgtest_main_static
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES Release)

    find_library(GTEST_LIBRARY_DEBUG libgtest_static
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES Debug)
    find_library(GTEST_MAIN_LIBRARY_DEBUG libgtest_main_static
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES Debug)

    set(GTEST_LIBRARY optimized ${GTEST_LIBRARY_RELEASE} debug ${GTEST_LIBRARY_DEBUG})
    set(GTEST_MAIN_LIBRARY optimized ${GTEST_MAIN_LIBRARY_RELEASE} debug ${GTEST_MAIN_LIBRARY_DEBUG})
else()
    find_library(GTEST_LIBRARY gtest
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES lib lib64)
    find_library(GTEST_MAIN_LIBRARY gtest_main
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES lib lib64)
endif()

find_package_handle_standard_args(Gtest DEFAULT_MSG GTEST_INCLUDE_DIR GTEST_LIBRARY GTEST_MAIN_LIBRARY)

if(GTEST_FOUND)
  set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})
  set(GTEST_LIBRARIES ${GTEST_LIBRARY})
  set(GTEST_MAIN_LIBRARIES ${GTEST_MAIN_LIBRARY})
  message(STATUS "Found googletest    (include: ${GTEST_INCLUDE_DIR}, library: ${GTEST_LIBRARY}, ${GTEST_MAIN_LIBRARY})")
  mark_as_advanced(GTEST_ROOT_DIR GTEST_INCLUDE_DIR
                   GTEST_LIBRARY_RELEASE GTEST_LIBRARY_DEBUG GTEST_LIBRARY
                   GTEST_MAIN_LIBRARY_RELEASE GTEST_MAIN_LIBRARY_DEBUG GTEST_MAIN_LIBRARY)
endif()
