
#Set binary and source directory..
if("" STREQUAL "${CTEST_SOURCE_DIRECTORY}")
   # Define source directory as current source directory
   file(GLOB _CTEST_SOURCE_DIR CTestTestfile.cmake)
   string(REGEX REPLACE "CTestTestfile.cmake" "" CTEST_SOURCE_DIR ${_CTEST_SOURCE_DIR})
else()
   # Define source directory from run script:
   set(CTEST_SOURCE_DIR ${CTEST_SOURCE_DIRECTORY})
endif()

if(NOT "" STREQUAL "${CTEST_BINARY_DIRECTORY}")
   # Define binary directory from run script:
   set(CTEST_BINARY_DIR ${CTEST_BINARY_DIRECTORY})
endif()
if(NOT DEFINED CTEST_BINARY_DIR)
   # Define binary directory as source directory (not through script)
   set(CTEST_BINARY_DIR ${CTEST_SOURCE_DIR})
endif()

# Note, for running in non-script mode,
# the variable CTEST_SOURCE_DIRECTORY is empty.. so this is fine
include(${CTEST_SOURCE_DIRECTORY}CTestSetup.cmake)

subdirs(test)
