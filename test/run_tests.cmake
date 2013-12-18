# This is a script for testing the source and submitting
# your results to a common server (cdash).
# You can see the server at:
# http://cern.ch/abp-cdash/index.php?project=pymad

# To run, change the source directory to a temporary path
# where you have the project checked out, then run the
# script with the command:
#   ctest -S run_tests.cmake

## -- SRC Dir
set(CTEST_SOURCE_DIRECTORY "")

## -- BIN Dir
set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")

## Remove old build folder if existing:
file(REMOVE_RECURSE ${CTEST_SOURCE_DIRECTORY}src/build)

## -- Dashboard type, possible values are 
##     'Experimental', 'Nightly' or 'Continuous'
set(DASHBOARD Experimental)


## -- Check that the user did what was told:
if(NOT CTEST_SOURCE_DIRECTORY)
    message(FATAL_ERROR "You did not set a source directory")
endif()

## -- Set hostname
## --------------------------
find_program(HOSTNAME_CMD NAMES hostname)
exec_program(${HOSTNAME_CMD} ARGS OUTPUT_VARIABLE HOSTNAME)

set(CTEST_SITE                          "$ENV{USER}@${HOSTNAME}")

## -- Set site / build name
## --------------------------

find_program(UNAME NAMES uname)
macro(getuname name flag)
  exec_program("${UNAME}" ARGS "${flag}" OUTPUT_VARIABLE "${name}")
endmacro(getuname)

getuname(osname -s)
getuname(cpu    -m)

set(CTEST_BUILD_NAME                    "${osname}-${cpu}")




# -----------------------------------------------------------  
# -- commands
# -----------------------------------------------------------  


## -- Update Command
set(CTEST_UPDATE_COMMAND "git")
set(CTEST_BUILD_COMMAND  "python setup.py build")

ctest_start(${DASHBOARD})
message(${CTEST_SOURCE_DIRECTORY}/src/build)
#ctest_update()
ctest_build(BUILD ${CTEST_SOURCE_DIRECTORY}/src)

file(GLOB build_libdir ${CTEST_SOURCE_DIRECTORY}/src/build/lib.*)
set(ENV{PYTHONPATH} "${build_libdir}:$ENV{PYTHONPATH}")

# only run quick tests:
ctest_test(EXCLUDE_LABEL SLOW)
# run all tests:
#ctest_test()

# uncomment this to submit to dashboard:
#ctest_submit()
