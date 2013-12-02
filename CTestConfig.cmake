## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.

set(CTEST_PROJECT_NAME "pymad")
set(CTEST_NIGHTLY_START_TIME "00:00:00 CET")
set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "abp-cdash.web.cern.ch/abp-cdash/")
set(CTEST_DROP_LOCATION "submit.php?project=pymad")
set(CTEST_DROP_SITE_CDASH TRUE)
set(UPDATE_TYPE "git")

set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "16384")
