
Tests can be run manually after a build has been performed.

If you are on a Linux 64bit, running python2.7, that command would look like:

PYTHONPATH=`pwd`/src/build/lib.linux-x86_64-2.7 ctest -LE SLOW

(the pythonpath is needed if you haven't installed yet)

The '-LE SLOW' argument excludes our rather slow tests

Tests can also be ran manually in the individual subfolders of test/

The file run_tests.cmake is configured and used when you want to submit
the results to the dashboard
