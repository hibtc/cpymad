
add_test(pymad_test_tfs ${PYTHON} test_tfs.py)
set_tests_properties(pymad_test_tfs PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
