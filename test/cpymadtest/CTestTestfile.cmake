
foreach(PYTEST clic aperture survey)
   add_test(test_${PYTEST} ${PYTHON} test_${PYTEST}.py)
endforeach()

# long test, can be excluded by running
# 'ctest -LE SLOW'
foreach(PYTEST lhc-b4 lhc)
   add_test(test_${PYTEST} ${PYTHON} test_${PYTEST}.py)
   set_tests_properties(test_${PYTEST} PROPERTIES LABELS SLOW FAIL_REGULAR_EXPRESSION "FAILED")
endforeach()
