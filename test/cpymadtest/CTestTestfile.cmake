
foreach(PYTEST clic aperture urvey)
   add_test(test_${PYTEST} ${PYTHON} test_${PYTEST}.py)
endforeach()

# long test, can be excluded by running
# 'ctest -LE SLOW'
foreach(PYTEST lhc-b4 lhc)
   add_test(test_${PYTEST} ${PYTHON} test_${PYTEST}.py)
   set_tests_properties(test_${PYTEST} PROPERTIES LABELS SLOW FAIL_REGULAR_EXPRESSION "FAILED" TIMEOUT 900)
endforeach()

add_test(test_madx ${PYTHON} test_madx.py)
add_test(test_resource ${PYTHON} test_resource.py)
add_test(test_model_locator ${PYTHON} test_model_locator.py)
