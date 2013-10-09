
foreach(PYTEST clic aperture survey)
   add_test(test_${PYTEST} ${PYTHON} test_${PYTEST}.py)
endforeach()

# long test, can be excluded by running
# 'ctest -E LONG'
foreach(PYTEST lhc-b4 lhc)
   add_test(test_${PYTEST}_LONG ${PYTHON} test_${PYTEST}.py)
endforeach()
