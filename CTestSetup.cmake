
# Find Python..
if(EXISTS "/usr/bin/python2.6") # For our official testing servers, should use this
   set(PYTHON "/usr/bin/python2.6")
else() # For other machines, hope ctest is clever enough..
   find_program(PYTHON python python2)
endif()

