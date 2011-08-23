These are the files needed to create a python library of madx using cython.

Prerequisites:
 # General:
  - python with numpy
 # For cpymad:
 - cython
 - gfortran or similar
 - Mad-X shared library
 - madextern.h in your include path
 # For jpymad:
 - Java
 - JMad
 - Py4J

Installation:
    See http://cern.ch/pymad/installation


Usage:
    # Once installed this is a nice example showing current
    # usability (run from examples folder):
    
    from cern import pymad
    
    # select backend, 
    # 'cpymad' is currently default if nothing is provided
    # Returns a pymad.service object
    pms=pymad.init('jpymad') 
    # Create a model:
    pm=pms.create_model('lhc')
    # Run twiss:
    # This returns a "lookup dictionary" containing
    # numpy arrays. Lowercase keys.
    twiss,summary=pm.twiss()
    # Your own analysis below:


See http://cern.ch/pymad/ for further documentation.