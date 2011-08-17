These are the files needed to create a python library of madx using cython.

Prerequisites:
 - python
 - cython
 - gfortran or similar
 - libmadx.so
 - madextern.h in your include path

Installation:
    # We recommend to add the optional "--prefix=~/.local/" to this command:
    python setup.py install


Usage:
    # Once installed this is a nice example showing current
    # usability (run from examples folder):
    
    import pymadx
    from matplotlib import pyplot

    m=pymadx.madx()
    m.call('initLHC.madx')
    # returns the twiss table and the parameters in two separate dictionaries
    tw,p=m.twiss('LHCB1') 
    m.finish()
    print p['ENERGY']
    pyplot.plot(tw['S'],tw['BETX'])
    pyplot.show()


Getting libmadx.so:
    # Create a build directory and enter it, i.e.
    mkdir build; cd build
    # Use cmake with some special options:
    cmake -DBUILD_SHARED_LIBS=ON -DMADX_FORCE_32=OFF -DCMAKE_Fortran_COMPILER=gfortran -DMADX_ONLINE=OFF /path/to/folder/madX
    # Then build with:
    make madx