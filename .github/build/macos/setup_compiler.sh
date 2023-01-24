# gcc-12 is the latest compiler version that includes gfortran that I could
# find on the MacOS runner. This may change, so be careful:
export CC=gcc-12
export CXX=g++-12
export FC=gfortran-12

which "$CC"
which "$CXX"
which "$FC"
