# gcc-12 is the latest compiler version that includes gfortran that I could
# find on the MacOS runner. This may change, so be careful:
export CC=gcc-12
export CPP=cpp-12
export CXX=g++-12
export FC=gfortran-12
export LD=gcc-12

which "$CC"
which "$CPP"
which "$CXX"
which "$FC"
which "$LD"
