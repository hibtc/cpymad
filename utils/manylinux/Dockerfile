FROM quay.io/pypa/manylinux1_x86_64

# Cache cython installation:
ENV PY="/opt/python/cp36-cp36m/bin"
RUN $PY/pip install cython

# NOTE: already installed:
# RUN yum install -y cmake28 devtoolset-2-gcc-gfortran.x86_64

# NOTE: not needed, we will disable online model and X11
# RUN yum install -y zlib-devel libX11-devel

WORKDIR /io
VOLUME /io
CMD ["/io/build_all"]
