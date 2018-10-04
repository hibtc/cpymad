FROM quay.io/pypa/manylinux1_x86_64

COPY ./build_all /opt

# NOTE: already installed:
# RUN yum install -y cmake28 devtoolset-2-gcc-gfortran.x86_64

# NOTE: not needed, we will disable online model and X11
# RUN yum install -y zlib-devel libX11-devel

# Setup IO volume for persisting the build artifacts:
VOLUME /io

# If the user wants to supply their own `/io/cpymad` as bind volume, we should
# mess with their data to avoid permission issues in both host and container.
# We will therefore generate the list of source files by running the `egg_info`
# command in order to copy them to a separate build folder. Naturally, this
# requires write access to the `cpymad.egg-info` subfolder, which we can grant
# by making at another volume:
VOLUME /io/cpymad/cpymad.egg-info

CMD ["/opt/build_all"]
