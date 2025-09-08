ARG BASE=manylinux1_x86_64
FROM quay.io/pypa/$BASE

ARG UID=1000
ARG GID=$UID
# Highest version supported by manylinux1:
ENV PATH="/opt/python/cp39-cp39/bin:${PATH}"

RUN groupadd -o -g $GID runner && \
    useradd -o -u $UID -g $GID runner && \
    if [[ $AUDITWHEEL_PLAT == manylinux2014_* ]]; then \
        yum install -y glibc-static; \
    fi && \
    pip install --upgrade cmake --only-binary=cmake && \
    pip install setuptools

VOLUME /mnt
WORKDIR /mnt
USER $UID:$GID
