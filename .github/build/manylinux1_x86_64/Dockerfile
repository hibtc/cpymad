ARG BASE=manylinux1_x86_64
FROM quay.io/pypa/$BASE

ARG UID=1000
ARG GID=$UID
ENV PY=/opt/python/cp36-cp36m/bin

RUN groupadd -o -g $GID runner && \
    useradd -o -u $UID -g $GID runner && \
    if [[ $AUDITWHEEL_PLAT == manylinux2014_* ]]; then \
        yum install -y glibc-static; \
    fi && \
    $PY/pip install --upgrade cmake cython && \
    for PYBIN in /opt/python/cp3*/bin; do \
        "${PYBIN}/pip" install -U setuptools; \
    done

VOLUME /mnt
WORKDIR /mnt
USER $UID:$GID
