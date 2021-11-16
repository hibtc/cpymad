ARG BASE=manylinux1_x86_64
FROM quay.io/pypa/$BASE

ARG UID=1000
ARG GID=$UID
ENV PY=/opt/python/cp36-cp36m/bin

RUN groupadd -o -g $GID runner && \
    useradd -o -u $UID -g $GID runner

VOLUME /mnt
WORKDIR /mnt
USER $UID:$GID
