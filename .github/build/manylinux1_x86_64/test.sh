#! /usr/bin/env bash
set -ex

for PY in $(cd /opt/python; eval "ls $@"); do
    /opt/python/$PY/bin/python -m venv env
    source env/bin/activate
    pip install cpymad -f dist --no-index --no-deps
    pip install cpymad pandas pytest
    pytest -v
    deactivate
done
