#! /usr/bin/env python
"""
Delete old branches from a github repository. This script is used in
`.travis.yml` to limit the number of space used by old builds on
https://github.com/hibtc/cpymad-artifacts.

Usage:
    python cleanup_artifacts.py REPO_URL CURRENT_REV NUM_KEEP

Arguments:
    REPO_URL        "https://{AUTH_TOKEN}@github.com/hibtc/cpymad-artifacts"
    CURRENT_REV     Newest build number, use $TRAVIS_BUILD_NUMBER
    NUM_KEEP        Number of old revisions to keep, e.g. 5
"""

from subprocess import run, check_output
import json
import sys
import os
import re

repo, build, num_keep = sys.argv[1:]

token, slug = re.match(
    r'https://([0-9a-f]{40})@github.com/(.*)', repo).groups()
min_build = int(build) - int(num_keep) + 1

with open(os.devnull) as null:
    data = check_output([
        'curl', '-H',
        'Authorization: token {}'.format(token),
        'https://api.github.com/repos/{}/git/refs/heads/'.format(slug),
    ], stderr=null)

branches = [r['ref'].split('/', 2)[2] for r in json.loads(data)]

travis = re.compile(r'_(\d+)-[0-9a-f]{40}')

cleanup = [
    ':' + branch
    for branch in branches
    for m in [travis.match(branch)]
    if m and int(m.group(1)) < min_build
]
print("Deleting:\n", "\n ".join(cleanup) or "(nothing)")

run(['git', 'push', repo] + cleanup)
