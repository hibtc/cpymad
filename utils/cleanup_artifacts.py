#! /usr/bin/env python

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
