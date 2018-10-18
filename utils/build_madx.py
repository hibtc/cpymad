"""
Simple setup utility to download and build MAD-X.
"""

from __future__ import print_function
from __future__ import division

import os
import sys
import zipfile
import subprocess
from contextlib import contextmanager

try:
    from urllib.request import urlretrieve
except ImportError:     # py2
    from urllib import urlretrieve

MADX_VERSION = '5.04.02'


def download(url, to=None):
    def report_progress(blocks_done, block_size, total_size):
        unit = 1024*1024
        size_done = blocks_done * block_size
        print("\rProgress: {:.1f}/{:.1f} MiB".format(
            size_done/unit, total_size/unit), end='')
    filename, http_result = urlretrieve(url, to, reporthook=report_progress)
    print()                 # terminate line
    return filename


def extract(filename, to=None):
    with zipfile.ZipFile(filename) as f:
        f.extractall(to)


def mkdir(dirname):
    try:
        os.mkdir(dirname)
        return True
    except OSError:
        return False


def build_madx(source_dir, build_dir, install_dir,
               static=False, shared=False, X11=True):
    cmake_args = [
        'cmake', os.path.abspath(source_dir),
        '-DMADX_ONLINE=OFF',
        '-DMADX_INSTALL_DOC=OFF',
        '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(install_dir),
        '-DCMAKE_BUILD_TYPE=Release',
        '-DMADX_X11='          + ('ON' if X11    else 'OFF'),
        '-DMADX_STATIC='       + ('ON' if static else 'OFF'),
        '-DBUILD_SHARED_LIBS=' + ('ON' if shared else 'OFF'),
    ]
    with chdir(build_dir):
        subprocess.call(cmake_args)
        subprocess.call(['make'])


@contextmanager
def chdir(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(old_cwd)


def install_madx(version=MADX_VERSION, prefix='.',
                 static=False, shared=False, X11=False):

    FILE    = '{}.zip'.format(version)
    BASE    = 'https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/'
    URL     = BASE + FILE
    ARCHIVE = os.path.join(prefix, 'MAD-X-{}.zip'.format(version))
    FOLDER  = os.path.join(prefix, 'MAD-X-{}'.format(version))
    BUILD   = os.path.join(FOLDER, 'build')
    INSTALL = os.path.join(FOLDER, 'install')

    try:
        os.makedirs(prefix)
    except OSError:
        pass

    print("Downloading: {}".format(ARCHIVE))
    if not os.path.exists(ARCHIVE):
        download(URL, ARCHIVE)
    else:
        print(" -> already downloaded.")
    print()

    print("Extracting to: {}".format(FOLDER))
    if not os.path.exists(FOLDER):
        extract(ARCHIVE, prefix)
    else:
        print(" -> already extracted!")
    print()

    print("Building MAD-X in: {}".format(BUILD))
    if mkdir(BUILD):
        build_madx(FOLDER, BUILD, INSTALL,
                   static=static, shared=shared, X11=False)
    else:
        print(" -> already built!")
    print()

    print("Installing MAD-X to: {}".format(INSTALL))
    if mkdir(INSTALL):
        with chdir(BUILD):
            subprocess.call(['make', 'install'])
    else:
        print(" -> already installed!")
    print()

    return INSTALL


if __name__ == '__main__':
    install_madx(*sys.argv[1:])
