"""
Simple setup utility to download and build MAD-X.
"""

from __future__ import print_function
from __future__ import division

import glob
import os
import sys
import zipfile
import subprocess
import platform
from contextlib import contextmanager

try:
    from urllib.request import urlretrieve
except ImportError:     # py2
    from urllib import urlretrieve

IS_WIN32 = platform.system() == 'Windows'
MAKE = 'mingw32-make' if IS_WIN32 else 'make'
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
               static=IS_WIN32, shared=False, X11=True):
    cmake_args = [
        'cmake', os.path.abspath(source_dir),
        '-G', ('MinGW Makefiles' if IS_WIN32 else 'Unix Makefiles'),
        '-DMADX_ONLINE=OFF',
        '-DMADX_INSTALL_DOC=OFF',
        '-DCMAKE_INSTALL_PREFIX=' + os.path.abspath(install_dir),
        '-DCMAKE_BUILD_TYPE=Release',
        '-DMADX_X11='          + ('ON' if X11    else 'OFF'),
        '-DMADX_STATIC='       + ('ON' if static else 'OFF'),
        '-DBUILD_SHARED_LIBS=' + ('ON' if shared else 'OFF'),
    ]
    with chdir(build_dir):
        subprocess.check_call(cmake_args)
        subprocess.check_call([MAKE])


def apply_patches(source_dir, patch_dir):
    import patch
    patches = (glob.glob(os.path.join(patch_dir, '*.diff')) +
               glob.glob(os.path.join(patch_dir, '*.patch')))
    for filename in patches:
        print("Applying patch: {!r}".format(filename))
        patchset = patch.fromfile(filename)
        success = patchset.apply(1, root=source_dir)
        if not success:
            print("Failed to apply patch! Exitting...")
            sys.exit(1)


@contextmanager
def chdir(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(old_cwd)


def install_madx(version=MADX_VERSION, prefix='.', install_dir='',
                 patches=None, static=IS_WIN32, shared=False, X11=False):

    FILE    = '{}.zip'.format(version)
    BASE    = 'https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/'
    URL     = BASE + FILE
    ARCHIVE = os.path.join(prefix, 'MAD-X-{}.zip'.format(version))
    FOLDER  = os.path.join(prefix, 'MAD-X-{}'.format(version))
    BUILD   = os.path.join(FOLDER, 'build')
    INSTALL = os.path.join(FOLDER, 'install')
    if install_dir:
        INSTALL = install_dir

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

    if patches:
        print("Applying patches: {}".format(FOLDER))
        apply_patches(FOLDER, patches)

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
            subprocess.check_call([MAKE, 'install'])
    else:
        print(" -> already installed!")
    print()

    return INSTALL


if __name__ == '__main__':
    install_madx(*sys.argv[1:])
