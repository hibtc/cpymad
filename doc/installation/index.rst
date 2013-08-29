

Installation Instructions
*************************

Installation of PyMad is different depending on which underlying modules you want.


JPyMad
------

For JPyMad, you first need JMad which you can obtain `here <http://cern.ch/jmad/>`_
Then, download the `source <https://github.com/pymad/pymad>`_, and in pymad/src, run
the command

.. code-block:: sh

    python setup.py install --install-platlib

The argument *--install-platlib* means we exclude external modules. The reason is that the external module cern.madx requires the dynamic library of Mad-X available on your system (which we here assume you do not have)

CPyMad
------

First method, use installation script:
    We provide an `installation script <install.sh>`_ which should do the full job for you. Download the script
    and run it. It will take a few minutes to finish. Upon successful completion, it will create an uninstall.sh
    script which you should keep somewhere. This script knows all the files you have installed,
    and will remove all files if you execute it (folders are not removed).

    Dependencies: cmake, compilers for c/fortran, python 2.6 or 2.7

Second method, source script:

  For CPymad on a 64 bit Linux machine with afs available, it is enough to source the script that we
  currently provide in

   /afs/cern.ch/user/y/ylevinse/public/setupCPYMAD.sh

  We hope to in the near future provide a tarball for Linux and OSX containing all needed files.

Third method, manual installation:

    * Download the Mad-X source from
      `svn <http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1>`_
      and unpack it.
    * Enter the folder madX and run the commands

      .. code-block:: sh

          mkdir build; cd build
          cmake -DMADX_STATIC=OFF -DBUILD_SHARED_LIBS=ON ../
          make install

    * Download the PyMad source from `github <https://github.com/pymad/pymad/zipball/master>`_
      and unpack it
    * In the folder pymad/src, run the command

      .. code-block:: sh

          python setup.py install

If you download JMad after following any of the methods described above for CPyMad,
you will immediately have JPyMad available as well.


Potential problems
------------------

In the following we will try to keep a list of the various issues users have reported during installation.

    * libmadx.so not found::

          from cern.madx import madx
          ImportError: libmadx.so: cannot open shared object file: No such file or directory

      Solution:
      Though we try to set the runtime path during compilation, it doesn't always work. Please set
      the LD_LIBRARY_PATH in your environment. Example, if libmadx.so is installed in
      $HOME/.local/lib, and you use bash, add to $HOME/.bashrc:

      .. code-block:: sh

          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/

      Please note, on OSX you might need to use the variable DYLD_LIBRARY_PATH instead of
      LD_LIBRARY_PATH. The authors are not very familiar with OSX, but know of at least one
      occurence where that was the problem.

    * Cython.Distutils not found:

      .. code-block:: sh

        Traceback (most recent call last):
         File "setup.py", line 22, in <module>
          from Cython.Distutils import build_ext
        ImportError: No module named Cython.Distutils

     Solution:
     In order to get cpymad, you need Cython installed on your system. If you cannot obtain that, use jpymad instead.
