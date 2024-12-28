cpymad
------
|Version| |Python| |Platform| |License| |Tests| |Coverage| |Citation|

cpymad is a Cython_ binding to MAD-X_ for giving full control and access to a
MAD-X interpreter in python.

.. _Cython: https://cython.org/
.. _MAD-X: https://cern.ch/mad


**Note:** Support for 32bit builds and python 2.7 has been removed in version
1.8.0. Support for python 3.5 has been removed in version 1.10.0.

**Note:** python 3.8 and below, as well as manylinux1 have reached EOL.
Support will be removed in a future release.


Links
~~~~~

- `Documentation`_:
    - `Installation`_
    - `Getting started`_

- `Source code`_:
    - `MAD-X source`_

- `Issue tracker`_
- `Releases`_

.. _Getting started: http://hibtc.github.io/cpymad/getting-started
.. _Installation: http://hibtc.github.io/cpymad/installation
.. _Source code: https://github.com/hibtc/cpymad
.. _Documentation: http://hibtc.github.io/cpymad
.. _Issue tracker: https://github.com/hibtc/cpymad/issues
.. _Releases: https://pypi.org/project/cpymad
.. _MAD-X source: https://github.com/MethodicalAcceleratorDesign/MAD-X


License
~~~~~~~

The cpymad source code itself is under free license, see COPYING.rst_.

However, the MAD-X software package and henceforth all binary cpymad package
distributions are **NOT FREE**., see `MAD-X license`_.

.. _COPYING.rst: https://github.com/hibtc/cpymad/blob/master/COPYING.rst
.. _MAD-X license: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/master/License.txt


CHANGELOG
~~~~~~~~~

The full changelog is available online in CHANGES.rst_.

.. _CHANGES.rst: https://github.com/hibtc/cpymad/blob/master/CHANGES.rst


Reporting issues
~~~~~~~~~~~~~~~~

Note that cpymad links against a custom build of MAD-X that may differ from
the official CERN command line client. This binary may have problems that the
official binary does not have and vice versa.

Therefore, before reporting issues, please make sure that you report to the
correct recipient. First try to check if that problem remains when using the
MAD-X command line client distributed by CERN, then report the issue:

- to CERN if it can be reproduced with the official MAD-X executable
- to us if it can not be reproduced with the official MAD-X executable

Please keep the code in the bug report as minimal as possible, i.e. remove
everything that can be removed such that the issue still occurs. This will
save us some effort in handling the error report.

Please post the code inline, don't upload zip files, or link to external
sources, if possible.

Bug reports should describe the issue and contain a minimal python script
similar to this:

.. code-block:: python

    from cpymad.madx import Madx
    m = Madx()
    m.call('commands.madx')

as well as the content of the ``commands.madx`` file.

You can create this file from your original python code with a minimal change
that tells cpymad to write all MAD-X commands to a file:

.. code-block:: python

   m = Madx(command_log='commands.madx')
   ...

Now run this file with the official MAD-X command line client::

    madx commands.madx

If ``madx`` reports the same error, check if there are any syntax errors in
the ``commands.madx`` file. These may result from incorrect usage of cpymad,
or bugs in cpymad.


.. Badges:

.. |Tests| image::      https://github.com/hibtc/cpymad/workflows/build/badge.svg
   :target:             https://github.com/hibtc/cpymad/actions?query=workflow%3A%22build%22
   :alt:                GitHub Actions Status

.. |Coverage| image::   https://coveralls.io/repos/hibtc/cpymad/badge.svg?branch=master
   :target:             https://coveralls.io/r/hibtc/cpymad
   :alt:                Coverage

.. |Version| image::    https://img.shields.io/pypi/v/cpymad.svg
   :target:             https://pypi.org/project/cpymad
   :alt:                Latest Version

.. |License| image::    https://img.shields.io/badge/license-Mixed-red.svg
   :target:             https://github.com/hibtc/cpymad/blob/master/COPYING.rst
   :alt:                License: Source: CC0, Apache | Binary: Non-Free

.. |Platform| image::   https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-blue
   :target:             https://pypi.org/project/cpymad#files
   :alt:                Supported platforms

.. |Python| image::     https://img.shields.io/pypi/pyversions/cpymad.svg
   :target:             https://pypi.org/project/cpymad#files
   :alt:                Python versions

.. |Citation| image::   https://zenodo.org/badge/DOI/10.5281/zenodo.4724856.svg
   :target:             https://doi.org/10.5281/zenodo.4724856
   :alt:                DOI and Citation
