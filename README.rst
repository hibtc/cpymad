cpymad
------
|Version| |License| |Python| |Tests| |Coverage|

cpymad_ is a Cython_ binding to MAD-X_ for giving full control and access to a
MAD-X interpreter within the python environment.

This version of cpymad should be built against MAD-X |VERSION|. It can be
installed on linux, Windows and Mac (but I can't offer much help for Mac).
Python 3 is recommended but not (yet) required.

.. _cpymad: https://github.com/hibtc/cpymad
.. _Cython: https://cython.org/
.. _MAD-X: https://cern.ch/mad
.. |VERSION| replace:: 5.04.01


Links
~~~~~

- `Getting started`_
- `Installation`_
- `Source code`_
- `Documentation`_
- `Issue tracker`_
- `Latest release`_
- `MAD-X source`_

.. _Getting started: http://hibtc.github.io/cpymad/getting-started
.. _Installation: http://hibtc.github.io/cpymad/installation
.. _Source code: https://github.com/hibtc/cpymad
.. _Documentation: http://hibtc.github.io/cpymad
.. _Issue tracker: https://github.com/hibtc/cpymad/issues
.. _Latest release: https://pypi.org/project/cpymad#files
.. _MAD-X source: https://github.com/MethodicalAcceleratorDesign/MAD-X


License
~~~~~~~

The cpymad source code itself is under free license.

However, note that the MAD-X software package and henceforth all binary
cpymad package distributions are **NOT FREE**.

For more details, see COPYING.rst_ and the `MAD-X license`_.

.. _COPYING.rst: https://github.com/hibtc/cpymad/blob/master/COPYING.rst
.. _MAD-X license: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/master/License.txt


Reporting issues
~~~~~~~~~~~~~~~~

cpymad is NOT maintained by CERN members and links against an unofficial build
of MAD-X that is not supported by CERN, i.e. this binary may have problems
that the official binary does not have and vice versa. See also: `Reporting
issues`_.

If you have a problem with a sequence file, first try to check if that
problem remains when using the MAD-X command line client distributed by
CERN, then:

- Report the issue to CERN only if it **can** be reproduced with their
  official command line client.
- Report the issue here only if it can **not** be reproduced with their
  official command line client.

For issues regarding the cpymad code itself or usage information, I'm happy to
answer. Just keep in mind to be **precise**, **specific**, **concise** and
provide all the necessary information.

.. Badges:

.. |Tests| image::      https://api.travis-ci.org/hibtc/cpymad.svg?branch=master
   :target:             https://travis-ci.org/hibtc/cpymad
   :alt:                Test Status

.. |Coverage| image::   https://coveralls.io/repos/hibtc/cpymad/badge.svg?branch=master
   :target:             https://coveralls.io/r/hibtc/cpymad
   :alt:                Coverage

.. |Version| image::    https://img.shields.io/pypi/v/cpymad.svg
   :target:             https://pypi.org/project/cpymad
   :alt:                Latest Version

.. |License| image::    https://img.shields.io/badge/license-CC0,_Apache,_Non--Free-red.svg
   :target:             https://github.com/hibtc/cpymad/blob/master/COPYING.rst
   :alt:                License: CC0, Apache, Non-Free

.. |Python| image::     https://img.shields.io/pypi/pyversions/cpymad.svg
   :target:             https://pypi.org/project/cpymad#files
   :alt:                Python versions
