Installation Instructions
*************************

In order to install cpymad, please try:

.. code-block:: bash

    pip install cpymad

This may need to build MAD-X in which case it can take a *long* time (roughly
between 5 and 30 minutes, depending on your internet connection and overall
system performance).

During this command some error messages may be generated in between even if
the overall command succeeds. The success of the installation can be judged
from messages printed to the screen near the end.

If the installation fails (which is not too unlikely due to the complexity of
the MAD-X dependency), you may have to install some dependencies first and
perhaps even build a MAD-X library version and the cpymad python extension
separately. In this case, please refer to the platform specific installation
instructions below:

.. toctree::
   :maxdepth: 1

   unix
   windows
   troubleshooting

Note that it is recommended to use python 3, preferrably 3.6 or later. As of
yet, python 2.7 is still supported, even if more likely to suffer latent
issues from the less stringent bytes and unicode handling in python 2. Support
for python 2 may be phased out in the upcoming versions.
