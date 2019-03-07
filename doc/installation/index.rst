Installation Instructions
*************************

In order to install cpymad, please try:

.. code-block:: bash

    pip install cpymad --only-binary

If this fails, it usually means that we haven't uploaded wheels for your
platform or python version. In this case, either ping us about adding a
corresponding wheel, or refer to the platform specific installation
instructions:

.. toctree::
   :maxdepth: 1

   unix
   windows
   offline
   troubleshooting

Note that it is recommended to use python 3, preferrably 3.6 or later. As of
yet, python 2.7 is still supported, even if more likely to suffer latent
issues from the less stringent bytes and unicode handling in python 2. Support
for python 2 may be phased out in the upcoming versions.
