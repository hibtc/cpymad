Installation Instructions
*************************

There are three basic steps when installing cpymad:

- install dependencies
- build MAD-X as a library
- build the cpymad binding

Depending on your platform, this can be more or less complicated.

Note that it is recommended to use python3, preferrably 3.6 or later. On
python2, there are likely unicode issues that will only surface once you pass
any non-ascii characters to/from MAD-X (which you probably should not do
anyway).


.. toctree::
   :maxdepth: 1

   unix
   windows
   troubleshooting
