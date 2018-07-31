Known issues
~~~~~~~~~~~~

On windows with python3.3, there is currently no satisfying way to close file
handles in the MAD-X process or prevent them from being inherited by default.
You have to make sure on your own that you close all file handles before
creating a new ``cpymad.madx.Madx`` instance!

On windows 10 with python≥3.5, 64bit, the prebuilt cpymad versions may crash.
There is no solution yet.
