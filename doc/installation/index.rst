

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

First method, source script:

  For CPymad on a 64 bit Linux machine with afs available, it is enough to source the script that we 
  currently provide in 
  
   /afs/cern.ch/user/y/ylevinse/public/setupCPYMAD.sh
  
  We hope to in the near future provide a tarball for Linux and OSX containing all needed files. 

Second method, manual installation:

    * Download the Mad-X source from 
      `svn <http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1>`_ 
      and unpack it. 
    * Enter the folder madX and run the commands
      
      .. code-block:: sh
      
          mkdir build; cd build; cmake -DMADX_STATIC=OFF ../;make install
    
    * Download the PyMad source from `github <https://github.com/pymad/pymad/zipball/master>`_
      and unpack it
    * In the folder pymad/src, run the command
    
      .. code-block:: sh
      
          python setup.py install

If you download JMad after following any of the methods described above for CPyMad,
you will immediately have JPyMad available as well.


