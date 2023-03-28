.. qp_flexzboost documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for ``qp_flexzboost``
===================================

The ``qp_flexzboost`` package allows for efficient, lossless storage of 
`Flexcode <https://github.com/lee-group-cmu/FlexCode>`_ [#]_ [#]_ 
conditional density estimates. It acts as an extension of `qp <https://github.com/LSSTDESC/qp>`_
and utilizes the functionality provided there. 

It will typically be used as a transparent shim between 
`qp <https://github.com/LSSTDESC/qp>`_, and 
`rail-flexzboost <https://github.com/LSSTDESC/rail_flexzboost>`_.

For detailed comparison of the performance of this package relative to built in 
``qp`` representations, please see the
:doc:`performance comparison <../source/performance_comparison>`.

Getting Started
---------------

The easiest way to install it is with ``pip``.

.. code:: bash

   >> pip install qp-flexzboost


Alternatively, the source code can be cloned from GitHub.

.. code:: bash

   >> git clone git@github.com:LSSTDESC/qp_flexzboost.git
   >> cd qp_flexzboost
   >> pip install -e .
   >> pip install .'[dev]' # Only required for development and contributions

This package is open source and uses the MIT license. 

Usage
-----

For the minimal usage take a look at this :doc:`minimal notebook <../notebooks/minimal_notebook>`.

Additional notebooks providing a general overview of usage can be found :doc:`here <../notebooks>`.

Note that this package will typically be used in conjunction with 
`RAIL <https://github.com/LSSTDESC/RAIL>`_, `qp <https://github.com/LSSTDESC/qp>`_, 
and `rail-flexzboost <https://github.com/LSSTDESC/rail_flexzboost>`_.


Citations
---------
.. [#] Rafael Izbicki and Ann B. Lee, “Converting high-dimensional regression to high-dimensional conditional density estimation”, Electron. J. Statist. 11(2): 2800-2831 (2017). DOI: 10.1214/17-EJS1302

.. [#] Schmidt et al, “Evaluation of probabilistic photometric redshift estimation approaches for The Rubin Observatory Legacy Survey of Space and Time (LSST)“, MNRAS, 449(2): 1587-1606. https://doi.org/10.1093/mnras/staa2799

.. toctree::
   :hidden:

   Home page <self>
   Performance Comparison <source/performance_comparison>
   API Reference <autoapi/index>
   Notebooks <notebooks>
