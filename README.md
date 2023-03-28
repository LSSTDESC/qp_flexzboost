![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/lsstdesc/qp_flexzboost/testing-and-coverage.yml?branch=main)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lsstdesc/qp_flexzboost/smoke-test.yml?label=smoke%20test)
![Codecov](https://img.shields.io/codecov/c/github/lsstdesc/qp_flexzboost)
[![Read the Docs](https://img.shields.io/readthedocs/qp-flexzboost)](https://qp-flexzboost.readthedocs.io/en/latest/index.html)

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

# qp_flexzboost

This package allows for efficient, lossless storage of [Flexcode](https://github.com/lee-group-cmu/FlexCode)[^1][^2] conditional density estimates and leverages the machinery provided by [qp](https://github.com/LSSTDESC/qp). 

The primary module in the package provides the `FlexzboostGen` class, a subclass of the `qp.Pdf_rows_gen` class. 

An API to retrieve PDF, CDF, and PPF values in addition to supporting simple plotting of PDFs is provided. 

While it is possible to use all of the standard `scipy.rvs_continuous` methods to work with a `qp.Ensemble` of CDEs stored as `FlexzboostGen` objects, it is much more efficient to convert the `FlexzboostGen` representation into a native `qp` representation, such as `qp.interp`. 

`FlexzboostGen` is not included as a part of `qp` by default for the following reasons: 
1) It is not possible to convert from a native `qp` representation into a `FlexzboostGen` representation because `FlexzboostGen` stores the output of machine learned model. However, it **is** possible to convert from `FlexzboostGen` to any other native `qp` representation.
2) The use case is very tightly coupled to `Flexcode` and currently supports one specific use case - efficient storage of `qp.Ensemble` objects produced as output from [rail_flexzboost](https://github.com/LSSTDESC/rail_flexzboost) stages.

For more information and usage examples, please see the documentation and API reference available here: https://qp-flexzboost.readthedocs.io/en/latest/index.html


## Attribution

This project was automatically generated using the LINCC Frameworks [Python Project Template](https://github.com/lincc-frameworks/python-project-template).

For more information about the project template see the [documentation](https://lincc-ppt.readthedocs.io/en/latest/).



[^1]: Rafael Izbicki and Ann B. Lee, “Converting high-dimensional regression to high-dimensional conditional density estimation”, Electron. J. Statist. 11(2): 2800-2831 (2017). DOI: 10.1214/17-EJS1302

[^2]: Schmidt et al, “Evaluation of probabilistic photometric redshift estimation approaches for The Rubin Observatory Legacy Survey of Space and Time (LSST)“, MNRAS, 449(2): 1587-1606. https://doi.org/10.1093/mnras/staa2799