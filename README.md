[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

# qp_flexzboost

This package allows for efficient, lossless storage of Flexcode conditional density estimates and leverages the machinery provided by qp. 

The primary module in the package proivdes the Flexzboost_Gen class, a subclass of qp's Pdf_rows_gen class. 

It provides an API to retrieve PDF, CDF, and PPF values in addition to supporting simple plotting of PDFs. 

While it is possible to use all of the standard `scipy.rvs_continuous` methods to work with a qp.Ensemble of CDEs stored as Flexzboost_Gen objects, it is much more efficient to convert the Flexzboost_Geb representation into a `qp.interp` (or any other native qp) representation. 

Flexzboost_Gen is not included as a part of qp by default primarily becuase it is only possible to convert Flexzboost_Gen objects to other qp representations. But it is not feasible to convert from a standard qp representation into a Flexzboost_Gen representation. 

For more informstion and examples, please see the ReadTheDocs documentation and API Reference available here: https://qp-flexzboost.readthedocs.io/en/latest/index.html


## Attribution

This project was automatically generated using the LINCC Frameworks [Python Project Template](https://github.com/lincc-frameworks/python-project-template).

For more information about the project template see the [documentation](https://lincc-ppt.readthedocs.io/en/latest/).
