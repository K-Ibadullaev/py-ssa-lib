# PY-SSA-LIB PACKAGE
## Intro
Welcome to the page of the **py-ssa-lib** package!
This package contains python implementations of the **Singular Spectrum Analysis(SSA)** and **Multichannel Singular Spectrum Analysis(MSSA)**. 

It can be used for the time series analysis and forecasting.

**Please, take a look at the guides for SSA and MSSA which are available in the [corresponding directory](https://github.com/K-Ibadullaev/py_ssa/tree/main/examples_and_guide)!**

## Mathematical Background
The [Wiki](https://github.com/K-Ibadullaev/py-ssa-lib/wiki) for the **py-ssa-lib** package is now available and will be periodically updated. It contains some theoretical background about the MSSA and SSA.
The API documentation is available [here](https://k-ibadullaev.github.io/py-ssa-lib/).

## Updates
**NEW:** Version 2.0.x is now available! 

- Simplified interface
- Improved visualization tools
- Extended SVD choice
- Enabled various weighting schemes for channels of the MSSA
- [API documentation](https://k-ibadullaev.github.io/py-ssa-lib/) 
...

## Installation
```shell
$ python -m pip install py-ssa-lib
```

## Requirements

The classes in the **py-ssa-lib** rely on the numpy, scipy, sklearn, pandas and matplotlib libraries.

## Similar Python Packages
Before the development of the **py-ssa-lib** I searched for the 
Python packages which implement both MSSA and SSA, and found only few decent packages with the similar functionality:

- https://github.com/AbdullahO/mSSA?tab=readme-ov-file
- https://github.com/kieferk/pymssa
  
(Arguably) the best functionality is provided by the RSSA-package in R (https://github.com/asl/rssa).



## List of the Core Packages
- NumPy https://numpy.org/
- SciPy https://scipy.org/
- Jupyter Lab https://jupyterlab.readthedocs.io/en/latest/index.html
- Scikit-learn https://scikit-learn.org/stable/
- Matplotlib https://matplotlib.org/


  
## Literature about SSA and MSSA
- https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition/notebook#2.-Introducing-the-SSA-Method
- https://link.springer.com/book/10.1007/978-3-642-34913-3
- https://link.springer.com/book/10.1007/978-3-662-57380-8
- https://www.gistatgroup.com/gus/mssa2.pdf
  



### Citation 
If you find this package useful, please, cite:

**Konstantin Ibadullaev, https://github.com/K-Ibadullaev/py_ssa-lib/**

(This file and the citation format will change over time.)




### Acknowledgements
This package is developed as a part of the research project "Intelligent Geosystems" (100693905) supported by ESF funding

![alt text](https://github.com/K-Ibadullaev/py-ssa-lib/blob/main/ESFICON.png)
