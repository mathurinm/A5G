# Aggressive gap screening rules for faster Lasso-type solvers

This repository hosts the implementation of fast solvers for the Lasso and multi-task Lasso problem.

The algorithms are in ```./a5g/*_fast.pyx```.
Currently implemented are:
* Lasso solver on dense and sparse data
* Multi-task Lasso solver on dense data (aka Group Lasso with groups equal to rows)

The algorithms are written in Cython, using calls to BLAS when possible.

# Installation
Clone the repository:

```
$git clone git@github.com:mathurinm/A5G.git
$cd A5G/
$conda env create --file environment.yml
$source activate a5g-env
$python setup.py build_ext -i
```

# Dependencies
All dependencies are in  ```./environment.yml```

# Cite
If you use this code, please cite [this paper](https://arxiv.org/abs/1703.07285):

Mathurin Massias, Alexandre Gramfort and Joseph Salmon
From safe screening rules to working sets for faster Lasso-type solvers
Arxiv preprint, 2017
