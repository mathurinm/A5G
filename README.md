The bash command to build the package is
```python setup.py build_ext --inplace```

If you want to install it system wide, the command is
```pip install -e .```

Then in a python console you can do
```from a5g.lasso_fast import a5g```


a5g for the lasso (lasso_fast.a5g) in fully coded in cython, whereas mt_a5g has its outer loop in python
and calls multitask_fast.mt_gram_solver (cython) on the inner loop.


Coming:
- sparse a5g solver
- full cython for mt_a5g
