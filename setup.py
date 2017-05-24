from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(name='a5g',
      version='0.1',
      description='A WS algorithm based on agressive screening rules',
      author='under review :)',
      author_email='xxx@ccc.com',
      url='',
      packages=['a5g'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('a5g.lasso_fast',
                    sources=['a5g/lasso_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
          Extension('a5g.multitask_fast',
                    sources=['a5g/multitask_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()])
                   ],
      )
