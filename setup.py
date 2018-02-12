from  setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

"""
TODO Add a custom command to install armadillo from through sudo apt install armadillo
Strategy is taken from
https://github.com/apache/beam/blob/master/sdks/python/apache_beam/examples/complete/juliaset/setup.py
"""

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name='ChannelAttribution',
    version='0.1',
    author='Davide Altromare & Fernando Gargiulo',
    description='Channel Attribution based on Markiv chains',
    packages = ['channelattribution'],
    package_dir = {'channelattribution': 'src'},
    ext_modules=[
        Extension(
          name='channelattribution._libc',
          sources=['src/libc/ChannelAttribution.cpp'],
          extra_compile_args=['-std=c++11'],
         )
        ],
    cmdclass={'build_ext': build_ext},
    install_requires=['numpy', 'pandas'],
    setup_requires=['numpy']
    )
