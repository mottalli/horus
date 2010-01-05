#!/usr/bin/python
from distutils.core import setup, Extension
import sys
import os
import fnmatch

SOURCES = ["horus_wrap.cxx"]
archivos = os.listdir('..')
for archivo in archivos:
	if fnmatch.fnmatch(archivo, '*.cpp') and archivo <> 'main.cpp' and archivo <> 'prueba_cuda.cpp':
		SOURCES.append('../'+archivo)

LIBRARIES = ['cv', 'cxcore', 'highgui']
INCLUDE_DIRS = ['c:\\Archivos de programa\\opencv\\__include']
LIBRARY_DIRS = ['c:\\Archivos de programa\\opencv\\lib']

if os.name == 'posix':
	horus_ext = Extension('_horus', sources=SOURCES, libraries=LIBRARIES)
elif os.name == 'nt':
	horus_ext = Extension('_horus', sources=SOURCES, libraries=LIBRARIES, 
					include_dirs=INCLUDE_DIRS, library_dirs=LIBRARY_DIRS)
else:
	raise Exception()

setup(name="horus",
	  version="0.1",
	  author="Marcelo L. Mottalli",
	  ext_modules=[horus_ext],
	  py_modules=["horus"]
	)
