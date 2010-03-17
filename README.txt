1) Run cmake (>= 2.8)
2) Run make. If CUDA support is enabled, this will create the "horus_cuda" library in the python_interface directory
3) Run "make install" EVERY TIME THE SOURCE CODE CHANGES
3) run python_interface/setup.py install. This will update the python interface

EASIER: run "sudo make install && sudo python setup.py install" in python_interface. This does everything.
