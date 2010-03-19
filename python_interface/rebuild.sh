#!/bin/bash
sudo make install
sudo rm horus_wrap.cpp
sudo rm horus.py
touch horus.i
sudo python setup.py install
sudo python setup.py install
