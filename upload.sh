#!/bin/bash

# clear cache
rm -rf build/ dist/

# build
python setup.py sdist bdist_wheel

# upload
twine upload dist/*
