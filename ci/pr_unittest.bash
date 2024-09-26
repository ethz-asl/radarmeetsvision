#!/bin/bash

if pip install .
then
    echo "Installation successful"
else
    echo "Installation failed"
    exit 1
fi

if python3 -m unittest
then
    echo "Unit tests successful"
else
    echo "Unit tests failed"
    exit 1
fi
