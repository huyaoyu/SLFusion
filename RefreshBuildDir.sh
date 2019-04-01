#!/bin/bash

echo "Delete build/."

rm -rf build/

echo "Create build/."

mkdir build

cd build/
cmake ../src

