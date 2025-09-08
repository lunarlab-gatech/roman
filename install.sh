#!/bin/bash

# Make script exit on first failure
set -euo pipefail
set -x   # show each command and its expanded args

# Navigate into the directory
ROMAN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $ROMAN_DIR
echo "HELLO"
echo $ROMAN_DIR

# Install gdown if not already installed
pip install gdown

echo "WHAT"
echo $CI_TESTS
# For GitHub Actions, set directory as safe so submodule update works
if [ "$CI_TESTS" = "true" ]; then
  git config --global --add safe.directory "$ROMAN_DIR"
  echo "ACTION RAN"
fi

# Install CLIPPER
git submodule update --init --recursive
mkdir dependencies/clipper/build
cd dependencies/clipper/build
cmake .. && make && make pip-install

# Install Kimera-RPGO
mkdir $ROMAN_DIR/dependencies/Kimera-RPGO/build
cd $ROMAN_DIR/dependencies/Kimera-RPGO/build
cmake .. && make

# Install robotdatapy
cd $ROMAN_DIR/dependencies/robotdatapy
pip install .

# pip install
cd $ROMAN_DIR
pip install .

# download weights
mkdir -p $ROMAN_DIR/weights
cd $ROMAN_DIR/weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
gdown 'https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv'