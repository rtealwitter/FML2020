#!/bin/bash

## Problem C.3
rm 3*
# Preprocessing data into libsvm format
python3 scriptpreprocess3.py
# Scaling
libsvm-3.24/svm-scale -s 3.range 3.train > 3scaled.train
libsvm-3.24/svm-scale -r 3.range 3.test > 3scaled.test

