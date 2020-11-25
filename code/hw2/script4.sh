#!/bin/bash

## Problem C.4
rm 4*
# Finding d* and C* via 10-fold cross-validation
for d in 1 2 3 4
do
for i in {-10..10}
do
C=$(bc -l <<< "2 ^($i)")
libsvm-3.24/svm-train -t 1 -d $d -c $C -v 10 -h 0 3scaled.train > 4.temp
python3 scriptextract.py 4.temp 4.accuracy $d $C
done
done
python3 scriptplot4.py
