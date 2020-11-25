#!/bin/bash

# Problem C.5
rm 5*
C=128 # degree is 2 and error is .193
# Testing on 10-fold cross-validation and test data
for d in 1 2 3 4 5 6
do
libsvm-3.24/svm-train -t 1 -d $d -c $C -h 0 -v 10 3scaled.train > 5.temp
python3 scriptextract.py 5.temp 5accuracy.cross $d $C
libsvm-3.24/svm-train -t 1 -d $d -c $C -h 0 3scaled.train 5.model > 5.temp
python3 scriptcount.py 5.model 5.counts
libsvm-3.24/svm-predict 3scaled.test 5.model 5.output > 5.temp
python3 scriptextract.py 5.temp 5accuracy.test $d $C
done
python3 scriptplot5.py

