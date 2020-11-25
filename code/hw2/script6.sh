#!/bin/bash

# Problem C.6.d
rm 6*
# Transform data into sparsity minimization problem
python3 scriptpreprocess6.py
libsvm-3.24/svm-scale -s 6.range 6.train > 6scaled.train
libsvm-3.24/svm-scale -r 6.range 6.test > 6scaled.test
# Finding C* via cross-validation
d=2
for i in {-8..8}
do
C=$(bc -l <<< "2 ^($i)")
libsvm-3.24/svm-train -t 1 -d $d -c $C -v 10 -h 0 6scaled.train > 6.temp
python3 scriptextract.py 6.temp 6.accuracy $d $C
done
C=32 # degree is 2 and error is .268
# Testing via 10-fold cross-validation and test data
for d in 1 2 3 4 5 6
do
libsvm-3.24/svm-train -t 1 -d $d -c $C -h 0 -v 10 6scaled.train > 6.temp
python3 scriptextract.py 6.temp 6accuracy.cross $d $C
libsvm-3.24/svm-train -t 1 -d $d -c $C -h 0 6scaled.train 6.model > 6.temp
libsvm-3.24/svm-predict 6scaled.test 6.model 6.output > 6.temp
python3 scriptextract.py 6.temp 6accuracy.test $d $C
done
python3 scriptplot6.py

