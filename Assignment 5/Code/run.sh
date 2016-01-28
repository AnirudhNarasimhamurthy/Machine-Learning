#!/bin/sh
python SVM.py
python CrossValidation.py
python decisiontree_train.py 4
python decisiontree_train.py 8
python decisiontree_train.py 20
python decisiontree_test.py 4
python decisiontree_test.py 8
python decisiontree_test.py 20
python CrossValidation_Ensemble.py 4
python CrossValidation_Ensemble.py 8
python CrossValidation_Ensemble.py 20
python SVM_Ensemble.py 4 0.001 10
python SVM_Ensemble.py 8 0.001 100
python SVM_Ensemble.py 20 0.001 100


