# Assignment 1: Linear Regression
The problem statement of the assignment is present [here](A1-PS.pdf). 

In this problem, we will use Linear Regression to predict the Total Costs of a hospital patient. We have been provided with training dataset of the SPARCS Hospital dataset. The dataset is present in this [link](https://drive.google.com/drive/folders/1fSytwJZXXVFNRCO7wcVyx9l8nAPFTfZc?usp=sharing).

The assignment consists of three parts-
- Analytic solution of Linear Regression using Moore-Penrose pseudo inverse.
- Analytic solution of Ridge Regression and 10-fold cross-validation to find the optimal regularization parameter.
- Feature Generation and Feature Selection using Lasso Regression. 

## How to run the code?
```
python3 linear.py Mode Parameters
```
The Mode corresponds to part [a,b,c] of the assignment.

The parameters are dependant on the mode:\
Part (a)
```
python3 linear.py a trainfile.csv testfile.csv outputfile.txt weightfile.txt
```
The predictions are stored in a line aligned ```outputfile.txt```. Also, the weights (including intercept in the very first line) are stored in the ```weightfile.txt```.
Part (b)
```
python3 linear.py b trainfile.csv testfile.csv regularization.txt outputfile.txt weightfile.txt bestparameter.txt
```
The best regularization parameter is reported in the file ```bestparameter.txt```.

Part (c)
```
python3 linear.py c trainfile.csv testfile.csv outputfile.txt
```
The features selected using Lasso Regression are used.

## Evaluation
The evaluation scripts are present in the [```evaluation```](evaluation/) folder. To evaluate different parts, run the following commands-

Part (a)
```
python3 grade_a.py outputfile_a.txt weightfile_a.txt model_outputfile_a.txt model_weightfile_a.txt
```
Part (b)
```
python3 grade_b.py outputfile_b.txt weightfile_b.txt model_outputfile_b.txt model_weightfile_b.txt
```

A  brief report on the feature generation and feature selection is present [here](A1-Report.pdf).

## Author
* Gaurav Jain [[GitHub](https://github.com/GauravJain28/)]
